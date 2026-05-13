"""
Xiaomi registration flow — Playwright + YOLOv8 multi-round reCAPTCHA solver.

Full pipeline:
  1. Mobile viewport (1080×1920, Android Chrome UA) to match Android Desktop-Site.
  2. Registration form:
       a. Click "Sign up" tab
       b. Fill Email / New password / Confirm password
       c. Tick "I've read and agreed to the Xiaomi Account User Agreement" checkbox
       d. Click orange "Next" button
  3. Multi-round reCAPTCHA solver loop (max 10 rounds):
       • If the anchor checkbox is visible and not yet checked → click it first
       • Detect targets with YOLOv8, click each with a ±5 px human-like jitter
       • If bottom button is "Next"   → click it, iterate to the next round
       • If bottom button is "Verify" → click it, wait for iframe to disappear → done
       • On round 10 timeout          → log CAPTCHA_FAILED, reload page
"""

import asyncio
import logging
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright_stealth import Stealth

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

XIAOMI_LOGIN_URL = "https://account.xiaomi.com/pass/serviceLogin?sid=passport"

# Mobile UA — must match the Android Desktop-Site layout we see in screenshots.
USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Mobile Safari/537.36"
)
VIEWPORT       = {"width": 1080, "height": 1920}
LOCALE         = "en-US"
TIMEZONE_ID    = "Asia/Manila"

# Debug / output files
SCREENSHOT_PATH  = "check.png"        # refreshed before every YOLO round
DEBUG_FORM_PATH  = "debug_form.png"   # taken after filling the form
CAPTCHA_CROP_PATH = "captcha_crop.png"

# YOLOv8
YOLO_MODEL_PATH           = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.20
TARGET_CLASSES = {
    "bus", "traffic light", "car", "truck",
    "bicycle", "motorcycle", "fire hydrant", "stop sign",
}

# Solver loop
MAX_CAPTCHA_ROUNDS  = 10    # give up after this many rounds
CLICK_DELAY_S       = 0.4   # pause between object clicks (seconds)
HUMAN_JITTER_PX     = 5     # ±px random offset on every click
ROUND_WAIT_MS       = 3000  # ms to wait for the grid to load each round

# Fallback full-page crop when bframe bounding box cannot be found
CAPTCHA_CROP_FALLBACK: Tuple[int, int, int, int] = (0, 0, 1080, 1920)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("xiaomi-captcha")

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_box(
    box: Tuple[int, int, int, int],
    max_w: int = 16384,
    max_h: int = 16384,
) -> Tuple[int, int, int, int]:
    """Ensure all coordinates are non-negative and box has positive area."""
    l = max(0, box[0])
    t = max(0, box[1])
    r = max(l + 1, min(box[2], max_w))
    b = max(t + 1, min(box[3], max_h))
    return (l, t, r, b)


def _jitter(coord: float, px: int = HUMAN_JITTER_PX) -> float:
    """Add a small random offset to a click coordinate."""
    return coord + random.uniform(-px, px)

# ─────────────────────────────────────────────────────────────────────────────
# YOLO helpers
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_yolo():
    from ultralytics import YOLO
    logger.info("Loading YOLO model: %s", YOLO_MODEL_PATH)
    return YOLO(YOLO_MODEL_PATH)


def _crop_screenshot(
    src: str,
    box: Tuple[int, int, int, int],
    dst: str,
) -> str:
    from PIL import Image
    if not Path(src).exists():
        raise FileNotFoundError(f"Screenshot not found: {src}")
    with Image.open(src) as img:
        img.crop(box).save(dst)
    logger.info("Saved crop %s → %s", box, dst)
    return dst


def _run_yolo(
    crop_path: str,
    origin: Tuple[int, int],
) -> List[Tuple[str, float, float, float]]:
    """Return list of (label, conf, viewport_cx, viewport_cy) sorted by conf desc."""
    model   = _load_yolo()
    results = model.predict(source=crop_path, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
    ox, oy  = origin
    hits: List[Tuple[str, float, float, float]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            label = result.names.get(int(box.cls[0]), "?")
            if label not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            hits.append((
                label,
                float(box.conf[0]),
                (x1 + x2) / 2.0 + ox,
                (y1 + y2) / 2.0 + oy,
            ))
    hits.sort(key=lambda d: d[1], reverse=True)
    return hits

# ─────────────────────────────────────────────────────────────────────────────
# reCAPTCHA iframe helpers
# ─────────────────────────────────────────────────────────────────────────────

# Anchor iframe = the small "I'm not a robot" widget
ANCHOR_SELS = (
    'iframe[title="reCAPTCHA"]',
    'iframe[src*="recaptcha/api2/anchor"]',
    'iframe[src*="recaptcha/enterprise/anchor"]',
)

# bframe = the image-selection challenge popup
BFRAME_SELS = (
    'iframe[src*="recaptcha/api2/bframe"]',
    'iframe[src*="recaptcha/enterprise/bframe"]',
    'iframe[title*="recaptcha challenge"]',
)


async def _bounding_box(
    page: Page,
    selectors: Tuple[str, ...],
) -> Optional[Tuple[int, int, int, int]]:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if await loc.count() == 0:
                continue
            raw = await loc.first.bounding_box()
        except Exception:
            continue
        if not raw:
            continue
        l = int(raw["x"])
        t = int(raw["y"])
        r = l + int(raw["width"])
        b = t + int(raw["height"])
        logger.info("bbox via %s → (%d,%d,%d,%d)", sel, l, t, r, b)
        return (l, t, r, b)
    return None


async def _anchor_is_visible(page: Page) -> bool:
    for sel in ANCHOR_SELS:
        try:
            loc = page.locator(sel)
            if await loc.count() > 0 and await loc.first.is_visible():
                return True
        except Exception:
            pass
    return False


async def _bframe_is_visible(page: Page) -> bool:
    for sel in BFRAME_SELS:
        try:
            loc = page.locator(sel)
            if await loc.count() > 0 and await loc.first.is_visible():
                return True
        except Exception:
            pass
    return False


async def _click_anchor_checkbox(page: Page) -> bool:
    """Enter the anchor iframe and click #recaptcha-anchor to open the challenge."""
    for sel in ANCHOR_SELS:
        try:
            if await page.locator(sel).count() == 0:
                continue
            frame    = page.frame_locator(sel)
            checkbox = frame.locator("#recaptcha-anchor")
            if await checkbox.count() == 0:
                checkbox = frame.locator("[role='checkbox']")
            if await checkbox.count() == 0:
                continue
            await checkbox.click(timeout=5000)
            logger.info("✓ Anchor checkbox clicked via: %s", sel)
            return True
        except Exception as exc:
            logger.warning("Anchor click failed (%s): %s", sel, exc)
    logger.warning("✗ Anchor checkbox not found")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# bframe button helpers (Next / Verify inside the challenge popup)
# ─────────────────────────────────────────────────────────────────────────────

async def _get_bframe_button_text(page: Page) -> str:
    """Return the lowercase text of the primary action button inside the bframe."""
    for sel in BFRAME_SELS:
        try:
            if await page.locator(sel).count() == 0:
                continue
            frame = page.frame_locator(sel)
            # reCAPTCHA uses #recaptcha-verify-button for Verify
            # and   #recaptcha-reload-button is "Get new challenge" (skip)
            for btn_sel in ("#recaptcha-verify-button", ".rc-button-default", "button"):
                btn = frame.locator(btn_sel)
                if await btn.count() == 0:
                    continue
                txt = (await btn.first.inner_text()).strip().lower()
                if txt:
                    return txt
        except Exception:
            pass
    return ""


async def _click_bframe_button(page: Page, label: str) -> bool:
    """Click a button by its text inside the bframe challenge popup."""
    label_lower = label.lower()
    for sel in BFRAME_SELS:
        try:
            if await page.locator(sel).count() == 0:
                continue
            frame = page.frame_locator(sel)
            for btn_sel in ("#recaptcha-verify-button", ".rc-button-default", "button"):
                btn = frame.locator(btn_sel)
                cnt = await btn.count()
                for i in range(cnt):
                    b   = btn.nth(i)
                    txt = (await b.inner_text()).strip().lower()
                    if label_lower in txt:
                        await b.click(timeout=5000)
                        logger.info("✓ bframe '%s' button clicked", label)
                        return True
        except Exception as exc:
            logger.warning("bframe button click failed (%s): %s", label, exc)
    logger.warning("✗ bframe '%s' button not found", label)
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Multi-round reCAPTCHA solver
# ─────────────────────────────────────────────────────────────────────────────

async def solve_captcha_loop(page: Page) -> bool:
    """
    Multi-round solver loop (up to MAX_CAPTCHA_ROUNDS).

    Each round:
      1. If the anchor checkbox is visible (challenge not open) → click it.
      2. Wait ROUND_WAIT_MS for image tiles to load.
      3. Take a fresh screenshot, crop to the bframe bounding box.
      4. Run YOLOv8 — click each detection with ±HUMAN_JITTER_PX offset.
      5. Read the primary bframe button label:
           "next"   → click it, continue to next round (new image set)
           "verify" → click it, wait for iframe to disappear → SUCCESS
      6. If no detections AND button is "verify" → click anyway (all tiles chosen).

    Returns True on success, False on timeout / repeated failure.
    """
    for rnd in range(1, MAX_CAPTCHA_ROUNDS + 1):
        logger.info("── Captcha round %d / %d ──", rnd, MAX_CAPTCHA_ROUNDS)

        # Step 1 ── trigger the challenge if anchor is still showing ─────────
        if await _anchor_is_visible(page) and not await _bframe_is_visible(page):
            logger.info("Anchor visible — clicking 'I'm not a robot'…")
            await _click_anchor_checkbox(page)
            await page.wait_for_timeout(ROUND_WAIT_MS)

        # Make sure bframe is actually there now
        if not await _bframe_is_visible(page):
            logger.warning("bframe still not visible after round %d anchor click", rnd)
            await page.wait_for_timeout(2000)
            if not await _bframe_is_visible(page):
                logger.warning("Giving up waiting for bframe")
                break

        # Step 2 ── wait for tiles to fully load ─────────────────────────────
        await page.wait_for_timeout(ROUND_WAIT_MS)

        # Step 3 ── screenshot + crop ─────────────────────────────────────────
        await page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        logger.info("[round %d] Screenshot saved → %s", rnd, SCREENSHOT_PATH)

        raw_box  = await _bounding_box(page, BFRAME_SELS)
        crop_box = _clamp_box(raw_box) if raw_box else _clamp_box(CAPTCHA_CROP_FALLBACK)
        logger.info("[round %d] Crop box: %s", rnd, crop_box)

        try:
            _crop_screenshot(SCREENSHOT_PATH, crop_box, CAPTCHA_CROP_PATH)
        except FileNotFoundError as exc:
            logger.error(exc)
            break

        # Step 4 ── YOLO detection + human-like clicks ────────────────────────
        loop       = asyncio.get_running_loop()
        detections = await loop.run_in_executor(
            None, _run_yolo, CAPTCHA_CROP_PATH, (crop_box[0], crop_box[1])
        )

        logger.info("[round %d] YOLO found %d target(s)", rnd, len(detections))
        for label, conf, cx, cy in detections:
            logger.info("  %-15s conf=%.2f  vp=(%.1f, %.1f)", label, conf, cx, cy)

        for label, _, cx, cy in detections:
            # Apply human-like jitter and clamp to viewport
            jx = max(0.0, min(_jitter(cx), float(VIEWPORT["width"]  - 1)))
            jy = max(0.0, min(_jitter(cy), float(VIEWPORT["height"] - 1)))
            logger.info("  Click '%s' at (%.1f, %.1f) [jittered from (%.1f, %.1f)]",
                        label, jx, jy, cx, cy)
            await page.mouse.click(jx, jy)
            await asyncio.sleep(CLICK_DELAY_S)

        # Short pause before checking the button
        await page.wait_for_timeout(800)

        # Step 5 ── read bframe button and act ────────────────────────────────
        btn_text = await _get_bframe_button_text(page)
        logger.info("[round %d] bframe button text: %r", rnd, btn_text)

        if "next" in btn_text:
            logger.info("[round %d] → Clicking 'Next' (new image set incoming)", rnd)
            await _click_bframe_button(page, "next")
            await page.wait_for_timeout(2000)
            continue  # next round

        elif "verify" in btn_text or btn_text == "":
            # Either "verify" is showing, or we couldn't read it — try verify
            logger.info("[round %d] → Clicking 'Verify'", rnd)
            await _click_bframe_button(page, "verify")
            await page.wait_for_timeout(3000)

            # Check whether the challenge disappeared (success) ───────────────
            if not await _bframe_is_visible(page):
                logger.info("✓ CAPTCHA SOLVED on round %d", rnd)
                return True

            # Challenge still showing — may have been wrong, continue
            logger.warning("[round %d] bframe still visible after Verify — retrying", rnd)
            await page.wait_for_timeout(2000)
            continue

        else:
            # Unknown button — just wait and retry
            logger.warning("[round %d] Unknown button %r — waiting 2 s", rnd, btn_text)
            await page.wait_for_timeout(2000)
            continue

    # ── Timeout: max rounds reached ───────────────────────────────────────────
    logger.error("CAPTCHA_FAILED — max rounds (%d) reached, reloading page", MAX_CAPTCHA_ROUNDS)
    await page.reload(wait_until="networkidle")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Browser context
# ─────────────────────────────────────────────────────────────────────────────

async def launch_stealth_context(playwright, headless: bool = True):
    browser = await playwright.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    context = await browser.new_context(
        user_agent=USER_AGENT,
        viewport=VIEWPORT,
        locale=LOCALE,
        timezone_id=TIMEZONE_ID,
        is_mobile=True,
        has_touch=True,
    )
    await Stealth().apply_stealth_async(context)
    return browser, context

# ─────────────────────────────────────────────────────────────────────────────
# Generic click / fill helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _try_click(page: Page, selectors: list, label: str) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0:
                continue
            if not await loc.is_visible():
                continue
            await loc.click(timeout=3000)
            logger.info("✓ %s — clicked via: %s", label, sel)
            return True
        except Exception:
            continue
    logger.warning("✗ %s — no selector matched", label)
    return False


async def _try_fill(page: Page, selectors: list, value: str, label: str) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0:
                continue
            await loc.fill(value, timeout=3000)
            logger.info("✓ %s — filled via: %s", label, sel)
            return True
        except Exception:
            continue
    logger.warning("✗ %s — no selector matched", label)
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Registration form
# ─────────────────────────────────────────────────────────────────────────────

async def _fill_registration_form(page: Page) -> None:
    """
    Complete registration flow:
      1. Sign up tab
      2. Email
      3. New password
      4. Confirm new password
      5. Agreement checkbox  ← CRITICAL: captcha won't appear without this
      6. Next button
    """
    W = 2000  # ms wait between steps

    # ── 1. Sign up tab ───────────────────────────────────────────────────────
    await _try_click(page, [
        "text=Sign up", "text=Create account", "text=Register", "text=注册",
        ".tab-item:has-text('Sign up')", ".tab-item:has-text('Register')",
        "[role='tab']:has-text('Sign up')", "[role='tab']:has-text('Register')",
        "a:has-text('Sign up')", "a:has-text('Register')", "a[href*='register']",
        "button:has-text('Sign up')",
        "[data-testid='signup-tab']", "[data-testid='register-tab']",
    ], "Sign up tab")
    await page.wait_for_timeout(W)

    # ── 2. Email ─────────────────────────────────────────────────────────────
    await _try_fill(page, [
        "input[type='email']", "input[name='email']", "input[name='account']",
        "input[placeholder*='email' i]", "input[placeholder*='Email' i]",
        "input[placeholder*='mail' i]", "input[id*='email' i]",
        "input[type='text']:visible",
    ], "testuser_debug@example.com", "Email")
    await page.wait_for_timeout(W)

    # ── 3. New password (first password field) ───────────────────────────────
    await _try_fill(page, [
        "input[name='password']", "input[name='newPassword']",
        "input[name='new_password']",
        "input[placeholder*='new password' i]",
        "input[placeholder*='Enter your new password' i]",
        "input[placeholder*='password' i]", "input[id*='password' i]",
        "(//input[@type='password'])[1]",
        "input[type='password']",
    ], "DebugPass123!", "New password")
    await page.wait_for_timeout(W)

    # ── 4. Confirm password (second password field) ──────────────────────────
    await _try_fill(page, [
        "input[name='confirmPassword']", "input[name='confirm_password']",
        "input[name='passwordConfirm']",
        "input[placeholder*='confirm' i]",
        "input[placeholder*='Confirm new password' i]",
        "input[type='password'] >> nth=1",
        "(//input[@type='password'])[2]",
    ], "DebugPass123!", "Confirm password")
    await page.wait_for_timeout(W)

    # ── Debug screenshot (post-fill) ─────────────────────────────────────────
    await page.screenshot(path=DEBUG_FORM_PATH, full_page=True)
    logger.info("Post-fill screenshot → %s", DEBUG_FORM_PATH)

    # ── 5. Agreement checkbox ─────────────────────────────────────────────────
    # CRITICAL: disabling the Next button until this is checked.
    await _try_click(page, [
        "input[type='checkbox']",
        ".agreement-checkbox input", ".agree-checkbox input",
        ".policy-checkbox input", ".checkbox input",
        "label.agreement input", "label.agree input",
        "label:has-text('agree') input", "label:has-text('Agreement') input",
        "label:has-text('User Agreement') input",
        "[class*='agree'] input[type='checkbox']",
        "[class*='policy'] input[type='checkbox']",
        "input[type='checkbox']:visible",
    ], "Agreement checkbox")
    await page.wait_for_timeout(W)

    # ── 6. Next button ────────────────────────────────────────────────────────
    await _try_click(page, [
        "button:has-text('Next')", "button:has-text('Continue')",
        "button:has-text('Sign up')", "button:has-text('Register')",
        "button:has-text('Create')",
        "button:has-text('下一步')", "button:has-text('注册')",
        "button:has-text('确定')",
        "text=Next", "text=Continue",
        "button[type='submit']", "input[type='submit']",
        ".n-footer .n-btn", ".n-footer button",
        ".submit-btn", ".btn-submit", ".btn-primary", ".next-btn",
        "[class*='submit']", "[class*='next']", "[class*='primary']",
        "[aria-label*='next' i]", "[aria-label*='submit' i]",
        "[data-testid*='next' i]", "[data-testid*='submit' i]",
        "button:visible",
    ], "Next button")

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def run(headless: bool = True) -> None:
    async with async_playwright() as pw:
        browser, context = await launch_stealth_context(pw, headless=headless)
        page: Optional[Page] = None
        try:
            page = await context.new_page()

            # ── Navigate ──────────────────────────────────────────────────────
            logger.info("Opening Xiaomi login page…")
            await page.goto(XIAOMI_LOGIN_URL, wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle")
            logger.info("Landed on: %s", page.url)

            # ── Registration form ─────────────────────────────────────────────
            logger.info("Filling registration form…")
            await _fill_registration_form(page)

            # ── Wait for reCAPTCHA anchor to render after Next ────────────────
            logger.info("Waiting 10 s for reCAPTCHA anchor…")
            await page.wait_for_timeout(10000)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)

            # ── Debug: dump iframe tags from page source ──────────────────────
            html        = await page.content()
            iframe_tags = re.findall(r"<iframe[^>]*>", html, re.IGNORECASE)
            logger.info("=== IFRAME DUMP (%d found) ===", len(iframe_tags))
            for i, tag in enumerate(iframe_tags):
                logger.info("  [%d] %s", i, tag)
            logger.info("=== END IFRAME DUMP ===")

            # ── Multi-round captcha solver ────────────────────────────────────
            solved = await solve_captcha_loop(page)
            if solved:
                logger.info("Registration captcha passed — continuing flow.")
            else:
                logger.error("Registration captcha could not be solved.")

            if not headless:
                logger.info("Headed mode — press Ctrl+C to exit.")
                await asyncio.Event().wait()

        finally:
            if page:
                await page.close()
            await context.close()
            await browser.close()


def main() -> None:
    try:
        asyncio.run(run(headless=True))
    except KeyboardInterrupt:
        logger.info("Interrupted.")


if __name__ == "__main__":
    main()
