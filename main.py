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

# Tile-drain inner loop
TILE_DRAIN_WAIT_MS  = 2000  # ms to wait after each click before re-detecting
MAX_TILE_CLICKS     = 3     # maximum click attempts per individual tile coordinate

# Session persistence — saves cookies/localStorage so future runs are
# already "recognised" by reCAPTCHA / Xiaomi, reducing captcha frequency.
AUTH_JSON_PATH      = "auth.json"

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

        # Step 3 ── wait for tiles to fully load, then enter the drain loop ──
        await page.wait_for_timeout(ROUND_WAIT_MS)

        # (initial screenshot/crop is taken inside the tile-drain loop below)

        # Step 4 ── Tile-drain inner loop ────────────────────────────────────
        # After every click a tile may fade out and be replaced by a new one.
        # We keep re-detecting and clicking until YOLO returns zero targets,
        # which means the grid is fully "drained" and we can move to Verify /
        # Next.  A per-coordinate click counter caps attempts at MAX_TILE_CLICKS
        # so a sticky tile that never fades cannot trap us in an infinite loop.

        event_loop  = asyncio.get_running_loop()
        click_counts: dict = {}   # key = (round_cx, round_cy), value = clicks so far

        drain_pass = 0
        while True:
            drain_pass += 1

            # Fresh screenshot + crop for this drain pass
            await page.screenshot(path=SCREENSHOT_PATH, full_page=True)
            raw_box_d  = await _bounding_box(page, BFRAME_SELS)
            crop_box_d = _clamp_box(raw_box_d) if raw_box_d else _clamp_box(CAPTCHA_CROP_FALLBACK)
            try:
                _crop_screenshot(SCREENSHOT_PATH, crop_box_d, CAPTCHA_CROP_PATH)
            except FileNotFoundError as exc:
                logger.error("Drain pass %d: %s", drain_pass, exc)
                break

            detections = await event_loop.run_in_executor(
                None, _run_yolo, CAPTCHA_CROP_PATH, (crop_box_d[0], crop_box_d[1])
            )

            logger.info("[round %d / drain %d] YOLO found %d target(s)",
                        rnd, drain_pass, len(detections))

            if not detections:
                logger.info("[round %d] Grid fully drained — proceeding to button", rnd)
                break

            clicked_any = False
            for label, conf, cx, cy in detections:
                # Round coords to a stable key (nearest 10 px grid)
                key = (round(cx / 10) * 10, round(cy / 10) * 10)
                attempts = click_counts.get(key, 0)

                if attempts >= MAX_TILE_CLICKS:
                    logger.warning(
                        "  Skipping '%s' at (%.1f,%.1f) — already clicked %d/%d times",
                        label, cx, cy, attempts, MAX_TILE_CLICKS,
                    )
                    continue

                # Human-like jitter + viewport clamp
                jx = max(0.0, min(_jitter(cx), float(VIEWPORT["width"]  - 1)))
                jy = max(0.0, min(_jitter(cy), float(VIEWPORT["height"] - 1)))
                logger.info(
                    "  [drain %d] Click '%s' conf=%.2f at (%.1f,%.1f) "
                    "[attempt %d/%d, jitter from (%.1f,%.1f)]",
                    drain_pass, label, conf, jx, jy,
                    attempts + 1, MAX_TILE_CLICKS, cx, cy,
                )
                await page.mouse.click(jx, jy)
                click_counts[key] = attempts + 1
                clicked_any = True
                await asyncio.sleep(CLICK_DELAY_S)

                # Wait for the tile fade/replace animation before re-detecting
                await page.wait_for_timeout(TILE_DRAIN_WAIT_MS)

            if not clicked_any:
                # All remaining detections hit their click cap — exit drain loop
                logger.warning(
                    "[round %d] All remaining tiles hit MAX_TILE_CLICKS (%d) — "
                    "exiting drain loop",
                    rnd, MAX_TILE_CLICKS,
                )
                break

        # Brief settle pause before reading the bframe button
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
    """Launch a stealth Chromium context.

    Session persistence via AUTH_JSON_PATH (auth.json):
      • If auth.json exists it is loaded as storage_state so cookies,
        localStorage and sessionStorage are restored.  reCAPTCHA and Xiaomi
        will treat the browser as a recognised session, reducing or skipping
        the image challenge entirely.
      • Call `await context.storage_state(path=AUTH_JSON_PATH)` at any point
        to persist the current session (see run() below).
    """
    browser = await playwright.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )

    # Load saved session if it exists
    auth_path = Path(AUTH_JSON_PATH)
    storage   = str(auth_path) if auth_path.exists() else None
    if storage:
        logger.info("Restoring session from %s", AUTH_JSON_PATH)
    else:
        logger.info("No saved session found — starting fresh (%s)", AUTH_JSON_PATH)

    context = await browser.new_context(
        user_agent=USER_AGENT,
        viewport=VIEWPORT,
        locale=LOCALE,
        timezone_id=TIMEZONE_ID,
        is_mobile=True,
        has_touch=True,
        storage_state=storage,   # None = fresh session; str path = restore
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
                # ── Persist session so future runs benefit from a recognised
                #    browser fingerprint / cookie jar (reduces captcha load).
                await context.storage_state(path=AUTH_JSON_PATH)
                logger.info("Session saved → %s", AUTH_JSON_PATH)
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
