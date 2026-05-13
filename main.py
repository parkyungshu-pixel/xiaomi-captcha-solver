"""
Xiaomi registration flow automation using Playwright with stealth mode.

Flow:
  1. Open Xiaomi login page in mobile viewport (1080x1920, Android Chrome UA).
  2. Click the "Sign up" tab.
  3. Fill Email, New password, Confirm new password.
  4. Tick the "I've read and agreed..." checkbox — required to enable Next.
  5. Click the orange "Next" button.
  6. Wait 10 s for the captcha to fully render.
  7. Auto-detect the captcha iframe, crop it from the screenshot, run YOLOv8.
  8. Click the detected object centers via page.mouse.click().
"""

import asyncio
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright_stealth import Stealth

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
XIAOMI_LOGIN_URL = "https://account.xiaomi.com/pass/serviceLogin?sid=passport"

# ---------------------------------------------------------------------------
# Browser / viewport config
# Mobile viewport to match the Android Desktop-Site view in the screenshot.
# page.mouse.click() coords must match the viewport, NOT the screenshot pixels
# when full_page=True captures content beyond the fold.
# ---------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Mobile Safari/537.36"
)
VIEWPORT = {"width": 1080, "height": 1920}
LOCALE = "en-US"
TIMEZONE_ID = "Asia/Manila"

# ---------------------------------------------------------------------------
# Screenshot / debug paths
# ---------------------------------------------------------------------------
SCREENSHOT_PATH   = "check.png"        # full-page after captcha triggered
DEBUG_FORM_PATH   = "debug_form.png"   # after fields are filled, before Next

# ---------------------------------------------------------------------------
# Captcha / YOLO config
# ---------------------------------------------------------------------------
# Fallback crop used when the bframe bounding-box lookup fails.
# (left, top, right, bottom) in screenshot coordinates.
CAPTCHA_CROP_BOX: Tuple[int, int, int, int] = (0, 0, 1080, 1920)

CAPTCHA_CROP_PATH = "captcha_crop.png"
YOLO_MODEL_PATH   = "yolov8n.pt"

TARGET_CLASSES = {
    "bus", "traffic light", "car", "truck",
    "bicycle", "motorcycle", "fire hydrant", "stop sign",
}

YOLO_CONFIDENCE_THRESHOLD = 0.20
CLICK_DELAY_SECONDS       = 0.4

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("xiaomi-captcha")


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_yolo_model():
    from ultralytics import YOLO
    logger.info("Loading YOLO model: %s", YOLO_MODEL_PATH)
    return YOLO(YOLO_MODEL_PATH)


def _crop_captcha(
    screenshot_path: str,
    crop_box: Tuple[int, int, int, int],
    out_path: str,
) -> str:
    from PIL import Image
    if not Path(screenshot_path).exists():
        raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
    with Image.open(screenshot_path) as img:
        img.crop(crop_box).save(out_path)
    logger.info("Saved captcha crop %s → %s", crop_box, out_path)
    return out_path


def _detect_targets(
    crop_path: str,
    crop_origin: Tuple[int, int],
) -> List[Tuple[str, float, float, float]]:
    model   = _load_yolo_model()
    results = model.predict(source=crop_path, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
    ox, oy  = crop_origin
    hits: List[Tuple[str, float, float, float]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            label = result.names.get(int(box.cls[0]), "?")
            if label not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            hits.append((label, float(box.conf[0]),
                          (x1 + x2) / 2 + ox,
                          (y1 + y2) / 2 + oy))
    hits.sort(key=lambda d: d[1], reverse=True)
    return hits


# ---------------------------------------------------------------------------
# reCAPTCHA helpers
# ---------------------------------------------------------------------------

def _clamp_box(box: Tuple[int, int, int, int],
               max_w: int = 16384,
               max_h: int = 16384) -> Tuple[int, int, int, int]:
    """Clamp a (left, top, right, bottom) crop box so every coordinate is
    non-negative and right > left, bottom > top.  Negative viewport positions
    can occur when the iframe is partially off-screen."""
    left   = max(0, box[0])
    top    = max(0, box[1])
    right  = max(left + 1, min(box[2], max_w))
    bottom = max(top  + 1, min(box[3], max_h))
    return (left, top, right, bottom)


async def _get_bounding_box(page: Page,
                            selectors: Tuple[str, ...]) -> Optional[Tuple[int, int, int, int]]:
    """Try each selector in turn; return the first valid bounding box found."""
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
        left  = int(raw["x"])
        top   = int(raw["y"])
        right  = left + int(raw["width"])
        bottom = top  + int(raw["height"])
        logger.info("Bounding box via %s → (%d,%d,%d,%d)", sel, left, top, right, bottom)
        return (left, top, right, bottom)
    return None


# Selectors for the reCAPTCHA ANCHOR iframe (the "I'm not a robot" checkbox).
RECAPTCHA_ANCHOR_SELECTORS = (
    'iframe[title="reCAPTCHA"]',
    'iframe[src*="recaptcha/api2/anchor"]',
    'iframe[src*="recaptcha/enterprise/anchor"]',
)

# Selectors for the reCAPTCHA CHALLENGE iframe (the image grid popup).
RECAPTCHA_BFRAME_SELECTORS = (
    'iframe[src*="recaptcha/api2/bframe"]',
    'iframe[src*="recaptcha/enterprise/bframe"]',
    'iframe[title*="recaptcha challenge"]',
)


async def _click_recaptcha_anchor(page: Page) -> bool:
    """Find the reCAPTCHA anchor iframe, switch into it, and click the
    checkbox.  This is what triggers the image-selection challenge to appear.

    Returns True if the checkbox was successfully clicked.
    """
    for sel in RECAPTCHA_ANCHOR_SELECTORS:
        try:
            loc = page.locator(sel)
            if await loc.count() == 0:
                continue

            # frame_locator lets us query elements inside a cross-origin iframe
            frame = page.frame_locator(sel)
            checkbox = frame.locator("#recaptcha-anchor")
            if await checkbox.count() == 0:
                # fallback: try any role=checkbox inside the iframe
                checkbox = frame.locator("[role='checkbox']")
            if await checkbox.count() == 0:
                logger.warning("No checkbox found inside anchor iframe (%s)", sel)
                continue

            await checkbox.click(timeout=5000)
            logger.info("✓ reCAPTCHA anchor checkbox clicked via iframe: %s", sel)
            return True
        except Exception as exc:
            logger.warning("Could not click anchor via %s: %s", sel, exc)
            continue

    logger.warning("✗ reCAPTCHA anchor iframe not found — captcha may not pop up")
    return False


async def _locate_challenge_box(page: Page) -> Optional[Tuple[int, int, int, int]]:
    """Return the bounding box of the reCAPTCHA challenge (bframe) iframe,
    clamped to non-negative coordinates."""
    raw = await _get_bounding_box(page, RECAPTCHA_BFRAME_SELECTORS)
    if raw is None:
        return None
    clamped = _clamp_box(raw)
    if clamped != raw:
        logger.info("Clamped challenge box %s → %s", raw, clamped)
    return clamped


# ---------------------------------------------------------------------------
# YOLO captcha solver
# ---------------------------------------------------------------------------
async def solve_captcha(page: Page) -> bool:
    """Full two-phase reCAPTCHA solve:

    Phase 1 — Trigger the image challenge:
        a. Find the reCAPTCHA anchor iframe and click its checkbox.
        b. Wait 3 s for the image-selection grid to fully render.

    Phase 2 — Detect and click targets:
        a. Take a fresh full-page screenshot AFTER the grid is visible.
        b. Get the bframe bounding box and crop captcha_crop.png from it.
        c. Run YOLOv8 on the crop to detect TARGET_CLASSES objects.
        d. Click each detection center via page.mouse.click().
    """
    # ── Phase 1: click the checkbox to open the challenge ───────────────────
    logger.info("Phase 1: clicking reCAPTCHA anchor checkbox…")
    await _click_recaptcha_anchor(page)

    # Wait for the image challenge popup (bframe) to fully load.
    logger.info("Waiting 3 s for image challenge to render…")
    await page.wait_for_timeout(3000)

    # ── Phase 2a: fresh screenshot now that the grid is visible ─────────────
    await page.screenshot(path=SCREENSHOT_PATH, full_page=True)
    logger.info("Saved post-challenge screenshot → %s", SCREENSHOT_PATH)

    # ── Phase 2b: locate bframe and crop ────────────────────────────────────
    crop_box = await _locate_challenge_box(page)
    if crop_box is None:
        logger.warning("bframe not found — falling back to full-page crop %s",
                       CAPTCHA_CROP_BOX)
        crop_box = _clamp_box(CAPTCHA_CROP_BOX)

    logger.info("Using crop box: %s", crop_box)

    try:
        crop_path = _crop_captcha(SCREENSHOT_PATH, crop_box, CAPTCHA_CROP_PATH)
    except FileNotFoundError as exc:
        logger.error(exc)
        return False

    # ── Phase 2c: YOLO inference ─────────────────────────────────────────────
    loop       = asyncio.get_running_loop()
    detections = await loop.run_in_executor(
        None, _detect_targets, crop_path, (crop_box[0], crop_box[1])
    )

    if not detections:
        logger.warning("No targets detected (classes=%s, conf≥%.2f)",
                       sorted(TARGET_CLASSES), YOLO_CONFIDENCE_THRESHOLD)
        return False

    logger.info("YOLO found %d target(s):", len(detections))
    for label, conf, cx, cy in detections:
        logger.info("  %-15s conf=%.2f  viewport=(%.1f, %.1f)", label, conf, cx, cy)

    # ── Phase 2d: click each detected object ─────────────────────────────────
    for label, _, cx, cy in detections:
        # Extra safety: clamp click coords to viewport bounds
        cx = max(0.0, min(cx, float(VIEWPORT["width"]  - 1)))
        cy = max(0.0, min(cy, float(VIEWPORT["height"] - 1)))
        logger.info("Clicking '%s' at (%.1f, %.1f)", label, cx, cy)
        await page.mouse.click(cx, cy)
        await asyncio.sleep(CLICK_DELAY_SECONDS)

    # TODO: click the captcha's Verify/Submit button and confirm success.
    return True


# ---------------------------------------------------------------------------
# Browser context factory
# ---------------------------------------------------------------------------
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
        # Tell sites this is a mobile device so they serve the mobile layout.
        is_mobile=True,
        has_touch=True,
    )
    await Stealth().apply_stealth_async(context)
    return browser, context


# ---------------------------------------------------------------------------
# Registration form flow
# ---------------------------------------------------------------------------
async def _try_click(page: Page, selectors: list, label: str) -> bool:
    """Attempt selectors in order; return True on first successful click."""
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
    """Attempt selectors in order; return True on first successful fill."""
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


async def _dump_buttons(page: Page) -> None:
    """Log every button/submit element so we can identify exact selectors."""
    logger.info("=== BUTTON DUMP ===")
    try:
        handles = await page.query_selector_all(
            "button, input[type='submit'], input[type='button'], a[role='button']"
        )
        if not handles:
            logger.info("  (none found)")
        for i, h in enumerate(handles):
            try:
                txt  = (await h.inner_text()).strip().replace("\n", " ")[:80]
                typ  = await h.get_attribute("type") or ""
                cls  = (await h.get_attribute("class") or "")[:80]
                vis  = await h.is_visible()
                logger.info("  [%d] vis=%-5s type=%-8s text=%r  class=%s",
                            i, vis, typ, txt, cls)
            except Exception:
                pass
    except Exception as e:
        logger.error("Button dump failed: %s", e)
    logger.info("=== END BUTTON DUMP ===")


async def _trigger_registration_form(page: Page) -> None:
    """
    Full registration sequence that forces the captcha to appear:

      Step 1 — Click the "Sign up" tab (switches from login to registration).
      Step 2 — Fill Email address field.
      Step 3 — Fill "Enter your new password" field.
      Step 4 — Fill "Confirm new password" field.
      Step 5 — Tick the "I've read and agreed..." checkbox.
               (Without this the Next button stays disabled and the captcha
                never appears.)
      Step 6 — Click the orange "Next" button.
    """
    WAIT = 2000   # ms between every step

    # ── Step 1 ── Sign up tab ────────────────────────────────────────────────
    await _try_click(page, [
        # Text-based (most reliable)
        "text=Sign up",
        "text=Create account",
        "text=Register",
        "text=注册",
        # Tab/link variants Xiaomi uses
        ".tab-item:has-text('Sign up')",
        ".tab-item:has-text('Register')",
        "[role='tab']:has-text('Sign up')",
        "[role='tab']:has-text('Register')",
        "a:has-text('Sign up')",
        "a:has-text('Register')",
        "a[href*='register']",
        "button:has-text('Sign up')",
        # Data attributes
        "[data-testid='signup-tab']",
        "[data-testid='register-tab']",
    ], "Sign up tab")
    await page.wait_for_timeout(WAIT)

    # ── Step 2 ── Email ──────────────────────────────────────────────────────
    await _try_fill(page, [
        "input[type='email']",
        "input[name='email']",
        "input[name='account']",
        "input[placeholder*='email' i]",
        "input[placeholder*='Email' i]",
        "input[placeholder*='mail' i]",
        "input[id*='email' i]",
        # Fallback: first visible text input that isn't a password field
        "input[type='text']:visible",
    ], "testuser_debug@example.com", "Email field")
    await page.wait_for_timeout(WAIT)

    # ── Step 3 ── New password ───────────────────────────────────────────────
    # On multi-password forms the FIRST password field is "Enter your new password"
    await _try_fill(page, [
        "input[name='password']",
        "input[name='newPassword']",
        "input[name='new_password']",
        "input[placeholder*='new password' i]",
        "input[placeholder*='Enter your new password' i]",
        "input[placeholder*='password' i]",
        "input[id*='password' i]",
        # If there are two fields, nth(0) is always the first
        "(//input[@type='password'])[1]",
        "input[type='password']",
    ], "DebugPass123!", "New password field")
    await page.wait_for_timeout(WAIT)

    # ── Step 4 ── Confirm password ───────────────────────────────────────────
    # The SECOND password field is "Confirm new password"
    await _try_fill(page, [
        "input[name='confirmPassword']",
        "input[name='confirm_password']",
        "input[name='passwordConfirm']",
        "input[placeholder*='confirm' i]",
        "input[placeholder*='Confirm new password' i]",
        # nth() picks the second password input when there are two
        "input[type='password'] >> nth=1",
        "(//input[@type='password'])[2]",
    ], "DebugPass123!", "Confirm password field")
    await page.wait_for_timeout(WAIT)

    # ── Debug screenshot after filling ──────────────────────────────────────
    await page.screenshot(path=DEBUG_FORM_PATH, full_page=True)
    logger.info("Saved post-fill screenshot → %s", DEBUG_FORM_PATH)

    # ── Step 5 ── Agreement checkbox ─────────────────────────────────────────
    # This is the small checkbox next to
    # "I've read and agreed to the Xiaomi Account User Agreement..."
    # Without ticking it, the Next button remains disabled.
    await _try_click(page, [
        # Checkbox by type
        "input[type='checkbox']",
        # Common class patterns for this checkbox on Xiaomi's page
        ".agreement-checkbox input",
        ".agree-checkbox input",
        ".policy-checkbox input",
        ".checkbox input",
        "label.agreement input",
        "label.agree input",
        # Text proximity selectors
        "label:has-text('agree') input",
        "label:has-text('Agreement') input",
        "label:has-text('User Agreement') input",
        "[class*='agree'] input[type='checkbox']",
        "[class*='policy'] input[type='checkbox']",
        # Fallback: any visible checkbox
        "input[type='checkbox']:visible",
    ], "Agreement checkbox")
    await page.wait_for_timeout(WAIT)

    # ── Button dump (debug) ──────────────────────────────────────────────────
    await _dump_buttons(page)

    # ── Step 6 ── Next / Submit button ───────────────────────────────────────
    await _try_click(page, [
        # Exact text matches
        "button:has-text('Next')",
        "button:has-text('Continue')",
        "button:has-text('Sign up')",
        "button:has-text('Register')",
        "button:has-text('Create')",
        "button:has-text('下一步')",
        "button:has-text('注册')",
        "button:has-text('确定')",
        "text=Next",
        "text=Continue",
        # Type-based
        "button[type='submit']",
        "input[type='submit']",
        # Xiaomi / NutUI CSS patterns
        ".n-footer .n-btn",
        ".n-footer button",
        ".submit-btn",
        ".btn-submit",
        ".btn-primary",
        ".next-btn",
        # Wildcard class attributes
        "[class*='submit']",
        "[class*='next']",
        "[class*='primary']",
        # ARIA / data
        "[aria-label*='next' i]",
        "[aria-label*='submit' i]",
        "[data-testid*='next' i]",
        "[data-testid*='submit' i]",
        # Absolute last resort: first visible button
        "button:visible",
    ], "Next / Submit button")


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------
async def run(headless: bool = True) -> None:
    async with async_playwright() as pw:
        browser, context = await launch_stealth_context(pw, headless=headless)
        page: Optional[Page] = None
        try:
            page = await context.new_page()

            logger.info("Opening Xiaomi login page…")
            await page.goto(XIAOMI_LOGIN_URL, wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle")
            logger.info("Landed on: %s", page.url)

            # Walk through the form to trigger the captcha.
            logger.info("Starting registration flow…")
            await _trigger_registration_form(page)

            # ── Wait 10 s AFTER clicking Next so the captcha anchor renders ──
            logger.info("Waiting 10 s for reCAPTCHA anchor to render…")
            await page.wait_for_timeout(10000)

            # Scroll to bottom in case the widget is below the fold.
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)

            # ── Dump raw iframe tags so we can see what loaded ─────────────
            html         = await page.content()
            iframe_tags  = re.findall(r"<iframe[^>]*>", html, re.IGNORECASE)
            logger.info("=== RAW <iframe> TAGS FOUND: %d ===", len(iframe_tags))
            for i, tag in enumerate(iframe_tags):
                logger.info("  [%d] %s", i, tag)
            if not iframe_tags:
                logger.info("  (none)")
            logger.info("=== END IFRAME DUMP ===")

            # ── Solve: click anchor → wait 3 s → crop bframe → YOLO ───────
            await solve_captcha(page)

            if not headless:
                logger.info("Press Ctrl+C to exit.")
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
