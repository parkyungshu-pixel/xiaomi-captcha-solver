"""
Xiaomi login flow automation using Playwright with stealth mode.

This script launches a Chromium browser patched against common bot-detection
signals (navigator.webdriver, missing plugins, HeadlessChrome UA, etc.) via
the `playwright-stealth` package, then navigates to the Xiaomi login page.

The captcha is solved by running YOLOv8 object detection against a crop of
the pre-captcha screenshot (`check.png`) and clicking the center of every
target object in viewport coordinates.
"""

import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright_stealth import Stealth

# Xiaomi account login endpoint. `sid=passport` returns the generic landing
# page; change this if you need to log into a specific Xiaomi service.
XIAOMI_LOGIN_URL = "https://account.xiaomi.com/pass/serviceLogin?sid=passport"

# Realistic desktop user agent. Stealth will still override several fingerprint
# fields, but pinning the UA avoids the default Playwright `HeadlessChrome`
# marker that triggers instant blocks.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

VIEWPORT = {"width": 1366, "height": 768}
LOCALE = "en-US"
TIMEZONE_ID = "Asia/Manila"

# Path for the post-load screenshot (captured before the captcha appears),
# useful for debugging selectors and verifying the page rendered correctly.
SCREENSHOT_PATH = "check.png"

# ---- Captcha / YOLO config --------------------------------------------------

# CSS selector used to locate the reCAPTCHA image-challenge iframe. Its
# bounding box is preferred over the hardcoded CAPTCHA_CROP_BOX fallback
# because reCAPTCHA renders the challenge at slightly different coords
# depending on viewport / prompt length.
RECAPTCHA_CHALLENGE_SELECTORS = (
    'iframe[src*="recaptcha/api2/bframe"]',
    'iframe[src*="recaptcha/enterprise/bframe"]',
    'iframe[title*="recaptcha challenge"]',
)

# Fallback pixel region of `check.png` that contains the object-selection
# captcha, used only if the iframe bounding box above cannot be resolved.
# The box is (left, top, right, bottom) in the screenshot's coordinate space.
# NOTE: page.mouse.click() uses viewport-relative coordinates. Since check.png
# is captured with full_page=True, this crop must lie within the initial
# viewport (no scroll offset) for the click coordinates to line up correctly.
# A typical reCAPTCHA v2 image challenge is roughly 400x580 px.
CAPTCHA_CROP_BOX: Tuple[int, int, int, int] = (400, 180, 800, 760)

# Where to write the cropped captcha for YOLO inference / debugging.
CAPTCHA_CROP_PATH = "captcha_crop.png"

# YOLOv8 weights. `yolov8n.pt` (nano) is downloaded on first use by
# ultralytics and is fast enough for a single captcha frame.
YOLO_MODEL_PATH = "yolov8n.pt"

# COCO classes we care about for Xiaomi's object-selection captcha. Extend or
# shrink this set based on the prompt text on the actual captcha.
TARGET_CLASSES = {
    "bus",
    "traffic light",
    "car",
    "truck",
    "bicycle",
    "motorcycle",
    "fire hydrant",
    "stop sign",
}

# Minimum confidence for a detection to be clicked. Kept low (0.20) because
# reCAPTCHA tiles are small crops of real objects and YOLO often returns
# modest confidences on them.
YOLO_CONFIDENCE_THRESHOLD = 0.20

# Small delay between clicks so the interaction looks more human.
CLICK_DELAY_SECONDS = 0.4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("xiaomi-login")


@lru_cache(maxsize=1)
def _load_yolo_model():
    """Load and cache the YOLOv8 model. Imported lazily so the browser-only
    code path does not pay the torch import cost."""
    from ultralytics import YOLO  # heavy import, keep it lazy

    logger.info("Loading YOLO model from %s ...", YOLO_MODEL_PATH)
    return YOLO(YOLO_MODEL_PATH)


def _crop_captcha(
    screenshot_path: str, crop_box: Tuple[int, int, int, int], out_path: str
) -> str:
    """Crop the captcha region out of the pre-captcha screenshot and save
    it so YOLO can run inference on just the relevant pixels."""
    from PIL import Image  # lazy import

    if not Path(screenshot_path).exists():
        raise FileNotFoundError(
            f"Pre-captcha screenshot not found: {screenshot_path}. "
            "Make sure run() captured it before calling solve_captcha()."
        )

    with Image.open(screenshot_path) as img:
        crop = img.crop(crop_box)
        crop.save(out_path)
    logger.info("Cropped captcha region %s -> %s", crop_box, out_path)
    return out_path


def _detect_targets(
    crop_path: str,
    crop_origin: Tuple[int, int],
) -> List[Tuple[str, float, float, float]]:
    """Run YOLOv8 on the cropped image and return a list of
    (label, confidence, viewport_cx, viewport_cy) tuples for every detection
    whose class is in TARGET_CLASSES.

    The crop's (0, 0) corresponds to `crop_origin` in the original
    screenshot / viewport, so we translate each center back to viewport
    coordinates before returning.
    """
    model = _load_yolo_model()
    results = model.predict(
        source=crop_path,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    ox, oy = crop_origin
    detections: List[Tuple[str, float, float, float]] = []

    for result in results:
        names = result.names  # {class_id: class_name}
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            label = names.get(cls_idx, str(cls_idx))
            if label not in TARGET_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx_crop = (x1 + x2) / 2.0
            cy_crop = (y1 + y2) / 2.0
            detections.append((label, conf, cx_crop + ox, cy_crop + oy))

    # Sort by confidence (highest first) so if the captcha expects N clicks
    # in order, the strongest predictions go first.
    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


async def _locate_recaptcha_box(
    page: Page,
) -> Optional[Tuple[int, int, int, int]]:
    """Try to resolve the reCAPTCHA challenge iframe's bounding box and
    return it as (left, top, right, bottom) in viewport coordinates.

    Returns None if no matching iframe is visible - callers should fall
    back to the hardcoded CAPTCHA_CROP_BOX in that case.
    """
    for selector in RECAPTCHA_CHALLENGE_SELECTORS:
        locator = page.locator(selector)
        try:
            count = await locator.count()
        except Exception:
            continue
        if count == 0:
            continue
        try:
            box = await locator.first.bounding_box()
        except Exception:
            box = None
        if not box:
            continue
        left = int(box["x"])
        top = int(box["y"])
        right = int(box["x"] + box["width"])
        bottom = int(box["y"] + box["height"])
        logger.info(
            "Resolved reCAPTCHA challenge via %s -> (%d, %d, %d, %d)",
            selector,
            left,
            top,
            right,
            bottom,
        )
        return (left, top, right, bottom)

    return None


async def solve_captcha(page: Page) -> bool:
    """Solve Xiaomi's object-selection captcha using YOLOv8.

    Pipeline:
      1. Crop the captcha region out of `check.png`.
      2. Run YOLOv8 on the crop to detect objects (bus, traffic light, ...).
      3. Translate each detection's center back to viewport coordinates.
      4. Click every center via `page.mouse.click()` with a small delay.

    Returns True when at least one target was detected and clicked.
    """
    # Prefer the live iframe bounding box over CAPTCHA_CROP_BOX - this way
    # we stay aligned even if reCAPTCHA shifts the challenge around.
    crop_box = await _locate_recaptcha_box(page)
    if crop_box is None:
        logger.warning(
            "reCAPTCHA iframe not found; falling back to hardcoded "
            "CAPTCHA_CROP_BOX=%s.",
            CAPTCHA_CROP_BOX,
        )
        crop_box = CAPTCHA_CROP_BOX

    try:
        crop_path = _crop_captcha(
            SCREENSHOT_PATH, crop_box, CAPTCHA_CROP_PATH
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return False

    # Inference is CPU/GPU bound and synchronous - run it off the event loop.
    loop = asyncio.get_running_loop()
    crop_origin = (crop_box[0], crop_box[1])
    detections = await loop.run_in_executor(
        None, _detect_targets, crop_path, crop_origin
    )

    if not detections:
        logger.warning(
            "No target objects detected in captcha crop (classes=%s, conf>=%.2f).",
            sorted(TARGET_CLASSES),
            YOLO_CONFIDENCE_THRESHOLD,
        )
        return False

    logger.info("YOLO detected %d target object(s):", len(detections))
    for label, conf, cx, cy in detections:
        logger.info("  - %-15s conf=%.2f  center=(%.1f, %.1f)", label, conf, cx, cy)

    # Click each detection center. page.mouse.click() takes viewport-relative
    # coordinates, which is why _detect_targets translated the crop-local
    # centers back using CAPTCHA_CROP_BOX's origin.
    for label, _conf, cx, cy in detections:
        logger.info("Clicking '%s' at (%.1f, %.1f)", label, cx, cy)
        await page.mouse.click(cx, cy)
        await asyncio.sleep(CLICK_DELAY_SECONDS)

    # TODO: after clicking, locate and press the captcha's submit/confirm
    # button, then verify success (e.g. wait for navigation or error toast).
    return True


async def launch_stealth_context(
    playwright, headless: bool = False
) -> tuple[Browser, BrowserContext]:
    """Launch Chromium and build a stealth-enabled browser context."""
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
    )

    # Apply stealth patches (navigator.webdriver, chrome runtime, WebGL
    # vendor, permissions query, plugins array, etc.) to every page opened
    # in this context.
    await Stealth().apply_stealth_async(context)

    return browser, context


async def run(headless: bool = False) -> None:
    async with async_playwright() as playwright:
        browser, context = await launch_stealth_context(playwright, headless=headless)
        page: Optional[Page] = None
        try:
            page = await context.new_page()

            logger.info("Navigating to Xiaomi login page...")
            await page.goto(XIAOMI_LOGIN_URL, wait_until="domcontentloaded")

            # Give the SPA a moment to hydrate form fields / anti-bot scripts.
            await page.wait_for_load_state("networkidle")
            logger.info("Landed on: %s", page.url)

            # Extra wait so the reCAPTCHA tile images (the 3x3 / 4x4 grid)
            # finish loading. networkidle alone is not enough because tiles
            # are lazily requested after the iframe renders.
            logger.info("Waiting 5s for reCAPTCHA tiles to finish loading...")
            await page.wait_for_timeout(5000)

            # Capture a full-page screenshot after load but before the captcha
            # is triggered - handy for inspecting layout and confirming that
            # stealth patches kept the page from redirecting to a block page.
            await page.screenshot(path=SCREENSHOT_PATH, full_page=True)
            logger.info("Saved pre-captcha screenshot to %s", SCREENSHOT_PATH)

            # TODO: fill in credentials
            #   await page.fill("input[name='account']", USERNAME)
            #   await page.fill("input[name='password']", PASSWORD)
            #   await page.click("button#login-button")

            # TODO: after clicking login, the captcha modal may appear.
            # Hand it off to the AI solver:
            await solve_captcha(page)

            # Keep the window open for manual inspection when running headed.
            if not headless:
                logger.info("Press Ctrl+C to exit.")
                await asyncio.Event().wait()
        finally:
            if page is not None:
                await page.close()
            await context.close()
            await browser.close()


def main() -> None:
    # headless=True so this also works in environments without a display
    # server (e.g. GitHub Codespaces, CI). Flip to False locally if you want
    # to watch the browser and keep the window open via the Ctrl+C loop.
    try:
        asyncio.run(run(headless=True))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
