"""
Xiaomi login flow automation using Playwright with stealth mode.

This script launches a Chromium browser patched against common bot-detection
signals (navigator.webdriver, missing plugins, HeadlessChrome UA, etc.) via
the `playwright-stealth` package, then navigates to the Xiaomi login page.

The captcha slot is left as a placeholder (see `solve_captcha`) so the AI
image solver can be plugged in later without touching the browser setup.
"""

import asyncio
import logging
from typing import Optional

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("xiaomi-login")


async def solve_captcha(page: Page) -> bool:
    """
    Placeholder for the AI-based captcha solver.

    TODO: Integrate the AI image solver here.
      1. Detect whether a captcha widget appeared on `page`
         (e.g. iframe or canvas with a known selector).
      2. Capture the challenge image (page.locator(...).screenshot()).
      3. Send it to the AI solver (local model or hosted API) and
         receive the target coordinates / slider offset / text.
      4. Replay the solution using page.mouse / page.keyboard so the
         interaction looks human (bezier path, jitter, small pauses).
      5. Return True on success, False otherwise so the caller can retry.
    """
    logger.info("Captcha solver not wired up yet - skipping.")
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
    try:
        asyncio.run(run(headless=False))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
