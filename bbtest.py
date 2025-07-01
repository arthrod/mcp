import asyncio
import logging
import os
import time

import uvloop
from browserbase import Browserbase
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def wait_for_user_auth(page, timeout_seconds=300) -> bool:
    """Wait for user to complete authentication with a timeout."""
    start_time = time.time()

    # Wait for user to complete auth (checking for NotebookLM home page)
    while time.time() - start_time < timeout_seconds:
        try:
            # Check if we're on the NotebookLM home page
            if "notebooklm.google.com" in page.url and "/notebook" not in page.url:
                await page.wait_for_selector('button[aria-label="Create new notebook"]', timeout=5000)
                return True
        except:
            pass
        await asyncio.sleep(2)

    return False


async def create_notebook_and_generate_lease() -> None:
    """Main function to create notebook and generate lease agreement."""
    # Initialize Browserbase
    bb = Browserbase(api_key=os.environ["BB_API_KEY"])

    # Create a session with keep alive for authentication
    session = bb.sessions.create(
        project_id=os.environ["BB_PROJECT_ID"],
        keep_alive=True,
        browser_settings={
            "viewport": {
                "width": 1920,
                "height": 1080,
            },
        },
    )

    logger.info(f"Created session: {session.id}")

    async with async_playwright() as playwright:
        # Connect to the browser
        browser = await playwright.chromium.connect_over_cdp(session.connect_url)
        context = browser.contexts[0]
        page = context.pages[0]

        try:
            # Navigate to NotebookLM
            logger.info("Navigating to NotebookLM...")
            await page.goto("https://notebooklm.google.com")

            # Wait for user authentication
            if not await wait_for_user_auth(page):
                msg = "Authentication failed or timed out"
                raise Exception(msg)

            # Create a new notebook
            logger.info("Creating new notebook...")
            create_button = await page.wait_for_selector('button[aria-label="Create new notebook"]')
            await create_button.click()

            # Wait for notebook creation dialog or page
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(2)

            # Check if we need to add a title or if we're already in the notebook
            try:
                # Try to find and fill the title input if it exists
                title_input = await page.wait_for_selector('input[placeholder*="title" i], input[aria-label*="title" i]', timeout=5000)
                await title_input.fill("Contracts")

                # Look for a create/continue button
                create_confirm = await page.wait_for_selector('button:has-text("Create"), button:has-text("Continue"), button:has-text("Done")', timeout=5000)
                await create_confirm.click()
            except:
                logger.info("No title dialog found, proceeding...")

            # Wait for the notebook to fully load
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(3)

            # Find the query/prompt input area
            logger.info("Looking for query input...")
            # NotebookLM might have different selectors for the query box
            query_selectors = [
                'textarea[placeholder*="Ask" i]',
                'textarea[placeholder*="What" i]',
                'textarea[placeholder*="Type" i]',
                'div[contenteditable="true"]',
                'input[type="text"][placeholder*="Ask" i]',
            ]

            query_input = None
            for selector in query_selectors:
                try:
                    query_input = await page.wait_for_selector(selector, timeout=5000)
                    if query_input:
                        logger.info(f"Found query input with selector: {selector}")
                        break
                except:
                    continue

            if not query_input:
                msg = "Could not find query input field"
                raise Exception(msg)

            # Enter the query
            logger.info("Entering lease agreement query...")
            await query_input.click()
            await query_input.fill("generate a lease agreement about a car")

            # Submit the query (try different methods)
            try:
                # Try pressing Enter
                await page.keyboard.press("Enter")
            except:
                # Try finding a submit button
                submit_button = await page.wait_for_selector('button[type="submit"], button:has-text("Send"), button:has-text("Generate")', timeout=5000)
                await submit_button.click()

            # Wait for response generation
            logger.info("Waiting for response generation...")
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(10)  # Give it time to generate

            # Look for the generated content
            logger.info("Looking for generated content...")
            # Try to find the response area
            response_selectors = [
                'div[role="article"]',
                "div.response-content",
                "div.generated-content",
                'div[data-message-role="assistant"]',
            ]

            response_content = None
            for selector in response_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        # Get the last element (most recent response)
                        response_content = elements[-1]
                        break
                except:
                    continue

            if not response_content:
                # Fallback: just get all text content
                logger.warning("Could not find specific response container, getting all text")
                response_content = await page.query_selector("body")

            # Extract the text content
            lease_text = await response_content.inner_text()
            logger.info(f"Generated lease agreement (preview): {lease_text[:200]}...")

            # Navigate to Google Docs
            logger.info("Creating new Google Doc...")
            await page.goto("https://docs.google.com/document/create")

            # Wait for the document to load
            await page.wait_for_selector("div.kix-appview-editor-container", timeout=30000)
            await asyncio.sleep(3)

            # Click on the document to focus
            editor = await page.wait_for_selector("div.kix-appview-editor")
            await editor.click()

            # Type the content
            logger.info("Inserting lease agreement into Google Doc...")
            await page.keyboard.type(lease_text)

            # Give the title
            await asyncio.sleep(2)
            title_input = await page.wait_for_selector("input.docs-title-input")
            await title_input.click()
            await title_input.fill("Car Lease Agreement - Generated from NotebookLM")

            logger.info("✅ Successfully created lease agreement in Google Docs!")

            # Keep session alive for user to review

            # Keep the session alive
            await asyncio.sleep(300)  # 5 minutes to review

        except Exception as e:
            logger.exception(f"Error occurred: {e!s}")
            raise
        finally:
            # Close the browser
            await browser.close()

            # End the session
            bb.sessions.update(session.id, status="REQUEST_RELEASE", project_id=os.environ["BB_PROJECT_ID"])
            logger.info("Session closed")


async def main() -> None:
    """Main entry point."""
    try:
        await create_notebook_and_generate_lease()
    except KeyboardInterrupt:
        pass
    except Exception:
        pass


if __name__ == "__main__":

    uvloop.run(main())
