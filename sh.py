# File 1: stagehand_tool.py
"""Simple Stagehand tool for PydanticAI using stagehand-py library."""
import asyncio
import os
import signal
import sys
from typing import Any

import requests
import uvloop
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from stagehand import AgentConfig
from stagehand.schemas import AgentExecuteOptions, AgentProvider
INSTRUCTIONS = """

# AI Automation Instructions: Google Sheets to ChatGPT Processing

## Core Workflow
Please assess the current screen and execute the following sequence:

### Step 1: Navigate to Google Sheets
- If not already in Google Sheets, navigate to the target spreadsheet
- Open the document entitled "prompts_to_be_processed"
- Locate the worksheet with the "Prompts" and "Results" columns

### Step 2: Identify Next Prompt to Process
- Scan the "Prompts" column from top to bottom
- Find the first row where:
  - The "Prompts" column contains text/content
  - The corresponding "Results" column is empty or blank
- Select and copy the prompt text from this identified row
- Remember the row number for later result placement

### Step 3: Navigate to ChatGPT
- Open a new tab or navigate to ChatGPT (chat.openai.com)
- If not logged in, complete authentication
- Start a new conversation or use existing chat interface

### Step 4: Process the Prompt
- Paste the copied prompt into ChatGPT's input field
- Submit the prompt and wait for the complete response
- Once the response is fully generated, select and copy the entire result

### Step 5: Return Results to Google Sheets
- Navigate back to the Google Sheets tab
- Locate the same row where the original prompt was found
- Click on the corresponding cell in the "Results" column
- Paste the ChatGPT response into this cell
- Ensure the result is properly formatted and saved

### Step 6: Continuation Logic
- After successfully completing one prompt-result cycle, automatically proceed to the next empty result row
- Repeat the entire process until all prompts have corresponding results
- If all prompts are processed, indicate completion

## Error Handling
- If ChatGPT is unavailable, note the error in the Results column
- If a prompt generates an error, paste the error message as the result
- If navigation fails, attempt to refresh and retry once before reporting failure
- Always maintain the row correspondence between prompts and results

## Quality Assurance
- Verify that each result is placed in the correct row
- Ensure complete responses are captured (don't truncate long outputs)
- Maintain original formatting when possible
- Double-check that no prompts are skipped or duplicated

## Completion Criteria
The task is complete when every non-empty cell in the "Prompts" column has a corresponding entry in the "Results" column.
"""


"Please assess the screen. If not in notebooklm, first go to notebooklm . If notebookl.m and there is a response to a question, copy the result of the conversation and paste into a new doc into docs.google.com. If in any other screen, attempt to go to notebooklm, create a mock lease agreement of a car as source. add the source. Then using the chat, make the question: whats the details of this agreement? then whatever it generates, even if garbage, copy it carefully, go to docs.google.com, create a new document and paste the content of the response there."

class StagehandTool:
    """Drop-in replacement - makes endpoint calls to Node.js bridge."""

    def __init__(self) -> None:
        self.base_url = "http://localhost:3001"
        self.session_id: str | None = None
        # Removed server process management per request

    def _request(self, method: str, endpoint: str, data: dict | None = None) -> dict[str, Any]:
        """Make request to bridge server."""
        try:
            if method == "GET":
                response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
            else:
                response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=30)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def initialize(self) -> dict[str, Any]:
        """Initialize session."""
        result = self._request("POST", "/init")
        if result.get("success"):
            self.session_id = "bridge_session"
        return result

    async def navigate(self, url: str) -> dict[str, Any]:
        """Navigate to URL."""
        return self._request("POST", "/navigate", {"url": url})

    async def act(self, action: str, dom_settle_timeout_ms: int = 1000) -> dict[str, Any]:
        """Perform action."""
        return self._request("POST", "/act", {"action": action})

    async def observe(self, instruction: str, draw_overlay: bool = True) -> dict[str, Any]:
        """Observe page."""
        return self._request("POST", "/observe", {"instruction": instruction})

    async def extract(self, instruction: str, schema_model) -> dict[str, Any]:
        """Extract data."""
        schema = schema_model.model_json_schema() if hasattr(schema_model, "model_json_schema") else schema_model
        return self._request("POST", "/extract", {"instruction": instruction, "schema": schema})

    async def paste_into_page(self, content: str) -> dict[str, Any]:
        """Paste content - uses agent for complex task."""
        return self._request("POST", "/agent", {
            "instructions": f"Type or enter the following text directly: {content} and then click add, continue, save or submit (the appropriate button to effectively submit the content). DO NOT try to copy/paste - instead TYPE the text directly into the field.",
            "maxSteps": 5,
        })

    async def extract_dom(self, instruction: str | None = None, selectors: list | None = None, use_iframes: bool | None = None) -> dict[str, Any]:
        """Extract content using enhanced Stagehand extraction with iframe support."""
        payload = {}
        if instruction:
            payload["instruction"] = instruction
        if selectors:
            payload["selectors"] = selectors
        if use_iframes is not None:
            payload["useIframes"] = use_iframes
        return self._request("POST", "/extract-dom", payload)

    async def verify_visual(self, description: str) -> dict[str, Any]:
        """Visual verification for canvas-based content like Google Docs."""
        return self._request("POST", "/verify-visual", {"description": description})

    async def paste_clipboard(self, content: str, target_description: str = "document content area") -> dict[str, Any]:
        """Paste content using proper clipboard and keyboard methods."""
        return self._request("POST", "/paste-clipboard", {"content": content, "target_description": target_description})

    async def close(self) -> dict[str, Any]:
        """Close session."""
        return self._request("POST", "/close")

    # Simple wrapper properties for backward compatibility
    @property
    def stagehand(self):
        """Stagehand wrapper."""
        return self

    @property
    def page(self):
        """Page wrapper."""
        class PageWrapper:
            def __init__(self, tool) -> None:
                self.tool = tool

            async def goto(self, url: str) -> None:
                result = await self.tool.navigate(url)
                if not result.get("success"):
                    raise Exception(result.get("error", "Navigation failed"))

            async def act(self, action: str) -> dict[str, Any]:
                """Perform action using the tool's act method."""
                return await self.tool.act(action)

            @property
            def url(self):
                result = self.tool._request("GET", "/url")
                return result.get("url", "unknown")

        return PageWrapper(self)


# from shtool import StagehandTool
# Configure loguru logger
logger.remove()  # Remove default handler
logger.add("automation.log", rotation="10 MB", level="DEBUG", format="{time} | {level} | {function} | {message}")
logger.add(lambda msg: print(msg, end=""), level="INFO", format="🔧 {function}: {message}")  # Console output

model = OpenAIModel("gpt-4.1")


# Global console for Rich formatting
console = Console()


async def wait_for_user_input_with_timer(message: str = "Waiting for user to complete manual action...") -> str:
    """Wait for user input with a 300s timeout and Rich formatting."""
    console.print(Panel(
        Text(message, style="bold yellow"),
        title="🔄 User Action Required",
        border_style="yellow",
    ))

    console.print("[bold green]Press Enter when you've completed the required action (300s timeout)...[/bold green]")

    try:
        # Use asyncio loop in executor for input with timeout
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: input("Press Enter to continue: ")),
            timeout=300.0,
        )
        console.print("✅ [bold green]User input received, continuing automation...[/bold green]")
        return f"✅ User provide the following feedback: {message} (it can be None, which means you should attempt to complete the task)"
    except TimeoutError:
        console.print("⏰ [bold red]Timeout reached (300s), continuing automation...[/bold red]")
        return f"⏰ Timeout: {message}"
    except (EOFError, KeyboardInterrupt):
        console.print("🚫 [bold red]Input interrupted, continuing automation...[/bold red]")
        return f"🚫 Input interrupted: {message}"


class TaskStatus(BaseModel):
    """Status of task execution."""
    completed: bool = Field(default=False, description="Is the task completed?")
    double_checked: bool = Field(default=False, description="Did you double check the completion?")
    needs_stagehand: bool = Field(default=False, description="Do you need to use Stagehand for next action?")
    next_action: str | None = Field(default=None, description="Next action to take if not completed")  # Added default=None
    reasoning: str = Field(default="", description="Explanation of current status")
    extracted_data: dict[str, Any] | None = Field(default=None, description="Any data extracted during execution")  # Added default=None


class AgentResult(BaseModel):
    """Result from Stagehand agent execution."""
    success: bool = Field(description="Was the action successful?")
    completed: bool = Field(description="Is the overall task completed?")
    message: str | None = Field(description="Result message")
    actions: list[str] = Field(default_factory=list, description="Actions performed")
    extracted_data: dict[str, Any] | None = Field(description="Data extracted during execution")


class AgentDependencies(BaseModel):
    """Dependencies for the agent."""
    stagehand_tool: Any  # Forward reference to StagehandTool
    instructions: str
    current_url: str | None = None
    execution_history: list[str] = Field(default_factory=list)


# Data extraction schemas
class PageData(BaseModel):
    """Schema for basic page data extraction."""
    title: str = Field(description="Page title")
    content: str = Field(description="Main content or description")


class PydanticAIWebAutomationAgent:
    """PydanticAI agent for web automation with Stagehand."""

    def __init__(self) -> None:
        self.agent = Agent(
            model,
            deps_type=AgentDependencies,
            result_type=TaskStatus,
            system_prompt=self._get_system_prompt(),
        )

        # Setup tools directly
        @self.agent.tool
        async def navigate_to_url(ctx: RunContext[AgentDependencies], url: str) -> str:
            """Navigate to a specific URL using Stagehand page.goto."""
            logger.info(f"🧭 TOOL CALLED: navigate_to_url with url='{url}'")
            try:
                if not ctx.deps.stagehand_tool.stagehand:
                    logger.error("❌ Stagehand not initialized")
                    return "❌ Stagehand not initialized"

                # Use Stagehand page.goto directly
                logger.debug(f"Using stagehand.page.goto({url})")
                await ctx.deps.stagehand_tool.stagehand.page.goto(url)
                ctx.deps.current_url = url
                ctx.deps.execution_history.append(f"Navigated to {url}")
                logger.success(f"✅ Successfully navigated to {url}")
                return f"✅ Successfully navigated to {url}"
            except Exception as e:
                logger.error(f"❌ Navigation failed: {e}")
                return f"❌ Failed to navigate: {e!s}"

        @self.agent.tool
        async def perform_action(ctx: RunContext[AgentDependencies], action: str) -> str:
            """Perform a web action using Stagehand AI with fallback mechanism."""
            logger.info(f"🎬 TOOL CALLED: perform_action with action='{action}'")

            # First attempt: Use Stagehand's page.act
            logger.debug(f"Attempting page.act with action: {action}")
            result = await ctx.deps.stagehand_tool.act(action)
            if result["success"]:
                ctx.deps.execution_history.append(f"Performed action: {action}")
                logger.success(f"✅ Action completed: {action}")
                return f"✅ Action completed: {action}"

            # If page.act failed, try alternative approach using Stagehand agent.execute
            logger.warning(f"⚠️ page.act failed: {result['error']}")
            console.print(f"⚠️ [yellow]page.act failed: {result['error']}[/yellow]")
            console.print("🔄 [blue]Attempting alternative approach with Stagehand agent.execute...[/blue]")

            try:
                # Use Stagehand's agent.execute as fallback
                if ctx.deps.stagehand_tool.stagehand:
                    logger.debug(f"Attempting Stagehand agent.execute fallback with action: {action}")

                    # Create agent with our context and instructions
                    agent_config = AgentConfig(
                        model="gpt-4.1",
                        provider=AgentProvider.OPENAI,
                        instructions=f"""
                        You are helping with browser automation.
                        Current task: {ctx.deps.instructions}
                        Failed action that needs alternative approach: {action}
                        Error encountered: {result['error']}

                        Please find an alternative way to accomplish this action.
                        """,
                        options={"apiKey": os.getenv("MODEL_API_KEY")},
                    )

                    # Execute with Stagehand agent
                    execute_options = AgentExecuteOptions(
                        instructions=action,
                        config=agent_config,
                    )

                    agent_result = await ctx.deps.stagehand_tool.stagehand.agent.execute(execute_options)

                    # Log and display the agent result with success/completion distinction
                    logger.info("🤖 STAGEHAND AGENT RESULT:")
                    logger.info(f"✅ Success: {agent_result.success}")
                    logger.info(f"🎯 Completed: {agent_result.completed}")
                    logger.info(f"💬 Message: {agent_result.message}")
                    logger.info(f"🔄 Actions: {len(agent_result.actions) if agent_result.actions else 0}")

                    console.print(Panel(
                        f"✅ Success: [bold]{'Yes' if agent_result.success else 'No'}[/]\n"
                        f"🎯 Completed: [bold]{'Yes' if agent_result.completed else 'No'}[/]\n"
                        f"💬 Message: [italic]{agent_result.message or 'None'}[/]\n"
                        f"🔄 Actions performed: [bold]{len(agent_result.actions) if agent_result.actions else 0}[/]",
                        title="🤖 Stagehand Agent Result",
                        border_style="blue",
                    ))

                    if agent_result.success:
                        ctx.deps.execution_history.append(f"Fallback action successful with Stagehand agent: {action}")
                        logger.success(f"✅ Stagehand agent fallback successful: {action}")
                        return f"✅ Stagehand agent alternative approach successful: {agent_result.message or action}"
                    logger.warning(f"⚠️ Stagehand agent execution was not successful: {agent_result.message}")
                    return f"⚠️ Stagehand agent attempted but not successful: {agent_result.message or 'No details'}"
                logger.error("❌ Stagehand not initialized for fallback")
                return "❌ Stagehand not initialized for fallback attempt"

            except Exception as fallback_error:
                logger.error(f"❌ Both action attempts failed. Original: {result['error']}, Fallback: {fallback_error}")
                ctx.deps.execution_history.append(f"Action failed (both attempts): {action}")
                return f"❌ Action failed: {result['error']} | Stagehand agent fallback also failed: {fallback_error!s}"

        @self.agent.tool
        async def extract_dom_content(ctx: RunContext[AgentDependencies], instruction: str | None = None, selectors: list | None = None) -> str:
            """Extract content using enhanced Stagehand extraction with iframe support - USE THIS FIRST before extract_page_data.

            Args:
                instruction: Natural language instruction for what to extract (preferred method)
                selectors: List of simple selector strings as fallback like ["p", ".chat-response", "xpath=/html/body/div"]
                          DO NOT pass objects - only simple strings!
            """
            logger.info(f"🔍 TOOL CALLED: extract_dom_content with instruction='{instruction}' selectors='{selectors}'")

            # Prefer using instruction for intelligent extraction
            if not instruction and not selectors:
                instruction = "Extract the main response or chat content from the current page"

            logger.debug(f"Enhanced extraction with instruction: {instruction}, selectors: {selectors}")

            # Add common NotebookLM selectors based on the logs
            default_selectors = [
                "xpath=/html[1]/body[1]/labs-tailwind-root[1]/div[1]/notebook[1]/mat-tab-group[1]/div[1]/mat-tab-body[2]/div[1]/div[1]/chat-panel[1]/div[1]/div[1]/span[3]/div[1]/p[1]",
                "xpath=/html[1]/body[1]/labs-tailwind-root[1]/div[1]/notebook[1]/mat-tab-group[1]/div[1]/mat-tab-body[2]/div[1]/div[1]/chat-panel[1]/div[1]/div[1]/span[3]/div[1]",
                "chat-panel", ".response-text", "p", "[data-testid='response']", "span",
            ]

            # Fix selector format if needed
            if selectors:
                fixed_selectors = []
                for sel in selectors:
                    if isinstance(sel, str):
                        fixed_selectors.append(sel)
                    elif isinstance(sel, dict) and "value" in sel:
                        # Handle malformed selector objects
                        fixed_selectors.append(sel["value"])
                        logger.warning(f"⚠️ Fixed malformed selector object: {sel}")
                    else:
                        logger.warning(f"⚠️ Skipping invalid selector: {sel}")
                selectors_to_use = fixed_selectors or default_selectors
            else:
                selectors_to_use = default_selectors
            logger.debug(f"Using selectors: {selectors_to_use}")

            # Use enhanced extraction with automatic iframe detection
            result = await ctx.deps.stagehand_tool.extract_dom(
                instruction=instruction,
                selectors=selectors_to_use,
                use_iframes=None  # Auto-detect
            )
            logger.debug(f"Raw DOM extraction result: {result}")

            if result["success"] and result.get("data"):
                data = result["data"]
                extracted_text = data.get("combinedText", "") or data.get("content", "")
                text_length = len(extracted_text) if extracted_text else 0
                method = data.get("method", "unknown")
                used_iframes = data.get("usedIframes", False)

                ctx.deps.execution_history.append(f"DOM extracted data: {text_length} characters via {method}")
                logger.success(f"✅ DOM extracted successfully using method: {method} (iframes: {used_iframes})")
                logger.success(f"📊 Extracted {text_length} characters")
                logger.info(f"📄 Extracted text preview: {extracted_text[:200]}...")

                if text_length > 0:
                    return f"✅ DOM extracted data ({text_length} chars, method: {method}): {extracted_text}"
                logger.warning("⚠️ DOM extraction succeeded but got empty text")
                return "⚠️ DOM extraction succeeded but found empty text"

            logger.warning("⚠️ DOM extraction had no results")
            logger.debug(f"Full result: {result}")
            return f"⚠️ DOM extraction found no content: {result.get('message', 'No data found')}"

        @self.agent.tool
        async def extract_iframe_content(ctx: RunContext[AgentDependencies], instruction: str) -> str:
            """Extract content from iframe-enabled sites using Stagehand's experimental iframe support - USE FOR NOTEBOOKLM."""
            logger.info(f"🖼️ TOOL CALLED: extract_iframe_content with instruction='{instruction}'")
            logger.debug(f"Using iframe-enabled extraction with instruction: {instruction}")

            # Force iframe usage for complex sites like NotebookLM
            result = await ctx.deps.stagehand_tool.extract_dom(
                instruction=instruction,
                selectors=None,  # Let AI handle selector discovery
                use_iframes=True  # Force iframe support
            )
            logger.debug(f"Raw iframe extraction result: {result}")

            if result["success"] and result.get("data"):
                data = result["data"]
                extracted_text = data.get("combinedText", "") or data.get("content", "")
                text_length = len(extracted_text) if extracted_text else 0
                method = data.get("method", "iframe_extraction")
                used_iframes = data.get("usedIframes", True)

                ctx.deps.execution_history.append(f"Iframe extracted data: {text_length} characters via {method}")
                logger.success(f"✅ Iframe extraction successful using method: {method} (iframes: {used_iframes})")
                logger.success(f"📊 Extracted {text_length} characters")
                logger.info(f"📄 Extracted text preview: {extracted_text[:200]}...")

                if text_length > 0:
                    return f"✅ Iframe extracted data ({text_length} chars, method: {method}): {extracted_text}"
                logger.warning("⚠️ Iframe extraction succeeded but got empty text")
                return "⚠️ Iframe extraction succeeded but found empty text"

            logger.warning("⚠️ Iframe extraction had no results")
            logger.debug(f"Full result: {result}")
            return f"⚠️ Iframe extraction found no content: {result.get('message', 'No data found')}"

        @self.agent.tool
        async def extract_page_data(ctx: RunContext[AgentDependencies], instruction: str) -> str:
            """Extract basic page data (title and content) - FALLBACK method, use extract_dom_content or extract_iframe_content first."""
            logger.info(f"📊 TOOL CALLED: extract_page_data with instruction='{instruction}'")
            logger.warning("⚠️ Using fallback extraction method - try extract_dom_content or extract_iframe_content first")
            result = await ctx.deps.stagehand_tool.extract(instruction, PageData)
            if result["success"]:
                data = result["data"]
                ctx.deps.execution_history.append(f"Extracted data: {instruction}")
                logger.success(f"✅ Extracted data: {data}")
                return f"✅ Extracted data: {data}"
            logger.error(f"❌ Extraction failed: {result['error']}")
            return f"❌ Extraction failed: {result['error']}"

        @self.agent.tool
        async def observe_page(ctx: RunContext[AgentDependencies], instruction: str) -> str:
            """Observe and get information about page elements."""
            logger.info(f"👁️ TOOL CALLED: observe_page with instruction='{instruction}'")
            result = await ctx.deps.stagehand_tool.observe(instruction)
            if result["success"]:
                observations = result["data"]
                ctx.deps.execution_history.append(f"Observed: {instruction}")
                logger.success(f"✅ Observations: {observations}")

                # Check for iframe warnings
                if isinstance(observations, list):
                    iframe_count = sum(1 for obs in observations if "iframe" in str(obs).lower())
                    if iframe_count > 0:
                        logger.warning(f"⚠️ Warning: found {iframe_count} iframe(s) on the page. If you wish to interact with iframe content, please make sure you are setting iframes: true")

                return f"✅ Observations: {observations}"
            logger.error(f"❌ Observation failed: {result['error']}")
            return f"❌ Observation failed: {result['error']}"

        @self.agent.tool
        async def wait_for_user_input(ctx: RunContext[AgentDependencies], message: str = "Waiting for user to complete manual action...") -> str:
            """Wait for user to complete manual actions like login and provide input."""
            logger.info(f"⏰ TOOL CALLED: wait_for_user_input with message='{message}'")
            ctx.deps.execution_history.append(f"Waiting for user: {message}")

            # Use the new Rich-formatted input function
            logger.debug("Starting 300s timer for user input")
            result_message = await wait_for_user_input_with_timer(message)

            ctx.deps.execution_history.append("User input received, continuing automation")
            logger.success(f"✅ User input completed: {result_message}")
            return result_message

        @self.agent.tool
        async def switch_to_frame(ctx: RunContext[AgentDependencies], frame_selector: str = "iframe") -> str:
            """Switch to iframe or frame context for interaction."""
            logger.info(f"🖼️ TOOL CALLED: switch_to_frame with frame_selector='{frame_selector}'")
            try:
                if not ctx.deps.stagehand_tool.stagehand:
                    logger.error("❌ Stagehand not initialized")
                    return "❌ Stagehand not initialized"

                # Get frame handle and switch to it
                frame = await ctx.deps.stagehand_tool.stagehand.page.query_selector(frame_selector)
                if frame:
                    frame_element = await frame.content_frame()
                    if frame_element:
                        ctx.deps.execution_history.append(f"Switched to frame: {frame_selector}")
                        logger.success(f"✅ Successfully switched to frame: {frame_selector}")
                        return f"✅ Successfully switched to frame: {frame_selector}"
                    logger.error(f"❌ Frame content not accessible: {frame_selector}")
                    return f"❌ Frame content not accessible: {frame_selector}"
                logger.error(f"❌ Frame not found: {frame_selector}")
                return f"❌ Frame not found: {frame_selector}"
            except Exception as e:
                logger.error(f"❌ Failed to switch to frame: {e}")
                return f"❌ Failed to switch to frame: {e!s}"

        @self.agent.tool
        async def validate_chat_response(ctx: RunContext[AgentDependencies]) -> str:
            """Validate if there's a chat response present before trying to extract it."""
            logger.info("🔍 TOOL CALLED: validate_chat_response")

            # Check for indicators that a response exists
            validation_selectors = [
                "xpath=/html[1]/body[1]/labs-tailwind-root[1]/div[1]/notebook[1]/mat-tab-group[1]/div[1]/mat-tab-body[2]/div[1]/div[1]/chat-panel[1]/div[1]/div[1]/span[3]",
                "chat-panel p", "chat-panel span", ".response", "[data-testid='chat-message']",
            ]

            result = await ctx.deps.stagehand_tool.extract_dom(
                instruction="Check if there is a chat response or conversation content on the page",
                selectors=validation_selectors,
                use_iframes=None  # Auto-detect
            )

            if result["success"] and result.get("data"):
                data = result["data"]
                text = data.get("combinedText", "")

                # Check for empty state messages
                empty_indicators = ["Add a source to get started", "0 sources", "No sources", "Upload a source"]
                if any(indicator in text for indicator in empty_indicators):
                    logger.warning("⚠️ Chat shows empty state - no sources or responses yet")
                    return f"⚠️ No chat response yet. Found: {text[:100]}"

                # Check for loading/generating indicators
                loading_indicators = ["Generating", "Loading", "Please wait", "...", "Thinking"]
                if any(indicator in text for indicator in loading_indicators):
                    logger.info("⏳ Response is being generated, wait longer")
                    return f"⏳ Response is being generated: {text[:100]}"

                if text and len(text.strip()) > 10:  # Reasonable content length
                    logger.success(f"✅ Chat response detected: {len(text)} characters")
                    return f"✅ Chat response is present ({len(text)} chars)"
                logger.warning("⚠️ Chat area exists but appears empty")
                return "⚠️ Chat area found but no substantial content"

            logger.warning("⚠️ No chat response area found")
            return "⚠️ No chat response area detected"

        @self.agent.tool
        async def submit_chat_question(ctx: RunContext[AgentDependencies], question: str) -> str:
            """Submit a question in NotebookLM chat and wait for response."""
            logger.info(f"💬 TOOL CALLED: submit_chat_question with question='{question}'")

            try:
                # Step 1: Type the question
                logger.debug("Step 1: Typing question into chat field")
                await ctx.deps.stagehand_tool.stagehand.page.act(f"Type '{question}' into the chat input field")

                # Step 2: Click the send button (arrow icon)
                logger.debug("Step 2: Looking for and clicking send button")
                await ctx.deps.stagehand_tool.stagehand.page.act("Click the send button or arrow icon to submit the chat message")

                ctx.deps.execution_history.append(f"Submitted chat question: {question}")
                logger.success(f"✅ Chat question submitted: {question}")

                # Step 3: Wait a moment for response to start generating
                import asyncio
                await asyncio.sleep(3)

                return f"✅ Question submitted: {question}. Use validate_chat_response to check when ready."

            except Exception as e:
                logger.error(f"❌ Failed to submit chat question: {e}")
                return f"❌ Failed to submit question: {e!s}"

        @self.agent.tool
        async def paste_content_clipboard(ctx: RunContext[AgentDependencies], content: str, target: str = "Google Docs document") -> str:
            """Paste content using proper clipboard and keyboard methods - USE FOR GOOGLE DOCS PASTING."""
            logger.info(f"📋 TOOL CALLED: paste_content_clipboard with {len(content)} characters to {target}")
            logger.debug(f"Content preview: {content[:100]}...")

            result = await ctx.deps.stagehand_tool.paste_clipboard(content, target)

            if result["success"]:
                method = result.get("method", "unknown")
                modifier = result.get("modifier_used", "unknown")
                ctx.deps.execution_history.append(f"Clipboard paste: {len(content)} chars via {method}")
                logger.success(f"✅ Clipboard paste successful using {modifier}+V method")
                return f"✅ Content pasted successfully using {method} ({modifier}+V). Content length: {len(content)} characters"

            logger.error(f"❌ Clipboard paste failed: {result.get('error', 'Unknown error')}")
            return f"❌ Clipboard paste failed: {result.get('error', 'Unknown error')}"

        @self.agent.tool
        async def verify_visual_content(ctx: RunContext[AgentDependencies], description: str) -> str:
            """Visual verification for canvas-based content like Google Docs - USE FOR GOOGLE DOCS VERIFICATION."""
            logger.info(f"📸 TOOL CALLED: verify_visual_content with description='{description}'")
            logger.debug("Starting visual verification process")

            result = await ctx.deps.stagehand_tool.verify_visual(description)

            if result["success"]:
                data = result["data"]
                verified = data.get("verified", False)
                analysis = data.get("analysis", "")

                ctx.deps.execution_history.append(f"Visual verification: {description}")

                if verified:
                    logger.success(f"✅ Visual verification PASSED: {analysis}")
                    return f"✅ Visual verification PASSED: Content is visible on screen. {analysis}"
                logger.warning(f"⚠️ Visual verification FAILED: {analysis}")
                return f"⚠️ Visual verification FAILED: Content not visible. {analysis}"

            logger.error(f"❌ Visual verification error: {result.get('error', 'Unknown error')}")
            return f"❌ Visual verification failed: {result.get('error', 'Unknown error')}"

        @self.agent.tool
        async def get_page_url(ctx: RunContext[AgentDependencies]) -> str:
            """Get the current page URL."""
            logger.info("🌐 TOOL CALLED: get_page_url")
            try:
                if not ctx.deps.stagehand_tool.stagehand:
                    logger.error("❌ Stagehand not initialized")
                    return "❌ Stagehand not initialized"

                url = ctx.deps.stagehand_tool.stagehand.page.url
                ctx.deps.current_url = url
                logger.success(f"✅ Current URL: {url}")
                return f"✅ Current URL: {url}"
            except Exception as e:
                logger.error(f"❌ Failed to get URL: {e}")
                return f"❌ Failed to get URL: {e!s}"

    def _get_system_prompt(self) -> str:
        return """
        You are a web automation agent that MUST take action to complete tasks.

        CRITICAL: You MUST actually DO the task, not just describe it. If the details are empty or missing, that means you haven't done the work yet!

        MANDATORY REQUIREMENTS:
        1. NEVER mark a task as completed if the actual data/content is empty or missing
        2. You MUST actually type text, click buttons, and interact with the page
        3. If you see empty results, that means you need to DO MORE WORK, not continue
        4. FOLLOW THE USER'S INSTRUCTIONS EXACTLY - do not modify or interpret them
        5. The instructions will be provided in the prompt - execute them precisely as written

        Available tools (USE THEM):
        - navigate_to_url: Go to a specific webpage using Stagehand page.goto
        - perform_action: Use AI to perform actions (click, type, etc.) - USE THIS TO ACTUALLY TYPE TEXT
        - submit_chat_question: **USE FOR NOTEBOOKLM CHAT** - Submit question and click send button
        - validate_chat_response: **USE BEFORE EXTRACTION** - Check if chat response exists/ready
        - extract_dom_content: **USE THIS FIRST** - Enhanced DOM content extraction with iframe support
          Parameters:
          * instruction: (recommended) Natural language instruction for what to extract
          * selectors: (fallback) List of CSS/XPath selectors as strings
          * Auto-detects iframes and uses proper Stagehand extraction methods
        - extract_page_data: Extract structured data from pages (FALLBACK - only if extract_dom_content fails)
        - paste_content_clipboard: **USE FOR GOOGLE DOCS PASTING** - Proper clipboard+keyboard paste method
          Parameters:
          * content: Text content to paste
          * target: Description of target area (e.g., "Google Docs document")
          * Uses cross-platform keyboard shortcuts (Meta+V on Mac, Ctrl+V on Windows/Linux)
        - verify_visual_content: **USE FOR GOOGLE DOCS** - Visual verification for canvas-based content
          Parameters:
          * description: What content to look for on the page
          * Uses AI screenshot analysis for canvas-rendered content like Google Docs
        - observe_page: Get information about page elements
        - wait_for_user_input: Use for authentication/login pages
        - switch_to_frame: Switch to iframe context when dealing with embedded content
        - get_page_url: Get current page URL
        - paste_into_page: Paste text into a page element (CRITICAL: See NotebookLM instructions below)

        CRITICAL RULES FOR IFRAME HANDLING:
        - extract_dom_content now auto-detects iframes and uses proper Stagehand extraction
        - If a site uses iframes, the tool will automatically set iframes: true
        - Iframe detection looks for <iframe> elements and adjusts extraction method
        - For complex iframe sites: Use instruction parameter for better AI-guided extraction
        - For simple sites: Selector fallback maintains backward compatibility

        CRITICAL RULES:
        - ALWAYS start with observe_page because you need to understand the page before interacting with it
        - WHEN YOU SEE "Sign in" OR "Google Accounts" PAGE: IMMEDIATELY use wait_for_user_input tool
        - DO NOT try to click "Create account" or fill login forms - use wait_for_user_input instead
        - BEFORE EXTRACTING: Use validate_chat_response to check if content exists
        - EMPTY RESULTS = TASK NOT DONE. Keep working until you have actual content.
        - Use perform_action to actually type the question into chat fields
        - DO NOT mark completed unless you have real extracted content to paste
        - FOLLOW THE EXACT INSTRUCTIONS PROVIDED IN THE PROMPT

        **CRITICAL NOTEBOOKLM SOURCE UPLOAD INSTRUCTIONS:**
        To add content that either is in memory or will be typed, you MUST:
        1. Go to "Add sources"
        2. Look for the words "Paste text" (this is a section header)
        3. Below "Paste text" there is a button called "Copied text"
        4. IMPORTANT: Even though it says "Copied text", this is the place to add ANY text that will be typed by the user
        5. Click the "Copied text" button to open the text input area
        6. Then type or paste your content into the text field that appears
        7. Submit/save the content to add it as a source

        **CRITICAL NOTEBOOKLM CHAT INSTRUCTIONS:**
        After typing a question in the NotebookLM chat:
        1. Look for the send/submit button - it looks like an arrow icon (→)
        2. Click the arrow/send button to submit your question
        3. WAIT for the response to be generated (this takes time)
        4. Use validate_chat_response to check if response is ready
        5. Only proceed to extract when response is complete

        **CRITICAL GOOGLE DOCS INSTRUCTIONS:**
        When in a new document in docs.google.com there are TWO forms:
        1. The FIRST form is for the title of the file (document title)
        2. The SECOND form is for the content (document body)
        3. **CRITICAL**: Use paste_content_clipboard(content, "Google Docs document") to paste content
        4. This method uses proper clipboard API + keyboard shortcuts (Cmd+V)
        5. Do NOT use regular paste_into_page for Google Docs - it won't work with canvas
        6. After pasting, use verify_visual_content to confirm content is visible
        7. Example workflow: paste_content_clipboard(response_text) → verify_visual_content("lease agreement response visible")

        **CRITICAL GOOGLE SHEETS INSTRUCTIONS:**
        When faced with sheets.google.com please ensure that:
        1. If the file is not immediately visible, type part of its name and click on the search button
        2. After opening the file, use verify_visual_content to confirm content is visible
        3. **CRITICAL**: Add the cursor into the cell you want to copy and then Use Cmd+C to copy the content
        4. **CRITICAL**: Use paste_content_clipboard(content, "Google Sheets document") to paste content to the cell you need to paste into
        5. Do NOT use regular paste_into_page for Google Sheets - it won't work with canvas
        6. After pasting, use verify_visual_content to confirm content is visible
        7. Example workflow: paste_content_clipboard(response_text) → verify_visual_content("lease agreement response visible")

        These are the ONLY ways to properly interact with NotebookLM and Google Docs and Google Sheets!
        """


class AgentExecutor:
    """Executes a single iteration of the PydanticAI agent."""

    def __init__(self, pydantic_agent: PydanticAIWebAutomationAgent) -> None:
        self.pydantic_agent = pydantic_agent

    async def execute_iteration(self, deps: AgentDependencies) -> TaskStatus:
        """Execute one iteration of the agent."""
        prompt = f"""
        USER INSTRUCTIONS TO FOLLOW EXACTLY: {deps.instructions}

        Current Status:
        - Current URL: {deps.current_url or 'None'}
        - Execution History: {deps.execution_history[-5:] if deps.execution_history else 'None'}

        CRITICAL: If you see "Sign in" or "Google Accounts" page, use wait_for_user_input immediately.

        Analyze the current state and determine:
        1. Are you following the user instructions exactly as written?
        2. If you see authentication pages, did you use wait_for_user_input?
        3. Is the task completed with real extracted content?
        4. If not completed, what specific action from the user instructions is needed next?

        Provide a structured response with your decision and reasoning.
        """

        # Log what's being sent to the model
        logger.info("🤖 SENDING TO MODEL:")
        logger.info(f"📝 PROMPT:\n{prompt}")
        logger.info(f"📦 DEPENDENCIES:\n  - instructions: {deps.instructions}\n  - current_url: {deps.current_url}\n  - execution_history: {deps.execution_history}")

        result = await self.pydantic_agent.agent.run(prompt, deps=deps)
        status = result.output

        # Log the model's response
        logger.info("🤖 MODEL RESPONSE:")
        logger.info(f"✅ completed: {status.completed}")
        logger.info(f"🔍 double_checked: {status.double_checked}")
        logger.info(f"🎯 needs_stagehand: {status.needs_stagehand}")
        logger.info(f"➡️ next_action: {status.next_action}")
        logger.info(f"💭 reasoning: {status.reasoning}")
        logger.info(f"📊 extracted_data: {status.extracted_data}")

        # Validation: Don't allow completion with empty/missing content UNLESS visual verification was used
        any("Visual verification" in step for step in deps.execution_history)
        visual_verification_passed = any("Visual verification" in step and ("PASSED" in step or "visible" in step) for step in deps.execution_history)

        if status.completed and not status.extracted_data and not visual_verification_passed:
            logger.warning("⚠️ Agent tried to mark completed but no content was extracted! Forcing continuation...")
            console.print("⚠️ [red]Agent tried to mark completed but no content was extracted! Forcing continuation...[/red]")
            status.completed = False
            status.double_checked = False
            status.reasoning = f"{status.reasoning} | FORCED CONTINUATION: No actual content extracted yet - must continue working"
        elif status.completed and visual_verification_passed:
            logger.success("✅ Task completed with visual verification - bypassing DOM extraction requirement")
            console.print("✅ [green]Task completed with visual verification - Google Docs canvas content confirmed[/green]")

        return status


class AgentManager:
    """Manages the execution of the web automation agent."""

    def __init__(self, max_iterations: int = 100) -> None:
        self.max_iterations = max_iterations
        self.pydantic_agent = PydanticAIWebAutomationAgent()
        self.executor = AgentExecutor(self.pydantic_agent)

    async def execute_instructions(self, instructions: str) -> TaskStatus:
        """Execute instructions with iterative agent management."""
        stagehand_tool = StagehandTool()

        # Auto-initialize Stagehand
        console.print(Panel(
            "🚀 Initializing Stagehand browser session...",
            title="Startup",
            border_style="blue",
        ))

        init_result = await stagehand_tool.initialize()
        if not init_result["success"]:
            console.print(f"❌ [bold red]Failed to initialize Stagehand: {init_result['error']}[/bold red]")
            return TaskStatus(
                completed=False,
                double_checked=False,
                needs_stagehand=False,
                reasoning=f"Failed to initialize Stagehand: {init_result['error']}",
            )

        console.print(f"✅ [bold green]Stagehand initialized successfully (Session: {init_result.get('sessionId', 'unknown')})[/bold green]")

        # Initialize dependencies
        deps = AgentDependencies(
            stagehand_tool=stagehand_tool,
            instructions=instructions,
        )

        try:
            # Iterative execution
            for iteration in range(self.max_iterations):
                console.print(Panel(
                    f"🔄 Iteration {iteration + 1}/{self.max_iterations}",
                    title="Processing",
                    border_style="cyan",
                ))

                status = await self.executor.execute_iteration(deps)

                console.print(Panel(
                    Text(status.reasoning, style="bold white"),
                    title="📊 Agent Status",
                    border_style="green" if status.completed else "yellow",
                ))

                # Check if task is completed and double-checked
                if status.completed and status.double_checked:
                    console.print(Panel(
                        "✅ Task completed successfully!",
                        title="Success",
                        border_style="green",
                        style="bold green",
                    ))
                    return status

                # If we need more iterations but reached the limit
                if iteration == self.max_iterations - 1:
                    status.reasoning += f" (⏰ Reached maximum iterations: {self.max_iterations})"
                    return status

            return status

        finally:
            # Always clean up
            console.print("🧹 [bold blue]Cleaning up browser session...[/bold blue]")
            await stagehand_tool.close()


async def main() -> None:
    """Main execution function with example instructions."""
    # Check for required environment variable
    if not os.getenv("MODEL_API_KEY"):
        console.print("❌ [bold red]Please set MODEL_API_KEY environment variable[/bold red]")
        return

    # Use the instructions defined at the top of the file
    instructions = INSTRUCTIONS

    # Display startup banner
    console.print(Panel(
        Text("🎭 PydanticAI Stagehand Automation", style="bold magenta", justify="center"),
        subtitle="📚 Using stagehand-py library for browser automation",
        border_style="magenta",
    ))

    console.print(Panel(
        Text(instructions, style="italic cyan"),
        title="📋 Mission Instructions",
        border_style="cyan",
    ))

    # Initialize and run the agent manager
    manager = AgentManager(max_iterations=5)

    try:
        final_status = await manager.execute_instructions(instructions)

        # Display final results
        console.print("\n")
        console.print(Panel(
            f"""
✅ [bold green]Completed:[/bold green] {final_status.completed}
🔍 [bold blue]Double Checked:[/bold blue] {final_status.double_checked}
💭 [bold yellow]Reasoning:[/bold yellow] {final_status.reasoning}
            """.strip(),
            title="📊 FINAL RESULT",
            border_style="green" if final_status.completed else "red",
        ))

        if final_status.extracted_data:
            console.print(Panel(
                str(final_status.extracted_data),
                title="📦 Extracted Data",
                border_style="blue",
            ))

    except Exception as e:
        console.print(Panel(
            f"❌ Error during execution: {e}",
            title="Error",
            border_style="red",
            style="bold red",
        ))



async def test_search_interaction() -> None:
    """Test search interaction."""
    instructions = """
    Navigate to Google (https://google.com).
    Search for "stagehand python library".
    Extract the title and description of the first search result.
    Verify the extraction is accurate.
    """

    manager = AgentManager()
    await manager.execute_instructions(instructions)


class SignalHandler:
    """Handles Ctrl+C interruptions with user input toggle behavior."""

    def __init__(self):
        self.is_in_user_input = False
        self.user_input_active = False
        self.manager = None
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        def handle_sigint(signum, frame):
            print('\n🛑 Received SIGINT (Ctrl+C)...')

            if self.is_in_user_input:
                # If already in user input, shutdown
                print('💀 Shutting down from user input screen...')
                self.graceful_shutdown()
            else:
                # Otherwise, trigger user input
                print('⏸️  Triggering user input...')
                self.trigger_user_input()

        def handle_sigterm(signum, frame):
            print('\n🛑 Received SIGTERM...')
            self.graceful_shutdown()

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigterm)

    def trigger_user_input(self):
        self.is_in_user_input = True
        print('\n📝 =================')
        print('📝 USER INPUT MODE')
        print('📝 Type your message and press Enter to continue automation')
        print('📝 Press Ctrl+C again to shutdown')
        print('📝 =================')

        try:
            user_input = input('💬 Your input: ')
            self.is_in_user_input = False

            if user_input.strip():
                logger.info(f"✅ User provided: {user_input}")
                print('🔄 Resuming automation...')
            else:
                print('📝 No input provided, resuming automation...')

        except KeyboardInterrupt:
            # This handles Ctrl+C during input
            self.graceful_shutdown()

    def graceful_shutdown(self):
        print('🛑 Initiating graceful shutdown...')

        try:
            if self.manager:
                print('🧹 Cleaning up agent manager...')
                # Add any cleanup for the agent manager if needed
            print('✅ Shutdown complete')
        except Exception as error:
            print(f'❌ Error during shutdown: {error}')

        sys.exit(0)

    def set_manager(self, manager):
        self.manager = manager


if __name__ == "__main__":
    console.print(Panel(
        Text("🎭 Welcome to Stagehand-powered PydanticAI Automation!", style="bold green", justify="center"),
        subtitle="🤖 Ready to automate browser tasks with AI",
        border_style="green",
    ))

    # Setup signal handling
    signal_handler = SignalHandler()

    try:
        # Run the main example
        uvloop.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Received KeyboardInterrupt during execution")
        signal_handler.graceful_shutdown()
    except Exception as e:
        logger.error(f"❌ Unhandled exception: {e}")
        signal_handler.graceful_shutdown()

    # Uncomment to run additional tests:
    # asyncio.run(test_product_extraction())
    # asyncio.run(test_search_interaction())
