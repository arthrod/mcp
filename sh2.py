"""Enhanced Stagehand tool with robust state management and beautiful console output."""
import asyncio
import json
import os
import signal
import sys
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import uvloop
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.layout import Layout
from rich.live import Live

# Constants
MAX_ITERATIONS = 10  # Made constant as requested
JSONL_OUTPUT_FILE = "automation_results.jsonl"
DOM_SETTLE_TIMEOUT_MS = 3000  # Increased for reliability
RESPONSE_WAIT_TIME = 5  # Seconds to wait for responses

# Configure logger
logger.remove()
logger.add("automation.log", rotation="10 MB", level="DEBUG", format="{time} | {level} | {function} | {message}")
logger.add(lambda msg: print(msg, end=""), level="INFO", format="🔧 {function}: {message}")

model = OpenAIModel("gpt-4.1")
console = Console()


class TaskState(BaseModel):
    """Persistent state for multi-step tasks."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = Field(default="prompt_processing")
    current_step: int = Field(default=0)
    total_steps: int = Field(default=0)
    data_buffer: Dict[str, Any] = Field(default_factory=dict)
    extracted_content: List[Dict[str, Any]] = Field(default_factory=list)
    processing_queue: List[str] = Field(default_factory=list)
    processed_items: List[str] = Field(default_factory=list)
    last_successful_extraction: Optional[str] = Field(default=None)
    last_extraction_method: Optional[str] = Field(default=None)
    last_observation: Optional[str] = Field(default=None)

    def add_extraction(self, content: str, method: str = "unknown") -> None:
        """Add successfully extracted content to state."""
        self.extracted_content.append({
            "content": content,
            "method": method,
            "timestamp": datetime.now(UTC).isoformat(),
            "step": self.current_step
        })
        self.last_successful_extraction = content
        self.last_extraction_method = method

    def mark_processed(self, item: str) -> None:
        """Mark an item as processed and move to next in queue."""
        if item in self.processing_queue:
            self.processing_queue.remove(item)
        if item not in self.processed_items:
            self.processed_items.append(item)
        self.current_step += 1

    def has_valid_extraction(self) -> bool:
        """Check if we have valid extracted content for current step."""
        return bool(self.last_successful_extraction and len(self.last_successful_extraction.strip()) > 50)

    def update_observation(self, observation: str) -> None:
        """Update last observation."""
        self.last_observation = observation


class StagehandTool:
    """Enhanced Stagehand tool with state management."""

    def __init__(self) -> None:
        self.base_url = "http://localhost:3001"
        self.session_id: str | None = None
        self.state = TaskState()

    def _request(self, method: str, endpoint: str, data: dict | None = None) -> dict[str, Any]:
        """Make request to bridge server."""
        try:
            if method == "GET":
                response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
            else:
                response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=60)
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

    async def act(self, action: str, dom_settle_timeout_ms: int = DOM_SETTLE_TIMEOUT_MS) -> dict[str, Any]:
        """Perform action."""
        return self._request("POST", "/act", {"action": action, "domSettleTimeoutMs": dom_settle_timeout_ms})

    async def observe(self, instruction: str, draw_overlay: bool = True) -> dict[str, Any]:
        """Observe page."""
        return self._request("POST", "/observe", {"instruction": instruction, "drawOverlay": draw_overlay})

    async def extract(self, instruction: str, schema_model) -> dict[str, Any]:
        """Extract data."""
        schema = schema_model.model_json_schema() if hasattr(schema_model, "model_json_schema") else schema_model
        return self._request("POST", "/extract", {"instruction": instruction, "schema": schema})

    async def extract_dom(self, instruction: str | None = None, selectors: list | None = None, use_iframes: bool | None = None) -> dict[str, Any]:
        """Extract content using enhanced Stagehand extraction."""
        payload = {}
        if instruction:
            payload["instruction"] = instruction
        if selectors:
            payload["selectors"] = selectors
        if use_iframes is not None:
            payload["useIframes"] = use_iframes
        return self._request("POST", "/extract-dom", payload)

    async def verify_visual(self, description: str) -> dict[str, Any]:
        """Visual verification for canvas-based content."""
        return self._request("POST", "/verify-visual", {"description": description})

    async def paste_clipboard(self, content: str, target_description: str = "input field") -> dict[str, Any]:
        """Paste content using clipboard."""
        return self._request("POST", "/paste-clipboard", {"content": content, "target_description": target_description})

    async def inject_prompt(self, prompt_text: str) -> dict[str, Any]:
        """Inject prompt directly into the input field."""
        return self._request("POST", "/inject-prompt", {"prompt": prompt_text})

    async def extract_response(self) -> dict[str, Any]:
        """Extract AI response using direct DOM scraping."""
        return self._request("POST", "/extract-response", {})

    async def agent_execute(self, instructions: str, max_steps: int = 10) -> dict[str, Any]:
        """Execute complex action using agent endpoint."""
        return self._request("POST", "/agent", {"instructions": instructions, "maxSteps": max_steps})

    async def close(self) -> dict[str, Any]:
        """Close session."""
        return self._request("POST", "/close")

    @property
    def stagehand(self):
        return self

    @property
    def page(self):
        class PageWrapper:
            def __init__(self, tool) -> None:
                self.tool = tool

            async def goto(self, url: str) -> None:
                result = await self.tool.navigate(url)
                if not result.get("success"):
                    raise Exception(result.get("error", "Navigation failed"))

            async def act(self, action: str) -> dict[str, Any]:
                return await self.tool.act(action)

            @property
            def url(self):
                result = self.tool._request("GET", "/url")
                return result.get("url", "unknown")

        return PageWrapper(self)


async def wait_for_user_input_with_timer(message: str = "Waiting for user to complete manual action...") -> str:
    """Wait for user input with a 300s timeout."""
    console.print(Panel(
        Text(message, style="bold yellow", justify="center"),
        title="🔄 User Action Required",
        border_style="yellow",
        expand=False
    ))

    console.print("[bold green]Press Enter when you've completed the required action (300s timeout)...[/bold green]")

    try:
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: input("➡️  ")),
            timeout=300.0,
        )
        console.print("✅ [bold green]User input received, continuing automation...[/bold green]\n")
        return f"✅ User completed manual action: {message}"
    except TimeoutError:
        console.print("⏰ [bold red]Timeout reached (300s), continuing automation...[/bold red]\n")
        return f"⏰ Timeout: {message}"
    except (EOFError, KeyboardInterrupt):
        console.print("🚫 [bold red]Input interrupted, continuing automation...[/bold red]\n")
        return f"🚫 Input interrupted: {message}"


def append_to_jsonl(data: dict) -> None:
    """Append data to JSONL file."""
    with open(JSONL_OUTPUT_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
    logger.info(f"📝 Appended result to {JSONL_OUTPUT_FILE}")


class TaskStatus(BaseModel):
    """Status of task execution."""
    completed: bool = Field(default=False, description="Is the current step completed?")
    all_steps_completed: bool = Field(default=False, description="Are all steps completed?")
    found_target_content: bool = Field(default=False, description="Did we find the content we were looking for?")
    next_action: str | None = Field(default=None, description="Next specific action to take")
    reasoning: str = Field(default="", description="Explanation of current status")
    extraction_method_used: str | None = Field(default=None, description="Which extraction method was successful")
    current_state: TaskState = Field(default_factory=TaskState, description="Current task state")


class AgentDependencies(BaseModel):
    """Dependencies for the agent."""
    stagehand_tool: Any
    instructions: str
    current_url: str | None = None
    execution_history: list[str] = Field(default_factory=list)
    task_state: TaskState = Field(default_factory=TaskState)


class PydanticAIWebAutomationAgent:
    """Enhanced PydanticAI agent with robust observation."""

    def __init__(self) -> None:
        self.agent = Agent(
            model,
            deps_type=AgentDependencies,
            result_type=TaskStatus,
            system_prompt=self._get_system_prompt(),
        )

        @self.agent.tool
        async def navigate_to_url(ctx: RunContext[AgentDependencies], url: str) -> str:
            """Navigate to a specific URL."""
            logger.info(f"🧭 TOOL CALLED: navigate_to_url with url='{url}'")
            try:
                await ctx.deps.stagehand_tool.stagehand.page.goto(url)
                ctx.deps.current_url = url
                ctx.deps.execution_history.append(f"Navigated to {url}")
                logger.success(f"✅ Successfully navigated to {url}")
                console.print(f"[green]✅ Navigated to:[/green] [blue]{url}[/blue]")
                return f"✅ Successfully navigated to {url}"
            except Exception as e:
                logger.error(f"❌ Navigation failed: {e}")
                console.print(f"[red]❌ Navigation failed:[/red] {e}")
                return f"❌ Failed to navigate: {e!s}"

        @self.agent.tool
        async def perform_action(ctx: RunContext[AgentDependencies], action: str) -> str:
            """Perform a web action - if act fails, automatically uses agent. ALWAYS observe after!"""
            logger.info(f"🎬 TOOL CALLED: perform_action with action='{action}'")
            console.print(f"[yellow]🎬 Performing:[/yellow] {action}")

            # First try act endpoint
            result = await ctx.deps.stagehand_tool.act(action, DOM_SETTLE_TIMEOUT_MS)
            if result["success"]:
                ctx.deps.execution_history.append(f"Performed action: {action}")
                logger.success(f"✅ Action completed: {action}")
                console.print(f"[green]✅ Action completed[/green]")
                return f"✅ Action completed: {action}. IMPORTANT: Now use observe to check if it worked!"

            # If act failed, automatically try agent endpoint
            logger.warning(f"⚠️ Act failed: {result['error']}, trying agent endpoint...")
            console.print(f"[yellow]⚠️ Act failed, using agent for complex action...[/yellow]")

            agent_result = await ctx.deps.stagehand_tool.agent_execute(action)
            if agent_result["success"]:
                ctx.deps.execution_history.append(f"Performed complex action via agent: {action}")
                logger.success(f"✅ Agent completed: {action}")
                console.print(f"[green]✅ Agent action completed[/green]")
                return f"✅ Action completed via agent: {action}. IMPORTANT: Now use observe to check!"

            logger.error(f"❌ Both act and agent failed: {result['error']} | {agent_result['error']}")
            console.print(f"[red]❌ Action failed completely[/red]")
            return f"❌ Action failed: {result['error']} | Agent also failed: {agent_result['error']}"

        @self.agent.tool
        async def extract_dom_content(ctx: RunContext[AgentDependencies], instruction: str | None = None, selectors: list | None = None) -> str:
            """Extract content from DOM - returns actual content if found."""
            logger.info(f"🔍 TOOL CALLED: extract_dom_content with instruction='{instruction}'")
            console.print(f"[cyan]🔍 Extracting content...[/cyan]")

            # Fix selector format if needed (handle objects passed as selectors)
            if selectors:
                fixed_selectors = []
                for sel in selectors:
                    if isinstance(sel, str):
                        fixed_selectors.append(sel)
                    elif isinstance(sel, dict) and "selector" in sel:
                        fixed_selectors.append(sel["selector"])
                    else:
                        logger.warning(f"⚠️ Skipping invalid selector: {sel}")
                selectors = fixed_selectors if fixed_selectors else None

            with console.status("[cyan]Extracting DOM content...[/cyan]", spinner="dots"):
                result = await ctx.deps.stagehand_tool.extract_dom(
                    instruction=instruction,
                    selectors=selectors,
                    use_iframes=None
                )

            if result["success"] and result.get("data"):
                data = result["data"]
                extracted_text = data.get("combinedText", "") or data.get("content", "")

                if extracted_text and len(extracted_text.strip()) > 50:
                    # Store in state
                    ctx.deps.task_state.add_extraction(extracted_text, "dom_extraction")
                    logger.success(f"✅ Extracted {len(extracted_text)} characters")
                    console.print(f"[green]✅ CONTENT FOUND:[/green] {len(extracted_text)} characters extracted")
                    console.print(Panel(extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                                      title="Extracted Content Preview",
                                      border_style="green"))
                    return f"✅ CONTENT FOUND ({len(extracted_text)} chars): {extracted_text}"

            console.print("[red]❌ NO CONTENT FOUND[/red] - Need to try a different approach")
            return "❌ NO CONTENT FOUND - try observing the page to understand why"

        @self.agent.tool
        async def observe_page(ctx: RunContext[AgentDependencies], instruction: str) -> str:
            """CRITICAL TOOL: Observe page to understand what's happening - USE THIS AFTER EVERY ACTION!"""
            logger.info(f"👁️ TOOL CALLED: observe_page with instruction='{instruction}'")
            console.print(f"[magenta]👁️ Observing:[/magenta] {instruction}")

            with console.status("[magenta]Observing page...[/magenta]", spinner="dots"):
                result = await ctx.deps.stagehand_tool.observe(instruction)

            if result["success"]:
                observations = result['data']
                ctx.deps.execution_history.append(f"Observed: {instruction}")
                ctx.deps.task_state.update_observation(str(observations))

                # Format observations nicely
                if isinstance(observations, list) and observations:
                    table = Table(title="Page Observations", border_style="magenta")
                    table.add_column("Element", style="cyan")
                    table.add_column("Description", style="white")

                    for obs in observations[:5]:  # Show first 5
                        if isinstance(obs, dict):
                            elem = obs.get('selector', 'Unknown')[:50]
                            desc = obs.get('description', 'No description')[:100]
                            table.add_row(elem, desc)

                    console.print(table)
                else:
                    console.print(Panel(str(observations)[:500], title="Observations", border_style="magenta"))

                return f"✅ Observations: {observations}"

            console.print(f"[red]❌ Observation failed:[/red] {result['error']}")
            return f"❌ Observation failed: {result['error']}"

        @self.agent.tool
        async def wait_for_user_input(ctx: RunContext[AgentDependencies], message: str = "Waiting for user...") -> str:
            """Wait for user input."""
            logger.info(f"⏰ TOOL CALLED: wait_for_user_input")
            result_message = await wait_for_user_input_with_timer(message)
            ctx.deps.execution_history.append("User input received")
            return result_message

        @self.agent.tool
        async def inject_prompt(ctx: RunContext[AgentDependencies], prompt_text: str = "") -> str:
            """Inject prompt directly into ChatGPT input field - USE THIS INSTEAD OF TYPING!"""
            # Get the current prompt from queue if not provided
            if not prompt_text and ctx.deps.task_state.processing_queue:
                prompt_text = ctx.deps.task_state.processing_queue[0]
                logger.info(f"📋 Using prompt from queue: {prompt_text[:50]}...")

            if not prompt_text:
                logger.error("❌ No prompt text provided and queue is empty")
                return "❌ No prompt text available"

            logger.info(f"💉 TOOL CALLED: inject_prompt with prompt='{prompt_text[:50]}...'")
            console.print(f"[cyan]💉 Injecting prompt directly into input field[/cyan]")

            result = await ctx.deps.stagehand_tool.inject_prompt(prompt_text)

            if result["success"]:
                ctx.deps.execution_history.append(f"Injected prompt: {prompt_text[:50]}...")
                logger.success("✅ Prompt injected successfully")
                console.print(f"[green]✅ Prompt injected into input field[/green]")
                return f"✅ Prompt injected. Now click the send button (arrow →)!"

            logger.error(f"❌ Injection failed: {result['error']}")
            console.print(f"[red]❌ Injection failed:[/red] {result['error']}")
            return f"❌ Injection failed: {result['error']}"

        @self.agent.tool
        async def extract_ai_response(ctx: RunContext[AgentDependencies]) -> str:
            """Extract AI response using direct DOM scraping - USE THIS FOR RELIABLE EXTRACTION!"""
            logger.info("🎯 TOOL CALLED: extract_ai_response")
            console.print("[cyan]🎯 Extracting AI response directly from DOM[/cyan]")

            with console.status("[cyan]Extracting response...[/cyan]", spinner="dots"):
                result = await ctx.deps.stagehand_tool.extract_response()

            if result["success"] and result.get("data"):
                extracted_text = result["data"].get("response", "")
                method = result["data"].get("method", "unknown")

                if extracted_text and len(extracted_text.strip()) > 50:
                    ctx.deps.task_state.add_extraction(extracted_text, method)
                    logger.success(f"✅ Extracted {len(extracted_text)} characters via {method}")
                    console.print(f"[green]✅ Response extracted:[/green] {len(extracted_text)} characters")
                    console.print(Panel(extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                                      title="Extracted Response",
                                      border_style="green"))
                    return f"✅ RESPONSE EXTRACTED ({len(extracted_text)} chars): {extracted_text}"

            console.print("[red]❌ No response found[/red] - May need to wait longer or try clipboard fallback")
            return "❌ No response found - try observe_page to check status"

        @self.agent.tool
        async def handle_security_challenge(ctx: RunContext[AgentDependencies], wait_seconds: int = 15) -> str:
            """Handle security challenges like 'Just a moment...' screens by waiting."""
            logger.info(f"🛡️ TOOL CALLED: handle_security_challenge with wait={wait_seconds}s")
            console.print(Panel(
                f"[yellow]🛡️ Security challenge detected, waiting {wait_seconds}s...[/yellow]",
                title="Cloudflare/DDoS Protection",
                border_style="yellow"
            ))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console
            ) as progress:
                task = progress.add_task(f"[yellow]Waiting for security check...[/yellow]", total=wait_seconds)
                for i in range(wait_seconds):
                    await asyncio.sleep(1)
                    progress.update(task, advance=1)

            console.print(f"[green]✅ Waited {wait_seconds} seconds for security check[/green]")
            return f"✅ Waited {wait_seconds} seconds. Now observe to check if the page loaded!"

        @self.agent.tool
        async def wait_for_response(ctx: RunContext[AgentDependencies], seconds: int = RESPONSE_WAIT_TIME) -> str:
            """Wait for a response to be generated (e.g., after submitting to ChatGPT)."""
            logger.info(f"⏳ TOOL CALLED: wait_for_response with seconds={seconds}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Waiting {seconds}s for response generation...[/cyan]", total=seconds)
                for i in range(seconds):
                    await asyncio.sleep(1)
                    progress.update(task, advance=1)

            console.print(f"[green]✅ Waited {seconds} seconds[/green]")
            return f"✅ Waited {seconds} seconds. Now use observe to check if response is ready!"

        @self.agent.tool
        async def save_extraction_result(ctx: RunContext[AgentDependencies], prompt: str = "", response: str = "") -> str:
            """Save a prompt-response pair to JSONL file."""
            logger.info("💾 TOOL CALLED: save_extraction_result")

            # Use the last successful extraction if no response provided
            if not response and ctx.deps.task_state.last_successful_extraction:
                response = ctx.deps.task_state.last_successful_extraction
                logger.info("📋 Using last successful extraction as response")

            # Use the current/just processed prompt from queue if not provided
            if not prompt and ctx.deps.task_state.processed_items:
                prompt = ctx.deps.task_state.processed_items[-1]
                logger.info(f"📋 Using last processed prompt: {prompt[:50]}...")
            elif not prompt and ctx.deps.task_state.processing_queue:
                prompt = ctx.deps.task_state.processing_queue[0]
                logger.info(f"📋 Using current prompt from queue: {prompt[:50]}...")

            if not prompt or not response:
                logger.error("❌ Missing prompt or response")
                return "❌ Cannot save: missing prompt or response"

            record = {
                "uuid": str(uuid.uuid4()),
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now(UTC).isoformat(),
                "task_id": ctx.deps.task_state.task_id,
                "extraction_method": ctx.deps.task_state.last_extraction_method
            }

            append_to_jsonl(record)
            ctx.deps.task_state.mark_processed(prompt)

            console.print(Panel(
                f"[green]✅ Saved to {JSONL_OUTPUT_FILE}[/green]\n"
                f"[yellow]Prompt:[/yellow] {prompt[:100]}...\n"
                f"[cyan]Response:[/cyan] {len(response)} characters",
                title="Result Saved",
                border_style="green"
            ))

            return f"✅ Saved result for prompt: {prompt[:50]}..."

        @self.agent.tool
        async def get_current_state(ctx: RunContext[AgentDependencies]) -> str:
            """Get current task state for decision making."""
            state = ctx.deps.task_state

            state_table = Table(title="Current Task State", border_style="blue")
            state_table.add_column("Property", style="cyan")
            state_table.add_column("Value", style="white")

            state_table.add_row("Task ID", state.task_id[:8] + "...")
            state_table.add_row("Current Step", str(state.current_step))
            state_table.add_row("Queue Size", str(len(state.processing_queue)))
            state_table.add_row("Processed", str(len(state.processed_items)))
            state_table.add_row("Has Valid Extract", "✅" if state.has_valid_extraction() else "❌")
            state_table.add_row("Last Method", state.last_extraction_method or "None")

            console.print(state_table)

            return f"""Current State:
- Task ID: {state.task_id}
- Current Step: {state.current_step}
- Processing Queue: {len(state.processing_queue)} items
- Processed: {len(state.processed_items)} items
- Has Valid Extraction: {state.has_valid_extraction()}
- Last Method: {state.last_extraction_method}
- Last Observation: {state.last_observation[:100] if state.last_observation else 'None'}"""

    def _get_system_prompt(self) -> str:
        return """
You are a robust web automation agent that executes tasks step by step.

🚨 CRITICAL RULES - FOLLOW EXACTLY:

1. ALWAYS OBSERVE AFTER ACTIONS:
   - After EVERY perform_action → MUST use observe_page to check if it worked
   - After navigate_to_url → observe to see what loaded
   - After wait_for_response → observe to check if response appeared
   - If something seems wrong → observe to understand why
   - NEVER assume an action worked without observing!

2. SECURITY CHALLENGES:
   - "Just a moment..." = Cloudflare/DDoS protection → use handle_security_challenge(15)
   - After security wait → observe to check if page loaded
   - If still stuck → wait_for_user_input("Security challenge detected, please complete")
   - Authentication errors → always use wait_for_user_input

3. FOR CHATGPT/AI SERVICES - NEW SIMPLIFIED STEPS:
   - Navigate to service
   - Observe if "Just a moment..." → handle_security_challenge(15)
   - Observe if logged in or need auth
   - If not logged in → wait_for_user_input("Please log in to ChatGPT")
   - After login → observe chat interface is ready
   - perform_action("Click on the message input field") → observe
   - inject_prompt("your prompt text here") ← USE THIS! NO TYPING!
   - observe_page("Verify prompt was injected")
   - perform_action("Click the send button (arrow icon →) to submit")
   - wait_for_response(10) → let it generate
   - observe_page("Check if AI response has appeared")
   - Keep observing until response is complete (no "..." or loading)
   - extract_ai_response() ← USE THIS! RELIABLE EXTRACTION!

4. SUBMITTING PROMPTS:
   - ChatGPT uses an ARROW BUTTON (→) to send messages, not Enter key
   - The send button is usually near the input field
   - MUST click this arrow/send button, typing alone won't submit!
   - If first action fails, it will automatically try agent endpoint

5. EXTRACTION RULES:
   - Only extract when observe confirms content exists
   - If "NO CONTENT FOUND" → observe why, don't keep trying same method
   - Content still generating → wait more and observe again
   - Need at least 50 characters for valid extraction

6. STATE AWARENESS:
   - Always check get_current_state() first
   - Track what's been tried to avoid repetition
   - Save results immediately after successful extraction

AVAILABLE TOOLS:
- navigate_to_url: Go to a webpage
- observe_page: Look at current page (USE CONSTANTLY!)
- perform_action: Do something (click, etc) - auto-fallback to agent
- inject_prompt: Inject prompt directly into input field (NO TYPING!)
- extract_ai_response: Extract response using DOM scraping (RELIABLE!)
- handle_security_challenge: Wait for Cloudflare/"Just a moment..." screens
- wait_for_response: Wait for AI to generate response
- extract_dom_content: Generic extraction (fallback only)
- wait_for_user_input: For login/manual steps
- save_extraction_result: Save prompt/response pair
- get_current_state: Check task progress

EXAMPLE WORKFLOW FOR CHATGPT:
1. get_current_state()
2. navigate_to_url("https://chat.openai.com")
3. observe_page("Check what loaded")
4. [If "Just a moment..."] handle_security_challenge(15)
5. observe_page("Check if security passed")
6. [If login needed] wait_for_user_input("Please log in to ChatGPT")
7. observe_page("Verify chat interface is ready")
8. perform_action("Click on the message input area")
9. observe_page("Verify input is focused and ready")
10. inject_prompt() ← Auto-uses prompt from queue!
11. observe_page("Verify prompt was injected")
12. perform_action("Click the send button arrow (→) to submit the message")
13. observe_page("Check if message was sent")
14. wait_for_response(10)
15. observe_page("Check if response started appearing")
16. [If still loading] wait_for_response(5) and observe again
17. extract_ai_response() ← Use this for extraction!
18. save_extraction_result() if successful

REMEMBER:
- observe_page is your eyes - use it constantly!
- The send button is an ARROW (→), not Enter key!
- If act fails, agent will be tried automatically
- Be patient with responses - they take time to generate
- Handle security screens properly - don't panic!
"""


class AgentExecutor:
    """Executes iterations with state awareness."""

    def __init__(self, pydantic_agent: PydanticAIWebAutomationAgent) -> None:
        self.pydantic_agent = pydantic_agent

    async def execute_iteration(self, deps: AgentDependencies) -> TaskStatus:
        """Execute one iteration."""
        # Current prompt to process
        current_prompt = deps.task_state.processing_queue[0] if deps.task_state.processing_queue else None

        state_summary = f"""
        📊 Current State:
        - Step: {deps.task_state.current_step + 1}/{deps.task_state.total_steps}
        - Current Prompt: {current_prompt[:50] + '...' if current_prompt and len(current_prompt) > 50 else current_prompt}
        - Queue: {len(deps.task_state.processing_queue)} items remaining
        - Last extraction valid: {deps.task_state.has_valid_extraction()}
        - Current URL: {deps.current_url}
        - Last observation: {deps.task_state.last_observation[:100] + '...' if deps.task_state.last_observation else 'None'}
        """

        prompt = f"""
        {state_summary}

        Instructions: {deps.instructions}

        Recent History (last 5 actions): {deps.execution_history[-5:] if deps.execution_history else 'None'}

        DECISION REQUIRED:
        1. If we have valid extracted content (>50 chars) for current prompt, save it and move to next
        2. If no valid content yet, what's the next action? Remember to OBSERVE after actions!
        3. If page shows login/auth, use wait_for_user_input
        4. If you submitted a prompt, did you wait_for_response and observe?

        Current prompt to process: {current_prompt}

        Return your analysis and next action.
        """

        console.print(Panel(state_summary, title="🤖 Agent Analysis", border_style="blue"))

        result = await self.pydantic_agent.agent.run(prompt, deps=deps)
        status = result.output
        status.current_state = deps.task_state

        return status


class AgentManager:
    """Manages the execution with state persistence."""

    def __init__(self) -> None:
        self.pydantic_agent = PydanticAIWebAutomationAgent()
        self.executor = AgentExecutor(self.pydantic_agent)

    async def execute_task(self, instructions: str, prompts: List[str]) -> None:
        """Execute task with given prompts."""
        stagehand_tool = StagehandTool()

        # Initialize with beautiful banner
        console.print(Panel(
            Text("🚀 Initializing Stagehand Browser Session", style="bold cyan", justify="center"),
            border_style="cyan"
        ))

        init_result = await stagehand_tool.initialize()
        if not init_result["success"]:
            console.print(Panel(
                f"❌ Failed to initialize: {init_result['error']}",
                title="Initialization Error",
                border_style="red"
            ))
            return

        console.print(f"[green]✅ Stagehand initialized successfully[/green]\n")

        # Setup state with prompts
        stagehand_tool.state.processing_queue = prompts.copy()
        stagehand_tool.state.total_steps = len(prompts)

        # Initialize dependencies
        deps = AgentDependencies(
            stagehand_tool=stagehand_tool,
            instructions=instructions,
            task_state=stagehand_tool.state
        )

        try:
            for iteration in range(MAX_ITERATIONS):
                console.print(Panel(
                    f"🔄 Iteration {iteration + 1}/{MAX_ITERATIONS} | Queue: {len(deps.task_state.processing_queue)} items",
                    title="Processing Status",
                    border_style="cyan",
                    expand=False
                ))

                status = await self.executor.execute_iteration(deps)

                # Display reasoning
                console.print(Panel(
                    status.reasoning,
                    title="💭 Agent Reasoning",
                    border_style="yellow"
                ))

                if status.all_steps_completed or len(deps.task_state.processing_queue) == 0:
                    console.print(Panel(
                        "✅ All tasks completed successfully!",
                        title="Success",
                        border_style="green",
                        style="bold green"
                    ))
                    break

                if iteration == MAX_ITERATIONS - 1:
                    console.print(Panel(
                        f"⏰ Reached maximum iterations ({MAX_ITERATIONS})",
                        title="Iteration Limit",
                        border_style="yellow"
                    ))

        finally:
            console.print("\n[blue]🧹 Cleaning up browser session...[/blue]")
            await stagehand_tool.close()
            console.print("[green]✅ Session closed[/green]")


async def main() -> None:
    """Main execution with user input."""
    if not os.getenv("MODEL_API_KEY"):
        console.print(Panel(
            "❌ Please set MODEL_API_KEY environment variable",
            title="Configuration Error",
            border_style="red"
        ))
        return

    # Beautiful welcome banner
    console.print(Panel(
        Text("🎭 Web Automation with State Management\n"
             "Powered by Stagehand & PydanticAI",
             style="bold magenta",
             justify="center"),
        border_style="magenta"
    ))

    # Get task type
    task_type = Prompt.ask(
        "\n[cyan]What would you like to generate?[/cyan]",
        choices=["legal_contract", "technical_spec", "creative_content", "custom"],
        default="custom"
    )

    if task_type == "custom":
        num_prompts = int(Prompt.ask("[cyan]How many prompts do you want to process?[/cyan]", default="3"))
        prompts = []
        for i in range(num_prompts):
            prompt = Prompt.ask(f"[yellow]Enter prompt {i+1}[/yellow]")
            prompts.append(prompt)
    else:
        # Predefined prompt sets
        prompt_sets = {
            "legal_contract": [
                "Generate a standard NDA between two companies",
                "Create a software licensing agreement template",
                "Draft a simple employment contract"
            ],
            "technical_spec": [
                "Write a REST API specification for a user management system",
                "Create a database schema for an e-commerce platform",
                "Design a microservices architecture document"
            ],
            "creative_content": [
                "Write a short story about time travel",
                "Create a poem about artificial intelligence",
                "Generate a movie plot synopsis"
            ]
        }
        prompts = prompt_sets.get(task_type, [])

    # Get target service
    service_url = Prompt.ask(
        "[cyan]Enter the URL of the service to use[/cyan]",
        default="https://chat.openai.com"
    )

    instructions = f"""
    Process each prompt in the queue by:
    1. Navigating to {service_url}
    2. Observing the page state and handling login if needed
    3. Locating and clicking the chat input area
    4. Using inject_prompt() to insert the prompt (NO TYPING!)
    5. Clicking the send button (arrow → icon)
    6. Waiting for the response to generate
    7. Using extract_ai_response() to get the response (RELIABLE!)
    8. Saving the result to JSONL
    9. Moving to next prompt

    CRITICAL: Use inject_prompt() and extract_ai_response() for reliability!
    CRITICAL: Click the ARROW BUTTON (→) to send messages!
    Total prompts to process: {len(prompts)}
    """

    # Summary table
    summary_table = Table(title="Task Summary", border_style="cyan")
    summary_table.add_column("Property", style="cyan")
    summary_table.add_column("Value", style="white")
    summary_table.add_row("Task Type", task_type)
    summary_table.add_row("Service URL", service_url)
    summary_table.add_row("Prompts to Process", str(len(prompts)))
    summary_table.add_row("Output File", JSONL_OUTPUT_FILE)

    console.print("\n")
    console.print(summary_table)
    console.print("\n")

    # Confirm
    if not Prompt.ask("[yellow]Ready to start?[/yellow]", choices=["y", "n"], default="y") == "y":
        console.print("[red]Cancelled by user[/red]")
        return

    manager = AgentManager()
    await manager.execute_task(instructions, prompts)

    # Show results summary
    if Path(JSONL_OUTPUT_FILE).exists():
        with open(JSONL_OUTPUT_FILE, "r") as f:
            results = [json.loads(line) for line in f]

        results_table = Table(title=f"Results Summary ({len(results)} items)", border_style="green")
        results_table.add_column("Prompt", style="cyan", width=50)
        results_table.add_column("Response Length", style="white")
        results_table.add_column("Timestamp", style="yellow")

        for r in results[-5:]:  # Show last 5
            prompt_preview = r["prompt"][:47] + "..." if len(r["prompt"]) > 50 else r["prompt"]
            results_table.add_row(
                prompt_preview,
                f"{len(r['response'])} chars",
                r["timestamp"].split("T")[1][:8]
            )

        console.print("\n")
        console.print(results_table)
        console.print(f"\n[green]✅ Full results saved to:[/green] [blue]{JSONL_OUTPUT_FILE}[/blue]")


class SignalHandler:
    """Handles Ctrl+C interruptions with graceful shutdown."""

    def __init__(self):
        self.is_shutting_down = False
        self.manager = None
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        def handle_sigint(signum, frame):
            if not self.is_shutting_down:
                self.is_shutting_down = True
                console.print('\n[red]🛑 Received SIGINT (Ctrl+C), shutting down gracefully...[/red]')
                self.graceful_shutdown()
            else:
                console.print('\n[red]💀 Force shutdown[/red]')
                sys.exit(1)

        def handle_sigterm(signum, frame):
            console.print('\n[red]🛑 Received SIGTERM, shutting down...[/red]')
            self.graceful_shutdown()

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigterm)

    def graceful_shutdown(self):
        console.print('[yellow]🛑 Initiating graceful shutdown...[/yellow]')

        try:
            if self.manager:
                console.print('[blue]🧹 Cleaning up browser session...[/blue]')
                # Cleanup would happen here if we had access to the stagehand instance
            console.print('[green]✅ Shutdown complete[/green]')
        except Exception as error:
            console.print(f'[red]❌ Error during shutdown: {error}[/red]')

        sys.exit(0)

    def set_manager(self, manager):
        self.manager = manager


if __name__ == "__main__":
    # Setup signal handling
    signal_handler = SignalHandler()

    try:
        uvloop.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]🛑 Shutdown requested[/red]")
        signal_handler.graceful_shutdown()
    except Exception as e:
        console.print(Panel(
            f"❌ Unexpected error: {e}",
            title="Error",
            border_style="red"
        ))
        logger.exception("Unexpected error in main")
        sys.exit(1)
