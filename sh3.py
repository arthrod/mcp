"""Simple Stagehand tool for PydanticAI using stagehand-py library - UPDATED VERSION."""
import asyncio
import os
import signal
import sys
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

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

# CONFIGURABLE CONSTANTS
MAX_ITERATIONS = 10  # Made configurable
DOM_SETTLE_MS = 1000
EXTRACTION_TIMEOUT = 30
JSONL_OUTPUT_FILE = "generation_results.jsonl"

# Configure loguru logger
logger.remove()
logger.add("automation.log", rotation="10 MB", level="DEBUG", format="{time} | {level} | {function} | {message}")
logger.add(lambda msg: print(msg, end=""), level="INFO", format="🔧 {function}: {message}")

model = OpenAIModel("gpt-4.1")
console = Console()


class ExtractionState(BaseModel):
    """Track what we've actually extracted, not just attempted."""
    has_content: bool = False
    content_type: str = ""  # "dom", "visual", "clipboard", etc.
    extracted_text: str = ""
    extraction_method: str = ""
    verified: bool = False


class TaskState(BaseModel):
    """Enhanced state tracking for multi-step tasks."""
    # Task definition
    user_goal: str = ""
    current_step: int = 0
    total_steps: int = 0

    # Navigation state
    current_url: Optional[str] = None
    authenticated: bool = False

    # Extraction state
    last_extraction: Optional[ExtractionState] = None
    successful_extractions: List[ExtractionState] = Field(default_factory=list)

    # Execution tracking
    execution_history: List[str] = Field(default_factory=list)
    attempted_methods: List[str] = Field(default_factory=list)

    # Results for JSONL
    generation_results: List[Dict[str, Any]] = Field(default_factory=list)


class StagehandTool:
    """Drop-in replacement - makes endpoint calls to Node.js bridge."""

    def __init__(self) -> None:
        self.base_url = "http://localhost:3001"
        self.session_id: str | None = None

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

    async def act(self, action: str, dom_settle_timeout_ms: int = DOM_SETTLE_MS) -> dict[str, Any]:
        """Perform action."""
        return self._request("POST", "/act", {"action": action})

    async def observe(self, instruction: str, draw_overlay: bool = True) -> dict[str, Any]:
        """Observe page."""
        return self._request("POST", "/observe", {"instruction": instruction})

    async def extract(self, instruction: str, schema_model) -> dict[str, Any]:
        """Extract data."""
        schema = schema_model.model_json_schema() if hasattr(schema_model, "model_json_schema") else schema_model
        return self._request("POST", "/extract", {"instruction": instruction, "schema": schema})

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


async def wait_for_user_input_with_timer(message: str = "Waiting for user to complete manual action...") -> str:
    """Wait for user input with a 300s timeout and Rich formatting."""
    console.print(Panel(
        Text(message, style="bold yellow"),
        title="🔄 User Action Required",
        border_style="yellow",
    ))

    console.print("[bold green]Press Enter when you've completed the required action (300s timeout)...[/bold green]")

    try:
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: input("Press Enter to continue: ")),
            timeout=300.0,
        )
        console.print("✅ [bold green]User input received, continuing automation...[/bold green]")
        return f"✅ User completed manual action: {message}"
    except TimeoutError:
        console.print("⏰ [bold red]Timeout reached (300s), continuing automation...[/bold red]")
        return f"⏰ Timeout: {message}"
    except (EOFError, KeyboardInterrupt):
        console.print("🚫 [bold red]Input interrupted, continuing automation...[/bold red]")
        return f"🚫 Input interrupted: {message}"


def save_to_jsonl(result: Dict[str, Any]) -> None:
    """Append result to JSONL file."""
    with open(JSONL_OUTPUT_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')
    logger.info(f"💾 Saved result to {JSONL_OUTPUT_FILE}")


async def get_user_goal() -> str:
    """Interactive initialization to get user's generation goal."""
    console.print(Panel(
        Text("Welcome! I'm your AI generation assistant.", style="bold cyan"),
        title="🤖 Generation Assistant",
        border_style="cyan",
    ))

    console.print("\n[bold yellow]What would you like me to help generate today?[/bold yellow]")
    console.print("[dim]Examples: product descriptions, lease agreements, technical documentation, etc.[/dim]\n")

    user_goal = input("Your request: ").strip()

    if not user_goal:
        user_goal = "Generate sample content based on prompts in Google Sheets"

    console.print(f"\n✅ [bold green]Got it! I'll help you generate: {user_goal}[/bold green]\n")
    return user_goal


class TaskStatus(BaseModel):
    """Status of task execution with state tracking."""
    completed: bool = Field(default=False, description="Is the task completed?")
    needs_action: bool = Field(default=True, description="Do we need to perform more actions?")
    next_action: str | None = Field(default=None, description="Next action to take if not completed")
    reasoning: str = Field(default="", description="Explanation of current status")

    # State tracking
    state: TaskState = Field(default_factory=TaskState, description="Current task state")


class AgentDependencies(BaseModel):
    """Dependencies for the agent."""
    stagehand_tool: Any
    state: TaskState  # Changed from execution_history to full state


class PydanticAIWebAutomationAgent:
    """PydanticAI agent for web automation with Stagehand."""

    def __init__(self) -> None:
        self.agent = Agent(
            model,
            deps_type=AgentDependencies,
            result_type=TaskStatus,
            system_prompt=self._get_system_prompt(),
        )

        # Tool definitions remain mostly the same, but with state tracking
        @self.agent.tool
        async def navigate_to_url(ctx: RunContext[AgentDependencies], url: str) -> str:
            """Navigate to a specific URL using Stagehand page.goto."""
            logger.info(f"🧭 TOOL CALLED: navigate_to_url with url='{url}'")
            try:
                await ctx.deps.stagehand_tool.stagehand.page.goto(url)
                ctx.deps.state.current_url = url
                ctx.deps.state.execution_history.append(f"Navigated to {url}")
                logger.success(f"✅ Successfully navigated to {url}")
                return f"✅ Successfully navigated to {url}"
            except Exception as e:
                logger.error(f"❌ Navigation failed: {e}")
                return f"❌ Failed to navigate: {e!s}"

        @self.agent.tool
        async def observe_page_structure(ctx: RunContext[AgentDependencies], instruction: str) -> str:
            """DETERMINISTIC: Observe page and return element selectors only - no extraction."""
            logger.info(f"🔍 TOOL CALLED: observe_page_structure with instruction='{instruction}'")
            result = await ctx.deps.stagehand_tool.observe(instruction)
            if result["success"]:
                observations = result["data"]
                ctx.deps.state.execution_history.append(f"Observed: {instruction}")

                # Extract just selectors/locations
                selectors = []
                if isinstance(observations, list):
                    for obs in observations:
                        if 'selector' in obs:
                            selectors.append(obs['selector'])

                logger.success(f"✅ Found {len(selectors)} elements")
                return f"✅ Found elements at selectors: {selectors}"
            return f"❌ Observation failed: {result['error']}"

        @self.agent.tool
        async def extract_at_selector(ctx: RunContext[AgentDependencies], selector: str, description: str = "") -> str:
            """DETERMINISTIC: Extract content at specific selector."""
            logger.info(f"📋 TOOL CALLED: extract_at_selector with selector='{selector}'")

            result = await ctx.deps.stagehand_tool.extract_dom(
                instruction=f"Extract text content from element at {selector}",
                selectors=[selector]
            )

            if result["success"] and result.get("data"):
                data = result["data"]
                extracted_text = data.get("combinedText", "") or data.get("content", "")

                if extracted_text:
                    # Create extraction state
                    extraction = ExtractionState(
                        has_content=True,
                        content_type="dom",
                        extracted_text=extracted_text,
                        extraction_method=f"selector: {selector}",
                        verified=True
                    )
                    ctx.deps.state.last_extraction = extraction
                    ctx.deps.state.successful_extractions.append(extraction)

                    logger.success(f"✅ Extracted {len(extracted_text)} characters")
                    return f"✅ Extracted content ({len(extracted_text)} chars): {extracted_text}"

            return f"❌ No content found at selector: {selector}"

        @self.agent.tool
        async def perform_action(ctx: RunContext[AgentDependencies], action: str) -> str:
            """Perform a web action - returns action completion, not result."""
            logger.info(f"🎬 TOOL CALLED: perform_action with action='{action}'")

            result = await ctx.deps.stagehand_tool.act(action)
            if result["success"]:
                ctx.deps.state.execution_history.append(f"Performed action: {action}")
                logger.success(f"✅ Action performed: {action}")
                return f"✅ Action performed successfully. (Note: This confirms the action was done, not that we have the result)"

            return f"❌ Action failed: {result['error']}"

        @self.agent.tool
        async def verify_extraction_success(ctx: RunContext[AgentDependencies]) -> str:
            """Check if we successfully extracted the content we need."""
            logger.info("✔️ TOOL CALLED: verify_extraction_success")

            if ctx.deps.state.last_extraction and ctx.deps.state.last_extraction.has_content:
                extraction = ctx.deps.state.last_extraction
                logger.success(f"✅ Have extracted content: {extraction.content_type} method, {len(extraction.extracted_text)} chars")
                return f"✅ Successfully extracted content using {extraction.extraction_method}"

            attempted = ", ".join(ctx.deps.state.attempted_methods) if ctx.deps.state.attempted_methods else "none"
            logger.warning(f"❌ No successful extraction yet. Attempted methods: {attempted}")
            return f"❌ No content extracted yet. Attempted methods: {attempted}"

        @self.agent.tool
        async def save_generation_result(ctx: RunContext[AgentDependencies], prompt: str, response: str) -> str:
            """Save a generation result to JSONL file."""
            logger.info("💾 TOOL CALLED: save_generation_result")

            result = {
                "uuid": str(uuid.uuid4()),
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "user_goal": ctx.deps.state.user_goal,
                "extraction_method": ctx.deps.state.last_extraction.extraction_method if ctx.deps.state.last_extraction else "unknown"
            }

            save_to_jsonl(result)
            ctx.deps.state.generation_results.append(result)

            return f"✅ Saved generation result with UUID: {result['uuid']}"

        @self.agent.tool
        async def wait_for_user_input(ctx: RunContext[AgentDependencies], message: str = "Waiting for user to complete manual action...") -> str:
            """Wait for user to complete manual actions like login."""
            logger.info(f"⏰ TOOL CALLED: wait_for_user_input with message='{message}'")
            ctx.deps.state.execution_history.append(f"Waiting for user: {message}")

            result_message = await wait_for_user_input_with_timer(message)

            # Check if this was for authentication
            if "log in" in message.lower() or "auth" in message.lower():
                ctx.deps.state.authenticated = True

            ctx.deps.state.execution_history.append("User input received, continuing automation")
            return result_message

    def _get_system_prompt(self) -> str:
        return """
        You are a web automation agent that helps users generate content based on their requests.

        CORE PRINCIPLES:
        1. DETERMINISTIC EXTRACTION: First locate elements, then extract at those locations
        2. VERIFY SUCCESS: Always check if you actually got the content before proceeding
        3. STATE TRACKING: Keep track of what you've extracted, not just what you've attempted
        4. ACTION vs RESULT: perform_action confirms the action was done, not that you got the result

        WORKFLOW PHASES:
        Phase 1: LOCATE - Use observe_page_structure to find element selectors
        Phase 2: EXTRACT - Use extract_at_selector with specific selectors from Phase 1
        Phase 3: VERIFY - Use verify_extraction_success to confirm you have content
        Phase 4: SAVE - Use save_generation_result to save prompt/response pairs

        TOOL USAGE RULES:
        - observe_page_structure: Returns selectors only - use this FIRST to find elements
        - extract_at_selector: Extract content at specific selector - use AFTER observing
        - verify_extraction_success: Check if extraction worked - use AFTER extracting
        - perform_action: Do page interactions - this does NOT get results
        - save_generation_result: Save each prompt/response pair to JSONL

        AUTHENTICATION:
        - If you see login/sign-in pages, use wait_for_user_input immediately
        - Track authentication state to avoid re-checking

        COMPLETION CRITERIA:
        - Task is complete when you have saved generation results
        - If extraction fails after trying different selectors, ask for user help
        - Do not mark complete without actual extracted content

        EXTRACTION STRATEGY:
        1. Try the most specific selector first
        2. If that fails, try parent elements
        3. If DOM extraction fails completely, note this and try alternative methods
        4. Always verify you have content before proceeding
        """


class AgentExecutor:
    """Executes a single iteration of the PydanticAI agent."""

    def __init__(self, pydantic_agent: PydanticAIWebAutomationAgent) -> None:
        self.pydantic_agent = pydantic_agent

    async def execute_iteration(self, deps: AgentDependencies) -> TaskStatus:
        """Execute one iteration of the agent."""
        prompt = f"""
        USER GOAL: {deps.state.user_goal}

        CURRENT STATE:
        - URL: {deps.state.current_url or 'None'}
        - Authenticated: {deps.state.authenticated}
        - Last extraction: {'Success' if deps.state.last_extraction and deps.state.last_extraction.has_content else 'None'}
        - Successful extractions: {len(deps.state.successful_extractions)}
        - Results saved: {len(deps.state.generation_results)}

        RECENT HISTORY: {deps.state.execution_history[-3:] if deps.state.execution_history else 'None'}

        Determine the next action based on:
        1. Have we successfully extracted content? (verify_extraction_success)
        2. If yes, can we proceed to the next prompt?
        3. If no, what selector should we try next?
        4. Are we done with all prompts?
        """

        logger.info("🤖 EXECUTING ITERATION")
        result = await self.pydantic_agent.agent.run(prompt, deps=deps)
        status = result.output

        # Update state in status
        status.state = deps.state

        # Log the decision
        logger.info(f"🎯 Decision: {status.reasoning}")
        logger.info(f"✅ Completed: {status.completed}")
        logger.info(f"➡️ Next action: {status.next_action}")

        return status


class AgentManager:
    """Manages the execution of the web automation agent."""

    def __init__(self, max_iterations: int = MAX_ITERATIONS) -> None:
        self.max_iterations = max_iterations
        self.pydantic_agent = PydanticAIWebAutomationAgent()
        self.executor = AgentExecutor(self.pydantic_agent)

    async def execute_task(self, user_goal: str) -> TaskStatus:
        """Execute task with state tracking."""
        stagehand_tool = StagehandTool()

        # Initialize
        console.print(Panel(
            "🚀 Initializing browser session...",
            title="Startup",
            border_style="blue",
        ))

        init_result = await stagehand_tool.initialize()
        if not init_result["success"]:
            console.print(f"❌ [bold red]Failed to initialize: {init_result['error']}[/bold red]")
            return TaskStatus(
                completed=False,
                reasoning=f"Failed to initialize: {init_result['error']}",
            )

        console.print(f"✅ [bold green]Browser initialized successfully[/bold green]")

        # Create initial state
        state = TaskState(user_goal=user_goal)

        # Initialize dependencies
        deps = AgentDependencies(
            stagehand_tool=stagehand_tool,
            state=state
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

                if status.completed:
                    console.print(Panel(
                        f"✅ Task completed! Generated {len(status.state.generation_results)} results.",
                        title="Success",
                        border_style="green",
                    ))

                    # Show summary
                    console.print(f"\n📊 [bold cyan]Results saved to: {JSONL_OUTPUT_FILE}[/bold cyan]")
                    console.print(f"📝 [bold cyan]Total generations: {len(status.state.generation_results)}[/bold cyan]\n")

                    return status

                # Update dependencies state for next iteration
                deps.state = status.state

            # Reached max iterations
            status.reasoning += f" (Reached maximum iterations: {self.max_iterations})"
            return status

        finally:
            console.print("🧹 [bold blue]Cleaning up browser session...[/bold blue]")
            await stagehand_tool.close()


async def main() -> None:
    """Main execution function with interactive initialization."""
    # Check environment
    if not os.getenv("MODEL_API_KEY"):
        console.print("❌ [bold red]Please set MODEL_API_KEY environment variable[/bold red]")
        return

    # Display banner
    console.print(Panel(
        Text("🎭 AI Generation Assistant", style="bold magenta", justify="center"),
        subtitle="Powered by Stagehand Browser Automation",
        border_style="magenta",
    ))

    # Get user goal interactively
    user_goal = await get_user_goal()

    # Execute with user's goal
    manager = AgentManager(max_iterations=MAX_ITERATIONS)

    try:
        final_status = await manager.execute_task(user_goal)

        # Display final results
        console.print("\n")
        console.print(Panel(
            f"""
✅ [bold green]Completed:[/bold green] {final_status.completed}
📝 [bold blue]Generated:[/bold blue] {len(final_status.state.generation_results)} items
💭 [bold yellow]Summary:[/bold yellow] {final_status.reasoning}
            """.strip(),
            title="📊 FINAL RESULT",
            border_style="green" if final_status.completed else "red",
        ))

    except Exception as e:
        console.print(Panel(
            f"❌ Error during execution: {e}",
            title="Error",
            border_style="red",
        ))
        raise


if __name__ == "__main__":
    console.print(Panel(
        Text("🎭 Welcome to AI Generation Assistant!", style="bold green", justify="center"),
        subtitle="🤖 Interactive content generation with browser automation",
        border_style="green",
    ))

    try:
        uvloop.run(main())
    except KeyboardInterrupt:
        console.print("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unhandled exception: {e}")
        raise
