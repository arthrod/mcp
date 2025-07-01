import asyncio
import uvloop
import os
import types
import inspect
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool, TextContent
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.tools import AgentDepsT

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from loguru import logger
from pydantic_ai.models.openai import OpenAIModel

SYSTEM = """
You are a browser automation tool. Your goal is to help users automate tasks in their web browser.
Use these tools and the others.
- navigate: Navigate to a URL
- go_back: Go back in browser history
- go_forward: Go forward in browser history
- snapshot: Take a snapshot of the page
- click: Click an element
- hover: Hover over an element
- type: Type text into an element
- select_option: Select an option from a dropdown
- press_key: Press a key
- wait: Wait for specified time
- get_console_logs: Get browser console logs
- screenshot: Take a screenshot
"""

model = OpenAIModel("gpt-4.1")
# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    "chatgpt_automation.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {function}:{line} | {message}"
)
logger.add(
    lambda msg: console.print(f"[dim cyan][LOG][/dim cyan] {msg}", end=""),
    level="INFO",
    colorize=True
)

# Global session for the inject_user_request tool
mcp_session: Optional[ClientSession] = None

def convert_schema_to_params(schema: Dict[str, Any]) -> List[inspect.Parameter]:
    """Convert a JSON schema to function parameters."""
    logger.debug(f"Converting schema to params: {schema}")
    parameters = []

    # Add context parameter first
    parameters.append(
        inspect.Parameter(
            name='ctx',
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation='RunContext[AgentDepsT]'
        )
    )

    # Extract properties from schema
    properties = schema.get('properties', {})
    required = schema.get('required', [])

    logger.debug(f"Schema properties: {list(properties.keys())}")
    logger.debug(f"Required fields: {required}")

    for prop_name, prop_schema in properties.items():
        # Determine the type annotation based on schema type
        prop_type = prop_schema.get('type', 'string')
        annotation = str  # default

        if prop_type == 'string':
            annotation = str
        elif prop_type == 'number':
            annotation = float
        elif prop_type == 'integer':
            annotation = int
        elif prop_type == 'boolean':
            annotation = bool
        elif prop_type == 'array':
            annotation = List[Any]
        elif prop_type == 'object':
            annotation = Dict[str, Any]

        logger.debug(f"Parameter {prop_name}: type={prop_type}, required={prop_name in required}")

        # Determine if parameter has default value
        if prop_name in required:
            param = inspect.Parameter(
                name=prop_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation
            )
        else:
            # Optional parameter with None default
            param = inspect.Parameter(
                name=prop_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
                default=None
            )

        parameters.append(param)

    return parameters

def create_function_from_schema(session: ClientSession, name: str, schema: Dict[str, Any]) -> types.FunctionType:
    """Create a function with a signature based on a JSON schema."""
    logger.info(f"Creating function from schema for tool: {name}")

    # Create parameter list from tool schema
    parameters = convert_schema_to_params(schema)

    # Create the signature
    sig = inspect.Signature(parameters=parameters)

    # Create function body
    async def function_body(ctx: RunContext[AgentDepsT], **kwargs) -> str:
        logger.info(f"Calling MCP tool: {name} with args: {kwargs}")

        # Remove None values from kwargs to avoid sending optional params
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        logger.debug(f"Filtered kwargs: {filtered_kwargs}")

        # Call the MCP tool with provided arguments
        result = await session.call_tool(name, arguments=filtered_kwargs)

        logger.debug(f"MCP tool {name} returned: {result}")

        # Handle response content
        if result.content:
            # Concatenate all text content
            text_parts = []
            for content in result.content:
                if isinstance(content, TextContent):
                    text_parts.append(content.text)

            if text_parts:
                response = '\n'.join(text_parts)
                logger.info(f"Tool {name} response: {response[:200]}...")
                return response
            else:
                logger.warning(f"Tool {name} returned no text content")
                return "Tool executed successfully (no text output)"
        else:
            logger.warning(f"Tool {name} returned no content")
            return "Tool executed successfully (no output)"

    # Create the function with the correct signature
    dynamic_function = types.FunctionType(
        function_body.__code__,
        function_body.__globals__,
        name=name,
        argdefs=function_body.__defaults__,
        closure=function_body.__closure__,
    )

    # Add signature and annotations
    dynamic_function.__signature__ = sig  # type: ignore
    dynamic_function.__annotations__ = {param.name: param.annotation for param in parameters}

    return dynamic_function

def pydantic_tool_from_mcp_tool(session: ClientSession, tool: MCPTool) -> Tool[AgentDepsT]:
    """Convert a MCP tool to a Pydantic AI tool."""
    logger.debug(f"Converting MCP tool to Pydantic tool: {tool.name}")
    tool_function = create_function_from_schema(session=session, name=tool.name, schema=tool.inputSchema)
    return Tool(name=tool.name, description=tool.description, function=tool_function, takes_ctx=True)

async def get_tools(session: ClientSession) -> List[Tool[AgentDepsT]]:
    """Get all tools from the MCP session and convert them to Pydantic AI tools."""
    logger.info("Fetching available MCP tools")
    tools_result = await session.list_tools()

    tool_names = [tool.name for tool in tools_result.tools]
    logger.info(f"Available MCP tools: {tool_names}")

    return [pydantic_tool_from_mcp_tool(session, tool) for tool in tools_result.tools]

async def inject_user_request(ctx: RunContext[AgentDepsT], prompt: str) -> str:
    """
    Custom tool to inject user prompt into ChatGPT interface.
    This tool will find the textarea and type the prompt.
    """
    logger.info(f"Injecting user request: {prompt[:100]}...")

    if not mcp_session:
        logger.error("MCP session not available")
        return "Error: MCP session not initialized"

    try:
        # First, take a snapshot to see current state
        logger.debug("Taking initial snapshot")
        snapshot_result = await mcp_session.call_tool("snapshot", {})

        # Find the main textarea (ChatGPT's input field)
        # Usually it has placeholder text like "Send a message..."
        logger.info("Looking for ChatGPT input textarea")

        # Wait longer for the page to fully load and avoid rate limiting
        await mcp_session.call_tool("wait", {"time": 5})

        # Take another snapshot after waiting
        snapshot_result = await mcp_session.call_tool("snapshot", {})
        logger.debug("Snapshot taken after wait")

        # Try to click on the textarea first to focus it
        # ChatGPT's textarea usually has a specific structure
        try:
            logger.info("Attempting to click on textarea")
            await mcp_session.call_tool("click", {
                "element": "Send a message textarea"
            })
        except Exception as e:
            logger.warning(f"Could not click textarea: {e}")
            # Try alternative selectors
            pass

        # Type the prompt
        logger.info("Typing the prompt into textarea")
        type_result = await mcp_session.call_tool("type", {
            "element": "message input textarea",
            "text": prompt
        })

        logger.debug(f"Type result: {type_result}")

        # Press Enter to submit
        logger.info("Pressing Enter to submit prompt")
        await mcp_session.call_tool("press_key", {"key": "Enter"})

        # Wait for response to generate
        logger.info("Waiting for ChatGPT to generate response...")
        await mcp_session.call_tool("wait", {"time": 10})

        # Take a final snapshot to see the response
        logger.debug("Taking final snapshot after response")
        final_snapshot = await mcp_session.call_tool("snapshot", {})

        logger.info("Successfully injected prompt and got response")
        return "Prompt injected successfully. ChatGPT is generating response."

    except Exception as e:
        logger.error(f"Error injecting prompt: {e}", exc_info=True)
        return f"Error injecting prompt: {str(e)}"

async def scrape_chatgpt_response(ctx: RunContext[AgentDepsT]) -> str:
    """
    Tool to scrape the latest ChatGPT response from the page.
    """
    logger.info("Scraping ChatGPT response")

    if not mcp_session:
        logger.error("MCP session not available")
        return "Error: MCP session not initialized"

    try:
        # Take a snapshot to analyze the page
        logger.debug("Taking snapshot for response scraping")
        snapshot_result = await mcp_session.call_tool("snapshot", {})

        if snapshot_result.content:
            snapshot_text = ""
            for content in snapshot_result.content:
                if isinstance(content, TextContent):
                    snapshot_text += content.text

            logger.debug(f"Snapshot length: {len(snapshot_text)} characters")

            # Parse the snapshot to find ChatGPT's response
            # This is a simplified approach - you might need to adjust based on actual structure
            logger.info("Parsing snapshot for ChatGPT response")

            # Return the full snapshot for now - in practice, you'd parse this
            # to extract just the ChatGPT response
            return f"Page snapshot:\n{snapshot_text}"
        else:
            logger.warning("No content in snapshot")
            return "Could not get page content"

    except Exception as e:
        logger.error(f"Error scraping response: {e}", exc_info=True)
        return f"Error scraping response: {str(e)}"

async def main():
    """Main function to run the ChatGPT automation."""
    global mcp_session

    # Set up uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("Starting ChatGPT automation with uvloop")

    console.print(Panel.fit(
        "[bold cyan]ChatGPT Automation with Browser MCP[/bold cyan]\n"
        "[dim]Powered by Pydantic AI and MCP[/dim]",
        border_style="cyan"
    ))

    # Configure the Browser MCP server
    # Using the built JavaScript file
    browser_mcp_server = StdioServerParameters(
        command="node",
        args=[
            "dist/index.js"  # Use the built JavaScript file
        ],
        env=os.environ.copy()
    )

    logger.info("Starting Browser MCP server")

    # Start server and connect client
    async with stdio_client(browser_mcp_server) as (read, write):
        async with ClientSession(read, write) as session:
            mcp_session = session  # Store session globally for custom tools

            # Initialize the connection
            logger.info("Initializing MCP session")
            await session.initialize()

            # Get available tools
            tools = await get_tools(session)

            # Add our custom tools
            custom_tools = [
                Tool(
                    name="inject_user_request",
                    description="Inject a user prompt into ChatGPT's input field and submit it",
                    function=inject_user_request,
                    takes_ctx=True
                ),
                Tool(
                    name="scrape_chatgpt_response",
                    description="Scrape the latest ChatGPT response from the page",
                    function=scrape_chatgpt_response,
                    takes_ctx=True
                )
            ]

            all_tools = tools + custom_tools

            console.print(f"\n[green]Available tools:[/green] {[tool.name for tool in all_tools]}")
            logger.info(f"Total tools available: {len(all_tools)}")

            # Create agent with all tools
            agent = Agent(
                model,
                tools=all_tools
            )

            console.print("\n[yellow]Make sure:[/yellow]")
            console.print("1. The Browser MCP Chrome extension is installed and connected")
            console.print("2. You have a browser tab ready to navigate\n")

            # Main automation loop
            while True:
                # Get user input
                user_prompt = console.input("\n[bold cyan]Enter your prompt for ChatGPT (or 'exit' to quit):[/bold cyan] ")

                if user_prompt.lower() == 'exit':
                    logger.info("User requested exit")
                    break

                logger.info(f"User prompt: {user_prompt}")

                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        # Navigate to ChatGPT
                        task = progress.add_task("[cyan]Navigating to ChatGPT...", total=None)
                        logger.info("Instructing agent to navigate to ChatGPT")

                        nav_response = await agent.run(
                            f"{SYSTEM} Navigate to https://chatgpt.com and wait for it to load completely"
                        )
                        console.print(f"[dim]{nav_response.data}[/dim]")

                        # Wait a bit more for full load
                        await asyncio.sleep(3)

                        # Inject the user's prompt
                        progress.update(task, description="[yellow]Injecting your prompt...")
                        logger.info("Instructing agent to inject user prompt")

                        inject_response = await agent.run(
                            f"Use the inject_user_request tool to type this exact prompt into ChatGPT: '{user_prompt}'"
                        )
                        console.print(f"[dim]{inject_response.data}[/dim]")

                        # Wait for ChatGPT to respond
                        progress.update(task, description="[green]Waiting for ChatGPT response...")
                        await asyncio.sleep(5)  # Adjust based on response time

                        # Scrape the response
                        progress.update(task, description="[blue]Scraping response...")
                        logger.info("Instructing agent to scrape ChatGPT response")

                        scrape_response = await agent.run(
                            f"{SYSTEM} Use the scrape_chatgpt_response tool to get ChatGPT's latest response"
                        )

                    # Display the result
                    console.print("\n[bold green]ChatGPT Response:[/bold green]")
                    console.print(Panel(scrape_response.data, border_style="green"))

                except Exception as e:
                    logger.error(f"Error in automation flow: {e}", exc_info=True)
                    console.print(f"[red]Error: {e}[/red]")

            logger.info("Automation completed")
            console.print("\n[cyan]Goodbye![/cyan]")

if __name__ == "__main__":
    # Run with uvloop
    uvloop.run(main())
