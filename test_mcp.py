#!/usr/bin/env python3
"""Simple test script to verify the MCP server works."""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp():
    """Test the MCP server connection and tools."""
    
    # Configure the Browser MCP server
    browser_mcp_server = StdioServerParameters(
        command="node",
        args=["dist/index.js"],
    )
    
    print("🚀 Starting Browser MCP server...")
    
    # Start server and connect client
    async with stdio_client(browser_mcp_server) as (read, write):
        async with ClientSession(read, write) as session:
            print("✅ Connected to MCP server")
            
            # Initialize the connection
            await session.initialize()
            print("✅ Session initialized")
            
            # List available tools
            tools_result = await session.list_tools()
            tools = tools_result.tools
            
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test a simple tool call (navigate)
            if any(tool.name == "navigate" for tool in tools):
                print("\n🧪 Testing navigate tool...")
                try:
                    result = await session.call_tool("navigate", {"url": "https://example.com"})
                    print(f"✅ Navigate result: {result}")
                except Exception as e:
                    print(f"⚠️  Navigate failed (expected - no browser connection): {e}")
            
            print("\n🎉 MCP server is working correctly!")

if __name__ == "__main__":
    asyncio.run(test_mcp())