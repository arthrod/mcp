#!/usr/bin/env python3

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_connection():
    """Test if MCP server can connect and list tools."""
    print("🔧 Testing MCP server connection...")
    
    # Configure the Browser MCP server
    browser_mcp_server = StdioServerParameters(
        command="node",
        args=["dist/index.js"],
        env=os.environ.copy()
    )

    print("📡 Starting MCP server...")
    
    try:
        # Start server and connect client
        async with stdio_client(browser_mcp_server) as (read, write):
            async with ClientSession(read, write) as session:
                print("🔗 Initializing MCP session...")
                await session.initialize()
                
                print("📋 Listing available tools...")
                tools_result = await session.list_tools()
                
                tool_names = [tool.name for tool in tools_result.tools]
                print(f"✅ Available tools: {tool_names}")
                
                return True
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_connection())
    if success:
        print("🎉 MCP connection test passed!")
    else:
        print("💥 MCP connection test failed!")