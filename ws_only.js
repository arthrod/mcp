#!/usr/bin/env node

// WebSocket server that bridges MCP server and browser extension
import { WebSocketServer } from "ws";

const port = 9009;
console.log(`🚀 Starting WebSocket bridge server on port ${port}...`);

const wss = new WebSocketServer({ port });

let browserExtension = null;
let mcpServer = null;

wss.on("connection", (ws) => {
  ws.on("message", (message) => {
    try {
      const data = JSON.parse(message.toString());
      
      // Identify client type
      if (data.type === 'mcp_server_connect') {
        mcpServer = ws;
        console.log("🔗 MCP server connected!");
        return;
      }
      
      // If no identification, assume it's browser extension
      if (!browserExtension) {
        browserExtension = ws;
        console.log("✅ Browser extension connected!");
      }
      
      // Route messages between MCP server and browser extension
      if (ws === mcpServer && browserExtension) {
        // Message from MCP server to browser extension
        console.log("📨 MCP→Browser:", data);
        browserExtension.send(message);
      } else if (ws === browserExtension && mcpServer) {
        // Message from browser extension to MCP server
        console.log("📨 Browser→MCP:", data);
        mcpServer.send(message);
      }
      
    } catch (e) {
      console.log("📨 Raw message:", message.toString());
      // Handle non-JSON messages
      if (ws === mcpServer && browserExtension) {
        browserExtension.send(message);
      } else if (ws === browserExtension && mcpServer) {
        mcpServer.send(message);
      }
    }
  });

  ws.on("close", () => {
    if (ws === browserExtension) {
      console.log("❌ Browser extension disconnected");
      browserExtension = null;
    } else if (ws === mcpServer) {
      console.log("❌ MCP server disconnected");
      mcpServer = null;
    }
  });
});

console.log(`✅ WebSocket bridge server ready on port ${port}`);
console.log("👆 Connect your browser extension and MCP server!");

// Keep the process alive
process.on("SIGINT", () => {
  console.log("\n👋 Shutting down WebSocket server...");
  wss.close();
  process.exit(0);
});
