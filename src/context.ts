import { WebSocket } from "ws";
import { createSocketMessageSender } from "./utils/messaging";
import { mcpConfig } from "./config";

const noConnectionMessage = `No connection to browser extension. In order to proceed, you must first connect a tab by clicking the Browser MCP extension icon in the browser toolbar and clicking the 'Connect' button.`;

export class Context {
  private _ws: WebSocket | undefined;

  constructor() {
    // Connect to existing WebSocket server as a client
    this.connectToWebSocketServer();
  }

  private async connectToWebSocketServer() {
    try {
      // Connect to the running WebSocket server on port 9009
      const ws = new WebSocket('ws://localhost:9009');
      
      ws.on('open', () => {
        console.log('🔗 MCP server connected to WebSocket server');
        // Identify as MCP server
        ws.send(JSON.stringify({ type: 'mcp_server_connect' }));
        this._ws = ws;
      });
      
      ws.on('error', (error) => {
        console.log('❌ WebSocket connection failed:', error.message);
      });
      
      ws.on('close', () => {
        console.log('🔌 WebSocket connection closed');
        this._ws = undefined;
      });
      
    } catch (error) {
      console.log('❌ Failed to connect to WebSocket server:', error);
    }
  }

  get ws(): WebSocket {
    if (!this._ws) {
      throw new Error(noConnectionMessage);
    }
    return this._ws;
  }

  set ws(ws: WebSocket) {
    this._ws = ws;
  }

  hasWs(): boolean {
    return !!this._ws;
  }

  async sendSocketMessage(
    type: string,
    payload: any,
    options: { timeoutMs?: number } = { timeoutMs: 30000 },
  ) {
    const { sendSocketMessage } = createSocketMessageSender<SocketMessageMap>(
      this.ws,
    );
    try {
      return await sendSocketMessage(type, payload, options);
    } catch (e) {
      if (e instanceof Error && e.message === mcpConfig.errors.noConnectedTab) {
        throw new Error(noConnectionMessage);
      }
      throw e;
    }
  }

  async close() {
    if (!this._ws) {
      return;
    }
    await this._ws.close();
  }
}
