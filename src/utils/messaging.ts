import { WebSocket } from "ws";

export function createSocketMessageSender(ws: WebSocket) {
  return {
    sendSocketMessage: async (type: string, payload: any, options: { timeoutMs?: number } = {}) => {
      const { timeoutMs = 30000 } = options;
      
      return new Promise((resolve, reject) => {
        const messageId = Math.random().toString(36).substring(7);
        const message = { id: messageId, type, payload };
        
        const timeout = setTimeout(() => {
          reject(new Error(`Message timeout after ${timeoutMs}ms`));
        }, timeoutMs);
        
        const handleMessage = (data: any) => {
          try {
            const response = JSON.parse(data.toString());
            if (response.id === messageId) {
              clearTimeout(timeout);
              ws.off('message', handleMessage);
              resolve(response.result);
            }
          } catch (e) {
            // Ignore parsing errors for non-matching messages
          }
        };
        
        ws.on('message', handleMessage);
        ws.send(JSON.stringify(message));
      });
    }
  };
}