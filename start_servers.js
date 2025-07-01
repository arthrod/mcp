#!/usr/bin/env node

// Simple script to start both WebSocket server for browser extension
// and keep it running while Python script connects via STDIO

import { createWebSocketServer } from './dist/index.js';

const port = 3001;

console.log(`🚀 Starting WebSocket server on port ${port}...`);

try {
  const wss = await createWebSocketServer(port);
  
  wss.on('connection', (ws) => {
    console.log('✅ Browser extension connected!');
    
    ws.on('close', () => {
      console.log('❌ Browser extension disconnected');
    });
    
    ws.on('error', (error) => {
      console.log('❌ WebSocket error:', error);
    });
  });
  
  console.log(`✅ WebSocket server ready on port ${port}`);
  console.log('👆 Connect your browser extension now!');
  
} catch (error) {
  console.error('❌ Failed to start WebSocket server:', error);
  process.exit(1);
}