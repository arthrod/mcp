#!/bin/bash

# Start script for Browser MCP automation

echo "🚀 Starting Browser MCP automation setup..."

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping all processes..."
    kill $WS_PID $PYTHON_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Start WebSocket bridge server
echo "📡 Starting WebSocket bridge server..."
node ws_only.js &
WS_PID=$!

# Wait for WebSocket server to start
sleep 2

echo "🔌 WebSocket bridge server started (PID: $WS_PID)"
echo ""
echo "📋 Setup Instructions:"
echo "1. Open Chrome browser"
echo "2. Install the Browser MCP extension from the Chrome Web Store"
echo "3. Click the Browser MCP extension icon in the toolbar"
echo "4. Click 'Connect' to connect to the WebSocket server"
echo "5. The Python script will now work properly"
echo ""
echo "🐍 Starting Python automation script..."

# Start Python script
python trick.py &
PYTHON_PID=$!

# Wait for processes
wait $WS_PID $PYTHON_PID