#!/bin/bash
# Start the Earthquake Prediction Dashboard server

echo "🌍 Starting Earthquake Prediction Dashboard..."
echo "This may take a few moments..."

# Kill any existing server processes
pkill -f serve.py 2>/dev/null

# Start the server in the background
cd /workspace && python3 serve.py > server.log 2>&1 &

# Wait a moment for the server to start
sleep 3

# Check if the server is running
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ Server is running successfully!"
    echo "🌐 Access the dashboard at: http://localhost:8000"
    echo ""
    echo "API Endpoints:"
    echo "   GET  http://localhost:8000/api/prediction-stats"
    echo "   GET  http://localhost:8000/api/recent-earthquakes" 
    echo "   POST http://localhost:8000/api/predict"
    echo ""
    echo "To stop the server, run: pkill -f serve.py"
else
    echo "❌ Server failed to start. Check server.log for details."
    cat /workspace/server.log
fi