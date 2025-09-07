#!/bin/bash

# Smart Attendance System Startup Script

echo "🚀 Starting Smart Attendance System..."

# Check if Python virtual environment exists
if [ ! -d "macenv" ]; then
    echo "❌ Python virtual environment not found!"
    echo "Please run: python -m venv macenv && source macenv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if Node.js dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start backend server in background
echo "🐍 Starting FastAPI backend server..."
source macenv/bin/activate
python server.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend development server
echo "⚛️ Starting React frontend..."
cd frontend
npm start &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

echo "✅ System started successfully!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
wait
