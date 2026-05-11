#!/bin/bash
# LAYOFF-NET — Start all services
# Usage: bash start.sh

BASE="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         LAYOFF-NET — Starting...         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# 1. ML API (FastAPI / Python)
echo "▶ Starting ML API on port 8000..."
cd "$BASE/server"
pip install fastapi uvicorn pydantic -q
python3 ml_api.py &
ML_PID=$!
echo "  ML API PID: $ML_PID"
sleep 2

# 2. Node.js backend
echo "▶ Starting Node.js backend on port 5000..."
npm install --silent 2>/dev/null
node index.js &
NODE_PID=$!
echo "  Node PID: $NODE_PID"
sleep 1

# 3. React frontend
echo "▶ Starting React frontend on port 3000..."
cd "$BASE/client"
npm install --silent 2>/dev/null
npm start &
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  LAYOFF-NET is running!                  ║"
echo "║  Frontend : http://localhost:3000        ║"
echo "║  Backend  : http://localhost:5000        ║"
echo "║  ML API   : http://localhost:8000        ║"
echo "║  API Docs : http://localhost:8000/docs   ║"
echo "╚══════════════════════════════════════════╝"

wait
