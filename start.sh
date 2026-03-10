#!/bin/bash

# ─── Hybrid RAG - Start Script ───
# Starts both the FastAPI backend and the React frontend

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV_DIR="$PROJECT_DIR/venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}          Hybrid RAG System - Starting Up            ${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}[ERROR] Virtual environment not found at $VENV_DIR${NC}"
    echo -e "${YELLOW}Run: python3 -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt${NC}"
    exit 1
fi

# Check if node_modules exist
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}[INFO] Installing frontend dependencies...${NC}"
    cd "$FRONTEND_DIR" && npm install
fi

# Cleanup function to kill background processes on exit
cleanup() {
    echo -e "\n${YELLOW}[INFO] Shutting down...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null && echo -e "${GREEN}[INFO] Backend stopped${NC}"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null && echo -e "${GREEN}[INFO] Frontend stopped${NC}"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# ─── Start Backend ───
echo -e "\n${GREEN}[1/2] Starting FastAPI backend on http://localhost:8000 ...${NC}"
cd "$BACKEND_DIR"
source "$VENV_DIR/bin/activate"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 3

# ─── Start Frontend ───
echo -e "${GREEN}[2/2] Starting React frontend on http://localhost:3000 ...${NC}"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo -e "\n${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Backend :  http://localhost:8000${NC}"
echo -e "${GREEN}  Frontend:  http://localhost:3000${NC}"
echo -e "${GREEN}  API Docs:  http://localhost:8000/docs${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Press Ctrl+C to stop both servers${NC}"
echo ""

# Wait for both processes
wait
