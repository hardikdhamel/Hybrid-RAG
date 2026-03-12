#!/bin/bash

# ─── Hybrid RAG - Start Script ───
# Starts Ollama, FastAPI backend, and React frontend

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

OLLAMA_PID=""
OLLAMA_STARTED_BY_US=false

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
    if [ "$OLLAMA_STARTED_BY_US" = true ] && [ ! -z "$OLLAMA_PID" ]; then
        kill "$OLLAMA_PID" 2>/dev/null && echo -e "${GREEN}[INFO] Ollama stopped${NC}"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# ─── Start Ollama ───
echo -e "\n${GREEN}[1/3] Starting Ollama server ...${NC}"

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}[INFO] Ollama is already running — skipping startup${NC}"
else
    # Check if ollama command exists
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}[ERROR] 'ollama' command not found. Please install Ollama first.${NC}"
        echo -e "${YELLOW}Visit: https://ollama.com/download${NC}"
        exit 1
    fi

    echo -e "${CYAN}[INFO] Starting Ollama server in background...${NC}"
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    OLLAMA_STARTED_BY_US=true

    # Wait for Ollama to be ready (max 30 seconds)
    echo -e "${CYAN}[INFO] Waiting for Ollama to be ready...${NC}"
    MAX_WAIT=30
    WAITED=0
    while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
        sleep 1
        WAITED=$((WAITED + 1))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo -e "${RED}[ERROR] Ollama failed to start within ${MAX_WAIT}s${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}[INFO] Ollama is ready! (took ${WAITED}s)${NC}"
fi

# ─── Start Backend ───
echo -e "\n${GREEN}[2/3] Starting FastAPI backend on http://localhost:8000 ...${NC}"
cd "$BACKEND_DIR"
source "$VENV_DIR/bin/activate"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 3

# ─── Start Frontend ───
echo -e "${GREEN}[3/3] Starting React frontend on http://localhost:3000 ...${NC}"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo -e "\n${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Ollama  :  http://localhost:11434${NC}"
echo -e "${GREEN}  Backend :  http://localhost:8000${NC}"
echo -e "${GREEN}  Frontend:  http://localhost:3000${NC}"
echo -e "${GREEN}  API Docs:  http://localhost:8000/docs${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Press Ctrl+C to stop all servers${NC}"
echo ""

# Wait for all processes
wait
