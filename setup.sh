#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# Road Surface Damage Detection — Local Setup & Run Script
# Usage:
#   bash setup.sh          → Full first-time setup + start everything
#   bash setup.sh backend  → Backend only
#   bash setup.sh frontend → Frontend only
#   bash setup.sh test     → Run backend test suite
#   bash setup.sh docker   → Start via Docker Compose
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
MODEL_DIR="$PROJECT_ROOT/model"
MODEL_FILE="$MODEL_DIR/best.pt"

log()  { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
err()  { echo -e "${RED}❌ $*${NC}"; }
info() { echo -e "${BLUE}ℹ️  $*${NC}"; }
step() { echo -e "\n${BOLD}── $* ──────────────────────────────────${NC}"; }

banner() {
  echo -e "${BLUE}"
  echo "  ██████╗  ██████╗  █████╗ ██████╗ "
  echo "  ██╔══██╗██╔═══██╗██╔══██╗██╔══██╗"
  echo "  ██████╔╝██║   ██║███████║██║  ██║"
  echo "  ██╔══██╗██║   ██║██╔══██║██║  ██║"
  echo "  ██║  ██║╚██████╔╝██║  ██║██████╔╝"
  echo "  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ "
  echo "  Road Surface Damage Detection System"
  echo -e "${NC}"
}

check_python() {
  if ! command -v python3 &>/dev/null; then
    err "Python 3 not found. Install from https://python.org"
    exit 1
  fi
  PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
  MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
  MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
  if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    err "Python 3.10+ required. Found: $PY_VERSION"
    exit 1
  fi
  log "Python $PY_VERSION"
}

check_node() {
  if ! command -v node &>/dev/null; then
    err "Node.js not found. Install from https://nodejs.org (v18+)"
    exit 1
  fi
  NODE_VERSION=$(node --version)
  log "Node.js $NODE_VERSION"
}

check_model() {
  if [ ! -f "$MODEL_FILE" ]; then
    warn "Model file not found: $MODEL_FILE"
    warn "The API will start but predictions will fail."
    warn "Place your trained best.pt file in: $MODEL_DIR/"
    warn "Run the Colab notebook first: colab_notebook/road_damage_training.py"
    return 1
  else
    SIZE=$(du -sh "$MODEL_FILE" | cut -f1)
    log "Model found: $MODEL_FILE ($SIZE)"
    return 0
  fi
}

setup_backend() {
  step "Setting up Backend"
  cd "$BACKEND_DIR"

  if [ ! -d "venv" ]; then
    info "Creating virtual environment..."
    python3 -m venv venv
    log "Virtual environment created"
  fi

  source venv/bin/activate

  info "Installing Python dependencies..."
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  log "Python dependencies installed"

  # Create .env if not exists
  if [ ! -f ".env" ]; then
    cp .env.example .env
    log ".env created from .env.example"
  fi

  # Create logs dir
  mkdir -p logs
  log "Log directory ready"

  # Check model
  check_model || true

  deactivate
  log "Backend setup complete"
}

setup_frontend() {
  step "Setting up Frontend"
  cd "$FRONTEND_DIR"

  check_node

  info "Installing Node.js dependencies..."
  npm install --silent
  log "Node dependencies installed"

  if [ ! -f ".env" ]; then
    cp .env.example .env
    log ".env created"
    info "Edit frontend/.env to set VITE_API_URL if needed"
  fi

  log "Frontend setup complete"
}

start_backend() {
  step "Starting Backend"
  cd "$BACKEND_DIR"

  if [ ! -d "venv" ]; then
    warn "Backend not set up. Run: bash setup.sh"
    exit 1
  fi

  source venv/bin/activate
  info "Starting FastAPI on http://localhost:8000"
  info "Press Ctrl+C to stop"
  echo ""
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
}

start_frontend() {
  step "Starting Frontend"
  cd "$FRONTEND_DIR"

  if [ ! -d "node_modules" ]; then
    warn "Frontend not set up. Run: bash setup.sh"
    exit 1
  fi

  info "Starting Vite dev server on http://localhost:5173"
  info "Press Ctrl+C to stop"
  echo ""
  npm run dev
}

run_tests() {
  step "Running Test Suite"
  cd "$BACKEND_DIR"

  if [ ! -d "venv" ]; then
    err "Backend not set up. Run: bash setup.sh first"
    exit 1
  fi

  source venv/bin/activate
  info "Installing test dependencies..."
  pip install pytest pytest-asyncio httpx -q

  info "Running pytest..."
  python -m pytest tests/ -v --tb=short
  deactivate
}

run_smoke_test() {
  step "Running API Smoke Test"
  cd "$BACKEND_DIR"
  source venv/bin/activate
  python api_test.py --url "${1:-http://localhost:8000}"
  deactivate
}

start_docker() {
  step "Starting with Docker Compose"
  cd "$PROJECT_ROOT"

  if ! command -v docker &>/dev/null; then
    err "Docker not found. Install from https://docker.com"
    exit 1
  fi

  if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null 2>&1; then
    err "docker-compose not found"
    exit 1
  fi

  check_model || warn "Starting without model — predictions will fail"

  info "Building and starting containers..."
  docker compose up --build
}

realtime() {
  step "Starting Real-Time Detection"
  cd "$BACKEND_DIR"

  if [ ! -d "venv" ]; then
    err "Backend not set up. Run: bash setup.sh first"
    exit 1
  fi

  source venv/bin/activate
  python realtime_detection.py --source "${1:-0}" --conf 0.30
  deactivate
}

# ─────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────

banner

MODE="${1:-all}"

case "$MODE" in
  all)
    check_python
    setup_backend
    setup_frontend
    echo ""
    log "Setup complete!"
    echo ""
    info "To start the application:"
    echo "  Terminal 1: bash setup.sh backend"
    echo "  Terminal 2: bash setup.sh frontend"
    echo ""
    info "Or with Docker:"
    echo "  bash setup.sh docker"
    echo ""
    info "API docs: http://localhost:8000/docs"
    info "Frontend: http://localhost:5173"
    ;;
  backend)
    check_python
    if [ ! -d "$BACKEND_DIR/venv" ]; then setup_backend; fi
    start_backend
    ;;
  frontend)
    check_node
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then setup_frontend; fi
    start_frontend
    ;;
  setup-backend)
    check_python
    setup_backend
    ;;
  setup-frontend)
    check_node
    setup_frontend
    ;;
  test)
    check_python
    run_tests
    ;;
  smoke)
    check_python
    run_smoke_test "${2:-}"
    ;;
  docker)
    start_docker
    ;;
  realtime)
    check_python
    realtime "${2:-0}"
    ;;
  *)
    echo "Usage: bash setup.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all           Full first-time setup (default)"
    echo "  backend       Start backend server"
    echo "  frontend      Start frontend dev server"
    echo "  setup-backend Install backend deps only"
    echo "  setup-frontend Install frontend deps only"
    echo "  test          Run pytest test suite"
    echo "  smoke [url]   Run API smoke test"
    echo "  docker        Start via Docker Compose"
    echo "  realtime [src] Start webcam detection"
    exit 1
    ;;
esac
