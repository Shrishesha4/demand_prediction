#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Starting E-commerce Demand Forecasting GUI..."
echo ""

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BE_DIR="$ROOT_DIR/be"
GUI_DIR="$ROOT_DIR/gui"
BE_VENV="$BE_DIR/.venv"

PY_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PY_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PY_BIN=python
else
    echo "❌ Error: python3 not found. Install Python 3.11 or newer."
    exit 1
fi

if [ ! -d "$BE_VENV" ]; then
    echo "🛠 Creating virtual environment for backend at $BE_VENV..."
    "$PY_BIN" -m venv "$BE_VENV"
fi

source "$BE_VENV/bin/activate"

echo "📦 Installing backend Python requirements..."

python -m pip install --upgrade pip setuptools wheel >/dev/null
if [ -f "$BE_DIR/requirements.txt" ]; then
    python -m pip install -r "$BE_DIR/requirements.txt"
else
    echo "⚠️  No requirements.txt found in $BE_DIR — skipping pip install"
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "❌ Error: npm not found. Install Node.js/npm (Node >= 20 recommended)."
    exit 1
fi

if [ ! -d "$GUI_DIR/node_modules" ]; then
    echo "📥 Installing frontend dependencies (npm ci)..."
    (cd "$GUI_DIR" && npm ci)
fi

cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
        kill "$FRONTEND_PID" || true
    fi
    if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
        kill "$BACKEND_PID" || true
    fi
    deactivate >/dev/null 2>&1 || true
    exit 0
}

trap cleanup INT TERM

echo "📡 Starting FastAPI backend on http://localhost:8000..."

cd "$BE_DIR"

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > /dev/null 2>&1 &
BACKEND_PID=$!
cd "$ROOT_DIR"

echo "⏳ Waiting for backend to become healthy..."
for i in {1..12}; do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "✅ Backend is healthy"
        break
    fi
    sleep 1
done

echo "🎨 Starting Svelte frontend on http://localhost:5173..."
cd "$GUI_DIR"

npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
cd "$ROOT_DIR"

sleep 2

echo ""
echo "✅ Both servers are running!"
echo ""
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

wait
