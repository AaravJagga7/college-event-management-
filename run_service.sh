#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "🏛️ RVCE 8th Mile Event Intelligence API - Local Runner"
echo "======================================================================"

PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}

if [ -f ".venv/bin/activate" ]; then
    echo "Activating local virtual environment (.venv)..."
    source .venv/bin/activate
fi

echo "Running test suite before starting server..."
pytest -q

echo "Starting Uvicorn server at http://${HOST}:${PORT} (Docs: http://localhost:${PORT}/docs)..."
uvicorn app.main:app --host "${HOST}" --port "${PORT}" --reload

