#!/usr/bin/env bash
# ==============================================================================
# Dagster Webserver & Orchestrator Launcher
# ==============================================================================
set -e

PORT=${1:-3000}
HOST="127.0.0.1"

echo "================================================================================"
echo "🚀 Starting Dagster Webserver & Model Orchestrator (with lakeFS)"
echo "================================================================================"
echo "📍 Code Location: dagster_orchestration.definitions"
echo "🌐 UI URL:        http://${HOST}:${PORT}"
echo "================================================================================"

exec .venv/bin/dagster-webserver \
  -m dagster_orchestration \
  -h "${HOST}" \
  -p "${PORT}"
