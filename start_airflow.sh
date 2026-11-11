#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "🌪️ RVCE 8th Mile - Apache Airflow MLOps Pipeline Orchestrator"
echo "======================================================================"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${PROJECT_DIR}/.venv/bin:$PATH"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export NO_PROXY="*"
export no_proxy="*"
export AIRFLOW_HOME="${PROJECT_DIR}/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_DIR}/airflow/dags"
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8081

AIRFLOW_BIN="${PROJECT_DIR}/.venv/bin/airflow"

if [ ! -f "${AIRFLOW_BIN}" ]; then
    echo "❌ Airflow not found in virtualenv. Installing..."
    "${PROJECT_DIR}/.venv/bin/pip" install apache-airflow==2.8.3
fi

echo "📦 Initializing Airflow Metadata Database..."
"${AIRFLOW_BIN}" db migrate

# Create Admin user if not exists
echo "👤 Provisioning Airflow Admin user (admin / admin)..."
"${AIRFLOW_BIN}" users create \
    --username admin \
    --firstname Aarav \
    --lastname Jagga \
    --role Admin \
    --email aarav@rvce.edu.in \
    --password admin 2>/dev/null || echo "✓ Admin user already exists."

echo "🚀 Starting Apache Airflow Standalone Server (Port 8081)..."
echo "======================================================================"
echo "🌐 Airflow Web UI:        http://127.0.0.1:8081"
echo "👤 Username:              admin"
echo "🔑 Password:              admin"
echo "🌿 Orchestrated DAG:      rvce_event_intelligence_mlops_pipeline"
echo "======================================================================"
echo "📋 Coexisting Cluster Ports:"
echo "   - Port 8000: FastAPI ML Inference & Dashboard (/ui)"
echo "   - Port 8080: Jenkins CI/CD Controller"
echo "   - Port 8081: Apache Airflow Orchestrator"
echo "======================================================================"

exec "${AIRFLOW_BIN}" standalone
