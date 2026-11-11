#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "⚡ RVCE 8th Mile - Jenkins Master-Slave Architecture & Multibranch Launcher"
echo "======================================================================"

JENKINS_PORT=${JENKINS_PORT:-8080}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not running."
    echo "👉 Please install Docker Desktop (https://www.docker.com/products/docker-desktop/)"
    exit 1
fi

# Ensure SSH keypair exists for Master -> Slave Agent connection
mkdir -p "${PROJECT_DIR}/jenkins/keys"
if [ ! -f "${PROJECT_DIR}/jenkins/keys/agent_rsa" ]; then
    echo "🔑 Generating master-slave SSH keypair..."
    ssh-keygen -t rsa -b 2048 -f "${PROJECT_DIR}/jenkins/keys/agent_rsa" -N "" -q
    chmod 600 "${PROJECT_DIR}/jenkins/keys/agent_rsa"
    chmod 644 "${PROJECT_DIR}/jenkins/keys/agent_rsa.pub"
fi

echo "📦 Launching Master (Controller) and Slave (Agent) via Docker Compose..."
cd "${PROJECT_DIR}/jenkins"

if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Neither 'docker compose' nor 'docker-compose' found."
    exit 1
fi

$COMPOSE_CMD up -d --build

echo ""
echo "⏳ Waiting for Jenkins Master Controller to complete startup..."
sleep 10

# Attempt to fetch admin password
for i in {1..30}; do
    PASSWORD=$(docker exec rvce_jenkins_controller cat /var/jenkins_home/secrets/initialAdminPassword 2>/dev/null || true)
    if [ -n "$PASSWORD" ]; then
        break
    fi
    sleep 2
done

echo "======================================================================"
echo "🎉 Jenkins Master-Slave Cluster is Live!"
echo "======================================================================"
echo "👑 Master (Controller):    http://localhost:${JENKINS_PORT}"
echo "💻 Slave Worker Node:      rvce_jenkins_agent (SSH Port 22)"
echo "🌿 Multibranch Job:        rvce-event-intelligence-multibranch"
echo "   Auto-scans:             https://github.com/AaravJagga7/college-event-management-.git"
if [ -n "$PASSWORD" ]; then
    echo "🔑 Admin Password:         ${PASSWORD}"
else
    echo "🔑 Admin Password:         docker exec rvce_jenkins_controller cat /var/jenkins_home/secrets/initialAdminPassword"
fi
echo "======================================================================"
echo "📋 Architecture Overview:"
echo "   - Master: 0 executors (pure orchestrator & web UI)"
echo "   - Slave:  2 executors (runs all Python, Pytest, Docker & Vercel builds)"
echo "======================================================================"
