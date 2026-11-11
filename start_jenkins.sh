#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "🏗️ RVCE 8th Mile - Automated Jenkins Server Launcher"
echo "======================================================================"

JENKINS_PORT=${JENKINS_PORT:-8080}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH."
    echo "👉 Please install Docker Desktop for Mac (https://www.docker.com/products/docker-desktop/)"
    echo "   Or run Jenkins directly with Java: java -jar jenkins.war --httpPort=${JENKINS_PORT}"
    exit 1
fi

echo "📦 Starting Jenkins Server container on port ${JENKINS_PORT}..."
cd "${PROJECT_DIR}/jenkins"

if command -v docker-compose &> /dev/null; then
    docker-compose up -d --build
else
    docker compose up -d --build
fi

echo ""
echo "⏳ Waiting for Jenkins to initialize and generate admin key..."
sleep 10

# Attempt to fetch admin password
for i in {1..30}; do
    PASSWORD=$(docker exec rvce_jenkins_server cat /var/jenkins_home/secrets/initialAdminPassword 2>/dev/null || true)
    if [ -n "$PASSWORD" ]; then
        break
    fi
    sleep 2
done

echo "======================================================================"
echo "🎉 Jenkins Server is Running Successfully!"
echo "======================================================================"
echo "🔗 Access URL:            http://localhost:${JENKINS_PORT}"
if [ -n "$PASSWORD" ]; then
    echo "🔑 Initial Admin Password: ${PASSWORD}"
else
    echo "🔑 Fetch Admin Password:   docker exec rvce_jenkins_server cat /var/jenkins_home/secrets/initialAdminPassword"
fi
echo "======================================================================"
echo "🚀 Next Steps to Run Your Pipeline:"
echo " 1. Open http://localhost:${JENKINS_PORT} in your browser."
echo " 2. Paste the Initial Admin Password."
echo " 3. Click 'Install Suggested Plugins'."
echo " 4. Create New Item -> Select 'Pipeline' -> Name: 'rvce-event-mlops'."
echo " 5. Under Pipeline definition, select 'Pipeline script from SCM' -> Git."
echo " 6. Set Script Path to 'Jenkinsfile' and click 'Save' -> 'Build Now'!"
echo "======================================================================"

# Open in macOS browser if supported
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:${JENKINS_PORT}" 2>/dev/null || true
fi

