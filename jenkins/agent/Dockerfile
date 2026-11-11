# ==============================================================================
# Jenkins Slave Agent Dockerfile (MLOps Worker Node)
# ==============================================================================
# Dedicated compute agent running Python 3.9+, Docker CLI, Node.js, and Vercel CLI.
# Connects to Controller via SSH on port 22 and executes all pipeline jobs.
# ==============================================================================
FROM jenkins/ssh-agent:latest-jdk17

USER root

# Install system dependencies, Python 3, pip, venv, git, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    ca-certificates \
    gnupg \
    lsb-release \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI to enable docker builds inside pipelines (via mounted host socket)
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js & Vercel CLI for automated frontend/serverless deployments
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g vercel@latest && \
    rm -rf /var/lib/apt/lists/*

# Grant jenkins user access to host Docker daemon socket
RUN usermod -aG root jenkins && \
    mkdir -p /home/jenkins/agent && \
    chown -R jenkins:jenkins /home/jenkins/agent

EXPOSE 22
