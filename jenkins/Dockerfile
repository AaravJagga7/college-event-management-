# ==============================================================================
# Custom Jenkins Server with Python 3, Pytest, Docker CLI & Node.js
# ==============================================================================
FROM jenkins/jenkins:lts

USER root

# Install system dependencies, Python 3, pip, venv, and Docker CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    gnupg \
    lsb-release \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI for container builds inside pipeline
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js for Vercel CLI deployments
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g vercel@latest && \
    rm -rf /var/lib/apt/lists/*

# Allow jenkins user to communicate with host Docker daemon
RUN usermod -aG root jenkins

USER jenkins

EXPOSE 8080 50000

