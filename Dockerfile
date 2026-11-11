# ==============================================================================
# Multi-Stage Production Dockerfile for RVCE Event Intelligence Service
# ==============================================================================

# Stage 1: Build & Dependency Resolution
FROM python:3.9-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Minimal Production Runtime
FROM python:3.9-slim AS runner

# Production Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    PORT=8000 \
    HOST=0.0.0.0

# Install runtime dependencies for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy python dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application code, model artifacts and baseline data
COPY --chown=appuser:appgroup app /app/app
COPY --chown=appuser:appgroup reports/models /app/reports/models
COPY --chown=appuser:appgroup reports/champion_metrics.json /app/reports/champion_metrics.json
COPY --chown=appuser:appgroup features_ready.csv /app/features_ready.csv
COPY --chown=appuser:appgroup config.yaml /app/config.yaml

# Switch to non-root user
USER appuser

# Expose HTTP port
EXPOSE 8000

# Docker native healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Execute FastAPI application with high-throughput Uvicorn workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]

