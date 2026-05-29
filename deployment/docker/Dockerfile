# ============================================================
# Deep RL Platform - Training Container
# ============================================================
# Multi-stage build for minimal production image size.
# Stage 1 (builder): Install dependencies + compile
# Stage 2 (runtime): Copy only necessary artifacts

# ────────── Stage 1: Builder ──────────
FROM python:3.11-slim AS builder

# System dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ────────── Stage 2: Runtime ──────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r rluser && useradd -r -g rluser rluser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/rluser/.local

# Application code
WORKDIR /app
COPY --chown=rluser:rluser . .

# Create output directories
RUN mkdir -p /app/outputs /app/checkpoints /app/experiments /app/logs \
    && chown -R rluser:rluser /app

USER rluser

# Expose telemetry port
EXPOSE 9090

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/rluser/.local/bin:${PATH}" \
    RL_OUTPUT_DIR="/app/outputs" \
    RL_CHECKPOINT_DIR="/app/checkpoints"

# Health check via telemetry endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Default: run training with config
ENTRYPOINT ["python", "-m", "scripts.train"]
CMD ["--config", "configs/default.yaml"]
