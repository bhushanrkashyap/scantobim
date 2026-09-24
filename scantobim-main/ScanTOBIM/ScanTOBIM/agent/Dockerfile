# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — ScanToBIM Agent (Multi-stage, hardened)
#
# Stages:
#   base        — shared OS deps (Open3D, scipy system libs)
#   deps        — Python package installation (layer-cached)
#   production  — final minimal image, non-root user, health check
#   dev         — production + dev tools (pytest, ruff) for local iteration
#
# Build production:  docker build --target production -t scantobim-api .
# Build dev:         docker build --target dev -t scantobim-dev .
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: base — OS dependencies ──────────────────────────────────────────
FROM python:3.11.14-slim AS base

# System libs required by open3d-cpu, scipy, laspy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — runs the API process without root privileges
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Stage 2: deps — pip install (cached unless pyproject.toml changes) ────────
FROM base AS deps

COPY agent/pyproject.toml ./agent/pyproject.toml

# Install production dependencies
# open3d-cpu is lighter; falls back to full open3d if CPU build unavailable
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.30.0" \
        "pydantic>=2.12.0" \
        "numpy>=1.26.0" \
        "scipy>=1.13.0" \
        "laspy[lazrs]>=2.5.0" \
        "python-dotenv>=1.0.0" \
        "structlog>=24.4.0" \
        "httpx>=0.27.0" \
        "aiosqlite>=0.20.0" \
        "python-multipart>=0.0.9" \
 && (pip install --no-cache-dir open3d-cpu || pip install --no-cache-dir open3d)

# ── Stage 3: production ───────────────────────────────────────────────────────
FROM deps AS production

# Copy application source
COPY agent/ ./agent/
COPY .env.example .env

# Persistent data directories (mounted as volumes)
RUN mkdir -p /data/scans /data/db \
 && chown -R appuser:appgroup /data /app

# Drop root — run as unprivileged user
USER appuser

# Environment defaults (overridden by docker-compose env_file / environment)
ENV DATABASE_URL="sqlite:////data/db/scantobim.db" \
    SCAN_FILES_DIR="/data/scans" \
    AGENT_HOST="0.0.0.0" \
    AGENT_PORT="8765" \
    LOG_LEVEL="info" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"

EXPOSE 8765

# Health check calls /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8765/health | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)"

CMD ["uvicorn", "agent.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8765", \
     "--workers", "2", \
     "--log-level", "info", \
     "--no-access-log"]

# ── Stage 4: dev — includes test + lint tools ─────────────────────────────────
FROM production AS dev

USER root
RUN pip install --no-cache-dir \
        "pytest>=8.0.0" \
        "pytest-asyncio>=0.24.0" \
        "pytest-cov>=5.0.0" \
        "ruff>=0.7.0" \
        "httpx>=0.27.0"
USER appuser

CMD ["uvicorn", "agent.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8765", \
     "--reload", \
     "--log-level", "debug"]
