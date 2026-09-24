# ─────────────────────────────────────────────────────────────────────────────
# Makefile — ScanToBIM Agent
# Works on Ubuntu / Debian / any POSIX Linux. Windows users: run make.ps1
# Run from the ScanToBIM/ root directory.
# ─────────────────────────────────────────────────────────────────────────────

# Discover a usable Python 3.11+ interpreter. Override on the command line
# or via environment: e.g. `make install PYTHON=python3.12`.
# Resolution order: env var → python3.11 → python3 → python.
PYTHON   ?= $(shell command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null)
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
RUFF     := $(VENV)/bin/ruff
UVICORN  := $(VENV)/bin/uvicorn

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "  ScanToBIM Agent"
	@echo "  ─────────────────────────────────────────────"
	@echo ""
	@echo "  Setup"
	@echo "    make install        Install Python deps (core + dev)"
	@echo "    make install-ai     Install OpenAI optional dep"
	@echo "    make env            Copy .env.example → .env (first time)"
	@echo ""
	@echo "  Run"
	@echo "    make server         Start FastAPI agent on port 8765"
	@echo "    make demo           Run standalone demo (no server needed)"
	@echo ""
	@echo "  Quality"
	@echo "    make test           Run pytest + coverage report"
	@echo "    make test-fast      Run pytest without coverage"
	@echo "    make lint           Run ruff linter"
	@echo "    make lint-fix       Run ruff linter with auto-fix"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-up      Start agent in Docker (port 8765)"
	@echo "    make docker-down    Stop and remove containers"
	@echo "    make docker-logs    Tail agent logs"
	@echo "    make docker-build   Rebuild Docker image"
	@echo ""
	@echo "  C# Tests"
	@echo "    make test-csharp    Run xUnit safety gate tests (no Revit needed)"
	@echo ""
	@echo "  Utilities"
	@echo "    make clean          Remove __pycache__, .db, coverage artifacts"
	@echo "    make check          Run lint + test in one shot (CI equivalent)"
	@echo ""
	@echo "  Windows users: use make.ps1 in PowerShell (same target names)."
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: check-python
check-python:
	@if [ -z "$(PYTHON)" ]; then \
	  echo "ERROR: No Python interpreter found on PATH."; \
	  echo "       Install python3.11 (Ubuntu: apt install python3.11 python3.11-venv)"; \
	  echo "       Or override: make install PYTHON=/custom/path/to/python"; \
	  exit 1; \
	fi
	@echo "→ Using Python: $(PYTHON)"
	@$(PYTHON) --version

.PHONY: install
install: check-python
	@echo "→ Creating virtual environment in $(VENV)/..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install \
	  fastapi \
	  "uvicorn[standard]" \
	  pydantic \
	  numpy \
	  scipy \
	  "laspy[lazrs]" \
	  python-dotenv \
	  structlog \
	  httpx \
	  aiosqlite \
	  python-multipart \
	  pytest \
	  pytest-asyncio \
	  pytest-cov \
	  ruff
	@echo "→ Installing Open3D (CPU variant by default — lighter download)..."
	$(PIP) install open3d-cpu || $(PIP) install open3d
	@echo ""
	@echo "✓ Done. Activate with:"
	@echo "    source $(VENV)/bin/activate"

.PHONY: install-ai
install-ai: install
	$(PIP) install 'openai>=1.50.0'

.PHONY: env
env:
	@if [ -f .env ]; then \
	  echo "  .env already exists — skipping."; \
	else \
	  cp .env.example .env; \
	  echo "✓ .env created from .env.example — edit AUDIT_HMAC_SECRET before use."; \
	fi

# ── Run ───────────────────────────────────────────────────────────────────────

.PHONY: server
server:
	@echo "→ Starting ScanToBIM agent on http://localhost:8765"
	@echo "  Docs: http://localhost:8765/docs"
	$(UVICORN) agent.main:app \
	  --host 0.0.0.0 \
	  --port 8765 \
	  --reload \
	  --log-level info

.PHONY: demo
demo:
	@echo "→ Running standalone demo (no server required)..."
	$(VENV)/bin/python demo.py

# ── Tests ─────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	@echo "→ Running Python tests with coverage..."
	cd agent && \
	../$(PYTEST) tests/ \
	  --cov=. \
	  --cov-report=term-missing \
	  --cov-report=html:../coverage-html \
	  --cov-fail-under=70 \
	  -v
	@echo ""
	@echo "✓ Coverage report: coverage-html/index.html"

.PHONY: test-fast
test-fast:
	@echo "→ Running Python tests (no coverage)..."
	cd agent && ../$(PYTEST) tests/ -v

.PHONY: test-csharp
test-csharp:
	@echo "→ Running C# xUnit safety gate tests..."
	dotnet test revit-addin/Tests/ScanToBIM.Tests.csproj \
	  --logger "console;verbosity=normal" \
	  --no-restore
	@echo "✓ C# tests complete."

# ── Linting ───────────────────────────────────────────────────────────────────

.PHONY: lint
lint:
	@echo "→ Running ruff linter..."
	$(RUFF) check agent/ demo.py

.PHONY: lint-fix
lint-fix:
	@echo "→ Running ruff linter with auto-fix..."
	$(RUFF) check --fix agent/ demo.py
	$(RUFF) format agent/ demo.py

# ── Docker ────────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up:
	@echo "→ Starting ScanToBIM stack..."
	docker compose up -d
	@echo ""
	@echo "✓ Agent running at http://localhost:8765"
	@echo "  Docs:    http://localhost:8765/docs"
	@echo "  Health:  http://localhost:8765/health"
	@echo "  Logs:    make docker-logs"

.PHONY: docker-down
docker-down:
	docker compose down
	@echo "✓ Stack stopped."

.PHONY: docker-logs
docker-logs:
	docker compose logs -f agent

.PHONY: docker-build
docker-build:
	docker compose build --no-cache agent

# ── Combined checks ───────────────────────────────────────────────────────────

.PHONY: check
check: lint test
	@echo ""
	@echo "✓ All checks passed."

# ── Cleanup ───────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	@echo "→ Cleaning build artifacts..."
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	rm -f demo_audit.db scantobim.db .coverage
	rm -rf coverage-html .pytest_cache agent/.pytest_cache
	@echo "✓ Clean."
