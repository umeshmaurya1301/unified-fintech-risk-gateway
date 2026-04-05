# ─────────────────────────────────────────────────────────────────────────────
# Unified Fintech Risk Gateway (UFRG) — Production Container
# ─────────────────────────────────────────────────────────────────────────────
# Lightweight, deterministic image that validates the Gymnasium environment
# on startup.  Build once → run anywhere → zero configuration.
#
#   docker build -t ufrg .
#   docker run --rm ufrg
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="Umesh Maurya <umeshmaurya1301>"                         \
      description="Unified Fintech Risk Gateway — Gymnasium RL Environment" \
      version="1.0.0"

# ── OS-level hardening ───────────────────────────────────────────────────────
# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# so every log line appears immediately in `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies (layer-cached) ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ─────────────────────────────────────────────────
COPY unified_gateway.py .
COPY dummy_test.py .

# ── Default entrypoint: run the validation + stress-test suite ───────────────
CMD ["python", "dummy_test.py"]
