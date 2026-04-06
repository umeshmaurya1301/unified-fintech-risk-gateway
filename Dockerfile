# ─────────────────────────────────────────────────────────────────────────────
# Unified Fintech Risk Gateway (UFRG) — Production Container
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="Umesh Maurya <umeshmaurya1301>" \
    description="Unified Fintech Risk Gateway — Gymnasium RL Environment" \
    version="1.0.0"

# ── OS-level hardening ───────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Copy ALL application source (including server folder) ────────────────────
COPY . /app

# ── Install dependencies (Explicit for Hugging Face) ─────────────────────────
RUN pip install --no-cache-dir openenv-core gymnasium numpy pydantic openai fastapi uvicorn

# ── Port configuration ───────────────────────────────────────────────────────
# Expose the exact port Hugging Face Spaces routes traffic to
EXPOSE 7860

# ── Default entrypoint: Start the FastAPI server ─────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]