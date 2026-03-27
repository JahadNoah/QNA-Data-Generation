# ── Base image ────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────
# Prevents Python from buffering stdout/stderr (important for Cloud Run logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────
# Copy requirements first so Docker layer cache is reused on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────
# .env is NOT copied — set environment variables in Cloud Run console instead
COPY core.py web.py ./
COPY prompts/ prompts/
COPY templates/ templates/

# ── Expose port ───────────────────────────────────────────────
EXPOSE 8080

# ── Start Gunicorn ────────────────────────────────────────────
# Worker class: gthread (threaded) — required for SSE streaming (text/event-stream)
# so long-lived SSE connections don't block other requests.
# --timeout 300 gives the LLM up to 5 minutes to respond per request.
# $PORT is set to 8080 by Cloud Run automatically.
CMD exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --worker-class gthread \
    --workers 2 \
    --threads 4 \
    --timeout 300 \
    --log-level info \
    web:app
