# Multi-stage Dockerfile for Quantum Realm Flask App
# Stage 1: Builder (install deps)
FROM python:3.12-slim AS builder

# Set workdir
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install deps (no dev tools)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (slim, secure)
FROM python:3.12-slim AS runtime

# Install runtime deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set env vars
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PORT=10000 \
    HOST=0.0.0.0

# Set workdir
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app files
COPY app.py .
COPY Procfile .

# Expose port (Render uses $PORT, but default 10000)
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Run with Gunicorn (as Procfile, but explicit for Docker)
CMD ["sh", "-c", "gunicorn -k eventlet -w 1 -b 0.0.0.0:${PORT} app:app --timeout 120"]
