# Use official Python slim runtime
FROM python:3.12-slim

# Set working directory
WORKDIR /code

# Install system deps (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (prevents hangs on old pip)
RUN pip install --no-cache-dir --upgrade pip

# Copy & install deps individually (avoids resolver hangs; per common fixes)
COPY requirements.txt /code/
RUN pip install --no-cache-dir fastapi==0.115.0
RUN pip install --no-cache-dir uvicorn[standard]==0.30.6
RUN pip install --no-cache-dir python-multipart==0.0.9

# Copy code & assets
COPY main.py /code/
COPY *.html /code/
RUN mkdir -p /code/uploads /code/static && chmod -R 755 /code/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run (exec form; fastapi run wraps uvicorn cleanly, no hang on startup)
CMD ["fastapi", "run", "main:app", "--host", "0.0.0.0", "--port", "8000"]
