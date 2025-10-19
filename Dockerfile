# Use slim Python 3.12 for smaller image (~150MB)
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install deps (no dev deps for prod)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY quantum_app.py .

# Expose port
EXPOSE 8000

# Health check: Ping localhost (Alice node)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn (0.0.0.0 for Docker/Render, workers for scale)
CMD ["uvicorn", "quantum_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
# Run application
CMD ["python", "quantum_app.py"]
