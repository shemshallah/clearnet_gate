FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY quantum_app.py .
COPY templates/ templates/
COPY static/ static/

# Create data directory
RUN mkdir -p data

# Create non-root user
RUN useradd -m -u 1000 quantum && \
    chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

# Run application
CMD ["python", "quantum_app.py"]
