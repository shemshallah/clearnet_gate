# Dockerfile
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Install runtime dependencies (network tools for QSH)
RUN apt-get update && apt-get install -y --no-install-recommends \
    iputils-ping \
    whois \
    traceroute \
    netcat-openbsd \
    dnsutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --chown=quantum:quantum qsh_foam_production.py .
COPY --chown=quantum:quantum blockchain.html .
COPY --chown=quantum:quantum email.html .

# Create necessary directories
RUN mkdir -p /app/data /app/static /app/logs && \
    chown -R quantum:quantum /app && \
    chmod 755 /app/data /app/static /app/logs

# Copy HTML files to static directory (if serving them statically)
RUN cp blockchain.html /app/static/ && \
    cp email.html /app/static/ && \
    chown -R quantum:quantum /app/static

# Switch to non-root user
USER quantum

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "qsh_foam_production:app", "--host", "0.0.0.0", "--port", "8000"]
