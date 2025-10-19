# Use official Python runtime as base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000 \
    DEBUG=false \
    HOLOGRAPHIC_CAPACITY_EB=6.0 \
    BELL_TEST_ITERATIONS=10000 \
    GHZ_TEST_ITERATIONS=10000 \
    TELEPORTATION_ITERATIONS=1000 \
    SECRET_KEY=your-super-secret-key-change-in-production

# Install system dependencies for ping and whois
RUN apt-get update && apt-get install -y \
    iputils-ping \
    whois \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /data && \
    mkdir -p data && \
    chmod -R 755 /data data

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
