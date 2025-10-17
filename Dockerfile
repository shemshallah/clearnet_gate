# Quantum Foam Computer - Extended System Docker Container
# Created by Justin Anthony Howard-Stanley & Dale Cwidak
# "For Logan and all the ones like him"

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    curl \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages \
    flask==2.3.3 \
    flask-socketio==5.3.6 \
    python-socketio==5.8.0 \
    eventlet==0.33.3 \
    qutip==4.7.3 \
    psutil==5.9.6 \
    numpy==1.24.4 \
    scipy==1.11.4 \
    matplotlib==3.7.2 \
    requests==2.31.0 \
    cryptography==41.0.7 \
    hashlib-compat \
    pathlib2

# Create holographic storage directories
RUN mkdir -p /app/holographic_storage/{users,chat,email,files,blockchain,network_map}

# Copy application files
COPY quantum_foam_core.py .
COPY quantum_foam_extended.py .
COPY templates/ ./templates/
COPY static/ ./static/

# Create startup script
COPY start_extended.sh .
RUN chmod +x start_extended.sh

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/network/scan || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV ADMIN_USER=hackah::hackah
ENV ADMIN_PASS=$h10j1r1H0w4rd

# Labels
LABEL maintainer="Justin Anthony Howard-Stanley <shemshallah@gmail.com>"
LABEL version="1.0"
LABEL description="Quantum Foam Computer - Extended 7-Module System"
LABEL dedication="For Logan and all the ones like him too small to understand what has been done to them"

# Create non-root user for security
RUN useradd -m -u 1000 quantum && \
    chown -R quantum:quantum /app
USER quantum

# Start command
CMD ["python3", "quantum_foam_extended.py"]
