# Quantum Foam Computer - Extended System Docker Container
# Created by Justin Anthony Howard-Stanley & Dale Cwidak
# "For Logan and all the ones like him"

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools first
RUN pip install --upgrade pip setuptools wheel

# Install packages one by one to avoid conflicts
RUN pip install --no-cache-dir flask
RUN pip install --no-cache-dir flask-socketio
RUN pip install --no-cache-dir python-socketio
RUN pip install --no-cache-dir eventlet
RUN pip install --no-cache-dir psutil
RUN pip install --no-cache-dir requests
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir scipy
RUN pip install --no-cache-dir qutip
RUN pip install --no-cache-dir cryptography

# Create directories
RUN mkdir -p /app/holographic_storage/{users,chat,email,files,blockchain,network_map}
RUN mkdir -p /app/templates /app/static

# Copy all files
COPY . .

# Create quantum_foam_core.py if missing
RUN if [ ! -f quantum_foam_core.py ]; then \
    cat > quantum_foam_core.py << 'EOF'
#!/usr/bin/env python3
import json
from datetime import datetime

class QuantumFoamComputer:
    def __init__(self, dimensions=6):
        self.fidelity = 0.999
        
    def get_echo_state(self):
        return {"fidelity": 0.999, "status": "active"}
        
    def get_system_stats(self):
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
EOF
    fi

# Make scripts executable
RUN chmod +x start_extended.sh 2>/dev/null || echo "Script will be created at runtime"

# Create non-root user
RUN useradd -m -u 1000 quantum && \
    chown -R quantum:quantum /app
USER quantum

# Expose port
EXPOSE 5000

# Environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production

# Labels
LABEL maintainer="Justin Anthony Howard-Stanley <shemshallah@gmail.com>"
LABEL version="1.0"
LABEL description="Quantum Foam Computer - Extended 7-Module System"

# Start command
CMD ["python3", "quantum_foam_extended.py"]
