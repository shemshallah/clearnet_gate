# Custom Python Image for QuTiP (g++ Entanglement)
FROM python:3.12-slim

# Install System Deps (Compilers + Git for Mirror)
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Working Dir
WORKDIR /app

# Copy Files
COPY . .

# Upgrade Pip & Install Python Deps (Sequential for QuTiP)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 && \
    pip install --no-cache-dir qutip==4.7.5 && \
    pip install --no-cache-dir -r requirements.txt

# Expose Port
EXPOSE 5000

# Run App
CMD ["python", "app.py"]
