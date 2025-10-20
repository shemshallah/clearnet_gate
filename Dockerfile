FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for cryptography and qutip/scipy
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libffi-dev \
    libssl-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY main.py .

# Create data directory for DB (writable on ephemeral FS)
RUN mkdir -p data

# Expose dynamic port (Render uses 10000)
EXPOSE ${PORT:-8000}

# Run the app with dynamic port (use Render's $PORT or default 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
