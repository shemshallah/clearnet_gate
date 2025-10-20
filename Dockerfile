FROM python:3.12-slim

# Install system build dependencies for qutip/scipy
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    python3-dev \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (Render uses 10000 by default)
EXPOSE 10000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
