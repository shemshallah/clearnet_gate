FROM python:3.12-slim

# Install compilers & libs for QuTiP (g++/gfortran + OpenBLAS/LAPACK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Pip upgrade & deps (numpy/scipy first for wheels)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT

CMD ["python", "app.py"]
