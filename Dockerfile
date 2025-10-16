# Use slim Python for smaller image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system deps for QuTiP/SciPy (gfortran for compilation)
RUN apt-get update && apt-get install -y \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port for Render
EXPOSE 8000

# Run with Uvicorn (production settings)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
