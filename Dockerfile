# Stage 1: Build Stage - Use a slim Python image
FROM python:3.11-slim as builder

# Set environment variables for non-interactive commands
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies needed for python packages (e.g., psycopg2)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Stage 2: Final Image - A lighter image for production
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the rest of the application code
# Assuming the modules are directly in the root directory
COPY . /app

# Ensure correct permissions for the app
RUN chmod -R 755 /app

# Expose the port defined in your docker-compose.yml
EXPOSE 8000

# Command to run the application using Gunicorn (the production standard)
# We use uvicorn workers for asynchronous performance.
# Assumes your fixed code is saved as 'main.py' and the FastAPI app is named 'app'.
# If you kept the file name 'main-2.py', change 'main:app' to 'main-2:app'.
CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--log-level", "info"]
# Labels
LABEL maintainer="Justin Anthony Howard-Stanley <shemshallah@gmail.com>"
LABEL version="1.0"
LABEL description="Quantum Foam Computer - Extended 7-Module System"

# Start command
CMD ["python3", "quantum_foam_extended.py"]
