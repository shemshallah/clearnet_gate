# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and rename the main application file from main-2.py to main.py
# NOTE: Ensure the local file 'main-2.py' is present.
COPY main-2.py main.py

# Copy the custom modules directory (REQUIRED for your application to boot)
# IMPORTANT: Ensure your local directory structure is: ./modules/quantum_core.py, etc.
COPY modules modules/

# Create necessary directories
RUN mkdir -p uploads static templates

# Expose the port the app runs on
EXPOSE 8000

# Run the application using Gunicorn with Uvicorn workers
# 'main:app' tells Gunicorn to load the object named 'app' from the file 'main.py'
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--log-level", "info"]
