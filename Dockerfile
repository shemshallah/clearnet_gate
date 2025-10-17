# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# This allows Docker to use the cache if requirements.txt hasn't changed
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# The application appears to use custom modules (modules/*)
# Copy the custom modules directory and the main application file
# NOTE: You must ensure a 'modules' directory with the imported files
# (e.g., quantum_core.py, security.py, etc.) exists on your host machine
# next to the Dockerfile when building.
COPY modules modules/
COPY main-2.py .

# Create necessary directories that the application initializes
# UPLOAD_DIR (default: ./uploads), STATIC_DIR (default: ./static), TEMPLATES_DIR (default: ./templates)
RUN mkdir -p uploads static templates

# Expose the port the app runs on (uvicorn default is 8000)
EXPOSE 8000

# Run the application using Gunicorn with Uvicorn workers
# 'main-2:app' refers to the 'app' object in the 'main-2.py' file.
# The number of workers is often set to (2 * $num_cores) + 1 for optimal performance.
# We'll default to 4 workers here for common environments.
CMD ["gunicorn", "main-2:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--log-level", "info"]
