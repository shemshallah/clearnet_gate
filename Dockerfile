# Use a Python slim base for a smaller final image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies first (for faster rebuilds when only code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file
# Assuming your local file is now called main.py, or you use: COPY main-2.py main.py
COPY main.py .

# Copy the entire custom module package
COPY modules modules/

# Create necessary directories for file storage and web serving
# The application (main.py) uses these for uploads and static content
RUN mkdir -p uploads static templates

# Expose the application port
EXPOSE 8000

# Run the application using Gunicorn with Uvicorn workers
# 'main:app' is the correct module:object format for your FastAPI instance
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--log-level", "info"]
