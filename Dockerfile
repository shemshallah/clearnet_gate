# Use official Python runtime as base image (slim for smaller size)
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Debug: Print requirements.txt content to logs (remove after fixing)
RUN cat requirements.txt

# Install dependencies (with timeout to fix network issues; no cache to reduce size)
RUN pip install --no-cache-dir --default-timeout=100 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the entire application code (includes main.py and all .html files)
COPY . .

# Create uploads directory if it doesn't exist
RUN mkdir -p uploads

# Expose the port the app runs on
EXPOSE 8000

# Run the application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
