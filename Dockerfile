# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# This step is essential and resolves the "requirements.txt: not found" error.
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file
COPY quantum_proxy.py /app/

# Expose the port the FastAPI application runs on (default 8000 in your code)
EXPOSE 8000

# Command to run the application using Uvicorn
# Format: uvicorn <module_name>:<app_object_name> --host <ip> --port <port>
CMD ["uvicorn", "quantum_proxy:app", "--host", "0.0.0.0", "--port", "8000"]
