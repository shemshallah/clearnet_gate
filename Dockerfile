# Multi-stage build for efficiency
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and templates
COPY quantum_proxy.py .
COPY templates/ ./templates/

# Production stage
FROM python:3.12-slim AS production

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --from=builder /app .

# Create upload directory (for local fallback)
RUN mkdir -p /opt/render/project/data/uploads

# Expose volume for persistent local storage (pull files from mounted /opt/render/project/data/uploads)
VOLUME ["/opt/render/project/data/uploads"]

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "quantum_proxy:app", "--host", "0.0.0.0", "--port", "8000"]
