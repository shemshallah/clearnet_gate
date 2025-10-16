
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY quantum_proxy.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "quantum_proxy:app", "--host", "0.0.0.0", "--port", "8000"]

