# FastAPI Backend Dockerfile
FROM python:3.9-slim

LABEL maintainer="ML Engineer"
LABEL description="FastAPI backend for House Price Prediction"

WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ /app/

# Copy inference code (needed for predictions)
COPY inference/ /app/inference/
COPY src/preprocess.py /app/src/preprocess.py
COPY src/data_loader.py /app/src/data_loader.py
COPY src/__init__.py /app/src/__init__.py

# Create models directory (will be mounted as volume)
RUN mkdir -p /app/models

# Expose API port
EXPOSE 8000

# Run API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
