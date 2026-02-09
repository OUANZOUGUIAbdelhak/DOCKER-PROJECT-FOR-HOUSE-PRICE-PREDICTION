# Dockerized Training Container - Fully Autonomous
FROM python:3.9-slim

LABEL maintainer="ML Engineer"
LABEL description="Dockerized training container for House Price Prediction"
LABEL version="1.0"

WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

# Copy Docker-specific requirements (TensorFlow excluded for smaller images)
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Copy only training code (fully autonomous)
COPY docker_training/train.py /app/train.py

# Create directories for volumes
RUN mkdir -p /app/models /app/data/raw /app/data/processed

# Default command: Train the selected model with full dataset
CMD ["python", "train.py"]
