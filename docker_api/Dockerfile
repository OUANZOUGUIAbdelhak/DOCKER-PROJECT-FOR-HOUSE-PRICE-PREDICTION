# FastAPI Backend Dockerfile - Fully Autonomous
FROM python:3.9-slim

LABEL maintainer="ML Engineer"
LABEL description="FastAPI backend for House Price Prediction"

WORKDIR /app

# Set Python path
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

# Copy Docker-specific requirements (TensorFlow excluded for smaller images)
# Use docker_api/requirements.txt if you need Keras model support
COPY docker_api/requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Copy only API code (fully autonomous)
COPY docker_api/app.py /app/app.py

# Create models directory (will be mounted as volume)
RUN mkdir -p /app/models

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
