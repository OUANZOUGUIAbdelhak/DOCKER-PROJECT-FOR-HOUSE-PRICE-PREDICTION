# Docker Image Rebuild Instructions

## Quick Rebuild (Recommended)

```bash
# Stop and remove containers
docker-compose down

# Rebuild images (without cache to ensure fresh build)
docker-compose build --no-cache

# Start services
docker-compose up -d
```

## Full Clean Rebuild (If you want to remove old images)

```bash
# Stop and remove containers
docker-compose down

# Remove old images (optional - saves disk space)
docker rmi house-price-training house-price-api house-price-frontend 2>$null

# Or remove all unused images
docker image prune -a

# Rebuild images
docker-compose build --no-cache

# Start services
docker-compose up -d
```

## Why Rebuild?

After recent changes:
- ✅ Dockerfiles now use `requirements-docker.txt` (TensorFlow excluded)
- ✅ Fixed logger initialization in `docker_api/app.py`
- ✅ Updated training code for neural network support

## Verify Rebuild

```bash
# Check images were rebuilt
docker images | grep house-price

# Check containers are running
docker-compose ps

# Check API logs (should show TensorFlow not available message)
docker-compose logs api | Select-String -Pattern "TensorFlow"
```

## Expected Output

After rebuild, API logs should show:
```
TensorFlow not available - Only sklearn models supported (this is fine for XGBoost/sklearn models)
```

This is **normal and expected** - TensorFlow is excluded from Docker builds for smaller images.
