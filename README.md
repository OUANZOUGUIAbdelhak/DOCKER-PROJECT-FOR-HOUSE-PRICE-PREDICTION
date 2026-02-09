# 🏠 House Price Prediction: Production ML/MLOps Pipeline with Docker

A complete, production-ready Machine Learning pipeline demonstrating Docker best practices, MLOps principles, and distributed deployment using Docker Swarm.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Local Training & Model Comparison](#3-local-training--model-comparison)
4. [Dockerized Training Pipeline](#4-dockerized-training-pipeline)
5. [API Inference Service](#5-api-inference-service)
6. [Frontend](#6-frontend)
7. [Running the Project](#7-running-the-project-step-by-step)
8. [Docker Swarm Deployment](#8-docker-swarm-deployment)
9. [MLOps & Reproducibility](#9-mlops--reproducibility)
10. [Common Errors & Debugging](#10-common-errors--debugging)

---

## 1. Project Overview

### Problem Description

Predicting house prices is a classic regression problem in Machine Learning. This project implements a complete ML pipeline that:

- Handles real-world data challenges (missing values, categorical variables, skewed distributions)
- Trains and compares multiple ML models
- Deploys models in production using Docker containers
- Provides a web interface for predictions
- Supports distributed deployment across multiple machines

### Why Docker?

**Docker solves critical ML deployment challenges:**

1. **Reproducibility**: Same environment everywhere (development, testing, production)
2. **Isolation**: Training and inference dependencies don't conflict
3. **Portability**: Run anywhere Docker runs (local, cloud, edge)
4. **Scalability**: Easy to scale services independently
5. **Consistency**: Eliminates "works on my machine" problems

### Why Separate Containers?

**Three production containers, each with a specific purpose:**

1. **Training Container**: Heavy dependencies, runs once, produces model artifacts
2. **API Container**: Lightweight, runs continuously, serves predictions
3. **Frontend Container**: UI only, no ML logic, stateless

**Benefits:**
- **Smaller production images**: API doesn't need training dependencies
- **Independent scaling**: Scale API without training overhead
- **Security**: Minimal attack surface in production
- **Cost efficiency**: Production containers are lightweight

---

## 2. Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOST MACHINE                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  LOCAL TRAINING (NO DOCKER)                              │ │
│  │  • Explore data                                          │ │
│  │  • Train multiple models                                 │ │
│  │  • Compare performance                                   │ │
│  │  • Select best model                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           │                                     │
│                           │ Model Selection                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  DOCKERIZED TRAINING CONTAINER                           │ │
│  │  • Retrain selected model                                │ │
│  │  • Use 100% of dataset                                   │ │
│  │  • Save to volume                                        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           │                                     │
│                           │ Writes                              │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  DOCKER VOLUME (models/)                                 │ │
│  │  • model.pkl                                             │ │
│  │  • preprocessor.pkl                                      │ │
│  │  • metrics.json                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ▲                                     │
│                           │ Reads                               │
│  ┌───────────────────────┴───────────────────────────────────┐ │
│  │  API CONTAINER (FastAPI)                                │ │
│  │  • Loads model from volume                              │ │
│  │  • Serves REST API                                      │ │
│  │  • Port 8000                                            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ▲                                     │
│                           │ HTTP Requests                       │
│  ┌───────────────────────┴───────────────────────────────────┐ │
│  │  FRONTEND CONTAINER (React + Nginx)                      │ │
│  │  • User interface                                        │ │
│  │  • Calls API                                             │ │
│  │  • Port 3000                                            │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Data Ingestion
   data/raw/train.csv (Kaggle dataset)
   
2. Local Experimentation (NO Docker)
   local_training/model_comparison/compare_models.py
   → Trains multiple models
   → Compares performance
   → Selects best model
   
3. Dockerized Training
   docker_training/train.py (inside container)
   → Loads data from volume
   → Retrains selected model with 100% data
   → Saves to models/ volume
   
4. API Inference
   docker_api/app.py (inside container)
   → Loads model from volume
   → Accepts HTTP requests
   → Returns predictions
   
5. Frontend Interaction
   docker_frontend/ (React app)
   → User inputs house features
   → Sends to API
   → Displays prediction
```

### Volume Usage

**Docker volumes persist data across container lifecycles:**

- **`./models` → `/app/models`**: Model artifacts (persisted on host)
- **`./data` → `/app/data`**: Dataset (read-only for containers)

**Why volumes?**
- Containers are ephemeral (data lost when container stops)
- Volumes persist data on host filesystem
- Multiple containers can share the same volume
- Models survive container restarts

---

## 3. Local Training & Model Comparison

### Purpose of Local Experimentation

**Local training runs OUTSIDE Docker** for:

1. **Rapid iteration**: Quick experiments with partial data
2. **Model exploration**: Try different architectures and hyperparameters
3. **Performance comparison**: Evaluate multiple models side-by-side
4. **Selection**: Choose the best model before Dockerized training

### How Models Are Compared

**Comparison Process:**

1. **Load data** from `data/raw/train.csv`
2. **Preprocess** (handle missing values, feature engineering)
3. **Split data** (60% train, 20% validation, 20% test)
4. **Train models**:
   - Linear Regression (baseline)
   - Random Forest (tree-based ensemble)
   - Gradient Boosting (sequential boosting)
   - Neural Network (deep learning with TensorFlow/Keras)
5. **Evaluate** on validation set using:
   - **RMSE** (Root Mean Squared Error): Penalizes large errors
   - **MAE** (Mean Absolute Error): Average error magnitude
   - **R²** (R-squared): Proportion of variance explained
6. **Select best model** based on lowest RMSE

### Criteria for Selecting Best Model

**Primary metric: RMSE** (Root Mean Squared Error)

- Lower RMSE = better predictions
- Penalizes large errors (important for expensive houses)
- Same units as target (dollars)

**Secondary metrics:**
- MAE: Average error (easier to interpret)
- R²: Overall model quality (higher is better)

### Explicit Comparison: Local vs Docker Training

| Aspect | Local Training | Dockerized Training |
|--------|---------------|---------------------|
| **Environment** | Local Python | Docker container |
| **Data Size** | Partial/reduced (for speed) | 100% of dataset |
| **Purpose** | Model selection | Final production model |
| **Reproducibility** | Depends on local setup | Guaranteed via Docker |
| **Output** | Comparison metrics | Trained model artifacts |
| **Speed** | Fast (less data) | Slower (full dataset) |
| **Isolation** | Uses local dependencies | Isolated environment |

**Why both?**
- **Local**: Fast experimentation and model selection
- **Docker**: Reproducible, production-ready training with full data

---

## 4. Dockerized Training Pipeline

### Training Container Responsibilities

**The training container (`docker_training/train.py`):**

1. **Loads data** from mounted volume (`/app/data`) - logic embedded in file
2. **Preprocesses** using embedded preprocessing functions (no shared modules)
3. **Retrains** the selected model using **100% of the dataset**
4. **Saves artifacts** to volume (`/app/models`):
   - `model.pkl`: Trained model
   - `preprocessor.pkl`: Fitted preprocessing pipeline
   - `metrics.json`: Evaluation metrics

**Fully autonomous:** All preprocessing, data loading, and training logic is contained within `docker_training/train.py`. No dependencies on shared modules.

### Full-Dataset Retraining

**Why retrain with 100% data?**

- **More data = better model**: Uses all available information
- **Production standard**: Final model should use maximum data
- **No test set leakage**: Test set used only for final evaluation

**Training process:**
1. Load full dataset
2. Preprocess (fit on full data)
3. Split 80/20 (train/validation)
4. Train model on 80%
5. Evaluate on 20%
6. **Retrain on 100%** for final model
7. Save to volume

### Artifact Persistence

**All artifacts saved to Docker volume:**

```
models/
├── model.pkl           # Trained model (sklearn)
├── preprocessor.pkl    # Fitted preprocessing pipeline
├── metrics.json        # Evaluation metrics
└── model_type.txt      # Model type identifier
```

**Why save preprocessor?**
- Inference must use **identical** preprocessing
- Prevents data leakage and inconsistencies
- Ensures same feature transformations

### Reproducibility Guarantees

**Docker ensures reproducibility:**

1. **Fixed dependencies**: `requirements.txt` pins versions
2. **Isolated environment**: No conflicts with host system
3. **Deterministic training**: Random seeds set (random_state=42)
4. **Version control**: Dockerfile defines exact environment
5. **Same results**: Same code + same data = same model

---

## 5. API Inference Service

### Model Loading Strategy

**API loads model at startup (`docker_api/app.py`):**

1. **Startup event**: `@app.on_event("startup")` loads model once
2. **Volume mount**: Reads from `/app/models` (mapped to `./models` on host)
3. **Error handling**: Fails fast if model not found
4. **Model type detection**: Supports sklearn models (extensible to Keras)

**Why load at startup?**
- **Performance**: Model loaded once, not per request
- **Fail fast**: Detects missing model immediately
- **Memory efficient**: Single model instance shared across requests

**Fully autonomous:** All preprocessing and prediction logic is embedded in `docker_api/app.py`. Preprocessing functions are re-implemented to match training preprocessing exactly (no shared modules).

### Prediction Workflow

```
1. User sends HTTP POST request with house features
   POST /predict
   {
     "LotArea": 8450,
     "YearBuilt": 2003,
     "OverallQual": 7,
     ...
   }

2. API validates input (Pydantic models)

3. Preprocessor transforms features
   - Handle missing values
   - Feature engineering
   - Encoding/scaling

4. Model predicts price
   model.predict(transformed_features)

5. API returns prediction
   {
     "predicted_price": 181500.0,
     "confidence": "high"
   }
```

### Runtime Behavior

**API characteristics:**

- **Stateless**: No training logic, only inference
- **Lightweight**: Minimal dependencies (FastAPI, sklearn, pandas)
- **Scalable**: Can run multiple replicas
- **Health checks**: `/health` endpoint for monitoring
- **CORS enabled**: Allows frontend requests

---

## 6. Frontend

### Frontend Responsibilities

**React frontend provides:**

1. **User interface**: Form for house features
2. **API interaction**: Sends requests to API container
3. **Result display**: Shows predicted price
4. **Error handling**: Displays API errors

### API Interaction

**Frontend calls API:**

```javascript
// Example API call
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(houseFeatures)
});

const prediction = await response.json();
```

**No ML logic in frontend:**
- Frontend is pure UI
- All ML processing happens in API
- Frontend only formats and displays results

---

## 7. Running the Project (Step-by-Step)

### Prerequisites

1. **Docker installed**: https://docs.docker.com/get-docker/
2. **Docker Compose**: Usually included with Docker Desktop
3. **Kaggle dataset**: House Prices - Advanced Regression Techniques
   - Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
   - Place `train.csv` in `data/raw/`

### Step 1: Dataset Placement

```bash
# Create data directory structure
mkdir -p data/raw data/processed

# Place train.csv in data/raw/
# Download from Kaggle and extract train.csv
# Copy to: data/raw/train.csv
```

### Step 2: Local Experimentation (Optional but Recommended)

```bash
# Install local dependencies (if not already installed)
pip install -r requirements.txt

# Run model comparison
python local_training/model_comparison/compare_models.py

# Review results
cat local_training/model_comparison/results.json
```

**Output:** Best model selection (e.g., `gradient_boosting`)

### Step 3: Docker Build Commands

**Important:** After code changes, rebuild images to apply updates:

```bash
# Stop existing containers
docker-compose down

# Rebuild all images (recommended after code changes)
docker-compose build --no-cache

# Or rebuild individual services
docker build -f docker_training/Dockerfile -t house-price-training .
docker build -f docker_api/Dockerfile -t house-price-api .
docker build -f docker_frontend/Dockerfile -t house-price-frontend .

# Start services
docker-compose up -d
```

**Note:** Docker builds use `requirements-docker.txt` which excludes TensorFlow (~475MB savings). 
- For local comparison with neural network: Use `requirements.txt` (includes TensorFlow)
- For Docker (sklearn/XGBoost models only): Uses `requirements-docker.txt` (no TensorFlow)
- To use neural network in Docker: Modify Dockerfiles to use `requirements.txt` instead

**Clean up old images (optional):**
```bash
# Remove old images to save disk space
docker rmi house-price-training house-price-api house-price-frontend
# Or remove all unused images
docker image prune -a
```

### Step 4: Docker Compose Execution

**Option A: Run training first, then API/Frontend (Recommended)**

```bash
# Step 1: Train model first
docker-compose run --rm training

# Or specify model
docker-compose run --rm training --model gradient_boosting

# Step 2: Start API and frontend (after training completes)
docker-compose up api frontend
```

**Option B: Run all services (training completes first automatically)**

```bash
# Docker Compose will:
# 1. Run training first (waits for completion)
# 2. Then start API (waits for training to finish)
# 3. Then start frontend (waits for API)
docker-compose up
```

**Option C: Run in background**

```bash
# Run all services in background
docker-compose up -d

# Check logs
docker-compose logs -f training  # Watch training progress
docker-compose logs -f api       # Watch API startup
```

**Note:** The API automatically waits up to 5 minutes for the model file if training is still running.

### Step 5: Volume Behavior

**Check model artifacts:**

```bash
# Models are saved to ./models/ on host
ls models/
# Should show:
# - model.pkl
# - preprocessor.pkl
# - metrics.json
# - model_type.txt
```

**Verify volume mounting:**

```bash
# Check container can see models
docker-compose exec api ls /app/models
```

### Step 6: API Testing Example

**Test API directly:**

```bash
# Health check
curl http://localhost:8000/health

# Prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LotArea": 8450,
    "YearBuilt": 2003,
    "OverallQual": 7,
    "OverallCond": 5,
    "TotalBsmtSF": 856,
    "GrLivArea": 1710,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "TotRmsAbvGrd": 8,
    "GarageCars": 2,
    "GarageArea": 548
  }'
```

**Expected response:**

```json
{
  "predicted_price": 181500.0,
  "message": "Prediction successful"
}
```

**Access frontend:**

Open browser: http://localhost:3000

---

## 8. Docker Swarm Deployment

### Why Docker Swarm?

**Docker Swarm enables:**

1. **Multi-machine deployment**: Distribute services across nodes
2. **High availability**: Automatic failover and restart
3. **Load balancing**: Distribute API requests across replicas
4. **Scalability**: Scale services independently
5. **Service discovery**: Automatic DNS resolution

### Swarm Initialization

**On manager node:**

```bash
# Initialize Swarm
docker swarm init

# Note the join token (for worker nodes)
# Example output:
# docker swarm join --token <token> <manager-ip>:2377
```

**On worker nodes:**

```bash
# Join Swarm
docker swarm join --token <token> <manager-ip>:2377
```

### Stack Deployment

**Build images on manager node:**

```bash
# Build images (TensorFlow excluded by default - smaller images)
docker build -f docker_training/Dockerfile -t house-price-training .
docker build -f docker_api/Dockerfile -t house-price-api .
docker build -f docker_frontend/Dockerfile -t house-price-frontend .
```

**Note:** Docker builds use `requirements-docker.txt` (TensorFlow excluded) for smaller images.

**Deploy stack:**

```bash
# Deploy stack
docker stack deploy -c docker-stack.yml house-price-stack

# Check services
docker stack services house-price-stack

# View logs
docker service logs house-price-stack_api
```

### Multi-Machine Execution Strategy

**Example deployment:**

```
Machine 1 (Manager):
  - Training service (when needed)
  - Swarm management

Machine 2 (Worker):
  - API service (2 replicas)
  - Frontend service

Machine 3 (Worker):
  - API service (2 replicas)
  - Additional frontend (if needed)
```

**Benefits:**
- **Training on manager**: Centralized model training
- **API on workers**: Distribute inference load
- **Frontend on workers**: Serve UI from multiple nodes
- **Shared volume**: Models accessible from all nodes (requires shared storage)

**Volume sharing across nodes:**

For production, use shared storage (NFS, Ceph, etc.):

```yaml
volumes:
  model-storage:
    driver: nfs
    driver_opts:
      type: nfs
      o: addr=nfs-server.example.com
      device: ":/exports/models"
```

---

## 9. MLOps & Reproducibility

### Separation of Concerns

**Clear boundaries:**

1. **Local Training**: Experimentation and model selection
2. **Dockerized Training**: Reproducible, production-ready training
3. **API**: Stateless inference service
4. **Frontend**: Pure UI, no ML logic

**Benefits:**
- **Maintainability**: Each component has single responsibility
- **Testability**: Test components independently
- **Scalability**: Scale services based on demand
- **Security**: Minimal attack surface

### Determinism

**Ensuring reproducible results:**

1. **Random seeds**: `random_state=42` in all ML code
2. **Pinned dependencies**: `requirements.txt` with exact versions
3. **Docker isolation**: Same environment every time
4. **Data versioning**: Use same dataset version
5. **Code versioning**: Git tracks all code changes

### Persistence

**Docker volumes ensure:**

- **Model persistence**: Models survive container restarts
- **Data persistence**: Dataset accessible across containers
- **Shared state**: Multiple containers can read same models
- **Backup**: Host filesystem can be backed up

### Scalability

**Horizontal scaling:**

- **API replicas**: Run multiple API containers
- **Load balancing**: Swarm distributes requests
- **Frontend replicas**: Serve UI from multiple nodes
- **Independent scaling**: Scale API without affecting training

**Vertical scaling:**

- **Resource limits**: Set CPU/memory limits per service
- **GPU support**: Add GPU nodes for training (if needed)

---

## 10. Common Errors & Debugging

### Error 1: "Model file not found"

**Symptom:**
```
FileNotFoundError: Model not found at /app/models/model.pkl
```

**Causes:**
- Training not run yet
- Volume not mounted correctly
- Wrong path in container

**Solutions:**

```bash
# Check if model exists on host
ls models/model.pkl

# Verify volume mount in docker-compose.yml
# Should have: - ./models:/app/models

# Re-run training
docker-compose run --rm training
```

### Error 2: "Permission denied" when writing to volume

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '/app/models/model.pkl'
```

**Causes:**
- File permissions mismatch (Linux containers on Windows)
- Volume mount path issues

**Solutions:**

```bash
# On Windows, use absolute paths
# In docker-compose.yml:
volumes:
  - C:/Users/hp/Desktop/DOCKER-PROJECT/models:/app/models

# Or fix permissions
chmod 777 models/
```

### Error 3: "Module not found" in container

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Causes:**
- PYTHONPATH not set
- Source code not copied to image
- Import path incorrect

**Solutions:**

```bash
# Check Dockerfile has:
ENV PYTHONPATH=/app

# Verify training code copied:
COPY docker_training/train.py /app/train.py

# Rebuild image
docker-compose build training
```

### Error 4: "Data not found"

**Symptom:**
```
FileNotFoundError: data/raw/train.csv not found
```

**Causes:**
- Dataset not downloaded
- Wrong path in container
- Volume not mounted

**Solutions:**

```bash
# Verify data exists on host
ls data/raw/train.csv

# Check volume mount
# Should have: - ./data:/app/data

# Download dataset from Kaggle
# Place in: data/raw/train.csv
```

### Error 5: "API connection refused"

**Symptom:**
```
ConnectionRefusedError: Connection refused
```

**Causes:**
- API container not running
- Wrong port
- Network issues

**Solutions:**

```bash
# Check API is running
docker-compose ps api

# Check API logs
docker-compose logs api

# Verify port mapping
# Should have: - "8000:8000"

# Restart API
docker-compose restart api
```

### Error 6: "Out of memory" during training

**Symptom:**
```
Container killed during training
```

**Causes:**
- Insufficient memory allocation
- Dataset too large
- Model too complex

**Solutions:**

```bash
# Increase memory limit
docker run --memory="4g" ...

# Or in docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 4G
```

---

## 📚 Project Structure

```
PROJECT_ROOT/
│
├── data/
│   ├── raw/                 # Original dataset (gitignored)
│   └── processed/           # Cleaned data (gitignored)
│
├── local_training/          # Local-only experiments (NO Docker)
│   ├── experiments/         # Model training experiments
│   ├── model_comparison/    # Metrics & comparison logic
│   │   ├── compare_models.py  # Fully autonomous (all logic embedded)
│   │   └── results.json
│   └── README.md            # Explains model comparison
│
├── docker_training/         # Dockerized training (Fully Autonomous)
│   ├── Dockerfile
│   └── train.py             # All logic embedded (preprocessing, training, etc.)
│
├── docker_api/              # Inference service (Fully Autonomous)
│   ├── Dockerfile
│   ├── app.py               # All logic embedded (preprocessing, prediction, etc.)
│   └── requirements.txt
│
├── docker_frontend/         # Frontend application
│   ├── Dockerfile
│   └── src/                 # React source code
│
├── models/                  # Docker volume (persisted artifacts, gitignored)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
│
├── docker-compose.yml       # Local multi-container orchestration
├── docker-stack.yml         # Docker Swarm deployment
├── requirements.txt         # Python dependencies
├── .dockerignore
├── .gitignore
└── README.md                # This file
```

### Key Architectural Principle

**Each container is fully autonomous:**
- **No shared Python modules** (`src/`, `inference/` folders removed)
- **All logic embedded** in container-specific files:
- `docker_training/train.py` - Contains all preprocessing, data loading, and training logic
- `docker_api/app.py` - Contains all preprocessing and prediction logic
  - `local_training/model_comparison/compare_models.py` - Contains all comparison logic
- **Explicit re-implementation** of shared concepts (e.g., preprocessing) where needed
- **Pedagogical clarity** - Easy to trace logic flow without abstraction layers

---

## 🎓 Key Takeaways

1. **Containers are ephemeral** - Use volumes for persistence
2. **Separate training and inference** - Different requirements, different containers
3. **Local experimentation first** - Select model before Dockerized training
4. **100% data for production** - Retrain selected model with full dataset
5. **Reproducibility is key** - Docker ensures consistent environments
6. **Separation of concerns** - Each container has a single responsibility
7. **Scalability** - Services can scale independently

---

## 📞 Next Steps

1. **Download dataset** from Kaggle
2. **Run local comparison** to select best model
3. **Train in Docker** with selected model
4. **Start API and frontend** to test predictions
5. **Deploy to Swarm** for multi-machine execution

**Happy Learning! 🚀**
