# 🏠 House Price Prediction: ML + Docker Production Pipeline

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Docker Deep Dive](#docker-deep-dive)
6. [Quick Start](#quick-start)
7. [Detailed Workflows](#detailed-workflows)
8. [Common Errors & Solutions](#common-errors--solutions)

---

## 🎯 Project Overview

This project demonstrates a **production-ready Machine Learning pipeline** using Docker, covering the complete ML lifecycle from data ingestion to model deployment.

### What We're Building

```
┌─────────────────────────────────────────────────────────────┐
│                    ML + Docker Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Kaggle Dataset]                                            │
│       │                                                       │
│       ▼                                                       │
│  [Data Ingestion] ──► [EDA] ──► [Preprocessing]              │
│                                                               │
│       │                                                       │
│       ▼                                                       │
│  [Training Container] ──► [Model Selection]                  │
│                                                               │
│       │                                                       │
│       ▼                                                       │
│  [Model Persistence] ──► [Docker Volume]                     │
│                                                               │
│       │                                                       │
│       ▼                                                       │
│  [Inference Container] ──► [Predictions]                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Why This Project?

**Industry Context:**
- ML models need to run in isolated, reproducible environments
- Training and inference often happen on different machines/servers
- Models must persist across container restarts
- Teams need to share identical environments

**Learning Objectives:**
- Understand Docker images vs containers
- Master volumes for data persistence
- Build separate training and inference pipelines
- Learn ML model serialization best practices
- Understand the complete ML deployment workflow

---

## 🏗️ Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      HOST MACHINE                             │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         TRAINING CONTAINER                            │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │  • Python 3.9                                  │   │   │
│  │  │  • scikit-learn, pandas, numpy                │   │   │
│  │  │  • Training scripts                           │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  │           │                                          │   │
│  │           │ writes model                             │   │
│  │           ▼                                          │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │  DOCKER VOLUME (models/)                     │   │   │
│  │  │  • model.pkl                                 │   │   │
│  │  │  • preprocessor.pkl                          │   │   │
│  │  │  • metrics.json                              │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│           ▲                                                  │
│           │ reads model                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         INFERENCE CONTAINER                           │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │  • Python 3.9 (lightweight)                  │   │   │
│  │  │  • scikit-learn, pandas                       │   │   │
│  │  │  • Inference scripts                          │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Key Concepts Explained

#### 1. **Docker Image vs Container**

**Image (Blueprint):**
- A **read-only template** with instructions for creating a container
- Contains: OS, dependencies, code
- Built once with `docker build`
- Stored in layers (cached for efficiency)

**Container (Running Instance):**
- A **running instance** of an image
- Has its own filesystem, network, and process space
- Created with `docker run`
- Ephemeral: changes inside container are lost when it stops (unless using volumes)

**Analogy:**
- Image = Class definition (blueprint)
- Container = Object instance (running process)

#### 2. **Why Separate Training and Inference Containers?**

**Training Container:**
- Heavy dependencies (Jupyter, matplotlib, seaborn for EDA)
- Large image size (~2-3 GB)
- Runs once to train model
- Needs access to full dataset

**Inference Container:**
- Minimal dependencies (only scikit-learn, pandas)
- Small image size (~500 MB)
- Runs continuously in production
- Only needs model file and prediction code

**Industry Practice:** This separation reduces production costs and attack surface.

#### 3. **Docker Volumes: The Persistence Solution**

**Problem:** Containers are ephemeral. When a container stops, all data inside is lost.

**Solution:** Docker Volumes

```
Container writes to: /app/models/model.pkl
         │
         │ (mapped via volume)
         ▼
Host directory: ./models/model.pkl (persists forever)
```

**Volume Types:**
- **Bind Mount:** Direct mapping to host directory (`-v ./models:/app/models`)
- **Named Volume:** Docker-managed storage (`-v model-storage:/app/models`)

**Why Volumes Matter:**
- Models trained in container persist on host
- Multiple containers can share the same volume
- Data survives container restarts

---

## 📊 Dataset

### Dataset Choice: House Prices - Advanced Regression Techniques

**Kaggle Link:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques

**Why This Dataset?**

1. **Realistic Business Use Case:** Predicting house prices is a common ML problem
2. **Rich Feature Set:** 79 features covering:
   - Property characteristics (size, age, quality)
   - Location features (neighborhood, zoning)
   - Temporal features (year built, year sold)
3. **Common ML Challenges:**
   - Missing values (many features have NA)
   - Categorical variables (need encoding)
   - Skewed distributions (need transformation)
   - Feature engineering opportunities
4. **Well-Documented:** Extensive community discussions and solutions

### Dataset Features Explained

**Target Variable:**
- `SalePrice`: Sale price of the house (continuous, regression problem)

**Key Feature Categories:**

1. **Property Characteristics:**
   - `LotArea`: Lot size in square feet
   - `YearBuilt`: Original construction date
   - `OverallQual`: Overall material and finish quality (1-10)
   - `OverallCond`: Overall condition rating (1-10)

2. **Location:**
   - `Neighborhood`: Physical locations within Ames city limits
   - `MSZoning`: Identifies general zoning classification

3. **Structure:**
   - `TotalBsmtSF`: Total square feet of basement area
   - `GrLivArea`: Above grade (ground) living area square feet
   - `FullBath`: Full bathrooms above grade

4. **Quality Indicators:**
   - `KitchenQual`: Kitchen quality
   - `GarageQual`: Garage quality
   - `FireplaceQu`: Fireplace quality

### Common Pitfalls in This Dataset

1. **Missing Values:**
   - `PoolQC`: 99.5% missing (most houses don't have pools)
   - `MiscFeature`: 96.3% missing
   - **Solution:** Use domain knowledge (missing = feature doesn't exist)

2. **Skewed Target:**
   - `SalePrice` is right-skewed
   - **Solution:** Log transformation

3. **Leakage Risk:**
   - `YrSold` and `MoSold` can cause data leakage if not handled carefully
   - **Solution:** Use only for validation, not feature engineering

4. **Categorical Encoding:**
   - Many ordinal categories (e.g., quality ratings)
   - **Solution:** Use ordinal encoding, not one-hot for ordered categories

---

## 📁 Project Structure

```
DOCKER-PROJECT/
│
├── data/
│   ├── raw/                    # Original Kaggle data (gitignored)
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/              # Cleaned, transformed data
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Load and validate data
│   ├── preprocess.py          # Feature engineering pipeline
│   ├── train.py               # Model training script
│   └── evaluate.py            # Model evaluation utilities
│
├── inference/
│   ├── __init__.py
│   └── predict.py             # Inference script
│
├── models/                     # Trained models (gitignored, persisted via volume)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
│
├── docker/
│   ├── Dockerfile.train       # Training container image
│   └── Dockerfile.inference   # Inference container image
│
├── docker-compose.yml         # Orchestrate both containers
├── requirements.txt           # Python dependencies
├── .dockerignore              # Exclude files from Docker build
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

### Structure Justification

**`data/raw/`:** Original, unmodified Kaggle data. Never modify this - it's our source of truth.

**`data/processed/`:** Cleaned data after preprocessing. Can be regenerated from raw data.

**`notebooks/`:** Jupyter notebooks for exploration. Not included in production containers.

**`src/`:** Production code. Modular, testable, importable.

**`inference/`:** Separate inference code. Can be deployed independently.

**`models/`:** Trained artifacts. Persisted via Docker volumes.

**`docker/`:** Dockerfiles separated for clarity. Could be in root, but separation improves organization.

---

## 🐳 Docker Deep Dive

### Understanding Docker Layers

When you build a Docker image, each instruction creates a **layer**:

```dockerfile
FROM python:3.9-slim          # Layer 1: Base OS + Python
RUN pip install pandas         # Layer 2: Install pandas
COPY src/ /app/src/            # Layer 3: Copy code
RUN python train.py            # Layer 4: Train model
```

**Why Layers Matter:**
- Layers are **cached**
- If `src/` changes, only layers 3-4 rebuild
- Layers 1-2 are reused (faster builds)

**Best Practice:** Order Dockerfile instructions from least to most frequently changing.

### COPY vs Volume

**COPY (in Dockerfile):**
- Copies files **into the image** at build time
- Files become part of the image
- Use for: code, static configs
- **Cannot** access host files at runtime

**Volume (at runtime):**
- Maps host directory **to container** at runtime
- Files are shared between host and container
- Use for: data, models, logs
- **Can** access host files at runtime

**Example:**
```dockerfile
# In Dockerfile (build time)
COPY src/ /app/src/              # Code goes into image

# At runtime (docker run)
docker run -v ./models:/app/models  # Models directory mapped from host
```

### RUN vs CMD vs ENTRYPOINT

**RUN:**
- Executes during **image build**
- Creates a new layer
- Use for: installing packages, compiling code
- Example: `RUN pip install pandas`

**CMD:**
- Default command when container **starts**
- Can be overridden: `docker run image echo "hello"`
- Use for: default behavior
- Example: `CMD ["python", "train.py"]`

**ENTRYPOINT:**
- Command that **always runs** when container starts
- Cannot be overridden (only arguments can be appended)
- Use for: fixed entry point
- Example: `ENTRYPOINT ["python"]` + `CMD ["train.py"]` = `python train.py`

**Best Practice:** Use `CMD` for flexibility, `ENTRYPOINT` when you need guaranteed execution.

---

## 🚀 Quick Start

### Prerequisites

1. **Docker installed:** https://docs.docker.com/get-docker/
2. **Kaggle API credentials:**
   - Go to Kaggle → Account → API → Create New Token
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### Step 1: Download Dataset

```bash
# Set Kaggle credentials (one-time setup)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Or on Windows PowerShell:
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"
```

### Step 2: Build Training Image

```bash
docker build -f docker/Dockerfile.train -t house-price-train .
```

**What Happens:**
1. Docker reads `Dockerfile.train`
2. Starts with `python:3.9-slim` base image
3. Installs dependencies from `requirements.txt`
4. Copies source code
5. Creates image tagged `house-price-train`

### Step 3: Run Training

```bash
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data house-price-train
```

**What Happens:**
1. Container starts from `house-price-train` image
2. Volume maps `./models` → `/app/models` (model persists on host)
3. Volume maps `./data` → `/app/data` (data accessible)
4. Training script runs
5. Model saved to `/app/models` (which is `./models` on host)
6. Container stops

### Step 4: Build Inference Image

```bash
docker build -f docker/Dockerfile.inference -t house-price-inference .
```

### Step 5: Run Inference

```bash
docker run -v ${PWD}/models:/app/models house-price-inference python predict.py --input data/processed/test_processed.csv
```

---

## 🔄 Detailed Workflows

### Training Workflow

```
1. Data Ingestion
   ├── Download from Kaggle
   ├── Validate schema
   └── Save to data/raw/

2. Preprocessing
   ├── Handle missing values
   ├── Encode categoricals
   ├── Scale numericals
   └── Save to data/processed/

3. Training
   ├── Split: train/val/test (60/20/20)
   ├── Train 3 models:
   │   ├── Linear Regression (baseline)
   │   ├── Random Forest
   │   └── Gradient Boosting
   ├── Evaluate on validation set
   └── Select best model

4. Final Evaluation
   ├── Evaluate best model on test set
   ├── Save metrics to metrics.json
   └── Save model to model.pkl
```

### Inference Workflow

```
1. Load Model
   ├── Load model.pkl from volume
   ├── Load preprocessor.pkl
   └── Validate model version

2. Preprocess Input
   ├── Apply same transformations
   └── Validate feature schema

3. Predict
   ├── Generate predictions
   └── Return results (JSON/CSV)
```

---

## ❌ Common Errors & Solutions

### Error 1: "Model file not found"

**Symptom:** Inference container can't find `model.pkl`

**Cause:** Volume not mounted or model not trained yet

**Solution:**
```bash
# Check if model exists
ls models/model.pkl

# Ensure volume is mounted
docker run -v ${PWD}/models:/app/models ...
```

### Error 2: "Permission denied" when writing to volume

**Symptom:** Container can't write to mounted volume

**Cause:** File permissions mismatch (Linux containers on Windows)

**Solution:**
```bash
# On Windows, ensure volume path is correct
docker run -v C:/Users/hp/Desktop/DOCKER-PROJECT/models:/app/models ...
```

### Error 3: "Module not found" in container

**Symptom:** ImportError when running container

**Cause:** Dependency not in `requirements.txt` or not installed in image

**Solution:**
1. Add to `requirements.txt`
2. Rebuild image: `docker build ...`

### Error 4: "Out of memory" during training

**Symptom:** Container killed during training

**Cause:** Insufficient memory allocation

**Solution:**
```bash
# Increase memory limit
docker run --memory="4g" ...
```

---

## 📚 Learning Roadmap

### Week 1: Understanding the Basics
- [ ] Understand Docker images vs containers
- [ ] Master volume mounting
- [ ] Run training container successfully

### Week 2: ML Pipeline
- [ ] Understand preprocessing pipeline
- [ ] Train multiple models
- [ ] Evaluate and select best model

### Week 3: Production Deployment
- [ ] Build inference container
- [ ] Test end-to-end workflow
- [ ] Understand Docker Compose

### Week 4: Advanced Topics
- [ ] Optimize Dockerfile layers
- [ ] Add model versioning
- [ ] Implement CI/CD pipeline

---

## 🎓 Key Takeaways

1. **Containers are ephemeral** - Use volumes for persistence
2. **Separate training and inference** - Different requirements, different containers
3. **Layer caching matters** - Order Dockerfile instructions wisely
4. **Model serialization is critical** - Use joblib/pickle, version your models
5. **Reproducibility is key** - Pin dependency versions, document everything

---

## 📞 Next Steps

1. Review the code in `src/` with inline comments
2. Run the training pipeline
3. Experiment with different models
4. Deploy inference container
5. Try Docker Compose for orchestration

**Happy Learning! 🚀**
