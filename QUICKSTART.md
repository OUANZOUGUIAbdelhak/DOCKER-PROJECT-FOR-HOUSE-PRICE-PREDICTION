# 🚀 Quick Start Guide

This guide will walk you through running the complete ML + Docker pipeline step-by-step.

## Prerequisites Checklist

- [ ] Docker installed and running
- [ ] Kaggle API credentials set up
- [ ] Git (optional, for cloning)

---

## Step 1: Set Up Kaggle Credentials

### Option A: Using Kaggle API File (Recommended)

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New Token"
4. This downloads `kaggle.json`

5. **On Windows:**
   ```powershell
   # Create .kaggle directory
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
   
   # Copy kaggle.json there
   Copy-Item kaggle.json "$env:USERPROFILE\.kaggle\kaggle.json"
   ```

6. **On Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Option B: Using Environment Variables

```powershell
# Windows PowerShell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"
```

```bash
# Linux/Mac
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

---

## Step 2: Download Dataset

### Option A: Using Python Script (Recommended)

```bash
python download_data.py
```

### Option B: Manual Download

1. Go to https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
2. Download `train.csv` and `test.csv`
3. Place them in `data/raw/` directory

---

## Step 3: Build Docker Images

### Build Training Image

```bash
docker build -f docker/Dockerfile.train -t house-price-train .
```

**What happens:**
- Docker reads `Dockerfile.train`
- Downloads Python base image
- Installs dependencies from `requirements.txt`
- Copies source code
- Creates image tagged `house-price-train`

**Expected output:**
```
Successfully built <image_id>
Successfully tagged house-price-train:latest
```

### Build Inference Image

```bash
docker build -f docker/Dockerfile.inference -t house-price-inference .
```

**Expected time:** 2-5 minutes per image (first time), faster on subsequent builds (caching)

---

## Step 4: Run Training

### Using Docker Run

```bash
# Windows PowerShell
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data house-price-train

# Linux/Mac
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data house-price-train
```

**What happens:**
1. Container starts from `house-price-train` image
2. Volume maps `./models` → `/app/models` (model persists on host)
3. Volume maps `./data` → `/app/data` (data accessible)
4. Training script runs:
   - Loads data
   - Preprocesses features
   - Trains 3 models (Linear Regression, Random Forest, Gradient Boosting)
   - Evaluates on validation set
   - Selects best model
   - Evaluates on test set
   - Saves model to `./models/model.pkl`
5. Container stops

**Expected output:**
```
Starting ML Training Pipeline
[Step 1] Loading data...
[Step 2] Preprocessing data...
[Step 3] Splitting data...
[Step 4] Training models...
Best Model: gradient_boosting
Test RMSE: $25,123.45
```

**Expected time:** 5-15 minutes (depending on hardware)

### Using Docker Compose

```bash
docker-compose run --rm training
```

---

## Step 5: Verify Model Saved

Check that model files were created:

```bash
# Windows PowerShell
ls models/

# Linux/Mac
ls models/
```

You should see:
- `model.pkl` - Trained model
- `preprocessor.pkl` - Fitted preprocessor
- `metrics.json` - Evaluation metrics

---

## Step 6: Run Inference

### Prepare Test Data

Create a sample input file or use the test set:

```bash
# Copy test data (if you have it)
cp data/raw/test.csv data/processed/test_input.csv
```

### Run Predictions

```bash
# Windows PowerShell
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data house-price-inference python inference/predict.py --input data/processed/test_input.csv --output predictions.csv

# Linux/Mac
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data house-price-inference python inference/predict.py --input data/processed/test_input.csv --output predictions.csv
```

**What happens:**
1. Container starts from `house-price-inference` image
2. Loads model from `/app/models` (mapped from `./models`)
3. Loads input data
4. Preprocesses data (same as training)
5. Generates predictions
6. Saves to `predictions.csv`

**Expected output:**
```
Loading model from /app/models/model.pkl
Generating predictions for 1459 samples
Predictions saved to predictions.csv

Prediction Summary:
  Number of predictions: 1459
  Mean predicted price: $180,921.23
```

### Using Docker Compose

```bash
docker-compose run --rm inference python inference/predict.py --input data/processed/test_input.csv --output predictions.csv
```

---

## Step 7: Explore Results

### View Predictions

```bash
# Windows PowerShell
cat predictions.csv

# Linux/Mac
head predictions.csv
```

### View Model Metrics

```bash
# Windows PowerShell
cat models/metrics.json

# Linux/Mac
cat models/metrics.json
```

---

## Common Issues & Solutions

### Issue: "Model file not found"

**Cause:** Training didn't complete or volume not mounted

**Solution:**
```bash
# Check if model exists
ls models/model.pkl

# Re-run training
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data house-price-train
```

### Issue: "Permission denied" on Windows

**Cause:** Docker volume path format

**Solution:**
```powershell
# Use forward slashes or full path
docker run -v C:/Users/hp/Desktop/DOCKER-PROJECT/models:/app/models ...
```

### Issue: "Kaggle API authentication failed"

**Cause:** Missing or incorrect credentials

**Solution:**
1. Verify `kaggle.json` is in correct location
2. Check credentials are valid
3. Ensure you've accepted competition rules on Kaggle

### Issue: "Out of memory" during training

**Cause:** Insufficient Docker memory allocation

**Solution:**
1. Increase Docker memory limit (Docker Desktop → Settings → Resources)
2. Reduce model complexity in `src/train.py`

---

## Next Steps

1. **Explore the Code:**
   - Read `src/train.py` to understand training pipeline
   - Read `src/preprocess.py` to understand feature engineering
   - Read `inference/predict.py` to understand inference

2. **Experiment:**
   - Modify hyperparameters in `src/train.py`
   - Add new features in `src/preprocess.py`
   - Try different models

3. **Learn Docker:**
   - Read Dockerfile comments (extensive explanations)
   - Experiment with Docker commands
   - Understand volumes and layers

4. **Production Deployment:**
   - Add API wrapper (Flask/FastAPI)
   - Set up CI/CD pipeline
   - Deploy to cloud (AWS, GCP, Azure)

---

## Learning Path

### Week 1: Basics
- [ ] Run complete pipeline end-to-end
- [ ] Understand Docker images vs containers
- [ ] Understand volumes

### Week 2: ML Pipeline
- [ ] Modify preprocessing
- [ ] Add new models
- [ ] Experiment with hyperparameters

### Week 3: Docker Advanced
- [ ] Optimize Dockerfile layers
- [ ] Use Docker Compose effectively
- [ ] Understand multi-stage builds

### Week 4: Production
- [ ] Add API layer
- [ ] Set up monitoring
- [ ] Deploy to cloud

---

## Getting Help

- Read `README.md` for detailed explanations
- Check Dockerfile comments for Docker concepts
- Review code comments for ML concepts
- Check Common Errors section in README

**Happy Learning! 🎉**
