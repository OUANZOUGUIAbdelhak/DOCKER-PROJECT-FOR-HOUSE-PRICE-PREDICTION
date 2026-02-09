# Local Training & Model Comparison

## Purpose

This directory contains scripts for **local experimentation** (outside Docker). This is where you:

1. **Explore the data** with partial/reduced datasets
2. **Train multiple models** and compare their performance
3. **Select the best model** based on validation metrics
4. **Compare local vs Dockerized execution** (required by project instructions)

## Key Principle

**Local training uses PARTIAL data** for quick experimentation.  
**Dockerized training uses 100% of data** for final production model.

## Structure

```
local_training/
├── experiments/          # Individual model training experiments
├── model_comparison/    # Comparison logic and metrics
└── README.md           # This file
```

## Usage

### Step 1: Run Model Comparison

```bash
# From project root
python local_training/model_comparison/compare_models.py
```

This will:
- Load data from `data/raw/`
- Train multiple models (Linear Regression, Random Forest, Gradient Boosting, Neural Network)
- Compare performance on validation set
- Save comparison results to `local_training/model_comparison/results.json`
- Display the best model

**Note:** Neural Network requires TensorFlow. Install it with: `pip install tensorflow`

### Step 2: Review Results

Check `local_training/model_comparison/results.json` to see:
- Validation metrics for each model
- Best model selection
- Comparison summary

### Step 3: Use Selected Model for Dockerized Training

After selecting the best model locally, use it in Dockerized training:

```bash
docker-compose run --rm training --model gradient_boosting
```

## Local vs Dockerized Training Comparison

| Aspect | Local Training | Dockerized Training |
|--------|---------------|---------------------|
| **Data Size** | Partial/reduced (for speed) | 100% of dataset |
| **Purpose** | Model selection & experimentation | Final production model |
| **Environment** | Local Python environment | Docker container |
| **Reproducibility** | Depends on local setup | Guaranteed via Docker |
| **Output** | Comparison metrics | Final trained model |

## Why This Separation?

1. **Speed**: Local experiments run faster with less data
2. **Iteration**: Easy to try different models/hyperparameters
3. **Comparison**: Explicitly demonstrates local vs Docker execution
4. **Production**: Dockerized training ensures reproducibility
