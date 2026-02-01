"""
Training Module

Purpose:
    - Train multiple ML models
    - Compare model performance
    - Select best model
    - Save model and metrics

Key ML Concepts:
    - Train/Validation/Test Split: Prevent overfitting
    - Model Comparison: Multiple algorithms, select best
    - Bias-Variance Tradeoff: Understanding model complexity
    - Cross-Validation: Robust performance estimation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
import logging
import sys
from typing import Dict, Any

# Add /app to Python path if not already there (for Docker container)
# This ensures 'src' module can be imported
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.data_loader import DataLoader
from src.preprocess import HousePricePreprocessor, handle_missing_values, create_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_test_split_with_validation(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Target
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Why Three Splits?
    
    1. Training Set (60%):
       - Used to train models
       - Models learn patterns from this data
    
    2. Validation Set (20%):
       - Used to tune hyperparameters
       - Used to compare models
       - Prevents overfitting to test set
    
    3. Test Set (20%):
       - Used ONLY for final evaluation
       - Never used for training or tuning
       - Gives unbiased performance estimate
    
    Data Leakage Warning:
    - Never use test set for model selection
    - Test set is "unseen" data - represents real-world performance
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (train_size + val_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of model (for logging)
    
    Returns:
        Dictionary of metrics
    
    Metrics Explained:
    
    1. RMSE (Root Mean Squared Error):
       - sqrt(mean((y_true - y_pred)^2))
       - Penalizes large errors heavily
       - Same units as target (dollars)
       - Lower is better
    
    2. MAE (Mean Absolute Error):
       - mean(|y_true - y_pred|)
       - Average error magnitude
       - Robust to outliers
       - Lower is better
    
    3. R² (R-squared):
       - Proportion of variance explained
       - 1.0 = perfect predictions
       - 0.0 = model is as good as predicting mean
       - Can be negative (worse than mean)
       - Higher is better
    
    Why Multiple Metrics?
    - RMSE: Focuses on large errors (important for expensive houses)
    - MAE: Average error (easier to interpret)
    - R²: Overall model quality
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    logger.info(f"{model_name} Metrics:")
    logger.info(f"  RMSE: ${rmse:,.2f}")
    logger.info(f"  MAE: ${mae:,.2f}")
    logger.info(f"  R²: {r2:.4f}")
    
    return metrics


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Train Linear Regression model.
    
    Why Linear Regression?
    - Baseline model (simple, interpretable)
    - Fast to train
    - Good for understanding data
    
    Bias-Variance Tradeoff:
    - High Bias: Assumes linear relationship (may be too simple)
    - Low Variance: Stable predictions (less sensitive to training data)
    - Good for: Linear relationships, small datasets
    """
    logger.info("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 20,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Train Random Forest model.
    
    Why Random Forest?
    - Handles non-linear relationships
    - Feature importance available
    - Robust to outliers
    
    Bias-Variance Tradeoff:
    - Lower Bias: Can capture complex patterns
    - Higher Variance: More sensitive to training data
    - Good for: Non-linear relationships, medium datasets
    
    How Random Forest Works:
    1. Train many decision trees on random subsets of data
    2. Each tree votes on prediction
    3. Average votes = final prediction
    4. Reduces overfitting through ensemble
    """
    logger.info("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    random_state: int = 42
) -> GradientBoostingRegressor:
    """
    Train Gradient Boosting model.
    
    Why Gradient Boosting?
    - Often best performance
    - Handles complex patterns
    - Good for structured data
    
    Bias-Variance Tradeoff:
    - Very Low Bias: Can capture very complex patterns
    - Higher Variance: Sensitive to training data
    - Good for: Complex relationships, large datasets
    
    How Gradient Boosting Works:
    1. Train first tree
    2. Calculate errors (residuals)
    3. Train next tree to predict errors
    4. Repeat, each tree corrects previous mistakes
    5. Sum all trees = final prediction
    
    Why Often Best?
    - Sequentially improves (unlike Random Forest which trains in parallel)
    - Can focus on hard examples
    """
    logger.info("Training Gradient Boosting...")
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, Any]:
    """
    Train multiple models and compare performance.
    
    Returns:
        Dictionary with models and their validation metrics
    
    Why Train Multiple Models?
    - No single "best" algorithm for all problems
    - Different models capture different patterns
    - Ensemble can outperform individual models
    
    Model Selection Process:
    1. Train all models on training set
    2. Evaluate on validation set
    3. Select best model based on validation performance
    4. Final evaluation on test set (only for best model)
    """
    models = {}
    results = {}
    
    # Train Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    lr_pred = lr_model.predict(X_val)
    lr_metrics = evaluate_model(y_val, lr_pred, "Linear Regression")
    models['linear_regression'] = lr_model
    results['linear_regression'] = lr_metrics
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    rf_metrics = evaluate_model(y_val, rf_pred, "Random Forest")
    models['random_forest'] = rf_model
    results['random_forest'] = rf_metrics
    
    # Train Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_pred = gb_model.predict(X_val)
    gb_metrics = evaluate_model(y_val, gb_pred, "Gradient Boosting")
    models['gradient_boosting'] = gb_model
    results['gradient_boosting'] = gb_metrics
    
    return models, results


def select_best_model(results: Dict[str, Dict[str, float]], metric: str = 'rmse') -> str:
    """
    Select best model based on validation metrics.
    
    Args:
        results: Dictionary of model results
        metric: Metric to optimize ('rmse', 'mae', or 'r2')
    
    Returns:
        Name of best model
    
    Selection Strategy:
    - For RMSE/MAE: Lower is better
    - For R²: Higher is better
    
    Why Validation Set?
    - Test set is "sacred" - only for final evaluation
    - Validation set prevents overfitting to test set
    - Gives unbiased model comparison
    """
    if metric in ['rmse', 'mae']:
        # Lower is better
        best_model = min(results.items(), key=lambda x: x[1][metric])
    else:  # r2
        # Higher is better
        best_model = max(results.items(), key=lambda x: x[1][metric])
    
    logger.info(f"\nBest model (by {metric}): {best_model[0]}")
    logger.info(f"  {metric}: {best_model[1][metric]}")
    
    return best_model[0]


def save_model_and_metrics(
    model: Any,
    preprocessor: HousePricePreprocessor,
    metrics: Dict[str, float],
    model_dir: str = "models"
) -> None:
    """
    Save model, preprocessor, and metrics to disk.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        metrics: Evaluation metrics
        model_dir: Directory to save files
    
    Why Save All Three?
    - Model: For predictions
    - Preprocessor: Must use same preprocessing for inference
    - Metrics: Track model performance over time
    
    Model Persistence Risks:
    1. Version Mismatch: sklearn version must match
    2. Missing Dependencies: All imports must be available
    3. Data Schema Changes: Features must match training data
    4. Model Drift: Performance degrades over time (retrain periodically)
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessor
    preprocessor_path = model_dir / "preprocessor.pkl"
    preprocessor.save(str(preprocessor_path))
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Save metrics
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


def main():
    """
    Main training pipeline.
    
    This is the entry point for training.
    Called by Docker container or directly.
    """
    logger.info("=" * 60)
    logger.info("Starting ML Training Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load data
    logger.info("\n[Step 1] Loading data...")
    loader = DataLoader()
    
    # Check if data exists, if not, try to download
    try:
        train_df = loader.load_train_data()
    except FileNotFoundError:
        logger.warning("Training data not found. Attempting to download...")
        loader.download_kaggle_dataset()
        train_df = loader.load_train_data()
    
    loader.validate_schema(train_df)
    
    # Step 2: Preprocess data
    logger.info("\n[Step 2] Preprocessing data...")
    X = train_df.drop('SalePrice', axis=1)
    y = train_df['SalePrice']
    
    # Handle missing values
    X = handle_missing_values(X)
    
    # Feature engineering
    X = create_features(X)
    
    # Create and fit preprocessor
    preprocessor = HousePricePreprocessor()
    X_transformed = preprocessor.fit_transform(X, y)
    
    # Step 3: Split data
    logger.info("\n[Step 3] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_with_validation(
        pd.DataFrame(X_transformed),  # Convert back to DataFrame for splitting
        y,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2
    )
    
    # Convert back to numpy for sklearn
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    
    # Step 4: Train models
    logger.info("\n[Step 4] Training models...")
    models, val_results = train_all_models(X_train, y_train.values, X_val, y_val.values)
    
    # Step 5: Select best model
    logger.info("\n[Step 5] Selecting best model...")
    best_model_name = select_best_model(val_results, metric='rmse')
    best_model = models[best_model_name]
    
    # Step 6: Final evaluation on test set
    logger.info("\n[Step 6] Final evaluation on test set...")
    test_pred = best_model.predict(X_test)
    test_metrics = evaluate_model(y_test.values, test_pred, f"{best_model_name} (Test Set)")
    
    # Step 7: Save model and metrics
    logger.info("\n[Step 7] Saving model and metrics...")
    save_model_and_metrics(best_model, preprocessor, test_metrics)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Test RMSE: ${test_metrics['rmse']:,.2f}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
