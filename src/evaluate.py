"""
Evaluation Module

Purpose:
    - Load saved model and evaluate on new data
    - Compare model versions
    - Generate evaluation reports

This module is useful for:
- Evaluating model on holdout test set
- Comparing different model versions
- Monitoring model performance over time
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
from typing import Dict

from src.preprocess import handle_missing_values, create_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_preprocessor(model_dir: str = "models"):
    """
    Load saved model and preprocessor.
    
    Args:
        model_dir: Directory containing model.pkl and preprocessor.pkl
    
    Returns:
        Tuple of (model, preprocessor)
    
    Why Load Both?
    - Model needs preprocessor to transform input data
    - Preprocessor must match training-time preprocessing
    - Both must be loaded together for consistency
    """
    model_dir = Path(model_dir)
    
    model_path = model_dir / "model.pkl"
    preprocessor_path = model_dir / "preprocessor.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    logger.info(f"Model and preprocessor loaded from {model_dir}")
    return model, preprocessor


def evaluate_saved_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: str = "models"
) -> Dict[str, float]:
    """
    Evaluate saved model on new data.
    
    Args:
        X: Features DataFrame
        y: Target Series
        model_dir: Directory containing saved model
    
    Returns:
        Dictionary of evaluation metrics
    
    Use Cases:
    - Evaluate on test set after training
    - Evaluate on new production data
    - Compare model versions
    """
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor(model_dir)
    
    # Preprocess data (same as training)
    X_processed = handle_missing_values(X)
    X_processed = create_features(X_processed)
    X_transformed = preprocessor.transform(X_processed)
    
    # Make predictions
    y_pred = model.predict(X_transformed)
    
    # Calculate metrics
    from src.train import evaluate_model
    metrics = evaluate_model(y.values, y_pred, "Saved Model")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from src.data_loader import DataLoader
    
    loader = DataLoader()
    test_df = loader.load_test_data()
    
    # Note: Test set doesn't have SalePrice in real scenario
    # This is just for demonstration
    print("Evaluation module loaded successfully")
