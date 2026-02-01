"""
Inference Module (Production Prediction)

Purpose:
    - Load trained model
    - Accept new input data
    - Generate predictions
    - Output results in production-ready format

Key Differences from Training:
- No training code (lighter dependencies)
- Focus on speed and reliability
- Error handling for production use
- Clear output format (JSON/CSV)
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
from pathlib import Path
import logging
import sys

# Add /app to Python path if not already there (for Docker container)
# This ensures 'src' module can be imported
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.preprocess import handle_missing_values, create_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HousePricePredictor:
    """
    Production inference class.
    
    Why a class?
    - Loads model once (expensive operation)
    - Reuses loaded model for multiple predictions
    - Better for API deployment (Flask/FastAPI)
    - Encapsulates prediction logic
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor by loading model and preprocessor.
        
        Args:
            model_dir: Directory containing model.pkl and preprocessor.pkl
        
        What happens here?
        1. Loads model from disk (expensive, do once)
        2. Loads preprocessor from disk
        3. Validates both are loaded correctly
        
        Error Handling:
        - FileNotFoundError: Model files don't exist
        - ValueError: Model/preprocessor incompatible
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load model and preprocessor from disk."""
        model_path = self.model_dir / "model.pkl"
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Train the model first using: python src/train.py"
            )
        
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. "
                f"Train the model first using: python src/train.py"
            )
        
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)
        
        self.is_loaded = True
        logger.info("Model and preprocessor loaded successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            X: DataFrame with features (must match training schema)
        
        Returns:
            Array of predicted house prices
        
        Prediction Pipeline:
        1. Validate input schema
        2. Handle missing values (same as training)
        3. Feature engineering (same as training)
        4. Transform using fitted preprocessor
        5. Generate predictions
        
        Why Same Preprocessing?
        - Must match training-time preprocessing exactly
        - Different preprocessing = wrong predictions
        - Preprocessor was fitted on training data
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Generating predictions for {len(X)} samples")
        
        # Preprocess (same as training)
        X_processed = handle_missing_values(X.copy())
        X_processed = create_features(X_processed)
        
        # Transform using fitted preprocessor
        X_transformed = self.preprocessor.transform(X_processed)
        
        # Predict
        predictions = self.model.predict(X_transformed)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def predict_single(self, house_data: dict) -> float:
        """
        Predict price for a single house.
        
        Args:
            house_data: Dictionary with house features
        
        Returns:
            Predicted price
        
        Use Case:
        - API endpoint: POST /predict with JSON body
        - Single prediction requests
        """
        df = pd.DataFrame([house_data])
        predictions = self.predict(df)
        return float(predictions[0])


def predict_from_file(input_file: str, output_file: str = None, model_dir: str = "models") -> pd.DataFrame:
    """
    Predict from CSV file and save results.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save predictions (optional)
        model_dir: Directory containing model
    
    Returns:
        DataFrame with predictions
    
    Input Format:
    - CSV file with feature columns
    - Must match training data schema
    - Can have 'Id' column (will be preserved)
    
    Output Format:
    - CSV with 'Id' (if present) and 'PredictedPrice' columns
    - Or JSON format
    """
    logger.info(f"Loading input data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Preserve ID column if present
    has_id = 'Id' in df.columns
    if has_id:
        ids = df['Id'].copy()
        df = df.drop('Id', axis=1)
    
    # Initialize predictor
    predictor = HousePricePredictor(model_dir=model_dir)
    
    # Generate predictions
    predictions = predictor.predict(df)
    
    # Create output DataFrame
    result_df = pd.DataFrame()
    if has_id:
        result_df['Id'] = ids
    result_df['PredictedPrice'] = predictions
    
    # Save results
    if output_file:
        if output_file.endswith('.json'):
            result_df.to_json(output_file, orient='records', indent=2)
            logger.info(f"Predictions saved to {output_file} (JSON)")
        else:
            result_df.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file} (CSV)")
    else:
        # Print to stdout
        print(result_df.to_string())
    
    return result_df


def main():
    """
    Command-line interface for inference.
    
    Usage:
        python inference/predict.py --input data/test.csv --output predictions.csv
        python inference/predict.py --input data/test.csv --output predictions.json
    """
    parser = argparse.ArgumentParser(description='Predict house prices using trained model')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with house features'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for predictions (CSV or JSON). If not specified, prints to stdout'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing model.pkl and preprocessor.pkl'
    )
    
    args = parser.parse_args()
    
    try:
        result_df = predict_from_file(
            input_file=args.input,
            output_file=args.output,
            model_dir=args.model_dir
        )
        
        logger.info("Prediction completed successfully")
        
        # Print summary statistics
        print("\nPrediction Summary:")
        print(f"  Number of predictions: {len(result_df)}")
        print(f"  Mean predicted price: ${result_df['PredictedPrice'].mean():,.2f}")
        print(f"  Min predicted price: ${result_df['PredictedPrice'].min():,.2f}")
        print(f"  Max predicted price: ${result_df['PredictedPrice'].max():,.2f}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
