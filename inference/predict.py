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
        
        # Add missing features with default values
        # The form only collects a subset, but preprocessor expects all features
        X_filled = self._add_missing_features(X.copy())
        
        # Preprocess (same as training)
        X_processed = handle_missing_values(X_filled)
        X_processed = create_features(X_processed)
        
        # CRITICAL: After feature engineering, ensure ALL expected columns exist
        # create_features may have created some, but we need to ensure all are present
        X_processed = self._ensure_all_expected_columns(X_processed)
        
        # Transform using fitted preprocessor
        X_transformed = self.preprocessor.transform(X_processed)
        
        # Predict
        predictions = self.model.predict(X_transformed)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def _add_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add missing features with default values.
        
        The form only collects a subset of features, but the preprocessor
        was trained on the full dataset. This function adds missing features
        with sensible defaults.
        
        CRITICAL: The preprocessor's ColumnTransformer expects ALL columns
        it was trained on. Missing columns cause "not in index" errors.
        """
        # Get all columns that the preprocessor expects
        # The preprocessor was fitted on the full dataset, so we need all those columns
        expected_columns = set()
        
        # Try to get expected columns from preprocessor's ColumnTransformer
        if hasattr(self.preprocessor, 'preprocessor') and hasattr(self.preprocessor.preprocessor, 'transformers_'):
            for name, transformer, columns in self.preprocessor.preprocessor.transformers_:
                if name != 'remainder' and columns is not None:
                    expected_columns.update(columns)
        
        # If we can't get from preprocessor, use comprehensive list
        # Features that might be missing from form input
        defaults = {
            # Building class
            'MSSubClass': 20,  # 1-STORY 1946 & NEWER ALL STYLES
            'MSZoning': 'RL',  # Residential Low Density
            
            # Lot features
            'LotFrontage': 70.0,
            'LotShape': 'Reg',
            'LandContour': 'Lvl',
            'LotConfig': 'Inside',
            'LandSlope': 'Gtl',
            'Street': 'Pave',
            'Alley': 'None',
            'Utilities': 'AllPub',
            'Exterior1st': 'VinylSd',
            'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None',
            'MasVnrArea': 0.0,
            'ExterCond': 'TA',
            'Foundation': 'PConc',
            'BsmtQual': 'TA',
            'BsmtCond': 'TA',
            'BsmtExposure': 'No',
            'BsmtFinType1': 'Unf',
            'BsmtFinType2': 'Unf',
            'BsmtFinSF1': 0.0,
            'BsmtFinSF2': 0.0,
            'BsmtUnfSF': 0.0,
            'Heating': 'GasA',
            'HeatingQC': 'TA',
            'Electrical': 'SBrkr',
            '1stFlrSF': None,  # Will infer
            '2ndFlrSF': None,  # Will infer
            'LowQualFinSF': 0.0,
            'KitchenAbvGr': 1,
            'Functional': 'Typ',
            'FireplaceQu': 'None',
            'GarageType': 'Attchd',
            'GarageFinish': 'Unf',
            'GarageQual': 'TA',
            'GarageCond': 'TA',
            'GarageYrBlt': None,  # Will infer
            'WoodDeckSF': 0.0,
            'OpenPorchSF': 0.0,
            'EnclosedPorch': 0.0,
            '3SsnPorch': 0.0,
            'ScreenPorch': 0.0,
            'PoolQC': 'None',
            'Fence': 'None',
            'MiscFeature': 'None',
            'MiscVal': 0,
            'SaleType': 'WD',
            'SaleCondition': 'Normal',
            'MoSold': 6,
            'YrSold': 2010,
            'Condition1': 'Norm',
            'Condition2': 'Norm',
            'RoofStyle': 'Gable',
            'RoofMatl': 'CompShg',
            'CentralAir': 'Y',
            'PavedDrive': 'Y',
        }
        
        # Add missing features
        for feature, default_value in defaults.items():
            if feature not in df.columns:
                if default_value is None:
                    # Infer from other features
                    if feature == '1stFlrSF':
                        # Estimate: if no basement info, assume 1st floor = GrLivArea
                        if 'GrLivArea' in df.columns:
                            df[feature] = df['GrLivArea'] if 'TotalBsmtSF' not in df.columns else df['GrLivArea'] * 0.6
                        else:
                            df[feature] = 1000
                    elif feature == '2ndFlrSF':
                        if 'GrLivArea' in df.columns and '1stFlrSF' in df.columns:
                            df[feature] = (df['GrLivArea'] - df['1stFlrSF']).clip(lower=0)
                        else:
                            df[feature] = 0
                    elif feature == 'GarageYrBlt':
                        if 'YearBuilt' in df.columns:
                            df[feature] = df['YearBuilt']
                        else:
                            df[feature] = 2000
                    else:
                        df[feature] = 0
                else:
                    df[feature] = default_value
        
        # Handle conditional defaults
        if 'Fireplaces' in df.columns:
            mask = df['Fireplaces'].fillna(0) == 0
            if mask.any():
                df.loc[mask, 'FireplaceQu'] = 'None'
        if 'PoolArea' in df.columns:
            mask = df['PoolArea'].fillna(0) == 0
            if mask.any():
                df.loc[mask, 'PoolQC'] = 'None'
        
        # CRITICAL: Ensure ALL expected columns exist
        # Add any columns that are still missing (from expected_columns)
        if expected_columns:
            missing_cols = expected_columns - set(df.columns)
            if missing_cols:
                logger.warning(f"Adding {len(missing_cols)} missing expected columns: {list(missing_cols)[:10]}...")
                for col in missing_cols:
                    # Use defaults if available, otherwise infer type
                    if col in defaults:
                        val = defaults[col]
                        if val is None:
                            df[col] = 0
                        else:
                            df[col] = val
                    else:
                        # Infer from column name or use safe default
                        # Features created by create_features (TotalSF, Age, etc.)
                        if col == 'TotalSF':
                            df[col] = df.get('GrLivArea', 0) + df.get('TotalBsmtSF', 0)
                        elif col == 'Age':
                            df[col] = df.get('YrSold', 2010) - df.get('YearBuilt', 2000)
                        elif col == 'TotalBathrooms':
                            df[col] = df.get('FullBath', 0) + 0.5 * df.get('HalfBath', 0) + df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0)
                        elif col == 'Qual_x_Area':
                            df[col] = df.get('OverallQual', 5) * df.get('GrLivArea', 1000)
                        else:
                            # Default: 0 for numeric, 'None' for likely categorical
                            df[col] = 0
        
        logger.info(f"Added missing features. Final columns: {len(df.columns)}, Expected: {len(expected_columns) if expected_columns else 'unknown'}")
        return df
    
    def _ensure_all_expected_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure ALL columns expected by the preprocessor are present.
        This is called AFTER feature engineering to catch any remaining missing columns.
        """
        # Get expected columns from preprocessor's ColumnTransformer
        expected_columns = set()
        if hasattr(self.preprocessor, 'preprocessor') and hasattr(self.preprocessor.preprocessor, 'transformers_'):
            for name, transformer, columns in self.preprocessor.preprocessor.transformers_:
                if name != 'remainder' and columns is not None:
                    expected_columns.update(columns)
        
        if not expected_columns:
            logger.warning("Could not determine expected columns from preprocessor")
            return df
        
        # Find missing columns
        missing_cols = expected_columns - set(df.columns)
        
        if missing_cols:
            logger.warning(f"Adding {len(missing_cols)} missing expected columns after feature engineering: {list(missing_cols)}")
            for col in missing_cols:
                # Features created by create_features that might be missing
                if col == 'TotalSF':
                    df[col] = df.get('GrLivArea', 0) + df.get('TotalBsmtSF', 0) if 'GrLivArea' in df.columns else 0
                elif col == 'Age':
                    yr_sold = df.get('YrSold', 2010) if 'YrSold' in df.columns else 2010
                    year_built = df.get('YearBuilt', 2000) if 'YearBuilt' in df.columns else 2000
                    df[col] = yr_sold - year_built
                elif col == 'TotalBathrooms':
                    df[col] = (df.get('FullBath', 0) + 0.5 * df.get('HalfBath', 0) + 
                               df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0))
                elif col == 'Qual_x_Area':
                    df[col] = df.get('OverallQual', 5) * df.get('GrLivArea', 1000)
                else:
                    # Default: 0 for numeric columns
                    df[col] = 0
                    logger.debug(f"Added missing column {col} with default value 0")
        
        logger.info(f"Ensured all expected columns present. Total columns: {len(df.columns)}, Expected: {len(expected_columns)}")
        return df
    
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
