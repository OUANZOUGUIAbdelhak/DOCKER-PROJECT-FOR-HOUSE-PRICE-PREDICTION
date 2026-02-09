"""
Dockerized Training Script - Fully Autonomous

Purpose:
    - Retrain the SELECTED model using 100% of the dataset
    - All preprocessing, data loading, and training logic contained here
    - No external dependencies on shared modules
    - Saves final model, preprocessor, and metrics to shared volume
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path
import logging
import argparse
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neural network imports (optional - only needed for neural network models)
# These are only used inside train_neural_network() which checks TENSORFLOW_AVAILABLE first
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Dummy values to prevent NameError (will never be used - function checks TENSORFLOW_AVAILABLE first)
    tf = None
    keras = None
    layers = None
    callbacks = None
    logger.info("TensorFlow not available - Only sklearn models supported (this is fine for XGBoost/sklearn models)")


# ============================================================================
# DATA LOADING (Embedded)
# ============================================================================

def load_train_data(data_dir: str = "/app/data") -> pd.DataFrame:
    """Load training data from CSV file."""
    data_path = Path(data_dir) / "raw" / "train.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Ensure data volume is mounted correctly."
        )
    
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if df.empty:
        raise ValueError("Training data is empty")
    
    if 'SalePrice' not in df.columns:
        raise ValueError("Target column 'SalePrice' not found in training data")
    
    return df


# ============================================================================
# PREPROCESSING (Embedded)
# ============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using domain knowledge."""
    df = df.copy()
    
    # Features where missing = feature doesn't exist
    none_features = {
        'PoolQC': 'None', 'MiscFeature': 'None', 'Alley': 'None', 'Fence': 'None',
        'FireplaceQu': 'None', 'GarageType': 'None', 'GarageFinish': 'None',
        'GarageQual': 'None', 'GarageCond': 'None', 'BsmtQual': 'None',
        'BsmtCond': 'None', 'BsmtExposure': 'None', 'BsmtFinType1': 'None',
        'BsmtFinType2': 'None', 'MasVnrType': 'None'
    }
    
    for feature, fill_value in none_features.items():
        if feature in df.columns:
            df[feature].fillna(fill_value, inplace=True)
    
    # Numerical features: missing = 0
    numerical_none_features = [
        'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    
    for feature in numerical_none_features:
        if feature in df.columns:
            df[feature].fillna(0, inplace=True)
    
    # Other features: use mode
    other_features = ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 
                     'Exterior2nd', 'KitchenQual', 'SaleType']
    for feature in other_features:
        if feature in df.columns and df[feature].isna().any():
            mode_value = df[feature].mode()[0] if not df[feature].mode().empty else 'None'
            df[feature].fillna(mode_value, inplace=True)
    
    # LotFrontage: Use median of neighborhood
    if 'LotFrontage' in df.columns:
        df['LotFrontage'].fillna(df.groupby('Neighborhood')['LotFrontage'].transform('median'), inplace=True)
        df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
    
    logger.info("Missing values handled using domain knowledge")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features through feature engineering."""
    df = df.copy()
    
    if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
        df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['Age'] = df['YrSold'] - df['YearBuilt']
        df['Age'].fillna(0, inplace=True)
    
    if all(col in df.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    
    if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
        df['Qual_x_Area'] = df['OverallQual'] * df['GrLivArea']
    
    logger.info("Feature engineering completed")
    return df


class HousePricePreprocessor:
    """Preprocessing pipeline for House Prices dataset."""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.is_fitted = False
    
    def _identify_feature_types(self, df: pd.DataFrame) -> Tuple[list, list, list]:
        """Identify numerical, categorical, and ordinal features."""
        exclude_cols = ['Id', 'SalePrice']
        cols = [c for c in df.columns if c not in exclude_cols]
        
        string_cols = df[cols].select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
        
        ordinal_features = [
            'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
            'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
            'GarageCond', 'PoolQC', 'LotShape', 'LandSlope', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish',
            'PavedDrive', 'Fence', 'Street', 'Alley', 'CentralAir'
        ]
        ordinal_features = [f for f in ordinal_features if f in cols and f in string_cols]
        
        categorical_features = [
            'MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical',
            'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition'
        ]
        categorical_features = [f for f in categorical_features if f in cols and f in string_cols]
        
        remaining_string_cols = [c for c in string_cols 
                                if c not in ordinal_features + categorical_features]
        categorical_features.extend(remaining_string_cols)
        
        numerical_features = [c for c in numeric_cols 
                             if c not in ordinal_features + categorical_features]
        numerical_features = [c for c in numerical_features if c not in string_cols]
        
        logger.info(f"Feature types: Numerical={len(numerical_features)}, "
                   f"Ordinal={len(ordinal_features)}, Categorical={len(categorical_features)}")
        
        return numerical_features, categorical_features, ordinal_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'HousePricePreprocessor':
        """Fit preprocessing pipeline on training data."""
        logger.info("Fitting preprocessing pipeline on training data...")
        
        numerical_features, categorical_features, ordinal_features = self._identify_feature_types(X)
        
        # Clean numerical features
        if numerical_features:
            numerical_features_clean = []
            for feat in numerical_features:
                if feat in X.columns and pd.api.types.is_numeric_dtype(X[feat]):
                    numerical_features_clean.append(feat)
            numerical_features = numerical_features_clean
        
        transformers = []
        
        if numerical_features:
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_pipeline, numerical_features))
        
        if ordinal_features:
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('ord', ordinal_pipeline, ordinal_features))
        
        if categorical_features:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, categorical_features))
        
        if not transformers:
            raise ValueError("No features to process!")
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        X_transformed = self.preprocessor.fit_transform(X)
        self.is_fitted = True
        
        logger.info(f"Preprocessing pipeline fitted. Output shape: {X_transformed.shape}")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform.")
        
        logger.info(f"Transforming data with shape {X.shape}")
        X_transformed = self.preprocessor.transform(X)
        logger.info(f"Transformed shape: {X_transformed.shape}")
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: str) -> None:
        """Save fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor.")
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")


# ============================================================================
# MODEL TRAINING (Embedded)
# ============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """Evaluate model performance."""
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


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    random_state: int = 42
):
    """Train Neural Network model using TensorFlow/Keras."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Install it with: pip install tensorflow")
    
    logger.info("Training Neural Network...")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    
    # Build model architecture
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)  # Output layer (no activation for regression)
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error as additional metric
    )
    
    # Callbacks for training
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    logger.info(f"Neural Network trained for {len(history.history['loss'])} epochs")
    
    return model


def train_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray = None, y_val: np.ndarray = None):
    """Train the specified model."""
    if model_name == 'linear_regression':
        logger.info("Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_name == 'random_forest':
        logger.info("Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
    elif model_name == 'gradient_boosting':
        logger.info("Training Gradient Boosting...")
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)
    elif model_name == 'neural_network':
        if X_val is None or y_val is None:
            raise ValueError("Neural network requires validation data for early stopping")
        input_dim = X_train.shape[1]
        model = train_neural_network(
            X_train, y_train, X_val, y_val, input_dim,
            epochs=100, batch_size=32, learning_rate=0.001
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def save_model_and_metrics(model, preprocessor: HousePricePreprocessor, 
                           metrics: dict, model_dir: str = "/app/models"):
    """Save model, preprocessor, and metrics to volume."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model type
    is_keras = False
    if TENSORFLOW_AVAILABLE and keras is not None:
        try:
            is_keras = isinstance(model, keras.Model)
        except (NameError, AttributeError, TypeError):
            is_keras = False
    model_type = "keras" if is_keras else "sklearn"
    
    # Save model based on type
    if model_type == "keras":
        model_path = model_dir / "model.keras"
        model.save(str(model_path))
        logger.info(f"Keras model saved to {model_path}")
    else:
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Sklearn model saved to {model_path}")
    
    # Save model type
    model_type_path = model_dir / "model_type.txt"
    with open(model_type_path, 'w') as f:
        f.write(model_type)
    
    # Save preprocessor
    preprocessor_path = model_dir / "preprocessor.pkl"
    preprocessor.save(str(preprocessor_path))
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Save metrics
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main Dockerized training pipeline."""
    parser = argparse.ArgumentParser(
        description='Dockerized training: Retrain selected model with full dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gradient_boosting',
        choices=['linear_regression', 'random_forest', 'gradient_boosting'] + (['neural_network'] if TENSORFLOW_AVAILABLE else []),
        help='Model to train (default: gradient_boosting). Neural network only available if TensorFlow is installed.'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Dockerized Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Training model: {args.model}")
    logger.info("Using 100% of dataset for final training")
    
    # Step 1: Load data
    logger.info("\n[Step 1] Loading data...")
    train_df = load_train_data()
    logger.info(f"Loaded {len(train_df)} samples")
    
    # Step 2: Preprocess data
    logger.info("\n[Step 2] Preprocessing data...")
    X = train_df.drop('SalePrice', axis=1)
    y = train_df['SalePrice']
    
    X = handle_missing_values(X)
    X = create_features(X)
    
    preprocessor = HousePricePreprocessor()
    X_transformed = preprocessor.fit_transform(X, y)
    
    # Step 3: Split for validation (80/20)
    logger.info("\n[Step 3] Splitting data (80% train, 20% validation)...")
    X_train, X_val, y_train, y_val = train_test_split(
        pd.DataFrame(X_transformed), y, test_size=0.2, random_state=42
    )
    
    X_train = X_train.values
    X_val = X_val.values
    
    # Step 4: Train model
    logger.info("\n[Step 4] Training model...")
    if args.model == 'neural_network':
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "Neural network requires TensorFlow. "
                "Docker builds exclude TensorFlow by default. "
                "Modify docker_training/Dockerfile to use requirements.txt instead of requirements-docker.txt"
            )
        model = train_model(args.model, X_train, y_train.values, X_val, y_val.values)
    else:
        model = train_model(args.model, X_train, y_train.values)
    
    # Step 5: Evaluate on validation set
    logger.info("\n[Step 5] Evaluating model...")
    if args.model == 'neural_network':
        val_pred = model.predict(X_val, verbose=0).flatten()
    else:
        val_pred = model.predict(X_val)
    metrics = evaluate_model(y_val.values, val_pred, f"{args.model} (Validation)")
    
    # Step 6: Retrain on full dataset (100%)
    logger.info("\n[Step 6] Retraining on 100% of data...")
    X_full = pd.DataFrame(X_transformed).values
    if args.model == 'neural_network':
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "Neural network requires TensorFlow. "
                "Docker builds exclude TensorFlow by default. "
                "Modify docker_training/Dockerfile to use requirements.txt instead of requirements-docker.txt"
            )
        # For neural network, use validation split instead of separate validation set
        # Split 80/20 for final training
        X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
            X_full, y.values, test_size=0.2, random_state=42
        )
        model_final = train_model(args.model, X_final_train, y_final_train, X_final_val, y_final_val)
    else:
        model_final = train_model(args.model, X_full, y.values)
    
    # Step 7: Save model and metrics
    logger.info("\n[Step 7] Saving artifacts...")
    save_model_and_metrics(model_final, preprocessor, metrics, model_dir="/app/models")
    
    logger.info("\n" + "=" * 60)
    logger.info("Dockerized Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Validation RMSE: ${metrics['rmse']:,.2f}")
    logger.info(f"Validation R²: {metrics['r2']:.4f}")
    logger.info(f"Model saved to /app/models (persisted via Docker volume)")


if __name__ == "__main__":
    main()
