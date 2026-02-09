"""
Local Model Comparison Script - Fully Autonomous

Purpose:
    - Train multiple models locally (outside Docker)
    - Compare performance on validation set
    - Select best model
    - Save comparison results

This runs BEFORE Dockerized training to select the model.
All preprocessing logic is embedded here (no shared modules).
"""

import sys
from pathlib import Path
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
import json
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neural network imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow available - Neural network will be included in comparison")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Neural network training will be disabled.")
    logger.warning("Install TensorFlow with: pip install tensorflow")

# Neural network imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Neural network training will be disabled.")


# ============================================================================
# DATA LOADING (Embedded)
# ============================================================================

def load_train_data(data_dir: str = "data") -> pd.DataFrame:
    """Load training data from CSV file."""
    data_path = Path(data_dir) / "raw" / "train.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Please download the dataset first"
        )
    
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if df.empty:
        raise ValueError("Training data is empty")
    
    if 'SalePrice' not in df.columns:
        raise ValueError("Target column 'SalePrice' not found")
    
    return df


# ============================================================================
# PREPROCESSING (Embedded - Same as training)
# ============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using domain knowledge."""
    df = df.copy()
    
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
    
    numerical_none_features = [
        'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    
    for feature in numerical_none_features:
        if feature in df.columns:
            df[feature].fillna(0, inplace=True)
    
    other_features = ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 
                     'Exterior2nd', 'KitchenQual', 'SaleType']
    for feature in other_features:
        if feature in df.columns and df[feature].isna().any():
            mode_value = df[feature].mode()[0] if not df[feature].mode().empty else 'None'
            df[feature].fillna(mode_value, inplace=True)
    
    if 'LotFrontage' in df.columns:
        df['LotFrontage'].fillna(df.groupby('Neighborhood')['LotFrontage'].transform('median'), inplace=True)
        df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
    
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
    
    return df


class HousePricePreprocessor:
    """Preprocessing pipeline for House Prices dataset."""
    
    def __init__(self):
        self.preprocessor = None
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
        
        return numerical_features, categorical_features, ordinal_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'HousePricePreprocessor':
        """Fit preprocessing pipeline on training data."""
        logger.info("Fitting preprocessing pipeline...")
        
        numerical_features, categorical_features, ordinal_features = self._identify_feature_types(X)
        
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
        
        self.preprocessor.fit_transform(X)
        self.is_fitted = True
        
        logger.info("Preprocessing pipeline fitted")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform.")
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


# ============================================================================
# MODEL TRAINING AND EVALUATION (Embedded)
# ============================================================================

def evaluate_model(y_true, y_pred, model_name=""):
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


def train_and_evaluate_model(model_name, X_train, y_train, X_val, y_val):
    """Train a model and evaluate on validation set."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name.upper()}")
    logger.info(f"{'='*60}")
    
    if model_name == 'linear_regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    elif model_name == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    elif model_name == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    elif model_name == 'neural_network':
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping neural network.")
            return None
        input_dim = X_train.shape[1]
        model = train_neural_network(
            X_train, y_train, X_val, y_val, input_dim,
            epochs=100, batch_size=32, learning_rate=0.001
        )
        y_pred = model.predict(X_val, verbose=0).flatten()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    metrics = evaluate_model(y_val, y_pred, model_name)
    
    return metrics


# ============================================================================
# MAIN COMPARISON PIPELINE
# ============================================================================

def main():
    """Main comparison pipeline."""
    logger.info("=" * 60)
    logger.info("LOCAL MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info("Purpose: Compare models locally before Dockerized training")
    
    # Step 1: Load data
    logger.info("\n[Step 1] Loading data...")
    train_df = load_train_data()
    
    # Step 2: Preprocess
    logger.info("\n[Step 2] Preprocessing data...")
    X = train_df.drop('SalePrice', axis=1)
    y = train_df['SalePrice']
    
    X = handle_missing_values(X)
    X = create_features(X)
    
    preprocessor = HousePricePreprocessor()
    X_transformed = preprocessor.fit_transform(X, y)
    
    # Step 3: Split data (60/20/20)
    logger.info("\n[Step 3] Splitting data (60% train, 20% val, 20% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        pd.DataFrame(X_transformed), y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    X_train = X_train.values
    X_val = X_val.values
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Validation: {len(X_val)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    
    # Step 4: Train and compare models
    logger.info("\n[Step 4] Training and comparing models...")
    
    models_to_compare = ['linear_regression', 'random_forest', 'gradient_boosting']
    
    # Add neural network if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        models_to_compare.append('neural_network')
        logger.info("TensorFlow available - Neural network will be included in comparison")
    else:
        logger.warning("TensorFlow not available - Neural network will be skipped")
        logger.warning("Install TensorFlow with: pip install tensorflow")
    
    results = {}
    
    for model_name in models_to_compare:
        try:
            metrics = train_and_evaluate_model(
                model_name, X_train, y_train.values, X_val, y_val.values
            )
            if metrics is not None:
                results[model_name] = metrics
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            logger.error(f"Skipping {model_name} and continuing with other models...")
    
    # Step 5: Select best model
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  RMSE: ${metrics['rmse']:,.2f}")
        logger.info(f"  MAE: ${metrics['mae']:,.2f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
    
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_model[0].upper()}")
    logger.info(f"  RMSE: ${best_model[1]['rmse']:,.2f}")
    logger.info(f"  MAE: ${best_model[1]['mae']:,.2f}")
    logger.info(f"  R²: {best_model[1]['r2']:.4f}")
    logger.info(f"{'='*60}")
    
    # Step 6: Save results
    output_dir = Path(__file__).parent
    results_file = output_dir / "results.json"
    
    comparison_results = {
        'best_model': best_model[0],
        'best_metrics': best_model[1],
        'all_results': results,
        'note': 'Use the best_model for Dockerized training with --model flag'
    }
    
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"\nNext step: Run Dockerized training with:")
    logger.info(f"  docker-compose run --rm training --model {best_model[0]}")


if __name__ == "__main__":
    main()
