"""
FastAPI Backend for House Price Prediction - Fully Autonomous

Purpose:
    - Serve ML model predictions via REST API
    - All preprocessing and prediction logic contained here
    - No external dependencies on shared modules
    - Loads model from Docker volume
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib
import sys
import pickle
from typing import Tuple
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize logger first (before any logger calls)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neural network imports (optional - only needed for Keras models)
# For sklearn/XGBoost models, TensorFlow is NOT required
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow available - Keras model support enabled")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.info("TensorFlow not available - Only sklearn models supported (this is fine for XGBoost/sklearn models)")

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="ML Model API for predicting house prices",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (loaded at startup)
predictor = None


# ============================================================================
# PREPROCESSING (Embedded - Must match training preprocessing exactly)
# ============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using domain knowledge (same as training)."""
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
    """Create new features through feature engineering (same as training)."""
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


def add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing features with default values (for API input)."""
    defaults = {
        'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 70.0, 'LotShape': 'Reg',
        'LandContour': 'Lvl', 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Street': 'Pave',
        'Alley': 'None', 'Utilities': 'AllPub', 'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd', 'MasVnrType': 'None', 'MasVnrArea': 0.0,
        'ExterCond': 'TA', 'Foundation': 'PConc', 'BsmtQual': 'TA', 'BsmtCond': 'TA',
        'BsmtExposure': 'No', 'BsmtFinType1': 'Unf', 'BsmtFinType2': 'Unf',
        'BsmtFinSF1': 0.0, 'BsmtFinSF2': 0.0, 'BsmtUnfSF': 0.0, 'Heating': 'GasA',
        'HeatingQC': 'TA', 'Electrical': 'SBrkr', 'LowQualFinSF': 0.0,
        'KitchenAbvGr': 1, 'Functional': 'Typ', 'FireplaceQu': 'None',
        'GarageType': 'Attchd', 'GarageFinish': 'Unf', 'GarageQual': 'TA',
        'GarageCond': 'TA', 'GarageYrBlt': None, 'WoodDeckSF': 0.0,
        'OpenPorchSF': 0.0, 'EnclosedPorch': 0.0, '3SsnPorch': 0.0,
        'ScreenPorch': 0.0, 'PoolQC': 'None', 'Fence': 'None',
        'MiscFeature': 'None', 'MiscVal': 0, 'SaleType': 'WD',
        'SaleCondition': 'Normal', 'MoSold': 6, 'YrSold': 2010,
        'Condition1': 'Norm', 'Condition2': 'Norm', 'RoofStyle': 'Gable',
        'RoofMatl': 'CompShg', 'CentralAir': 'Y', 'PavedDrive': 'Y'
    }
    
    for feature, default_value in defaults.items():
        if feature not in df.columns:
            if default_value is None:
                if feature == 'GarageYrBlt':
                    df[feature] = df.get('YearBuilt', 2000)
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
    
    # Infer missing floor areas
    if '1stFlrSF' not in df.columns:
        if 'GrLivArea' in df.columns:
            df['1stFlrSF'] = df['GrLivArea'] if 'TotalBsmtSF' not in df.columns else df['GrLivArea'] * 0.6
        else:
            df['1stFlrSF'] = 1000
    
    if '2ndFlrSF' not in df.columns:
        if 'GrLivArea' in df.columns and '1stFlrSF' in df.columns:
            df['2ndFlrSF'] = (df['GrLivArea'] - df['1stFlrSF']).clip(lower=0)
        else:
            df['2ndFlrSF'] = 0
    
    return df


# ============================================================================
# PREPROCESSOR CLASS (Embedded - Must match training script exactly)
# ============================================================================

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


# ============================================================================
# PREDICTION CLASS (Embedded)
# ============================================================================

class HousePricePredictor:
    """Production inference class."""
    
    def __init__(self, model_dir: str = "/app/models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load model and preprocessor from disk."""
        model_path = self.model_dir / "model.pkl"
        keras_model_path = self.model_dir / "model.keras"
        model_type_path = self.model_dir / "model_type.txt"
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        
        # Determine model type
        self.model_type = "sklearn"  # default
        if model_type_path.exists():
            with open(model_type_path, 'r') as f:
                self.model_type = f.read().strip()
        
        # Load model based on type
        if self.model_type == "keras":
            if not TENSORFLOW_AVAILABLE:
                raise ImportError(
                    "Keras model detected but TensorFlow is not available. "
                    "Docker builds exclude TensorFlow by default (for smaller images). "
                    "Options:\n"
                    "1. Train a sklearn model instead (recommended): docker-compose run --rm training --model gradient_boosting\n"
                    "2. Or enable TensorFlow: modify docker_api/Dockerfile to use docker_api/requirements.txt instead of requirements-docker.txt"
                )
            if not keras_model_path.exists():
                raise FileNotFoundError(
                    f"Keras model not found at {keras_model_path}. "
                    f"Train the model first: docker-compose run --rm training --model neural_network"
                )
            logger.info(f"Loading Keras model from {keras_model_path}")
            self.model = keras.models.load_model(str(keras_model_path))
        else:
            # sklearn/XGBoost models - TensorFlow NOT needed
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    f"Train the model first: docker-compose run --rm training"
                )
            logger.info(f"Loading sklearn model from {model_path}")
            self.model = joblib.load(model_path)
        
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. "
                f"Train the model first: docker-compose run --rm training"
            )
        
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        # Register HousePricePreprocessor in __main__ module for pickle compatibility
        # The preprocessor was pickled as __main__.HousePricePreprocessor in training script
        # We need to register it in sys.modules['__main__'] so pickle can find it
        if '__main__' not in sys.modules:
            sys.modules['__main__'] = types.ModuleType('__main__')
        sys.modules['__main__'].HousePricePreprocessor = HousePricePreprocessor
        
        # Now joblib.load can find the class when unpickling
        self.preprocessor = joblib.load(preprocessor_path)
        
        logger.info(f"Model and preprocessor loaded successfully (type: {self.model_type})")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data."""
        logger.info(f"Generating predictions for {len(X)} samples")
        
        # Add missing features
        X_filled = add_missing_features(X.copy())
        
        # Preprocess (same as training)
        X_processed = handle_missing_values(X_filled)
        X_processed = create_features(X_processed)
        
        # Ensure all expected columns exist
        X_processed = self._ensure_all_expected_columns(X_processed)
        
        # Transform using fitted preprocessor
        X_transformed = self.preprocessor.transform(X_processed)
        
        # Predict (different methods for sklearn vs Keras)
        if self.model_type == "keras":
            predictions = self.model.predict(X_transformed, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_transformed)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def _ensure_all_expected_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ALL columns expected by the preprocessor are present."""
        expected_columns = set()
        if hasattr(self.preprocessor, 'preprocessor') and hasattr(self.preprocessor.preprocessor, 'transformers_'):
            for name, transformer, columns in self.preprocessor.preprocessor.transformers_:
                if name != 'remainder' and columns is not None:
                    expected_columns.update(columns)
        
        if not expected_columns:
            logger.warning("Could not determine expected columns from preprocessor")
            return df
        
        missing_cols = expected_columns - set(df.columns)
        if missing_cols:
            logger.warning(f"Adding {len(missing_cols)} missing expected columns")
            for col in missing_cols:
                if col == 'TotalSF':
                    df[col] = df.get('GrLivArea', 0) + df.get('TotalBsmtSF', 0)
                elif col == 'Age':
                    df[col] = df.get('YrSold', 2010) - df.get('YearBuilt', 2000)
                elif col == 'TotalBathrooms':
                    df[col] = (df.get('FullBath', 0) + 0.5 * df.get('HalfBath', 0) + 
                              df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0))
                elif col == 'Qual_x_Area':
                    df[col] = df.get('OverallQual', 5) * df.get('GrLivArea', 1000)
                else:
                    df[col] = 0
        
        return df
    
    def predict_single(self, house_data: dict) -> float:
        """Predict price for a single house."""
        df = pd.DataFrame([house_data])
        predictions = self.predict(df)
        return float(predictions[0])


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    global predictor
    import time
    
    model_dir_str = "/app/models"
    model_dir = Path(model_dir_str)
    model_path = model_dir / "model.pkl"
    
    # Wait for model file to exist (in case training is still running)
    max_wait_time = 300  # 5 minutes
    wait_interval = 2  # Check every 2 seconds
    elapsed = 0
    
    logger.info(f"Waiting for model file at {model_path}...")
    while not model_path.exists() and elapsed < max_wait_time:
        logger.info(f"Model not found yet. Waiting... ({elapsed}s/{max_wait_time}s)")
        time.sleep(wait_interval)
        elapsed += wait_interval
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path} after waiting {max_wait_time}s. "
            f"Please train the model first: docker-compose run --rm training"
        )
    
    try:
        predictor = HousePricePredictor(model_dir=model_dir_str)
        logger.info(f"Model loaded successfully from {model_dir_str}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


class HouseFeatures(BaseModel):
    """House features for prediction."""
    LotArea: float = Field(..., description="Lot size in square feet", gt=0)
    YearBuilt: int = Field(..., description="Year built", ge=1800, le=2024)
    YearRemodAdd: Optional[int] = Field(None, description="Remodel year", ge=1800, le=2024)
    OverallQual: int = Field(..., description="Overall quality (1-10)", ge=1, le=10)
    OverallCond: int = Field(..., description="Overall condition (1-10)", ge=1, le=10)
    GrLivArea: float = Field(..., description="Above grade living area (sq ft)", gt=0)
    TotalBsmtSF: Optional[float] = Field(None, description="Total basement area (sq ft)", ge=0)
    FirstFlrSF: Optional[float] = Field(None, description="First floor area (sq ft)", ge=0)
    SecondFlrSF: Optional[float] = Field(None, description="Second floor area (sq ft)", ge=0)
    FullBath: int = Field(..., description="Full bathrooms above grade", ge=0)
    HalfBath: int = Field(0, description="Half bathrooms above grade", ge=0)
    BsmtFullBath: Optional[int] = Field(None, description="Basement full bathrooms", ge=0)
    BsmtHalfBath: Optional[int] = Field(None, description="Basement half bathrooms", ge=0)
    BedroomAbvGr: int = Field(..., description="Bedrooms above grade", ge=0)
    TotRmsAbvGrd: int = Field(..., description="Total rooms above grade", ge=0)
    GarageCars: Optional[int] = Field(None, description="Garage car capacity", ge=0)
    GarageArea: Optional[float] = Field(None, description="Garage area (sq ft)", ge=0)
    Neighborhood: str = Field(..., description="Neighborhood")
    HouseStyle: str = Field(..., description="House style")
    BldgType: str = Field(..., description="Building type")
    ExterQual: str = Field(..., description="Exterior quality (Ex, Gd, TA, Fa, Po)")
    KitchenQual: str = Field(..., description="Kitchen quality (Ex, Gd, TA, Fa, Po)")
    Fireplaces: Optional[int] = Field(0, description="Number of fireplaces", ge=0)
    PoolArea: Optional[float] = Field(0, description="Pool area (sq ft)", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "LotArea": 8450, "YearBuilt": 2003, "YearRemodAdd": 2003,
                "OverallQual": 7, "OverallCond": 5, "GrLivArea": 1710,
                "TotalBsmtSF": 856, "FirstFlrSF": 856, "SecondFlrSF": 854,
                "FullBath": 2, "HalfBath": 1, "BsmtFullBath": 1, "BsmtHalfBath": 0,
                "BedroomAbvGr": 3, "TotRmsAbvGrd": 8, "GarageCars": 2, "GarageArea": 548,
                "Neighborhood": "Veenker", "HouseStyle": "2Story", "BldgType": "1Fam",
                "ExterQual": "Gd", "KitchenQual": "Gd", "Fireplaces": 0, "PoolArea": 0
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    predicted_price: float = Field(..., description="Predicted house price in USD")
    predicted_price_formatted: str = Field(..., description="Formatted price string")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """Predict house price based on features."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_dict = features.model_dump()
        predicted_price = predictor.predict_single(features_dict)
        predicted_price_formatted = f"${predicted_price:,.2f}"
        
        logger.info(f"Prediction: ${predicted_price:,.2f}")
        
        return PredictionResponse(
            predicted_price=predicted_price,
            predicted_price_formatted=predicted_price_formatted
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/features/options")
async def get_feature_options():
    """Get available options for categorical features."""
    return {
        "Neighborhood": [
            "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
            "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
            "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt",
            "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr",
            "Timber", "Veenker"
        ],
        "HouseStyle": [
            "1.5Fin", "1.5Unf", "1Story", "2.5Fin", "2.5Unf", "2Story",
            "SFoyer", "SLvl", "Split", "SplitFoyer"
        ],
        "BldgType": ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
        "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"],
        "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"]
    }
