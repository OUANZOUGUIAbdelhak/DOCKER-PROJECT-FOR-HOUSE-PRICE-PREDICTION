"""
FastAPI Backend for House Price Prediction

Purpose:
    - Serve ML model predictions via REST API
    - Accept house features from frontend
    - Return predicted prices

Key Features:
    - FastAPI for modern async API
    - Pydantic for request validation
    - CORS enabled for React frontend
    - Auto-generated API documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predict import HousePricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="ML Model API for predicting house prices",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loads model once at startup)
predictor = None

@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    global predictor
    try:
        # Model directory is /app/models (mounted volume)
        # Use absolute path to ensure correct resolution
        model_dir_str = "/app/models"
        model_dir = Path(model_dir_str)
        
        # Verify model exists before loading
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path.absolute()}. "
                f"Please ensure model.pkl exists in ./models/ directory on host. "
                f"Train the model first: docker-compose run --rm training"
            )
        
        predictor = HousePricePredictor(model_dir=model_dir_str)
        logger.info(f"Model loaded successfully from {model_dir_str}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Current working directory: {Path.cwd()}")
        logger.error(f"Model dir path: {model_dir if 'model_dir' in locals() else 'N/A'}")
        raise


# Pydantic models for request validation
class HouseFeatures(BaseModel):
    """House features for prediction."""
    
    # Property characteristics
    LotArea: float = Field(..., description="Lot size in square feet", gt=0)
    YearBuilt: int = Field(..., description="Year built", ge=1800, le=2024)
    YearRemodAdd: Optional[int] = Field(None, description="Remodel year", ge=1800, le=2024)
    OverallQual: int = Field(..., description="Overall quality (1-10)", ge=1, le=10)
    OverallCond: int = Field(..., description="Overall condition (1-10)", ge=1, le=10)
    
    # Living space
    GrLivArea: float = Field(..., description="Above grade living area (sq ft)", gt=0)
    TotalBsmtSF: Optional[float] = Field(None, description="Total basement area (sq ft)", ge=0)
    FirstFlrSF: Optional[float] = Field(None, description="First floor area (sq ft)", ge=0)
    SecondFlrSF: Optional[float] = Field(None, description="Second floor area (sq ft)", ge=0)
    
    # Bathrooms
    FullBath: int = Field(..., description="Full bathrooms above grade", ge=0)
    HalfBath: int = Field(0, description="Half bathrooms above grade", ge=0)
    BsmtFullBath: Optional[int] = Field(None, description="Basement full bathrooms", ge=0)
    BsmtHalfBath: Optional[int] = Field(None, description="Basement half bathrooms", ge=0)
    
    # Bedrooms
    BedroomAbvGr: int = Field(..., description="Bedrooms above grade", ge=0)
    TotRmsAbvGrd: int = Field(..., description="Total rooms above grade", ge=0)
    
    # Garage
    GarageCars: Optional[int] = Field(None, description="Garage car capacity", ge=0)
    GarageArea: Optional[float] = Field(None, description="Garage area (sq ft)", ge=0)
    
    # Location & type
    Neighborhood: str = Field(..., description="Neighborhood")
    HouseStyle: str = Field(..., description="House style")
    BldgType: str = Field(..., description="Building type")
    
    # Quality ratings
    ExterQual: str = Field(..., description="Exterior quality (Ex, Gd, TA, Fa, Po)")
    KitchenQual: str = Field(..., description="Kitchen quality (Ex, Gd, TA, Fa, Po)")
    
    # Additional features
    Fireplaces: Optional[int] = Field(0, description="Number of fireplaces", ge=0)
    PoolArea: Optional[float] = Field(0, description="Pool area (sq ft)", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "LotArea": 8450,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "OverallQual": 7,
                "OverallCond": 5,
                "GrLivArea": 1710,
                "TotalBsmtSF": 856,
                "FirstFlrSF": 856,
                "SecondFlrSF": 854,
                "FullBath": 2,
                "HalfBath": 1,
                "BsmtFullBath": 1,
                "BsmtHalfBath": 0,
                "BedroomAbvGr": 3,
                "TotRmsAbvGrd": 8,
                "GarageCars": 2,
                "GarageArea": 548,
                "Neighborhood": "Veenker",
                "HouseStyle": "2Story",
                "BldgType": "1Fam",
                "ExterQual": "Gd",
                "KitchenQual": "Gd",
                "Fireplaces": 0,
                "PoolArea": 0
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    predicted_price: float = Field(..., description="Predicted house price in USD")
    predicted_price_formatted: str = Field(..., description="Formatted price string")
    confidence: Optional[str] = Field(None, description="Confidence level (if available)")


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
    """
    Predict house price based on features.
    
    Args:
        features: House features (validated by Pydantic)
    
    Returns:
        Predicted price and formatted string
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to dict
        features_dict = features.model_dump()
        
        # Generate prediction
        predicted_price = predictor.predict_single(features_dict)
        
        # Format price
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
    """
    Get available options for categorical features.
    Helps frontend populate dropdowns.
    
    These are the actual values from the House Prices dataset.
    """
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
        "BldgType": [
            "1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"
        ],
        "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"],
        "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"]
    }
