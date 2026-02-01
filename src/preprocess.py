"""
Preprocessing Module

Purpose:
    - Feature engineering
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    - Create preprocessing pipeline for consistent train/test transformation

Key ML Concepts:
    - Data Leakage: Never use test data statistics for training
    - Fit-Transform Pattern: Fit on train, transform both train and test
    - Pipeline: Chain transformations for reproducibility
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousePricePreprocessor:
    """
    Preprocessing pipeline for House Prices dataset.
    
    Why a class?
    - Encapsulates all preprocessing logic
    - Can be serialized (saved) and reused for inference
    - Maintains state (fitted transformers)
    
    Critical ML Principle:
    - Fit ONLY on training data
    - Transform both train and test using fitted transformers
    - This prevents data leakage (using future information)
    """
    
    def __init__(self):
        """
        Initialize preprocessor.
        
        We don't fit transformers here - that happens in fit().
        This allows us to create the preprocessor without data.
        """
        self.preprocessor = None
        self.feature_names = None
        self.is_fitted = False
    
    def _identify_feature_types(self, df: pd.DataFrame) -> Tuple[list, list, list]:
        """
        Identify numerical, categorical, and ordinal features.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Tuple of (numerical_cols, categorical_cols, ordinal_cols)
        
        Why separate feature types?
        - Different preprocessing for different types
        - Numerical: scaling, imputation
        - Categorical: one-hot encoding
        - Ordinal: ordinal encoding (preserves order)
        
        House Prices Dataset Features:
        - Numerical: LotArea, YearBuilt, GrLivArea, etc.
        - Ordinal: OverallQual (1-10), OverallCond (1-10), etc.
        - Categorical: Neighborhood, MSZoning, etc.
        """
        # Exclude target and ID columns
        exclude_cols = ['Id', 'SalePrice']
        cols = [c for c in df.columns if c not in exclude_cols]
        
        # Ordinal features (ordered categories)
        # These have inherent order, so we use ordinal encoding, not one-hot
        ordinal_features = [
            'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
            'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
            'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
            'LotShape', 'LandSlope', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'GarageFinish', 'PavedDrive',
            'Fence', 'Street', 'Alley', 'CentralAir'
        ]
        
        # Filter to only features that exist in dataset
        ordinal_features = [f for f in ordinal_features if f in cols]
        
        # Categorical features (no inherent order)
        # These get one-hot encoded
        categorical_features = [
            'MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical',
            'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition'
        ]
        categorical_features = [f for f in categorical_features if f in cols]
        
        # Numerical features (everything else)
        numerical_features = [c for c in cols if c not in ordinal_features + categorical_features]
        
        logger.info(f"Feature types identified:")
        logger.info(f"  Numerical: {len(numerical_features)}")
        logger.info(f"  Ordinal: {len(ordinal_features)}")
        logger.info(f"  Categorical: {len(categorical_features)}")
        
        return numerical_features, categorical_features, ordinal_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'HousePricePreprocessor':
        """
        Fit preprocessing pipeline on training data.
        
        Args:
            X: Training features (DataFrame)
            y: Training target (Series) - optional, not used for preprocessing
        
        Returns:
            Self (for method chaining)
        
        What happens here?
        1. Identify feature types
        2. Create transformers for each type
        3. Fit transformers on training data
        4. Store fitted transformers
        
        CRITICAL: This must ONLY be called on training data!
        """
        logger.info("Fitting preprocessing pipeline on training data...")
        
        numerical_features, categorical_features, ordinal_features = self._identify_feature_types(X)
        
        # Create preprocessing pipelines for each feature type
        
        # Numerical: Impute missing values with median, then standardize
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Median is robust to outliers
            ('scaler', StandardScaler())  # Mean=0, Std=1
        ])
        
        # Ordinal: Impute with 'missing' category, then ordinal encode
        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Categorical: Impute with 'missing', then one-hot encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding
        ])
        
        # Combine all transformers
        # ColumnTransformer applies different transformers to different columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('ord', ordinal_pipeline, ordinal_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )
        
        # Fit the preprocessor
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Store feature names (for later inspection)
        self.feature_names = self._get_feature_names(X, numerical_features, ordinal_features, categorical_features)
        self.is_fitted = True
        
        logger.info(f"Preprocessing pipeline fitted. Output shape: {X_transformed.shape}")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Data to transform (can be train or test)
        
        Returns:
            Transformed numpy array
        
        Why separate fit() and transform()?
        - Fit learns parameters from training data
        - Transform applies those parameters to any data
        - This prevents data leakage (test data doesn't influence training)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        logger.info(f"Transforming data with shape {X.shape}")
        X_transformed = self.preprocessor.transform(X)
        logger.info(f"Transformed shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform in one step (convenience method)."""
        return self.fit(X, y).transform(X)
    
    def _get_feature_names(self, X: pd.DataFrame, numerical_features, ordinal_features, categorical_features) -> list:
        """
        Get feature names after transformation.
        
        This is tricky because ColumnTransformer changes feature names.
        We need this for model interpretability.
        """
        feature_names = []
        
        # Numerical features (same names)
        feature_names.extend(numerical_features)
        
        # Ordinal features (same names)
        feature_names.extend(ordinal_features)
        
        # Categorical features (expanded by one-hot encoding)
        # Note: Feature names for one-hot encoded features are handled by ColumnTransformer
        # We'll get them from the transformer after fitting
        for cat_feat in categorical_features:
            if cat_feat in X.columns:
                # Approximate feature names (exact names depend on OneHotEncoder)
                unique_vals = X[cat_feat].fillna('missing').unique()
                for val in sorted(unique_vals):
                    feature_names.append(f"{cat_feat}_{val}")
        
        return feature_names
    
    def save(self, filepath: str) -> None:
        """
        Save fitted preprocessor to disk.
        
        Args:
            filepath: Path to save preprocessor
        
        Why save the preprocessor?
        - Must use SAME preprocessing for inference as training
        - Prevents inconsistencies
        - Enables model deployment
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor. Call fit() first.")
        
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HousePricePreprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            filepath: Path to saved preprocessor
        
        Returns:
            Loaded HousePricePreprocessor instance
        
        Why classmethod?
        - Can be called without instance: HousePricePreprocessor.load(...)
        - Returns new instance with loaded state
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def handle_missing_values(df: pd.DataFrame, strategy: str = "domain_knowledge") -> pd.DataFrame:
    """
    Handle missing values using domain knowledge.
    
    Args:
        df: DataFrame with missing values
        strategy: Strategy to use
    
    Returns:
        DataFrame with handled missing values
    
    Domain Knowledge Approach:
    - For House Prices dataset, missing often means "feature doesn't exist"
    - Example: Missing PoolQC = no pool
    - Example: Missing GarageType = no garage
    
    Why domain knowledge > simple imputation?
    - Preserves information (missing is meaningful)
    - More accurate than mean/median imputation
    - Better model performance
    """
    df = df.copy()
    
    # Features where missing = feature doesn't exist
    # Fill with appropriate "none" value
    none_features = {
        'PoolQC': 'None',
        'MiscFeature': 'None',
        'Alley': 'None',
        'Fence': 'None',
        'FireplaceQu': 'None',
        'GarageType': 'None',
        'GarageFinish': 'None',
        'GarageQual': 'None',
        'GarageCond': 'None',
        'BsmtQual': 'None',
        'BsmtCond': 'None',
        'BsmtExposure': 'None',
        'BsmtFinType1': 'None',
        'BsmtFinType2': 'None',
        'MasVnrType': 'None'
    }
    
    for feature, fill_value in none_features.items():
        if feature in df.columns:
            df[feature].fillna(fill_value, inplace=True)
    
    # Numerical features: missing = 0 (doesn't exist)
    numerical_none_features = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    
    for feature in numerical_none_features:
        if feature in df.columns:
            df[feature].fillna(0, inplace=True)
    
    # Other features: use mode (most common value)
    other_features = ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']
    for feature in other_features:
        if feature in df.columns and df[feature].isna().any():
            mode_value = df[feature].mode()[0] if not df[feature].mode().empty else 'None'
            df[feature].fillna(mode_value, inplace=True)
    
    # LotFrontage: Use median of neighborhood (more sophisticated)
    if 'LotFrontage' in df.columns:
        df['LotFrontage'].fillna(df.groupby('Neighborhood')['LotFrontage'].transform('median'), inplace=True)
        df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)  # Fallback
    
    logger.info("Missing values handled using domain knowledge")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features through feature engineering.
    
    Args:
        df: Original DataFrame
    
    Returns:
        DataFrame with new features
    
    Why feature engineering?
    - Domain knowledge can create better predictors
    - Example: TotalSF = GrLivArea + TotalBsmtSF (total living space)
    - Example: Age = YearSold - YearBuilt (house age)
    
    Feature Engineering Ideas:
    1. Combine related features (TotalSF)
    2. Create ratios (Bathrooms per bedroom)
    3. Extract temporal features (Age)
    4. Create interaction terms (Quality × Area)
    """
    df = df.copy()
    
    # Total square footage
    if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
        df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    
    # House age
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['Age'] = df['YrSold'] - df['YearBuilt']
        df['Age'].fillna(0, inplace=True)  # Handle missing
    
    # Total bathrooms
    if all(col in df.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    
    # Quality × Area interactions (high quality + large area = expensive)
    if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
        df['Qual_x_Area'] = df['OverallQual'] * df['GrLivArea']
    
    logger.info("Feature engineering completed")
    return df


def main():
    """Example usage of preprocessor."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    train_df = loader.load_train_data()
    
    # Separate features and target
    X = train_df.drop('SalePrice', axis=1)
    y = train_df['SalePrice']
    
    # Handle missing values
    X = handle_missing_values(X)
    
    # Feature engineering
    X = create_features(X)
    
    # Create and fit preprocessor
    preprocessor = HousePricePreprocessor()
    X_transformed = preprocessor.fit_transform(X, y)
    
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    
    # Save preprocessor
    preprocessor.save("models/preprocessor.pkl")


if __name__ == "__main__":
    main()
