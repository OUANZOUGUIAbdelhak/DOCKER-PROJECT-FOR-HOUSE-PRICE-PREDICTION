"""
Data Loader Module

Purpose:
    - Download dataset from Kaggle
    - Load and validate data
    - Provide clean interface for data access

Key Concepts:
    - Data validation: Ensure data matches expected schema
    - Error handling: Graceful failures with clear messages
    - Separation of concerns: Data loading separate from processing
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data loading and validation.
    
    Why a class instead of functions?
    - Encapsulates state (data paths, validation rules)
    - Easier to extend (add new data sources)
    - Better for testing (mock the class)
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Root directory for data (contains raw/ and processed/)
        
        Why use Path instead of string?
        - Cross-platform compatibility (handles / vs \)
        - Better path manipulation methods
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_name: str = "c/house-prices-advanced-regression-techniques") -> None:
        """
        Download dataset from Kaggle using Kaggle API.
        
        Args:
            dataset_name: Kaggle dataset identifier (format: owner/dataset or competition/c/competition-name)
        
        How Kaggle API works:
        1. Requires kaggle.json credentials file
        2. Location: ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)
        3. Contains: {"username": "...", "key": "..."}
        
        What can go wrong?
        - Missing credentials → AuthenticationError
        - Invalid dataset name → NotFoundError
        - Network issues → ConnectionError
        
        Solution: Check credentials exist before calling
        """
        try:
            import kaggle
            
            logger.info(f"Downloading dataset: {dataset_name}")
            logger.info(f"Target directory: {self.raw_dir}")
            
            # Kaggle API downloads to current directory
            # We need to change to raw_dir, download, then change back
            original_dir = os.getcwd()
            os.chdir(self.raw_dir)
            
            try:
                kaggle.api.dataset_download_files(
                    dataset_name,
                    path=str(self.raw_dir),
                    unzip=True
                )
                logger.info("Dataset downloaded successfully")
            finally:
                os.chdir(original_dir)
                
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.error("Make sure you have:")
            logger.error("1. Kaggle API credentials at ~/.kaggle/kaggle.json")
            logger.error("2. Valid dataset name")
            logger.error("3. Internet connection")
            raise
    
    def load_train_data(self, filename: str = "train.csv") -> pd.DataFrame:
        """
        Load training data from raw directory.
        
        Args:
            filename: Name of training CSV file
        
        Returns:
            DataFrame with training data
        
        Why separate load functions?
        - Train and test may have different schemas
        - Different validation rules
        - Clearer intent in code
        """
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Training data not found at {filepath}. "
                f"Run download_kaggle_dataset() first."
            )
        
        logger.info(f"Loading training data from {filepath}")
        df = pd.read_csv(filepath)
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_test_data(self, filename: str = "test.csv") -> pd.DataFrame:
        """Load test data from raw directory."""
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Test data not found at {filepath}. "
                f"Run download_kaggle_dataset() first."
            )
        
        logger.info(f"Loading test data from {filepath}")
        df = pd.read_csv(filepath)
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def validate_schema(self, df: pd.DataFrame, expected_target: str = "SalePrice") -> bool:
        """
        Validate that DataFrame has expected structure.
        
        Args:
            df: DataFrame to validate
            expected_target: Name of target column (for training data)
        
        Returns:
            True if valid, raises error otherwise
        
        Why validate?
        - Catch data issues early
        - Fail fast with clear error messages
        - Prevent downstream errors
        
        What we check:
        1. DataFrame is not empty
        2. Target column exists (for training data)
        3. No completely empty columns
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if expected_target in df.columns:
            if df[expected_target].isna().all():
                raise ValueError(f"Target column '{expected_target}' is completely empty")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            logger.warning(f"Found completely empty columns: {empty_cols}")
        
        logger.info("Schema validation passed")
        return True
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, subdir: str = "processed") -> None:
        """
        Save processed DataFrame to disk.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            subdir: Subdirectory (processed or raw)
        
        Why save processed data?
        - Avoid reprocessing expensive operations
        - Reproducibility: Same processed data = same results
        - Debugging: Inspect intermediate results
        """
        if subdir == "processed":
            output_dir = self.processed_dir
        else:
            output_dir = self.raw_dir
        
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")


def main():
    """
    Example usage of DataLoader.
    
    This demonstrates how to use the class.
    In production, this would be called from training script.
    """
    loader = DataLoader()
    
    # Download dataset (one-time operation)
    # loader.download_kaggle_dataset()
    
    # Load data
    train_df = loader.load_train_data()
    loader.validate_schema(train_df)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()[:10]}...")  # First 10 columns


if __name__ == "__main__":
    main()
