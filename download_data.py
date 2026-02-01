import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

class DataLoader:
    """
    Helper class to download Kaggle datasets using username + API key.

    Requires environment variables:
        - KAGGLE_USERNAME
        - KAGGLE_KEY

    Saves datasets to `data/raw` relative to project root.
    """

    def __init__(self, raw_dir: str = "data/raw"):
        # Base directory (project root)
        self.base_dir = Path(__file__).resolve().parent.parent
        # Directory to store raw datasets
        self.raw_dir = self.base_dir / raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Check that env vars exist
        if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
            raise ValueError(
                "KAGGLE_USERNAME and KAGGLE_KEY must be set as environment variables.\n"
                "Windows (PowerShell):\n"
                '  setx KAGGLE_USERNAME "your_username"\n'
                '  setx KAGGLE_KEY "your_api_key"\n'
                "Linux/macOS:\n"
                "  export KAGGLE_USERNAME=your_username\n"
                "  export KAGGLE_KEY=your_api_key"
            )

        # Initialize Kaggle API and authenticate via environment variables
        self.api = KaggleApi()
        self.api.authenticate()

    def download_kaggle_dataset(self, competition: str):
        """
        Download a Kaggle competition dataset.

        Args:
            competition (str): Kaggle competition name, e.g.,
                "c/house-prices-advanced-regression-techniques"
        """
        print(f"Downloading dataset from Kaggle competition: {competition}")
        print(f"Saving to: {self.raw_dir}")

        # Download all competition files to raw_dir
        self.api.competition_download_files(
            competition, path=str(self.raw_dir), quiet=False
        )

        # Unzip all downloaded zip files
        self._unzip_all_files()

    def _unzip_all_files(self):
        """
        Unzip all .zip files in raw_dir and remove the zip files.
        """
        for zip_path in self.raw_dir.glob("*.zip"):
            print(f"Unzipping {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
            zip_path.unlink()  # remove the zip after extraction
        print("All files extracted successfully.")
