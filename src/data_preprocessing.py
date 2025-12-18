# src/data_preprocessing.py

import os
import pandas as pd
from pathlib import Path

def create_data_dirs():
    """Create raw and processed data directories if they don't exist."""
    # Get the project root directory (parent of src folder)
    project_root = Path(__file__).parent.parent
    
    # Create data directories relative to project root
    try:
        os.makedirs(project_root / "data" / "raw", exist_ok=True)
        os.makedirs(project_root / "data" / "processed", exist_ok=True)
    except PermissionError as e:
        print(f"Warning: Unable to create data directories - {e}")

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, handle missing values."""
    df.drop_duplicates(inplace=True)
    df.dropna(how="any", inplace=True)
    return df
     
