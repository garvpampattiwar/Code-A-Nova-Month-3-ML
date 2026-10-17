"""
Data loading utilities for CineMatch Movie Recommendation System.
Handles loading raw and processed MovieLens datasets safely with schema verification.
"""

import os
import pandas as pd
from typing import Dict, Tuple, Optional

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


def get_data_paths(base_dir: Optional[str] = None) -> Dict[str, str]:
    """Return dictionary of standard data paths."""
    root = base_dir or BASE_DIR
    return {
        "raw_movies": os.path.join(root, "data", "raw", "movies.csv"),
        "raw_ratings": os.path.join(root, "data", "raw", "ratings.csv"),
        "raw_tags": os.path.join(root, "data", "raw", "tags.csv"),
        "raw_links": os.path.join(root, "data", "raw", "links.csv"),
        "processed_movies": os.path.join(root, "data", "processed", "movies_processed.csv"),
        "models_dir": os.path.join(root, "models"),
    }


def verify_raw_data(raw_dir: str = RAW_DATA_DIR) -> Tuple[bool, str]:
    """Check if all required raw MovieLens dataset files exist."""
    required = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
    missing = [f for f in required if not os.path.exists(os.path.join(raw_dir, f))]
    if missing:
        return False, f"Missing raw dataset files in {raw_dir}: {', '.join(missing)}"
    return True, "All raw dataset files present."


def load_raw_movies(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw movies.csv and validate essential columns."""
    path = file_path or get_data_paths()["raw_movies"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"movies.csv not found at: {path}")
    
    df = pd.read_csv(path)
    expected_cols = {"movieId", "title", "genres"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"movies.csv missing expected columns. Found: {list(df.columns)}")
    return df


def load_raw_ratings(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw ratings.csv and validate columns."""
    path = file_path or get_data_paths()["raw_ratings"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"ratings.csv not found at: {path}")
    
    df = pd.read_csv(path)
    expected_cols = {"userId", "movieId", "rating"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"ratings.csv missing expected columns. Found: {list(df.columns)}")
    return df


def load_raw_tags(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw tags.csv."""
    path = file_path or get_data_paths()["raw_tags"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    
    df = pd.read_csv(path)
    return df


def load_raw_links(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw links.csv."""
    path = file_path or get_data_paths()["raw_links"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
    
    df = pd.read_csv(path)
    return df


def load_processed_movies(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load preprocessed movies dataset with fallback to preprocessing pipeline."""
    path = file_path or get_data_paths()["processed_movies"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at: {path}. "
            "Run 'python scripts/build_model.py' to generate processed data."
        )
    return pd.read_csv(path)
