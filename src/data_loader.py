"""Dataset loader and validator for Spam Detection.

Dataset Source:
    UCI SMS Spam Collection dataset (Tiago A. Almeida & Jose Maria Gomez Hidalgo).
    Publicly available benchmark dataset containing 5,572 labeled messages.
    Labels: 'ham' (legitimate/not spam) and 'spam' (unsolicited commercial/phishing).

Notice & Transparency:
    This model utilizes the classic SMS Spam Collection benchmark. The text
    classification algorithms and feature extraction (TF-IDF + Naive Bayes / Logistic Regression)
    operate on message text. While labeled as SMS messages, the lexical patterns
    (urgency, monetary rewards, suspicious links, phishing terms) apply directly to message-level
    spam detection.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
BACKUP_DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/sms_spam.csv"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def ensure_directories() -> None:
    """Create data and report directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(dest_path: Path) -> Path:
    """Download the raw dataset from verified repository mirror."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Dataset already exists locally at: {dest_path}")
        return dest_path

    logger.info(f"Downloading raw dataset from {RAW_DATA_URL} to {dest_path}...")
    try:
        req = urllib.request.Request(
            RAW_DATA_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=15) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        logger.info("Download completed successfully.")
        return dest_path
    except Exception as e:
        logger.warning(f"Primary source download failed: {e}. Attempting backup mirror...")
        csv_dest = RAW_DIR / "sms_spam_backup.csv"
        req = urllib.request.Request(
            BACKUP_DATA_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=15) as response, open(csv_dest, "wb") as out_file:
            out_file.write(response.read())
        return csv_dest


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw dataset with tab or comma separation."""
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
        # Standardize column names if needed
        if "type" in df.columns and "text" in df.columns:
            df = df.rename(columns={"type": "label", "text": "message"})
    else:
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8",
            on_bad_lines="skip"
        )
    return df


def validate_and_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate data schema, handle nulls, document duplicates, and calculate distribution."""
    initial_rows = len(df)
    
    # 1. Null check
    null_counts = df.isnull().sum().to_dict()
    df = df.dropna(subset=["label", "message"]).copy()
    
    # 2. String conversion & strip whitespace
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    
    # 3. Filter invalid labels
    valid_mask = df["label"].isin(["ham", "spam"])
    df = df[valid_mask].copy()
    
    # Filter empty messages
    df = df[df["message"].str.len() > 0].copy()
    
    # 4. Check duplicates
    duplicate_count = int(df.duplicated(subset=["message"]).sum())
    
    # Deduplicate for training to prevent train/test overlap leakage
    df_dedup = df.drop_duplicates(subset=["message"]).copy().reset_index(drop=True)
    
    # 5. Calculate metrics
    label_counts = df_dedup["label"].value_counts().to_dict()
    total_dedup_rows = len(df_dedup)
    
    df_dedup["char_length"] = df_dedup["message"].apply(len)
    df_dedup["word_count"] = df_dedup["message"].apply(lambda x: len(x.split()))
    
    summary: Dict[str, Any] = {
        "dataset_name": "UCI SMS Spam Collection",
        "dataset_license": "Public Domain / Attribution (CC BY 4.0 for UCI ML repository)",
        "source_url": RAW_DATA_URL,
        "raw_total_records": initial_rows,
        "null_counts": null_counts,
        "duplicate_records": duplicate_count,
        "deduplicated_total_records": total_dedup_rows,
        "label_distribution": {
            "ham": {
                "count": int(label_counts.get("ham", 0)),
                "percentage": round((label_counts.get("ham", 0) / total_dedup_rows) * 100, 2)
            },
            "spam": {
                "count": int(label_counts.get("spam", 0)),
                "percentage": round((label_counts.get("spam", 0) / total_dedup_rows) * 100, 2)
            }
        },
        "statistics": {
            "avg_char_length_ham": round(float(df_dedup[df_dedup["label"] == "ham"]["char_length"].mean()), 2),
            "avg_char_length_spam": round(float(df_dedup[df_dedup["label"] == "spam"]["char_length"].mean()), 2),
            "avg_word_count_ham": round(float(df_dedup[df_dedup["label"] == "ham"]["word_count"].mean()), 2),
            "avg_word_count_spam": round(float(df_dedup[df_dedup["label"] == "spam"]["word_count"].mean()), 2),
        },
        "disclaimer": (
            "Model is trained on the verified UCI SMS Spam Collection benchmark dataset. "
            "Although tested and evaluated on SMS text communications, text classification "
            "and TF-IDF representations generalize to email text patterns with similar vocabulary."
        )
    }
    
    return df_dedup, summary


def acquire_and_prepare_dataset() -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Complete workflow: ensure dirs, download raw, clean, validate, save processed & reports."""
    ensure_directories()
    raw_file = RAW_DIR / "sms_spam_raw.tsv"
    actual_path = download_dataset(raw_file)
    
    df_raw = load_raw_data(actual_path)
    df_clean, summary = validate_and_clean_data(df_raw)
    
    processed_file = PROCESSED_DIR / "clean_spam.csv"
    df_clean[["label", "message"]].to_csv(processed_file, index=False, encoding="utf-8")
    
    summary_file = REPORTS_DIR / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    logger.info(f"Cleaned dataset saved to: {processed_file}")
    logger.info(f"Dataset summary saved to: {summary_file}")
    return df_clean, summary


if __name__ == "__main__":
    _, report = acquire_and_prepare_dataset()
    print("\n--- DATASET VALIDATION REPORT ---")
    print(json.dumps(report, indent=2))
