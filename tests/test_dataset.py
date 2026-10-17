"""Automated tests for dataset validation and integrity."""

import json
from pathlib import Path
import pandas as pd
import pytest


def test_processed_dataset_exists():
    data_path = Path("data/processed/clean_spam.csv")
    assert data_path.exists(), "Processed dataset file clean_spam.csv must exist."
    
    df = pd.read_csv(data_path)
    assert len(df) > 4000, f"Expected >4000 records, got {len(df)}"
    assert "label" in df.columns
    assert "message" in df.columns


def test_dataset_no_nulls():
    data_path = Path("data/processed/clean_spam.csv")
    df = pd.read_csv(data_path)
    assert df["label"].isnull().sum() == 0
    assert df["message"].isnull().sum() == 0


def test_dataset_valid_labels():
    data_path = Path("data/processed/clean_spam.csv")
    df = pd.read_csv(data_path)
    unique_labels = set(df["label"].unique())
    assert unique_labels == {"ham", "spam"}, f"Unexpected labels in dataset: {unique_labels}"


def test_dataset_summary_report():
    report_path = Path("reports/dataset_summary.json")
    assert report_path.exists(), "reports/dataset_summary.json must exist."
    
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "dataset_name" in data
    assert "label_distribution" in data
    assert "statistics" in data
    assert data["label_distribution"]["spam"]["count"] > 0
    assert data["label_distribution"]["ham"]["count"] > 0
