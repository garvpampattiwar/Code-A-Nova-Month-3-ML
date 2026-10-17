"""Model training and evaluation pipeline for Spam Detection.

Trains and evaluates scikit-learn pipelines with TF-IDF feature extraction
and classifiers (MultinomialNB and LogisticRegression).
Ensures zero data leakage by strictly fitting vectorizers on training data only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data_loader import acquire_and_prepare_dataset
from src.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "clean_spam.csv"


def load_dataset() -> pd.DataFrame:
    """Load validated dataset or trigger acquisition if missing."""
    if not DATA_FILE.exists():
        logger.info(f"{DATA_FILE} not found. Acquiring dataset...")
        df, _ = acquire_and_prepare_dataset()
        return df
    return pd.read_csv(DATA_FILE)


def build_pipeline(classifier_type: str = "multinomial_nb") -> Pipeline:
    """Construct a full TF-IDF + Classifier Pipeline.

    Args:
        classifier_type: 'multinomial_nb' or 'logistic_regression'.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    if classifier_type == "multinomial_nb":
        clf = MultinomialNB(alpha=0.2)
    elif classifier_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    return Pipeline([("tfidf", vectorizer), ("classifier", clf)])


def evaluate_cross_validation(X_train: pd.Series, y_train: pd.Series) -> Dict[str, Any]:
    """Perform Stratified 5-Fold Cross-Validation on the training set to compare architectures."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]

    results: Dict[str, Any] = {}
    candidates = {
        "MultinomialNB (alpha=0.2)": build_pipeline("multinomial_nb"),
        "LogisticRegression (balanced)": build_pipeline("logistic_regression")
    }

    logger.info("Running 5-Fold Stratified Cross Validation on Training Set...")
    for name, pipe in candidates.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(scores["test_precision"])),
            "cv_recall_mean": float(np.mean(scores["test_recall"])),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
        }
        logger.info(
            f"[{name}] CV F1: {results[name]['cv_f1_mean']:.4f} | "
            f"Precision: {results[name]['cv_precision_mean']:.4f} | "
            f"Recall: {results[name]['cv_recall_mean']:.4f}"
        )

    return results


def train_and_evaluate() -> Dict[str, Any]:
    """Full training, cross-validation, final evaluation, and artifact serialization."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    logger.info(f"Loaded dataset containing {len(df)} records.")

    # Target: 0 for ham (not spam), 1 for spam
    X = df["message"]
    y = (df["label"] == "spam").astype(int)

    # Stratified Train/Test split: 80% train, 20% test, fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info(f"Train split size: {len(X_train)} (Spam: {sum(y_train)}, Ham: {len(y_train) - sum(y_train)})")
    logger.info(f"Test split size:  {len(X_test)} (Spam: {sum(y_test)}, Ham: {len(y_test) - sum(y_test)})")

    # 1. Compare models with CV on training set
    cv_comparison = evaluate_cross_validation(X_train, y_train)

    # 2. Select and fit primary model on full training set
    primary_pipe = build_pipeline("multinomial_nb")
    primary_pipe.fit(X_train, y_train)

    # 3. Predict on held-out test set
    y_pred = primary_pipe.predict(X_test)
    y_prob = primary_pipe.predict_proba(X_test)[:, 1]

    # 4. Compute Test Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    test_metrics = {
        "test_size": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_spam": round(float(precision_score(y_test, y_pred, pos_label=1)), 4),
        "recall_spam": round(float(recall_score(y_test, y_pred, pos_label=1)), 4),
        "f1_spam": round(float(f1_score(y_test, y_pred, pos_label=1)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives_ham": int(tn),
            "false_positives_ham_as_spam": int(fp),
            "false_negatives_spam_as_ham": int(fn),
            "true_positives_spam": int(tp),
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["NOT SPAM (ham)", "SPAM"], output_dict=True
        ),
        "cross_validation_comparison": cv_comparison,
    }

    # 5. Save metrics report
    metrics_file = REPORTS_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
    logger.info(f"Test metrics report saved to: {metrics_file}")

    # 6. Save Confusion Matrix visualization
    cm_plot_file = REPORTS_DIR / "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Spam (Ham)", "Spam"],
        yticklabels=["Not Spam (Ham)", "Spam"],
        cbar=False
    )
    plt.title("Confusion Matrix (Held-out Test Set)", fontsize=12, pad=12)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("Actual Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_plot_file, dpi=200)
    plt.close()
    logger.info(f"Confusion matrix plot saved to: {cm_plot_file}")

    # 7. Collect and document False Positives and False Negatives for error analysis
    test_df = pd.DataFrame({
        "message": X_test.values,
        "actual": ["spam" if label == 1 else "ham" for label in y_test.values],
        "predicted": ["spam" if label == 1 else "ham" for label in y_pred],
        "spam_probability": [round(float(p), 4) for p in y_prob]
    })
    
    fps = test_df[(test_df["actual"] == "ham") & (test_df["predicted"] == "spam")].to_dict(orient="records")
    fns = test_df[(test_df["actual"] == "spam") & (test_df["predicted"] == "ham")].to_dict(orient="records")

    error_analysis = {
        "false_positive_count": len(fps),
        "false_negative_count": len(fns),
        "false_positives_sample": fps[:5],
        "false_negatives_sample": fns[:5],
    }
    error_file = REPORTS_DIR / "error_analysis.json"
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=4)
    logger.info(f"Error analysis saved to: {error_file}")

    # 8. Save the best pipeline model artifact
    model_file = MODELS_DIR / "spam_pipeline.joblib"
    joblib.dump(primary_pipe, model_file)
    logger.info(f"Trained pipeline model saved to: {model_file}")

    return test_metrics


if __name__ == "__main__":
    report = train_and_evaluate()
    print("\n--- MODEL TRAINING & EVALUATION REPORT ---")
    print(f"Accuracy:       {report['accuracy'] * 100:.2f}%")
    print(f"Spam Precision: {report['precision_spam'] * 100:.2f}%")
    print(f"Spam Recall:    {report['recall_spam'] * 100:.2f}%")
    print(f"Spam F1-Score:  {report['f1_spam'] * 100:.2f}%")
    print(f"ROC-AUC:        {report['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN (Ham correctly classified):  {report['confusion_matrix']['true_negatives_ham']}")
    print(f"  FP (Ham misclassified as Spam): {report['confusion_matrix']['false_positives_ham_as_spam']}")
    print(f"  FN (Spam missed as Ham):        {report['confusion_matrix']['false_negatives_spam_as_ham']}")
    print(f"  TP (Spam correctly caught):     {report['confusion_matrix']['true_positives_spam']}")
