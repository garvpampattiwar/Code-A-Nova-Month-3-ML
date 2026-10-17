"""Safe inference module for Spam Detection.

Loads serialized pipeline and performs validation, classification,
and probability calibration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "spam_pipeline.joblib"

# Module-level model cache to avoid reloading from disk on every invocation
_CACHED_PIPELINE: Optional[Pipeline] = None


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the trained scikit-learn pipeline from disk with in-memory caching."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is not None:
        return _CACHED_PIPELINE

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Please run 'python -m src.train_model' to train and save the pipeline."
        )

    logger.info(f"Loading trained model pipeline from {model_path}...")
    pipeline = joblib.load(model_path)
    _CACHED_PIPELINE = pipeline
    return pipeline


def validate_input(text: Any) -> Tuple[bool, str]:
    """Validate user input before inference.

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if text is None:
        return False, "Input message cannot be null."

    if not isinstance(text, str):
        return False, f"Expected string input, received {type(text).__name__}."

    stripped = text.strip()
    if len(stripped) == 0:
        return False, "Please enter an email or message to analyze. Input is empty."

    if len(stripped) < 3:
        return False, "Message is too short to analyze (minimum 3 characters required)."

    return True, ""


def predict_message(text: str, model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Classify a single email or text message.

    Args:
        text: Raw email or message text.
        model_path: Optional path to serialized model artifact.

    Returns:
        Dictionary containing label, probabilities, cleaned text, and confidence status.
    """
    is_valid, err_msg = validate_input(text)
    if not is_valid:
        return {
            "success": False,
            "error": err_msg,
            "label": None,
            "spam_probability": 0.0,
            "ham_probability": 0.0,
            "confidence_pct": 0.0,
        }

    pipeline = load_model(model_path)
    cleaned = clean_text(text)

    # Note: TfidfVectorizer inside pipeline applies clean_text,
    # but we pass the raw text to let pipeline do its full transformation.
    probabilities = pipeline.predict_proba([text])[0]
    ham_prob = float(probabilities[0])
    spam_prob = float(probabilities[1])

    is_spam = spam_prob >= 0.50
    label = "SPAM" if is_spam else "NOT SPAM"
    confidence = spam_prob if is_spam else ham_prob

    # Extract informative tokens if vectorizer vocabulary matches
    top_signals = []
    try:
        vectorizer = pipeline.named_steps.get("tfidf")
        classifier = pipeline.named_steps.get("classifier")
        if vectorizer and classifier:
            feature_names = vectorizer.get_feature_names_out()
            # Transform single text to get TF-IDF weights
            transformed_vec = vectorizer.transform([text]).toarray()[0]
            non_zero_indices = transformed_vec.nonzero()[0]
            
            # For Naive Bayes, feature log prob difference between spam and ham
            if hasattr(classifier, "feature_log_prob_"):
                spam_log_prob = classifier.feature_log_prob_[1]
                ham_log_prob = classifier.feature_log_prob_[0]
                diff = spam_log_prob - ham_log_prob
                
                scored_tokens = [
                    (feature_names[idx], float(transformed_vec[idx]), float(diff[idx]))
                    for idx in non_zero_indices
                ]
                # Sort by strongest spam weight or strongest ham weight
                scored_tokens.sort(key=lambda x: abs(x[2]), reverse=True)
                top_signals = [
                    {"word": word, "tfidf_weight": round(weight, 3), "spam_tendency": "Spam" if d > 0 else "Legitimate"}
                    for word, weight, d in scored_tokens[:8]
                ]
    except Exception as e:
        logger.debug(f"Token signal extraction skipped: {e}")

    return {
        "success": True,
        "error": None,
        "raw_text": text,
        "cleaned_preview": cleaned,
        "is_spam": bool(is_spam),
        "label": label,
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(ham_prob, 4),
        "confidence_pct": round(confidence * 100, 2),
        "top_signals": top_signals,
    }


if __name__ == "__main__":
    test_samples = [
        "Congratulations! You won a $1,000 Walmart gift card. Click http://fake-claim.biz now!",
        "Hi John, hope you are doing well. Are we still on for our project review meeting tomorrow at 2 PM?",
        "",  # Invalid input test
    ]
    for sample in test_samples:
        result = predict_message(sample)
        print(f"\nMessage: '{sample}'")
        print(f"Result:  {result['label']} (Spam Prob: {result.get('spam_probability')}, Conf: {result.get('confidence_pct')}%)")
        if result["error"]:
            print(f"Error:   {result['error']}")
