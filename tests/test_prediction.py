"""Automated tests for prediction logic and model loading."""

from pathlib import Path
import pytest
from src.predict import load_model, predict_message, validate_input


def test_model_artifact_exists():
    model_path = Path("models/spam_pipeline.joblib")
    assert model_path.exists(), "Trained model artifact models/spam_pipeline.joblib must exist."
    assert model_path.stat().st_size > 1000, "Model artifact file appears corrupt or empty."


def test_model_loading():
    pipeline = load_model()
    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_validate_input_empty_and_short():
    valid, msg = validate_input("")
    assert not valid
    assert "empty" in msg.lower()

    valid, msg = validate_input("   ")
    assert not valid
    assert "empty" in msg.lower()

    valid, msg = validate_input(None)
    assert not valid
    assert "null" in msg.lower()

    valid, msg = validate_input("ab")
    assert not valid
    assert "too short" in msg.lower()

    valid, msg = validate_input("This is a legitimate message.")
    assert valid
    assert msg == ""


def test_predict_obvious_spam():
    message = "URGENT! You have won £1,000 cash guaranteed! Call 09061743810 now or claim at http://claim-award.biz"
    res = predict_message(message)
    assert res["success"] is True
    assert res["label"] == "SPAM"
    assert res["is_spam"] is True
    assert res["spam_probability"] > 0.70
    assert 0.0 <= res["confidence_pct"] <= 100.0


def test_predict_obvious_ham():
    message = "Hey mom, just checking in to see what time we are having dinner tonight. Love you!"
    res = predict_message(message)
    assert res["success"] is True
    assert res["label"] == "NOT SPAM"
    assert res["is_spam"] is False
    assert res["spam_probability"] < 0.30
    assert res["ham_probability"] > 0.70


def test_predict_empty_string_graceful():
    res = predict_message("")
    assert res["success"] is False
    assert res["error"] is not None
    assert res["label"] is None
