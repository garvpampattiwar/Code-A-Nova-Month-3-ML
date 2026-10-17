"""Automated tests for text preprocessing."""

import pytest
from src.preprocessing import clean_text, preprocess_corpus


def test_clean_text_normal_string():
    raw = "Hello! Are we meeting at 4pm tomorrow?"
    cleaned = clean_text(raw)
    assert "hello" in cleaned
    assert "meeting" in cleaned
    assert "tomorrow" in cleaned


def test_clean_text_url_preservation():
    raw = "Click here: https://win-prizes-now.fake/login to claim your reward"
    cleaned = clean_text(raw)
    assert "http_token" in cleaned
    assert "click" in cleaned
    assert "claim" in cleaned
    assert "reward" in cleaned


def test_clean_text_currency_and_numbers():
    raw = "Congratulations, you won £10000 or $500 cash! Call 0800123456 now."
    cleaned = clean_text(raw)
    assert "currency_token" in cleaned
    assert "number_token" in cleaned
    assert "won" in cleaned


def test_clean_text_email_address():
    raw = "Contact support at billing-alert@phishing-update.com immediately."
    cleaned = clean_text(raw)
    assert "email_token" in cleaned
    assert "immediately" in cleaned


def test_clean_text_excessive_exclamation():
    raw = "URGENT ACTION REQUIRED!!!!"
    cleaned = clean_text(raw)
    assert "excl_token" in cleaned
    assert "urgent" in cleaned


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""


def test_preprocess_corpus():
    inputs = ["Hey there", "Win $500 now!"]
    outputs = preprocess_corpus(inputs)
    assert len(outputs) == 2
    assert "hey" in outputs[0]
    assert "currency_token" in outputs[1]
