"""Text preprocessing pipeline for email and message spam detection.

Preserves critical spam signals:
- URLs / web addresses (normalized to 'http_token')
- Currency symbols ($, £, €, normalized to 'currency_token')
- Phone numbers and numeric sequences (normalized to 'number_token')
- Excessive exclamation/question marks (normalized to 'punct_excl_token')
- Case folding (lowercasing)
- Whitespace stripping & contraction
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List

# Regex patterns for key spam artifacts
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
CURRENCY_PATTERN = re.compile(r"[\$£€¥₹]")
PHONE_OR_NUM_PATTERN = re.compile(r"\b\d{3,}\b")  # 3 or more consecutive digits
EXCLAMATION_PATTERN = re.compile(r"!{2,}")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    """Preprocess a single raw message string while preserving spam indicators.

    Args:
        text: Input string or object.

    Returns:
        Cleaned, normalized text string.
    """
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase early for consistent regex matching
    text = text.lower().strip()
    if not text:
        return ""

    # 2. Normalize URLs
    text = URL_PATTERN.sub(" http_token ", text)

    # 3. Normalize Email Addresses
    text = EMAIL_PATTERN.sub(" email_token ", text)

    # 4. Normalize Currency Symbols
    text = CURRENCY_PATTERN.sub(" currency_token ", text)

    # 5. Normalize Multiple Exclamation marks (strong spam signal)
    text = EXCLAMATION_PATTERN.sub(" excl_token ", text)

    # 6. Normalize Multi-digit numbers (account numbers, verification codes, cash amounts)
    text = PHONE_OR_NUM_PATTERN.sub(" number_token ", text)

    # 7. Strip non-alphanumeric characters except basic tokens
    # Keep letters, numbers, and underscores (which form our special tokens like http_token)
    text = re.sub(r"[^\w\s]", " ", text)

    # 8. Collapse whitespace
    text = MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()

    return text


def preprocess_corpus(corpus: Iterable[str]) -> List[str]:
    """Apply clean_text across a sequence of messages.

    Args:
        corpus: Iterable collection of message strings.

    Returns:
        List of preprocessed strings.
    """
    return [clean_text(item) for item in corpus]
