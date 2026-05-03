"""
utils.py — Text preprocessing utilities for the Sentiment Analysis Dashboard.
Handles cleaning, normalization, and helper functions shared across modules.
"""

import re
import string
import nltk
from nltk.corpus import stopwords

# Download required NLTK data (runs once)
def download_nltk_resources():
    """Download all required NLTK resources silently."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


download_nltk_resources()


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text for NLP processing.

    Steps:
      - Lowercase
      - Remove URLs
      - Remove HTML tags
      - Remove punctuation
      - Collapse whitespace

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"<.*?>", "", text)                    # Remove HTML tags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)  # Punctuation
    text = re.sub(r"\s+", " ", text).strip()             # Collapse whitespace
    return text


def truncate_text(text: str, max_words: int = 512) -> str:
    """
    Truncate text to a maximum number of words.
    Used to stay within transformer model token limits.

    Args:
        text: Input string.
        max_words: Maximum number of words allowed.

    Returns:
        Truncated string.
    """
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def polarity_to_label(polarity: float, threshold: float = 0.05) -> str:
    """
    Map a continuous polarity score to a sentiment label.

    Args:
        polarity: Float in [-1, 1].
        threshold: Dead-zone around zero for neutral classification.

    Returns:
        One of 'Positive', 'Negative', or 'Neutral'.
    """
    if polarity > threshold:
        return "Positive"
    elif polarity < -threshold:
        return "Negative"
    else:
        return "Neutral"


def compound_to_label(compound: float, pos_thresh: float = 0.05, neg_thresh: float = -0.05) -> str:
    """
    Map VADER compound score to a sentiment label.

    Args:
        compound: VADER compound score in [-1, 1].
        pos_thresh: Threshold above which text is positive.
        neg_thresh: Threshold below which text is negative.

    Returns:
        One of 'Positive', 'Negative', or 'Neutral'.
    """
    if compound >= pos_thresh:
        return "Positive"
    elif compound <= neg_thresh:
        return "Negative"
    else:
        return "Neutral"


def label_to_color(label: str) -> str:
    """
    Return a consistent hex color for a sentiment label.

    Args:
        label: 'Positive', 'Negative', or 'Neutral'.

    Returns:
        Hex color string.
    """
    palette = {
        "Positive": "#2ECC71",
        "Negative": "#E74C3C",
        "Neutral":  "#95A5A6",
    }
    return palette.get(label, "#BDC3C7")


SENTIMENT_COLORS = {
    "Positive": "#2ECC71",
    "Negative": "#E74C3C",
    "Neutral":  "#95A5A6",
}
