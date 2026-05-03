"""
sentiment.py — Sentiment analysis model implementations.

Three models are provided:
  1. VADERAnalyzer   — Rule-based, uses NLTK's VADER lexicon.
  2. TextBlobAnalyzer — Lexicon + pattern-based, uses TextBlob.
  3. DistilBERTAnalyzer — Transformer-based, uses HuggingFace pipeline.

Each analyzer exposes:
  - analyze(text: str) -> dict
  - analyze_batch(texts: list[str]) -> list[dict]

All results include keys:
  - label  : 'Positive' | 'Negative' | 'Neutral'
  - score  : float in [0, 1] (confidence or normalized polarity)
  - details: model-specific breakdown dict
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from utils import clean_text, truncate_text, compound_to_label, polarity_to_label


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------

class VADERAnalyzer:
    """
    Sentiment analysis using NLTK's Valence Aware Dictionary and sEntiment Reasoner.

    VADER is optimized for social-media text and works well on short reviews.
    It does NOT require cleaned/lowercased text — it handles raw text natively.
    """

    def __init__(self):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self._sia = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text string.

        Returns:
            {label, score, details: {pos, neg, neu, compound}}
        """
        if not isinstance(text, str) or not text.strip():
            return {"label": "Neutral", "score": 0.0, "details": {}}

        scores = self._sia.polarity_scores(text)
        label  = compound_to_label(scores["compound"])

        # Normalize compound [-1,1] → confidence [0,1]
        confidence = (scores["compound"] + 1) / 2

        return {
            "label": label,
            "score": round(confidence, 4),
            "details": {
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral":  round(scores["neu"], 4),
                "compound": round(scores["compound"], 4),
            },
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze a list of texts. Returns a list of result dicts."""
        return [self.analyze(t) for t in texts]


# ---------------------------------------------------------------------------
# TextBlob
# ---------------------------------------------------------------------------

class TextBlobAnalyzer:
    """
    Sentiment analysis using TextBlob's pattern-based approach.

    Produces a polarity score in [-1, 1] and a subjectivity score in [0, 1].
    Works best on clean, grammatically correct English text.
    """

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text string.

        Returns:
            {label, score, details: {polarity, subjectivity}}
        """
        from textblob import TextBlob

        if not isinstance(text, str) or not text.strip():
            return {"label": "Neutral", "score": 0.0, "details": {}}

        blob = TextBlob(clean_text(text))
        polarity     = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        label        = polarity_to_label(polarity)

        # Normalize polarity [-1,1] → confidence [0,1]
        confidence = (polarity + 1) / 2

        return {
            "label": label,
            "score": round(confidence, 4),
            "details": {
                "polarity":     round(polarity, 4),
                "subjectivity": round(subjectivity, 4),
            },
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze a list of texts. Returns a list of result dicts."""
        return [self.analyze(t) for t in texts]


# ---------------------------------------------------------------------------
# DistilBERT
# ---------------------------------------------------------------------------

class DistilBERTAnalyzer:
    """
    Sentiment analysis using a fine-tuned DistilBERT model from HuggingFace.

    Model: distilbert-base-uncased-finetuned-sst-2-english
    Labels from model: POSITIVE / NEGATIVE  (no explicit NEUTRAL)
    We add a NEUTRAL band around the 0.5 confidence boundary.

    The pipeline is loaded once and cached via Streamlit's @st.cache_resource.
    """

    # Class-level neutral threshold — scores within this band of 0.5 are Neutral
    NEUTRAL_BAND = 0.15

    def __init__(self, pipeline):
        self._pipeline = pipeline

    @staticmethod
    @st.cache_resource(show_spinner="Loading DistilBERT model…")
    def load_pipeline():
        """
        Load and cache the HuggingFace pipeline.
        Called once per Streamlit session; subsequent calls return the cached object.
        """
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )

    def _map_result(self, raw: Dict) -> Dict[str, Any]:
        """
        Map a raw HuggingFace pipeline result to our unified format.

        The model returns POSITIVE/NEGATIVE with a confidence score.
        We apply a neutral band: if confidence < 0.5 + NEUTRAL_BAND, label → Neutral.
        """
        hf_label = raw["label"]          # 'POSITIVE' or 'NEGATIVE'
        hf_score = raw["score"]          # confidence for that label

        # Derive polarity confidence: how far from 0.5?
        if hf_label == "POSITIVE":
            confidence = hf_score
        else:
            confidence = 1.0 - hf_score  # flip so 0=neg, 1=pos

        # Apply neutral band
        if abs(confidence - 0.5) < self.NEUTRAL_BAND:
            label = "Neutral"
        elif confidence > 0.5:
            label = "Positive"
        else:
            label = "Negative"

        return {
            "label": label,
            "score": round(confidence, 4),
            "details": {
                "hf_label":    hf_label,
                "hf_score":    round(hf_score, 4),
                "pos_confidence": round(confidence, 4),
            },
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze a single text string."""
        if not isinstance(text, str) or not text.strip():
            return {"label": "Neutral", "score": 0.5, "details": {}}

        text = truncate_text(text, max_words=400)
        raw  = self._pipeline(text)[0]
        return self._map_result(raw)

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Analyze texts in batches for efficiency.

        Args:
            texts: List of input strings.
            batch_size: Number of texts per forward pass.

        Returns:
            List of result dicts.
        """
        cleaned = [truncate_text(t, 400) if isinstance(t, str) else "" for t in texts]
        results = []
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            raw_batch = self._pipeline(batch)
            results.extend([self._map_result(r) for r in raw_batch])
        return results


# ---------------------------------------------------------------------------
# Convenience: run all models on a DataFrame
# ---------------------------------------------------------------------------

def run_all_models(
    df: pd.DataFrame,
    vader: VADERAnalyzer,
    textblob: TextBlobAnalyzer,
    distilbert: DistilBERTAnalyzer,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run VADER, TextBlob, and DistilBERT on every row of the DataFrame.

    Adds columns:
      vader_label, vader_score,
      textblob_label, textblob_score,
      distilbert_label, distilbert_score

    Args:
        df: DataFrame with a 'text' column.
        vader, textblob, distilbert: Analyzer instances.
        progress_callback: Optional callable(step: int, total: int, message: str).

    Returns:
        The original DataFrame with new sentiment columns appended.
    """
    texts = df["text"].tolist()
    total = len(texts)

    if progress_callback:
        progress_callback(0, total, "Running VADER…")
    vader_results = vader.analyze_batch(texts)

    if progress_callback:
        progress_callback(1, total, "Running TextBlob…")
    tb_results = textblob.analyze_batch(texts)

    if progress_callback:
        progress_callback(2, total, "Running DistilBERT…")
    db_results = distilbert.analyze_batch(texts)

    result_df = df.copy()
    result_df["vader_label"]      = [r["label"] for r in vader_results]
    result_df["vader_score"]      = [r["score"] for r in vader_results]
    result_df["textblob_label"]   = [r["label"] for r in tb_results]
    result_df["textblob_score"]   = [r["score"] for r in tb_results]
    result_df["distilbert_label"] = [r["label"] for r in db_results]
    result_df["distilbert_score"] = [r["score"] for r in db_results]

    if progress_callback:
        progress_callback(3, total, "Done.")

    return result_df
