"""
data_loader.py — CSV ingestion and validation for the Sentiment Analysis Dashboard.

Responsibilities:
  - Load uploaded or sample CSV files into a pandas DataFrame.
  - Validate required columns.
  - Parse timestamps and categories if present.
  - Provide a clean preview and summary for the UI.
"""

import io
import pandas as pd
import streamlit as st


REQUIRED_COLUMN = "text"
OPTIONAL_TIMESTAMP = "timestamp"
OPTIONAL_CATEGORY  = "category"


def load_dataframe(source) -> pd.DataFrame:
    """
    Load a CSV from either an uploaded file object or a file path string.

    Args:
        source: A Streamlit UploadedFile, a file-like object, or a path string.

    Returns:
        A pandas DataFrame with at least a 'text' column.

    Raises:
        ValueError: If the CSV does not contain a 'text' column.
    """
    if isinstance(source, str):
        df = pd.read_csv(source)
    else:
        content = source.read()
        df = pd.read_csv(io.BytesIO(content))

    # Normalize column names: strip whitespace, lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    if REQUIRED_COLUMN not in df.columns:
        raise ValueError(
            f"CSV must contain a '{REQUIRED_COLUMN}' column. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows where text is null or empty
    df = df[df[REQUIRED_COLUMN].notna()]
    df = df[df[REQUIRED_COLUMN].astype(str).str.strip() != ""]
    df[REQUIRED_COLUMN] = df[REQUIRED_COLUMN].astype(str).str.strip()

    # Parse timestamp column if present
    if OPTIONAL_TIMESTAMP in df.columns:
        try:
            df[OPTIONAL_TIMESTAMP] = pd.to_datetime(df[OPTIONAL_TIMESTAMP], infer_datetime_format=True)
        except Exception:
            # Keep as-is if parsing fails
            pass

    # Normalize category column if present
    if OPTIONAL_CATEGORY in df.columns:
        df[OPTIONAL_CATEGORY] = df[OPTIONAL_CATEGORY].astype(str).str.strip().str.title()

    df = df.reset_index(drop=True)
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Compute a quick summary of the loaded dataset.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Dictionary with keys: total_reviews, has_timestamp, has_category,
        categories (list or None), date_range (tuple or None).
    """
    summary = {
        "total_reviews": len(df),
        "has_timestamp": OPTIONAL_TIMESTAMP in df.columns and pd.api.types.is_datetime64_any_dtype(df[OPTIONAL_TIMESTAMP]),
        "has_category":  OPTIONAL_CATEGORY in df.columns,
        "categories":    None,
        "date_range":    None,
    }

    if summary["has_category"]:
        summary["categories"] = sorted(df[OPTIONAL_CATEGORY].unique().tolist())

    if summary["has_timestamp"]:
        summary["date_range"] = (
            df[OPTIONAL_TIMESTAMP].min().strftime("%Y-%m-%d"),
            df[OPTIONAL_TIMESTAMP].max().strftime("%Y-%m-%d"),
        )

    return summary


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    """
    Load the bundled sample dataset (cached so it only reads once).

    Returns:
        DataFrame with 'text', 'timestamp', and 'category' columns.
    """
    return load_dataframe("sample_data.csv")
