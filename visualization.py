"""
visualization.py — Plotly chart builders for the Sentiment Analysis Dashboard.

All functions accept a pandas DataFrame (already enriched with sentiment columns)
and return a plotly Figure object ready to be passed to st.plotly_chart().

Color palette:
  Positive  → #2ECC71 (emerald green)
  Negative  → #E74C3C (alizarin red)
  Neutral   → #95A5A6 (concrete grey)
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from utils import SENTIMENT_COLORS

# ── Shared theme helpers ────────────────────────────────────────────────────

_FONT_FAMILY = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"

_LAYOUT_DEFAULTS = dict(
    font=dict(family=_FONT_FAMILY, size=13, color="#ECF0F1"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        bgcolor="rgba(30,39,46,0.7)",
        bordercolor="#2C3E50",
        borderwidth=1,
    ),
    margin=dict(l=30, r=30, t=50, b=30),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply shared layout defaults and an optional title."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#ECF0F1")),
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#95A5A6")
    fig.update_yaxes(showgrid=True, gridcolor="#2C3E50", zeroline=False, color="#95A5A6")
    return fig


# ── 1. Sentiment Distribution Bar Chart ────────────────────────────────────

def sentiment_distribution_chart(df: pd.DataFrame, label_col: str, title: str = "Sentiment Distribution") -> go.Figure:
    """
    Horizontal bar chart showing count of each sentiment label.

    Args:
        df: DataFrame containing the label column.
        label_col: Name of the column with sentiment labels.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    counts = df[label_col].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0).reset_index()
    counts.columns = ["Sentiment", "Count"]
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    counts["Color"] = counts["Sentiment"].map(SENTIMENT_COLORS)

    fig = go.Figure()
    for _, row in counts.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Count"]],
            y=[row["Sentiment"]],
            orientation="h",
            marker_color=row["Color"],
            text=f'{row["Count"]} ({row["Percentage"]}%)',
            textposition="outside",
            name=row["Sentiment"],
            hovertemplate=f"<b>{row['Sentiment']}</b><br>Count: {row['Count']}<br>Share: {row['Percentage']}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        showlegend=False,
        height=280,
    )
    _apply_defaults(fig, title)
    fig.update_xaxes(title_text="Number of Reviews")
    return fig


# ── 2. Sentiment Over Time (Line Chart) ────────────────────────────────────

def sentiment_over_time_chart(df: pd.DataFrame, label_col: str, timestamp_col: str = "timestamp", title: str = "Sentiment Over Time") -> go.Figure:
    """
    Line chart of daily/weekly sentiment label counts over time.

    Args:
        df: DataFrame with timestamp and label columns.
        label_col: Column with sentiment labels.
        timestamp_col: Column with datetime values.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    if timestamp_col not in df.columns:
        return go.Figure()

    ts_df = df[[timestamp_col, label_col]].copy()
    ts_df[timestamp_col] = pd.to_datetime(ts_df[timestamp_col])
    ts_df = ts_df.set_index(timestamp_col)

    # Resample by week; fall back to day if ≤14 data points
    n_days = (ts_df.index.max() - ts_df.index.min()).days
    freq = "W" if n_days > 14 else "D"

    pivot = (
        ts_df.groupby([pd.Grouper(freq=freq), label_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    fig = go.Figure()
    for label in ["Positive", "Negative", "Neutral"]:
        if label in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot[timestamp_col],
                y=pivot[label],
                name=label,
                mode="lines+markers",
                line=dict(color=SENTIMENT_COLORS[label], width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Count: %{{y}}<extra></extra>",
            ))

    fig.update_layout(height=350, hovermode="x unified")
    _apply_defaults(fig, title)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Count")
    return fig


# ── 3. Category Breakdown Grouped Bar Chart ─────────────────────────────────

def category_breakdown_chart(df: pd.DataFrame, label_col: str, category_col: str = "category", title: str = "Sentiment by Category") -> go.Figure:
    """
    Grouped bar chart: categories on X-axis, bars colored by sentiment.

    Args:
        df: DataFrame with category and label columns.
        label_col: Column with sentiment labels.
        category_col: Column with category names.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    if category_col not in df.columns:
        return go.Figure()

    pivot = (
        df.groupby([category_col, label_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    fig = go.Figure()
    for label in ["Positive", "Negative", "Neutral"]:
        if label in pivot.columns:
            fig.add_trace(go.Bar(
                x=pivot[category_col],
                y=pivot[label],
                name=label,
                marker_color=SENTIMENT_COLORS[label],
                hovertemplate=f"<b>{label}</b><br>Category: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))

    fig.update_layout(barmode="group", height=350)
    _apply_defaults(fig, title)
    fig.update_xaxes(title_text="Category")
    fig.update_yaxes(title_text="Count")
    return fig


# ── 4. Model Comparison Grouped Bar Chart ───────────────────────────────────

def model_comparison_chart(df: pd.DataFrame, title: str = "Model Comparison — Sentiment Distribution") -> go.Figure:
    """
    Side-by-side grouped bar chart comparing sentiment label distributions
    across VADER, TextBlob, and DistilBERT.

    Args:
        df: DataFrame with vader_label, textblob_label, distilbert_label columns.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    models = {
        "VADER":      "vader_label",
        "TextBlob":   "textblob_label",
        "DistilBERT": "distilbert_label",
    }
    labels = ["Positive", "Negative", "Neutral"]

    # Build a summary DataFrame
    records = []
    for model_name, col in models.items():
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        for lbl in labels:
            records.append({
                "Model":     model_name,
                "Sentiment": lbl,
                "Count":     counts.get(lbl, 0),
            })

    summary = pd.DataFrame(records)

    fig = px.bar(
        summary,
        x="Model",
        y="Count",
        color="Sentiment",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        hover_data=["Count"],
    )

    fig.update_layout(height=380, legend_title_text="Sentiment")
    _apply_defaults(fig, title)
    fig.update_xaxes(title_text="Model")
    fig.update_yaxes(title_text="Count")
    return fig


# ── 5. Score Distribution (Violin / Box) ────────────────────────────────────

def score_distribution_chart(df: pd.DataFrame, score_col: str, label_col: str, model_name: str = "") -> go.Figure:
    """
    Violin plot of confidence scores split by sentiment label.

    Args:
        df: DataFrame with score and label columns.
        score_col: Column with float confidence scores.
        label_col: Column with sentiment labels.
        model_name: Used in the chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    for label in ["Positive", "Negative", "Neutral"]:
        subset = df[df[label_col] == label][score_col]
        if subset.empty:
            continue
        fig.add_trace(go.Violin(
            y=subset,
            name=label,
            box_visible=True,
            meanline_visible=True,
            fillcolor=SENTIMENT_COLORS[label],
            opacity=0.7,
            line_color=SENTIMENT_COLORS[label],
        ))

    title = f"{model_name} — Confidence Score Distribution" if model_name else "Score Distribution"
    fig.update_layout(height=350, violinmode="group")
    _apply_defaults(fig, title)
    fig.update_yaxes(title_text="Confidence Score (0–1)")
    return fig


# ── 6. Agreement Heatmap (Compare All) ──────────────────────────────────────

def agreement_heatmap(df: pd.DataFrame, title: str = "Model Agreement Heatmap") -> go.Figure:
    """
    Heatmap showing how often each pair of models agree on a label.

    Args:
        df: DataFrame with vader_label, textblob_label, distilbert_label columns.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    model_cols = {
        "VADER":      "vader_label",
        "TextBlob":   "textblob_label",
        "DistilBERT": "distilbert_label",
    }
    # Filter to columns actually present
    available = {k: v for k, v in model_cols.items() if v in df.columns}
    model_names = list(available.keys())

    n = len(model_names)
    matrix = [[0.0] * n for _ in range(n)]

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            col1, col2 = available[m1], available[m2]
            agreement = (df[col1] == df[col2]).mean() * 100
            matrix[i][j] = round(agreement, 1)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=model_names,
        y=model_names,
        colorscale="RdYlGn",
        zmin=0,
        zmax=100,
        text=[[f"{v}%" for v in row] for row in matrix],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} vs %{x}</b><br>Agreement: %{z}%<extra></extra>",
        colorbar=dict(title="Agreement %", ticksuffix="%"),
    ))

    fig.update_layout(height=340)
    _apply_defaults(fig, title)
    return fig


# ── 7. Word Cloud (PIL-based, returned as image bytes) ───────────────────────

def generate_wordcloud(df: pd.DataFrame, label_filter: Optional[str] = None, label_col: str = "vader_label") -> Optional[bytes]:
    """
    Generate a word-cloud image for the given sentiment subset.

    Args:
        df: DataFrame with 'text' column.
        label_filter: If provided, only use rows where label_col == label_filter.
        label_col: Column to filter by.

    Returns:
        PNG image bytes, or None if wordcloud is not installed.
    """
    try:
        from wordcloud import WordCloud
        import io
        from PIL import Image
    except ImportError:
        return None

    subset = df.copy()
    if label_filter and label_col in df.columns:
        subset = subset[subset[label_col] == label_filter]

    text = " ".join(subset["text"].astype(str).tolist())
    if not text.strip():
        return None

    color = {
        "Positive": "#2ECC71",
        "Negative": "#E74C3C",
        "Neutral":  "#95A5A6",
    }.get(label_filter, "#3498DB")

    wc = WordCloud(
        width=800,
        height=400,
        background_color="#1A252F",
        colormap="viridis" if label_filter != "Negative" else "Reds",
        max_words=120,
        collocations=False,
    ).generate(text)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()
