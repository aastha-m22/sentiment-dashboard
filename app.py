"""
app.py — Streamlit entry point for the Sentiment Analysis Dashboard.

Run with:
    streamlit run app.py

Structure:
  ├── Sidebar  : data source + model selection
  ├── Page 1   : Data Preview & Dataset Summary
  ├── Page 2   : Single-Model Analysis (VADER / TextBlob / DistilBERT)
  └── Page 3   : Compare All Models
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* ── Dark background ── */
  .stApp {
    background: linear-gradient(135deg, #0D1B2A 0%, #1A252F 60%, #0D1B2A 100%);
    color: #ECF0F1;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(26, 37, 47, 0.95);
    border-right: 1px solid #2C3E50;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: rgba(44, 62, 80, 0.5);
    border: 1px solid #2C3E50;
    border-radius: 12px;
    padding: 16px;
  }

  /* ── Positive / Negative / Neutral colored badges ── */
  .badge-positive { background:#1E8449; color:#fff; padding:2px 10px; border-radius:20px; font-size:0.8rem; }
  .badge-negative { background:#C0392B; color:#fff; padding:2px 10px; border-radius:20px; font-size:0.8rem; }
  .badge-neutral  { background:#616A6B; color:#fff; padding:2px 10px; border-radius:20px; font-size:0.8rem; }

  /* ── Section headers ── */
  h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; }
  h1 { font-weight: 700; letter-spacing: -0.5px; }
  h2 { font-weight: 600; color: #85C1E9; }
  h3 { font-weight: 600; color: #AED6F1; }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] { border-radius: 8px; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    background: rgba(44,62,80,0.4);
    color: #AED6F1;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(52,152,219,0.2) !important;
    color: #3498DB !important;
    border-bottom: 2px solid #3498DB;
  }

  /* ── Divider ── */
  hr { border-color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

# ── Local imports (after page config) ───────────────────────────────────────
from data_loader import load_dataframe, load_sample_data, get_dataset_summary
from sentiment   import VADERAnalyzer, TextBlobAnalyzer, DistilBERTAnalyzer, run_all_models
from visualization import (
    sentiment_distribution_chart,
    sentiment_over_time_chart,
    category_breakdown_chart,
    model_comparison_chart,
    score_distribution_chart,
    agreement_heatmap,
    generate_wordcloud,
)


# ═══════════════════════════════════════════════════════════════════════════
# Model initialisation (cached so they live for the whole session)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_vader():
    return VADERAnalyzer()

@st.cache_resource(show_spinner=False)
def get_textblob():
    return TextBlobAnalyzer()

@st.cache_resource(show_spinner=False)
def get_distilbert():
    pipeline = DistilBERTAnalyzer.load_pipeline()
    return DistilBERTAnalyzer(pipeline)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Sentiment Dashboard")
    st.markdown("---")

    # ── Data source ───────────────────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Use sample data", "Upload CSV"],
        label_visibility="collapsed",
    )

    df_raw: pd.DataFrame | None = None

    if data_source == "Use sample data":
        df_raw = load_sample_data()
        st.success("✅ Sample data loaded (40 reviews)")

    else:
        uploaded = st.file_uploader(
            "Upload a CSV with a 'text' column",
            type=["csv"],
            help="Required column: `text`. Optional: `timestamp`, `category`.",
        )
        if uploaded:
            try:
                df_raw = load_dataframe(uploaded)
                st.success(f"✅ Loaded {len(df_raw):,} reviews")
            except ValueError as e:
                st.error(str(e))

    st.markdown("---")

    # ── Model selector ────────────────────────────────────────────────────
    st.markdown("### 🔬 Model")
    model_choice = st.selectbox(
        "Sentiment model",
        ["VADER", "TextBlob", "DistilBERT", "Compare All"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Compare rule-based **(VADER, TextBlob)** and "
        "transformer-based **(DistilBERT)** sentiment models "
        "on your customer reviews."
    )
    st.markdown("[GitHub](https://github.com/) · Built with ❤️ & Streamlit")


# ═══════════════════════════════════════════════════════════════════════════
# Guard: nothing to show until data is loaded
# ═══════════════════════════════════════════════════════════════════════════

if df_raw is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 0;'>
      <h1>🧠 Sentiment Analysis Dashboard</h1>
      <p style='color:#95A5A6; font-size:1.1rem;'>
        Upload a CSV file or use the sample data to get started.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

summary = get_dataset_summary(df_raw)

st.markdown(f"# 🧠 Sentiment Analysis Dashboard")
st.markdown(f"Analyzing **{summary['total_reviews']:,} reviews** — Model: **{model_choice}**")
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

tab_data, tab_analysis, tab_compare = st.tabs([
    "📋 Data Preview",
    "📊 Analysis",
    "⚖️ Compare Models",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — Data Preview
# ───────────────────────────────────────────────────────────────────────────

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews",  f"{summary['total_reviews']:,}")
    c2.metric("Has Timestamps", "✅ Yes" if summary["has_timestamp"] else "❌ No")
    c3.metric("Has Categories", "✅ Yes" if summary["has_category"]  else "❌ No")
    c4.metric("Categories",     len(summary["categories"]) if summary["categories"] else "—")

    st.markdown("### Preview (first 10 rows)")
    st.dataframe(df_raw.head(10), use_container_width=True, height=320)

    if summary["categories"]:
        st.markdown(f"**Categories found:** {', '.join(summary['categories'])}")

    if summary["date_range"]:
        st.markdown(f"**Date range:** {summary['date_range'][0]} → {summary['date_range'][1]}")


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — Single Model Analysis
# ───────────────────────────────────────────────────────────────────────────

with tab_analysis:
    if model_choice == "Compare All":
        st.info("Switch the model selector to VADER, TextBlob, or DistilBERT to see single-model analysis.")
        st.stop()

    # ── Run selected model ───────────────────────────────────────────────
    @st.cache_data(show_spinner="Analysing reviews…", hash_funcs={pd.DataFrame: lambda df: df.to_json()})
    def run_single_model(df: pd.DataFrame, model: str) -> pd.DataFrame:
        texts = df["text"].tolist()

        if model == "VADER":
            analyzer  = get_vader()
            results   = analyzer.analyze_batch(texts)
            label_col = "vader_label"
            score_col = "vader_score"
        elif model == "TextBlob":
            analyzer  = get_textblob()
            results   = analyzer.analyze_batch(texts)
            label_col = "textblob_label"
            score_col = "textblob_score"
        else:  # DistilBERT
            analyzer  = get_distilbert()
            results   = analyzer.analyze_batch(texts)
            label_col = "distilbert_label"
            score_col = "distilbert_score"

        out = df.copy()
        out[label_col] = [r["label"] for r in results]
        out[score_col] = [r["score"] for r in results]
        return out, label_col, score_col

    result_df, label_col, score_col = run_single_model(df_raw, model_choice)

    # ── KPI row ──────────────────────────────────────────────────────────
    counts = result_df[label_col].value_counts()
    total  = len(result_df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews", total)
    k2.metric("😊 Positive",   f"{counts.get('Positive', 0)} ({counts.get('Positive',0)/total*100:.0f}%)")
    k3.metric("😐 Neutral",    f"{counts.get('Neutral',  0)} ({counts.get('Neutral', 0)/total*100:.0f}%)")
    k4.metric("😞 Negative",   f"{counts.get('Negative', 0)} ({counts.get('Negative',0)/total*100:.0f}%)")

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Distribution")
        st.plotly_chart(
            sentiment_distribution_chart(result_df, label_col),
            use_container_width=True,
        )

    with col_right:
        if summary["has_timestamp"]:
            st.markdown("#### Sentiment Over Time")
            st.plotly_chart(
                sentiment_over_time_chart(result_df, label_col),
                use_container_width=True,
            )
        else:
            st.markdown("#### Score Distribution")
            st.plotly_chart(
                score_distribution_chart(result_df, score_col, label_col, model_choice),
                use_container_width=True,
            )

    if summary["has_category"]:
        st.markdown("#### Category Breakdown")
        st.plotly_chart(
            category_breakdown_chart(result_df, label_col),
            use_container_width=True,
        )

    # ── Word cloud ───────────────────────────────────────────────────────
    st.markdown("#### ☁️ Word Clouds")
    wc_cols = st.columns(3)
    for idx, sentiment_filter in enumerate(["Positive", "Negative", "Neutral"]):
        img_bytes = generate_wordcloud(result_df, label_filter=sentiment_filter, label_col=label_col)
        with wc_cols[idx]:
            st.markdown(f"**{sentiment_filter}**")
            if img_bytes:
                st.image(img_bytes, use_container_width=True)
            else:
                st.caption("Install `wordcloud` for word clouds: `pip install wordcloud`")

    # ── Results table ────────────────────────────────────────────────────
    st.markdown("#### 📄 Review-Level Results")
    display_cols = ["text", label_col, score_col]
    if "timestamp" in result_df.columns:
        display_cols.insert(1, "timestamp")
    if "category" in result_df.columns:
        display_cols.insert(2, "category")

    st.dataframe(
        result_df[display_cols].rename(columns={
            label_col: "Sentiment",
            score_col: "Confidence",
        }),
        use_container_width=True,
        height=400,
    )

    # ── Download ─────────────────────────────────────────────────────────
    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download results as CSV",
        data=csv_out,
        file_name=f"sentiment_{model_choice.lower()}.csv",
        mime="text/csv",
    )


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — Compare All Models
# ───────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.markdown("Run all three models and compare their predictions side-by-side.")

    run_btn = st.button("▶️  Run All Models", type="primary")

    if "compare_df" not in st.session_state:
        st.session_state["compare_df"] = None

    if run_btn:
        with st.spinner("Running VADER, TextBlob, and DistilBERT…"):
            prog = st.progress(0, text="Starting…")

            def _progress(step, total, msg):
                prog.progress(step / 3, text=msg)

            st.session_state["compare_df"] = run_all_models(
                df_raw,
                get_vader(),
                get_textblob(),
                get_distilbert(),
                progress_callback=_progress,
            )
            prog.empty()
        st.success("✅ All models finished!")

    cdf = st.session_state["compare_df"]

    if cdf is not None:
        # ── KPI summary table ─────────────────────────────────────────────
        st.markdown("#### Model Summary")
        model_summary = []
        for m_name, l_col, s_col in [
            ("VADER",      "vader_label",      "vader_score"),
            ("TextBlob",   "textblob_label",   "textblob_score"),
            ("DistilBERT", "distilbert_label", "distilbert_score"),
        ]:
            cnts = cdf[l_col].value_counts()
            model_summary.append({
                "Model":    m_name,
                "Positive": cnts.get("Positive", 0),
                "Negative": cnts.get("Negative", 0),
                "Neutral":  cnts.get("Neutral",  0),
                "Avg Confidence": round(cdf[s_col].mean(), 3),
            })
        st.dataframe(pd.DataFrame(model_summary), use_container_width=True, hide_index=True)

        # ── Charts ────────────────────────────────────────────────────────
        st.markdown("#### Distribution Comparison")
        st.plotly_chart(model_comparison_chart(cdf), use_container_width=True)

        st.markdown("#### Agreement Heatmap")
        st.plotly_chart(agreement_heatmap(cdf), use_container_width=True)

        if summary["has_timestamp"]:
            st.markdown("#### Sentiment Over Time (per model)")
            for m_name, l_col in [
                ("VADER",      "vader_label"),
                ("TextBlob",   "textblob_label"),
                ("DistilBERT", "distilbert_label"),
            ]:
                st.plotly_chart(
                    sentiment_over_time_chart(cdf, l_col, title=f"{m_name} — Over Time"),
                    use_container_width=True,
                )

        # ── Full results table ────────────────────────────────────────────
        st.markdown("#### Full Comparison Table")
        show_cols = ["text", "vader_label", "textblob_label", "distilbert_label",
                     "vader_score", "textblob_score", "distilbert_score"]
        st.dataframe(cdf[show_cols], use_container_width=True, height=400)

        csv_all = cdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download full comparison CSV",
            data=csv_all,
            file_name="sentiment_comparison_all_models.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **Run All Models** to start the comparison.")
