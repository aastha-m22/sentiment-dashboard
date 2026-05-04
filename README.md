# Sentiment Analysis Dashboard

> An interactive NLP dashboard that compares **VADER**, **TextBlob**, and **DistilBERT** sentiment models on customer reviews — built with Streamlit and Plotly.

---

##  Features

| Feature | Details |
|---|---|
| **3 Sentiment Models** | VADER (rule-based), TextBlob (lexicon), DistilBERT (transformer) |
| **Interactive Charts** | Distribution bars, time-series lines, category breakdowns, model comparison |
| **Model Comparison** | Side-by-side label counts + agreement heatmap |
| **Word Clouds** | Per-sentiment word clouds (requires `wordcloud`) |
| **CSV Upload** | Drop in any CSV with a `text` column |
| **Download Results** | Export enriched CSV with sentiment labels per row |

---

##  Project Structure

```
sentiment-dashboard/
│
├── app.py             # Streamlit UI — all pages and layout
├── sentiment.py       # VADER, TextBlob, DistilBERT analyzers
├── visualization.py   # Plotly chart builders
├── data_loader.py     # CSV ingestion and validation
├── utils.py           # Text cleaning and helper functions
├── requirements.txt   # Python dependencies
├── README.md
└── sample_data.csv    # 40-row demo dataset
```

---

## Setup

### 1. Clone / download

```bash
git clone https://github.com/your-repo/sentiment-dashboard.git
cd sentiment-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** replace `torch>=2.2.0` in `requirements.txt` with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Download NLTK data (automatic)

The app downloads VADER and tokenizer data automatically on first run. No action needed.

---

##  Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

##  How to Use

### Data Input
- **Sample data** — click *Use sample data* in the sidebar. Loads 40 pre-labelled customer reviews.
- **Upload CSV** — your file must have a `text` column (required). Optional columns:
  - `timestamp` — enables the *Sentiment Over Time* chart (any parseable date format).
  - `category` — enables the *Category Breakdown* chart.

### Model Selection (sidebar)
| Option | Description |
|---|---|
| **VADER** | Fast, rule-based. Great for social-media / informal text. |
| **TextBlob** | Pattern-based lexicon. Best on grammatically correct text. |
| **DistilBERT** | Fine-tuned transformer. Most accurate; downloads ~270 MB on first use. |
| **Compare All** | Runs all three, shows agreement heatmap and side-by-side comparison. |

### Tabs
| Tab | What you see |
|---|---|
|  Data Preview | Dataset stats, first 10 rows, column summary |
|  Analysis | KPI cards, charts, word clouds, per-row results table, CSV download |
|  Compare Models | Run all three models and compare label distributions and agreements |

---

##  Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Data | pandas |
| Charts | Plotly |
| Rule-based NLP | NLTK (VADER) |
| Lexicon NLP | TextBlob |
| Transformer | HuggingFace `transformers` + PyTorch |
| Word Cloud | wordcloud + Pillow |

---

##  Notes

- The DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) is downloaded from HuggingFace on first use (~270 MB). Subsequent runs use the local cache.
- All models are cached for the Streamlit session — no redundant re-loading on re-runs.
- The neutral band for DistilBERT is ±0.15 around the 0.5 confidence boundary (configurable in `sentiment.py`).

---

##  License

MIT — free to use, modify, and distribute.
