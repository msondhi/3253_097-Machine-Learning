# E-Commerce Sentiment Analysis

A sentiment analysis system for e-commerce product reviews. Reviews are classified as **positive** or **negative** using four different ML approaches — from a simple TF-IDF baseline to a TensorFlow neural network on top of dense sentence embeddings. Results are served through a FastAPI REST API and a Gradio web app.

---

## Project Structure

```
├── app/
│   ├── data/
│   │   ├── reviews.csv                              # Women's clothing reviews (~28k rows)
│   │   ├── amazon_listing_review.csv                # Amazon fine food reviews (~568k rows)
│   │   ├── amazon-review/
│   │   │   ├── products.csv                         # Amazon apparel product metadata (728 products)
│   │   │   └── reviews.csv                          # Amazon apparel reviews (~6k rows)
│   │   ├── combined_reviews.csv                     # Merged dataset (~389k rows, generated)
│   │   └── embeddings/
│   │       ├── bge_clothing.npz                     # Pre-computed BGE embeddings — clothing
│   │       └── bge_combined.npz                     # Pre-computed BGE embeddings — combined
│   ├── model/
│   │   ├── sentiment_model.joblib                   # TF-IDF + Logistic Regression
│   │   ├── sentiment_model_svm.joblib               # TF-IDF + LinearSVC
│   │   ├── sentiment_model_bge.joblib               # BGE embeddings + Logistic Regression
│   │   ├── sentiment_model_bge_mlp.joblib           # BGE embeddings + TF MLP (wrapper)
│   │   └── sentiment_model_bge_mlp.keras            # TensorFlow Keras model weights
│   ├── resources/
│   │   └── api.py                                   # FastAPI application
│   ├── training/
│   │   ├── bge_model.py                             # BGELogisticRegressionModel class
│   │   ├── bge_mlp_model.py                         # BGEMLPModel class
│   │   ├── combine_datasets.py                      # Merges all three datasets
│   │   ├── generate_bge_embeddings.py               # Step 1: embed and cache to disk
│   │   ├── train_logistic_regression.py             # TF-IDF + LR trainer
│   │   ├── train_svm.py                             # TF-IDF + SVM trainer
│   │   ├── train_bge_logistic_regression.py         # BGE + LR trainer
│   │   └── train_bge_mlp.py                         # Step 2: TF MLP trainer
│   ├── gradio_app.py                                # Gradio web interface
│   ├── bge_mlp_sentiment.ipynb                      # Step-by-step BGE + MLP notebook
│   └── __main__.py                                  # FastAPI / Uvicorn entrypoint
├── requirements.txt
└── imdb_sentiment_classifcation.ipynb
```

---

## Datasets

| Dataset | Domain | Rows (raw) | Rows after cleaning |
|---|---|---|---|
| `reviews.csv` | Women's clothing | ~28,000 | ~19,800 |
| `amazon-review/reviews.csv` | Amazon apparel | ~6,300 | ~5,900 |
| `amazon_listing_review.csv` | Amazon fine food | ~568,000 | ~525,800 |
| `combined_reviews.csv` | All three combined | — | ~389,500 (after dedup) |

**Preprocessing applied across all models:**
- Rows with missing review text or rating are dropped
- 3-star (neutral) reviews are excluded — binary classification only
- Ratings 1–2 → **negative (0)**, ratings 4–5 → **positive (1)**

---

## Models

### 1. TF-IDF + Logistic Regression
Converts raw text into sparse TF-IDF vectors (up to 50k features, unigrams + bigrams) and fits a logistic regression classifier. Fast baseline.

### 2. TF-IDF + LinearSVC
Same TF-IDF vectorisation, but uses a Support Vector Machine. `CalibratedClassifierCV` wraps `LinearSVC` to add probability outputs. Often higher accuracy than LR on sparse text.

### 3. BGE Embeddings + Logistic Regression
Uses [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) to encode each review into a 768-dimensional dense vector, then fits logistic regression on those embeddings. Captures semantic meaning and handles synonyms better than TF-IDF.

### 4. BGE Embeddings + TensorFlow MLP
Same BGE embeddings as above, but replaces the logistic regression head with a small TensorFlow neural network (`Dense(128) → Dropout → Dense(64) → Dropout → sigmoid`). Learns non-linear decision boundaries in embedding space.

| Model | Vectorisation | Classifier | Train time (clothing) |
|---|---|---|---|
| TF-IDF + LR | Sparse TF-IDF | Logistic Regression | ~10 sec |
| TF-IDF + SVM | Sparse TF-IDF | LinearSVC | ~20 sec |
| BGE + LR | BGE 768-dim | Logistic Regression | ~2 min |
| BGE + MLP | BGE 768-dim | TensorFlow MLP | ~3 min |

---

## Setup

**Requirements:** Python 3.11+

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`BAAI/bge-base-en-v1.5` (~440 MB) is downloaded automatically on first use and cached in `~/.cache/huggingface`.

---

## Training

### Step 1 — Prepare the combined dataset (optional)

```bash
python -m app.training.combine_datasets
```

Merges all three source datasets, deduplicates, and writes `app/data/combined_reviews.csv`.

### Step 2 — Train models

All trainers accept an optional `--combined` flag to use the merged dataset instead of clothing-only.

```bash
# TF-IDF + Logistic Regression
python -m app.training.train_logistic_regression
python -m app.training.train_logistic_regression --combined

# TF-IDF + SVM
python -m app.training.train_svm
python -m app.training.train_svm --combined

# BGE + Logistic Regression
python -m app.training.train_bge_logistic_regression
python -m app.training.train_bge_logistic_regression --combined

# BGE + MLP (TensorFlow) — two separate steps to avoid PyTorch/TF process conflicts
python -m app.training.generate_bge_embeddings   # generates and caches embeddings
python -m app.training.train_bge_mlp             # trains TF MLP on cached embeddings
```

Each trainer prints accuracy, a classification report, and a confusion matrix on the held-out test set (80/20 stratified split).

---

## Running

### Gradio Web App

```bash
python -m app.gradio_app
```

Opens at `http://localhost:7860`. Enter any review and compare all four models side by side in real time. Models that haven't been trained yet are skipped gracefully.

![Gradio app — four model cards showing positive/negative confidence bars side by side]

### FastAPI REST API

```bash
python -m app
```

Opens at `http://localhost:8080`.
- Swagger UI — `http://localhost:8080/docs`
- ReDoc — `http://localhost:8080/redoc`

Models that haven't been trained yet return `503` until you train them.

---

## API Endpoints

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /analyse` · `POST /analyse/svm` · `POST /analyse/bge` · `POST /analyse/bge-mlp`

All four endpoints share the same request and response schema.

**Request body:**
```json
{
  "comment": "This dress fits perfectly and the fabric feels luxurious.",
  "product_id": "PROD-001",
  "product_name": "Silk Wrap Dress",
  "category": "clothing",
  "rating": 5.0,
  "verified_purchase": true,
  "reviewer_id": "USR-123"
}
```

Only `comment` is required.

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.9312,
  "comment_preview": "This dress fits perfectly and the fabric feels luxurious."
}
```

### Request fields

| Field | Type | Required | Description |
|---|---|---|---|
| `comment` | string | Yes | Review text (1–5000 characters) |
| `product_id` | string | No | Product identifier |
| `product_name` | string | No | Product name (max 255 chars) |
| `category` | enum | No | `electronics` · `clothing` · `home_and_garden` · `sports` · `beauty` · `toys` · `books` · `other` |
| `rating` | float | No | Star rating 1.0–5.0 |
| `verified_purchase` | bool | No | Whether the reviewer is a verified buyer |
| `reviewer_id` | string | No | Anonymised reviewer identifier |

---

## Known Limitations

- **Domain shift** — models trained on clothing data perform worse on electronics, food, etc.
- **Language** — English only; non-English input is not detected or rejected.
- **Sarcasm / irony** — linear decision boundaries cannot capture sentiment reversals.
- **Neutral reviews** — 3-star reviews are excluded from training; genuinely ambiguous input is forced into one class.
- **BGE token limit** — reviews longer than 512 tokens are silently truncated before embedding.
- **Label noise** — labels are derived from star ratings, which sometimes contradict the review text (e.g. 1-star due to shipping, but a positive review body).
- **Class imbalance** — positive reviews dominate all three datasets (64–70%). All models use `class_weight="balanced"` to compensate.
