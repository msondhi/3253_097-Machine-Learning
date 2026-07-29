import argparse
import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from app.training.plot_utils import save_confusion_matrix, save_classification_report


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "app" / "data" / "reviews.csv"
COMBINED_DATA_PATH = PROJECT_ROOT / "app" / "data" / "combined_reviews.csv"
MODEL_PATH = PROJECT_ROOT / "app" / "model" / "sentiment_model.joblib"

TEXT_COL = "Review Text"
RATING_COL = "Rating"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = df[[TEXT_COL, RATING_COL]].dropna()
    df = df[df[RATING_COL] != 3]
    df["label"] = df[RATING_COL].apply(lambda x: 1 if x >= 4 else 0)
    X = df[TEXT_COL].astype(str)
    y = df["label"]
    return X, y


def load_combined_data():
    """Load the combined dataset produced by combine_datasets.py.
    Expects columns: text, rating (1,2,4,5 — no 3-stars)."""
    if not COMBINED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Combined dataset not found at {COMBINED_DATA_PATH}. "
            "Run: python -m app.training.combine_datasets"
        )
    df = pd.read_csv(COMBINED_DATA_PATH, usecols=["text", "rating"])
    df = df.dropna(subset=["text", "rating"])
    df["label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    X = df["text"].astype(str)
    y = df["label"]
    return X, y


def train_model(use_combined: bool = False):
    X, y = load_combined_data() if use_combined else load_and_prepare_data()
    print(f"Dataset: {'combined' if use_combined else 'clothing'} — {len(X):,} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            max_features=50000
        )),
        ("classifier", LogisticRegression(
            C=0.3,
            max_iter=10000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    print("\nSaving plots …")
    save_confusion_matrix(y_test, y_pred, "tfidf_lr")
    save_classification_report(y_test, y_pred, "tfidf_lr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true", help="Train on combined_reviews.csv instead of clothing only")
    args = parser.parse_args()
    train_model(use_combined=args.combined)
