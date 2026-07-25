"""
Step 1 of 2 — generate and cache BGE embeddings to disk.
Run this BEFORE train_bge_mlp.py. TensorFlow is never imported here.

Usage
-----
python -m app.training.generate_bge_embeddings             # clothing
python -m app.training.generate_bge_embeddings --combined  # all datasets
"""

import argparse
import os
from pathlib import Path

# Must be set before any HuggingFace / tokenizers import —
# disables Rust tokenizer parallelism which deadlocks on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

PROJECT_ROOT   = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "app" / "data" / "embeddings"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

CLOTHING_CSV  = PROJECT_ROOT / "app" / "data" / "reviews.csv"
COMBINED_CSV  = PROJECT_ROOT / "app" / "data" / "combined_reviews.csv"


def load_clothing():
    df = pd.read_csv(CLOTHING_CSV, usecols=["Review Text", "Rating"]).dropna()
    df = df[df["Rating"] != 3]
    df["label"] = (df["Rating"] >= 4).astype(int)
    return df["Review Text"].astype(str), df["label"]


def load_combined():
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(
            f"{COMBINED_CSV} not found. "
            "Run: python -m app.training.combine_datasets"
        )
    df = pd.read_csv(COMBINED_CSV, usecols=["text", "rating"]).dropna()
    df["label"] = (df["rating"] >= 4).astype(int)
    return df["text"].astype(str), df["label"]


def generate(use_combined: bool = False):
    dataset_tag = "combined" if use_combined else "clothing"
    out_path = EMBEDDINGS_DIR / f"bge_{dataset_tag}.npz"

    X, y = load_combined() if use_combined else load_clothing()
    print(f"Dataset : {dataset_tag} — {len(X):,} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Loading encoder: {EMBEDDING_MODEL_NAME}")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    print(f"Embedding {len(X_train):,} training samples …")
    X_train_emb = encoder.encode(
        X_train.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    print(f"Embedding {len(X_test):,} test samples …")
    X_test_emb = encoder.encode(
        X_test.tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=X_train_emb,
        X_test=X_test_emb,
        y_train=np.array(y_train),
        y_test=np.array(y_test),
    )
    print(f"\nEmbeddings saved to: {out_path}")
    print(f"  X_train : {X_train_emb.shape}")
    print(f"  X_test  : {X_test_emb.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()
    generate(use_combined=args.combined)
