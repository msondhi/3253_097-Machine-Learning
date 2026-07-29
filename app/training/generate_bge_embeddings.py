"""
Step 1 of 2 — generate and cache BGE embeddings to disk.
Run this BEFORE train_bge_mlp.py. TensorFlow is never imported here.

Usage
-----
# Full clothing dataset
python -m app.training.generate_bge_embeddings

# Full combined dataset
python -m app.training.generate_bge_embeddings --combined

# Combined with 50 000 samples per source
python -m app.training.generate_bge_embeddings --combined --limit 50000

# Balanced: equal negative and positive samples globally
python -m app.training.generate_bge_embeddings --combined --limit 50000 --balance

# Control parallelism (default 4 threads)
python -m app.training.generate_bge_embeddings --combined --limit 5000 --workers 6

Parallel strategy
-----------------
Data is split into N shards. Each shard runs in its own thread with its own
encoder instance — no shared state, no fork(), no mutex issues on macOS.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT   = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "app" / "data" / "embeddings"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CLOTHING_CSV = PROJECT_ROOT / "app" / "data" / "reviews.csv"
COMBINED_CSV = PROJECT_ROOT / "app" / "data" / "combined_reviews.csv"


def load_clothing(limit: int | None = None):
    df = pd.read_csv(CLOTHING_CSV, usecols=["Review Text", "Rating"]).dropna()
    df = df[df["Rating"] != 3]
    df["label"] = (df["Rating"] >= 4).astype(int)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    return df["Review Text"].astype(str), df["label"]


def load_combined(limit_per_source: int | None = None, balance: bool = False):
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(
            f"{COMBINED_CSV} not found. "
            "Run: python -m app.training.combine_datasets"
        )
    df = pd.read_csv(COMBINED_CSV, usecols=["text", "rating", "source"]).dropna()
    df["label"] = (df["rating"] >= 4).astype(int)

    if limit_per_source:
        parts = []
        for source, group in df.groupby("source"):
            n = min(limit_per_source, len(group))
            parts.append(group.sample(n=n, random_state=42))
            print(f"  {source:<12} → {n:,} samples (of {len(group):,})")
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
    else:
        for source, group in df.groupby("source"):
            print(f"  {source:<12} → {len(group):,} samples")

    if balance:
        neg = df[df["label"] == 0]
        pos = df[df["label"] == 1]
        n = min(len(neg), len(pos))
        df = pd.concat([
            neg.sample(n=n, random_state=42),
            pos.sample(n=n, random_state=42),
        ]).sample(frac=1, random_state=42)
        print(f"\n  Balanced → {n:,} negative + {n:,} positive = {len(df):,} total")

    return df["text"].astype(str), df["label"]


def _encode_shard(shard_texts: list, shard_id: int) -> np.ndarray:
    enc = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return enc.encode(
        shard_texts,
        normalize_embeddings=True,
        batch_size=512,
        show_progress_bar=False,
    )


def parallel_encode(texts: list, n_workers: int, label: str) -> np.ndarray:
    # Clamp workers — no point having more shards than 5k samples each
    n_workers = min(n_workers, max(1, len(texts) // 5_000))
    shards    = np.array_split(texts, n_workers)

    print(f"  {label}: {len(texts):,} texts  |  {n_workers} threads  |  batch_size=512")
    results = [None] * n_workers

    with tqdm(total=len(texts), unit="review", ncols=80) as bar:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_encode_shard, shard.tolist(), idx): (idx, len(shard))
                for idx, shard in enumerate(shards)
            }
            for future in as_completed(futures):
                idx, size = futures[future]
                results[idx] = future.result()
                bar.update(size)

    return np.vstack(results)


def generate(use_combined: bool = False, limit: int | None = None, n_workers: int = 4, balance: bool = False):
    dataset_tag = "combined" if use_combined else "clothing"
    tag_suffix  = f"_limit{limit}" if limit else ""
    if balance:
        tag_suffix += "_balanced"
    out_path    = EMBEDDINGS_DIR / f"bge_{dataset_tag}{tag_suffix}.npz"

    print(f"Dataset : {dataset_tag}" + (f"  (limit {limit:,} per source)" if limit else "") + ("  [balanced]" if balance else ""))

    if use_combined:
        X, y = load_combined(limit_per_source=limit, balance=balance)
    else:
        X, y = load_clothing(limit=limit)

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    print(f"Total   : {len(X):,} samples  |  negative={neg:,} ({neg/len(X)*100:.1f}%)  positive={pos:,} ({pos/len(X)*100:.1f}%)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_emb = parallel_encode(X_train.tolist(), n_workers, label="train")
    X_test_emb  = parallel_encode(X_test.tolist(),  max(1, n_workers // 2), label="test")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=X_train_emb,
        X_test=X_test_emb,
        y_train=np.array(y_train),
        y_test=np.array(y_test),
    )
    print(f"\nSaved → {out_path}")
    print(f"  X_train : {X_train_emb.shape}")
    print(f"  X_test  : {X_test_emb.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true",  help="Use combined_reviews.csv")
    parser.add_argument("--limit",   type=int, default=None, help="Max samples per source dataset")
    parser.add_argument("--workers", type=int, default=4,    help="Parallel threads (default: 4)")
    parser.add_argument("--balance", action="store_true",    help="Downsample majority class to match minority (50/50 split)")
    args = parser.parse_args()
    generate(use_combined=args.combined, limit=args.limit, n_workers=args.workers, balance=args.balance)
