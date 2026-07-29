"""
Train Logistic Regression on pre-computed BGE embeddings.
Run generate_bge_embeddings.py first.

Usage
-----
python -m app.training.train_bge_logistic_regression --combined --balance
python -m app.training.train_bge_logistic_regression --combined --limit 50000
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from app.training.bge_model import BGELogisticRegressionModel
from app.training.plot_utils import save_confusion_matrix, save_classification_report

PROJECT_ROOT     = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR   = PROJECT_ROOT / "app" / "data" / "embeddings"
MODEL_PATH       = PROJECT_ROOT / "app" / "model" / "sentiment_model_bge.joblib"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embeddings(use_combined: bool, limit: int | None = None, balance: bool = False) -> tuple:
    dataset_tag = "combined" if use_combined else "clothing"
    tag_suffix  = f"_limit{limit}" if limit else ""
    if balance:
        tag_suffix += "_balanced"
    path = EMBEDDINGS_DIR / f"bge_{dataset_tag}{tag_suffix}.npz"
    if not path.exists():
        cmd = "python -m app.training.generate_bge_embeddings"
        if use_combined: cmd += " --combined"
        if limit:        cmd += f" --limit {limit}"
        if balance:      cmd += " --balance"
        available = sorted(EMBEDDINGS_DIR.glob("*.npz"))
        hint = "\nAvailable embedding files:\n" + "\n".join(f"  {f.name}" for f in available) if available else "\nNo embedding files found in embeddings/."
        raise FileNotFoundError(f"Embeddings not found at {path}.\nRun first: {cmd}{hint}")
    data = np.load(path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def train_model(use_combined: bool = False, limit: int | None = None, balance: bool = False):
    tag = ("combined" if use_combined else "clothing") + (f" limit={limit}" if limit else "") + (" balanced" if balance else "")
    print(f"Loading pre-computed embeddings ({tag}) …")
    X_train_emb, X_test_emb, y_train, y_test = load_embeddings(use_combined, limit, balance)
    print(f"  X_train : {X_train_emb.shape}")
    print(f"  X_test  : {X_test_emb.shape}")

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    print(f"  Labels  : negative={neg:,}  positive={pos:,}")

    print("\nTraining LogisticRegression on BGE embeddings …")
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(X_train_emb, y_train)

    y_pred = classifier.predict(X_test_emb)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Build wrapper with lazy encoder (loaded only at inference time)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    model = BGELogisticRegressionModel(encoder=encoder, classifier=classifier)

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    print("\nSaving plots …")
    save_confusion_matrix(y_test, y_pred, "bge_lr")
    save_classification_report(y_test, y_pred, "bge_lr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true",   help="Use combined embeddings")
    parser.add_argument("--limit",   type=int, default=None, help="Must match the limit used in generate_bge_embeddings")
    parser.add_argument("--balance", action="store_true",    help="Must match the --balance flag used in generate_bge_embeddings")
    args = parser.parse_args()
    train_model(use_combined=args.combined, limit=args.limit, balance=args.balance)
