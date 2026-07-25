import argparse
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from app.training.bge_model import BGELogisticRegressionModel
from app.training.train_logistic_regression import load_and_prepare_data, load_combined_data

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "app" / "model" / "sentiment_model_bge.joblib"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def train_model(use_combined: bool = False):
    X, y = load_combined_data() if use_combined else load_and_prepare_data()
    print(f"Dataset: {'combined' if use_combined else 'clothing'} — {len(X):,} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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

    print("Training LogisticRegression on BGE embeddings …")
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(X_train_emb, y_train)

    y_pred = classifier.predict(X_test_emb)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model = BGELogisticRegressionModel(encoder=encoder, classifier=classifier)

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true", help="Train on combined_reviews.csv instead of clothing only")
    args = parser.parse_args()
    train_model(use_combined=args.combined)
