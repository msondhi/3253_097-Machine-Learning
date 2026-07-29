import argparse
import os
from pathlib import Path

import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from app.training.plot_utils import save_confusion_matrix, save_classification_report

from app.training.train_logistic_regression import load_and_prepare_data, load_combined_data, DATA_PATH

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "app" / "model" / "sentiment_model_svm.joblib"


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

    # LinearSVC is fast on sparse TF-IDF data.
    # CalibratedClassifierCV wraps it to add predict_proba support.
    svm = LinearSVC(
        C=0.5,
        max_iter=20000,
        class_weight="balanced",
        random_state=42,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            max_features=50000,
        )),
        ("classifier", CalibratedClassifierCV(svm, cv=5)),
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
    save_confusion_matrix(y_test, y_pred, "tfidf_svm")
    save_classification_report(y_test, y_pred, "tfidf_svm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true", help="Train on combined_reviews.csv instead of clothing only")
    args = parser.parse_args()
    train_model(use_combined=args.combined)
