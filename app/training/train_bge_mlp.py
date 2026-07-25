"""
Step 2 of 2 — train TensorFlow MLP on pre-computed BGE embeddings.
Run generate_bge_embeddings.py first. sentence-transformers is never
imported here so there is no PyTorch/TensorFlow mutex conflict.

Usage
-----
python -m app.training.train_bge_mlp                # clothing embeddings
python -m app.training.train_bge_mlp --combined     # combined embeddings
"""

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from app.training.bge_mlp_model import BGEMLPModel

PROJECT_ROOT   = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "app" / "data" / "embeddings"
KERAS_MODEL_PATH = PROJECT_ROOT / "app" / "model" / "sentiment_model_bge_mlp.keras"
WRAPPER_PATH     = PROJECT_ROOT / "app" / "model" / "sentiment_model_bge_mlp.joblib"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_embeddings(use_combined: bool) -> tuple:
    dataset_tag = "combined" if use_combined else "clothing"
    path = EMBEDDINGS_DIR / f"bge_{dataset_tag}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {path}.\n"
            f"Run first: python -m app.training.generate_bge_embeddings"
            + (" --combined" if use_combined else "")
        )
    data = np.load(path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def train_model(use_combined: bool = False):
    print(f"Loading pre-computed embeddings ({'combined' if use_combined else 'clothing'}) …")
    X_train_bge, X_test_bge, y_train, y_test = load_embeddings(use_combined)
    print(f"  X_train : {X_train_bge.shape}")
    print(f"  X_test  : {X_test_bge.shape}")

    input_dim = X_train_bge.shape[1]

    mlp_model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    mlp_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    mlp_model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = mlp_model.fit(
        X_train_bge,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
    )

    y_prob = mlp_model.predict(X_test_bge).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(KERAS_MODEL_PATH.parent, exist_ok=True)
    mlp_model.save(KERAS_MODEL_PATH)
    print(f"\nKeras model saved to : {KERAS_MODEL_PATH}")

    wrapper = BGEMLPModel(
        encoder_name=EMBEDDING_MODEL_NAME,
        keras_model_path=str(KERAS_MODEL_PATH),
    )
    joblib.dump(wrapper, WRAPPER_PATH)
    print(f"Wrapper saved to     : {WRAPPER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true", help="Use combined embeddings")
    args = parser.parse_args()
    train_model(use_combined=args.combined)
