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
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from app.training.bge_mlp_model import BGEMLPModel
from app.training.plot_utils import save_confusion_matrix, save_classification_report, save_training_curves

PROJECT_ROOT   = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "app" / "data" / "embeddings"
KERAS_MODEL_PATH = PROJECT_ROOT / "app" / "model" / "sentiment_model_bge_mlp.keras"
WRAPPER_PATH     = PROJECT_ROOT / "app" / "model" / "sentiment_model_bge_mlp.joblib"

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
    X_train_bge, X_test_bge, y_train, y_test = load_embeddings(use_combined, limit, balance)
    print(f"  X_train : {X_train_bge.shape}")
    print(f"  X_test  : {X_test_bge.shape}")

    input_dim = X_train_bge.shape[1]

    # Compute class weights to correct for the positive-skew in e-commerce data
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}
    print(f"  Class weights → negative: {weights[0]:.3f}, positive: {weights[1]:.3f}")

    l2 = regularizers.l2(1e-4)
    mlp_model = models.Sequential([
        layers.Input(shape=(input_dim,)), #384
        layers.Dense(128, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_regularizer=l2),
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
        patience=5,
        restore_best_weights=True,
    )

    history = mlp_model.fit(
        X_train_bge,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
    )

    y_prob = mlp_model.predict(X_test_bge).ravel()
    y_pred = (y_prob >= 0.65).astype(int)

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

    print("\nSaving plots …")
    save_training_curves(history, "bge_mlp")
    save_confusion_matrix(y_test, y_pred, "bge_mlp")
    save_classification_report(y_test, y_pred, "bge_mlp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true",   help="Use combined embeddings")
    parser.add_argument("--limit",   type=int, default=None, help="Must match the limit used in generate_bge_embeddings")
    parser.add_argument("--balance", action="store_true",    help="Must match the --balance flag used in generate_bge_embeddings")
    args = parser.parse_args()
    train_model(use_combined=args.combined, limit=args.limit, balance=args.balance)
