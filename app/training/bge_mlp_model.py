import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np


class BGEMLPModel:
    """BGE encoder (frozen) + TensorFlow/Keras MLP classifier.

    Stores only strings so joblib can serialise this wrapper cleanly.
    The encoder and Keras model are loaded lazily on first inference call.
    """

    def __init__(self, encoder_name: str, keras_model_path: str):
        self.encoder_name = encoder_name
        self.keras_model_path = keras_model_path
        self._encoder = None
        self._keras_model = None

    def _load(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name, device="cpu")
        if self._keras_model is None:
            import tensorflow as tf
            self._keras_model = tf.keras.models.load_model(self.keras_model_path)
            # Validate embedding dimension matches what the Keras model expects
            keras_input_dim = self._keras_model.input_shape[-1]
            encoder_dim = self._encoder.get_sentence_embedding_dimension()
            if keras_input_dim != encoder_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: encoder '{self.encoder_name}' produces "
                    f"{encoder_dim}-dim vectors but the Keras model expects {keras_input_dim}-dim. "
                    f"Delete the model files and retrain with a consistent encoder."
                )

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        return self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    THRESHOLD = 0.65

    def predict(self, texts: list[str]) -> np.ndarray:
        self._load()
        probs = self._keras_model.predict(self._embed(texts), verbose=0).ravel()
        return (probs >= self.THRESHOLD).astype(int)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self._load()
        probs = self._keras_model.predict(self._embed(texts), verbose=0).ravel()
        return np.column_stack([1 - probs, probs])
