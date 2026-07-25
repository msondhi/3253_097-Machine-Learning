import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


class BGELogisticRegressionModel:
    def __init__(self, encoder: SentenceTransformer, classifier: LogisticRegression):
        self.encoder = encoder
        self.classifier = classifier

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict(self._embed(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict_proba(self._embed(texts))
