"""
Anomaly Scoring and Compatibility Metrics for XENON-QAI.

Computes anomaly scores for detecting rare or unmodeled physics events:
- Distance-to-background manifold in latent space
- Reconstruction error from Autoencoder
- Isolation Forest anomaly scores
"""

from typing import Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


class XenonAnomalyScorer:
    """
    Computes calibrated anomaly scores in [0, 1] using background reference events.
    Score ~ 1 means highly anomalous (potential dark matter candidate or novel physics).
    Score ~ 0 means typical background.
    """
    
    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.is_fitted: bool = False
        
    def fit(self, X_background: np.ndarray) -> "XenonAnomalyScorer":
        """Fits on known background representations."""
        self.model.fit(X_background)
        raw_scores = -self.model.score_samples(X_background)
        self.scaler.fit(raw_scores.reshape(-1, 1))
        self.is_fitted = True
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns normalized anomaly score in [0, 1]."""
        raw_scores = -self.model.score_samples(X)
        norm_scores = self.scaler.transform(raw_scores.reshape(-1, 1)).flatten()
        return np.clip(norm_scores, 0.0, 1.0)
