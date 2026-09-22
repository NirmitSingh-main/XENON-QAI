"""
Prediction Intervals and Confidence Bounds.
"""

from typing import Tuple
import numpy as np


def compute_prediction_intervals(
    mean: np.ndarray,
    std: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Gaussian prediction interval bounds [lower, upper] for a given confidence level.
    """
    z_score = 1.96 if confidence_level == 0.95 else (2.576 if confidence_level == 0.99 else 1.645)
    lower = np.clip(mean - z_score * std, 0.0, 1.0)
    upper = np.clip(mean + z_score * std, 0.0, 1.0)
    return lower, upper
