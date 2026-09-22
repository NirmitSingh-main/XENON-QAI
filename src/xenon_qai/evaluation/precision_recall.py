"""
Precision-Recall Curve Analysis for Rare-Event Search.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def compute_pr_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """
    Computes Precision-Recall curve and Average Precision (PR-AUC).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "average_precision": float(ap)
    }
