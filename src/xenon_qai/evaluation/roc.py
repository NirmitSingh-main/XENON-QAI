"""
ROC Curve and Discrimination Analysis for XENON-QAI.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def compute_roc_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """
    Computes false positive rates, true positive rates, and AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc)
    }
