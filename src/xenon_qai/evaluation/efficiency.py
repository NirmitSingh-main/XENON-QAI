"""
Signal Efficiency and Background Rejection Benchmarking.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_curve


def compute_efficiency_at_rejection_levels(
    y_true: np.ndarray,
    y_score: np.ndarray,
    rejection_levels: Tuple[float, ...] = (0.90, 0.99, 0.999, 0.9999)
) -> Dict[str, float]:
    """
    Computes signal efficiency (True Positive Rate) at specified background rejection levels.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    results = {}
    
    for rej in rejection_levels:
        target_fpr = 1.0 - rej
        valid_indices = np.where(fpr <= target_fpr)[0]
        if len(valid_indices) > 0:
            eff = float(tpr[valid_indices[-1]])
        else:
            eff = 0.0
        key = f"signal_eff_at_{rej*100:.2f}%_bkg_rej"
        results[key] = eff
        
    return results
