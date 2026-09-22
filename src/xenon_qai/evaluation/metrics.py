"""
Physics-Informed Evaluation Metrics for XENON-QAI.

Implements metrics required per dataanalysis.txt Section 2:
- ROC-AUC (invariant to class ratio)
- Precision-Recall AUC
- Signal Efficiency at fixed Background Rejection (99.9%, 99.99%)
- Multi-class confusion matrix
- Per-class metrics
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes all binary classification metrics for signal vs background.
    
    Args:
        y_true: Ground truth binary labels (0=background, 1=signal).
        y_score: Predicted probability/score for the positive class.
        y_pred: Optional hard predictions. If None, derived from y_score > threshold.
        threshold: Decision threshold for hard predictions.
    
    Returns:
        Dictionary of metric_name -> value.
    """
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    
    metrics: Dict[str, float] = {}
    
    # ROC-AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = 0.0
    
    # PR-AUC
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["pr_auc"] = 0.0
    
    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    # F1
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0.0))
    
    # Signal Efficiency at fixed Background Rejection
    metrics["sig_eff_at_99.9_bkg_rej"] = signal_efficiency_at_background_rejection(
        y_true, y_score, target_bkg_rejection=0.999
    )
    metrics["sig_eff_at_99.99_bkg_rej"] = signal_efficiency_at_background_rejection(
        y_true, y_score, target_bkg_rejection=0.9999
    )
    
    return metrics


def signal_efficiency_at_background_rejection(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_bkg_rejection: float = 0.999
) -> float:
    """
    Computes the signal efficiency (true positive rate) at a specific background rejection level.
    
    Background Rejection = 1 - FPR (False Positive Rate).
    This is the key metric in rare-event physics searches.
    
    Args:
        y_true: Binary ground truth (0=background, 1=signal).
        y_score: Predicted scores.
        target_bkg_rejection: Target background rejection (e.g. 0.999 = 99.9%).
    
    Returns:
        Signal efficiency (TPR) at the given background rejection, or 0.0 if not achievable.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    target_fpr = 1.0 - target_bkg_rejection
    
    # Find the TPR at or below the target FPR
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0
    return float(tpr[valid_idx[-1]])


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes multi-class classification metrics.
    """
    n_classes = len(set(y_true))
    
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0.0)),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Per-class report
    if class_names is not None:
        report = classification_report(
            y_true, y_pred, target_names=class_names,
            output_dict=True, zero_division=0.0
        )
        metrics["per_class_report"] = report
    
    return metrics


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Formats a metrics dictionary as a printable table.
    """
    lines = ["+" + "-" * 40 + "+" + "-" * 15 + "+"]
    lines.append(f"| {'Metric':<38} | {'Value':>13} |")
    lines.append("+" + "-" * 40 + "+" + "-" * 15 + "+")
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"| {key:<38} | {val:>13.6f} |")
        else:
            lines.append(f"| {key:<38} | {str(val):>13} |")
    lines.append("+" + "-" * 40 + "+" + "-" * 15 + "+")
    return "\n".join(lines)
