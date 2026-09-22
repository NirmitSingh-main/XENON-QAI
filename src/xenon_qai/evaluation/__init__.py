"""
Evaluation, metrics, and comparative benchmarking for XENON-QAI.
"""

from xenon_qai.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    signal_efficiency_at_background_rejection,
    format_metrics_table,
)
from xenon_qai.evaluation.roc import compute_roc_curve_data
from xenon_qai.evaluation.precision_recall import compute_pr_curve_data
from xenon_qai.evaluation.efficiency import compute_efficiency_at_rejection_levels
from xenon_qai.evaluation.significance import compute_discovery_significance
from xenon_qai.evaluation.comparison import compare_models, format_comparison_table

__all__ = [
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "signal_efficiency_at_background_rejection",
    "format_metrics_table",
    "compute_roc_curve_data",
    "compute_pr_curve_data",
    "compute_efficiency_at_rejection_levels",
    "compute_discovery_significance",
    "compare_models",
    "format_comparison_table",
]
