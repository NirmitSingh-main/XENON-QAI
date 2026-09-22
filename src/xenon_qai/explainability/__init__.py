"""
Explainability and physics validation modules for XENON-QAI.
"""

from xenon_qai.explainability.feature_importance import compute_integrated_gradients
from xenon_qai.explainability.physics_checks import check_s2_s1_monotonicity

__all__ = [
    "compute_integrated_gradients",
    "check_s2_s1_monotonicity",
]
