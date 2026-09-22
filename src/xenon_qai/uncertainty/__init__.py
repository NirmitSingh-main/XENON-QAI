"""
Uncertainty estimation and calibration modules for XENON-QAI.
"""

from xenon_qai.uncertainty.uncertainty import estimate_mc_dropout_uncertainty
from xenon_qai.uncertainty.calibration import TemperatureScaler, compute_ece
from xenon_qai.uncertainty.confidence import compute_prediction_intervals

__all__ = [
    "estimate_mc_dropout_uncertainty",
    "TemperatureScaler",
    "compute_ece",
    "compute_prediction_intervals",
]
