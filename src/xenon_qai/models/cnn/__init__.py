"""
Multi-Modal Convolutional Neural Network package for XENON-QAI.
"""

from xenon_qai.models.cnn.model import (
    XenonMultimodalCNN,
    S1WaveformCNN,
    S2WaveformCNN,
    PMTSpatialMLP,
    PhysicsFeatureMLP,
)
from xenon_qai.models.cnn.train import train_xenon_cnn, evaluate_model

__all__ = [
    "XenonMultimodalCNN",
    "S1WaveformCNN",
    "S2WaveformCNN",
    "PMTSpatialMLP",
    "PhysicsFeatureMLP",
    "train_xenon_cnn",
    "evaluate_model",
]
