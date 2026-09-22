"""
Model architectures for XENON-QAI.

Provides:
- Baselines (Logistic Regression, Random Forest, XGBoost, MLP)
- Deep CNN Multi-Modal Architecture (S1/S2/PMT/Physics)
"""

from xenon_qai.models.baselines import (
    XenonLogisticRegression,
    XenonRandomForest,
    XenonXGBoost,
    XenonMLP,
    XenonMLPModule,
)
from xenon_qai.models.cnn import (
    XenonMultimodalCNN,
    S1WaveformCNN,
    S2WaveformCNN,
    PMTSpatialMLP,
    PhysicsFeatureMLP,
    train_xenon_cnn,
    evaluate_model,
)

__all__ = [
    "XenonLogisticRegression",
    "XenonRandomForest",
    "XenonXGBoost",
    "XenonMLP",
    "XenonMLPModule",
    "XenonMultimodalCNN",
    "S1WaveformCNN",
    "S2WaveformCNN",
    "PMTSpatialMLP",
    "PhysicsFeatureMLP",
    "train_xenon_cnn",
    "evaluate_model",
]
