"""
Baseline models for XENON-QAI.

Provides standard classical machine learning baselines for physics feature benchmark:
- XenonLogisticRegression
- XenonRandomForest
- XenonXGBoost
- XenonMLP
"""

from xenon_qai.models.baselines.logistic import XenonLogisticRegression
from xenon_qai.models.baselines.random_forest import XenonRandomForest
from xenon_qai.models.baselines.xgboost import XenonXGBoost
from xenon_qai.models.baselines.mlp import XenonMLP, XenonMLPModule

__all__ = [
    "XenonLogisticRegression",
    "XenonRandomForest",
    "XenonXGBoost",
    "XenonMLP",
    "XenonMLPModule",
]
