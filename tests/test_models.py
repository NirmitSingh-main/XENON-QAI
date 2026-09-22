"""
Unit and integration tests for XENON-QAI classical baselines and deep learning models.
"""

import pytest
import numpy as np
import torch
from xenon_qai.models.baselines import (
    XenonLogisticRegression,
    XenonRandomForest,
    XenonXGBoost,
    XenonMLP
)
from xenon_qai.models.cnn import (
    XenonMultimodalCNN,
    S1WaveformCNN,
    S2WaveformCNN,
    PMTSpatialMLP,
    PhysicsFeatureMLP
)
from xenon_qai.data.types import MULTIMODAL_DIM, PHYSICS_FEATURE_DIM


@pytest.fixture
def synthetic_data():
    """Generates synthetic tabular and multimodal batches."""
    np.random.seed(42)
    n_samples = 100
    
    # 13D physics features
    x_phys = np.random.randn(n_samples, PHYSICS_FEATURE_DIM).astype(np.float32)
    
    # 908D multimodal vectors
    x_multi = np.random.randn(n_samples, MULTIMODAL_DIM).astype(np.float32)
    
    # Binary targets (0: background, 1: signal)
    y_bin = np.random.randint(0, 2, size=n_samples)
    
    # Multiclass targets (0..11)
    y_multi = np.random.randint(0, 12, size=n_samples)
    
    return {
        "x_phys": x_phys,
        "x_multi": x_multi,
        "y_bin": y_bin,
        "y_multi": y_multi
    }


def test_logistic_regression(synthetic_data):
    X = synthetic_data["x_phys"]
    y = synthetic_data["y_bin"]
    
    model = XenonLogisticRegression(max_iter=100)
    model.fit(X[:80], y[:80])
    assert model.is_fitted
    
    metrics = model.evaluate(X[80:], y[80:])
    assert "roc_auc" in metrics
    assert "accuracy" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_random_forest(synthetic_data):
    X = synthetic_data["x_phys"]
    y = synthetic_data["y_bin"]
    
    model = XenonRandomForest(n_estimators=20)
    model.fit(X[:80], y[:80])
    assert model.is_fitted
    
    importances = model.feature_importances()
    assert len(importances) == PHYSICS_FEATURE_DIM
    assert np.isclose(np.sum(importances), 1.0)
    
    metrics = model.evaluate(X[80:], y[80:])
    assert "roc_auc" in metrics


def test_xgboost(synthetic_data):
    X = synthetic_data["x_phys"]
    y = synthetic_data["y_bin"]
    
    model = XenonXGBoost(n_estimators=15, max_depth=3)
    model.fit(X[:80], y[:80])
    assert model.is_fitted
    
    metrics = model.evaluate(X[80:], y[80:])
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_mlp_baseline(synthetic_data):
    X = synthetic_data["x_phys"]
    y = synthetic_data["y_bin"]
    
    model = XenonMLP(input_dim=PHYSICS_FEATURE_DIM, epochs=5, batch_size=32)
    model.fit(X[:80], y[:80])
    assert model.is_fitted
    
    latents = model.extract_latent(X[:10])
    assert latents.shape == (10, 32)
    
    metrics = model.evaluate(X[80:], y[80:])
    assert "roc_auc" in metrics


def test_multimodal_cnn_forward():
    batch_size = 8
    x_multi = torch.randn(batch_size, MULTIMODAL_DIM)
    x_phys = torch.randn(batch_size, PHYSICS_FEATURE_DIM)
    
    model = XenonMultimodalCNN(latent_dim=16, num_classes=1, include_physics_features=True)
    logits, z = model(x_multi, x_phys)
    
    assert logits.shape == (batch_size, 1)
    assert z.shape == (batch_size, 16)
    
    scores = model.predict_candidate_score(x_multi, x_phys)
    assert scores.shape == (batch_size,)
    assert torch.all((scores >= 0.0) & (scores <= 1.0))
