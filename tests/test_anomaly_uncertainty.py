"""
Unit tests for anomaly detection, calibration, uncertainty, and explainability.
"""

import pytest
import numpy as np
import torch
from xenon_qai.anomaly import XenonAnomalyScorer, CandidateRanker
from xenon_qai.uncertainty import compute_ece, compute_prediction_intervals
from xenon_qai.explainability import compute_integrated_gradients, check_s2_s1_monotonicity
from xenon_qai.models.cnn import XenonMultimodalCNN
from xenon_qai.representation import LatentReducer
from xenon_qai.data.types import MULTIMODAL_DIM, PHYSICS_FEATURE_DIM


def test_anomaly_scorer():
    bg_data = np.random.randn(50, 16)
    test_data = np.random.randn(10, 16)
    
    scorer = XenonAnomalyScorer()
    scorer.fit(bg_data)
    scores = scorer.score(test_data)
    
    assert scores.shape == (10,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_candidate_ranker():
    event_ids = [f"ev_{i}" for i in range(10)]
    c_scores = np.linspace(0.1, 0.9, 10)
    a_scores = np.linspace(0.2, 0.8, 10)
    
    ranker = CandidateRanker()
    df = ranker.rank_events(event_ids, c_scores, a_scores, top_k=5)
    
    assert len(df) == 5
    assert df["rank"].iloc[0] == 1
    assert df["priority_score"].iloc[0] >= df["priority_score"].iloc[-1]


def test_ece_computation():
    probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0, 0])
    ece = compute_ece(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_prediction_intervals():
    mean = np.array([0.5, 0.8])
    std = np.array([0.1, 0.05])
    lower, upper = compute_prediction_intervals(mean, std)
    assert np.all(lower <= upper)


def test_integrated_gradients():
    model = XenonMultimodalCNN(latent_dim=16, num_classes=1, include_physics_features=False)
    x = torch.randn(2, MULTIMODAL_DIM)
    
    res = compute_integrated_gradients(model, x, steps=5)
    assert "attributions" in res
    assert res["attributions"].shape == (2, MULTIMODAL_DIM)


def test_latent_reducer():
    latents = np.random.randn(30, 16)
    reducer = LatentReducer(method="pca", n_components=2)
    embedding = reducer.fit_transform(latents)
    assert embedding.shape == (30, 2)
