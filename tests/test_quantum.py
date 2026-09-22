"""
Unit tests for Quantum ML modules, circuits, hybrid layers, and resource estimations.
"""

import pytest
import numpy as np
import torch
from xenon_qai.quantum import (
    normalize_features_for_encoding,
    estimate_circuit_resources,
    ParameterMatchedClassicalHead,
    QuantumVariationalClassifier,
    QuantumKernelEstimator
)


def test_feature_normalization():
    raw_feats = np.array([-10.0, 0.0, 10.0])
    norm_feats = normalize_features_for_encoding(raw_feats, method="arctan")
    
    assert np.all(norm_feats >= -np.pi / 2)
    assert np.all(norm_feats <= np.pi / 2)


def test_parameter_matched_classical_head():
    in_features = 8
    target_params = 64
    head = ParameterMatchedClassicalHead(in_features=in_features, target_param_count=target_params)
    param_count = head.count_parameters()
    
    # Check that parameter count is within reasonable range of target
    assert abs(param_count - target_params) < 30
    
    x = torch.randn(4, in_features)
    out = head(x)
    assert out.shape == (4, 1)


def test_quantum_variational_classifier_forward():
    n_qubits = 4
    n_layers = 2
    in_features = 8
    batch_size = 3
    
    qnn = QuantumVariationalClassifier(
        n_qubits=n_qubits,
        n_layers=n_layers,
        in_features=in_features,
        num_classes=1
    )
    
    x = torch.randn(batch_size, in_features)
    logits = qnn(x)
    
    assert logits.shape == (batch_size, 1)
    assert not torch.isnan(logits).any()


def test_quantum_kernel_estimator():
    n_qubits = 2
    estimator = QuantumKernelEstimator(n_qubits=n_qubits, reps=1)
    
    x1 = np.array([0.5, 0.5])
    x2 = np.array([0.5, 0.5])
    x3 = np.array([1.5, 1.5])
    
    # Overlap with identical point must be 1.0
    self_overlap = estimator.compute_kernel_element(x1, x2)
    assert np.isclose(self_overlap, 1.0, atol=1e-3)
    
    # Overlap with distinct point should be <= 1.0
    cross_overlap = estimator.compute_kernel_element(x1, x3)
    assert 0.0 <= cross_overlap <= 1.0


def test_circuit_resource_estimation():
    res = estimate_circuit_resources(n_qubits=4, n_layers=3, ansatz_type="hardware_efficient")
    
    assert res["n_qubits"] == 4
    assert res["n_layers"] == 3
    assert res["trainable_parameters"] == 24  # 4 * 3 * 2
    assert res["two_qubit_cnot_gates"] > 0
    assert res["estimated_circuit_depth"] > 0
