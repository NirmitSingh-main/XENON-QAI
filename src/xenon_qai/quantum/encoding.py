"""
Quantum State Encodings for XENON-QAI.

Implements feature encoding strategies for mapping classical latent representations
z in R^D to N-qubit quantum states |psi(z)>:
- Angle Encoding: Maps D features to D single-qubit rotations (Rx, Ry, Rz)
- Dense Angle Encoding: Maps 2D features to D qubits using 2 Euler angles per qubit
- Amplitude Encoding: Maps 2^N features to N qubits normalized state vector
- IQP (Instantaneous Quantum Polynomial) Encoding: Entangling feature map with non-linear correlations
"""

from typing import Optional, Callable
import numpy as np
import pennylane as qml
import torch


def angle_encoding(features: np.ndarray, wires: list, rotation: str = "Y") -> None:
    """
    Applies Angle Encoding to map features into qubit rotation angles.
    
    Args:
        features: Array of shape (n_qubits,) with normalized values.
        wires: List of qubit wire indices.
        rotation: Single-qubit rotation axis ('X', 'Y', 'Z').
    """
    rot_map = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}
    rot_gate = rot_map.get(rotation.upper(), qml.RY)
    
    for i, wire in enumerate(wires):
        if i < len(features):
            rot_gate(features[i], wires=wire)


def dense_angle_encoding(features: np.ndarray, wires: list) -> None:
    """
    Dense Angle Encoding: Encodes 2 classical features per qubit using (RY, RZ) rotations.
    Maps 2N classical dimensions onto N qubits.
    """
    n_qubits = len(wires)
    for i, wire in enumerate(wires):
        theta = features[2 * i] if 2 * i < len(features) else 0.0
        phi = features[2 * i + 1] if 2 * i + 1 < len(features) else 0.0
        qml.RY(theta, wires=wire)
        qml.RZ(phi, wires=wire)


def iqp_encoding(features: np.ndarray, wires: list, repetitions: int = 2) -> None:
    """
    Instantaneous Quantum Polynomial (IQP) Style Encoding.
    Introduces ZZ-interaction phases for feature-feature interactions:
    exp(i * theta_i * Z_i) and exp(i * theta_i * theta_j * Z_i Z_j).
    """
    n_qubits = len(wires)
    for rep in range(repetitions):
        # Hadamard layer for superposition
        for wire in wires:
            qml.Hadamard(wires=wire)
        
        # Single qubit RZ phase rotations
        for i, wire in enumerate(wires):
            if i < len(features):
                qml.RZ(features[i], wires=wire)
        
        # Two-qubit ZZ entangling phase interactions
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if i < len(features) and j < len(features):
                    phase = features[i] * features[j]
                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.RZ(phase, wires=wires[j])
                    qml.CNOT(wires=[wires[i], wires[j]])


def normalize_features_for_encoding(
    features: np.ndarray,
    method: str = "arctan",
    feature_range: tuple = (-np.pi / 2, np.pi / 2)
) -> np.ndarray:
    """
    Normalizes classical features into appropriate angular ranges for quantum gates.
    """
    if method == "arctan":
        # Maps (-inf, inf) -> (-pi/2, pi/2)
        return np.arctan(features)
    elif method == "minmax":
        min_val, max_val = feature_range
        f_min, f_max = np.min(features), np.max(features)
        if f_max - f_min < 1e-8:
            return np.zeros_like(features)
        scaled = (features - f_min) / (f_max - f_min)
        return scaled * (max_val - min_val) + min_val
    elif method == "tanh":
        return np.tanh(features) * np.pi
    else:
        return features
