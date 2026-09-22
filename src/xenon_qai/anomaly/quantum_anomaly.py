"""
Quantum Anomaly Detection using Quantum Fidelity & Compression.
"""

from typing import Optional
import numpy as np
import pennylane as qml
from xenon_qai.quantum.feature_map import ZZFeatureMap


class QuantumFidelityAnomalyDetector:
    """
    Computes anomaly scores based on quantum state overlap fidelity
    with an empirical background reference state.
    """
    
    def __init__(self, n_qubits: int = 4, reps: int = 2):
        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_map = ZZFeatureMap(n_qubits=n_qubits, reps=reps)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.reference_centroid: Optional[np.ndarray] = None
        
        @qml.qnode(self.dev)
        def overlap_circuit(x1, x2):
            self.feature_map(x1)
            qml.adjoint(self.feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))
            
        self._circuit = overlap_circuit
        
    def fit(self, X_background: np.ndarray) -> "QuantumFidelityAnomalyDetector":
        """Computes background reference centroid."""
        self.reference_centroid = np.mean(X_background, axis=0)
        return self
    
    def score_sample(self, x: np.ndarray) -> float:
        """Anomaly score = 1.0 - fidelity with background reference."""
        if self.reference_centroid is None:
            raise ValueError("Detector must be fitted before scoring.")
        probs = self._circuit(x, self.reference_centroid)
        fidelity = float(probs[0])
        return 1.0 - fidelity
    
    def score(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.score_sample(x) for x in X])
