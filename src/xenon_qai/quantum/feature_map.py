"""
Parameterized Quantum Feature Maps for XENON-QAI.

Maps classical latent representations to Hilbert space via quantum circuits:
- ZZFeatureMap (Hadamard + RZ + CNOT + RZ + CNOT)
- PauliFeatureMap (configurable Pauli strings)
- ChebyshevFeatureMap (orthogonal polynomial quantum embedding)
"""

from typing import List, Optional
import numpy as np
import pennylane as qml


class ZZFeatureMap:
    """
    Second-order Pauli-Z Feature Map (Havlicek et al., Nature 2019).
    Encodes feature products (x_i * x_j) into non-classical quantum correlations.
    """
    
    def __init__(self, n_qubits: int, reps: int = 2, entanglement: str = "linear"):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        
    def __call__(self, x: np.ndarray, wires: Optional[List[int]] = None) -> None:
        if wires is None:
            wires = list(range(self.n_qubits))
            
        for rep in range(self.reps):
            # Superposition layer
            for w in wires:
                qml.Hadamard(wires=w)
            
            # 1st-order phase: phi_i(x) = x_i
            for i, w in enumerate(wires):
                if i < len(x):
                    qml.RZ(2.0 * x[i], wires=w)
            
            # 2nd-order phase: phi_{i,j}(x) = (pi - x_i)(pi - x_j)
            if self.entanglement == "linear":
                pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
            elif self.entanglement == "circular":
                pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
            elif self.entanglement == "full":
                pairs = [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]
            else:
                pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
                
            for i, j in pairs:
                if i < len(x) and j < len(x):
                    phi_ij = 2.0 * (np.pi - x[i]) * (np.pi - x[j])
                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.RZ(phi_ij, wires=wires[j])
                    qml.CNOT(wires=[wires[i], wires[j]])


class ChebyshevFeatureMap:
    """
    Chebyshev Quantum Feature Map (Kyriienko et al., 2021).
    Encodes features using Chebyshev polynomials T_n(x) via cos(n * arccos(x)).
    Provides rich non-linear polynomial basis expansions on single qubits.
    """
    
    def __init__(self, n_qubits: int, max_degree: int = 3):
        self.n_qubits = n_qubits
        self.max_degree = max_degree
        
    def __call__(self, x: np.ndarray, wires: Optional[List[int]] = None) -> None:
        if wires is None:
            wires = list(range(self.n_qubits))
            
        # x is assumed clipped to [-1, 1]
        x_clipped = np.clip(x, -0.999, 0.999)
        angles = np.arccos(x_clipped)
        
        for i, w in enumerate(wires):
            if i < len(angles):
                for degree in range(1, self.max_degree + 1):
                    qml.RY(2.0 * degree * angles[i], wires=w)
                    qml.RZ(1.0 * degree * angles[i], wires=w)
