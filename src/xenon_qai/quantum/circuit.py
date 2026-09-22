"""
Variational Quantum Circuits and Parameterized Ansatzes for XENON-QAI.

Implements variational circuits:
- StronglyEntanglingLayers: High expressibility using 3-axis Euler rotations + multi-range CNOTs
- HardwareEfficientAnsatz: Alternating single-qubit rotations (RY/RZ) and linear nearest-neighbor CNOTs
- RealAmplitudesAnsatz: Real-amplitude circuit using RY and CNOTs (pure real state vector)
- BasicEntanglerAnsatz: Lightweight standard benchmark
"""

from typing import List, Optional, Tuple
import numpy as np
import pennylane as qml


def hardware_efficient_ansatz(
    weights: np.ndarray,
    wires: List[int],
    entanglement: str = "linear"
) -> None:
    """
    Hardware-efficient ansatz with alternating single-qubit RY/RZ and CNOT gates.
    
    Args:
        weights: Shape (n_layers, n_qubits, 2)
        wires: List of wire indices
        entanglement: 'linear', 'circular', or 'all-to-all'
    """
    n_layers, n_qubits, _ = weights.shape
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for i, wire in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=wire)
            qml.RZ(weights[layer, i, 1], wires=wire)
        
        # Entangling layer
        if entanglement == "linear":
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        elif entanglement == "circular":
            for i in range(n_qubits):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])
        elif entanglement == "all-to-all":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[wires[i], wires[j]])


def real_amplitudes_ansatz(
    weights: np.ndarray,
    wires: List[int]
) -> None:
    """
    RealAmplitudes ansatz (rotations restricted to Y-axis for real wavefunctions).
    
    Args:
        weights: Shape (n_layers + 1, n_qubits)
    """
    n_layers_plus_1, n_qubits = weights.shape
    n_layers = n_layers_plus_1 - 1
    
    # Initial rotation layer
    for i, wire in enumerate(wires):
        qml.RY(weights[0, i], wires=wire)
        
    for layer in range(n_layers):
        # Linear CNOT ladder
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            
        # Subsequent RY rotation
        for i, wire in enumerate(wires):
            qml.RY(weights[layer + 1, i], wires=wire)


def get_ansatz_weight_shape(
    ansatz_type: str,
    n_qubits: int,
    n_layers: int
) -> Tuple[int, ...]:
    """
    Returns required parameter weight tensor shape for the given ansatz.
    """
    if ansatz_type == "hardware_efficient":
        return (n_layers, n_qubits, 2)
    elif ansatz_type == "real_amplitudes":
        return (n_layers + 1, n_qubits)
    elif ansatz_type == "strongly_entangling":
        return (n_layers, n_qubits, 3)
    elif ansatz_type == "basic_entangler":
        return (n_layers, n_qubits)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
