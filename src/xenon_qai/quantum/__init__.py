"""
Quantum Machine Learning modules for XENON-QAI.
"""

from xenon_qai.quantum.encoding import (
    angle_encoding,
    dense_angle_encoding,
    iqp_encoding,
    normalize_features_for_encoding,
)
from xenon_qai.quantum.feature_map import ZZFeatureMap, ChebyshevFeatureMap
from xenon_qai.quantum.circuit import (
    hardware_efficient_ansatz,
    real_amplitudes_ansatz,
    get_ansatz_weight_shape,
)
from xenon_qai.quantum.classical_head import (
    ParameterMatchedClassicalHead,
    ClassicalMLPHead,
)
from xenon_qai.quantum.qnn import (
    QuantumVariationalClassifier,
    QuantumHybridClassifier,
    train_quantum_classifier,
)
from xenon_qai.quantum.qkernel import QuantumKernelEstimator, XenonQSVC
from xenon_qai.quantum.noise import NoisyDeviceManager
from xenon_qai.quantum.resources import estimate_circuit_resources, compute_meyer_wallach_entanglement

__all__ = [
    "angle_encoding",
    "dense_angle_encoding",
    "iqp_encoding",
    "normalize_features_for_encoding",
    "ZZFeatureMap",
    "ChebyshevFeatureMap",
    "hardware_efficient_ansatz",
    "real_amplitudes_ansatz",
    "get_ansatz_weight_shape",
    "ParameterMatchedClassicalHead",
    "ClassicalMLPHead",
    "QuantumVariationalClassifier",
    "QuantumHybridClassifier",
    "train_quantum_classifier",
    "QuantumKernelEstimator",
    "XenonQSVC",
    "NoisyDeviceManager",
    "estimate_circuit_resources",
    "compute_meyer_wallach_entanglement",
]
