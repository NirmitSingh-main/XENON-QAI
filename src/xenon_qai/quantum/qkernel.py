"""
Quantum Kernel Estimation and Quantum Support Vector Classifier (QSVC).

Computes quantum transition fidelities |<psi(x)|psi(x')>|^2 using parameterized
quantum feature maps, constructing positive semi-definite Gram matrices for QSVC.
"""

from typing import Optional, Tuple
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from xenon_qai.quantum.feature_map import ZZFeatureMap
from xenon_qai.evaluation.metrics import compute_binary_metrics


class QuantumKernelEstimator:
    """
    Computes quantum kernel Gram matrix K(x_i, x_j) = |<0| U(x_i)^\dagger U(x_j) |0>|^2.
    """
    
    def __init__(self, n_qubits: int = 4, reps: int = 2, dev_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_map = ZZFeatureMap(n_qubits=n_qubits, reps=reps)
        self.dev = qml.device(dev_name, wires=n_qubits)
        
        # Define overlap circuit
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            self.feature_map(x1)
            qml.adjoint(self.feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))
            
        self._circuit = kernel_circuit
        
    def compute_kernel_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes overlap fidelity between two data points."""
        probs = self._circuit(x1, x2)
        # Probability of state |00...0>
        return float(probs[0])
    
    def compute_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the Gram matrix between X1 and X2 (or symmetric Gram matrix if X2 is None).
        """
        n1 = len(X1)
        if X2 is None:
            K = np.zeros((n1, n1))
            for i in range(n1):
                K[i, i] = 1.0
                for j in range(i + 1, n1):
                    val = self.compute_kernel_element(X1[i], X1[j])
                    K[i, j] = val
                    K[j, i] = val
            return K
        else:
            n2 = len(X2)
            K = np.zeros((n1, n2))
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.compute_kernel_element(X1[i], X2[j])
            return K


class XenonQSVC:
    """
    Quantum Support Vector Classifier for XENON-QAI.
    """
    
    def __init__(self, n_qubits: int = 4, C: float = 1.0, reps: int = 2):
        self.n_qubits = n_qubits
        self.C = C
        self.estimator = QuantumKernelEstimator(n_qubits=n_qubits, reps=reps)
        self.svc = SVC(C=C, kernel="precomputed", probability=True, random_state=42)
        self.X_train_cached: Optional[np.ndarray] = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XenonQSVC":
        self.X_train_cached = X_train
        K_train = self.estimator.compute_matrix(X_train)
        self.svc.fit(K_train, y_train)
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        K_test = self.estimator.compute_matrix(X_test, self.X_train_cached)
        return self.svc.predict(K_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        K_test = self.estimator.compute_matrix(X_test, self.X_train_cached)
        return self.svc.predict_proba(K_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        return compute_binary_metrics(y_test, y_proba, y_pred)
