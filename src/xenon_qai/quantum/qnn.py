"""
Quantum Neural Network (QNN) Layer & Hybrid Classifier for XENON-QAI.

Implements PyTorch-differentiable Variational Quantum Classifiers (VQC)
interfaced downstream of learned deep-learning representations.
"""

from typing import Dict, Any, Optional, Tuple, List, Callable
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

from xenon_qai.quantum.encoding import angle_encoding, normalize_features_for_encoding
from xenon_qai.quantum.circuit import hardware_efficient_ansatz, get_ansatz_weight_shape
from xenon_qai.evaluation.metrics import compute_binary_metrics


def create_qnn_qnode(
    n_qubits: int,
    n_layers: int,
    dev_name: str = "default.qubit",
    diff_method: str = "best"
) -> Tuple[Callable, Tuple[int, ...]]:
    """
    Creates a PennyLane QNode for a Variational Quantum Classifier.
    """
    dev = qml.device(dev_name, wires=n_qubits)
    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    
    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def qnode(inputs, weights):
        # 1. Feature encoding layer
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
            
        # 2. Variational Ansatz layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                
        # 3. Measurement: Expectation of PauliZ on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
    return qnode, weight_shapes


class QuantumVariationalClassifier(nn.Module):
    """
    PyTorch-compatible Variational Quantum Classifier (VQC) layer.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        in_features: int = 4,
        num_classes: int = 1,
        dev_name: str = "default.qubit"
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.in_features = in_features
        self.num_classes = num_classes
        
        # Linear projection if classical latent dimension != n_qubits
        if in_features != n_qubits:
            self.pre_project = nn.Linear(in_features, n_qubits)
        else:
            self.pre_project = None
            
        qnode, weight_shapes = create_qnn_qnode(n_qubits, n_layers, dev_name)
        self.qnn_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # Post-quantum readout projection
        self.post_project = nn.Linear(n_qubits, num_classes)
        
    def count_quantum_parameters(self) -> int:
        """Returns number of variational quantum circuit parameters."""
        return sum(p.numel() for p in self.qnn_layer.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-projection and angular scaling to [-pi/2, pi/2]
        if self.pre_project is not None:
            x = self.pre_project(x)
        angles = torch.tanh(x) * (np.pi / 2.0)
        
        # Quantum forward execution
        q_out = self.qnn_layer(angles)
        
        # Classical post-readout projection
        logits = self.post_project(q_out)
        return logits


class QuantumHybridClassifier(nn.Module):
    """
    Complete Hybrid Classical-Quantum Classifier:
    Deep Classical Representation Encoder -> Quantum Variational Head.
    """
    
    def __init__(
        self,
        classical_encoder: nn.Module,
        latent_dim: int = 32,
        n_qubits: int = 4,
        n_layers: int = 3,
        num_classes: int = 1
    ):
        super().__init__()
        self.encoder = classical_encoder
        self.quantum_head = QuantumVariationalClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            in_features=latent_dim,
            num_classes=num_classes
        )
        
    def forward(
        self,
        x: torch.Tensor,
        physics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract latent representation from classical encoder
        if hasattr(self.encoder, "encode"):
            if hasattr(self.encoder, "_slice_multimodal_vector"):
                s1, s2, pmts = self.encoder._slice_multimodal_vector(x)
                z = self.encoder.encode(s1, s2, pmts, physics)
            else:
                z = self.encoder(x)
        elif hasattr(self.encoder, "extract_features"):
            z = self.encoder.extract_features(x)
        else:
            z = self.encoder(x)
            
        logits = self.quantum_head(z)
        return logits, z


def train_quantum_classifier(
    qnn: QuantumVariationalClassifier,
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    val_latents: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 25,
    lr: float = 0.02,
    batch_size: int = 32,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Trains quantum variational classifier on pre-extracted classical latent embeddings.
    """
    x_train_t = torch.tensor(train_latents, dtype=torch.float32)
    y_train_t = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    
    train_ds = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(qnn.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {"train_loss": [], "val_auc": []}
    
    for epoch in range(1, epochs + 1):
        qnn.train()
        total_loss = 0.0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = qnn(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
            
        avg_loss = total_loss / len(train_ds)
        history["train_loss"].append(avg_loss)
        
        # Validation evaluation
        qnn.eval()
        with torch.no_grad():
            val_x_t = torch.tensor(val_latents, dtype=torch.float32)
            val_logits = qnn(val_x_t)
            val_scores = torch.sigmoid(val_logits).cpu().numpy().squeeze(1)
            val_metrics = compute_binary_metrics(val_labels, val_scores)
            val_auc = val_metrics["roc_auc"]
            history["val_auc"].append(val_auc)
            
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"QNN Epoch {epoch:2d}/{epochs} | Train Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")
            
    return {"model": qnn, "history": history, "final_val_auc": history["val_auc"][-1]}
