"""
Parameter-Matched Classical Inference Heads for XENON-QAI.

Provides classical benchmark heads with identical or closely matched parameter budgets
to the Quantum Neural Network (QNN) heads, ensuring scientifically rigorous and fair
quantum-vs-classical comparative benchmarking.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterMatchedClassicalHead(nn.Module):
    """
    Classical Multi-Layer Perceptron head designed to match the parameter count
    of an N-qubit L-layer variational quantum circuit.
    
    If QNN has N_params = L * N * 2 (e.g. 8 qubits * 4 layers * 2 = 64 parameters),
    this classical head is configured with hidden dimensions matching ~64 parameters.
    """
    
    def __init__(
        self,
        in_features: int,
        target_param_count: int,
        out_features: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.target_param_count = target_param_count
        self.out_features = out_features
        
        # Calculate hidden dimension H such that: in_features*H + H + H*out_features + out_features ~ target_param_count
        # H * (in_features + 1 + out_features) ~ target_param_count - out_features
        denom = in_features + 1 + out_features
        hidden_dim = max(2, (target_param_count - out_features) // max(1, denom))
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_features)
        )
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalMLPHead(nn.Module):
    """
    Standard classical MLP head for comparison against QNN classifier.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        out_features: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
