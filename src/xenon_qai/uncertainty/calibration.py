"""
Model Probability Calibration and Expected Calibration Error (ECE).
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler:
    """
    Post-hoc temperature scaling for neural network logit calibration (Guo et al., 2017).
    """
    
    def __init__(self, lr: float = 0.01, max_iter: int = 50):
        self.temperature = 1.0
        self.lr = lr
        self.max_iter = max_iter
        
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Optimizes single scalar temperature T using validation NLL loss."""
        t_param = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LBFGS([t_param], lr=self.lr, max_iter=self.max_iter)
        
        logits_t = torch.tensor(logits, dtype=torch.float32)
        if logits.ndim == 1 or logits.shape[1] == 1:
            labels_t = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            labels_t = torch.tensor(labels, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
            
        def eval_loss():
            optimizer.zero_grad()
            scaled = logits_t / t_param
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        self.temperature = float(t_param.item())
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / max(1e-3, self.temperature)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(probs)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            ece += (bin_size / n_samples) * np.abs(avg_accuracy - avg_confidence)
            
    return float(ece)
