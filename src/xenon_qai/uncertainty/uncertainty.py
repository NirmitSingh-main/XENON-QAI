"""
Epistemic and Aleatoric Uncertainty Estimation using Monte Carlo Dropout.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


def estimate_mc_dropout_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    physics: Optional[torch.Tensor] = None,
    n_forward_passes: int = 30
) -> Dict[str, np.ndarray]:
    """
    Performs Monte Carlo Dropout sampling to quantify model predictive uncertainty.
    
    Returns:
        Dictionary with:
          - 'mean_score': Predictive mean probability
          - 'epistemic_uncertainty': Variance across MC passes
          - 'std_score': Standard deviation of predictions
    """
    # Enable dropout during inference
    model.train()
    samples = []
    
    with torch.no_grad():
        for _ in range(n_forward_passes):
            logits, _ = model(x, physics)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            samples.append(probs)
            
    samples_arr = np.stack(samples, axis=0)  # shape (n_passes, n_samples)
    mean_score = np.mean(samples_arr, axis=0)
    variance = np.var(samples_arr, axis=0)
    std_dev = np.std(samples_arr, axis=0)
    
    return {
        "mean_score": mean_score,
        "epistemic_uncertainty": variance,
        "std_score": std_dev
    }
