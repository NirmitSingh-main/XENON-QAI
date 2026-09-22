"""
Feature Attribution and Saliency Analysis for XENON-QAI.

Implements Integrated Gradients (Sundararajan et al., 2017) to compute
input attribution maps for 1D waveforms (S1/S2), PMTs, and physics features.
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn


def compute_integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    physics: Optional[torch.Tensor] = None,
    target_class: int = 0,
    steps: int = 40,
    baseline: Optional[torch.Tensor] = None
) -> Dict[str, np.ndarray]:
    """
    Computes Integrated Gradients attribution for input features:
      IG_i(x) = (x_i - x'_i) * integral_0^1 (d F(x' + alpha*(x - x')) / d x_i) d alpha
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
        
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    # (steps + 1, batch_size, feature_dim)
    interpolated = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * (x - baseline).unsqueeze(0)
    
    total_grads = torch.zeros_like(x)
    
    for alpha_idx in range(steps + 1):
        step_x = interpolated[alpha_idx].clone().detach().requires_grad_(True)
        if physics is not None:
            step_phys = physics.clone().detach()
        else:
            step_phys = None
            
        logits, _ = model(step_x, step_phys)
        if logits.shape[-1] == 1:
            score = logits.sum()
        else:
            score = logits[:, target_class].sum()
            
        grads = torch.autograd.grad(score, step_x)[0]
        total_grads += grads
        
    avg_grads = total_grads / (steps + 1)
    attributions = (x - baseline) * avg_grads
    
    return {
        "attributions": attributions.detach().cpu().numpy(),
        "input": x.detach().cpu().numpy()
    }
