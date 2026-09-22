"""
Physics Consistency Checks for Learned Decision Boundaries.

Validates that model predictions obey fundamental detector physics:
- Monotonicity with respect to log10(S2/S1)
- Invariance under azimuthal rotation of PMT coordinate grids
- Proper handling of physical limits
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn


def check_s2_s1_monotonicity(
    model: nn.Module,
    sample_multimodal: torch.Tensor,
    s2_scale_factors: np.ndarray = np.linspace(0.2, 5.0, 20)
) -> Dict[str, Any]:
    """
    Checks whether candidate score decreases as S2/S1 ratio increases
    (nuclear recoil signal should have lower S2/S1 than ER background).
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for factor in s2_scale_factors:
            modified = sample_multimodal.clone()
            # Scale S2 region (160 to 660)
            modified[:, 160:660] *= factor
            logits, _ = model(modified)
            prob = torch.sigmoid(logits).item()
            scores.append(prob)
            
    scores_arr = np.array(scores)
    # Correlation between scale factor and signal score
    # Expected negative correlation (higher S2 -> ER-like -> lower dark matter signal score)
    corr = np.corrcoef(s2_scale_factors, scores_arr)[0, 1]
    is_physically_consistent = corr < 0.0
    
    return {
        "correlation": float(corr),
        "is_physically_consistent": bool(is_physically_consistent),
        "scores_vs_s2_scale": scores_arr.tolist()
    }
