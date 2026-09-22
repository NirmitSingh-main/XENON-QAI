"""
Statistical Discovery Significance for Dark Matter Searches.

Implements Profile Likelihood Ratio asymptotic discovery significance:
  Z = sqrt(2 * [ (s + b) * ln(1 + s/b) - s ])  (Cowan et al., 2011 / Asimov dataset)
"""

from typing import Union
import numpy as np


def compute_discovery_significance(
    s: Union[float, np.ndarray],
    b: Union[float, np.ndarray],
    b_syst: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Computes Poisson discovery significance Z (in units of standard deviations sigma).
    
    Args:
        s: Expected number of signal events.
        b: Expected number of background events.
        b_syst: Fractional systematic uncertainty on background (e.g. 0.10 for 10%).
        
    Returns:
        Significance Z in Gaussian sigmas (Z >= 5 corresponds to 5-sigma discovery).
    """
    s_arr = np.maximum(np.asarray(s), 1e-8)
    b_arr = np.maximum(np.asarray(b), 1e-8)
    
    if b_syst == 0.0:
        # Standard Cowan asymptotic formula without systematic uncertainty
        val = 2.0 * ((s_arr + b_arr) * np.log(1.0 + s_arr / b_arr) - s_arr)
        return np.sqrt(np.maximum(0.0, val))
    else:
        # Asymptotic formula with Gaussian systematic uncertainty sigma_b = b * b_syst
        sig_b = b_arr * b_syst
        sig_b2 = sig_b ** 2
        term1 = (s_arr + b_arr) * np.log(((s_arr + b_arr) * (b_arr + sig_b2)) / (b_arr ** 2 + (s_arr + b_arr) * sig_b2))
        term2 = (b_arr ** 2 / sig_b2) * np.log(1.0 + (sig_b2 * s_arr) / (b_arr * (b_arr + sig_b2)))
        val = 2.0 * (term1 - term2)
        return np.sqrt(np.maximum(0.0, val))
