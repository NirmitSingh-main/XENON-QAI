"""
S1-S2 Scintillation vs Ionization Discrimination Physics.

In liquid xenon, Nuclear Recoils (NR, e.g. WIMPs, neutrons, CEvNS) have higher
stopping power (dE/dx) than Electronic Recoils (ER, e.g. gammas, betas), leading
to higher charge recombination and therefore smaller S2/S1 ratios.

Discriminator: log10(S2/S1) or log10(S2_bottom / S1) as a function of S1.
"""

from typing import Tuple, Union, Optional
import numpy as np


def compute_log_s2_over_s1(
    s1: Union[float, np.ndarray],
    s2: Union[float, np.ndarray],
    eps: float = 1e-4
) -> Union[float, np.ndarray]:
    """
    Computes standard discrimination variable log10(S2 / S1).
    """
    s1_safe = np.maximum(np.asarray(s1), eps)
    s2_safe = np.maximum(np.asarray(s2), eps)
    return np.log10(s2_safe / s1_safe)


def er_band_mean(s1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Empirical Electronic Recoil (ER) median discrimination band in log10(S2/S1) vs S1 (PE).
    """
    s1_arr = np.maximum(np.asarray(s1), 1.0)
    # Typical empirical parametrization in LXe TPCs
    return 2.45 - 0.18 * np.log10(s1_arr)


def nr_band_mean(s1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Empirical Nuclear Recoil (NR) median discrimination band in log10(S2/S1) vs S1 (PE).
    NR events systematically lie below ER events by ~0.6-0.8 decades in log10(S2/S1).
    """
    s1_arr = np.maximum(np.asarray(s1), 1.0)
    return 1.75 - 0.15 * np.log10(s1_arr)


def er_band_width(s1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    1-sigma Gaussian width of the ER band as a function of S1 (Poisson + Fano fluctuation).
    """
    s1_arr = np.maximum(np.asarray(s1), 1.0)
    return 0.12 + 0.35 / np.sqrt(s1_arr)


def compute_er_leakage_fraction(
    s1: np.ndarray,
    log_s2_s1: np.ndarray,
    target_nr_acceptance: float = 0.50
) -> float:
    """
    Computes fraction of ER background events leaking below the NR median (50% NR acceptance line).
    In pure classical cut-based physics analyses, this ER leakage is typically ~ 10^-3 (99.9% ER rejection).
    """
    nr_median = nr_band_mean(s1)
    leaked_events = np.sum(log_s2_s1 < nr_median)
    total_events = len(s1)
    return float(leaked_events / max(1, total_events))
