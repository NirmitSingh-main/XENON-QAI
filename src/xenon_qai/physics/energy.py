"""
Liquid Xenon Energy Reconstruction Physics for XENON-QAI.

Implements combined energy scale reconstruction using prompt scintillation (S1)
and proportional ionization (S2) signals:
  E = W * ( S1 / g1 + S2 / g2 )
where:
  W = 13.7 eV (average work function per electron-ion/photon pair)
  g1 = primary scintillation photon detection efficiency
  g2 = secondary proportional scintillation gain per extracted electron
"""

from typing import Union, Tuple
import numpy as np

# Standard XENON detector physical constants
W_VALUE_EV: float = 13.7          # Average energy to produce one quantum (eV)
G1_DEFAULT: float = 0.142          # Photon detection efficiency (PE / photon)
G2_DEFAULT: float = 31.2           # Secondary gain (PE / electron)
ELECTRON_LIFETIME_US: float = 650.0  # Typical electron drift lifetime (microseconds)


def reconstruct_energy_combined(
    s1: Union[float, np.ndarray],
    s2: Union[float, np.ndarray],
    g1: float = G1_DEFAULT,
    g2: float = G2_DEFAULT,
    w_value_ev: float = W_VALUE_EV
) -> Union[float, np.ndarray]:
    """
    Reconstructs total deposited energy in keV using the combined energy scale.
    
    Args:
        s1: S1 pulse area (PE)
        s2: S2 pulse area (PE, corrected for electron lifetime)
        g1: Photon detection efficiency
        g2: Single-electron gain
        w_value_ev: Average work function in eV
        
    Returns:
        Deposited energy in keV.
    """
    n_photons = np.asarray(s1) / g1
    n_electrons = np.asarray(s2) / g2
    total_quanta = n_photons + n_electrons
    energy_ev = total_quanta * w_value_ev
    energy_kev = energy_ev / 1000.0
    return energy_kev


def correct_s2_lifetime(
    s2_raw: Union[float, np.ndarray],
    drift_time_us: Union[float, np.ndarray],
    electron_lifetime_us: float = ELECTRON_LIFETIME_US
) -> Union[float, np.ndarray]:
    """
    Corrects S2 signal for electron absorption by electronegative impurities during drift:
      S2_corr = S2_raw * exp(drift_time / electron_lifetime)
    """
    s2_arr = np.asarray(s2_raw)
    dt_arr = np.asarray(drift_time_us)
    correction = np.exp(dt_arr / electron_lifetime_us)
    return s2_arr * correction
