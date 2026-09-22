"""
Nuclear Recoil (NR) and Electronic Recoil (ER) Quenching Physics.

Implements Lindhard quenching theory for nuclear recoils in liquid xenon:
  L(E_R) = (k * g(eps)) / (1 + k * g(eps))
where eps = 11.5 * E_R * Z^(-7/3) is the reduced energy.
"""

from typing import Union
import numpy as np

# Physical constants for Xenon (Z=54, A=131.293)
XENON_Z: float = 54.0
XENON_A: float = 131.293
LINDHARD_K: float = 0.1394  # Typical Lindhard proportionality factor in liquid Xe


def lindhard_quenching(
    e_nr_kev: Union[float, np.ndarray],
    z: float = XENON_Z,
    k: float = LINDHARD_K
) -> Union[float, np.ndarray]:
    """
    Computes the Lindhard energy-dependent quenching factor for nuclear recoils in xenon.
    
    Args:
        e_nr_kev: Nuclear recoil true deposited energy in keV.
        z: Atomic number of target nucleus (Z=54 for Xe).
        k: Electronic stopping proportionality factor.
        
    Returns:
        Lindhard fraction L(E_R) in [0, 1] representing energy converted to electronic excitation.
    """
    e_arr = np.maximum(np.asarray(e_nr_kev), 1e-4)
    # Dimensionless reduced energy
    eps = 11.5 * e_arr * (z ** (-7.0 / 3.0))
    # Universal screening function g(eps)
    g_eps = 3.0 * (eps ** 0.15) + 0.7 * (eps ** 0.6) + eps
    l_factor = (k * g_eps) / (1.0 + k * g_eps)
    return l_factor


def nr_to_ee_energy(e_nr_kev: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts nuclear recoil energy (keV_nr) to equivalent electronic recoil energy (keV_ee).
    """
    l_fact = lindhard_quenching(e_nr_kev)
    return np.asarray(e_nr_kev) * l_fact
