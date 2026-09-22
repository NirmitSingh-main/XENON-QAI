"""
Coherent Elastic Neutrino-Nucleus Scattering (CEvNS) Physics.

Implements Standard Model CEvNS differential cross section:
  dsigma/dE_R = (G_F^2 * M) / (4 * pi) * Q_W^2 * F^2(q^2) * (1 - (M * E_R) / (2 * E_nu^2))
where:
  G_F = Fermi coupling constant
  M = Xenon nuclear mass
  Q_W = N - (1 - 4 * sin^2(theta_W)) * Z (weak nuclear charge ~ N)
  F(q^2) = Helm nuclear form factor
"""

from typing import Union
import numpy as np

# Physical constants
G_FERMI_GEV2: float = 1.1663787e-5    # GeV^-2
HBAR_C_GEV_FM: float = 0.19732698     # GeV * fm
SIN2_THETA_W: float = 0.23122         # Weak mixing angle
M_XENON_GEV: float = 131.293 * 0.9315 # Xe nuclear mass ~ 122.3 GeV


def helm_form_factor(e_r_kev: Union[float, np.ndarray], a: float = 131.293) -> Union[float, np.ndarray]:
    """
    Computes the Helm nuclear form factor F(q^2) representing spatial coherence over the nucleus.
    
    Args:
        e_r_kev: Nuclear recoil energy in keV.
        a: Target atomic mass number.
    """
    e_arr = np.asarray(e_r_kev)
    # Momentum transfer q in fm^-1: q = sqrt(2 * M * E_R) / hbar_c
    e_gev = e_arr * 1e-6
    m_gev = a * 0.9315
    q_gev = np.sqrt(2.0 * m_gev * e_gev)
    q_fm = q_gev / HBAR_C_GEV_FM  # in fm^-1
    
    # Nuclear radius parameters (Lewin & Smith / Helm)
    r_n = 1.2 * (a ** (1.0 / 3.0))  # Nuclear skin radius (fm)
    s = 0.9  # Skin thickness parameter (fm)
    r0 = np.sqrt(np.maximum(1e-4, r_n ** 2 - 5 * s ** 2))
    
    qr = q_fm * r0
    # Spherical Bessel j1(qr) / (qr) * 3
    # For small qr, j1(qr)/qr -> 1/3, so 3 * j1(qr)/qr -> 1
    qr_safe = np.where(qr < 1e-4, 1e-4, qr)
    spherical_bessel = (np.sin(qr_safe) - qr_safe * np.cos(qr_safe)) / (qr_safe ** 3)
    j1_term = 3.0 * spherical_bessel
    
    # Gaussian damping for surface skin thickness
    form_factor = j1_term * np.exp(-0.5 * (q_fm * s) ** 2)
    return np.clip(form_factor, 0.0, 1.0)


def weak_nuclear_charge(z: float = 54.0, a: float = 131.293) -> float:
    """
    Computes weak nuclear charge Q_W = N - (1 - 4 * sin^2(theta_W)) * Z.
    Since (1 - 4*sin^2(theta_W)) ~ 0.075 is small, Q_W is dominated by neutron count N.
    """
    n = a - z
    return n - (1.0 - 4.0 * SIN2_THETA_W) * z


def cevns_differential_cross_section(
    e_r_kev: Union[float, np.ndarray],
    e_nu_mev: float,
    z: float = 54.0,
    a: float = 131.293
) -> Union[float, np.ndarray]:
    """
    Computes Standard Model CEvNS differential cross section dsigma/dE_R in cm^2/keV.
    """
    e_r_arr = np.asarray(e_r_kev)
    e_r_gev = e_r_arr * 1e-6
    e_nu_gev = e_nu_mev * 1e-3
    m_gev = a * 0.9315
    
    # Maximum kinematic recoil energy E_R_max = 2 * E_nu^2 / (M + 2*E_nu)
    e_r_max_gev = 2.0 * (e_nu_gev ** 2) / (m_gev + 2.0 * e_nu_gev)
    
    qw = weak_nuclear_charge(z, a)
    ff = helm_form_factor(e_r_arr, a)
    
    # Differential cross section in natural units (GeV^-3)
    # dsigma/dE_R = (G_F^2 * M) / (4 * pi) * Q_W^2 * F^2 * (1 - (M * E_R)/(2 * E_nu^2))
    kinematic_factor = 1.0 - (m_gev * e_r_gev) / (2.0 * (e_nu_gev ** 2))
    kinematic_factor = np.maximum(0.0, kinematic_factor)
    
    # Only valid where E_R <= E_R_max
    kinematic_factor = np.where(e_r_gev <= e_r_max_gev, kinematic_factor, 0.0)
    
    dsigma_dE_gev3 = (G_FERMI_GEV2 ** 2 * m_gev) / (4.0 * np.pi) * (qw ** 2) * (ff ** 2) * kinematic_factor
    
    # Conversion: 1 GeV^-2 = 0.389379 mb = 0.389379e-27 cm^2
    # 1 GeV^-3 = 0.389379e-27 cm^2 / GeV = 0.389379e-33 cm^2 / keV
    conv_factor = 0.389379e-33
    return dsigma_dE_gev3 * conv_factor
