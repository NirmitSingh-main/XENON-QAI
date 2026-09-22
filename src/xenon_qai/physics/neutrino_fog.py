"""
Neutrino Fog & Background Limits for Direct Detection Dark Matter Searches.

Models the coherent neutrino scattering backgrounds that form the ultimate
sensitivity floor ('Neutrino Fog') for liquid xenon direct detection experiments.
Per dataanalysis.txt Section 11: CNNS is the irreducible coherent background.
"""

from typing import Dict, Any, Tuple
import numpy as np

# Standard solar / atmospheric / DSNB integrated neutrino fluxes at Earth (cm^-2 s^-1)
SOLAR_8B_FLUX: float = 5.46e6      # 8B solar neutrinos (up to ~15 MeV)
SOLAR_HEP_FLUX: float = 8.04e3     # hep solar neutrinos (up to ~18.8 MeV)
DSNB_FLUX: float = 40.0            # Diffuse supernova neutrino background
ATMOSPHERIC_FLUX: float = 1.0e3    # Atmospheric neutrinos (GeV scale)


def estimate_solar_8b_annual_events(
    exposure_tonne_years: float,
    threshold_kev_nr: float = 1.0,
    max_energy_kev_nr: float = 5.0
) -> float:
    """
    Estimates expected number of solar 8B CEvNS events in LXe for a given exposure (tonne-years).
    In xenon, 8B neutrinos produce recoils primarily between 0.5 and 4 keV_nr (~10-20 events / tonne-year at low threshold).
    """
    # Approximate rate in [1.0, 5.0] keV_nr window ~ 12 events / tonne-year
    base_rate_per_tonne_yr = 12.5
    # Threshold suppression factor
    suppression = np.exp(-0.8 * max(0.0, threshold_kev_nr - 1.0))
    return float(exposure_tonne_years * base_rate_per_tonne_yr * suppression)


def neutrino_fog_cross_section_floor(wimp_mass_gev: float) -> float:
    """
    Returns approximate spin-independent WIMP-nucleon cross-section floor (cm^2)
    defined by the Neutrino Fog for a given WIMP mass (GeV/c^2).
    """
    # Empirical parameterization of the O'Hare (2021) neutrino floor in xenon
    if wimp_mass_gev < 10.0:
        # Dominated by 8B solar neutrinos
        return 1.0e-44 * (10.0 / wimp_mass_gev) ** 4.5
    elif 10.0 <= wimp_mass_gev <= 100.0:
        # Minimum around 40-50 GeV (DSNB / atmospheric floor)
        return 2.0e-49 * (1.0 + ((wimp_mass_gev - 45.0) / 30.0) ** 2)
    else:
        # High mass tail rising linearly with mass
        return 2.0e-49 * (wimp_mass_gev / 50.0)
