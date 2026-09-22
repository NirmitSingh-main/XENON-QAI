"""
Physics modules for XENON-QAI.
"""

from xenon_qai.physics.energy import (
    reconstruct_energy_combined,
    correct_s2_lifetime,
    W_VALUE_EV,
    G1_DEFAULT,
    G2_DEFAULT,
)
from xenon_qai.physics.s1_s2 import (
    compute_log_s2_over_s1,
    er_band_mean,
    nr_band_mean,
    er_band_width,
    compute_er_leakage_fraction,
)
from xenon_qai.physics.recoil import lindhard_quenching, nr_to_ee_energy
from xenon_qai.physics.cevns import (
    helm_form_factor,
    weak_nuclear_charge,
    cevns_differential_cross_section,
)
from xenon_qai.physics.neutrino_fog import (
    estimate_solar_8b_annual_events,
    neutrino_fog_cross_section_floor,
)
from xenon_qai.physics.backgrounds import BACKGROUND_PROPERTIES, get_background_info
from xenon_qai.physics.detector_geometry import is_in_fiducial_volume, drift_time_to_z
from xenon_qai.physics.timing import expected_s2_width_ns

__all__ = [
    "reconstruct_energy_combined",
    "correct_s2_lifetime",
    "W_VALUE_EV",
    "G1_DEFAULT",
    "G2_DEFAULT",
    "compute_log_s2_over_s1",
    "er_band_mean",
    "nr_band_mean",
    "er_band_width",
    "compute_er_leakage_fraction",
    "lindhard_quenching",
    "nr_to_ee_energy",
    "helm_form_factor",
    "weak_nuclear_charge",
    "cevns_differential_cross_section",
    "estimate_solar_8b_annual_events",
    "neutrino_fog_cross_section_floor",
    "BACKGROUND_PROPERTIES",
    "get_background_info",
    "is_in_fiducial_volume",
    "drift_time_to_z",
    "expected_s2_width_ns",
]
