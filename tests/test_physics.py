"""
Unit tests for XENON-QAI detector physics, Lindhard quenching, CEvNS, and discrimination bands.
"""

import pytest
import numpy as np
from xenon_qai.physics import (
    reconstruct_energy_combined,
    correct_s2_lifetime,
    compute_log_s2_over_s1,
    er_band_mean,
    nr_band_mean,
    compute_er_leakage_fraction,
    lindhard_quenching,
    nr_to_ee_energy,
    helm_form_factor,
    weak_nuclear_charge,
    cevns_differential_cross_section,
    estimate_solar_8b_annual_events,
    is_in_fiducial_volume,
    drift_time_to_z,
    expected_s2_width_ns
)


def test_energy_reconstruction():
    # S1 = 50 PE, S2 = 3000 PE
    s1, s2 = 50.0, 3000.0
    energy_kev = reconstruct_energy_combined(s1, s2)
    assert energy_kev > 0.0
    assert 5.0 < energy_kev < 10.0  # Typical low-energy physical expectation


def test_s1_s2_discrimination_bands():
    s1_vals = np.array([10.0, 50.0, 100.0])
    er_means = er_band_mean(s1_vals)
    nr_means = nr_band_mean(s1_vals)
    
    # Fundamental physics invariant: ER band mean > NR band mean
    assert np.all(er_means > nr_means)
    # Difference should be ~0.7 decades
    assert np.all((er_means - nr_means) > 0.5)


def test_lindhard_quenching():
    energies_nr = np.array([1.0, 10.0, 50.0, 100.0])
    l_factors = lindhard_quenching(energies_nr)
    
    # Lindhard factor must be bounded in (0, 1)
    assert np.all((l_factors > 0.0) & (l_factors < 1.0))
    # Quenching increases monotonically with energy
    assert np.all(np.diff(l_factors) > 0.0)


def test_cevns_physics():
    # Weak charge for Xenon-131 (Z=54, N=77)
    qw = weak_nuclear_charge(z=54, a=131)
    assert 70.0 < qw < 80.0  # Dominantly neutron number N=77
    
    # Helm form factor at E_R = 1 keV should be near 1 (coherent)
    ff = helm_form_factor(1.0)
    assert 0.95 <= ff <= 1.0
    
    # Differential cross section should be positive
    diff_xs = cevns_differential_cross_section(e_r_kev=1.0, e_nu_mev=15.0)
    assert diff_xs > 0.0


def test_fiducial_volume_and_drift_z():
    # Event inside fiducial
    assert is_in_fiducial_volume(r_cm=30.0, z_cm=-50.0) == True
    # Event outside (wall)
    assert is_in_fiducial_volume(r_cm=65.0, z_cm=-50.0) == False
    
    # Z coordinate should be negative for positive drift time
    z = drift_time_to_z(100.0)
    assert z < 0.0
