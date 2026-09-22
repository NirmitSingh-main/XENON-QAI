"""
Pulse Timing Physics and Electron Diffusion in Liquid Xenon.

Models:
- Prompt S1 scintillation timing (< 100 ns)
- S2 pulse duration and longitudinal electron diffusion:
    sigma_t^2 = sigma_0^2 + (2 * D_L * t_drift) / v_drift^2
- Distinguishing physical single-site recoils from multiple scatters and accidental coincidences.
"""

from typing import Union
import numpy as np

# Longitudinal diffusion constant in liquid Xe at ~100 V/cm (cm^2 / s)
DIFFUSION_COEFFICIENT_DL: float = 30.0  # cm^2 / s
DRIFT_VELOCITY_CM_S: float = 1.33e5     # cm / s (1.33 mm / us)


def expected_s2_width_ns(
    drift_time_us: Union[float, np.ndarray],
    initial_width_ns: float = 200.0,
    d_l_cm2_s: float = DIFFUSION_COEFFICIENT_DL,
    v_drift_cm_s: float = DRIFT_VELOCITY_CM_S
) -> Union[float, np.ndarray]:
    """
    Computes expected S2 pulse temporal width (1-sigma Gaussian, in ns) as a function of drift depth.
    Longer drift paths broaden the S2 pulse due to longitudinal electron cloud diffusion.
    """
    dt_arr = np.asarray(drift_time_us)
    t_drift_s = dt_arr * 1e-6
    # Spatial diffusion variance sigma_z^2 = 2 * D_L * t
    sigma_z2_cm2 = 2.0 * d_l_cm2_s * t_drift_s
    # Temporal variance sigma_t^2 = sigma_z^2 / v_drift^2
    sigma_t2_s2 = sigma_z2_cm2 / (v_drift_cm_s ** 2)
    sigma_t_ns = np.sqrt(sigma_t2_s2) * 1e9
    
    # Add quadrature initial extraction width
    total_width_ns = np.sqrt(initial_width_ns ** 2 + sigma_t_ns ** 2)
    return total_width_ns
