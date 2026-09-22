"""
Liquid Xenon Dual-Phase Time Projection Chamber (TPC) Geometry Physics.

Models:
- Top PMT array (127 PMTs) and bottom PMT array (121 PMTs)
- Drift region, gate grid, liquid-gas interface, and anode grid
- Radial and depth fiducial volume cuts for background shielding
"""

from typing import Tuple, Union
import numpy as np

# Typical XENONnT / XENON1T scale geometry constants
TPC_RADIUS_CM: float = 66.0         # TPC internal radius (cm)
TPC_HEIGHT_CM: float = 150.0        # TPC drift length (cm)
DRIFT_VELOCITY_MM_US: float = 1.33  # Electron drift velocity in liquid Xe at ~100 V/cm (mm / us)
FIDUCIAL_R_CUT_CM: float = 60.0     # Standard fiducial radius cut (cm)
FIDUCIAL_Z_MIN_CM: float = -140.0   # Lower Z fiducial boundary (cm)
FIDUCIAL_Z_MAX_CM: float = -10.0    # Upper Z fiducial boundary (cm)


def is_in_fiducial_volume(
    r_cm: Union[float, np.ndarray],
    z_cm: Union[float, np.ndarray],
    r_cut_cm: float = FIDUCIAL_R_CUT_CM,
    z_min_cm: float = FIDUCIAL_Z_MIN_CM,
    z_max_cm: float = FIDUCIAL_Z_MAX_CM
) -> Union[bool, np.ndarray]:
    """
    Evaluates whether event coordinates (R, Z) lie within the low-background fiducial volume.
    Fiducial cuts reject >99.9% of external gamma backgrounds and wall surface events.
    """
    r_arr = np.asarray(r_cm)
    z_arr = np.asarray(z_cm)
    in_r = r_arr <= r_cut_cm
    in_z = (z_arr >= z_min_cm) & (z_arr <= z_max_cm)
    return in_r & in_z


def drift_time_to_z(
    drift_time_us: Union[float, np.ndarray],
    drift_velocity_mm_us: float = DRIFT_VELOCITY_MM_US
) -> Union[float, np.ndarray]:
    """
    Converts electron drift time t_drift (us) to interaction vertical coordinate Z (cm):
      Z = - (t_drift * v_drift) / 10
    (Z = 0 at the liquid surface / gate grid; negative downwards into the detector).
    """
    dt_arr = np.asarray(drift_time_us)
    z_mm = - dt_arr * drift_velocity_mm_us
    return z_mm / 10.0  # Convert mm to cm
