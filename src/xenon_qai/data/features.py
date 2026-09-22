"""
Physics Feature Extraction for XENON-QAI Dataset.
Implements the 13 core physics-derived features defined in dataanalysis.txt.
"""

import math
from typing import Dict, List, Optional, Union
import numpy as np
import h5py

from xenon_qai.data.types import PHYSICS_FEATURE_NAMES, PHYSICS_FEATURE_DIM


def compute_physics_features(
    s1_hits: np.ndarray,
    s2_hits: np.ndarray,
    s1_raw: Optional[np.ndarray],
    s2_raw: Optional[np.ndarray],
    hits_top: np.ndarray,
    hits_bot: np.ndarray,
) -> np.ndarray:
    """
    Computes 13 physically justified features from raw event representations.
    
    Args:
        s1_hits: 1D array of peakS1_onlyhits (photoelectrons / bin).
        s2_hits: 1D array of peakS2_onlyhits (photoelectrons / bin).
        s1_raw: 1D array of peakS1_raw (optional, if None s1_hits used).
        s2_raw: 1D array of peakS2_raw (optional, if None s2_hits used).
        hits_top: 1D array of length 127 for top PMT array.
        hits_bot: 1D array of length 121 for bottom PMT array.
        
    Returns:
        1D float32 array of shape (13,) containing all engineered physics features.
    """
    if s1_raw is None:
        s1_raw = s1_hits
    if s2_raw is None:
        s2_raw = s2_hits
        
    s1_area_hits = float(np.sum(s1_hits))
    s2_area_hits = float(np.sum(s2_hits))
    s1_area_raw = float(np.sum(s1_raw))
    s2_area_raw = float(np.sum(s2_raw))
    
    s1_len = float(len(s1_hits))
    s2_len = float(len(s2_hits))
    
    s1_max = float(np.max(s1_hits)) if len(s1_hits) > 0 else 0.0
    s2_max = float(np.max(s2_hits)) if len(s2_hits) > 0 else 0.0
    
    top_sum = float(np.sum(hits_top))
    bot_sum = float(np.sum(hits_bot))
    total_pmt_hits = top_sum + bot_sum
    
    # Top-to-bottom charge asymmetry (Z-depth proxy)
    top_bot_asym = float((top_sum - bot_sum) / (total_pmt_hits + 1e-6))
    
    # Recoil type discriminator: log10(S2/S1)
    s2_s1_ratio = (s2_area_hits + 1e-3) / (s1_area_hits + 1e-3)
    log_s2_s1 = float(math.log10(max(1e-3, s2_s1_ratio)))
    
    features = np.array([
        s1_area_hits,
        s2_area_hits,
        s1_area_raw,
        s2_area_raw,
        log_s2_s1,
        s1_len,
        s2_len,
        s1_max,
        s2_max,
        top_sum,
        bot_sum,
        total_pmt_hits,
        top_bot_asym
    ], dtype=np.float32)
    
    return features


def extract_features_from_hdf5_event(h5_file: h5py.File, event_key: str) -> np.ndarray:
    """
    Extracts 13 physics features directly from an open HDF5 file and event key.
    """
    s1_hits = h5_file["peakS1_onlyhits"][event_key][()]
    s2_hits = h5_file["peakS2_onlyhits"][event_key][()]
    s1_raw = h5_file["peakS1_raw"][event_key][()]
    s2_raw = h5_file["peakS2_raw"][event_key][()]
    hits_top = h5_file["HITS_top"][event_key][()]
    hits_bot = h5_file["HITS_bot"][event_key][()]
    
    return compute_physics_features(
        s1_hits=s1_hits,
        s2_hits=s2_hits,
        s1_raw=s1_raw,
        s2_raw=s2_raw,
        hits_top=hits_top,
        hits_bot=hits_bot
    )
