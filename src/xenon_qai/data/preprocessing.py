"""
Preprocessing, Waveform Resampling, and Leakage-Free Scaling for XENON-QAI.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import interp1d

from xenon_qai.data.types import S1_MAX_LEN, S2_TARGET_LEN, TOP_PMT_COUNT, BOT_PMT_COUNT, MULTIMODAL_DIM


def pad_s1_waveform(s1_wave: np.ndarray, max_len: int = S1_MAX_LEN) -> np.ndarray:
    """
    Pads S1 waveform to fixed length max_len with zeros.
    Observed range: 45 to 155 bins; max_len=160 accommodates 100% of events without truncation.
    """
    out = np.zeros(max_len, dtype=np.float32)
    n = min(len(s1_wave), max_len)
    out[:n] = s1_wave[:n]
    return out


def resample_s2_waveform(s2_wave: np.ndarray, target_len: int = S2_TARGET_LEN) -> np.ndarray:
    """
    Resamples variable-length S2 waveform (191 to 1484 bins) to fixed target_len=500 using linear interpolation.
    Preserves pulse integral and peak morphology.
    """
    n_in = len(s2_wave)
    if n_in == target_len:
        return s2_wave.astype(np.float32)
    if n_in == 0:
        return np.zeros(target_len, dtype=np.float32)
    if n_in == 1:
        out = np.zeros(target_len, dtype=np.float32)
        out[0] = s2_wave[0]
        return out
        
    x_orig = np.linspace(0.0, 1.0, n_in, endpoint=True)
    x_target = np.linspace(0.0, 1.0, target_len, endpoint=True)
    
    # 1D linear interpolation
    resampled = np.interp(x_target, x_orig, s2_wave).astype(np.float32)
    
    # Rescale by length ratio so area under curve is preserved
    area_scale = n_in / float(target_len)
    return resampled * area_scale


def build_multimodal_vector(
    s1_hits: np.ndarray,
    s2_hits: np.ndarray,
    hits_top: np.ndarray,
    hits_bot: np.ndarray,
    s1_max_len: int = S1_MAX_LEN,
    s2_target_len: int = S2_TARGET_LEN
) -> np.ndarray:
    """
    Constructs the standard 908-dimensional multimodal input vector:
    [s1_padded (160) | s2_resampled (500) | hits_top (127) | hits_bot (121)]
    """
    s1_proc = pad_s1_waveform(s1_hits, max_len=s1_max_len)
    s2_proc = resample_s2_waveform(s2_hits, target_len=s2_target_len)
    
    top_proc = hits_top.astype(np.float32)
    bot_proc = hits_bot.astype(np.float32)
    
    vec = np.concatenate([s1_proc, s2_proc, top_proc, bot_proc], axis=0).astype(np.float32)
    assert len(vec) == MULTIMODAL_DIM, f"Expected multimodal dimension {MULTIMODAL_DIM}, got {len(vec)}"
    return vec


class XenonPreprocessor:
    """
    Leak-free feature and waveform preprocessor.
    Must be fitted STRICTLY on the training split.
    """
    def __init__(self):
        self.is_fitted: bool = False
        self.physics_mean: Optional[np.ndarray] = None
        self.physics_std: Optional[np.ndarray] = None
        self.top_pmt_mean: Optional[np.ndarray] = None
        self.top_pmt_std: Optional[np.ndarray] = None
        self.bot_pmt_mean: Optional[np.ndarray] = None
        self.bot_pmt_std: Optional[np.ndarray] = None

    def fit(self, train_physics_features: np.ndarray, train_top_pmt: Optional[np.ndarray] = None, train_bot_pmt: Optional[np.ndarray] = None) -> "XenonPreprocessor":
        """
        Fits scalers on training data only.
        """
        self.physics_mean = np.mean(train_physics_features, axis=0).astype(np.float32)
        self.physics_std = np.std(train_physics_features, axis=0).astype(np.float32)
        # Avoid division by zero
        self.physics_std[self.physics_std < 1e-6] = 1.0
        
        if train_top_pmt is not None:
            top_log = np.log1p(np.maximum(0.0, train_top_pmt))
            self.top_pmt_mean = np.mean(top_log, axis=0).astype(np.float32)
            self.top_pmt_std = np.std(top_log, axis=0).astype(np.float32)
            self.top_pmt_std[self.top_pmt_std < 1e-6] = 1.0
            
        if train_bot_pmt is not None:
            bot_log = np.log1p(np.maximum(0.0, train_bot_pmt))
            self.bot_pmt_mean = np.mean(bot_log, axis=0).astype(np.float32)
            self.bot_pmt_std = np.std(bot_log, axis=0).astype(np.float32)
            self.bot_pmt_std[self.bot_pmt_std < 1e-6] = 1.0
            
        self.is_fitted = True
        return self

    def transform_physics(self, physics_features: np.ndarray) -> np.ndarray:
        """Standardizes 13D physics features using training parameters."""
        if not self.is_fitted:
            return physics_features.astype(np.float32)
        return ((physics_features - self.physics_mean) / self.physics_std).astype(np.float32)

    def transform_multimodal(
        self,
        s1_hits: np.ndarray,
        s2_hits: np.ndarray,
        hits_top: np.ndarray,
        hits_bot: np.ndarray
    ) -> np.ndarray:
        """Builds and normalizes the 908D multimodal vector."""
        s1_proc = pad_s1_waveform(s1_hits, S1_MAX_LEN)
        s2_proc = resample_s2_waveform(s2_hits, S2_TARGET_LEN)
        
        top_proc = hits_top.astype(np.float32)
        bot_proc = hits_bot.astype(np.float32)
        
        if self.is_fitted and self.top_pmt_mean is not None:
            top_proc = (np.log1p(np.maximum(0.0, top_proc)) - self.top_pmt_mean) / self.top_pmt_std
        if self.is_fitted and self.bot_pmt_mean is not None:
            bot_proc = (np.log1p(np.maximum(0.0, bot_proc)) - self.bot_pmt_mean) / self.bot_pmt_std
            
        return np.concatenate([s1_proc, s2_proc, top_proc, bot_proc], axis=0).astype(np.float32)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "XenonPreprocessor":
        with open(filepath, "rb") as f:
            return pickle.load(f)
