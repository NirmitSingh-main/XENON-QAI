"""
PyTorch Dataset and DataLoader Implementation for XENON-QAI.
Provides:
1. Lazy HDF5 event loading without loading 65+ GB into memory.
2. Safe multi-worker HDF5 handle lifecycle management.
3. Dual-mode output: 13D Physics Features, 908D Multimodal Tensors, or Both.
4. Correct numerical key indexing and label encoding.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from xenon_qai.data.types import (
    ALL_CLASSES,
    CLASS_TO_IDX,
    WIMP_CLASSES,
    S1_MAX_LEN,
    S2_TARGET_LEN,
    PHYSICS_FEATURE_DIM,
    MULTIMODAL_DIM
)
from xenon_qai.data.features import compute_physics_features
from xenon_qai.data.preprocessing import (
    build_multimodal_vector,
    pad_s1_waveform,
    resample_s2_waveform,
    XenonPreprocessor
)


class XenonHDF5Dataset(Dataset):
    """
    Lazy PyTorch Dataset for XENON LXe-TPC event data.
    
    Reads directly from HDF5 archives on demand. Safe for multi-process PyTorch DataLoaders.
    """
    def __init__(
        self,
        event_manifest: List[Dict[str, Any]],
        mode: str = "multimodal",  # 'multimodal', 'physics', or 'both'
        preprocessor: Optional[XenonPreprocessor] = None,
        return_dict: bool = False
    ):
        """
        Args:
            event_manifest: List of event metadata dicts containing file_path, event_key, class_idx, is_signal.
            mode: 'multimodal' (908D), 'physics' (13D), or 'both'.
            preprocessor: Optional fitted XenonPreprocessor for standardizing features.
            return_dict: If True, returns a structured dictionary instead of (x, y_binary, y_multiclass) tuple.
        """
        super().__init__()
        self.events: List[Dict[str, Any]] = event_manifest
        self.mode: str = mode
        self.preprocessor: Optional[XenonPreprocessor] = preprocessor
        self.return_dict: bool = return_dict
        
        # Per-worker / per-process lazy HDF5 file handles
        self._h5_handles: Dict[str, h5py.File] = {}

    def _get_h5_file(self, file_path: str) -> h5py.File:
        """Lazily opens and caches HDF5 file handles per process worker."""
        if file_path not in self._h5_handles:
            self._h5_handles[file_path] = h5py.File(file_path, "r", swmr=True)
        return self._h5_handles[file_path]

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, ...], Dict[str, Any]]:
        item = self.events[idx]
        file_path = item["file_path"]
        event_key = item["event_key"]
        is_signal = item["is_signal"]
        class_idx = item["class_idx"]
        class_name = item["class"]
        event_id = item["event_id"]
        
        f = self._get_h5_file(file_path)
        
        # Extract raw arrays from HDF5 groups
        s1_hits = f["peakS1_onlyhits"][event_key][()]
        s2_hits = f["peakS2_onlyhits"][event_key][()]
        s1_raw = f["peakS1_raw"][event_key][()]
        s2_raw = f["peakS2_raw"][event_key][()]
        hits_top = f["HITS_top"][event_key][()]
        hits_bot = f["HITS_bot"][event_key][()]
        
        # Construct requested representations
        physics_feat: Optional[np.ndarray] = None
        multimodal_vec: Optional[np.ndarray] = None
        
        if self.mode in ("physics", "both"):
            physics_feat = compute_physics_features(
                s1_hits=s1_hits,
                s2_hits=s2_hits,
                s1_raw=s1_raw,
                s2_raw=s2_raw,
                hits_top=hits_top,
                hits_bot=hits_bot
            )
            if self.preprocessor is not None and self.preprocessor.is_fitted:
                physics_feat = self.preprocessor.transform_physics(physics_feat)
                
        if self.mode in ("multimodal", "both"):
            if self.preprocessor is not None and self.preprocessor.is_fitted:
                multimodal_vec = self.preprocessor.transform_multimodal(
                    s1_hits=s1_hits,
                    s2_hits=s2_hits,
                    hits_top=hits_top,
                    hits_bot=hits_bot
                )
            else:
                multimodal_vec = build_multimodal_vector(
                    s1_hits=s1_hits,
                    s2_hits=s2_hits,
                    hits_top=hits_top,
                    hits_bot=hits_bot
                )

        y_sig = torch.tensor(is_signal, dtype=torch.long)
        y_cls = torch.tensor(class_idx, dtype=torch.long)
        
        if self.return_dict:
            res: Dict[str, Any] = {
                "is_signal": y_sig,
                "class_idx": y_cls,
                "class_name": class_name,
                "event_id": event_id
            }
            if physics_feat is not None:
                res["physics_features"] = torch.from_numpy(physics_feat)
            if multimodal_vec is not None:
                res["multimodal"] = torch.from_numpy(multimodal_vec)
                # Also include structured components
                res["s1"] = torch.from_numpy(pad_s1_waveform(s1_hits, S1_MAX_LEN))
                res["s2"] = torch.from_numpy(resample_s2_waveform(s2_hits, S2_TARGET_LEN))
                res["top_pmt"] = torch.from_numpy(hits_top.astype(np.float32))
                res["bot_pmt"] = torch.from_numpy(hits_bot.astype(np.float32))
            return res
            
        if self.mode == "physics":
            return torch.from_numpy(physics_feat), y_sig, y_cls
        elif self.mode == "multimodal":
            return torch.from_numpy(multimodal_vec), y_sig, y_cls
        else:  # both
            return torch.from_numpy(multimodal_vec), torch.from_numpy(physics_feat), y_sig, y_cls

    def close(self) -> None:
        """Closes all open HDF5 file handles."""
        for h in self._h5_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._h5_handles.clear()

    def __del__(self) -> None:
        self.close()
