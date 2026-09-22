"""
DataLoader Factory and Pipeline Orchestrator for XENON-QAI.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from xenon_qai.data.types import (
    ALL_CLASSES,
    CLASS_TO_IDX,
    WIMP_CLASSES,
    DatasetConfig
)
from xenon_qai.data.dataset import XenonHDF5Dataset
from xenon_qai.data.splitting import create_stratified_split, save_split_manifest, load_split_manifest
from xenon_qai.data.preprocessing import XenonPreprocessor
from xenon_qai.data.features import compute_physics_features


def _fit_preprocessor_on_train_split(
    train_events: List[Dict[str, Any]],
    sample_size: int = 5000,
    random_seed: int = 42
) -> XenonPreprocessor:
    """
    Fits XenonPreprocessor strictly on training events.
    To be memory and time efficient, draws a balanced sample from the training manifest.
    """
    rng = np.random.RandomState(random_seed)
    n_train = len(train_events)
    sample_indices = np.arange(n_train)
    if n_train > sample_size:
        sample_indices = rng.choice(n_train, size=sample_size, replace=False)
        
    physics_list = []
    top_pmt_list = []
    bot_pmt_list = []
    
    # Cache handles during fitting
    handles: Dict[str, h5py.File] = {}
    
    try:
        for idx in sample_indices:
            item = train_events[idx]
            fpath = item["file_path"]
            k = item["event_key"]
            if fpath not in handles:
                handles[fpath] = h5py.File(fpath, "r")
            f = handles[fpath]
            
            s1_hits = f["peakS1_onlyhits"][k][()]
            s2_hits = f["peakS2_onlyhits"][k][()]
            s1_raw = f["peakS1_raw"][k][()]
            s2_raw = f["peakS2_raw"][k][()]
            hits_top = f["HITS_top"][k][()]
            hits_bot = f["HITS_bot"][k][()]
            
            feat = compute_physics_features(s1_hits, s2_hits, s1_raw, s2_raw, hits_top, hits_bot)
            physics_list.append(feat)
            top_pmt_list.append(hits_top)
            bot_pmt_list.append(hits_bot)
    finally:
        for h in handles.values():
            h.close()
            
    preprocessor = XenonPreprocessor()
    preprocessor.fit(
        train_physics_features=np.array(physics_list, dtype=np.float32),
        train_top_pmt=np.array(top_pmt_list, dtype=np.float32),
        train_bot_pmt=np.array(bot_pmt_list, dtype=np.float32)
    )
    return preprocessor


def create_dataloaders(
    data_dir: Path = Path("data/extracted"),
    batch_size: int = 64,
    mode: str = "multimodal",  # 'multimodal', 'physics', or 'both'
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42,
    num_workers: int = 0,
    fit_preprocessor: bool = True,
    classes: Optional[List[str]] = None,
    max_events_per_class: Optional[int] = None,
    return_dict: bool = False,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[XenonPreprocessor], Dict[str, List[Dict[str, Any]]]]:
    """
    End-to-end DataLoader factory for XENON-QAI.
    
    Returns:
        (train_loader, val_loader, test_loader, preprocessor, split_manifest)
    """
    splits = create_stratified_split(
        data_dir=data_dir,
        split_ratios=split_ratios,
        random_seed=random_seed,
        classes=classes,
        max_events_per_class=max_events_per_class
    )
    
    preprocessor: Optional[XenonPreprocessor] = None
    if fit_preprocessor:
        preprocessor = _fit_preprocessor_on_train_split(
            train_events=splits["train"],
            sample_size=min(5000, len(splits["train"])),
            random_seed=random_seed
        )
        
    train_ds = XenonHDF5Dataset(
        event_manifest=splits["train"],
        mode=mode,
        preprocessor=preprocessor,
        return_dict=return_dict
    )
    val_ds = XenonHDF5Dataset(
        event_manifest=splits["val"],
        mode=mode,
        preprocessor=preprocessor,
        return_dict=return_dict
    )
    test_ds = XenonHDF5Dataset(
        event_manifest=splits["test"],
        mode=mode,
        preprocessor=preprocessor,
        return_dict=return_dict
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, preprocessor, splits
