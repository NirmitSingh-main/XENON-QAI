"""
Reproducible Train / Validation / Test Splitting for XENON-QAI.
Guarantees:
1. Strict numerical event-key sorting (ev_1, ev_2, ... ev_N) avoiding lexicographical h5py pitfalls.
2. Exact class-stratified 70/15/15 partitioning with fixed seeds.
3. Zero train/test leakage.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import h5py

from xenon_qai.data.types import ALL_CLASSES, CLASS_TO_IDX, WIMP_CLASSES


def get_sorted_event_keys(h5_group: h5py.Group) -> List[str]:
    """
    Extracts and numerically sorts HDF5 event keys.
    Prevents the lexicographical ordering bug (['ev_1', 'ev_10', 'ev_100'...]).
    
    Returns:
        List of event key strings sorted by integer ID (['ev_1', 'ev_2', ... 'ev_N']).
    """
    raw_keys = list(h5_group.keys())
    return sorted(raw_keys, key=lambda k: int(k.split("_")[1]))


def get_hdf5_file_for_class(data_dir: Path, class_name: str) -> Path:
    """Resolves the exact HDF5 filepath for a given class name."""
    class_dir = data_dir / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")
    
    # Check standard filenames
    candidates = [
        class_dir / f"{class_name}_all.hdf5",
        class_dir / f"WP_{class_name}_sig1045.hdf5"
    ]
    for c in candidates:
        if c.exists():
            return c
            
    # Fallback to any hdf5 in directory
    found = list(class_dir.glob("*.hdf5"))
    if found:
        return found[0]
        
    raise FileNotFoundError(f"No HDF5 file found in {class_dir}")


def create_stratified_split(
    data_dir: Path = Path("data/extracted"),
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42,
    classes: Optional[List[str]] = None,
    max_events_per_class: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates a deterministic, stratified train/validation/test split.
    
    Args:
        data_dir: Path to extracted dataset directory.
        split_ratios: (train_frac, val_frac, test_frac) summing to 1.0.
        random_seed: Global random seed for reproducible permutation.
        classes: List of class names to include (default: all 12 classes).
        max_events_per_class: Optional cap on events per class (for fast testing/debugging).
        
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing a list of event metadata dicts:
        {
            'class': str,
            'file_path': str,
            'event_key': str,
            'event_id': int,
            'class_idx': int,
            'is_signal': int
        }
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-5, f"Split ratios must sum to 1.0, got {split_ratios}"
    train_frac, val_frac, test_frac = split_ratios
    
    if classes is None:
        classes = list(ALL_CLASSES)
        
    rng = np.random.RandomState(random_seed)
    
    splits: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": []
    }
    
    for class_name in classes:
        h5_path = get_hdf5_file_for_class(data_dir, class_name)
        with h5py.File(h5_path, "r") as f:
            first_group_name = list(f.keys())[0]
            sorted_keys = get_sorted_event_keys(f[first_group_name])
            
        n_events = len(sorted_keys)
        if max_events_per_class is not None and max_events_per_class < n_events:
            sorted_keys = sorted_keys[:max_events_per_class]
            n_events = len(sorted_keys)
            
        indices = np.arange(n_events)
        rng.shuffle(indices)
        
        n_train = int(n_events * train_frac)
        n_val = int(n_events * val_frac)
        # Remaining goes to test to ensure exact total conservation
        n_test = n_events - n_train - n_val
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        is_sig = 1 if class_name in WIMP_CLASSES else 0
        cls_idx = CLASS_TO_IDX[class_name]
        
        for idx_array, split_name in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
            for i in idx_array:
                k = sorted_keys[i]
                event_id = int(k.split("_")[1])
                splits[split_name].append({
                    "class": class_name,
                    "file_path": str(h5_path),
                    "event_key": k,
                    "event_id": event_id,
                    "class_idx": cls_idx,
                    "is_signal": is_sig
                })
                
    return splits


def save_split_manifest(splits: Dict[str, List[Dict[str, Any]]], output_path: Path) -> None:
    """Saves split manifest to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)


def load_split_manifest(manifest_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Loads split manifest from JSON file."""
    with open(manifest_path, "r") as f:
        return json.load(f)
