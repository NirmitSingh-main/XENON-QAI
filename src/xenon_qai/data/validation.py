"""
Validation and Integrity Checks for XENON-QAI Dataset Pipeline.
"""

from typing import Dict, List, Any, Tuple
import numpy as np

from xenon_qai.data.types import ALL_CLASSES, PHYSICS_FEATURE_DIM, MULTIMODAL_DIM


def validate_split_integrity(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Validates split disjointness, coverage, and class distribution.
    
    Returns:
        Dictionary of validation results and integrity checks.
    """
    train_keys = set((item["class"], item["event_key"]) for item in splits["train"])
    val_keys = set((item["class"], item["event_key"]) for item in splits["val"])
    test_keys = set((item["class"], item["event_key"]) for item in splits["test"])
    
    train_val_overlap = len(train_keys.intersection(val_keys))
    train_test_overlap = len(train_keys.intersection(test_keys))
    val_test_overlap = len(val_keys.intersection(test_keys))
    
    total_events = len(train_keys) + len(val_keys) + len(test_keys)
    is_disjoint = (train_val_overlap == 0) and (train_test_overlap == 0) and (val_test_overlap == 0)
    
    class_counts_train: Dict[str, int] = {}
    class_counts_val: Dict[str, int] = {}
    class_counts_test: Dict[str, int] = {}
    
    for item in splits["train"]:
        class_counts_train[item["class"]] = class_counts_train.get(item["class"], 0) + 1
    for item in splits["val"]:
        class_counts_val[item["class"]] = class_counts_val.get(item["class"], 0) + 1
    for item in splits["test"]:
        class_counts_test[item["class"]] = class_counts_test.get(item["class"], 0) + 1
        
    return {
        "is_disjoint": is_disjoint,
        "train_count": len(train_keys),
        "val_count": len(val_keys),
        "test_count": len(test_keys),
        "total_count": total_events,
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "class_counts_train": class_counts_train,
        "class_counts_val": class_counts_val,
        "class_counts_test": class_counts_test
    }


def validate_batch(
    batch: Any,
    expected_mode: str = "multimodal"
) -> Tuple[bool, str]:
    """
    Validates shapes, dtypes, and NaN/Inf status of a loaded PyTorch batch.
    """
    if expected_mode == "multimodal":
        x, y_sig, y_cls = batch
        if len(x.shape) != 2 or x.shape[1] != MULTIMODAL_DIM:
            return False, f"Expected multimodal shape (B, {MULTIMODAL_DIM}), got {x.shape}"
    elif expected_mode == "physics":
        x, y_sig, y_cls = batch
        if len(x.shape) != 2 or x.shape[1] != PHYSICS_FEATURE_DIM:
            return False, f"Expected physics shape (B, {PHYSICS_FEATURE_DIM}), got {x.shape}"
    elif expected_mode == "both":
        x_multi, x_phys, y_sig, y_cls = batch
        if x_multi.shape[1] != MULTIMODAL_DIM or x_phys.shape[1] != PHYSICS_FEATURE_DIM:
            return False, f"Shapes mismatch: multimodal {x_multi.shape}, physics {x_phys.shape}"
        x = x_multi
    elif isinstance(batch, dict):
        if "multimodal" in batch and batch["multimodal"].shape[1] != MULTIMODAL_DIM:
            return False, f"Dict multimodal shape mismatch: {batch['multimodal'].shape}"
        if "physics_features" in batch and batch["physics_features"].shape[1] != PHYSICS_FEATURE_DIM:
            return False, f"Dict physics shape mismatch: {batch['physics_features'].shape}"
        x = batch.get("multimodal", batch.get("physics_features"))
    else:
        return False, f"Unknown batch format: {type(batch)}"
        
    if np.isnan(x.detach().cpu().numpy()).any():
        return False, "NaN values detected in batch tensor"
    if np.isinf(x.detach().cpu().numpy()).any():
        return False, "Inf values detected in batch tensor"
        
    return True, "Batch is valid"
