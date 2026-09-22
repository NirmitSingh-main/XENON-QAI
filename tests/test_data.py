"""
Unit and Integration Tests for XENON-QAI Dataset Pipeline.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xenon_qai.data import (
    ALL_CLASSES,
    WIMP_CLASSES,
    BACKGROUND_CLASSES,
    PHYSICS_FEATURE_DIM,
    MULTIMODAL_DIM,
    S1_MAX_LEN,
    S2_TARGET_LEN,
    get_sorted_event_keys,
    create_stratified_split,
    validate_split_integrity,
    compute_physics_features,
    pad_s1_waveform,
    resample_s2_waveform,
    build_multimodal_vector,
    XenonPreprocessor,
    XenonHDF5Dataset,
    create_dataloaders,
    validate_batch
)

DATA_DIR = Path("data/extracted")


def test_numerical_sorting():
    """Verifies that HDF5 keys are sorted numerically (ev_1, ev_2, ... ev_10) not lexicographically."""
    import h5py
    ac_file = DATA_DIR / "AC" / "AC_all.hdf5"
    if not ac_file.exists():
        pytest.skip("Dataset not found")
        
    with h5py.File(ac_file, "r") as f:
        grp = f["HITS_top"]
        sorted_keys = get_sorted_event_keys(grp)
        
        # Check first 15 keys
        expected = [f"ev_{i}" for i in range(1, 16)]
        assert sorted_keys[:15] == expected, f"Expected {expected}, got {sorted_keys[:15]}"
        
        # Check integer IDs are monotonically increasing
        int_ids = [int(k.split("_")[1]) for k in sorted_keys]
        assert int_ids == sorted(int_ids)
        assert len(int_ids) == len(set(int_ids))


def test_physics_feature_computation():
    """Verifies 13D physics feature vector calculation."""
    s1_hits = np.array([0.0, 1.2, 3.5, 2.1, 0.0], dtype=np.float32)
    s2_hits = np.array([0.0, 10.5, 45.2, 80.1, 30.0, 5.0, 0.0], dtype=np.float32)
    s1_raw = s1_hits + 0.05
    s2_raw = s2_hits + 0.1
    top_pmt = np.ones(127, dtype=np.float32) * 2.0
    bot_pmt = np.ones(121, dtype=np.float32) * 1.0
    
    feats = compute_physics_features(s1_hits, s2_hits, s1_raw, s2_raw, top_pmt, bot_pmt)
    
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (PHYSICS_FEATURE_DIM,)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()
    
    # Check physics invariants
    s1_area = feats[0]
    s2_area = feats[1]
    log_s2_s1 = feats[4]
    top_sum = feats[9]
    bot_sum = feats[10]
    total_pmts = feats[11]
    asym = feats[12]
    
    assert np.isclose(s1_area, np.sum(s1_hits))
    assert np.isclose(s2_area, np.sum(s2_hits))
    assert np.isclose(top_sum, 127 * 2.0)
    assert np.isclose(bot_sum, 121 * 1.0)
    assert np.isclose(total_pmts, top_sum + bot_sum)
    assert np.isclose(asym, (top_sum - bot_sum) / total_pmts)


def test_waveform_preprocessing():
    """Verifies waveform padding and resampling."""
    # S1 padding
    s1_short = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    s1_padded = pad_s1_waveform(s1_short, max_len=S1_MAX_LEN)
    assert s1_padded.shape == (S1_MAX_LEN,)
    assert np.array_equal(s1_padded[:3], s1_short)
    assert np.all(s1_padded[3:] == 0.0)
    
    # S2 resampling
    s2_raw = np.sin(np.linspace(0, np.pi, 300)).astype(np.float32) * 50.0
    s2_resampled = resample_s2_waveform(s2_raw, target_len=S2_TARGET_LEN)
    assert s2_resampled.shape == (S2_TARGET_LEN,)
    assert not np.isnan(s2_resampled).any()
    
    # Multimodal concatenation
    top_pmt = np.ones(127, dtype=np.float32)
    bot_pmt = np.ones(121, dtype=np.float32)
    multi_vec = build_multimodal_vector(s1_short, s2_raw, top_pmt, bot_pmt)
    assert multi_vec.shape == (MULTIMODAL_DIM,)
    assert multi_vec.shape == (908,)


def test_stratified_split_integrity():
    """Verifies 70/15/15 stratified split disjointness and reproducibility."""
    splits = create_stratified_split(
        data_dir=DATA_DIR,
        split_ratios=(0.70, 0.15, 0.15),
        random_seed=42,
        max_events_per_class=100  # fast test
    )
    
    val_info = validate_split_integrity(splits)
    assert val_info["is_disjoint"] is True
    assert val_info["train_val_overlap"] == 0
    assert val_info["train_test_overlap"] == 0
    assert val_info["val_test_overlap"] == 0
    
    # Check that all 12 classes are present in every split
    assert len(val_info["class_counts_train"]) == len(ALL_CLASSES)
    assert len(val_info["class_counts_val"]) == len(ALL_CLASSES)
    assert len(val_info["class_counts_test"]) == len(ALL_CLASSES)
    
    # Check reproducibility with same seed
    splits2 = create_stratified_split(
        data_dir=DATA_DIR,
        split_ratios=(0.70, 0.15, 0.15),
        random_seed=42,
        max_events_per_class=100
    )
    assert splits["train"][0] == splits2["train"][0]
    assert splits["test"][-1] == splits2["test"][-1]


def test_dataloader_end_to_end():
    """Runs an end-to-end DataLoader test with batching and feature transformations."""
    train_loader, val_loader, test_loader, preprocessor, splits = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=16,
        mode="both",
        split_ratios=(0.70, 0.15, 0.15),
        random_seed=42,
        num_workers=0,
        fit_preprocessor=True,
        max_events_per_class=50
    )
    
    assert preprocessor is not None
    assert preprocessor.is_fitted is True
    
    # Test batch from train_loader
    for batch in train_loader:
        x_multi, x_phys, y_sig, y_cls = batch
        assert x_multi.shape == (16, MULTIMODAL_DIM)
        assert x_phys.shape == (16, PHYSICS_FEATURE_DIM)
        assert y_sig.shape == (16,)
        assert y_cls.shape == (16,)
        
        # Validate data integrity
        is_valid, msg = validate_batch(batch, expected_mode="both")
        assert is_valid, msg
        
        # Check label constraints
        assert torch.all((y_sig == 0) | (y_sig == 1))
        assert torch.all((y_cls >= 0) & (y_cls < len(ALL_CLASSES)))
        break

    # Test batch from test_loader
    for batch in test_loader:
        x_multi, x_phys, y_sig, y_cls = batch
        assert x_multi.shape[1] == MULTIMODAL_DIM
        assert x_phys.shape[1] == PHYSICS_FEATURE_DIM
        break


if __name__ == "__main__":
    print("Running manual test execution...")
    test_numerical_sorting()
    print("✓ Numerical sorting passed")
    test_physics_feature_computation()
    print("✓ Physics features passed")
    test_waveform_preprocessing()
    print("✓ Waveform preprocessing passed")
    test_stratified_split_integrity()
    print("✓ Stratified split integrity passed")
    test_dataloader_end_to_end()
    print("✓ End-to-end DataLoader test passed")
    print("\nALL TESTS PASSED SUCCESSFULLY!")
