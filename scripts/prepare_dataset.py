"""
XENON-QAI Dataset Preparation & Verification Entry Point.
Executes stratified splitting, preprocessor fitting, and DataLoader verification.
"""

import sys
from pathlib import Path
import torch

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xenon_qai.data import (
    create_dataloaders,
    validate_split_integrity,
    save_split_manifest,
    ALL_CLASSES,
    PHYSICS_FEATURE_DIM,
    MULTIMODAL_DIM
)


def main():
    print("=" * 80)
    print("XENON-QAI DATASET PIPELINE VERIFICATION")
    print("=" * 80)
    
    data_dir = Path("data/extracted")
    if not data_dir.exists():
        print(f"ERROR: Dataset path not found: {data_dir}")
        return 1
        
    print(f"Data Directory: {data_dir.resolve()}")
    print(f"Classes ({len(ALL_CLASSES)}): {ALL_CLASSES}\n")
    
    print("1. Creating stratified 70/15/15 DataLoaders (mode='both')...")
    train_loader, val_loader, test_loader, preprocessor, splits = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        mode="both",
        split_ratios=(0.70, 0.15, 0.15),
        random_seed=42,
        num_workers=0,
        fit_preprocessor=True,
        max_events_per_class=200  # fast verification sample across all 12 classes
    )
    
    # 2. Split integrity
    split_info = validate_split_integrity(splits)
    print("\n2. Split Validation Results:")
    print(f"  Is Disjoint          : {split_info['is_disjoint']}")
    print(f"  Train Events         : {split_info['train_count']}")
    print(f"  Validation Events    : {split_info['val_count']}")
    print(f"  Test Events          : {split_info['test_count']}")
    print(f"  Total Sampled Events : {split_info['total_count']}")
    print(f"  Train/Val Overlap    : {split_info['train_val_overlap']}")
    print(f"  Train/Test Overlap   : {split_info['train_test_overlap']}")
    print(f"  Val/Test Overlap     : {split_info['val_test_overlap']}")
    
    # Save manifest artifact
    manifest_path = Path("results/split_manifest.json")
    save_split_manifest(splits, manifest_path)
    print(f"  Saved split manifest to: {manifest_path}")
    
    # 3. DataLoader Batch Inspection
    print("\n3. Inspecting DataLoader Batches:")
    for split_name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
        x_multi, x_phys, y_sig, y_cls = next(iter(loader))
        print(f"  [{split_name:10s}] Batches: {len(loader):3d} | Multimodal Shape: {tuple(x_multi.shape)} | Physics Shape: {tuple(x_phys.shape)} | Signals in batch: {int(y_sig.sum())}/{len(y_sig)}")
        assert x_multi.shape[1] == MULTIMODAL_DIM, f"Multimodal dim mismatch: {x_multi.shape}"
        assert x_phys.shape[1] == PHYSICS_FEATURE_DIM, f"Physics dim mismatch: {x_phys.shape}"
        assert not torch.isnan(x_multi).any(), "NaN found in multimodal tensor"
        assert not torch.isnan(x_phys).any(), "NaN found in physics tensor"
        
    print("\n" + "=" * 80)
    print("PIPELINE VERIFICATION SUCCESSFUL")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
