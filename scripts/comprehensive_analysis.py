import os
import sys
import glob
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
from PIL import Image

DATA_DIR = Path("data/extracted")

print("=" * 80)
print("XENON-QAI COMPREHENSIVE DATASET ANALYSIS")
print("=" * 80)

# 1. Inspect all HDF5 files
hdf5_files = sorted(DATA_DIR.rglob("*.hdf5"))
print(f"Total HDF5 files found: {len(hdf5_files)}")

hdf5_summary = {}

for hdf5_path in hdf5_files:
    class_name = hdf5_path.parent.name
    file_size_mb = hdf5_path.stat().st_size / (1024 * 1024)
    print(f"\nAnalyzing {class_name} ({hdf5_path.name}) - {file_size_mb:.2f} MB...")
    
    with h5py.File(hdf5_path, "r") as f:
        groups = list(f.keys())
        counts = {g: len(f[g]) for g in groups}
        
        # Check event key consistency across groups
        first_group_keys = set(f[groups[0]].keys())
        all_match = all(set(f[g].keys()) == first_group_keys for g in groups)
        num_events = len(first_group_keys)
        
        # Check key format and sorting
        sample_keys = list(f[groups[0]].keys())[:5]
        
        hdf5_summary[class_name] = {
            "path": str(hdf5_path),
            "size_mb": file_size_mb,
            "num_events": num_events,
            "groups": groups,
            "keys_match_across_groups": all_match,
            "sample_keys": sample_keys
        }
        print(f"  Events: {num_events}, Groups: {groups}, Keys strictly match across groups: {all_match}")

print("\n" + "=" * 80)
print("HDF5 EVENT COUNTS SUMMARY:")
total_hdf5_events = 0
for k, v in hdf5_summary.items():
    print(f"  {k:15s} : {v['num_events']:6d} events ({v['size_mb']:8.2f} MB)")
    total_hdf5_events += v['num_events']
print(f"  {'TOTAL':15s} : {total_hdf5_events:6d} events")

# 2. Inspect cs1cs2 files
cs1cs2_dir = DATA_DIR / "cs1cs2"
print("\n" + "=" * 80)
print("CS1/CS2 SUMMARY:")
cs1cs2_summary = {}
if cs1cs2_dir.exists():
    for fpath in sorted(cs1cs2_dir.glob("*.txt")):
        with open(fpath, "r") as f:
            lines = [float(l.strip()) for l in f if l.strip()]
        arr = np.array(lines)
        cs1cs2_summary[fpath.name] = {
            "count": len(arr),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "zeros": int(np.sum(arr == 0)),
            "nans": int(np.isnan(arr).sum())
        }
        print(f"  {fpath.name:15s}: count={len(arr):6d}, min={arr.min():10.4f}, max={arr.max():10.4f}, median={np.median(arr):10.4f}, mean={arr.mean():10.4f}")

# 3. Inspect 00-images directory
img_dir = DATA_DIR / "00-images"
print("\n" + "=" * 80)
print("00-IMAGES SUMMARY:")
img_summary = {}
if img_dir.exists():
    for sub in sorted(img_dir.iterdir()):
        if not sub.is_dir():
            continue
        ok_file = sub / "ok-index.txt"
        ok_count = 0
        if ok_file.exists():
            with open(ok_file) as f:
                ok_count = len([l for l in f if l.strip()])
        
        subdirs = [d.name for d in sub.iterdir() if d.is_dir()]
        counts = {}
        img_stats = {}
        for dname in subdirs:
            folder = sub / dname
            pngs = list(folder.glob("*.png"))
            counts[dname] = len(pngs)
            if pngs and dname not in img_stats:
                # Sample one image
                with Image.open(pngs[0]) as im:
                    im_arr = np.array(im)
                    img_stats[dname] = {
                        "size": im.size,
                        "mode": im.mode,
                        "shape": im_arr.shape,
                        "dtype": str(im_arr.dtype),
                        "min": int(im_arr.min()),
                        "max": int(im_arr.max())
                    }
        
        img_summary[sub.name] = {
            "ok_count": ok_count,
            "counts": counts,
            "stats": img_stats
        }
        print(f"  {sub.name:15s}: ok-index={ok_count:5d}, resolutions={counts}")
        for rname, rstat in img_stats.items():
            print(f"      {rname}: size={rstat['size']}, mode={rstat['mode']}, shape={rstat['shape']}, min={rstat['min']}, max={rstat['max']}")

