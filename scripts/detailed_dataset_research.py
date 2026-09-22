import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from collections import defaultdict
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/extracted")

def print_section(title):
    print("\n" + "=" * 80, flush=True)
    print(f" {title.upper()}", flush=True)
    print("=" * 80, flush=True)

# -----------------------------------------------------------------------------
# 4. DEEP DIVE INTO HDF5 REPRESENTATIONS PER CLASS
# -----------------------------------------------------------------------------
print_section("4. Detailed Inspection of HDF5 Groups & Representations")

sample_classes = ["ER", "RN", "WALL", "AC", "CNNS", "mDM10GeV", "mDM50GeV", "mDM500GeV"]
rep_stats = defaultdict(lambda: defaultdict(list))

for cname in sample_classes:
    hpath = DATA_DIR / cname / (f"{cname}_all.hdf5" if (DATA_DIR / cname / f"{cname}_all.hdf5").exists() else f"WP_{cname}_sig1045.hdf5")
    if not hpath.exists():
        continue
    
    print(f"\n--- Sampling 200 events from {cname} ({hpath.name}) ---", flush=True)
    with h5py.File(hpath, "r") as f:
        groups = list(f.keys())
        keys = list(f[groups[0]].keys())[:200]
        
        for g in groups:
            shapes = []
            dtypes = []
            mins = []
            maxs = []
            means = []
            zeros = []
            nans = []
            negs = []
            
            for k in keys:
                ds = f[g][k]
                arr = ds[()]
                shapes.append(arr.shape)
                dtypes.append(str(arr.dtype))
                mins.append(float(np.min(arr)))
                maxs.append(float(np.max(arr)))
                means.append(float(np.mean(arr)))
                zeros.append(float(np.mean(arr == 0)))
                nans.append(int(np.isnan(arr).sum()) + int(np.isinf(arr).sum()))
                negs.append(float(np.mean(arr < 0)))
            
            len_min = min([s[0] for s in shapes])
            len_max = max([s[0] for s in shapes])
            len_mean = np.mean([s[0] for s in shapes])
            val_min = min(mins)
            val_max = max(maxs)
            val_mean = np.mean(means)
            zero_frac = np.mean(zeros)
            neg_frac = np.mean(negs)
            nan_count = sum(nans)
            
            shape_str = str(shapes[0]) if len_min == len_max else f"Var({len_min}-{len_max}, mean={len_mean:.1f})"
            print(f"  [{g:16s}] Shape: {shape_str:20s} | Dtype: {dtypes[0]:8s} | ValRange: [{val_min:8.2f}, {val_max:8.2f}] | Zero%: {zero_frac*100:5.1f}% | Neg%: {neg_frac*100:5.1f}% | NaNs: {nan_count}", flush=True)

# -----------------------------------------------------------------------------
# 5. EXACT INVESTIGATION OF HITS_bot_img & HITS_top_img
# -----------------------------------------------------------------------------
print_section("5. Investigation of HITS_bot_img & HITS_top_img in HDF5")

test_path = DATA_DIR / "ER" / "ER_all.hdf5"
with h5py.File(test_path, "r") as f:
    for gname in ["HITS_top_img", "HITS_bot_img"]:
        sample_keys = list(f[gname].keys())[:10]
        print(f"\nGroup {gname} analysis across 10 sample events:")
        for k in sample_keys:
            arr = f[gname][k][()]
            uniques, counts = np.unique(arr, return_counts=True)
            print(f"  Event {k}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, unique_values_count={len(uniques)}")
            if len(uniques) <= 10:
                print(f"    Unique values & counts: {dict(zip(uniques, counts))}")
            else:
                top5_idx = np.argsort(-counts)[:5]
                top5 = {float(uniques[i]): int(counts[i]) for i in top5_idx}
                print(f"    Top 5 frequent values: {top5}, Non-1.0 count: {int(np.sum(arr != 1.0))}")

# -----------------------------------------------------------------------------
# 6. INVESTIGATION OF peakS1_raw vs peakS1_onlyhits AND S2
# -----------------------------------------------------------------------------
print_section("6. Comparison of _raw vs _onlyhits Waveforms")

with h5py.File(test_path, "r") as f:
    for peak_type in ["peakS1", "peakS2"]:
        raw_grp = f[f"{peak_type}_raw"]
        hits_grp = f[f"{peak_type}_onlyhits"]
        sample_keys = list(raw_grp.keys())[:5]
        print(f"\n--- {peak_type} Comparison (Raw vs OnlyHits) ---")
        for k in sample_keys:
            raw_w = raw_grp[k][()]
            hits_w = hits_grp[k][()]
            diff = raw_w - hits_w
            print(f"  Event {k}:")
            print(f"    Raw      : len={len(raw_w)}, min={raw_w.min():.4f}, max={raw_w.max():.4f}, sum={raw_w.sum():.4f}, zero_frac={(raw_w==0).mean():.2f}")
            print(f"    OnlyHits : len={len(hits_w)}, min={hits_w.min():.4f}, max={hits_w.max():.4f}, sum={hits_w.sum():.4f}, zero_frac={(hits_w==0).mean():.2f}")
            print(f"    Diff (Raw - Hits): min={diff.min():.4f}, max={diff.max():.4f}, non-zero diff frac={(diff!=0).mean():.2f}")

# -----------------------------------------------------------------------------
# 7. INVESTIGATION OF HITS_bot (121) vs HITS_top (127)
# -----------------------------------------------------------------------------
print_section("7. Investigation of HITS_bot (121 PMTs) and HITS_top (127 PMTs)")

with h5py.File(test_path, "r") as f:
    top_grp = f["HITS_top"]
    bot_grp = f["HITS_bot"]
    print("XENON1T PMT configuration: Top array has 127 PMTs, Bottom array has 121 PMTs (Total = 248 PMTs).")
    for k in list(top_grp.keys())[:5]:
        top_v = top_grp[k][()]
        bot_v = bot_grp[k][()]
        print(f"  Event {k}: Top sum={top_v.sum():.2f} (active PMTs: {(top_v>0).sum()}/127), Bot sum={bot_v.sum():.2f} (active PMTs: {(bot_v>0).sum()}/121), Total Hits={top_v.sum()+bot_v.sum():.2f}, Top/Total={top_v.sum()/(top_v.sum()+bot_v.sum()+1e-6):.3f}")

# -----------------------------------------------------------------------------
# 8. PHYSICS FEATURES EXTRACTION & CLASS COMPARISON
# -----------------------------------------------------------------------------
print_section("8. Physics Features Extraction & Separability Analysis")

# Let's extract physics features from a balanced sample across all 12 classes
feature_records = []
all_hdf5s = sorted(DATA_DIR.rglob("*.hdf5"))

np.random.seed(42)

for hpath in all_hdf5s:
    cname = hpath.parent.name
    print(f"Extracting features from {cname}...", flush=True)
    with h5py.File(hpath, "r") as f:
        keys = list(f["HITS_top"].keys())
        # sample up to 1000 events per class
        sample_size = min(1000, len(keys))
        chosen_indices = np.random.choice(len(keys), sample_size, replace=False)
        chosen_keys = [keys[i] for i in chosen_indices]
        
        for k in chosen_keys:
            top_hits = f["HITS_top"][k][()]
            bot_hits = f["HITS_bot"][k][()]
            s1_raw = f["peakS1_raw"][k][()]
            s1_hits = f["peakS1_onlyhits"][k][()]
            s2_raw = f["peakS2_raw"][k][()]
            s2_hits = f["peakS2_onlyhits"][k][()]
            
            # Physics features
            s1_area_hits = float(np.sum(s1_hits))
            s1_area_raw = float(np.sum(s1_raw))
            s2_area_hits = float(np.sum(s2_hits))
            s2_area_raw = float(np.sum(s2_raw))
            
            s1_len = len(s1_hits)
            s2_len = len(s2_hits)
            
            top_sum = float(np.sum(top_hits))
            bot_sum = float(np.sum(bot_hits))
            total_pmt_hits = top_sum + bot_sum
            top_bot_asym = (top_sum - bot_sum) / (total_pmt_hits + 1e-6)
            top_active = int(np.sum(top_hits > 0))
            bot_active = int(np.sum(bot_hits > 0))
            
            s2_s1_ratio = (s2_area_hits + 1e-3) / (s1_area_hits + 1e-3)
            log_s2_s1 = math.log10(max(1e-3, s2_s1_ratio))
            
            # S1 prompt fraction or peakiness
            s1_max = float(np.max(s1_hits)) if len(s1_hits) > 0 else 0.0
            s2_max = float(np.max(s2_hits)) if len(s2_hits) > 0 else 0.0
            
            feature_records.append({
                "class": cname,
                "is_signal": 1 if cname.startswith("mDM") else 0,
                "is_wimp": 1 if cname.startswith("mDM") else 0,
                "is_er": 1 if cname == "ER" else 0,
                "is_rn": 1 if cname == "RN" else 0,
                "is_wall": 1 if cname == "WALL" else 0,
                "is_ac": 1 if cname == "AC" else 0,
                "is_cnns": 1 if cname == "CNNS" else 0,
                "s1_area_hits": s1_area_hits,
                "s1_area_raw": s1_area_raw,
                "s2_area_hits": s2_area_hits,
                "s2_area_raw": s2_area_raw,
                "s1_len": s1_len,
                "s2_len": s2_len,
                "top_sum": top_sum,
                "bot_sum": bot_sum,
                "total_pmt_hits": total_pmt_hits,
                "top_bot_asym": top_bot_asym,
                "top_active": top_active,
                "bot_active": bot_active,
                "log_s2_s1": log_s2_s1,
                "s1_max": s1_max,
                "s2_max": s2_max
            })

df_feat = pd.DataFrame(feature_records)
print(f"\nExtracted features DataFrame: {df_feat.shape}")

print("\n--- Summary Statistics by Class ---")
class_summary = df_feat.groupby("class")[["s1_area_hits", "s2_area_hits", "s1_len", "s2_len", "log_s2_s1", "top_bot_asym", "total_pmt_hits"]].mean()
print(class_summary.to_string())

# -----------------------------------------------------------------------------
# 9. BASELINE CLASSIFIERS & SEPARABILITY BENCHMARKS
# -----------------------------------------------------------------------------
print_section("9. Baseline Classification & Separability Benchmarks")

# Task A: Binary Signal (All WIMPs) vs Backgrounds (ER, RN, WALL, AC, CNNS)
feature_cols = ["s1_area_hits", "s2_area_hits", "s1_len", "s2_len", "top_sum", "bot_sum", "total_pmt_hits", "top_bot_asym", "top_active", "bot_active", "log_s2_s1", "s1_max", "s2_max"]

X = df_feat[feature_cols].values
y_bin = df_feat["is_signal"].values

X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42, stratify=df_feat["class"])

rf_bin = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bin.fit(X_train, y_train)
preds_prob = rf_bin.predict_proba(X_test)[:, 1]
preds_cls = rf_bin.predict(X_test)

auc_bin = roc_auc_score(y_test, preds_prob)
acc_bin = accuracy_score(y_test, preds_cls)
print(f"\n[Task A] Binary WIMP (all masses) vs All Backgrounds (Tabular/Physics Features RF):")
print(f"  ROC-AUC  : {auc_bin:.4f}")
print(f"  Accuracy : {acc_bin:.4f}")

# Task B: ER vs WIMP (The classical electronic vs nuclear recoil problem)
er_wimp_df = df_feat[df_feat["class"].isin(["ER", "mDM50GeV"])]
X_er_wimp = er_wimp_df[feature_cols].values
y_er_wimp = (er_wimp_df["class"] == "mDM50GeV").astype(int).values

X_tr, X_te, y_tr, y_te = train_test_split(X_er_wimp, y_er_wimp, test_size=0.3, random_state=42)
rf_er_wimp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_er_wimp.fit(X_tr, y_tr)
auc_er_wimp = roc_auc_score(y_te, rf_er_wimp.predict_proba(X_te)[:, 1])
print(f"\n[Task B] ER vs 50 GeV WIMP (Electronic Recoil vs Nuclear Recoil):")
print(f"  ROC-AUC  : {auc_er_wimp:.4f}")

# Task C: CNNS vs Low-Mass WIMPs (10 GeV, 20 GeV) - The Neutrino Fog challenge
cnns_10_df = df_feat[df_feat["class"].isin(["CNNS", "mDM10GeV"])]
X_cnns_10 = cnns_10_df[feature_cols].values
y_cnns_10 = (cnns_10_df["class"] == "mDM10GeV").astype(int).values

X_tr, X_te, y_tr, y_te = train_test_split(X_cnns_10, y_cnns_10, test_size=0.3, random_state=42)
rf_cnns_10 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cnns_10.fit(X_tr, y_tr)
auc_cnns_10 = roc_auc_score(y_te, rf_cnns_10.predict_proba(X_te)[:, 1])
print(f"\n[Task C] CNNS vs 10 GeV WIMP (Neutrino Fog Challenge):")
print(f"  ROC-AUC  : {auc_cnns_10:.4f}")

# Baseline Logistic Regression for Task A
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)
lr_bin = LogisticRegression(max_iter=1000, random_state=42)
lr_bin.fit(X_tr_sc, y_train)
auc_lr = roc_auc_score(y_test, lr_bin.predict_proba(X_te_sc)[:, 1])
acc_lr = accuracy_score(y_test, lr_bin.predict(X_te_sc))
print(f"\n[Task A - Logistic Regression] Binary WIMP vs Backgrounds:")
print(f"  ROC-AUC  : {auc_lr:.4f}")
print(f"  Accuracy : {acc_lr:.4f}")

# Task D: Multiclass Classification (12 classes)
class_labels = sorted(df_feat["class"].unique())
class_to_idx = {c: i for i, c in enumerate(class_labels)}
y_multi = np.array([class_to_idx[c] for c in df_feat["class"]])

X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X, y_multi, test_size=0.3, random_state=42, stratify=y_multi)
rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multi.fit(X_tr_m, y_tr_m)
preds_multi = rf_multi.predict(X_te_m)
acc_multi = accuracy_score(y_te_m, preds_multi)
print(f"\n[Task D] 12-Class Multiclass Benchmark (RF on 13 Physics Features):")
print(f"  Accuracy : {acc_multi:.4f}")
print("Classification Report (12 classes):")
print(classification_report(y_te_m, preds_multi, target_names=class_labels))

# Feature importances
print("\nFeature Importances for Binary WIMP vs Backgrounds:")
for f_name, imp in sorted(zip(feature_cols, rf_bin.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f_name:20s}: {imp:.4f}")

# -----------------------------------------------------------------------------
# 10. DUPLICATE & NEAR-DUPLICATE / LEAKAGE RISK ANALYSIS
# -----------------------------------------------------------------------------
print_section("10. Leakage & Duplicate Analysis")

# Check if there are exact duplicates within and across classes
seen_hashes = defaultdict(list)
print("Checking for duplicate PMT hit patterns across sample events...")
for i, row in df_feat.iterrows():
    # Hash the tuple of top PMT hits and S1/S2 lengths
    h = (round(row["top_sum"], 3), round(row["bot_sum"], 3), row["s1_len"], row["s2_len"], round(row["s1_area_hits"], 3), round(row["s2_area_hits"], 3))
    seen_hashes[h].append((row["class"], i))

duplicates = {k: v for k, v in seen_hashes.items() if len(v) > 1}
print(f"Total unique feature tuples tested: {len(seen_hashes)}")
print(f"Duplicate tuples found: {len(duplicates)}")
cross_class_dups = 0
for k, v in duplicates.items():
    classes_in_dup = set([item[0] for item in v])
    if len(classes_in_dup) > 1:
        cross_class_dups += 1
print(f"Cross-class duplicate tuples: {cross_class_dups}")

print("\nAnalysis finished successfully!", flush=True)

