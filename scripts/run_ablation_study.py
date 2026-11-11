"""
Modality Ablation Study for XENON-QAI.
Evaluates the contribution of each detector modality on identical splits:
1. Full Model (S1 + S2 + PMT + Physics)
2. No S1 (Mask S1)
3. No S2 (Mask S2)
4. No PMT (Mask PMT)
5. No Physics (include_physics_features=False)
6. S1 Only
7. S2 Only
8. PMT Only
9. Physics Only

Uses the exact 45k split manifest to guarantee fairness and zero leakage.
Trains on a standardized training budget of 10,000 events for 10 epochs.
Saves all outputs to results/ablation_study/
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from xenon_qai.utils.seed import set_seed
from xenon_qai.data.types import (
    S1_MAX_LEN,
    S2_TARGET_LEN,
    TOP_PMT_COUNT,
    BOT_PMT_COUNT,
    MULTIMODAL_DIM,
    PHYSICS_FEATURE_DIM
)
from xenon_qai.data.splitting import load_split_manifest
from xenon_qai.data.dataset import XenonHDF5Dataset
from xenon_qai.data.loader import _fit_preprocessor_on_train_split
from xenon_qai.models.cnn.model import XenonMultimodalCNN
from xenon_qai.evaluation.metrics import compute_binary_metrics


def get_process_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def extract_arrays_from_loader(loader: DataLoader):
    multi_list, phys_list, sig_list = [], [], []
    for batch in loader:
        x_m, x_p, y_s, _ = batch
        multi_list.append(x_m.numpy())
        phys_list.append(x_p.numpy())
        sig_list.append(y_s.numpy())
    return (
        np.concatenate(multi_list, axis=0),
        np.concatenate(phys_list, axis=0),
        np.concatenate(sig_list, axis=0)
    )


def train_ablation_model(
    x_multi_train, x_phys_train, y_train,
    x_multi_val, x_phys_val, y_val,
    x_multi_test, x_phys_test, y_test,
    config_name: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3
):
    print(f"\n--- Running Ablation: {config_name} ---")
    include_physics = True
    x_tr_m = x_multi_train.copy()
    x_va_m = x_multi_val.copy()
    x_te_m = x_multi_test.copy()
    
    x_tr_p = x_phys_train.copy()
    x_va_p = x_phys_val.copy()
    x_te_p = x_phys_test.copy()
    
    # Slices: S1 is 0..160, S2 is 160..660, PMT is 660..908
    s1_slice = slice(0, S1_MAX_LEN)
    s2_slice = slice(S1_MAX_LEN, S1_MAX_LEN + S2_TARGET_LEN)
    pmt_slice = slice(S1_MAX_LEN + S2_TARGET_LEN, MULTIMODAL_DIM)
    
    if config_name == "Full Model":
        pass
    elif config_name == "No S1":
        x_tr_m[:, s1_slice] = 0.0
        x_va_m[:, s1_slice] = 0.0
        x_te_m[:, s1_slice] = 0.0
    elif config_name == "No S2":
        x_tr_m[:, s2_slice] = 0.0
        x_va_m[:, s2_slice] = 0.0
        x_te_m[:, s2_slice] = 0.0
    elif config_name == "No PMT":
        x_tr_m[:, pmt_slice] = 0.0
        x_va_m[:, pmt_slice] = 0.0
        x_te_m[:, pmt_slice] = 0.0
    elif config_name == "No Physics":
        include_physics = False
        x_tr_p = np.zeros_like(x_tr_p)
        x_va_p = np.zeros_like(x_va_p)
        x_te_p = np.zeros_like(x_te_p)
    elif config_name == "S1 Only":
        include_physics = False
        x_tr_m[:, s2_slice] = 0.0
        x_va_m[:, s2_slice] = 0.0
        x_te_m[:, s2_slice] = 0.0
        x_tr_m[:, pmt_slice] = 0.0
        x_va_m[:, pmt_slice] = 0.0
        x_te_m[:, pmt_slice] = 0.0
        x_tr_p = np.zeros_like(x_tr_p)
        x_va_p = np.zeros_like(x_va_p)
        x_te_p = np.zeros_like(x_te_p)
    elif config_name == "S2 Only":
        include_physics = False
        x_tr_m[:, s1_slice] = 0.0
        x_va_m[:, s1_slice] = 0.0
        x_te_m[:, s1_slice] = 0.0
        x_tr_m[:, pmt_slice] = 0.0
        x_va_m[:, pmt_slice] = 0.0
        x_te_m[:, pmt_slice] = 0.0
        x_tr_p = np.zeros_like(x_tr_p)
        x_va_p = np.zeros_like(x_va_p)
        x_te_p = np.zeros_like(x_te_p)
    elif config_name == "PMT Only":
        include_physics = False
        x_tr_m[:, s1_slice] = 0.0
        x_va_m[:, s1_slice] = 0.0
        x_te_m[:, s1_slice] = 0.0
        x_tr_m[:, s2_slice] = 0.0
        x_va_m[:, s2_slice] = 0.0
        x_te_m[:, s2_slice] = 0.0
        x_tr_p = np.zeros_like(x_tr_p)
        x_va_p = np.zeros_like(x_va_p)
        x_te_p = np.zeros_like(x_te_p)
    elif config_name == "Physics Only":
        x_tr_m[:, :] = 0.0
        x_va_m[:, :] = 0.0
        x_te_m[:, :] = 0.0
    else:
        raise ValueError(f"Unknown ablation configuration: {config_name}")
        
    train_ds = TensorDataset(
        torch.tensor(x_tr_m, dtype=torch.float32),
        torch.tensor(x_tr_p, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = XenonMultimodalCNN(latent_dim=16, num_classes=1, include_physics_features=include_physics, dropout=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    t0 = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        for bx_m, bx_p, by in train_ldr:
            optimizer.zero_grad()
            p_arg = bx_p if include_physics else None
            logits, _ = model(bx_m, p_arg)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0
    
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        te_p_arg = torch.tensor(x_te_p, dtype=torch.float32) if include_physics else None
        test_logits, _ = model(torch.tensor(x_te_m, dtype=torch.float32), te_p_arg)
        test_probs = torch.sigmoid(test_logits).numpy().squeeze(1)
    infer_time = time.time() - t0
    
    metrics = compute_binary_metrics(y_test, test_probs)
    print(f"[{config_name:14s}] Train: {train_time:5.2f}s | Infer: {infer_time:5.3f}s | ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | SigEff@99.9%: {metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    return {
        "metrics": metrics,
        "timings": {"train_s": train_time, "infer_s": infer_time},
        "probabilities": test_probs.tolist()
    }


def run_ablation_study():
    print("=" * 90)
    print("XENON-QAI: MODALITY ABLATION STUDY (9 CONFIGURATIONS ON IDENTICAL DATA)")
    print("=" * 90)
    
    set_seed(42)
    out_dir = Path("results/ablation_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = Path("results/45k_benchmark/45k_split_manifest.json")
    splits = load_split_manifest(manifest_path)
    
    # Standardized ablation subset: 10,000 train events, 3,000 test events for rapid exact comparison
    rng = np.random.RandomState(42)
    n_sub_train = min(10000, len(splits["train"]))
    n_sub_test = min(3000, len(splits["test"]))
    
    sub_train_idx = rng.choice(len(splits["train"]), size=n_sub_train, replace=False)
    sub_test_idx = rng.choice(len(splits["test"]), size=n_sub_test, replace=False)
    
    sub_train_manifest = [splits["train"][i] for i in sub_train_idx]
    sub_test_manifest = [splits["test"][i] for i in sub_test_idx]
    sub_val_manifest = splits["val"][:1000]
    
    print(f"Fitting preprocessor strictly on ablation train events (N={n_sub_train})...")
    preprocessor = _fit_preprocessor_on_train_split(sub_train_manifest, sample_size=min(3000, n_sub_train), random_seed=42)
    
    print("Extracting feature arrays for ablation dataset...")
    tr_ds = XenonHDF5Dataset(sub_train_manifest, mode="both", preprocessor=preprocessor)
    va_ds = XenonHDF5Dataset(sub_val_manifest, mode="both", preprocessor=preprocessor)
    te_ds = XenonHDF5Dataset(sub_test_manifest, mode="both", preprocessor=preprocessor)
    
    tr_ldr = DataLoader(tr_ds, batch_size=64, shuffle=False)
    va_ldr = DataLoader(va_ds, batch_size=64, shuffle=False)
    te_ldr = DataLoader(te_ds, batch_size=64, shuffle=False)
    
    x_m_tr, x_p_tr, y_tr = extract_arrays_from_loader(tr_ldr)
    x_m_va, x_p_va, y_va = extract_arrays_from_loader(va_ldr)
    x_m_te, x_p_te, y_te = extract_arrays_from_loader(te_ldr)
    print(f"Feature arrays ready: Train {x_m_tr.shape}, Test {x_m_te.shape}")
    
    ablation_configs = [
        "Full Model",
        "No S1",
        "No S2",
        "No PMT",
        "No Physics",
        "S1 Only",
        "S2 Only",
        "PMT Only",
        "Physics Only"
    ]
    
    results = {}
    csv_rows = [["Configuration", "ROC-AUC", "PR-AUC", "Accuracy", "SigEff@99.9%", "Train_Time_s", "Infer_Time_s"]]
    
    for cfg in ablation_configs:
        res = train_ablation_model(
            x_m_tr, x_p_tr, y_tr,
            x_m_va, x_p_va, y_va,
            x_m_te, x_p_te, y_te,
            config_name=cfg,
            epochs=10,
            batch_size=64,
            lr=1e-3
        )
        results[cfg] = res
        m = res["metrics"]
        t = res["timings"]
        csv_rows.append([
            cfg,
            f"{m['roc_auc']:.6f}",
            f"{m['pr_auc']:.6f}",
            f"{m['accuracy']:.6f}",
            f"{m['sig_eff_at_99.9_bkg_rej']:.6f}",
            f"{t['train_s']:.2f}",
            f"{t['infer_s']:.4f}"
        ])
        
    csv_path = out_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nSaved ablation summary CSV to {csv_path}")
    
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablation results JSON to {json_path}")


if __name__ == "__main__":
    run_ablation_study()

