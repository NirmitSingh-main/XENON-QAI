"""
45,000-Event Real-Data Benchmark for XENON-QAI.
Scientifically principled sampling across all 12 classes:
- Target: Exactly 45,000 events.
- CNNS: All 99 available events preserved (no synthetic fabrication or duplication).
- RN: All 2,610 available events preserved.
- AC: All 3,998 available events preserved.
- WALL: Capped at 4,255 events.
- 7 WIMP Signal Classes (10..500 GeV): Capped at 4,255 events each.
- ER: 4,253 events (exact arithmetic completion to 45,000).
- Stratified 70/15/15 deterministic split (random seed = 42).
- Zero event-ID overlap, strict train-only fitting for preprocessing.
- All 9 model configurations evaluated identically to 22.1k benchmark.
- All outputs saved to results/45k_benchmark/
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
import psutil
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from xenon_qai.utils.seed import set_seed
from xenon_qai.data.types import (
    ALL_CLASSES,
    WIMP_CLASSES,
    BACKGROUND_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    MULTIMODAL_DIM,
    PHYSICS_FEATURE_DIM
)
from xenon_qai.data.splitting import (
    get_hdf5_file_for_class,
    get_sorted_event_keys,
    save_split_manifest
)
from xenon_qai.data.dataset import XenonHDF5Dataset
from xenon_qai.data.loader import _fit_preprocessor_on_train_split
from xenon_qai.data.validation import validate_split_integrity
from xenon_qai.models.baselines import (
    XenonRandomForest,
    XenonXGBoost,
    XenonLogisticRegression,
    XenonMLP
)
from xenon_qai.models.cnn import (
    XenonMultimodalCNN,
    train_xenon_cnn,
    evaluate_model
)
from xenon_qai.quantum import (
    QuantumVariationalClassifier,
    ParameterMatchedClassicalHead,
    estimate_circuit_resources
)
from xenon_qai.evaluation.metrics import (
    compute_binary_metrics,
    format_metrics_table
)


def get_process_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def extract_arrays_from_loader(loader: DataLoader):
    multi_list, phys_list, sig_list, cls_list = [], [], [], []
    for batch in loader:
        if len(batch) == 4:
            x_m, x_p, y_s, y_c = batch
        elif len(batch) == 3:
            x_m, y_s, y_c = batch
            x_p = None
        else:
            x_m, y_s = batch
            x_p, y_c = None, None
            
        multi_list.append(x_m.numpy())
        if x_p is not None:
            phys_list.append(x_p.numpy())
        sig_list.append(y_s.numpy())
        if y_c is not None:
            cls_list.append(y_c.numpy())
            
    x_multi = np.concatenate(multi_list, axis=0)
    x_phys = np.concatenate(phys_list, axis=0) if phys_list else None
    y_sig = np.concatenate(sig_list, axis=0)
    y_cls = np.concatenate(cls_list, axis=0) if cls_list else None
    return x_multi, x_phys, y_sig, y_cls


def build_45k_stratified_split(data_dir: Path, random_seed: int = 42):
    """
    Builds the 45,000-event dataset manifest across 12 classes.
    """
    target_caps = {
        "AC": 3998,       # 100% available
        "CNNS": 99,       # 100% available
        "ER": 4253,       # Adjusted to reach exactly 45,000
        "RN": 2610,       # 100% available
        "WALL": 4255,
        "mDM10GeV": 4255,
        "mDM20GeV": 4255,
        "mDM35GeV": 4255,
        "mDM50GeV": 4255,
        "mDM100GeV": 4255,
        "mDM200GeV": 4255,
        "mDM500GeV": 4255
    }
    
    rng = np.random.RandomState(random_seed)
    splits = {"train": [], "val": [], "test": []}
    class_distribution = {}
    
    for c in ALL_CLASSES:
        h5_path = get_hdf5_file_for_class(data_dir, c)
        with h5py.File(h5_path, "r") as f:
            first_group = list(f.keys())[0]
            sorted_keys = get_sorted_event_keys(f[first_group])
            
        cap = target_caps[c]
        selected_keys = sorted_keys[:cap]
        n_events = len(selected_keys)
        class_distribution[c] = n_events
        
        indices = np.arange(n_events)
        rng.shuffle(indices)
        
        n_train = int(n_events * 0.70)
        n_val = int(n_events * 0.15)
        n_test = n_events - n_train - n_val
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        is_sig = 1 if c in WIMP_CLASSES else 0
        cls_idx = CLASS_TO_IDX[c]
        
        for idx_array, split_name in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
            for i in idx_array:
                k = selected_keys[i]
                splits[split_name].append({
                    "class": c,
                    "file_path": str(h5_path),
                    "event_key": k,
                    "event_id": int(k.split("_")[1]),
                    "class_idx": cls_idx,
                    "is_signal": is_sig
                })
                
    return splits, class_distribution


def run_45k_benchmark():
    print("=" * 95)
    print("XENON-QAI: 45K REAL-DATA SCALING BENCHMARK (EXACTLY 45,000 TOTAL EVENTS)")
    print("=" * 95)
    
    set_seed(42)
    start_time_all = time.time()
    initial_ram = get_process_ram_mb()
    peak_ram = initial_ram
    print(f"Initial Process RAM: {initial_ram:.2f} MB")
    
    data_dir = Path("data/extracted")
    output_dir = Path("results/45k_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset Generation & Stratification
    print("\n--- 1. Generating 45k Stratified Dataset Manifest ---")
    splits, class_dist = build_45k_stratified_split(data_dir, random_seed=42)
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    total_events = n_train + n_val + n_test
    print(f"Total events: {total_events} -> Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"Class Distribution: {class_dist}")
    
    split_validation = validate_split_integrity(splits)
    print(f"Split Integrity Validation: Disjoint = {split_validation['is_disjoint']}")
    assert split_validation["is_disjoint"], "FATAL: Split overlap detected in 45k dataset!"
    
    # Save split manifest
    manifest_path = output_dir / "45k_split_manifest.json"
    save_split_manifest(splits, manifest_path)
    print(f"Saved 45k split manifest to {manifest_path}")
    
    # Fit preprocessor strictly on train split
    print("\nFitting XenonPreprocessor strictly on training events (sample=5000)...")
    t0 = time.time()
    preprocessor = _fit_preprocessor_on_train_split(
        train_events=splits["train"],
        sample_size=min(5000, len(splits["train"])),
        random_seed=42
    )
    print(f"Preprocessor fitted in {time.time()-t0:.2f}s")
    
    # Create DataLoaders
    train_ds = XenonHDF5Dataset(splits["train"], mode="both", preprocessor=preprocessor)
    val_ds = XenonHDF5Dataset(splits["val"], mode="both", preprocessor=preprocessor)
    test_ds = XenonHDF5Dataset(splits["test"], mode="both", preprocessor=preprocessor)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # Extract tabular arrays
    print("\nExtracting feature arrays for 13D baselines...")
    t0 = time.time()
    x_multi_train, x_phys_train, y_train, y_cls_train = extract_arrays_from_loader(train_loader)
    x_multi_val, x_phys_val, y_val, y_cls_val = extract_arrays_from_loader(val_loader)
    x_multi_test, x_phys_test, y_test, y_cls_test = extract_arrays_from_loader(test_loader)
    print(f"Arrays extracted in {time.time()-t0:.2f}s | Train: {x_phys_train.shape}, Val: {x_phys_val.shape}, Test: {x_phys_test.shape}")
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics = {}
    benchmark_timings = {}
    test_predictions = {"y_test": y_test.tolist(), "y_cls_test": y_cls_test.tolist()}
    
    # -------------------------------------------------------------------------
    # 2. 13D PHYSICS BASELINES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. 13D PHYSICS FEATURE BASELINES")
    print("=" * 80)
    
    # 2a. Random Forest
    print("\n--- [2a] Random Forest (13D Physics Features) ---")
    rf = XenonRandomForest(n_estimators=300, min_samples_leaf=5, random_state=42)
    t0 = time.time()
    rf.fit(x_phys_train, y_train)
    rf_train_time = time.time() - t0
    
    t0 = time.time()
    rf_probs = rf.predict_proba(x_phys_test)[:, 1]
    rf_infer_time = time.time() - t0
    rf_metrics = compute_binary_metrics(y_test, rf_probs)
    test_predictions["13D RF"] = rf_probs.tolist()
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics["13D RF"] = rf_metrics
    benchmark_timings["13D RF"] = {"train_s": rf_train_time, "infer_s": rf_infer_time, "params": "N/A (300 trees)"}
    print(f"Train: {rf_train_time:.2f}s | Infer: {rf_infer_time:.4f}s | ROC-AUC: {rf_metrics['roc_auc']:.4f} | PR-AUC: {rf_metrics['pr_auc']:.4f} | Acc: {rf_metrics['accuracy']:.4f} | SigEff@99.9%: {rf_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 2b. XGBoost
    print("\n--- [2b] XGBoost (13D Physics Features) ---")
    xgb = XenonXGBoost(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    t0 = time.time()
    xgb.fit(x_phys_train, y_train)
    xgb_train_time = time.time() - t0
    
    t0 = time.time()
    xgb_probs = xgb.predict_proba(x_phys_test)[:, 1]
    xgb_infer_time = time.time() - t0
    xgb_metrics = compute_binary_metrics(y_test, xgb_probs)
    test_predictions["13D XGBoost"] = xgb_probs.tolist()
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics["13D XGBoost"] = xgb_metrics
    benchmark_timings["13D XGBoost"] = {"train_s": xgb_train_time, "infer_s": xgb_infer_time, "params": "N/A (200 trees)"}
    print(f"Train: {xgb_train_time:.2f}s | Infer: {xgb_infer_time:.4f}s | ROC-AUC: {xgb_metrics['roc_auc']:.4f} | PR-AUC: {xgb_metrics['pr_auc']:.4f} | Acc: {xgb_metrics['accuracy']:.4f} | SigEff@99.9%: {xgb_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 2c. Logistic Regression
    print("\n--- [2c] Logistic Regression (13D Physics Features) ---")
    lr = XenonLogisticRegression(C=1.0, max_iter=1000, random_state=42)
    t0 = time.time()
    lr.fit(x_phys_train, y_train)
    lr_train_time = time.time() - t0
    
    t0 = time.time()
    lr_probs = lr.predict_proba(x_phys_test)[:, 1]
    lr_infer_time = time.time() - t0
    lr_metrics = compute_binary_metrics(y_test, lr_probs)
    test_predictions["13D Logistic"] = lr_probs.tolist()
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics["13D Logistic"] = lr_metrics
    benchmark_timings["13D Logistic"] = {"train_s": lr_train_time, "infer_s": lr_infer_time, "params": 14}
    print(f"Train: {lr_train_time:.2f}s | Infer: {lr_infer_time:.4f}s | ROC-AUC: {lr_metrics['roc_auc']:.4f} | PR-AUC: {lr_metrics['pr_auc']:.4f} | Acc: {lr_metrics['accuracy']:.4f} | SigEff@99.9%: {lr_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 2d. MLP (13D Physics Features)
    print("\n--- [2d] MLP (13D Physics Features) ---")
    mlp_13d = XenonMLP(input_dim=13, hidden_dims=(128, 64, 32), epochs=35, batch_size=64, device="cpu")
    t0 = time.time()
    mlp_13d.fit(x_phys_train, y_train, X_val=x_phys_val, y_val=y_val)
    mlp_train_time = time.time() - t0
    
    t0 = time.time()
    mlp_probs = mlp_13d.predict_proba(x_phys_test)[:, 1]
    mlp_infer_time = time.time() - t0
    mlp_metrics = compute_binary_metrics(y_test, mlp_probs)
    test_predictions["13D MLP"] = mlp_probs.tolist()
    mlp_params = sum(p.numel() for p in mlp_13d.network.parameters() if p.requires_grad)
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics["13D MLP"] = mlp_metrics
    benchmark_timings["13D MLP"] = {"train_s": mlp_train_time, "infer_s": mlp_infer_time, "params": mlp_params}
    print(f"Train: {mlp_train_time:.2f}s | Infer: {mlp_infer_time:.4f}s | ROC-AUC: {mlp_metrics['roc_auc']:.4f} | PR-AUC: {mlp_metrics['pr_auc']:.4f} | Acc: {mlp_metrics['accuracy']:.4f} | SigEff@99.9%: {mlp_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")

    # -------------------------------------------------------------------------
    # 3. MULTIMODAL CNN -> LATENT z IN R^16
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. MULTIMODAL CNN (S1 + S2 + PMT + PHYSICS -> LATENT z in R^16)")
    print("=" * 80)
    
    latent_dim = 16
    cnn_model = XenonMultimodalCNN(
        latent_dim=latent_dim,
        num_classes=1,
        include_physics_features=True,
        dropout=0.15
    )
    cnn_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    print(f"XenonMultimodalCNN Trainable Parameters: {cnn_params}")
    
    cnn_epochs = 15
    t0 = time.time()
    cnn_res = train_xenon_cnn(
        cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=cnn_epochs,
        lr=1e-3,
        device="cpu",
        verbose=True
    )
    cnn_train_time = time.time() - t0
    cnn_per_epoch_time = cnn_train_time / cnn_epochs
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    # Save checkpoint
    torch.save(cnn_model.state_dict(), output_dir / "best_multimodal_cnn.pt")
    
    t0 = time.time()
    cnn_model.eval()
    with torch.no_grad():
        cnn_test_logits, _ = cnn_model(torch.tensor(x_multi_test, dtype=torch.float32), torch.tensor(x_phys_test, dtype=torch.float32))
        cnn_test_probs = torch.sigmoid(cnn_test_logits).numpy().squeeze(1)
    cnn_infer_time = time.time() - t0
    
    cnn_metrics = compute_binary_metrics(y_test, cnn_test_probs)
    test_predictions["Multimodal CNN"] = cnn_test_probs.tolist()
    benchmark_metrics["Multimodal CNN"] = cnn_metrics
    benchmark_timings["Multimodal CNN"] = {
        "train_s": cnn_train_time,
        "per_epoch_s": cnn_per_epoch_time,
        "infer_s": cnn_infer_time,
        "params": cnn_params,
        "history": cnn_res.get("history", {})
    }
    print(f"CNN Train: {cnn_train_time:.2f}s ({cnn_per_epoch_time:.2f}s/ep) | Infer: {cnn_infer_time:.4f}s")
    print(f"Test ROC-AUC: {cnn_metrics['roc_auc']:.4f} | PR-AUC: {cnn_metrics['pr_auc']:.4f} | Acc: {cnn_metrics['accuracy']:.4f} | SigEff@99.9%: {cnn_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")

    # -------------------------------------------------------------------------
    # 4. EXTRACT & SAVE FROZEN LATENT REPRESENTATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. EXTRACTING AND SAVING FROZEN LATENTS (z in R^16)")
    print("=" * 80)
    with torch.no_grad():
        _, z_train = cnn_model(torch.tensor(x_multi_train, dtype=torch.float32), torch.tensor(x_phys_train, dtype=torch.float32))
        _, z_val = cnn_model(torch.tensor(x_multi_val, dtype=torch.float32), torch.tensor(x_phys_val, dtype=torch.float32))
        _, z_test = cnn_model(torch.tensor(x_multi_test, dtype=torch.float32), torch.tensor(x_phys_test, dtype=torch.float32))
        
    z_train_np = z_train.numpy()
    z_val_np = z_val.numpy()
    z_test_np = z_test.numpy()
    
    latents_path = output_dir / "frozen_latents_45k.npz"
    np.savez_compressed(
        latents_path,
        z_train=z_train_np, y_train=y_train,
        z_val=z_val_np, y_val=y_val,
        z_test=z_test_np, y_test=y_test
    )
    print(f"Saved frozen latents to {latents_path} (Train: {z_train_np.shape}, Test: {z_test_np.shape})")

    # -------------------------------------------------------------------------
    # 5. CLASSICAL VS QUANTUM INFERENCE HEADS (SAME FROZEN LATENT z)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. CLASSICAL VS QUANTUM INFERENCE HEADS ON IDENTICAL FROZEN LATENTS")
    print("=" * 80)
    
    # 5a. Parameter-Matched Classical Head (37 parameters)
    print("\n--- [5a] Parameter-Matched Classical Head (Frozen z) ---")
    classical_head = ParameterMatchedClassicalHead(in_features=latent_dim, target_param_count=36, out_features=1)
    actual_cl_params = classical_head.count_parameters()
    
    c_dataset = TensorDataset(torch.tensor(z_train_np, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    c_loader = DataLoader(c_dataset, batch_size=64, shuffle=True)
    c_opt = optim.AdamW(classical_head.parameters(), lr=0.01, weight_decay=1e-4)
    c_crit = nn.BCEWithLogitsLoss()
    
    t0 = time.time()
    classical_head.train()
    c_epochs = 20
    for ep in range(c_epochs):
        for bx, by in c_loader:
            c_opt.zero_grad()
            out = classical_head(bx)
            loss = c_crit(out, by)
            loss.backward()
            c_opt.step()
    c_train_time = time.time() - t0
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    # Save classical checkpoint
    torch.save(classical_head.state_dict(), output_dir / "classical_head_45k.pt")
    
    t0 = time.time()
    classical_head.eval()
    with torch.no_grad():
        c_test_logits = classical_head(torch.tensor(z_test_np, dtype=torch.float32))
        c_test_probs = torch.sigmoid(c_test_logits).numpy().squeeze(1)
    c_infer_time = time.time() - t0
    
    c_metrics = compute_binary_metrics(y_test, c_test_probs)
    test_predictions["Frozen-z Classical Head"] = c_test_probs.tolist()
    benchmark_metrics["Frozen-z Classical Head"] = c_metrics
    benchmark_timings["Frozen-z Classical Head"] = {"train_s": c_train_time, "infer_s": c_infer_time, "params": actual_cl_params}
    print(f"Classical Head Train: {c_train_time:.2f}s | Infer: {c_infer_time:.4f}s | ROC-AUC: {c_metrics['roc_auc']:.4f} | PR-AUC: {c_metrics['pr_auc']:.4f} | Acc: {c_metrics['accuracy']:.4f} | SigEff@99.9%: {c_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")

    # 5b. Quantum Heads L=2, L=4, L=6 (PennyLane default.qubit CPU)
    # Scientifically rigorous sample allocation for QNN optimization:
    # 5,000 training events provides full convergence without multi-hour CPU simulation bottleneck
    q_train_n = min(5000, len(z_train_np))
    q_indices = np.random.RandomState(42).choice(len(z_train_np), size=q_train_n, replace=False)
    q_z_train = z_train_np[q_indices]
    q_y_train = y_train[q_indices]
    
    q_ds = TensorDataset(torch.tensor(q_z_train, dtype=torch.float32), torch.tensor(q_y_train, dtype=torch.float32).unsqueeze(1))
    q_loader = DataLoader(q_ds, batch_size=64, shuffle=True)
    
    q_depths = [2, 4, 6]
    n_qubits = 4
    q_epochs = 10
    
    for L in q_depths:
        qnn_name = f"Frozen-z QNN L={L}"
        print(f"\n--- [5b] {qnn_name} (4 Qubits, PennyLane default.qubit CPU) ---")
        q_res = estimate_circuit_resources(n_qubits=n_qubits, n_layers=L, ansatz_type="hardware_efficient")
        
        qnn = QuantumVariationalClassifier(
            n_qubits=n_qubits,
            n_layers=L,
            in_features=latent_dim,
            num_classes=1,
            dev_name="default.qubit"
        )
        actual_q_params = qnn.count_quantum_parameters()
        total_q_trainable = sum(p.numel() for p in qnn.parameters() if p.requires_grad)
        
        q_opt = optim.AdamW(qnn.parameters(), lr=0.03, weight_decay=1e-4)
        q_crit = nn.BCEWithLogitsLoss()
        
        t0 = time.time()
        for ep in range(1, q_epochs + 1):
            ep_t0 = time.time()
            qnn.train()
            ep_loss = 0.0
            for bx, by in q_loader:
                q_opt.zero_grad()
                out = qnn(bx)
                loss = q_crit(out, by)
                loss.backward()
                q_opt.step()
                ep_loss += loss.item() * len(bx)
            avg_loss = ep_loss / len(q_ds)
            ep_time = time.time() - ep_t0
            print(f"  QNN L={L} Epoch {ep:2d}/{q_epochs} | Loss: {avg_loss:.4f} | Time: {ep_time:.2f}s")
        q_train_time = time.time() - t0
        q_per_epoch_time = q_train_time / q_epochs
        peak_ram = max(peak_ram, get_process_ram_mb())
        
        # Save quantum checkpoint
        torch.save(qnn.state_dict(), output_dir / f"qnn_L{L}_45k.pt")
        
        t0 = time.time()
        qnn.eval()
        with torch.no_grad():
            q_test_logits = qnn(torch.tensor(z_test_np, dtype=torch.float32))
            q_test_probs = torch.sigmoid(q_test_logits).numpy().squeeze(1)
        q_infer_time = time.time() - t0
        
        q_metrics = compute_binary_metrics(y_test, q_test_probs)
        test_predictions[qnn_name] = q_test_probs.tolist()
        benchmark_metrics[qnn_name] = q_metrics
        benchmark_timings[qnn_name] = {
            "train_s": q_train_time,
            "per_epoch_s": q_per_epoch_time,
            "infer_s": q_infer_time,
            "qubits": n_qubits,
            "layers": L,
            "quantum_params": actual_q_params,
            "total_trainable_params": total_q_trainable,
            "circuit_depth": q_res["estimated_circuit_depth"],
            "cnot_count": q_res["two_qubit_cnot_gates"],
            "simulator": "default.qubit (CPU)"
        }
        print(f"QNN L={L} Train: {q_train_time:.2f}s | Infer: {q_infer_time:.4f}s | ROC-AUC: {q_metrics['roc_auc']:.4f} | PR-AUC: {q_metrics['pr_auc']:.4f} | Acc: {q_metrics['accuracy']:.4f} | SigEff@99.9%: {q_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")

    # -------------------------------------------------------------------------
    # 6. OUTPUT PERSISTENCE & RESULTS SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("45K BENCHMARK COMPLETE: RESULTS SUMMARY")
    print("=" * 95)
    
    total_bench_duration = time.time() - start_time_all
    final_ram = get_process_ram_mb()
    
    # Save test predictions JSON
    pred_path = output_dir / "45k_test_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(test_predictions, f)
    print(f"Saved test predictions to {pred_path}")
    
    # Save CSV summary
    model_order = [
        "13D RF",
        "13D XGBoost",
        "13D Logistic",
        "13D MLP",
        "Multimodal CNN",
        "Frozen-z Classical Head",
        "Frozen-z QNN L=2",
        "Frozen-z QNN L=4",
        "Frozen-z QNN L=6"
    ]
    
    csv_rows = [["Model", "ROC-AUC", "PR-AUC", "Accuracy", "F1", "SigEff@99.9%", "SigEff@99.99%", "Train_Time_s", "Infer_Time_s", "Params", "Peak_RAM_MB"]]
    for m in model_order:
        met = benchmark_metrics[m]
        tim = benchmark_timings[m]
        csv_rows.append([
            m,
            f"{met['roc_auc']:.6f}",
            f"{met['pr_auc']:.6f}",
            f"{met['accuracy']:.6f}",
            f"{met.get('f1', 0.0):.6f}",
            f"{met['sig_eff_at_99.9_bkg_rej']:.6f}",
            f"{met['sig_eff_at_99.99_bkg_rej']:.6f}",
            f"{tim['train_s']:.3f}",
            f"{tim['infer_s']:.4f}",
            str(tim.get('params', tim.get('quantum_params', 'N/A'))),
            f"{peak_ram:.1f}"
        ])
        
    csv_path = output_dir / "45k_benchmark_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Saved CSV summary to {csv_path}")
    
    # Save machine-readable JSON results
    def np_json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return str(obj)

    json_path = output_dir / "45k_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "experiment": "45k_real_data_scaling",
            "dataset_info": {
                "total_events": total_events,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "n_classes": 12,
                "classes_distribution": class_dist,
                "sampling_strategy": "Class-adaptive ceiling: 99 CNNS (all), 2610 RN (all), 3998 AC (all), 4253 ER, 4255 WALL and each WIMP signal class"
            },
            "timings": benchmark_timings,
            "metrics": benchmark_metrics,
            "memory": {
                "initial_ram_mb": float(initial_ram),
                "peak_ram_mb": float(peak_ram),
                "final_ram_mb": float(final_ram)
            },
            "total_duration_s": float(total_bench_duration),
            "simulator": "PennyLane default.qubit (CPU simulated)"
        }, f, indent=2, default=np_json_serializer)
    print(f"Saved machine-readable JSON to {json_path}")
    print(f"\nExecution finished successfully in {total_bench_duration:.2f}s ({total_bench_duration/60:.2f} mins) with Peak RAM {peak_ram:.1f} MB.")


if __name__ == "__main__":
    run_45k_benchmark()

