"""
25k Real-Data Scaling Benchmark for XENON-QAI (~2,000 events/class, ~24,000 events total).

Executes:
1. Data loading & stratified splitting (~2,000 events/class across 12 classes).
2. Leakage-free preprocessor fitting on train split only.
3. 13D Physics feature baselines (Random Forest, XGBoost, Logistic Regression, MLP).
4. Multi-modal CNN (1D-CNNs + PMT MLP + Physics MLP -> Latent z in R^16).
5. Latent space extraction & saving (train/val/test latents).
6. Parameter-matched Classical Head on frozen latents.
7. Quantum Heads with varying circuit depths (L=2, L=4, L=6) on the EXACT SAME frozen latents.
8. Fairness validation & computational monitoring (RAM, CPU times, NaN/Inf checks).
9. Output JSON & CSV results to results/25k_benchmark_results.json and results/25k_benchmark_summary.csv.
"""

from pathlib import Path
import time
import json
import csv
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from xenon_qai.utils.seed import set_seed
from xenon_qai.data import (
    create_dataloaders,
    validate_split_integrity,
    save_split_manifest
)
from xenon_qai.data.types import MULTIMODAL_DIM, PHYSICS_FEATURE_DIM
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
from xenon_qai.evaluation.comparison import compare_models, format_comparison_table


def get_process_ram_mb() -> float:
    """Returns RSS memory in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def extract_arrays_from_loader(loader: DataLoader):
    """Extracts numpy feature arrays from DataLoader."""
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


def run_25k_benchmark():
    print("=" * 95)
    print("XENON-QAI: 25K REAL-DATA SCALING BENCHMARK (~2,000 events/class, ~24,000 total events)")
    print("=" * 95)
    
    set_seed(42)
    start_time_all = time.time()
    initial_ram = get_process_ram_mb()
    peak_ram = initial_ram
    print(f"Initial Process RAM: {initial_ram:.2f} MB")
    
    data_dir = Path("data/extracted")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. DATA SAMPLING, SPLITTING & PREPROCESSING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print("1. DATA SAMPLING, SPLITTING & LEAKAGE-FREE PREPROCESSING (~25k EVENTS)")
    print("-" * 95)
    
    events_per_class = 2000
    print(f"Sampling {events_per_class} events per class across 12 classes (~24,000 total events)...")
    
    t0 = time.time()
    train_loader, val_loader, test_loader, preprocessor, splits = create_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        mode="both",
        split_ratios=(0.70, 0.15, 0.15),
        random_seed=42,
        max_events_per_class=events_per_class,
        fit_preprocessor=True
    )
    load_and_prep_time = time.time() - t0
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    total_events = n_train + n_val + n_test
    print(f"Total events: {total_events} -> Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"Data loading & preprocessor fitting time: {load_and_prep_time:.2f}s")
    print(f"Current RAM: {get_process_ram_mb():.2f} MB (Delta: +{get_process_ram_mb() - initial_ram:.2f} MB)")
    
    # Disjointness and split validation check
    split_validation = validate_split_integrity(splits)
    print(f"Split Integrity Validation: Disjoint = {split_validation['is_disjoint']}")
    print(f"  - Overlaps: train-val={split_validation['train_val_overlap']}, train-test={split_validation['train_test_overlap']}, val-test={split_validation['val_test_overlap']}")
    print(f"  - Classes covered: {len(split_validation['class_counts_train'])}/12")
    assert split_validation["is_disjoint"], "FATAL: Split overlap detected!"
    
    # Save split manifest for complete scientific auditability
    manifest_path = results_dir / "25k_split_manifest.json"
    save_split_manifest(splits, manifest_path)
    print(f"Saved split manifest to {manifest_path}")
    
    # Verify first batch shapes & check NaN/Inf
    first_batch = next(iter(train_loader))
    x_multi_b, x_phys_b, y_sig_b, y_cls_b = first_batch
    print(f"\nBatch Tensor Shapes (batch_size=64):")
    print(f"  - Multimodal Vector (S1 + S2 + PMTs): {x_multi_b.shape}, dtype={x_multi_b.dtype}")
    print(f"  - Physics Features (13D):             {x_phys_b.shape}, dtype={x_phys_b.dtype}")
    print(f"  - Binary Label (is_signal):           {y_sig_b.shape}, dtype={y_sig_b.dtype}")
    print(f"  - Multi-class Label (0..11):          {y_cls_b.shape}, dtype={y_cls_b.dtype}")
    
    nan_check_data = bool(torch.isnan(x_multi_b).any() or torch.isnan(x_phys_b).any())
    inf_check_data = bool(torch.isinf(x_multi_b).any() or torch.isinf(x_phys_b).any())
    print(f"Data Batch Numerical Check: NaN = {nan_check_data}, Inf = {inf_check_data}")
    assert not nan_check_data and not inf_check_data, "NaN/Inf in preprocessed data!"
    
    # Extract tabular arrays for 13D baselines
    print("\nExtracting feature arrays for all splits...")
    t0 = time.time()
    x_multi_train, x_phys_train, y_train, _ = extract_arrays_from_loader(train_loader)
    x_multi_val, x_phys_val, y_val, _ = extract_arrays_from_loader(val_loader)
    x_multi_test, x_phys_test, y_test, _ = extract_arrays_from_loader(test_loader)
    feature_extract_time = time.time() - t0
    peak_ram = max(peak_ram, get_process_ram_mb())
    print(f"Extracted arrays in {feature_extract_time:.2f}s | Train shape: {x_phys_train.shape}, Val: {x_phys_val.shape}, Test shape: {x_phys_test.shape}")
    
    benchmark_metrics = {}
    benchmark_timings = {}
    
    # -------------------------------------------------------------------------
    # 2. 13D PHYSICS BASELINES
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print("2. 13D PHYSICS FEATURE BASELINES")
    print("-" * 95)
    
    # 2a. Random Forest
    print("\n--- [2a] Random Forest (13D Physics Features) ---")
    rf = XenonRandomForest(n_estimators=300, min_samples_leaf=5, random_state=42)
    t0 = time.time()
    rf.fit(x_phys_train, y_train)
    rf_train_time = time.time() - t0
    
    t0 = time.time()
    rf_metrics = rf.evaluate(x_phys_test, y_test)
    rf_infer_time = time.time() - t0
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics["13D RF"] = rf_metrics
    benchmark_timings["13D RF"] = {"train_s": rf_train_time, "infer_s": rf_infer_time}
    print(f"Train time: {rf_train_time:.3f}s | Infer time: {rf_infer_time:.3f}s")
    print(f"Test ROC-AUC: {rf_metrics['roc_auc']:.4f} | PR-AUC: {rf_metrics['pr_auc']:.4f} | Acc: {rf_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {rf_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 2b. XGBoost
    print("\n--- [2b] XGBoost (13D Physics Features) ---")
    xgb = XenonXGBoost(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    t0 = time.time()
    xgb.fit(x_phys_train, y_train)
    xgb_train_time = time.time() - t0
    
    t0 = time.time()
    xgb_metrics = xgb.evaluate(x_phys_test, y_test)
    xgb_infer_time = time.time() - t0
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics["13D XGBoost"] = xgb_metrics
    benchmark_timings["13D XGBoost"] = {"train_s": xgb_train_time, "infer_s": xgb_infer_time}
    print(f"Train time: {xgb_train_time:.3f}s | Infer time: {xgb_infer_time:.3f}s")
    print(f"Test ROC-AUC: {xgb_metrics['roc_auc']:.4f} | PR-AUC: {xgb_metrics['pr_auc']:.4f} | Acc: {xgb_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {xgb_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 2c. Logistic Regression
    print("\n--- [2c] Logistic Regression (13D Physics Features) ---")
    lr = XenonLogisticRegression(C=1.0, max_iter=1000, random_state=42)
    t0 = time.time()
    lr.fit(x_phys_train, y_train)
    lr_train_time = time.time() - t0
    
    t0 = time.time()
    lr_metrics = lr.evaluate(x_phys_test, y_test)
    lr_infer_time = time.time() - t0
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics["13D Logistic"] = lr_metrics
    benchmark_timings["13D Logistic"] = {"train_s": lr_train_time, "infer_s": lr_infer_time}
    print(f"Train time: {lr_train_time:.3f}s | Infer time: {lr_infer_time:.3f}s")
    print(f"Test ROC-AUC: {lr_metrics['roc_auc']:.4f} | PR-AUC: {lr_metrics['pr_auc']:.4f} | Acc: {lr_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {lr_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 2d. MLP (13D Physics Features)
    print("\n--- [2d] MLP (13D Physics Features) ---")
    mlp_13d = XenonMLP(input_dim=13, hidden_dims=(128, 64, 32), epochs=35, batch_size=64, device="cpu")
    t0 = time.time()
    mlp_13d.fit(x_phys_train, y_train, X_val=x_phys_val, y_val=y_val)
    mlp_train_time = time.time() - t0
    
    t0 = time.time()
    mlp_metrics = mlp_13d.evaluate(x_phys_test, y_test)
    mlp_infer_time = time.time() - t0
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    benchmark_metrics["13D MLP"] = mlp_metrics
    benchmark_timings["13D MLP"] = {"train_s": mlp_train_time, "infer_s": mlp_infer_time}
    print(f"Train time (35 epochs): {mlp_train_time:.3f}s ({mlp_train_time/35:.3f}s/epoch) | Infer time: {mlp_infer_time:.3f}s")
    print(f"Test ROC-AUC: {mlp_metrics['roc_auc']:.4f} | PR-AUC: {mlp_metrics['pr_auc']:.4f} | Acc: {mlp_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {mlp_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")

    # -------------------------------------------------------------------------
    # 3. MULTIMODAL CNN -> LATENT z IN R^16
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print("3. MULTIMODAL CNN (S1 1D-CNN + S2 1D-CNN + PMT MLP + PHYSICS MLP -> z in R^16)")
    print("-" * 95)
    
    latent_dim = 16
    cnn_model = XenonMultimodalCNN(
        latent_dim=latent_dim,
        num_classes=1,
        include_physics_features=True,
        dropout=0.15
    )
    
    cnn_epochs = 15
    print(f"Training XenonMultimodalCNN for {cnn_epochs} epochs on CPU (batch_size=64)...")
    
    t0 = time.time()
    cnn_train_res = train_xenon_cnn(
        cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=cnn_epochs,
        lr=1e-3,
        device="cpu",
        verbose=True
    )
    cnn_total_train_time = time.time() - t0
    cnn_per_epoch_time = cnn_total_train_time / cnn_epochs
    cnn_test_metrics = cnn_train_res["test_metrics"]
    peak_ram = max(peak_ram, get_process_ram_mb())
    
    t0 = time.time()
    with torch.no_grad():
        _ = evaluate_model(cnn_model, test_loader, device=torch.device("cpu"), task="binary")
    cnn_infer_time = time.time() - t0
    
    benchmark_metrics["Multimodal CNN"] = cnn_test_metrics
    benchmark_timings["Multimodal CNN"] = {
        "train_s": cnn_total_train_time,
        "per_epoch_s": cnn_per_epoch_time,
        "infer_s": cnn_infer_time,
        "history": cnn_train_res.get("history", {})
    }
    print(f"\nCNN Total Train time ({cnn_epochs} epochs): {cnn_total_train_time:.2f}s ({cnn_per_epoch_time:.3f}s/epoch) | Infer: {cnn_infer_time:.3f}s")
    print(f"Test ROC-AUC: {cnn_test_metrics['roc_auc']:.4f} | PR-AUC: {cnn_test_metrics['pr_auc']:.4f} | Acc: {cnn_test_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {cnn_test_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")

    # -------------------------------------------------------------------------
    # 4. FROZEN LATENT EXTRACTION & SAVING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print("4. FROZEN LATENT EXTRACTION & AUDIT TRAIL (z in R^16)")
    print("-" * 95)
    
    cnn_model.eval()
    with torch.no_grad():
        _, z_train = cnn_model(torch.tensor(x_multi_train, dtype=torch.float32), torch.tensor(x_phys_train, dtype=torch.float32))
        _, z_val = cnn_model(torch.tensor(x_multi_val, dtype=torch.float32), torch.tensor(x_phys_val, dtype=torch.float32))
        _, z_test = cnn_model(torch.tensor(x_multi_test, dtype=torch.float32), torch.tensor(x_phys_test, dtype=torch.float32))
        
    z_train_np = z_train.numpy()
    z_val_np = z_val.numpy()
    z_test_np = z_test.numpy()
    
    nan_count_z = int(np.isnan(z_train_np).sum() + np.isnan(z_val_np).sum() + np.isnan(z_test_np).sum())
    inf_count_z = int(np.isinf(z_train_np).sum() + np.isinf(z_val_np).sum() + np.isinf(z_test_np).sum())
    
    print(f"Latents Extracted -> Train: {z_train_np.shape} | Val: {z_val_np.shape} | Test: {z_test_np.shape}")
    print(f"Latent Statistics: Mean={z_train_np.mean():.4f}, Std={z_train_np.std():.4f}, Range=[{z_train_np.min():.3f}, {z_train_np.max():.3f}]")
    print(f"Latent NaN Count: {nan_count_z} | Inf Count: {inf_count_z}")
    assert nan_count_z == 0 and inf_count_z == 0, "FATAL: NaN/Inf in extracted latent space!"
    
    # Save frozen latents to disk
    latents_save_path = results_dir / "frozen_latents_25k.npz"
    np.savez_compressed(
        latents_save_path,
        z_train=z_train_np, y_train=y_train,
        z_val=z_val_np, y_val=y_val,
        z_test=z_test_np, y_test=y_test
    )
    print(f"Saved frozen latents to {latents_save_path}")

    # -------------------------------------------------------------------------
    # 5. FAIR CLASSICAL VS QUANTUM EXPERIMENT ON IDENTICAL FROZEN LATENTS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print("5. FAIR CLASSICAL VS QUANTUM EXPERIMENT (EXACT SAME FROZEN LATENTS)")
    print("-" * 95)
    
    # 5a. Parameter-Matched Classical Head (37 parameters)
    print("\n--- [5a] Parameter-Matched Classical Head (Frozen z in R^16) ---")
    classical_head = ParameterMatchedClassicalHead(
        in_features=latent_dim,
        target_param_count=36,
        out_features=1
    )
    actual_classical_params = classical_head.count_parameters()
    print(f"Classical Head Parameter Count: {actual_classical_params} parameters")
    
    c_train_t = torch.tensor(z_train_np, dtype=torch.float32)
    c_y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    c_dataset = TensorDataset(c_train_t, c_y_train_t)
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
    
    t0 = time.time()
    classical_head.eval()
    with torch.no_grad():
        c_test_logits = classical_head(torch.tensor(z_test_np, dtype=torch.float32))
        c_test_probs = torch.sigmoid(c_test_logits).numpy().squeeze(1)
    c_infer_time = time.time() - t0
    
    c_metrics = compute_binary_metrics(y_test, c_test_probs)
    benchmark_metrics["Frozen-z Classical Head"] = c_metrics
    benchmark_timings["Frozen-z Classical Head"] = {"train_s": c_train_time, "infer_s": c_infer_time, "params": actual_classical_params}
    print(f"Classical Head Train time ({c_epochs} epochs): {c_train_time:.3f}s | Infer: {c_infer_time:.4f}s")
    print(f"Test ROC-AUC: {c_metrics['roc_auc']:.4f} | PR-AUC: {c_metrics['pr_auc']:.4f} | Acc: {c_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {c_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 5b. Quantum Head Experiments: L = 2, 4, 6 (Classically Simulated on CPU)
    q_depths = [2, 4, 6]
    n_qubits = 4
    q_epochs = 10
    q_batch_size = 64
    
    for L in q_depths:
        qnn_name = f"Frozen-z QNN L={L}"
        print(f"\n--- [5b] Quantum Head (QNN) Depth L={L} (4 Qubits, Classically Simulated on CPU) ---")
        
        q_res = estimate_circuit_resources(n_qubits=n_qubits, n_layers=L, ansatz_type="hardware_efficient")
        
        qnn = QuantumVariationalClassifier(
            n_qubits=n_qubits,
            n_layers=L,
            in_features=latent_dim,
            num_classes=1,
            dev_name="default.qubit"
        )
        actual_q_params = qnn.count_quantum_parameters()
        print(f"Circuit Config: {n_qubits} qubits, {L} layers | Estimated Depth: {q_res['estimated_circuit_depth']} | 2-qubit CNOTs: {q_res['two_qubit_cnot_gates']} | Quantum Params: {actual_q_params}")
        
        q_ds = TensorDataset(torch.tensor(z_train_np, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        q_loader = DataLoader(q_ds, batch_size=q_batch_size, shuffle=True)
        
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
            ep_time = time.time() - ep_t0
            avg_loss = ep_loss / len(q_ds)
            print(f"  QNN L={L} Epoch {ep:2d}/{q_epochs} | Loss: {avg_loss:.4f} | Time: {ep_time:.2f}s ({ep_time/len(q_loader):.3f}s/batch)")
            
        q_train_time = time.time() - t0
        q_per_epoch_time = q_train_time / q_epochs
        peak_ram = max(peak_ram, get_process_ram_mb())
        
        t0 = time.time()
        qnn.eval()
        with torch.no_grad():
            q_test_logits = qnn(torch.tensor(z_test_np, dtype=torch.float32))
            q_test_probs = torch.sigmoid(q_test_logits).numpy().squeeze(1)
        q_infer_time = time.time() - t0
        
        q_metrics = compute_binary_metrics(y_test, q_test_probs)
        benchmark_metrics[qnn_name] = q_metrics
        benchmark_timings[qnn_name] = {
            "train_s": q_train_time,
            "per_epoch_s": q_per_epoch_time,
            "infer_s": q_infer_time,
            "qubits": n_qubits,
            "layers": L,
            "quantum_params": actual_q_params,
            "circuit_depth": q_res["estimated_circuit_depth"],
            "cnot_count": q_res["two_qubit_cnot_gates"],
            "simulator": "default.qubit (CPU)"
        }
        print(f"QNN L={L} Total Train ({q_epochs} ep): {q_train_time:.2f}s ({q_per_epoch_time:.2f}s/ep) | Infer: {q_infer_time:.4f}s")
        print(f"Test ROC-AUC: {q_metrics['roc_auc']:.4f} | PR-AUC: {q_metrics['pr_auc']:.4f} | Acc: {q_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {q_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")

    # -------------------------------------------------------------------------
    # 6. SUMMARY REPORT & BENCHMARK COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("25K BENCHMARK: SUMMARY COMPARISON TABLE (~24,000 EVENTS)")
    print("=" * 95)
    
    total_bench_duration = time.time() - start_time_all
    final_ram = get_process_ram_mb()
    
    print(f"\nRAM Monitoring: Start: {initial_ram:.1f} MB -> Peak: {peak_ram:.1f} MB -> Final: {final_ram:.1f} MB")
    print(f"Total Benchmark Execution Time: {total_bench_duration:.2f}s ({total_bench_duration/60:.2f} minutes)")
    
    header = f"{'Model / Pipeline Stage':<28} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Accuracy':<9} | {'SigEff@99.9%':<12} | {'Train (s)':<10} | {'Infer (s)':<9} | {'Peak RAM':<9}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
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
    
    csv_rows = []
    csv_rows.append(["Model", "ROC-AUC", "PR-AUC", "Accuracy", "SigEff@99.9%", "Train_Time_s", "Infer_Time_s", "Peak_RAM_MB"])
    
    for m_name in model_order:
        if m_name in benchmark_metrics:
            m = benchmark_metrics[m_name]
            auc = m.get("roc_auc", 0.0)
            pr_auc = m.get("pr_auc", 0.0)
            acc = m.get("accuracy", 0.0)
            sig_eff = m.get("sig_eff_at_99.9_bkg_rej", 0.0)
            t_tr = benchmark_timings.get(m_name, {}).get("train_s", 0.0)
            t_inf = benchmark_timings.get(m_name, {}).get("infer_s", 0.0)
            print(f"{m_name:<28} | {auc:<8.4f} | {pr_auc:<8.4f} | {acc:<9.4f} | {sig_eff:<12.4f} | {t_tr:<10.2f} | {t_inf:<9.3f} | {peak_ram:<6.1f} MB")
            csv_rows.append([m_name, f"{auc:.6f}", f"{pr_auc:.6f}", f"{acc:.6f}", f"{sig_eff:.6f}", f"{t_tr:.3f}", f"{t_inf:.4f}", f"{peak_ram:.1f}"])
    print("-" * len(header))
    
    # Save CSV summary
    csv_summary_path = results_dir / "25k_benchmark_summary.csv"
    with open(csv_summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Saved CSV summary to {csv_summary_path}")
    
    # Save machine-readable results JSON
    def np_json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return str(obj)
        
    json_results_path = results_dir / "25k_benchmark_results.json"
    with open(json_results_path, "w") as f:
        json.dump({
            "experiment": "25k_real_data_scaling",
            "dataset_info": {
                "total_events": total_events,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "events_per_class": events_per_class,
                "n_classes": 12,
                "classes_distribution": split_validation.get("class_counts_train", {})
            },
            "timings": benchmark_timings,
            "metrics": {
                k: {mk: (mv if not isinstance(mv, np.ndarray) else None) for mk, mv in v.items() if mk != "latents"}
                for k, v in benchmark_metrics.items()
            },
            "latent_info": {
                "latent_dim": latent_dim,
                "mean": float(z_train_np.mean()),
                "std": float(z_train_np.std()),
                "min": float(z_train_np.min()),
                "max": float(z_train_np.max()),
                "nan_count": nan_count_z,
                "inf_count": inf_count_z
            },
            "memory": {
                "initial_ram_mb": float(initial_ram),
                "peak_ram_mb": float(peak_ram),
                "final_ram_mb": float(final_ram)
            },
            "total_duration_s": float(total_bench_duration),
            "simulator": "PennyLane default.qubit (CPU simulated)"
        }, f, indent=2, default=np_json_serializer)
    print(f"Saved machine-readable benchmark results to {json_results_path}")


if __name__ == "__main__":
    run_25k_benchmark()
