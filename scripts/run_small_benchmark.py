"""
Small Real-Data Benchmark for XENON-QAI.

Executes a representative small real-data benchmark on CPU:
- Data loader & preprocessor timing and memory
- 13D Physics feature baselines (RF, XGB, Logistic, MLP)
- Multimodal CNN (1D-CNN + PMT MLP + Physics MLP -> Latent z in R^16)
- Parameter-matched Classical Head on frozen latents
- Quantum Head (QNN) on the same frozen latents
- Detailed profiling of loading time, training time, inference time, RAM, NaN/Inf checks, and feasibility analysis.
"""

from pathlib import Path
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from xenon_qai.utils.seed import set_seed
from xenon_qai.data import create_dataloaders
from xenon_qai.data.types import MULTIMODAL_DIM, PHYSICS_FEATURE_DIM
from xenon_qai.models.baselines import (
    XenonRandomForest,
    XenonXGBoost,
    XenonLogisticRegression,
    XenonMLP
)
from xenon_qai.models.cnn import XenonMultimodalCNN, train_xenon_cnn, evaluate_model
from xenon_qai.quantum import (
    QuantumVariationalClassifier,
    ParameterMatchedClassicalHead,
    estimate_circuit_resources
)
from xenon_qai.evaluation.metrics import compute_binary_metrics, format_metrics_table
from xenon_qai.evaluation.comparison import compare_models, format_comparison_table


def get_ram_mb() -> float:
    """Returns current process RSS memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def extract_features_and_labels_from_loader(loader: DataLoader):
    """Extracts numpy arrays from DataLoader for tabular baselines."""
    multi_list, phys_list, sig_list = [], [], []
    for batch in loader:
        if len(batch) == 4:
            x_m, x_p, y_s, _ = batch
        elif len(batch) == 3:
            x_m, y_s, _ = batch
            x_p = None
        else:
            x_m, y_s = batch
            x_p = None
            
        multi_list.append(x_m.numpy())
        if x_p is not None:
            phys_list.append(x_p.numpy())
        sig_list.append(y_s.numpy())
        
    x_multi = np.concatenate(multi_list, axis=0)
    x_phys = np.concatenate(phys_list, axis=0) if phys_list else None
    y_sig = np.concatenate(sig_list, axis=0)
    return x_multi, x_phys, y_sig


def run_benchmark():
    print("=" * 80)
    print("XENON-QAI: SMALL REAL-DATA CPU BENCHMARK")
    print("=" * 80)
    
    set_seed(42)
    start_bench_time = time.time()
    initial_ram = get_ram_mb()
    print(f"Initial Process RAM: {initial_ram:.2f} MB")
    
    data_dir = Path("data/extracted")
    if not data_dir.exists():
        raise FileNotFoundError(f"Extracted dataset directory not found at {data_dir}")
        
    # -------------------------------------------------------------------------
    # 1. DATA LOADING & PREPROCESSING TIMING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("1. DATA LOADING & PREPROCESSING BENCHMARK")
    print("-" * 80)
    
    events_per_class = 100  # 12 classes * 100 = 1,200 events
    print(f"Loading representative subset: {events_per_class} events per class across 12 classes (1,200 total events)...")
    
    t0 = time.time()
    train_loader, val_loader, test_loader, preprocessor, splits = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        mode="both",
        split_ratios=(0.70, 0.15, 0.15),
        random_seed=42,
        max_events_per_class=events_per_class,
        fit_preprocessor=True
    )
    load_and_prep_time = time.time() - t0
    ram_after_load = get_ram_mb()
    
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    print(f"Split sizes -> Train: {n_train} | Val: {n_val} | Test: {n_test} (Total: {n_train + n_val + n_test})")
    print(f"Loading & Preprocessing setup time: {load_and_prep_time:.2f} seconds")
    print(f"RAM after loading: {ram_after_load:.2f} MB (Delta: +{ram_after_load - initial_ram:.2f} MB)")
    
    # Verify first batch shapes & check NaN/Inf
    first_batch = next(iter(train_loader))
    x_multi_batch, x_phys_batch, y_sig_batch, y_cls_batch = first_batch
    print(f"Batch Tensor Shapes:")
    print(f"  - Multimodal (S1 + S2 + PMTs): {x_multi_batch.shape}, dtype={x_multi_batch.dtype}")
    print(f"  - Physics Features (13D):      {x_phys_batch.shape}, dtype={x_phys_batch.dtype}")
    print(f"  - Binary Label (is_signal):    {y_sig_batch.shape}, dtype={y_sig_batch.dtype}")
    print(f"  - Class Index (0..11):         {y_cls_batch.shape}, dtype={y_cls_batch.dtype}")
    
    has_nan = torch.isnan(x_multi_batch).any() or torch.isnan(x_phys_batch).any()
    has_inf = torch.isinf(x_multi_batch).any() or torch.isinf(x_phys_batch).any()
    print(f"Integrity Check: NaN detected = {bool(has_nan)}, Inf detected = {bool(has_inf)}")
    assert not has_nan and not has_inf, "Error: NaN or Inf detected in data batch!"
    
    # Extract tabular arrays for 13D baselines
    print("\nExtracting preprocessed feature arrays...")
    t0 = time.time()
    x_multi_train, x_phys_train, y_train = extract_features_and_labels_from_loader(train_loader)
    x_multi_val, x_phys_val, y_val = extract_features_and_labels_from_loader(val_loader)
    x_multi_test, x_phys_test, y_test = extract_features_and_labels_from_loader(test_loader)
    feature_extract_time = time.time() - t0
    print(f"Extracted feature arrays in {feature_extract_time:.2f}s | Train shape: {x_phys_train.shape}, Test shape: {x_phys_test.shape}")
    
    results = {}
    timings = {}
    
    # -------------------------------------------------------------------------
    # 2. PRIORITY 1: 13D PHYSICS BASELINES
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("2. PRIORITY 1: 13D PHYSICS BASELINES")
    print("-" * 80)
    
    # 2a. Random Forest
    print("\n--- [2a] Random Forest (13D Physics Features) ---")
    rf = XenonRandomForest(n_estimators=100, min_samples_leaf=5, random_state=42)
    t0 = time.time()
    rf.fit(x_phys_train, y_train)
    rf_train_time = time.time() - t0
    
    t0 = time.time()
    rf_metrics = rf.evaluate(x_phys_test, y_test)
    rf_infer_time = time.time() - t0
    
    results["Random Forest (13D)"] = rf_metrics
    timings["Random Forest (13D)"] = {"train_time_s": rf_train_time, "infer_time_s": rf_infer_time}
    print(f"Train time: {rf_train_time:.3f}s | Infer time: {rf_infer_time:.3f}s")
    print(f"Test ROC-AUC: {rf_metrics['roc_auc']:.4f} | PR-AUC: {rf_metrics['pr_auc']:.4f} | Acc: {rf_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {rf_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 2b. XGBoost
    print("\n--- [2b] XGBoost (13D Physics Features) ---")
    xgb = XenonXGBoost(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    t0 = time.time()
    xgb.fit(x_phys_train, y_train)
    xgb_train_time = time.time() - t0
    
    t0 = time.time()
    xgb_metrics = xgb.evaluate(x_phys_test, y_test)
    xgb_infer_time = time.time() - t0
    
    results["XGBoost (13D)"] = xgb_metrics
    timings["XGBoost (13D)"] = {"train_time_s": xgb_train_time, "infer_time_s": xgb_infer_time}
    print(f"Train time: {xgb_train_time:.3f}s | Infer time: {xgb_infer_time:.3f}s")
    print(f"Test ROC-AUC: {xgb_metrics['roc_auc']:.4f} | PR-AUC: {xgb_metrics['pr_auc']:.4f} | Acc: {xgb_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {xgb_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # 2c. Logistic Regression
    print("\n--- [2c] Logistic Regression (13D Physics Features) ---")
    lr = XenonLogisticRegression(C=1.0, max_iter=500, random_state=42)
    t0 = time.time()
    lr.fit(x_phys_train, y_train)
    lr_train_time = time.time() - t0
    
    t0 = time.time()
    lr_metrics = lr.evaluate(x_phys_test, y_test)
    lr_infer_time = time.time() - t0
    
    results["Logistic Regression (13D)"] = lr_metrics
    timings["Logistic Regression (13D)"] = {"train_time_s": lr_train_time, "infer_time_s": lr_infer_time}
    print(f"Train time: {lr_train_time:.3f}s | Infer time: {lr_infer_time:.3f}s")
    print(f"Test ROC-AUC: {lr_metrics['roc_auc']:.4f} | PR-AUC: {lr_metrics['pr_auc']:.4f} | Acc: {lr_metrics['accuracy']:.4f}")
    
    # 2d. MLP (13D Physics Features)
    print("\n--- [2d] MLP (13D Physics Features) ---")
    mlp_13d = XenonMLP(input_dim=13, hidden_dims=(64, 32, 16), epochs=30, batch_size=32, device="cpu")
    t0 = time.time()
    mlp_13d.fit(x_phys_train, y_train, X_val=x_phys_val, y_val=y_val)
    mlp_train_time = time.time() - t0
    
    t0 = time.time()
    mlp_metrics = mlp_13d.evaluate(x_phys_test, y_test)
    mlp_infer_time = time.time() - t0
    
    results["MLP (13D)"] = mlp_metrics
    timings["MLP (13D)"] = {"train_time_s": mlp_train_time, "infer_time_s": mlp_infer_time}
    print(f"Train time (30 epochs): {mlp_train_time:.3f}s ({mlp_train_time/30:.3f}s/epoch) | Infer time: {mlp_infer_time:.3f}s")
    print(f"Test ROC-AUC: {mlp_metrics['roc_auc']:.4f} | PR-AUC: {mlp_metrics['pr_auc']:.4f} | Acc: {mlp_metrics['accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # 3. PRIORITY 2: MULTIMODAL CNN -> LATENT z IN R^16
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("3. PRIORITY 2: MULTIMODAL CNN (WAVEFORMS + PMTS + PHYSICS) -> LATENT z")
    print("-" * 80)
    
    latent_dim = 16
    cnn_model = XenonMultimodalCNN(
        latent_dim=latent_dim,
        num_classes=1,
        include_physics_features=True,
        dropout=0.15
    )
    
    cnn_epochs = 15
    print(f"Training XenonMultimodalCNN for {cnn_epochs} epochs on CPU...")
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
    
    results["Multimodal CNN (End-to-End)"] = cnn_test_metrics
    timings["Multimodal CNN"] = {"train_time_s": cnn_total_train_time, "per_epoch_s": cnn_per_epoch_time}
    print(f"\nCNN Total Train time ({cnn_epochs} epochs): {cnn_total_train_time:.2f}s ({cnn_per_epoch_time:.3f}s/epoch)")
    print(f"Test ROC-AUC: {cnn_test_metrics['roc_auc']:.4f} | PR-AUC: {cnn_test_metrics['pr_auc']:.4f} | Acc: {cnn_test_metrics['accuracy']:.4f} | Sig Eff @ 99.9% Rej: {cnn_test_metrics['sig_eff_at_99.9_bkg_rej']:.4f}")
    
    # Extract frozen latent representations z for all splits
    print("\nExtracting frozen latent embeddings z in R^16 from trained CNN...")
    cnn_model.eval()
    dev = torch.device("cpu")
    with torch.no_grad():
        _, z_train = cnn_model(torch.tensor(x_multi_train, dtype=torch.float32), torch.tensor(x_phys_train, dtype=torch.float32))
        _, z_val = cnn_model(torch.tensor(x_multi_val, dtype=torch.float32), torch.tensor(x_phys_val, dtype=torch.float32))
        _, z_test = cnn_model(torch.tensor(x_multi_test, dtype=torch.float32), torch.tensor(x_phys_test, dtype=torch.float32))
        
    z_train_np = z_train.numpy()
    z_val_np = z_val.numpy()
    z_test_np = z_test.numpy()
    print(f"Frozen latents extracted -> Train: {z_train_np.shape} | Val: {z_val_np.shape} | Test: {z_test_np.shape}")
    print(f"Latent range: [{z_train_np.min():.3f}, {z_train_np.max():.3f}], Mean: {z_train_np.mean():.3f}, Std: {z_train_np.std():.3f}")

    # -------------------------------------------------------------------------
    # 4. PRIORITY 3: PARAMETER-MATCHED CLASSICAL HEAD ON FROZEN LATENTS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("4. PRIORITY 3: PARAMETER-MATCHED CLASSICAL HEAD ON FROZEN LATENTS")
    print("-" * 80)
    
    # Target parameter budget matching 4-qubit, 2-layer QNN: ~24-32 parameters
    q_resource = estimate_circuit_resources(n_qubits=4, n_layers=2, ansatz_type="hardware_efficient")
    target_param_count = q_resource["trainable_parameters"] + 4  # ~20 parameters
    print(f"Target Parameter Budget to match QNN: {target_param_count} parameters")
    
    classical_head = ParameterMatchedClassicalHead(
        in_features=latent_dim,
        target_param_count=target_param_count,
        out_features=1
    )
    actual_classical_params = classical_head.count_parameters()
    print(f"Classical Head Parameter Count: {actual_classical_params} parameters")
    
    # Train classical head on frozen latents
    t0 = time.time()
    c_train_t = torch.tensor(z_train_np, dtype=torch.float32)
    c_y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    c_dataset = TensorDataset(c_train_t, c_y_train_t)
    c_loader = DataLoader(c_dataset, batch_size=32, shuffle=True)
    
    c_opt = optim.AdamW(classical_head.parameters(), lr=0.01, weight_decay=1e-4)
    c_crit = nn.BCEWithLogitsLoss()
    
    c_epochs = 20
    classical_head.train()
    for ep in range(c_epochs):
        for bx, by in c_loader:
            c_opt.zero_grad()
            out = classical_head(bx)
            loss = c_crit(out, by)
            loss.backward()
            c_opt.step()
    c_train_time = time.time() - t0
    
    # Evaluate classical head
    t0 = time.time()
    classical_head.eval()
    with torch.no_grad():
        c_test_logits = classical_head(torch.tensor(z_test_np, dtype=torch.float32))
        c_test_probs = torch.sigmoid(c_test_logits).numpy().squeeze(1)
    c_infer_time = time.time() - t0
    
    c_metrics = compute_binary_metrics(y_test, c_test_probs)
    results["Classical Head (Frozen z)"] = c_metrics
    timings["Classical Head (Frozen z)"] = {"train_time_s": c_train_time, "infer_time_s": c_infer_time}
    print(f"Classical Head Train time ({c_epochs} epochs): {c_train_time:.3f}s | Infer time: {c_infer_time:.3f}s")
    print(f"Test ROC-AUC: {c_metrics['roc_auc']:.4f} | PR-AUC: {c_metrics['pr_auc']:.4f} | Acc: {c_metrics['accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # 5. PRIORITY 4: QUANTUM HEAD ON THE SAME FROZEN LATENTS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("5. PRIORITY 4: QUANTUM HEAD (QNN) ON THE SAME FROZEN LATENTS")
    print("-" * 80)
    
    n_qubits = 4
    n_layers = 2
    qnn = QuantumVariationalClassifier(
        n_qubits=n_qubits,
        n_layers=n_layers,
        in_features=latent_dim,
        num_classes=1,
        dev_name="default.qubit"
    )
    q_params = qnn.count_quantum_parameters()
    print(f"Quantum Variational Head: {n_qubits} qubits, {n_layers} layers | Quantum Parameters: {q_params}")
    
    # Train QNN on the exact same frozen latents
    q_epochs = 10
    q_batch_size = 32
    print(f"Training QuantumVariationalClassifier for {q_epochs} epochs (batch_size={q_batch_size}) on CPU...")
    
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
        print(f"  QNN Epoch {ep:2d}/{q_epochs} | Loss: {avg_loss:.4f} | Time: {ep_time:.2f}s ({ep_time/len(q_loader):.3f}s/batch)")
        
    q_train_time = time.time() - t0
    q_per_epoch_time = q_train_time / q_epochs
    
    # Evaluate quantum head
    t0 = time.time()
    qnn.eval()
    with torch.no_grad():
        q_test_logits = qnn(torch.tensor(z_test_np, dtype=torch.float32))
        q_test_probs = torch.sigmoid(q_test_logits).numpy().squeeze(1)
    q_infer_time = time.time() - t0
    
    q_metrics = compute_binary_metrics(y_test, q_test_probs)
    results["Quantum Head (Frozen z)"] = q_metrics
    timings["Quantum Head (Frozen z)"] = {"train_time_s": q_train_time, "infer_time_s": q_infer_time, "per_epoch_s": q_per_epoch_time}
    print(f"\nQuantum Head Total Train time ({q_epochs} epochs): {q_train_time:.2f}s ({q_per_epoch_time:.2f}s/epoch)")
    print(f"Test ROC-AUC: {q_metrics['roc_auc']:.4f} | PR-AUC: {q_metrics['pr_auc']:.4f} | Acc: {q_metrics['accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # 6. SUMMARY REPORT & BENCHMARK COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY & PERFORMANCE COMPARISON")
    print("=" * 80)
    
    final_ram = get_ram_mb()
    total_bench_duration = time.time() - start_bench_time
    
    print(f"\nMemory Footprint: Start: {initial_ram:.1f} MB -> Peak/Final: {final_ram:.1f} MB (Delta: +{final_ram - initial_ram:.1f} MB)")
    print(f"Total Benchmark Execution Time: {total_bench_duration:.2f} seconds ({total_bench_duration/60:.2f} minutes)")
    
    # Print metrics table
    print("\n" + "-" * 90)
    print(f"{'Model / Pipeline Stage':<32} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'Accuracy':<9} | {'SigEff@99.9%':<12} | {'Train (s)':<9}")
    print("-" * 90)
    for model_name, m in results.items():
        auc = m.get("roc_auc", 0.0)
        pr_auc = m.get("pr_auc", 0.0)
        acc = m.get("accuracy", 0.0)
        sig_eff = m.get("sig_eff_at_99.9_bkg_rej", 0.0)
        t_tr = timings.get(model_name, {}).get("train_time_s", 0.0)
        print(f"{model_name:<32} | {auc:<9.4f} | {pr_auc:<9.4f} | {acc:<9.4f} | {sig_eff:<12.4f} | {t_tr:<9.2f}")
    print("-" * 90)
    
    # Side-by-side Classical vs Quantum Head comparison on identical frozen latents
    comp = compare_models(c_metrics, q_metrics, model_names=("Classical Head", "Quantum Head"))
    print("\n" + format_comparison_table(comp, title="Classical Head vs Quantum Head (Identical Frozen Latents)"))
    
    return {
        "results": results,
        "timings": timings,
        "ram": {"initial": initial_ram, "final": final_ram, "delta": final_ram - initial_ram},
        "total_duration_s": total_bench_duration
    }


if __name__ == "__main__":
    run_benchmark()
