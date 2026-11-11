# REPOSITORY INVENTORY & CODEBASE MAP

**Repository:** XENON-QAI  
**Audit Date:** 2026-11-11  
**Status:** Complete Technical Inventory & Operational Verification  

---

## 1. Directory Tree & Architecture Overview

```
XENON-QAI/
├── configs/                      # Configuration YAML files (stubbed, 0 bytes; config in code)
├── data/                         # Compressed (.tar.xz) and extracted HDF5 datasets
│   └── extracted/                # Extracted HDF5 files per class (250,778 raw events available)
├── docs/                         # Theoretical documentation & architecture guides
├── experiments/                  # Experiment tracking stubs (001_dataset .. 016_explainability)
├── figures/                      # Publication-quality vector & raster figures (600 DPI PDF/PNG)
├── notebooks/                    # Jupyter notebooks for prototyping
├── research_audit/               # 12 rigorous audit reports & validation matrices
├── results/                      # Saved experimental outputs, checkpoints, and predictions
│   ├── 45k_benchmark/            # 45,000-event benchmark results (JSON, CSV, PT, NPZ)
│   ├── ablation_study/           # 9-modality ablation outputs
│   ├── per_class_evaluation/     # 12-class breakdown & 1000-sample bootstrap CIs
│   ├── 25k_benchmark_results.json # Verified 22,099-event scaling benchmark
│   └── intermediate_benchmark_results.json # Verified 5,599-event benchmark
├── scripts/                      # Benchmark runners, analysis, and figure generation scripts
├── src/xenon_qai/                # Core modular Python library
│   ├── anomaly/                  # Classical and quantum anomaly detection
│   ├── data/                     # Ingestion, padding, resampling, feature extraction, splitting
│   ├── domain/                   # Domain adaptation routines
│   ├── evaluation/               # Physics metrics, ROC, PR, SigEff@99.9%, bootstrap CIs
│   ├── explainability/           # Integrated gradients and feature importance
│   ├── models/                   # Classical baselines (RF, XGB, LogReg, MLP) and Multimodal CNN
│   ├── physics/                  # Physics consistency and pulse validation
│   ├── quantum/                  # PennyLane VQC, circuit encoders, and classical-matched heads
│   ├── representation/           # Latent bottleneck, PCA, UMAP, visualization
│   ├── representations/          # Multi-channel data wrappers
│   ├── uncertainty/              # Calibration, MC-Dropout, temperature scaling
│   └── utils/                    # Seed control, device allocation, checkpoints
└── tests/                        # Pytest suite
```

---

## 2. Core Python Source Modules

| Subpackage | Key Files | Functionality & Implementation Status |
|:---|:---|:---|
| `xenon_qai.data` | `types.py`<br>`features.py`<br>`preprocessing.py`<br>`splitting.py`<br>`dataset.py`<br>`loader.py`<br>`validation.py` | **Fully operational.** Computes 13D physics features, pads S1 (160), resamples S2 (500), standardizes top/bot PMTs (248), generates deterministic stratified splits with zero leakage. |
| `xenon_qai.models.baselines` | `random_forest.py`<br>`xgboost.py`<br>`logistic.py`<br>`mlp.py` | **Fully operational.** Scikit-learn, XGBoost, and PyTorch MLP architectures with unified `fit()`, `predict_proba()`, and `evaluate()` interfaces. |
| `xenon_qai.models.cnn` | `model.py`<br>`train.py` | **Fully operational.** 4-branch deep multimodal network: S1 1D-CNN, S2 1D-CNN (dilated), PMT Spatial MLP, Physics MLP with 16D bottleneck latent $z$. |
| `xenon_qai.quantum` | `qnn.py`<br>`classical_head.py`<br>`circuit.py`<br>`encoding.py`<br>`feature_map.py`<br>`resources.py`<br>`noise.py`<br>`qkernel.py` | **Fully operational (Simulated).** PennyLane `default.qubit` statevector simulation on CPU. Parameter-matched classical head (37 params) vs QNN $L=2, 4, 6$. |
| `xenon_qai.evaluation` | `metrics.py`<br>`comparison.py` | **Fully operational.** ROC-AUC, PR-AUC, accuracy, $F_1$, and rare-event physics metric: signal efficiency at 99.9% and 99.99% background rejection. |
| `xenon_qai.anomaly` | `detector.py`<br>`quantum_anomaly.py`<br>`ranking.py`<br>`scoring.py` | **Code implemented, partially tested.** Standalone unsupervised scoring; not part of primary supervised benchmark tables. |
| `xenon_qai.uncertainty` | `calibration.py`<br>`confidence.py`<br>`uncertainty.py` | **Code implemented.** Temperature scaling, ECE calculation, and MC-Dropout routines. |
| `xenon_qai.explainability` | `attribution.py`<br>`importance.py`<br>`integrated_gradients.py` | **Code implemented.** Captum-compatible feature attributions. |
| `xenon_qai.representation` | `dimensionality_reduction.py`<br>`visualization.py` | **Code implemented.** Latent space projection, PCA, and UMAP wrappers. |

---

## 3. Data Availability & Verified Raw Counts

Across `data/extracted/`, 250,778 raw dual-phase xenon simulation events exist in HDF5 format:
- **AC (Accidental Coincidence):** 3,998 events (`AC_all.hdf5`)
- **CNNS (Coherent Neutrino-Nucleus Scattering):** 99 events (`CNNS_all.hdf5`)
- **ER (Electronic Recoil Background):** 35,549 events (`ER_all.hdf5`)
- **RN (Radon / Radioactive Alpha/Beta Background):** 2,610 events (`RN_all.hdf5`)
- **WALL (Surface Wall Charge Accumulation):** 4,642 events (`WALL_all.hdf5`)
- **WIMP Signals (10, 20, 35, 50, 100, 200, 500 GeV):** 203,880 total events (~20k to 32k per mass class)

