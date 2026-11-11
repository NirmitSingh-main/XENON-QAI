# COMPUTATIONAL RESOURCE REPORT

**Repository:** XENON-QAI  
**Audit Date:** 2026-11-11  

---

## 1. Hardware Environment Specifications

| Component | Hardware Specification / Setting |
|:---|:---|
| **CPU Model** | Intel Core i7-12700H (Intel64 Family 6 Model 154 Stepping 3, GenuineIntel) |
| **CPU Core Topology** | 16 Logical Threads (10 Physical Cores: 6 Performance + 4 Efficient) |
| **System RAM** | 15.68 GB Total Physical Memory |
| **GPU Model & VRAM** | None active / CUDA: False (`torch.cuda.is_available() == False`) |
| **Operating System** | Windows 11 Enterprise (Build 10.0.26200-SP0) |
| **Python Environment** | Python 3.11.9 (tags/v3.11.9:de54cf5) |

---

## 2. Software Dependencies & Library Versions

| Package | Version |
|:---|:---|
| **PyTorch** | `2.14.0+cpu` |
| **PennyLane** | `0.45.1` |
| **Scikit-Learn** | `1.9.1` |
| **XGBoost** | `3.2.0` |
| **NumPy** | `1.26.4` |
| **h5py** | `3.11.0` |
| **psutil** | `5.9.8` |

---

## 3. Operational Profile: 45K Benchmark Resource Consumption

| Pipeline Stage | Train Time (s) | Inference Time (s) | Memory Delta / Peak RAM |
|:---|:---:|:---:|:---:|
| **Data Partitioning & Split Manifest** | 0.85s | — | Peak RAM: 412 MB |
| **Preprocessor Fitting (5,000 samples)** | 34.25s | — | Peak RAM: 580 MB |
| **Tabular Array Extraction (45,000 events)** | 106.58s | — | Peak RAM: 820 MB |
| **13D Random Forest** | 2.18s | 0.0692s | Peak RAM: 890 MB |
| **13D XGBoost** | 1.96s | 0.0040s | Peak RAM: 910 MB |
| **13D Logistic Regression** | 0.04s | 0.0010s | Peak RAM: 910 MB |
| **13D MLP (35 epochs)** | 91.18s | 0.0090s | Peak RAM: 935 MB |
| **Multimodal CNN (15 epochs)** | 2309.45s | 2.6606s | Peak RAM: 1,212.0 MB |
| **Frozen Latent Space Extraction** | 3.12s | — | Peak RAM: 1,212.0 MB |
| **Parameter-Matched Classical Head** | 30.84s | 0.0020s | Peak RAM: 1,212.0 MB |
| **Frozen-z QNN $L=2$** | 31.81s | 0.0600s | Peak RAM: 1,212.0 MB |
| **Frozen-z QNN $L=4$** | 42.05s | 0.1052s | Peak RAM: 1,212.0 MB |
| **Frozen-z QNN $L=6$** | 64.56s | 0.1613s | Peak RAM: 1,212.0 MB |
| **Total 45k Benchmark Execution** | **2,752.95s** (~45.88 min) | — | **Peak RAM: 1,212.0 MB** |

---

## 4. Quantum Simulation vs. Hardware Execution Statement

> [!IMPORTANT]
> All quantum variational circuits evaluated in this study were executed via statevector algebra on classical CPU cores using PennyLane's `default.qubit` device. They represent classical numerical simulations of idealized, noiseless quantum states. They **do not** reflect execution on physical quantum processors (such as superconducting qubits, trapped ions, or neutral atoms) and do not provide empirical evidence of quantum speedup or physical fault tolerance.

