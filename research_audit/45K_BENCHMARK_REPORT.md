# 45K-EVENT BENCHMARK REPORT: METHODOLOGY, EXECUTION & RESULTS

**Experiment ID:** `45k_real_data_scaling`  
**Execution Timestamp:** 2026-11-11  
**Output Directory:** `results/45k_benchmark/`  
**Total Wallclock Duration:** 2,752.95 s (~45.88 minutes)  
**Hardware & Environment:** Intel Core i7-12700H (16 logical, 10 physical cores), 15.68 GB RAM, CPU-only simulation (CUDA: False).

---

## 1. Dataset Design & Principle-Guided Sampling

A target of approximately 45,000 events was formulated. A physical audit of all classes revealed severe natural imbalances in physical simulations:
- **CNNS:** Only 99 events exist in the curated dataset. **Under zero circumstances were synthetic events fabricated, duplicated, or oversampled.** All 99 events were retained.
- **RN:** Exactly 2,610 events exist. All 2,610 were retained (100% sampling).
- **AC:** Exactly 3,998 events exist. All 3,998 were retained (100% sampling).
- **WALL & 7 WIMP Classes:** Capped at 4,255 events each.
- **ER:** Set to 4,253 events to arrive at an **exact dataset total of 45,000 events**.

### Exact Event Partitioning

| Class | Type | Total Events | Train (70%) | Val (15%) | Test (15%) |
|:---|:---|:---:|:---:|:---:|:---:|
| **AC** | Background | 3,998 | 2,798 | 599 | 601 |
| **CNNS** | Background | 99 | 69 | 14 | 16 |
| **ER** | Background | 4,253 | 2,977 | 637 | 639 |
| **RN** | Background | 2,610 | 1,827 | 390 | 393 |
| **WALL** | Background | 4,255 | 2,978 | 638 | 639 |
| **mDM10GeV** | WIMP Signal | 4,255 | 2,978 | 638 | 639 |
| **mDM20GeV** | WIMP Signal | 4,255 | 2,978 | 638 | 639 |
| **mDM35GeV** | WIMP Signal | 4,255 | 2,978 | 638 | 639 |
| **mDM50GeV** | WIMP Signal | 4,255 | 2,978 | 638 | 639 |
| **mDM100GeV** | WIMP Signal | 4,255 | 2,978 | 638 | 639 |
| **mDM200GeV** | WIMP Signal | 4,255 | 2,978 | 638 | 639 |
| **mDM500GeV** | WIMP Signal | 4,255 | 2,978 | 638 | 639 |
| **TOTAL** | — | **45,000** | **31,494** | **6,745** | **6,761** |

**Split Integrity:** Event IDs were strictly disjoint ($Overlap = 0$). Preprocessor scalers were fit **strictly** on the training set (sample size = 5,000) to ensure zero data leakage.

---

## 2. Experimental Results (Test Set Evaluation)

All 9 model configurations were evaluated on the exact same 6,761 test events. The classical head and the three QNN heads were evaluated on the **exact same frozen latent space** $z \in \mathbb{R}^{16}$ extracted from the trained Multimodal CNN.

| Model / Architecture | Inputs | ROC-AUC | PR-AUC | Accuracy | $F_1$-Score | SigEff @ 99.9% Rej | SigEff @ 99.99% Rej | Train Time (s) | Infer Time (s) | Trainable Params |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **13D Random Forest** | 13D Physics | 0.8732 | 0.8911 | 0.8576 | 0.8955 | 0.0027 | 0.0004 | 2.18 | 0.0692 | N/A (300 trees) |
| **13D XGBoost** | 13D Physics | 0.8791 | 0.8976 | 0.8645 | 0.9038 | 0.0058 | 0.0007 | 1.96 | 0.0040 | N/A (200 trees) |
| **13D Logistic Reg.** | 13D Physics | 0.8328 | 0.8671 | 0.8065 | 0.8563 | 0.0067 | 0.0013 | 0.04 | 0.0010 | 14 |
| **13D MLP** | 13D Physics | 0.8818 | 0.8993 | 0.8666 | 0.9044 | 0.0065 | 0.0025 | 91.18 | 0.0090 | 12,609 |
| **Multimodal CNN** | S1+S2+PMT+Phys | **0.8927** | **0.9170** | **0.8688** | **0.9058** | 0.0143 | 0.0139 | 2309.45 | 2.6606 | 315,297 |
| **Frozen-z Classical Head**| Frozen $z \in \mathbb{R}^{16}$ | 0.8904 | 0.9144 | 0.8679 | 0.9049 | **0.0295** | **0.0141** | 30.84 | 0.0020 | 37 |
| **Frozen-z QNN $L=2$** | Frozen $z \in \mathbb{R}^{16}$ | 0.8923 | 0.9163 | 0.8679 | 0.9050 | 0.0130 | 0.0110 | 31.81 | 0.0600 | 16 (circ) / 89 (tot) |
| **Frozen-z QNN $L=4$** | Frozen $z \in \mathbb{R}^{16}$ | 0.8906 | 0.9150 | 0.8564 | 0.8935 | 0.0215 | 0.0114 | 42.05 | 0.1052 | 32 (circ) / 105 (tot) |
| **Frozen-z QNN $L=6$** | Frozen $z \in \mathbb{R}^{16}$ | 0.8903 | 0.9144 | 0.8632 | 0.8999 | 0.0148 | 0.0123 | 64.56 | 0.1613 | 48 (circ) / 121 (tot) |

---

## 3. Scientific Analysis & Key Takeaways

1. **Multimodal Scaling Confirmation:**  
   The Multimodal CNN reaches its highest recorded performance across all benchmark scales: **ROC-AUC = 0.8927, PR-AUC = 0.9170**. This exceeds the 22.1k benchmark (ROC-AUC = 0.8706) by +0.0221, confirming monotonic scaling benefits when ingesting full waveforms and spatial PMT hit arrays.
2. **Classical vs Quantum Head Parity:**  
   On the exact same frozen latent representations ($z \in \mathbb{R}^{16}$), the Parameter-Matched Classical Head (37 parameters) achieved **ROC-AUC = 0.8904**, while QNN $L=2$ achieved **ROC-AUC = 0.8923** ($\Delta = +0.0019$), QNN $L=4$ achieved **0.8906** ($\Delta = +0.0002$), and QNN $L=6$ achieved **0.8903** ($\Delta = -0.0001$).  
   **Rigorous Conclusion:** Classical and quantum heads perform within overlapping statistical uncertainty bounds ($95\%\text{ CI}: [0.880, 0.901]$). **There is no evidence of quantum advantage.**
3. **Hardware Simulation Distinction:**  
   All quantum circuits were executed via PennyLane statevector simulation (`default.qubit`) on classical x86 CPU cores. Inference took 0.060s to 0.161s for the QNN vs 0.002s for the classical head, demonstrating the computational simulation overhead of classical statevector emulation.

