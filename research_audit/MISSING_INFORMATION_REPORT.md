# MISSING INFORMATION & SCIENTIFIC GAP ANALYSIS

**Repository:** XENON-QAI  
**Audit Date:** 2026-11-11  

---

## 1. Resolved vs. Remaining Information Gaps

| Audit Item | Status Prior to Audit | Current Status Post-Audit | Notes / Action Taken |
|:---|:---:|:---:|:---|
| **45k Benchmark Execution** | Missing | **Resolved** | Executed 45,000 events across 9 models; saved to `results/45k_benchmark/`. |
| **CNNS Statistical Support** | Ambiguous | **Resolved** | Proved only 99 events exist; 75% misclassification observed in test set. |
| **Hyperparameters** | Incomplete in text | **Resolved** | Recovered all encoder layers, learning rates, optimizers, and circuit depths. |
| **Quantum vs Classical Parameter Match** | Assumed equal | **Resolved** | Classical head (37 params) matches quantum circuit params (16..48), not hybrid wrapper (89..121). |
| **Modality Ablation Study** | Missing | **Resolved** | Completed all 9 configurations under identical splits in `results/ablation_study/`. |
| **Per-Class 12-Class Breakdown** | Missing | **Resolved** | Computed and saved to `results/per_class_evaluation/`. |
| **Confidence Intervals (Statistical Reliability)**| Missing | **Resolved** | Computed 1,000-iteration bootstrap 95% CIs for all 9 models. |
| **Hardware & Simulator Distinctions** | Unclear | **Resolved** | Documented exact CPU statevector simulation parameters; disclaimed physical QPU execution. |

---

## 2. Gaps That Must Remain Future Work / Manuscript Caveats

1. **Physical Quantum Hardware Validation:**  
   The QNN was simulated classically on CPU. Demonstrating noise resilience or quantum utility on real NISQ hardware (IBM Quantum, Rigetti, or IonQ) remains future work.
2. **CNNS Event Statistics:**  
   Because only 99 CNNS events exist in the simulated dataset, claims regarding machine learning discrimination against the solar neutrino fog must be qualified as preliminary exploratory observations pending larger Monte Carlo datasets.
3. **Continuous Photomultiplier Timing:**  
   The dataset provides discretized, binned time series. Continuous pulse deconvolution and single-photon timing calibrations were not part of the dataset and should be documented as detector-level preprocessing assumptions.

