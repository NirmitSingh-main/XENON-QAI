# PAPER COMPLETION MATRIX: MANUSCRIPT PREPARATION MAP

**Manuscript Title:** *XENON-QAI: Deep Learning and Quantum Machine Learning for Dark Matter Candidate Identification in Liquid-Xenon Detectors*  
**Status:** All Empirical Data and Audit Evidence Prepared for Author Manuscript Update  

---

## 1. Manuscript Section-by-Section Alignment Matrix

| Manuscript Section | Required Information | Audit Output / File Reference | Status |
|:---|:---|:---|:---:|
| **Abstract** | Final 45k scaling metrics, model ranking, classical-quantum parity statement. | `45K_BENCHMARK_REPORT.md`<br>`results/45k_benchmark/45k_benchmark_results.json` | **Ready** |
| **Section 3: Dataset & Simulation** | 12 classes, exact 45,000 sampling distribution, CNNS 99-event caveat, 70/15/15 disjoint splits. | `45K_BENCHMARK_REPORT.md`<br>`PHYSICS_PIPELINE_AUDIT.md`<br>`45k_split_manifest.json` | **Ready** |
| **Section 4: Multimodal Architecture** | CNN layers, filter counts, kernel sizes, GELU, LayerNorm, 16D bottleneck, 315,297 parameters. | `HYPERPARAMETER_AUDIT.md`<br>`src/xenon_qai/models/cnn/model.py` | **Ready** |
| **Section 5: Variational Quantum Inference** | 4 qubits, $L \in \{2, 4, 6\}$, hardware-efficient ansatz, 16..48 circuit params vs 37 classical params, PennyLane CPU simulator. | `HYPERPARAMETER_AUDIT.md`<br>`COMPUTATIONAL_RESOURCE_REPORT.md` | **Ready** |
| **Section 6: Benchmark & Scaling Results** | 4-scale trajectory (1.2k, 5.6k, 22.1k, 45k), ROC-AUC, PR-AUC, accuracy, SigEff@99.9%, SigEff@99.99%. | `45k_benchmark_summary.csv`<br>`figures/benchmark_scaling_4scales.pdf` | **Ready** |
| **Section 7: Modality Ablation Study** | 9 configurations (Full, No S1, No S2, No PMT, No Phys, S1 only, S2 only, PMT only, Phys only). | `ABLATION_STUDY_REPORT.md`<br>`figures/modality_ablation_comparison.pdf` | **Ready** |
| **Section 8: Per-Class & Neutrino Fog Analysis**| 12-class breakdown, CNNS 75% misclassification, Wall 97.2% rejection, WIMP detection rates (>91%). | `PER_CLASS_EVALUATION_REPORT.md`<br>`figures/per_class_acceptance_rates.pdf` | **Ready** |
| **Section 9: Statistical Significance** | 1,000-bootstrap 95% confidence intervals, paired hypothesis testing, non-overlapping vs overlapping bounds. | `STATISTICAL_RELIABILITY_REPORT.md` | **Ready** |
| **Section 10: Discussion & Scientific Limitations** | No dark matter discovery claim, no quantum advantage claim, CPU simulation disclosure, data scarcity caveats. | `FINAL_RESEARCH_READINESS_REPORT.md` | **Ready** |

---

## 2. Table & Figure Placement Guide

1. **Table 1: Main 45k Benchmark Comparison** $\rightarrow$ Source: `results/45k_benchmark/45k_benchmark_summary.csv`.
2. **Table 2: Four-Scale Data Scaling Trajectory** $\rightarrow$ Source: `results/45k_benchmark/45k_benchmark_results.json` combined with 1.2k, 5.6k, and 22.1k verified logs.
3. **Table 3: Modality Ablation Matrix** $\rightarrow$ Source: `results/ablation_study/ablation_summary.csv`.
4. **Table 4: Per-Class Test Set Detection Statistics** $\rightarrow$ Source: `results/per_class_evaluation/per_class_breakdown.csv`.
5. **Figure 1: Main 45k Benchmark Bar Chart** $\rightarrow$ Source: `figures/benchmark_45k_performance.pdf`.
6. **Figure 2: Scaling Trajectory across 4 Empirical Scales** $\rightarrow$ Source: `figures/benchmark_scaling_4scales.pdf`.
7. **Figure 3: Modality Ablation Performance** $\rightarrow$ Source: `figures/modality_ablation_comparison.pdf`.
8. **Figure 4: Per-Class Acceptance & Confusion Matrix** $\rightarrow$ Source: `figures/per_class_acceptance_rates.pdf` and `figures/confusion_matrix_binary_45k.pdf`.

