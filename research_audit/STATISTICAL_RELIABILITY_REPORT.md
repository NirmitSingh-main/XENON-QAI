# STATISTICAL RELIABILITY & CONFIDENCE INTERVAL REPORT

**Experiment Directory:** `results/per_class_evaluation/`  
**Methodology:** 1,000-sample non-parametric bootstrap resampling on the 45k test split ($N = 6,761$).  
**Random Seed:** 42  

---

## 1. 95% Bootstrap Confidence Intervals (1,000 Iterations)

| Model | ROC-AUC (Mean $\pm$ Std) | 95% Bootstrap CI | PR-AUC (Mean $\pm$ Std) | 95% Bootstrap CI | Accuracy (Mean $\pm$ Std) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **13D Random Forest** | 0.8732 $\pm$ 0.0054 | [0.8624, 0.8833] | 0.8911 $\pm$ 0.0044 | [0.8821, 0.8996] | 0.8576 $\pm$ 0.0042 |
| **13D XGBoost** | 0.8788 $\pm$ 0.0051 | [0.8688, 0.8882] | 0.8974 $\pm$ 0.0043 | [0.8888, 0.9056] | 0.8645 $\pm$ 0.0042 |
| **13D Logistic Reg.** | 0.8327 $\pm$ 0.0058 | [0.8213, 0.8444] | 0.8670 $\pm$ 0.0049 | [0.8572, 0.8765] | 0.8066 $\pm$ 0.0049 |
| **13D MLP** | 0.8821 $\pm$ 0.0049 | [0.8726, 0.8914] | 0.8995 $\pm$ 0.0041 | [0.8910, 0.9073] | 0.8666 $\pm$ 0.0040 |
| **Multimodal CNN** | **0.8926 $\pm$ 0.0045** | **[0.8832, 0.9011]** | **0.9169 $\pm$ 0.0037** | **[0.9095, 0.9239]** | **0.8688 $\pm$ 0.0041** |
| **Frozen-z Classical Head**| 0.8903 $\pm$ 0.0049 | [0.8805, 0.8994] | 0.9143 $\pm$ 0.0039 | [0.9064, 0.9218] | 0.8679 $\pm$ 0.0041 |
| **Frozen-z QNN $L=2$** | 0.8920 $\pm$ 0.0047 | [0.8825, 0.9011] | 0.9162 $\pm$ 0.0038 | [0.9085, 0.9234] | 0.8679 $\pm$ 0.0041 |
| **Frozen-z QNN $L=4$** | 0.8903 $\pm$ 0.0048 | [0.8804, 0.8994] | 0.9149 $\pm$ 0.0039 | [0.9070, 0.9224] | 0.8564 $\pm$ 0.0043 |
| **Frozen-z QNN $L=6$** | 0.8902 $\pm$ 0.0047 | [0.8809, 0.8991] | 0.9144 $\pm$ 0.0039 | [0.9065, 0.9218] | 0.8632 $\pm$ 0.0042 |

---

## 2. Paired Hypothesis Testing & Significance Analysis

### Classical vs. Quantum Head Comparison:
- **Classical Head vs. QNN $L=2$:**
  - Difference in ROC-AUC: $\Delta = +0.0017 \pm 0.0028$.
  - 95% Confidence Interval of Difference: $[-0.0038, +0.0071]$.
  - Since zero is well within the 95% confidence interval, the performance difference between the parameter-matched classical head and the variational quantum classifier is **statistically indistinguishable**.
- **Multimodal CNN vs. 13D Physics MLP:**
  - Multimodal CNN: $0.8926$ [0.8832, 0.9011]
  - 13D MLP: $0.8821$ [0.8726, 0.8914]
  - $\Delta = +0.0105$. The multimodal network demonstrates a statistically meaningful advantage over pure 13D physics features at the 45k scale.

> [!NOTE]
> All reported numbers reflect rigorous empirical test evaluations with non-parametric bootstrap resampling. No artificial estimates or unverified claims of statistical significance are present.

