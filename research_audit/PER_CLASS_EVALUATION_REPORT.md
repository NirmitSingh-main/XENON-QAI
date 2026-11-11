# PER-CLASS EVALUATION & CNNS NEUTRINO-FOG AUDIT

**Output Directory:** `results/per_class_evaluation/`  
**Dataset:** 45,000-event test split (6,761 total events: 2,288 background, 4,473 WIMP signal).  
**Evaluator:** Multimodal CNN (`best_multimodal_cnn.pt`).  

---

## 1. Per-Class Test Breakdown (12 Classes)

| Class Name | Physical Category | Test Support ($N$) | Mean Model Confidence | Std. Dev. | Classified as Signal ($\hat{y}=1$) | Signal Acceptance Rate (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AC** | Background | 601 | 0.3874 | 0.3728 | 248 | 41.3% |
| **CNNS** | Neutrino Fog Bkg | **16** | **0.7425** | 0.2407 | **12** | **75.0%** |
| **ER** | Electronic Recoil | 639 | 0.2150 | 0.2919 | 128 | 20.0% |
| **RN** | Radiogenic Bkg | 393 | 0.6504 | 0.3223 | 275 | 70.0% |
| **WALL** | Surface Background | 639 | 0.0369 | 0.1210 | 18 | **2.8%** |
| **mDM10GeV** | WIMP Signal | 639 | 0.8585 | 0.1313 | 622 | 97.3% |
| **mDM20GeV** | WIMP Signal | 639 | 0.8844 | 0.1206 | 623 | 97.5% |
| **mDM35GeV** | WIMP Signal | 639 | 0.8708 | 0.1280 | 620 | 97.0% |
| **mDM50GeV** | WIMP Signal | 639 | 0.8578 | 0.1413 | 617 | 96.6% |
| **mDM100GeV** | WIMP Signal | 639 | 0.8356 | 0.1685 | 603 | 94.4% |
| **mDM200GeV** | WIMP Signal | 639 | 0.8253 | 0.1782 | 595 | 93.1% |
| **mDM500GeV** | WIMP Signal | 639 | 0.8268 | 0.1881 | 587 | 91.9% |

---

## 2. In-Depth Audit of CNNS & The Neutrino Fog

### The Statistical Reality of CNNS
- Across the entire curated simulated dataset, **only 99 CNNS events exist in total**.
- In the 45k benchmark test set, there are **only 16 CNNS events**.
- The Multimodal CNN assigned CNNS a mean confidence score of **0.7425**, with **75.0% (12 of 16 events)** classified as signal at the default threshold ($\theta = 0.5$).
- RN background likewise exhibited a high false-positive acceptance of **70.0%**.

### Physical Reasons for High CNNS Acceptance
CNNS (Coherent Neutrino-Nucleus Scattering, predominantly from solar $^8\text{B}$ neutrinos) and low-mass WIMPs (10–20 GeV) both induce single nuclear recoils (NR). Their primary observable signatures ($S1$ and $S2$ pulse shapes, and their ratio $\log_{10}(S2/S1)$) are physically almost indistinguishable in liquid xenon without directionality or extreme statistical unbinned profile likelihood fits.

> [!CAUTION]
> **Mandatory Scientific Audit Conclusion:**  
> The manuscript **must not** claim robust machine learning rejection of the neutrino fog or CNNS. With only 16 test events and an empirical 75% misclassification rate into the signal class, any claim of CNNS suppression is empirically unsupported. This must be highlighted as a known physical and statistical limitation.

