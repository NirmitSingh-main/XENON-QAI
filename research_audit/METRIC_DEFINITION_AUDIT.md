# EVALUATION METHODOLOGY & METRIC AUDIT

**Repository:** XENON-QAI  
**Audited File:** `src/xenon_qai/evaluation/metrics.py`  

---

## 1. Mathematical Definitions and Implementation Audit

### Signal vs. Background Mapping
- **WIMP Signal (Positive Class = 1):**  
  All 7 simulated WIMP masses (`mDM10GeV`, `mDM20GeV`, `mDM35GeV`, `mDM50GeV`, `mDM100GeV`, `mDM200GeV`, `mDM500GeV`).
- **Physical Background (Negative Class = 0):**  
  All 5 physical backgrounds:
  1. `AC` (Accidental Coincidence)
  2. `CNNS` (Coherent Neutrino-Nucleus Scattering)
  3. `ER` (Electronic Recoil)
  4. `RN` (Radon / Radiogenic)
  5. `WALL` (Surface charge accumulation)
- **CNNS Classification Status:**  
  **Yes, CNNS is explicitly included in the background class (Class 0).**

---

## 2. Metric Calculations & Threshold Selection

1. **ROC-AUC & PR-AUC:**
   - Evaluated using `sklearn.metrics.roc_auc_score(y_true, y_score)` and `sklearn.metrics.average_precision_score(y_true, y_score)`.
   - Threshold-independent area under the curve metrics.

2. **Accuracy & $F_1$-score:**
   - Evaluated using a fixed threshold $\theta = 0.5$ on sigmoid probabilities: $\hat{y} = \mathbb{I}(p \ge 0.5)$.

3. **Signal Efficiency at Extreme Background Rejection:**
   - **Formula:**
     $$\text{SigEff} = \text{TPR} \quad \text{at} \quad \text{FPR} \le 1.0 - \text{Rejection}$$
     - At $99.9\%$ background rejection: $\text{FPR} \le 1.0 \times 10^{-3} = 0.001$.
     - At $99.99\%$ background rejection: $\text{FPR} \le 1.0 \times 10^{-4} = 0.0001$.
   - **Implementation:**
     ```python
     fpr, tpr, _ = roc_curve(y_true, y_score)
     target_fpr = 1.0 - target_bkg_rejection
     valid_idx = np.where(fpr <= target_fpr)[0]
     return float(tpr[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0
     ```

---

## 3. Discrepancy & Statistical Limitations at Extreme Rejection

### Why Do Earlier Benchmarks Contain 0.0000 for Signal Efficiency?
In rare-event physics, measuring true positive rates at $99.9\%$ or $99.99\%$ background rejection requires substantial background test statistics:
- At the **5.6K benchmark**, there were only **350 background test events**.
  $$\text{Allowed False Positives at } 99.9\% = 350 \times 0.001 = 0.35 \text{ events}.$$
  Since fewer than 1 false positive was permitted, no background event could leak, requiring the decision threshold to be set above all background scores. If the highest background score exceeded or matched top signal scores, the signal efficiency collapsed to **0.0000**.
- At the **22.1K benchmark**, there were **1,216 background test events**:
  - Allowed FP at $99.9\% = 1.216$ events (only 1 event permitted).
  - Allowed FP at $99.99\% = 0.1216$ events (requires $\ge 10,000$ background events). Hence, $99.99\%$ signal efficiency was identically **0.0000** for several baselines.
- At the **45K benchmark**, with **2,288 background test events**:
  - Allowed FP at $99.9\% = 2.288$ events (up to 2 events permitted).
  - This allowed non-zero signal efficiencies to be reliably computed for all models (Multimodal CNN: 0.0143, Classical Head: 0.0295, QNN $L=2$: 0.0130).

> [!WARNING]
> While calculating signal efficiency at $99.9\%$ and $99.99\%$ background rejection is scientifically conventional in dark matter literature, reporting non-zero values on test sets with fewer than $10^4$ background events carries high binomial variance. The paper must explicitly state the exact number of retained test background events.

