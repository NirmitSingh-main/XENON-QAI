# RECOVERED HYPERPARAMETERS & ARCHITECTURAL SPECIFICATIONS

**Repository:** XENON-QAI  
**Audit Source:** Python source code (`src/xenon_qai/`), execution scripts, and saved model weights.  

---

## 1. Multimodal CNN Architecture & Training Setup

### Layer-by-Layer Encoder Specifications

1. **S1 Waveform Encoder (`S1WaveformCNN`):**
   - **Input:** $x_{S1} \in \mathbb{R}^{160}$ (160 time bins, zero-padded from variable length 45–155 bins).
   - `Conv1d(1, 32, kernel_size=7, stride=2, padding=3)` $\rightarrow$ `BatchNorm1d(32)` $\rightarrow$ `GELU`
   - `Conv1d(32, 64, kernel_size=5, stride=2, padding=2)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `GELU`
   - `Conv1d(64, 128, kernel_size=3, stride=2, padding=1)` $\rightarrow$ `BatchNorm1d(128)` $\rightarrow$ `GELU`
   - `AdaptiveAvgPool1d(1)` $\rightarrow$ `Linear(128, 64)` $\rightarrow$ `GELU`
   - **Output representation:** 64D vector.

2. **S2 Waveform Encoder (`S2WaveformCNN`):**
   - **Input:** $x_{S2} \in \mathbb{R}^{500}$ (500 time bins, linearly interpolated & area-normalized from 191–1484 bins).
   - `Conv1d(1, 32, kernel_size=11, stride=2, padding=5)` $\rightarrow$ `BatchNorm1d(32)` $\rightarrow$ `GELU`
   - `Conv1d(32, 64, kernel_size=7, stride=2, padding=3, dilation=2)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `GELU`
   - `Conv1d(64, 128, kernel_size=5, stride=2, padding=2, dilation=2)` $\rightarrow$ `BatchNorm1d(128)` $\rightarrow$ `GELU`
   - `Conv1d(128, 256, kernel_size=3, stride=2, padding=1)` $\rightarrow$ `BatchNorm1d(256)` $\rightarrow$ `GELU`
   - `AdaptiveAvgPool1d(1)` $\rightarrow$ `Linear(256, 128)` $\rightarrow$ `GELU`
   - **Output representation:** 128D vector.

3. **PMT Spatial Hit Encoder (`PMTSpatialMLP`):**
   - **Input:** $x_{PMT} \in \mathbb{R}^{248}$ (127 top PMTs + 121 bottom PMTs, log-transformed and standardized).
   - `Linear(248, 128)` $\rightarrow$ `LayerNorm(128)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.10)`
   - `Linear(128, 64)` $\rightarrow$ `LayerNorm(64)` $\rightarrow$ `GELU`
   - **Output representation:** 64D vector.

4. **13D Physics Feature Encoder (`PhysicsFeatureMLP`):**
   - **Input:** $x_{phys} \in \mathbb{R}^{13}$ (13 domain features standardized on training split).
   - `Linear(13, 64)` $\rightarrow$ `LayerNorm(64)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.10)`
   - `Linear(64, 32)` $\rightarrow$ `LayerNorm(32)` $\rightarrow$ `GELU`
   - **Output representation:** 32D vector.

5. **Fusion Bottleneck & Output Head:**
   - **Concatenated representation:** $64 + 128 + 64 + 32 = 288\text{D}$.
   - `Linear(288, 128)` $\rightarrow$ `LayerNorm(128)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.15)`
   - `Linear(128, 16)` $\rightarrow$ `LayerNorm(16)` $\rightarrow$ **Bottleneck Latent** $z \in \mathbb{R}^{16}$.
   - **Classification Head:** `Linear(16, 1)` (Binary Logits).
   - **Total Trainable Parameters:** **315,297**.

### Training Protocols
- **Optimizer:** `AdamW` ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$)
- **Learning Rate:** $1.0 \times 10^{-3}$ with `CosineAnnealingLR(T_max=15)`
- **Weight Decay:** $1.0 \times 10^{-4}$
- **Batch Size:** 64
- **Epochs:** 15 (convergence reached around epoch 5–8)
- **Loss Function:** `BCEWithLogitsLoss()`
- **Checkpoint Selection:** State with highest validation ROC-AUC.

---

## 2. Classical Tabular Baselines

| Model | Hyperparameters Recovered from Code |
|:---|:---|
| **Random Forest** | `n_estimators=300`, `min_samples_leaf=5`, `class_weight='balanced'`, `max_depth=None`, `random_state=42`, `n_jobs=-1` |
| **XGBoost** | `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, `n_jobs=-1`, `tree_method='hist'` |
| **Logistic Regression** | `C=1.0`, `penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`, `random_state=42` |
| **13D MLP** | Hidden layers: `(128, 64, 32)` with `LayerNorm` + `GELU` + `Dropout(0.10)`, `lr=1e-3`, `weight_decay=1e-4`, `batch_size=64`, `epochs=35`, Parameters: **12,609**. |

---

## 3. Quantum Variational Classifiers & Classical Head Audit

### Quantum Neural Network (QNN) Configuration
- **Simulator Backend:** PennyLane `default.qubit` (Classically simulated via CPU statevector algebra; no real quantum hardware used).
- **Number of Qubits:** $N = 4$.
- **Circuit Depths:** $L \in \{2, 4, 6\}$.
- **Ansatz:** Hardware-Efficient Ansatz with alternating $R_Y(\theta)$, $R_Z(\phi)$ single-qubit rotations and linear adjacent CNOT entangling gates.
- **Encoding Strategy:** Linear pre-projection from $16\text{D} \rightarrow 4\text{D}$, followed by $\tanh(x) \times \frac{\pi}{2}$ angle encoding into $R_Y$ gates.
- **Measurement Observables:** Expectation values of Pauli-Z on all 4 qubits: $\langle \sigma_z^{(i)} \rangle$ for $i \in \{0, 1, 2, 3\}$.
- **Classical Post-Readout:** `Linear(4, 1)` mapped to binary logits.
- **Differentiation:** Automatic adjoint / backpropagation differentiation (`diff_method='best'`).
- **Optimizer:** `AdamW(lr=0.03, weight_decay=1e-4)`, Batch size = 64, Epochs = 10.

### Parameter-Matching Rigor Assessment: Is the Classical Head Genuinely Matched?

The parameter-matched classical head was designed via:
$$\text{Linear}(16 \rightarrow 2) \rightarrow \text{GELU} \rightarrow \text{Linear}(2 \rightarrow 1)$$
- Weights: $16 \times 2 + 2 = 34$; $2 \times 1 + 1 = 3$. Total classical parameters = **37**.
- Pure quantum variational circuit parameters:
  - $L=2$: $2 \times 4 \times 2 = \mathbf{16}$ quantum parameters.
  - $L=4$: $4 \times 4 \times 2 = \mathbf{32}$ quantum parameters.
  - $L=6$: $6 \times 4 \times 2 = \mathbf{48}$ quantum parameters.
- **Audit Finding:**  
  While the variational *circuit* parameters (16, 32, 48) closely align with the classical head's 37 parameters, the hybrid PyTorch QNN wrapper contains classical linear projection layers:
  - Pre-projection ($16 \rightarrow 4$): $16 \times 4 + 4 = 68$ parameters.
  - Post-readout ($4 \rightarrow 1$): $4 \times 1 + 1 = 5$ parameters.
  - **Total Trainable Parameters in Hybrid QNN:** 89 ($L=2$), 105 ($L=4$), 121 ($L=6$).
  
> [!IMPORTANT]
> In scientific publications, one must clearly differentiate between **pure quantum gate parameters** (16, 32, 48) and the **total hybrid model trainable parameters** (89, 105, 121). The classical head (37 parameters) is strictly matched to the pure quantum circuit parameters, not the hybrid classical-quantum interface.

