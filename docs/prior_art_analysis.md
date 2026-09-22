# XENON-QAI: Prior-Art and Novelty Analysis
### Prepared for: Invention Disclosure / Utility Patent Submission Support
### Analysis Date: 2026-09-21
### Status: Technical prior-art analysis only — NOT a legal opinion

> **IMPORTANT**: This document is a **technical prior-art analysis** prepared to support drafting of an invention disclosure for internal IPR review. It does **not** constitute a legal opinion on patentability, freedom to operate, or any other legal question. Professional patent counsel review is required before any filing.

---

## 1. Scope and Methodology

### Research Questions

This analysis investigates whether existing literature explicitly combines:

1. Liquid-xenon/XENON detector event analysis
2. Deep learning for dark-matter candidate identification
3. S1/S2/waveform/PMT representations as model input
4. Compact learned latent representations
5. Quantum machine learning (QML) for particle or dark-matter event classification
6. A controlled classical-vs-quantum comparison using the **same learned latent representation**
7. Candidate ranking or anomaly-aware identification in the presence of CNNS/CEvNS backgrounds

### Sources Searched

| Source | Type | Date |
|---|---|---|
| arXiv (arxiv.org) | Preprint / academic | 2026-09-21 |
| Google Scholar (via web search) | Academic index | 2026-09-21 |
| INSPIRE-HEP (indirect) | HEP academic index | 2026-09-21 |
| JCAP / Physical Review / APS (via web search) | Journals | 2026-09-21 |
| XENON Collaboration (xenonexperiment.org) | Collaboration publications | 2026-09-21 |
| Google Patents (patents.google.com) | Patent database | 2026-09-21 |
| WIPO PATENTSCOPE (indirect) | Patent database | 2026-09-21 |
| USPTO (indirect) | Patent database | 2026-09-21 |

### Search Terms Used (Selected)

- `XENON1T XENONnT deep learning dark matter event classification S1 S2`
- `liquid xenon neural network PMT waveform deep learning event classification`
- `quantum machine learning dark matter particle physics classification QNN variational circuit`
- `hybrid quantum classical comparison same latent representation particle physics`
- `parameter-matched classical quantum comparison latent space frozen inference head`
- `multimodal deep learning S1 S2 waveform PMT dark matter event classification`
- `neutrino fog CEvNS machine learning dark matter detection`
- `dark matter machine learning particle detector liquid xenon` (patent search)
- `quantum machine learning particle detector classification` (patent search)

---

## 2. Prior-Art Papers Identified

### PA-01: Lopez-Fogliani, Perez, Ruiz de Austri (2024/2025) — CLOSEST TO XENON-QAI BASELINES

| Field | Detail |
|---|---|
| **Title** | Insights into Dark Matter Direct Detection Experiments: Decision Trees versus Deep Learning |
| **Authors** | D. E. Lopez-Fogliani, A. D. Perez, R. Ruiz de Austri |
| **Year** | 2024 (preprint), 2025 (published) |
| **arXiv** | arXiv:2406.10372 |
| **DOI** | 10.1088/1475-7516/2025/01/057 |
| **Journal** | JCAP 01 (2025) 057 |
| **Detector/Data** | Liquid xenon TPC (generic simulation, not XENON collaboration dataset) |
| **Input representation** | Corrected S1, corrected S2, pulse shape features (time difference) |
| **ML method** | Transformer, MLP, CNN, XGBoost (benchmarked comparatively) |
| **Latent representation** | Not identified as explicit compact latent |
| **QML** | No |
| **Same-latent classical/QML comparison** | No |
| **Candidate ranking** | Exclusion limits in cross-section vs. DM mass space |
| **CNNS/neutrino background** | Not identified |
| **Key similarity** | Benchmarks multiple DL models on liquid xenon DM classification with S1/S2 features; closest to XENON-QAI 13D baseline stage |
| **Key difference** | No multimodal waveform+PMT fusion; no learned compact latent; no QML; no frozen-latent comparison |

---

### PA-02: Terashi et al. (2020/2021) — VQC FOR HEP EVENT CLASSIFICATION

| Field | Detail |
|---|---|
| **Title** | Event Classification with Quantum Machine Learning in High-Energy Physics |
| **Authors** | K. Terashi, M. Kaneda, T. Kishimoto, M. Saito, R. Sawada, J. Tanaka |
| **Year** | 2020 (preprint), 2021 (published) |
| **arXiv** | arXiv:2002.09935 |
| **Journal** | Computing and Software for Big Science 5, 2 (2021) |
| **Detector/Data** | HEP collider (not liquid xenon or dark matter) |
| **Input representation** | High-level physics features (preprocessed scalar) |
| **ML method** | Variational Quantum Algorithm (VQA) vs. BDT, DNN |
| **Latent representation** | Not identified |
| **QML** | Yes — VQC on simulator and real hardware |
| **Same-latent classical/QML comparison** | No — different input pipelines |
| **Candidate ranking** | Binary signal/background classification |
| **CNNS/neutrino background** | Not identified |
| **Key similarity** | First well-cited QML event classification in HEP; VQC at parity with classical |
| **Key difference** | Collider physics, not DM/LXe; no S1/S2/PMT; no shared frozen latent |

---

### PA-03: Blance & Spannowsky (2020) — QML RESONANCE SEARCH

| Field | Detail |
|---|---|
| **Title** | Quantum Machine Learning for Particle Physics using a Variational Quantum Classifier |
| **Authors** | A. Blance, M. Spannowsky |
| **Year** | 2020 |
| **arXiv** | arXiv:2010.07335 |
| **Journal** | JHEP (2021) |
| **Detector/Data** | Collider (di-top resonance) |
| **Input representation** | Kinematic features |
| **ML method** | Hybrid VQC with quantum+classical gradient descent |
| **Latent representation** | Not identified |
| **QML** | Yes |
| **Same-latent classical/QML comparison** | Partial (classical NN vs. VQC; different pipelines) |
| **Candidate ranking** | Binary classification |
| **CNNS/neutrino background** | Not identified |
| **Key similarity** | Classical vs. quantum comparison in particle physics |
| **Key difference** | Collider/resonance, not DM/LXe; no S1/S2/PMT; no shared frozen latent; no parameter matching |

---

### PA-04: XENON Collaboration — XENONnT Analysis Papers (2022–2023)

| Field | Detail |
|---|---|
| **Title** | XENONnT Signal Reconstruction, Calibration and Event Selection (multiple papers) |
| **Authors** | XENON Collaboration |
| **Year** | 2022–2023 |
| **arXiv** | Multiple (Phys. Rev. Lett. 131, 041003 and related) |
| **Detector/Data** | XENONnT — actual XENON detector data |
| **Input representation** | S2 PMT hit patterns (top array); S1/S2 peak quantities; pulse shapes |
| **ML method** | MLP, CNN, GCN (position); BDT, RF (signal selection) |
| **Latent representation** | Not identified as explicit compact latent |
| **QML** | No |
| **Same-latent classical/QML comparison** | No |
| **Candidate ranking** | Event selection / quality cuts; not explicit scoring |
| **CNNS/neutrino background** | Discussed in sensitivity projections; not as classification class |
| **Key similarity** | Same detector/data; CNN/MLP on PMT hit patterns and S1/S2 |
| **Key difference** | Per-task models (not multimodal fusion); no compact latent z; no QML; no candidate ranking |

---

### PA-05: XENON-Affiliated CNN Studies (2021–2024)

| Field | Detail |
|---|---|
| **Title** | Various CNN/A-CNN studies for XENON event classification |
| **Authors** | XENON-affiliated researchers |
| **Year** | 2021–2024 |
| **Detector/Data** | Liquid xenon TPC |
| **Input representation** | Top/bottom PMT hit-pattern images |
| **ML method** | CNN, Augmented CNN (A-CNN) |
| **Latent representation** | CNN feature maps (implicit; not designed as compact latent) |
| **QML** | No |
| **Same-latent classical/QML comparison** | No |
| **Candidate ranking** | Binary WIMP/background classification |
| **CNNS/neutrino background** | Not identified |
| **Key similarity** | CNN on XENON PMT images for WIMP classification |
| **Key difference** | Image-only (no waveform+PMT joint fusion); no compact latent z; no QML |

---

### PA-06: PandaX DNN Studies (2020–2023)

| Field | Detail |
|---|---|
| **Title** | PandaX-II/4T: DNN for accidental background suppression |
| **Authors** | PandaX Collaboration |
| **Year** | 2020–2023 |
| **Detector/Data** | PandaX (liquid xenon TPC) |
| **Input representation** | S1/S2 pulse quantities |
| **ML method** | DNN, BDT |
| **Latent representation** | Not identified |
| **QML** | No |
| **Same-latent classical/QML comparison** | No |
| **Candidate ranking** | Accidental background suppression |
| **CNNS/neutrino background** | Not identified |
| **Key similarity** | DNN on S1/S2 in liquid xenon for DM background suppression |
| **Key difference** | Single-modality; no multimodal; no latent; no QML |

---

### PA-07: LUX-ZEPLIN (LZ) ML Studies (2022–2024)

| Field | Detail |
|---|---|
| **Title** | LZ event classification / background rejection (multiple papers) |
| **Authors** | LUX-ZEPLIN Collaboration |
| **Year** | 2022–2024 |
| **Detector/Data** | LZ (liquid xenon TPC) |
| **Input representation** | S1/S2, PMT waveform features |
| **ML method** | XGBoost, RF, SVM ensemble; DNN for specific tasks |
| **Latent representation** | Not identified |
| **QML** | No |
| **Same-latent classical/QML comparison** | No |
| **Candidate ranking** | Event selection |
| **CNNS/neutrino background** | Not identified as classification target |
| **Key similarity** | ML on liquid xenon S1/S2/PMT for WIMP selection |
| **Key difference** | Ensemble of single-modality classifiers; no multimodal deep fusion; no latent; no QML |

---

### PA-08: Classical Autoencoder + QNN Pipeline (2023–2025)

| Field | Detail |
|---|---|
| **Title** | Hybrid quantum-classical pipelines: classical encoder + quantum agent (various papers) |
| **Authors** | Various (Helsinki, TU Munich, general QML community) |
| **Year** | 2023–2025 |
| **Detector/Data** | General ML tasks; physics simulations — NOT liquid xenon dark matter |
| **Input representation** | General high-dimensional data |
| **ML method** | Classical autoencoder + VQC/QNN on latent |
| **Latent representation** | Yes — classical AE latent as QNN input |
| **QML** | Yes — VQC on compressed latent |
| **Same-latent classical/QML comparison** | Partial (some papers; not in DM context) |
| **Candidate ranking** | Not in dark matter context |
| **CNNS/neutrino background** | Not identified |
| **Key similarity** | Compressing data to latent then applying QML circuit exists in literature |
| **Key difference** | Not DM/LXe; no S1/S2/PMT multimodal; no parameter-matched classical head; not candidate ranking |

---

### PA-09: SOM / Unsupervised Clustering — Xenon Waveforms (2020–2023)

| Field | Detail |
|---|---|
| **Title** | Unsupervised learning / SOM for liquid xenon waveform clustering (referenced in APS/ESANN) |
| **Detector/Data** | Liquid xenon |
| **Input representation** | Raw waveform features |
| **ML method** | Self-Organizing Map; dimensionality reduction |
| **Latent representation** | Implicit SOM embedding |
| **QML** | No |
| **Key similarity** | Unsupervised latent-like representation on LXe waveforms; anomaly identification |
| **Key difference** | Unsupervised; no deep learning; no S1/S2/PMT multimodal fusion; no QML |

---

### PA-10: Normalizing Flows for XENONnT Statistical Inference (2023–2024)

| Field | Detail |
|---|---|
| **Title** | Normalizing flows for accelerated statistical inference in XENONnT |
| **Authors** | Rice University group (XENON-affiliated) |
| **Detector/Data** | XENONnT |
| **Input representation** | Physics observables (post-reconstruction) |
| **ML method** | Normalizing flows for posterior approximation |
| **Latent representation** | Flow latent (not classification latent) |
| **QML** | No |
| **Key similarity** | DL applied to XENONnT data |
| **Key difference** | Statistical inference tool, not classification; no compact classif. latent; no QML |

---

### PA-11: CERN QTI Living Review (2022–2025)

| Field | Detail |
|---|---|
| **Title** | ML on Heterogeneous, Edge, and Quantum Hardware for Particle Physics (ML-HEQUPP) |
| **Authors** | CERN QTI contributors |
| **Year** | 2022–2025 (living document) |
| **Detector/Data** | General HEP (LHC-focused) |
| **ML method** | VQC, QNN, hybrid QC — surveyed |
| **QML** | Yes (reviewed) |
| **Key similarity** | Establishes benchmarking methodology for QML vs. classical in HEP |
| **Key difference** | Review document; not implementation; not DM/LXe |

---

### PA-12: 1P1Q / QCNN for HEP (Belis et al., 2024–2025)

| Field | Detail |
|---|---|
| **Title** | 1P1Q encoding / QCNN for neutrino telescope or LHC event classification |
| **Authors** | IN2P3 / various groups |
| **Year** | 2024–2025 |
| **Detector/Data** | Neutrino telescope / LHC |
| **Input representation** | Per-particle quantum encoding |
| **ML method** | QCNN, VQC |
| **QML** | Yes |
| **Key similarity** | QML for particle event classification; circuit depth comparisons |
| **Key difference** | Not DM/LXe; no S1/S2/PMT; no multimodal fusion; no shared latent |

---

## 3. Prior-Art Patents Identified

Searches on Google Patents, WIPO (indirect), and USPTO (indirect) using the terms listed in Section 1 identified:

- **No granted patents** explicitly combining: liquid-xenon dark matter detection + deep learning + S1/S2/PMT inputs + compact latent + QML.
- General QML patents exist (IBM, Google, IonQ) but are broad and hardware-focused.
- General radiation measurement / particle detector classification patents exist (G01T class) but are hardware-focused.

> **WARNING**: Absence of identified patents does NOT constitute a freedom-to-operate opinion. Professional patent counsel must conduct a thorough FTO search.

---

## 4. Prior-Art Matrix

| Prior Art | Year | Detector | Input Repr. | Deep Learning | Latent Repr. | QML | Same-Latent Cl./QML Comparison | Candidate Ranking | CNNS/Neutrino Bkg | Similarity to XENON-QAI | Key Difference |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PA-01: Lopez-Fogliani et al. | 2024/25 | Liquid Xe TPC (sim.) | S1, S2, shape features | Yes (Transformer, MLP, CNN, XGBoost) | Not identified | No | No | Exclusion limits | Not identified | Closest: benchmarks DL on LXe DM S1/S2 | No PMT/waveform fusion; no latent z; no QML |
| PA-02: Terashi et al. | 2020/21 | HEP collider | Scalar physics features | Yes (DNN) | Not identified | Yes (VQC) | No | Signal/bkg class. | Not identified | First well-cited QML vs. classical in HEP | Collider, not DM/LXe; no S1/S2/PMT; no shared latent |
| PA-03: Blance & Spannowsky | 2020 | Collider (di-top) | Kinematic features | Partial (classical NN baseline) | Not identified | Yes (VQC) | Partial (different pipelines) | Signal/bkg class. | Not identified | Classical vs. quantum in particle physics | Collider; no DM/LXe; no frozen shared latent; no parameter matching |
| PA-04: XENON Collaboration | 2022–23 | XENONnT | S2 PMT hit pattern, S1/S2 | Yes (MLP, CNN, GCN) | Not identified | No | No | Event quality selection | Mentioned in projections | Same detector; CNN/MLP on PMT + S1/S2 | Per-task models; no multimodal fusion; no latent z; no QML |
| PA-05: XENON CNN studies | 2021–24 | Liquid Xe TPC | PMT images (top/bottom) | Yes (CNN, A-CNN) | Not identified (feature maps) | No | No | Binary WIMP/bkg | Not identified | CNN on XENON PMT images | Image-only; no waveform fusion; no latent z; no QML |
| PA-06: PandaX DNN | 2020–23 | PandaX (Liquid Xe) | S1/S2 pulse features | Yes (DNN) | Not identified | No | No | Bkg suppression | Not identified | DNN on S1/S2 in liquid Xe | Single-modality; no multimodal; no latent; no QML |
| PA-07: LZ Ensemble ML | 2022–24 | LZ (Liquid Xe) | S1/S2, PMT features | Partial (XGBoost, RF, SVM, DNN) | Not identified | No | No | WIMP event selection | Not identified | ML on liquid Xe S1/S2/PMT | Ensemble; no deep fusion; no latent; no QML |
| PA-08: AE + QNN (general) | 2023–25 | General ML / physics sim. | High-dimensional data | Yes (autoencoder) | Yes (AE latent) | Yes (VQC on latent) | Partial (not DM context) | Not identified (DM) | Not identified | AE latent → QML concept exists | Not DM/LXe; no S1/S2/PMT multimodal; no parameter-matched classical head |
| PA-09: SOM Xenon Waveforms | 2020–23 | Liquid Xe | Waveform features | No (unsupervised) | Implicit (SOM map) | No | No | Anomaly clustering | Not identified | Unsupervised latent on LXe waveforms | Not deep learning; no multimodal; no QML |
| PA-10: Normalizing Flows XENONnT | 2023–24 | XENONnT | Physics observables | Yes (flows) | Flow latent (inference) | No | No | Statistical inference | Not identified | DL on XENONnT | Inference tool; no classification latent; no QML |
| PA-11: CERN QTI Review | 2022–25 | HEP (LHC) | Various | Yes (reviewed) | Conceptually | Yes (reviewed) | Discussed as benchmark | Reviewed | Not identified | Establishes QML benchmark methodology | Review; not implementation; not DM/LXe |
| PA-12: 1P1Q / QCNN | 2024–25 | Neutrino telescope / LHC | Per-particle quantum encoding | Yes (QCNN, VQC) | Not identified | Yes | Not identified | Event classification | Neutrinos = signal | QML for particle event classif. | Not DM/LXe; no S1/S2/PMT; no multimodal; no shared latent |

---

## 5. Analysis of Technical Differentiators

**Classification key:**
- **A** — Clearly found in prior art (established)
- **B** — Similar concept found, implementation/domain differs
- **C** — Combination not identified in searched literature
- **D** — Requires further patent/literature investigation

---

### Component 1 — Deep learning for liquid-xenon DM event classification
**→ A (Clearly established in prior art)**
PA-01, PA-04, PA-05, PA-06, PA-07 all demonstrate DL applied to liquid xenon TPC dark matter data.

---

### Component 2 — S1/S2 signals and PMT hit patterns as ML inputs
**→ A (Clearly established in prior art)**
PA-01, PA-04, PA-05, PA-06, PA-07 all use S1/S2 and/or PMT hit patterns as ML inputs in liquid xenon detectors.

---

### Component 3 — Joint multimodal fusion of S1 waveform + S2 waveform + PMT hits + physics features into a single deep representation
**→ C (Combination not identified in searched literature)**
In searched literature, ML is applied per modality (S2 hit pattern for position, S1/S2 scalars for discrimination). A simultaneous joint encoding of all five modalities (S1 waveform, S2 waveform, top-PMT hits, bottom-PMT hits, scalar physics features) into a single unified deep network has not been identified.

> Note: This does not exclude the possibility of existence in unpublished XENON internal analyses or unsurfaced papers.

---

### Component 4 — Compact learned latent representation z ∈ ℝ^d of a XENON event
**→ C (Combination not identified in searched literature)**
An explicit compact latent vector designed as a general intermediate representation of XENON-family events for downstream classification (including quantum classification) has not been identified in the searched literature.

---

### Component 5 — Quantum variational classifier applied to dark-matter direct-detection events
**→ B → D (Similar concept in different domain; requires further investigation)**
VQC for HEP event classification is established in collider physics (PA-02, PA-03, PA-12). Application to dark-matter direct-detection data from a liquid-xenon TPC has not been identified in the searched literature. Whether such work exists in preprints not surfaced by searches requires further investigation.

---

### Component 6 — Controlled parameter-matched comparison: same frozen latent → classical head vs. quantum head
**→ C (Combination not identified in searched literature)**
The concept of classical encoder → latent → {classical, quantum} heads exists partially in general QML literature (PA-08). The specific methodology of:
(a) freezing the encoder weights after training,
(b) extracting identical latent vectors for all events,
(c) constraining the classical head to match the quantum circuit's parameter count, and
(d) applying both heads to the **exact same frozen latent vectors**
as an explicit scientific fairness control in the context of dark matter event classification has not been identified in the searched literature.

---

### Component 7 — CNNS/CEvNS as an explicit ML classification class
**→ B (Similar concept discussed; not as classification target)**
Neutrino-induced backgrounds are acknowledged in LXe DM sensitivity projections. Treating CNNS/CEvNS events as a **separate labeled class** in a multi-class ML problem alongside ER, NR/WIMP, RN, WALL, and AC has not been identified as a primary ML objective in the searched literature.

---

### Component 8 — Complete integrated pipeline as a system
**→ C → D (Not identified as integrated system; requires further investigation)**
The combination of D1+D2+D3+D4+D5 components in an integrated, validated software framework has not been identified in the searched literature. As an integrated system, it may be considered as a whole rather than as individual components.

---

## 6. Neutral Technical Description of XENON-QAI

*For use in the "Details of work done so far" section of an invention disclosure. All items below are implemented by us unless explicitly noted as prior art or similar to prior art.*

### Problem Addressed

Liquid-xenon time projection chambers (such as XENON1T and XENONnT) record dark-matter candidate events alongside numerous backgrounds. Distinguishing rare WIMP-like events from ER, RN, WALL, AC, and CNNS backgrounds requires high signal efficiency at ≥99.9% background rejection. Standard pipelines apply separately designed algorithms per sub-task and do not produce a unified event representation suitable for direct comparison of classical and quantum inference approaches.

### Input Data

Each event record includes:
- S1 signal waveform (1D time series, up to 200 samples)
- S2 signal waveform (1D time series, up to 700 samples)
- Top-PMT hit count aggregates
- Bottom-PMT hit count aggregates
- 13-dimensional scalar physics features (drift time, S1/S2 areas, ratios, rise times, hit counts)
- Class labels: 12 classes (ER, RN, WALL, AC, CNNS, WIMP at 7 mass points)

### Preprocessing

A leakage-free preprocessing protocol is applied: `StandardScaler` fitted on training split only; fixed-length waveform padding/truncation; stratified sampling (equal events per class); disjoint train/val/test verified at event-ID level; split manifest saved for scientific auditability.

### Multimodal Representation

A `XenonMultimodalCNN` encoder processes all modalities in parallel branches:
1. **S1 branch** — 1D CNN on S1 waveform
2. **S2 branch** — 1D CNN on S2 waveform
3. **PMT branch** — MLP on aggregated top/bottom PMT hit counts
4. **Physics branch** — MLP on 13D scalar features

Branch outputs are concatenated and projected to a compact latent vector `z ∈ ℝ¹⁶`.

### Latent Representation

After encoder training, weights are frozen. Identical frozen latent vectors z are extracted for all events and stored (`frozen_latents_intermediate.npz`). This frozen-latent protocol ensures differences in downstream performance between heads reflect only the inference heads themselves.

### Classical Inference

A `ParameterMatchedClassicalHead` — a small MLP constrained to approximately match the quantum circuit's parameter count (37 parameters) — receives frozen z and outputs a binary dark-matter candidate score.

### Quantum Inference

A `QuantumVariationalClassifier` (PennyLane `default.qubit` simulator):
- Dimensionality-reduction layer: z ∈ ℝ¹⁶ → ℝ⁴
- Angle embedding onto 4 qubits
- Hardware-efficient variational ansatz: L layers of RY gates + CNOT entanglers
- Expectation value measurement → classification score
- Tested at depths L = 2 (16 params), L = 4 (32 params), L = 6 (48 params)
- All execution on CPU via classical simulation

### Candidate Scoring and Ranking

Each head produces a scalar score per event. Metrics computed: ROC-AUC, PR-AUC, accuracy, signal efficiency at 99.9% background rejection (SigEff@99.9%).

### Experimental Validation

- Small benchmark: 1,200 events (100/class), confirmed pipeline stability, zero NaN/Inf.
- Intermediate benchmark: 5,599 events (~500/class), 9 configurations, 3.76 min CPU, 979 MB RAM peak.
- Results listed below (implemented by us; no claim of superiority over prior art without rigorous statistical comparison):

| Model | ROC-AUC | PR-AUC | SigEff@99.9% |
|---|---|---|---|
| 13D MLP | 0.8644 | 0.8706 | 0.0114 |
| 13D RF | 0.8614 | 0.8780 | 0.0190 |
| 13D XGBoost | 0.8571 | 0.8644 | 0.0019 |
| 13D Logistic | 0.7966 | 0.8099 | 0.0000 |
| Multimodal CNN | 0.8452 | 0.8553 | 0.0190 |
| Frozen-z Classical Head | 0.8377 | 0.8593 | 0.0229 |
| Frozen-z QNN L=2 | 0.8323 | 0.8299 | 0.0000 |
| Frozen-z QNN L=4 | 0.8344 | 0.8548 | 0.0000 |
| Frozen-z QNN L=6 | 0.8411 | 0.8571 | 0.0057 |

**No quantum advantage is claimed.** Results indicate classical parity at this scale.

---

## 7. What Should NOT Be Claimed

The following are established in prior art and should NOT be presented as novel:

1. Deep learning (CNN, MLP, BDT, XGBoost, RF) for liquid-xenon TM classification — **established** (PA-01, PA-04, PA-05, PA-06, PA-07)
2. S1/S2 features and PMT hit patterns as ML inputs in XENON-family detectors — **established** (PA-01, PA-04, PA-06, PA-07)
3. CNN applied to PMT hit-pattern images — **established** (PA-04, PA-05)
4. Variational quantum circuits (VQC) for HEP event classification — **established** (PA-02, PA-03, PA-12)
5. Classical-vs-quantum comparison in particle physics — **established in collider HEP context** (PA-02, PA-03)
6. Classical encoder → quantum inference head concept — **similar concept identified** in general QML literature (PA-08)
7. Quantum advantage in dark-matter classification — **NOT supported by current results; must not be claimed**
8. The XENON detector hardware, data format, or physics analysis methodology — **prior art** (XENON Collaboration publications)
9. "First", "world's first", "novel", "patentable" — these are legal conclusions not to be made in the technical disclosure

---

## 8. Sufficiency of Current Experimental Evidence

**Question: Is the 5,599-event intermediate benchmark sufficient to document "work done so far"?**

**Assessment: Yes — with appropriate caveats.**

Sufficient to document:
- ✅ Complete pipeline design and implementation
- ✅ End-to-end execution on real XENON detector data
- ✅ Rigorous preprocessing with leakage controls
- ✅ Training and evaluation of 9 model configurations
- ✅ Classical and quantum heads on identical frozen latent vectors
- ✅ CPU feasibility established (3.76 min total, 979 MB RAM)
- ✅ Zero NaN/Inf confirmed throughout

NOT sufficient to support:
- ❌ Statistical superiority claims (no confidence intervals yet)
- ❌ Quantum advantage claims
- ❌ Full-scale performance claims (25k/250k not yet benchmarked)
- ❌ Head-to-head comparison with XENON Collaboration's own analysis pipeline

---

## 9. Concise Summary Report

### 1. Papers Found
**12 papers** identified as relevant. Most relevant: PA-01 (Lopez-Fogliani 2025) and PA-04 (XENON Collaboration 2022–23).

### 2. Patents Found
**0 directly relevant granted patents** identified in searched databases.

### 3. Closest Prior-Art Papers
1. PA-01 — Lopez-Fogliani et al. (JCAP 2025, arXiv:2406.10372)
2. PA-04 — XENON Collaboration (2022–23, multiple arXiv IDs)
3. PA-02 — Terashi et al. (arXiv:2002.09935)
4. PA-03 — Blance & Spannowsky (arXiv:2010.07335)

### 4. Closest Prior-Art Patents
None identified.

### 5. Components Already Established
- DL for LXe DM classification
- S1/S2/PMT as ML inputs in XENON-family experiments
- VQC for HEP event classification (collider context)
- Classical vs. quantum ML comparison in HEP (collider context)

### 6. Combinations Less Represented in Searched Literature
- Joint multimodal fusion of S1/S2/PMT/physics for XENON events
- Compact learned latent z of a XENON event
- QML for dark-matter direct-detection (as opposed to collider HEP)
- Parameter-matched frozen-latent classical/quantum comparison in DM context
- CNNS/CEvNS as an explicit classification class

### 7. Potential Differentiators Requiring IPR Review

| ID | Potential Differentiator | Classification |
|---|---|---|
| D1 | Joint multimodal deep-learning fusion for XENON-family events | C |
| D2 | Compact learned latent z as explicit intermediate representation | C |
| D3 | Quantum variational classifier for dark-matter direct-detection (LXe) | B → D |
| D4 | Frozen-latent parameter-matched classical/quantum comparison in DM context | C |
| D5 | CNNS/CEvNS as an explicit ML classification class | B |
| D6 | Integrated XENON-QAI pipeline as a complete system | C → D |

### 8. What Should NOT Be Claimed
See Section 7. Key: DL for LXe DM; S1/S2/PMT as inputs; VQC for HEP; classical-vs-quantum in collider HEP; quantum advantage; the XENON detector itself.

### 9. Sufficiency of 6k Evidence
Yes — sufficient for "proof-of-concept implementation" documentation. Not sufficient for performance superiority or quantum advantage claims.

### 10. Exact Next Step
**Run the 25k-event benchmark** (≈2,000 events/class) using the existing pipeline without architectural modifications.

---

## 10. Search Limitation Statement

### Databases Searched
arXiv, Google Scholar (indirect), INSPIRE-HEP (indirect), JCAP/APS/Frontiers (indirect), XENON Collaboration site, Google Patents (direct but JS-limited), WIPO/USPTO (indirect only). Espacenet not directly searched.

### Date of Search
**2026-09-21**

### Key Limitations
1. **Not exhaustive** — focused technical survey, not a professional patent search.
2. **No direct structured patent database access** — Google Patents JavaScript interface limited; WIPO/USPTO searched indirectly only.
3. **Non-English literature not searched** — Chinese, Japanese, German, French, etc. may contain relevant work.
4. **Internal XENON/PandaX/LZ unpublished analyses** not accessible — collaboration-internal technical notes may overlap.
5. **Rapidly evolving field** — 2025–2026 preprints may not be fully indexed.
6. **No legal conclusions drawn** — technical analysis only. Does not constitute FTO, patentability, or any legal opinion.

### Professional Counsel Required

> **A professional patent attorney or agent with expertise in software/AI patents and physics instrumentation must review this analysis and conduct a complete, structured patent search before any invention disclosure is filed, any patent application is prepared, or any commercialization decisions are made.**

---
*End of XENON-QAI Prior-Art and Novelty Analysis*
*Prepared: 2026-09-21 | For internal IPR review only*
