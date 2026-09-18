# XENON-QAI 🔬⚛️

## Deep Learning and Quantum Machine Learning for Dark Matter Candidate Identification in Liquid-Xenon Detectors

> A research framework for analyzing liquid-xenon detector events using deep learning, latent representations, and quantum machine learning.

---

## 🚧 Project Status

**Status:** Research / Early Development

XENON-QAI is currently in the **physics and dataset understanding phase**.

The repository structure has been established, while the actual data pipeline, model architecture, experiments, and quantum components will be developed progressively after inspecting the available XENON event representations.

---

# 🔬 Project Overview

XENON-QAI investigates whether learned representations from deep-learning models can improve the identification and ranking of dark-matter-like candidate events in liquid-xenon detector data.

The project focuses particularly on situations where conventional backgrounds and neutrino-induced events become difficult to distinguish from expected dark-matter interactions.

A quantum machine-learning component will be introduced downstream of the deep-learning representation and compared against an equivalent classical inference model.

### Core Pipeline

```text
XENON Detector/Event Data
          ↓
Data Preprocessing
          ↓
Physics-Aware Representation
          ↓
Deep-Learning Model
          ↓
Learned Latent Representation
          ↓
Quantum-Compatible Compression
          ↓
 ┌─────────────────┬─────────────────┐
 │                 │                 │
Classical Head   Quantum Head
 │                 │
 └─────────────────┬─────────────────┘
                   ↓
       Candidate / Anomaly Score
                   ↓
              Event Ranking
                   ↓
       Scientific Validation
```

---

# 🎯 Research Question

> **Can a deep-learning representation of liquid-xenon detector events improve the identification and ranking of dark-matter-like candidates in the presence of conventional backgrounds and neutrino-induced events, and under what data and computational conditions does a quantum inference layer become competitive with an equivalent classical classifier?**

The project does **not** assume that quantum machine learning will outperform classical machine learning.

Possible outcomes include:

- Quantum model performs better under specific conditions
- Classical model performs better
- Both perform similarly
- Quantum performance changes significantly with data availability, qubit count, circuit depth, or noise

A scientifically useful result can therefore be obtained regardless of whether a quantum advantage is observed.

---

# 🧪 What XENON-QAI Is

XENON-QAI is:

- A software-based scientific analysis framework
- A deep-learning representation pipeline
- A candidate-identification and ranking system
- A hybrid classical–quantum machine-learning experiment
- An investigation of neutrino-induced backgrounds
- An anomaly-analysis framework
- A platform for studying simulation and real-data generalization

# 🚫 What XENON-QAI Is Not

XENON-QAI is **not**:

- A physical liquid-xenon detector
- A replacement for the XENON experiment
- A claim of discovering dark matter
- A system that can prove an individual event is dark matter
- A claim that quantum computing is automatically superior to classical computing

The output of the system should be interpreted as a **candidate score, compatibility score, anomaly score, or ranking**, indicating which events may deserve further investigation.

---

# ⚛️ XENON Detector Physics

The project is based on liquid-xenon time-projection-chamber (LXe-TPC) event data.

Important physical quantities include:

### S1

Prompt scintillation light produced by an interaction in liquid xenon.

### S2

Ionization electrons drift through the detector and are extracted into gaseous xenon, producing secondary scintillation light.

### Event Information

Depending on the representation, an event may contain information related to:

- S1
- S2
- S2/S1
- Timing
- Pulse characteristics
- Energy
- Spatial information
- Waveform information
- Detector-response features

These quantities can provide information about the physical interaction responsible for an event.

---

# 🌌 Dark Matter Candidate Identification

The detector observes physical interactions rather than events explicitly labeled as "dark matter."

The analysis therefore involves distinguishing possible dark-matter-like interactions from different backgrounds.

Relevant event categories include:

```text
Candidate Dark Matter
        │
        ├── WIMP interactions
        │
        ▼
   Candidate Signal


Backgrounds
        │
        ├── Electronic Recoils (ER)
        ├── Surface Events
        ├── Radiogenic Neutrons
        ├── Accidental Coincidences
        └── Neutrino-Induced Events
```

---

# 🌊 Neutrino Fog

A major focus of XENON-QAI is the **neutrino fog**.

Solar neutrinos can interact with xenon through processes including coherent elastic neutrino-nucleus scattering (CEvNS).

These interactions can produce nuclear-recoil events that resemble expected signals from some light dark-matter interactions.

As dark-matter experiments become increasingly sensitive, neutrino-induced events become an increasingly important background.

Therefore, XENON-QAI will explicitly investigate candidate ranking under neutrino-induced backgrounds.

---

# 📊 Dataset Strategy

The initial development will use publicly available **simulated XENON1T event data** containing labeled signal and background events.

The dataset provides multiple event representations, including:

- Images
- Arrays
- Resumed features
- cS1-cS2 representations

The simulated dataset includes multiple background categories and WIMP signal masses.

The first step is **not** to assume which representation is best.

Instead:

```text
Dataset
   ↓
Inspect representations
   ↓
Understand event structure
   ↓
Analyze features
   ↓
Determine suitable architecture
```

Real/public XENON data will be treated differently because real detector events do not contain ground-truth labels identifying dark matter.

Real data will therefore be investigated through:

- Distribution analysis
- Candidate ranking
- Anomaly analysis
- Generalization studies
- Detector-consistency checks

---

# 🧠 Deep Learning Strategy

XENON-QAI will not assume a particular neural architecture before inspecting the dataset.

Potential architectures include:

```text
Sequence / Waveform
        ↓
Transformer / 1D CNN

Spatial / Relational Structure
        ↓
Graph Neural Network

Image / PMT Map
        ↓
CNN

Other Structured Representations
        ↓
Architecture selected according to data
```

A Transformer, GNN, CNN, RNN, or another architecture will only be used where its inductive bias is appropriate for the actual event representation.

The goal is **not to build a model zoo**.

The goal is to identify a meaningful learned representation of XENON detector events.

---

# 🧩 Learned Latent Representation

The deep-learning model will produce a compact learned representation:

```text
Detector Event
      ↓
Deep Learning Encoder
      ↓
Latent Vector
```

The latent representation becomes the common input for the later classical and quantum inference experiments.

This creates a controlled comparison:

```text
                 Learned Latent Vector
                         │
              ┌──────────┴──────────┐
              │                     │
       Classical Head         Quantum Head
              │                     │
              └──────────┬──────────┘
                         ↓
                  Compare Results
```

---

# ⚛️ Quantum Machine Learning

The quantum component will operate **after** deep-learning representation learning.

The full deep-learning model will not be transferred to a quantum computer.

Instead:

```text
Deep Learning
      ↓
Latent Representation
      ↓
Dimensionality Reduction
      ↓
Quantum-Compatible Vector
      ↓
Feature Encoding
      ↓
Quantum Circuit
      ↓
Measurement
      ↓
Classification / Score
```

Potential quantum approaches include:

- Variational Quantum Classifier
- Quantum Neural Network
- Quantum Kernel Methods
- Parameterized Quantum Circuits

The number of qubits and circuit complexity will be determined experimentally rather than assumed beforehand.

---

# ⚖️ Classical vs Quantum Comparison

The central controlled experiment is:

```text
                 Same Deep-Learning Latent Vector
                              │
                 ┌────────────┴────────────┐
                 │                         │
          Classical Head             Quantum Head
                 │                         │
                 └────────────┬────────────┘
                              ↓
                       Fair Comparison
```

Both approaches should receive the **same learned representation**.

The comparison will consider more than accuracy.

### Classical / Quantum Performance

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Signal efficiency
- Background rejection
- Significance-related measures where appropriate

### Quantum Resources

- Number of qubits
- Circuit depth
- Trainable parameters
- Measurement shots
- Noise sensitivity
- Computational requirements
- Scaling behavior

---

# 🔎 Research Experiments

The project will progressively investigate:

### Experiment 1 — Dataset Exploration

Understand:

- Event structure
- Representations
- Features
- Labels
- Class distributions
- Data quality

### Experiment 2 — Classical Reference Baselines

Establish reference performance using simple and strong classical methods.

### Experiment 3 — Deep-Learning Architecture Selection

Determine which architecture is appropriate for the actual XENON representation.

### Experiment 4 — Deep Representation Learning

Train the selected deep-learning model and extract latent representations.

### Experiment 5 — Latent-Space Analysis

Investigate:

- Representation quality
- Class separation
- Clustering
- Dimensionality
- Physics-related structure

### Experiment 6 — Classical Latent Classifier

Train a classical inference head using the learned latent representation.

### Experiment 7 — Quantum Latent Classifier

Use the same representation with a quantum inference model.

### Experiment 8 — Classical vs Quantum

Perform a controlled comparison between the two inference approaches.

### Experiment 9 — Data Scarcity

Investigate performance using different training-data fractions:

```text
1%
5%
10%
25%
50%
100%
```

### Experiment 10 — Neutrino Fog

Explicitly investigate performance in the presence of neutrino-induced backgrounds such as CEvNS.

### Experiment 11 — Anomaly Detection

Train primarily on known backgrounds and investigate whether unusual or dark-matter-like events receive high anomaly scores.

### Experiment 12 — Simulation Domain Shift

Evaluate whether the model learns meaningful physics rather than simulator-specific artifacts.

### Experiment 13 — Simulation to Real

Where suitable public XENON data are available, investigate generalization and event-score distributions.

### Experiment 14 — Quantum Resource Scaling

Study:

- Qubit count
- Circuit depth
- Parameter count
- Measurement shots

### Experiment 15 — Quantum Noise

Compare ideal quantum simulation with realistic noisy simulation.

### Experiment 16 — Explainability

Investigate:

- Important detector features
- Learned representations
- Attention or feature attribution
- Physics-consistency behavior

---

# 🔬 Research Architecture

```text
                  XENON EVENT DATA
                         │
                         ▼
              ┌────────────────────┐
              │ Data Preprocessing  │
              │                    │
              │ S1 / S2            │
              │ Timing             │
              │ Spatial Features   │
              │ Waveforms          │
              │ Event Features     │
              └─────────┬──────────┘
                        │
                        ▼
              Physics-Aware Analysis
                        │
                        ▼
              Deep Learning Encoder
                        │
             Transformer / CNN / GNN*
                        │
                        ▼
                 Latent Space
                        │
                        ▼
             Quantum-Compatible
                 Compression
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
       Classical Head       Quantum Head
              │                   │
              └─────────┬─────────┘
                        ▼
              Candidate / Anomaly
                    Score
                        │
                        ▼
                    Ranking
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Neutrino Fog   Domain Shift   Real Data
          │             │             │
          └─────────────┼─────────────┘
                        ▼
              Explainability +
             Uncertainty Analysis
```

`*` Architecture will be selected after dataset inspection.

---

# 🧪 Repository Structure

```text
xenon-qai/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── training.yaml
│   ├── quantum.yaml
│   └── experiments.yaml
│
├── data/
│   └── README.md
│
├── docs/
│   ├── physics/
│   ├── dataset/
│   ├── research/
│   └── ip/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_visualization.ipynb
│   ├── 03_physics_feature_analysis.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_deep_learning_experiments.ipynb
│   ├── 06_latent_space_analysis.ipynb
│   ├── 07_quantum_experiments.ipynb
│   ├── 08_classical_vs_quantum.ipynb
│   ├── 09_neutrino_fog.ipynb
│   ├── 10_anomaly_detection.ipynb
│   ├── 11_domain_shift.ipynb
│   ├── 12_real_xenon_inference.ipynb
│   └── 13_explainability.ipynb
│
├── src/
│   └── xenon_qai/
│       ├── data/
│       ├── physics/
│       ├── representations/
│       ├── models/
│       ├── representation/
│       ├── quantum/
│       ├── anomaly/
│       ├── domain/
│       ├── uncertainty/
│       ├── explainability/
│       ├── evaluation/
│       └── utils/
│
├── experiments/
│   ├── 001_dataset/
│   ├── 002_baselines/
│   ├── 003_architecture_selection/
│   ├── 004_deep_learning/
│   ├── 005_latent_space/
│   ├── 006_classical_latent_head/
│   ├── 007_quantum_model/
│   ├── 008_classical_vs_quantum/
│   ├── 009_data_scarcity/
│   ├── 010_neutrino_fog/
│   ├── 011_anomaly_detection/
│   ├── 012_simulation_domain_shift/
│   ├── 013_simulation_to_real/
│   ├── 014_quantum_resources/
│   ├── 015_quantum_noise/
│   └── 016_explainability/
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── checkpoints/
│   ├── predictions/
│   ├── latent_spaces/
│   └── experiment_logs/
│
├── scripts/
└── tests/
```

---

# 📌 Development Roadmap

```text
Stage 1   Physics + Dataset Understanding
    ↓
Stage 2   Dataset Preparation + Reproducible Pipeline
    ↓
Stage 3   Classical Reference Baselines
    ↓
Stage 4   Deep-Learning Architecture Selection
    ↓
Stage 5   Deep-Learning Training
    ↓
Stage 6   Latent-Space Extraction + Analysis
    ↓
Stage 7   Classical Inference Head
    ↓
Stage 8   Quantum Inference Head
    ↓
Stage 9   Controlled Classical-vs-Quantum Experiments
    ↓
Stage 10  Neutrino Fog + Data Scarcity
    ↓
Stage 11  Anomaly Detection
    ↓
Stage 12  Simulation Domain Shift
    ↓
Stage 13  Public/Real XENON Data Analysis
    ↓
Stage 14  Quantum Noise + Resource Analysis
    ↓
Stage 15  Explainability + Uncertainty
    ↓
Stage 16  Research Paper + Prototype + IP Review
```

---

# 💡 Research Novelty Direction

The project does **not** claim novelty from simply using:

- Transformers
- Deep learning
- Quantum machine learning
- Anomaly detection
- Quantum computing in particle physics

These areas already have prior research.

The potential research contribution is instead investigated through the **specific combination and experimental methodology**, including:

```text
Learned liquid-xenon event representation
                +
Quantum-compatible latent compression
                +
Controlled classical-vs-quantum inference
                +
Neutrino-induced background analysis
                +
Data-scarcity experiments
                +
Simulation/domain-shift evaluation
                +
Quantum resource/noise analysis
                +
Candidate ranking / anomaly analysis
```

The final novelty claim will only be made after implementation and prior-art analysis establish what is genuinely differentiated.

---

# 🧠 Scientific Philosophy

XENON-QAI follows several principles:

### 1. Physics before architecture

The model should be selected according to the structure of the detector data.

### 2. No assumed quantum advantage

Quantum machine learning is an experimental component, not a predetermined winner.

### 3. Fair comparison

Classical and quantum inference models receive the same learned representation.

### 4. Candidate identification, not discovery

A high model score does not prove that an event is dark matter.

### 5. Reproducible experiments

Dataset versions, preprocessing, configurations, random seeds, model parameters, and experimental results should be tracked.

### 6. Research over model quantity

The objective is to understand the scientific problem rather than accumulate as many architectures as possible.

---

# 🔐 Intellectual Property

The project is being developed with potential intellectual-property protection in mind.

The repository should remain private during active development and prior-art/IP evaluation.

Potential areas of investigation include:

- Adaptive compression of detector representations into quantum-compatible latent spaces
- Resource-aware quantum circuit selection
- Hybrid deep-learning and quantum candidate ranking
- Uncertainty-aware candidate prioritization
- Robust simulation-to-real adaptation

These are **research directions, not established patent claims**.

Any eventual patent filing should be reviewed through the appropriate institutional IP office or patent professional.

---

# 🖥️ Hardware

Hardware is **not part of the core implementation**.

The primary project is software and simulation based.

A future extension could investigate an FPGA or embedded preprocessing layer capable of:

```text
Detector-like Pulse Signal
          ↓
Digitization
          ↓
Filtering
          ↓
Feature Extraction
          ↓
Compact Representation
          ↓
Existing AI + QML Pipeline
```

Hardware will only be considered if a concrete technical function and realistic prototype can be defined.

---

# 📚 Current Development Principle

The project will proceed incrementally.

The first implementation milestone is:

```text
Public XENON Dataset
        ↓
Dataset Exploration
        ↓
Data Representation Analysis
        ↓
Reproducible Data Pipeline
```

Only after understanding the dataset will the deep-learning architecture be finalized.

The minimum strong version of the project is:

```text
Simulated XENON Data
        ↓
Deep Learning
        ↓
Latent Representation
        ↓
Classical Head vs Quantum Head
        ↓
Fair Evaluation
        ↓
Neutrino-Background Experiment
```

Advanced experiments such as domain shift, real-data analysis, anomaly detection, quantum noise, resource scaling, uncertainty, and explainability will be added incrementally.

---

## 🔬 Project Status

**Current Phase:**

```text
[████░░░░░░░░░░░░░░░░] Physics + Dataset Preparation
```

**Current priorities:**

- [ ] Understand XENON detector physics
- [ ] Download and inspect the public dataset
- [ ] Analyze available event representations
- [ ] Build dataset exploration notebook
- [ ] Establish reproducible preprocessing
- [ ] Select appropriate deep-learning architecture
- [ ] Train first research model
- [ ] Extract latent representation
- [ ] Implement classical latent head
- [ ] Implement quantum latent head
- [ ] Begin controlled experiments

---

## ⚠️ Data Policy

Raw and processed datasets are **not committed to Git**.

Large files such as:

```text
*.h5
*.hdf5
*.npy
*.npz
*.csv
*.parquet
*.root
```

should remain outside the repository or be excluded through `.gitignore`.

Dataset provenance and download information should be documented in:

```text
docs/dataset/
```

---

## 📄 License

MIT License

---

## 🧑‍🔬 Project

**XENON-QAI**

Deep Learning + Quantum Machine Learning for Dark Matter Candidate Identification in Liquid-Xenon Detectors.

**Research project — active development**
