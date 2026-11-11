# PHYSICS & DATA PIPELINE SCIENTIFIC AUDIT

**Repository:** XENON-QAI  
**Audit Target:** `src/xenon_qai/data/`, feature extraction routines, and HDF5 physical integrity.  

---

## 1. Detector Physics & Feature Extraction Verification

### S1 and S2 Pulse Extraction
- **S1 Prompt Scintillation:**
  - Extracted from `peakS1_onlyhits` (isolated photoelectron counts per nanosecond bin).
  - Raw length varies from 45 to 155 bins.
  - Zero-padded to fixed length $S1_{\text{len}} = 160$. Because $160 > \max(\text{length})$, 100% of physical S1 pulses are captured with zero truncation.
- **S2 Ionization Drift Waveform:**
  - Extracted from `peakS2_onlyhits`.
  - Raw length varies broadly between 191 and 1,484 bins due to z-dependent electron diffusion.
  - Standardized to fixed length $S2_{\text{len}} = 500$ using linear 1D interpolation.
  - **Area Conservation:** The resampled waveform is multiplied by $\frac{N_{\text{raw}}}{500}$, rigorously preserving the total integrated charge ($\int S2 \, dt$) across interpolation.

### PMT Spatial Normalization
- **Top PMT Array:** 127 channels (`HITS_top`).
- **Bottom PMT Array:** 121 channels (`HITS_bot`).
- **Total Physical PMTs:** 248 channels.
- **Transform:** Applied $\log(1 + x)$ followed by z-score standardization:
  $$x_{\text{norm}} = \frac{\log(1 + x) - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  All $\mu_{\text{train}}, \sigma_{\text{train}}$ parameters were fitted strictly on training events.

### 13D Domain-Engineered Physics Features
The pipeline extracts 13 hand-crafted features derived from Noble Element Simulation Technique (NEST) principles:
1. `s1_area_hits`: Integrated photoelectrons (PE) of S1 pulse.
2. `s2_area_hits`: Integrated photoelectrons (PE) of S2 pulse.
3. `s1_area_raw`: Integrated PE of raw S1 pulse before baseline filtering.
4. `s2_area_raw`: Integrated PE of raw S2 pulse.
5. `log_s2_s1`: $\log_{10}\left(\frac{S2_{\text{area}} + 10^{-3}}{S1_{\text{area}} + 10^{-3}}\right)$ — Primary nuclear vs electronic recoil discriminator.
6. `s1_len`: Pulse width / duration of S1 in nanoseconds.
7. `s2_len`: Pulse width / envelope duration of S2.
8. `s1_max`: Peak amplitude of S1 pulse.
9. `s2_max`: Peak amplitude of S2 pulse.
10. `top_sum`: Total charge detected across top PMT array.
11. `bot_sum`: Total charge detected across bottom PMT array.
12. `total_pmt_hits`: `top_sum` + `bot_sum`.
13. `top_bot_asym`: $\frac{\text{top\_sum} - \text{bot\_sum}}{\text{total\_pmt\_hits} + 10^{-6}}$ — Depth proxy for interaction z-coordinate.

---

## 2. Inconsistencies & Scientific Safeguards

1. **Multimodal Vector Composition:**  
   The 908D multimodal vector consists of:
   $$160 \text{ (S1)} + 500 \text{ (S2)} + 127 \text{ (Top PMT)} + 121 \text{ (Bot PMT)} = 908 \text{ dimensions}.$$
   The 13 domain features enter the Multimodal CNN through a dedicated auxiliary MLP encoder branch (`PhysicsFeatureMLP`), which fuses with the 908D latent projection before the bottleneck.
2. **Missing Detector Metadata:**  
   The raw HDF5 files contain simulated pulse waveforms and spatial PMT hit patterns from Geant4/NEST. However, exact photomultiplier calibration gain maps and unbinned photon arrival times are absent in the curated arrays. The manuscript should acknowledge that waveform representations are pre-integrated into binned time arrays rather than continuous analog pulses.

