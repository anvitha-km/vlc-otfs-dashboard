# AI-Assisted OTFS mmWave 6G Communication, Sensing & Metasurface System

## Complete Multi-Metric Project Package (28 GHz / 24.25 GHz 6G-V2X)

This package contains the complete, newly built physical-layer simulation engine and interactive demonstration dashboard for **AI-Assisted OTFS mmWave (28 GHz / 24.25 GHz) Systems**, designed for high-mobility 6G vehicular and aerial networks (speeds up to 500 km/h).

It directly synthesizes the principles, channel formulations, and deep learning algorithms from **nine high-impact reference papers**:
1. **Mutescu et al. (Sensors 2025):** 1D-CNN Waveform Classifier (OFDM vs. OTFS 99.5%) & Multitask Numerology Detection.
2. **Li et al. (IEEE/arXiv 2022):** Hybrid RIS (HRIS) mmWave OTFS with Joint Channel Estimation & Data Detection (JCEDD).
3. **Anilkumar & Sengan (Sci Rep 2026):** Deep Bayesian Gaussian Processes & Compressive Sensing (DBGP-CS) reducing pilot overhead by 50%.
4. **Wang et al. (IEEE Access 2025):** Precision Sensing-Aided Multi-Target Beamforming Prediction in ISAC-OTFS (MDC-OMP).
5. **Jin et al. (IEEE TVT 2026):** Two-Stage Super-Resolution SBL-OMP & Predictable Beamforming Optimization in MIMO-OTFS ISAC.
6. **Shokoohifard & Fapojuwo (IEEE Access 2026):** Noncoherent Differential Modulation (DM) with OTFS & SECTNet CNN-Transformer (DDRNet).
7. **Abid et al. (IEEE Access 2023):** Reconfigurable Intelligent Surface-Aided OTFS Channel Matrix & Deep Learning Signal Detection.
8. **Singh et al. (IEEE OJVT 2026):** Unsupervised Deep Learning Hybrid Beamforming (DLBF) & Data-Augmented CNN Detector under Hardware Impairments.
9. **Noor-A-Rahim, Liu, Saad, Poor et al. (Proc. IEEE 2022):** 6G for Vehicle-to-Everything (V2X) Communications: Enabling Technologies & Architectures.

---

### Directory Structure & Deliverables:

```
AI_Assisted_OTFS_mmWave_Project/
├── 01_Reports_and_Documentation/
│   └── AI_Assisted_OTFS_mmWave_Technical_Report.md     <-- Complete technical report & Viva Q&A
├── 02_Interactive_Demonstration_Dashboard/
│   ├── rf_mmwave_otfs_dashboard.html                   <-- 60 FPS Standalone Interactive Web Dashboard
│   ├── index.html                                      <-- Instant 1-click Netlify / GitHub Pages mirror
│   └── app_streamlit_rf.py                             <-- Python Streamlit web dashboard
├── 03_Simulation_Source_Code/
│   ├── ai_otfs_rf_mmwave_sim.py                        <-- Core PyTorch / NumPy simulation engine
│   └── generate_all_mmwave_plots.py                    <-- Standalone high-res plot generation script
└── 04_Figures_and_Plots/                               <-- 8 Publication-grade generated PNG figures
    ├── metric1_ber_vs_snr_waterfall.png
    ├── metric2_ber_vs_speed_doppler.png
    ├── metric3_constellations_and_beamforming.png
    ├── metric4_goodput_and_spectral_efficiency.png
    ├── metric5_delay_doppler_sparsity_heatmap.png
    ├── metric6_ai_waveform_and_numerology.png
    ├── metric7_isac_trajectory_tracking.png
    └── metric8_hris_metasurface_phase_profile.png
```

---

### Quick Start Instructions:

#### 1. Instant Interactive Web Dashboard (Zero Installation Required)
Simply double click or open in any browser:
* `rf_mmwave_otfs_dashboard.html` (located in root and in `02_Interactive_Demonstration_Dashboard/`)

It features:
* Responsive, high-tech dark theme with live 60 FPS client-side physical recalculation.
* Interactive sidebar controlling carrier frequency ($28\text{ GHz}$, $24.25\text{ GHz}$, $3.5\text{ GHz}$), vehicle velocity ($0\text{–}500\text{ km/h}$), ETU multipath, Massive MIMO array size, Hybrid RIS, and hardware impairments.
* 3 organized view tabs displaying all 8 interactive demonstration plots.
* 4 pre-configured review presets for fast defense demonstration.

#### 2. Running the Python Simulation Engine
```bash
python AI_Assisted_OTFS_mmWave_Project/03_Simulation_Source_Code/ai_otfs_rf_mmwave_sim.py
```

#### 3. Running the Streamlit Web Application
```bash
pip install streamlit plotly
streamlit run app_streamlit_rf.py
```
