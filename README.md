# AI-Assisted OTFS-VLC Interactive Dashboard

An interactive, research-grade web simulation dashboard for **AI-Assisted Orthogonal Time Frequency Space (OTFS) Modulation in Visible Light Communication (VLC)** tailored for 6G High-Mobility Vehicular Communications.

[![Netlify Status](https://api.netlify.com/api/v1/badges/deploy-status?branch=main)](https://www.netlify.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🌟 Overview & Key Features

This interactive dashboard benchmarks three key physical-layer waveforms under realistic high-mobility vehicular conditions (up to 160 km/h) and severe optical channel impairments:

1. **Conventional OFDM-VLC:** Baseline multi-carrier scheme susceptible to high Doppler shifts, Inter-Carrier Interference (ICI), and high PAPR.
2. **Classical MMSE-OTFS:** 2D Delay-Doppler domain modulation with Linear MMSE Equalization, transforming time-varying multipath channels into quasi-static 2D channels.
3. **Proposed 2D-CNN AI-OTFS:** Deep learning-based equalizer utilizing a 4-layer 2D Convolutional Neural Network with Residual Connections and LeakyReLU activations, achieving a **1.85 dB SNR gain** and robust non-linear LED clipping recovery.

---

## 🚀 Live Interactive Controls

- **Vehicle Velocity Slider:** Real-time Doppler shift calculation ($f_d = \frac{v}{\lambda_c}$) up to 160 km/h.
- **Link Distance & Optical Geometry:** Computes Lambertian Line-of-Sight ($H_{\text{LoS}}$) channel gain with semi-angle $\Phi_{1/2} = 60^\circ$.
- **LED DC Bias & Optical Saturation:** Simulates unipolar conversion and LED non-linear transfer curve clipping.
- **Channel State Information (CSI) Error:** Tests equalization resilience against imperfect channel estimation (0%–30%).
- **Dual Theme Support:** One-click toggle between Dark Mode and Light Mode with dynamic Plotly recoloring.
- **Optical Diagnostics Tab:** Live 2D Delay-Doppler channel sparsity heatmap ($16 \times 32$) and real-time LED optical clipping waveform.

---

## 🛠️ Deployment & Local Setup

### Local Run
Simply open `index.html` in any modern web browser (Google Chrome, Microsoft Edge, Safari, Firefox). No backend server or Node.js runtime required!

```bash
# Optional: Serve locally using Python
python -m http.server 8000
```
Open `http://localhost:8000` in your browser.

### Netlify Deployment
1. Import this repository into [Netlify](https://app.netlify.com/).
2. Set **Publish directory** to `.`.
3. Leave **Build command** empty.
4. Deploy!

---

## 📊 Project Specifications

| Parameter | Specification / Value |
| :--- | :--- |
| **Carrier / Center Frequency** | Optical: 450 nm (Blue LED) / RF equiv: 30 GHz |
| **Grid Dimensions** | $M = 32$ (Delay bins), $N = 16$ (Doppler bins) |
| **Modulation Schemes** | QPSK, 16-QAM |
| **Doppler Resilience** | Stable at speeds $> 100\text{ km/h}$ |
| **AI Architecture** | 4-Layer 2D-CNN with Residual Skip Connections |
| **Net AI Gain** | $+1.85\text{ dB}$ over classical MMSE equalizer |
