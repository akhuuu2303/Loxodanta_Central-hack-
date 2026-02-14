# 🐘 Tusk Trust Sentinel

> **The Next Generation of Wildlife Sentinel Systems. Utilizing Deep Learning & Geospatial Fusion for Extinction Prevention.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)
![GEE](https://img.shields.io/badge/Google_Earth_Engine-Geospatial-34A853.svg)

Tusk Trust Sentinel is a proactive, AI-driven triage dashboard built for conservationists and veterinarians. By merging non-invasive biological sampling (gut metagenomics) with global satellite telemetry, it acts as an early warning radar for ecological collapse in keystone species like elephants.



---

## 📖 Table of Contents
- [The Problem](#-the-problem)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Future Roadmap](#-future-roadmap)

---

## 🚨 The Problem
Currently, wildlife conservation is deeply **reactive**. Vets are only deployed *after* an animal is visibly sick, which is often too late. Furthermore, traditional "Black Box" classification models require thousands of labeled "sick" examples to learn—data that simply doesn't exist for endangered species.

**The Solution:** Tusk Trust Sentinel uses an unsupervised deep learning approach trained entirely on *healthy* biological baselines, combined with live environmental satellite data, to catch microscopic health deviations before they become systemic failures.

---

## 🚀 Key Features

* **🧠 Unsupervised Biological Inference:** Routes high-dimensional genomic data (326 bacterial features) through a custom PyTorch Variational Autoencoder (VAE) to calculate a continuous 'Anomaly Score'.
* **🛰️ Geospatial Telemetry (GEE Fusion):** Automatically extracts coordinates from uploaded samples and pings the Google Earth Engine API to fetch real-time landscape data (NDVI, Rainfall, Temperature).
* **🌍 Eco-Biological Correlation:** Maps internal biological anomalies against hardcoded ecological danger zones (e.g., >35°C heat stress, <20mm severe drought) to determine the root cause of the biological shift.
* **📦 Bulk Interpretability Export:** In-memory ZIP compilation allows researchers to instantly generate and download Microbiome Deviation Heatmaps for an entire herd with a single click.
* **📊 Automated Triage Reporting:** Calculates a localized 'Risk Severity %' and appends specific landscape warnings into a structured, downloadable CSV for rapid veterinary dispatch.

---

## ⚙️ System Architecture

Our platform employs a multi-layered approach to ensure interpretability and scientific rigor:

1. **Dimensionality Reduction:** The VAE Encoder compresses 326 high-dimensional genomic features down to a dense 12D 'latent space'.
2. **Reconstruction Penalty:** The Decoder attempts to reconstruct the original healthy microbiome. Stressed/sick data results in a high Reconstruction Error (used as our Anomaly Score).
3. **Contextual Arbitration:** The Anomaly Score is plotted against satellite telemetry to separate landscape-driven stress (drought) from internal disease (pathogens).

---

## 💻 Installation & Setup

### Prerequisites
* Python 3.9+
* (Optional) Google Earth Engine authenticated account. *Note: The system will safely fall back to simulated telemetry if GEE is not authenticated locally.*

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/Tusk-Trust-Sentinel.git](https://github.com/YourUsername/Tusk-Trust-Sentinel.git)
cd Tusk-Trust-Sentinel



---

##  Installation & Usage
```bash
# Clone the repository
git clone [https://github.com/your-username/TUSK TRUST-ai.git](https://github.com/your-username/TUSK TRUST-ai.git)

# Install dependencies
pip install torch pandas numpy scikit-learn seaborn
