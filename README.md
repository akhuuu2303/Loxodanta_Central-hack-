# 🐘 TUSK TRUST AI: Unsupervised Elephant Health Monitoring

TUSK TRUST AI is a computational biology platform designed to monitor African elephant health through non-invasive fecal microbiome analysis. By leveraging **Variational Autoencoders (VAEs)**, the system identifies gut dysbiosis and physiological stress signatures without the need for invasive blood tests or physical immobilization.



---

## The Biological Thesis
The gut microbiome of the African elephant (*Loxodonta africana*) is a highly sensitive indicator of metabolic and environmental stress.
* **Healthy Baseline:** Characterized by a dominant Firmicutes population and a balanced Firmicutes/Bacteroidetes (F/B) ratio of **~2.79**.
* **Dysbiosis Signature:** Stress factors (such as human-wildlife conflict or diet shifts) trigger a crash in Firmicutes and a corresponding spike in Bacteroidetes, leading to a measurable "F/B Inversion."

---

## AI Architecture: Variational Autoencoder (VAE)
Unlike traditional supervised models that require labeled "sick" data, TUSK TRUST AI uses an **Unsupervised Anomaly Detection** approach.

### Key Features:
* **Unsupervised Latent Space:** The VAE is trained exclusively on healthy African elephant profiles, learning the complex mathematical "manifold" of a healthy gut.
* **Reconstruction Error as a Proxy for Health:** When presented with an unseen stressed profile, the VAE fails to accurately reconstruct the data. This **Mean Squared Error (MSE)** is then mapped to a standardized **Risk Score**.
* **Traffic-Light Diagnostics:** Automated classification into three actionable zones:
  *  **Green (<30%):** Stable Baseline.
  *  **Yellow (30-70%):** Early Warning / Minor Dysbiosis.
  *  **Red (>70%):** Critical Anomaly / Acute Stress.



---

##  Data Pipeline & Preprocessing
To ensure scientific validity and prevent data leakage, the pipeline implements:
1. **Biological Standardization:** Unified 328-feature genus-level abundance matrix.
2. **Anti-Leakage Protocol:** Mathematical purging of test subjects from the training pool to ensure 100% blind testing.
3. **Synthetic Dysbiosis Injection:** Generation of realistic testing sets using a literature-backed **2.8x - 3.0x Bacteroidetes spike** to validate detection sensitivity.

---

##  Model Performance
Validated on unseen African elephant data, the TUSK TRUST VAE currently demonstrates:
* **Accuracy:** 97% (on validated Test_1 datasets)
* **F1-Score:** 0.88
* **ROC-AUC:** 0.89



---

##  Installation & Usage
```bash
# Clone the repository
git clone [https://github.com/your-username/TUSK TRUST-ai.git](https://github.com/your-username/TUSK TRUST-ai.git)

# Install dependencies
pip install torch pandas numpy scikit-learn seaborn
