# A Multi-Stage Hybrid Deep Learning Framework for Interpretable Anomaly Detection in Environmental Sensor Networks

This repository implements the framework presented in our manuscript titled "A Multi-Stage Hybrid Deep Learning Framework for Interpretable Anomaly Detection in Environmental Sensor Networks." Our approach integrates convolutional and recurrent neural architectures with interpretable AI techniques for anomaly detection in real-time sensor data collected from diverse environmental settings.
🔍 Overview

Environmental sensor networks generate high-frequency data prone to anomalies due to sensor drift, environmental noise, or malicious attacks. This project presents a multi-stage hybrid deep learning (MS-HDL) architecture that performs:

    Temporal and spatial feature extraction

    Anomaly detection using unsupervised deep learning

    Interpretable analysis using SHAP (SHapley Additive exPlanations)

    Performance benchmarking on SMAP, SWaT, and Niger Delta real-world datasets

📊 Key Contributions

    Multi-Stage Pipeline combining CNNs and Bi-LSTM layers for robust temporal-spatial anomaly learning.

    Anomaly Detection Module using thresholding on reconstruction errors and predictive residuals.

    SHAP-based Interpretability to explain anomaly decisions and identify faulty sensors.

    Ablation Study & Latency Profiling for deployment in low-power IoT settings.

🧠 Architecture

graph TD
A[Sensor Data Input] --> B[Preprocessing & Normalization]
B --> C[CNN Feature Extraction]
C --> D[Bi-LSTM Sequence Modeling]
D --> E[Reconstruction & Prediction]
E --> F[Error Analysis]
F --> G[Anomaly Detection + SHAP Explanations]

🗂️ Repository Structure

├── data/
│   ├── smap/                 # NASA SMAP dataset
│   ├── swat/                 # SWaT ICS dataset
│   └── niger_delta/          # Real-world environmental data
├── models/
│   ├── cnn_bilstm.py         # Hybrid architecture
│   ├── autoencoder.py        # AE baseline
├── utils/
│   ├── preprocessing.py      # Data loaders and transformers
│   ├── evaluation.py         # Metrics and plots
│   └── shap_utils.py         # SHAP explainability functions
├── notebooks/
│   ├── EDA.ipynb             # Exploratory analysis
│   ├── train_model.ipynb     # Training workflow
│   └── shap_explanation.ipynb# SHAP visualizations
├── results/
│   ├── performance_tables/
│   └── figures/
├── main.py                   # Main training & inference script
├── requirements.txt
└── README.md

📈 Results
Dataset	Precision	Recall	F1-Score	AUC-ROC
SMAP	0.94	0.91	0.925	0.97
SWaT	0.96	0.92	0.94	0.98
Niger Delta	0.89	0.87	0.88	0.95

    Latency (inference): ~40ms per window on edge hardware (Raspberry Pi 4)

    Interpretability: Top 5 SHAP values identify faulty sensors and feature contributions

⚙️ Installation

git clone https://github.com/your-username/environmental-anomaly-detection.git
cd environmental-anomaly-detection
pip install -r requirements.txt

▶️ Usage
Train the model

python main.py --train --dataset smap

Evaluate and visualize SHAP

python main.py --eval --shap --dataset swat

🧪 Datasets

    SMAP: NASA Soil Moisture Active Passive Satellite Link

    SWaT: Secure Water Treatment Dataset Link

    Niger Delta: Real-world environmental data (to be made available upon request or after publication)

📘 Citation

If you use this code or dataset in your research, please cite:

@article{akajiaku2025deep,
  title={A Multi-Stage Hybrid Deep Learning Framework for Interpretable Anomaly Detection in Environmental Sensor Networks},
  author={Akajiaku, Ugochukwu Charles and et al.},
  journal={Machine Learning (Springer)},
  year={2025}
}

🔐 License

This project is licensed under the MIT License.
📬 Contact

For questions, feedback, or dataset access, please reach out to:

    Ugochukwu Charles Akajiaku

    Email: [Insert Email]

    GitHub: https://github.com/Akajiaku1

    LinkedIn: https://www.linkedin.com/in/akajiaku
