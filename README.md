App Link: https://predictivemaintenanceofmillingmachine-zggtvf8z6xuoxgqmwgdjtj.streamlit.app/
# 🏭 Predictive Maintenance of Milling Machine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ishivamm/Predictive_maintenance_of_milling_machine/blob/main/Predictive_Maintenance_ML_Pipeline.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictivemaintenanceofmillingmachine-zggtvf8z6xuoxgqmwgdjtj.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest%20%7C%20Logistic%20Regression-green)

An end-to-end machine learning pipeline that predicts equipment failures in CNC milling machines **before they occur** — enabling proactive maintenance, reducing downtime, and cutting operational costs.

---

## 📌 Project Overview

Unplanned machine failures in manufacturing can halt production lines and cost thousands per hour. This project applies supervised machine learning to the **AI4I 2020 Predictive Maintenance Dataset** to build a binary classifier that identifies whether a milling machine is at risk of failure based on real-time sensor readings.

Key challenges tackled:
- Severe class imbalance (only **3.39% failure rate**)
- Physics-informed feature engineering from raw sensor data
- Multi-model benchmarking with hyperparameter tuning
- Deployment as a live interactive Streamlit dashboard

---

## 🚀 Live Demo

🔗 **[Try the Streamlit App →](https://predictivemaintenanceofmillingmachine-zggtvf8z6xuoxgqmwgdjtj.streamlit.app/)**

Enter real-time machine parameters and get an instant failure prediction with probability scores.

---

## 📊 Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository

| Property | Value |
|---|---|
| Samples | 10,000 |
| Features | 14 (raw) → 13 (engineered) |
| Target | `Machine failure` (binary: 0 / 1) |
| Failure rate | 3.39% (severe class imbalance) |
| Failure types | TWF, HDF, PWF, OSF, RNF |

**Raw Features:**

| Feature | Description |
|---|---|
| `Type` | Product quality tier — L (Low), M (Medium), H (High) |
| `Air temperature [K]` | Ambient air temperature in Kelvin |
| `Process temperature [K]` | Process/coolant temperature in Kelvin |
| `Rotational speed [rpm]` | Spindle speed |
| `Torque [Nm]` | Cutting torque |
| `Tool wear [min]` | Cumulative tool wear time |

---

## 🧠 ML Pipeline

```
Raw Data → EDA → Feature Engineering → Preprocessing → SMOTE → Model Training → Evaluation → Deployment
```

### 1. Exploratory Data Analysis (EDA)
- Target variable distribution & class imbalance visualization
- Feature distributions split by failure status (KDE + histograms)
- Box plots for outlier and spread analysis
- Correlation heatmap
- Pair plots for key process features
- Failure-type frequency breakdown (TWF, HDF, PWF, OSF, RNF)

### 2. Physics-Informed Feature Engineering

Five domain-driven features derived from mechanical engineering principles:

| New Feature | Formula | Physical Rationale |
|---|---|---|
| `Temp_diff` | Process Temp − Air Temp | Temperature gradient → heat stress |
| `Power` | Torque × Rotational Speed | Mechanical power draw |
| `Torque_speed_ratio` | Torque / Speed | High ratio = high cutting strain |
| `Wear_torque` | Tool Wear × Torque | Worn tools under load → failure risk |
| `Failure_mode_count` | Sum of TWF, HDF, PWF, OSF, RNF | Concurrent fault indicator |

### 3. Preprocessing
- Label encoding + one-hot encoding for `Type` (L/M/H)
- Stratified 80/20 train-test split
- `StandardScaler` for numerical feature normalization
- **SMOTE** (Synthetic Minority Oversampling Technique) to handle the 3.39% minority class

### 4. Models Trained

| Model | Tuning |
|---|---|
| Logistic Regression | Default |
| Random Forest | RandomizedSearchCV |
| XGBoost | RandomizedSearchCV |

Hyperparameter search uses `StratifiedKFold` (k=3) with F1-score as the optimization metric.

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices for all models
- ROC curves and Precision-Recall curves
- Feature importance (Gini for RF, Gain for XGBoost)
- Logistic Regression coefficient analysis

---

## 📁 Repository Structure

```
Predictive_maintenance_of_milling_machine/
│
├── Predictive_Maintenance_ML_Pipeline.ipynb  # Full ML pipeline notebook
├── app_basic.py                              # Streamlit deployment app
├── ai4i2020.csv                              # Dataset
├── requirements.txt                          # Python dependencies
├── LICENSE
│
└── model_artifacts/
    ├── best_model.joblib                     # Trained best model
    ├── scaler.joblib                         # Fitted StandardScaler
    ├── label_encoder_type.joblib             # Fitted LabelEncoder
    ├── feature_names.joblib                  # Feature order for inference
    └── model_comparison.csv                  # Metrics for all models
```

---

## ⚙️ Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Installation

```bash
git clone https://github.com/ishivamm/Predictive_maintenance_of_milling_machine.git
cd Predictive_maintenance_of_milling_machine
pip install -r requirements.txt
```

### Run the Notebook

Open `Predictive_Maintenance_ML_Pipeline.ipynb` in Jupyter or directly in Google Colab via the badge at the top.

### Run the Streamlit App Locally

```bash
streamlit run app_basic.py
```

---

## 🔮 Inference — Prediction Function

A ready-to-use prediction function is included for integrating the trained model into external systems:

```python
from predict import predict_machine_failure

result = predict_machine_failure(
    air_temp=300.0,           # Kelvin
    process_temp=310.0,       # Kelvin
    rotational_speed=1500,    # rpm
    torque=40.0,              # Nm
    tool_wear=100,            # minutes
    product_type="M"          # 'L', 'M', or 'H'
)

print(result["label"])               # ✅ No Failure  /  ⚠️ FAILURE PREDICTED
print(result["probability_failure"]) # e.g., 0.0821
```

**Returns:** `prediction` (0/1), `label`, `probability_no_failure`, `probability_failure`, `input_features`

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
scipy
joblib
streamlit
shap
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔑 Key Design Decisions

**Why SMOTE?** With only 3.39% failures, a naive model predicts "No Failure" almost always and gets 96%+ accuracy while being useless. SMOTE generates synthetic minority-class samples to force the model to learn genuine failure patterns.

**Why drop individual failure-type columns?** TWF, HDF, PWF, OSF, and RNF are direct sub-indicators of the target variable — using them would be data leakage. They are dropped before training; only the aggregate `Failure_mode_count` is considered.

**Why physics-derived features?** Features like `Power = Torque × ω` mirror real engineering quantities (in watts). This gives tree-based models meaningful split boundaries that align with physical failure thresholds rather than relying solely on raw sensor correlations.

---

## 👤 Author

**Shivam Maurya**
B.Tech Mechanical Engineering, MMMUT Gorakhpur
Data Science & ML Portfolio | [GitHub](https://github.com/ishivamm)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [Stephan Matzka, 2020 — UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- Inspired by real-world IIoT and Industry 4.0 predictive maintenance use cases
