# 🛡️ UPI Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A full-stack Machine Learning web app that detects fraudulent UPI transactions in real time — **no database, no complex setup, just 3 commands.**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏠 Dashboard | Live charts — fraud rate, amounts, locations, hour of day |
| 🔮 Predictor | Enter any transaction → get fraud risk score 0–100% |
| 📊 Data Explorer | Filter and browse transactions interactively |
| 📁 CSV Upload | Upload your own dataset for bulk fraud detection |
| 🤖 Dual Models | Random Forest + XGBoost compared side by side |
| 🎲 Data Generator | Generate synthetic UPI data — no CSV needed |
| 🌑 Dark Mode | Fully styled dark theme UI |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/upi-fraud-detection.git
cd upi-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

That's it. The app auto-generates data and trains models on first run.

---

## 🗂️ Project Structure

```
upi-fraud-detection/
├── app.py                        ← Dashboard (entry point)
├── pages/
│   ├── 1_🔮_Predict.py           ← Single transaction predictor
│   ├── 2_📊_Data_Explorer.py     ← Filter & browse data
│   └── 3_📁_Upload_CSV.py        ← Bulk CSV analysis
├── src/
│   ├── data_generator.py         ← Synthetic data generator
│   ├── train_model.py            ← RF + XGBoost training
│   └── utils.py                  ← Shared helpers
├── models/                       ← Saved .pkl model files
├── data/                         ← Auto-generated CSV
├── notebooks/
│   └── UPI_Fraud_Detection.ipynb ← Google Colab version
├── .streamlit/config.toml        ← Dark theme
└── requirements.txt
```

---

## 🤖 Models

| Model | Algorithm | Tuning |
|-------|-----------|--------|
| Random Forest | Ensemble (100 trees) | `class_weight='balanced'` |
| XGBoost | Gradient Boosting | `scale_pos_weight` for imbalance |

**Features used:**
- Transaction amount, Hour of day, Location, Transaction type
- Sender/Receiver bank, New device flag, Failed PIN attempts
- Engineered: `is_night`, `is_high_amount`, `is_unknown_location`

---

## 📊 Risk Score Logic

| Score | Level | Verdict |
|-------|-------|---------|
| 0–30% | 🟢 LOW | Legitimate |
| 30–60% | 🟡 MEDIUM | Review |
| 60–100% | 🔴 HIGH | Fraud |

---

## 🧪 Google Colab

Open the notebook for a zero-install version:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** — web UI
- **Scikit-learn** — Random Forest
- **XGBoost** — gradient boosting
- **Plotly** — interactive charts
- **Pandas / NumPy** — data processing
- **Joblib** — model serialisation

---

## 👨‍💻 Author

**Prince** — Big Data Capstone Project
"# upi-fraud-detection" 
