# 🏠 End-to-End California House Price Prediction App

This repository contains a production-ready Machine Learning project that predicts median house values in California neighborhoods using an advanced regression pipeline. It includes data cleaning, automated feature transformation, model evaluation, and an interactive web application built with Streamlit.

---

## 🚀 Live Interactive Dashboard
The project features a real-time web interface where users can adjust geographical location, property specifications, and neighborhood demographics to see instant valuation estimates.

---

## 📊 Performance & Model Evaluation
We trained and evaluated a baseline model against an advanced ensemble learning model to ensure high prediction accuracy:

| Model | Mean Absolute Error (MAE) | R² Score (Variance Explained) |
| :--- | :---: | :---: |
| **Linear Regression (Baseline)** | \$50,670.49 | 62.54% |
| **Random Forest Regressor** | **\$31,636.35** | **81.69%** |

*The Random Forest model outperformed the baseline model by explaining 81.69% of the variance and reducing individual prediction error by over \$19,000.*

---

## 🛠️ Project Structure
```text
├── data/              # Raw data files (housing.csv)
├── notebooks/         # Exploratory Data Analysis (exploration.ipynb)
├── src/               # Production source code
│   ├── train.py       # ML Pipeline & model training script
│   ├── app.py         # Streamlit Web Application interface
│   └── random_forest_model.pkl  # Trained, serialized winning model
└── requirements.txt   # Required Python libraries
