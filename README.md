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
```

---

## ⚙️ How to Run This Project Locally
Follow these steps to set up and run the application on your computer:

### 1. Clone the Repository
```bash
git clone https://github.com/krishMoradiyaa/house-price-prediction.git
cd house-price-prediction
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web App
```bash
streamlit run src/app.py
```

This will automatically open the application in your browser.
