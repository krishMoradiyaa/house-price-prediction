# 🏠 End-to-End California House Price Prediction App

This repository contains a production-ready Machine Learning project that predicts California housing prices using an end-to-end regression pipeline with automated preprocessing, feature engineering, and ensemble learning techniques.

The project includes data cleaning, exploratory data analysis, model training, evaluation, and an interactive Streamlit web application for real-time house price predictions.

---

# 🚀 Live Interactive Dashboard

The application features a real-time web interface where users can:

- Select geographical location
- Adjust property specifications
- Configure neighborhood demographics
- Generate instant house price predictions

The dashboard is designed to simulate a real-world property valuation system powered by Machine Learning.

---

# 📊 Model Performance & Evaluation

We trained and evaluated multiple regression models to ensure high prediction accuracy and strong generalization performance.

| Model | Mean Absolute Error (MAE) | R² Score |
|---|---|---|
| Linear Regression (Baseline) | $50,670.49 | 62.54% |
| Random Forest Regressor | $31,636.35 | 81.69% |

The Random Forest model significantly outperformed the baseline model by:

- Explaining over 81% of variance
- Reducing prediction error by approximately 37%
- Improving overall prediction stability and robustness

---

# 🔥 Key Features

- End-to-End Machine Learning Pipeline
- Automated Data Preprocessing
- Exploratory Data Analysis (EDA)
- Ensemble Learning with Random Forest
- Interactive Streamlit Dashboard
- Real-time House Price Prediction
- Production-ready Modular Architecture
- Model Serialization & Reusability

---

# 🧠 Tech Stack

## Programming Language
- Python

## Data Analysis & Visualization
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Machine Learning
- Scikit-learn
- Random Forest Regressor
- Linear Regression

## Web Application
- Streamlit

## Tools & Platforms
- Git
- GitHub
- VS Code

---

# 🛠️ Project Structure

```bash
house-price-prediction/
│
├── data/                          # Raw dataset files
│   └── housing.csv
│
├── notebooks/                     # Exploratory Data Analysis
│   └── exploration.ipynb
│
├── src/                           # Production source code
│   ├── train.py                   # Model training pipeline
│   ├── app.py                     # Streamlit web application
│   └── random_forest_model.pkl    # Serialized trained model
│
├── requirements.txt               # Required dependencies
├── README.md
└── .gitignore
```

---

# ⚙️ How to Run This Project Locally

Follow these steps to set up and run the application on your computer.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/house-price-prediction.git

cd house-price-prediction
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Launch the Streamlit Web Application

```bash
streamlit run src/app.py
```

The application will automatically open in your browser.

---

# 🌐 Live Demo

```bash
streamlit run src/app.py
```

---

# 📈 Example Prediction Workflow

1. User enters housing and demographic details
2. Data preprocessing pipeline transforms inputs
3. Trained Random Forest model generates predictions
4. Streamlit dashboard displays estimated housing price instantly

---

# 🚀 Future Improvements

- Hyperparameter Optimization
- XGBoost & Gradient Boosting Models
- Docker Deployment
- CI/CD Pipeline Integration
- Cloud Deployment (AWS/GCP/Azure)
- User Authentication
- Real-time API Integration
- Advanced Feature Engineering


---

# 📄 License

This project is licensed under the MIT License.
