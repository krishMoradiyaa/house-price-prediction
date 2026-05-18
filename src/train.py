import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. LOAD THE DATA
# ==========================================
print("📦 Loading dataset...")
df = pd.read_csv("data/housing.csv")

# Separate features (X) and target variable (y)
# We want to predict 'median_house_value'
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. DATA PREPROCESSING (CLEANING)
# ==========================================
# Group columns by type
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Pipeline for numerical data: Fill missing values with the median
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Pipeline for categorical data: Convert text categories to numbers (One-Hot Encoding)
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both cleaning processes into a single preprocessor tool
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# ==========================================
# 3. TRAIN AND EVALUATE A BASELINE MODEL (Linear Regression)
# ==========================================
print("🤖 Training Linear Regression model...")
lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print(f"   Linear Regression MAE: ${lr_mae:,.2f}")
print(f"   Linear Regression R² Score: {lr_r2:.4f}\n")

# ==========================================
# 4. TRAIN AND EVALUATE AN ADVANCED MODEL (Random Forest)
# ==========================================
print("🌲 Training Random Forest Regressor (This might take a minute)...")
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print(f"   Random Forest MAE: ${rf_mae:,.2f}")
print(f"   Random Forest R² Score: {rf_r2:.4f}\n")

print("✅ Model training complete!")

# Save the winning Random Forest model to a file
# Save the winning Random Forest model to a file
print("💾 Saving the Random Forest model...")
joblib.dump(rf_model, "src/random_forest_model.pkl")
print("🎉 Model saved successfully as 'src/random_forest_model.pkl'!")