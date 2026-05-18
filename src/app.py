import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("src/random_forest_model.pkl")

model = load_model()

# 2. App Title & Description
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 California House Price Predictor")
st.write("Enter the details of the neighborhood to estimate the median house value.")

st.markdown("---")

# 3. Create Input Fields for Users
st.header("📍 Location & Geography")
col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input("Longitude", value=-122.23, step=0.01, help="California ranges from roughly -124 to -114")
with col2:
    latitude = st.number_input("Latitude", value=37.88, step=0.01, help="California ranges from roughly 32 to 42")

st.header("📊 Property Details")
col3, col4, col5 = st.columns(3)
with col3:
    housing_median_age = st.slider("Median House Age (Years)", min_value=1, max_value=52, value=25)
with col4:
    total_rooms = st.number_input("Total Rooms in Block", min_value=1, value=2000, step=50)
with col5:
    total_bedrooms = st.number_input("Total Bedrooms in Block", min_value=1, value=400, step=10)

col6, col7, col8 = st.columns(3)
with col6:
    population = st.number_input("Population in Block", min_value=1, value=1000, step=50)
with col7:
    households = st.number_input("Households in Block", min_value=1, value=300, step=10)
with col8:
    median_income = st.number_input("Median Income (in tens of thousands $)", min_value=0.5, max_value=15.0, value=4.0, step=0.1)

st.header("🌊 Proximity to Ocean")
ocean_proximity = st.selectbox(
    "Select Ocean Proximity Group:",
    ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
)

st.markdown("---")

# 4. Make Prediction when Button is Clicked
if st.button("🔮 Predict House Value", type="primary"):
    # Put input data into a DataFrame with the exact same column names as training
    input_data = pd.DataFrame([{
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }])
    
    # Predict using the saved model pipeline
    prediction = model.predict(input_data)[0]
    
    # Display the result beautifully
    st.balloons()
    st.success(f"### 🎯 Estimated Median House Value: ${prediction:,.2f}")