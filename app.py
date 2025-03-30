import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained XGBoost model
with open("xgboost_ev_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define preprocessing pipeline
categorical_cols = ['platform', 'facilityType', 'season']
numeric_cols = ['stationId', 'distance', 'startHour', 'is_peak_hour', 'is_weekend', 'startMonth', 'charging_speed']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Streamlit UI
st.title("XGBoost Prediction App")
st.write("Enter input features to get a prediction")

# Create input fields
def user_input():
    station_id = st.number_input("Station ID", value=0)
    distance = st.number_input("Distance (km)", value=0.0)
    platform = st.selectbox("Platform", ["A", "B", "C"])
    facility_type = st.selectbox("Facility Type", ["Parking", "Charging Hub", "Other"])
    start_hour = st.number_input("Start Hour", min_value=0, max_value=23, value=12)
    is_peak_hour = st.selectbox("Is Peak Hour", [0, 1])
    is_weekend = st.selectbox("Is Weekend", [0, 1])
    start_month = st.number_input("Start Month", min_value=1, max_value=12, value=1)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    charging_speed = st.number_input("Charging Speed (kW)", value=0.0)
    
    return pd.DataFrame({
        "stationId": [station_id],
        "distance": [distance],
        "platform": [platform],
        "facilityType": [facility_type],
        "startHour": [start_hour],
        "is_peak_hour": [is_peak_hour],
        "is_weekend": [is_weekend],
        "startMonth": [start_month],
        "season": [season],
        "charging_speed": [charging_speed]
    })

# Get input data
input_df = user_input()

# Button to get prediction
if st.button("Predict"):
    st.write("Fetching prediction...")
    
    # Preprocess input
    input_processed = preprocessor.fit_transform(input_df)
    
    # Get predictions for both outputs
    predictions = model.predict(input_processed)
    kwh_total, charge_time_hrs = predictions[0]
    
    st.success(f"Predicted kWh Total: {kwh_total}")
    st.success(f"Predicted Charge Time (hrs): {charge_time_hrs}")

if __name__ == "__main__":
    st.write("App is running...")
