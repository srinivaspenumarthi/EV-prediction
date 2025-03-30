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

# Load dataset for statistical imputation


# Streamlit UI
st.title("XGBoost Prediction App")
st.write("Enter input features to get a prediction")

# Create input fields
def user_input():
    station_id = st.number_input("Station ID", value=0)
    distance = st.number_input("Distance (km)", value=0.0)
    platform = st.selectbox("Platform", ["android", "ios", "web"])
    facility_type = st.selectbox("Facility Type", [1, 2, 3, 4])
    start_time = st.time_input("Start Time")
    start_date = st.date_input("Start Date")
    
    # Compute derived features
    start_hour = start_time.hour
    start_month = start_date.month
    is_peak_hour = 1 if start_hour in [7, 8, 9, 17, 18, 19, 20] else 0
    is_weekend = 1 if start_date.weekday() >= 5 else 0
    season = 1 if start_month in [12, 1, 2] else 2 if start_month in [3, 4, 5] else 3 if start_month in [6, 7, 8] else 4
    
    # Compute charging speed using median if missing
    charging_speed = 5.809629 / (2.841488 + 1e-6)
    
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
    input_processed = preprocessor.transform(input_df)
    
    # Get predictions for both outputs
    predictions = model.predict(input_processed)
    kwh_total_pred, charge_time_hrs_pred = predictions[0]
    
    st.success(f"Predicted kWh Total: {kwh_total_pred}")
    st.success(f"Predicted Charge Time (hrs): {charge_time_hrs_pred}")

if __name__ == "__main__":
    st.write("App is running...")
