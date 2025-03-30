import streamlit as st import joblib import numpy as np import time from sklearn.ensemble import RandomForestRegressor

Load the trained model

model = joblib.load("best_random_forest.pkl")

Streamlit app design

st.set_page_config(page_title="AI Power Predictor", page_icon="🌟", layout="wide")

Custom CSS for ultra-premium look

st.markdown( """ <style> body { background-color: #0e1117; color: #fff; font-family: 'Arial', sans-serif; } .big-font { font-size: 36px !important; font-weight: bold; text-align: center; margin-bottom: 20px; } .stTextInput, .stNumberInput, .stButton > button { border-radius: 20px; padding: 10px; font-size: 18px; } .stButton > button { background: linear-gradient(45deg, #ff416c, #ff4b2b); color: white; transition: all 0.3s ease; } .stButton > button:hover { transform: scale(1.1); } </style> """, unsafe_allow_html=True )

App title

st.markdown("<p class='big-font'>⚡ AI Power Consumption Predictor ⚡</p>", unsafe_allow_html=True)

Input Fields (31 Features)

st.sidebar.header("🔢 Enter Features") features = [] feature_names = ['sessionId', 'dollars', 'startTime', 'endTime', 'chargeTimeHrs', 'distance', 'userId', 'stationId', 'locationId', 'managerVehicle', 'facilityType', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun', 'reportedZip', 'start_hour', 'end_hour', 'day_of_week', 'month', 'kwhTotal_lag_1', 'kwhTotal_lag_2', 'kwhTotal_lag_3', 'distance_x_chargeTimeHrs', 'day_of_week_x_start_hour', 'kwhTotal_x_kwhTotal_lag_1', 'kwhTotal_x_kwhTotal_lag_2', 'kwhTotal_x_kwhTotal_lag_3']

for feature in feature_names: value = st.sidebar.number_input(f"{feature}", value=0.0, step=0.01, format="%.2f") features.append(value)

Prediction Button

if st.sidebar.button("🚀 Predict", key="predict"): with st.spinner("Predicting..."): time.sleep(2)  # Simulate loading time prediction = model.predict([features]) st.success(f"⚡ Predicted Power Consumption: {prediction[0]:.4f} kWh")

