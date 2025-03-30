import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained XGBoost model
model_filename = "xgboost_ev_model.pkl"
model = joblib.load(model_filename)

# Define preprocessing pipeline
categorical_cols = ['platform', 'facilityType', 'season']
numeric_cols = ['stationId', 'distance', 'startHour', 'is_peak_hour', 'is_weekend', 'startMonth', 'charging_speed']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Dummy data for fitting preprocessor
dummy_data = pd.DataFrame({
    'stationId': [0], 'distance': [0.0], 'platform': ['android'],
    'facilityType': [1], 'startHour': [0], 'is_peak_hour': [0],
    'is_weekend': [0], 'startMonth': [1], 'season': [1], 'charging_speed': [1.0]
})
preprocessor.fit(dummy_data)

# Set Streamlit Page Config
st.set_page_config(page_title="EV Charging AI 🚀", layout="wide")

st.markdown("""
    <style>
        body { font-family: 'Arial', sans-serif; color: #ffffff; background-color: #0e1117; }
        .stApp { background: linear-gradient(135deg, #1f1f1f, #2c2c2c); padding: 2rem; }
        .title { font-size: 3rem; font-weight: 700; text-align: center; color: #61dafb; }
        .subtitle { font-size: 1.2rem; text-align: center; color: #a0a0a0; }
        .prediction-box { background: #222; padding: 20px; border-radius: 15px; font-size: 1.5rem;
                         text-align: center; font-weight: bold; color: #61dafb; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 class='title'>⚡ EV Charging Prediction AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered predictions for Electric Vehicle charging analysis</p>", unsafe_allow_html=True)
st.write("---")

# Geolocation Feature
def get_user_location():
    geolocator = Nominatim(user_agent="ev_app")
    location = geolocator.geocode("Your City")  # Replace with dynamic detection if needed
    return location.latitude, location.longitude

lat, lon = get_user_location()

# Layout: Split into Two Columns
col1, col2 = st.columns(2)

# User Inputs
input_data = {}
with col1:
    st.subheader("🔢 Enter Feature Values")
    input_data['stationId'] = st.number_input("Station ID", value=0, step=1)
    input_data['distance'] = st.number_input("Distance (km)", value=0.0, format="%.4f")
    input_data['platform'] = st.selectbox("Platform", ["android", "ios", "web"])
    input_data['facilityType'] = st.selectbox("Facility Type", [1, 2, 3, 4])
    start_time = st.time_input("Start Time")
    start_date = st.date_input("Start Date")

    # Compute derived features
    input_data['startHour'] = start_time.hour
    input_data['startMonth'] = start_date.month
    input_data['is_peak_hour'] = 1 if input_data['startHour'] in [7, 8, 9, 17, 18, 19, 20] else 0
    input_data['is_weekend'] = 1 if start_date.weekday() >= 5 else 0
    input_data['season'] = 1 if input_data['startMonth'] in [12, 1, 2] else 2 if input_data['startMonth'] in [3, 4, 5] else 3 if input_data['startMonth'] in [6, 7, 8] else 4
    input_data['charging_speed'] = 5.809629 / (2.841488 + 1e-6)

# Make Predictions
with col2:
    st.subheader("🎯 Prediction Output")
    if st.button("🚀 Predict Now"):
        input_df = pd.DataFrame([input_data])
        input_processed = preprocessor.transform(input_df)
        predictions = model.predict(input_processed)
        kwh_total_pred, charge_time_hrs_pred = predictions[0]
        
        # Display Predictions
        st.markdown(f"<div class='prediction-box'>🔋 Predicted kWh Total: {kwh_total_pred:.4f} kWh</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-box'>⏳ Predicted Charge Time: {charge_time_hrs_pred:.4f} hrs</div>", unsafe_allow_html=True)

# Interactive Data Visualization
st.subheader("📊 Charging Trends")
dummy_chart_data = pd.DataFrame({
    'Time': pd.date_range(start='1/1/2024', periods=24, freq='H'),
    'Charging Demand': np.random.randint(1, 100, 24)
})
fig = px.line(dummy_chart_data, x='Time', y='Charging Demand', title='Hourly Charging Demand')
st.plotly_chart(fig)

# Map Visualization
st.subheader("📍 Nearby Charging Stations")
m = folium.Map(location=[lat, lon], zoom_start=12)
folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color='blue')).add_to(m)
folium_static(m)

st.write("---")
st.markdown("<p style='text-align: center; color: #a0a0a0;'>🚀 Built with ❤️ using Streamlit | AI-Powered ⚡</p>", unsafe_allow_html=True)
