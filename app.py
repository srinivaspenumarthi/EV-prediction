import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from streamlit_js_eval import get_geolocation
import folium
from streamlit_folium import folium_static

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

# Fit preprocessor with dummy data to prevent errors
dummy_data = pd.DataFrame({
    'stationId': [0], 'distance': [0.0], 'platform': ['android'],
    'facilityType': [1], 'startHour': [0], 'is_peak_hour': [0],
    'is_weekend': [0], 'startMonth': [1], 'season': [1], 'charging_speed': [1.0]
})
preprocessor.fit(dummy_data)

# Custom Styling
st.set_page_config(page_title="EV Charging AI 🚀", layout="wide")

st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            color: #ffffff;
            background-color: #0e1117;
        }
        .stApp {
            background: linear-gradient(135deg, #1f1f1f, #2c2c2c);
            padding: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 class='title'>⚡ EV Charging Prediction AI</h1>", unsafe_allow_html=True)
st.write("---")

# Tabs for Prediction and Charging Stations
tab1, tab2 = st.tabs(["Prediction", "Find Stations"])

# Prediction Tab
with tab1:
    col1, col2 = st.columns([1, 1])
    input_data = {}
    with col1:
        input_data['stationId'] = st.number_input("Station ID", value=0, step=1)
        input_data['distance'] = st.number_input("Distance (km)", value=0.0, format="%.4f")
        input_data['platform'] = st.selectbox("Platform", ["android", "ios", "web"])
        input_data['facilityType'] = st.selectbox("Facility Type", [1, 2, 3, 4])
        start_time = st.time_input("Start Time")
        start_date = st.date_input("Start Date")
        input_data['startHour'] = start_time.hour
        input_data['startMonth'] = start_date.month
        input_data['is_peak_hour'] = 1 if input_data['startHour'] in [7, 8, 9, 17, 18, 19, 20] else 0
        input_data['is_weekend'] = 1 if start_date.weekday() >= 5 else 0
        input_data['season'] = 1 if input_data['startMonth'] in [12, 1, 2] else 2 if input_data['startMonth'] in [3, 4, 5] else 3 if input_data['startMonth'] in [6, 7, 8] else 4
        input_data['charging_speed'] = 5.809629 / (2.841488 + 1e-6)
    with col2:
        if st.button("🚀 Predict Now"):
            input_df = pd.DataFrame([input_data])
            input_processed = preprocessor.transform(input_df)
            predictions = model.predict(input_processed)
            kwh_total_pred, charge_time_hrs_pred = predictions[0]
            st.write(f"Predicted kWh Total: {kwh_total_pred:.4f} kWh")
            st.write(f"Predicted Charge Time: {charge_time_hrs_pred:.4f} hrs")

# Map Tab - for displaying nearby charging stations
with tab2:
    city_name = st.text_input("Enter a city or address:", placeholder="e.g., New York, NY")
    
    def get_coordinates(city):
        url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key=4cda5084fabf428aa8e6564d16b7ad8c"
        response = requests.get(url).json()
        if response and 'results' in response and len(response['results']) > 0:
            return float(response['results'][0]['geometry']['lat']), float(response['results'][0]['geometry']['lng'])
        return None, None
    
    def get_nearby_ev_stations(lat, lon):
        api_url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={lat}&longitude={lon}&maxresults=10&key=a1f5b87f-3209-4eb2-afc1-c9d379acfa10"
        response = requests.get(api_url).json()
        return response
    
    lat, lon = None, None
    if city_name:
        lat, lon = get_coordinates(city_name)
    
    if lat is None or lon is None:
        if st.button("📱 Use My Location"):
            location = get_geolocation()
            if location and 'coords' in location:
                lat, lon = location['coords'].get('latitude'), location['coords'].get('longitude')
    
    if lat and lon:
        m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
        folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color="blue", icon="user", prefix="fa")).add_to(m)
        
        stations = get_nearby_ev_stations(lat, lon)
        if stations:
            for station in stations:
                station_name = station.get('AddressInfo', {}).get('Title', 'Unknown Station')
                station_lat = station.get('AddressInfo', {}).get('Latitude')
                station_lon = station.get('AddressInfo', {}).get('Longitude')
                if station_lat and station_lon:
                    folium.Marker([station_lat, station_lon], popup=station_name, icon=folium.Icon(color="green", icon="bolt", prefix="fa")).add_to(m)
        folium_static(m, width=1000, height=500)
    else:
        st.info("Enter a location or use your current position to see charging stations on the map.")
