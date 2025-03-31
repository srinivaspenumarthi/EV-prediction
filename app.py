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
from geopy.distance import geodesic

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

# Tabs for Prediction and Location
tab1, tab2 = st.tabs(["🔢 Prediction", "📍 Location & Maps"])

with tab1:
    st.markdown("##  EV Charging Prediction ")
    st.write("---")
   


    # Layout: Split into Two Columns
    col1, col2 = st.columns(2)
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

    with col2:
        st.subheader("🎯 Prediction Output")
        if st.button("🚀 Predict Now"):
            input_df = pd.DataFrame([input_data])
            input_processed = preprocessor.transform(input_df)
            predictions = model.predict(input_processed)
            kwh_total_pred, charge_time_hrs_pred = predictions[0]
            
            # Display Predictions
            st.success(f"🔋 Predicted kWh Total: {kwh_total_pred:.4f} kWh")
            st.success(f"⏳ Predicted Charge Time: {charge_time_hrs_pred:.4f} hrs")

with tab2:
    st.markdown("## 📍 Location & Nearby Charging Stations")
    city_name = st.text_input("Enter a city name to search (optional):", "")
    
    def get_coordinates(city):
        url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key=4cda5084fabf428aa8e6564d16b7ad8c"
        response = requests.get(url).json()
        if response and 'results' in response and len(response['results']) > 0:
            return float(response['results'][0]['geometry']['lat']), float(response['results'][0]['geometry']['lng'])
        return None, None
    
    def get_nearby_ev_stations(lat, lon):
        api_url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={lat}&longitude={lon}&maxresults=5&key=a1f5b87f-3209-4eb2-afc1-c9d379acfa10"
        response = requests.get(api_url).json()
        return response
    
    lat, lon = None, None
    if city_name:
        lat, lon = get_coordinates(city_name)
        if lat and lon:
            st.success(f"📍 Location set to: {city_name} ({lat}, {lon})")
        else:
            st.warning("⚠️ Could not find the entered city. Using default user location.")
    
    if lat is None or lon is None:
        location = get_geolocation()
        if location and isinstance(location, dict) and 'coords' in location:
            lat, lon = location['coords'].get('latitude'), location['coords'].get('longitude')
            if lat and lon:
                st.success(f"📍 Your Location: {lat}, {lon}")
            else:
                st.warning("⚠️ Unable to retrieve precise location. Check your browser settings.")
    
    if lat and lon:
        stations = get_nearby_ev_stations(lat, lon)
        if stations:
            station_data = pd.DataFrame([
                {"Name": s['AddressInfo']['Title'], 
                 "Distance (km)": geodesic((lat, lon), (s['AddressInfo']['Latitude'], s['AddressInfo']['Longitude'])).km} 
                for s in stations
            ])
            st.table(station_data)
        
        m = folium.Map(location=[lat, lon], zoom_start=12)
        folium.Marker([lat, lon], popup="You are here", icon=folium.Icon(color="blue")).add_to(m)
        for station in stations:
            folium.Marker([station['AddressInfo']['Latitude'], station['AddressInfo']['Longitude']],
                          popup=station['AddressInfo']['Title'],
                          icon=folium.Icon(color="green")).add_to(m)
        folium_static(m)
