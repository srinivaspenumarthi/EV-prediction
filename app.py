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
from streamlit_lottie import st_lottie
import time
import json


# Load model
model = joblib.load("xgboost_ev_model.pkl")

# Define preprocessor
categorical_cols = ['platform', 'facilityType', 'season']
numeric_cols = ['stationId', 'distance', 'startHour', 'is_peak_hour', 'is_weekend', 'startMonth', 'charging_speed']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Dummy fit
dummy_data = pd.DataFrame({
    'stationId': [0], 'distance': [0.0], 'platform': ['android'],
    'facilityType': [1], 'startHour': [0], 'is_peak_hour': [0],
    'is_weekend': [0], 'startMonth': [1], 'season': [1], 'charging_speed': [1.0]
})
preprocessor.fit(dummy_data)

# Config
st.set_page_config(page_title="EV Prediction", layout="wide")

# Initialize session state variables for location refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'refresh_location' not in st.session_state:
    st.session_state.refresh_location = False
if 'lat' not in st.session_state:
    st.session_state.lat = None
if 'lon' not in st.session_state:
    st.session_state.lon = None

# Lottie loader
@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the uploaded animation JSON
lottie_json = load_lottiefile("Animation - 1746651823424.json")

# Center layout using 3 columns
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    if lottie_json:
        st_lottie(lottie_json, height=350, speed=1.5)
    st.markdown("<h1 style='margin-top: -30px;'>EV Charging System</h1>", unsafe_allow_html=True)


# Tabs
with st.container():
    tab1, tab2 = st.tabs(["⚡ Prediction", "📍 Location & Maps"])

with tab1:
    st.markdown("## EV Charging Prediction")
    with st.container():
        col1, col2 = st.columns([1.2, 1])
        input_data = {}

        with col1:
            st.subheader("Input Parameters")
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
            input_data['season'] = (
                1 if input_data['startMonth'] in [12, 1, 2] else
                2 if input_data['startMonth'] in [3, 4, 5] else
                3 if input_data['startMonth'] in [6, 7, 8] else 4
            )
            input_data['charging_speed'] = 5.809629 / (2.841488 + 1e-6)

        with col2:
            st.subheader("Prediction Results")
            if st.button("🔍 Predict Now", use_container_width=True):
                input_df = pd.DataFrame([input_data])
                input_processed = preprocessor.transform(input_df)
                predictions = model.predict(input_processed)
                kwh_total_pred, charge_time_hrs_pred = predictions[0]

                st.success(f"⚡ Predicted kWh Total: {kwh_total_pred:.4f} kWh")
                st.success(f"⏱️ Predicted Charge Time: {charge_time_hrs_pred:.4f} hrs")

with tab2:
    st.markdown("## Location & Nearby Stations")
    
    # Location refresh functions
    def refresh_location():
        st.session_state.refresh_location = True
        st.session_state.last_refresh = time.time()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        city_name = st.text_input("Enter city name (optional):")
    with col2:
        st.button("🔄 Refresh Location", on_click=refresh_location, use_container_width=True)

    def get_coordinates(city):
        url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key=3518ba3e6620418cab166e34afd6ad4e"
        res = requests.get(url).json()
        if res and 'results' in res and len(res['results']) > 0:
            return float(res['results'][0]['geometry']['lat']), float(res['results'][0]['geometry']['lng'])
        return None, None

    def get_nearby_ev_stations(lat, lon):
        url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={lat}&longitude={lon}&maxresults=5&key=a1f5b87f-3209-4eb2-afc1-c9d379acfa10"
        return requests.get(url).json()

    def get_directions(start_lat, start_lon, end_lat, end_lon):
        url = f"https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": "5b3ce3597851110001cf62483ee31511b8e745d0b635b37fbaeb4f57"}
        params = {
            "start": f"{start_lon},{start_lat}",
            "end": f"{end_lon},{end_lat}"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()['features'][0]['geometry']['coordinates']
        return []

    # Location logic
    lat, lon = None, None
    
    if city_name:
        lat, lon = get_coordinates(city_name)
        if lat and lon:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.success(f"📍 Set to {city_name}: ({lat}, {lon})")
    
    # Use get_geolocation if needed
    if st.session_state.refresh_location or not st.session_state.lat or not st.session_state.lon:
        location_placeholder = st.empty()
        with location_placeholder:
            st.info("📡 Getting your location...")
        
        location = get_geolocation()
        if location and 'coords' in location:
            lat = location['coords'].get('latitude')
            lon = location['coords'].get('longitude')
            if lat and lon:
                st.session_state.lat = lat
                st.session_state.lon = lon
                location_placeholder.success(f"📍 Your Location: ({lat}, {lon})")
                st.session_state.refresh_location = False
    else:
        lat = st.session_state.lat
        lon = st.session_state.lon

    # Show map and stations if we have location
    if lat and lon:
        stations = get_nearby_ev_stations(lat, lon)
        if stations:
            station_options = [s['AddressInfo']['Title'] for s in stations]
            selected_station = st.selectbox("Select a station to show route:", station_options)
            selected_info = next((s for s in stations if s['AddressInfo']['Title'] == selected_station), stations[0])
            end_lat = selected_info['AddressInfo']['Latitude']
            end_lon = selected_info['AddressInfo']['Longitude']

            df = pd.DataFrame([{
                "Name": s['AddressInfo']['Title'],
                "Distance (km)": round(geodesic((lat, lon), (s['AddressInfo']['Latitude'], s['AddressInfo']['Longitude'])).km, 2),
                "Status": "Available" if s.get('StatusType', {}).get('IsOperational', True) else "Unavailable"
            } for s in stations])
            st.dataframe(df, use_container_width=True)

            m = folium.Map(location=[lat, lon], zoom_start=13)
            folium.Marker([lat, lon], popup="You are here", 
                         icon=folium.Icon(color="blue", icon="user", prefix='fa')).add_to(m)

            for s in stations:
                color = "green" if s.get('StatusType', {}).get('IsOperational', True) else "red"
                folium.Marker(
                    [s['AddressInfo']['Latitude'], s['AddressInfo']['Longitude']],
                    popup=s['AddressInfo']['Title'],
                    icon=folium.Icon(color=color, icon="bolt", prefix='fa')
                ).add_to(m)

            route_coords = get_directions(lat, lon, end_lat, end_lon)
            if route_coords:
                folium.PolyLine(locations=[(coord[1], coord[0]) for coord in route_coords], 
                               color='#0B3D91', weight=4, opacity=0.8).add_to(m)

            folium_static(m)
        else:
            st.warning("⚠️ No charging stations found nearby. Try a different location.")
    else:
        st.warning("📡 Location not available. Please enter a city name or allow location access.")

# Adaptive CSS for both light and dark modes
st.markdown("""
<style>
/* Base Styles */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Card Styling */
.card {
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #00C6FF, #0072FF);
    color: white !important;
    padding: 0.75em 2em;
    border: none !important;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 114, 255, 0.4);
    filter: brightness(110%);
}

/* Data Elements */
.dataframe {
    border: none !important;
}
.dataframe th {
    background: #0072FF !important;
    color: white !important;
    font-weight: 600 !important;
}
.dataframe td {
    font-size: 14px !important;
}

/* Alert Styling */
.stAlert {
    border-radius: 10px !important;
}
.st-success {
    background-color: #00FFA3 !important;
    color: #003B2C !important;
    border: none !important;
}
.st-info {
    background-color: #00C6FF !important;
    color: #002847 !important;
    border: none !important;
}
.st-warning {
    background-color: #FFD166 !important;
    color: #5E4200 !important;
    border: none !important;
}
.st-error {
    background-color: #FF6B6B !important;
    color: #5E0000 !important;
    border: none !important;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 24px;
    border-radius: 4px 4px 0px 0px;
    font-weight: 600;
}

/* Input Fields */
input, select, textarea {
    border-radius: 8px !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .stButton > button {
        width: 100%;
        padding: 0.6em;
    }
}

/* General improvements */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)
