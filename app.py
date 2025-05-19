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
import xgboost

# Load only the model, preprocessor will be implemented inline
model = joblib.load("xgboost_ev_model.pkl")

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

# Define built-in preprocessing function to replace preprocessor.pkl
def preprocess_data(input_data):
    """
    Custom preprocessing function to replace the need for preprocessor.pkl
    
    This function:
    1. Handles categorical features with one-hot encoding
    2. Performs numerical scaling
    3. Returns preprocessed data ready for model prediction
    """
    df = input_data.copy()
    
    # Categorical features
    categorical_features = ['platform', 'facilityType']
    
    # Create one-hot encoding for categorical features
    for feature in categorical_features:
        if feature == 'platform':
            # One-hot encode platform (android, ios, web)
            df[f'{feature}_android'] = (df[feature] == 'android').astype(int)
            df[f'{feature}_ios'] = (df[feature] == 'ios').astype(int)
            df[f'{feature}_web'] = (df[feature] == 'web').astype(int)
        elif feature == 'facilityType':
            # One-hot encode facilityType (1, 2, 3, 4)
            df[f'{feature}_1'] = (df[feature] == 1).astype(int)
            df[f'{feature}_2'] = (df[feature] == 2).astype(int)
            df[f'{feature}_3'] = (df[feature] == 3).astype(int)
            df[f'{feature}_4'] = (df[feature] == 4).astype(int)
    
    # Numerical features to scale
    numerical_features = ['stationId', 'distance', 'startHour', 'is_peak_hour', 
                         'is_weekend', 'startMonth', 'season', 'charging_speed']
    
    # Feature scaling for numerical features (StandardScaler equivalent)
    # Define scaling parameters (mean and std) based on training data
    # Note: These values should be the same as in your original preprocessor
    scaling_params = {
        'stationId': {'mean': 50, 'std': 25},  # Example values - replace with actual values
        'distance': {'mean': 3.5, 'std': 2.8},  # from your training data
        'startHour': {'mean': 12, 'std': 6},
        'is_peak_hour': {'mean': 0.5, 'std': 0.5},
        'is_weekend': {'mean': 0.3, 'std': 0.45},
        'startMonth': {'mean': 6.5, 'std': 3.5},
        'season': {'mean': 2.5, 'std': 1.12},
        'charging_speed': {'mean': 2.04, 'std': 0.85}
    }
    
    # Apply scaling
    for feature in numerical_features:
        if feature in scaling_params:
            mean = scaling_params[feature]['mean']
            std = scaling_params[feature]['std']
            df[feature] = (df[feature] - mean) / std
    
    # Drop original categorical columns and create final feature set
    df = df.drop(['platform', 'facilityType'], axis=1)
    
    # Make sure columns are in the same order as expected by the model
    expected_columns = ['stationId', 'distance', 'startHour', 'is_peak_hour', 
                        'is_weekend', 'startMonth', 'season', 'charging_speed',
                        'platform_android', 'platform_ios', 'platform_web',
                        'facilityType_1', 'facilityType_2', 'facilityType_3', 'facilityType_4']
    
    # Ensure all expected columns exist (add missing ones with zeros)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
            
    # Return preprocessed data with columns in correct order
    return df[expected_columns].values

with tab1:
    st.markdown("## EV Charging Prediction")
    with st.container():
        col1, col2 = st.columns([1.2, 1])
        input_data = {}

with col1:
    st.subheader("Input Parameters")  
    stationId = st.number_input("Station ID", min_value=0, step=1)
    distance = st.number_input("Distance (km)", min_value=0.0, format="%.4f")
    platform = st.selectbox("Platform", ["android", "ios", "web"])
    facilityType = st.selectbox("Facility Type", [1, 2, 3, 4])
    startHour = st.number_input("Start Hour (0-23)", min_value=0, max_value=23)
    startMonth = st.number_input("Start Month (1-12)", min_value=1, max_value=12)

    # Engineered features
    is_peak_hour = 1 if startHour in [7,8,9,17,18,19,20] else 0
    # Here you need day info to check weekend; assuming user provides weekday or date
    weekday = st.selectbox("Weekday", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    is_weekend = 1 if weekday in ['Saturday', 'Sunday'] else 0

    season = (
        1 if startMonth in [12,1,2] else
        2 if startMonth in [3,4,5] else
        3 if startMonth in [6,7,8] else 4
    )

    charging_speed = 5.809629 / (2.841488 + 1e-6)
    
with col2:
    st.subheader("Prediction Results")
    if st.button("🔍 Predict Now", use_container_width=True):
        
        # Base features 
        input_data = pd.DataFrame([{
            'stationId': stationId,
            'distance': distance,
            'platform': platform,
            'facilityType': facilityType,
            'startHour': startHour,
            'is_peak_hour': is_peak_hour,
            'is_weekend': is_weekend,
            'startMonth': startMonth,
            'season': season,
            'charging_speed': charging_speed
        }])

        # Preprocess using our custom function
        input_processed = preprocess_data(input_data)

        # Predict
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
