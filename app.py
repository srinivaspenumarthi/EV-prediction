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
import polyline

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

# Page config
st.set_page_config(page_title="EV Charging AI", layout="wide")

# Load Lottie animation
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_json = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_3rwasyjy.json")
if lottie_json:
    st_lottie(lottie_json, height=220, speed=1.2)

# Tabs
with st.container():
    tab1, tab2 = st.tabs([" Prediction", " Location & Maps"])

with tab1:
    st.markdown("### EV Charging Demand Prediction")
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
        if st.button("Predict Now"):
            input_df = pd.DataFrame([input_data])
            input_processed = preprocessor.transform(input_df)
            predictions = model.predict(input_processed)
            kwh_total_pred, charge_time_hrs_pred = predictions[0]

            st.success(f"Predicted kWh Total: {kwh_total_pred:.4f} kWh")
            st.success(f"Predicted Charge Time: {charge_time_hrs_pred:.4f} hrs")

with tab2:
    st.markdown("### Location & Nearby Stations")
    city_name = st.text_input("Enter city name (optional):")

    def get_coordinates(city):
        url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key=3518ba3e6620418cab166e34afd6ad4e"
        res = requests.get(url).json()
        if res and 'results' in res and len(res['results']) > 0:
            return float(res['results'][0]['geometry']['lat']), float(res['results'][0]['geometry']['lng'])
        return None, None

    def get_nearby_ev_stations(lat, lon):
        url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={lat}&longitude={lon}&maxresults=5&key=a1f5b87f-3209-4eb2-afc1-c9d379acfa10"
        return requests.get(url).json()

    def get_route(start_lat, start_lon, end_lat, end_lon):
        url = f"https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {
            'Authorization': '5b3ce3597851110001cf62483ee31511b8e745d0b635b37fbaeb4f57',
            'Content-Type': 'application/json'
        }
        payload = {
            "coordinates": [[start_lon, start_lat], [end_lon, end_lat]]
        }
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 200:
            data = res.json()
            geometry = data['routes'][0]['geometry']
            return polyline.decode(geometry)
        return None

    lat, lon = None, None
    if city_name:
        lat, lon = get_coordinates(city_name)
        if lat and lon:
            st.success(f"Set to {city_name}: ({lat}, {lon})")
    if not lat or not lon:
        location = get_geolocation()
        if location and 'coords' in location:
            lat = location['coords'].get('latitude')
            lon = location['coords'].get('longitude')
            if lat and lon:
                st.success(f"Your Location: ({lat}, {lon})")

    if lat and lon:
        stations = get_nearby_ev_stations(lat, lon)
        if stations:
            station_names = [s['AddressInfo']['Title'] for s in stations]
            selected_station_name = st.selectbox("Select a Station for Directions", station_names)

            selected_station = next(s for s in stations if s['AddressInfo']['Title'] == selected_station_name)
            end_lat = selected_station['AddressInfo']['Latitude']
            end_lon = selected_station['AddressInfo']['Longitude']

            df = pd.DataFrame([{
                "Name": s['AddressInfo']['Title'],
                "Distance (km)": geodesic((lat, lon), (s['AddressInfo']['Latitude'], s['AddressInfo']['Longitude'])).km
            } for s in stations])
            st.dataframe(df)

            m = folium.Map(location=[lat, lon], zoom_start=13)
            folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color="blue")).add_to(m)
            for s in stations:
                folium.Marker(
                    [s['AddressInfo']['Latitude'], s['AddressInfo']['Longitude']],
                    popup=s['AddressInfo']['Title'],
                    icon=folium.Icon(color="green", icon="bolt", prefix='fa')
                ).add_to(m)

            route_coords = get_route(lat, lon, end_lat, end_lon)
            if route_coords:
                folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8).add_to(m)

            folium_static(m)

# Premium CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #f9fafc;
    color: #222;
}
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif;
    color: #0a2540;
    font-weight: 700;
}
.stButton > button {
    background: linear-gradient(to right, #667eea, #764ba2);
    color: white;
    padding: 0.75em 2em;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 16px;
    box-shadow: 0 4px 14px rgba(118, 75, 162, 0.4);
}
.stButton > button:hover {
    background: linear-gradient(to right, #5a67d8, #6b46c1);
    transform: translateY(-1px);
}
.css-1kyxreq, .stColumn {
    background: white;
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
thead {
    background: #eef1f7;
}
tbody tr:hover {
    background: #f5f7fa;
}
.stAlert {
    background-color: #ebf8ff;
    border-left: 5px solid #3182ce;
    color: #2c5282;
    padding: 1rem;
    border-radius: 10px;
}
.block-container {
    padding: 3rem 2rem;
}
</style>
""", unsafe_allow_html=True)
