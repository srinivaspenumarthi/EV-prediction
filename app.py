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
from datetime import datetime, time

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
st.set_page_config(page_title="EV Charging AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS with improved design
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-color: #4CAF50;
            --secondary-color: #2196F3;
            --bg-color: #0e1117;
            --text-color: #ffffff;
            --card-bg: rgba(30, 30, 30, 0.7);
            --accent-color: #00BFA5;
        }
        
        body {
            font-family: 'Montserrat', sans-serif;
            color: var(--text-color);
            background-color: var(--bg-color);
            background-image: linear-gradient(135deg, #121212 25%, #1E1E1E 25%, #1E1E1E 50%, #121212 50%, #121212 75%, #1E1E1E 75%, #1E1E1E 100%);
            background-size: 20px 20px;
        }
        
        .stApp {
            padding: 1rem;
        }
        
        h1, h2, h3 {
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
        }
        
        .title-container {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        
        .title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #ffffff, #e0e0e0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .subtitle {
            font-size: 1.3rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 400;
        }
        
        .card {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1.5rem;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px rgba(0, 0, 0, 0.3);
        }
        
        .card-title {
            color: var(--primary-color);
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border-bottom: 2px solid rgba(76, 175, 80, 0.3);
            padding-bottom: 0.5rem;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.7rem 2rem;
            border-radius: 30px;
            transition: all 0.3s;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            width: 100%;
            font-size: 1.1rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25);
        }
        
        .stTextInput > div > div > input, .stNumberInput > div > div > input {
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(30, 30, 30, 0.5);
            color: white;
            padding: 0.7rem 1rem;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .stSelectbox > div > div {
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(30, 30, 30, 0.5);
        }
        
        .prediction-box {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(33, 150, 243, 0.2));
            padding: 1.5rem;
            border-radius: 15px;
            font-size: 1.7rem;
            text-align: center;
            font-weight: bold;
            margin-top: 1rem;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .prediction-icon {
            font-size: 2rem;
            color: var(--primary-color);
        }
        
        .metric-container {
            display: flex;
            justify-content: space-between;
            margin-top: 1rem;
        }
        
        .metric-card {
            background: rgba(30, 30, 30, 0.5);
            border-radius: 15px;
            padding: 1rem;
            width: 48%;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent-color);
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.7);
        }
        
        .folium-map {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Improve radio buttons and other input elements */
        .stRadio > div {
            background-color: rgba(30, 30, 30, 0.5);
            border-radius: 10px;
            padding: 10px;
        }
        
        /* Tooltip styling */
        div[data-baseweb="tooltip"] {
            background-color: #2d2d2d !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: var(--primary-color) !important;
        }
        
        /* Animation for loading state */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 1.5s infinite;
        }
    </style>
""", unsafe_allow_html=True)

# App Title in a nice container
st.markdown("""
    <div class="title-container">
        <h1 class="title">⚡ EV Charging Prediction AI</h1>
        <p class="subtitle">Smart predictions for your electric vehicle charging experience</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs for better navigation
tab1, tab2 = st.tabs(["📊 Prediction", "🗺️ Charging Map"])

with tab1:
    # Main content in two columns
    col1, col2 = st.columns([1, 1])
    
    # Input Form
    with col1:
        st.markdown('<div class="card"><h3 class="card-title">Input Parameters</h3>', unsafe_allow_html=True)
        
        # Create a form for better user experience
        with st.form(key='prediction_form'):
            # Station & Location Information
            st.subheader("📍 Station Details")
            input_data = {}
            
            col_a, col_b = st.columns(2)
            with col_a:
                input_data['stationId'] = st.number_input("Station ID", value=1, min_value=1, step=1)
            with col_b:
                input_data['distance'] = st.number_input("Distance (km)", value=2.5, min_value=0.1, max_value=100.0, step=0.1, format="%.1f")
            
            input_data['platform'] = st.selectbox("Platform", ["android", "ios", "web"], index=0)
            facility_options = {1: "Residential", 2: "Commercial", 3: "Public Fast Charger", 4: "Supercharger"}
            facility_choice = st.selectbox("Facility Type", list(facility_options.keys()), format_func=lambda x: f"Type {x} - {facility_options[x]}")
            input_data['facilityType'] = facility_choice
            
            # Time & Date Selection
            st.subheader("⏰ Charging Session Time")
            
            col_c, col_d = st.columns(2)
            with col_c:
                start_time = st.time_input("Start Time", time(12, 0))
            with col_d:
                today = datetime.now()
                start_date = st.date_input("Start Date", today)
            
            # Charging Speed with a slider for better UX
            st.subheader("🔋 Charging Parameters")
            input_data['charging_speed'] = st.slider(
                "Charging Speed (kW)",
                min_value=1.0,
                max_value=10.0,
                value=5.8,
                step=0.1,
                help="Higher values mean faster charging"
            )
            
            # Compute derived features
            input_data['startHour'] = start_time.hour
            input_data['startMonth'] = start_date.month
            input_data['is_peak_hour'] = 1 if input_data['startHour'] in [7, 8, 9, 17, 18, 19, 20] else 0
            input_data['is_weekend'] = 1 if start_date.weekday() >= 5 else 0
            
            # Map month to season (with nicer display)
            season_mapping = {
                (12, 1, 2): (1, "Winter ❄️"),
                (3, 4, 5): (2, "Spring 🌱"),
                (6, 7, 8): (3, "Summer ☀️"),
                (9, 10, 11): (4, "Fall 🍂")
            }
            season_id, season_name = next((v for k, v in season_mapping.items() if input_data['startMonth'] in k), (1, "Winter"))
            input_data['season'] = season_id
            
            # Display derived values to the user
            st.markdown("""
            <div style="background: rgba(30, 30, 30, 0.5); border-radius: 10px; padding: 10px; margin-top: 10px;">
                <h4 style="color: #4CAF50; margin-bottom: 8px;">Detected Parameters</h4>
                <ul style="list-style-type: none; padding-left: 5px; margin-bottom: 0;">
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <li>Season: {season_name}</li>
                <li>{'Peak Hours' if input_data['is_peak_hour'] else 'Off-Peak Hours'}</li>
                <li>{'Weekend' if input_data['is_weekend'] else 'Weekday'}</li>
            """, unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Submit button
            predict_button = st.form_submit_button(label=" Generate Prediction")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Results Display
    with col2:
        st.markdown('<div class="card"><h3 class="card-title">Prediction Results</h3>', unsafe_allow_html=True)
        
        if predict_button:
            # Show a loading spinner
            with st.spinner("Calculating your prediction..."):
                # Prepare input for model
                input_df = pd.DataFrame([input_data])
                input_processed = preprocessor.transform(input_df)
                
                # Make prediction
                predictions = model.predict(input_processed)
                kwh_total_pred, charge_time_hrs_pred = predictions[0]
                
                # Display Predictions with improved visualization
                st.markdown(f"""
                    <div class="prediction-box">
                        <span class="prediction-icon">🔋</span>
                        <div>
                            <span style="font-size: 0.8rem; display: block; text-align: left; opacity: 0.8;">Predicted Energy:</span>
                            <span>{kwh_total_pred:.2f} kWh</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="prediction-box">
                        <span class="prediction-icon">⏱️</span>
                        <div>
                            <span style="font-size: 0.8rem; display: block; text-align: left; opacity: 0.8;">Predicted Time:</span>
                            <span>{charge_time_hrs_pred:.2f} hours</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics (cost and environmental impact)
                """avg_electricity_cost = 0.15  # $/kWh
                cost_estimate = kwh_total_pred * avg_electricity_cost
                co2_saved = kwh_total_pred * 0.4  # kg CO2 equivalent
                
                st.markdown("""
                    <div class="metric-container">
                        <div class="metric-card">
                            <div class="metric-value">$%.2f</div>
                            <div class="metric-label">Estimated Cost</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">%.1f kg</div>
                            <div class="metric-label">CO₂ Saved</div>
                        </div>
                    </div>
                """ % (cost_estimate, co2_saved), unsafe_allow_html=True)
                
                # Charging efficiency visualization
                efficiency = min(100, 100 * (kwh_total_pred / (charge_time_hrs_pred * input_data['charging_speed'])))
                
                st.markdown("<h4 style='margin-top: 1.5rem;'>Charging Efficiency:</h4>", unsafe_allow_html=True)
                st.progress(efficiency / 100)
                st.markdown(f"<div style='text-align: center; font-size: 0.9rem;'>{efficiency:.1f}% efficiency</div>", unsafe_allow_html=True)
                
                # Tips based on prediction
                st.subheader("💡 Smart Tips")
                tips = []
                
                if input_data['is_peak_hour']:
                    tips.append("Consider charging during off-peak hours to reduce costs.")
                
                if charge_time_hrs_pred > 3:
                    tips.append("For faster charging, consider using a higher-powered charging station.")
                
                if efficiency < 70:
                    tips.append("Your charging efficiency is below average. Check battery health.")
                
                for tip in tips:
                    st.markdown(f"<div style='background: rgba(76, 175, 80, 0.1); padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #4CAF50;'>{tip}</div>", unsafe_allow_html=True)
        """
        else:
            # Default display when no prediction has been made
            st.markdown("""
                <div style="text-align: center; padding: 2rem 0;">
                    <img src="https://img.icons8.com/fluency/96/000000/tesla-model-3.png" style="width: 80px; height: 80px; margin-bottom: 1rem; opacity: 0.8;">
                    <h3 style="color: #808080; font-weight: 400; margin-bottom: 1rem;">Ready for your prediction</h3>
                    <p style="color: #A0A0A0; font-size: 0.9rem;">Fill in the parameters and click 'Generate Prediction' to see your results.</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Map Tab - for displaying nearby charging stations
with tab2:
    st.markdown('<div class="card"><h3 class="card-title">Find Nearby Charging Stations</h3>', unsafe_allow_html=True)
    
    # Search for location
    city_name = st.text_input("Enter a city or address:", placeholder="e.g., New York, NY")
    
    # Get coordinates for the location
    def get_coordinates(city):
        url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key=4cda5084fabf428aa8e6564d16b7ad8c"
        response = requests.get(url).json()
        if response and 'results' in response and len(response['results']) > 0:
            return float(response['results'][0]['geometry']['lat']), float(response['results'][0]['geometry']['lng'])
        return None, None
    
    # Get nearby EV stations
    def get_nearby_ev_stations(lat, lon):
        api_url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={lat}&longitude={lon}&maxresults=10&key=a1f5b87f-3209-4eb2-afc1-c9d379acfa10"
        response = requests.get(api_url).json()
        return response
    
    # Determine location
    lat, lon = None, None
    if city_name:
        lat, lon = get_coordinates(city_name)
        if lat and lon:
            st.success(f"📍 Location set to: {city_name}")
        else:
            st.warning("⚠️ Could not find the entered location. Please try another search or use your current location.")
    
    # Get user's current location if no city is provided
    if lat is None or lon is None:
        st.info("🔍 Click 'Use My Location' to find charging stations near you")
        
        if st.button("📱 Use My Location"):
            try:
                location = get_geolocation()
                if location and isinstance(location, dict) and 'coords' in location:
                    lat, lon = location['coords'].get('latitude'), location['coords'].get('longitude')
                    if lat and lon:
                        st.success(f"📍 Location detected: {lat:.6f}, {lon:.6f}")
                    else:
                        st.error("Could not retrieve your location coordinates.")
                else:
                    st.error("Location access failed. Please check your browser settings or enter a location manually.")
            except Exception as e:
                st.error(f"Error accessing location: {e}")
    
    # Display the map if coordinates are available
    if lat and lon:
        # Create a map centered at the user's location
        m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
        
        # Add marker for user location
        folium.Marker(
            [lat, lon],
            popup="Your Location",
            icon=folium.Icon(color="blue", icon="user", prefix="fa"),
            tooltip="You are here"
        ).add_to(m)
        
        # Get and display nearby charging stations
        with st.spinner("Searching for nearby charging stations..."):
            stations = get_nearby_ev_stations(lat, lon)
            
            if stations:
                # Create a station info table
                st.subheader(f"Found {len(stations)} Charging Stations Nearby")
                
                station_data = []
                for station in stations:
                    try:
                        station_name = station.get('AddressInfo', {}).get('Title', 'Unknown Station')
                        station_lat = station.get('AddressInfo', {}).get('Latitude')
                        station_lon = station.get('AddressInfo', {}).get('Longitude')
                        
                        if station_lat and station_lon:
                            # Add marker to the map
                            station_address = station.get('AddressInfo', {}).get('AddressLine1', '')
                            connector_types = [c.get('ConnectionType', {}).get('Title', 'Unknown') for c in station.get('Connections', [])]
                            connector_str = ", ".join(set(connector_types)) if connector_types else "Unknown"
                            
                            # Determine station status and color
                            status = "Available"
                            marker_color = "green"
                            for connection in station.get('Connections', []):
                                if connection.get('StatusType', {}).get('IsOperational') == False:
                                    status = "Unavailable"
                                    marker_color = "red"
                                    break
                            
                            # Create popup content
                            popup_html = f"""
                                <div style="width: 200px;">
                                    <h4 style="margin-bottom: 5px;">{station_name}</h4>
                                    <p style="margin-bottom: 5px;"><b>Address:</b> {station_address}</p>
                                    <p style="margin-bottom: 5px;"><b>Connectors:</b> {connector_str}</p>
                                    <p><b>Status:</b> <span style="color: {'green' if status == 'Available' else 'red'};">{status}</span></p>
                                </div>
                            """
                            
                            folium.Marker(
                                [station_lat, station_lon],
                                popup=folium.Popup(popup_html, max_width=300),
                                icon=folium.Icon(color=marker_color, icon="bolt", prefix="fa"),
                                tooltip=station_name
                            ).add_to(m)
                            
                            # Add to table data
                            station_data.append({
                                "Name": station_name,
                                "Distance": f"{station.get('AddressInfo', {}).get('Distance', 0):.1f} km",
                                "Connectors": connector_str,
                                "Status": status
                            })
                    except Exception as e:
                        print(f"Error processing station: {e}")
                
                # Display the station table
                if station_data:
                    station_df = pd.DataFrame(station_data)
                    st.dataframe(station_df, use_container_width=True, hide_index=True)
        
        # Display the map with a custom CSS class for styling
        st.markdown('<div class="folium-map">', unsafe_allow_html=True)
        folium_static(m, width=1000, height=500)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add travel directions
        st.markdown("""
            <div style="background: rgba(33, 150, 243, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px; border: 1px solid rgba(33, 150, 243, 0.3);">
                <h4 style="color: #2196F3; margin-top: 0;">Need directions?</h4>
                <p>Click on a station marker and get directions in your preferred navigation app.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Display a placeholder map or message
        st.info("Enter a location or use your current position to see charging stations on the map.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
        <p style="color: #808080; font-size: 0.8rem;">EV Charging Prediction AI © 2023 | Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
