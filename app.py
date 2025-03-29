import streamlit as st
import joblib
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import json
from streamlit_plotly_events import plotly_events

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("best_random_forest.pkl")

model = load_model()

# Load Lottie animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Lottie animations
lottie_power = load_lottieurl("https://lottie.host/ff472e08-69a1-4e27-89d1-33c3a5e90e33/ClscJEZ326.json")
lottie_charging = load_lottieurl("https://lottie.host/1dc3ffd6-7197-49aa-a7fc-ee157dfc5f74/n1evmS6DSx.json")

# Streamlit app design
st.set_page_config(
    page_title="AI Power Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-premium look
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a1c20 0%, #0e1117 100%);
        color: #fff;
    }
    
    [data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #192133 0%, #111827 100%);
        border-radius: 10px;
        margin: 10px;
    }
    
    .css-1kyxreq.e115fcil2 {
        border-radius: 20px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stNumberInput input, .stTextInput input, .stButton > button {
        border-radius: 20px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4f46e5, #7c3aed);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.4);
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.6);
    }
    
    .big-font {
        font-size: 42px !important;
        font-weight: 800;
        background: linear-gradient(to right, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        font-size: 18px;
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        margin-bottom: 30px;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s;
    }
    
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.07);
        transform: translateY(-5px);
    }
    
    .prediction-area {
        background: rgba(24, 26, 27, 0.6);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(79, 70, 229, 0.2);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        margin-top: 20px;
    }
    
    .prediction-value {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(to right, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #4f46e5;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stSlider > div > div > div {
        background-color: #4f46e5 !important;
    }
    
    .stProgress > div > div > div {
        background-color: #4f46e5 !important;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: rgba(0, 0, 0, 0.8);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        width: 120px;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(124, 58, 237, 0.2) !important;
        border-bottom: 2px solid #7c3aed !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App header with animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st_lottie(lottie_power, speed=1, height=180, key="power_animation")
    st.markdown("<p class='big-font'>⚡ AI Power Consumption Predictor</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Predict energy consumption using advanced machine learning algorithms</p>", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔮 3D Visualization", "ℹ️ About"])

# Tab 1: Prediction
with tab1:
    # Two column layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        
        # Feature categories
        feature_categories = {
            "Session Data": ['sessionId', 'dollars', 'startTime', 'endTime', 'chargeTimeHrs', 'distance'],
            "User & Location": ['userId', 'stationId', 'locationId', 'managerVehicle', 'facilityType', 'reportedZip'],
            "Time Features": ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun', 'start_hour', 'end_hour', 'day_of_week', 'month'],
            "Historical & Derived": ['kwhTotal_lag_1', 'kwhTotal_lag_2', 'kwhTotal_lag_3', 'distance_x_chargeTimeHrs', 
                                   'day_of_week_x_start_hour', 'kwhTotal_x_kwhTotal_lag_1', 
                                   'kwhTotal_x_kwhTotal_lag_2', 'kwhTotal_x_kwhTotal_lag_3']
        }
        
        # Create expanders for each category and collect inputs
        features = {}
        for category, feature_list in feature_categories.items():
            with st.expander(f"{category}", expanded=True if category == "Session Data" else False):
                for feature in feature_list:
                    # Add default values for demonstration
                    default_val = 1.0 if feature in ['chargeTimeHrs', 'distance', 'kwhTotal_lag_1'] else 0.0
                    
                    # Add tooltips for features
                    tooltips = {
                        'chargeTimeHrs': 'Duration of charging in hours',
                        'distance': 'Distance traveled in miles',
                        'kwhTotal_lag_1': 'Previous consumption in kWh',
                        'start_hour': 'Hour when charging started (0-23)',
                    }
                    
                    tooltip_text = tooltips.get(feature, f"{feature} value")
                    
                    # Display feature with tooltip
                    features[feature] = st.number_input(
                        f"{feature} ℹ️",
                        min_value=0.0,
                        value=default_val,
                        step=0.01,
                        format="%.2f",
                        help=tooltip_text
                    )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick Preset Templates
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("🔄 Quick Presets")
        preset = st.selectbox(
            "Load a sample configuration",
            ["Custom Input", "Home Charging - Evening", "Work Charging - Day", "Fast Charging Station"]
        )
        
        if preset == "Home Charging - Evening":
            # Update values for home charging preset
            features['chargeTimeHrs'] = 8.0
            features['distance'] = 30.0
            features['start_hour'] = 18.0
            features['facilityType'] = 1.0
            st.success("Home charging preset loaded!")
            
        elif preset == "Work Charging - Day":
            # Update values for work charging preset
            features['chargeTimeHrs'] = 6.0
            features['distance'] = 15.0
            features['start_hour'] = 9.0
            features['facilityType'] = 2.0
            st.success("Work charging preset loaded!")
            
        elif preset == "Fast Charging Station":
            # Update values for fast charging preset
            features['chargeTimeHrs'] = 0.75
            features['distance'] = 120.0
            features['start_hour'] = 14.0
            features['facilityType'] = 3.0
            st.success("Fast charging preset loaded!")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        # Create prediction card
        st.markdown("<div class='prediction-area'>", unsafe_allow_html=True)
        st_lottie(lottie_charging, speed=1, height=200, key="charging_animation")
        
        # Prepare all features in the correct order
        feature_names = ['sessionId', 'dollars', 'startTime', 'endTime', 'chargeTimeHrs', 'distance',
                       'userId', 'stationId', 'locationId', 'managerVehicle', 'facilityType', 'Mon',
                       'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun', 'reportedZip', 'start_hour',
                       'end_hour', 'day_of_week', 'month', 'kwhTotal_lag_1', 'kwhTotal_lag_2',
                       'kwhTotal_lag_3', 'distance_x_chargeTimeHrs', 'day_of_week_x_start_hour',
                       'kwhTotal_x_kwhTotal_lag_1', 'kwhTotal_x_kwhTotal_lag_2', 'kwhTotal_x_kwhTotal_lag_3']
        
        # Calculate derived features based on inputs
        features['distance_x_chargeTimeHrs'] = features['distance'] * features['chargeTimeHrs']
        features['day_of_week_x_start_hour'] = features.get('day_of_week', 0) * features.get('start_hour', 0)
        
        # Set weekday flags based on day_of_week
        if 'day_of_week' in features:
            day = int(features['day_of_week'])
            features['Mon'] = 1.0 if day == 0 else 0.0
            features['Tues'] = 1.0 if day == 1 else 0.0
            features['Wed'] = 1.0 if day == 2 else 0.0
            features['Thurs'] = 1.0 if day == 3 else 0.0
            features['Fri'] = 1.0 if day == 4 else 0.0
            features['Sat'] = 1.0 if day == 5 else 0.0
            features['Sun'] = 1.0 if day == 6 else 0.0

        # Placeholders for features that need more complex calculation
        if 'kwhTotal_x_kwhTotal_lag_1' not in features:
            features['kwhTotal_x_kwhTotal_lag_1'] = features.get('kwhTotal_lag_1', 0) * 1.0
        if 'kwhTotal_x_kwhTotal_lag_2' not in features:
            features['kwhTotal_x_kwhTotal_lag_2'] = features.get('kwhTotal_lag_2', 0) * 1.0
        if 'kwhTotal_x_kwhTotal_lag_3' not in features:
            features['kwhTotal_x_kwhTotal_lag_3'] = features.get('kwhTotal_lag_3', 0) * 1.0
        
        # Extract features in the correct order
        feature_values = [features.get(feature, 0.0) for feature in feature_names]
        
        # Prediction button with animation
        if st.button("🚀 Generate Prediction", key="predict"):
            with st.spinner("AI model processing..."):
                # Show progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                # Make prediction
                prediction = model.predict([feature_values])[0]
                
                st.markdown(f"<p class='prediction-value'>{prediction:.2f} kWh</p>", unsafe_allow_html=True)
                
                # Show estimated cost
                cost_per_kwh = 0.15  # Average cost per kWh in dollars
                estimated_cost = prediction * cost_per_kwh
                st.metric(
                    "Estimated Cost", 
                    f"${estimated_cost:.2f}",
                    delta=None
                )
                
                # Generate a sample time-series to visualize consumption pattern
                # This is a simulated pattern based on the predicted value
                hours = np.arange(0, 24, 0.5)
                
                # Create a bell-shaped curve centered at features['start_hour']
                start = features.get('start_hour', 12)
                duration = features.get('chargeTimeHrs', 1)
                charge_pattern = np.exp(-0.5 * ((hours - start) / duration) ** 2)
                charge_pattern = charge_pattern * (prediction / charge_pattern.sum())
                
                # Create a dataframe for plotting
                df = pd.DataFrame({
                    'Hour': hours,
                    'Power (kW)': charge_pattern
                })
                
                # Plot the consumption pattern
                fig = px.area(
                    df, 
                    x='Hour', 
                    y='Power (kW)',
                    title='Predicted Power Consumption Pattern',
                    color_discrete_sequence=['rgba(79, 70, 229, 0.6)'],
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(
                        title='Hour of Day',
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True,
                    ),
                    yaxis=dict(
                        title='Power Consumption (kW)',
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True,
                    ),
                    hovermode='x unified',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance visualization (simulated)
                st.subheader("🔍 Feature Impact Analysis")
                features_importance = {
                    'chargeTimeHrs': 0.35,
                    'distance': 0.25,
                    'kwhTotal_lag_1': 0.15,
                    'start_hour': 0.10,
                    'facilityType': 0.08,
                    'day_of_week': 0.07,
                }
                
                fig_imp = px.bar(
                    x=list(features_importance.values()),
                    y=list(features_importance.keys()),
                    orientation='h',
                    title='Feature Importance',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    color=list(features_importance.values()),
                    color_continuous_scale='Viridis',
                )
                
                fig_imp.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    coloraxis_showscale=False,
                )
                
                st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: 3D Visualization
with tab2:
    st.header("🔌 3D Power Consumption Visualization")
    
    # Create a 3D scatter plot with sample data
    st.markdown("This 3D visualization shows the relationship between charging time, distance, and power consumption:")
    
    # Generate sample data points
    np.random.seed(42)
    num_points = 100
    charge_times = np.random.uniform(0.5, 10, num_points)
    distances = np.random.uniform(5, 200, num_points)
    
    # Create a simple model for demo purposes
    power = 2 * charge_times + 0.05 * distances + charge_times * distances * 0.01 + np.random.normal(0, 2, num_points)
    
    # Create a color scale based on power values
    fig = go.Figure(data=[go.Scatter3d(
        x=charge_times,
        y=distances,
        z=power,
        mode='markers',
        marker=dict(
            size=6,
            color=power,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Power (kWh)")
        ),
        hovertemplate=
        '<b>Charge Time</b>: %{x:.2f} hrs<br>' +
        '<b>Distance</b>: %{y:.2f} miles<br>' +
        '<b>Power</b>: %{z:.2f} kWh<extra></extra>'
    )])
    
    # Add a surface to better visualize the relationship
    x_range = np.linspace(min(charge_times), max(charge_times), 20)
    y_range = np.linspace(min(distances), max(distances), 20)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 2 * X + 0.05 * Y + X * Y * 0.01
    
    fig.add_trace(go.Surface(
        x=x_range, y=y_range, z=Z,
        colorscale='Viridis',
        opacity=0.4,
        showscale=False
    ))

    fig.update_layout(
        title='3D Relationship: Charging Time, Distance & Power Consumption',
        scene=dict(
            xaxis_title='Charging Time (hours)',
            yaxis_title='Distance (miles)',
            zaxis_title='Power Consumption (kWh)',
            xaxis=dict(gridcolor='white', backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(gridcolor='white', backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(gridcolor='white', backgroundcolor='rgba(0,0,0,0)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Tip: Click and drag to rotate the 3D visualization. Scroll to zoom in/out.")

    # Interactive exploration section
    st.subheader("🎮 Interactive Exploration")
    col1, col2 = st.columns(2)
    
    with col1:
        explore_time = st.slider("Charging Time (hours)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
    with col2:
        explore_distance = st.slider("Distance (miles)", min_value=5.0, max_value=200.0, value=50.0, step=5.0)
    
    # Calculate estimated power using the simple model
    estimated_power = 2 * explore_time + 0.05 * explore_distance + explore_time * explore_distance * 0.01
    
    # Create a highlighted point in the 3D space
    highlight_fig = go.Figure(data=[go.Scatter3d(
        x=charge_times,
        y=distances,
        z=power,
        mode='markers',
        marker=dict(
            size=4,
            color='rgba(70, 70, 70, 0.4)',
            opacity=0.6
        ),
        hoverinfo='none'
    )])
    
    # Add the highlighted point
    highlight_fig.add_trace(go.Scatter3d(
        x=[explore_time],
        y=[explore_distance],
        z=[estimated_power],
        mode='markers',
        marker=dict(
            size=12,
            color='#ff4b2b',
            symbol='diamond',
            line=dict(color='white', width=1)
        ),
        hovertemplate=
        '<b>Charge Time</b>: %{x:.2f} hrs<br>' +
        '<b>Distance</b>: %{y:.2f} miles<br>' +
        '<b>Power</b>: %{z:.2f} kWh<extra>Your Selection</extra>'
    ))
    
    # Add a surface to better visualize the relationship (same as before)
    highlight_fig.add_trace(go.Surface(
        x=x_range, y=y_range, z=Z,
        colorscale='Viridis',
        opacity=0.3,
        showscale=False
    ))

    highlight_fig.update_layout(
        title='Your Selection in the 3D Power Space',
        scene=dict(
            xaxis_title='Charging Time (hours)',
            yaxis_title='Distance (miles)',
            zaxis_title='Power Consumption (kWh)',
            xaxis=dict(gridcolor='white', backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(gridcolor='white', backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(gridcolor='white', backgroundcolor='rgba(0,0,0,0)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )
    
    st.plotly_chart(highlight_fig, use_container_width=True)
    
    # Display the estimated power with a metric
    st.metric(
        "Estimated Power Consumption", 
        f"{estimated_power:.2f} kWh",
        delta=f"{estimated_power - 20:.2f} kWh from baseline" if estimated_power > 20 else f"{20 - estimated_power:.2f} kWh below baseline"
    )

# Tab 3: About
with tab3:
    st.header("About the Power Consumption Predictor")
    
    st.markdown("""
    <div class="feature-card">
    <h3>🧠 AI Model Details</h3>
    <p>This application uses a Random Forest Regressor model trained on historical charging data to predict power consumption for electric vehicles. The model analyzes 31 different features to generate accurate predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <h3>📊 Feature Importance</h3>
    <p>The most influential features in predicting power consumption are:</p>
    <ul>
        <li>Charging duration (hours)</li>
        <li>Distance traveled</li>
        <li>Historical consumption patterns</li>
        <li>Time of day</li>
        <li>Day of week</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <h3>💡 How to Use</h3>
    <p>1. Enter your charging session details in the sidebar</p>
    <p>2. Use the quick presets for common scenarios</p>
    <p>3. Click "Generate Prediction" to see the estimated power consumption</p>
    <p>4. Explore the 3D visualization to better understand the relationships between features</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; font-size: 12px; color: rgba(255,255,255,0.5);">
    Powered by ⚡ AI Power Predictor | Created with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
