import streamlit as st
import joblib
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("best_random_forest.pkl")

# Streamlit app design
st.set_page_config(page_title="AI Power Predictor", page_icon="⚡", layout="wide")

# Custom CSS for better appearance
st.markdown(
    """
    <style>
    .header-font {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        color: #FF4B2B;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1E1E1E;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<p class='header-font'>⚡ AI Power Consumption Predictor ⚡</p>", unsafe_allow_html=True)

# Organize features into groups for better usability
col1, col2 = st.columns(2)
with col1:
    st.write("### Basic Information")
    sessionId = st.number_input("Session ID", value=0.0, step=1.0)
    dollars = st.number_input("Cost (Dollars)", value=0.0, step=0.1, format="%.2f")
    chargeTimeHrs = st.number_input("Charge Time (Hours)", value=0.0, step=0.1, format="%.2f")
    distance = st.number_input("Distance", value=0.0, step=0.1, format="%.2f")

with col2:
    st.write("### User & Station Data")
    userId = st.number_input("User ID", value=0.0, step=1.0)
    stationId = st.number_input("Station ID", value=0.0, step=1.0)
    locationId = st.number_input("Location ID", value=0.0, step=1.0)
    managerVehicle = st.number_input("Manager Vehicle", value=0.0, step=1.0)
    facilityType = st.number_input("Facility Type", value=0.0, step=1.0)
    reportedZip = st.number_input("Reported ZIP", value=0.0, step=1.0)

# Expandable sections for additional features
with st.expander("Time Features"):
    time_col1, time_col2 = st.columns(2)
    with time_col1:
        startTime = st.number_input("Start Time", value=0.0, step=0.1, format="%.2f")
        endTime = st.number_input("End Time", value=0.0, step=0.1, format="%.2f")
        start_hour = st.number_input("Start Hour (0-23)", value=0.0, min_value=0.0, max_value=23.0, step=1.0)
        end_hour = st.number_input("End Hour (0-23)", value=0.0, min_value=0.0, max_value=23.0, step=1.0)
    
    with time_col2:
        day_of_week = st.number_input("Day of Week (0-6)", value=0.0, min_value=0.0, max_value=6.0, step=1.0)
        month = st.number_input("Month (1-12)", value=1.0, min_value=1.0, max_value=12.0, step=1.0)
        
        # Day checkboxes
        st.write("Days of Week:")
        Mon = 1.0 if st.checkbox("Monday") else 0.0
        Tues = 1.0 if st.checkbox("Tuesday") else 0.0
        Wed = 1.0 if st.checkbox("Wednesday") else 0.0
        Thurs = 1.0 if st.checkbox("Thursday") else 0.0
        Fri = 1.0 if st.checkbox("Friday") else 0.0
        Sat = 1.0 if st.checkbox("Saturday") else 0.0
        Sun = 1.0 if st.checkbox("Sunday") else 0.0

with st.expander("Advanced Features"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        kwhTotal_lag_1 = st.number_input("kWh Total Lag 1", value=0.0, step=0.01, format="%.2f")
        kwhTotal_lag_2 = st.number_input("kWh Total Lag 2", value=0.0, step=0.01, format="%.2f")
        kwhTotal_lag_3 = st.number_input("kWh Total Lag 3", value=0.0, step=0.01, format="%.2f")
        distance_x_chargeTimeHrs = st.number_input("Distance × Charge Time", value=0.0, step=0.01, format="%.2f")
    
    with adv_col2:
        day_of_week_x_start_hour = st.number_input("Day of Week × Start Hour", value=0.0, step=0.01, format="%.2f")
        kwhTotal_x_kwhTotal_lag_1 = st.number_input("kWh × kWh Lag 1", value=0.0, step=0.01, format="%.2f")
        kwhTotal_x_kwhTotal_lag_2 = st.number_input("kWh × kWh Lag 2", value=0.0, step=0.01, format="%.2f")
        kwhTotal_x_kwhTotal_lag_3 = st.number_input("kWh × kWh Lag 3", value=0.0, step=0.01, format="%.2f")

# Add a simple preset option
st.sidebar.markdown("### Quick Presets")
preset = st.sidebar.selectbox(
    "Select a preset scenario:",
    ["Custom (No preset)", "Typical Weekday Charge", "Weekend Long Charge", "Short Quick Charge"]
)

# Apply preset values if selected
if preset == "Typical Weekday Charge":
    if st.sidebar.button("Apply Preset"):
        sessionId = 1001.0
        dollars = 15.75
        chargeTimeHrs = 2.5
        distance = 30.0
        # Would set other values too
        st.experimental_rerun()
elif preset == "Weekend Long Charge":
    if st.sidebar.button("Apply Preset"):
        sessionId = 2001.0
        dollars = 25.50
        chargeTimeHrs = 4.0
        distance = 15.0
        # Would set other values too
        st.experimental_rerun()
elif preset == "Short Quick Charge":
    if st.sidebar.button("Apply Preset"):
        sessionId = 3001.0
        dollars = 8.25
        chargeTimeHrs = 0.75
        distance = 5.0
        # Would set other values too
        st.experimental_rerun()

# Create the features array
features = [
    sessionId, dollars, startTime, endTime, chargeTimeHrs, distance,
    userId, stationId, locationId, managerVehicle, facilityType, Mon,
    Tues, Wed, Thurs, Fri, Sat, Sun, reportedZip, start_hour,
    end_hour, day_of_week, month, kwhTotal_lag_1, kwhTotal_lag_2,
    kwhTotal_lag_3, distance_x_chargeTimeHrs, day_of_week_x_start_hour,
    kwhTotal_x_kwhTotal_lag_1, kwhTotal_x_kwhTotal_lag_2, kwhTotal_x_kwhTotal_lag_3
]

# Prediction Button with better styling
st.markdown("")  # Add some space
predict_button = st.button("⚡ Predict Power Consumption", key="predict", use_container_width=True)

if predict_button:
    # Show spinner during prediction
    with st.spinner("Analyzing data..."):
        time.sleep(1)  # Simulated delay for better UX
        prediction = model.predict([features])
    
    # Display the prediction with animation
    st.markdown("""
    <div class='result-box'>
        <h2>Prediction Result</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Use progress bar for visual effect
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    st.success(f"⚡ Predicted Power Consumption: {prediction[0]:.4f} kWh")
    
    # Display a simple visualization of the result
    fig, ax = plt.subplots()
    ax.bar(['Predicted kWh'], [prediction[0]], color='#FF4B2B')
    ax.set_ylabel('kWh')
    ax.set_title('Power Consumption Prediction')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Power Consumption Predictor v1.0 | Built with Streamlit")
