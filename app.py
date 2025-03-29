import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained models
models = {
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "KNN": "knn_model.pkl",
    "Decision Tree": "decision_tree_model.pkl"
}

st.title("EV Charging Prediction")
st.write("Select a model and enter values to predict energy consumption (kWhTotal)")

# Model selection
selected_model = st.selectbox("Choose a Model", list(models.keys()))
model = joblib.load(models[selected_model])

# Generate random lag values
kwhTotal_lag_1 = np.random.uniform(0, 1)
kwhTotal_lag_2 = np.random.uniform(0, 1)
kwhTotal_lag_3 = np.random.uniform(0, 1)

# User input fields
input_data = {
    "dollars": st.number_input("Dollars", value=0.0),
    "startTime": st.number_input("Start Time (Hour)", value=13.0),
    "endTime": st.number_input("End Time (Hour)", value=16.0),
    "chargeTimeHrs": st.number_input("Charge Time (Hrs)", value=2.8089),
    "distance": st.number_input("Distance (miles)", value=21.02),
    "managerVehicle": st.selectbox("Manager Vehicle", [0, 1]),
    "facilityType": st.selectbox("Facility Type", [1, 2, 3, 4]),
    "reportedZip": st.number_input("Reported Zip", value=1.0),
    "month": st.number_input("Month", value=1),
    "is_weekend": st.selectbox("Is Weekend", [0, 1]),
    "kwhTotal_lag_1": kwhTotal_lag_1,
    "kwhTotal_lag_2": kwhTotal_lag_2,
    "kwhTotal_lag_3": kwhTotal_lag_3,
}

# Single dropdown for weekday selection
selected_day = st.selectbox("Select Day of the Week", ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"])
for day in ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"]:
    input_data[day] = 1 if day == selected_day else 0

# Auto-calculate dependent values
input_data["distance_x_chargeTimeHrs"] = input_data["distance"] * input_data["chargeTimeHrs"]
input_data["day_of_week_x_start_hour"] = input_data[selected_day] * input_data["startTime"]
input_data["kwhTotal_x_kwhTotal_lag_1"] = input_data["kwhTotal_lag_1"] * input_data["kwhTotal_lag_1"]
input_data["kwhTotal_x_kwhTotal_lag_2"] = input_data["kwhTotal_lag_2"] * input_data["kwhTotal_lag_2"]
input_data["kwhTotal_x_kwhTotal_lag_3"] = input_data["kwhTotal_lag_3"] * input_data["kwhTotal_lag_3"]

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted kWhTotal: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
