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

# User input fields
input_data = {
    "dollars": st.number_input("Dollars", value=0.0),
    "startTime": st.number_input("Start Time (Hour)", value=13),
    "endTime": st.number_input("End Time (Hour)", value=16),
    "chargeTimeHrs": st.number_input("Charge Time (Hrs)", value=2.8),
    "distance": st.number_input("Distance (miles)", value=21.02),
    "month": st.number_input("Month", value=1),
    "is_weekend": st.selectbox("Is Weekend", [0, 1]),
    "kwhTotal_lag_1": st.number_input("kWhTotal Lag 1", value=0.0),
    "kwhTotal_lag_2": st.number_input("kWhTotal Lag 2", value=0.0),
    "kwhTotal_lag_3": st.number_input("kWhTotal Lag 3", value=0.0),
}

# Dropdown for weekday selection (Convert to numerical format)
selected_day = st.selectbox("Select Day of the Week", ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"])
day_mapping = {"Mon": 0, "Tues": 1, "Wed": 2, "Thurs": 3, "Fri": 4, "Sat": 5, "Sun": 6}
input_data["weekday"] = day_mapping[selected_day]

# Auto-calculate dependent values
input_data["distance_x_chargeTimeHrs"] = input_data["distance"] * input_data["chargeTimeHrs"]
input_data["day_of_week_x_start_hour"] = input_data["weekday"] * input_data["startTime"]
input_data["kwhTotal_x_kwhTotal_lag_1"] = input_data["kwhTotal_lag_1"] * input_data["kwhTotal_lag_1"]
input_data["kwhTotal_x_kwhTotal_lag_2"] = input_data["kwhTotal_lag_2"] * input_data["kwhTotal_lag_2"]
input_data["kwhTotal_x_kwhTotal_lag_3"] = input_data["kwhTotal_lag_3"] * input_data["kwhTotal_lag_3"]

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure input features match model expectations
missing_features = set(model.feature_names_in_) - set(input_df.columns)
if missing_features:
    st.error(f"Missing features: {missing_features}")
else:
    # Predict
    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)
            st.success(f"Predicted kWhTotal: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
