import streamlit as st
import joblib
import pandas as pd

# Load models
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
dollars = st.number_input("Dollars", value=0.0)
startTime = st.number_input("Start Time (Hour)", value=13.0)
endTime = st.number_input("End Time (Hour)", value=16.0)
chargeTimeHrs = st.number_input("Charge Time (Hrs)", value=2.8089)
distance = st.number_input("Distance (miles)", value=21.02)
managerVehicle = st.selectbox("Manager Vehicle", [0, 1])
facilityType = st.selectbox("Facility Type", [1, 2, 3, 4])
Mon = st.selectbox("Monday", [0, 1])
Tues = st.selectbox("Tuesday", [0, 1])
Wed = st.selectbox("Wednesday", [0, 1])
Thurs = st.selectbox("Thursday", [0, 1])
Fri = st.selectbox("Friday", [0, 1])
Sat = st.selectbox("Saturday", [0, 1])
Sun = st.selectbox("Sunday", [0, 1])
reportedZip = st.number_input("Reported Zip", value=1.0)

# Create input dataframe
input_data = pd.DataFrame([[
    dollars, startTime, endTime, chargeTimeHrs, distance,
    managerVehicle, facilityType, Mon, Tues, Wed, Thurs, Fri, Sat, Sun, reportedZip
]], columns=[
    "dollars", "startTime", "endTime", "chargeTimeHrs", "distance", 
    "managerVehicle", "facilityType", "Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun", "reportedZip"
])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted kWhTotal: {prediction[0]:.2f}")
