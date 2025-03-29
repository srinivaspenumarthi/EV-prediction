import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_filename = "best_random_forest.pkl"
model = joblib.load(model_filename)

# Get the feature names from the trained model
feature_names = model.feature_names_in_

# Streamlit App
st.title("🔮 EV Charging Prediction App")
st.markdown("Enter the required features to predict the output using the trained model.")

# Create input fields for each feature dynamically
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, format="%.4f")

# Convert input data to DataFrame for prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    
    # Ensure input data matches feature names
    input_df = input_df[feature_names]  

    # Make Prediction
    prediction = model.predict(input_df)

    st.success(f"🔮 Prediction: {prediction[0]:.4f}")

st.markdown("---")
st.markdown("🚀 Built with Streamlit | Trained with Scikit-Learn")
