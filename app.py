import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_filename = "best_random_forest.pkl"
model = joblib.load(model_filename)

# Get feature names
feature_names = model.feature_names_in_

# Custom Styling
st.set_page_config(page_title="EV Charging AI 🚀", layout="wide")

# Custom CSS for Billion-Dollar Look
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            color: #ffffff;
            background-color: #0e1117;
        }
        .stApp {
            background: linear-gradient(135deg, #1f1f1f, #2c2c2c);
            padding: 2rem;
        }
        .title {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            color: #61dafb;
        }
        .subtitle {
            font-size: 1.2rem;
            text-align: center;
            color: #a0a0a0;
        }
        .stTextInput, .stNumberInput, .stButton > button {
            border-radius: 10px;
        }
        .stButton > button {
            background: #61dafb;
            color: black;
            font-size: 1.1rem;
            padding: 10px 20px;
            border-radius: 10px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: #40a9ff;
        }
        .prediction-box {
            background: #222;
            padding: 20px;
            border-radius: 15px;
            font-size: 1.5rem;
            text-align: center;
            font-weight: bold;
            color: #61dafb;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 class='title'>⚡ EV Charging Prediction AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered predictions for Electric Vehicle charging analysis</p>", unsafe_allow_html=True)
st.write("---")

# Layout: Split into Two Columns
col1, col2 = st.columns(2)

# User Inputs
input_data = {}
with col1:
    st.subheader("🔢 Enter Feature Values")
    for feature in feature_names:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, format="%.4f")

# Make Predictions
with col2:
    st.subheader("🎯 Prediction Output")
    if st.button("🚀 Predict Now"):
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]  # Ensure correct column order
        prediction = model.predict(input_df)
        
        # Display Prediction in a Styled Box
        st.markdown(f"<div class='prediction-box'>🔮 Prediction: {prediction[0]:.4f} kWh</div>", unsafe_allow_html=True)

st.write("---")
st.markdown("<p style='text-align: center; color: #a0a0a0;'>🚀 Built with ❤️ using Streamlit | AI-Powered ⚡</p>", unsafe_allow_html=True)
