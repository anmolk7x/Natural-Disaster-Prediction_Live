import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load both models
forest_model = pickle.load(open('model.pkl', 'rb'))
earthquake_model = pickle.load(open('earthquake_model.pkl', 'rb'))

# Pre-load necessary data for earthquake model (e.g., unique places)
dataset = pd.read_csv("Earthquake_dataset.csv")
dataset['place'] = dataset['place'].str.replace('^(.*of )', '', regex=True)
unique_places = dataset['place'].unique()

# Get the feature names the earthquake model was trained on
expected_cols = earthquake_model.estimators_[0].feature_names_in_

# Sidebar Dropdown
st.sidebar.markdown("<h2 style='font-size: 20px;'>🛰️ Select Prediction Task</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.selectbox(
    "",
    ["🌲 Forest Fire Prediction", "🌍 Earthquake Magnitude Prediction"]
)

# Forest Fire Prediction
if app_mode == "🌲 Forest Fire Prediction":
    st.markdown("<h1 style='font-size: 32px;'>🌲 Forest Fire Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Enter the environmental parameters to predict forest fire risk:</p>", unsafe_allow_html=True)

    # Now stacked vertically
    oxygen = st.number_input("🧪 Oxygen Level (%)", min_value=0, max_value=100, value=20, step=1)
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10, max_value=60, value=25, step=1)
    humidity = st.number_input("💧 Humidity (%)", min_value=0, max_value=100, value=50, step=1)

    if st.button("🔥 Predict Fire Risk"):
        input_data = np.array([[oxygen, temperature, humidity]])
        prediction = forest_model.predict_proba(input_data)
        output = round(prediction[0][1], 2)

        if output > 0.5:
            st.error(f"🔥 Forest is in Danger! Fire probability: {output}")
            st.warning("⚠️ Take Preventive Measures!")
        else:
            st.success(f"✅ Forest is Safe! Fire probability: {output}")

# Earthquake Magnitude Prediction
elif app_mode == "🌍 Earthquake Magnitude Prediction":
    st.markdown("<h1 style='font-size: 32px;'>🌍 Earthquake Magnitude Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>Enter earthquake event details to predict magnitude:</p>", unsafe_allow_html=True)

    # Also stacked vertically now
    nst = st.number_input("📡 NST (stations reporting)", min_value=0, max_value=500, value=20)
    gap = st.number_input("🧭 Gap (degrees)", min_value=0.0, max_value=360.0, value=50.0)
    rms = st.number_input("📊 RMS (Root Mean Square)", min_value=0.0, max_value=10.0, value=1.0)
    depth = st.number_input("🌐 Depth (km)", min_value=0.0, max_value=700.0, value=10.0)

    place = st.selectbox("📍 Place", unique_places)

    if st.button("🌍 Predict Magnitude"):
        input_df = pd.DataFrame({
            'nst': [nst],
            'gap': [gap],
            'rms': [rms],
            'depth': [depth],
            'place': [place]
        })

        input_df = pd.get_dummies(input_df, columns=['place'])

        # Add missing columns from model training
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure the order matches model training
        input_df = input_df[expected_cols]

        prediction = earthquake_model.predict(input_df)
        st.success(f"🔔 Predicted Earthquake Magnitude: {round(prediction[0][0], 2)}")
