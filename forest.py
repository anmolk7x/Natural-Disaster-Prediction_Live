import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Load world cities dataset
world_df = pd.read_csv('worldcities.csv')

# Streamlit App
st.set_page_config(page_title="🌲 Forest Fire Predictor", layout="centered", page_icon=":evergreen_tree:")

st.markdown("<h1 style='text-align: center;'>🌲 Forest Fire Prediction App (Real-Time)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select country and city to get real-time weather data and predict forest fire risk.</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🌍 Select Country & City")

# Country selection
countries = sorted(world_df['country'].unique())
selected_country = st.sidebar.selectbox("Select Country", countries)

# City selection based on country
filtered_df = world_df[world_df['country'] == selected_country]
cities = sorted(filtered_df['city'].unique())
selected_city = st.sidebar.selectbox("Select City", cities)

# Fetch button
if st.sidebar.button("🔍 Fetch & Predict Fire Risk"):
    st.subheader(f"🌆 City: {selected_city}, {selected_country}")

    # Get Latitude and Longitude for timezone detection
    city_info = filtered_df[filtered_df['city'] == selected_city].iloc[0]
    lat = city_info['lat']
    lng = city_info['lng']

    # Detect timezone
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=lng, lat=lat)
    if timezone_str:
        tz = pytz.timezone(timezone_str)
        local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"🕒 Local Time in {selected_city}: {local_time}")
    else:
        st.warning("⚠️ Timezone not found, showing UTC time.")
        local_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"🕒 UTC Time: {local_time}")

    # WeatherAPI call
    api_key = '7f2e881529c54a67892192355252303'  # Your API key here
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={selected_city}"

    try:
        response = requests.get(url)
        data = response.json()

        if "current" in data:
            temperature = data['current']['temp_c']
            humidity = data['current']['humidity']
            oxygen = round(20.95 + np.random.uniform(-0.5, 0.5), 2)  # Approx oxygen %

            st.write(f"🌡️ Temperature: {temperature} °C")
            st.write(f"💧 Humidity: {humidity} %")
            st.write(f"🧪 Oxygen Level: {oxygen} % (estimated)")

            # Model prediction
            input_data = np.array([[oxygen, temperature, humidity]])
            prediction = model.predict_proba(input_data)
            output = round(prediction[0][1], 2)

            if output > 0.5:
                st.error(f"🔥 Forest is in Danger! Fire probability: {output}")
            else:
                st.success(f"✅ Forest is Safe! Fire probability: {output}")

        else:
            st.warning("⚠️ City not found or API limit reached!")

    except Exception as e:
        st.error(f"API Error: {e}")
