import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
import pytz

# Load models
forest_fire_model = pickle.load(open('model.pkl', 'rb'))
earthquake_model = joblib.load("earthquake_model_25.pkl")

# Load world cities dataset
@st.cache_data
def load_city_data():
    return pd.read_csv("worldcities.csv")[["country", "city", "lat", "lng"]]

# Streamlit App Config
st.set_page_config(page_title="🌋 Disaster Risk App", layout="centered", page_icon="🌋")
cities_df = load_city_data()


# Fetch average depth for earthquakes
def fetch_avg_depth(lat, lng):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=90)
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 3.0,
        "maxradiuskm": 300,
        "latitude": lat,
        "longitude": lng,
    }
    response = requests.get(url, params=params)
    data = response.json()
    features = data.get("features", [])
    depths = [feature["geometry"]["coordinates"][2] for feature in features[:100] if
              feature["geometry"]["coordinates"][2] is not None]
    return sum(depths) / len(depths) if depths else None

# Fetch recent earthquake data
def fetch_live_earthquake_data(lat, lng):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=90)
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 3.0,
        "maxradiuskm": 300,
        "latitude": lat,
        "longitude": lng,
    }
    response = requests.get(url, params=params)
    data = response.json()
    features = data.get("features", [])
    earthquakes = [{
        "TIME": datetime.utcfromtimestamp(f["properties"]["time"] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
        "MAGNITUDE": f["properties"]["mag"],
        "DEPTH": f["geometry"]["coordinates"][2],
        "PLACE": f["properties"]["place"],
        "STATUS": "🟢 Safe" if f["properties"]["mag"] < 4.5 else (
            "🟡 Moderate Risk" if 4.5 <= f["properties"]["mag"] < 6.0 else "🔴 High Risk"),
    } for f in features]
    return pd.DataFrame(earthquakes)


# Sidebar
st.sidebar.header("🌍 Select Prediction Model")
selected_model = st.sidebar.selectbox("🔮 Choose Prediction Type",
                                      ("Select an option", "Forest Fire Prediction", "Earthquake Prediction"))

# Show title only if no model is selected
if selected_model == "Select an option":
    st.markdown("<h1 style='text-align: center;'> 🔴Live Natural Disaster Risk Predictor</h1>", unsafe_allow_html=True)

# Common dropdowns
selected_country = st.sidebar.selectbox("🌎 Select Country",
                                        ["Select a country"] + sorted(cities_df["country"].unique()))
filtered_cities = cities_df[
    cities_df["country"] == selected_country] if selected_country != "Select a country" else pd.DataFrame()
selected_city = st.sidebar.selectbox("🏙️ Select City", ["Select a city"] + sorted(
    filtered_cities["city"].unique()) if not filtered_cities.empty else ["Select a city"])

local_time = "Please select Location !"
if selected_city and selected_city != "Select a city":
    city_info = filtered_cities[filtered_cities["city"] == selected_city].iloc[0]
    latitude, longitude = city_info["lat"], city_info["lng"]
    st.sidebar.write(f"📍 **Latitude:** {latitude:.6f}, **Longitude:** {longitude:.6f}")

    # Detect timezone
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=longitude, lat=latitude)
    if timezone_str:
        local_time = datetime.now(pytz.timezone(timezone_str)).strftime("%Y-%m-%d %H:%M:%S")

if selected_model == "Forest Fire Prediction":
    st.title("🔥 Forest Fire Prediction")
    st.info(f"🕒 **Local Time:** {local_time}")
    if st.sidebar.button("🔍 Predict Fire Risk"):
        # Fetch weather data
        api_key = '7f2e881529c54a67892192355252303'  # Replace with your API key
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={selected_city}"
        try:
            response = requests.get(url)
            data = response.json()
            if "current" in data:
                temp = data['current']['temp_c']
                humidity = data['current']['humidity']
                oxygen = round(20.95 + np.random.uniform(-0.5, 0.5), 2)
                st.write(f"🌡️ Temperature: {temp} °C")
                st.write(f"💧 Humidity: {humidity} %")
                st.write(f"🧪 Oxygen Level: {oxygen} %")

                # Prediction
                input_data = np.array([[oxygen, temp, humidity]])
                prediction = forest_fire_model.predict_proba(input_data)
                output = round(prediction[0][1], 2)
                if output > 0.5:
                    st.error(f"🔥 Forest is in Danger! Fire probability: {output}")
                else:
                    st.success(f"✅ Forest is Safe! Fire probability: {output}")
            else:
                st.warning("⚠️ City not found or API limit reached!")
        except Exception as e:
            st.error(f"API Error: {e}")

elif selected_model == "Earthquake Prediction":
    st.title("🌍 Earthquake Prediction")
    st.info(f"🕒 **Local Time:** {local_time}")
    avg_depth = fetch_avg_depth(latitude, longitude) if selected_city != "Select a city" else None
    if avg_depth is not None:
        st.sidebar.write(f"🔽 **Avg Depth (300km Range):** {avg_depth:.2f} km")

    if st.sidebar.button("🔍 Predict Earthquake Risk"):
        input_data = pd.DataFrame([[avg_depth if avg_depth else 10, latitude, longitude]],
                                  columns=["Depth (km)", "Latitude", "Longitude"])
        magnitude = earthquake_model.predict(input_data)[0]
        probability = min(max((magnitude - 3) / 7, 0), 1) * 100
        status = "✅ Area is SAFE" if magnitude < 4.5 else ("⚠️ Area in MODERATE RISK" if 4.5 <= magnitude < 6.0 else "🔴 AREA in HIGH RISK")

        st.success(f"🌍 **Predicted Magnitude:** {magnitude:.2f}")
        st.info(f"⚠️ **Earthquake Probability:** {probability:.2f}%")
        if magnitude < 4.5:
            st.success(status)
        elif 4.5 <= magnitude < 6.0:
            st.warning(status)
        else:
            st.error(status)

        st.header(f"📊 Recent Earthquakes near {selected_city}, {selected_country}")
        st.write(f"🕒 **Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        live_data = fetch_live_earthquake_data(latitude, longitude)
        if not live_data.empty:
            st.dataframe(live_data.head(10))
        else:
            st.write("✅ No significant earthquakes detected in the past 3 months.")
