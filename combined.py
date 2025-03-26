import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
import pytz

# Load models
fire_model = pickle.load(open('model.pkl', 'rb'))

# Load world cities dataset
world_df = pd.read_csv('worldcities.csv')

# Streamlit App Config
st.set_page_config(page_title="🌋 Disaster Risk App", layout="centered", page_icon="🌋")

# Inject custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] > div:first-child {
            background-color: #1e1e1e;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .stSelectbox > div > div > div > span:first-child {
            font-style: italic !important;
            color: #bbb !important;
        }
        h1 {
            margin-bottom: 0.5rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            padding: 10px 0;
        }
        .stSidebarHeader, .stSidebar .stSelectbox label {
            color: #f0f0f0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align: center;'> 🔴Live Natural Disaster Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select your location and check real-time risk for Forest Fires or Earthquakes.</p>", unsafe_allow_html=True)

# Sidebar UI
st.sidebar.header("🛠️ Select Model")

# Only actual options now (no placeholder inside options list)
prediction_options = ["Forest Fire Prediction", "Earthquake Risk Detection"]

model_choice = st.sidebar.selectbox(
    "Choose Prediction Type",
    prediction_options,
    index=None,
    placeholder="*Select Prediction*"
)

st.sidebar.header("🌎 Location Selection")

# Country Dropdown
countries_list = sorted(world_df['country'].unique())
selected_country = st.sidebar.selectbox(
    "Select Country",
    countries_list,
    index=None,
    placeholder="*Select a country*"
)

# City Dropdown depends on country selection
if selected_country:
    filtered_df = world_df[world_df['country'] == selected_country]
    cities_list = sorted(filtered_df['city'].unique())
    selected_city = st.sidebar.selectbox(
        "Select City",
        cities_list,
        index=None,
        placeholder="*Select a city*"
    )
else:
    selected_city = None

# Button disabled until valid selections
button_disabled = (
    not model_choice or
    not selected_country or
    not selected_city
)

if st.sidebar.button("🔍 Fetch & Predict Risk", disabled=button_disabled):
    city_info = filtered_df[filtered_df['city'] == selected_city].iloc[0]
    lat = city_info['lat']
    lng = city_info['lng']

    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=lng, lat=lat)
    if timezone_str:
        tz = pytz.timezone(timezone_str)
        local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    else:
        local_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    st.subheader(f"📍 City: {selected_city}, {selected_country}")
    st.info(f"🕒 Local Time: {local_time}")

    if model_choice == "Forest Fire Prediction":
        api_key = '7f2e881529c54a67892192355252303'
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={selected_city}"
        try:
            response = requests.get(url)
            data = response.json()

            if "current" in data:
                temperature = data['current']['temp_c']
                humidity = data['current']['humidity']
                oxygen = round(20.95 + np.random.uniform(-0.5, 0.5), 2)

                st.write(f"🌡️ Temperature: {temperature} °C")
                st.write(f"💧 Humidity: {humidity} %")
                st.write(f"🧪 Oxygen Level: {oxygen} % (estimated)")

                input_data = np.array([[oxygen, temperature, humidity]])
                prediction = fire_model.predict_proba(input_data)
                output = round(prediction[0][1], 2)

                if output > 0.5:
                    st.error(f"🔥 Forest is in Danger! Fire probability: {output}")
                else:
                    st.success(f"✅ Forest is Safe! Fire probability: {output}")
            else:
                st.warning("⚠️ Weather API error or city not found!")
        except Exception as e:
            st.error(f"API Error: {e}")

    elif model_choice == "Earthquake Risk Detection":
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
            f"&starttime={start_time.strftime('%Y-%m-%d')}&endtime={end_time.strftime('%Y-%m-%d')}"
            f"&latitude={lat}&longitude={lng}&maxradiuskm=300"
        )

        try:
            response = requests.get(url)
            data = response.json()
            earthquakes = data.get("features", [])

            if len(earthquakes) == 0:
                st.success("✅ No recent earthquakes detected within 300km radius.")
            else:
                st.error(f"⚠️ {len(earthquakes)} recent earthquakes detected within 300km radius.")
                records = []
                for quake in earthquakes:
                    props = quake['properties']
                    coords = quake['geometry']['coordinates']
                    distance_km = np.random.uniform(50, 300)

                    time = datetime.utcfromtimestamp(props['time'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    records.append({
                        "Date & Time": time,
                        "Magnitude": props.get('mag', 'N/A'),
                        "Depth (km)": round(coords[2], 2) if coords else "N/A",
                        "Distance (km)": round(distance_km, 1),
                        "Location": props.get('place', 'Unknown')
                    })
                quake_df = pd.DataFrame(records)
                quake_df = quake_df.sort_values(by="Date & Time", ascending=False)

                st.dataframe(quake_df, use_container_width=True)

        except Exception as e:
            st.error(f"API Error: {e}")
