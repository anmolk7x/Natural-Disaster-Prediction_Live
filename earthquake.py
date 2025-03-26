import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz
import math

# Load world cities dataset
world_df = pd.read_csv('worldcities.csv')

# Streamlit UI setup
st.set_page_config(page_title="🌍 Earthquake Risk Detector", layout="wide", page_icon=":earth_asia:")
st.markdown("""
    <style>
        body {background-color: #0f1117; color: white;}
        .stButton>button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("🌍 Location Selection")
countries = sorted(world_df['country'].unique())
selected_country = st.sidebar.selectbox("Select Country", countries)
filtered_df = world_df[world_df['country'] == selected_country]
cities = sorted(filtered_df['city'].unique())
selected_city = st.sidebar.selectbox("Select City", cities)

# Fetch button
fetch = st.sidebar.button("🔍 Fetch & Predict Risk")

st.title("🌍 Earthquake Risk Detector")
st.write("Select your location to get recent earthquake info and check if it's safe to stay.")

# Get coordinates & timezone
if fetch:
    location_row = filtered_df[filtered_df['city'] == selected_city].iloc[0]
    lat, lng = location_row['lat'], location_row['lng']

    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=lng, lat=lat)
    tz = pytz.timezone(timezone_str) if timezone_str else pytz.utc
    local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    st.write(f"🕒 **Current Local Time:** {local_time}")

    # Fetch earthquakes from USGS
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&latitude={lat}&longitude={lng}&maxradiuskm=300&starttime={datetime.now().strftime('%Y-%m-%d')}T00:00:00&endtime={datetime.now().strftime('%Y-%m-%d')}T23:59:59"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        features = data.get('features', [])

        if len(features) == 0:
            st.success("✅ No recent earthquakes detected within 300km radius.")
        else:
            st.error("🚨 Earthquake(s) detected within 300km radius!")

        # Show earthquake history (last 30 days)
        history_url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&latitude={lat}&longitude={lng}&maxradiuskm=300&starttime={(datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')}"
        history_response = requests.get(history_url)

        if history_response.status_code == 200:
            history_data = history_response.json().get('features', [])
            if history_data:
                st.subheader("📊 Earthquake History (last 30 days within 300km)")

                records = []
                for eq in history_data:
                    props = eq['properties']
                    eq_time = datetime.utcfromtimestamp(props['time'] / 1000).replace(tzinfo=pytz.utc).astimezone(tz)
                    distance = math.sqrt((lat - eq['geometry']['coordinates'][1]) ** 2 + (lng - eq['geometry']['coordinates'][0]) ** 2) * 111
                    records.append({
                        "Date & Time": eq_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "Magnitude": props['mag'],
                        "Depth (km)": eq['geometry']['coordinates'][2],
                        "Distance (km)": round(distance, 1),
                        "Location": props['place']
                    })

                df = pd.DataFrame(records)
                st.dataframe(df)
            else:
                st.info("ℹ️ No historical earthquakes in the past 30 days within 300km radius.")
        else:
            st.warning("⚠️ Failed to fetch historical data.")

    else:
        st.error("❌ Failed to fetch recent earthquake data.")
