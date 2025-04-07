import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
import pytz
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import time

# Load models
forest_fire_model = pickle.load(open('forestmodel.pkl', 'rb'))
earthquake_model = joblib.load("earthquake_model_25.pkl")

# Load world cities dataset
@st.cache_data
def load_city_data():
    return pd.read_csv("worldcities.csv", usecols=["country", "city", "lat", "lng"], encoding="latin-1")


# Streamlit App Config
st.set_page_config(page_title="Real-Time Natural Disaster Risk Prediction", layout="centered", page_icon="🌍")
cities_df = load_city_data()


# Common functions
def create_base_map(lat, lon, zoom=6):
    return folium.Map(location=[lat, lon], zoom_start=zoom, tiles='CartoDB positron')


def get_current_location():
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        response.raise_for_status()
        data = response.json()
        loc = data.get('loc', '0,0').split(',')
        return float(loc[0]), float(loc[1]), data.get('city', 'Unknown')
    except Exception as e:
        st.error(f"Location detection error: {str(e)}")
        return None, None, None


# Initialize session states
if 'earthquake_data' not in st.session_state:
    st.session_state.update({
        'earthquake_data': pd.DataFrame(),
        'fire_data': {},
        'eq_results': None,
        'current_location': None
    })

# Sidebar controls
st.sidebar.header("🌍 Prediction Settings")
use_current_location = st.sidebar.checkbox("Use Current Location", value=False)

if use_current_location:
    if not st.session_state.current_location:
        lat, lon, city = get_current_location()
        if lat and lon:
            st.session_state.current_location = {'lat': lat, 'lon': lon, 'city': city}
    if st.session_state.current_location:
        st.sidebar.write(f"📍 Detected Location: {st.session_state.current_location['city']}")
else:
    selected_country = st.sidebar.selectbox(
        "🌎 Select Country",
        ["Select a country"] + sorted(cities_df["country"].unique())
    )
    if selected_country != "Select a country":
        filtered_cities = cities_df[cities_df["country"] == selected_country]
        selected_city = st.sidebar.selectbox(
            "🏙️ Select City",
            ["Select a city"] + sorted(filtered_cities["city"].unique()))
    else:
        selected_city = None

# Main prediction parameters
selected_model = st.sidebar.selectbox(
    "🔎 Choose Prediction Type",
    ("Select an option", "Forest Fire Prediction", "Earthquake Prediction")
)


# Earthquake functions
def fetch_earthquake_data(lat, lng, radius_km=300, days=90):
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%d"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": 3.0,
            "maxradiuskm": radius_km,
            "latitude": lat,
            "longitude": lng,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("features", [])
    except Exception as e:
        st.error(f"Error fetching earthquake data: {str(e)}")
        return []


def process_earthquake_data(features):
    earthquakes = []
    for f in features:
        try:
            props = f.get("properties", {})
            coords = f.get("geometry", {}).get("coordinates", [0, 0, 0])
            earthquakes.append({
                "TIME": datetime.utcfromtimestamp(props.get("time", 0) / 1000),
                "MAGNITUDE": props.get("mag", 0),
                "DEPTH": coords[2] if len(coords) > 2 else 0,
                "LAT": coords[1] if len(coords) > 1 else 0,
                "LON": coords[0] if len(coords) > 0 else 0,
                "PLACE": props.get("place", "Unknown location"),
                "STATUS": ("🟢 Safe" if props.get("mag", 0) < 4.5 else
                           "🟡 Moderate Risk" if 4.5 <= props.get("mag", 0) < 6.0 else
                           "🔴 High Risk")
            })
        except Exception as e:
            st.error(f"Error processing earthquake feature: {str(e)}")
    return pd.DataFrame(earthquakes)


# Forest Fire functions
def get_weather_data(api_key, city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            'temp': data['current']['temp_c'],
            'humidity': data['current']['humidity'],
            'rain': data['current'].get('precip_mm', 0)
        }
    except Exception as e:
        st.error(f"Weather API Error: {str(e)}")
        return None


def create_fire_risk_map(lat, lon, risk_level):
    m = create_base_map(lat, lon)
    color = '#ff0000' if risk_level == 'High' else '#ffa500' if risk_level == 'Moderate' else '#00ff00'

    folium.Marker(
        [lat, lon],
        popup=f"Fire Risk: {risk_level}",
        icon=folium.Icon(color='white', icon_color=color, icon='fire', prefix='fa')
    ).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=5000,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.2
    ).add_to(m)

    return m


# Main app
if selected_model != "Select an option":
    if use_current_location and st.session_state.current_location:
        lat = st.session_state.current_location['lat']
        lon = st.session_state.current_location['lon']
        location_name = st.session_state.current_location['city']
    elif not use_current_location and selected_city and selected_city != "Select a city":
        city_info = filtered_cities[filtered_cities["city"] == selected_city].iloc[0]
        lat, lon = city_info["lat"], city_info["lng"]
        location_name = selected_city
    else:
        lat, lon = None, None
        location_name = None

    if lat and lon:
        # Timezone display
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=lon, lat=lat)
        if timezone_str:
            local_time = datetime.now(pytz.timezone(timezone_str)).strftime("%Y-%m-%d %H:%M:%S")
            st.info(f"🕒 Local Time: {local_time} ({timezone_str})")

        if selected_model == "Earthquake Prediction":
            st.title("🌍 Earthquake Prediction")

            # Earthquake specific sidebar parameters
            days_back = st.sidebar.slider("Data Period (days)", 1, 365, 90)
            radius_km = st.sidebar.slider("Search Radius (km)", 100, 1000, 300)

            if st.sidebar.button("🔍 Predict Earthquake Risk"):
                with st.spinner("Analyzing seismic data..."):
                    features = fetch_earthquake_data(lat, lon, radius_km, days_back)
                    live_data = process_earthquake_data(features)
                    st.session_state.earthquake_data = live_data

                    depths = [f["geometry"]["coordinates"][2] for f in features if
                              len(f.get("geometry", {}).get("coordinates", [])) > 2]
                    avg_depth = sum(depths) / len(depths) if depths else 10

                    input_data = pd.DataFrame([[avg_depth, lat, lon]],
                                              columns=["Depth (km)", "Latitude", "Longitude"])
                    magnitude = earthquake_model.predict(input_data)[0]
                    probability = min(max((magnitude - 3) / 7, 0), 1) * 100
                    status = ("✅ Area is SAFE" if magnitude < 4.5 else
                              "⚠️ Area in MODERATE RISK" if 4.5 <= magnitude < 6.0 else
                              "🔴 AREA in HIGH RISK")

                    st.session_state.eq_results = {
                        'magnitude': magnitude,
                        'probability': probability,
                        'status': status,
                        'quake_count': len(live_data)
                    }

            # Display persistent earthquake results
            if st.session_state.eq_results:
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Magnitude", f"{st.session_state.eq_results['magnitude']:.1f}")
                col2.metric("Risk Probability", f"{st.session_state.eq_results['probability']:.1f}%")
                col3.metric("Recent Quakes", st.session_state.eq_results['quake_count'])
                st.markdown(f"**Risk Assessment:** {st.session_state.eq_results['status']}")

                if not st.session_state.earthquake_data.empty:
                    st.subheader("🗺️ Recent Earthquake Activity Map")
                    with st.spinner("Rendering map..."):
                        time.sleep(0.5)
                        quake_map = create_base_map(lat, lon)
                        for _, quake in st.session_state.earthquake_data.iterrows():
                            color = ('green' if quake['STATUS'] == '🟢 Safe' else
                                     'orange' if quake['STATUS'] == '🟡 Moderate Risk' else
                                     'red')
                            folium.CircleMarker(
                                location=[quake['LAT'], quake['LON']],
                                radius=quake['MAGNITUDE'] * 1.5,
                                color=color,
                                fill=True,
                                fill_color=color,
                                popup=f"Mag: {quake['MAGNITUDE']}<br>Depth: {quake['DEPTH']}km",
                                tooltip=quake['PLACE']
                            ).add_to(quake_map)
                        st_folium(quake_map, width=700, height=500, key="earthquake_map")

                    st.subheader("📊 Seismic Activity Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Magnitude Distribution**")
                        fig, ax = plt.subplots()
                        ax.hist(st.session_state.earthquake_data['MAGNITUDE'], bins=10, edgecolor='black')
                        ax.set_xlabel("Magnitude")
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)

                    with col2:
                        st.write("**Depth vs Magnitude**")
                        fig, ax = plt.subplots()
                        ax.scatter(st.session_state.earthquake_data['MAGNITUDE'],
                                   st.session_state.earthquake_data['DEPTH'], alpha=0.6)
                        ax.set_xlabel("Magnitude")
                        ax.set_ylabel("Depth (km)")
                        st.pyplot(fig)

                    st.subheader("📜 Recent Earthquake Events")
                    # Remove Latitude and Longitude columns
                    earthquake_table = st.session_state.earthquake_data.drop(columns=['LAT', 'LON'])
                    st.dataframe(earthquake_table.sort_values('TIME', ascending=False).head(10))

                    csv = st.session_state.earthquake_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Earthquake Data",
                        data=csv,
                        file_name='earthquake_data.csv',
                        mime='text/csv',
                    )

        elif selected_model == "Forest Fire Prediction":
            st.title("🔥 Forest Fire Prediction")
            WEATHER_API_KEY = '7f2e881529c54a67892192355252303'

            if st.sidebar.button("🔍 Predict Fire Risk"):
                with st.spinner("Analyzing weather conditions..."):
                    weather_data = get_weather_data(WEATHER_API_KEY, location_name)

                    if weather_data:
                        oxygen = round(20.95 + np.random.uniform(-0.5, 0.5), 2)
                        input_data = np.array([[oxygen, weather_data['temp'], weather_data['humidity']]])
                        prediction = forest_fire_model.predict_proba(input_data)
                        fire_prob = round(prediction[0][1] * 100, 2)
                        risk_level = "High" if fire_prob > 50 else "Moderate" if fire_prob > 35 else "Low"

                        st.session_state.fire_data = {
                            'lat': lat,
                            'lon': lon,
                            'risk_level': risk_level,
                            'metrics': {
                                'temp': weather_data['temp'],
                                'humidity': weather_data['humidity'],
                                'oxygen': oxygen,
                                'probability': fire_prob
                            }
                        }

            # Display persistent fire results
            if st.session_state.fire_data:
                st.subheader("🔥 Fire Risk Assessment")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🌡️ Temperature", f"{st.session_state.fire_data['metrics']['temp']}°C")
                col2.metric("💧 Humidity", f"{st.session_state.fire_data['metrics']['humidity']}%")
                col3.metric("🫁 Oxygen", f"{st.session_state.fire_data['metrics']['oxygen']}%")
                col4.metric("📈 Fire Probability", f"{st.session_state.fire_data['metrics']['probability']}%")

                risk_color = "#ff0000" if st.session_state.fire_data['risk_level'] == 'High' else \
                    "#ffa500" if st.session_state.fire_data['risk_level'] == 'Moderate' else "#00ff00"
                st.markdown(f"""
                    <div style="background-color: {risk_color}; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: white;">{st.session_state.fire_data['risk_level']} Fire Risk</h2>
                    </div>
                """, unsafe_allow_html=True)

                st.subheader("🗺️ Fire Risk Map")
                with st.spinner("Rendering fire risk map..."):
                    time.sleep(0.5)
                    fire_map = create_fire_risk_map(
                        st.session_state.fire_data['lat'],
                        st.session_state.fire_data['lon'],
                        st.session_state.fire_data['risk_level']
                    )
                    st_folium(fire_map, width=700, height=500, key="fire_map")

                st.subheader("📋 Safety Recommendations")
                if st.session_state.fire_data['risk_level'] == 'High':
                    st.error("""
                    🚨 Immediate Action Required:
                    - Evacuate if recommended by authorities
                    - Report any signs of fire immediately
                    - Prepare emergency supplies
                    """)
                elif st.session_state.fire_data['risk_level'] == 'Moderate':
                    st.warning("""
                    ⚠️ Stay Alert:
                    - Monitor local fire warnings
                    - Clear vegetation around property
                    - Prepare evacuation plan
                    """)
                else:
                    st.success("""
                    ✅ Low Risk:
                    - Maintain fire safety protocols
                    - Keep fire extinguishers ready
                    - Stay informed about weather changes
                    """)

# Information expanders
with st.expander("📖 Understanding the Risk Levels"):
    st.markdown("""
    **Earthquake Risk Guide:**
    - 🟢 Safe (Magnitude < 4.5): Minor or no impact
    - 🟡 Moderate (4.5-6.0): Possible light damage
    - 🔴 High (≥6.0): Significant damage potential

    **Fire Risk Guide:**
    - 🔴 High (>50%): Immediate danger
    - 🟠 Moderate (30-50%): Increased caution needed
    - 🟢 Low (<30%): Normal precautions

    **Data Sources:**
    - Real-time earthquake data from USGS
    - Weather data from weatherapi.com
    - Predictive models trained on historical data
    """)


with st.expander("ℹ️ Application Guide"):
    st.markdown("""
    **How to Use:**
    1. Select prediction type from sidebar
    2. Choose to use current location or select manually
    3. Adjust parameters as needed
    4. Click prediction button
    5. Explore persistent results including maps

    **Features:**
    - Live location detection
    - Historical data analysis
    - Interactive visualizations
    - Safety recommendations
    - Data export capabilities

    **Note:** - Enable location access in browser for live detection
    - Maps may take 2-3 seconds to load
    - API keys required for weather data
    """)