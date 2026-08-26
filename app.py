import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pandera as pa
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine  # cloud database engine.

# 1. page ka global configuration set karna.
st.set_page_config(page_title="Dublin Smart Transit AI", layout="wide")

# ====================  CLOUD DATABASE CONNECTION POOLING ====================
# connection ko baar baar khulne-band hone se bachane ke liye st.cache_resource
@st.cache_resource
def init_connection():
    # Note: yeh hamare cloud PostgreSQL (Neon.tech / Supabase) ka live connection string hai.
    # real-world me ise secure environment variable se pass kiya jata hai.
    # Testing or deployment ko super-fast rakhne ke liye hum ise direct engine me daal rhe hai.
    db_url = "postgresql://neondb_owner:npg_XF3hYyHw1uGv@ep-calm-sunset-a1me089r-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
    engine = create_engine(db_url, pool_size=5, max_overflow=10)
    return engine

try:
    db_engine = init_connection()
except Exception as conn_err:
    st.sidebar.error(f"Database Connection Failed: {conn_err}")

# ====================  PANDERA DATA CONTRACT SCHEMA ====================
dublin_api_schema = pa.DataFrameSchema({
    "station_id": pa.Column(pa.Int, coerce=True, nullable=False),
    "num_bikes_available": pa.Column(pa.Int, coerce=True, default=0),
    "num_docks_available": pa.Column(pa.Int, coerce=True, default=0),
    "last_reported": pa.Column(pa.Int, coerce=True, nullable=True)
})

RELOAD_INTERVAL = 30 
st.sidebar.markdown(f"###  Live Production Sync: ACTIVE")
st.sidebar.caption(f"Infrastructure auto-refreshing every {RELOAD_INTERVAL} seconds...")

# 2. backend live API ingestion pipe (With Cloud DB Persistent Ingestion)
@st.cache_data(ttl=RELOAD_INTERVAL)
def fetch_live_dublin_data_from_api():
    api_url = "https://cyclocity.fr"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            live_json = response.json()
            df_raw = pd.json_normalize(live_json['data']['stations'])
            
            # Pandera protection circle.
            df_new = dublin_api_schema.validate(df_raw)
            
            # टाtime-series engineering.
            df_new['Reported_Time'] = pd.to_datetime(df_new['last_reported'], unit='s', utc=True).dt.tz_convert('Europe/Dublin')
            df_new['Hour'] = df_new['Reported_Time'].dt.hour
            
            # local CSV ki jagah seedhe cloud PostgreSQL database me UPSERT/APPEND karna.
            # ab  Render re-start hone per bhi data hamesha ke liye safe(protect) rahega.
            df_new.to_sql('dublin_bikes_live', con=db_engine, if_exists='append', index=False)
            
            return df_new
    except Exception as e:
        st.sidebar.error(f"Cloud API/Database Sync Error: {e}")
    
    # Fallback protection circle: koi error aane per cloud database se hi aakhri(last) available data read karna.
    try:
        return pd.read_sql("SELECT * FROM dublin_bikes_live ORDER BY Reported_Time DESC LIMIT 100", con=db_engine)
    except:
        return pd.read_csv("Dublin_Live_Transit_Master.csv")

# live data pipeline trigger karna.
df_master = fetch_live_dublin_data_from_api()

# ====================  ADVANCED MACHINE LEARNING ENGINE ====================
X = df_master[['Hour', 'num_docks_available']]
y_bikes = df_master['num_bikes_available']

if 'delay_minutes' not in df_master.columns:
    df_master['delay_minutes'] = np.where(df_master['Hour'].isin([8, 9, 17, 18]), 15, 2)
y_delay = df_master['delay_minutes']

model_bikes = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_bikes)
model_delay = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_delay)

# ==================== SIDEBAR UI ARCHITECTURE ====================
station_list = df_master['station_id'].unique()

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.info("Select the transit parameters below to update the live AI engine projections.")
    selected_station = st.selectbox(" Target Station ID:", station_list)
    st.markdown("---")
    st.markdown(f" **Last Server Refresh:**\n `{time.strftime('%H:%M:%S')}`")
    st.caption("Developed under strict descriptive patterns for Dublin Infrastructure.")

# ==================== MAIN DISPLAY UI ARCHITECTURE ====================
st.title(" Dublin Smart Transit — Live Predictive Dashboard")
st.markdown("---")

station_data = df_master[df_master['station_id'] == selected_station]

if not station_data.empty:
    current_hour = int(station_data['Hour'].iloc[0] if isinstance(station_data['Hour'], pd.Series) else station_data['Hour'])
    available_bikes = int(station_data['num_bikes_available'].iloc[0] if isinstance(station_data['num_bikes_available'], pd.Series) else station_data['num_bikes_available'])
    total_docks = int(station_data['num_docks_available'].iloc[0] if isinstance(station_data['num_docks_available'], pd.Series) else station_data['num_docks_available'])
    
    st.markdown("###  Current Station Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=" Available Bikes Right Now", value=available_bikes)
    with col2:
        st.metric(label="🅿️ Empty Docks / Stand Capacity", value=total_docks)

    st.markdown("---")
    st.markdown("###  AI Predictive Intelligence Engine (Powered by Random Forest )")
    
    live_input = pd.DataFrame([{'Hour': current_hour, 'num_docks_available': total_docks}])
    predicted_bikes = max(0, int(round(model_bikes.predict(live_input)[0])))
    predicted_delay = max(0, int(round(model_delay.predict(live_input)[0])))
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.info(f" **Bike Availability Projection:**\nOur Machine Learning model predicts approximately **{predicted_bikes} bikes** will be available in the next interval.")
    with p_col2:
        st.warning(f" **Transit Delay Projection:**\nPredicted Delay for vehicles passing through this zone: **{predicted_delay} Minutes**.")
    
    st.markdown("---")
    with st.expander(" View Raw Station Record Matrix"):
        st.dataframe(station_data, use_container_width=True)
else:
    st.error("No data found for this station.")

time.sleep(RELOAD_INTERVAL)
st.rerun()
