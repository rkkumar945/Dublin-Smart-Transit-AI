import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestRegressor

# 1.page ka global configuration set karna (Wide Layout)
st.set_page_config(page_title="Dublin Smart Transit AI", layout="wide")

# ====================  GAP 3: LIVE AUTOMATION & SCHEDULING CORE ====================
# hamare data project condition ke according har 10 minute (600 sec) me data sink karne ka time circle.
# (note: testing ke liye ise 30 sec per set kar rahe hai taki aap live refresh dekh sake)
RELOAD_INTERVAL = 30 

st.sidebar.markdown(f"###  Live Production Sync: ACTIVE")
st.sidebar.caption(f"Infrastructure auto-refreshing every {RELOAD_INTERVAL} seconds...")

# backend live API ingestion pipeline (phase 4,pahse 8,task 3 wala work)
@st.cache_data(ttl=RELOAD_INTERVAL) # @st.cache_data ram level per performance optimize karta hai. 
def fetch_live_dublin_data_from_api():
    # dublin goverment ka bilkul live official API URL.
    api_url = "https://cyclocity.fr"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            live_json = response.json()
            # JSON KO seedhe live flat dataframe me change karna.
            df_new = pd.json_normalize(live_json['data']['stations'])
            
            # timestamp ko local dublin time me convert karna.
            df_new['Reported_Time'] = pd.to_datetime(df_new['last_reported'], unit='s', utc=True).dt.tz_convert('Europe/Dublin')
            df_new['Hour'] = df_new['Reported_Time'].dt.hour
            
            # is fresh data ko master csv me overwrite/update kar dena (Data Persistence)
            df_new.to_csv("Dublin_Live_Transit_Master.csv", index=False)
            return df_new
    except Exception as e:
        st.sidebar.error(f"Cloud API Sync Error: {e}")
    
    # agar internet ya API fail ho to local backup file load karna (Fallback Strategy)
    return pd.read_csv("Dublin_Live_Transit_Master.csv")

# live data pipeline ko trigger karke master dataframe load karna.
df_master = fetch_live_dublin_data_from_api()

# ====================  ADVANCED MACHINE LEARNING ENGINE ====================
X = df_master[['Hour', 'num_docks_available']]
y_bikes = df_master['num_bikes_available']

# buses ki live delay (Predicted Delay in Minutes - simulation based bussiness logic)
if 'delay_minutes' not in df_master.columns:
    df_master['delay_minutes'] = np.where(df_master['Hour'].isin([8, 9, 17, 18]), 15, 2)
y_delay = df_master['delay_minutes']

# 100 decision tree wale rendom forest models ko train karna.
model_bikes = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_bikes)
model_delay = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_delay)

# ==================== SIDEBAR UI ARCHITECTURE ====================
station_list = df_master['station_id'].unique()

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.info("Select the transit parameters below to update the live AI engine projections.")
    
    # dropdown box.
    selected_station = st.selectbox(" Target Station ID:", station_list)
    
    st.markdown("---")
    # yeh watch ki sui user ko live dikhagi ki backend data kab update hua.
    st.markdown(f" **Last Server Refresh:**\n `{time.strftime('%H:%M:%S')}`")
    st.caption(" Developed under strict descriptive patterns for Dublin Infrastructure.")

# ==================== MAIN DISPLAY UI ARCHITECTURE ====================
st.title(" Dublin Smart Transit — Live Predictive Dashboard")
st.markdown("---")

station_data = df_master[df_master['station_id'] == selected_station]

if not station_data.empty:
    current_hour = int(station_data['Hour'].values[0] if isinstance(station_data['Hour'].values, np.ndarray) else station_data['Hour'].values)
    available_bikes = int(station_data['num_bikes_available'].values[0] if isinstance(station_data['num_bikes_available'].values, np.ndarray) else station_data['num_bikes_available'].values)
    total_docks = int(station_data['num_docks_available'].values[0] if isinstance(station_data['num_docks_available'].values, np.ndarray) else station_data['num_docks_available'].values)
    
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

# time corcle: ye line browser khula rehne per har 30 second me poore page ko khud re-run kar degi.
time.sleep(RELOAD_INTERVAL)
st.rerun()
