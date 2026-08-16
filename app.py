import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 1. page ka global configuration set karna (wide layout)
st.set_page_config(page_title="Dublin Smart Transit AI", layout="wide")

# 2. backend per hamari master csv file ko load karna.
df_master = pd.read_csv("Dublin_Live_Transit_Master.csv")

# ====================  ADVANCED MACHINE LEARNING ENGINE ====================
# projetc ke according random forest apply karna.
# features (X) or targets (y) ko alag karna.
X = df_master[['Hour', 'num_docks_available']]

# target 1:(Bikes Availability)
y_bikes = df_master['num_bikes_available']

# target 2: buses ki live delay (Predicted Delay in Minutes - simlulation based)
# agar aapke live data me delay ka column nahi hai , to hum ek domain-logic based
# live delay (Minutes) ka target variable sink kar rahe hai.
if 'delay_minutes' not in df_master.columns:
    # ek logical delay ka column banana: peak hours (morning 8, evening 5) per delay jyada hogi.
    df_master['delay_minutes'] = np.where(df_master['Hour'].isin([8, 17]), 15, 2)
y_delay = df_master['delay_minutes']

# 100 decision tree wala ek strong rendom forest model tyar karna.
model_bikes = RandomForestRegressor(n_estimators=100, random_state=42)
model_delay = RandomForestRegressor(n_estimators=100, random_state=42)

# dono advance models ko live data per train karna (.fit)
model_bikes.fit(X, y_bikes)
model_delay.fit(X, y_delay)

# ==================== SIDEBAR UI ARCHITECTURE ====================
station_list = df_master['station_id'].unique()

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.info("Select the transit parameters below to update the live AI engine projections.")
    
    # dropdown box
    selected_station = st.selectbox("Target Station ID:", station_list)
    
    st.markdown("---")
    st.caption("Developed under strict descriptive patterns for Dublin Infrastructure.")

# ==================== MAIN DISPLAY UI ARCHITECTURE ====================
st.title(" Dublin Smart Transit — Live Predictive Dashboard")
st.markdown("---")

# user ke choose kiye hue station ke base per data ko filter karna.
station_data = df_master[df_master['station_id'] == selected_station]

if not station_data.empty:
    current_hour = int(station_data['Hour'].values[0])
    available_bikes = int(station_data['num_bikes_available'].values[0])
    total_docks = int(station_data['num_docks_available'].values[0])
    
    # 3. live metrices cards (KPIs) dikhana.
    st.markdown("###  Current Station Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=" Available Bikes Right Now", value=available_bikes)
    with col2:
        st.metric(label="🅿️ Empty Docks / Stand Capacity", value=total_docks)

    st.markdown("---")
    
    # 4. AI pediction windpw box (project ke phase 7,task4 ke according dono prediction dikhana.)
    st.markdown("###  AI Predictive Intelligence Engine (Powered by Random Forest )")
    
    # live input vector banana.
    live_input = pd.DataFrame([{'Hour': current_hour, 'num_docks_available': total_docks}])
    
    #dono advance prediction nikalna.
    pred_bikes_raw = model_bikes.predict(live_input)[0]
    pred_delay_raw = model_delay.predict(live_input)[0]
    
    # boundary controls lagana (Safety Bounds)
    predicted_bikes = max(0, int(round(pred_bikes_raw)))
    predicted_delay = max(0, int(round(pred_delay_raw)))
    
    # do barabar side by side prediction box ko dikhana.
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.info(f" **Bike Availability Projection:**\nOur Machine Learning model predicts approximately **{predicted_bikes} bikes** will be available in the next interval.")
        
    with p_col2:
        # bilkul accurate"Predicted Delay: 12 Minutes" wala visual component.
        st.warning(f" **Transit Delay Projection:**\nPredicted Delay for vehicles passing through this zone: **{predicted_delay} Minutes**.")
    
    st.markdown("---")
    
    # 5. raw records grid ko expender ke andar dikhana.
    with st.expander(" View Raw Station Record Matrix"):
        st.dataframe(station_data, use_container_width=True)
else:
    st.error("No data found for this station.")
