#task5 phase7
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. page ka configuration set karna (website ka tab naam or wide layout)
st.set_page_config(page_title="Dublin Transit AI", layout="wide")

# 2.backend per hamari master csv file ko load karna.
df_master = pd.read_csv("Dublin_Live_Transit_Master.csv")

# 3. backend per lightweight linear regression model ko live train karna.
X_train = df_master[['Hour', 'num_docks_available']]
y_train = df_master['num_bikes_available']
ai_model = LinearRegression()
ai_model.fit(X_train, y_train)

# 4. dataset me se sabhi unique station_id ki list nikalna.
station_list = df_master['station_id'].unique()

# ==================== SIDEBAR UI ARCHITECTURE ====================
# st.sidebar ka use karke sabhi input controls ko left side me shift karna.
with st.sidebar:
    st.image("https://wikimedia.org", width=100)
    st.markdown("## ⚙️ Control Panel")
    st.info("Select the transit parameters below to update the live AI engine projections.")
    
    # magical step: dropdown box ko sidebar ke andar banana.
    selected_station = st.selectbox(" Target Station ID:", station_list) #user jo bhi station choose karega uski value 'selected_station me save ho jaigi.
    
    st.markdown("---")
    st.caption(" Developed as part of the Dublin Smart Transit Capstone Project.")

# ==================== MAIN DISPLAY UI ARCHITECTURE ====================
# main screen ab keval clean or bade results dikhagi.
st.title(" Dublin Smart Transit — Live Predictive Dashboard")
st.markdown("---")

# user ke chune hue station ke base per data ko filter karna.
station_data = df_master[df_master['station_id'] == selected_station]

if not station_data.empty:
    current_hour = int(station_data['Hour'].values[0])
    available_bikes = int(station_data['num_bikes_available'].values[0])
    total_docks = int(station_data['num_docks_available'].values[0])
    
    # 5. live metrics cards (KPIs) dikhana.
    st.markdown("### Current Station Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=" Available Bikes Right Now", value=available_bikes)
    with col2:
        st.metric(label="🅿️ Empty Docks / Stand Capacity", value=total_docks)

    st.markdown("---")
    
    # 6. AI prediction window box ko dikhana.
    st.markdown("###  AI Predictive Intelligence Engine")
    live_input = pd.DataFrame([{'Hour': current_hour, 'num_docks_available': total_docks}])
    ai_prediction = ai_model.predict(live_input)
    predicted_bikes = max(0, int(round(ai_prediction[0])))
    
    st.info(f" **AI Prediction:** Based on current transit trends at Hour **{current_hour}**, our Machine Learning model predicts approximately **{predicted_bikes} bikes** will be available at this station in the next interval.")
    
    st.markdown("---")
    
    # 7. row record grid ko expender ke andar chupana (clean UI best practice)
    with st.expander("View Raw Station Record Matrix"):
        st.dataframe(station_data, use_container_width=True)
else:
    st.error("No data found for this station.")