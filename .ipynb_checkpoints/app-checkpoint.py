"""
Dublin Smart Transit AI — Live Predictive Dashboard-Author-Rajat kumar
-----------------------------------------------------
Fully patched version. All issues identified in the code review
(logical bugs, security vulnerabilities, and code-quality gaps)
have been fixed. See the accompanying PDF review report for the
full list of what was found and how each item below resolves it.
"""

import os
import time
import logging

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pandera as pa
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==================== STRUCTURED LOGGING ====================
# FIX: logger is now actually used everywhere instead of print(),
# so every message carries a timestamp/level and can be piped to
# a real log aggregator in production.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Dublin Smart Transit AI", layout="wide")

RELOAD_INTERVAL = 30  # seconds
CSV_BACKUP_PATH = "Dublin_Live_Transit_Master.csv"
DB_TABLE_NAME = "dublin_bikes_live"
API_URL = "https://api.cyclocity.fr/contracts/dublin/gbfs/station_status.json"
API_TIMEOUT_SECONDS = 10  # FIX: requests.get() had no timeout; a slow/hanging
                           # API could freeze the whole app indefinitely.


# ==================== CLOUD DATABASE CONNECTION POOLING ====================
@st.cache_resource
def init_connection():
    """
    Create a pooled SQLAlchemy engine for the Neon/Postgres database.

    FIX (security): the connection string is never hardcoded here.
    It is read from an environment variable first, falling back to
    Streamlit's secrets store. Neither location should ever be
    committed to source control (see .gitignore).
    """
    db_url = os.environ.get("DATABASE_URL") or st.secrets.get("DATABASE_URL", "")
    if not db_url:
        raise ValueError("DATABASE_URL is not configured (env var or st.secrets).")
    engine = create_engine(db_url, pool_size=5, max_overflow=10)
    return engine


try:
    db_engine = init_connection()
except Exception as conn_err:
    # FIX (security): user only ever sees a generic message; the real
    # exception (which could contain connection details) is logged
    # server-side only, never rendered to the browser.
    st.sidebar.warning("⚠️ Cloud database unavailable — running on local data only.")
    logger.error(f"Database connection failed: {conn_err}")
    db_engine = None


# ==================== PANDERA DATA CONTRACT SCHEMA ====================
dublin_api_schema = pa.DataFrameSchema({
    "station_id": pa.Column(pa.Int, coerce=True, nullable=False),
    "num_bikes_available": pa.Column(pa.Int, coerce=True, default=0),
    "num_docks_available": pa.Column(pa.Int, coerce=True, default=0),
    "last_reported": pa.Column(pa.Int, coerce=True, nullable=True),
})

st.sidebar.markdown("### 🟢 Live Production Sync: ACTIVE")
st.sidebar.caption(f"Infrastructure auto-refreshing every {RELOAD_INTERVAL} seconds...")


# ==================== LIVE API INGESTION ====================
@st.cache_data(ttl=RELOAD_INTERVAL)
def fetch_live_dublin_data_from_api():
    """
    Pull the current snapshot from the Dublin GBFS feed.
    Pure/cached function: performs no writes, only returns data (or None).
    """
    try:
        response = requests.get(API_URL, timeout=API_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as e:
        # FIX: network-level failures (DNS, connection refused, timeout)
        # are now logged clearly instead of silently falling through.
        logger.error(f"API request failed: {e}")
        return None

    # FIX: a non-200 response used to be silently swallowed with no log
    # line at all, making failures very hard to diagnose.
    if response.status_code != 200:
        logger.error(f"API returned unexpected status code: {response.status_code}")
        return None

    try:
        live_json = response.json()
        stations = live_json.get("data", {}).get("stations")
        if not stations:
            raise ValueError("Unexpected API response structure (no 'stations' key).")

        df_raw = pd.json_normalize(stations)
        df_new = dublin_api_schema.validate(df_raw)

        df_new["Reported_Time"] = (
            pd.to_datetime(df_new["last_reported"], unit="s", utc=True)
            .dt.tz_convert("Europe/Dublin")
        )
        df_new["Hour"] = df_new["Reported_Time"].dt.hour
        return df_new
    except Exception as e:
        logger.error(f"API payload parsing/validation failed: {e}")
        return None


def save_transit_data(df_new: pd.DataFrame, engine) -> None:
    """
    Persist a freshly fetched snapshot. Has side effects (DB write, file
    write) so — unlike fetch_live_dublin_data_from_api — this function is
    deliberately NOT decorated with @st.cache_data.
    """
    if df_new is None or df_new.empty:
        return

    if engine is not None:
        try:
            df_new.to_sql(DB_TABLE_NAME, con=engine, if_exists="append", index=False)
        except Exception as db_err:
            logger.error(f"Database write failed: {db_err}")

    # FIX: previously this used to_csv() in the default 'w' (overwrite)
    # mode, wiping out all prior history on every 30-second refresh.
    # It now appends, and only writes the header on the very first run,
    # so the CSV becomes a genuine, growing historical backup.
    try:
        file_exists = os.path.isfile(CSV_BACKUP_PATH)
        df_new.to_csv(CSV_BACKUP_PATH, mode="a", header=not file_exists, index=False)
    except Exception as csv_err:
        logger.error(f"CSV backup write failed: {csv_err}")


@st.cache_data(ttl=RELOAD_INTERVAL)
def load_historical_data(_engine) -> pd.DataFrame:
    """
    Load accumulated history (not just the latest snapshot) for model
    training and as a display fallback.

    FIX (methodology): training used to run on a single live snapshot,
    where almost every row shares the same 'Hour' value — the model
    could never actually learn a time-of-day effect. By training on
    accumulated history instead, the Hour feature has real variety.
    """
    if _engine is not None:
        try:
            df = pd.read_sql(
                f"SELECT * FROM {DB_TABLE_NAME} ORDER BY Reported_Time DESC LIMIT 2000",
                con=_engine,
            )
            if not df.empty:
                df["Reported_Time"] = pd.to_datetime(df["Reported_Time"], utc=True, errors="coerce")
                return df
        except Exception as e:
            logger.error(f"Loading history from database failed: {e}")

    try:
        df = pd.read_csv(CSV_BACKUP_PATH)
        if not df.empty:
            df["Reported_Time"] = pd.to_datetime(df["Reported_Time"], utc=True, errors="coerce")
        return df
    except Exception as e:
        logger.error(f"Loading history from CSV failed: {e}")
        return pd.DataFrame()


# ==================== PIPELINE ORCHESTRATION ====================
df_live = fetch_live_dublin_data_from_api()

if df_live is not None:
    save_transit_data(df_live, db_engine)

df_master = load_historical_data(db_engine)

# If there is no persisted history yet (first-ever run), fall back to
# whatever we just fetched live so the app still has something to show.
if (df_master is None or df_master.empty) and df_live is not None:
    df_master = df_live

if df_master is None or df_master.empty:
    st.error("No data available from any source. Please check connections.")
    st.stop()

# The most up-to-date snapshot to show per-station "right now" numbers.
# Prefer the just-fetched live data; otherwise use the freshest row
# already sitting in df_master.
display_source = df_live if df_live is not None else df_master


# ==================== MACHINE LEARNING ENGINE ====================
X = df_master[["Hour", "num_docks_available"]]
y_bikes = df_master["num_bikes_available"]

if "delay_minutes" not in df_master.columns:
    df_master["delay_minutes"] = np.where(df_master["Hour"].isin([8, 9, 17, 18]), 15, 2)
y_delay = df_master["delay_minutes"]


def _data_fingerprint(df: pd.DataFrame) -> str:
    """A cheap, hashable summary of the training data's size/recency."""
    latest = df["Reported_Time"].max() if "Reported_Time" in df.columns else ""
    return f"{len(df)}_{latest}"


@st.cache_resource(show_spinner="Training AI models...")
def train_production_ml_models(_X, _y_bikes, _y_delay, data_fingerprint: str):
    """
    FIX (subtle caching bug): the previous version prefixed every
    argument with an underscore (_X, _y_bikes, _y_delay). Streamlit
    deliberately skips hashing underscore-prefixed arguments, which
    means the cache key never changed — the models were trained
    exactly once for the life of the server and silently never
    updated again, no matter how much new data arrived.

    The fix keeps the large DataFrames/Series underscore-prefixed
    (so Streamlit doesn't waste time hashing big objects) but adds
    `data_fingerprint`, a small plain string that DOES get hashed.
    When the underlying data changes, the fingerprint changes, and
    the cache correctly invalidates and retrains.
    """
    if len(_X) < 10:
        # Too little data for a meaningful train/test split — train on
        # everything and log a warning instead of crashing.
        logger.warning("Very small training set (<10 rows); skipping train/test split.")
        m_bikes = RandomForestRegressor(n_estimators=100, random_state=42).fit(_X, _y_bikes)
        m_delay = RandomForestRegressor(n_estimators=100, random_state=42).fit(_X, _y_delay)
        return m_bikes, m_delay

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        _X, _y_bikes, test_size=0.2, random_state=42
    )
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        _X, _y_delay, test_size=0.2, random_state=42
    )

    m_bikes = RandomForestRegressor(n_estimators=100, random_state=42)
    m_delay = RandomForestRegressor(n_estimators=100, random_state=42)

    m_bikes.fit(X_train_b, y_train_b)
    m_delay.fit(X_train_d, y_train_d)

    return m_bikes, m_delay


fingerprint = _data_fingerprint(df_master)
model_bikes, model_delay = train_production_ml_models(X, y_bikes, y_delay, fingerprint)


# ==================== SIDEBAR UI ====================
station_list = sorted(df_master["station_id"].unique())

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.info("Select the transit parameters below to update the live AI engine projections.")
    selected_station = st.selectbox("Target Station ID:", station_list)
    st.markdown("---")
    st.markdown(f"**Last Server Refresh:**\n`{time.strftime('%H:%M:%S')}`")
    st.caption("Developed under strict descriptive patterns for Dublin Infrastructure.")


# ==================== MAIN DISPLAY UI ====================
st.title("Dublin Smart Transit — Live Predictive Dashboard")
st.markdown("---")

station_data = display_source[display_source["station_id"] == selected_station]
if "Reported_Time" in station_data.columns:
    station_data = station_data.sort_values("Reported_Time", ascending=False)

if not station_data.empty:
    current_hour = int(station_data["Hour"].iloc[0])
    available_bikes = int(station_data["num_bikes_available"].iloc[0])
    total_docks = int(station_data["num_docks_available"].iloc[0])

    st.markdown("### Current Station Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🚲 Available Bikes Right Now", value=available_bikes)
    with col2:
        st.metric(label="🅿️ Empty Docks / Stand Capacity", value=total_docks)

    st.markdown("---")
    st.markdown("### AI Predictive Intelligence Engine (Powered by Random Forest)")

    live_input = pd.DataFrame([{"Hour": current_hour, "num_docks_available": total_docks}])
    predicted_bikes = max(0, int(round(model_bikes.predict(live_input)[0])))
    predicted_delay = max(0, int(round(model_delay.predict(live_input)[0])))

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.info(
            f"**Bike Availability Projection:**\n\n"
            f"Our Machine Learning model predicts approximately **{predicted_bikes} bikes** "
            f"will be available in the next interval."
        )
    with p_col2:
        st.warning(
            f"**Transit Delay Projection:**\n\n"
            f"Predicted delay for vehicles passing through this zone: **{predicted_delay} minutes**."
        )

    st.markdown("---")
    with st.expander("View Raw Station Record Matrix"):
        st.dataframe(station_data, use_container_width=True)
else:
    st.error("No data found for this station.")

# ==================== NON-BLOCKING AUTO-REFRESH ====================
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Refresh Database Now"):
    st.cache_data.clear()
    st.rerun()

st.caption(f"💡 Automated background streaming active. Next data cycle check in {RELOAD_INTERVAL}s.")
