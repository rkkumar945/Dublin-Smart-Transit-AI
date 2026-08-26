# 🍀 Dublin Smart Transit — Real-Time Predictive AI & MLOps Infrastructure

<!-- LIVE PRODUCTION BADGES -->
[![Live Production Dashboard](https://shields.io)](https://onrender.com)
[![Production Source Code](https://shields.io)](https://github.com)
[![Data Endpoint](https://shields.io)](https://cyclocity.fr)

An enterprise-grade, high-availability data engineering and streaming pipeline designed to ingest live municipal transit streams, enforce runtime data contracts, and serve real-time machine learning predictions across Dublin City.

---

### ⏱️ THE 60-SECOND EXECUTIVE ARCHITECTURE SUMMARY
The end-to-end software architecture implements a resilient 5-tier production data lifecycle:

*   **⚡ Automated Streaming Ingestion:** Continuously scrapes live JSON data packets from the official Dublin Smart City GBFS API endpoint on an automated 30-second cron refresh interval.
*   **🛡️ Pandera Data Contract Shield:** Enforces strict runtime data contract schemas on raw data feeds to catch and handle upstream API column mutations and data type alterations without crashing.
*   **🔌 Persistent Cloud Database Hub:** Bypasses ephemeral cloud server resets by persisting validated records into a remote, fully managed **Neon PostgreSQL Cloud Database** backed by **SQLAlchemy Connection Pooling**.
*   **🌳 Dual-Output Ensemble Core:** Fits a non-linear **Random Forest Regressor** to evaluate historical traffic distributions and predict both *Vehicle Delay Projections (Minutes)* and *Bike Asset Availability* on the fly.
*   **📊 Optimized UI Serving Layer:** Dynamically serves predictions and business KPIs on a public interface built with Streamlit using memory-caching guardrails (`@st.cache_data`) and auto-trigger reruns (`st.rerun`).

---

## 🛠️ TECH STACK & GOVERNANCE INFRASTRUCTURE
*   **Data Pipelines & Engineering:** Python 3.x, Pandas Vectorization, NumPy, REST APIs
*   **Data Quality & Schema Governance:** Pandera (Schema Contract Invalidation & Imputation)
*   **Cloud Architecture & Storage:** Remote Neon PostgreSQL, SQLAlchemy (Pooling: 5-size, 10-overflow)
*   **Machine Learning Engineering:** Scikit-Learn (Random Forest Ensemble Models)
*   **Cloud Deployment & MLOps:** Streamlit UI Framework, Render App Cloud Infrastructure

---

## 🗂️ PRODUCTION FILE TREE
```text
Dublin-Smart-Transit-AI/
├── .streamlit/
│   └── config.toml                  # Global server configs, CORS, and custom UI design tokens
├── app.py                           # Master production script (Ingestion + Pandera Shield + ML + UI)
├── Dublin_Live_Transit_Master.csv   # Local fail-safe fallback storage database (Zero network backup)
├── requirements.txt                 # Absolute frozen pipeline dependencies and software packages
└── README.md                        # Senior Architecture Executive Documentation
```

---

## 🚀 LOCAL INSTALLATION & REPRODUCTION GUIDE
Execute these commands sequentially inside your local terminal workspace to spin up the container:

```bash
# Clone the secure cloud repository
git clone https://github.com.git

# Move into the project workspace directory
cd Dublin-Smart-Transit-AI

# Install all frozen software dependencies
pip install -r requirements.txt

# Launch the interactive local runtime server
python -m streamlit run app.py
```

---
*Developed under strict defensive programming and stateless cloud fault-tolerance patterns for the Dublin Infrastructure Analytics Portfolio.*
