# 🍀 Dublin Smart Transit — Live Predictive Analytics System

An enterprise-grade, real-time data engineering and machine learning streaming pipeline designed to predict bus delays and transit bike availability across Dublin City.

## 📊 Project Architecture
The end-to-end infrastructure follows a strict 4-tier software engineering methodology:
1. **Data Ingestion:** Automated scraping from official Dublin GBFS live production endpoints using `requests`.
2. **Database Governance:** Local staging, indexing, and aggregations handled via an encapsulated `SQLite3` database engine.
3. **Exploratory Analytics (EDA):** Statistical validation, skewness monitoring, and outlier detection using `Seaborn` histograms and box plots.
4. **Machine Learning Core:** Baseline forecasting system using specialized `LinearRegression` with feature importance metrics.
5. **Production UI:** Responsive web interface built with `Streamlit` utilizing dynamic caching algorithms (`@st.cache_data`).

## 🛠️ Tech Stack & Dependencies
- **Backend:** Python 3.x, Pandas, NumPy, SQLite3
- **Machine Learning:** Scikit-Learn (Model Evaluation Matrix)
- **UI Framework:** Streamlit Open-Source UI Components
- **Version Control:** Distributed Git Architecture

## 🚀 Installation & Local Deployment
To execute this production script on your local machine, run the following commands sequentially:

```bash
# Clone the repository
git clone https://github.com

# Navigate to project workspace
cd Dublin-Smart-Transit-AI

# Install explicit software dependencies
pip install -r requirements.txt

# Launch the interactive local server
python -m streamlit run app.py
```

---
*Developed under strict defensive programming patterns for the Dublin Transit Infrastructure Analytics Portfolio.*
