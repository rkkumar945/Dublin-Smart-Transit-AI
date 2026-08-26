# 🚲 Dublin Smart Transit AI — Live Predictive Dashboard
https://dublin-smart-transit-ai-xenjgmqgfkvjturadmyqsm.streamlit.app/

A Streamlit dashboard that ingests live Dublin Bikes (GBFS) station data,
persists it to a cloud Postgres database, and uses a Random Forest model to
predict near-term bike availability and transit delay per station.

> **Status:** Personal / portfolio project. Not affiliated with Dublin City
> Council or the official Dublin Bikes service.

---

## 📋 Table of Contents
- [What This Project Does](#-what-this-project-does)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [Environment Variables & Secrets](#-environment-variables--secrets)
- [Running the App](#-running-the-app)
- [Known Limitations](#-known-limitations)
- [Security Notes](#-security-notes)
- [License](#-license)

---

## 🚀 What This Project Does

1. **Fetches live data** every 30 seconds from Dublin's public GBFS bike-share
   feed (station status: bikes available, docks available, last reported time).
2. **Validates the data** against a strict schema (using [Pandera](https://pandera.readthedocs.io/))
   before trusting it, so a malformed or changed API response doesn't crash the app.
3. **Persists it** to a managed PostgreSQL database (e.g. [Neon](https://neon.tech)),
   with a local CSV file as an automatic fallback/backup.
4. **Trains two Random Forest models** on the accumulated history — one predicts
   bike availability, the other predicts expected transit delay — and re-trains
   automatically whenever new data comes in.
5. **Serves an interactive dashboard** where you can pick a station and see its
   current stats plus the model's next-interval predictions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI / App framework | [Streamlit](https://streamlit.io/) |
| Data handling | Pandas, NumPy |
| Data validation | [Pandera](https://pandera.readthedocs.io/) (schema contracts) |
| Machine learning | scikit-learn (`RandomForestRegressor`, `train_test_split`) |
| Database | PostgreSQL (tested with [Neon](https://neon.tech)), via SQLAlchemy |
| Live data source | [Dublin Bikes GBFS feed](https://api.cyclocity.fr/contracts/dublin/gbfs/station_status.json) |
| Fallback storage | Local CSV file |

---

## 🗂️ Project Structure

```text
Dublin-Smart-Transit-AI/
├── .streamlit/
│   ├── config.toml              # Theme + server settings (CSRF protection enabled)
│   └── secrets.toml.example     # Template for local secrets — copy to secrets.toml
├── .gitignore                   # Excludes secrets.toml, __pycache__, *.db, etc.
├── app.py                       # Main app: ingestion → validation → DB write → ML → UI
├── Core.ipynb                   # Exploratory/practice notebook (pandas + SQL practice).
│                                 # Not part of the production pipeline — kept for reference only.
├── Dublin_Live_Transit_Master.csv  # Local fallback data store (auto-updated by app.py)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Dublin GBFS API │ ──▶ │ Pandera Validation│ ──▶ │  PostgreSQL + CSV  │
│ (every 30s fetch)│     │  (schema contract) │     │   (persistence)    │
└─────────────────┘     └──────────────────┘     └─────────┬──────────┘
                                                             │
                                                             ▼
                                              ┌───────────────────────────┐
                                              │  Random Forest training    │
                                              │  (auto-retrains on new     │
                                              │   data via a data          │
                                              │   fingerprint cache key)   │
                                              └─────────────┬─────────────┘
                                                             │
                                                             ▼
                                              ┌───────────────────────────┐
                                              │   Streamlit Dashboard UI   │
                                              │ (station picker + metrics  │
                                              │  + predictions)            │
                                              └───────────────────────────┘
```

If the live API call fails, the app falls back to the most recent data already
saved in PostgreSQL; if the database is also unavailable, it falls back further
to the local CSV backup — so a single point of failure doesn't take the whole
dashboard down.

---

## 🏁 Getting Started

### Prerequisites
- Python 3.9 or newer
- A PostgreSQL database (a free [Neon](https://neon.tech) instance works well),
  **or** you can run without one — the app will automatically fall back to the
  local CSV file.

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd Dublin-Smart-Transit-AI
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables & Secrets

The app needs a `DATABASE_URL` to connect to PostgreSQL. **Never commit a real
connection string to the repository.** Choose one of the two options below.

**Option A — Streamlit secrets file (local development):**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Then edit `.streamlit/secrets.toml` and fill in your real connection string:
```toml
DATABASE_URL = "postgresql://<user>:<password>@<host>/<database>?sslmode=require"
```
This file is already listed in `.gitignore` and will not be committed.

**Option B — Environment variable (recommended for deployment):**
```bash
export DATABASE_URL="postgresql://<user>:<password>@<host>/<database>?sslmode=require"
```
On hosting platforms (Render, Streamlit Community Cloud, etc.), set this as a
secret/environment variable in the platform's dashboard — never in code.

If neither is set, the app will show a warning in the sidebar and automatically
run in **local CSV-only mode**.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints in your terminal (usually `http://localhost:8501`).
The dashboard auto-refreshes its data every 30 seconds; you can also trigger an
immediate refresh with the **"🔄 Force Refresh Database Now"** button in the sidebar.

---

## ⚠️ Known Limitations

- **Simple feature set:** predictions currently use only `Hour` and
  `num_docks_available` as inputs. Weather, day-of-week, and station-specific
  seasonality are not yet modeled.
- **Synthetic delay labels:** `delay_minutes` is currently a rule-of-thumb
  placeholder (higher during 8–9 AM and 5–6 PM) rather than a real measured
  delay, since no ground-truth delay data source is wired in yet.
- **`Core.ipynb`** is a personal learning/practice notebook (pandas and SQL
  exercises) and is not used by `app.py` at runtime.
- This project is not officially affiliated with or endorsed by Dublin City
  Council or the Dublin Bikes scheme; it simply consumes their public GBFS feed.

---

## 🔒 Security Notes

- Database credentials are read only from an environment variable or Streamlit
  secrets — never hardcoded in `app.py`.
- `.gitignore` excludes `secrets.toml`, local `.db`/`.sqlite3` files, and
  Python cache directories.
- `enableXsrfProtection` is enabled in `.streamlit/config.toml`.
- If you ever suspect a database credential has been exposed (e.g. committed
  by mistake), rotate it immediately from your database provider's dashboard.

---

## 📄 License

This project is provided as-is for educational/portfolio purposes. Add a
license file (e.g. MIT) here if you intend to make the repository public and
accept external use or contributions.
