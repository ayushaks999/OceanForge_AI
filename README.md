# 🌊 ARGO RAG Explorer

**Tagline:** *An intelligent geospatial and ML-driven explorer for ARGO oceanographic data, combining RAG, LLM reasoning, and scientific visualization.*

---

## 🚀 Overview

**ARGO RAG Explorer** is a full-stack **Streamlit application** designed for interactive exploration, analysis, and reasoning over **ARGO ocean profile datasets**. It combines **Relational Database Management**, **Machine Learning**, and **Retrieval-Augmented Generation (RAG)** to make oceanographic data accessible, explainable, and predictive.

The app integrates advanced tools — from **xarray** and **netCDF** parsing to **LangChain-based natural language querying** — enabling both data scientists and ocean researchers to intuitively query, visualize, and model ARGO float data.

---

## 🧠 Key Features

### 🔍 1. Intelligent Data Ingestion

* Fetches and parses official **IFREMER ARGO index files**.
* Converts raw `.nc` (NetCDF) profiles into structured relational data.
* Handles missing or malformed data gracefully with detailed logging.

### 🌍 2. Interactive Map Exploration

* **Nearest Float Search:** Find floats near a location or coordinates.
* **Dynamic Plotly Maps:** Visualize ARGO positions, trajectories, and profiles.
* Supports **place search** via OpenStreetMap’s Nominatim API.

### 🧩 3. Conversational RAG Querying

* Ask natural-language questions like:

  > “Show floats near the Bay of Bengal with temperature below 10°C.”
* Powered by **Google Gemini or local LLMs**, with **structured query parsing** fallback.
* Integrates **ChromaDB** for vector retrieval of ARGO profile summaries.

### ⚙️ 4. Data Processing & Exports

* Clean ingestion into SQLite/PostgreSQL databases.
* Downloadable exports in CSV, Parquet, and even reconstructed NetCDF.
* Includes background workers for long ingestion or embedding tasks.

### 📈 5. ML-Based Temperature Prediction

* Train custom regression models (RandomForest, XGBoost, LightGBM, HistGB, etc.).
* Feature engineering on geospatial and temporal attributes.
* Visualize prediction quality with RMSE, R², and scatter plots.
* Save, reload, and use models for **real-time predictions**.

### 🧭 6. Trajectories & Profile Comparison

* Compare multiple float trajectories side-by-side.
* Plot temperature-depth and time-series trends.
* Perform statistical aggregations (mean, median, etc.).

### 💬 7. Feedback & User Interaction

* Integrated feedback logging for RAG responses.
* Local user DB stores chat history and model evaluation metadata.

---

## 🧱 System Architecture

```text
+-----------------------------------------------------------+
|                       Streamlit UI                        |
|  Tabs: Nearest | Explore | Ingest | Chat | Compare | ML   |
+-----------------------------------------------------------+
            |                   |                  |
            ↓                   ↓                  ↓
     Data Layer (SQLite / Postgres)     ⟶   ML Engine (sklearn, xgboost)
            ↓                   ↑                  ↓
     netCDF Parser (xarray)  LLM/RAG Layer (Gemini, Chroma)
```

**Key Components:**

* **Database:** SQLite by default; PostgreSQL optional.
* **Storage:** Configurable local paths for index, profiles, and status files.
* **Models:** On-demand loading of Gemini API, embeddings, and Chroma vector stores.

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/argo-rag-explorer.git
cd argo-rag-explorer
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set up environment variables

Create a `.env` file with your API keys and paths:

```env
GEMINI_API_KEY=your_gemini_key_here
IFREMER_INDEX_URL=https://example.com/argo_index.csv.gz
SQLITE_PATH=argo.db
```

### 5️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📦 Example `.env`

```env
# LLMs and Embeddings
GEMINI_API_KEY=AIzaSy...
USE_CHROMADB=true

# Database and Paths
SQLITE_PATH=data/argo.db
PROFILE_STORAGE_PATH=data/profiles
INDEX_FILE_PATH=data/argo_index.csv.gz
```

---

## 🧩 Core Modules

| Module     | Description                                            |
| ---------- | ------------------------------------------------------ |
| `app.py`   | Streamlit UI and orchestration layer                   |
| `utils.py` | Helpers for parsing, ingestion, and data normalization |
| `ml.py`    | Machine learning models, pipelines, and training logic |
| `rag.py`   | LLM and RAG-based question answering (Gemini, Chroma)  |
| `db.py`    | Database setup and ORM helpers using SQLAlchemy        |

---

## 🧪 Supported Libraries

* **Core:** pandas, numpy, xarray, requests, streamlit
* **ML:** scikit-learn, xgboost, lightgbm, joblib
* **Visualization:** plotly, folium (optional)
* **RAG & LLM:** langchain, chromadb, google-generativeai
* **Database:** sqlalchemy, sqlite/postgres

---

## 🌐 Deployment

### Streamlit Cloud

Add your `.env` secrets in **Streamlit Cloud → Settings → Secrets**.

### Docker

```bash
docker build -t argo-rag-explorer .
docker run -p 8501:8501 argo-rag-explorer
```

### Local with PostgreSQL

Edit the `.env`:

```env
POSTGRES_URL=postgresql+psycopg2://user:password@localhost/argo
```

---

## 📊 Example Use Cases

| Use Case            | Description                                         |
| ------------------- | --------------------------------------------------- |
| Oceanographers      | Study thermal changes along ARGO trajectories       |
| Climate Researchers | Query temperature-depth profiles by region/time     |
| ML Engineers        | Build predictive models for temperature or salinity |
| Students            | Learn geospatial data pipelines & scientific ML     |

---

## 🧠 Future Roadmap

* [ ] Add fine-tuned RAG retrievers for oceanographic contexts.
* [ ] Integrate satellite overlays for sea-surface temperature.
* [ ] Add temporal animation of float movements.
* [ ] Deploy multi-user session management with authentication.

---

## 🤝 Contributors

**Author:** Ayush Kumar Shaw ([GitHub](https://github.com/ayushaks999))
B.Tech CSE, NIT Durgapur — Specializing in AI, ML, and Data Systems.

---

## 🏁 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

> *"Exploring the deep blue with data and intelligence."* 🌐
