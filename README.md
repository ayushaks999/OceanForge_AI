# 🌊 ARGO RAG Explorer — Interactive Ocean Data Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.9%2B-orange) ![Streamlit](https://img.shields.io/badge/streamlit-%3E%3D1.20-%23FF4B4B)

> **ARGO RAG Explorer** is a professional-grade **interactive oceanographic data exploration platform**. It integrates **ARGO float datasets, NetCDF processing, Retrieval-Augmented Generation (RAG), geospatial analysis, and conversational AI** into a single unified application.

---

## 🚀 Features

### 🔍 Data Indexing & Ingestion

* Automated ingestion of **ARGO float index files** (GDAC / AOML).
* Seamless handling of **NetCDF profiles** (`.nc`) with robust error handling.
* Structured storage in **SQLite database** with schema:

  * `argo_index` → metadata (float ID, position, date, file path).
  * `argo_info` → profile variables (temperature, salinity, pressure, etc.).
* Background workers for **asynchronous ingestion & updates**.

### 📊 Data Exploration

* **Index Browser**: filter ARGO floats by ID, parameter, date, and region.
* **Nearest Float Search**: query by lat/lon or place name (geocoding via Nominatim) with interactive map results.
* **Profile Comparison**: select multiple floats and compare:

  * Temperature vs Depth
  * Salinity vs Depth
  * Pressure vs Depth
  * Temperature evolution over time (shallowest, median, mean, or max profile values).
* **Trajectory Plots**: visualize float movement over time.

### 🧠 Conversational RAG (Retrieval-Augmented Generation)

* **LLM-powered querying** with optional integration of:

  * Google Gemini API
  * LangChain
  * ChromaDB (vector search)
* Hybrid answers combining:

  * Structured **SQL queries** over ARGO datasets.
  * **RAG-based summaries** enriched with metadata and context.
* Intelligent parsing of natural language into filters and database queries.
* Place lookup, automatic nearest-float context retrieval, and `.nc` previews.

### 🌐 Geospatial Intelligence

* Haversine-based **nearest float computation**.
* Interactive **Mapbox visualizations** for float locations, trajectories, and query results.

### 💾 Exports & Downloads

* Export query results and measurements as:

  * **CSV** (timeseries data)
  * **Parquet** (complete argo\_info)
  * **NetCDF** (curated dataset)

### 🧩 Advanced Architecture

* Modular design with reusable helpers:

  * Retry-enabled HTTP fetcher
  * Safe SQL builder
  * NetCDF parsers
  * Async job queue
* Robust error handling (file parsing, DB ingestion, LLM fallbacks).
* Clean separation of **tabs** for workflows:

  1. Nearest Floats
  2. Explore Index
  3. Ingest Profiles
  4. Chat (RAG)
  5. Trajectories & Profile Comparison
  6. Data Exports

---

## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/) with Plotly + Mapbox visualizations.
* **Backend:** Python, SQLite, SQLAlchemy ORM.
* **Data Processing:** xarray, netCDF4, pandas, numpy.
* **Geospatial:** geopy (Nominatim), haversine formula.
* **AI / RAG:**

  * Google Gemini API
  * LangChain + LangGraph
  * ChromaDB (optional vector database)
* **Misc:** dotenv, requests, tqdm.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/argo-rag-explorer.git
cd argo-rag-explorer

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
AGENTIC_RAG_STORAGE=./storage
```

---

## ▶️ Usage

Run the app locally:

```bash
streamlit run app.py --server.port 8501
```

Navigate to `http://localhost:8501` and explore:

* 📍 Nearest Floats → lookup by place or coordinates.
* 📊 Explore Index → filter and map floats.
* 📂 Ingest Profiles → add `.nc` profiles.
* 💬 Chat (RAG) → conversational ARGO assistant.
* 📈 Trajectories → visualize movement.
* 📤 Exports → download curated datasets.

---

## 📑 Example Workflows

### 1. Nearest Float Lookup

* Enter `Indian Ocean` or `Arabian Sea` → app finds nearest floats.
* Visualize results on an interactive map.

### 2. Compare Profiles

* Select multiple floats → compare **temperature, salinity, pressure** over depth.
* Download processed timeseries as CSV.

### 3. Chat with ARGO Data

* Ask: *"Find floats near Pacific Ocean in 2021 with temperature data."*
* System runs SQL + RAG queries → returns table + contextual summary.

---

## 📈 Roadmap

* [ ] Multi-user session management
* [ ] Persistent cache for processed NetCDF files
* [ ] Advanced RAG with hybrid retrievers
* [ ] Real-time streaming from live ARGO feeds
* [ ] Automated anomaly detection in profiles

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a PR.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* **ARGO Project** for global float datasets.
* **AOML / GDAC** for index and profile hosting.
* **Streamlit** for enabling rapid data apps.
* **LangChain & Google Gemini** for powering RAG-based exploration.
