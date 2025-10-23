# 🌊 ARGO RAG Explorer — Full-feature README

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-app-orange.svg)](https://streamlit.io)

**Tagline:** *An enterprise-grade Streamlit application for ingestion, exploration, RAG-powered question answering, visualization, and ML modelling on ARGO oceanographic profile data.*

---

## 🔎 Project summary

ARGO RAG Explorer is a production-ready research and engineering toolkit built around ARGO float profile data (NetCDF). It ingests official IFREMER/ARGO index listings and `.nc` profile files, stores flattened per-profile measurements into a relational store, provides a human-friendly Streamlit UI for geospatial searches and interactive plotting, offers RAG-enabled natural-language queries (with optional LLMs & vector DB), and supports model training, export, and inference for oceanographic variables (e.g., temperature predictions).

This README documents architecture, installation, operational concerns, and advanced features — everything needed to run, extend, or deploy the project in research or engineering settings.

---

## 📂 What this repo contains

* `app.py` — Single-file Streamlit application (UI + orchestration). Can be split into modules for maintainability.
* Core helper functions (embedded in `app.py`):

  * netCDF parsing (xarray), robust variable detection, and per-sample row generation.
  * Index ingestion and remote download helpers (IFREMER index support).
  * SQLAlchemy schema/DB helpers for `argo_index` and `argo_info` tables and runtime migrations.
  * RAG helpers: context assembly, LLM prompts, Chroma vector retrieval integration.
  * Background workers for asynchronous index and chroma builds.
* ML training utilities (sklearn pipelines, safe wrappers for XGBoost/LightGBM, model persistence via joblib).
* Export utilities: Parquet and NetCDF generation from ingested rows.

> Note: The code was intentionally written to be robust across environments: many optional libraries (Chroma, Gemini LangChain adapter, Folium, XGBoost, LightGBM) are optional and gracefully degrade if not present.

---

## ⭐ Key Features (detailed)

### 1. Data ingestion

* **Index ingestion**: Download IFREMER index file, parse and persist to `argo_index` table.
* **NetCDF parsing**: Convert profile `.nc` into many `argo_info` rows (one per measurement) with rich metadata mapping (global attributes, calibration history, QC flags).
* **Schema**: `argo_index` (index metadata) + `argo_info` (per-profile measurements and metadata) — designed to work on SQLite and Postgres.
* **Idempotent ingestion**: Partial/migration helpers to add missing columns to `argo_info` when the schema evolves.

### 2. Interactive exploration & visualization

* **Nearest floats**: Haversine-based nearest float search.
* **Place geocoding**: Integrated Nominatim geocoder with a small catalog for common seas/regions.
* **Grid-based clustering**: Fast cluster-on-the-fly for map visualizations to keep interactive performance.
* **Plotly + Mapbox**: Rich interactive maps for points and trajectories.
* **NC previews**: On-demand `.nc` preview parsing for accurate variable views when DB rows are not yet present.

### 3. Conversational RAG (Retrieval-Augmented Generation)

* **LLM integration**: Optional Gemini (via `langchain_google_genai`) — structured parser fallback if LLM unavailable.
* **Context assembly (MCP)**: Combine index samples, `.nc` previews, and vector hits into a concise context for LLM prompts.
* **Chroma vector store**: Optional vector retrieval to find semantically similar profiles.
* **Structured JSON responses**: LLM responses are parsed to JSON to extract recommended SQL or final answers.

### 4. ML / Modeling

* **Training UI**: Train multiple tree-based regressors with configurable hyperparameters.
* **Safe wrappers**: Encapsulates XGBoost / LightGBM compatibility issues with sklearn by lazy instantiation and tag-safe wrappers.
* **Model persistence**: Save best and all models; load for inference in-session.
* **Inference UI**: Predict temperature for single samples and evaluate saved models on DB samples.

### 5. Exports & reproducibility

* **Parquet and NetCDF exports** from `argo_info` for portability.
* **Model artifacts** saved via `joblib`.

---

## 🏗️ Architecture & Dataflow

```
[IFREMER index (remote)] → (ensure_index_file) → index_local.txt
  └→ parse_index_file → argo_index (DB)

[.nc files] → (download_netcdf_for_index_path) → local .nc → parse_profile_netcdf_to_info_rows → argo_info (DB)

Streamlit UI → (ask_argo_question) → safe_sql_builder → SQL → DB (argo_index/argo_info)
                      ↘ optional: read_netcdf_variables_to_df for direct previews

LLM/Embeddings (optional): Gemini ↔ Embeddings → ChromaDB ↔ assemble_mcp_context → rag_answer_with_mcp

ML: read argo_info → feature engineering → sklearn/xgboost/lightgbm pipelines → train/save model
```

---

## 🛠️ Quickstart — Local Development

### 1. Prerequisites

* Python 3.8+ (3.10/3.11 recommended)
* Optional: PostgreSQL for production-scale DB
* Disk space for downloaded `.nc` files (depends on index subset)

### 2. Clone & virtualenv

```bash
git clone https://github.com/your-username/argo-rag-explorer.git
cd argo-rag-explorer
python -m venv venv
source venv/bin/activate
```

### 3. Minimal `requirements.txt` (suggested)

```
streamlit
pandas
numpy
xarray
requests
sqlalchemy
python-dotenv
plotly
joblib
scikit-learn
netcdf4
```

Optional (install if you need features):

```
langchain-google-genai
chromadb
xgboost
lightgbm
folium
streamlit-folium
```

### 4. Environment variables (`.env`)

```env
# Storage / DB
ARGO_SQLITE_PATH=./data/argo.db
AGENTIC_RAG_STORAGE=./storage
AGENTIC_RAG_DB_PATH=./storage/agentic_rag_meta.db

# IFREMER index URL (defaults included)
IFREMER_INDEX_URL=https://data-argo.ifremer.fr/ar_index_global_prof.txt
IFREMER_BASE=https://data-argo.ifremer.fr/dac

# Optional LLM / Embeddings
GEMINI_API_KEY=your_gemini_key_here
ARGO_PG_URL=postgresql+psycopg2://user:pass@host:5432/dbname
```

### 5. Run the app

```bash
streamlit run app.py
```

Visit `http://localhost:8501`.

---

## ⚙️ Configuration & Tuning

* **Storage location**: `AGENTIC_RAG_STORAGE` — where `.nc` files and Chroma DB will be persisted.
* **DB selection**: If `ARGO_PG_URL` is set, the app will use PostgreSQL. Otherwise it defaults to SQLite at `ARGO_SQLITE_PATH`.
* **Concurrency & background tasks**: Long tasks (index ingest, chroma build) are executed in separate `multiprocessing.Process`. On low-memory systems, reduce batch sizes or run sequentially to avoid swapping.

---

## ✅ Recommended Deployment Options

1. **Streamlit Cloud / Streamlit for Teams** — simplest for interactive demos & small datasets.
2. **Docker** (recommended for reproducibility). Example Dockerfile snippet:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

3. **Cloud VM (GCP/AWS/Azure)** behind a basic Nginx reverse proxy for production access. Use Postgres for DB and a managed volumes for storage.
4. **Kubernetes** — package as a deployment with a PersistentVolume for storage and a separate job for large index/chroma builds.

---

## 🔒 Security & Data Governance

* **API keys**: Never hardcode API keys. Use `.env` for local or platform secrets for deployment.
* **Public data**: ARGO data is generally public, but be mindful of any institutional constraints when exporting.
* **Rate-limits**: Respect Nominatim usage policy — include a descriptive `User-Agent` and avoid automated heavy requests. Consider caching geocoding results.

---

## 🧪 Testing & CI

* Unit-testable pieces:

  * `parse_index_file()` with synthetic index lines.
  * `parse_profile_netcdf_to_info_rows()` using small `.nc` fixtures.
  * `safe_sql_builder()` parsing logic.
* CI: run `pytest` for parser and helper tests; add `black` and `flake8` for formatting/linting.

---

## 🚩 Troubleshooting & Tips

* **Large downloads / slow network**: Limit index rows when building Chroma. Use `rows_to_index` slider in UI.
* **Missing variables in .nc**: The parser aggressively searches for common variable names — if your dataset uses other conventions, extend `_maybe_get` and candidate variable lists.
* **Memory errors during Chroma build**: Increase swap / use smaller batch sizes. Consider building Chroma on a machine with more RAM or a streaming approach.

---

## 🧾 Examples: Useful Commands & Queries

* Find nearest floats:

  * In UI: Nearest ARGO floats → enter lat/lon → Find nearest floats.
* Ask natural-language question:

  * "List floats in the Arabian Sea from 2019 with temperature measurements" → Chat tab.
* Export ingested info to Parquet (UI): Exports → Export ingested info to Parquet.

---

## 🔬 Reproducibility & Research Best Practices

* Pin Python package versions in `requirements.txt` for reproducible runs.
* Use small, versioned subsets of the IFREMER index when iterating.
* Store trained model artifacts with metadata (features, training sample size, metrics) — current `joblib` artifacts embed that metadata.

---

## 📚 Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repo and create a feature branch.
2. Run tests and linting locally.
3. Open a PR describing the feature/bugfix and tests.

Suggested areas for contributions:

* Modularize `app.py` into `ui.py`, `ingest.py`, `rag.py`, and `ml.py`.
* Add unit tests for the NetCDF parser with small fixtures.
* Add streaming ingestion for ultra-large indexes.

---

## 🧾 Attribution & Credits

* Built by **Ayush Kumar Shaw** (NIT Durgapur).
* Inspired by open ARGO datasets and modern RAG/LLM workflows.

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE`.

---

## 📌 Changelog (high level)

* v0.1 — Initial ingestion + UI + NetCDF parser.
* v0.2 — RAG integration + Chroma indexing + background workers.
* v0.3 — ML training UI + model persistence + export features.

---












