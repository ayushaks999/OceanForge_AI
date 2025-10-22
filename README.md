# Argo Ocean Data Explorer & Temperature Predictor

> Streamlit app for exploring Argo float profiles, ingesting NetCDF profile data, running retrieval-augmented QA, and training/serving multi-model temperature predictors.

---

## Overview

This repository contains a production-ready Streamlit application that ingests Argo float `.nc` profile files, stores per-profile measurement rows in a SQL database, enables interactive geospatial exploration and comparison of float trajectories, and trains multiple machine learning regressors to predict seawater temperature from features such as latitude, longitude, pressure, salinity and date.

The app is designed for researchers and engineers who want a single, reproducible interface to explore Argo-derived data, build lightweight ML models, and perform retrieval-augmented question answering (RAG) over the indexed float records and short NetCDF previews.

---

## Mermaid Architecture Diagram

Below is a high-level architecture and data-flow diagram rendered with Mermaid. It shows the main components, data stores, and key interactions between modules.

```mermaid
flowchart TD
  subgraph UI [Streamlit UI]
    A1[Tabs & Controls]
    A2[Maps & Visualizations]
    A3[Chat (RAG) Input]
    A4[ML Training & Prediction]
  end

  subgraph App [Application Logic]
    B1[Indexing & Ingest]
    B2[NetCDF Parsing]
    B3[SQL Builder & Queries]
    B4[Retrieval (MCP) & RAG]
    B5[Model Training Pipeline]
    B6[Model Serving (Session)]
  end

  subgraph Storage [Storage & Services]
    C1[(Local NetCDF files)]
    C2[(SQLite / PostgreSQL DB)]
    C3[(Model Artifacts - joblib)]
    C4[(Chroma / Vector DB) - optional]
  end

  subgraph Ext [External Services]
    D1[IFREMER / Argo index URL]
    D2[Nominatim / OSM Geocoding]
    D3[Gemini / LLM (optional)]
  end

  A1 -->|requests| B3
  A2 -->|visualize| B3
  A3 -->|ask| B4
  A4 -->|train/eval| B5

  B1 -->|download| C1
  B2 -->|parse| B1
  B1 -->|insert rows| C2
  B3 -->|read/write| C2
  B4 -->|context| C4
  B5 -->|save| C3
  B6 -->|load| C3

  D1 -->|index file| B1
  D2 -->|bbox| B3
  D3 -->|LLM| B4

  style UI fill:#f9f,stroke:#333,stroke-width:1px
  style App fill:#ffe6a7,stroke:#333,stroke-width:1px
  style Storage fill:#cde6f7,stroke:#333,stroke-width:1px
  style Ext fill:#e6e6e6,stroke:#333,stroke-width:1px
```

---

## Quickstart (Run locally)

```bash
git clone https://github.com/yourusername/argo-ocean-predictor.git
cd argo-ocean-predictor
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

> The app will create a local SQLite DB under `storage/` by default and download cached NetCDF files to `storage/`.

---

## Configuration

Environment variables (use a `.env` file):

* `ARGO_SQLITE_PATH` — path for the default SQLite database (defaults to `argo.db`).
* `ARGO_PG_URL` — optional Postgres URL (if set, Postgres is used instead of SQLite).
* `AGENTIC_RAG_STORAGE` — `STORAGE_ROOT` for local cache, models, and index files.
* `IFREMER_INDEX_URL` — remote index file URL (defaults to Ifremer public index).
* `GEMINI_API_KEY` — optional API key for Gemini/LLM embeddings (if using LangChain / Gemini).

---

## Main Streamlit Tabs & Interactions

**Nearest ARGO floats**

* Inputs: Latitude, Longitude, Limit
* Core functionality: `nearest_floats(lat, lon, limit)` → computes haversine distances and returns nearest index rows.
* Map visualization with clustering and download of mapped points.

**Explore Index**

* Inputs: lat/lon bounding box, ocean code, institution, date range, limit
* Uses: `safe_sql_builder(filters, target='index')` to build parameterized index queries.
* Visualizes index rows and a small map preview.

**Ingest Profiles**

* Bulk ingestion UI: paste index paths or trigger async index ingest.
* Core functions: `ensure_index_file()`, `parse_index_file()`, `download_netcdf_for_index_path()`, `parse_profile_netcdf_to_info_rows()`, `ingest_info_rows()`.
* Background ingestion: `start_index_ingest_async()` runs ingestion in a separate process and writes status to `storage/chromadb/build_status.json`.

**Chat (RAG)**

* Natural-language queries routed via `ask_argo_question(llm, emb_model, question, ...)`.
* LLM conversion to structured filters: `llm_to_structured()` (falls back to `_simple_parse_question()` where necessary).
* Assembles retrieval context with `assemble_mcp_context()` and uses `rag_answer_with_mcp()` to ask the LLM with indexed context.

**Trajectories & Profile comparison**

* Select up to 3 floats; uses `get_measurements_for_float()` and `find_common_vars_for_floats()`.
* Visualizations: trajectory map (`plotly Scattermapbox`) and two-panel profile plots (temp vs depth, pressure vs depth) using `plotly.subplots`.

**Exports**

* Export `argo_info` to Parquet or a simple NetCDF via in-app buttons.

**ML: Temperature predictor**

* Load a sampled training dataset via `load_training_df(max_rows)`.
* Train multiple candidates: RandomForest, GradientBoosting, HistGradientBoosting, XGBoost (wrapped), LightGBM (optional).
* Choice of model selection metric: `RMSE` or `R2`.
* Save artifacts: `temp_predictor_best.joblib` and `temp_predictor_all.joblib` under `STORAGE_ROOT/models/`.
* Inference UI supports ensemble std calculation for tree-based models exposing `estimators_`.

---

## Detailed Function Reference

> The following functions are implemented in `app.py` (or companion modules). This section documents their responsibilities and expected I/O.

### Data ingestion & parsing

* `ensure_index_file(local_path=INDEX_LOCAL_PATH, remote_url=INDEX_REMOTE_URL, timeout=60) -> str`

  * Ensures the IFREMER index file is present locally, downloads if missing.
  * Returns: local path to the index file.

* `parse_index_file(path=INDEX_LOCAL_PATH) -> pd.DataFrame`

  * Reads the index text file and extracts rows with fields (file, date, latitude, longitude, ocean, profiler_type, institution, date_update).
  * Returns: DataFrame of index rows.

* `download_netcdf_for_index_path(index_file_path, dest_root=STORAGE_ROOT, timeout=60) -> str`

  * Downloads an `.nc` file from `IFREMER_BASE` and stores it under `STORAGE_ROOT` mirroring the index path.
  * Returns: local file path.

* `parse_profile_netcdf_to_info_rows(nc_path) -> List[Dict]`

  * Deep parser: opens a NetCDF with `xarray`, attempts to locate latitude, longitude, juld/time, pres, and candidate measurement variables (temp, psal, etc.).
  * Produces *one row per measurement* (so multiple rows per profile) formatted to match the `argo_info` schema.

* `read_netcdf_variables_to_df(nc_path, prof_index=0) -> pd.DataFrame`

  * Extracts 1D arrays for depth/temp/psal for a single profile index and returns a tidy DataFrame.

* `ingest_info_rows(rows: List[Dict]) -> int`

  * Inserts rows in batches into the `argo_info` table. Returns number of rows inserted.

### DB & Query helpers

* `_ensure_info_table_schema()` — Idempotent helper that adds missing columns to an existing `argo_info` table.
* `safe_sql_builder(filters: dict, target: str) -> (sql, params)` — Builds parameterized SQL for either the `argo_index` or `argo_info` queries while validating inputs.
* `get_local_netcdf_path_from_indexfile(index_file)` — Local path mapping helper.

### Geolocation & retrieval

* `get_bbox_for_place(place_name)` — Uses Nominatim to obtain a bounding box for a place name (with a small fallback heuristic for specific names).
* `nearest_floats(lat0, lon0, limit, max_candidates)` — Returns the nearest floats based on a pre-populated `argo_index` table by computing distances.
* `get_measurements_for_float(float_id, variable_hint=None)` — Loads measurements for a float (optionally filtered by parameter name) from `argo_info`.
* `find_common_vars_for_floats(float_ids)` — Detects variable name mappings (e.g., which column corresponds to `temp` or `psal`) across floats.

### RAG & LLM helpers

* `_simple_parse_question(question)` — Rule-based question-to-filter parser used as a fallback.
* `llm_to_structured(llm, question)` — Uses the LLM to convert free-text to a JSON structured query (falls back to `_simple_parse_question()` on failure).
* `assemble_mcp_context(index_rows, nc_previews, chroma_client, question, emb_model, top_k) -> dict` — Builds context chunks for the multi-context prompt (MCP) used for RAG.
* `rag_answer_with_mcp(llm, emb_model, question, index_rows, nc_previews, chroma_client) -> dict` — Sends the MCP prompt to the LLM and expects structured JSON (answer, recommended SQL, references).
* `ask_argo_question(llm, emb_model, question, user_id, chroma_client)` — Top-level orchestration for chat queries; performs parsing, index queries, optional `nc` preview reads, and returns a structured response.

### ML utilities

* `load_training_df(max_rows)` — Loads a sampled training DataFrame from `argo_info` and converts `juld` to epoch seconds (`juld_ts`).
* `XGBRegressorSafe(BaseEstimator, RegressorMixin)` — A shim class that wraps XGBoost's scikit-learn interface to avoid `sklearn_tags` compatibility failures in some environments.
* `_LazyXGB` — A fallback on-demand wrapper that instantiates real `xgboost.XGBRegressor` at `fit()` time.
* Training logic (UI-driven) — builds candidate pipelines with `StandardScaler` + model, trains on `train_test_split`, computes `RMSE` and `R2`, and saves the best model and all models as `.joblib` artifacts under `STORAGE_ROOT/models/`.
* Prediction logic — loads a model pipeline into `st.session_state`, prepares a single-sample DataFrame, and predicts. If the underlying model exposes `estimators_`, it computes per-tree ensemble predictions to estimate an empirical stddev for the prediction.

### Background & helper utilities

* `_requests_session_with_retries()` — Returns a requests Session configured with retries and exponential backoff.
* `_write_status()` / `_read_status()` — Writes and reads a small JSON status file used by background workers (indexing / chroma build).
* `start_index_ingest_async()` / `start_chroma_build_async()` — Spawn background processes that run ingestion or chroma indexing.

---

## Best Practices, Performance & Notes

* **Batch ingest**: `parse_profile_netcdf_to_info_rows()` can generate many rows per file; ingest in batches and avoid holding all rows in memory for very large ingest jobs.
* **Database choice**: For small experiments, SQLite is sufficient. For large-scale ingestion or concurrent access, use PostgreSQL and set `ARGO_PG_URL`.
* **Model serving**: Streamlit is convenient for demos, but for production serving of heavy models, export the pipeline and serve via FastAPI or a model server (e.g., BentoML, TorchServe).
* **LLM costs**: If using Gemini/LLM for RAG, be mindful of API usage and rate limits. Consider batching, caching, and local vector stores (Chroma) to reduce calls.
* **Dependency mismatches**: The repo includes shims (`XGBRegressorSafe`, `_LazyXGB`) to mitigate sklearn/XGBoost compatibility issues.

---

## Troubleshooting & FAQs

**Q: XGBoost raises `super object has no attribute sklearn_tags`.**

A: Use the provided `XGBRegressorSafe` wrapper or install an xgboost version compatible with your scikit-learn release. The app also attempts a `_LazyXGB` fallback.

**Q: Downloads are slow or failing.**

A: Check connectivity to IFREMER; the app uses retries and streaming downloads. For large downloads, run ingestion during off-peak hours.

**Q: My NetCDF parsing missed variables.**

A: Argo NetCDFs can vary. The parser searches common variable names and falls back to scanning 1–2D variables. Inspect `read_netcdf_variables_to_df()` to add custom variable name mappings for your dataset.

---

## Contribution & Contact

Contributions and pull requests are welcome. Open an issue for bugs, feature requests, or to request help with large-scale ingestion.

If you’d like a tailored deployment (Docker, Cloud Run, or Hugging Face Spaces + autoscaling), I can provide a `Dockerfile` and deployment steps.

**Author**: Ayush — Data Engineer & ML Developer

---

## License

MIT © 2025

---

*End of README — The Mermaid diagram above will render on platforms that support Mermaid syntax (GitHub rendering may require enabling mermaid preview or converting to an image).*
