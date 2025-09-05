# 🚀 ARGO RAG Explorer — README

**Professional, advanced README for the ARGO RAG Explorer project**

---

## 📚 Table of contents

1. Project overview
2. Key features (implemented)
3. Architecture & components
4. Quick start (install & run)
5. Environment variables & configuration
6. Database schema & migration notes
7. Data flow and pipelines
8. Core functions & modules (API-style summary)
9. Streamlit UI — tabs & interactions
10. Ingestion, parsing & previewing .nc files
11. Retrieval-Augmented Generation (RAG) & MCP
12. Background workers & async tasks
13. Plotting, QC and aggregation choices
14. Performance, scaling & operational guidance
15. Testing, debugging & troubleshooting
16. Security, privacy & data governance
17. Contribution, code style & tests
18. License & acknowledgements

---

# 1. Project overview 🔎

ARGO RAG Explorer is a production-oriented, end-to-end system for ingesting, indexing, previewing and interacting with ARGO oceanographic profiles. It fuses traditional relational storage of parsed NetCDF metadata (`argo_index`, `argo_info`) with optional vector search (Chroma + embeddings) and an LLM (Gemini) for Retrieval-Augmented Generation (RAG). A Streamlit front-end provides exploration, visualizations, and RAG-driven Q\&A.

**Design goals:**

* ✅ Robust, idempotent ingestion of IFREMER ARGO NetCDF inventory
* ✅ Fine-grained per-measurement ingestion (each depth/sample becomes a row) into a relational store
* ✅ Fast index queries + local `.nc` previews when high-fidelity measurements are requested
* ✅ Optional semantic/vector index (Chroma) and LLM-assisted answers (Gemini via LangChain wrappers)
* ✅ Developer-friendly Streamlit UI for exploration, ingestion, RAG chat, and exports

# 2. Key features (implemented) ✨

* **Index ingestion from IFREMER** — download and parse `ar_index_global_prof.txt` and ingest to `argo_index` table.
* **NetCDF parsing & per-measurement ingestion** — `parse_profile_netcdf_to_info_rows` converts `.nc` files into multiple `argo_info` rows (temp/psal/pres/parameter/value etc.).
* **NetCDF previews** — `read_netcdf_variables_to_df` extracts `depth/temp/psal` per profile for fast preview without DB round-trips.
* **Relational storage** — SQLAlchemy-based schema (`argo_index`, `argo_info`) supports SQLite or Postgres via env configuration.
* **Migration helper** — `_ensure_info_table_schema` adds missing columns if schema evolves.
* **Search & filtering** — `safe_sql_builder` generates safe parameterized SQL for both index and measurement queries.
* **Place geocoding fallback** — Nominatim geocoding with light fallback for e.g., ‘Arabian Sea’. 🌍
* **Nearest floats** — Haversine distance-based nearest float lookup.
* **RAG + MCP assembly** — `assemble_mcp_context` and `rag_answer_with_mcp` build contextual prompts combining index rows, `.nc` previews and vector hits.
* **Optional vector index** — chromadb client integration and embedding support (Gemini embeddings via LangChain wrapper). 🧠
* **Streamlit app** — multi-tab UI including nearest floats, index explorer, bulk ingest, RAG chat, trajectories / profile comparison, exports (Parquet, NetCDF).
* **Background workers** — index ingest and Chroma build launchable as background processes (`multiprocessing`) with status tracking.
* **Plotting & QC-aware displays** — Temperature vs Depth, Pressure vs Depth, Temperature vs Time with aggregation options and sentinel/QC handling (e.g., ignore `temp == 1`). 📈

# 3. Architecture & components 🏗️

* **app.py / part1** — core helpers, DB, parsing & RAG functions
* **app.py / part2** — Streamlit UI that imports/uses part1 functions (can be combined into a single `app.py` or split into modules)
* **Storage** — configurable root storage directory for downloaded `.nc` files and vector DB
* **DB** — SQLAlchemy engine (SQLite default, Postgres optional via `ARGO_PG_URL`)
* **Optional ML components** — `langchain_google_genai` wrappers for Gemini LLM + embeddings; `chromadb` for vector search
* **Background workers** — small multiprocess worker pattern with JSON status file for progress monitoring

# 4. Quick start (install & run) ▶️

```bash
# recommended: create venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# or minimal:
pip install streamlit sqlalchemy pandas xarray requests plotly python-dotenv
# optional
pip install chromadb langchain-google-genai streamlit-folium

# run the Streamlit app
streamlit run app.py
```

# 5. Environment variables & configuration ⚙️

* `ARGO_SQLITE_PATH` — local sqlite filename (default: `argo.db`)
* `ARGO_PG_URL` — optional Postgres URL; if present, Postgres is used instead of SQLite
* `AGENTIC_RAG_STORAGE` — root folder for downloads, index and chroma DB (default: `./storage`)
* `IFREMER_INDEX_URL` — remote ar\_index file URL (default pointed to IFREMER)
* `IFREMER_BASE` — base URL for NetCDF files on the IFREMER server
* `AGENTIC_RAG_DB_PATH` — local sqlite for chat/feedback (default under storage)
* `GEMINI_API_KEY` — required if LLM/embeddings features are desired 🔑

Store environment variables in a `.env` file or export them into your shell prior to running.

# 6. Database schema & migration notes 🗄️

* `argo_index` table: columns — `id, file, date, latitude, longitude, ocean, profiler_type, institution, date_update`
* `argo_info` table: wide schema - stores global file attrs and per-measurement fields; includes `pres`, `temp`, `psal`, `parameter`, `juld` (datetime) plus many metadata columns.
* `_ensure_info_table_schema` inspects existing DB and runs `ALTER TABLE ADD COLUMN` for missing columns. This is intentionally tolerant/fail-safe.
* Helpful indexes are created on `(latitude, longitude)`, `juld`, and `parameter` to speed geographic/time/variable queries.

# 7. Data flow and pipelines 🔁

1. Ensure index file present: `ensure_index_file()` downloads IFREMER index if missing.
2. `parse_index_file()` → DataFrame of index rows → `ingest_index_to_sqlite()` stores to `argo_index`.
3. For selected index rows (file paths), download `.nc` via `download_netcdf_for_index_path()`.
4. Parse `.nc` with `parse_profile_netcdf_to_info_rows()` to produce per-measurement rows.
5. Insert per-measurement rows via `ingest_info_rows()` into `argo_info`.
6. For quick preview, `read_netcdf_variables_to_df()` extracts depth/temp/psal per profile without full ingestion.
7. Optional vectorization: for large-scale semantic search, build a Chroma collection from index/preview texts.

# 8. Core functions & modules (API-style summary 🔧)

* `ensure_index_file(local_path, remote_url)` — downloads index
* `parse_index_file(path)` — parses index file into DataFrame
* `ingest_index_to_sqlite(df)` — writes index DataFrame to DB
* `download_netcdf_for_index_path(index_file_path)` — download `.nc`
* `parse_profile_netcdf_to_info_rows(nc_path)` — parse .nc -> list\[dict] rows for `argo_info`
* `read_netcdf_variables_to_df(nc_path, prof_index=0)` — fast depth/temp/psal preview as DataFrame
* `ingest_info_rows(rows)` — batch-insert rows into `argo_info`
* `safe_sql_builder(filters, target)` — builds SQL and params for index or measurements
* `nearest_floats(lat, lon, limit)` — returns nearest floats (haversine)
* `get_measurements_for_float(float_id, variable_hint=None)` — retrieve measurements for float from DB
* `llm_to_structured(llm, question)` — LLM or fallback parse to structured filters
* `ask_argo_question(llm, emb_model, question, chroma_client)` — end-to-end RAG-enabled question pipeline
* `assemble_mcp_context(...)` / `rag_answer_with_mcp(...)` — create context and ask LLM for JSON response

# 9. Streamlit UI — tabs & interactions 🖥️

**Tabs implemented:**

* **Nearest ARGO floats** — lat/lon input, nearest lookup, interactive map (Plotly Mapbox) 🧭
* **Explore Index** — parameterized index queries, map visualization, last-index caching in `session_state` 🔎
* **Ingest Profiles** — paste index paths, download & ingest `.nc` files (batch) ⬇️
* **Chat (RAG)** — LLM-assisted chat (auto/force/fallback), place lookup integration, `.nc` previews used preferentially for numeric variables 💬
* **Trajectories & Profile comparison** — pick up to 3 floats, show index metadata, trajectories from `argo_info`, raw `.nc` previews and ingested rows, comparison plots 🗺️
* **Exports** — export `argo_info` to Parquet or a simple NetCDF representation 📤

UI niceties:

* Status JSON in sidebar
* Background job start buttons for index ingest and Chroma build
* Download buttons for CSV / NetCDF / Parquet previews

# 10. Ingestion, parsing & previewing .nc files 🧾

* Parsing is tolerant: `_maybe_get()` finds attributes/variables using candidate names
* `parse_profile_netcdf_to_info_rows()` emits one row per measurement with `parameter`, `temp`, `psal`, `pres`, `juld`, `latitude`, `longitude` when available
* The parser attempts to handle a variety of ARGO NetCDF shapes (multi-profile dims, 1-D and 2-D arrays) and maps common variable name flavours
* Previews use `read_netcdf_variables_to_df()` which returns a cleaned `depth/temp/psal` DataFrame for quick in-UI inspection

# 11. Retrieval-Augmented Generation (RAG) & MCP 🤖

* `assemble_mcp_context()` aggregates index samples, `.nc` previews and vector hits into a short context for the LLM
* `rag_answer_with_mcp()` builds a system+context prompt and expects the LLM to return a JSON object `{answer, sql, references}`
* When `.nc` previews are available for requested variables, these are preferred over DB rows for numeric precision in the RAG pipeline

# 12. Background workers & async tasks 🛠️

* `start_index_ingest_async()` & `start_chroma_build_async()` spawn lightweight processes and write status to `STATUS_FILE`
* Worker processes call `_write_status(...)` to persist progress; UI reads status with `_read_status()`
* `multiprocessing.set_start_method('fork')` is attempted for performance but code gracefully falls back on platforms where unavailable

# 13. Plotting, QC and aggregation choices 📊

* Temperature-depth and pressure-depth plots use Plotly with spline lines and reversed Y-axes (depth increasing downwards)
* The UI explicitly filters sentinel values like `temp == 1` (often used as QC flag) — this is applied before plotting
* Temperature vs Time supports aggregation per profile: Shallowest, Median, Mean, Max (user-selectable)
* Hover & download controls provide CSV exports of plotted/aggregated tables

# 14. Performance, scaling & operational guidance ⚡

* For small-to-moderate datasets, SQLite is convenient. For large-scale ingestion (> millions of rows), use Postgres (`ARGO_PG_URL`)
* Use `ingest_info_rows()` batch size tuning (default flush at 500 rows) to balance memory and transaction overhead
* Index heavy queries should use the DB indexes created on `(latitude, longitude)`, `juld`, `parameter`
* Vector build: prepare a pre-processing step that trims/normalizes docs before embedding to control vector DB size
* For faster downloads use `requests` session with retries (`_requests_session_with_retries()`)

# 15. Testing, debugging & troubleshooting 🐞

* Unit test suggestions: parse a small set of IFREMER `.nc` files, validate row counts & presence of temp/psal/pres
* Repro steps if `.nc` parsing fails: open file locally with `xarray.open_dataset()` to inspect variables & dims
* If index ingestion stalls: check `STATUS_FILE` JSON and `ensure_index_file()` network reachability
* If LLM/Embeddings fail: verify `GEMINI_API_KEY` and that `langchain_google_genai` and `chromadb` packages are installed

# 16. Security, privacy & data governance 🔐

* Do not commit `GEMINI_API_KEY` or DB credentials to Git. Use `.env` (excluded from VCS) or secrets manager
* Access controls: Streamlit app is primarily for internal use — protect with network-level auth or use Streamlit Cloud permissions
* Data retention: store only what you need; exports are local files and should be stored with governance in mind

# 17. Contribution, code style & tests 🤝

* Follow PEP8 and type hints where practical
* Add unit tests for `parse_profile_netcdf_to_info_rows` and `read_netcdf_variables_to_df`
* Keep the Streamlit UI components declarative and avoid long-running synchronous work on main thread (use background workers for heavy tasks)

# 18. License & acknowledgements 📝

This README ships as part of the ARGO RAG Explorer project. Choose an appropriate license (e.g., MIT) and add `LICENSE` file. Acknowledge IFREMER and ARGO program data sources where applicable.

---

## 📎 Appendix — Frequently used commands & examples

**Download index and ingest (manual):**

```python
local = ensure_index_file()
df = parse_index_file(local)
print(len(df))
ingest_index_to_sqlite(df)
```

**Bulk ingest selected files (example snippet used by Streamlit UI):**

```python
for p in paths:
    local_nc = download_netcdf_for_index_path(p)
    rows = parse_profile_netcdf_to_info_rows(local_nc)
    n = ingest_info_rows(rows)
    print(f"Ingested {n} rows from {p}")
```

**Ask via RAG (programmatic):**

```python
out = ask_argo_question(llm=None, emb_model=None, question="salinity near the equator in March 2023")
print(out['explanation'])
```

**Export to parquet:**

```python
df = pd.read_sql_query("SELECT * FROM argo_info", engine)
df.to_parquet("argo_info.parquet", index=False)
```

---

If you want, I can:

* ✂️ Produce a shorter `README.md` summary for project landing pages.
* 📘 Generate `CONTRIBUTING.md`, `DEPLOYMENT.md` or `ARCHITECTURE.md` based on this document.
* 🧪 Create example unit tests and CI pipeline snippets.
