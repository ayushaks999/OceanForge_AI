# ARGO RAG Explorer

> Professional Streamlit application for ingesting, exploring and answering natural‑language queries over Argo profile NetCDF data.

**Author:** Ayush Kumar Shaw
**Contact:** [ayushaks099@gmail.com](mailto:ayushaks099@gmail.com)

---

## Quick links

* **Copy‑paste ready:** this file is a drop‑in `README.md`.
* **Run:** `streamlit run app.py` (see Quickstart).

---

## Overview

ARGO RAG Explorer is an operational, modular toolkit that converts Argo NetCDF profiles into an ingestible measurement table, provides fast indexed search and visualization, and offers an experimental Retrieval‑Augmented Generation (RAG) chat interface using optional LLM + embeddings. It is designed for reproducible ingestion, interactive analysis (maps, profiles, timeseries), and pragmatic research workflows.

The codebase is split into:

* **Backend** — NetCDF parsing, DB schema, ingestion, query builders, and RAG helpers.
* **Frontend** — Streamlit UI with prebuilt workflows: nearest‑float lookup, index exploration, bulk ingest, chat (RAG), profile comparison, and exports.

---

## Highlights / Capabilities

* Robust NetCDF → relational ingestion (per‑profile, per‑sample rows) for SQL‑friendly analytics.
* Lightweight `argo_index` for file metadata and `argo_info` for measurement rows (auto‑migrates new columns).
* `.nc` preview extraction (fast depth/temp/psal previews without full ingestion).
* Interactive Streamlit tabs: nearest floats, index explorer, bulk ingest, RAG chat, trajectories & profile comparison, and exports (Parquet/NetCDF).
* Safe, parameterized SQL builder to produce queries from filters or parsed LLM output.
* Optional Chroma vector index + embeddings + Gemini (via LangChain) for semantic retrieval and RAG responses.
* Background workers (multiprocessing) for index ingestion and vector indexing with status reporting.

---

## Minimal quickstart (copy & paste)

```bash
# clone or copy repo
git clone <REPO_URL> argo-rag-explorer
cd argo-rag-explorer

# create and activate venv
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# create .env (see example below)
# run the app
streamlit run app.py
```

Open the URL printed by Streamlit (typically `http://localhost:8501`).

---

## Recommended `.env` (drop into project root)

```ini
# Local DB & storage
ARGO_SQLITE_PATH=argo.db
AGENTIC_RAG_STORAGE=./storage
AGENTIC_RAG_DB_PATH=./storage/agentic_rag_meta.db
CHROMA_DIR=./storage/chromadb

# IFREMER defaults (public mirror)
IFREMER_INDEX_URL=https://data-argo.ifremer.fr/ar_index_global_prof.txt
IFREMER_BASE=https://data-argo.ifremer.fr/dac

# Optional: Postgres (recommended for production)
# ARGO_PG_URL=postgresql://user:pass@host:5432/dbname

# Optional: Gemini API key for LLM/embeddings
GEMINI_API_KEY=
```

---

## Example `requirements.txt` (baseline)

```
streamlit
pandas
numpy
xarray
netcdf4
h5netcdf
sqlalchemy
requests
python-dotenv
plotly

# optional (enable semantic/RAG features)
chromadb
langchain-google-genai
streamlit-folium
folium
```

Install with `pip install -r requirements.txt`.

---

## How the system works (concise)

1. Download IFREMER index (`ar_index_global_prof.txt`) and ingest into `argo_index` (file metadata).
2. Download remote NetCDFs (to `STORAGE_ROOT`) and parse per‑profile samples with `xarray`. Each numeric sample becomes a row in `argo_info` (columns: `file`, `juld`, `latitude`, `longitude`, `pres`, `temp`, `psal`, `parameter`, ...).
3. Streamlit UI queries `argo_index` / `argo_info` for nearest‑float lookups, filtering, previews, and plots.
4. For natural‑language queries, `ask_argo_question()` parses the query (LLM or rule fallback), prefers `.nc` previews for measurement requests, or queries `argo_info` otherwise. Optionally a RAG step builds context (`assemble_mcp_context`) and asks the LLM (`rag_answer_with_mcp`).

---

## Usage patterns & examples

### Index ingestion (programmatic)

```python
from app import ensure_index_file, parse_index_file, ingest_index_to_sqlite
local = ensure_index_file()
df = parse_index_file(local)
ingest_index_to_sqlite(df)
```

### Download + ingest a single profile

```python
from app import download_netcdf_for_index_path, parse_profile_netcdf_to_info_rows, ingest_info_rows
local = download_netcdf_for_index_path('aoml/13857/profiles/R13857_001.nc')
rows = parse_profile_netcdf_to_info_rows(local)
count = ingest_info_rows(rows)
print('ingested rows', count)
```

### Example Chat queries

* `list floats in Indian Ocean`
* `salinity near the equator in March 2023`
* `show temperature profiles for float R13857`

---

## Production recommendations

* Use **Postgres** (`ARGO_PG_URL`) for concurrent ingestion or large datasets (SQLite may lock on concurrent writes).
* Run heavy index or chroma builds in a worker/container with increased CPU/RAM.
* Keep `storage/` on disk with sufficient space; NetCDFs can accumulate quickly.

---

## Exports

* **Parquet**: full `argo_info` export for analytics (fast, columnar).
* **NetCDF**: a simple row‑oriented NetCDF with `temp` as primary variable and coordinates for `file`, `juld`, `lat`, `lon`, `pres`, `parameter`.

---

## Observability & maintenance

* The app writes a `build_status.json` in `CHROMA_DIR` with build/ingest status and timestamps.
* Logging: wrap long operations in try/except and inspect `build_status.json` for background task results.

---

## Troubleshooting (common issues)

* **NetCDF open errors**: ensure `netCDF4`/`h5netcdf` and `xarray` are installed and the file is intact.
* **Empty previews**: some NetCDFs lack standard variable names; the parser attempts many name variants but may still find nothing.
* **Map rendering fails**: Plotly's mapbox uses external tiles (open‑street‑map) — network required.
* **SQLite locks**: switch to Postgres for heavy or concurrent write workloads.

---

## Development & contribution

* Suggested tests: parsing (`parse_profile_netcdf_to_info_rows`), SQL builder (`safe_sql_builder`), ingestion roundtrips.
* PR checklist: small commits, unit tests, linting (`black`/`flake8`).
* Database schema changes must include migration logic — see `_ensure_info_table_schema()`.

---

## Security & privacy notes

* The app downloads public Argo files by default. If using private datasets, ensure secure storage and appropriate access controls.
* Do not commit secrets (`GEMINI_API_KEY`, DB credentials) to source control — keep them in `.env` or a secure secret manager.

---

## Licensing & attribution

* **Author:** Ayush Kumar Shaw
* **License:** MIT (change in repo to apply your preferred license).

---







