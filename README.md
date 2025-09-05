# ARGO RAG Explorer

**A production-minded Streamlit app for downloading, parsing and exploring Argo NetCDF profiles, ingesting per-profile measurements into a relational store, and answering natural-language questions using optional RAG (LLM + embeddings + Chroma) capabilities.**

**Author / Maintainer:** Ayush Kumar Shaw
**Email:** [ayushaks099@gmail.com](mailto:ayushaks099@gmail.com)
**Repo / Home:** (replace with your repo URL)

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Architecture](#architecture)
* [Quickstart (copy & paste)](#quickstart-copy--paste)
* [Environment variables / `.env` example](#environment-variables--env-example)
* [Dependencies (`requirements.txt`)](#dependencies-requirementstxt)
* [Database & Schema summary](#database--schema-summary)
* [How it works (high level)](#how-it-works-high-level)
* [Usage examples & common commands](#usage-examples--common-commands)
* [RAG / LLM / Chroma notes](#rag--llm--chroma-notes)
* [Exports & Data formats](#exports--data-formats)
* [Troubleshooting & tips](#troubleshooting--tips)
* [Development & contribution](#development--contribution)
* [License](#license)

---

## Project Overview

ARGO RAG Explorer provides an end-to-end developer workflow to fetch Argo index entries and NetCDF profile files, parse per-profile measurements into a relational database, explore the dataset with an interactive Streamlit UI (maps, plots, table views), and experiment with Retrieval-Augmented Generation (RAG) via optional LLM + embeddings + Chroma.

It is designed to be modular: usable without any LLM/embeddings, while providing optional semantic/RAG features when you enable them.

---

## Features

* Download & parse IFREMER Argo index (`ar_index_global_prof.txt`) and remote NetCDF profile files.
* Convert each NetCDF profile into many per-measurement rows (depth/temp/psal/other variables) and store them in `argo_info`.
* Lightweight `argo_index` table for quick file-level lookup and filtering.
* Streamlit UI with tabs: Nearest floats, Index explorer, Bulk ingest, Chat (RAG + MCP), Trajectories & Profile comparison, Exports.
* `.nc` preview parsing for quick inspection without full ingestion.
* Safe SQL builder to convert user filters to parameterized SQL (reduces injection risk).
* Optional Chroma vector index + embeddings and Gemini (via `langchain_google_genai`) integration for RAG answers.
* Background worker helpers for index ingestion and Chroma building.

---

## Architecture

* **Backend (PART 1)**

  * Configuration & env parsing
  * SQLAlchemy engine & schema (`argo_index`, `argo_info`)
  * NetCDF parsing helpers using `xarray` (`parse_profile_netcdf_to_info_rows`, `read_netcdf_variables_to_df`)
  * Bulk ingestion helper (`ingest_info_rows`) and safe SQL builder (`safe_sql_builder`)
  * Retrieval helpers: `nearest_floats`, `get_measurements_for_float`
  * RAG helpers: `assemble_mcp_context`, `rag_answer_with_mcp`, `ask_argo_question`

* **Frontend (PART 2 / PART 3)**

  * Streamlit app with tabs and Plotly visualizations (mapbox/line charts)
  * CSV/Parquet/NetCDF export utilities
  * Optional folium support (via `streamlit-folium`) for maps

---

## Quickstart (copy & paste)

1. **Clone the repo** (or copy `app.py` and supporting modules into a folder):

```bash
git clone <your-repo-url> argo-rag-explorer
cd argo-rag-explorer
```

2. **Create and activate a virtual environment** (recommended):

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

3. **Create `requirements.txt`** (see example below) and install dependencies:

```bash
pip install -r requirements.txt
```

4. **Create a `.env` file** in the project root (see example below). At minimum you can use defaults and the app will create `storage/` and `argo.db` locally.

5. **Run the Streamlit app**:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`).

---

## Environment variables / `.env` example

Create a `.env` file in your project root and update values as needed.

```ini
# Storage & DB
ARGO_SQLITE_PATH=argo.db
AGENTIC_RAG_STORAGE=./storage
AGENTIC_RAG_DB_PATH=./storage/agentic_rag_meta.db
CHROMA_DIR=./storage/chromadb

# IFREMER index/download endpoints (defaults are suitable for public IFREMER mirror)
IFREMER_INDEX_URL=https://data-argo.ifremer.fr/ar_index_global_prof.txt
IFREMER_BASE=https://data-argo.ifremer.fr/dac

# Optional: use a Postgres DB instead of SQLite
# ARGO_PG_URL=postgresql://user:pass@host:5432/dbname

# Optional: API key for Gemini (if you plan to use LLM/embeddings via LangChain)
GEMINI_API_KEY=
```

---

## Dependencies (`requirements.txt`)

Below is a conservative list — trim/add packages based on which optional features you enable.

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
sqlalchemy-utils

# optional (enable for LLM/embeddings and vector DB)
chromadb
langchain-google-genai
streamlit-folium
folium
```

Install with `pip install -r requirements.txt`.

---

## Database & Schema summary

* **argo\_index**: file-level index (file path, date, latitude, longitude, ocean, profiler\_type, institution, date\_update)
* **argo\_info**: ingested per-profile/per-measurement table. Useful columns include:

  * `file`, `juld` (DateTime), `latitude`, `longitude`, `pres` (pressure/depth), `temp`, `psal`, `parameter`, and numerous metadata columns.

The code creates tables on startup and attempts to add missing columns (migration helper) when the DB exists.

---

## How it works (high level)

1. **Index**: `ensure_index_file()` downloads IFREMER index (if missing). `parse_index_file()` converts it to a DataFrame and `ingest_index_to_sqlite()` stores it in `argo_index`.
2. **Download NetCDF**: `download_netcdf_for_index_path(path)` downloads NetCDF into `STORAGE_ROOT`.
3. **Parse**: `parse_profile_netcdf_to_info_rows(nc_path)` opens `.nc` with `xarray` and emits many rows (one per measurement sample). There is also `read_netcdf_variables_to_df()` for quick depth/temp/psal previews.
4. **Ingest**: `ingest_info_rows(rows)` bulk-inserts rows into `argo_info`.
5. **Query / UI**: Streamlit UI calls `safe_sql_builder()` to construct parameterized queries, shows index & measurement results, plots trajectories and profiles, and offers exports.
6. **RAG**: For natural language questions, `ask_argo_question()` uses LLM parsing (or fallback rules) and prefers `.nc` previews for measurement queries; it falls back to `argo_info` DB rows when needed.

---

## Usage examples & common commands

### Start the app

```bash
streamlit run app.py
```

### Programmatic examples (Python REPL or script)

```python
from app import ensure_index_file, parse_index_file, ingest_index_to_sqlite
local = ensure_index_file()
df = parse_index_file(local)
print('index rows', len(df))
ingest_index_to_sqlite(df)
```

```python
# Download and ingest a single NetCDF by index path
from app import download_netcdf_for_index_path, parse_profile_netcdf_to_info_rows, ingest_info_rows
local = download_netcdf_for_index_path('aoml/13857/profiles/R13857_001.nc')
rows = parse_profile_netcdf_to_info_rows(local)
count = ingest_info_rows(rows)
print('ingested rows', count)
```

### Example natural-language queries (Chat tab)

* `list floats in Indian Ocean`
* `salinity near the equator in March 2023`
* `show temperature profiles for float R13857`

---

## RAG / LLM / Chroma notes

* LLM & embeddings are optional. If `GEMINI_API_KEY` is set and `langchain_google_genai` is installed, `ensure_models()` will return LLM & embeddings objects.
* The app tries to build a Chroma collection `floats_user_0` for vector retrieval. This is optional and can be skipped.
* If no LLM is available the app uses a conservative fallback parser (`_simple_parse_question`) to convert text -> filters.
* `assemble_mcp_context()` builds context chunks from index rows and .nc previews for the LLM.

---

## Exports & Data formats

* **Parquet**: full `argo_info` table export (fast, columnar). Use the Export tab.
* **NetCDF**: simple row-oriented NetCDF generated from `argo_info` (the app creates a `value` variable from `temp` and stores `file`, `juld`, `lat`, `lon`, `pres`, `parameter` as coordinates).

---

## Troubleshooting & tips

* \*\*`xarray` fails to parse a .nc`**: check file integrity and that `netCDF4`/`h5netcdf\` are installed.
* **Large ingests**: ingesting hundreds of thousands of files is slow and memory/disk intensive. Start small (100–1000 rows) for testing.
* **Concurrent DB access (SQLite)**: SQLite can show locks under concurrent writes — switch to Postgres for heavy parallel ingestion and set `ARGO_PG_URL`.
* **Map rendering**: Plotly mapbox uses `open-street-map` tiles that require network access. If offline, maps may fail gracefully.
* **Sentinel values**: Many datasets use sentinel values (e.g. `temp == 1`) or QC flags. The UI filters `temp == 1` by default for plotting. Adjust logic if your dataset differs.

---

## Development & contribution

* Use feature branches for new work and include tests for parsing/SQL logic.
* Suggested test targets:

  * `parse_profile_netcdf_to_info_rows` with known NetCDF samples
  * `safe_sql_builder` with diverse filter combos
  * DB ingest + readback roundtrips

Pull request checklist:

* Lint (black/flake8), tests passing, small & descriptive commits
* Backwards-compatible DB changes must include a migration helper (see `_ensure_info_table_schema`)

---

## License

MIT License (or replace with your preferred OSS license).

---

## Further help

If you want I can also:

* Produce a `requirements.txt` tailored to enabled features, or
* Produce a `Dockerfile` + `docker-compose.yml` to containerize the app, or
* Create a small example dataset and a step-by-step ingestion notebook.

Just tell me which you prefer and I will add it to the repo.
