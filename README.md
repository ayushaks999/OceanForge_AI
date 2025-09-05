# ARGO RAG Explorer

> **A professional Retrieval-Augmented-Generation (RAG) platform for ARGO oceanographic profiles** — index, ingest, explore and chat with Argo `.nc` profiles. ⚓️🌊

---

## 🚀 One-line summary

ARGO RAG Explorer is a streamlined web application to search the IFREMER ARGO index, download NetCDF profiles, parse per-sample measurements into a relational store, and answer natural-language questions using a Retrieval-Augmented-Generation pipeline with Multi-Context Prompting (MCP).

---

## ✨ Highlights & Features

* Safe, idempotent ingestion of IFREMER `ar_index_global_prof.txt` into `argo_index`.
* Robust `.nc` parsing via `xarray` to create per-sample rows (depth / temp / psal + metadata) in `argo_info`.
* Flexible database backends: SQLite for quick demos and Postgres for production.
* Optional vector index (Chroma) + embeddings for semantic retrieval of profile metadata and previews.
* MCP: assemble index samples, `.nc` previews, and vector hits to ground the LLM and reduce hallucination.
* Streamlit UI with polished tabs: Nearest floats, Index Explorer, Bulk Ingest, RAG Chat, Trajectories & Comparison, Exports.
* Export ingested data to Parquet / NetCDF for analysis or sharing.

---

## 🧭 Technology stack

* Python 3.9+
* xarray, numpy, pandas — NetCDF parsing and numerics
* SQLAlchemy — database schema & lightweight migrations
* Streamlit + Plotly — interactive UI & visualizations
* Optional: chromadb (vectors), langchain-google-genai (Gemini LLM / embeddings), folium (maps)

---

## 📁 Recommended repository layout

```
app/
  ├─ core.py           # ingestion, parsing, DB helpers
  ├─ rag.py            # MCP & RAG helpers (assemble contexts, call LLM)
  ├─ ui.py             # Streamlit application (tabs & views)
  ├─ utils.py          # helpers (HTTP session, geocoding)
  └─ config.py         # env config loader

storage/               # downloaded .nc files, chroma dir, exports
.env                   # environment variables
requirements.txt
README.md
```

---

## 🔧 Environment variables (example)

```ini
ARGO_SQLITE_PATH=./storage/argo.db
ARGO_PG_URL=                     # optional (postgres://user:pass@host:port/db)
IFREMER_INDEX_URL=https://data-argo.ifremer.fr/ar_index_global_prof.txt
IFREMER_BASE=https://data-argo.ifremer.fr/dac
AGENTIC_RAG_STORAGE=./storage
GEMINI_API_KEY=YOUR_GOOGLE_API_KEY    # optional for LLM/embeddings
```

> Tip: Use Postgres in production; SQLite is fine for local testing.

---

## 🧭 High-level architecture

Below is a compact diagram showing the main components and how they interact.

```mermaid
flowchart TB
  A[Index_File] --> B[Parse_Index_File]
  B --> C[Ingest_Index_to_DB]
  D[Download_NetCDF] --> E[Parse_NetCDF_to_Info_Rows]
  E --> F[Ingest_Info_to_DB]
  C & F --> G[DB_Query_Service]
  H[Chroma_Embeddings] --> I[Vector_Retrieval]
  G & I --> J[Assemble_MCP_Context]
  J --> K[LLM_Prompting]
  K --> L[RAG_Answer]
  M[Streamlit_UI] --> G
  M --> D
  M --> L
```

**Component notes:**

* **Index ingestion:** `ensure_index_file()` → `parse_index_file()` → `ingest_index_to_sqlite()`.
* **Profile ingest:** `download_netcdf_for_index_path()` → `parse_profile_netcdf_to_info_rows()` → `ingest_info_rows()`.
* **Retrieval:** `safe_sql_builder()` constructs parameterized SQL for `argo_index` and `argo_info` queries.
* **RAG / MCP:** `assemble_mcp_context()` merges index samples, `.nc` previews, and vector hits for the LLM prompt.
* **UI:** Streamlit orchestrates user flows: nearest floats, place lookup, ingestion, chat, comparisons, exports.

---

## 📚 Data model

* **argo\_index** — IFREMER index rows: `file`, `date`, `latitude`, `longitude`, `ocean`, `institution`, `date_update`, ...
* **argo\_info** — per-measurement rows (ingested from `.nc`): `juld`, `latitude`, `longitude`, `pres`, `temp`, `psal`, `parameter`, plus calibration & history fields.

Schema is created via SQLAlchemy and a small migration helper `_ensure_info_table_schema()` adds missing columns when required.

---

## 🧪 Usage examples

* **Nearest floats:** `nearest_floats(lat, lon, limit)` — returns closest index rows using Haversine distance.
* **Place lookup:** free-text geocode (Nominatim) → bounding box → index query via `safe_sql_builder()`.
* **Bulk ingest:** paste index file paths → download `.nc` → parse → ingest into `argo_info`.
* **Chat (RAG):** user asks a question; system prefers `.nc` previews for measurement queries, otherwise queries DB; MCP used to ground LLM.
* **Compare:** choose up to 3 floats → plot trajectories from `argo_info` and compare temp/psal profiles.

---

## 🧠 RAG & MCP guidance

* Prefer `.nc` previews for numeric-variable queries when available; they are the closest representation of the raw instrument measurements.
* The MCP context should be compact: include a small sample of index rows, brief `.nc` CSV previews (head), and vector hits metadata.
* Use LLM temperature=0 for deterministic behavior in prompt-to-JSON responses.

---

## ⚙️ Deployment & scaling

* For demos: a VM with 4–8 cores and 16–32 GB RAM is sufficient. Use SQLite or a small Postgres instance.
* For production: containerize with Docker, use managed Postgres, and move `.nc` files to object storage (S3 / GCS). Run Chroma as a separate service.
* For heavy geospatial loads consider PostGIS and dedicated geospatial queries.

---

## ✅ Testing checklist

* Unit tests for `parse_profile_netcdf_to_info_rows()` with representative NetCDF variants.
* Tests for `_to_float_array()` and `_maybe_get()` to ensure robustness across variable shapes.
* Integration test: small IFREMER subset (100–1k rows) to validate download → parse → ingest → query → UI.

---

## 📦 Exports & interoperability

* Export `argo_info` to Parquet (fast analytics) and NetCDF (scientific compatibility). Both are included in the UI.
* Parquet files are portable to DuckDB, Spark, and BI tools.

---

## 🤝 Contributing

* Fork → Branch → PR with tests and updated README. Include one or two `.nc` samples for parser regression tests.

---

## 🧾 License & contact

* Suggested license: MIT or Apache-2.0 (add LICENSE file).
* Contact / Author: **Ayush Kumar Shaw** — add GitHub and email in project metadata.

---

✨ *Want this README exported to `README.md` in the repo or a concise SIH submission one-pager (with diagram as an image)?*
