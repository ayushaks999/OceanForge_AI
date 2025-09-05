# ARGO RAG Explorer — Professional README

> **A production-ready, research-grade platform for exploring ARGO float data with integrated RAG and MCP grounding.**

---

## 1. Project Overview

**ARGO RAG Explorer** is an end-to-end application designed for oceanographers, data scientists, and research engineers who need to explore, query, compare, and summarize ARGO float profile data at scale. It combines robust NetCDF parsing, structured relational storage, geospatial queries, interactive visualizations, and Retrieval-Augmented Generation (RAG) driven by a Model Context Protocol (MCP) to produce reproducible, grounded LLM answers.

This README documents the complete functionality implemented in the repository and provides practical guidance for installation, deployment, operation, and extension.

---

## 2. Highlights & What’s Implemented

### Core data capabilities

* **Index ingestion**: download and parse IFREMER / GDAC index files and persist `argo_index` metadata (file path, float id, lat/lon, date, ocean, institution).
* **NetCDF parsing**: robust parser that reads `.nc` profiles (via `xarray`) and extracts per-sample rows (temp, psal, pres, parameter, timestamps) into `argo_info` table.
* **Schema migration helper**: idempotent creation of `argo_index` and `argo_info` tables plus logic to add missing columns to an existing `argo_info` table.
* **Batch ingestion**: high-throughput ingestion with 500-row batching and graceful error handling.

### Search & retrieval

* **Flexible SQL builder** (`safe_sql_builder`) supporting spatial, temporal, variable, ocean, and institution filters for both the index and measurement queries.
* **Nearest-float lookup** by coordinates (Haversine) with distance ranking and configurable candidate limits.
* **Place-based lookup** via Nominatim geocoding with a conservative fallback bounding box for common names (e.g., Arabian Sea).

### Interactive UI (Streamlit)

* **Nearest ARGO floats** tab (coordinate input, Plotly Mapbox visualization, top-N results).
* **Explore Index** tab with rich filters, map preview, and query runner.
* **Ingest Profiles** tab for manual index path ingestion and per-file progress/feedback.
* **Chat (RAG)** tab: LLM-assisted conversational interface with place lookup, `.nc` preview prioritization for measurement queries, and downloadable CSVs of results.
* **Trajectories & Profile comparison** tab for multi-float comparison (trajectories, temp vs depth, pressure vs depth, temp vs time with aggregation methods).
* **Exports** tab to dump ingested `argo_info` to Parquet or NetCDF for downstream analysis.

### RAG, MCP & Vector Search

* **MCP integration** (`assemble_mcp_context()`): constructs a tightly-scoped, traceable context (index samples, `.nc` previews, vector hits and chunk metadata) to ground LLM responses.
* **Structured RAG responses**: `rag_answer_with_mcp()` prompts LLM to return a single JSON object with `answer`, `sql`, and `references` keys — machine readable and reproducible.
* **ChromaDB optional integration**: vector hits are included in MCP context when a Chroma collection exists.
* **Alternating fallbacks**: the application works without LLMs — using deterministic rule-based parsing (`_simple_parse_question`) and SQL only queries.

### Observability & background tasks

* **Status file** and helper functions (`_write_status`, `_read_status`) to track background jobs.
* **Background worker processes** for index ingestion and Chroma build (launched via `multiprocessing.Process` and exposed in the UI).

### Robustness & pragmatic handling

* **Retry-enabled HTTP session** with `requests.adapters.Retry` for index and NetCDF downloads.
* **Byte/array decoding**, `np`/`pandas` coercions and fail-safe conversions to avoid ingestion failures.
* **Heuristics for variable detection** (temp, psal, pres) across non-standard NetCDF variable naming patterns.
* **QC/sentinel handling**: explicit removal of `temp == 1` sentinel rows during visualization workflows.

---

## 3. Architecture & Data Flow

1. **Index retrieval**: `ensure_index_file()` downloads a GDAC/IFREMER index file if missing.
2. **Index parsing**: `parse_index_file()` converts CSV-like index into a DataFrame and `ingest_index_to_sqlite()` persists `argo_index`.
3. **Profile download**: `download_netcdf_for_index_path()` fetches `.nc` files into a structured storage tree (`AGENTIC_RAG_STORAGE`).
4. **Profile parsing**: `parse_profile_netcdf_to_info_rows()` extracts per-sample profile rows and returns them as Python dicts.
5. **DB ingestion**: `ingest_info_rows()` inserts parsed rows in batches into `argo_info`.
6. **Querying**: `safe_sql_builder()` + SQLAlchemy are used to run queries from UI and programmatic callers.
7. **RAG**: `assemble_mcp_context()` and `rag_answer_with_mcp()` combine DB results, previews and vectors into an MCP payload that is sent to LLM.
8. **Visualization & export**: Streamlit + Plotly produce interactive charts; CSV/Parquet/NetCDF exports are available.

---

## 4. Installation & Environment

### Requirements

* Python 3.9+
* Recommended libraries (non exhaustive): `streamlit`, `xarray`, `netCDF4`, `pandas`, `numpy`, `sqlalchemy`, `requests`, `plotly`, `chromadb` (optional), `langchain-google-genai` (optional), `python-dotenv`.

### Install

```bash
git clone <repo>
cd repo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment variables (create `.env`)

```
# required (unless you don't use LLM/Chroma features)
GEMINI_API_KEY=YOUR_GOOGLE_GEMINI_KEY
AGENTIC_RAG_STORAGE=./storage
ARGO_SQLITE_PATH=./storage/argo.db
AGENTIC_RAG_DB_PATH=./storage/agentic_rag_meta.db
# optional
IFREMER_INDEX_URL=https://data-argo.ifremer.fr/ar_index_global_prof.txt
IFREMER_BASE=https://data-argo.ifremer.fr/dac
ARGO_PG_URL=postgresql://user:pass@host:5432/db
```

### Run

```bash
streamlit run app.py --server.port 8501
```

---

## 5. Configuration & Tuning

* **DB backend**: set `ARGO_PG_URL` to a Postgres URL to move from SQLite to Postgres for concurrent workloads.
* **Chroma**: configure persistence by setting `CHROMA_DIR` and ensure `chromadb` is installed. Use the UI to trigger Chroma builds for first-N rows.
* **Multiprocessing start method**: code attempts `fork`; on Windows it will fallback to the platform default.
* **Index refresh**: call `start_index_ingest_async()` or use the Streamlit sidebar button.

---

## 6. Developer Notes & Extensibility

### Key modules (single-file layout)

* `ensure_index_file()`, `parse_index_file()` — index lifecycle.
* `download_netcdf_for_index_path()`, `read_netcdf_variables_to_df()`, `parse_profile_netcdf_to_info_rows()` — NetCDF helpers.
* `safe_sql_builder()` — secure, parameterized SQL generation for both index and measurement queries.
* `assemble_mcp_context()`, `rag_answer_with_mcp()` — MCP/RAG orchestration.
* Streamlit UI section (tabs) — user workflows and IO.

### Best places to extend

* Add pluggable retrievers for RAG (FAISS, OpenSearch, or cloud vector DBs).
* Add a celery/RQ worker layer for robust background processing in production.
* Introduce schema versioning for `argo_info` and migration scripts (Alembic) for Postgres deployment.
* Integrate authentication and multi-tenant logic for shared deployments.

---

## 7. MCP / RAG — Implementation Details (Advanced)

* **assemble\_mcp\_context()** builds a compact context with:

  * `index_sample` text (sample rows from `argo_index`).
  * `.nc` previews (CSV-like snippet of depth/temp/psal for top-K files).
  * Vector hits from Chroma (if available) — included as short metadata lines.
  * A `chunks` list describing each piece (type, id, text/meta) — stored in the returned MCP object for auditability.

* **rag\_answer\_with\_mcp()** creates a deterministic prompt: instructing the LLM to respond with a single JSON object with fields:

  * **answer** — human-readable explanation/summary
  * **sql** — optional recommended SQL query
  * **references** — short text pointers to sources used (index rows, file names, vector hits)

* **Fallback behavior**: if LLM or embeddings are unavailable, the app uses `_simple_parse_question()` as a strict rule-based parser and executes SQL directly.

* **Why MCP matters**: by providing the LLM with only vetted, structured context and by requesting JSON-only output, the responses are reproducible, auditable, and suitable for automation (eg. building dashboards, triggering further SQL queries, or programmatic alerts).

---

## 8. Usage Examples (Common Commands)

**Nearest floats around a coordinate** (in the UI): input lat/lon, click *Find nearest*.

**Programmatic call to `ask_argo_question()`** (Python REPL):

```python
from app import ask_argo_question, ensure_models
llm, emb = None, None
# If you have Gemini / embeddings available
# llm, emb = ensure_models()
out = ask_argo_question(llm, emb, "salinity near the equator in March 2023", user_id=0)
print(out['explanation'])
if out.get('measurement_rows') is not None:
    df = out['measurement_rows']
```

**Example of LLM structured response** (what the app expects):

```json
{
  "answer": "Salinity near equator in March 2023 shows a mean surface salinity of ~35 PSU based on 12 profiles.",
  "sql": "SELECT ... FROM argo_info WHERE ... LIMIT 500",
  "references": ["file=R12345_001.nc", "vector_hit: file=R23456_010.nc"]
}
```

---

## 9. Known Limitations & Troubleshooting

* **NetCDF heterogeneity**: ARGO files can differ — the parser uses heuristics (variable name lists, dimensional fallbacks) which are intentionally permissive. Some profiles may still fail to yield measurements.
* **`temp == 1` sentinel**: many NetCDF datasets use integer flags; the UI removes rows where `temp == 1` before plotting. Adjust logic if your datasets use different flagging.
* **Streamlit rebuilding**: large DB or heavy background activity can cause Streamlit to show rebuilds; prefer a production server or separate worker processes for ingestion.
* **LLM availability**: Gemini integration is optional — without it the chat mode falls back to rule-based parsing.

---

## 10. Production Deployment Recommendations

* Use **Postgres** for `ARGO_PG_URL` with proper connection pooling and migrations (Alembic).
* Replace `multiprocessing` background tasks with **Celery + Redis** (or a cloud task queue) for robustness and monitoring.
* Host Streamlit behind a reverse proxy (NGINX) or use Streamlit Server for auth & scaling.
* Use a managed vector DB (Chroma cloud / Pinecone / Milvus) if you expect high-volume vector retrieval.
* Periodically rebuild the Chroma index (nightly) and retain MCP chunk metadata for auditing.

---

## 11. Testing & Validation

* Unit test parsers on a small corpus of `.nc` files covering common variant names (`TEMP`, `temperature`, `PSAL`, `salinity`, `PRES`).
* Integration tests for index ingestion and `safe_sql_builder()` outputs.
* Validate LLM JSON outputs with a strict JSON schema or JSON-LD validator if automating downstream use.

---

## 12. Contributing

Contributions are welcome. Please follow typical OSS workflow: fork, branch, PR. Include tests and maintain backward compatibility for DB columns.

---

## 13. License & Credits

MIT License. Credits to ARGO, IFREMER/GDAC, AOML for dataset availability; Streamlit, Plotly, Xarray, Pandas and the open-source ML tooling ecosystem for enabling RAG workflows.

---

If you want, I can now:

* Add an **architecture diagram** (SVG/PNG) into the repo and the README.
* Add **example screenshots** of each Streamlit tab.
* Generate a `requirements.txt` pinned to tested versions.

Tell me which of those you'd like and I will add them directly to the canvas file.
