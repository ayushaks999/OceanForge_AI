# ARGO RAG Explorer

> **A professional Retrieval-Augmented-Generation (RAG) platform for ARGO oceanographic profiles** — index, ingest, explore, compare trajectories, and chat with Argo `.nc` profiles. ⚓️🌊

---

## 🚀 One-line summary

ARGO RAG Explorer is a polished Streamlit application that ingests the IFREMER ARGO index, downloads and parses NetCDF profiles into a tabular `argo_info` table, supports geospatial queries (nearest floats & bounding-box), plots trajectories, and answers natural-language questions using an MCP-grounded RAG pipeline with optional vector embeddings.

---

## ✨ Key features

* ✅ **Index ingestion**: download and parse `ar_index_global_prof.txt` into `argo_index`.
* ✅ **NetCDF parsing**: robust parsing of `.nc` files into per-sample rows (`argo_info`) with `pres`, `temp`, `psal`, timestamps and provenance.
* ✅ **Nearest-float search**: compute haversine distances to return nearest floats (`nearest_floats()`).
* ✅ **Trajectory extraction**: build per-float trajectories from `argo_info` (`juld`, `latitude`, `longitude`, `pres`).
* ✅ **Safe SQL builder**: `safe_sql_builder()` produces parameterized SQL for index/measurement queries.
* ✅ **RAG + MCP**: assemble index samples + `.nc` previews + vector hits for LLM grounding (`assemble_mcp_context()`).
* ✅ **Streamlit UI**: polished tabs — Nearest floats, Index Explorer, Bulk Ingest, RAG Chat, Trajectories & Comparison, Exports.
* ✅ **Exports**: Parquet and NetCDF exports for interoperability.

---

## 🧭 Tech stack

Python ecosystem:

* Python 3.9+
* xarray, numpy, pandas — NetCDF parsing & array handling
* SQLAlchemy — DB schema & migration helper
* Streamlit + Plotly — interactive UI and visualizations
* Optional: chromadb (vectors), `langchain-google-genai` (Gemini LLM/embeddings), folium (maps)

---

## 📂 Recommended repo layout

```
app/
  ├─ core.py           # ingestion, parsing, DB helpers
  ├─ rag.py            # MCP & RAG helpers
  ├─ ui.py             # Streamlit app (all tabs)
  ├─ utils.py          # helpers (HTTP session, geocoding)
  └─ config.py         # env loader

storage/               # downloaded .nc, chroma dir, exports
.env                   # environment variables
requirements.txt
README.md
```

---

## 🔑 Environment variables (example)

```ini
ARGO_SQLITE_PATH=./storage/argo.db
ARGO_PG_URL=                     # optional Postgres URL
IFREMER_INDEX_URL=https://data-argo.ifremer.fr/ar_index_global_prof.txt
IFREMER_BASE=https://data-argo.ifremer.fr/dac
AGENTIC_RAG_STORAGE=./storage
GEMINI_API_KEY=YOUR_GOOGLE_API_KEY    # optional for LLM/embeddings
```

> Use Postgres for production; SQLite is fine for local testing.

---

## 🏗️ Full architecture (concise & correct)

> This Mermaid block is formatted to avoid parsing issues in many renderers. For best results, open in a Mermaid-capable viewer if you need the diagram image.

```mermaid
flowchart TB
  IFREMER[IFREMER Index File]
  ParseIndex[parse_index_file()]
  IngestIndex[ingest_index_to_sqlite()
(argo_index)]
  DownloadNC[download_netcdf_for_index_path()]
  ParseNC[parse_profile_netcdf_to_info_rows()]
  IngestInfo[ingest_info_rows()
(argo_info)]
  DBQuery[safe_sql_builder() & DB Query Service]
  Chroma[Chroma & Embeddings (optional)]
  Vector[Vector Retrieval (Chroma)]
  MCP[assemble_mcp_context()]
  LLM[LLM Prompting & RAG]
  RAGAns[rag_answer_with_mcp()]
  Streamlit[Streamlit UI]
  Nearest[nearest_floats() — Haversine]
  TrajSQL[Trajectory extraction (argo_info juld/lat/lon)]

  IFREMER --> ParseIndex --> IngestIndex --> DBQuery
  DownloadNC --> ParseNC --> IngestInfo --> DBQuery
  Chroma --> Vector --> MCP
  DBQuery --> MCP --> LLM --> RAGAns
  Streamlit --> DBQuery
  Streamlit --> DownloadNC
  Streamlit --> RAGAns
  Streamlit --> Nearest
  Streamlit --> TrajSQL
```

**Notes:**

* `nearest_floats()` loads index lat/lon (or filtered subset) and computes Haversine distances in Python (fast and numerically stable).
* Trajectory extraction queries `argo_info` for `juld`, `latitude`, `longitude`, (and `pres` if available) and orders by `juld` to draw paths.
* MCP merges a small index sample, `.nc` CSV heads, and vector metadata (if present) into a short, structured prompt for the LLM.

---

## 📦 Data model & SQL examples

**Tables:**

* `argo_index` — `file`, `date`, `latitude`, `longitude`, `ocean`, `profiler_type`, `institution`, `date_update`, ...
* `argo_info` — `file`, `juld` (datetime), `latitude`, `longitude`, `pres`, `temp`, `psal`, `parameter`, calibration/history fields.

**Nearest floats (client-side distance):**

```sql
SELECT file, latitude, longitude, date FROM argo_index
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
LIMIT 10000;
```

(Then compute distances in Python with `haversine_np()` and return top-N.)

**Trajectory SQL (per-float positions):**

```sql
SELECT file, juld, latitude, longitude, pres
FROM argo_info
WHERE file LIKE '%<FLOAT_ID>.nc' AND latitude IS NOT NULL AND longitude IS NOT NULL
ORDER BY juld ASC;
```

**Example measurement query (from `safe_sql_builder`):**

```sql
SELECT * FROM argo_info
WHERE argo_info.latitude >= :lat_min AND argo_info.latitude <= :lat_max
  AND lower(argo_info.parameter) LIKE :var
  AND argo_info.juld >= :t0_dt AND argo_info.juld <= :t1_dt
ORDER BY juld DESC LIMIT 500;
```

---

## 🧠 RAG & MCP behaviour

1. Parse question into structured filters (via `llm_to_structured()` or `_simple_parse_question()`).
2. If it's a variable-specific query (e.g., "salinity"), prefer `.nc` previews where available and use them as primary data.
3. Otherwise, query `argo_info` using `safe_sql_builder()`.
4. Build MCP context (index sample + `.nc` heads + vector hits) and send a compact prompt to the LLM (temperature=0 recommended).
5. Parse JSON response and show answer + optional SQL / references.

This flow reduces hallucination and prioritises raw instrument data when possible.

---

## 🧪 Testing & validation checklist

* Unit tests: `parse_profile_netcdf_to_info_rows()`, `_to_float_array()`, and variable detection across NetCDF variants.
* Nearest tests: accuracy and edge-cases near dateline/poles.
* SQL builder tests: ensure correct parameterization & filter application.
* Integration: ingest small IFREMER subset and run end-to-end UI/RAG scenarios.

---

## ⚙️ Run & usage

1. Install requirements and create `.env`.
2. Quick index test:

```bash
python -c "from core import ensure_index_file, parse_index_file, ingest_index_to_sqlite; p=ensure_index_file(); df=parse_index_file(p); ingest_index_to_sqlite(df.head(100))"
```

3. Start Streamlit UI:

```bash
streamlit run ui.py
```

---

## 📤 Exports & interoperability

* Export `argo_info` to Parquet (analytics) and NetCDF (scientific consumption) via UI buttons.
* Parquet files are portable to DuckDB, Spark, and BI tools.

---

## 🚀 Deployment suggestions

* Demo: VM with 4–8 vCPUs and 16–32 GB RAM, SQLite or small Postgres.
* Production: Docker/Kubernetes, managed Postgres, S3 for `.nc` storage, dedicated Chroma service.
* Store secrets (GEMINI\_API\_KEY, DB creds) in a secrets manager.

---

## 🤝 Contributing

* Fork, add tests, update README, and open PR. Please include at least one `.nc` sample for parser regression tests.

---

## 🧾 License & contact

* Suggested license: MIT or Apache-2.0 (add LICENSE file).
* Author / Contact: **Ayush Kumar Shaw** — please put GitHub & email in project metadata.

---

✨ This README has been saved to the canvas. Open the canvas to view and export. If you want, I can also:

* Export this README to `README.md` in the repo (file ready for download), or
* Produce a one-page SIH submission PDF or PPTX slide from this content.
