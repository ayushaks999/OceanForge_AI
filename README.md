# ARGO RAG Explorer

> **A professional Retrieval-Augmented-Generation (RAG) platform for ARGO oceanographic profiles** — index, ingest, explore, compare trajectories, and chat with Argo `.nc` profiles. ⚓️🌊

---

## 🚀 One-line summary

ARGO RAG Explorer is a polished Streamlit application that ingests the IFREMER ARGO index, downloads and parses NetCDF profiles into a tabular `argo_info` table, supports geospatial queries (nearest floats & bounding-box), plots trajectories, and answers natural-language questions using an MCP-grounded RAG pipeline with optional vector embeddings.

---

## ✨ Highlights & Features

* Robust ingestion of IFREMER `ar_index_global_prof.txt` into `argo_index` table.
* Parse `.nc` NetCDF profiles via `xarray` into per-sample rows (depth / temp / psal + rich metadata) in `argo_info`.
* Nearest-float search using Haversine distance (`nearest_floats()`), and trajectory extraction from `argo_info` for per-float path plotting.
* Safe SQL builder (`safe_sql_builder()`) for parameterized spatial, temporal, and variable queries.
* Optional semantic retrieval with Chroma (vector index) + embeddings for better RAG results.
* MCP (Multi-Context Prompting): assemble index samples, `.nc` CSV previews, and vector hits to ground the LLM.
* Streamlit UI with tabs for Nearest floats, Index Explorer, Bulk Ingest, RAG Chat, Trajectories & Comparison, and Exports.
* Export ingested data to Parquet and NetCDF for downstream analytics and scientific workflows.

---

## 🧭 Technical stack

* Python 3.9+
* xarray, numpy, pandas — NetCDF parsing and numeric manipulations
* SQLAlchemy — DB schema management & migration helper
* Streamlit + Plotly — UI and visualizations
* Optional: chromadb (vector store), langchain-google-genai (Gemini LLM/embeddings), folium (maps)

---

## 📁 Recommended repo layout

```
app/
  ├─ core.py           # ingestion, parsing, DB helpers
  ├─ rag.py            # MCP & RAG helpers (assemble contexts, call LLM)
  ├─ ui.py             # Streamlit app (tabs & views)
  ├─ utils.py          # helpers (HTTP session, geocoding)
  └─ config.py         # env config loader

storage/               # downloaded .nc, chroma dir, exports
.env                   # environment variables
requirements.txt
README.md
```

---

## 🔧 Environment variables (example)

```ini
ARGO_SQLITE_PATH=./storage/argo.db
ARGO_PG_URL=                     # optional Postgres URL
IFREMER_INDEX_URL=https://data-argo.ifremer.fr/ar_index_global_prof.txt
IFREMER_BASE=https://data-argo.ifremer.fr/dac
AGENTIC_RAG_STORAGE=./storage
GEMINI_API_KEY=YOUR_GOOGLE_API_KEY    # optional for LLM/embeddings
```

> Use Postgres for production; SQLite is convenient for development and demos.

---

## 🧭 Full architecture (with trajectory & nearest-float components)

Here is an updated architecture diagram that explicitly includes the trajectory extraction and nearest-float logic.

```mermaid
flowchart TB
  subgraph Index
    A[IFREMER Index File] --> B[parse_index_file()]
    B --> C[ingest_index_to_sqlite() 
(argo_index)]
  end

  subgraph Download_Parse
    D[download_netcdf_for_index_path()] --> E[parse_profile_netcdf_to_info_rows()]
    E --> F[ingest_info_rows() 
(argo_info)]
  end

  subgraph Retrieval
    C --> G[safe_sql_builder()]
    F --> G
    G --> H[DB Query Service]
    I[Chroma & Embeddings] --> J[Vector Retrieval]
    H & J --> K[assemble_mcp_context()]
    K --> L[LLM Prompting & RAG]
    L --> M[rag_answer_with_mcp()]
  end

  subgraph UI
    N[Streamlit UI] --> H
    N --> D
    N --> M
    N --> O[nearest_floats() 
(Haversine)]
    N --> P[trajectory SQL 
(extract per-profile lat/lon from argo_info)]
  end
```

**Key additions:**

* **nearest\_floats():** reads `argo_index` lat/lon and computes haversine distances to return nearest floats; used by the "Nearest ARGO floats" UI tab.
* **Trajectory SQL:** queries `argo_info` for per-profile `juld`, `latitude`, `longitude` and optional `pres` to build trajectories and time-series — used by the Trajectories & Profile comparison tab.

---

## 📚 Data model & example SQL snippets

### Tables

* **argo\_index** — columns: `file`, `date`, `latitude`, `longitude`, `ocean`, `profiler_type`, `institution`, `date_update`, etc.
* **argo\_info** — columns: `file`, `juld` (datetime), `latitude`, `longitude`, `pres`, `temp`, `psal`, `parameter`, calibration/history fields, etc.

### Example: Nearest floats (haversine approach)

```sql
SELECT file, latitude, longitude, date
FROM argo_index
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
LIMIT 10000; -- then compute distances client-side and pick top-N
```

(Implementation computes distances in Python using `haversine_np()` for speed & numeric stability.)

### Example: Trajectories — per-float positions

```sql
SELECT file, juld, latitude, longitude, pres
FROM argo_info
WHERE file LIKE '%<FLOAT_ID>.nc' AND latitude IS NOT NULL AND longitude IS NOT NULL
ORDER BY juld ASC;
```

### Example: Measurement query (safe\_sql\_builder output)

```sql
SELECT * FROM argo_info
WHERE argo_info.latitude >= :lat_min AND argo_info.latitude <= :lat_max
  AND lower(argo_info.parameter) LIKE :var
  AND argo_info.juld >= :t0_dt AND argo_info.juld <= :t1_dt
ORDER BY juld DESC LIMIT 500;
```

---

## 🧪 How the RAG flow prefers sources

1. Parse user question (LLM `llm_to_structured()` or fallback rule parser).
2. If variable-specific (e.g., "salinity"), try to load `.nc` previews for matched index rows and use them as primary source.
3. Otherwise, query `argo_info` with `safe_sql_builder()` to fetch measurement rows.
4. Build MCP context including a small index sample, `.nc` CSV heads, and vector hits (if available).
5. Send compact, clearly-formatted prompt to the LLM and parse the returned JSON answer.

This ordering reduces hallucination and uses raw instrument data where possible.

---

## 🧪 Testing checklist (expanded)

* Unit tests for `parse_profile_netcdf_to_info_rows()` across NetCDF variants (dimension names & shapes).
* Tests for `nearest_floats()` correctness (haversine) and edge cases near the dateline / poles.
* Tests for `safe_sql_builder()` making sure parameterized SQL is safe and filters are applied correctly.
* Integration test: ingest a small IFREMER subset and validate end-to-end RAG answers and visualizations.

---

## ⚙️ Run & usage

1. Install requirements and create `.env`.
2. Optionally index a small sample first for fast experimentation:

```bash
python -c "from core import ensure_index_file, parse_index_file, ingest_index_to_sqlite; p=ensure_index_file(); df=parse_index_file(p); ingest_index_to_sqlite(df.head(100))"
```

3. Start UI:

```bash
streamlit run ui.py
```

---

## 📦 Exports

* Export `argo_info` to Parquet or NetCDF (provided by UI). Parquet for analytics (DuckDB/Spark), NetCDF for scientific tools.

---

## 🚀 Deployment suggestions

* Small demo: 4–8 CPU cores, 16–32 GB RAM, SQLite or small Postgres instance.
* Production: Docker + Kubernetes, managed Postgres, S3 for `.nc` files, dedicated Chroma/embeddings service.
* Use secrets manager for `GEMINI_API_KEY` and DB credentials.

---

## 🧾 License & Contact

* Suggested license: MIT or Apache-2.0 (add LICENSE file).
* Contact: **Ayush Kumar Shaw** — add GitHub and email in project metadata.

---

✨ I updated the README to include the **nearest-floats** and **trajectory SQL** details and expanded the architecture and SQL examples. Want a condensed SIH one-sheet or a PPT slide deck export next?
