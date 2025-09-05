# ARGO RAG Explorer

> **A professional Retrieval-Augmented-Generation (RAG) platform for ARGO oceanographic profiles** — index, ingest, explore and chat with Argo `.nc` profiles. Built with Python, xarray, SQLAlchemy, Streamlit and optional LLM + embeddings for RAG (Gemini / Chroma). ⚓️🌊

---

## 🚀 Quick snapshot

* **Purpose:** Provide a single-entry web UI to search IFREMER Argo index, download `.nc` profiles, ingest per-sample rows into a relational store, and answer natural-language queries using RAG + MCP (Multi-Context-Prompting).
* **Core pieces:** Index ingestion, `.nc` parsing, `argo_info` measurement table, optional vector index (Chroma), LLM + embeddings adapters, and a Streamlit frontend for exploration, chat, and profile comparisons.
* **Ideal for:** oceanographers, data scientists, and SIH teams building an entry-level RAG system on scientific netCDF data.

---

## ✨ Highlights & Features

* Ingests IFREMER Argo index and downloads NetCDF profiles.
* Extracts per-sample rows (depth/temp/psal + metadata) to `argo_info` table (SQLite or Postgres).
* Powerful query builder with safe parameterization & geospatial bbox filters.
* Optional vector embedding index (Chroma) for semantic retrieval over profile metadata and previews.
* Multi-Context Prompting (MCP) — assemble index samples + `.nc` previews + vector hits as retrieval context for the LLM.
* Streamlit UI with tabs: Nearest floats, Index explorer, Bulk ingest, RAG Chat, Trajectories & Profile comparison, Exports.
* Export ingested data to Parquet / NetCDF for downstream analysis.

---

## ⚙️ Technologies

* Python 3.9+
* xarray, numpy, pandas — netCDF parsing & numeric handling
* SQLAlchemy — DB schema + migration helpers
* Streamlit + Plotly — interactive UI & plotting
* Optional: chromadb, Gemini (via `langchain-google-genai`) for embeddings & LLMs
* Optional: folium / streamlit\_folium for map rendering

---

## 🗂 Repository structure (recommended)

```
app/                     # main package (parts 1..3 combined)
  │
  ├─ core.py             # ingestion, parsing, DB helpers (Part 1)
  ├─ rag.py              # MCP & RAG helpers
  ├─ ui.py               # Streamlit app (Part 2 & 3)
  ├─ utils.py            # small helpers, geocoding, http session
  └─ config.py           # env/config loader

storage/                 # downloaded .nc files, chroma dir, exports
.env                     # GEMINI_API_KEY, DB urls, other env
README.md                # this document
requirements.txt
```

---

## 📥 Installation (local dev)

1. Clone repo and create venv:

```bash
git clone <repo-url>
cd repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` (see **Environment variables** below).
3. Run Streamlit:

```bash
streamlit run ui.py
```

---

## 🧩 Environment variables (recommended)

Create a `.env` with these keys (example values):

```ini
# DB
ARGO_SQLITE_PATH=./storage/argo.db
ARGO_PG_URL=          # optional Postgres URL (postgres://user:pass@host/db)

# IFREMER index
IFREMER_INDEX_URL=https://data-argo.ifremer.fr/ar_index_global_prof.txt
IFREMER_BASE=https://data-argo.ifremer.fr/dac

# Storage
AGENTIC_RAG_STORAGE=./storage

# Optional LLM + embedding keys
GEMINI_API_KEY=YOUR_GOOGLE_API_KEY
```

> Tip: use Postgres for production; SQLite is convenient for development.

---

## 🧭 High-level architecture

Below is a concise view of the major components and how they interact.

```mermaid
flowchart TB
  subgraph Ingest
    A[Index file (IFREMER)] --> B[ensure_index_file()
(parse_index_file)]
    B --> C[ingest_index_to_sqlite()
(argo_index table)]
  end

  subgraph Download & Parse
    D[download_netcdf_for_index_path()] --> E[parse_profile_netcdf_to_info_rows()]
    E --> F[ingest_info_rows()
(argo_info table)]
  end

  subgraph Retrieval
    G[Safe SQL builder] --> H[DB queries (argo_index / argo_info)]
    I[Chroma / embeddings] --> J[vector hits]
  end

  subgraph RAG
    H & J --> K[assemble_mcp_context()]
    K --> L[LLM prompt]
    L --> M[rag_answer_with_mcp()]
  end

  subgraph UI
    N[Streamlit] --> H
    N --> D
    N --> M
    N --> F
  end

  B --> N
  F --> N
```

> The MCP block combines index samples, `.nc` previews, and vector hits into a single retrieval context sent to the LLM.

---

## 📚 Data model

* `argo_index` — raw IFREMER index rows (file path, date, lat, lon, ocean, institution, etc.)
* `argo_info` — per-measurement rows: `juld`, `latitude`, `longitude`, `pres`, `temp`, `psal`, `parameter`, calibration & history fields.

Schema created dynamically via SQLAlchemy; migration helper `_ensure_info_table_schema()` adds missing columns.

---

## 🧪 Usage & examples

* **Nearest floats:** give lat/lon → `nearest_floats()` computes haversine distances and returns top N.
* **Place lookup:** free-text geocoding (Nominatim fallback) → bounding box → index query.
* **Bulk ingest:** paste index paths → download `.nc` → parse → ingest per-sample rows.
* **Chat (RAG):** ask natural language questions; system prefers `.nc` previews for measurement queries and falls back to DB rows.
* **Compare:** select up to 3 floats to compare trajectories (from `argo_info`) and profile variables (temp/psal).

---

## 🧠 RAG & MCP notes (best practices)

* **Prefer `.nc` previews as primary source** for variable-specific questions (RAG uses parsed numeric rows directly when available).
* **MCP assembly** merges index sample text, `.nc` CSV previews, and vector hits (if chroma present) — reduces LLM hallucination by grounding prompts.
* **Embedding model** should be small & consistent; regenerate Chroma when index changes.

---

## ⚙️ Deployment suggestions

* **Small team / demo:** host on a VM (4–8 cores, 16–32GB RAM) with Postgres; schedule async index ingestion and Chroma builds.
* **Production:** containerize (Docker) + Kubernetes; use object storage for `.nc` files (S3) and DuckDB/Parquet or Postgres for metadata.
* **Secrets:** store `GEMINI_API_KEY` and DB creds in a secure secrets manager (Vault / cloud secret store).

---

## ✅ Testing & validation

* Unit-test: parsing of representative `.nc` files (different dim layouts), `_to_float_array()` behaviors, and SQL builder edge cases.
* Integration: small IFREMER index subset (100–1k rows) — verify download → parse → ingest → queries → UI flows.
* LLM tests: use deterministic prompt responses (zero temperature) to validate MCP prompt shape and JSON parsing.

---

## 📈 Performance & scaling tips

* Use Postgres with PostGIS for heavy geospatial filtering and indexing.
* Avoid eager download of all `.nc`; preview first and download only selected files.
* Batch DB inserts (already implemented) and tune batch size depending on DB throughput.
* For very large vector indices, run Chroma on a separate service with its own persist directory.

---

## 📦 Exports & interoperability

* Export `argo_info` to Parquet (fast analytics) and NetCDF (scientific compatibility) — included in UI.
* Downstream users can mount Parquet into Spark / DuckDB for large-scale analysis.

---

## 🤝 Contributing

* Fork → PR with feature branch; include tests and update README when adding major features.
* Please add one `.nc` sample and expected parsed CSV for parser regression tests.

---

## 🧾 License & Contact

* *Suggested*: MIT or Apache-2.0 (pick one and add LICENSE file).
* Contact: **Ayush Kumar Shaw** — include your GitHub / email in project metadata.

---

✨ *Want this README exported to `README.md` in the repo or a tailored version for SIH submission (short summary + architecture slide)?*
