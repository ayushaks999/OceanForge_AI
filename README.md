# ARGO RAG Explorer — Professional README

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-orange.svg)](#) [![Streamlit](https://img.shields.io/badge/streamlit-%3E%3D1.20-%23FF4B4B.svg)](#) [![Status](https://img.shields.io/badge/status-production%2Fprototype-yellowgreen.svg)](#)

> **A production-grade, research-focused platform for ARGO float exploration — combining NetCDF parsing, geospatial indexing, interactive visualization, and grounded LLM answers (RAG + MCP).**

---

## Table of Contents

1. [Quick Visual](#quick-visual)
2. [Why this project](#why-this-project)
3. [Features (short)](#features-short)
4. [Architecture & Data Flow](#architecture--data-flow)
5. [Getting Started (Quickstart)](#getting-started-quickstart)
6. [UI Walkthrough (what each tab does)](#ui-walkthrough-what-each-tab-does)
7. [MCP & RAG — Deep dive](#mcp--rag---deep-dive)
8. [Exports & Integration points](#exports--integration-points)
9. [Production Notes & Recommendations](#production-notes--recommendations)
10. [Developer Guide & Extensibility](#developer-guide--extensibility)
11. [Troubleshooting & Known Limitations](#troubleshooting--known-limitations)
12. [Contribute / License / Credits](#contribute--license--credits)

---

## Quick Visual

*(Add screenshots or GIFs here for: Nearest Floats Map, Chat RAG response, Profile Comparison plots)*

**Hero screenshot placeholder**

> ![hero-placeholder](./docs/screenshots/hero.png)

> Tip: add a 900×380 hero image (map + profile plots side-by-side) to make the repo look polished on GitHub.

---

## Why this project

This repo is built for reproducible oceanographic analysis workflows where domain data (ARGO floats) must be: parsed reliably, stored for efficient queries, visualized interactively, and explained or summarized by LLMs in a grounded/traceable way. It is suitable for researchers, operations teams, and engineers who need a hybrid SQL+RAG approach with auditability.

---

## Features (short)

* Robust NetCDF (.nc) parsing to structured `argo_info` rows (temp, salinity, pressure, time, metadata).
* Index ingestion (`argo_index`) from IFREMER/GDAC index files.
* Place-based geocoding (Nominatim) with sane fallbacks.
* Nearest-float Haversine search, interactive Plotly Mapbox maps.
* Profile comparison (temp/depth, pres/depth, temp vs time) with aggregation options.
* Conversational interface (RAG) with MCP grounding: returns structured JSON answers (answer, recommended SQL, references).
* Optional vector search via ChromaDB — integrated into MCP context.
* Background ingestion & Chroma build workers, status tracking.
* Exports: CSV / Parquet / NetCDF for downstream workflows.

---

## Architecture & Data Flow

**High-level flow**:

1. Index download → parse → persist to `argo_index`.
2. `.nc` download → parse to per-sample rows → persist to `argo_info`.
3. UI queries use `safe_sql_builder()` to construct parameterized SQL (index or measurement queries).
4. For measurement-oriented LLM queries, `.nc` previews are used first (local parsing), else fall back to DB rows.
5. MCP context is constructed (index sample, previews, vector hits) → fed to LLM → structured JSON returned.
6. Visualizations are built via Plotly and served through Streamlit.

**Key components**:

* `parse_profile_netcdf_to_info_rows()` — resilient parser for many NetCDF naming conventions.
* `safe_sql_builder()` — parameterized SQL builder avoiding injection and enforcing constraints.
* `assemble_mcp_context()` & `rag_answer_with_mcp()` — MCP orchestration & deterministic prompting.

---

## Getting Started (Quickstart)

1. Clone the repo and create a virtualenv

```bash
git clone <repo>
cd repo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env`:

```
AGENTIC_RAG_STORAGE=./storage
ARGO_SQLITE_PATH=./storage/argo.db
AGENTIC_RAG_DB_PATH=./storage/agentic_rag_meta.db
# Optional (for LLM/embeddings)
GEMINI_API_KEY=YOUR_GOOGLE_GEMINI_KEY
```

3. Run the app:

```bash
streamlit run app.py --server.port 8501
```

4. Recommended: seed the index by clicking **Ensure index downloaded & ingested (async)** on the sidebar.

---

## UI Walkthrough (what each tab does)

* **Nearest ARGO floats**: input lat/lon or use place lookup → returns nearest N floats + interactive map with distance bubbles.
* **Explore Index**: rich filters (spatial, temporal, ocean, institution) → displays index rows and heatmap-style map preview.
* **Ingest Profiles**: paste index paths (e.g., `aoml/13857/profiles/R13857_001.nc`), download & ingest to populate `argo_info`.
* **Chat (RAG)**: natural language interface. Internally uses rule-based parsing or LLM parsing (if available), prioritizes `.nc` previews for measurement queries, and returns tables + an LLM-generated structured JSON answer.
* **Trajectories & Profile comparison**: select floats (up to 3), visualize trajectories, plot temp/pres vs depth, and compute aggregated per-profile representative temperatures.
* **Exports**: export `argo_info` to Parquet or NetCDF for archival or downstream analysis.

---

## MCP & RAG — Deep dive (what’s implemented and why it matters)

* **MCP (Model Context Protocol)**: the app collects short, high-fidelity context pieces (index samples, parsed `.nc` preview snippets, and vector search metadata) and passes them as the LLM prompt context.
* **Deterministic Output**: LLM is instructed to return a single JSON object containing `answer`, `sql` and `references`. This makes outputs machine-readable and auditable.
* **Chunks & metadata**: every context piece is captured in a `chunks` list for traceability — useful when auditing why a model cited a file or a vector hit.
* **Fallback workflows**: if LLM/embeddings are not available, the app performs deterministic SQL queries using `_simple_parse_question()` to maintain full functionality.

---

## Exports & Integration points

* CSV: per-query downloads for measurement tables and aggregated timeseries.
* Parquet: full `argo_info` export for fast analytics.
* NetCDF: curated netCDF export for interoperability with other oceanographic tools.
* Programmatic API: import `ask_argo_question()` and helper functions in Python to integrate into pipelines.

---

## Production Notes & Recommendations

* Use **Postgres** (set `ARGO_PG_URL`) for high concurrency and to avoid SQLite locks.
* Offload ingestion & Chroma builds to dedicated workers (Celery + Redis recommended).
* Store Chroma / vectors in a managed vector DB for scale (Pinecone, Milvus, or Chroma with remote storage).
* Add Alembic for explicit DB migrations when running on Postgres.

---

## Developer Guide & Extensibility

* **Where to extend**:

  * Add new retrievers (FAISS, ElasticSearch) — integrate with `assemble_mcp_context()`.
  * Improve NetCDF heuristics — add more alias names or per-deployment config mapping.
  * Add auth to Streamlit (proxy with OAuth or Streamlit Enterprise).
* **Testing**: create a small corpus of `.nc` samples representing different DACs and run unit tests against `parse_profile_netcdf_to_info_rows()`.

---

## Troubleshooting & Known Limitations

* NetCDF heterogeneity may lead to missing variables in some profiles — parser uses heuristics and best-effort fallbacks.
* Visualizations drop rows with `temp == 1` (commonly a QC flag). Adjust this behavior if your dataset uses different sentinel values.
* If Streamlit rebuilds frequently when ingesting, run ingestion in a separate process or external worker to avoid UI interruptions.

---

## Contribute / License / Credits

* Contributions welcome. Fork → Branch → PR. Include tests for parsers and SQL builders.
* Licensed under MIT.
* Credits: ARGO program, IFREMER/GDAC, Streamlit, Plotly, Xarray, Pandas, and the open-source ML ecosystem.

---

### Next options I can do for you

* Add polished screenshots and an architecture diagram (I will provide placeholders and instructions for capturing images).
* Produce a pinned `requirements.txt` with pinned versions.
* Generate a sample `.env.example` and a contributors guide.

Tell me which of the above to add and I will embed them into the README in the canvas (screenshots/diagram + requirements + .env example).

---

## Architecture Blueprint — High-level Design & Diagram

Below is a clear, production-oriented architecture for **ARGO RAG Explorer** including components, data flow, and recommended deployment/layouts. The architecture supports both a simple single-host setup (dev/prototype) and a scalable production deployment using Docker/Kubernetes and managed services.

flowchart LR
  A[Researcher / Analyst] -->|interacts via browser| UI[Streamlit UI]
  UI -->|HTTP| App[Streamlit App (app.py)]
  App --> DB[(Relational DB\n(SQLite / Postgres))]
  App --> Storage[(Object Storage / Local FS)]
  App --> IFREMER[IFREMER / GDAC]
  App --> Nom[Nominatim (Geocoding)]
  App --> LLM[LLM / Embeddings\n(Google Gemini via LangChain)]
  App --> Chroma[Chroma DB / Vector DB]
  App --> Status[status.json]

  subgraph Workers
    Queue[Message Queue\n(Redis / RabbitMQ)]
    Worker[Ingest Worker\n(Celery / RQ)]
    Queue --> Worker
    App -->|enqueue jobs| Queue
    Worker --> Storage
    Worker --> DB
    Worker --> Chroma
  end

  subgraph Optional
    Chroma -->|persist| ObjStorage[(S3 / Blob Storage)]
  end

  style App fill:#f9f,stroke:#333,stroke-width:1px
  style Worker fill:#ff9,stroke:#333,stroke-width:1px
