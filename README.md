# ARGO RAG Explorer — Full Architecture Diagram

This document contains a complete architecture drawing (ASCII/flow) and detailed explanation of every component, storage location, data flow, and the key functions that move data between components. Use this as a single-source reference for how the app works end‑to‑end.

---

## Visual architecture (ASCII flow)

```
                         +------------------------+
                         |   User (browser)       |
                         |  - Streamlit UI        |
                         +-----------+------------+
                                     |
                                     | HTTP / Interactive
                                     v
               +-----------------------------------------------+
               | STREAMLIT APP (parts 2 & 3)                  |
               | - UI tabs: Nearest, Explore, Ingest, Chat,   |
               |   Compare, Exports                            |
               | - Orchestrates user actions / displays plots  |
               +-----------+-----------------------------------+
                           |                    |
          ask / UI action |                    | Background/async
                           v                    v
        +-------------------------------+   +------------------------+
        | ask_argo_question(...)        |   | Background workers     |
        | - llm_to_structured / parse   |   | - _index_ingest_worker |
        | - safe_sql_builder(index)     |   | - _chroma_build_worker |
        | - query argo_index            |   | - spawn via multiprocessing
        | - prefer .nc previews         |   +------------------------+
        | - assemble_mcp_context()      |
        | - rag_answer_with_mcp()       |
        +---------+---------------------+
                  |
     +------------+-------------------------------+
     |                                            |
     v                                            v
+-----------+                               +----------------+
| IFREMER   |                               | Local STORAGE   |
| Remote    |  <-- index URL (IFREMER_BASE)  | (STORAGE_ROOT)  |
| (index &  |                               | - ar_index_global_prof.txt
|  .nc host)|                               | - downloaded .nc files
+-----------+                               | - chromadb dir
                                            | - exports (parquet/netcdf)
                                            +---------+------+
                                                      |
                                                      v
                                         +------------------------------+
                                         | NetCDF parsing & ingestion   |
                                         | - download_netcdf_for_index_path()
                                         | - read_netcdf_variables_to_df()
                                         | - parse_profile_netcdf_to_info_rows()
                                         | - ingest_info_rows() -> argo_info table
                                         +------------------------------+
                                                      |
                                                      v
             +--------------------------------+    +---------------------------+
             | Relational DB (SQLite/Postgres)|<---| SQLAlchemy engine         |
             | (DB_ENGINE_URL)                |    | - tables: argo_index,    |
             | - argo_index                   |    |   argo_info               |
             | - argo_info (per-sample rows)  |    +---------------------------+
             +--------------------------------+
                            ^  ^
                            |  |
      queries (safe_sql_builder) | measurement queries (argo_info)


+----------------------+    +------------------------+
| Embeddings / LLM     |    | Chroma vector DB       |
| (Gemini via LangChain)|   | (optional)             |
| - ensure_models()    |    | - persisted in CHROMA_DIR
| - llm.invoke(prompt) |    | - collections per user
+--------+-------------+    +----------+-------------+
         |                              |
         | embeddings & context         | vector search results
         v                              v
  rag_answer_with_mcp() <--------------- assemble_mcp_context()
  (LLM uses context_text + chunks)     (index sample + .nc previews + vector hits)

+------------------+
| USER_DB (sqlite) |
| USER_DB_PATH      |
| - chats, feedback |
+------------------+

+------------------+
| STATUS_FILE JSON  |
| CHROMA_DIR/build_status.json
+------------------+
```

---

## Component descriptions & primary storage

* **Streamlit UI** (parts 2 & 3): user facing. No long-term storage beyond session state. Triggers `ask_argo_question`, ingestion, exports, and background workers.

* **STORAGE\_ROOT** (filesystem): single place for big artifacts:

  * `ar_index_global_prof.txt` (IFREMER index copy)
  * downloaded `.nc` files (kept under the same relative paths as in the index)
  * `chromadb/` (Chroma persistence directory)
  * exported files (parquet, netcdf)

* **Relational DB (engine)**:

  * `argo_index` table — compact file index produced from the IFREMER index (fast lookups of floats with lat/lon/date).
  * `argo_info` table — detailed ingested per-sample rows (one row per depth/sample) with many metadata columns. This is the main analytical table.
  * Engine uses `SQLAlchemy` and defaults to a local SQLite DB (`ARGO_SQLITE_PATH`) or Postgres if `ARGO_PG_URL` provided.

* **Chroma vector DB (optional)**: stores embeddings for retrieval (collections named e.g. `floats_user_0`). Persisted to `CHROMA_DIR` (duckdb+parquet or other) so vector search survives restarts.

* **User DB** (`USER_DB_PATH`) — small sqlite used for storing chat history and user feedback (tables: `chats`, `feedback`).

* **STATUS\_FILE** — build status and background-job state (JSON) for UI to show progress.

---

## Key function-role mapping (how data moves)

* **Index acquisition**: `ensure_index_file()`, `parse_index_file()`, `ingest_index_to_sqlite()`

  * Remote IFREMER index → local file → `argo_index` DB table.

* **NetCDF download & previews**: `get_local_netcdf_path_from_indexfile()`, `download_netcdf_for_index_path()`, `read_netcdf_variables_to_df()`

  * `.nc` files downloaded into `STORAGE_ROOT`, previews used by UI & RAG.

* **Parse & ingest to argo\_info**: `parse_profile_netcdf_to_info_rows()`, `ingest_info_rows()`

  * Converts `.nc` file contents into many per-measurement rows to populate `argo_info`.

* **Query building**: `safe_sql_builder(filters, target)`

  * Converts structured filters into parameterized SQL for `argo_index` or `argo_info`.

* **Nearest floats**: `nearest_floats(lat, lon)`

  * Pulls candidates from `argo_index`, computes Haversine distances and returns nearest.

* **LLM/RAG helpers**: `llm_to_structured()`, `assemble_mcp_context()`, `rag_answer_with_mcp()`

  * Build LLM prompts, gather context (index sample + .nc previews + Chroma hits), call LLM and parse JSON output.

* **Top-level orchestration**: `ask_argo_question()`

  * Uses parser → index query → previews or DB measurement query → optional RAG answer → returns everything to UI.

* **Background async**: `start_index_ingest_async()`, `start_chroma_build_async()`

  * Spawn processes to build index and vector DB; write status into `STATUS_FILE`.

---

## End-to-end sequence (user query example)

1. User types: "salinity in Arabian Sea in March 2023" in Streamlit chat.
2. `ask_argo_question()` runs `llm_to_structured()` or `_simple_parse_question()` → produces filters (ocean=I, time range, variable=psal).
3. `safe_sql_builder(..., target='index')` → SQL executed against `argo_index` → returns candidate index rows.
4. Since variable is measurement-like, code prefers `.nc` previews:

   * For each index row: `get_local_netcdf_path_from_indexfile()` → `download_netcdf_for_index_path()` if missing → `read_netcdf_variables_to_df()` to produce previews.
5. If previews available, combine previews and skip `argo_info` query; otherwise `safe_sql_builder(..., target='measurements')` queries `argo_info`.
6. `assemble_mcp_context()` gathers context text (index sample, preview CSVs, vector hits) and `rag_answer_with_mcp()` calls the LLM to create a JSON answer.
7. UI displays RAG answer, index rows, measurement rows, plots, and file download buttons.

---

## Notes, caveats & suggestions

* **Chroma build**: implementation of `build_chroma_from_index()` is referenced but must embed sensible metadata (file path, mean lat/lon, date, sample text) for useful retrieval.
* **Scaling**: `argo_info` can become large. Consider partitioning by year or platform\_number, and use Postgres for large-scale usage.
* **Caching**: cache `.nc` previews (CSV) after first parse to speed repeated queries.
* **Parallel ingestion**: parse `.nc` files in parallel worker pool instead of one-by-one for bulk ingest.
* **Robust QC**: many datasets use sentinel values (temp==1) — you already filter these in plotting code; consider cleaning upstream during ingest.

---

If you'd like, I can now:

* produce a **PNG/SVG** version of this diagram,
* convert the ASCII flow into a **Mermaid** diagram for embedding, or
* create a **sequence diagram** for `ask_argo_question()` and background workers.

Tell me which format you want (PNG / SVG / Mermaid / Sequence), and I’ll generate it next.

## Mermaid diagram

Below is a Mermaid flowchart representation of the architecture. You can copy the block and render it in any Mermaid-compatible renderer (Mermaid Live Editor, VS Code with Mermaid preview, MkDocs, etc.).

```mermaid
flowchart LR
  U[User (browser)] --> S[Streamlit App]
  S --> A[ask_argo_question]
  A --> IDX[argo_index (DB)]
  A --> INFO[argo_info (DB)]
  A --> NC[Downloaded .nc files (STORAGE_ROOT)]
  A --> RAG[RAG / LLM (Gemini via LangChain)]
  RAG --> CHR[Chroma (vector DB) - optional]
  S --> BG[Background workers]
  BG --> IDX
  BG --> CHR
  IDX -->|file paths| NC
  NC -->|parse -> rows| INFO

  subgraph Storage[Storage & Local State]
    IDX
    INFO
    NC
    CHR
    STATUS[STATUS_FILE (build_status.json)]
    USERDB[USER_DB (chats, feedback)]
  end

  RAG -->|answers & context| S
  USERDB --> S
  S -->|exports| EXPORTS[Parquet / NetCDF files]
```

If you'd like, I can now export this Mermaid as a PNG or SVG and attach it. Which format do you prefer?
