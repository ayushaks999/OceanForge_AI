flowchart TB
  IFREMER["IFREMER Index File"]
  ParseIndex["parse_index_file()"]
  IngestIndex["ingest_index_to_sqlite()\nargo_index"]
  DownloadNC["download_netcdf_for_index_path()"]
  ParseNC["parse_profile_netcdf_to_info_rows()"]
  IngestInfo["ingest_info_rows()\nargo_info"]
  DBQuery["safe_sql_builder()\n& DB Query Service"]
  Chroma["Chroma & Embeddings (optional)"]
  Vector["Vector Retrieval\nChroma"]
  MCP["assemble_mcp_context()"]
  LLM["LLM Prompting & RAG"]
  RAGAns["rag_answer_with_mcp()"]
  Streamlit["Streamlit UI"]
  Nearest["nearest_floats() — Haversine"]
  TrajSQL["Trajectory extraction\n(argo_info juld/lat/lon)"]

  IFREMER --> ParseIndex
  ParseIndex --> IngestIndex
  DownloadNC --> ParseNC
  ParseNC --> IngestInfo
  IngestIndex --> DBQuery
  IngestInfo --> DBQuery
  Chroma --> Vector
  Vector --> MCP
  DBQuery --> MCP
  MCP --> LLM
  LLM --> RAGAns
  Streamlit --> DBQuery
  Streamlit --> DownloadNC
  Streamlit --> RAGAns
  Streamlit --> Nearest
  Streamlit --> TrajSQL
