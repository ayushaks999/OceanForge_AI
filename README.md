flowchart LR
  U[User_browser] --> S[Streamlit_App]
  S --> A[ask_argo_question]
  A --> IDX[argo_index_DB]
  A --> INFO[argo_info_DB]
  A --> NC[Downloaded_nc_files]
  A --> RAG[RAG_LLM_Gemini]
  RAG --> CHR[Chroma_vector_DB_optional]
  S --> BG[Background_workers]
  BG --> IDX
  BG --> CHR
  IDX -->|file_paths| NC
  NC -->|parse_to_rows| INFO

  subgraph Storage_and_Local_State
    IDX
    INFO
    NC
    CHR
    STATUS[STATUS_FILE_build_status.json]
    USERDB[USER_DB_chats_feedback]
  end

  RAG -->|answers_and_context| S
  USERDB --> S
  S -->|exports| EXPORTS[Parquet_NetCDF_files]
