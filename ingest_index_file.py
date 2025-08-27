# ingest_index_file.py
import os
import pandas as pd
from sqlalchemy import create_engine, text

# adjust to match your app settings or .env values
STORAGE_ROOT = os.path.abspath("./storage")
DB_PATH = os.path.abspath(os.getenv("ARGO_SQLITE_PATH", "argo.db"))
SQLITE_URL = f"sqlite:///{DB_PATH}"

INDEX_LOCAL_FILE = "ar_index_this_week_prof.txt"   # your file in repo root or storage
INDEX_LOCAL_PATHS = [
    os.path.join(".", INDEX_LOCAL_FILE),
    os.path.join(STORAGE_ROOT, INDEX_LOCAL_FILE),
]

def detect_and_read_index(path):
    # Read first non-empty line to detect header
    with open(path, "r", encoding="utf-8") as fh:
        first = ""
        for line in fh:
            s = line.strip()
            if s:
                first = s
                break
    hdr = first.lower().replace(" ", "")
    if hdr.startswith("file,date,latitude"):
        # CSV (header present)
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        expected = ["file","date","latitude","longitude","ocean","profiler_type","institution","date_update"]
        # Ensure all expected columns exist
        for c in expected:
            if c not in df.columns:
                df[c] = None
        # Coerce lat/lon
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        # Normalize ocean codes (optional)
        df["ocean"] = df["ocean"].astype(str).str.strip().str.upper().str[:1].replace({"": None})
        return df[expected]
    # else fallback to original parser
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            if len(parts) < 8:
                continue
            try:
                rows.append({
                    "file": parts[0].strip(),
                    "date": parts[1].strip(),
                    "latitude": float(parts[2]) if parts[2] not in ("", "NA") else None,
                    "longitude": float(parts[3]) if parts[3] not in ("", "NA") else None,
                    "ocean": parts[4].strip(),
                    "profiler_type": parts[5].strip(),
                    "institution": parts[6].strip(),
                    "date_update": parts[7].strip(),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)

def main():
    # find file
    path = None
    for p in INDEX_LOCAL_PATHS:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        print("Index file not found. Put ar_index_this_week_prof.txt in repo root or ./storage and retry.")
        return

    df = detect_and_read_index(path)
    if df is None or df.empty:
        print("No rows detected in index file.")
        return

    # write to sqlite
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    # backup existing table if desired (optional)
    # with engine.begin() as conn:
    #     conn.execute(text("ALTER TABLE argo_index RENAME TO argo_index_backup;"))
    df.to_sql("argo_index", con=engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM argo_index")).scalar()
    print(f"Ingest complete — {int(n)} rows inserted into argo_index (DB: {DB_PATH})")

if __name__ == "__main__":
    from sqlalchemy import create_engine
    main()
