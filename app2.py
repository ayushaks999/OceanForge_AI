# PART 1: ARGO RAG Explorer - BACKEND (db, ingestion, parsers, helpers)
# Paste this above PART 2 in a single app.py, or save as a module and import in PART 2.

import os
import re
import json
import time
import tempfile
import sqlite3
import multiprocessing
import datetime
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import requests

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, text, inspect
from dotenv import load_dotenv

# Optional LLM + embeddings (Gemini via LangChain or other wrappers). Keep graceful.
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except Exception:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

# Optional chromadb - optional
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

load_dotenv()

# ---------------------------
# Config / Paths
# ---------------------------

def _get_env(k, default=None):
    v = os.environ.get(k)
    return v if v is not None else default

# Primary local sqlite store (can be swapped for PostgreSQL by setting ARGO_PG_URL env)
ARGO_SQLITE_PATH = _get_env("ARGO_SQLITE_PATH", "argo.db")
DB_PATH = os.path.abspath(ARGO_SQLITE_PATH)
SQLITE_URL = f"sqlite:///{DB_PATH}"

# Optional PostgreSQL: set ARGO_PG_URL (e.g. postgres://user:pass@host:5432/dbname)
POSTGRES_URL = _get_env("ARGO_PG_URL", None)

STORAGE_ROOT = os.path.abspath(_get_env("AGENTIC_RAG_STORAGE", "./storage"))
os.makedirs(STORAGE_ROOT, exist_ok=True)

INDEX_LOCAL_PATH = os.path.join(STORAGE_ROOT, "ar_index_global_prof.txt")
INDEX_REMOTE_URL = _get_env("IFREMER_INDEX_URL", "https://data-argo.ifremer.fr/ar_index_global_prof.txt")
IFREMER_BASE = _get_env("IFREMER_BASE", "https://data-argo.ifremer.fr/dac")

USER_DB_PATH = os.path.abspath(_get_env("AGENTIC_RAG_DB_PATH", os.path.join(STORAGE_ROOT, "agentic_rag_meta.db")))

CHROMA_DIR = os.path.join(STORAGE_ROOT, "chromadb")
os.makedirs(CHROMA_DIR, exist_ok=True)

STATUS_FILE = os.path.join(CHROMA_DIR, "build_status.json")

GEMINI_API_KEY = _get_env("GEMINI_API_KEY")

# Multiprocessing start method (best-effort)
try:
    multiprocessing.set_start_method('fork')
except Exception:
    pass

# Choose engine: prefer Postgres if configured otherwise sqlite
DB_ENGINE_URL = POSTGRES_URL if POSTGRES_URL else SQLITE_URL
engine = create_engine(DB_ENGINE_URL, connect_args={"check_same_thread": False} if SQLITE_URL in DB_ENGINE_URL else {})
metadata = MetaData()

# ---------------------------
# Schema (lightweight) — suitable for both SQLite & Postgres
# ---------------------------
argo_index_table = Table(
    "argo_index", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("file", String, index=True),
    Column("date", String, index=True),
    Column("latitude", Float, index=True),
    Column("longitude", Float, index=True),
    Column("ocean", String, index=True),
    Column("profiler_type", String),
    Column("institution", String),
    Column("date_update", String, index=True),
)

# New table name: argo_info (stores ingested .nc profile metadata + per-profile measurements as rows)
argo_info_table = Table(
    "argo_info", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("file", String, index=True),
    Column("data_type", String),
    Column("format_version", String),
    Column("handbook_version", String),
    Column("reference_date_time", String),
    Column("date_creation", String),
    Column("date_update", String),
    Column("platform_number", String, index=True),
    Column("project_name", String),
    Column("pi_name", String),
    Column("station_parameters", String),
    Column("cycle_number", String),
    Column("direction", String),
    Column("data_centre", String),
    Column("dc_reference", String),
    Column("data_state_indicator", String),
    Column("data_mode", String),
    Column("platform_type", String),
    Column("float_serial_no", String),
    Column("firmware_version", String),
    Column("wmo_inst_type", String),
    Column("juld", DateTime, index=True),
    Column("juld_qc", String),
    Column("juld_location", String),
    Column("latitude", Float, index=True),
    Column("longitude", Float, index=True),
    Column("position_qc", String),
    Column("positioning_system", String),
    Column("vertical_sampling_scheme", String),
    Column("config_mission_number", String),
    Column("profile_pres_qc", String),
    Column("profile_temp_qc", String),
    Column("profile_psal_qc", String),
    Column("pres", Float, index=True),
    Column("pres_qc", String),
    Column("pres_adjusted", Float),
    Column("pres_adjusted_qc", String),
    Column("pres_adjusted_error", Float),
    Column("temp", Float, index=True),
    Column("temp_qc", String),
    Column("temp_adjusted", Float),
    Column("temp_adjusted_qc", String),
    Column("temp_adjusted_error", Float),
    Column("psal", Float, index=True),
    Column("psal_qc", String),
    Column("psal_adjusted", Float),
    Column("psal_adjusted_qc", String),
    Column("psal_adjusted_error", Float),
    Column("parameter", String, index=True),
    Column("scientific_calib_equation", String),
    Column("scientific_calib_coefficient", String),
    Column("scientific_calib_comment", String),
    Column("scientific_calib_date", String),
    Column("history_institution", String),
    Column("history_step", String),
    Column("history_software", String),
    Column("history_software_release", String),
    Column("history_reference", String),
    Column("history_date", String),
    Column("history_action", String),
    Column("history_parameter", String),
    Column("history_start_pres", Float),
    Column("history_stop_pres", Float),
    Column("history_previous_value", String),
    Column("history_qctest", String),
)

# Create tables if not present (idempotent)
metadata.create_all(engine)

# Provide a small migration helper: add missing columns to argo_info if DB pre-exists
def _ensure_info_table_schema():
    try:
        insp = inspect(engine)
        if not insp.has_table('argo_info'):
            return
        existing = [c['name'] for c in insp.get_columns('argo_info')]
        def _col_type_sql(col):
            from sqlalchemy import Integer as _Int, Float as _F, DateTime as _DT
            t = col.type
            if isinstance(t, _Int):
                return 'INTEGER'
            if isinstance(t, _F):
                return 'REAL'
            if isinstance(t, _DT):
                return 'TEXT'
            return 'TEXT'
        with engine.begin() as conn:
            for col in argo_info_table.columns:
                if col.name not in existing:
                    sqltype = _col_type_sql(col)
                    try:
                        conn.execute(text(f'ALTER TABLE argo_info ADD COLUMN "{col.name}" {sqltype}'))
                    except Exception:
                        pass
    except Exception:
        pass

# run migration attempt now
_ensure_info_table_schema()

# small meta DB for chat history and feedback (sqlite)
_user_conn = sqlite3.connect(USER_DB_PATH, check_same_thread=False)

def _init_user_db(conn):
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, ts REAL, role TEXT, content TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, ts REAL, question TEXT, snippet TEXT, label INTEGER)')
    conn.commit()

_init_user_db(_user_conn)

# helpful indexes (best-effort)
with engine.begin() as conn:
    try:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_argo_index_latlon ON argo_index(latitude, longitude)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_argo_index_date ON argo_index(date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_argo_info_juld ON argo_info(juld)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_argo_info_latlon ON argo_info(latitude, longitude)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_argo_info_param ON argo_info(parameter)"))
    except Exception:
        pass

# ---------------------------
# Status helpers
# ---------------------------

def _write_status(status: str, info: Dict[str, Any] = None):
    try:
        payload = {"status": status, "info": info or {}, "ts": datetime.datetime.utcnow().isoformat() + "Z"}
        with open(STATUS_FILE + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(STATUS_FILE + ".tmp", STATUS_FILE)
    except Exception:
        pass


def _read_status():
    try:
        if not os.path.exists(STATUS_FILE):
            return {"status": "idle", "info": {}, "ts": None}
        with open(STATUS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"status": "unknown", "info": {}, "ts": None}

# ---------------------------
# HTTP session with retries
# ---------------------------
from requests.adapters import HTTPAdapter, Retry
SESSION = None

def _requests_session_with_retries(total_retries=3, backoff=0.5, status_forcelist=(429,500,502,503,504)):
    global SESSION
    if SESSION is None:
        s = requests.Session()
        retries = Retry(total=total_retries, backoff_factor=backoff, status_forcelist=status_forcelist, raise_on_status=False)
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))
        SESSION = s
    return SESSION

# ---------------------------
# Geocoding helper (Nominatim)
# ---------------------------

def get_bbox_for_place(place_name: str, user_agent: str = "ARGO-RAG-Explorer/1.0 (+https://example.com)"):
    session = _requests_session_with_retries()
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": user_agent}
    params = {"q": place_name, "format": "json", "limit": 1, "polygon_geojson": 0}
    try:
        r = session.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, list) and len(j) > 0:
                bb = j[0].get("boundingbox")
                if bb and len(bb) == 4:
                    lat_min = float(bb[0]); lat_max = float(bb[1])
                    lon_min = float(bb[2]); lon_max = float(bb[3])
                    return {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max, "source": "nominatim"}
    except Exception:
        pass
    if "arabian" in (place_name or "").lower():
        return {"lat_min": -1.0, "lat_max": 30.0, "lon_min": 32.0, "lon_max": 78.0, "source": "fallback"}
    return None

# ---------------------------
# Model helpers (LLM + Embeddings)
# ---------------------------

def ensure_models():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    if ChatGoogleGenerativeAI is None or GoogleGenerativeAIEmbeddings is None:
        raise RuntimeError("langchain-google-genai is not installed")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    return llm, emb

# ---------------------------
# Index file helpers (IFREMER index)
# ---------------------------

def ensure_index_file(local_path=INDEX_LOCAL_PATH, remote_url=INDEX_REMOTE_URL, timeout=60) -> str:
    if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
        return local_path
    session = _requests_session_with_retries()
    r = session.get(remote_url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(local_path, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
    return local_path


def parse_index_file(path=INDEX_LOCAL_PATH) -> pd.DataFrame:
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


def ingest_index_to_sqlite(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    df.to_sql("argo_index", con=engine, if_exists="replace", index=False)
    return len(df)

# ---------------------------
# small helpers
# ---------------------------

def _decode_if_bytes(x):
    try:
        if isinstance(x, (bytes, bytearray, np.bytes_)):
            return x.decode("utf-8", "ignore")
        return x
    except Exception:
        return x


def _scalar_or_list_to_python(v):
    if v is None:
        return None
    # convert xarray numpy scalars and bytes
    if isinstance(v, (np.generic,)):
        try:
            return v.item()
        except Exception:
            pass
    # arrays -> list
    try:
        if hasattr(v, 'tolist'):
            return v.tolist()
    except Exception:
        pass
    return v


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    v2 = _scalar_or_list_to_python(v)
    if isinstance(v2, list):
        out = []
        for x in v2:
            x = _decode_if_bytes(x)
            out.append("" if x is None else str(x))
        return ",".join(out)
    v2 = _decode_if_bytes(v2)
    return "" if v2 is None else str(v2)

# ---------------------------
# netCDF helpers & parser for argo_info rows
# ---------------------------

def get_local_netcdf_path_from_indexfile(index_file: str) -> str:
    return os.path.join(STORAGE_ROOT, index_file)


def download_netcdf_for_index_path(index_file_path: str, dest_root: str = STORAGE_ROOT, timeout=60) -> str:
    url = f"{IFREMER_BASE}/{index_file_path}"
    local_dir = os.path.join(dest_root, os.path.dirname(index_file_path))
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(index_file_path))
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path
    session = _requests_session_with_retries()
    r = session.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, dir=local_dir) as tmp:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.flush()
    os.replace(tmp.name, local_path)
    return local_path


def _to_float_array(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(x)
    except Exception:
        try:
            return np.asarray(pd.to_numeric(pd.Series(x).ravel(), errors="coerce"), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    flat = arr.ravel()
    try:
        conv = pd.to_numeric(flat, errors="coerce").astype(float).to_numpy()
    except Exception:
        out = []
        for v in flat:
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
        conv = np.asarray(out, dtype=float)
    return conv


def _maybe_get(ds, names):
    """Try to find a variable/attribute in xarray Dataset by list of candidate names."""
    if ds is None:
        return None
    # check attributes
    for n in names:
        if n in ds.attrs:
            return ds.attrs.get(n)
    # check vars
    for n in names:
        if n in ds.variables:
            try:
                v = ds[n].values
                if np.size(v) == 1:
                    return v.tolist()
                return v
            except Exception:
                continue
    return None


def parse_profile_netcdf_to_info_rows(nc_path: str) -> List[Dict[str, Any]]:
    """Parse .nc file and return list of dict rows matching argo_info table schema.
    This function attempts to map global attributes and per-profile variables to the schema.
    Each measurement (depth/sample) produces one row with temp/psal/pres fields filled when present.
    """
    try:
        ds = xr.open_dataset(nc_path, decode_times=True, mask_and_scale=True, decode_cf=True)
    except Exception as e:
        print(f"parse_profile_netcdf_to_info_rows: open failed {nc_path}: {e}")
        return []

    def gget(cands):
        v = _maybe_get(ds, cands)
        return v

    fname = os.path.basename(nc_path)
    float_id = fname.replace('.nc','')

    # Gather many global attrs (best-effort)
    common = {
        'file': os.path.relpath(nc_path, STORAGE_ROOT),
        'data_type': _safe_str(gget(['DATA_TYPE','data_type','dataType'])),
        'format_version': _safe_str(gget(['FORMAT_VERSION','format_version'])),
        'handbook_version': _safe_str(gget(['HANDBOOK_VERSION','handbook_version'])),
        'reference_date_time': _safe_str(gget(['REFERENCE_DATE_TIME','reference_date_time'])),
        'date_creation': _safe_str(gget(['DATE_CREATION','date_creation'])),
        'date_update': _safe_str(gget(['DATE_UPDATE','date_update'])),
        'platform_number': _safe_str(gget(['PLATFORM_NUMBER','platform_number','platform'])),
        'project_name': _safe_str(gget(['PROJECT_NAME','project_name'])),
        'pi_name': _safe_str(gget(['PI_NAME','pi_name'])),
        'station_parameters': _safe_str(gget(['STATION_PARAMETERS','station_parameters'])),
        'cycle_number': _safe_str(gget(['CYCLE_NUMBER','cycle_number'])),
        'direction': _safe_str(gget(['DIRECTION','direction'])),
        'data_centre': _safe_str(gget(['DATA_CENTRE','data_centre'])),
        'dc_reference': _safe_str(gget(['DC_REFERENCE','dc_reference'])),
        'data_state_indicator': _safe_str(gget(['DATA_STATE_INDICATOR','data_state_indicator'])),
        'data_mode': _safe_str(gget(['DATA_MODE','data_mode'])),
        'platform_type': _safe_str(gget(['PLATFORM_TYPE','platform_type'])),
        'float_serial_no': _safe_str(gget(['FLOAT_SERIAL_NO','float_serial_no'])),
        'firmware_version': _safe_str(gget(['FIRMWARE_VERSION','firmware_version'])),
        'wmo_inst_type': _safe_str(gget(['WMO_INST_TYPE','wmo_inst_type'])),
        'positioning_system': _safe_str(gget(['POSITIONING_SYSTEM','positioning_system'])),
        'vertical_sampling_scheme': _safe_str(gget(['VERTICAL_SAMPLING_SCHEME','vertical_sampling_scheme'])),
        'config_mission_number': _safe_str(gget(['CONFIG_MISSION_NUMBER','config_mission_number'])),
    }

    # find common variable names
    lat_v = None
    lon_v = None
    time_v = None
    pres_v = None
    for cand in ['LATITUDE','latitude','lat','LAT']:
        if cand in ds.variables:
            lat_v = ds[cand]
            break
    for cand in ['LONGITUDE','longitude','lon','LON']:
        if cand in ds.variables:
            lon_v = ds[cand]
            break
    for cand in ['JULD','time','TIME','juld','date']:
        if cand in ds.variables:
            time_v = ds[cand]
            break
    for cand in ['PRES','pres','pressure','DEPTH','depth']:
        if cand in ds.variables:
            pres_v = ds[cand]
            break

    # candidate measurement vars
    candidate_vars = {}
    for name in ds.data_vars:
        nlow = name.lower()
        if any(k in nlow for k in ['temp','psal','doxy','o2','chla','chlorophyll','nitrate','nitr','salin']):
            candidate_vars[name] = ds[name]
    if not candidate_vars:
        for name in ds.data_vars:
            if ds[name].ndim <= 2:
                candidate_vars[name] = ds[name]
    if not candidate_vars:
        for name in ds.data_vars:
            try:
                flat = _to_float_array(ds[name].values)
                if np.isfinite(flat).any():
                    candidate_vars[name] = ds[name]
            except Exception:
                continue

    rows = []
    prof_dim = None
    for d in ds.dims:
        dl = d.lower()
        if 'prof' in dl or 'trajectory' in dl or 'n_prof' in dl or 'nprof' in dl:
            prof_dim = d
            break

    try:
        if prof_dim is not None:
            nprof = int(ds.dims[prof_dim])
            for p in range(nprof):
                try:
                    latv = float(_to_float_array(lat_v.isel({prof_dim:p}).values).mean()) if lat_v is not None else None
                except Exception:
                    latv = None
                try:
                    lonv = float(_to_float_array(lon_v.isel({prof_dim:p}).values).mean()) if lon_v is not None else None
                except Exception:
                    lonv = None
                try:
                    traw = time_v.isel({prof_dim:p}).values if time_v is not None else None
                    tval = pd.to_datetime(str(traw)) if traw is not None else None
                except Exception:
                    tval = None

                depths = None
                if pres_v is not None:
                    try:
                        depths = _to_float_array(pres_v.isel({prof_dim:p}).values) if prof_dim in pres_v.dims else _to_float_array(pres_v.values)
                    except Exception:
                        depths = None

                for vname, var in candidate_vars.items():
                    try:
                        if prof_dim in var.dims:
                            vals = _to_float_array(var.isel({prof_dim:p}).values)
                        else:
                            if var.ndim == 2:
                                dims = list(var.dims)
                                try:
                                    cand = _to_float_array(var.isel({dims[0]: p}).values)
                                    vals = cand if np.isfinite(cand).any() else _to_float_array(var.isel({dims[1]: p}).values)
                                except Exception:
                                    vals = _to_float_array(var.values)
                            else:
                                vals = _to_float_array(var.values)
                    except Exception:
                        continue

                    if depths is None or len(depths) == 0:
                        for i, val in enumerate(np.atleast_1d(vals)):
                            if not np.isfinite(val): continue
                            row = common.copy()
                            row.update({
                                'juld': pd.to_datetime(tval).to_pydatetime() if tval is not None else None,
                                'latitude': latv, 'longitude': lonv,
                                'pres': float(i), 'parameter': vname, 'temp': None, 'psal': None, 'value': float(val)
                            })
                            # map temps/psal if variable name indicates
                            if any(k in vname.lower() for k in ['temp','theta']):
                                row['temp'] = float(val)
                            if any(k in vname.lower() for k in ['psal','salin']):
                                row['psal'] = float(val)
                            rows.append(row)
                    else:
                        for dpt, val in zip(np.atleast_1d(depths), np.atleast_1d(vals)):
                            if not np.isfinite(val): continue
                            dv = float(dpt) if np.isfinite(dpt) else None
                            row = common.copy()
                            row.update({
                                'juld': pd.to_datetime(tval).to_pydatetime() if tval is not None else None,
                                'latitude': latv, 'longitude': lonv,
                                'pres': dv, 'parameter': vname, 'temp': None, 'psal': None, 'value': float(val)
                            })
                            if any(k in vname.lower() for k in ['temp','theta']):
                                row['temp'] = float(val)
                            if any(k in vname.lower() for k in ['psal','salin']):
                                row['psal'] = float(val)
                            rows.append(row)
        else:
            try:
                latv = float(_to_float_array(lat_v.values).mean()) if lat_v is not None else None
            except Exception:
                latv = None
            try:
                lonv = float(_to_float_array(lon_v.values).mean()) if lon_v is not None else None
            except Exception:
                lonv = None
            try:
                tval = pd.to_datetime(str(time_v.values)) if time_v is not None else None
            except Exception:
                tval = None

            depths = None
            if pres_v is not None:
                try:
                    depths = _to_float_array(pres_v.values)
                except Exception:
                    depths = None

            for vname, var in candidate_vars.items():
                try:
                    vals = _to_float_array(var.values)
                except Exception:
                    continue
                if depths is None or len(depths) == 0:
                    for i, val in enumerate(np.atleast_1d(vals)):
                        if not np.isfinite(val): continue
                        row = common.copy()
                        row.update({
                            'juld': pd.to_datetime(tval).to_pydatetime() if tval is not None else None,
                            'latitude': latv, 'longitude': lonv,
                            'pres': float(i), 'parameter': vname, 'temp': None, 'psal': None, 'value': float(val)
                        })
                        if any(k in vname.lower() for k in ['temp','theta']):
                            row['temp'] = float(val)
                        if any(k in vname.lower() for k in ['psal','salin']):
                            row['psal'] = float(val)
                        rows.append(row)
                else:
                    for dpt, val in zip(np.atleast_1d(depths), np.atleast_1d(vals)):
                        if not np.isfinite(val): continue
                        dv = float(dpt) if np.isfinite(dpt) else None
                        row = common.copy()
                        row.update({
                            'juld': pd.to_datetime(tval).to_pydatetime() if tval is not None else None,
                            'latitude': latv, 'longitude': lonv,
                            'pres': dv, 'parameter': vname, 'temp': None, 'psal': None, 'value': float(val)
                        })
                        if any(k in vname.lower() for k in ['temp','theta']):
                            row['temp'] = float(val)
                        if any(k in vname.lower() for k in ['psal','salin']):
                            row['psal'] = float(val)
                        rows.append(row)
    finally:
        try:
            ds.close()
        except Exception:
            pass

    return rows


def read_netcdf_variables_to_df(nc_path: str, prof_index: int = 0) -> pd.DataFrame:
    try:
        ds = xr.open_dataset(nc_path, decode_times=True, mask_and_scale=True, decode_cf=True)
    except Exception as e:
        print(f"read_netcdf_variables_to_df: failed to open {nc_path}: {e}")
        return pd.DataFrame()

    def find_first(subs):
        for name in ds.data_vars:
            nl = name.lower()
            if any(s in nl for s in subs):
                return name
        return None

    depth_name = find_first(['pres','depth','pressure'])
    temp_name = find_first(['temp','theta'])
    psal_name = find_first(['psal','salin','sal'])

    def extract_1d(name):
        if not name or name not in ds.variables:
            return np.array([], dtype=float)
        var = ds[name]
        try:
            dims = list(var.dims)
            if len(dims) == 0:
                return _to_float_array(var.values)
            if len(dims) == 1:
                return _to_float_array(var.values)
            if len(dims) == 2:
                prof_dim = None
                for d in dims:
                    if 'prof' in d.lower() or 'trajectory' in d.lower():
                        prof_dim = d; break
                if prof_dim is not None:
                    return _to_float_array(var.isel({prof_dim: prof_index}).values)
                try:
                    cand = _to_float_array(var.isel({dims[0]: prof_index}).values)
                    if np.isfinite(cand).any(): return cand
                except Exception: pass
                try:
                    cand2 = _to_float_array(var.isel({dims[1]: prof_index}).values)
                    if np.isfinite(cand2).any(): return cand2
                except Exception: pass
                return _to_float_array(var.values).ravel()
            return _to_float_array(var.values).ravel()
        except Exception:
            return _to_float_array(var.values).ravel()

    depth = extract_1d(depth_name)
    temp = extract_1d(temp_name)
    psal = extract_1d(psal_name)

    n = 0
    for arr in (depth, temp, psal):
        if getattr(arr, "size", 0) > n:
            n = int(arr.size)
    if n == 0:
        ds.close()
        return pd.DataFrame()

    rows = []
    for i in range(n):
        d = depth[i] if i < len(depth) else np.nan
        t = temp[i] if i < len(temp) else np.nan
        s = psal[i] if i < len(psal) else np.nan
        rows.append({"depth": (float(d) if np.isfinite(d) else None),
                     "temp": (float(t) if np.isfinite(t) else None),
                     "psal": (float(s) if np.isfinite(s) else None)})
    ds.close()
    df = pd.DataFrame(rows)
    df = df.dropna(how='all', subset=["depth","temp","psal"]).reset_index(drop=True)
    return df

# ---------------------------
# ingest info rows into DB
# ---------------------------

def ingest_info_rows(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    inserted = 0
    with engine.begin() as conn:
        batch = []
        for r in rows:
            # ensure datetime types
            if isinstance(r.get("juld"), (pd.Timestamp, np.datetime64)):
                try:
                    r["juld"] = pd.to_datetime(r["juld"]).to_pydatetime()
                except Exception:
                    r["juld"] = None
            # coerce scalars and decode any bytes
            for k,v in list(r.items()):
                if isinstance(v, (bytes, bytearray, np.bytes_)):
                    try:
                        r[k] = v.decode('utf-8', 'ignore')
                    except Exception:
                        r[k] = str(v)
            batch.append(r)
            if len(batch) >= 500:
                conn.execute(argo_info_table.insert(), batch)
                inserted += len(batch)
                batch = []
        if batch:
            conn.execute(argo_info_table.insert(), batch)
            inserted += len(batch)
    return inserted

# ---------------------------
# SQL builder + parsing helpers
# ---------------------------
ALLOWED_VAR_SUBSTR = ["temp", "psal", "doxy", "o2", "chla", "oxygen", "salin"]

OCEAN_NAME_TO_CODE = {
    'indian': 'I', 'india': 'I', 'arabian': 'I',
    'pacific': 'P', 'pac': 'P',
    'atlantic': 'A', 'atl': 'A',
    'southern': 'S', 'antarctic': 'S', 'antarctica': 'S'
}


def normalize_ocean_token(tok: str) -> Optional[str]:
    if not tok:
        return None
    t = tok.strip().lower()
    if t in ('i','p','a','s'):
        return t.upper()
    for k,v in OCEAN_NAME_TO_CODE.items():
        if t.startswith(k):
            return v
    return None


def safe_sql_builder(filters: Dict[str, Any], target: str = "index") -> Tuple[str, Dict[str, Any]]:
    where = []
    params = {}
    filters = filters or {}
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    lat_min = filters.get("lat_min"); lat_max = filters.get("lat_max")
    lon_min = filters.get("lon_min"); lon_max = filters.get("lon_max")

    if target == "index":
        if lat_min is not None and (v := to_float(lat_min)) is not None:
            where.append("latitude >= :lat_min"); params["lat_min"] = v
        if lat_max is not None and (v := to_float(lat_max)) is not None:
            where.append("latitude <= :lat_max"); params["lat_max"] = v
        if lon_min is not None and (v := to_float(lon_min)) is not None:
            where.append("longitude >= :lon_min"); params["lon_min"] = v
        if lon_max is not None and (v := to_float(lon_max)) is not None:
            where.append("longitude <= :lon_max"); params["lon_max"] = v
    else:
        if lat_min is not None and (v := to_float(lat_min)) is not None:
            where.append("argo_info.latitude >= :lat_min"); params["lat_min"] = v
        if lat_max is not None and (v := to_float(lat_max)) is not None:
            where.append("argo_info.latitude <= :lat_max"); params["lat_max"] = v
        if lon_min is not None and (v := to_float(lon_min)) is not None:
            where.append("argo_info.longitude >= :lon_min"); params["lon_min"] = v
        if lon_max is not None and (v := to_float(lon_max)) is not None:
            where.append("argo_info.longitude <= :lon_max"); params["lon_max"] = v

    def parse_ts(x):
        if not x:
            return None
        try:
            dt = pd.to_datetime(x)
        except Exception:
            return None
        try:
            py = dt.to_pydatetime()
        except Exception:
            return dt.isoformat()
        if getattr(py, "tzinfo", None) is not None:
            py = py.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return py
    t0 = parse_ts(filters.get("time_start"))
    t1 = parse_ts(filters.get("time_end"))
    if target == "index":
        if t0 is not None:
            where.append("date >= :t0"); params["t0"] = t0.strftime("%Y%m%d%H%M%S") if isinstance(t0, datetime.datetime) else str(t0)
        if t1 is not None:
            where.append("date <= :t1"); params["t1"] = t1.strftime("%Y%m%d%H%M%S") if isinstance(t1, datetime.datetime) else str(t1)
        ocean_raw = filters.get("ocean")
        if ocean_raw:
            ocean_list = []
            if isinstance(ocean_raw, (list, tuple)):
                candidates = list(ocean_raw)
            else:
                candidates = re.split(r"[,;]|\band\b", str(ocean_raw))
            for c in candidates:
                code = normalize_ocean_token(c)
                if code:
                    ocean_list.append(code)
            ocean_list = list(dict.fromkeys([x for x in ocean_list if x]))
            if len(ocean_list) == 1:
                where.append("ocean = :ocean"); params["ocean"] = ocean_list[0]
            elif len(ocean_list) > 1:
                placeholders = []
                for i,code in enumerate(ocean_list):
                    key = f"ocean_{i}"
                    placeholders.append(f":{key}")
                    params[key] = code
                where.append(f"ocean IN ({', '.join(placeholders)})")
        institution = (filters.get("institution") or "").strip()
        if institution:
            where.append("upper(institution) = upper(:institution)"); params["institution"] = institution
    else:
        if t0 is not None:
            where.append("argo_info.juld >= :t0_dt"); params["t0_dt"] = t0
        if t1 is not None:
            where.append("argo_info.juld <= :t1_dt"); params["t1_dt"] = t1
        var = (filters.get("variable") or "").lower().strip()
        if var and any(s in var for s in ALLOWED_VAR_SUBSTR):
            where.append("lower(argo_info.parameter) LIKE :var"); params["var"] = f"%{var}%"
        institution = (filters.get("institution") or "").strip()
        if institution:
            where.append("upper(argo_info.data_centre) = upper(:institution)"); params["institution"] = institution

    limit = filters.get("limit") or 500
    try:
        limit = int(limit)
        if limit <= 0 or limit > 10000:
            limit = 500
    except Exception:
        limit = 500
    where_clause = " AND ".join(where) if where else "1=1"
    if target == "index":
        sql = f"SELECT * FROM argo_index WHERE {where_clause} ORDER BY date DESC LIMIT {limit}"
    else:
        # query argo_info
        sql = f"SELECT * FROM argo_info WHERE {where_clause} ORDER BY juld DESC LIMIT {limit}"
    return sql, params

# ---------------------------
# fallback parser
# ---------------------------

def _simple_parse_question(question: str) -> Dict[str, Any]:
    q = (question or "").lower()
    parsed = {"action": "answer", "filters": {}}
    if any(k in q for k in ["list float", "list floats", "floats in", "show floats", "find floats"]):
        parsed["action"] = "index"
        oceans = []
        names = re.findall(r"\b(indian|pacific|atlantic|southern|antarctic|arabian)\b", q, flags=re.I)
        for n in names:
            code = normalize_ocean_token(n)
            if code:
                oceans.append(code)
        if not oceans and 'ocean' in q:
            letters = re.findall(r"\b([IPAS])\b", q)
            for L in letters:
                code = normalize_ocean_token(L)
                if code:
                    oceans.append(code)
        if oceans:
            parsed["filters"]["ocean"] = list(dict.fromkeys(oceans))
        return parsed
    for var in ALLOWED_VAR_SUBSTR + ["salinity", "temperature"]:
        if var in q:
            parsed["action"] = "measurements"
            parsed["filters"]["variable"] = var
            break
    latm = re.search(r"([0-9]{1,2}(?:\.[0-9]+)?)\s*[°º]?\s*([nNsS])", question)
    lonm = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*[°º]?\s*([eEwW])", question)
    if latm:
        lat = float(latm.group(1))
        if latm.group(2).lower() == "s":
            lat = -lat
        parsed["filters"]["lat_min"] = lat - 0.5
        parsed["filters"]["lat_max"] = lat + 0.5
    if lonm:
        lon = float(lonm.group(1))
        if lonm.group(2).lower() == "w":
            lon = -lon
        parsed["filters"]["lon_min"] = lon - 0.5
        parsed["filters"]["lon_max"] = lon + 0.5
    if parsed["action"] == "answer":
        parsed["action"] = "index"
    return parsed

# ---------------------------
# LLM-to-structured helper
# ---------------------------

def llm_to_structured(llm, question: str) -> Dict[str, Any]:
    if llm is None:
        return _simple_parse_question(question)
    prompt = (
        "You are a strict parser. Convert the user's question into JSON: "
        "{action: 'index'|'measurements'|'answer', "
        "filters:{lat_min,lat_max,lon_min,lon_max,time_start,time_end,ocean,variable,institution,limit}}. "
        "Respond ONLY with JSON. Use ISO8601 timestamps or null. "
        "User question: " + question
    )
    try:
        r = llm.invoke(prompt)
        out = getattr(r, "content", str(r))
        m = re.search(r"\{[\s\S]*\}", out)
        if m:
            try:
                parsed = json.loads(m.group(0))
                f = parsed.get('filters', {}) or {}
                ocean_raw = f.get('ocean')
                if ocean_raw:
                    if isinstance(ocean_raw, str):
                        candidates = re.split(r"[,;]|\band\b", ocean_raw)
                    elif isinstance(ocean_raw, (list,tuple)):
                        candidates = list(ocean_raw)
                    else:
                        candidates = [str(ocean_raw)]
                    ocean_codes = []
                    for c in candidates:
                        code = normalize_ocean_token(str(c))
                        if code:
                            ocean_codes.append(code)
                    if ocean_codes:
                        parsed['filters']['ocean'] = list(dict.fromkeys(ocean_codes))
                return parsed
            except Exception:
                return _simple_parse_question(question)
    except Exception:
        pass
    return _simple_parse_question(question)

# ---------------------------
# Haversine & nearest helper
# ---------------------------

def haversine_np(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.minimum(1, np.sqrt(a)))
    R = 6371.0
    return R * c


def nearest_floats(lat0: float, lon0: float, limit: int = 10, max_candidates: int = 10000) -> pd.DataFrame:
    with engine.connect() as conn:
        res = conn.execute(text("SELECT file, latitude, longitude, date, ocean, profiler_type, institution FROM argo_index WHERE latitude IS NOT NULL AND longitude IS NOT NULL LIMIT :L"), {"L": max_candidates})
        df = pd.DataFrame(res.fetchall(), columns=res.keys())
    if df.empty:
        return df
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
    df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')
    df = df.dropna(subset=["latitude", "longitude"])
    if df.empty:
        return df
    df["distance_km"] = haversine_np(lat0, lon0, df["latitude"].values, df["longitude"].values)
    df = df.sort_values("distance_km").reset_index(drop=True)
    return df.head(limit)


def get_measurements_for_float(float_id: str, variable_hint: Optional[str] = None) -> pd.DataFrame:
    with engine.connect() as conn:
        if variable_hint:
            res = conn.execute(text("SELECT juld, latitude, longitude, pres as depth, parameter, temp, psal FROM argo_info WHERE file LIKE :pat AND lower(parameter) LIKE :v ORDER BY juld ASC"), {"pat": f"%{float_id}.nc", "v": f"%{variable_hint.lower()}%"})
        else:
            res = conn.execute(text("SELECT juld, latitude, longitude, pres as depth, parameter, temp, psal FROM argo_info WHERE file LIKE :pat ORDER BY juld ASC"), {"pat": f"%{float_id}.nc"})
        df = pd.DataFrame(res.fetchall(), columns=res.keys())
    if df.empty:
        return df
    df["juld"] = pd.to_datetime(df["juld"])
    return df


def find_common_vars_for_floats(float_ids: List[str]) -> Dict[str, str]:
    var_counts = {}
    with engine.connect() as conn:
        for fid in float_ids:
            pattern = f"%{fid}.nc"
            res = conn.execute(text("SELECT DISTINCT parameter FROM argo_info WHERE file LIKE :p"), {"p": pattern})
            for (v,) in res.fetchall():
                if not v: continue
                nlow = v.lower()
                var_counts.setdefault(nlow, set()).add(fid)
    mapping = {}
    for name in var_counts.keys():
        if any(k in name for k in ["temp", "theta"]):
            mapping.setdefault("temp", name)
        if any(k in name for k in ["psal", "salin"]):
            mapping.setdefault("psal", name)
    return mapping

# ---------------------------
# Retrieval-Augmented Generation helpers (MCP)
# ---------------------------

def assemble_mcp_context(index_rows: Optional[pd.DataFrame], nc_previews: Dict[str, pd.DataFrame], chroma_client=None, question: str = "", emb_model=None, top_k: int = 5) -> Dict[str, Any]:
    parts = []
    chunks = []
    if index_rows is not None and not index_rows.empty:
        sample = index_rows.head(10).copy()
        sample['file'] = sample['file'].astype(str)
        idx_text = f"Index matches ({len(index_rows)} rows). Sample:\n"
        for _, r in sample.iterrows():
            idx_text += f"- {os.path.basename(r['file'])}: lat={r.get('latitude')}, lon={r.get('longitude')}, date={r.get('date')}, ocean={r.get('ocean')}, inst={r.get('institution')}\n"
        parts.append(idx_text)
        chunks.append({'type':'index_sample','text': idx_text})

    if nc_previews:
        for fid, info in list(nc_previews.items())[:top_k]:
            dfp = info.get('preview')
            if isinstance(dfp, pd.DataFrame) and not dfp.empty:
                s = dfp.head(10).to_csv(index=False)
                txt = f"Preview {fid}:\n{s}\n"
                parts.append(txt)
                chunks.append({'type':'nc_preview','id':fid,'text': s})

    if chroma_client is not None and emb_model is not None:
        try:
            q_emb = emb_model.embed_query(question) if hasattr(emb_model, 'embed_query') else emb_model.embed_documents([question])[0]
            coll_name = f"floats_user_0"
            coll = chroma_client.get_collection(name=coll_name)
            results = coll.query(query_embeddings=[q_emb], n_results=top_k)
            docs = results.get('documents') if isinstance(results, dict) else None
            metas = results.get('metadatas') if isinstance(results, dict) else None
            if metas:
                for m in metas[0][:top_k]:
                    txt = f"Vector hit: file={m.get('file')}, lat={m.get('mean_lat')}, lon={m.get('mean_lon')}, date={m.get('date')}\n"
                    parts.append(txt)
                    chunks.append({'type':'vector_hit','meta': m,'text':txt})
        except Exception:
            pass

    context_text = "\n\n".join(parts)
    return {"context_text": context_text, "chunks": chunks}


def rag_answer_with_mcp(llm, emb_model, question: str, index_rows: Optional[pd.DataFrame], nc_previews: Dict[str, pd.DataFrame], chroma_client=None) -> Dict[str, Any]:
    mcp = assemble_mcp_context(index_rows, nc_previews, chroma_client=chroma_client, question=question, emb_model=emb_model)
    system = (
        "You are an expert assistant for ARGO oceanographic data. Use the provided context to answer the user's question. "
        "Return a JSON object only, with keys: answer (string), sql (optional string of recommended SQL to run), references (array of short text references)."
    )
    prompt = system + "\nCONTEXT:\n" + mcp['context_text'] + "\nUSER QUESTION:\n" + question + "\nRespond ONLY with a single JSON object."
    try:
        r = llm.invoke(prompt)
        out = getattr(r, 'content', str(r))
        m = re.search(r"\{[\s\S]*\}", out)
        if m:
            parsed = json.loads(m.group(0))
            parsed['_mcp_chunks'] = mcp['chunks']
            return parsed
    except Exception:
        pass
    return {"answer": "(LLM failed to produce a structured response)", "sql": None, "references": [], "_mcp_chunks": mcp['chunks']}

# ---------------------------
# ask_argo_question: uses .nc previews first for temp-like queries then falls back to DB.
# ---------------------------

def ask_argo_question(llm, emb_model, question: str, user_id: int = 0, chroma_client=None) -> Dict[str, Any]:
    out = {"action": None, "parsed": None, "sql_index": None, "index_rows": None,
           "sql_measurements": None, "measurement_rows": None, "nc_previews": {}, "explanation": None, 'rag_answer': None}
    if llm is not None:
        parsed = llm_to_structured(llm, question)
    else:
        parsed = _simple_parse_question(question)
    out["parsed"] = parsed

    try:
        have_latlon = any(k in parsed.get("filters", {}) for k in ("lat_min","lat_max","lon_min","lon_max"))
        if not have_latlon and question and question.strip():
            bbox = get_bbox_for_place(question.strip())
            if bbox:
                filters = parsed.get("filters", {}) or {}
                filters.update({
                    "lat_min": bbox["lat_min"], "lat_max": bbox["lat_max"],
                    "lon_min": bbox["lon_min"], "lon_max": bbox["lon_max"],
                })
                parsed["filters"] = filters
                out["parsed"] = parsed
    except Exception:
        pass

    action = parsed.get("action", "answer")
    filters = parsed.get("filters", {}) or {}
    out["action"] = action

    try:
        sql_index, params_index = safe_sql_builder(filters, target="index")
        out["sql_index"] = sql_index
        with engine.connect() as conn:
            res = conn.execute(text(sql_index), params_index)
            df_index = pd.DataFrame(res.fetchall(), columns=res.keys())
        out["index_rows"] = df_index
    except Exception as e:
        out["explanation"] = f"Index query failed: {e}"
        return out

    var_val = (filters.get("variable") or "").lower()
    wants_nc_preferred = False
    if action == "measurements" and var_val:
        if any(s in var_val for s in ALLOWED_VAR_SUBSTR):
            wants_nc_preferred = True

    nc_previews = {}
    if wants_nc_preferred and out["index_rows"] is not None and not out["index_rows"].empty:
        for _, row in out["index_rows"].iterrows():
            fpath = row.get("file")
            if not fpath:
                continue
            local = get_local_netcdf_path_from_indexfile(fpath)
            if not os.path.exists(local):
                try:
                    local = download_netcdf_for_index_path(fpath)
                except Exception:
                    local = None
            if local and os.path.exists(local):
                try:
                    df_nc = read_netcdf_variables_to_df(local, prof_index=0)
                    if df_nc is not None and not df_nc.empty:
                        fid = os.path.basename(fpath).replace('.nc','')
                        nc_previews[fid] = {"file": fpath, "preview": df_nc}
                except Exception:
                    pass
    out["nc_previews"] = nc_previews

    if wants_nc_preferred and nc_previews:
        combined = []
        for fid, info in nc_previews.items():
            dfp = info["preview"].copy()
            dfp["float_id"] = fid
            dfp["file"] = info["file"]
            combined.append(dfp)
        if combined:
            df_combined = pd.concat(combined, ignore_index=True, sort=False)
            out["explanation"] = f"Found {len(nc_previews)} .nc previews (used as primary source for requested variable)."
            out["measurement_rows"] = df_combined
            out["sql_measurements"] = None
            if llm is not None:
                out['rag_answer'] = rag_answer_with_mcp(llm, emb_model, question, out['index_rows'], out['nc_previews'], chroma_client=chroma_client)
            return out

    try:
        sql_meas, params_meas = safe_sql_builder(filters, target="measurements")
        out["sql_measurements"] = sql_meas
        with engine.connect() as conn:
            res = conn.execute(text(sql_meas), params_meas)
            df_meas = pd.DataFrame(res.fetchall(), columns=res.keys())
        out["measurement_rows"] = df_meas
        if df_meas.empty:
            out["explanation"] = "No measurement rows matched your query (and no suitable .nc previews were found)."
        else:
            out["explanation"] = f"Found {len(df_meas)} measurement rows from DB."
            if llm is not None:
                out['rag_answer'] = rag_answer_with_mcp(llm, emb_model, question, out['index_rows'], out['nc_previews'], chroma_client=chroma_client)
    except Exception as e:
        out["explanation"] = f"Measurement query failed: {e}"
    return out

# ---------------------------
# Background workers
# ---------------------------

def _chroma_build_worker(limit_rows, user_id=0):
    _write_status('building_chroma', {'limit_rows': int(limit_rows)})
    try:
        llm, emb = ensure_models()
        build_chroma_from_index(emb, user_id=int(user_id), limit_rows=int(limit_rows))
        _write_status('chroma_done', {'limit_rows': int(limit_rows)})
    except Exception as e:
        _write_status('chroma_error', {'error': str(e)})


def start_chroma_build_async(limit_rows, user_id=0):
    p = multiprocessing.Process(target=_chroma_build_worker, args=(int(limit_rows), int(user_id)), daemon=True)
    p.start()
    _write_status('chroma_started', {'pid': p.pid, 'limit_rows': int(limit_rows)})
    return p.pid


def _index_ingest_worker():
    _write_status('ingesting_index', {})
    try:
        local = ensure_index_file()
        df = parse_index_file(local)
        n = ingest_index_to_sqlite(df)
        _write_status('index_done', {'rows': int(n)})
    except Exception as e:
        _write_status('index_error', {'error': str(e)})


def start_index_ingest_async():
    p = multiprocessing.Process(target=_index_ingest_worker, daemon=True)
    p.start()
    _write_status('index_started', {'pid': p.pid})
    return p.pid

# End of PART 1
# PART 2: ARGO RAG Explorer - STREAMLIT UI
# Paste this below PART 1 in a single app.py (or keep as a separate module and import functions from PART 1).

import os
import re
import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional folium for maps (kept optional)
try:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    folium = None
    MarkerCluster = None
    st_folium = None
    FOLIUM_AVAILABLE = False

st.set_page_config(page_title="ARGO RAG Explorer", layout="wide")
st.title("ARGO RAG Explorer — Nearest floats, RAG Chat & Profile comparison")

st.sidebar.header("Config")
st.sidebar.write(f"Storage: `{STORAGE_ROOT}`")
st.sidebar.write(f"DB: `{DB_ENGINE_URL}`")

if st.sidebar.button("Ensure index downloaded & ingested (async)", key="ensure_index_btn"):
    try:
        pid = start_index_ingest_async()
        st.sidebar.success(f"Index ingest started (pid {pid}). Refresh status below to see progress.")
    except Exception as e:
        st.sidebar.error(f"Failed to start index ingest: {e}")

st.sidebar.markdown("**Chroma / Vector options**")
rows_to_index = int(st.sidebar.number_input("Rows to index (Chroma)", min_value=10, max_value=500000, value=1000, step=10))
if st.sidebar.button("(Re)build Chroma float index (first N rows) (async)", key="rebuild_chroma"):
    if ChatGoogleGenerativeAI is None or GoogleGenerativeAIEmbeddings is None:
        st.sidebar.warning("Embeddings are required to build Chroma. Set GEMINI_API_KEY and install embedding package.")
    elif chromadb is None:
        st.sidebar.warning("chromadb not installed.")
    else:
        try:
            pid = start_chroma_build_async(rows_to_index, user_id=0)
            st.sidebar.success(f"Chroma build started (pid {pid}). Refresh status below to see progress.")
        except Exception as e:
            st.sidebar.error(f"Chroma build failed to start: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Background job status")
status = _read_status()
try:
    st.sidebar.json(status)
except Exception:
    st.sidebar.code(json.dumps(status, indent=2))

chat_map_toggle = st.sidebar.checkbox("Show maps in Chat?", value=False, key="chat_show_maps")

tabs = st.tabs(["Nearest ARGO floats", "Explore Index", "Ingest Profiles", "Chat (RAG)", "Trajectories & Profile comparison", "Exports"]) 

# Small session state defaults
if "last_index_df" not in st.session_state:
    st.session_state["last_index_df"] = pd.DataFrame()

# --- Nearest tab ---
with tabs[0]:
    st.header("Nearest ARGO floats to a coordinate")
    c1, c2, c3 = st.columns([3,2,1])
    lat0 = c1.number_input("Latitude", value=15.0, format="%f", key="nearest_lat")
    lon0 = c2.number_input("Longitude", value=72.0, format="%f", key="nearest_lon")
    nlimit = c3.number_input("Limit", min_value=1, max_value=200, value=10, key="nearest_limit")

    if "nearest_query" not in st.session_state:
        st.session_state["nearest_query"] = None
    if "nearest_df" not in st.session_state:
        st.session_state["nearest_df"] = pd.DataFrame()

    if st.button("Find nearest floats", key="nearest_find"):
        try:
            df_near = nearest_floats(lat0, lon0, limit=int(nlimit))
            st.session_state["nearest_df"] = df_near
            st.session_state["nearest_query"] = {"lat": float(lat0), "lon": float(lon0)}
        except Exception as e:
            st.error(f"Nearest lookup failed: {e}")

    df_near = st.session_state.get("nearest_df", pd.DataFrame())
    nq = st.session_state.get("nearest_query")
    if not df_near.empty:
        st.write(df_near[["file","latitude","longitude","date","distance_km","institution"]].head(50))
        df_map = df_near.dropna(subset=["latitude", "longitude"])[:2000]
        if not df_map.empty:
            try:
                center_lat = float(nq.get('lat', lat0)) if nq else float(lat0)
                center_lon = float(nq.get('lon', lon0)) if nq else float(lon0)
                df_map = df_map.copy()
                df_map['ocean_cat'] = df_map['ocean'].fillna('Unknown')
                df_map['map_size'] = df_map['distance_km'].replace(0, 0.1)
                df_map['map_size'] = (df_map['map_size'].max() / df_map['map_size']).clip(1, 20)

                fig = px.scatter_mapbox(
                    df_map,
                    lat='latitude', lon='longitude',
                    hover_name='file',
                    hover_data={'date':True, 'distance_km':True, 'institution':True},
                    color='ocean_cat',
                    size='map_size',
                    size_max=16,
                    zoom=3,
                    center={'lat': center_lat, 'lon': center_lon},
                    height=520
                )
                fig.update_layout(mapbox_style='open-street-map', margin={'r':0,'t':0,'l':0,'b':0}, legend=dict(title='Ocean'))
                fig.add_trace(px.scatter_mapbox(pd.DataFrame([{'lat': center_lat, 'lon': center_lon, 'label':'Query Center'}]), lat='lat', lon='lon').data[0])
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Map rendering failed.")
    else:
        st.info("No nearest-floats result yet. Press 'Find nearest floats' after entering coordinates.")

# --- Explore Index ---
with tabs[1]:
    st.header("Explore ARGO index (SQLite/Postgres)")
    try:
        idx_count = 0
        inspector = inspect(engine)
        if inspector.has_table("argo_index"):
            with engine.connect() as conn:
                idx_count = int(conn.execute(text("SELECT COUNT(*) FROM argo_index")).scalar() or 0)
    except Exception:
        idx_count = 0
    try:
        with engine.connect() as conn:
            info_count = conn.execute(text("SELECT COUNT(*) FROM argo_info")).scalar()
            info_count = int(info_count) if info_count is not None else 0
    except Exception:
        info_count = 0
    st.caption(f"Index rows available: {idx_count} — Ingested info rows: {info_count}")

    with st.expander("Filters"):
        with st.form(key="index_filter_form"):
            c1, c2, c3 = st.columns(3)
            lat_min = c1.number_input("lat_min", value=-90.0, format="%f")
            lat_max = c1.number_input("lat_max", value=90.0, format="%f")
            lon_min = c2.number_input("lon_min", value=-180.0, format="%f")
            lon_max = c2.number_input("lon_max", value=180.0, format="%f")
            ocean = c3.text_input("ocean (A/P/I/S)", value="")
            institution = c3.text_input("institution (e.g. AO, IN)", value="")
            date_from = c1.text_input("date_from (YYYYMMDDHHMMSS)", value="")
            date_to = c2.text_input("date_to (YYYYMMDDHHMMSS)", value="")
            limit = c3.number_input("limit", min_value=1, max_value=5000, value=500)
            run = st.form_submit_button("Run index query")

        if run:
            filters = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max,
                       "ocean": ocean.strip() or None, "institution": institution.strip() or None,
                       "time_start": date_from or None, "time_end": date_to or None,
                       "limit": limit}
            sql, params = safe_sql_builder(filters, target="index")
            try:
                with engine.connect() as conn:
                    res = conn.execute(text(sql), params)
                    df = pd.DataFrame(res.fetchall(), columns=res.keys())
            except Exception as e:
                st.error(f"Query failed: {e}")
                df = pd.DataFrame()
            st.write(f"Returned rows: {len(df)}")
            if not df.empty:
                st.session_state["last_index_df"] = df
                st.session_state["last_index_query"] = {"filters": filters}
                st.dataframe(df.head(200))
                df_map = df.dropna(subset=["latitude","longitude"])[:2000]
                if not df_map.empty:
                    try:
                        center = {'lat': float(df_map['latitude'].mean()), 'lon': float(df_map['longitude'].mean())}
                        df_map = df_map.copy()
                        df_map['ocean_cat'] = df_map['ocean'].fillna('Unknown')
                        fig = px.scatter_mapbox(
                            df_map,
                            lat='latitude', lon='longitude', hover_name='file',
                            hover_data={'date':True, 'institution':True}, color='ocean_cat',
                            size_max=12, zoom=1, center=center, height=520
                        )
                        fig.update_layout(mapbox_style='open-street-map', margin={'r':0,'t':0,'l':0,'b':0}, legend=dict(title='Ocean'))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.write("Map rendering failed.")

# --- Ingest Profiles ---
with tabs[2]:
    st.header("Bulk ingest by manual index paths")
    st.markdown("Paste index file paths (one per line), e.g. `aoml/13857/profiles/R13857_001.nc`")
    paths_text = st.text_area("Index file paths", height=160)
    max_files = st.number_input("Max files to download", min_value=1, max_value=200, value=10, key="bulk_max_files")
    if st.button("Download & ingest paths", key="bulk_ingest"):
        lines = [l.strip() for l in paths_text.splitlines() if l.strip()]
        lines = lines[:max_files]
        total = 0
        for p in lines:
            try:
                local_nc = download_netcdf_for_index_path(p)
                rows = parse_profile_netcdf_to_info_rows(local_nc)
                n = ingest_info_rows(rows)
                total += n
                st.success(f"Ingested {n} rows from {p}")
            except Exception as e:
                st.error(f"Failed {p}: {e}")
        st.info(f"Total info rows ingested: {total}")

# --- Chat (RAG) ---
with tabs[3]:
    st.header("Chat with ARGO data (RAG + MCP)")
    have_models = True
    llm = None
    emb = None
    chroma_client = None
    try:
        llm, emb = ensure_models()
    except Exception:
        have_models = False
        st.info("LLM/Embeddings not available; using fallback parser.")

    if chromadb is not None:
        try:
            chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=os.path.join(CHROMA_DIR, "user_0")))
        except Exception:
            try:
                chroma_client = chromadb.Client()
            except Exception:
                chroma_client = None

    mode = st.radio("Mode", ["Auto (LLM if available)", "LLM (force)", "Fallback (rule-based)"], index=0)
    question = st.text_area("Ask (e.g., 'salinity near the equator in March 2023', 'list floats in Indian Ocean')", height=140)

    st.markdown("---")
    st.subheader("Place lookup (optional)")
    pcol1, pcol2, pcol3 = st.columns([3,1,1])
    place_name = pcol1.text_input("Place name (e.g., 'Arabian Sea')", value="Arabian Sea", key="chat_place_name")
    place_year = pcol2.text_input("Year (optional, YYYY)", value="", key="chat_place_year")
    place_limit = pcol3.number_input("Limit", min_value=1, max_value=5000, value=200, key="chat_place_limit")
    place_institution = st.text_input("Institution (optional, e.g., AO)", key="chat_place_institution")

    if st.button("Find place lat/lon and nearby ARGO floats", key="chat_place_find"):
        if not place_name.strip():
            st.warning("Type a place name.")
        else:
            with st.spinner("Resolving place and searching index..."):
                bbox = get_bbox_for_place(place_name.strip())
                if bbox is None:
                    st.error("Could not resolve place name and no fallback available.")
                else:
                    lat_c = (bbox["lat_min"] + bbox["lat_max"]) / 2.0
                    lon_c = (bbox["lon_min"] + bbox["lon_max"]) / 2.0
                    st.success(f"Place resolved (source: {bbox.get('source','unknown')}). Center: {lat_c:.4f}, {lon_c:.4f}")
                    st.info(f"Bounding box: lat [{bbox['lat_min']}, {bbox['lat_max']}], lon [{bbox['lon_min']}, {bbox['lon_max']}]")

                    filters = {"lat_min": bbox["lat_min"], "lat_max": bbox["lat_max"], "lon_min": bbox["lon_min"], "lon_max": bbox["lon_max"], "institution": place_institution.strip() or None, "limit": int(place_limit)}
                    if place_year and re.match(r"^\d{4}$", place_year.strip()):
                        y = int(place_year.strip())
                        filters["time_start"] = f"{y:04d}0101000000"
                        filters["time_end"] = f"{y:04d}1231235959"

                    sql, params = safe_sql_builder(filters, target="index")
                    try:
                        with engine.connect() as conn:
                            res = conn.execute(text(sql), params)
                            df = pd.DataFrame(res.fetchall(), columns=res.keys())
                    except Exception as e:
                        st.error(f"Query failed: {e}")
                        df = pd.DataFrame()

                    st.session_state["chat_place_df"] = df
                    st.session_state["chat_place_center"] = (lat_c, lon_c)

    chat_df = st.session_state.get("chat_place_df", pd.DataFrame())
    chat_center = st.session_state.get("chat_place_center")
    st.write(f"Returned rows: {len(chat_df)}")
    if not chat_df.empty:
        st.dataframe(chat_df.head(200))
        if chat_map_toggle:
            df_map = chat_df.dropna(subset=["latitude","longitude"])[:2000]
            if not df_map.empty:
                try:
                    center = {'lat': float(df_map['latitude'].mean()), 'lon': float(df_map['longitude'].mean())}
                    df_map = df_map.copy()
                    df_map['ocean_cat'] = df_map['ocean'].fillna('Unknown')
                    fig = px.scatter_mapbox(df_map, lat='latitude', lon='longitude', hover_name='file', hover_data={'date':True, 'institution':True}, color='ocean_cat', zoom=2, center=center, height=520)
                    fig.update_layout(mapbox_style='open-street-map', margin={'r':0,'t':0,'l':0,'b':0}, legend=dict(title='Ocean'))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.write("Map render failed.")

    if st.button("Ask", key="ask_btn"):
        if mode == "LLM (force)" and not have_models:
            st.error("LLM not available.")
        else:
            use_llm = (mode == "Auto (LLM if available)" and have_models) or (mode == "LLM (force)" and have_models)
            llm_obj = llm if use_llm else None
            emb_obj = emb if have_models else None
            if not question.strip():
                st.warning("Type a question.")
            else:
                with st.spinner("Processing..."):
                    out = ask_argo_question(llm_obj, emb_obj, question, user_id=0, chroma_client=chroma_client)
                st.subheader("Answer / Explanation")
                st.write(out.get("explanation", ""))

                if out.get('rag_answer'):
                    st.markdown("**LLM-generated answer (RAG)**")
                    try:
                        st.json(out['rag_answer'])
                    except Exception:
                        st.write(out['rag_answer'])

                st.markdown("**Parsed filters (debug)**")
                try:
                    st.json(out.get("parsed", {}))
                except Exception:
                    st.write(out.get("parsed", {}))

                st.markdown("### Index rows (matched floats/files)")
                idx = out.get("index_rows")
                if isinstance(idx, pd.DataFrame) and not idx.empty:
                    st.dataframe(idx.head(200))
                else:
                    st.write("No index rows matched.")

                st.markdown("### Measurement rows used (DB or .nc previews)")
                meas = out.get("measurement_rows")
                if isinstance(meas, pd.DataFrame) and not meas.empty:
                    st.dataframe(meas.head(200))
                    st.download_button("Download results as CSV", meas.to_csv(index=False).encode("utf-8"), file_name="argo_query_results.csv")
                else:
                    st.write("No measurement rows returned.")

                st.markdown("### .nc previews (if used or available)")
                nc_previews = out.get("nc_previews", {}) or {}
                if nc_previews:
                    for fid, info in nc_previews.items():
                        st.markdown(f"**{fid}** — `{info.get('file')}`")
                        dfp = info.get("preview")
                        if isinstance(dfp, pd.DataFrame) and not dfp.empty:
                            st.dataframe(dfp.head(200))
                            st.download_button(f"Download {fid} .nc preview CSV", dfp.to_csv(index=False).encode("utf-8"), file_name=f"{fid}_preview.csv")
                        else:
                            st.write("Preview empty.")

# --- Trajectories & Profile comparison ---
with tabs[4]:
    st.header("Profile & Index metadata comparison (trajectories from argo_info)")

    # candidate floats from last index query or fallback to index table
    candidate_floats = []
    if not st.session_state.get("last_index_df", pd.DataFrame()).empty:
        candidate_floats = [os.path.basename(x).replace('.nc','') for x in st.session_state["last_index_df"].get('file',[]).dropna().unique().tolist()]
    if not candidate_floats:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT DISTINCT file FROM argo_index WHERE file IS NOT NULL LIMIT 500"))
            rows = res.fetchall()
            candidate_floats = [os.path.basename(r[0]).replace('.nc','') for r in rows if r and r[0]]

    comp_sel = st.multiselect("Select floats for comparison", options=candidate_floats, max_selections=3, key="comp_sel")

    if not comp_sel:
        st.info("Pick up to 3 floats to compare. Trajectories will be plotted from the ingested `argo_info` table (per-profile lat/lon).")
    else:
        # --- index metadata for selected floats ---
        metadata = []
        with engine.connect() as conn:
            for fid in comp_sel:
                pattern = f"%{fid}.nc"
                res = conn.execute(text(
                    "SELECT file, date, date_update, latitude, longitude, ocean, profiler_type, institution "
                    "FROM argo_index WHERE file LIKE :p LIMIT 1"), {"p": pattern})
                row = res.fetchone()
                if row:
                    metadata.append({
                        "float_id": fid,
                        "file": row[0],
                        "date": pd.to_datetime(row[1]) if row[1] is not None else None,
                        "date_update": pd.to_datetime(row[2]) if row[2] is not None else None,
                        "latitude": row[3],
                        "longitude": row[4],
                        "ocean": row[5],
                        "profiler_type": row[6],
                        "institution": row[7]
                    })
                else:
                    metadata.append({"float_id": fid, "file": None, "date": None, "date_update": None,
                                     "latitude": None, "longitude": None, "ocean": None,
                                     "profiler_type": None, "institution": None})

        meta_df = pd.DataFrame(metadata).set_index("float_id")
        st.subheader("Index metadata for selected floats")
        st.table(meta_df.reset_index())

        # --- positions & trajectories from argo_info ---
        pos_frames = []
        with engine.connect() as conn:
            for fid in comp_sel:
                pattern = f"%{fid}.nc"
                try:
                    res = conn.execute(text(
                        "SELECT file, juld, latitude, longitude, pres FROM argo_info "
                        "WHERE file LIKE :p AND latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY juld ASC"), {"p": pattern})
                    dfp = pd.DataFrame(res.fetchall(), columns=res.keys())
                    if not dfp.empty:
                        dfp['float_id'] = fid
                        pos_frames.append(dfp)
                except Exception as e:
                    st.write(f"Failed fetching positions for {fid}: {e}")

        if pos_frames:
            df_positions = pd.concat(pos_frames, ignore_index=True, sort=False)
            df_positions['juld'] = pd.to_datetime(df_positions['juld'], errors='coerce')
            df_positions['latitude'] = pd.to_numeric(df_positions['latitude'], errors='coerce')
            df_positions['longitude'] = pd.to_numeric(df_positions['longitude'], errors='coerce')
            df_positions['pres'] = pd.to_numeric(df_positions.get('pres', pd.NA), errors='coerce')
            df_positions = df_positions.dropna(subset=['latitude','longitude']).copy()

            if df_positions.empty:
                st.warning("`argo_info` contains no positions for the selected floats (or coordinates are null).")
            else:
                center_lat = float(df_positions['latitude'].mean())
                center_lon = float(df_positions['longitude'].mean())

                try:
                    fig_traj = px.line_mapbox(
                        df_positions.sort_values(['float_id','juld']),
                        lat='latitude', lon='longitude',
                        color='float_id',
                        hover_name='file' if 'file' in df_positions.columns else None,
                        hover_data={k: True for k in ['juld','pres'] if k in df_positions.columns},
                        zoom=3,
                        center={'lat': center_lat, 'lon': center_lon},
                        height=420,
                    )
                    scatter = px.scatter_mapbox(
                        df_positions,
                        lat='latitude', lon='longitude',
                        color='float_id',
                        hover_name='file' if 'file' in df_positions.columns else None,
                        hover_data={k: True for k in ['juld','pres'] if k in df_positions.columns},
                        zoom=3,
                    )
                    for t in scatter.data:
                        fig_traj.add_trace(t)
                    fig_traj.update_layout(mapbox_style='open-street-map', margin={'r':0,'t':0,'l':0,'b':0}, legend=dict(title='Float ID'))
                    st.subheader("Trajectories (from argo_info positions)")
                    st.plotly_chart(fig_traj, use_container_width=True, key=f"traj_{uuid.uuid4().hex}")
                except Exception:
                    st.write("Map plotting failed for trajectories (falling back to table view).")
                    st.dataframe(df_positions.head(200))

                st.markdown("### Sample positions (from argo_info)")
                st.dataframe(df_positions[['float_id','file','juld','latitude','longitude','pres']].sort_values(['float_id','juld']).head(500) if 'file' in df_positions.columns else df_positions[['float_id','juld','latitude','longitude','pres']].sort_values(['float_id','juld']).head(500))
        else:
            st.info("No per-profile positions found in `argo_info` for selected floats.")

        # --- raw .nc previews (unchanged) ---
        st.subheader("Raw values (from .nc) — depth / temp / psal")
        for fid in comp_sel:
            file_path = meta_df.loc[fid, "file"] if fid in meta_df.index else None
            if not file_path:
                st.info(f"{fid}: no index file found for this float.")
                continue
            with st.expander(f"{fid} — {file_path}"):
                local_path = get_local_netcdf_path_from_indexfile(file_path) if file_path else None
                if not local_path or not os.path.exists(local_path):
                    try:
                        if file_path:
                            local_path = download_netcdf_for_index_path(file_path)
                            st.success(f"Downloaded {file_path} to {local_path}")
                        else:
                            st.info("No index file path to download.")
                    except Exception as e:
                        st.error(f"Failed to download {file_path}: {e}")
                        continue
                else:
                    st.write(f"Using local file: {local_path}")
                df_nc = read_netcdf_variables_to_df(local_path, prof_index=0)
                if df_nc is None or df_nc.empty:
                    st.info("No numeric depth/temp/psal values found or parsing returned empty table.")
                else:
                    st.download_button(f"Download {fid} .nc variables as CSV", df_nc.to_csv(index=False).encode("utf-8"), file_name=f"{fid}_vars.csv")
                    st.dataframe(df_nc.head(200))

        # --- ingested rows comparison & plotting (explicit pres & psal handling) ---
        with engine.connect() as conn:
            info_cnt = conn.execute(text("SELECT COUNT(*) FROM argo_info")).scalar()
            info_cnt = int(info_cnt) if info_cnt is not None else 0
        if info_cnt == 0:
            st.warning("No ingested info rows in DB — raw .nc values above (if any).")
        else:
            profs = {fid: get_measurements_for_float(fid) for fid in comp_sel}
            varmap = find_common_vars_for_floats(comp_sel)
            st.write("Detected variables summary (lowercased names):")
            st.json(list(varmap.values()) or [])

            import plotly.graph_objects as go
            import plotly.express as px
            import uuid

            # -----------------------
            # Improved profile plots (robust hover_data handling)
            # -----------------------

            # build cleaned per-float dfs and then a combined df for display
            cleaned_frames = []
            for fid, dfp in profs.items():
                if dfp is None or dfp.empty:
                    st.write(f"Float {fid}: no measurement rows returned by get_measurements_for_float().")
                    continue

                # normalize names
                if 'pres' not in dfp.columns and 'depth' in dfp.columns:
                    dfp['pres'] = dfp['depth']
                if 'psal' not in dfp.columns and 'psa' in dfp.columns:
                    dfp['psal'] = dfp['psa']
                dfp['pres'] = pd.to_numeric(dfp.get('pres', pd.NA), errors='coerce')
                dfp['temp'] = pd.to_numeric(dfp.get('temp', pd.NA), errors='coerce')
                dfp['psal'] = pd.to_numeric(dfp.get('psal', pd.NA), errors='coerce')
                if 'juld' in dfp.columns:
                    dfp['juld'] = pd.to_datetime(dfp['juld'], errors='coerce')

                dfp['float_id'] = fid
                cleaned_frames.append(dfp)

            if cleaned_frames:
                combined_df = pd.concat(cleaned_frames, ignore_index=True, sort=False)
            else:
                combined_df = pd.DataFrame()

            # ensure backward compatibility: define fig_sal if older code references it
            fig_sal = None

            # helper to choose hover columns only if present
            def pick_hover_cols(df, want):
                return [c for c in want if c in df.columns]

            # --- Temperature vs Depth (modern, clear) ---
            # prefer explicit 'depth' column for y-axis, fall back to 'pres' if needed
            depth_col = 'depth' if 'depth' in combined_df.columns else ('pres' if 'pres' in combined_df.columns else None)

            if depth_col and not combined_df.empty and combined_df[['temp', depth_col]].dropna().shape[0] > 0:
                tmp_plot_df = combined_df.dropna(subset=['temp', depth_col]).copy().sort_values(['float_id', depth_col])
                hover_cols = pick_hover_cols(tmp_plot_df, ['file', 'juld', 'parameter', 'float_id'])
                fig_temp = px.line(
                    tmp_plot_df,
                    x='temp', y=depth_col,
                    color='float_id',
                    line_group='float_id',
                    markers=True,
                    hover_data=hover_cols,
                    title=f'Temperature vs {depth_col.capitalize()} (profile)',
                    labels={'temp': 'Temperature (units in data)', depth_col: f'{depth_col.capitalize()} (instrument units)'}
                )
                fig_temp.update_traces(mode='lines+markers', marker=dict(size=6))
                fig_temp.update_layout(template='plotly_white', legend_title_text='Float ID', height=520, margin=dict(t=50))
                fig_temp.update_yaxes(autorange='reversed')
            else:
                fig_temp = None

            # --- Pressure vs Depth (modern) ---
            # plot pres (x) against depth (y) when both columns exist and are meaningful
            if 'pres' in combined_df.columns and depth_col and not combined_df[['pres', depth_col]].dropna().empty:
                pres_plot_df = combined_df.dropna(subset=['pres', depth_col]).copy().sort_values(['float_id', depth_col])
                hover_cols = pick_hover_cols(pres_plot_df, ['file', 'juld', 'parameter', 'float_id'])
                fig_pres_depth = px.line(
                    pres_plot_df,
                    x='pres', y=depth_col,
                    color='float_id',
                    line_group='float_id',
                    markers=True,
                    hover_data=hover_cols,
                    title=f'Pressure vs {depth_col.capitalize()} (profile)',
                    labels={'pres': 'Pressure (instrument units)', depth_col: f'{depth_col.capitalize()} (instrument units)'}
                )
                fig_pres_depth.update_traces(mode='lines+markers', marker=dict(size=6))
                fig_pres_depth.update_layout(template='plotly_white', legend_title_text='Float ID', height=520, margin=dict(t=50))
                fig_pres_depth.update_yaxes(autorange='reversed')
            else:
                fig_pres_depth = None

            # --- Pressure time-series (juld vs pres) - keep existing style but improved look ---
            fig_pres_time = go.Figure()
            pres_time_traces = 0
            if not combined_df.empty and combined_df[['juld', 'pres']].dropna().shape[0] > 0:
                for fid in combined_df['float_id'].unique():
                    ppp = combined_df.loc[combined_df['float_id'] == fid].dropna(subset=['juld', 'pres']).sort_values('juld')
                    if not ppp.empty:
                        fig_pres_time.add_trace(go.Scatter(x=ppp['juld'], y=ppp['pres'], mode='lines+markers', name=fid))
                        pres_time_traces += 1
                fig_pres_time.update_layout(template='plotly_white', legend_title_text='Float ID', height=520, margin=dict(t=50))
                fig_pres_time.update_yaxes(title_text='Pressure (instrument units)')
                fig_pres_time.update_xaxes(title_text='Time (juld)')
            else:
                fig_pres_time = None

            # --- Display: give Temp vs Depth and Pres vs Depth prominence (two wide cols), and time-series as third ---
            # ---------- Display block (robust, compatibility-safe) ----------
            # Provide compatibility shim so older references to fig_pres / pres_traces won't raise NameError.
            if 'fig_pres_time' in locals():
                fig_pres = fig_pres_time
                pres_traces = pres_time_traces
            else:
                fig_pres = locals().get('fig_pres', None)
                pres_traces = locals().get('pres_traces', 0)

            # Choose layout: Temp (left), Pres-vs-Depth (optional middle), Pres-vs-Time (right)
            cols = st.columns([1.1, 1.1, 1])

            # Left: Temperature vs Depth
            if 'fig_temp' in locals() and fig_temp is not None:
                cols[0].plotly_chart(
                    fig_temp,
                    use_container_width=True,
                    key=f"fig_temp_{'_'.join(comp_sel) if comp_sel else 'none'}"
                )
            else:
                cols[0].info('No temperature-depth profiles available for selected floats.')

            # Middle: Pressure vs Depth (optional)
            if 'fig_pres_depth' in locals() and fig_pres_depth is not None:
                cols[1].plotly_chart(
                    fig_pres_depth,
                    use_container_width=True,
                    key=f"fig_pres_depth_{'_'.join(comp_sel) if comp_sel else 'none'}"
                )
            else:
                cols[1].info('No pressure-depth profiles available for selected floats.')

            # Right: Pressure vs Time
            if fig_pres is not None and pres_traces > 0:
                cols[2].plotly_chart(
                    fig_pres,
                    use_container_width=True,
                    key=f"fig_pres_time_{'_'.join(comp_sel) if comp_sel else 'none'}"
                )
            else:
                cols[2].info('No pressure time-series available for selected floats.')

            # --- Show combined argo_info rows for selected floats ---
            st.subheader("Combined argo_info rows for selected floats")
            if combined_df.empty:
                st.info("No measurement rows available to display from argo_info.")
            else:
                # choose columns to present (you can add/remove columns as needed)
                display_cols = [c for c in combined_df.columns]
                st.write(f"Showing {len(combined_df)} rows from `argo_info` for selected floats.")
                st.dataframe(combined_df[display_cols].sort_values(['float_id', 'juld'] if 'juld' in combined_df.columns else ['float_id']).reset_index(drop=True))
            cols = st.columns(3)
            if fig_temp:
                cols[0].plotly_chart(fig_temp, use_container_width=True)
            else:
                cols[0].info('No temperature-depth profiles available for selected floats.')

            if fig_sal:
                cols[1].plotly_chart(fig_sal, use_container_width=True)
            else:
                cols[1].info('No salinity-depth profiles available for selected floats.')

            if fig_pres and pres_traces > 0:
                cols[2].plotly_chart(fig_pres, use_container_width=True)
            else:
                cols[2].info('No pressure time-series available for selected floats.')

            # --- Show combined argo_info rows for selected floats (full, scrollable) ---
            st.subheader("Combined argo_info rows for selected floats")
            if combined_df.empty:
                st.info("No measurement rows available to display from argo_info.")
            else:
                # choose columns to present (you can add/remove columns as needed)
                display_cols = [c for c in combined_df.columns]
                st.write(f"Showing {len(combined_df)} rows from `argo_info` for selected floats.")
                st.dataframe(combined_df[display_cols].sort_values(['float_id','juld'] if 'juld' in combined_df.columns else ['float_id']).reset_index(drop=True))

# --- Exports ---
with tabs[5]:
    st.header("Exports")
    if st.button("Export ingested info to Parquet", key="export_parquet"):
        path = os.path.join(STORAGE_ROOT, f"argo_info_{int(time.time())}.parquet")
        df = pd.read_sql_query("SELECT * FROM argo_info", engine)
        df.to_parquet(path, index=False)
        with open(path, "rb") as fh:
            st.download_button("Download Parquet", fh.read(), file_name=os.path.basename(path), mime="application/octet-stream")

    if st.button("Export ingested info to NetCDF (simple)", key="export_netcdf"):
        path = os.path.join(STORAGE_ROOT, f"argo_info_{int(time.time())}.nc")
        df = pd.read_sql_query("SELECT * FROM argo_info", engine)
        if df.empty:
            st.warning("No ingested info to export.")
        else:
            ds = xr.Dataset({
                "value": (("row",), df.get("temp", pd.Series([np.nan]*len(df))).astype(float).fillna(np.nan).values)
            }, coords={
                "row": np.arange(len(df)),
                "file": (("row",), df["file"].astype(str).values),
                "juld": (("row",), pd.to_datetime(df["juld"]).astype("datetime64[ns]").values),
                "lat": (("row",), df["latitude"].astype(float).values),
                "lon": (("row",), df["longitude"].astype(float).values),
                "pres": (("row",), df["pres"].astype(float).values),
                "parameter": (("row",), df["parameter"].astype(str).values),
            })
            ds.to_netcdf(path)
            with open(path, "rb") as fh:
                st.download_button("Download NetCDF", fh.read(), file_name=os.path.basename(path), mime="application/octet-stream")

st.sidebar.markdown("---")
st.sidebar.write("Tip: For quick tests, index only a small number of rows (e.g., 100–1000). Large builds can be slow.")
