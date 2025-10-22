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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
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

tabs = st.tabs(["Nearest ARGO floats", "Explore Index", "Ingest Profiles", "Chat (RAG)", "Trajectories & Profile comparison", "Exports","ML: Temperature predictor"]) 

# Small session state defaults
if "last_index_df" not in st.session_state:
    st.session_state["last_index_df"] = pd.DataFrame()

# --- Nearest tab ---
# ---------------- Tab 0: Nearest floats + Place lookup (single OpenStreetMap Plotly map) ----------------
with tabs[0]:
    st.header("Nearest ARGO floats / Place lookup")
    # --- Nearest floats inputs ---
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
            # clear place lookup results when user asks nearest
            st.session_state.pop("chat_place_df", None)
            st.session_state.pop("chat_place_center", None)
        except Exception as e:
            st.error(f"Nearest lookup failed: {e}")

    # small separator
    st.markdown("---")

    # ---------- Place lookup (in-tab; single map) ----------
    st.subheader("Place lookup (optional)")

    # small fast catalog of common seas/regions (keeps your catalog)
    COMMON_SEAS = {
        'arabian sea': {'lat_min': 5.0, 'lat_max': 28.0, 'lon_min': 50.0, 'lon_max': 77.0, 'source': 'catalog'},
        'bay of bengal': {'lat_min': 5.0, 'lat_max': 22.0, 'lon_min': 80.0, 'lon_max': 100.0, 'source': 'catalog'},
        'red sea': {'lat_min': 12.0, 'lat_max': 30.0, 'lon_min': 32.0, 'lon_max': 44.0, 'source': 'catalog'},
        'mediterranean sea': {'lat_min': 30.0, 'lat_max': 46.0, 'lon_min': -6.0, 'lon_max': 36.0, 'source': 'catalog'},
        'south china sea': {'lat_min': 0.0, 'lat_max': 23.0, 'lon_min': 100.0, 'lon_max': 121.0, 'source': 'catalog'},
        'gulf of mexico': {'lat_min': 18.0, 'lat_max': 31.0, 'lon_min': -98.0, 'lon_max': -80.0, 'source': 'catalog'},
        'caribbean sea': {'lat_min': 9.0, 'lat_max': 23.0, 'lon_min': -89.0, 'lon_max': -60.0, 'source': 'catalog'},
        'andaman sea': {'lat_min': 5.0, 'lat_max': 15.0, 'lon_min': 92.0, 'lon_max': 100.0, 'source': 'catalog'},
        'persian gulf': {'lat_min': 23.0, 'lat_max': 30.0, 'lon_min': 48.0, 'lon_max': 57.0, 'source': 'catalog'},
        'gulf of aden': {'lat_min': 10.0, 'lat_max': 16.0, 'lon_min': 42.0, 'lon_max': 52.0, 'source': 'catalog'},
    }

    # UI inputs for place lookup (kept inside tab)
    pcol1, pcol2, pcol3 = st.columns([3,1,1])
    place_name = pcol1.text_input("Place name (e.g., 'Arabian Sea')", value=st.session_state.get("chat_place_name","Arabian Sea"), key="chat_place_name")
    place_year = pcol2.text_input("Year (optional, YYYY)", value=st.session_state.get("chat_place_year",""), key="chat_place_year")
    place_limit = pcol3.number_input("Limit", min_value=1, max_value=5000, value=200, key="chat_place_limit")
    place_institution = st.text_input("Institution (optional, e.g., AO)", value=st.session_state.get("chat_place_institution",""), key="chat_place_institution")

    # Resolve when user clicks
    if st.button("Find place lat/lon and nearby ARGO floats (geocode)", key="chat_place_find"):
        if not place_name.strip():
            st.warning("Type a place name.")
        else:
            with st.spinner("Resolving place and searching index..."):
                chosen_bbox = None
                source = None

                # 1) fast catalog check
                cat = COMMON_SEAS.get(place_name.strip().lower())
                if cat:
                    chosen_bbox = cat
                    source = "catalog"
                else:
                    # 2) check cache then geocode (uses your helper get_place_candidates)
                    cache_key = place_name.strip().lower()
                    candidates = st.session_state["place_lookup_cache"].get(cache_key) if "place_lookup_cache" in st.session_state else None
                    if candidates is None:
                        candidates = get_place_candidates(place_name.strip(), limit=5)
                        st.session_state.setdefault("place_lookup_cache", {})[cache_key] = candidates

                    if not candidates:
                        st.error("No geocoding candidates found.")
                    elif len(candidates) == 1:
                        chosen_bbox = candidates[0]
                        source = "nominatim"
                    else:
                        # present choices for disambiguation (single selectbox displayed inline)
                        sel = st.selectbox("Multiple places found — pick the correct one", options=[c["display_name"] for c in candidates], key="place_candidate_select")
                        for c in candidates:
                            if c["display_name"] == sel:
                                chosen_bbox = c
                                source = "nominatim"
                                break

                if chosen_bbox is not None:
                    lat_c = (chosen_bbox["lat_min"] + chosen_bbox["lat_max"]) / 2.0
                    lon_c = (chosen_bbox["lon_min"] + chosen_bbox["lon_max"]) / 2.0
                    st.success(f"Place resolved (source: {source}). Center: {lat_c:.4f}, {lon_c:.4f}")
                    st.info(f"Bounding box: lat [{chosen_bbox['lat_min']}, {chosen_bbox['lat_max']}], lon [{chosen_bbox['lon_min']}, {chosen_bbox['lon_max']}]")

                    # build filters and query index (same as before)
                    filters = {"lat_min": chosen_bbox["lat_min"], "lat_max": chosen_bbox["lat_max"], "lon_min": chosen_bbox["lon_min"], "lon_max": chosen_bbox["lon_max"], "institution": place_institution.strip() or None, "limit": int(place_limit)}
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
                    # clear nearest when user chooses place lookup so map shows place results
                    st.session_state.pop("nearest_df", None)
                    st.session_state.pop("nearest_query", None)

    # --- Map rendering: choose place results first, else nearest results ---
    chat_df = st.session_state.get("chat_place_df", pd.DataFrame())
    near_df = st.session_state.get("nearest_df", pd.DataFrame())
    # prefer place results if present
    map_source_df = chat_df if (isinstance(chat_df, pd.DataFrame) and not chat_df.empty) else (near_df if (isinstance(near_df, pd.DataFrame) and not near_df.empty) else pd.DataFrame())
    map_center = st.session_state.get("chat_place_center") or st.session_state.get("nearest_query")

    st.write(f"Mapped rows: {len(map_source_df)} (showing place results first, then nearest results)")

    # Sidebar cluster controls (kept in sidebar for global control)
    cluster_km = st.sidebar.slider("Cluster cell size (km)", min_value=5, max_value=200, value=25, step=5, key="cluster_km")
    min_cluster_size = st.sidebar.slider("Min points to show as cluster", min_value=2, max_value=50, value=3, step=1, key="min_cluster_size")

    # Render single OpenStreetMap plotly map with optional clustering
    if not map_source_df.empty:
        try:
            import math, plotly.graph_objects as go
            df_map = map_source_df.dropna(subset=["latitude","longitude"]).copy()
            df_map["latitude"] = pd.to_numeric(df_map["latitude"], errors="coerce")
            df_map["longitude"] = pd.to_numeric(df_map["longitude"], errors="coerce")
            df_map = df_map.dropna(subset=["latitude","longitude"])
            if df_map.empty:
                st.info("No mapped points (latitude/longitude missing or non-numeric).")
            else:
                # map center: prefer explicit center, otherwise mean of points
                if map_center:
                    try:
                        center = {'lat': float(map_center[0]), 'lon': float(map_center[1])}
                    except Exception:
                        center = {'lat': float(df_map['latitude'].mean()), 'lon': float(df_map['longitude'].mean())}
                else:
                    center = {'lat': float(df_map['latitude'].mean()), 'lon': float(df_map['longitude'].mean())}

                # lightweight grid clustering (fast)
                lat_mean = center["lat"]
                dlat = cluster_km / 110.574
                dlon = cluster_km / (111.320 * max(0.0001, math.cos(math.radians(lat_mean))))
                lat0 = df_map["latitude"].min()
                lon0 = df_map["longitude"].min()
                cell_i = ((df_map["latitude"] - lat0) // dlat).astype(int)
                cell_j = ((df_map["longitude"] - lon0) // dlon).astype(int)
                df_map["_cell"] = cell_i.astype(str) + "_" + cell_j.astype(str)

                grouped = df_map.groupby("_cell")
                clusters = []
                singles = []
                for name, grp in grouped:
                    cnt = len(grp)
                    mean_lat = grp["latitude"].mean()
                    mean_lon = grp["longitude"].mean()
                    sample_files = grp["file"].dropna().astype(str).unique().tolist()[:5]
                    oceans = grp["ocean"].fillna("Unknown").unique().tolist()
                    item = {
                        "cell": name,
                        "count": int(cnt),
                        "lat": float(mean_lat),
                        "lon": float(mean_lon),
                        "sample_files": sample_files,
                        "oceans": oceans,
                        "df_idx": grp.index.tolist()
                    }
                    if cnt >= min_cluster_size:
                        clusters.append(item)
                    else:
                        for _, row in grp.iterrows():
                            singles.append({
                                "lat": float(row["latitude"]),
                                "lon": float(row["longitude"]),
                                "file": row.get("file"),
                                "date": row.get("date"),
                                "institution": row.get("institution"),
                                "ocean": row.get("ocean", "Unknown")
                            })

                fig = go.Figure()

                # add single points via px (preserve color mapping)
                if singles:
                    sf = pd.DataFrame(singles)
                    sf["ocean_cat"] = sf["ocean"].fillna("Unknown")
                    px_single = px.scatter_mapbox(
                        sf,
                        lat="lat", lon="lon",
                        hover_name="file",
                        hover_data={"date": True, "institution": True, "ocean": True},
                        color="ocean_cat",
                        zoom=3,
                        center=center,
                        height=560
                    )
                    for tr in px_single.data:
                        fig.add_trace(tr)

                # cluster markers
                if clusters:
                    cl_df = pd.DataFrame(clusters)
                    cl_df["size"] = np.clip(6 + (np.log1p(cl_df["count"]) * 8), 8, 60)
                    cl_df["text"] = cl_df.apply(lambda r: f"Cluster: {r['count']} pts<br/>Oceans: {', '.join(map(str, r['oceans']))}<br/>Sample: {', '.join(r['sample_files'])}", axis=1)
                    fig.add_trace(go.Scattermapbox(
                        lat=cl_df["lat"],
                        lon=cl_df["lon"],
                        mode="markers+text",
                        marker=go.scattermapbox.Marker(size=cl_df["size"], color="rgba(30,150,240,0.85)"),
                        text=cl_df["count"].astype(str),
                        textposition="middle center",
                        hoverinfo="text",
                        hovertext=cl_df["text"],
                        name=f"Clusters (>= {min_cluster_size})"
                    ))

                # draw bbox if available in last_index_query session state
                last_filters = st.session_state.get('last_index_query', {}).get('filters') or {}
                if all(k in last_filters for k in ('lat_min','lat_max','lon_min','lon_max')):
                    lla = float(last_filters['lat_min']); llo = float(last_filters['lon_min']); hra = float(last_filters['lat_max']); hlo = float(last_filters['lon_max'])
                    bbox_lats = [lla, lla, hra, hra, lla]
                    bbox_lons = [llo, hlo, hlo, llo, llo]
                    fig.add_trace(go.Scattermapbox(
                        lat=bbox_lats, lon=bbox_lons,
                        mode="lines",
                        line=go.scattermapbox.Line(width=2, color="yellow"),
                        hoverinfo="none",
                        name="Query bbox"
                    ))

                # approximate zoom
                lat_extent = df_map["latitude"].max() - df_map["latitude"].min()
                lon_extent = df_map["longitude"].max() - df_map["longitude"].min()
                max_extent = max(lat_extent, lon_extent)
                if max_extent <= 0:
                    zoom = 3
                else:
                    zoom = max(1, min(12, int(6 - math.log(max_extent+1) + 2)))

                fig.update_layout(
                    mapbox=dict(style="open-street-map", center=center, zoom=zoom),
                    margin={"r":0,"t":0,"l":0,"b":0},
                    legend=dict(title="Legend"),
                    height=560
                )

                st.plotly_chart(fig, use_container_width=True)

                # small cluster summary expander (not nested inside other expanders)
                with st.expander("Cluster summary and sample points (click to open)"):
                    if clusters:
                        cl_small = pd.DataFrame(clusters)[["cell","count","lat","lon","sample_files","oceans"]].sort_values("count", ascending=False).head(20)
                        st.dataframe(cl_small)
                    if singles:
                        st.subheader("Sample individual points (small groups)")
                        st.dataframe(pd.DataFrame(singles).head(200))

                # CSV download of mapped points
                csv_bytes = df_map.to_csv(index=False).encode("utf-8")
                st.download_button("Download mapped points as CSV", csv_bytes, file_name="argo_place_query_points.csv", mime="text/csv")

        except Exception as e:
            st.write(f"Map render failed: {e}")
    else:
        st.info("No results yet. Use 'Find nearest floats' or 'Find place lat/lon and nearby ARGO floats'.")

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
# --- Chat (RAG) ---
with tabs[3]:
    st.header("Chat with ARGO data (RAG + MCP)")

    # Try to initialize LLM / embeddings adapters (graceful fallback)
    have_models = True
    llm = None
    emb = None
    chroma_client = None
    try:
        llm, emb = ensure_models()
    except Exception:
        have_models = False
        st.info("LLM/Embeddings not available; using fallback parser.")

    # chroma client (optional)
    if chromadb is not None:
        try:
            chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=os.path.join(CHROMA_DIR, "user_0")))
        except Exception:
            try:
                chroma_client = chromadb.Client()
            except Exception:
                chroma_client = None

    # UI controls
    mode = st.radio("Mode", ["Auto (LLM if available)", "LLM (force)", "Fallback (rule-based)"], index=0)
    question = st.text_area("Ask (e.g., 'salinity near the equator in March 2023', 'list floats in Indian Ocean')", height=140)
    st.markdown("---")

    # Optional place lookup / context fields (re-use values from sidebar/tab if set)
    place_hint = st.text_input("Place hint (optional)", value="", key="chat_place_hint")
    place_limit = st.number_input("Place result limit (used if place hint given)", min_value=1, max_value=5000, value=200, key="chat_place_limit_small")
    st.markdown("---")

    # Ask button
    if st.button("Ask", key="ask_btn"):
        # guard for forced-LLM mode
        if mode == "LLM (force)" and not have_models:
            st.error("LLM not available (GEMINI_API_KEY missing or model client not installed).")
        else:
            if not question or not question.strip():
                st.warning("Type a question first.")
            else:
                # Choose whether to use llm/emb based on mode and availability
                use_llm = (mode == "Auto (LLM if available)" and have_models) or (mode == "LLM (force)" and have_models)
                llm_obj = llm if use_llm else None
                emb_obj = emb if have_models else None

                # If user provided a place hint, try to resolve it and add bounding-box filters to initial parsing:
                if place_hint and place_hint.strip():
                    try:
                        bbox = get_bbox_for_place(place_hint.strip())
                        if bbox:
                            # stash a quick place-derived filter in session (ask_argo_question already attempts geocoding internally if needed)
                            st.info(f"Place hint resolved (center {((bbox['lat_min']+bbox['lat_max'])/2):.3f}, {((bbox['lon_min']+bbox['lon_max'])/2):.3f})")
                            # We don't mutate the question; ask_argo_question will attempt to geocode again if it lacks lat/lon
                        else:
                            st.info("Place hint could not be resolved by the geocoder.")
                    except Exception:
                        pass

                # Run the query
                with st.spinner("Processing your question (this may take a few seconds)..."):
                    try:
                        out = ask_argo_question(llm_obj, emb_obj, question, user_id=0, chroma_client=chroma_client)
                    except Exception as e:
                        out = {"explanation": f"ask_argo_question failed: {e}", "parsed": None, "index_rows": None, "measurement_rows": None, "nc_previews": None, "rag_answer": None}
                # Display outputs
                st.subheader("Answer / Explanation")
                st.write(out.get("explanation", ""))

                # LLM RAG answer (if present)
                if out.get("rag_answer"):
                    st.markdown("**LLM-generated answer (RAG)**")
                    try:
                        # prefer pretty JSON view
                        st.json(out["rag_answer"])
                    except Exception:
                        st.write(out["rag_answer"])

                # Parsed filters debug
                st.markdown("**Parsed filters (debug)**")
                try:
                    st.json(out.get("parsed", {}))
                except Exception:
                    st.write(out.get("parsed", {}))

                # Index rows
                st.markdown("### Index rows (matched floats/files)")
                idx = out.get("index_rows")
                if isinstance(idx, pd.DataFrame) and not idx.empty:
                    st.dataframe(idx.head(200))
                    # allow download
                    csv_idx = idx.to_csv(index=False).encode("utf-8")
                    st.download_button("Download index rows CSV", csv_idx, file_name="argo_index_rows.csv", mime="text/csv")
                else:
                    st.write("No index rows matched.")

                # Measurement rows (from DB or .nc previews)
                st.markdown("### Measurement rows used (DB or .nc previews)")
                meas = out.get("measurement_rows")
                if isinstance(meas, pd.DataFrame) and not meas.empty:
                    st.dataframe(meas.head(200))
                    st.download_button("Download measurement rows CSV", meas.to_csv(index=False).encode("utf-8"), file_name="argo_measurements.csv", mime="text/csv")
                else:
                    st.write("No measurement rows returned.")

                # .nc previews (if any)
                st.markdown("### .nc previews (if used or available)")
                nc_previews = out.get("nc_previews", {}) or {}
                if nc_previews:
                    for fid, info in nc_previews.items():
                        st.markdown(f"**{fid}** — `{info.get('file')}`")
                        dfp = info.get("preview")
                        if isinstance(dfp, pd.DataFrame) and not dfp.empty:
                            st.dataframe(dfp.head(200))
                            st.download_button(f"Download {fid} .nc preview CSV", dfp.to_csv(index=False).encode("utf-8"), file_name=f"{fid}_preview.csv", mime="text/csv")
                        else:
                            st.write("Preview empty.")
                else:
                    st.write("No .nc previews available.")

                # Show MCP chunks if rag_answer included them for debugging
                if isinstance(out.get("rag_answer"), dict) and out["rag_answer"].get("_mcp_chunks"):
                    st.markdown("### MCP / Retrieval chunks (debug)")
                    try:
                        st.write(out["rag_answer"].get("_mcp_chunks"))
                    except Exception:
                        st.write("Could not display MCP chunks.")

                # Feedback quick buttons (store minimal feedback in user DB)
                st.markdown("### Feedback")
                fb_col1, fb_col2 = st.columns([1,1])
                if fb_col1.button("👍 Useful", key="fb_useful"):
                    try:
                        _user_conn.execute("INSERT INTO feedback (ts, question, snippet, label) VALUES (?, ?, ?, ?)",
                                           (time.time(), question[:400], (str(out.get("explanation"))[:400] if out.get("explanation") else ""), 1))
                        _user_conn.commit()
                        st.success("Thanks for the feedback.")
                    except Exception:
                        st.warning("Feedback not saved.")
                if fb_col2.button("👎 Not useful", key="fb_notuseful"):
                    try:
                        _user_conn.execute("INSERT INTO feedback (ts, question, snippet, label) VALUES (?, ?, ?, ?)",
                                           (time.time(), question[:400], (str(out.get("explanation"))[:400] if out.get("explanation") else ""), 0))
                        _user_conn.commit()
                        st.success("Thanks, noted.")
                    except Exception:
                        st.warning("Feedback not saved.")

# --- Trajectories & Profile comparison ---
# --- Trajectories & Profile comparison (improved) ---
with tabs[4]:
    st.header("Profile & Index metadata comparison (trajectories from argo_info)")

    # local imports (safe even if already imported elsewhere)
    import uuid
    import os
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sqlalchemy import text
    import datetime

    # Gather candidate floats from previous index query or DB fallback
    candidate_floats = []
    last_index_df = st.session_state.get("last_index_df", pd.DataFrame())
    if isinstance(last_index_df, pd.DataFrame) and not last_index_df.empty:
        candidate_floats = [os.path.basename(x).replace('.nc', '') for x in last_index_df.get('file', []).dropna().unique().tolist()]

    if not candidate_floats:
        try:
            with engine.connect() as conn:
                res = conn.execute(text("SELECT DISTINCT file FROM argo_index WHERE file IS NOT NULL LIMIT 500"))
                rows = res.fetchall()
                candidate_floats = [os.path.basename(r[0]).replace('.nc', '') for r in rows if r and r[0]]
        except Exception as e:
            st.warning(f"Could not fetch candidate floats from DB: {e}")
            candidate_floats = []

    comp_sel = st.multiselect("Select floats for comparison (max 3)", options=candidate_floats, max_selections=3, key="comp_sel")

    if not comp_sel:
        st.info("Pick up to 3 floats to compare. Trajectories will be plotted from the ingested `argo_info` table (per-profile lat/lon).")
    else:
        # --- Index metadata for selected floats ---
        metadata = []
        try:
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
        except Exception as e:
            st.error(f"Error fetching index metadata: {e}")

        meta_df = pd.DataFrame(metadata).set_index("float_id")
        st.subheader("Index metadata for selected floats")
        st.table(meta_df.reset_index())

        # --- Positions & trajectories from argo_info ---
        pos_frames = []
        try:
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
                        st.warning(f"Failed fetching positions for {fid}: {e}")
        except Exception as e:
            st.warning(f"DB error while fetching positions: {e}")

        if pos_frames:
            df_positions = pd.concat(pos_frames, ignore_index=True, sort=False)
            # clean types
            df_positions['juld'] = pd.to_datetime(df_positions['juld'], errors='coerce')
            df_positions['latitude'] = pd.to_numeric(df_positions['latitude'], errors='coerce')
            df_positions['longitude'] = pd.to_numeric(df_positions['longitude'], errors='coerce')
            df_positions['pres'] = pd.to_numeric(df_positions.get('pres', pd.NA), errors='coerce')
            df_positions = df_positions.dropna(subset=['latitude', 'longitude']).copy()

            if df_positions.empty:
                st.warning("`argo_info` contains no positions for the selected floats (or coordinates are null).")
            else:
                center_lat = float(df_positions['latitude'].mean())
                center_lon = float(df_positions['longitude'].mean())

                st.subheader("Trajectories (from argo_info positions)")
                try:
                    fig_traj = go.Figure()
                    palette = px.colors.qualitative.Dark24
                    floats_unique = sorted(df_positions['float_id'].unique())
                    for i, fid in enumerate(floats_unique):
                        df_f = df_positions[df_positions['float_id'] == fid].sort_values('juld')
                        if df_f.empty:
                            continue
                        color = palette[i % len(palette)]
                        # line (track)
                        fig_traj.add_trace(go.Scattermapbox(
                            lat=df_f['latitude'], lon=df_f['longitude'], mode='lines',
                            line=dict(width=2.0, color=color), name=f"{fid} (track)",
                            hoverinfo='text',
                            text=df_f.apply(lambda r: f"{fid}<br>juld: {r['juld']}<br>pres: {r.get('pres')}", axis=1)
                        ))
                        # points
                        fig_traj.add_trace(go.Scattermapbox(
                            lat=df_f['latitude'], lon=df_f['longitude'], mode='markers',
                            marker=dict(size=7, color=color), name=f"{fid} (pts)",
                            hoverinfo='text',
                            text=df_f.apply(lambda r: f"{fid}<br>juld: {r['juld']}<br>pres: {r.get('pres')}", axis=1)
                        ))

                    fig_traj.update_layout(
                        mapbox=dict(style='open-street-map', center={'lat': center_lat, 'lon': center_lon}, zoom=3),
                        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                        height=480,
                        legend=dict(orientation='v', x=1.02, y=0.95)
                    )
                    st.plotly_chart(fig_traj, use_container_width=True)
                except Exception:
                    st.write("Map plotting failed for trajectories (falling back to table view).")
                    st.dataframe(df_positions.head(200))

                st.markdown("### Sample positions (from argo_info)")
                pos_cols = ['float_id', 'file', 'juld', 'latitude', 'longitude', 'pres'] if 'file' in df_positions.columns else ['float_id', 'juld', 'latitude', 'longitude', 'pres']
                st.dataframe(df_positions[pos_cols].sort_values(['float_id', 'juld']).head(500))
        else:
            st.info("No per-profile positions found in `argo_info` for selected floats.")

        # --- Raw .nc previews per selected float ---
        st.subheader("Raw values (from .nc) — depth / temp / psal")
        for fid in comp_sel:
            file_path = meta_df.loc[fid, "file"] if fid in meta_df.index else None
            if not file_path:
                st.info(f"{fid}: no index file found for this float.")
                continue
            with st.expander(f"{fid} — {file_path}"):
                local_path = None
                try:
                    local_path = get_local_netcdf_path_from_indexfile(file_path) if file_path else None
                except Exception:
                    local_path = None
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

                try:
                    df_nc = read_netcdf_variables_to_df(local_path, prof_index=0)
                except Exception as e:
                    df_nc = pd.DataFrame()
                    st.warning(f"Preview parse failed: {e}")

                if df_nc is None or df_nc.empty:
                    st.info("No numeric depth/temp/psal values found or parsing returned empty table.")
                else:
                    st.download_button(f"Download {fid} .nc variables as CSV", df_nc.to_csv(index=False).encode("utf-8"), file_name=f"{fid}_vars.csv")
                    st.dataframe(df_nc.head(200))

        # --- Ingested rows comparison & plotting ---
        try:
            with engine.connect() as conn:
                info_cnt = conn.execute(text("SELECT COUNT(*) FROM argo_info")).scalar()
                info_cnt = int(info_cnt) if info_cnt is not None else 0
        except Exception as e:
            st.error(f"DB error checking argo_info rows: {e}")
            info_cnt = 0

        if info_cnt == 0:
            st.warning("No ingested info rows in DB — raw .nc values above (if any).")
        else:
            profs = {fid: get_measurements_for_float(fid) for fid in comp_sel}
            varmap = find_common_vars_for_floats(comp_sel)
            st.write("Detected variables summary (lowercased names):")
            st.json(list(varmap.values()) or [])

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

                # numeric conversions & types
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

            # Prepare plotting df and filter sentinel values
            plot_df = combined_df.copy()
            if 'temp' in plot_df.columns:
                plot_df['temp'] = pd.to_numeric(plot_df['temp'], errors='coerce')
                plot_df = plot_df[plot_df['temp'] != 1].copy()
            if 'juld' in plot_df.columns:
                plot_df['juld'] = pd.to_datetime(plot_df['juld'], errors='coerce')
            if 'pres' in plot_df.columns:
                plot_df['pres'] = pd.to_numeric(plot_df['pres'], errors='coerce')

            # Plot controls in sidebar
            st.sidebar.markdown("---")
            st.sidebar.subheader("Profile plotting options")
            depth_min, depth_max = st.sidebar.slider("Depth range (m)", min_value=0, max_value=6000, value=(0, 1000), step=10, key="depth_range_comp")
            marker_size = st.sidebar.slider("Marker size", min_value=3, max_value=12, value=6, key="marker_size_comp")
            opacity = st.sidebar.slider("Marker opacity", min_value=0.1, max_value=1.0, value=0.9, step=0.05, key="marker_opacity_comp")

            depth_col = 'depth' if 'depth' in plot_df.columns else ('pres' if 'pres' in plot_df.columns else None)
            palette = px.colors.qualitative.Dark24

            # Two-panel: Temperature vs Depth (left), Pressure vs Depth (right)
            left_title = 'Temperature vs ' + (depth_col.capitalize() if depth_col else 'Depth')
            right_title = 'Pressure vs ' + (depth_col.capitalize() if depth_col else 'Depth')

            if depth_col and not plot_df.empty and plot_df[['temp', depth_col]].dropna().shape[0] > 0:
                pf = plot_df[(plot_df[depth_col] >= depth_min) & (plot_df[depth_col] <= depth_max)].copy()
                floats = sorted(pf['float_id'].unique())
                color_map = {f: palette[i % len(palette)] for i, f in enumerate(floats)}

                fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08,
                                    subplot_titles=(left_title, right_title))

                # Temp vs Depth (left)
                for fid in floats:
                    dff = pf[pf['float_id'] == fid].dropna(subset=['temp', depth_col]).sort_values(depth_col)
                    if dff.empty:
                        continue
                    color = color_map[fid]
                    for juld, grp in dff.groupby('juld'):
                        customdata = np.stack([grp['juld'].astype(str)], axis=1) if 'juld' in grp.columns else None
                        fig.add_trace(go.Scatter(x=grp['temp'], y=grp[depth_col], mode='lines+markers',
                                                 marker=dict(size=marker_size, opacity=opacity),
                                                 line=dict(width=1.2, color=color), name=f"{fid}",
                                                 hovertemplate="Float: %{text}<br>juld: %{customdata[0]}<br>pres: %{y}<br>temp: %{x}",
                                                 text=[fid] * len(grp), customdata=customdata
                                                 ), row=1, col=1)

                # Pressure vs Depth (right)
                if 'pres' in pf.columns:
                    for fid in floats:
                        dff = pf[pf['float_id'] == fid].dropna(subset=['pres', depth_col]).sort_values(depth_col)
                        if dff.empty:
                            continue
                        color = color_map[fid]
                        for juld, grp in dff.groupby('juld'):
                            customdata = np.stack([grp['juld'].astype(str)], axis=1) if 'juld' in grp.columns else None
                            fig.add_trace(go.Scatter(x=grp['pres'], y=grp[depth_col], mode='lines+markers',
                                                     marker=dict(size=marker_size, opacity=opacity),
                                                     line=dict(width=1.2, color=color), name=f"{fid}",
                                                     hovertemplate="Float: %{text}<br>juld: %{customdata[0]}<br>pres: %{x}<br>depth: %{y}",
                                                     text=[fid] * len(grp), customdata=customdata
                                                     ), row=1, col=2)

                fig.update_yaxes(autorange='reversed')
                fig.update_xaxes(title_text='Temperature (units in data)', row=1, col=1)
                fig.update_xaxes(title_text='Pressure (units in data)', row=1, col=2)
                fig.update_layout(height=600, template='plotly_dark',
                                  legend=dict(title='Float ID', orientation='v', x=1.02, y=0.95),
                                  margin=dict(t=80))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No temperature-depth profiles available for selected floats (after filtering).")

            # -----------------------------
            # Temperature vs Time (improved)
            # - allow aggregation method selection
            # - allow user-specified date range to filter plotted points
            # -----------------------------
            required = ['float_id', 'juld', 'temp']
            st.subheader('Temperature vs Time (per-profile representative)')

            if all(rc in plot_df.columns for rc in required):
                # aggregation method
                agg_method = st.selectbox("Representative temp per profile (aggregation)", options=[
                    "Shallowest (temp at minimum pres) — preferred if pres available",
                    "Median (per profile)",
                    "Mean (per profile)",
                    "Max (per profile)"
                ], index=1, key=f"temp_agg_method_{'_'.join(comp_sel)}")

                # Build df_time (drop NA)
                df_time = plot_df[['float_id', 'juld', 'temp'] + [c for c in ['file', 'pres', 'psal', 'temp_qc'] if c in plot_df.columns]].copy()
                df_time = df_time.dropna(subset=['juld', 'temp']).sort_values(['float_id', 'juld'])

                if df_time.empty:
                    st.info("No valid temperature/time rows to aggregate.")
                else:
                    # compute aggregated dataframe: one row per float_id + juld
                    if agg_method.startswith("Shallowest") and 'pres' in df_time.columns:
                        df_time_shallow = df_time.dropna(subset=['pres']).sort_values(['float_id', 'juld', 'pres'])
                        df_rep = df_time_shallow.groupby(['float_id', 'juld'], as_index=False).first()
                        # fallback median for groups without pres
                        all_groups = set(df_time.groupby(['float_id', 'juld']).size().index)
                        rep_groups = set(df_rep.set_index(['float_id', 'juld']).index)
                        missing_groups = all_groups - rep_groups
                        if missing_groups:
                            med = df_time.groupby(['float_id', 'juld'], as_index=False)['temp'].median()
                            med = med[med.set_index(['float_id','juld']).index.isin(missing_groups)]
                            if not med.empty:
                                for c in df_rep.columns:
                                    if c not in med.columns:
                                        med[c] = pd.NA
                                df_rep = pd.concat([df_rep, med[df_rep.columns]], ignore_index=True, sort=False)
                    else:
                        agg_func = 'median' if agg_method.startswith("Median") else ('mean' if agg_method.startswith("Mean") else 'max')
                        agg_dict = {'temp': agg_func}
                        optional = [c for c in ['file', 'pres', 'psal', 'temp_qc'] if c in df_time.columns]
                        for c in optional:
                            agg_dict[c] = 'first'
                        df_rep = df_time.groupby(['float_id', 'juld'], as_index=False).agg(agg_dict)

                    df_rep['juld'] = pd.to_datetime(df_rep['juld'], errors='coerce')
                    df_rep = df_rep.dropna(subset=['juld', 'temp']).sort_values(['float_id', 'juld'])
                    if df_rep.empty:
                        st.warning("No aggregated temperature/time points available after grouping.")
                    else:
                        # date-range chooser (defaults to whole range)
                        min_date = df_rep['juld'].min().date()
                        max_date = df_rep['juld'].max().date()
                        start_date, end_date = st.date_input(
                            "Filter date range for time plot",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key=f"temp_time_date_filter_{'_'.join(comp_sel)}"
                        )
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        df_plot_time = df_rep[(df_rep['juld'] >= start_dt) & (df_rep['juld'] <= end_dt)].copy()

                        if df_plot_time.empty:
                            st.warning("No aggregated temperature data in selected date range.")
                        else:
                            # improved color map
                            floats_time = sorted(df_plot_time['float_id'].unique())
                            color_map_time = {f: palette[i % len(palette)] for i, f in enumerate(floats_time)}

                            hover_cols = [c for c in ['file', 'pres', 'psal', 'float_id'] if c in df_plot_time.columns]
                            fig_temp_time = px.line(
                                df_plot_time,
                                x='juld',
                                y='temp',
                                color='float_id',
                                markers=True,
                                hover_data=hover_cols,
                                title='Temperature vs Time (per-profile representative)',
                                labels={'juld': 'Time (juld)', 'temp': 'Temperature (units in data)'},
                                color_discrete_map=color_map_time,
                                line_shape='spline'
                            )
                            # aesthetics
                            fig_temp_time.update_traces(mode='lines+markers', marker=dict(size=6, opacity=0.9), line=dict(width=1.5))
                            fig_temp_time.update_layout(
                                template='plotly_dark',
                                height=560,
                                legend_title_text='Float ID',
                                margin=dict(t=90),
                                legend=dict(x=1.02, y=0.95),
                                hovermode='x unified',
                                font=dict(size=12)
                            )
                            fig_temp_time.update_xaxes(rangeslider_visible=True)
                            fig_temp_time.update_layout(hoverlabel=dict(bgcolor="rgba(0,0,0,0.7)", font_size=12, font_color="white"))

                            cols_tt = st.columns([1, 0.18])
                            cols_tt[0].plotly_chart(fig_temp_time, use_container_width=True, key=f"temp_time_{uuid.uuid4().hex}")

                            # download aggregated timeseries CSV
                            csv_bytes = df_plot_time.to_csv(index=False).encode('utf-8')
                            cols_tt[1].download_button("Download CSV", csv_bytes, file_name=f"temp_time_{'_'.join(comp_sel)}.csv")
            else:
                st.info("Temperature vs date plot unavailable — `float_id`, `juld`, and `temp` columns are required.")

            # Show combined argo_info rows
            st.subheader("Combined argo_info rows for selected floats")
            if combined_df.empty:
                st.info("No measurement rows available to display from argo_info.")
            else:
                display_cols = [c for c in combined_df.columns]
                st.write(f"Showing {len(combined_df)} rows from `argo_info` for selected floats.")
                st.dataframe(combined_df[display_cols].sort_values(['float_id', 'juld'] if 'juld' in combined_df.columns else ['float_id']).reset_index(drop=True))
   
# ---------------------------
# Tab 5: Exports (tabs[5]) and Tab 6: ML: Temp Predictor (tabs[6])
# Paste these two blocks into your Streamlit UI section where you defined `tabs`.
# ---------------------------

# --- Exports tab ---
with tabs[5]:
    st.header("Exports")
    st.write("Export ingested `argo_info` rows to Parquet or a simple NetCDF file.")

    # determine a safe SQL selection for argo_info (only existing columns)
    try:
        available_cols = [c.name for c in argo_info_table.columns]
    except Exception:
        available_cols = None

    if st.button("Export ingested info to Parquet", key="export_parquet"):
        try:
            if available_cols:
                df = pd.read_sql_query("SELECT * FROM argo_info", engine)
            else:
                df = pd.read_sql_query("SELECT * FROM argo_info", engine)
        except Exception as e:
            st.error(f"Failed to read argo_info from DB: {e}")
            df = pd.DataFrame()

        if df.empty:
            st.warning("No ingested info to export.")
        else:
            path = os.path.join(STORAGE_ROOT, f"argo_info_{int(time.time())}.parquet")
            try:
                df.to_parquet(path, index=False)
                with open(path, "rb") as fh:
                    st.download_button("Download Parquet", fh.read(), file_name=os.path.basename(path), mime="application/octet-stream")
            except Exception as e:
                st.error(f"Failed to write Parquet: {e}")

    st.markdown("---")

    if st.button("Export ingested info to NetCDF (simple)", key="export_netcdf"):
        try:
            df = pd.read_sql_query("SELECT * FROM argo_info", engine)
        except Exception as e:
            st.error(f"Failed to read argo_info from DB: {e}")
            df = pd.DataFrame()

        if df.empty:
            st.warning("No ingested info to export.")
        else:
            try:
                # ensure columns exist and coerce types safely
                n = len(df)
                temp_arr = df.get("temp", pd.Series([np.nan] * n)).astype(float).fillna(np.nan).values
                file_arr = df.get("file", pd.Series([""] * n)).astype(str).values
                juld_ts = pd.to_datetime(df.get("juld")) if "juld" in df.columns else pd.to_datetime(pd.Series([pd.NaT] * n))
                lat_arr = df.get("latitude", pd.Series([np.nan] * n)).astype(float).values
                lon_arr = df.get("longitude", pd.Series([np.nan] * n)).astype(float).values
                pres_arr = df.get("pres", pd.Series([np.nan] * n)).astype(float).values
                param_arr = df.get("parameter", pd.Series([""] * n)).astype(str).values

                ds = xr.Dataset(
                    {"temp": (("row",), temp_arr)},
                    coords={
                        "row": np.arange(n),
                        "file": (("row",), file_arr),
                        "juld": (("row",), juld_ts.astype("datetime64[ns]").values),
                        "lat": (("row",), lat_arr),
                        "lon": (("row",), lon_arr),
                        "pres": (("row",), pres_arr),
                        "parameter": (("row",), param_arr),
                    },
                )
                path = os.path.join(STORAGE_ROOT, f"argo_info_{int(time.time())}.nc")
                ds.to_netcdf(path)
                with open(path, "rb") as fh:
                    st.download_button("Download NetCDF", fh.read(), file_name=os.path.basename(path), mime="application/octet-stream")
            except Exception as e:
                st.error(f"Failed to build NetCDF: {e}")

    st.sidebar.markdown("---")
    st.sidebar.write("Tip: For quick tests, index only a small number of rows (e.g., 100–1000). Large builds can be slow.")


# --- ML tab: Temperature predictor ---
# --- ML tab: Temperature predictor ---
# streamlit_temp_predictor_multi_model_fixed.py
# Full corrected code — trains multiple regressors and handles XGBoost sklearn/tag incompatibilities.
import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st

# guarded local imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.base import BaseEstimator, RegressorMixin
    import joblib
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# optional libs
_has_xgb = False
try:
    import xgboost as xgb  # noqa: F401
    _has_xgb = True
except Exception:
    _has_xgb = False

_has_lgb = False
try:
    import lightgbm as lgb  # noqa: F401
    _has_lgb = True
except Exception:
    _has_lgb = False

if not SKLEARN_OK:
    st.error("scikit-learn or joblib not available. Install `scikit-learn` and `joblib` to use ML features.")
    st.stop()

# ---- XGBoost safe wrapper --------------------------------------------------
# A lightweight wrapper around xgboost.XGBRegressor that exposes sklearn-compatible
# methods/tags so pipelines and some sklearn internals don't break when versions mismatch.
class XGBRegressorSafe(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        # keep parameters in self.params so get_params/set_params work
        self.params = kwargs.copy()

    def fit(self, X, y, **fit_kwargs):
        # create the underlying model lazily to avoid import-time issues
        from xgboost import XGBRegressor  # import locally
        # XGBoost's native sklearn wrapper can have compatibility problems with certain sklearn versions.
        # We instantiate and delegate to it.
        self._model = XGBRegressor(**self.params)
        # prefer numpy arrays for safety
        Xarr = X if isinstance(X, np.ndarray) else np.asarray(X)
        yarr = y if isinstance(y, np.ndarray) else np.asarray(y)
        # call fit, propagate fit_kwargs if provided
        self._model.fit(Xarr, yarr, **fit_kwargs)
        return self

    def predict(self, X):
        return self._model.predict(X if isinstance(X, np.ndarray) else np.asarray(X))

    def get_params(self, deep=True):
        # scikit-learn expects dict-like parameters
        return self.params.copy()

    def set_params(self, **params):
        self.params.update(params)
        return self

    # Provide _get_tags for newer sklearn expectations (safe default)
    def _get_tags(self):
        return {
            "requires_y": True,
            "multioutput": False,
            "preserves_dtype": None,
            "no_validation": False,
        }

# ---------------------------------------------------------------------------

# --- model storage path ---
# Make sure STORAGE_ROOT is defined in your main app; we rely on it here.
model_dir = os.path.join(STORAGE_ROOT, "models")
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, "temp_predictor_best.joblib")
all_models_path = os.path.join(model_dir, "temp_predictor_all.joblib")

# UI: header
with st.expander("ML: Temperature predictor (from argo_info) — multi-model (fixed)", expanded=True):
    st.write(
        "Train multiple regressors and automatically choose the best model for prediction.\n"
        "This version includes a safe XGBoost wrapper to avoid sklearn/xgboost tag incompatibilities."
    )

    # Training controls
    st.subheader("Training controls & dataset sampling")
    sample_size = st.number_input("Max training rows to sample", min_value=100, max_value=500000, value=20000, step=100)
    test_size_frac = st.slider("Test fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    metric_choice = st.selectbox("Model selection metric (best model will be the lowest RMSE or highest R²)", ["RMSE", "R2"])

    # choose models to train
    st.markdown("**Choose models to train**")
    train_rf = st.checkbox("RandomForest", value=True)
    train_gb = st.checkbox("GradientBoosting", value=True)
    train_hist = st.checkbox("HistGradientBoosting", value=False)
    train_xgb = st.checkbox(f"XGBoost (available={_has_xgb})", value=_has_xgb and False)
    train_lgb = st.checkbox(f"LightGBM (available={_has_lgb})", value=_has_lgb and False)

    # simple hyperparams (global)
    n_estimators = st.number_input("n_estimators (for tree ensembles)", min_value=10, max_value=2000, value=200, step=10)
    max_depth = st.number_input("max_depth (0=None)", min_value=0, max_value=100, value=15, step=1)

    run_train = st.button("Train selected models")

    @st.cache_data(ttl=3600)
    def load_training_df(max_rows):
        wanted = ["temp", "psal", "pres", "latitude", "longitude", "juld"]
        try:
            available = [c.name for c in argo_info_table.columns]
            select_cols = [c for c in wanted if c in available]
            if not select_cols:
                select_cols = wanted
            sql = "SELECT " + ", ".join(select_cols) + " FROM argo_info"
            df = pd.read_sql_query(sql, engine)
        except Exception:
            try:
                df = pd.read_sql_query("SELECT temp, psal, pres, latitude, longitude, juld FROM argo_info", engine)
            except Exception:
                df = pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(subset=["temp", "latitude", "longitude"], how="any")
        for c in ["temp", "psal", "pres", "latitude", "longitude"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "juld" in df.columns:
            df["juld"] = pd.to_datetime(df["juld"], errors="coerce")
            def _to_epoch(x):
                try:
                    if pd.isna(x):
                        return np.nan
                    return int(pd.Timestamp(x).timestamp())
                except Exception:
                    return np.nan
            df["juld_ts"] = df["juld"].apply(_to_epoch)
        else:
            df["juld_ts"] = np.nan
        df = df.dropna(subset=["latitude", "longitude", "temp"], how="any")
        if max_rows and len(df) > max_rows:
            df = df.sample(n=int(max_rows), random_state=42)
        return df.reset_index(drop=True)

    df_train = load_training_df(int(sample_size))
    st.write(f"Loaded {len(df_train)} rows for training (after basic cleaning).")
    if not df_train.empty:
        st.dataframe(df_train.head(8))

    # Train multiple models
    if run_train:
        if df_train.empty:
            st.error("No training data available in `argo_info` — ingest profiles first.")
        else:
            with st.spinner("Preparing features and training selected models..."):
                use_cols = [c for c in ["latitude", "longitude", "pres", "psal", "juld_ts"] if c in df_train.columns]
                if not all(c in use_cols for c in ["latitude", "longitude", "pres"]):
                    st.error("Required features missing (latitude, longitude, pres).")
                else:
                    X = df_train[use_cols].copy()
                    # fill numeric NaNs with medians
                    for col in X.columns:
                        if X[col].dtype.kind in "fiu":
                            med = X[col].median(skipna=True)
                            X[col] = X[col].fillna(med if not pd.isna(med) else 0.0)
                    y = pd.to_numeric(df_train["temp"], errors="coerce")
                    ok = ~(y.isna() | X.isna().any(axis=1))
                    X = X.loc[ok].astype(float)
                    y = y.loc[ok].astype(float)

                    if len(X) < 50:
                        st.warning("Very few rows available after filtering — model may not generalize.")

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size_frac), random_state=42)

                    # build candidate models
                    candidates = {}
                    if train_rf:
                        candidates["RandomForest"] = RandomForestRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=(None if int(max_depth) == 0 else int(max_depth)),
                            n_jobs=-1, random_state=42
                        )
                    if train_gb:
                        candidates["GradientBoosting"] = GradientBoostingRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=(None if int(max_depth) == 0 else int(max_depth)),
                            random_state=42
                        )
                    if train_hist:
                        candidates["HistGradientBoosting"] = HistGradientBoostingRegressor(
                            max_iter=int(n_estimators),
                            max_depth=(None if int(max_depth) == 0 else int(max_depth)),
                            random_state=42
                        )
                    if train_xgb and _has_xgb:
                        # use safe wrapper to avoid sklearn/xgboost tag mismatch issues
                        candidates["XGBoost"] = XGBRegressorSafe(
                            n_estimators=int(n_estimators),
                            max_depth=(0 if int(max_depth) == 0 else int(max_depth)),
                            objective='reg:squarederror',
                            n_jobs=-1,
                            random_state=42
                        )
                    if train_lgb and _has_lgb:
                        # LightGBM's sklearn wrapper tends to be okay, but keep it optional
                        try:
                            from lightgbm import LGBMRegressor
                            candidates["LightGBM"] = LGBMRegressor(
                                n_estimators=int(n_estimators),
                                max_depth=(-1 if int(max_depth) == 0 else int(max_depth)),
                                random_state=42
                            )
                        except Exception:
                            st.warning("LightGBM import failed at instantiation time; skipping LightGBM.")

                    if not candidates:
                        st.error("No models selected or available to train.")
                    else:
                        results = []
                        trained_models = {}
                        for name, model in candidates.items():
                            st.write(f"Training: {name} ...")
                            try:
                                scaler = StandardScaler()
                                pipe = Pipeline([("scaler", scaler), ("model", model)])
                                # Fit inside try/except — some wrappers may still fail depending on env versions
                                pipe.fit(X_train, y_train)
                                y_pred = pipe.predict(X_test)
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = float(np.sqrt(mse))
                                r2 = float(r2_score(y_test, y_pred))
                                results.append({"model": name, "rmse": rmse, "r2": r2})
                                trained_models[name] = pipe
                                st.write(f"{name} done — RMSE: {rmse:.4f}, R²: {r2:.4f}")
                            except Exception as e:
                                # detailed warning so user can inspect cause; continue training other models
                                st.warning(f"Training {name} failed: {e}")
                                # If the error looks like the sklearn_tags issue and name == "XGBoost" attempt fallback:
                                if name == "XGBoost" and "sklearn_tags" in str(e):
                                    st.info("Attempting fallback: re-train XGBoost using direct XGBRegressor inside pipeline.")
                                    try:
                                        # fallback: construct pipeline with a lambda/wrapper that instantiates real XGBRegressor at fit time
                                        class _LazyXGB(BaseEstimator, RegressorMixin):
                                            def __init__(self, **kw):
                                                self.kw = kw
                                            def fit(self, X_local, y_local):
                                                from xgboost import XGBRegressor as _X
                                                self._m = _X(**self.kw)
                                                self._m.fit(X_local, y_local)
                                                return self
                                            def predict(self, X_local):
                                                return self._m.predict(X_local)
                                            def get_params(self, deep=True): return self.kw.copy()
                                            def set_params(self, **p): self.kw.update(p); return self
                                            def _get_tags(self): return {"requires_y": True}
                                        lazy = _LazyXGB(n_estimators=int(n_estimators),
                                                        max_depth=(0 if int(max_depth) == 0 else int(max_depth)),
                                                        objective='reg:squarederror', n_jobs=-1, random_state=42)
                                        pipe2 = Pipeline([("scaler", StandardScaler()), ("model", lazy)])
                                        pipe2.fit(X_train, y_train)
                                        y_pred2 = pipe2.predict(X_test)
                                        mse2 = mean_squared_error(y_test, y_pred2)
                                        rmse2 = float(np.sqrt(mse2))
                                        r22 = float(r2_score(y_test, y_pred2))
                                        results.append({"model": name + "_fallback", "rmse": rmse2, "r2": r22})
                                        trained_models[name + "_fallback"] = pipe2
                                        st.write(f"{name} fallback done — RMSE: {rmse2:.4f}, R²: {r22:.4f}")
                                    except Exception as e2:
                                        st.warning(f"XGBoost fallback also failed: {e2}")
                                # continue to next model
                                continue

                        if not results:
                            st.error("All model trainings failed.")
                        else:
                            df_results = pd.DataFrame(results)
                            # sort depending on metric
                            if metric_choice == "RMSE":
                                df_results = df_results.sort_values(by="rmse", ascending=True)
                            else:
                                df_results = df_results.sort_values(by="r2", ascending=False)
                            st.subheader("Model comparison (test set)")
                            st.dataframe(df_results.reset_index(drop=True))

                            # pick best according to chosen metric
                            best_row = df_results.iloc[0]
                            best_name = best_row["model"]
                            best_pipe = trained_models.get(best_name)

                            # save best and all models
                            try:
                                joblib.dump(
                                    {"pipeline": best_pipe, "features": use_cols, "model_name": best_name, "metrics": df_results.to_dict(orient="records")},
                                    best_model_path
                                )
                                joblib.dump(
                                    {"models": trained_models, "features": use_cols, "metrics": df_results.to_dict(orient="records")},
                                    all_models_path
                                )
                                st.success(f"Best model: {best_name}. Saved to: {best_model_path}")
                            except Exception as e:
                                st.warning(f"Model save failed: {e}")

                            # quick plot for the best model
                            try:
                                y_pred_best = best_pipe.predict(X_test)
                                import plotly.express as px
                                fig = px.scatter(pd.DataFrame({"actual": y_test, "pred": y_pred_best}), x="actual", y="pred", title=f"{best_name}: predicted vs actual")
                                fig.add_shape(
                                    type="line",
                                    x0=min(y_test.min(), np.min(y_pred_best)), x1=max(y_test.max(), np.max(y_pred_best)),
                                    y0=min(y_test.min(), np.min(y_pred_best)), y1=max(y_test.max(), np.max(y_pred_best)),
                                    line=dict(dash="dash")
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass

    # --- Model load & inference UI ---
    st.subheader("Load model(s) & predict")

    loaded = False
    if os.path.exists(best_model_path):
        st.write(f"Saved best model exists at `{best_model_path}`")
        if st.button("Load saved best model"):
            try:
                blob = joblib.load(best_model_path)
                st.session_state["ml_model_pipeline"] = blob.get("pipeline")
                st.session_state["ml_model_features"] = blob.get("features", [])
                st.session_state["ml_model_name"] = blob.get("model_name", "best_model")
                st.success("Best model loaded into session.")
                loaded = True
            except Exception as e:
                st.error(f"Failed to load best model: {e}")
    if os.path.exists(all_models_path):
        st.write(f"Saved all trained models exists at `{all_models_path}`")
        if st.button("Load all saved models"):
            try:
                blob = joblib.load(all_models_path)
                st.session_state["ml_models"] = blob.get("models", {})
                st.session_state["ml_model_features"] = blob.get("features", [])
                st.success("All models loaded into session.")
                loaded = True
            except Exception as e:
                st.error(f"Failed to load all models: {e}")

    # auto-load if best exists and nothing in session
    if st.session_state.get("ml_model_pipeline") is None and os.path.exists(best_model_path):
        try:
            blob = joblib.load(best_model_path)
            st.session_state["ml_model_pipeline"] = blob.get("pipeline")
            st.session_state["ml_model_features"] = blob.get("features", [])
            st.session_state["ml_model_name"] = blob.get("model_name", "best_model")
        except Exception:
            pass

    model_pipeline = st.session_state.get("ml_model_pipeline")
    model_features = st.session_state.get("ml_model_features", [])
    models_dict = st.session_state.get("ml_models", {})

    if model_pipeline is None and not models_dict:
        st.warning("No model loaded. Train or load a model first.")
    else:
        # choose which loaded model to use for prediction
        use_best_auto = st.checkbox("Auto-use best loaded model for prediction (if available)", value=True)
        chosen_model_name = None
        if use_best_auto and model_pipeline is not None:
            chosen_model = model_pipeline
            chosen_model_name = st.session_state.get("ml_model_name", "best_model")
        else:
            # show dropdown of loaded models
            options = []
            if model_pipeline is not None:
                options.append(st.session_state.get("ml_model_name", "best_model"))
            options.extend(list(models_dict.keys()))
            if not options:
                st.warning("No models available in session.")
                chosen_model = None
            else:
                chosen_model_name = st.selectbox("Pick a loaded model for prediction", options)
                if chosen_model_name == st.session_state.get("ml_model_name"):
                    chosen_model = st.session_state.get("ml_model_pipeline")
                else:
                    chosen_model = models_dict.get(chosen_model_name)

        st.write(f"Model features: {model_features}")
        st.markdown("**Predict temperature for a single sample**")
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        p_lat = pred_col1.number_input("Latitude", value=0.0, format="%f", key="pred_lat")
        p_lon = pred_col1.number_input("Longitude", value=0.0, format="%f", key="pred_lon")
        p_pres = pred_col2.number_input("Pressure / depth (pres)", value=10.0, format="%f", key="pred_pres")
        p_psal = pred_col2.number_input("Practical Salinity (psal) — optional", value=float("nan"), format="%f", key="pred_psal")
        p_juld = pred_col3.date_input("Date (juld) — optional", value=datetime.date.today(), key="pred_juld")

        if st.button("Predict temperature (single)"):
            if chosen_model is None:
                st.error("No model selected for prediction.")
            else:
                feat_vals = {}
                for f in model_features:
                    if f == "latitude":
                        feat_vals[f] = float(p_lat)
                    elif f == "longitude":
                        feat_vals[f] = float(p_lon)
                    elif f == "pres":
                        feat_vals[f] = float(p_pres)
                    elif f == "psal":
                        try:
                            v = float(p_psal)
                            if np.isnan(v):
                                v = df_train["psal"].median() if "psal" in df_train.columns else 0.0
                            feat_vals[f] = v
                        except Exception:
                            feat_vals[f] = df_train["psal"].median() if "psal" in df_train.columns else 0.0
                    elif f == "juld_ts":
                        try:
                            dt = pd.Timestamp(p_juld)
                            feat_vals[f] = int(dt.timestamp())
                        except Exception:
                            feat_vals[f] = int(pd.Timestamp.now().timestamp())
                    else:
                        feat_vals[f] = 0.0

                Xpred = pd.DataFrame([feat_vals])
                Xpred = Xpred[model_features].astype(float)

                try:
                    model_obj = chosen_model
                    # pipeline case: get last step as leaf model for ensemble std calculation
                    if hasattr(model_obj, "named_steps"):
                        leaf_model = list(model_obj.named_steps.values())[-1]
                    else:
                        leaf_model = model_obj

                    # prefer pipeline predict (includes preprocessing)
                    pred = None
                    if hasattr(model_obj, "predict"):
                        pred = model_obj.predict(Xpred)[0]

                    # If underlying model has `estimators_` (sklearn RF or GB), compute tree-wise predictions for std
                    std_pred = None
                    if hasattr(leaf_model, "estimators_"):
                        try:
                            if hasattr(model_obj, "named_steps"):
                                prep = model_obj[:-1]
                                Xprep = prep.transform(Xpred)
                            else:
                                Xprep = Xpred.values
                            tree_preds = np.vstack([t.predict(Xprep) for t in leaf_model.estimators_])
                            mean_pred = float(np.mean(tree_preds, axis=0)[0])
                            std_pred = float(np.std(tree_preds, axis=0)[0])
                            st.success(f"Predicted temp = {mean_pred:.4f} ± {std_pred:.4f} (ensemble std)")
                        except Exception:
                            if pred is not None:
                                st.success(f"Predicted temp = {float(pred):.4f}")
                            else:
                                st.error("Prediction failed (ensemble).")
                    else:
                        if pred is not None:
                            st.success(f"Predicted temp = {float(pred):.4f}")
                        else:
                            st.error("Prediction failed: no predict() method on model.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Evaluate saved model on DB sample
    st.markdown("**Evaluate saved model on a DB sample**")
    if st.button("Evaluate saved model on DB sample"):
        # choose eval model: prefer best pipeline
        eval_model = None
        if st.session_state.get("ml_model_pipeline") is not None and use_best_auto:
            eval_model = st.session_state.get("ml_model_pipeline")
        elif models_dict and not use_best_auto and chosen_model_name:
            eval_model = models_dict.get(chosen_model_name)
        elif st.session_state.get("ml_model_pipeline") is not None:
            eval_model = st.session_state.get("ml_model_pipeline")

        if eval_model is None:
            st.error("Load or train a model first.")
        else:
            df_eval = load_training_df(int(sample_size))
            if df_eval.empty:
                st.error("No eval data available.")
            else:
                use_cols = model_features
                X_all = df_eval[use_cols].copy()
                for col in X_all.columns:
                    if X_all[col].dtype.kind in "fiu":
                        med = X_all[col].median(skipna=True)
                        X_all[col] = X_all[col].fillna(med if not pd.isna(med) else 0.0)
                y_all = pd.to_numeric(df_eval["temp"], errors="coerce")
                ok = ~(y_all.isna() | X_all.isna().any(axis=1))
                X_all = X_all.loc[ok].astype(float)
                y_all = y_all.loc[ok].astype(float)
                if len(X_all) < 20:
                    st.warning("Too few rows to evaluate robustly.")
                else:
                    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, random_state=42)
                    yhat = eval_model.predict(X_te)
                    mse_eval = mean_squared_error(y_te, yhat)
                    rmse_eval = float(np.sqrt(mse_eval))
                    r2_eval = float(r2_score(y_te, yhat))
                    st.write(f"Eval RMSE = {rmse_eval:.4f}, R² = {r2_eval:.4f}")
                    try:
                        import plotly.express as px
                        fig = px.scatter(pd.DataFrame({"actual": y_te, "pred": yhat}), x="actual", y="pred", title="Eval: predicted vs actual")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

    st.info("This example trains multiple tree-based regressors and picks the best by test performance. "
            "For heavier / production models consider using dedicated training scripts (parameter search, cross-validation) "
            "and serving via FastAPI or a model server.")
