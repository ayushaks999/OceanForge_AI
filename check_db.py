import sqlite3
import os
import sys

# Default DB path (safe for Windows)
DEFAULT_DB_PATH = r"C:\Users\Ayush\Desktop\Apple\SIH\argo.db"

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_PATH

# Check if DB exists
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"❌ Database file not found: {DB_PATH}")

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Check if argo_profiles table exists
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='argo_profiles';")
if cur.fetchone() is None:
    raise Exception("❌ Table 'argo_profiles' not found in DB. Did you run ingest.py?")

# Show row count
cur.execute("SELECT COUNT(*) FROM argo_profiles;")
print("✅ Total rows:", cur.fetchone()[0])

# Show preview
cur.execute("SELECT * FROM argo_profiles LIMIT 5;")
rows = cur.fetchall()
print("🔍 Preview:", rows)

conn.close()
