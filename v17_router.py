import os, glob, sqlite3
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

v17_router = APIRouter()
VIDEO_DIR = '/kaggle/working/Insta-Vault/DataLake/videos'
DB_PATH = '/kaggle/working/Insta-Vault/DataLake/lake.db'

@v17_router.get("/v17", response_class=HTMLResponse)
async def serve_ui():
    with open("v17_ui.html", "r") as f: return f.read()

@v17_router.get("/api/database")
def get_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM videos ORDER BY msg_id DESC")
        return {"folders": [dict(r) for r in c.fetchall()]}
    except: return {"folders": []}
