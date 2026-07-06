"""Local UI preview — serves the V18 workstation against the sample DataLake.
No Telegram, no Redis, no GPU: only the FastAPI endpoints the UI needs.
Run: python local_preview.py  →  http://localhost:8000
"""

import os

os.environ.setdefault('VIOS_BASE_DIR',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Insta-Vault'))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import VIDEO_DIR, THUMB_DIR
from v17_backend import v17_router

app = FastAPI(title="VIOS V18 Local Preview")
app.mount("/data", StaticFiles(directory=VIDEO_DIR), name="v17_data")
app.mount("/thumbs", StaticFiles(directory=THUMB_DIR), name="v17_thumbs")
app.include_router(v17_router)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v17_ui.html"),
              "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
