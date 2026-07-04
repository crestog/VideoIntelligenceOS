"""
VIOS local test harness (dev only — NOT part of production boot).

Runs the REAL admin_backend + v17_backend routers against seeded data so
every fixed feature can be verified in a browser without Telegram/OpenCV:
  • 3 categories, 6 videos with staggered download timestamps
  • real JPEG frame folders (PIL-generated) + thumbnails + dummy MP4s
  • Redis running locally for category control / dedup / queue state

Usage: python test_harness.py   (serves on port 3000)
"""
import os
import sqlite3
import struct
import time

from config import LAKE_DIR, DB_PATH, VIDEO_DIR, THUMB_DIR

# ── Seed data ────────────────────────────────────────────────────────────
def seed():
    from PIL import Image

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS categories
                 (id INTEGER PRIMARY KEY, name TEXT UNIQUE)""")
    c.execute("""CREATE TABLE IF NOT EXISTS posts
                 (video_id INTEGER PRIMARY KEY, category_id INTEGER,
                  creator TEXT, likes INTEGER, caption TEXT,
                  local_video_path TEXT, status TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS videos
                 (msg_id INTEGER PRIMARY KEY, folder_id TEXT, title TEXT,
                  frames INTEGER, duration_sec REAL, duration_str TEXT,
                  thumb TEXT, first_frame TEXT, file_size_mb REAL,
                  abs_path TEXT, created_at REAL)""")

    cats = ["Nature", "Sports", "Cooking"]
    for i, name in enumerate(cats, start=1):
        c.execute("INSERT OR IGNORE INTO categories (id, name) VALUES (?, ?)", (i, name))

    now = time.time()
    videos = [
        # (msg_id, cat_id, hours_ago, n_frames, base_hue)
        (1001, 1, 48, 64, 0.60),
        (1002, 1, 30, 48, 0.33),
        (1003, 2, 20, 72, 0.08),
        (1004, 2, 10, 56, 0.75),
        (1005, 3, 4, 40, 0.15),
        (1006, 3, 1, 80, 0.50),
    ]

    import colorsys

    for msg_id, cat_id, hours_ago, n_frames, hue in videos:
        folder = f"frames_{msg_id}"
        fdir = os.path.join(VIDEO_DIR, folder)
        os.makedirs(fdir, exist_ok=True)
        created = now - hours_ago * 3600

        first = None
        for i in range(n_frames):
            ts = i / 6.0  # 6 fps
            fname = f"frame_{i:05d}_ts_{ts:.3f}s.jpg"
            fpath = os.path.join(fdir, fname)
            if first is None:
                first = fname
            if not os.path.exists(fpath):
                h = (hue + i / n_frames * 0.25) % 1.0
                r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, 0.65, 0.55 + 0.4 * (i % 10) / 10)]
                img = Image.new("RGB", (320, 180), (r, g, b))
                # add a moving bright square so histogram visibly changes
                px = img.load()
                sq = int((i / max(n_frames - 1, 1)) * 280)
                for x in range(sq, min(sq + 40, 320)):
                    for y in range(70, 110):
                        px[x, y] = (250, 250, 245)
                img.save(fpath, "JPEG", quality=82)

        # thumbnail + dummy mp4
        thumb = os.path.join(THUMB_DIR, f"{folder}.jpg")
        if not os.path.exists(thumb):
            Image.open(os.path.join(fdir, first)).save(thumb, "JPEG", quality=80)
        mp4 = os.path.join(VIDEO_DIR, f"video_{msg_id}.mp4")
        if not os.path.exists(mp4):
            with open(mp4, "wb") as fh:
                fh.write(os.urandom(256 * 1024))  # 256 KB dummy

        dur = (n_frames - 1) / 6.0
        c.execute(
            """INSERT OR REPLACE INTO videos
               (msg_id, folder_id, title, frames, duration_sec, duration_str,
                thumb, first_frame, file_size_mb, abs_path, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (msg_id, folder, f"Video #{msg_id}", n_frames, dur, f"{dur:.1f}s",
             f"/thumbs/{folder}.jpg", f"/data/{folder}/{first}", 0.25, fdir, created),
        )
        c.execute(
            """INSERT OR REPLACE INTO posts
               (video_id, category_id, creator, likes, caption, local_video_path, status)
               VALUES (?,?,?,?,?,?,?)""",
            (msg_id, cat_id, f"creator_{msg_id}", msg_id * 7, f"Caption for {msg_id}",
             mp4, "Harvested"),
        )
        os.utime(fdir, (created, created))

    # extra Metadata_Only posts so category priority has pending work
    for vid, cat in [(2001, 1), (2002, 2), (2003, 3), (2004, 1)]:
        c.execute(
            """INSERT OR REPLACE INTO posts
               (video_id, category_id, creator, likes, caption, local_video_path, status)
               VALUES (?,?,?,?,?,NULL,'Metadata_Only')""",
            (vid, cat, f"creator_{vid}", vid, f"Pending {vid}"),
        )

    conn.commit()
    conn.close()
    print(f"Seeded {len(videos)} videos across {len(cats)} categories")


# ── App ──────────────────────────────────────────────────────────────────
def build_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    from admin_backend import admin_router
    from v17_backend import v17_router

    app = FastAPI(title="VIOS Test Harness")
    app.include_router(admin_router)
    app.include_router(v17_router)
    app.mount("/data", StaticFiles(directory=VIDEO_DIR), name="data")
    app.mount("/thumbs", StaticFiles(directory=THUMB_DIR), name="thumbs")

    @app.get("/", response_class=HTMLResponse)
    def root():
        with open("main_ui.html", "r", encoding="utf-8") as fh:
            return fh.read()

    return app


if __name__ == "__main__":
    import uvicorn

    seed()
    uvicorn.run(build_app(), host="0.0.0.0", port=3000, log_level="warning")
