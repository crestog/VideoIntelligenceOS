import os, glob, time, json, shutil, struct, sqlite3
from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import HTMLResponse

v17_router = APIRouter()

# Mapped precisely to the integrated system's DataLake
VIDEO_DIR = '/kaggle/working/Insta-Vault/DataLake/videos'
THUMB_DIR = '/kaggle/working/Insta-Vault/DataLake/.thumbnails'
FLAG_DIR = '/kaggle/working/Insta-Vault/DataLake/_Flagged_Dataset'
DB_PATH = '/kaggle/working/Insta-Vault/DataLake/lake.db'
KAGGLE_LOGS = ["[SYSTEM] V17 Workspace Booted & Synced with DataLake."]

@v17_router.get("/api/database")
def get_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM frame_videos ORDER BY msg_id DESC")
        rows = c.fetchall()
        conn.close()
        payload = []
        for r in rows:
            d = dict(r)
            d["id"] = d["folder_id"]
            d["numeric_id"] = d["msg_id"]
            d["duration"] = d["duration_str"]
            d["size"] = f"{d['file_size_mb']:.1f} MB"
            payload.append(d)
        return {"folders": payload}
    except Exception: return {"folders": []}

@v17_router.get("/api/workspace/{folder_name}")
def get_workspace_data(folder_name: str):
    folder_path = os.path.join(VIDEO_DIR, folder_name)
    if not os.path.exists(folder_path): raise HTTPException(status_code=404, detail="Not Found")
    frames = sorted(glob.glob(os.path.join(folder_path, "frame_*.jpg")))
    filenames = [os.path.basename(f) for f in frames]
    fps = 30.0
    if len(frames) > 1:
        try:
            ts_start = float(filenames[0].split('_ts_')[1].replace('s.jpg', ''))
            ts_end = float(filenames[-1].split('_ts_')[1].replace('s.jpg', ''))
            dur = ts_end - ts_start
            if dur > 0: fps = len(frames) / dur
        except: pass
    return {"frames": filenames, "native_fps": fps}

@v17_router.get("/api/batch/{folder_name}")
def get_frame_batch(folder_name: str, start: int = 0, count: int = 100):
    folder_path = os.path.join(VIDEO_DIR, folder_name)
    frames = sorted(glob.glob(os.path.join(folder_path, "frame_*.jpg")))
    if not frames: raise HTTPException(status_code=404)
    packet = bytearray()
    end = min(start + count, len(frames))
    for i in range(start, end):
        try:
            with open(frames[i], "rb") as f:
                img_data = f.read()
                packet.extend(struct.pack('<I', len(img_data)))
                packet.extend(img_data)
        except: pass
    return Response(content=bytes(packet), media_type="application/octet-stream")

@v17_router.get("/api/logs")
def get_live_logs():
    return {"logs": KAGGLE_LOGS}

@v17_router.post("/api/flag/{folder_name}/{frame_filename}")
def flag_frame(folder_name: str, frame_filename: str):
    source_path = os.path.join(VIDEO_DIR, folder_name, frame_filename)
    target_path = os.path.join(FLAG_DIR, f"{folder_name}_{frame_filename}")
    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, target_path)
            return {"status": "success"}
        except: raise HTTPException(status_code=500)
    raise HTTPException(status_code=404)

@v17_router.get("/v17", response_class=HTMLResponse)
def serve_v17_ui():
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Insta-Vault V17 Workspace</title>
    <style>
        :root { --bg: #0f172a; --panel: #1e293b; --border: #334155; --text: #f8fafc; --accent: #0ea5e9; --red: #ef4444; --green: #10b981; --console: #020617;}
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; touch-action: manipulation; -webkit-tap-highlight-color: transparent;}
        body { background: var(--bg); color: var(--text); overflow-x: hidden; }
        
        #toast-container { position: fixed; top: 70px; left: 50%; transform: translateX(-50%); z-index: 9999; display: flex; flex-direction: column; gap: 10px; pointer-events: none; width: 90%; max-width: 400px;}
        .toast { background: rgba(15, 23, 42, 0.95); color: #fff; padding: 12px 20px; border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 4px 12px rgba(0,0,0,0.5); font-size: 14px; font-weight: bold; opacity: 0; transform: translateY(-20px); transition: all 0.2s ease-out; backdrop-filter: blur(8px); text-align: center; }
        .toast.show { opacity: 1; transform: translateY(0); }
        .toast.success { border-left: 4px solid var(--green); } .toast.error { border-left: 4px solid var(--red); }
        
        .tab-bar { display: flex; gap: 8px; background: rgba(15, 23, 42, 0.98); padding: 12px 16px; border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 100; backdrop-filter: blur(10px); }
        .tab-btn { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid transparent; background: transparent; color: #94a3b8; font-weight: bold; cursor: pointer; transition: 0.2s; font-size: 14px;}
        .tab-btn.active { background: var(--panel); color: var(--accent); border-color: var(--border); box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        .tab-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        
        .view { display: none; padding: 16px; max-width: 1200px; margin: 0 auto; animation: fadeIn 0.2s ease-out;}
        .view.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        
        .toolbar { display: grid; grid-template-columns: 1fr 200px 90px; gap: 8px; margin-bottom: 16px; width: 100%;}
        @media (max-width: 600px) { .toolbar { grid-template-columns: 1fr; } .toolbar > * { width: 100%; } }
        
        .toolbar input { padding: 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg); color: var(--text); outline: none; width: 100%;}
        .toolbar button { padding: 12px; border-radius: 8px; border: none; background: #334155; color: var(--text); cursor: pointer; font-weight: bold; width: 100%;}
        
        .custom-dropdown { position: relative; width: 100%; user-select: none; }
        .dropdown-selected { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; color: var(--text); font-weight: bold; cursor: pointer; display: flex; justify-content: space-between; align-items: center; width: 100%;}
        .dropdown-options { position: absolute; top: 100%; left: 0; right: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; margin-top: 4px; z-index: 50; overflow: hidden; display: none; box-shadow: 0 10px 25px rgba(0,0,0,0.5);}
        .dropdown-options.show { display: block; animation: slideDown 0.15s ease-out; }
        .dropdown-option { padding: 12px; color: #cbd5e1; cursor: pointer; font-size: 13px; font-weight: bold; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .dropdown-option:last-child { border-bottom: none; }
        .dropdown-option:active, .dropdown-option.selected { background: var(--accent); color: #fff; }
        @keyframes slideDown { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 14px; }
        @media (min-width: 768px) { .grid { grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; } }
        
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; cursor: pointer; position: relative;}
        .card:active { transform: scale(0.97); border-color: var(--accent); }
        .card-thumb { width: 100%; aspect-ratio: 16/9; background: #000; position: relative; }
        .card-thumb img { width: 100%; height: 100%; object-fit: cover; }
        .badge { position: absolute; bottom: 6px; right: 6px; background: rgba(15,23,42,0.85); font-size: 10px; padding: 4px 8px; border-radius: 4px; font-weight: bold; backdrop-filter: blur(4px);}
        .card-info { padding: 12px; }
        .card-title { font-size: 14px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 4px; }
        .card-meta { font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;}
        .card-id-badge { background: #334155; padding: 2px 6px; border-radius: 4px; color: #fff;}
        
        .player-container { 
            width: 100%; height: 52vh; background: #000; position: relative; display: flex;
            justify-content: center; align-items: center; overflow: hidden; border-radius: 12px 12px 0 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5); border: 1px solid var(--border); border-bottom: none;
        }
        canvas#viewport { max-width: 100%; max-height: 100%; object-fit: contain; cursor: grab; will-change: transform; }
        
        .overlay-hist { position: absolute; top: 12px; left: 12px; pointer-events: none; z-index: 10; border-radius: 4px; overflow: hidden;}
        canvas#histogram { width: 160px; height: 35px; display: block; background: rgba(0,0,0,0.4); backdrop-filter: blur(2px);}
        
        .overlay-filename { 
            position: absolute; bottom: 8px; right: 8px; 
            font-family: 'Courier New', monospace; font-size: 11px; color: #fff; font-weight: bold; 
            pointer-events: none; z-index: 10; 
            text-shadow: 1px 1px 2px #000, -1px -1px 2px #000, 0px 2px 5px rgba(0,0,0,0.9); 
        }
        
        .precision-info-strip {
            background: #1e293b; border: 1px solid var(--border); border-top: none; padding: 8px 14px;
            font-family: 'Courier New', monospace; font-size: 11px; color: #cbd5e1; display: flex;
            justify-content: space-between; align-items: center; border-radius: 0 0 12px 12px; gap: 10px;
            flex-wrap: wrap; margin-bottom: 12px;
        }
        .strip-segment { display: flex; gap: 14px; align-items: center;}
        .highlight-data { color: var(--accent); font-weight: bold;}
        
        .controls { background: var(--panel); padding: 16px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 16px; }
        
        .scrubber-container { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; width: 100%;}
        .scrub-time { font-family: 'Courier New', monospace; font-size: 13px; font-weight: bold; color: var(--accent); min-width: 60px; text-align: center; }
        .scrubber { flex-grow: 1; height: 8px; border-radius: 4px; accent-color: var(--accent); cursor: pointer;}
        
        .transport-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 14px;}
        .btn { background: #334155; border: none; color: var(--text); padding: 14px 0; border-radius: 8px; cursor: pointer; font-weight: bold; display: flex; justify-content: center; align-items: center; font-size: 16px; transition: 0.1s; user-select: none; -webkit-touch-callout: none;}
        .btn:active { transform: scale(0.92); background: #475569;}
        .btn.primary { background: var(--accent); }
        .btn.danger { background: var(--red); }
        .btn.success { background: var(--green); grid-column: span 5;}
        
        .util-row { display: flex; gap: 12px; flex-wrap: wrap; background: rgba(15, 23, 42, 0.4); padding: 12px; border-radius: 8px;}
        .util-box { flex: 1; min-width: 160px; display: flex; flex-direction: column; gap: 8px; }
        .util-label { font-size: 11px; color: #94a3b8; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;}
        
        .speed-control { display: flex; flex-direction: column; gap: 8px; }
        .speed-toggles { display: flex; gap: 4px; }
        .speed-btn { flex: 1; padding: 6px; background: #334155; border: 1px solid transparent; color: #94a3b8; border-radius: 4px; font-size: 11px; font-weight: bold; cursor: pointer;}
        .speed-btn.active { background: var(--accent); color: #fff; border-color: #0284c7;}
        
        .spinbox { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--bg);}
        .spinbox button { background: #334155; color: #fff; border: none; width: 36px; cursor: pointer; font-weight: bold; font-size: 14px;}
        .spinbox button:active { background: var(--accent); }
        .spinbox input { flex-grow: 1; background: transparent; border: none; color: #fff; text-align: center; font-weight: bold; font-family: 'Courier New', monospace; outline: none; -moz-appearance: textfield;}
        
        .jump-row { display: flex; gap: 6px; width: 100%; flex-wrap: nowrap;}
        .jump-row input { flex: 1; min-width: 50px; padding: 8px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: #fff; text-align: center; font-weight: bold;}
        .jump-row button { flex-shrink: 0; padding: 8px 12px; border-radius: 6px; border: none; background: var(--accent); color: #fff; font-weight: bold; cursor: pointer;}
        .buffer-pill { display: inline-block; background: rgba(14, 165, 233, 0.15); padding: 4px 10px; border-radius: 12px; font-weight: bold; color: var(--accent); border: 1px solid rgba(14, 165, 233, 0.3); font-size: 11px; margin-top: 10px; text-align: center;}

        .system-console-panel { background: var(--console); border: 1px solid var(--border); border-radius: 12px; padding: 14px; font-family: 'Courier New', monospace; margin-top: 30px;}
        .console-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 10px; color: var(--accent); font-size: 12px; font-weight: bold;}
        .metadata-dump-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; font-size: 11px; color: #94a3b8; margin-bottom: 14px;}
        .metadata-dump-grid div { background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
        .logs-stream-container { background: #000; border: 1px solid #1e293b; border-radius: 6px; height: 180px; overflow-y: auto; padding: 12px; font-size: 11px; color: #34d399; line-height: 1.6; }
    </style>
</head>
<body>

    <div id="toast-container"></div>

    <div class="tab-bar">
        <button class="tab-btn active" id="tab-db" onclick="switchView('database')">🗄️ Library</button>
        <button class="tab-btn" id="tab-work" onclick="switchView('workspace')" disabled>🎬 Workspace</button>
    </div>

    <div id="view-database" class="view active">
        <div class="toolbar">
            <input type="text" id="search-input" placeholder="🔍 Search ID or Name..." oninput="filterGrid()">
            
            <div class="custom-dropdown" id="sort-dropdown">
                <div class="dropdown-selected" onclick="toggleDropdown()">
                    <span id="sort-selected-text">Newest Telegram Videos</span>
                    <span>▼</span>
                </div>
                <div class="dropdown-options" id="sort-options">
                    <div class="dropdown-option selected" onclick="selectSort('id-desc', 'Newest Telegram Videos')">Newest Telegram Videos</div>
                    <div class="dropdown-option" onclick="selectSort('id-asc', 'Oldest Telegram Videos')">Oldest Telegram Videos</div>
                    <div class="dropdown-option" onclick="selectSort('name-asc', 'Video Name (A-Z)')">Video Name (A-Z)</div>
                    <div class="dropdown-option" onclick="selectSort('name-desc', 'Video Name (Z-A)')">Video Name (Z-A)</div>
                    <div class="dropdown-option" onclick="selectSort('frames-desc', 'Most Frames')">Most Frames</div>
                    <div class="dropdown-option" onclick="selectSort('size-desc', 'Largest File Size')">Largest File Size</div>
                </div>
            </div>
            
            <button onclick="fetchDatabase()">🔄 Sync</button>
        </div>
        
        <div class="grid" id="grid-container"></div>
        
        <div class="system-console-panel">
            <div class="console-header">
                <span>⚡ KAGGLE SYSTEM TELEMETRY & LOGS</span>
                <span style="color: #64748b; font-size: 10px;">V17 CORE ENGINE</span>
            </div>
            <div class="metadata-dump-grid">
                <div>📁 <b>Active Mount:</b> <span id="dbg-abs-path">Awaiting Mount...</span></div>
                <div>🗄️ <b>Storage Key:</b> <span id="dbg-folder-id">--</span></div>
                <div>📦 <b>Source Size:</b> <span id="dbg-file-size">--</span></div>
            </div>
            <div class="logs-stream-container" id="logs-stream-box">
                [SYSTEM START] Awaiting operations...
            </div>
        </div>
    </div>

    <div id="view-workspace" class="view">
        <div class="player-container" id="player-wrap">
            <div class="overlay-hist"><canvas id="histogram"></canvas></div>
            <div class="overlay-filename" id="hud-filename">frame_00000.jpg</div>
            <canvas id="viewport"></canvas>
        </div>
        
        <div class="precision-info-strip">
            <div class="strip-segment">
                <span>DIM: <span class="highlight-data" id="strip-res">0 x 0</span></span>
                <span>RATIO: <span class="highlight-data" id="strip-ratio">0:0</span></span>
                <span>NATIVE: <span class="highlight-data" id="strip-native-fps">00.00 FPS</span></span>
            </div>
            <div class="strip-segment">
                <span>FRAME: <span id="info-frame-idx" style="color:#fff; font-weight:bold;">0 / 0</span></span>
            </div>
        </div>

        <div class="controls">
            <div class="scrubber-container">
                <span class="scrub-time" id="scrub-current">0.000s</span>
                <input type="range" class="scrubber" id="timeline" min="0" max="0" value="0" oninput="scrubTo(this.value)">
                <span class="scrub-time" id="scrub-total" style="color:#94a3b8;">0.000s</span>
            </div>
            
            <div class="transport-grid">
                <button class="btn" id="btn-back-10">⏪</button>
                <button class="btn" id="btn-back-1">◀</button>
                <button class="btn primary" id="btn-play" onclick="togglePlay()">▶</button>
                <button class="btn" id="btn-fwd-1">▶</button>
                <button class="btn" id="btn-fwd-10">⏩</button>
                <button class="btn success" onclick="flagCurrentFrame()">📌 FLAG & EXPORT RAW FRAME</button>
            </div>

            <div class="util-row">
                <div class="util-box">
                    <span class="util-label">Playback Velocity</span>
                    <div class="speed-control">
                        <div class="speed-toggles">
                            <button id="btn-native" class="speed-btn active" onclick="setSpeedMode('native')">NATIVE</button>
                            <button class="speed-btn" onclick="injectFps(24)">24</button>
                            <button class="speed-btn" onclick="injectFps(30)">30</button>
                            <button class="speed-btn" onclick="injectFps(60)">60</button>
                        </div>
                        <div class="spinbox">
                            <button onclick="stepFps(-1)">-</button>
                            <input type="number" id="fps-val" value="30" onchange="updateCustomFps(this.value)">
                            <button onclick="stepFps(1)">+</button>
                        </div>
                    </div>
                </div>
                
                <div class="util-box">
                    <span class="util-label">Chronological Jump</span>
                    <div class="jump-row">
                        <input type="number" id="jump-sec" placeholder="Sec" step="0.1" min="0">
                        <button onclick="jumpToSecond()">JUMP</button>
                    </div>
                    <div class="buffer-pill" id="buffer-status">RAM: EMPTY</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let dbCache = [];
        const MULTI_CACHE = new Map(); 
        const MAX_CACHED_VIDEOS = 3;

        let activeFolder = "";
        let framesList = [];
        let currentIdx = 0;
        let nativeFps = 30;
        let activeFps = 30;
        let speedMode = 'native';
        let isPlaying = false;
        let animationFrameId;
        let lastDrawTime = 0;
        let activeRes = "Wait";
        
        let networkAborter = null;
        let activeSortVal = 'id-desc';

        const canvas = document.getElementById('viewport');
        const ctx = canvas.getContext('2d', { alpha: false });
        let scale = 1; let offsetX = 0; let offsetY = 0;
        let isDragging = false; let startX, startY;

        const histCanvas = document.getElementById('histogram');
        const hCtx = histCanvas.getContext('2d', { alpha: false });
        const dpr = window.devicePixelRatio || 1;
        const histW = 160; const histH = 35;
        histCanvas.width = histW * dpr;
        histCanvas.height = histH * dpr;
        hCtx.scale(dpr, dpr);
        
        const offCanvas = document.createElement('canvas');
        offCanvas.width = 32; offCanvas.height = 18; 
        const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
        let histTick = 0;

        const viewDB = document.getElementById('view-database');
        const viewWork = document.getElementById('view-workspace');
        const tabDB = document.getElementById('tab-db');
        const tabWork = document.getElementById('tab-work');
        const gridContainer = document.getElementById('grid-container');
        const searchInput = document.getElementById('search-input');
        const timeline = document.getElementById('timeline');
        const btnPlay = document.getElementById('btn-play');
        const bufferStatus = document.getElementById('buffer-status');
        
        const hudFilename = document.getElementById('hud-filename');
        const stripRes = document.getElementById('strip-res');
        const stripRatio = document.getElementById('strip-ratio');
        const stripNativeFps = document.getElementById('strip-native-fps');
        const scrubCurrent = document.getElementById('scrub-current');
        const scrubTotal = document.getElementById('scrub-total');
        const infoFrameIdx = document.getElementById('info-frame-idx');

        const dbgAbsPath = document.getElementById('dbg-abs-path');
        const dbgFolderId = document.getElementById('dbg-folder-id');
        const dbgFileSize = document.getElementById('dbg-file-size');
        const logsStreamBox = document.getElementById('logs-stream-box');

        window.addEventListener('popstate', (event) => {
            if(event.state && event.state.view === 'workspace') switchView('workspace', false);
            else switchView('database', false);
        });

        function toggleDropdown() {
            document.getElementById('sort-options').classList.toggle('show');
        }
        function selectSort(val, text) {
            activeSortVal = val;
            document.getElementById('sort-selected-text').innerText = text;
            document.querySelectorAll('.dropdown-option').forEach(el => el.classList.remove('selected'));
            event.target.classList.add('selected');
            toggleDropdown();
            filterGrid();
        }
        window.onclick = function(e) {
            if (!e.target.matches('.dropdown-selected') && !e.target.matches('.dropdown-selected *')) {
                const dropdowns = document.getElementsByClassName("dropdown-options");
                for (let i = 0; i < dropdowns.length; i++) {
                    dropdowns[i].classList.remove('show');
                }
            }
        }

        let holdTimer = null;
        let holdInterval = null;
        function bindHold(btnId, dir) {
            const btn = document.getElementById(btnId);
            const startHold = (e) => {
                if(e.cancelable) e.preventDefault();
                stopPlayback(); scrubTo(currentIdx + dir);
                holdTimer = setTimeout(() => {
                    holdInterval = setInterval(() => scrubTo(currentIdx + dir), 50);
                }, 300);
            };
            const stopHold = () => { clearTimeout(holdTimer); clearInterval(holdInterval); };
            
            btn.addEventListener('mousedown', startHold);
            btn.addEventListener('mouseup', stopHold);
            btn.addEventListener('mouseleave', stopHold);
            btn.addEventListener('touchstart', startHold, {passive: false});
            btn.addEventListener('touchend', stopHold);
        }
        bindHold('btn-back-10', -10);
        bindHold('btn-back-1', -1);
        bindHold('btn-fwd-1', 1);
        bindHold('btn-fwd-10', 10);

        function switchView(target, pushState = true) {
            if(target === 'database') {
                stopPlayback();
                if(networkAborter) networkAborter.abort(); 
                viewWork.classList.remove('active');
                viewDB.classList.add('active');
                tabWork.classList.remove('active');
                tabDB.classList.add('active');
                if(pushState) history.pushState({view: 'database'}, "", "#");
            } else {
                if(!activeFolder) return; 
                viewDB.classList.remove('active');
                viewWork.classList.add('active');
                tabDB.classList.remove('active');
                tabWork.classList.add('active');
                tabWork.disabled = false;
                resetZoom();
                window.scrollTo(0, 0);
                if(pushState) history.pushState({view: 'workspace'}, "", "#workspace");
            }
        }

        function showToast(message, type="info") {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerText = message;
            container.appendChild(toast);
            void toast.offsetWidth;
            toast.classList.add('show');
            setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 250); }, 2000);
        }

        function extractTimestamp(filename) {
            try { return filename.split('_ts_')[1].replace('s.jpg', '') + 's'; } 
            catch(e) { return "0.000s"; }
        }

        function calculateAspectRatio(width, height) {
            function gcd(a, b) { return b ? gcd(b, a % b) : a; }
            const divisor = gcd(width, height);
            return `${width / divisor}:${height / divisor}`;
        }

        function updateHistogram(img) {
            offCtx.drawImage(img, 0, 0, 32, 18);
            const data = offCtx.getImageData(0, 0, 32, 18).data;
            let r = new Array(256).fill(0); let g = new Array(256).fill(0); let b = new Array(256).fill(0);
            let maxVal = 0;
            
            for(let i=0; i<data.length; i+=4) { r[data[i]]++; g[data[i+1]]++; b[data[i+2]]++; }
            for(let i=0; i<256; i++) {
                if(r[i] > maxVal) maxVal = r[i];
                if(g[i] > maxVal) maxVal = g[i];
                if(b[i] > maxVal) maxVal = b[i];
            }

            hCtx.clearRect(0, 0, histW, histH);
            hCtx.globalCompositeOperation = "screen"; 
            
            const w = histW / 256; const h = histH;
            for(let i=0; i<256; i++) {
                hCtx.fillStyle = "rgba(239, 68, 68, 0.7)"; hCtx.fillRect(i*w, h-(r[i]/maxVal)*h, w, (r[i]/maxVal)*h);
                hCtx.fillStyle = "rgba(16, 185, 129, 0.7)"; hCtx.fillRect(i*w, h-(g[i]/maxVal)*h, w, (g[i]/maxVal)*h);
                hCtx.fillStyle = "rgba(14, 165, 233, 0.7)"; hCtx.fillRect(i*w, h-(b[i]/maxVal)*h, w, (b[i]/maxVal)*h);
            }
            hCtx.globalCompositeOperation = "source-over";
        }

        async function renderCanvas() {
            if(framesList.length === 0 && currentIdx !== 0) return; 
            
            const cacheBlock = MULTI_CACHE.get(activeFolder);
            if(!cacheBlock) return;

            const img = new Image();
            if (cacheBlock.blobMap.has(currentIdx)) {
                img.src = cacheBlock.blobMap.get(currentIdx);
            } else if(framesList[currentIdx]) {
                img.src = `/data/${activeFolder}/${framesList[currentIdx]}`;
            } else { return; }

            try {
                await img.decode(); 
                if(currentIdx != timeline.value && !isPlaying) return; 
                
                if(canvas.width !== img.width || canvas.height !== img.height) {
                    canvas.width = img.width; canvas.height = img.height;
                    activeRes = `${img.width} x ${img.height}`;
                    stripRes.innerText = activeRes;
                    stripRatio.innerText = calculateAspectRatio(img.width, img.height);
                }

                ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.save(); ctx.translate(offsetX, offsetY); ctx.scale(scale, scale);
                ctx.drawImage(img, 0, 0); ctx.restore();
                
                const filename = framesList[currentIdx];
                if(filename) {
                    scrubCurrent.innerText = extractTimestamp(filename);
                    hudFilename.innerText = filename;
                }
                infoFrameIdx.innerText = `${currentIdx} / ${Math.max(0, framesList.length - 1)}`;
                
                histTick++;
                if (!isPlaying || histTick % 3 === 0) { updateHistogram(img); }

            } catch(e) { }
        }

        function zoomCanvas(factor) {
            const oldScale = scale; scale *= factor;
            if(scale < 1) scale = 1; if(scale > 10) scale = 10;
            if (scale === 1) { offsetX = 0; offsetY = 0; }
            else {
                const cx = canvas.width / 2; const cy = canvas.height / 2;
                offsetX = cx - (cx - offsetX) * (scale / oldScale);
                offsetY = cy - (cy - offsetY) * (scale / oldScale);
            }
            renderCanvas();
        }
        function resetZoom() { scale = 1; offsetX = 0; offsetY = 0; renderCanvas(); }

        canvas.addEventListener('mousedown', e => { isDragging = true; startX = e.clientX - offsetX; startY = e.clientY - offsetY; });
        canvas.addEventListener('mousemove', e => { if(!isDragging || scale === 1) return; offsetX = e.clientX - startX; offsetY = e.clientY - startY; renderCanvas(); });
        canvas.addEventListener('mouseup', () => isDragging = false);
        canvas.addEventListener('mouseleave', () => isDragging = false);
        canvas.addEventListener('touchstart', e => {
            if(scale === 1) return; 
            if(e.touches.length === 1) { isDragging = true; startX = e.touches[0].clientX - offsetX; startY = e.touches[0].clientY - offsetY; }
        }, {passive: true}); 
        canvas.addEventListener('touchmove', e => {
            if(scale === 1) return; e.preventDefault(); 
            if(!isDragging) return; offsetX = e.touches[0].clientX - startX; offsetY = e.touches[0].clientY - startY; renderCanvas();
        }, {passive: false});
        canvas.addEventListener('touchend', () => isDragging = false);

        async function fetchDatabase() {
            gridContainer.innerHTML = "<p style='color:#94a3b8; padding: 20px;'>Querying Deep Database...</p>";
            try {
                const res = await fetch('/api/database');
                const data = await res.json();
                dbCache = data.folders;
                filterGrid();
                fetchSystemLogs();
            } catch(e) { gridContainer.innerHTML = "<p style='color:var(--red)'>Error connecting to API.</p>"; }
        }

        async function fetchSystemLogs() {
            try {
                const res = await fetch('/api/logs');
                const data = await res.json();
                logsStreamBox.innerHTML = data.logs.reverse().join("<br>");
            } catch(e) {}
        }

        function filterGrid() {
            const query = searchInput.value.toLowerCase();
            const sortMode = activeSortVal;
            
            let sortedCache = [...dbCache];
            sortedCache.sort((a, b) => {
                if (sortMode === 'id-desc') return b.numeric_id - a.numeric_id;
                if (sortMode === 'id-asc') return a.numeric_id - b.numeric_id;
                if (sortMode === 'name-asc') return a.title.localeCompare(b.title);
                if (sortMode === 'name-desc') return b.title.localeCompare(a.title);
                if (sortMode === 'frames-desc') return b.frames - a.frames;
                if (sortMode === 'size-desc') return b.file_size_mb - a.file_size_mb;
                return 0;
            });

            let htmlStr = "";
            let count = 0;
            sortedCache.forEach(item => {
                if(item.title.toLowerCase().includes(query) || item.id.toLowerCase().includes(query)) {
                    count++;
                    htmlStr += `
                        <div class="card" onclick="openWorkspace('${item.id}')">
                            <div class="card-thumb">
                                <img src="${item.thumb}" loading="lazy">
                                <span class="badge">${item.duration}</span>
                            </div>
                            <div class="card-info">
                                <div class="card-title">${item.title}</div>
                                <div class="card-meta">
                                    <span>${item.frames} Frm | ${item.size}</span> 
                                    <span class="card-id-badge">#${item.numeric_id}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }
            });
            gridContainer.innerHTML = count === 0 ? "<p style='color:#94a3b8; padding: 20px;'>No matching assets found.</p>" : htmlStr;
        }

        async function openWorkspace(folderId) {
            const itemData = dbCache.find(x => x.id === folderId);
            if(!itemData) return;

            if(networkAborter) networkAborter.abort();
            networkAborter = new AbortController();
            let fetchSessionId = Date.now(); 
            const currentSession = fetchSessionId;
            
            activeFolder = folderId;
            
            dbgAbsPath.innerText = itemData.abs_path;
            dbgFolderId.innerText = folderId;
            dbgFileSize.innerText = itemData.size;
            
            switchView('workspace', true);
            
            if (MULTI_CACHE.has(folderId)) {
                const cacheBlock = MULTI_CACHE.get(folderId);
                framesList = cacheBlock.frames;
                nativeFps = cacheBlock.fps;
                currentIdx = cacheBlock.lastIdx;
                
                timeline.max = framesList.length - 1;
                timeline.value = currentIdx;
                setSpeedMode(speedMode); 
                
                bufferStatus.innerText = `RAM: CACHED`;
                stripNativeFps.innerText = `${nativeFps.toFixed(2)} FPS`;
                scrubTotal.innerText = itemData.duration;
                
                renderCanvas();
                fetchSystemLogs();
                return;
            }

            if (MULTI_CACHE.size >= MAX_CACHED_VIDEOS) {
                const firstKey = MULTI_CACHE.keys().next().value;
                const block = MULTI_CACHE.get(firstKey);
                block.blobMap.forEach(url => URL.revokeObjectURL(url)); 
                MULTI_CACHE.delete(firstKey);
            }

            const newBlock = { frames: [], fps: 30.0, blobMap: new Map(), lastIdx: 0 };
            MULTI_CACHE.set(folderId, newBlock);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            hCtx.clearRect(0, 0, histW, histH);
            
            if (itemData.first_frame) {
                const ghostImg = new Image(); ghostImg.src = itemData.first_frame;
                ghostImg.onload = () => {
                    if (currentSession !== fetchSessionId) return; 
                    if(canvas.width !== ghostImg.width) {
                        canvas.width = ghostImg.width; canvas.height = ghostImg.height;
                        activeRes = `${ghostImg.width} x ${ghostImg.height}`;
                        stripRes.innerText = activeRes;
                        stripRatio.innerText = calculateAspectRatio(ghostImg.width, ghostImg.height);
                    }
                    ctx.drawImage(ghostImg, 0, 0);
                    scrubCurrent.innerText = "0.000s"; hudFilename.innerText = "Loading...";
                    infoFrameIdx.innerText = `0 / ?`; scrubTotal.innerText = itemData.duration;
                    updateHistogram(ghostImg);
                };
            }

            bufferStatus.innerText = "RAM: 0%";
            framesList = []; currentIdx = 0; timeline.value = 0;
            
            try {
                const res = await fetch(`/api/workspace/${folderId}`, { signal: networkAborter.signal });
                if (!res.ok) throw new Error("Mount 404");
                
                const data = await res.json();
                framesList = data.frames;
                nativeFps = data.native_fps;
                
                newBlock.frames = framesList;
                newBlock.fps = nativeFps;
                
                setSpeedMode('native');
                timeline.max = framesList.length - 1;
                stripNativeFps.innerText = `${nativeFps.toFixed(2)} FPS`;
                
                startBinaryBuffer(folderId, newBlock, networkAborter.signal, currentSession);
                
            } catch(e) { 
                if(e.name === 'AbortError') return; 
                showToast("Mount Failed", "error"); MULTI_CACHE.delete(folderId); 
            }
            fetchSystemLogs();
        }

        async function startBinaryBuffer(folder, cacheBlock, signal, session) {
            let startIdx = 0; const CHUNK_SIZE = 100; let retryCount = 0;
            
            while(startIdx < framesList.length) {
                try {
                    const res = await fetch(`/api/batch/${folder}?start=${startIdx}&count=${CHUNK_SIZE}`, { signal });
                    if(!res.ok) throw new Error("Chunk Failed");
                    
                    const buffer = await res.arrayBuffer();
                    const view = new DataView(buffer);
                    let offset = 0; let framePointer = startIdx;
                    
                    while(offset < buffer.byteLength) {
                        const size = view.getUint32(offset, true); offset += 4;
                        if (!cacheBlock.blobMap.has(framePointer)) {
                            const blob = new Blob([buffer.slice(offset, offset + size)], {type: 'image/jpeg'});
                            cacheBlock.blobMap.set(framePointer, URL.createObjectURL(blob));
                        }
                        framePointer++; offset += size;
                    }
                    
                    const pct = Math.round((cacheBlock.blobMap.size / framesList.length) * 100);
                    bufferStatus.innerText = `RAM: ${Math.min(pct, 100)}%`;
                    startIdx += CHUNK_SIZE; retryCount = 0;
                    
                    if(startIdx === CHUNK_SIZE) { renderCanvas(); } 
                    
                } catch(e) { 
                    if(e.name === 'AbortError') return; 
                    retryCount++;
                    if(retryCount > 3) { showToast("Buffer Sync Stalled.", "error"); break; }
                    await new Promise(r => setTimeout(r, 1000));
                }
            }
            if (startIdx >= framesList.length) { bufferStatus.innerText = "RAM: SECURED"; }
        }

        function scrubTo(idx) {
            idx = parseInt(idx);
            if(idx < 0) idx = 0;
            if(idx >= framesList.length) idx = framesList.length - 1;
            currentIdx = idx;
            timeline.value = idx;
            
            if(MULTI_CACHE.has(activeFolder)) MULTI_CACHE.get(activeFolder).lastIdx = currentIdx;
            renderCanvas();
        }

        function jumpToSecond() {
            const sec = parseFloat(document.getElementById('jump-sec').value);
            if(isNaN(sec) || sec < 0) return;
            scrubTo(Math.round(sec * nativeFps));
        }

        function setSpeedMode(mode) {
            speedMode = mode;
            const spinInput = document.getElementById('fps-val');
            const btnNative = document.getElementById('btn-native');
            const btns = document.querySelectorAll('.speed-btn');
            
            btns.forEach(b => b.classList.remove('active'));
            
            if (mode === 'native') {
                activeFps = nativeFps;
                btnNative.classList.add('active');
                spinInput.value = Math.round(nativeFps);
            } else {
                activeFps = parseFloat(spinInput.value);
            }
        }

        function updateCustomFps(val) {
            setSpeedMode('custom');
            let v = parseFloat(val);
            if(v < 1) v = 1; if(v > 120) v = 120;
            activeFps = v;
            document.getElementById('fps-val').value = v;
        }

        function stepFps(dir) {
            setSpeedMode('custom');
            let v = parseFloat(document.getElementById('fps-val').value) + dir;
            if(v < 1) v = 1; if(v > 120) v = 120;
            activeFps = v;
            document.getElementById('fps-val').value = v;
        }

        function injectFps(val) {
            document.getElementById('fps-val').value = val;
            updateCustomFps(val);
            event.target.classList.add('active'); 
        }
        
        async function flagCurrentFrame() {
            if(!activeFolder || framesList.length === 0) return;
            const filename = framesList[currentIdx];
            try {
                const res = await fetch(`/api/flag/${activeFolder}/${filename}`, {method: 'POST'});
                if(res.ok) { showToast(`✅ Flagged: FRM ${currentIdx}`, "success"); fetchSystemLogs(); }
                else showToast(`❌ Failed to flag.`, "error");
            } catch(e) { showToast(`❌ Network error.`, "error"); }
        }

        function playLoop(timestamp) {
            if (!isPlaying) return;
            const delay = 1000 / (activeFps > 0 ? activeFps : 30);
            if (timestamp - lastDrawTime >= delay) {
                let nextIdx = currentIdx + 1;
                if (nextIdx >= framesList.length) nextIdx = 0; 
                
                const cacheBlock = MULTI_CACHE.get(activeFolder);
                if(cacheBlock && (cacheBlock.blobMap.has(nextIdx) || framesList[nextIdx])) {
                    scrubTo(nextIdx);
                    lastDrawTime = timestamp;
                }
            }
            animationFrameId = requestAnimationFrame(playLoop);
        }

        function togglePlay() {
            if(isPlaying) { stopPlayback(); } 
            else {
                if(currentIdx >= framesList.length - 1) scrubTo(0); 
                isPlaying = true;
                btnPlay.innerText = "⏸";
                btnPlay.classList.replace('primary', 'danger');
                lastDrawTime = performance.now();
                animationFrameId = requestAnimationFrame(playLoop);
            }
        }

        function stopPlayback() {
            isPlaying = false;
            btnPlay.innerText = "▶";
            btnPlay.classList.replace('danger', 'primary');
            cancelAnimationFrame(animationFrameId);
        }

        document.addEventListener('keydown', (e) => {
            if(activeFolder === "" || document.activeElement.tagName === 'INPUT') return;
            if(e.code === 'Space') { e.preventDefault(); togglePlay(); }
            else if(e.code === 'ArrowRight') { e.preventDefault(); stopPlayback(); scrubTo(currentIdx + (e.shiftKey ? 10 : 1)); }
            else if(e.code === 'ArrowLeft') { e.preventDefault(); stopPlayback(); scrubTo(currentIdx - (e.shiftKey ? 10 : 1)); }
            else if(e.code === 'KeyF') { e.preventDefault(); flagCurrentFrame(); }
        });

        if(window.location.hash === '#workspace') history.replaceState({view: 'database'}, "", "#");
        fetchDatabase();
        setInterval(fetchSystemLogs, 3000); 
    </script>
</body>
</html>
"""
