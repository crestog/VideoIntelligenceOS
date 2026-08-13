"""
vios.capture.routes — the HTTP surface the admin tab talks to.

Thin on purpose. Every route is a translation of one engine call into JSON;
none of them contain logic that would be missed if the UI were replaced. The
engine is the product, this is the wiring.

Two rules govern everything here:

**Nothing blocks.** A channel scan takes a minute and a ZIP import takes a few
seconds, so both run on a thread and the browser polls. A route that parks for
sixty seconds makes the whole FastAPI worker look dead, and the operator's
first instinct will be to restart the session — which is the one thing that
costs real time.

**Credentials go in and never come out.** `POST /api/capture/config` accepts a
bot token; no route anywhere returns one. The status payload reports presence
(`bot_token_set: true`) and nothing else. This is not decoration: a status
endpoint that echoed the token would put it in the browser's network log, in
any screenshot of this tab, and in the reply of anyone who pasted the JSON into
a chat asking for help.
"""

from __future__ import annotations

import json
import os
import threading
import time

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .engine import get_engine

capture_router = APIRouter()

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

# State for the two long operations, so the UI can poll them.
_task = {"kind": "", "running": False, "message": "", "result": None,
         "error": "", "at": 0.0}
_task_lock = threading.Lock()


def _ok(**kw):
    return JSONResponse({"ok": True, **kw})


def _err(message: str, status: int = 400):
    return JSONResponse({"ok": False, "error": str(message)[:600]},
                        status_code=status)


def _run_task(kind: str, fn):
    """Start a background job, refusing to start a second one.

    One at a time is a deliberate limit, not laziness: a channel scan and a
    ledger import both write the same rows, and letting them interleave would
    produce a ledger whose state depends on timing.
    """
    with _task_lock:
        if _task["running"]:
            return _err(f"Busy: {_task['kind']} is still running.", 409)
        _task.update({"kind": kind, "running": True, "message": "Starting…",
                      "result": None, "error": "", "at": time.time()})

    def _wrap():
        try:
            _task["result"] = fn(lambda m: _task.update({"message": m}))
            _task["message"] = "Done."
        except Exception as exc:
            _task["error"] = f"{type(exc).__name__}: {exc}"
            _task["message"] = _task["error"]
        finally:
            _task["running"] = False

    threading.Thread(target=_wrap, name=f"vios-{kind}", daemon=True).start()
    return _ok(started=kind)


# ═══════════════════════════════════════════════════════════════════════
# Page
# ═══════════════════════════════════════════════════════════════════════
@capture_router.get("/capture", response_class=HTMLResponse)
def capture_page():
    path = os.path.join(_REPO, "capture_ui.html")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return HTMLResponse(fh.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>capture_ui.html not found</h1>", 404)


# ═══════════════════════════════════════════════════════════════════════
# Reading
# ═══════════════════════════════════════════════════════════════════════
@capture_router.get("/api/capture/status")
def status():
    try:
        s = get_engine().status()
        s["task"] = {k: v for k, v in _task.items() if k != "result"}
        return JSONResponse(s)
    except Exception as exc:
        return _err(exc, 500)


@capture_router.get("/api/capture/activity")
def activity(limit: int = 40):
    try:
        return _ok(events=get_engine().activity(min(int(limit), 200)))
    except Exception as exc:
        return _err(exc, 500)


@capture_router.get("/api/capture/task")
def task_state():
    return JSONResponse(dict(_task))


@capture_router.get("/api/capture/queue")
def queue(state: str = "", limit: int = 50, offset: int = 0):
    """A window into the ledger, for the operator who wants to see the list."""
    try:
        eng = get_engine()
        limit, offset = min(int(limit), 500), max(int(offset), 0)
        sql = ("SELECT key,url,state,attempts,last_error,added_at,done_at,"
               "uploader,views,likes,file_size,duration,msg_id FROM item")
        params = []
        if state:
            sql += " WHERE state=?"
            params.append(state)
        sql += " ORDER BY position LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = [dict(r) for r in eng.ledger.conn.execute(sql, params)]
        return _ok(items=rows, counts=eng.ledger.counts(),
                   offset=offset, limit=limit)
    except Exception as exc:
        return _err(exc, 500)


@capture_router.get("/api/capture/collections")
def collections():
    try:
        return _ok(collections=get_engine().ledger.collections())
    except Exception as exc:
        return _err(exc, 500)


@capture_router.get("/api/capture/failures")
def failures(limit: int = 100):
    try:
        return _ok(failures=get_engine().ledger.failures(min(int(limit), 500)))
    except Exception as exc:
        return _err(exc, 500)


# ═══════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════
@capture_router.post("/api/capture/config")
async def config(
    bot_token: str = Form(""),
    channel_id: str = Form(""),
    api_id: str = Form(""),
    api_hash: str = Form(""),
    cookies_text: str = Form(""),
    target_seconds: str = Form(""),
    quiet_hours: str = Form(""),
    breaks: str = Form(""),
    skip_collections: str = Form(""),
    max_attempts: str = Form(""),
    gallery_dl: str = Form(""),
    speed: str = Form(""),
):
    """Take the admin form. Blank means "leave it alone", so the operator can
    change the pace on day four without re-typing a token."""
    def _int(v, d=None):
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return d

    def _bool(v):
        s = str(v).strip().lower()
        return None if s == "" else s in ("1", "true", "on", "yes")

    try:
        out = get_engine().configure(
            bot_token=bot_token.strip(),
            channel_id=_int(channel_id),
            api_id=_int(api_id, 0) or 0,
            api_hash=api_hash.strip(),
            cookies_text=cookies_text,
            target=_int(target_seconds) or None,
            quiet_hours=_bool(quiet_hours),
            breaks=_bool(breaks),
            skip_collections=([c.strip() for c in skip_collections.split(",")
                               if c.strip()] if skip_collections else None),
            max_attempts=_int(max_attempts),
            allow_gallery_dl=_bool(gallery_dl),
            # Only "fast" or "safe" reach the engine; anything else is a typo
            # and must not silently mean one of them.
            speed=(speed.strip().lower()
                   if speed.strip().lower() in ("fast", "safe") else None),
        )
        return _ok(settings=out)
    except Exception as exc:
        return _err(exc)


@capture_router.post("/api/capture/preflight")
def preflight():
    try:
        res = get_engine().preflight()
        # The engine's own `ok` means "ready to run"; the envelope's `ok` means
        # "the request succeeded". Collapsing the two would make a correctly
        # reported "you have no cookies" look like a server error to the UI.
        res["ready"] = res.pop("ok", False)
        return _ok(**res)
    except Exception as exc:
        return _err(exc, 500)


# ═══════════════════════════════════════════════════════════════════════
# Filling the queue
# ═══════════════════════════════════════════════════════════════════════
@capture_router.post("/api/capture/import")
async def import_file(file: UploadFile = File(None), text: str = Form(""),
                      path: str = Form("")):
    """Accept the export ZIP, the markdown list, pasted links, or a local path.

    The upload is read into memory rather than streamed to disk: an Instagram
    export ZIP is a few MB of JSON, and a temp file that must be cleaned up on
    every error path is more moving parts than the size justifies.

    Parsing then happens on the task thread, not here. It is only a second or
    two, but this route is `async` — doing it inline parks the event loop, so
    the status poll that drives the whole tab stops answering and the page
    looks hung at the exact moment the user most wants to see progress.

    `path` is the escape hatch for when the browser upload is the problem: on
    Kaggle the export is usually already on disk as a dataset, and pointing at
    it skips a 20 MB round trip through the tunnel entirely.
    """
    try:
        eng = get_engine()

        if path.strip():
            src = os.path.expanduser(path.strip().strip('"').strip("'"))
            if not os.path.isfile(src):
                return _err(f"No file at {src}. Give the full path to the "
                            f"export ZIP as it exists on this machine.")
            return _run_task("import", lambda say: (
                say(f"Reading {os.path.basename(src)}…"),
                eng.import_file(src))[-1])

        if file is not None and file.filename:
            data = await file.read()
            if not data:
                return _err("That upload arrived empty — the connection "
                            "probably dropped mid-transfer. Try again, or use "
                            "the file-path box below.")
            if len(data) > 200 * 1024 * 1024:
                return _err("That file is over 200 MB — it is probably the "
                            "full media export. Only the JSON export is "
                            "needed; the media itself gets re-downloaded.")
            name = file.filename
            return _run_task("import", lambda say: (
                say(f"Parsing {name} ({len(data) / 1048576:.1f} MB)…"),
                eng.import_file(name, data))[-1])

        if text.strip():
            body = text
            return _run_task("import", lambda say: (
                say("Reading pasted links…"),
                eng.import_text(body, source="pasted"))[-1])

        return _err("Nothing to import — choose a file, give a path, or paste "
                    "some links.")
    except Exception as exc:
        return _err(exc)


@capture_router.post("/api/capture/seed/channel")
def seed_channel():
    """Scan the channel and adopt everything already uploaded."""
    eng = get_engine()
    if eng.telegram is None:
        return _err("Save the bot token and channel id first.")

    def _job(say):
        # The engine owns this now, so the route, the capture loop and the asset
        # backfill all seed the same way — one place that knows the credentials
        # and what to do when the scan fails.
        return eng.seed_ledger(
            on_progress=lambda at, head, n:
                say(f"Scanning message {at:,} of {head:,} — {n} reels found"))

    return _run_task("channel scan", _job)


@capture_router.post("/api/capture/seed/rescan")
def seed_rescan():
    """Forget the scan watermark so the next scan re-reads the whole channel."""
    try:
        eng = get_engine()
        eng.ledger.set_meta("scan_high_water", "0")
        eng.ledger.conn.commit()
        return _ok(message="Next scan will read the channel from message 1.")
    except Exception as exc:
        return _err(exc, 500)


@capture_router.post("/api/capture/seed/urls")
def seed_urls(text: str = Form("")):
    try:
        from .seed import seed_from_urls
        return _ok(**seed_from_urls(get_engine().ledger, text))
    except Exception as exc:
        return _err(exc)


@capture_router.post("/api/capture/requeue")
def requeue(state: str = Form("failed")):
    """Give up-and-retry, for after the cookies have been refreshed."""
    try:
        return _ok(requeued=get_engine().ledger.requeue(state))
    except Exception as exc:
        return _err(exc)


# ═══════════════════════════════════════════════════════════════════════
# Running
# ═══════════════════════════════════════════════════════════════════════
@capture_router.post("/api/capture/start")
def start(seed_first: str = Form("1")):
    try:
        res = get_engine().start(
            seed_first=str(seed_first).lower() not in ("0", "false", ""))
        return JSONResponse(res, status_code=200 if res["ok"] else 400)
    except Exception as exc:
        return _err(exc)


@capture_router.post("/api/capture/pause")
def pause():
    return JSONResponse(get_engine().pause())


@capture_router.post("/api/capture/resume")
def resume():
    return JSONResponse(get_engine().resume())


@capture_router.post("/api/capture/stop")
def stop():
    return JSONResponse(get_engine().stop())


@capture_router.post("/api/capture/snapshot")
def snapshot():
    """Push the ledger to Telegram now, rather than waiting for the interval."""
    try:
        return _ok(**get_engine().snapshot_now())
    except Exception as exc:
        return _err(exc, 500)


@capture_router.post("/api/capture/restore")
def restore():
    """Pull the pinned ledger back — the first thing a fresh Kaggle session
    does when the working directory is empty."""
    try:
        return _ok(**get_engine().restore_ledger())
    except Exception as exc:
        return _err(exc, 500)


@capture_router.get("/api/capture/export")
def export():
    """The whole ledger as JSON, for the operator who wants it elsewhere."""
    try:
        eng = get_engine()
        rows = [dict(r) for r in eng.ledger.conn.execute(
            "SELECT * FROM item ORDER BY position")]
        return JSONResponse(
            {"exported_at": time.time(), "counts": eng.ledger.counts(),
             "items": rows},
            headers={"Content-Disposition":
                     'attachment; filename="vios_capture_ledger.json"'})
    except Exception as exc:
        return _err(exc, 500)


# ═══════════════════════════════════════════════════════════════════════
# Asset sets for the videos that were captured before they existed
# ═══════════════════════════════════════════════════════════════════════
# Deliberately outside `_run_task`. A backfill runs for the better part of an
# hour, and the single task slot exists so a scan and an import cannot interleave
# their writes to the same rows — a rule that does not apply here, because a
# backfill only ever UPDATEs the four asset columns of a row that is already
# `uploaded`. Putting it in the slot would mean an operator could not scan the
# channel or paste a link for the whole hour, which is the wrong trade.
@capture_router.post("/api/capture/backfill/start")
def backfill_start(limit: str = Form("")):
    try:
        from .backfill import get_backfill
        try:
            n = int(str(limit).strip())
        except (TypeError, ValueError):
            n = 0
        res = get_backfill().start(get_engine(), limit=max(n, 0))
        return JSONResponse(res, status_code=200 if res["ok"] else 400)
    except Exception as exc:
        return _err(exc, 500)


@capture_router.post("/api/capture/backfill/stop")
def backfill_stop():
    try:
        from .backfill import get_backfill
        return JSONResponse(get_backfill().stop())
    except Exception as exc:
        return _err(exc, 500)


@capture_router.get("/api/capture/backfill")
def backfill_status():
    """State plus the archive-wide count, so the card can be read on its own."""
    try:
        from .backfill import autostart_state, get_backfill
        out = get_backfill().status()
        out["autostart"] = autostart_state()
        try:
            out["counts"] = get_engine().ledger.asset_counts()
        except Exception as exc:
            # The state of the worker is worth reporting even when the ledger
            # cannot be counted; collapsing the two would blank the whole card.
            out["counts"] = {"error": f"{type(exc).__name__}: {exc}"}
        return _ok(**out)
    except Exception as exc:
        return _err(exc, 500)
