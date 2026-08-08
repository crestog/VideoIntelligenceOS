"""
vios.process.routes — the HTTP surface the processing tab talks to.

Same two rules as the capture plane, for the same reasons.

**Nothing blocks.** Restoring shards from a channel is a scan over thousands of
messages; a preflight sends a probe message and waits on Telegram. Both run on
a thread and the browser polls. A route that parks for sixty seconds makes the
worker look dead, and the operator's first instinct — restart the Kaggle
session — is the one move that actually costs hours.

**Credentials go in and never come out.** `POST /api/process/config` accepts a
bot token, an API hash and a Hugging Face token. No route here returns any of
them. `settings()` reports presence as a boolean and nothing more, so a
screenshot of this tab, or the network log behind it, contains no secret.

There is one route with real teeth: `reset`. It re-queues a pass that has
already run everywhere, which on a full sweep is hours of GPU time. It takes
the component id in the body rather than the path so it cannot be triggered by
a stray link, and it says in its response exactly how many rows it moved.
"""

from __future__ import annotations

import os
import threading
import time

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, JSONResponse

from . import registry
from .engine import get_engine

process_router = APIRouter()

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

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

    Serialised deliberately: a shard restore and a ledger sync both write the
    video table, and interleaving them would make the result depend on timing.
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

    threading.Thread(target=_wrap, name=f"vios-proc-{kind}",
                     daemon=True).start()
    return _ok(started=kind)


def _int(v, d=None):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return d


def _bool(v):
    s = str(v).strip().lower()
    return None if s == "" else s in ("1", "true", "on", "yes")


# ═══════════════════════════════════════════════════════════════════════
# Page
# ═══════════════════════════════════════════════════════════════════════
@process_router.get("/process", response_class=HTMLResponse)
def process_page():
    path = os.path.join(_REPO, "process_ui.html")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return HTMLResponse(fh.read())
    except OSError as exc:
        return HTMLResponse(
            f"<h1>process_ui.html is missing</h1><p>{exc}</p>", status_code=500)


# ═══════════════════════════════════════════════════════════════════════
# Reading
# ═══════════════════════════════════════════════════════════════════════
@process_router.get("/api/process/status")
def status(fresh: int = 0):
    """The one call the tab polls. Cached two seconds inside the engine."""
    try:
        s = get_engine().status(fresh=bool(fresh))
        s["task"] = {k: v for k, v in _task.items() if k != "result"}
        return JSONResponse(s)
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/activity")
def activity(limit: int = 60):
    try:
        return _ok(events=get_engine().activity(min(int(limit), 300)))
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/task")
def task_state():
    return JSONResponse(dict(_task))


@process_router.get("/api/process/catalog")
def catalog():
    """Every pass this system knows how to run, with what it costs and what it
    writes. The tab's list — the thing the operator actually reads before
    ticking a box."""
    try:
        eng = get_engine()
        return _ok(components=eng.catalog(),
                   stages=registry.STAGE_NAMES,
                   selected=list(eng.selected),
                   settings=eng.settings())
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/plan")
def plan():
    """What this machine would do, without doing it."""
    try:
        return _ok(**get_engine().plan())
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/coverage")
def coverage():
    """The matrix: one row per pass, one column per state."""
    try:
        eng = get_engine()
        return _ok(matrix=eng.coverage.matrix(), counts=eng.coverage.counts(),
                   running=eng.coverage.running())
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/failures")
def failures(limit: int = 100):
    try:
        return _ok(failures=get_engine().coverage.failures(
            min(int(limit), 500)))
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/videos")
def videos(limit: int = 50, offset: int = 0):
    """A window into the evidence store, so the operator can see what is in it
    rather than trusting a count."""
    try:
        eng = get_engine()
        limit, offset = min(int(limit), 500), max(int(offset), 0)
        rows = [dict(r) for r in eng.store.conn.execute(
            "SELECT video_key,url,uploader,duration,width,height,bytes,"
            "shots,taken_at,added_at FROM video "
            "ORDER BY added_at DESC LIMIT ? OFFSET ?", (limit, offset))]
        return _ok(videos=rows, offset=offset, limit=limit,
                   total=len(eng.store.video_keys()))
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/video/{key}")
def video_detail(key: str):
    """Everything known about one reel: shots, coverage, artifacts, claims.

    This is the route that makes the database inspectable instead of a number
    on a dashboard — the operator can read what a pass actually wrote and judge
    whether it was worth running.
    """
    try:
        out = get_engine().video_detail(key)
        return _ok(**out) if out else _err(f"no video {key}", 404)
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/search")
def search(q: str = "", limit: int = 50):
    """Full-text over every claim. The first taste of what the database is
    being built for."""
    try:
        if not q.strip():
            return _ok(results=[])
        return _ok(results=get_engine().store.search(q.strip(),
                                                     min(int(limit), 200)))
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/observers")
def observers():
    """Who said what. Every claim points at one of these rows, and this is how
    a disagreement between two models is traced back to the models."""
    try:
        return _ok(observers=get_engine().store.observers())
    except Exception as exc:
        return _err(exc, 500)


@process_router.get("/api/process/shards")
def shards():
    try:
        return _ok(shards=get_engine().store.shards())
    except Exception as exc:
        return _err(exc, 500)


# ═══════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════
@process_router.post("/api/process/config")
async def config(
    bot_token: str = Form(""),
    channel_id: str = Form(""),
    api_id: str = Form(""),
    api_hash: str = Form(""),
    hf_token: str = Form(""),
    components: str = Form(""),
    partitions: str = Form(""),
    index: str = Form(""),
    follow: str = Form(""),
    sync_on_start: str = Form(""),
    restore_on_start: str = Form(""),
    publish_every: str = Form(""),
    vram_headroom_mb: str = Form(""),
    cache_budget_mb: str = Form(""),
    video_limit: str = Form(""),
    ledger_path: str = Form(""),
    video_dirs: str = Form(""),
):
    """Take the tab's form. Blank means "leave it alone", so the operator can
    change the partition on day two without re-typing three credentials."""
    try:
        picked = ([c.strip() for c in components.split(",") if c.strip()]
                  if components.strip() else None)
        # Newline or comma: a path list is pasted, and a Windows path has no
        # comma in it but a Kaggle input mount is easier to paste one per line.
        dirs = ([d.strip() for d in video_dirs.replace(",", "\n").split("\n")
                 if d.strip()] if video_dirs.strip() else None)
        out = get_engine().configure(
            bot_token=bot_token.strip(),
            channel_id=_int(channel_id),
            api_id=_int(api_id, 0) or 0,
            api_hash=api_hash.strip(),
            hf_token=hf_token.strip(),
            components=picked,
            partitions=_int(partitions),
            index=_int(index),
            follow=_bool(follow),
            sync_on_start=_bool(sync_on_start),
            restore_on_start=_bool(restore_on_start),
            publish_every=_int(publish_every),
            vram_headroom_mb=_int(vram_headroom_mb),
            cache_budget_mb=_int(cache_budget_mb),
            video_limit=_int(video_limit),
            ledger_path=ledger_path.strip(),
            video_dirs=dirs,
        )
        return _ok(settings=out)
    except ValueError as exc:
        return _err(exc)
    except Exception as exc:
        return _err(exc, 500)


@process_router.post("/api/process/preflight")
def preflight():
    """Runs on a thread: it probes Telegram, which can take a few seconds, and
    a tab that freezes on the readiness check teaches the operator to skip it.
    """
    def _job(say):
        say("Checking hardware, packages and Telegram…")
        res = get_engine().preflight()
        res["ready"] = res.pop("ok", False)
        return res
    return _run_task("preflight", _job)


# ═══════════════════════════════════════════════════════════════════════
# Filling the work table
# ═══════════════════════════════════════════════════════════════════════
@process_router.post("/api/process/sync")
def sync():
    """Pull newly captured reels out of the capture ledger."""
    def _job(say):
        say("Reading the capture ledger…")
        return get_engine().sync_now()
    return _run_task("sync", _job)


@process_router.post("/api/process/restore")
def restore():
    """Replay every evidence shard the channel holds into this database.

    The first thing a fresh Kaggle session should do, and the reason a session
    dying at hour twelve costs nothing: the evidence is in the channel, not in
    this container.
    """
    def _job(say):
        eng = get_engine()
        if not (eng._tg and eng._tg.token):
            raise RuntimeError("No bot token — there is nothing to restore "
                               "from. Save your credentials first.")
        opened = False
        if not (eng._channel and eng._channel.ready):
            eng._open_channel()
            opened = True
        try:
            say("Scanning the channel for evidence shards…")
            return intake_restore(eng, say)
        finally:
            if opened and eng.state == "idle" and eng._channel:
                eng._channel.stop()
                eng._channel = None
    return _run_task("restore", _job)


def intake_restore(eng, say):
    from . import intake  # noqa: PLC0415 — keeps the import cost off page load
    return intake.restore_shards(
        eng.store, eng._tg, eng._channel,
        on_progress=lambda seen, head, n: say(
            f"{seen}/{head} messages scanned · {n} shards imported"))


@process_router.post("/api/process/adopt")
def adopt(folder: str = Form(...)):
    """Take a folder of video files into the work table.

    The counterpart to `sync`: reels that are already on this disk — a Kaggle
    dataset, a rescued scratch directory, the capture engine's own working
    folder — need neither a ledger row nor a Telegram message to be processed.
    """
    def _job(say):
        say(f"Scanning {folder} for videos…")
        return get_engine().adopt_folder_now(folder)
    return _run_task("adopt", _job)


@process_router.post("/api/process/publish")
def publish():
    """Export and upload everything written since the last shard. Press this
    before closing a Kaggle session early."""
    try:
        return _ok(**get_engine().publish_now())
    except Exception as exc:
        return _err(exc, 500)


# ═══════════════════════════════════════════════════════════════════════
# Running
# ═══════════════════════════════════════════════════════════════════════
@process_router.post("/api/process/start")
def start():
    try:
        res = get_engine().start()
        return JSONResponse(res, status_code=200 if res.get("ok") else 400)
    except Exception as exc:
        return _err(exc, 500)


@process_router.post("/api/process/pause")
def pause():
    return JSONResponse(get_engine().pause())


@process_router.post("/api/process/resume")
def resume():
    return JSONResponse(get_engine().resume())


@process_router.post("/api/process/stop")
def stop():
    return JSONResponse(get_engine().stop())


# ═══════════════════════════════════════════════════════════════════════
# Repair
# ═══════════════════════════════════════════════════════════════════════
@process_router.post("/api/process/requeue")
def requeue(component: str = Form(""), state: str = Form("failed")):
    """Give failed rows another chance — after installing the package they were
    missing, or after the model download that timed out."""
    try:
        return _ok(**get_engine().requeue(component.strip(), state.strip()))
    except Exception as exc:
        return _err(exc, 500)


@process_router.post("/api/process/reset")
def reset(component: str = Form(...), confirm: str = Form("")):
    """Re-run one pass over every video, from scratch.

    Expensive and irreversible in the sense that matters — it spends GPU hours.
    So it requires the component id echoed back in `confirm`, which the UI fills
    from a typed field. The claims already written are kept: they carry the old
    observer id and the new run carries a new one, and an append-only database
    that quietly forgot its previous opinion could not be audited at all.
    """
    cid = component.strip()
    if confirm.strip() != cid:
        return _err("Type the component id to confirm — this re-runs the pass "
                    "over every video and costs GPU hours.")
    try:
        registry.get(cid)
    except KeyError:
        return _err(f"no component {cid}", 404)
    try:
        return _ok(**get_engine().reset_component(cid))
    except Exception as exc:
        return _err(exc, 500)
