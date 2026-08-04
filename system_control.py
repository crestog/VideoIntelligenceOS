"""
system_control.py — one pause switch for the whole system, and one reset that
actually empties the machine.

Two operations live here because they answer the same question from opposite
ends: "make everything stop" and "make everything gone".

═══════════════════════════════════════════════════════════════════════════
GLOBAL PAUSE
═══════════════════════════════════════════════════════════════════════════
Before this there were two pause flags and neither covered the system:

  * CV_PAUSED   — frame_worker, and only in the branch taken when the queue
                  came back EMPTY. With a backlog the worker never reached that
                  branch, so pressing Pause on a busy machine did nothing until
                  the backlog drained — which is precisely when nobody needs a
                  pause button.
  * OMNI_PAUSED — the Omniscient vision and oracle loops.
  * the harvester had no flag at all, so downloads kept filling the disk while
    the panel said "paused".

Now one key, VIOS_PAUSED, covers every loop that consumes work:

    harvest   ui_server.background_downloader   Telegram downloads
    cv        frame_worker                      frame extraction
    analyze   model_manager                     the GPU analysis queue
    omni      omni_engine                       vision + oracle workers

The per-component flags still work — pausing only the CV engine is a real
thing to want — so `is_paused(component)` is the OR of the global key and that
component's own key. `pause_all` writes all of them, which also means any loop
that still reads only the legacy key stops correctly.

A flag on its own is a claim, not a fact, so each loop calls `heartbeat()` on
every pass. The panel shows what each component last reported (running, paused,
idle) with the age of that report, so "paused" means observed rather than
requested. Heartbeats expire after HEARTBEAT_TTL: a component that stops
reporting reads as stale instead of silently keeping its last state forever.

What pause does NOT cover, deliberately: an in-flight export or restore. Those
are single bounded transfers with their own Cancel, and freezing one mid-HTTP
request would hold a socket open for no benefit. The panel says so.

═══════════════════════════════════════════════════════════════════════════
FACTORY RESET
═══════════════════════════════════════════════════════════════════════════
Kaggle's own "factory reset" restarts the container and hands back an empty
disk. There is no equivalent from inside a running session — and a running
session is where you actually discover you want one, usually because 19.5 GB of
OUTPUT quota is full and every write is failing. This is that button.

It is genuinely destructive: files are removed, the Postgres schema is dropped,
Redis is flushed. Two things bound the damage:

  * Scope. Each target is opt-in and priced in bytes before you commit, because
    "delete everything" means something different when the model cache is 28 GB
    of re-downloadable weights and lake.db is the only record of what was ever
    harvested.
  * The channel is never touched. Exported bundles live in Telegram, which is
    the one tier that survives a container, so a reset is always recoverable
    through Database Restore — as long as an export ran first. The preview says
    exactly that, including whether a bundle exists.

The system is paused before the wipe and the workers are restarted after it, so
nothing is holding a deleted file or a stale connection when it is over.
"""

import os
import shutil
import subprocess
import threading
import time

from config import (BASE_DIR, LAKE_DIR, DB_PATH, VIDEO_DIR, THUMB_DIR,
                    ARCHIVE_DIR, QDRANT_PATH, NEO4J_DATA_DIR, NEO4J_HOME,
                    MODEL_CACHE_DIR, SCRATCH_DIR, SESSION_DIR,
                    OMNI_PG_DB, OMNI_PG_USER, OMNI_PG_PASSWORD, OMNI_PG_HOST)
from logger import vios_log

# ═══════════════════════════════════════════════════════════
# PAUSE
# ═══════════════════════════════════════════════════════════
PAUSE_KEY = "VIOS_PAUSED"
PAUSE_REASON_KEY = "VIOS_PAUSED_REASON"
PAUSE_SINCE_KEY = "VIOS_PAUSED_AT"

# Component → its own pause key. The two legacy names are kept exactly as they
# were: other code (and any half-updated worker) still reads them.
COMPONENT_KEYS = {
    "harvest": "HARVEST_PAUSED",
    "cv": "CV_PAUSED",
    "analyze": "ANALYZE_PAUSED",
    "omni": "OMNI_PAUSED",
}

COMPONENT_LABELS = {
    "harvest": "Harvester (Telegram downloads)",
    "cv": "CV engine (frame extraction)",
    "analyze": "Analysis queue (GPU models)",
    "omni": "Omniscient engine (vision + oracle)",
}

# Components that run more than one loop under one pause flag. Each loop
# heartbeats separately (VIOS_HB:omni:vision) because "the Omniscient engine is
# running" is not useful when the vision worker is busy and the oracle worker
# died — the panel shows both and rolls them up.
SUBWORKERS = {"omni": ("vision", "oracle")}

# Ordered by how much they matter when rolling several sub-workers into one
# line: anything still running outranks anything idle.
_STATE_RANK = {"running": 4, "idle": 3, "paused": 2, "stale": 1, "unknown": 0}

# A heartbeat older than this is reported as stale rather than trusted. Long
# enough that a worker inside one slow job (a 4-minute Qwen chunk) does not
# flicker; short enough that a crashed worker is obvious within a poll or two.
HEARTBEAT_TTL = 90


def _redis():
    """Redis handle or None. Never raises: every caller here is either a status
    read or a control action, and both should degrade to a clear message rather
    than a 500."""
    try:
        from queue_manager import get_redis
        r = get_redis()
        r.ping()
        return r
    except Exception:
        return None


def is_paused(component: str | None = None) -> bool:
    """True when the global switch is on, or when this component's own is.

    Workers call this on every loop pass, so it must be cheap and it must not
    raise — a Redis blip should not stop the pipeline, and the safe reading of
    "I cannot tell" is "not paused" (the alternative silently halts a working
    system).
    """
    r = _redis()
    if r is None:
        return False
    try:
        if r.get(PAUSE_KEY) == "1":
            return True
        key = COMPONENT_KEYS.get(component or "")
        return bool(key) and r.get(key) == "1"
    except Exception:
        return False


def heartbeat(component: str, state: str, detail: str = "") -> None:
    """Report what this component is actually doing. Best-effort by design.

    `state` is one of running | paused | idle. `component` may name a
    sub-worker as "omni:vision" so two loops sharing one pause flag do not
    overwrite each other's report. This is what turns the pause indicator from
    "we wrote a flag" into "every component confirmed it stopped", which is the
    difference the old panel could not show.
    """
    r = _redis()
    if r is None:
        return
    try:
        r.hset(f"VIOS_HB:{component}",
               mapping={"state": state, "detail": detail[:120],
                        "at": f"{time.time():.0f}"})
        r.expire(f"VIOS_HB:{component}", HEARTBEAT_TTL * 4)
    except Exception:
        pass


def wait_while_paused(component: str, poll: float = 2.0,
                      on_pause=None, on_resume=None) -> None:
    """Block here for as long as the system is paused.

    Callers use this at the top of their consume loop, before claiming a job —
    not after, which was frame_worker's bug: a pause that only takes effect
    once the queue empties is not a pause.
    """
    if not is_paused(component):
        return
    if on_pause:
        on_pause()
    heartbeat(component, "paused")
    while is_paused(component):
        time.sleep(poll)
        heartbeat(component, "paused")
    if on_resume:
        on_resume()
    heartbeat(component, "running")


def pause_all(reason: str = "") -> dict:
    """Stop every consumer. Writes the global key and each component key, so a
    loop reading only its legacy flag halts too."""
    r = _redis()
    if r is None:
        return {"ok": False, "error": "Redis is unavailable — cannot pause."}
    try:
        r.set(PAUSE_KEY, "1")
        r.set(PAUSE_SINCE_KEY, f"{time.time():.0f}")
        r.set(PAUSE_REASON_KEY, (reason or "paused from the admin panel")[:200])
        for key in COMPONENT_KEYS.values():
            r.set(key, "1")
    except Exception as e:
        return {"ok": False, "error": f"Redis write failed: {str(e)[:160]}"}
    vios_log(f"SYSTEM PAUSED — {reason or 'admin panel'}", "ADMIN", "WARN")
    return {"ok": True, "paused": True}


def resume_all() -> dict:
    """Release every flag pause_all set."""
    r = _redis()
    if r is None:
        return {"ok": False, "error": "Redis is unavailable — cannot resume."}
    try:
        r.delete(PAUSE_KEY, PAUSE_SINCE_KEY, PAUSE_REASON_KEY,
                 *COMPONENT_KEYS.values())
    except Exception as e:
        return {"ok": False, "error": f"Redis write failed: {str(e)[:160]}"}
    vios_log("SYSTEM RESUMED", "ADMIN", "SUCCESS")
    return {"ok": True, "paused": False}


def set_component(component: str, paused: bool) -> dict:
    """Pause or resume one component without touching the others."""
    key = COMPONENT_KEYS.get(component)
    if not key:
        return {"ok": False, "error": f"Unknown component {component!r}."}
    r = _redis()
    if r is None:
        return {"ok": False, "error": "Redis is unavailable."}
    try:
        r.set(key, "1") if paused else r.delete(key)
    except Exception as e:
        return {"ok": False, "error": f"Redis write failed: {str(e)[:160]}"}
    vios_log(f"{COMPONENT_LABELS[component]} "
             f"{'PAUSED' if paused else 'RESUMED'}", "ADMIN", "INFO")
    return {"ok": True, "component": component, "paused": paused}


def _read_hb(r, key: str) -> dict:
    """One heartbeat, with its age resolved into a state the panel can render.

    A heartbeat older than HEARTBEAT_TTL is reported as "stale" rather than
    believed — a worker that died mid-job would otherwise show its last state
    forever, which is exactly the kind of frozen-but-confident display this
    whole change exists to remove.
    """
    hb = {}
    try:
        hb = r.hgetall(f"VIOS_HB:{key}") or {}
    except Exception:
        pass
    if not hb:
        return {"state": "unknown", "detail": "", "age": None}
    age = None
    try:
        age = round(time.time() - float(hb.get("at", 0)))
    except (TypeError, ValueError):
        pass
    state = hb.get("state") or "unknown"
    if age is None or age > HEARTBEAT_TTL:
        state = "stale"
    return {"state": state, "detail": hb.get("detail", ""), "age": age}


def pause_state() -> dict:
    """Requested state plus observed state, per component.

    The two are reported separately on purpose. `paused` is what was asked for;
    `state` is what the component last said about itself. They disagree for a
    few seconds after a click (the worker is finishing a chunk), and they
    disagree permanently when a worker is dead — which is worth seeing.
    """
    r = _redis()
    if r is None:
        return {"ok": False, "redis": False, "global_paused": False,
                "components": [], "error": "Redis is unavailable."}

    out = {"ok": True, "redis": True, "components": []}
    try:
        out["global_paused"] = r.get(PAUSE_KEY) == "1"
        out["reason"] = r.get(PAUSE_REASON_KEY) or ""
        since = r.get(PAUSE_SINCE_KEY)
        out["paused_since"] = float(since) if since else None
        out["paused_for_s"] = (round(time.time() - float(since))
                               if since else None)

        for name, key in COMPONENT_KEYS.items():
            own = r.get(key) == "1"
            subs = SUBWORKERS.get(name)
            if subs:
                reports = {s: _read_hb(r, f"{name}:{s}") for s in subs}
                # Roll up to the busiest sub-worker: one live vision loop means
                # the engine has not stopped, whatever the oracle says.
                lead = max(reports.values(),
                           key=lambda h: _STATE_RANK.get(h["state"], 0))
                detail = " · ".join(
                    f"{s}: {h['state']}" + (f" ({h['detail']})" if h["detail"]
                                            else "")
                    for s, h in reports.items())
                hb = {"state": lead["state"], "detail": detail,
                      "age": lead["age"]}
            else:
                hb = _read_hb(r, name)

            out["components"].append({
                "name": name,
                "label": COMPONENT_LABELS[name],
                "paused": bool(own or out["global_paused"]),
                "own_flag": own,
                "state": hb["state"],
                "detail": hb["detail"],
                "heartbeat_age_s": hb["age"],
            })
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)[:200]
    return out


# ═══════════════════════════════════════════════════════════
# FACTORY RESET
# ═══════════════════════════════════════════════════════════
def _du(path: str) -> int:
    """Bytes under `path`, counting each inode once and never following links.

    Same rule as the disk panel: a HuggingFace cache stores one blob and points
    at it from every snapshot, so a naive walk reports several times the real
    size — which would make the reset preview promise space it cannot free.
    """
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.stat(path, follow_symlinks=False).st_size
        except OSError:
            return 0
    total, seen, stack = 0, set(), [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                            continue
                        st = entry.stat(follow_symlinks=False)
                        if st.st_nlink > 1:
                            ident = (st.st_dev, st.st_ino)
                            if ident in seen:
                                continue
                            seen.add(ident)
                        total += st.st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def _export_dir() -> str:
    from db_export import EXPORT_DIR
    return EXPORT_DIR


def _rm(path: str) -> None:
    """Remove a file or tree, if it is there."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path) or os.path.islink(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _empty_dir(path: str) -> None:
    """Delete everything *inside* a directory but keep the directory.

    Several of these paths are held open or were created at import time by
    config; removing the directory itself would make the next write fail with
    ENOENT rather than recreate it.
    """
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        _rm(os.path.join(path, name))


# Each target: what it is, what it costs to lose, how to measure it, how to
# wipe it. `safe` marks the ones that are re-derivable from something else —
# they default on. The rest are opt-in per click.
def _targets() -> list:
    return [
        {
            "id": "media",
            "label": "Videos, frames and thumbnails",
            "paths": [VIDEO_DIR, THUMB_DIR, ARCHIVE_DIR],
            "loss": "Downloaded reels and every extracted frame. Re-downloadable "
                    "from the channel; frames re-extract on the next pass.",
            "safe": True,
        },
        {
            "id": "vectors",
            "label": "Qdrant vector store",
            "paths": [QDRANT_PATH],
            "loss": "Semantic search index. Rebuilt by an encoder pass over the "
                    "frames — never shipped in a bundle for that reason.",
            "safe": True,
        },
        {
            "id": "graph",
            "label": "Neo4j knowledge graph",
            "paths": [NEO4J_DATA_DIR],
            "loss": "Entity graph. Projected from the Postgres narratives, so it "
                    "returns when those are re-ingested.",
            "safe": True,
        },
        {
            "id": "queues",
            "label": "Redis queues, dedup sets and logs",
            "paths": [],
            "loss": "In-flight jobs and the processed-video set. Session state "
                    "only — the dedup set is rebuilt from the database at boot.",
            "safe": True,
        },
        {
            "id": "exports",
            "label": "Local export bundles",
            "paths": [_export_dir()],
            "loss": "Bundle copies still on disk. Anything already uploaded is "
                    "in the channel and is NOT affected.",
            "safe": True,
        },
        {
            "id": "postgres",
            "label": "PostgreSQL omnidb (frames, chunks, narratives)",
            "paths": [],
            "loss": "Every Qwen narrative this machine produced. GPU hours, not "
                    "bandwidth. Recoverable only from an exported bundle.",
            "safe": False,
        },
        {
            "id": "sqlite",
            "label": "lake.db (harvest index)",
            "paths": [DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm"],
            "loss": "The record of which reels exist, their creators, categories "
                    "and captions. Nothing else reproduces it. Recoverable only "
                    "from an exported bundle.",
            "safe": False,
        },
        {
            "id": "models",
            "label": "Model weights cache",
            "paths": [MODEL_CACHE_DIR],
            "loss": "~28 GB of weights. Nothing is lost permanently, but the "
                    "next boot re-downloads them before anything can run.",
            "safe": False,
        },
        {
            "id": "telegram_session",
            "label": "Telegram session files",
            "paths": [SESSION_DIR, SESSION_DIR + ".session",
                      os.path.join(LAKE_DIR, "bot_session_export.session")],
            "loss": "Cached auth. The bot re-authenticates from the token on the "
                    "next start; nothing user-visible.",
            "safe": True,
        },
    ]


def reset_preview() -> dict:
    """What each target holds right now, and whether the channel can undo it.

    Sizes are measured, not estimated, because the number is the entire reason
    to press this: on a Kaggle box the question is always "will this free
    enough to keep going".
    """
    targets = []
    for t in _targets():
        size = sum(_du(p) for p in t["paths"])
        if t["id"] == "postgres":
            size = _pg_size()
        if t["id"] == "queues":
            size = _redis_size()
        targets.append({
            "id": t["id"], "label": t["label"], "loss": t["loss"],
            "safe": t["safe"], "bytes": size,
            "size_mb": round(size / 1048576, 1),
            "paths": [p for p in t["paths"] if os.path.exists(p)],
        })

    return {
        "targets": targets,
        "total_mb": round(sum(t["bytes"] for t in targets) / 1048576, 1),
        "confirm_phrase": CONFIRM_PHRASE,
        "recovery": _recovery_note(),
        "channel_note": (
            "Nothing in the Telegram channel is deleted. Reels and uploaded "
            "bundles stay exactly where they are — this only clears what is on "
            "this Kaggle container."),
    }


def _pg_size() -> int:
    """Size of the Omniscient database as Postgres itself reports it."""
    if not shutil.which("psql"):
        return 0
    env = dict(os.environ, PGPASSWORD=OMNI_PG_PASSWORD)
    try:
        p = subprocess.run(
            ["psql", "-h", OMNI_PG_HOST, "-U", OMNI_PG_USER, "-d", OMNI_PG_DB,
             "-tAc", f"SELECT pg_database_size('{OMNI_PG_DB}')"],
            env=env, capture_output=True, text=True, timeout=20)
        return int((p.stdout or "0").strip() or 0)
    except (subprocess.SubprocessError, OSError, ValueError):
        return 0


def _redis_size() -> int:
    r = _redis()
    if r is None:
        return 0
    try:
        return int(r.info("memory").get("used_memory", 0))
    except Exception:
        return 0


def _recovery_note() -> dict:
    """Whether a reset is undoable, answered with the actual bundle list rather
    than a reassurance. A reset with no bundle anywhere is permanent, and that
    should be on screen before the click, not after."""
    local = []
    try:
        from db_export import list_local_bundles
        local = list_local_bundles()
    except Exception:
        pass
    return {
        "local_bundles": len(local),
        "newest_local": local[0]["name"] if local else None,
        "advice": (
            "A bundle in the channel is the only way back. Run Database Export "
            "first if this database holds anything you want to keep — restore "
            "reads it from Telegram, which a reset cannot touch."),
    }


CONFIRM_PHRASE = "DELETE EVERYTHING"

_lock = threading.Lock()
_job: dict = {
    "state": "idle",          # idle | running | done | error
    "stage": "",
    "pct": 0,
    "detail": "",
    "started_at": None,
    "finished_at": None,
    "results": [],
    "freed_mb": 0,
    "error": None,
    "log": [],
}


def _set(**kw):
    with _lock:
        _job.update(kw)
        if "stage" in kw:
            line = f"{time.strftime('%H:%M:%S')} · {kw['stage']}"
            if kw.get("detail"):
                line += f" — {kw['detail']}"
            _job["log"] = (_job["log"] + [line])[-40:]


def reset_status() -> dict:
    with _lock:
        return dict(_job)


def is_resetting() -> bool:
    with _lock:
        return _job["state"] == "running"


def _wipe_postgres() -> str:
    """Drop and recreate the public schema. Returns "" on success."""
    if not shutil.which("psql"):
        return "psql not installed"
    env = dict(os.environ, PGPASSWORD=OMNI_PG_PASSWORD)
    try:
        p = subprocess.run(
            ["psql", "-h", OMNI_PG_HOST, "-U", OMNI_PG_USER, "-d", OMNI_PG_DB,
             "-q", "-v", "ON_ERROR_STOP=1",
             "-c", "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"],
            env=env, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "timed out — a connection is holding a lock"
    except (subprocess.SubprocessError, OSError) as e:
        return f"psql would not run ({type(e).__name__})"
    if p.returncode != 0:
        return (p.stderr or "psql failed").strip()[:160]
    return ""


def _stop_neo4j() -> None:
    """Neo4j must not be running while its store is deleted — a live store
    manager rewrites the files it is told to forget, and the restart then finds
    a half-deleted database it refuses to open."""
    binary = os.path.join(NEO4J_HOME, "bin", "neo4j")
    if not os.path.exists(binary):
        return
    try:
        subprocess.run([binary, "stop"], capture_output=True, timeout=120,
                       env=dict(os.environ, NEO4J_HOME=NEO4J_HOME))
    except (subprocess.SubprocessError, OSError):
        pass


def _restart_workers() -> list:
    """Kill the worker processes so boot.py's watchdog restarts them clean.

    ui_server is deliberately NOT killed: this code is running inside it, and
    taking it down would drop the response that reports the result. Everything
    else holds file handles, DB connections and loaded models that refer to
    things that no longer exist, so a restart is the only honest way back.
    """
    restarted = []
    for name in ("frame_worker.py", "model_manager.py", "omni_engine.py"):
        try:
            p = subprocess.run(["pkill", "-f", name], capture_output=True,
                               timeout=20)
            if p.returncode == 0:
                restarted.append(name)
        except (subprocess.SubprocessError, OSError):
            pass
    return restarted


def _run(scope: list, restart: bool) -> None:
    """Body of the reset thread."""
    chosen = [t for t in _targets() if t["id"] in scope]
    results = []
    freed = 0
    try:
        _set(state="running", stage="Pausing the system", pct=2,
             detail="stopping every consumer before deleting what it reads",
             started_at=time.time(), finished_at=None, error=None,
             results=[], freed_mb=0, log=[])

        pause_all(reason="factory reset")
        # A worker mid-chunk keeps its file handles for a few seconds. This is
        # not a lock — it just avoids the common case of deleting a frame
        # directory the CV engine is writing into at that exact moment.
        time.sleep(3)

        if "graph" in scope:
            _set(stage="Stopping Neo4j", pct=6, detail="before its store is removed")
            _stop_neo4j()

        for n, t in enumerate(chosen, 1):
            pct = 8 + int(80 * (n - 1) / max(len(chosen), 1))
            _set(stage=f"Clearing {t['label']}", pct=pct, detail="measuring")

            before = (_pg_size() if t["id"] == "postgres" else
                      _redis_size() if t["id"] == "queues" else
                      sum(_du(p) for p in t["paths"]))
            note = ""

            try:
                if t["id"] == "postgres":
                    note = _wipe_postgres()
                elif t["id"] == "queues":
                    r = _redis()
                    if r is None:
                        note = "Redis unavailable"
                    else:
                        r.flushall()
                        # flushall took the pause flags with it. The reset is
                        # not over, so put them back immediately.
                        pause_all(reason="factory reset")
                else:
                    for p in t["paths"]:
                        # Directories config creates at import are emptied, not
                        # removed: the next write must not hit ENOENT.
                        if p in (VIDEO_DIR, THUMB_DIR, ARCHIVE_DIR, LAKE_DIR):
                            _empty_dir(p)
                        else:
                            _rm(p)
                        _set(detail=f"{os.path.basename(p) or p} cleared")
                    for p in t["paths"]:
                        if p in (VIDEO_DIR, THUMB_DIR, ARCHIVE_DIR):
                            os.makedirs(p, exist_ok=True)
            except Exception as e:
                note = f"{type(e).__name__}: {str(e)[:140]}"

            # lake.db is the one target whose owner does not get restarted:
            # this code runs inside ui_server, and init_db() is only called at
            # its __main__. Without this, the harvester's next INSERT hits
            # "no such table: posts" and keeps hitting it until the whole
            # container is rebooted. Everything else self-heals — frame_worker,
            # model_manager and omni_engine recreate their own tables (and
            # Postgres/Neo4j) when the watchdog brings them back.
            if t["id"] == "sqlite" and not note:
                try:
                    from lake_schema import ensure_lake_schema
                    schema = ensure_lake_schema()
                    _set(detail="lake.db schema recreated — empty, not missing")
                    if schema["skipped"]:
                        note = "; ".join(schema["skipped"])[:140]
                except Exception as e:
                    note = (f"schema NOT recreated ({type(e).__name__}) — "
                            f"restart the container")

            after = (_pg_size() if t["id"] == "postgres" else
                     _redis_size() if t["id"] == "queues" else
                     sum(_du(p) for p in t["paths"]))
            gained = max(0, before - after)
            freed += gained
            results.append({
                "id": t["id"], "label": t["label"],
                "before_mb": round(before / 1048576, 1),
                "after_mb": round(after / 1048576, 1),
                "freed_mb": round(gained / 1048576, 1),
                "ok": not note, "note": note,
            })
            _set(results=list(results), freed_mb=round(freed / 1048576, 1))
            vios_log(f"reset · {t['label']}: freed {gained / 1048576:.0f} MB"
                     + (f" ({note})" if note else ""), "ADMIN",
                     "WARN" if note else "INFO")

        restarted = []
        if restart:
            _set(stage="Restarting workers", pct=92,
                 detail="they hold handles to files that no longer exist")
            restarted = _restart_workers()

        _set(stage="Resuming the system", pct=97,
             detail="the workers come back to an empty machine")
        resume_all()

        _set(state="done", stage="Reset complete", pct=100,
             finished_at=time.time(),
             detail=f"{round(freed / 1048576, 1)} MB freed across "
                    f"{len(results)} target(s)"
                    + (f" · restarted {', '.join(restarted)}" if restarted
                       else ""),
             results=results, freed_mb=round(freed / 1048576, 1))
        vios_log(f"FACTORY RESET complete — {freed / 1048576:.0f} MB freed",
                 "ADMIN", "SUCCESS")

    except Exception as e:
        # Leaving the system paused after a failed reset would be a second
        # outage on top of the first. Resume, then report.
        resume_all()
        _set(state="error", stage="Failed", detail=str(e)[:260],
             error=str(e)[:300], finished_at=time.time(), results=results)
        vios_log(f"factory reset failed: {e}", "ADMIN", "ERROR")


def start_reset(scope: list, confirm: str, restart: bool = True) -> dict:
    """Validate the request, then run it on a thread. Poll reset_status().

    The typed phrase is checked here rather than in the route so it cannot be
    bypassed by calling the module directly — this is the one operation in VIOS
    with no undo on the machine it runs on.
    """
    if (confirm or "").strip().upper() != CONFIRM_PHRASE:
        return {"ok": False,
                "error": f"Type {CONFIRM_PHRASE} exactly to confirm."}
    valid = {t["id"] for t in _targets()}
    scope = [s for s in (scope or []) if s in valid]
    if not scope:
        return {"ok": False, "error": "Nothing selected to delete."}
    if is_resetting():
        return {"ok": False, "error": "A reset is already running."}
    for mod in ("db_export", "db_restore"):
        try:
            m = __import__(mod)
            if m.is_running():
                return {"ok": False,
                        "error": f"{mod.split('_')[1].title()} is running — "
                                 f"wait for it to finish."}
        except Exception:
            pass

    vios_log(f"FACTORY RESET requested: {', '.join(scope)}", "ADMIN", "WARN")
    t = threading.Thread(target=_run, args=(scope, restart),
                         name="vios-factory-reset", daemon=True)
    t.start()
    return {"ok": True, "scope": scope}
