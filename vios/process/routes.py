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

# The v1 plane's liveness, supplied by whoever mounted this router.
#
# Registered rather than imported: `ui_server` imports this module, so importing
# it back would be a cycle, and the v2 plane is meant to run without it — Atlas
# on a laptop mounts these routes with no Redis, no Postgres and no ghost
# worker. A missing probe means "there is no v1 plane here", which is a true
# statement and not an error.
_v1_probe = None


def set_v1_probe(fn) -> None:
    """Tell the process tab how to ask the v1 plane what it is doing.

    `fn()` returns a dict — status line, queue depths, whether anything is
    moving. This is the fix for the tab announcing "Nothing running" while the
    frame worker is visibly grinding: the two planes were never idle at the same
    time, they simply had no shared surface to say so on.
    """
    global _v1_probe
    _v1_probe = fn


def _v1_plane() -> dict:
    if _v1_probe is None:
        return {"present": False}
    try:
        out = _v1_probe()
        return {"present": True, **(out if isinstance(out, dict) else {})}
    except Exception as exc:                       # noqa: BLE001
        # A dead Redis must degrade this panel, not the whole status call the
        # tab polls once a second.
        return {"present": True, "error": f"{type(exc).__name__}: {exc}"[:200]}


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
# Autostart
# ═══════════════════════════════════════════════════════════════════════
# The v2 engine had no starter. Every "Idle / Nothing running" the tab has ever
# shown was true — a Kaggle session booted, mounted these routes, and then sat
# there until somebody opened the page and pressed Start. On a twelve-hour
# session begun and then left alone, that is twelve hours of nothing.
_autostart = {"state": "off", "step": "", "message": "", "error": "",
              "at": 0.0, "started": False}


def autostart_state() -> dict:
    return dict(_autostart)


def autostart(delay: float = 0.0) -> dict:
    """Bring the processing plane up by itself.

    Runs on a thread and reports through the same `_task` slot the manual
    buttons use, which is the point of routing it here rather than doing it in
    `ui_server`: a restore kicked off by boot and a restore kicked off by a
    person must not run at the same time, and the only way to guarantee that is
    for both to ask the same lock.

    `VIOS_PROCESS_AUTOSTART=0` opts out entirely — for a session opened to
    inspect a database rather than to add to it, where a sweep starting on its
    own would be an unwelcome surprise on someone else's GPU.
    """
    if str(os.environ.get("VIOS_PROCESS_AUTOSTART", "1")).strip().lower() in (
            "0", "false", "no", "off"):
        _autostart.update({"state": "off", "at": time.time(),
                           "message": "VIOS_PROCESS_AUTOSTART=0"})
        return dict(_autostart)
    if _autostart["started"]:
        return dict(_autostart)
    _autostart.update({"started": True, "state": "waiting", "at": time.time(),
                       "message": "Waiting for the server to settle…"})

    def _boot():
        if delay > 0:
            time.sleep(delay)
        try:
            _autostart_steps()
        except Exception as exc:                   # noqa: BLE001
            _autostart.update({
                "state": "error",
                "error": f"{type(exc).__name__}: {exc}"[:400],
                "message": "Autostart failed — the tab still works, press "
                           "Start when the reason is fixed."})

    threading.Thread(target=_boot, name="vios-proc-autostart",
                     daemon=True).start()
    return dict(_autostart)


def _autostart_steps() -> None:
    """Restore first, then reconcile, then sweep — in that order and no other.

    Starting the engine before the restore finishes would have it claim work
    that the channel already holds the answer to, which is precisely the
    "reprocesses everything after a session dies" complaint. So the sweep waits
    for the evidence to land, even though waiting is the slower-looking choice.
    """
    eng = get_engine()

    _autostart.update({"state": "restoring", "step": "restore",
                       "message": "Restoring what is already known…"})
    res = _run_task("autostart-restore", lambda say: _startup_restore(eng, say))
    # `_run_task` returns immediately; wait for the slot it took, so the sweep
    # starts after the restore rather than beside it.
    if getattr(res, "status_code", 200) == 409:
        _autostart["message"] = ("Another job was already running — waiting "
                                 "for it")
    deadline = time.time() + _RESTORE_WAIT_SECONDS
    while _task["running"] and time.time() < deadline:
        _autostart["message"] = _task.get("message") or "Restoring…"
        time.sleep(1.0)
    if _task.get("error"):
        # A restore that failed is not a reason to refuse to process. The
        # coverage table is the authority on what is done; a failed restore
        # means it may be thinner than it should be, which costs repeated work
        # and nothing else.
        _autostart["error"] = str(_task["error"])[:400]

    _autostart.update({"state": "starting", "step": "start",
                       "message": "Starting the sweep…"})
    out = eng.start()
    if out.get("ok"):
        _autostart.update({"state": "running", "step": "",
                           "message": "Processing started automatically.",
                           "at": time.time()})
    else:
        _autostart.update({"state": "blocked", "step": "",
                           "message": str(out.get("error") or
                                          "The engine refused to start.")[:400],
                           "at": time.time()})


def _startup_restore(eng, say):
    """What a booting session must do before it processes anything.

    Four steps, in this order and no other, because each one's input is the
    previous one's output:

    1. **Wait for the services.** `boot.py:251` explains why the restore cannot
       live there — Postgres is not up before ignition, and half a restore is
       worse than none. That reasoning is right, so the action moved and the
       check did not: this runs from the web process, where the services are
       already live, and still waits for them rather than assuming.
    1b. **Wait for the credentials, but only if they are still coming.** Both
       restores read the channel, and this whole phase is a one-shot: burning it
       ninety seconds before a throttled Kaggle Secrets retry succeeds is how a
       session ends up with an empty coverage table and reprocesses an archive it
       already has bundles for. The wait is conditional on `creds.recoverable` —
       a session with nothing stored waits zero seconds, because nothing is on
       its way.
    2. **The v1 harvest database**, if it is empty. `db_restore.start_restore`
       is the same call the admin panel makes; it refuses to run beside an
       export and knows both schema versions. Only when `posts == 0` — a
       populated lake means this session already has its catalogue, and
       overwriting it from a channel bundle would be a destructive act nobody
       asked for.
    3. **The v2 evidence shards**, through the engine's own channel. Every
       claim, vector, per-frame embedding and per-frame metric any worker has
       ever pushed, replayed by uid, order-independent and idempotent.
    4. **The reconcile.** Restoring the evidence and leaving the work table
       queued is the failure this whole phase exists to fix — the rows come
       back and the sweep processes the archive again anyway. This is the step
       that makes "nothing is ever reprocessed" true rather than hopeful.

    Every step reports through `say`, which lands in `/api/process/status.task`
    and therefore on the tab. Nothing here raises: a boot-time restore that
    fails must cost repeated work, never the session.
    """
    out = {"waited": 0.0, "creds_waited": 0.0, "lake": {}, "shards": {},
           "reconciled": {}}

    say("Waiting for the services to come up…")
    out["waited"] = _wait_for_services(say)

    out["creds_waited"] = _wait_for_credentials(say)

    say("Checking the harvest database…")
    try:
        out["lake"] = _restore_lake(say)
    except Exception as exc:                       # noqa: BLE001
        out["lake"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        say(f"Harvest restore skipped — {out['lake']['error']}")

    say("Scanning the channel for evidence shards…")
    try:
        out["shards"] = _restore_evidence(eng, say)
    except Exception as exc:                       # noqa: BLE001
        out["shards"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        say(f"Shard restore failed — {out['shards']['error']}")

    # Runs even when the shard restore failed. The evidence this container
    # already holds — from a bundle loaded by hand, from a previous run of this
    # same session — is just as much a reason not to redo the work.
    say("Matching restored evidence against the work table…")
    try:
        # The work table is empty on a fresh container, and a reconcile against
        # no rows would report "0 already done" and be believed. `plan` is
        # idempotent and is what the sweep does at the top of every rotation
        # anyway; doing it here is what makes the number on the tab real.
        eng.coverage.plan(eng.selected)
        rec = eng.reconcile_now()
        out["reconciled"] = rec
        say(f"{rec.get('rows', 0)} passes across {rec.get('videos', 0)} videos "
            f"were already done — they will not run again.")
    except Exception as exc:                       # noqa: BLE001
        out["reconciled"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        say(f"Reconcile failed — {out['reconciled']['error']}")

    return out


def _wait_for_credentials(say, timeout: float = 330.0) -> float:
    """Hold the restore while a recoverable credential failure is still healing.

    Returns the seconds waited. Zero is the normal answer.

    The restore is the one step here that cannot be retried cheaply: it is a
    one-shot per session (`_autostart["started"]`), it reads the channel, and if
    it runs without Telegram it reports "not configured" and the sweep then
    reprocesses an archive whose evidence was sitting in a bundle the whole time.
    Meanwhile `ui_server` may be four minutes into re-asking a Kaggle Secrets
    store that answered HTTP 429 during Phase 0 — a limit that clears in about a
    minute, in a session that lasts twelve hours.

    So this waits, and the important part is when it does *not*:

      * credentials already present → returns immediately;
      * the sweep only ever failed to find rows, or this is not Kaggle at all →
        returns immediately, because nothing is coming and the sweep should get
        on with the work it does not need Telegram for;
      * only a `recoverable` reason (throttled / unreachable / backend) buys the
        wait, and only up to `timeout` — 330 s, which covers the first three
        rungs of `creds._RECOVER_LADDER`.
    """
    try:
        import config                              # noqa: PLC0415
        from vios import creds                     # noqa: PLC0415
    except Exception:                              # noqa: BLE001
        return 0.0
    if getattr(config, "telegram_ready", None) and config.telegram_ready():
        return 0.0
    try:
        if not creds.recoverable(creds.kaggle_report()):
            return 0.0
    except Exception:                              # noqa: BLE001
        return 0.0

    say("Telegram credentials are still being re-read after a temporary Kaggle "
        "Secrets failure — holding the restore rather than burning it.")
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(5.0)
        if config.telegram_ready():
            waited = time.time() - t0
            say(f"Credentials arrived after {int(waited)}s — restoring.")
            return waited
        left = int(timeout - (time.time() - t0))
        if left % 60 < 5:
            say(f"Still waiting for Telegram credentials ({left}s left before "
                f"the restore goes ahead without them)…")
    say("Credentials never arrived — restoring what can be restored without "
        "the channel. Type them on the Setup page and press Restore.")
    return time.time() - t0


def _wait_for_services(say, timeout: float = 180.0) -> float:
    """Block until Postgres answers, or until it is clear it never will.

    Bounded, and a timeout is not an error: `OMNI_ENABLED=0` is a legitimate
    configuration, and Atlas on a laptop has no Postgres at all. Both mean "go
    ahead without it", which is a true answer and not a failure to report.
    """
    t0 = time.time()
    try:
        import config                              # noqa: PLC0415
        if not getattr(config, "OMNI_ENABLED", True):
            return 0.0
    except Exception:                              # noqa: BLE001
        return 0.0
    try:
        import omni_db                             # noqa: PLC0415
    except Exception:                              # noqa: BLE001
        return 0.0

    last = ""
    while time.time() - t0 < timeout:
        try:
            conn = omni_db.get_pg_conn()
            if conn is not None:
                try:
                    conn.close()
                except Exception:                  # noqa: BLE001
                    pass
                return round(time.time() - t0, 1)
        except Exception as exc:                   # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"[:120]
        say(f"Waiting for Postgres ({int(time.time() - t0)}s)"
            + (f" — {last}" if last else "") + "…")
        time.sleep(3.0)
    say("Postgres did not come up in time — continuing without it.")
    return round(time.time() - t0, 1)


def _restore_lake(say) -> dict:
    """Pull the harvest database back, but only into an empty one.

    A container that already holds a populated `posts` table has its
    catalogue — restoring over it would replace live rows with a channel
    snapshot that may be hours older, which is a destructive act and not one
    a boot sequence should take on its own.
    """
    try:
        import config                              # noqa: PLC0415
        import db_restore                          # noqa: PLC0415
    except Exception as exc:                       # noqa: BLE001
        return {"skipped": f"unavailable: {type(exc).__name__}: {exc}"[:200]}

    posts = -1
    try:
        import sqlite3                             # noqa: PLC0415
        conn = sqlite3.connect(config.DB_PATH, timeout=10.0)
        try:
            posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        finally:
            conn.close()
    except Exception:                              # noqa: BLE001
        posts = 0        # no file, or no table — the fresh-container case

    if posts > 0:
        return {"skipped": f"{posts} posts already here", "posts": posts}
    if db_restore.is_running():
        return {"skipped": "a restore was already running"}

    say("Harvest database is empty — restoring it from the channel…")
    res = db_restore.start_restore("apply")
    if not res.get("ok"):
        return {"skipped": str(res.get("error") or "refused")[:200]}

    # `start_restore` returns immediately. Followed here rather than left to
    # run behind us because step 3 replays shards into a database this restore
    # may be in the middle of replacing.
    deadline = time.time() + _LAKE_WAIT_SECONDS
    while time.time() < deadline:
        st = db_restore.restore_status()
        if st.get("state") != "running":
            return {"posts_before": posts, "state": st.get("state"),
                    "detail": str(st.get("detail") or "")[:200],
                    "error": str(st.get("error") or "")[:200]}
        say(f"Harvest restore — {st.get('stage') or 'working'}"
            + (f" · {st['detail']}" if st.get("detail") else ""))
        time.sleep(2.0)
    return {"posts_before": posts, "state": "timeout",
            "detail": f"still running after {_LAKE_WAIT_SECONDS}s"}


def _restore_evidence(eng, say) -> dict:
    """Replay every evidence shard the channel holds that this database lacks.

    Opens the channel if the engine has not, and closes it again if it opened
    it and nothing else needs it — a boot restore should not leave an MTProto
    session held open on a container that is about to sit idle.
    """
    if not (eng._tg and eng._tg.token):
        return {"skipped": "no bot token — nothing to restore from"}
    opened = False
    if not (eng._channel and eng._channel.ready):
        eng._open_channel()
        opened = True
    try:
        if not (eng._channel and eng._channel.ready):
            why = getattr(eng._channel, "reason", "") if eng._channel else ""
            return {"skipped": f"no MTProto session{' — ' + why if why else ''}"}
        got = intake_restore(eng, say)
        # Tell the sweep this walk already happened. Without it `_sweep` repeats
        # the identical scan over thousands of messages seconds later, downloads
        # nothing, and costs the same minutes twice on every boot.
        eng._shards_at = time.time()
        return got
    finally:
        if opened and eng.state == "idle" and eng._channel:
            try:
                eng._channel.stop()
            except Exception:                      # noqa: BLE001
                pass
            eng._channel = None


# A harvest restore downloads a multi-part bundle over HTTP. Fifteen minutes is
# past any observed run and short enough that a wedged one does not cost the
# session: the sweep starts anyway, and the coverage table decides what is left.
_LAKE_WAIT_SECONDS = 15 * 60

# How long the autostart waits for the whole restore before starting the sweep
# anyway. It has to exceed the sum of its parts — three minutes for the
# services, fifteen for a harvest bundle, and a channel scan over thousands of
# messages — or the sweep would start on top of a restore still writing to the
# same database. Past it, the coverage table, not the restore, decides what
# still needs doing: nothing is lost, some work is repeated.
_RESTORE_WAIT_SECONDS = 45 * 60


@process_router.get("/api/process/autostart")
def autostart_status():
    """Whether this session started itself, and how far it got."""
    return JSONResponse({"ok": True, "autostart": autostart_state()})


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
    """The one call the tab polls. Cached two seconds inside the engine.

    `planes` is added here rather than in the engine because the engine knows
    nothing about the v1 plane and should not: it is one process's two halves,
    joined at the HTTP surface they share and nowhere else.
    """
    try:
        eng = get_engine()
        s = eng.status(fresh=bool(fresh))
        s["task"] = {k: v for k, v in _task.items() if k != "result"}
        s["planes"] = {"v1": _v1_plane(),
                       "v2": {"state": s.get("state", "idle"),
                              "message": s.get("message", ""),
                              "autostart": autostart_state()}}
        try:
            s["errors"] = eng.coverage.recent_errors(40)
        except Exception as exc:                   # noqa: BLE001
            s["errors"] = []
            s["errors_error"] = f"{type(exc).__name__}: {exc}"[:200]
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


@process_router.get("/api/process/stages")
def stages():
    """Per-stage coverage, errors and last channel bundle.

    The panel that answers "which stage went wrong", which is the question
    actually asked when a twelve-hour sweep produces a database that looks thin.
    A component matrix with thirty-four rows does not answer it at a glance.
    """
    try:
        return _ok(stages=get_engine().stage_status())
    except Exception as exc:
        return _err(exc, 500)


@process_router.post("/api/process/publish_stage")
def publish_stage(stage: str = Form("")):
    """Cut a stage bundle by hand. Empty `stage` cuts every planned stage.

    Worth pressing before closing a Kaggle session early: the bundle is a whole
    database, so it restores without replaying anything.
    """
    try:
        return _ok(**get_engine().publish_stage_now(stage.strip()))
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


@process_router.post("/api/process/reconcile")
def reconcile():
    """Match evidence this database already holds against the work table.

    The restore's other half, exposed on its own because the two fail
    independently. A bundle loaded by hand, a shard pushed by another worker, a
    revision reverted on purpose — each puts evidence here without touching
    coverage, and each leaves the sweep about to re-earn with the GPU what is
    already on disk. Pressing this is how an operator asks "how much of the
    archive is genuinely still to do?" and gets a number rather than a guess.

    Idempotent and safe while processing: `Coverage.reconcile` will not touch a
    `running` row, so it cannot steal a lease from the sweep this may be racing.
    """
    def _job(say):
        eng = get_engine()
        say("Planning the work table…")
        # Without this the reconcile has nothing to update on a fresh container
        # and would honestly report zero. `plan` is idempotent — it is what the
        # sweep does at the top of every rotation.
        planned = eng.coverage.plan(eng.selected)
        say("Matching evidence against the work table…")
        out = eng.reconcile_now()
        out["planned"] = planned
        return out
    return _run_task("reconcile", _job)


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
