"""
vios.capture.backfill — asset sets for the videos that were captured first.

`publish_assets` runs inside a capture, in the one window where the bytes are on
disk and the anchor message id exists. That is the right place for it and it is
also the reason the archive is split in two: every video captured before Phase J
has a message in the channel and nothing beside it, so Atlas can only reach
those through a media session and a byte-range seek — the slow path the clips
exist to replace. Sixty-two videos on this channel are in that state, and no
amount of running the capture engine fixes them, because they are already
`uploaded` and will never be claimed again.

This module is the second window. It works from the ledger, which is the record
of what is in the channel, and for each video with no asset set it obtains the
bytes, cuts the clips, uploads them under the video's own message, and writes
the manifest id back. Three properties, each bought deliberately:

  Resumable.
      The ledger row is the state. A run that dies at video 40 resumes at 41
      because `needs_assets()` no longer returns the first 39. Nothing is held
      in memory that a restart would need.

  Paced.
      Fifteen documents per video against a channel with a per-minute budget is
      the shape that trips a 429, and a 429 during backfill would slow the
      capture engine sharing the same bot token. So there is a pause between
      clips (`assets.CHUNK_PAUSE`) and a longer one between videos
      (`VIOS_BACKFILL_PAUSE`).

  Cheap on bytes.
      The mp4 is already in Telegram, so getting it back is a download rather
      than an Instagram request — no cookies, no rate ladder, no account risk.
      Local disk is checked first, the Bot API second (it carries anything under
      20 MB, which is nearly every reel), and MTProto last, for the ones the Bot
      API refuses.

What it deliberately does not do: re-fetch anything from Instagram, touch a row
that is not `uploaded`, or fail a video for anything short of "the bytes cannot
be obtained". An asset set is an optimisation over data that is already safe.

One thing it must do before any of that, and did not: learn what the channel
holds. The ledger is a cache, it lives in scratch, and on a fresh session it is
empty while the channel holds every video this module exists for — so a pass that
counts before it scans concludes, correctly and uselessly, that there is nothing
to do. `_cycle` therefore seeds first and counts second, and repeats on an
interval so a video harvested at hour three is covered in the same session
instead of the next one.
"""

from __future__ import annotations

import os
import shutil
import threading
import time

from . import assets as assets_mod
from .ledger import Ledger

# Between videos. Long enough that fifteen documents per video averages out well
# under Telegram's channel budget, short enough that 62 videos is under an hour.
VIDEO_PAUSE = float(os.environ.get("VIOS_BACKFILL_PAUSE", "4.0"))

# The Bot API's download ceiling. Above it, only MTProto can fetch the file.
BOT_DOWNLOAD_LIMIT = 20 * 1024 * 1024

# How long after a pass drains before looking again. The arm is one-shot
# otherwise, so a video harvested at hour three would wait for the next session.
# `0` keeps the old one-shot behaviour.
REARM_EVERY = float(os.environ.get("VIOS_BACKFILL_EVERY", "1800"))

# How many times one video is attempted per session. The re-arm looks again every
# half hour and the ledger row of a failed video is indistinguishable from a new
# one, so without this a video whose bytes cannot be obtained — or every video, if
# ffmpeg is missing — would be re-downloaded twenty-four times over twelve hours
# to fail identically each time. Two attempts covers the transient case (an
# MTProto session refused on the first pass); a third is the next session's job,
# or the operator's button.
SESSION_ATTEMPTS = max(int(os.environ.get("VIOS_BACKFILL_TRIES", "2") or 2), 1)


def log(msg: str) -> None:
    """Say it on the console.

    Nothing under `vios/` printed anything, so the backfill's only voice was the
    `_AUTO` dict on the capture setup page. On the console the operator saw
    `asset-set backfill armed` at boot and then silence — which reads exactly
    like a feature that is broken, whether it is working, idle or switched off.

    The ascii retry is not decoration: Kaggle's console is not always utf-8, and
    a UnicodeEncodeError raised out of a log line would take the caller with it.
    """
    try:
        print(f"🎞️ [CAPTURE] {msg}", flush=True)
    except Exception:                                  # noqa: BLE001
        try:
            print(f"[CAPTURE] {msg}".encode("ascii", "replace").decode("ascii"),
                  flush=True)
        except Exception:                              # noqa: BLE001
            pass


class Backfill:
    """One per process. Start it, watch it, stop it. Never raises to the caller.

    Deliberately independent of the capture engine's own thread and of the
    routes' single task slot: a backfill runs for the better part of an hour and
    must not block a channel scan or a link import for that long. The two share
    the ledger, and every write here is a single UPDATE on one row.
    """

    def __init__(self):
        self.state = "idle"            # idle | running | stopping | error
        self.message = "Not started."
        self.error = ""
        self.started_at: float = 0.0
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.clips = 0
        self.uploads = 0
        self.total = 0
        self.current: dict = {}
        self.notes: list = []          # the last few per-video notes, newest last
        self._tries: dict = {}         # key → attempts this session, across passes
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()

    # ── control ──────────────────────────────────────────────────────────
    def start(self, engine, limit: int = 0) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"ok": False, "state": self.state,
                        "message": "A backfill is already running."}
            if not engine.telegram:
                return {"ok": False, "state": self.state,
                        "message": "Set the bot token and channel id first."}
            self._stop.clear()
            self.state = "running"
            self.error = ""
            self.message = "Starting…"
            self.started_at = time.time()
            self.done = self.failed = self.skipped = 0
            self.clips = self.uploads = 0
            self.notes = []
            self._thread = threading.Thread(
                target=self._run, args=(engine, int(limit or 0)),
                name="vios-backfill", daemon=True)
            self._thread.start()
        return {"ok": True, "state": self.state, "message": "Started."}

    def stop(self, wait: float = 5.0) -> dict:
        """Ask for a stop after the video in flight. Never mid-upload.

        Interrupting between clips would leave a video whose clips are in the
        channel and whose manifest is not — invisible to Atlas and invisible to
        the next run, because the row still says it has no asset set and the
        retry would upload the clips a second time. Finishing the current video
        costs seconds and avoids that entirely.
        """
        self._stop.set()
        if self.state == "running":
            self.state = "stopping"
            self.message = "Stopping after the current video…"
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=wait)
        return {"ok": True, "state": self.state}

    def status(self) -> dict:
        return {
            "state": self.state, "message": self.message, "error": self.error,
            "started_at": self.started_at or None,
            "done": self.done, "failed": self.failed, "skipped": self.skipped,
            "clips": self.clips, "uploads": self.uploads, "total": self.total,
            "current": dict(self.current), "notes": list(self.notes[-12:]),
            "video_pause": VIDEO_PAUSE,
        }

    def outstanding(self, engine, limit: int = 0) -> list:
        """The rows a run started right now would actually attempt.

        `needs_assets()` cannot tell a video that has never been tried from one
        that failed four minutes ago — the row is identical either way, because a
        failure leaves the asset columns untouched by design. This is the
        difference, and it is what makes the re-arm loop safe: a caller asks
        before starting, and a pass with nothing new to do does not run at all
        rather than running to fail the same sixty-two videos again.
        """
        rows = [r for r in engine.ledger.needs_assets(100000)
                if self._tries.get(str(r["key"]), 0) < SESSION_ATTEMPTS]
        return rows[:limit] if limit > 0 else rows

    # ── the loop ─────────────────────────────────────────────────────────
    def _run(self, engine, limit: int) -> None:
        led: Ledger = engine.ledger
        tg = engine.telegram
        channel = None
        try:
            rows = self.outstanding(engine, int(limit or 0))
            self.total = len(rows)
            if not rows:
                left = led.asset_counts()["without_assets"]
                self.message = (
                    f"{left} video(s) still have no asset set, and each has "
                    f"already been tried {SESSION_ATTEMPTS}× this session — the "
                    f"next session retries them." if left else
                    "Every captured video already has an asset set — nothing "
                    "to backfill.")
                self.state = "idle"
                log(self.message)
                return
            self.message = f"{len(rows)} video(s) without an asset set."
            # Said once, up front: without ffmpeg every single video fails the
            # same way, and the operator should learn that on line one rather
            # than from sixty-two identical notes on a page they are not reading.
            have_ffmpeg = bool(shutil.which(assets_mod.FFMPEG) or
                               os.path.exists(assets_mod.FFMPEG))
            log(f"{len(rows)} video(s) without an asset set · ffmpeg: "
                + (assets_mod.FFMPEG if have_ffmpeg else
                   f"NOT FOUND ({assets_mod.FFMPEG!r}) — no clips can be cut"))

            work_root = os.path.join(engine.scratch, "backfill")
            os.makedirs(work_root, exist_ok=True)

            for n, row in enumerate(rows, 1):
                if self._stop.is_set():
                    break
                key = str(row["key"])
                # Counted before the attempt, not after it: an attempt that dies
                # in a way this loop does not catch still used the bytes and the
                # rate limit, and must not be free to repeat every half hour.
                self._tries[key] = self._tries.get(key, 0) + 1
                self.current = {"key": key, "n": n, "of": len(rows),
                                "phase": "fetching", "started": time.time()}
                self.message = f"{key} ({n}/{len(rows)})"
                work = os.path.join(work_root, key)
                try:
                    channel = self._one(engine, led, tg, channel, row, work)
                except Exception as exc:
                    self.failed += 1
                    self._note(key, f"{type(exc).__name__}: {str(exc)[:200]}")
                    log(f"{key} ({n}/{len(rows)}) — failed: "
                        f"{type(exc).__name__}: {str(exc)[:160]}")
                    led.log("backfill-failed",
                            f"{type(exc).__name__}: {str(exc)[:300]}", key)
                finally:
                    shutil.rmtree(work, ignore_errors=True)
                    self.current = {}
                if self._stop.is_set():
                    break
                if VIDEO_PAUSE:
                    # Sleep in slices so a stop is felt immediately rather than
                    # four seconds later, sixty-two times over.
                    end = time.time() + VIDEO_PAUSE
                    while time.time() < end and not self._stop.is_set():
                        time.sleep(0.25)

            led.checkpoint()
            self.message = (
                f"{self.done} asset set(s) built, {self.clips} clip(s) "
                f"uploaded, {self.failed} video(s) could not be built"
                + (" — stopped early." if self._stop.is_set() else "."))
            led.log("backfill", self.message)
            log(self.message)
            if self.done:
                self._tell_atlas()
            self.state = "idle"
        except Exception as exc:
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            self.message = self.error
            log(f"backfill stopped — {self.error}")
        finally:
            self.current = {}
            if channel is not None:
                try:
                    channel.stop()
                except Exception:
                    pass
            if self.state == "stopping":
                self.state = "idle"

    def _one(self, engine, led: Ledger, tg, channel, row: dict, work: str):
        """Build and upload one video's asset set. Returns the MTProto session.

        The session is threaded through rather than opened per video because the
        handshake is the expensive part — but it is only opened at all if some
        video actually needs it, which on an archive of reels is a minority.
        """
        key = str(row["key"])
        os.makedirs(work, exist_ok=True)
        video = os.path.join(work, f"{key}.mp4")

        channel, how = self._obtain(engine, tg, channel, row, video)
        if not os.path.exists(video) or os.path.getsize(video) < 4096:
            self.skipped += 1
            self._note(key, f"bytes unavailable ({how})")
            log(f"{key} ({self.current.get('n')}/{self.current.get('of')}) — "
                f"skipped: bytes unavailable ({how})")
            led.mark_assets(key, 0, note=f"backfill: bytes unavailable ({how})")
            return channel

        self.current["phase"] = "cutting"
        # A synthetic capture result, so this reuses `publish_assets` exactly as
        # the capture loop calls it rather than reimplementing the upload order,
        # the manifest shape or the failure handling. The record here is the
        # ledger row — which is all we know about a video captured months ago —
        # merged with the Instagram export slice if one is configured.
        result = {
            "video": video,
            "bytes": os.path.getsize(video),
            "sha256": row.get("sha256") or "",
            "record": {
                "source": "backfill",
                "post": {"uploader": row.get("uploader") or "",
                         "title": row.get("title") or "",
                         "duration": row.get("duration"),
                         "width": row.get("width"),
                         "height": row.get("height"),
                         "taken_at": row.get("taken_at")},
                "url": row.get("url") or "",
                "key": key,
                "kind": row.get("kind") or "",
                "collections": engine._collections(led, key),
            },
        }
        sent = {"msg_id": int(row.get("msg_id") or 0),
                "record_msg_id": int(row.get("record_msg_id") or 0),
                "file_id": row.get("file_id") or "",
                "duration": row.get("duration"),
                "width": row.get("width"), "height": row.get("height")}

        self.current["phase"] = "uploading"
        got = assets_mod.publish_assets(
            tg, result, sent, key, work,
            ig_slice=engine._ig_slice(key),
            on_note=lambda m: self._note(key, m))

        self.clips += int(got.get("clips") or 0)
        self.uploads += int(got.get("uploaded") or 0)
        led.mark_assets(key, got.get("manifest_msg_id") or 0,
                        clips=got.get("clips") or 0,
                        note="; ".join(got.get("notes") or [])[:400])
        where = f"{self.current.get('n')}/{self.current.get('of')}"
        if got.get("manifest_msg_id"):
            self.done += 1
            led.log("assets", f"backfill · {got['clips']} clip(s), "
                              f"{got['uploaded']} message(s) · via {how}", key)
            log(f"{key} ({where}) — {got['clips']} clip(s), "
                f"{got['uploaded']} message(s) via {how}")
        else:
            self.failed += 1
            self._note(key, "manifest was not uploaded — will retry next run")
            log(f"{key} ({where}) — no manifest uploaded, will retry next run"
                + (f" · {'; '.join(got.get('notes') or [])[:200]}"
                   if got.get("notes") else ""))
        return channel

    # ── getting the bytes back ───────────────────────────────────────────
    def _obtain(self, engine, tg, channel, row: dict, dest: str):
        """Put the video at `dest`. Returns (channel, how) and never raises.

        Order is by cost. A file already on this machine is free; the Bot API is
        one HTTPS GET with the token we already hold; MTProto needs a login and
        a handshake and is the only thing that can fetch a file over 20 MB or a
        message whose `file_id` was never recorded.
        """
        key = str(row["key"])

        local = self._find_local(engine, key)
        if local:
            try:
                shutil.copyfile(local, dest)
                return channel, "local disk"
            except OSError:
                pass

        size = int(row.get("file_size") or 0)
        file_id = row.get("file_id") or ""
        if file_id and (not size or size <= BOT_DOWNLOAD_LIMIT):
            try:
                if tg.download(file_id, dest):
                    return channel, "bot api"
            except Exception as exc:
                self._note(key, f"bot api download failed: "
                                f"{type(exc).__name__}: {str(exc)[:120]}")

        msg_id = int(row.get("msg_id") or 0)
        if msg_id:
            channel = self._session(engine, channel)
            if channel is not None and channel.ready:
                msgs = channel.messages([msg_id])
                msg = msgs.get(msg_id)
                if msg is not None and channel.download(msg, dest):
                    return channel, "mtproto"
                return channel, (channel.last_error or
                                 f"message {msg_id} did not download")
            return channel, (getattr(channel, "reason", "") or
                             "no MTProto session")
        return channel, "no message id and no local copy"

    @staticmethod
    def _find_local(engine, key: str) -> str:
        """The mp4 the capture or processing plane may have left behind.

        Both planes work in directories named after the key, so a session that
        captured or processed this video recently still has the file and the
        backfill is instant for it. The two name it differently — the processing
        plane always writes `source.mp4`, while the capture plane keeps whatever
        yt-dlp called it — so after the exact candidates there is one shallow
        scan of the key's own directory. Shallow, never recursive: a full walk of
        a mounted Kaggle input runs for minutes per video.
        """
        stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in key)[:64]
        roots = [engine.scratch,
                 os.path.join(os.path.dirname(engine.scratch), "process")]
        for root in roots:
            for cand in (os.path.join(root, stem, "source.mp4"),
                         os.path.join(root, stem, f"{stem}.mp4"),
                         os.path.join(root, f"{stem}.mp4")):
                try:
                    if os.path.exists(cand) and os.path.getsize(cand) > 4096:
                        return cand
                except OSError:
                    continue
            best, size = "", 0
            try:
                for entry in os.scandir(os.path.join(root, stem)):
                    if not entry.is_file():
                        continue
                    if not entry.name.lower().endswith(
                            (".mp4", ".mov", ".mkv", ".webm", ".m4v")):
                        continue
                    n = entry.stat().st_size
                    if n > size:
                        best, size = entry.path, n
            except OSError:
                pass
            if size > 4096:
                return best
        return ""

    def _session(self, engine, channel):
        """Open the shared MTProto session, once, and only if it is needed."""
        if channel is not None:
            return channel
        try:
            from vios.process.intake import Channel  # noqa: PLC0415
        except Exception as exc:
            self._note("", f"MTProto unavailable: {type(exc).__name__}")
            return None
        ch = Channel(engine.telegram, log=lambda m: self._note("", m))
        ch.start()
        if not ch.ready:
            self._note("", f"MTProto session refused: {ch.reason}")
        return ch

    def _note(self, key: str, message: str) -> None:
        self.notes.append({"at": time.time(), "key": key,
                           "text": str(message)[:300]})
        del self.notes[:-60]

    def _tell_atlas(self) -> None:
        """Ask Atlas to read the manifests this run just wrote.

        Atlas scans the channel once, in its own boot, and builds `parts` from
        the manifests it finds there. Every manifest written after that moment is
        invisible to it — so a backfill that finishes at hour two would leave the
        clips it cut unused until the next session, which is the whole benefit
        arriving a day late. The plane that created the fast path is the one that
        should say so.

        A note, never an exception: Atlas may not be importable in this process,
        may not have booted, or may already be scanning. None of those are
        reasons to fail a run whose real work is already durable in the channel.
        """
        try:
            from atlas.server import rescan  # noqa: PLC0415
        except Exception as exc:
            said = (f"Atlas not reachable from here ({type(exc).__name__}) — "
                    f"the clips will be picked up at its next boot scan")
            self._note("", said)
            log(said)
            return
        try:
            started = rescan(full=True)
        except Exception as exc:
            said = (f"Atlas rescan failed: {type(exc).__name__}: "
                    f"{str(exc)[:160]}")
            self._note("", said)
            log(said)
            return
        said = ("Atlas is re-reading the channel — the new clips become the "
                "fast path within a minute" if started else
                "Atlas is busy or has not been opened yet; it reads the new "
                "clips in its own scan")
        self._note("", said)
        log(said)


_BACKFILL: Backfill | None = None


def get_backfill() -> Backfill:
    global _BACKFILL
    if _BACKFILL is None:
        _BACKFILL = Backfill()
    return _BACKFILL


# ── starting without being asked ─────────────────────────────────────────
# The 62 videos with no asset set are the reason Atlas is slow on this archive,
# and a button nobody presses fixes nothing. So the session arms this itself,
# the same way the processing plane arms its own sweep, and it is a no-op on the
# common path: once every uploaded video has a manifest, `needs_assets()` returns
# an empty list and the thread exits without sending a single message.
_AUTO = {"state": "off", "message": "", "at": 0.0, "armed": False}

ATLAS_WAIT = float(os.environ.get("VIOS_BACKFILL_ATLAS_WAIT", "900"))


def _state(state: str, message: str = "") -> None:
    """Change the reported state — on the setup page and on the console at once.

    Every state this thread passes through goes through here, and every reason it
    gives up goes through here verbatim. The page was the wrong place for those
    to live alone: the operator watching a session start sees `asset-set backfill
    armed`, and if nothing follows it, there is no way to tell a run that turned
    itself off in its first second from one that is quietly working through
    sixty-two videos. That ambiguity is the whole reason this feature reads as
    broken rather than as off.
    """
    _AUTO.update({"state": state, "message": message, "at": time.time()})
    log(f"{state}{(' — ' + message) if message else ''}")


def _wait_for_atlas() -> None:
    """Hold until Atlas has finished booting, if it is booting at all.

    Atlas's boot reads the whole channel and imports every database bundle it
    finds, and it is the slowest thing in a session as well as the only one the
    operator is sitting in front of. A backfill starting underneath it competes
    for the same Bot API — the same rate limit, on the same chat — so the work
    whose entire purpose is to make Atlas fast would first make Atlas slow to
    open, which is exactly backwards.

    Three states, three answers. `starting` means nobody has opened Atlas, so
    there is nothing to wait for and waiting would postpone the backfill for the
    whole session. `ready` or `error` means the boot is over. Anything between is
    a boot in progress: wait, but on a clock, because a scan that never finishes
    must not silently cost the archive its clips.
    """
    try:
        from atlas.server import boot_phase  # noqa: PLC0415
    except Exception:
        return                              # no Atlas here; nothing to yield to
    try:
        if boot_phase() in ("starting", "ready", "error", ""):
            return
        _state("waiting", "Atlas is reading the channel — starting once it is "
                          "ready, so the two are not competing for the same "
                          "rate limit")
        end = time.time() + max(0.0, ATLAS_WAIT)
        while time.time() < end:
            time.sleep(2.0)
            if boot_phase() in ("ready", "error", ""):
                return
        _state("waiting", f"Atlas still booting after {int(ATLAS_WAIT)}s — "
                          f"starting anyway")
    except Exception:
        return


def autostart_state() -> dict:
    return dict(_AUTO)


def _cycle(eng, first: bool) -> None:
    """One seed → count → run, held until the run drains.

    The order is the fix. The ledger is a cache of what the channel holds, and on
    a fresh Kaggle session it lives in scratch and is empty — while the channel
    holds every video this exists for. Counting first and concluding "nothing to
    do" is how a twelve-hour session leaves the archive exactly as slow as it
    found it, which is what happened on the last run.

    Quiet after the first pass. The re-arm looks again every half hour, and forty
    lines saying "idle" is not information; a pass that finds work always says so.
    """
    counts = eng.ledger.asset_counts()
    if first and not counts["videos"]:
        _state("restoring", "no ledger yet — restoring the pinned copy from "
                            "the channel")
        try:
            eng.restore_ledger()
        except Exception as exc:                            # noqa: BLE001
            # No longer fatal, which it used to be. The snapshot is a shortcut
            # for a ledger that was built once; the channel walk below learns the
            # same thing the long way, and it is the only thing that can see a
            # video uploaded by a plane that never wrote a ledger row at all.
            log(f"no pinned ledger to restore ({type(exc).__name__}: "
                f"{str(exc)[:140]}) — reading the channel instead")

    if first:
        _state("seeding", "reading the channel for videos the ledger has not "
                          "seen")
    # A first walk of a few thousand messages takes a minute or two, and a minute
    # of silence during the one step that has to work is what made this look
    # broken in the first place. Throttled, because the callback fires per batch
    # and the log is not a progress bar. Later passes read one batch, so they say
    # nothing until there is something to say.
    last = [0.0]

    def _say(at, head, n):
        if time.time() - last[0] < 15.0:
            return
        last[0] = time.time()
        log(f"channel scan: message {at:,} of {head:,} — {n} video(s) found")

    res = eng.seed_ledger(on_progress=_say if first else None)
    if res.get("error"):
        log(f"channel scan failed: {res['error'][:160]} — working from the "
            f"ledger as it stands")
    elif first or res.get("adopted"):
        log(f"channel scan: {res.get('in_channel', 0)} video(s) seen, "
            f"{res.get('adopted', 0)} new to the ledger"
            + (" (unchanged since the last scan)" if res.get("skipped") else ""))

    counts = eng.ledger.asset_counts()
    bf = get_backfill()
    pending = bf.outstanding(eng)
    if not pending:
        if first:
            _state("done", (f"all {counts['videos']} captured video(s) already "
                            f"have an asset set — {counts['clips']} clip(s) in "
                            f"the channel") if not counts["without_assets"] else
                           (f"{counts['without_assets']} video(s) have no asset "
                            f"set and every one of them has already been tried "
                            f"this session"))
        return

    started = bf.start(eng)
    # `start` refuses while a run is alive, and on the second pass that is the
    # likely answer: the operator pressed the button, or the previous pass is
    # still going. Waiting is the right response to that, and reporting "off"
    # would be a lie about a backfill that is working.
    if not started.get("ok") and bf.status()["state"] not in ("running",
                                                              "stopping"):
        _state("off", started.get("message") or "the backfill refused to start")
        return
    _state("running", f"{len(pending)} video(s) without an asset set"
                      if started.get("ok") else
                      "a backfill is already running — waiting for it to finish")
    # Held until it drains, so the re-arm interval is measured from the end of a
    # pass and not from its start — otherwise a run longer than the interval would
    # be re-entered every half hour and refused every time.
    while bf.status()["state"] in ("running", "stopping"):
        time.sleep(5.0)


def autostart(delay: float = 0.0) -> dict:
    """Arm the backfill for after boot. Never raises; reports instead.

    One thread, then a loop: seed the ledger from the channel, count what has no
    asset set, run, wait for it to drain, and look again `VIOS_BACKFILL_EVERY`
    seconds later. The repeat is what covers a video harvested at hour three by a
    producer that never calls `publish_assets` — which on this deployment is every
    producer. `VIOS_BACKFILL_EVERY=0` stops after one pass.

    `VIOS_BACKFILL_AUTOSTART=0` opts out entirely — for a session opened to read
    the archive rather than to fill in what it is missing.
    """
    if str(os.environ.get("VIOS_BACKFILL_AUTOSTART", "1")).strip().lower() in (
            "0", "false", "no", "off"):
        _state("off", "VIOS_BACKFILL_AUTOSTART=0 — no clips will be built this "
                      "session")
        return dict(_AUTO)
    if _AUTO["armed"]:
        return dict(_AUTO)
    _AUTO.update({"armed": True, "state": "waiting", "at": time.time(),
                  "message": f"starting in {int(delay)}s"})
    log(f"armed — first pass in {int(delay)}s")

    def _boot():
        try:
            time.sleep(max(0.0, delay))
            _state("checking", "reading the ledger")
            from .engine import get_engine  # noqa: PLC0415
            eng = get_engine()

            # Before anything touches the channel, not just before the uploads.
            # The seed below walks history over MTProto against the same chat
            # Atlas is importing bundles from, so it belongs on this side of the
            # wait too.
            _wait_for_atlas()

            passes, said = 0, False
            while True:
                if eng.telegram:
                    _cycle(eng, passes == 0)
                    passes += 1
                elif passes == 0:
                    # Not the end of it, as long as the re-arm is on. The token is
                    # set in the capture tab, and an operator who sets it after
                    # boot should not have to know that this thread gave up ninety
                    # seconds in.
                    _state("off", "no bot token yet — set it in the capture tab; "
                                  "this picks it up on its own"
                                  if REARM_EVERY > 0 else
                                  "no bot token — set it in the capture tab and "
                                  "press Build the missing asset sets")
                if REARM_EVERY <= 0:
                    return              # one-shot: the old behaviour, on request
                if passes and not said:
                    said = True
                    log(f"watching for videos harvested later in this session — "
                        f"looking again every "
                        f"{max(1, int(REARM_EVERY // 60))} min "
                        f"(VIOS_BACKFILL_EVERY=0 to stop after one pass)")
                # Silent from here: a line every half hour for twelve hours says
                # nothing, and `_cycle` speaks whenever it finds work.
                _AUTO.update({"state": "waiting", "at": time.time(),
                              "message": "watching for videos harvested later "
                                         "in this session"})
                time.sleep(REARM_EVERY)
        except Exception as exc:
            _state("off", f"{type(exc).__name__}: {str(exc)[:200]}")

    threading.Thread(target=_boot, name="vios-backfill-arm",
                     daemon=True).start()
    return dict(_AUTO)
