"""
vios.capture.engine — the loop that runs for a week.

Everything else in this package is a part; this is the machine. It claims one
item from the ledger, fetches it, uploads it, records it, waits a couple of
minutes, and does it again — for eight days if that is what 5,000 reels takes,
across however many sessions the operator feels like starting.

The design constraints are unusual and they drive every decision here.

**It will be killed.** Kaggle sessions end at 12 hours; the operator stops it
after two hours because they feel like it. So there is no "run" concept at all:
there is only the ledger, and starting is just "keep going". Nothing is held in
memory that would be missed. The unit of progress is one item, committed.

**It must never re-download.** The ledger answers that, and the channel scan
repairs the ledger, and both run before the first fetch of every session. Six
months and a new laptop later, the answer is still correct.

**It must not look like a robot.** The pacer owns that entirely. The loop's
only obligation is to actually honour it — including after a failure, where
the tempting bug is to retry immediately and hand Instagram the exact burst
pattern it watches for.

**It must be stoppable instantly.** Every sleep is sliced, every long operation
is bounded by a timeout, and the flag is checked between phases. Stop means
stop within about a second, even mid-break.

Credentials live here only as instance attributes, set from the admin form and
gone when the process exits. They are never written to disk, never logged, and
never read from a module constant.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback

# Imported by name, not as `from . import fetch as fetchmod`: the package's
# __init__ re-exports the *function* `fetch`, which rebinds the attribute
# `vios.capture.fetch` from the submodule to the function. A relative module
# import would then silently pick up the function and fail on the first
# `fetchmod.cleanup(...)` — an hour into a run, not at import time.
from .fetch import (fetch as fetch_one, cleanup as fetch_cleanup,
                    FetchError, tool_versions)
from .ledger import Ledger, open_ledger
from .pacing import Pacer
from .upload import Telegram, UploadError, publish, upload_snapshot

IDLE, RUNNING, PAUSED, STOPPING, ERROR = "idle", "running", "paused", "stopping", "error"

SNAPSHOT_EVERY = 25          # items between ledger uploads to Telegram
MIN_FREE_BYTES = 2 * 1024 ** 3
HOSTILE_STREAK_STOP = 5      # consecutive hostile replies that halt the run


def _default_base() -> str:
    """Where the ledger and scratch live.

    Prefers the existing VIOS layout when the repo's config is importable, so
    a Kaggle session that already has `/kaggle/working/Insta-Vault` keeps
    everything in one place, and falls back to a local directory so this
    package can be run on its own.
    """
    try:
        import config
        return getattr(config, "BASE_DIR", os.path.abspath("vios_data"))
    except Exception:
        return os.path.abspath("vios_data")


def _default_scratch() -> str:
    try:
        import config
        return getattr(config, "SCRATCH_DIR", os.path.join(_default_base(), "_scratch"))
    except Exception:
        return os.path.join(_default_base(), "_scratch")


class CaptureEngine:
    """One instance per process. Start it, watch it, stop it."""

    def __init__(self, base_dir: str | None = None):
        self.base = base_dir or _default_base()
        self.ledger_path = os.path.join(self.base, "capture_ledger.db")
        self.scratch = os.path.join(_default_scratch(), "capture")
        os.makedirs(self.base, exist_ok=True)
        os.makedirs(self.scratch, exist_ok=True)

        self.state = IDLE
        self.message = "Not started."
        self.error = ""
        self.started_at: float | None = None
        self.session_done = 0
        self.session_failed = 0
        self.hostile_streak = 0
        self.since_snapshot = 0
        self.current: dict = {}
        self.waiting_until: float | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.RLock()

        # Credentials. Held in memory for the life of the process and never
        # written by this class — but *read* from Kaggle Secrets, the
        # environment and the laptop file at construction, so a session that
        # has them stored is already configured before the page loads. See
        # vios/creds.py.
        self._tg: Telegram | None = None
        self._api_id = 0
        self._api_hash = ""
        self._cookies_path = ""
        self.cred_sources: dict = {}

        self.pacer = Pacer()
        self.skip_collections: tuple = ()
        self.max_attempts = 5
        self.allow_gallery_dl = True
        self.snapshot_every = SNAPSHOT_EVERY

        self._ledger: Ledger | None = None
        self._adopt_stored_credentials()

    def _adopt_stored_credentials(self) -> None:
        """Configure from the stored credentials, if there are any.

        Failure here is silent on purpose: a broken secret store must not stop
        the tab from loading, because the tab is where you would go to fix it.
        """
        try:
            from vios.creds import resolve  # noqa: PLC0415
            got = resolve()
        except Exception:
            return
        v = got.get("values") or {}
        if not v:
            return
        self.cred_sources = got.get("sources") or {}
        try:
            self.configure(
                bot_token=v.get("bot_token", ""),
                channel_id=v.get("channel_id") or None,
                api_id=int(v.get("api_id") or 0),
                api_hash=v.get("api_hash", ""),
                cookies_text=v.get("ig_cookies", ""))
        except Exception:
            self.cred_sources = {}

    # ── ledger ───────────────────────────────────────────────────────────
    @property
    def ledger(self) -> Ledger:
        with self._lock:
            if self._ledger is None:
                self._ledger = open_ledger(self.ledger_path)
            return self._ledger

    # ── configuration ────────────────────────────────────────────────────
    def configure(self, bot_token: str = "", channel_id=None, api_id=0,
                  api_hash: str = "", cookies_text: str = "",
                  target: float | None = None, quiet_hours: bool | None = None,
                  breaks: bool | None = None, skip_collections=None,
                  max_attempts: int | None = None,
                  allow_gallery_dl: bool | None = None,
                  speed: str | None = None) -> dict:
        """Accept settings from the admin form. Blank fields keep their value,
        so the operator can change the pace mid-week without re-typing a token.
        """
        with self._lock:
            if bot_token or channel_id:
                token = bot_token or (self._tg.token if self._tg else "")
                chan = channel_id or (self._tg.channel if self._tg else None)
                self._api_id = int(api_id or self._api_id or 0)
                self._api_hash = (api_hash or self._api_hash or "").strip()
                self._tg = Telegram(token, chan, self._api_id, self._api_hash)
            elif api_id or api_hash:
                self._api_id = int(api_id or self._api_id or 0)
                self._api_hash = (api_hash or self._api_hash or "").strip()
                if self._tg:
                    self._tg.api_id = self._api_id
                    self._tg.api_hash = self._api_hash

            if cookies_text and cookies_text.strip():
                self._write_cookies(cookies_text)
            # Before `target`: switching profile re-aims the target, so an
            # explicit target sent in the same call has to win over it.
            if speed:
                self.pacer.set_profile(speed)
            if target:
                self.pacer.target = max(self.pacer.floor, float(target))
            if quiet_hours is not None:
                self.pacer.quiet_hours = bool(quiet_hours)
            if breaks is not None:
                self.pacer.breaks = bool(breaks)
            if skip_collections is not None:
                self.skip_collections = tuple(
                    c for c in skip_collections if c)
            if max_attempts:
                self.max_attempts = max(1, int(max_attempts))
            if allow_gallery_dl is not None:
                self.allow_gallery_dl = bool(allow_gallery_dl)

        out = self.settings()
        # Outside the lock: this touches the ledger, and the ledger is what the
        # worker thread holds. A channel change is surfaced, never acted on
        # silently — requeueing 6,000 reels is the operator's decision.
        if self._tg and self._tg.channel:
            try:
                out["channel_check"] = self.ledger.bind_channel(self._tg.channel)
            except Exception as exc:
                out["channel_check"] = {"state": "unknown", "error": str(exc)}
        return out

    def accept_channel_change(self, requeue: bool = True) -> dict:
        """Confirm a channel switch. See `Ledger.rebind_channel`."""
        if not (self._tg and self._tg.channel):
            raise RuntimeError("Set the channel id first.")
        return self.ledger.rebind_channel(self._tg.channel, requeue=requeue)

    def _write_cookies(self, text: str):
        """Persist the cookie jar for the length of the process only.

        yt-dlp needs a file, so one is written — into scratch, mode 0600, and
        removed on stop. A session cookie is as good as a password: it does not
        go near the repo directory and it is never included in a snapshot.
        """
        path = os.path.join(self.scratch, ".ig_cookies.txt")
        body = text if text.lstrip().startswith("#") else "# Netscape HTTP Cookie File\n" + text
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(body if body.endswith("\n") else body + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        self._cookies_path = path

    def settings(self) -> dict:
        """Never returns a secret. The UI shows presence, not value."""
        try:
            from vios.creds import describe  # noqa: PLC0415
            stored = describe()
        except Exception:
            stored = {}
        return {
            "base": self.base,
            "ledger": self.ledger_path,
            "bot_token_set": bool(self._tg and self._tg.token),
            "channel": (self._tg.channel if self._tg else None),
            "api_credentials_set": bool(self._api_id and self._api_hash),
            "credential_sources": dict(self.cred_sources),
            "stored_credentials": stored,
            "cookies_set": bool(self._cookies_path
                                and os.path.isfile(self._cookies_path)),
            "speed": self.pacer.profile,
            "target_seconds": round(self.pacer.target, 1),
            "quiet_hours": self.pacer.quiet_hours,
            "breaks": self.pacer.breaks,
            "skip_collections": list(self.skip_collections),
            "max_attempts": self.max_attempts,
            "gallery_dl_fallback": self.allow_gallery_dl,
            "snapshot_every": self.snapshot_every,
        }

    # ── readiness ────────────────────────────────────────────────────────
    def preflight(self) -> dict:
        """Everything that could stop a week-long run, checked in one call.

        Run by the admin tab's "Check" button before Start. A missing cookie
        file discovered at reel one costs an hour of confusion; discovered here
        it costs a sentence.
        """
        out = {"ok": False, "checks": [], "blocking": []}

        def check(name, ok, detail, blocking=True):
            out["checks"].append({"name": name, "ok": bool(ok),
                                  "detail": detail})
            if not ok and blocking:
                out["blocking"].append(name)

        tools = tool_versions()
        check("yt-dlp", bool(tools["yt_dlp"]),
              tools["yt_dlp"] or "not installed — pip install -U yt-dlp")
        check("gallery-dl", bool(tools["gallery_dl"]),
              tools["gallery_dl"] or "not installed (optional fallback)",
              blocking=False)
        check("ffmpeg", bool(tools["ffmpeg"]),
              tools["ffmpeg"] or "not on PATH (optional)", blocking=False)

        if self._tg:
            probe = self._tg.probe()
            check("Telegram", probe["ok"],
                  (f"@{probe['bot']} → {probe['channel']}" if probe["ok"]
                   else probe["error"]))
        else:
            check("Telegram", False, "No bot token or channel id yet.")

        check("Instagram cookies",
              bool(self._cookies_path and os.path.isfile(self._cookies_path)),
              "loaded" if self._cookies_path else
              "none — public reels may still work, saved/private ones will not",
              blocking=False)

        free = shutil.disk_usage(self.base).free
        check("Disk", free > MIN_FREE_BYTES, f"{free / 1024**3:.1f} GB free")

        counts = self.ledger.counts()
        check("Queue", counts.get("remaining", 0) > 0,
              f"{counts.get('remaining', 0)} to capture, "
              f"{counts.get('uploaded', 0)} already done")

        out["ok"] = not out["blocking"]
        out["counts"] = counts
        out["eta_hours"] = round(
            self.pacer.eta_seconds(counts.get("remaining", 0)) / 3600, 1)
        return out

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self, seed_first: bool = True) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                if self.state == PAUSED:
                    self.resume()
                    return {"ok": True, "state": self.state,
                            "message": "Resumed."}
                return {"ok": False, "state": self.state,
                        "message": "Already running."}
            if not self._tg:
                return {"ok": False, "state": self.state,
                        "message": "Set the bot token and channel id first."}
            self._stop.clear()
            self._pause.clear()
            self.error = ""
            self.session_done = self.session_failed = 0
            self.hostile_streak = 0
            self.started_at = time.time()
            self.state = RUNNING
            self.message = "Starting…"
            self._thread = threading.Thread(
                target=self._run, args=(seed_first,),
                name="vios-capture", daemon=True)
            self._thread.start()
        return {"ok": True, "state": self.state, "message": "Started."}

    def pause(self) -> dict:
        self._pause.set()
        if self.state == RUNNING:
            self.state = PAUSED
            self.message = "Paused — will finish the current item first."
        return {"ok": True, "state": self.state}

    def resume(self) -> dict:
        self._pause.clear()
        if self.state == PAUSED:
            self.state = RUNNING
            self.message = "Resumed."
        return {"ok": True, "state": self.state}

    def stop(self, wait: float = 20.0) -> dict:
        """Ask the loop to finish. The in-flight item is abandoned, not
        corrupted: its row is left `fetching` and the next start repairs it."""
        self._stop.set()
        self._pause.clear()
        self.state = STOPPING
        self.message = "Stopping…"
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=wait)
        if self.state == STOPPING and not (t and t.is_alive()):
            # The loop's own exit path normally does this; cover the case
            # where the thread was already gone when stop() was called.
            self.state = IDLE
            self.message = "Stopped."
        return {"ok": True, "state": self.state}

    def _should_stop(self) -> bool:
        return self._stop.is_set()

    # ── the loop ─────────────────────────────────────────────────────────
    def _run(self, seed_first: bool):
        led = self.ledger
        try:
            led.repair_stale()
            if seed_first:
                self._seed(led)

            while not self._should_stop():
                if self._pause.is_set():
                    self.state = PAUSED
                    time.sleep(1.0)
                    continue
                self.state = RUNNING

                item = led.claim_next(self.skip_collections)
                if item is None:
                    # Nothing claimable — but that has two very different
                    # meanings. If items are merely cooling off after a
                    # transient failure, exiting here would strand them until
                    # the operator noticed and pressed Start again, which over
                    # a week-long unattended run is exactly the failure this
                    # engine exists to avoid.
                    due = led.next_due_at(self.skip_collections)
                    if due is not None:
                        wait = max(5.0, min(due - time.time(), 900.0))
                        self.waiting_until = time.time() + wait
                        self.message = (
                            f"Waiting {wait / 60:.0f} min to retry "
                            f"{led.counts().get('failed', 0)} item(s) that "
                            f"failed earlier.")
                        self.pacer.sleep(wait, should_stop=self._should_stop)
                        self.waiting_until = None
                        continue
                    self.message = ("Everything in the queue is captured. "
                                    "Import more links to continue.")
                    self._snapshot(led, force=True)
                    break

                ok = self._one(led, item)
                if self._should_stop():
                    break

                if self.since_snapshot >= self.snapshot_every:
                    self._snapshot(led)

                if self.hostile_streak >= HOSTILE_STREAK_STOP:
                    self.state = ERROR
                    self.error = (
                        f"Instagram refused {self.hostile_streak} requests in a "
                        f"row. Stopped on purpose. Refresh the cookies from a "
                        f"browser that is logged in, wait a few hours, then "
                        f"start again — the ledger has lost nothing.")
                    self.message = self.error
                    led.log("halt", self.error)
                    self._snapshot(led, force=True)
                    return

                self._wait(ok)

            if self._should_stop():
                self.message = (f"Stopped. {self.session_done} captured this "
                                f"session — nothing is lost, start again "
                                f"whenever you like.")
        except Exception as exc:
            self.state = ERROR
            self.error = f"{type(exc).__name__}: {exc}"
            self.message = self.error
            try:
                led.log("crash", traceback.format_exc()[-1800:])
            except Exception:
                pass
        finally:
            self.current = {}
            self.waiting_until = None
            if self.state not in (ERROR,):
                self.state = IDLE

    def _seed(self, led: Ledger):
        """Teach the ledger what is already in the channel, before fetching.

        Failure here is reported and survived. The scan is an optimisation —
        an important one, but a run that cannot read history should still
        capture, and the ledger's own record is usually sufficient.
        """
        from .seed import seed_from_channel
        self.message = "Reading the channel to see what is already captured…"
        try:
            res = seed_from_channel(
                led, self._tg, self._api_id, self._api_hash,
                on_progress=lambda at, head, n: setattr(
                    self, "message",
                    f"Scanning channel {at}/{head} — {n} reels found"),
                should_stop=self._should_stop)
            if res.get("skipped"):
                self.message = "Channel unchanged since the last scan."
            else:
                self.message = (f"Channel has {res['in_channel']} reels; "
                                f"{res['adopted']} were new to the ledger.")
        except Exception as exc:
            self.message = (f"Could not read the channel ({exc}). Continuing "
                            f"from the local ledger.")
            led.log("seed-failed", str(exc)[:400])

    def _one(self, led: Ledger, item: dict) -> bool:
        """Capture a single reel. Returns False if it failed."""
        key, url = item["key"], item["url"]
        work = os.path.join(self.scratch, key)
        self.current = {"key": key, "url": url, "phase": "fetching",
                        "attempt": item.get("attempts", 1),
                        "started": time.time()}
        try:
            free = shutil.disk_usage(self.base).free
            if free < MIN_FREE_BYTES:
                raise RuntimeError(
                    f"only {free / 1024**3:.1f} GB free — pausing to avoid "
                    f"a half-written upload")

            result = fetch_one(
                url, key, work, cookies=self._cookies_path or None,
                collections=self._collections(led, key),
                allow_gallery_dl=self.allow_gallery_dl,
                fast=(self.pacer.profile == "fast"))

            self.current["phase"] = "uploading"
            self.current["bytes"] = result["bytes"]
            sent = publish(self._tg, result, self._collections(led, key),
                           progress=self._progress)

            rec = result["record"]
            post = rec.get("post", {})
            eng = rec.get("engagement", {})
            media = rec.get("media", {}) or {}
            is_photo = not result.get("video")
            led.mark_uploaded(
                key,
                msg_id=sent["msg_id"], record_msg_id=sent["record_msg_id"],
                file_id=sent["file_id"], file_size=result["bytes"],
                sha256=result["sha256"],
                # A photo post has no filename to take an extension from, so
                # say what it is. Without this the row's `ext` stays empty and
                # the processing plane cannot tell a photo from a failed video.
                ext=((media.get("filename", "").rsplit(".", 1)[-1][:8])
                     or ("photo" if is_photo else "")),
                duration=sent.get("duration") or post.get("duration"),
                width=sent.get("width") or post.get("width"),
                height=sent.get("height") or post.get("height"),
                uploader=post.get("uploader"), title=post.get("title"),
                views=eng.get("views"), likes=eng.get("likes"),
                comment_count=eng.get("comments"),
                comments_got=rec.get("comments_captured"),
                taken_at=post.get("taken_at"))
            what = (f"{sent.get('slides', 0)} photo(s)" if is_photo
                    else f"{result['bytes'] / 1048576:.1f} MB")
            led.log("captured",
                    f"{post.get('uploader') or '?'} · {what} · "
                    f"{rec.get('comments_captured', 0)} comments", key)
            for miss in (sent.get("slides_failed") or []):
                led.log("slide-missing", miss, key)

            self.session_done += 1
            self.since_snapshot += 1
            self.hostile_streak = 0
            self.pacer.note_success()
            self.message = f"Captured {key} ({self.session_done} this session)"
            return True

        except FetchError as exc:
            if exc.terminal:
                led.mark_unavailable(key, str(exc))
                # Not a failure of ours: the post is gone. The pacer should
                # count it as a normal, successful request, because it was one.
                self.pacer.note_success()
                self.session_failed += 1
                self.message = f"{key} is no longer on Instagram"
                return True
            if exc.hostile:
                self.hostile_streak += 1
                self.pacer.note_hostile()
                led.mark_failed(key, str(exc), retry_in=3600 * 2,
                                max_attempts=self.max_attempts)
                led.log("hostile", str(exc)[:400], key)
                self.message = (f"Instagram pushed back on {key} — backing off "
                                f"{self.pacer.backoff:.1f}×")
            else:
                self.pacer.note_failure()
                led.mark_failed(key, str(exc), max_attempts=self.max_attempts)
                self.message = f"{key} failed: {str(exc)[:120]}"
            self.session_failed += 1
            return False

        except UploadError as exc:
            # The bytes exist but Telegram would not take them. Retry sooner
            # than an Instagram failure — nothing about this involves the
            # account being watched, and re-fetching is the expensive part.
            self.pacer.note_failure()
            led.mark_failed(key, f"upload: {exc}", retry_in=300,
                            max_attempts=self.max_attempts)
            led.log("upload-failed", str(exc)[:400], key)
            self.session_failed += 1
            self.message = f"Upload failed for {key}: {str(exc)[:120]}"
            return False

        except Exception as exc:
            self.pacer.note_failure()
            led.mark_failed(key, f"{type(exc).__name__}: {exc}",
                            max_attempts=self.max_attempts)
            self.session_failed += 1
            self.message = f"{key}: {type(exc).__name__}: {str(exc)[:120]}"
            return False

        finally:
            fetch_cleanup(work)
            self.current = {}

    def _collections(self, led: Ledger, key: str) -> list:
        rows = led.conn.execute(
            "SELECT collection FROM membership WHERE key=?", (key,)).fetchall()
        return [r["collection"] for r in rows]

    def _progress(self, sent: int, total: int):
        if self.current:
            self.current["sent"] = sent
            self.current["total"] = total

    def _wait(self, ok: bool):
        """The deliberate idle between reels — the whole anti-detection budget.

        A failure does not shorten this. Retrying quickly after a refusal is
        the single behaviour most likely to escalate a soft rate limit into a
        hard block, so a failed fetch waits exactly as long as a good one.
        """
        if self.pacer.due_for_break():
            span = self.pacer.take_break()
            self.message = (f"Taking a {span / 60:.0f} minute break — "
                            f"{self.session_done} captured so far")
            self.waiting_until = time.time() + span
            self.pacer.sleep(span, should_stop=self._should_stop)
        else:
            gap = self.pacer.next_interval()
            self.waiting_until = time.time() + gap
            self.pacer.sleep(gap, should_stop=self._should_stop)
        self.waiting_until = None

    def _snapshot(self, led: Ledger, force: bool = False):
        """Push the ledger to Telegram so the session becomes disposable."""
        if not (force or self.since_snapshot >= self.snapshot_every):
            return
        self.since_snapshot = 0
        try:
            counts = led.counts()
            # Without this the uploaded file is missing every commit still
            # sitting in the write-ahead log — which is all the recent ones.
            led.checkpoint()
            upload_snapshot(
                self._tg, self.ledger_path,
                note=(f"{counts.get('uploaded', 0)} captured · "
                      f"{counts.get('remaining', 0)} to go"))
            led.set_meta("last_snapshot", str(time.time()))
            led.conn.commit()
        except Exception as exc:
            led.log("snapshot-failed", str(exc)[:300])

    # ── what the admin tab reads ─────────────────────────────────────────
    @property
    def telegram(self) -> Telegram | None:
        """The configured client, or None. Read-only on purpose — callers ask
        whether it exists, they do not get to swap it mid-run."""
        return self._tg

    def snapshot_now(self) -> dict:
        if not self._tg:
            raise RuntimeError("Set the bot token and channel id first.")
        self._snapshot(self.ledger, force=True)
        return {"message": "Ledger uploaded and pinned in the channel."}

    def restore_ledger(self) -> dict:
        """Pull the pinned ledger out of the channel and put it in place.

        The first thing a fresh Kaggle session does. The existing file is moved
        aside rather than deleted: restoring the wrong snapshot over a good
        local ledger would otherwise be an unrecoverable click, and the cost of
        keeping it is a few megabytes.
        """
        from .upload import restore_snapshot
        if not self._tg:
            raise RuntimeError("Set the bot token and channel id first.")
        if self.state in (RUNNING, PAUSED):
            raise RuntimeError("Stop the run before restoring a ledger over it.")
        tmp = self.ledger_path + ".incoming"
        if not restore_snapshot(self._tg, tmp):
            raise RuntimeError("No pinned ledger found in the channel.")
        with self._lock:
            if self._ledger:
                self._ledger.close()
                self._ledger = None
            if os.path.exists(self.ledger_path):
                os.replace(self.ledger_path, self.ledger_path + ".replaced")
            # A stale -wal/-shm beside a replaced database is how SQLite ends up
            # applying one ledger's journal to another one's pages.
            for side in ("-wal", "-shm"):
                stale = self.ledger_path + side
                if os.path.exists(stale):
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
            os.replace(tmp, self.ledger_path)
        counts = self.ledger.counts()
        self.ledger.log("restore", f"ledger restored from the channel: "
                                   f"{counts.get('uploaded', 0)} captured")
        return {"counts": counts,
                "message": "Ledger restored from the channel."}

    def status(self) -> dict:
        led = self.ledger
        counts = led.counts()
        remaining = counts.get("remaining", 0)
        elapsed = time.time() - self.started_at if self.started_at else 0
        return {
            "state": self.state,
            "message": self.message,
            "error": self.error,
            "counts": counts,
            "session": {
                "captured": self.session_done,
                "failed": self.session_failed,
                "elapsed": round(elapsed),
                "started_at": self.started_at,
            },
            "current": dict(self.current),
            "waiting_seconds": (round(self.waiting_until - time.time())
                                if self.waiting_until else 0),
            "pacer": self.pacer.describe(),
            "eta_hours": round(self.pacer.eta_seconds(remaining) / 3600, 1),
            "per_hour": led.throughput(3600),
            "hostile_streak": self.hostile_streak,
            "settings": self.settings(),
        }

    def activity(self, limit: int = 40) -> list:
        return self.ledger.events(limit=limit)

    # ── importing ────────────────────────────────────────────────────────
    def import_file(self, path: str, data: bytes | None = None) -> dict:
        """Parse an export ZIP or markdown file and enqueue everything in it."""
        from .inputs import parse_any
        parsed = parse_any(path, data)
        res = self.ledger.add_many(parsed["items"],
                                   source=os.path.basename(path))
        res.update({"format": parsed["format"],
                    "collections": parsed["collections"],
                    "found": len(parsed["items"])})
        return res

    def import_text(self, text: str, source: str = "pasted") -> dict:
        from .inputs import parse_markdown
        items = parse_markdown(text)
        res = self.ledger.add_many(items, source=source)
        res["found"] = len(items)
        return res

    # ── teardown ─────────────────────────────────────────────────────────
    def shutdown(self):
        self.stop()
        if self._cookies_path and os.path.isfile(self._cookies_path):
            try:
                os.remove(self._cookies_path)
            except OSError:
                pass
        if self._ledger:
            self._ledger.close()
            self._ledger = None


_ENGINE: CaptureEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> CaptureEngine:
    """The process-wide engine the admin routes talk to."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = CaptureEngine()
        return _ENGINE
