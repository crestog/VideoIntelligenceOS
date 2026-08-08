"""
vios.capture.fetch — one permalink in, one complete evidence bundle out.

The old Colab script asked yt-dlp for a video and an info JSON. That threw away
the single richest signal Instagram hands out for free: **the comments**. For a
system whose stated purpose is understanding why a reel worked, the audience's
own reaction is not a nice-to-have — it is the only ground truth about
reception that exists anywhere in the pipeline. yt-dlp's Instagram extractor
has a dedicated comments endpoint and fetching it costs exactly one extra HTTP
request per reel, which at our pacing is free.

So this module extracts the maximum, once, and never comes back:

  video      the original mp4, untouched, never re-encoded
  info json  yt-dlp's metadata with --no-clean-info-json, so the raw Instagram
             API fields survive rather than being pruned to the fields yt-dlp
             itself models
  comments   --write-comments: author, text, like count, and parent id, which
             is what gives the reply threading
  thumbnail  the creator's own chosen cover frame — an authored artifact, not
             a frame we picked
  record     a normalised, schema-versioned summary written by us on top of all
             of the above, so the processing plane never has to guess which of
             Instagram's several view-count fields is populated today

Everything lands in one temp directory per reel and is uploaded together.

Failure taxonomy matters here, because the pacer's reaction depends on it:

  unavailable  the post is gone or private. Terminal — nothing retries.
  hostile      401/429/checkpoint/login-required. Back off hard; this is the
               response that precedes a block.
  transient    network, timeout, a bad minute. Retry later at the same pace.

Conflating the last two is the mistake that turns a rate limit into a ban:
retrying a 429 promptly is precisely the behaviour being watched for.

Invocation is `python -m yt_dlp`, not the `yt-dlp` console script. On Kaggle a
`pip install --user` puts the script somewhere off PATH often enough that the
console-script form fails intermittently on a machine where the module imports
perfectly — and an intermittent failure in an unattended week-long run is much
worse than an obvious one.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

RECORD_SCHEMA = 1

# Classification of the fetcher's own complaint. Hostile is tested first: a
# login wall says both "not available" and "login required", and the second is
# the one that must win.
_HOSTILE = re.compile(
    r"(rate.?limit|\b429\b|too many requests|login required|"
    r"requested content is not available|checkpoint|challenge.?required|"
    r"please wait a few minutes|\b401\b|\b403\b|forbidden|"
    r"cookies are no longer valid|empty media response|"
    r"sign in to confirm|temporarily blocked)", re.IGNORECASE)

_UNAVAILABLE = re.compile(
    r"(unable to (fetch|extract)|has been removed|has been deleted|"
    r"(may have been|been) (deleted|removed)|"
    r"content is (not available|unavailable)|page not found|"
    # "Post not available", "Video unavailable" — the same fact as the phrasings
    # above, and without this they fall through to `transient` and spend a full
    # retry ladder on a post that is definitively gone.
    r"\b(post|video|media|reel|photo) (is )?(not available|unavailable)\b|"
    r"post is private|account is private|"
    r"unsupported url|\b404\b|does not exist|nothing to download)",
    re.IGNORECASE)

# "No video formats found" is yt-dlp saying *this post has no video*, which for
# a saved Instagram post usually means it is a photo or a carousel of photos.
# That is a post worth keeping — the caption, the comments and the images are
# all signal — and `fetch` has always known how to return an image-only result.
# What it could not do was get there: this pattern used to sit in _UNAVAILABLE,
# so the error was terminal, gallery-dl was never tried, and every photo post
# in the export was written off as "unavailable". It is its own class now, and
# the only one that says "try the other downloader, it fetches images".
_PHOTO_ONLY = re.compile(
    r"(no video formats found|no video could be found|"
    r"requested format is not available|there are no video)",
    re.IGNORECASE)

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".heic")

# A current desktop Chrome UA. It has to be plausible next to the cookies,
# which came from a desktop browser; a mismatched or absent UA is a cheap tell.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36")


class FetchError(RuntimeError):
    """A fetch failed.

    `kind` is one of **unavailable | hostile | transient | photo**, and the last
    one is internal to this module. `photo` means yt-dlp found the post but it
    holds no video, which is a routing instruction rather than a failure:
    `fetch` catches it and hands the URL to gallery-dl, so it resolves into
    either a success or an `unavailable`. It never reaches the engine, whose
    handler branches on `terminal` and `hostile` and would otherwise treat a
    photo post as a transient fault worth retrying.
    """

    def __init__(self, message: str, kind: str = "transient"):
        super().__init__(message)
        self.kind = kind

    @property
    def hostile(self) -> bool:
        return self.kind == "hostile"

    @property
    def terminal(self) -> bool:
        return self.kind == "unavailable"

    @property
    def photo_only(self) -> bool:
        """yt-dlp found the post but it holds no video. gallery-dl can."""
        return self.kind == "photo"


def classify(text: str) -> str:
    blob = text or ""
    if _HOSTILE.search(blob):
        return "hostile"
    # Ordered before _UNAVAILABLE: a photo post's log often carries a generic
    # "unable to extract" alongside the specific "no video formats found", and
    # the specific one is the one that tells you what to do next.
    if _PHOTO_ONLY.search(blob):
        return "photo"
    if _UNAVAILABLE.search(blob):
        return "unavailable"
    return "transient"


def tool_versions() -> dict:
    """What the fetcher can actually reach, for the admin panel's readiness
    check. Reported before a run rather than discovered at reel one."""
    out = {"yt_dlp": "", "gallery_dl": "", "ffmpeg": "", "ok": False}
    probes = (
        ("yt_dlp", [sys.executable, "-m", "yt_dlp", "--version"]),
        ("gallery_dl", [sys.executable, "-m", "gallery_dl", "--version"]),
        ("ffmpeg", ["ffmpeg", "-version"]),
    )
    for name, args in probes:
        try:
            res = subprocess.run(args, capture_output=True, text=True,
                                 timeout=60)
            if res.returncode == 0 and res.stdout:
                line = res.stdout.strip().splitlines()[0]
                # ffmpeg answers with a whole banner ("ffmpeg version 8.1.2-full
                # _build-www.gyan.dev Copyright (c) 2000-2026 …"); the readiness
                # panel wants the version, not the build's marketing.
                m = re.search(r"\bversion\s+(\S+)", line)
                out[name] = (m.group(1) if m else line)[:40]
        except (OSError, subprocess.SubprocessError):
            pass
    out["ok"] = bool(out["yt_dlp"])
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _find(work: str, exts) -> list:
    out = []
    for ext in exts:
        out.extend(glob.glob(os.path.join(work, f"*{ext}")))
        out.extend(glob.glob(os.path.join(work, "**", f"*{ext}"),
                             recursive=True))
    return sorted(set(out))


def _first_error(blob: str) -> str:
    for line in (blob or "").splitlines():
        if "ERROR" in line or "error" in line.lower():
            return line.strip()[:400]
    return (blob or "unknown fetch failure").strip()[-400:]


# ═══════════════════════════════════════════════════════════════════════
# Running the fetchers
# ═══════════════════════════════════════════════════════════════════════
def _run(argv: list, timeout: float) -> tuple[int, str]:
    try:
        res = subprocess.run(argv, capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FetchError(f"fetch timed out after {timeout:.0f}s", "transient")
    except OSError as e:
        raise FetchError(f"cannot run fetcher: {e}", "transient")
    combined = ((res.stdout or "") + "\n" + (res.stderr or ""))
    return res.returncode, combined[-8000:]


def _ytdlp_argv(url: str, work: str, cookies: str | None,
                comments: bool, fast: bool = False) -> list:
    argv = [
        sys.executable, "-m", "yt_dlp",
        "--no-warnings", "--no-progress", "--no-color",
        # A stray ~/.config/yt-dlp must not silently change our format
        # selection or pacing on someone else's machine.
        "--ignore-config",
        "--no-mtime",
        "--retries", "2",
        "--fragment-retries", "2",
        "--socket-timeout", "45",
        # yt-dlp's own inter-request sleep, on top of our pacer. This spaces
        # the two or three requests a single reel makes (page, media,
        # comments) so one reel is not itself a small burst. In fast mode it
        # drops to nothing: it was costing ~9 s per reel, which at a 6 s target
        # would have been most of the run time.
        "--sleep-requests", "0" if fast else "3",
        "--user-agent", USER_AGENT,
        "--add-header", "Accept-Language:en-US,en;q=0.9",

        # Maximum metadata. --no-clean-info-json is the important one: without
        # it yt-dlp strips every field it does not model, which is most of
        # what Instagram returns.
        "--write-info-json",
        "--no-clean-info-json",
        "--write-thumbnail",

        # Original bytes. Prefer a single progressive mp4 so nothing is
        # remuxed; fall back to best available.
        "-f", "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "-o", "%(id)s.%(ext)s",
        "-P", work,
    ]
    if fast:
        # Split the file across connections. Worth it only in fast mode: at a
        # two-minute pace the download is not the bottleneck and four parallel
        # range requests per reel is a louder fingerprint than one.
        argv += ["--concurrent-fragments", "4"]
    if comments:
        argv.append("--write-comments")
    if cookies and os.path.isfile(cookies):
        argv += ["--cookies", cookies]
    argv.append(url)
    return argv


def run_ytdlp(url: str, work: str, cookies: str | None = None,
              timeout: float = 420.0, fast: bool = False) -> str:
    """Fetch with yt-dlp. Returns the combined output for the journal.

    Comments are requested first and dropped on the retry: `--write-comments`
    exercises a separate Instagram endpoint that fails independently of the
    media one, and losing a reel because its comment page had a bad minute
    would be a poor trade. The retry is not paced differently — it is the same
    post, seconds later, which is ordinary browsing behaviour.
    """
    code, out = _run(_ytdlp_argv(url, work, cookies, True, fast), timeout)
    if code == 0 and _find(work, VIDEO_EXT + IMAGE_EXT):
        return out

    kind = classify(out)
    if kind == "hostile":
        raise FetchError(_first_error(out), "hostile")

    code2, out2 = _run(_ytdlp_argv(url, work, cookies, False, fast), timeout)
    if code2 == 0 and _find(work, VIDEO_EXT + IMAGE_EXT):
        return out + "\n[retried without comments]\n" + out2

    blob = out2 or out
    raise FetchError(_first_error(blob), classify(blob))


def run_gallery_dl(url: str, work: str, cookies: str | None = None,
                   timeout: float = 420.0) -> str:
    """Fallback path. gallery-dl parses Instagram differently enough that it
    often succeeds where yt-dlp's extractor has fallen behind a site change —
    which is the most common way a pipeline like this breaks over a year."""
    argv = [sys.executable, "-m", "gallery_dl",
            "--quiet", "--write-metadata", "--no-download-archive",
            "--user-agent", USER_AGENT,
            "-D", work]
    if cookies and os.path.isfile(cookies):
        argv += ["--cookies", cookies]
    argv.append(url)
    code, out = _run(argv, timeout)
    if code != 0 and not _find(work, VIDEO_EXT + IMAGE_EXT):
        raise FetchError(_first_error(out), classify(out))
    return out


# ═══════════════════════════════════════════════════════════════════════
# Normalisation
# ═══════════════════════════════════════════════════════════════════════
def _load_info(work: str) -> dict:
    """The richest metadata JSON in the directory.

    A carousel writes one .info.json per entry plus one for the playlist; the
    largest file is the one with the most metadata, which is the one worth
    normalising from. Every one of them is still uploaded — the record keeps
    the raw blob so nothing is lost by choosing here.
    """
    files = (glob.glob(os.path.join(work, "**", "*.info.json"), recursive=True)
             or glob.glob(os.path.join(work, "**", "*.json"), recursive=True))
    best, best_size = None, -1
    for path in files:
        if path.endswith(".vios.json"):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size > best_size:
            best, best_size = path, size
    if not best:
        return {}
    try:
        with open(best, "r", encoding="utf-8", errors="replace") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _first_number(info: dict, *keys) -> int | None:
    """Instagram populates whichever of its several count fields it feels like
    on a given day, and yt-dlp passes them through under different names
    depending which endpoint answered. Take the first one that is present."""
    for k in keys:
        v = info.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v >= 0:
            return int(v)
    return None


def _comments_from(info: dict) -> list:
    raw = info.get("comments")
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        out.append({
            "id": str(c.get("id") or ""),
            "parent": str(c.get("parent") or "root"),
            "author": c.get("author") or c.get("author_id") or "",
            "text": (c.get("text") or "")[:4000],
            "likes": c.get("like_count"),
            "at": c.get("timestamp"),
            "is_creator": bool(c.get("author_is_uploader")),
        })
    return out


def build_record(url: str, key: str, work: str, info: dict,
                 collections: list, fetch_log: str, tool: str) -> dict:
    """The normalised summary that travels with the video, forever.

    Deliberately a *superset*: the flattened fields the rest of VIOS reads,
    plus `raw`, the untouched info JSON. Anything we failed to anticipate today
    is still recoverable in two years without re-fetching — which is the whole
    reason capture is treated as irreversible and processing as disposable.
    """
    comments = _comments_from(info)
    desc = info.get("description") or info.get("title") or ""
    return {
        "schema": RECORD_SCHEMA,
        "captured_at": time.time(),
        "captured_by": tool,
        "key": key,
        "url": url,
        "collections": sorted({c for c in (collections or []) if c}),

        "post": {
            "id": info.get("id") or key,
            "uploader": info.get("uploader") or info.get("channel") or "",
            "uploader_id": info.get("uploader_id") or info.get("channel_id") or "",
            "uploader_url": info.get("uploader_url") or "",
            "title": (info.get("title") or "")[:500],
            "description": desc,
            "hashtags": sorted(set(re.findall(r"#(\w{2,50})", desc))),
            "mentions": sorted(set(re.findall(r"@([\w.]{2,40})", desc))),
            "taken_at": info.get("timestamp") or info.get("release_timestamp"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "width": info.get("width"),
            "height": info.get("height"),
            "fps": info.get("fps"),
            "vcodec": info.get("vcodec"),
            "acodec": info.get("acodec"),
            "thumbnail": info.get("thumbnail"),
            "track": info.get("track"),
            "artist": info.get("artist"),
        },

        "engagement": {
            # Frozen at capture time and never re-polled. Re-polling 5,000
            # posts is the single riskiest thing this system could do, and
            # `overperformance` is computed against the creator's own baseline,
            # which cancels most of the staleness anyway.
            "views": _first_number(info, "view_count", "play_count"),
            "likes": _first_number(info, "like_count"),
            "comments": _first_number(info, "comment_count"),
            "reposts": _first_number(info, "repost_count"),
            "at": time.time(),
        },

        "comments": comments,
        "comments_captured": len(comments),

        "files": sorted(
            os.path.relpath(p, work)
            for p in glob.glob(os.path.join(work, "**", "*"), recursive=True)
            if os.path.isfile(p)),
        "fetch_log": (fetch_log or "")[-3000:],
        "raw": info,
    }


# ═══════════════════════════════════════════════════════════════════════
# The one call the engine makes
# ═══════════════════════════════════════════════════════════════════════
def fetch(url: str, key: str, work: str, cookies: str | None = None,
          collections: list | None = None,
          allow_gallery_dl: bool = True,
          timeout: float = 420.0,
          fast: bool = False) -> dict:
    """Fetch one permalink into `work`. Raises FetchError on failure.

    Returns {video, images, record, record_path, info, bytes, sha256, tool}.
    `video` is None for a photo-only post — the caller decides what to do with
    that, and the right answer is to keep it: the caption and comments are
    still signal, and the ledger should not pretend the post never existed.

    `fast` drops yt-dlp's own inter-request sleep and splits the download
    across connections. See `vios.capture.pacing` for what that costs.
    """
    os.makedirs(work, exist_ok=True)
    tool = "yt-dlp"
    try:
        log = run_ytdlp(url, work, cookies, timeout, fast=fast)
    except FetchError as first:
        # A photo post is the one failure where the second downloader is not a
        # long shot but the correct tool: yt-dlp only does video, gallery-dl
        # does images. So it overrides `allow_gallery_dl`, which exists to keep
        # a *retry* off a hostile host — and this is not a retry, it is the
        # first attempt by the only program that can do the job.
        if first.hostile or (first.terminal and not first.photo_only):
            raise
        if not allow_gallery_dl and not first.photo_only:
            raise
        try:
            log = run_gallery_dl(url, work, cookies, timeout)
            tool = "gallery-dl"
        except FetchError as second:
            if first.photo_only:
                # Now it is genuinely gone: the post has no video, and the
                # image fetcher could not find images either.
                raise FetchError(
                    f"post has no video and no images could be fetched "
                    f"({second})", "unavailable")
            # Otherwise report the first failure: yt-dlp's diagnostics are far
            # better, and gallery-dl's "no results" says nothing about why.
            raise FetchError(f"{first} (gallery-dl also failed: {second})",
                             first.kind)

    videos = _find(work, VIDEO_EXT)
    images = _find(work, IMAGE_EXT)
    # yt-dlp writes the cover next to the video with the same stem. That file
    # is a thumbnail, not a photo post, and must not be counted as media.
    stems = {os.path.splitext(os.path.basename(v))[0] for v in videos}
    images = [p for p in images
              if os.path.splitext(os.path.basename(p))[0] not in stems]

    video = max(videos, key=os.path.getsize) if videos else None
    if not video and not images:
        raise FetchError("fetch produced no media", "unavailable")

    info = _load_info(work)
    record = build_record(url, key, work, info, collections or [], log, tool)

    # For a photo post the "media" is the slides. Reporting 0 bytes and an
    # empty digest — which is what measuring only `video` did — made every
    # photo look like a zero-byte capture in the ledger and gave the dedup
    # check nothing to compare, so two imports of the same carousel could not
    # be told apart.
    if video:
        size = os.path.getsize(video)
        digest = _sha256(video)
    else:
        size = sum(os.path.getsize(p) for p in images)
        digest = _sha256(images[0]) if images else ""

    record["media"] = {
        "filename": os.path.basename(video) if video else "",
        "bytes": size,
        "sha256": digest,
        "kind": "video" if video else "photo",
        "images": [os.path.basename(p) for p in images],
    }

    record_path = os.path.join(work, f"{key}.vios.json")
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, separators=(",", ":"))

    return {"video": video, "images": images, "record": record,
            "record_path": record_path, "info": info,
            "bytes": size, "sha256": digest, "tool": tool, "log": log}


def cleanup(work: str):
    shutil.rmtree(work, ignore_errors=True)
