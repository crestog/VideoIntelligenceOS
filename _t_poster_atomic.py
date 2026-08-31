"""Does a poster render land atomically, and can a reader still be torn?

Two claims to check, and only one of them needs ffmpeg.

1. `media._render_frame` never leaves a partially written file at `dest`, even
   when several threads render different seek positions of the same second at
   once — which is the normal case, because the cache filename keeps whole
   seconds while the URL keeps decimals. Checked by having readers stat and read
   `dest` in a tight loop throughout and asserting every non-empty observation is
   a complete JPEG (SOI..EOI), never a prefix.

2. `server._jpeg_response` derives Content-Length from the same bytes it sends,
   so no interleaving can make the two disagree. Checked without a server by
   asserting `content-length == len(body)` for a file that is rewritten to a
   different size between the response being built and being read.

Run from the repo root: python _t_poster_atomic.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def check(name, good, detail=""):
    print(f"  {'PASS' if good else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not good:
        FAILS.append(name)


def make_source(path):
    """A four-second test clip whose frames differ, so seeks differ in size."""
    cmd = [shutil.which("ffmpeg"), "-nostdin", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=12:duration=4",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    return subprocess.run(cmd, capture_output=True, timeout=90).returncode == 0


def is_whole_jpeg(raw):
    return len(raw) > 2 and raw[:2] == b"\xff\xd8" and raw[-2:] == b"\xff\xd9"


def test_atomic_render(tmpdir):
    from atlas import media

    if not media._FFMPEG:
        check("ffmpeg present", False, "cannot exercise the render path")
        return

    src = os.path.join(tmpdir, "src.mp4")
    if not make_source(src):
        check("test clip built", False, "ffmpeg refused to build the fixture")
        return
    check("test clip built", True, f"{os.path.getsize(src)} bytes")

    dest = os.path.join(tmpdir, "key_1.jpg")
    stop = threading.Event()
    torn, seen, sizes = [], [], set()
    blocked = [0]
    empty_reads = [0]

    def reader():
        while not stop.is_set():
            try:
                with open(dest, "rb") as fh:
                    raw = fh.read()
            except OSError:
                # On Windows a writer holds the output open exclusively, so a
                # concurrent reader is refused rather than served a prefix. On
                # Linux — which is where the engine runs — this does not happen
                # and the reader sees whatever bytes are there, which is why the
                # tear was reachable in the first place.
                blocked[0] += 1
                continue
            if not raw:
                empty_reads[0] += 1
                continue
            seen.append(1)
            sizes.add(len(raw))
            if not is_whole_jpeg(raw):
                torn.append(len(raw))

    def writer(seek):
        # Same integer second, different decimals — the exact collision the
        # cache filename creates.
        for _ in range(4):
            media._render_frame(src, seek, dest)

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
    for t in readers:
        t.start()
    writers = [threading.Thread(target=writer, args=(s,))
               for s in (1.05, 1.35, 1.65, 1.95)]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    time.sleep(0.15)
    stop.set()
    for t in readers:
        t.join(timeout=5)

    check("readers saw the file at all", bool(seen),
          f"{len(seen)} read(s), {len(sizes)} distinct size(s)")
    check("no reader ever saw a torn JPEG", not torn,
          "clean" if not torn else f"{len(torn)} torn read(s): {torn[:5]}")
    check("no reader ever saw a zero-byte file", not empty_reads[0],
          "clean" if not empty_reads[0] else
          f"{empty_reads[0]} — a truncate was visible")
    print(f"        (informational: {blocked[0]} read(s) refused by the OS — "
          f"Windows exclusive-open; expected 0 on Linux)")
    check("a complete poster is left at dest",
          os.path.exists(dest) and is_whole_jpeg(open(dest, "rb").read()))

    litter = [f for f in os.listdir(tmpdir) if ".part" in f]
    check("no .part litter left behind", not litter, str(litter[:4]))

    # Different seeks really do produce different sizes — otherwise the race
    # this guards against would have been benign and the test proves nothing.
    one = os.path.join(tmpdir, "a.jpg")
    two = os.path.join(tmpdir, "b.jpg")
    media._render_frame(src, 1.05, one)
    media._render_frame(src, 1.95, two)
    n1 = os.path.getsize(one) if os.path.exists(one) else 0
    n2 = os.path.getsize(two) if os.path.exists(two) else 0
    check("two seeks in one second differ in size", n1 != n2,
          f"{n1} vs {n2} bytes — a mid-response rewrite would change the length")


def test_length_matches_body(tmpdir):
    """Content-Length and the body come from one read, so they cannot disagree."""
    from atlas import server

    path = os.path.join(tmpdir, "len.jpg")
    with open(path, "wb") as fh:
        fh.write(b"\xff\xd8" + b"A" * 4000 + b"\xff\xd9")

    resp = server._jpeg_response(path)
    # Rewrite to a different size *after* the response exists — under
    # FileResponse this is exactly the window that produced the RuntimeError.
    with open(path, "wb") as fh:
        fh.write(b"\xff\xd8" + b"B" * 40 + b"\xff\xd9")

    declared = int(resp.headers["content-length"])
    check("content-length equals the body it will send",
          declared == len(resp.body),
          f"declared {declared}, body {len(resp.body)}")
    check("a rewrite after the fact cannot change either", declared == 4004,
          f"{declared} — still the bytes that were read")
    check("media type survives", resp.headers.get("content-type") == "image/jpeg")
    check("cache header survives",
          "immutable" in resp.headers.get("cache-control", ""))

    missing = server._jpeg_response(os.path.join(tmpdir, "nope.jpg"))
    check("a vanished poster is 204, not a traceback",
          missing.status_code == 204, f"status {missing.status_code}")

    empty = os.path.join(tmpdir, "zero.jpg")
    open(empty, "wb").close()
    check("a zero-byte poster is 204, not an empty 200",
          server._jpeg_response(empty).status_code == 204)


def main():
    tmpdir = tempfile.mkdtemp(prefix="vios_poster_")
    try:
        print("atomic render under concurrent readers")
        test_atomic_render(tmpdir)
        print("content-length from the bytes actually sent")
        test_length_matches_body(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} — " + "; ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
