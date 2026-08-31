"""Every page and every read-only endpoint, against the real app. No stubs.

The first version of this walked `app.routes` and found six API routes, which
felt thorough and was not: `include_router` stores a wrapper whose real routes
hang off `original_router`, and a `Mount` hides a whole second application. The
four routers this server includes — v17, admin, capture, process — and all of
Atlas were invisible to it. `_walk` descends through both, which is the
difference between checking 6 endpoints and checking 138.

Only parameter-free GETs are called: they are the ones a browser hits on its
own, and they are safe to call in any order. A GET that needs a `{video_id}`
has no correct value to pass without inventing state, and a POST is not a thing
you probe.
"""
import os, tempfile

BASE = tempfile.mkdtemp()
os.environ.update(VIOS_BASE_DIR=BASE, ATLAS_HOME=os.path.join(BASE, "atlas"),
                  ATLAS_CACHE_DIR=os.path.join(BASE, "cache"), VIOS_OMNI="0")
os.makedirs(os.environ["ATLAS_HOME"], exist_ok=True)

from fastapi.testclient import TestClient
import ui_server, sitemap


def _walk(routes, prefix=""):
    """Every (path, methods) the app can serve, mounts and routers included."""
    out = []
    for r in routes:
        inner = getattr(r, "original_router", None)      # app.include_router(...)
        if inner is not None:
            ctx = getattr(r, "include_context", None)
            out += _walk(inner.routes, prefix + (getattr(ctx, "prefix", "") or ""))
        elif hasattr(r, "methods") and hasattr(r, "path"):
            out.append((prefix + r.path,
                        {m for m in r.methods if m not in ("HEAD", "OPTIONS")}))
        elif hasattr(r, "routes") or getattr(r, "app", None) is not None:
            sub = getattr(r, "routes", None) or getattr(r.app, "routes", [])
            out += _walk(sub, prefix + (getattr(r, "path", "") or ""))
    return out


c = TestClient(ui_server.app)
bad = []

print("== pages ==")
for path, title, _desc in sitemap.PAGES:
    r = c.get(path)
    footer = "sitemap.js" in r.text
    print(f"   {r.status_code}  {path:12s} {title:14s} footer={footer}")
    if r.status_code >= 500 or not footer:
        bad.append((path, r.status_code, f"footer={footer}"))

ALL = _walk(ui_server.app.routes)
GETS = sorted({p for p, m in ALL if "GET" in m and "{" not in p})
SKIP = {"/openapi.json", "/redoc", "/docs", "/sitemap.js", "/atlas/openapi.json",
        "/atlas/sitemap.js", "/atlas/favicon.ico"}
GETS = [p for p in GETS if p not in SKIP and not p.endswith((".css", ".js"))]

print(f"\n== {len(GETS)} parameter-free GETs of {len(ALL)} routes ==")
for path in GETS:
    try:
        r = c.get(path)
    except Exception as e:
        print(f"   RAISED  {path}: {type(e).__name__}: {e}")
        bad.append((path, "raised", f"{type(e).__name__}: {e}"[:120]))
        continue
    if r.status_code >= 500:
        print(f"   {r.status_code}  {path}   <-- {r.text[:110]}")
        bad.append((path, r.status_code, r.text[:160]))
    else:
        print(f"   {r.status_code}  {path}")

print("\n== the dense lane, on a host that refuses to load a model ==")
# A parameter-free GET never reaches `search._dense`: it needs `q`, and `_dense`
# returns early unless a vector matrix is resident. Both were true above, which
# is why the sweep passed while the dense path was broken.
#
# The defect: `encoder.get_encoder` guarded `import torch` with `except
# ImportError`. Torch can be installed and *forbidden* — under Windows Smart App
# Control its unsigned DLLs raise `OSError: [WinError 4551] An Application
# Control policy has blocked this file` — so on such a host the OSError left
# `get_encoder`, left `_dense` (which called it outside its own try), and left
# the request as a 500. Absent and refused are different facts and neither is a
# 500.
#
# The matrix is injected rather than built, the way `_t_vsearch.py` installs its
# resident index: that is the state the function actually reads, and building it
# for real would need a working encoder — the one thing this host cannot have.
#
# It is `EMBED_DIM` wide, not the `np.eye(4)` it used to be. Four columns only
# ever worked on a host that could not load a model, because such a host returns
# from `_dense` before the matmul. Run the same file on a host where torch *does*
# load and the encoder produced a 384-d query against a 4-wide matrix, which is
# `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
# (size 384 is different from 4)` raised straight out of the handler — a 500 on
# the one route the site is, found by running this test on the laptop rather than
# on Kaggle. `_dense` now checks the width and falls back to lexical, and the
# fixture is the right shape so this assertion means the same thing on both
# kinds of host.
import numpy as np
from atlas import config as _cfg, encoder as _enc, search as _search

_DIM = int(_cfg.EMBED_DIM)
with _search._VEC_LOCK:
    _search._VECTORS = np.eye(4, _DIM, dtype="float32")
    _search._VEC_IDS = np.arange(4, dtype="int64")
_enc._ENCODER, _enc._TRIED, _enc._ERROR = None, False, ""

try:
    r = c.get("/atlas/api/search", params={"q": "cooking pasta", "limit": 3})
    print(f"   {r.status_code}  /atlas/api/search?q=cooking+pasta")
    if r.status_code >= 500:
        bad.append(("/atlas/api/search?q=", r.status_code, r.text[:160]))
except Exception as e:
    print(f"   RAISED  /atlas/api/search: {type(e).__name__}: {e}")
    bad.append(("/atlas/api/search?q=", "raised", f"{type(e).__name__}: {e}"[:120]))

# A mismatched index is the other half of the same fault, and the one that
# actually happened on Kaggle: the run rebuilt the dense index ~76 times, so a
# file from an earlier encoder outliving the one now loaded is not hypothetical.
# The route must answer lexically, not 500.
with _search._VEC_LOCK:
    _search._VECTORS = np.eye(4, max(2, _DIM - 1), dtype="float32")
    _search._VEC_IDS = np.arange(4, dtype="int64")
try:
    r = c.get("/atlas/api/search", params={"q": "cooking pasta", "limit": 3})
    print(f"   {r.status_code}  /atlas/api/search?q=... with a {_DIM - 1}d index")
    if r.status_code >= 500:
        bad.append(("/atlas/api/search mismatched dim", r.status_code,
                    r.text[:160]))
except Exception as e:
    print(f"   RAISED  /atlas/api/search (mismatched dim): "
          f"{type(e).__name__}: {e}")
    bad.append(("/atlas/api/search mismatched dim", "raised",
                f"{type(e).__name__}: {e}"[:120]))
with _search._VEC_LOCK:
    _search._VECTORS = np.eye(4, _DIM, dtype="float32")
    _search._VEC_IDS = np.arange(4, dtype="int64")

# And the loader itself, which needs its own reset: `get_encoder` sets `_TRIED`
# *before* the import it is guarding, so the first caller to hit a broken host
# eats the exception and every caller after it gets the memoised None. At boot
# that first caller is `index._embed_all`, which has always guarded it — which is
# exactly why the narrow guard survived so long. Reset here or this assertion
# reads a cached answer and proves nothing.
_enc._ENCODER, _enc._TRIED, _enc._ERROR = None, False, ""
try:
    got = _enc.get_encoder()
    print(f"   get_encoder -> {'an encoder' if got else 'None'}; "
          f"reason: {(_enc.error() or 'none needed')[:80]}")
    if got is None and not _enc.error():
        bad.append(("encoder.get_encoder", "silent", "declined without saying why"))
except Exception as e:
    print(f"   RAISED  encoder.get_encoder: {type(e).__name__}: {e}")
    bad.append(("encoder.get_encoder", "raised", f"{type(e).__name__}: {e}"[:120]))

with _search._VEC_LOCK:
    _search._VECTORS = _search._VEC_IDS = None

print("\n== result ==")
if bad:
    for b in bad:
        print("   FAIL", b)
    raise SystemExit(1)
print(f"   {len(sitemap.PAGES)} pages + {len(GETS)} GETs, no 500s, footer everywhere")
