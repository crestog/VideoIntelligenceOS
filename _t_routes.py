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

print("\n== result ==")
if bad:
    for b in bad:
        print("   FAIL", b)
    raise SystemExit(1)
print(f"   {len(sitemap.PAGES)} pages + {len(GETS)} GETs, no 500s, footer everywhere")
