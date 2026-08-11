"""
atlas.maps — the whole archive as one picture.

Three maps, built from what the database already holds, all of them clickable
down to the claim that produced the point.

  * **Semantic map** — every passage embedding projected to two dimensions.
    Distance on screen approximates distance in meaning, so the archive's
    actual shape becomes visible: the tight knot of gym reels, the long tail of
    one-off cooking clips, the bridge of videos that talk about both.
  * **Cluster map** — the same projection, coloured by k-means, with each
    cluster *named* from the words that distinguish its members. This is the
    part that answers "what is even in here" without anyone typing a query.
  * **Scatter plot** — any two numeric columns against each other, with the
    axis list derived from the schema rather than written here. Duration
    against likes, moment count against duration, cut rate against engagement:
    whatever the imported bundle happens to carry.

Why the projection is stored, not computed per request
──────────────────────────────────────────────────────
A projection is a global fit — every point's position depends on every other
point, so it cannot be computed for a viewport or a page. It is built once
against the whole matrix, written to `map_point`, and served from there. A
rebuild is triggered when the dense index changes, which is the same moment
the search ranking changes, so the map is never showing a different archive
than the search box is.

Degradation, deliberately
─────────────────────────
`moments.vec` only exists once the encoder has run (`index._embed_all`). When
there is no encoder — no GPU, no model download, a lexical-only session — the
semantic and cluster maps have nothing to project and say so. **The scatter
plot still works**, because it is pure SQL over `video_index` and needs no
embedding at all. One of the three maps is always available.

Which projection ran is reported, never hidden
──────────────────────────────────────────────
Three tiers, best available wins:

  1. **UMAP** — preserves neighbourhoods and global structure, has a real
     `transform()` so a subsample fit extends to the full set exactly.
  2. **t-SNE** — better local structure than PCA, no `transform()`, so
     out-of-sample points are placed by their nearest fitted neighbours.
  3. **PCA** — pure numpy via SVD. Always available, always fast, and honest
     about being a linear projection: it will show the big splits and blur the
     fine ones.

The tier used is stored with the map and shown in the interface, because a PCA
map and a UMAP map invite different conclusions and a viewer is entitled to
know which one they are reading.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time

from . import config, reflect
from .tgchannel import log

# ══════════════════════════════════════════════════════════════════════════
# TUNING
# ══════════════════════════════════════════════════════════════════════════
# Above this many points, the expensive projections fit on a random subsample
# and place the remainder by nearest-neighbour extension. UMAP on 200k rows is
# minutes; on 25k it is seconds, and the extension is accurate because the
# subsample already covers the space densely at that size.
FIT_SAMPLE = int(os.environ.get("ATLAS_MAP_FIT_SAMPLE", "25000"))
EXTEND_K = 12               # neighbours used to place an out-of-sample point

# Cluster count scales with the archive: too few and everything is "video",
# too many and the legend is unreadable. sqrt(n/2) is the standard rule of
# thumb, clamped to a range a person can actually scan.
MIN_CLUSTERS = 6
MAX_CLUSTERS = 48
KMEANS_ITERS = 60

# Words that cluster everything together and name nothing.
_STOP = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have he her hers him his how i if in into is it its me my not of on or our out
she so than that the their them then there these they this those to too us was
we were what when where which who why will with would you your it's don't
video reel clip shows showing seen see look looks like just really very much
one two three get got make made take taken thing things way
""".split())

_WORD = re.compile(r"[a-z][a-z'-]{2,}")

_DDL = (
    # One row per plotted point. `level` is what a point *is*: a whole video,
    # or one passage inside one. Both are built from the same fit so the two
    # views are registered against each other — a video sits where its
    # passages sit, and switching level does not teleport the map.
    "CREATE TABLE IF NOT EXISTS map_point ("
    "  level TEXT NOT NULL,"          # video | moment
    "  ref TEXT NOT NULL,"            # video_key, or moment id as text
    "  video_key TEXT NOT NULL,"
    "  x REAL NOT NULL,"
    "  y REAL NOT NULL,"
    "  cluster INTEGER,"
    "  t_start REAL,"                 # NULL at video level
    "  source TEXT,"
    "  PRIMARY KEY(level, ref))",

    "CREATE INDEX IF NOT EXISTS map_point_level ON map_point(level, cluster)",
    "CREATE INDEX IF NOT EXISTS map_point_video ON map_point(level, video_key)",
    # Box selection is a range scan on x, then a filter on y. Without this a
    # lasso over 180k points is a full table scan per drag frame.
    "CREATE INDEX IF NOT EXISTS map_point_xy ON map_point(level, x, y)",

    "CREATE TABLE IF NOT EXISTS map_cluster ("
    "  level TEXT NOT NULL,"
    "  cluster INTEGER NOT NULL,"
    "  label TEXT,"
    "  terms TEXT,"                   # JSON list of the distinguishing words
    "  size INTEGER,"
    "  videos INTEGER,"
    "  cx REAL, cy REAL,"
    "  PRIMARY KEY(level, cluster))",
)

_LOCK = threading.RLock()
_STATE = {
    "phase": "idle",     # idle | reading | fitting | clustering | writing | done | error | unavailable
    "detail": "",
    "method": "",
    "points": 0,
    "clusters": 0,
    "running": False,
    "error": "",
    "started_at": 0.0,
    "finished_at": 0.0,
    "at": 0.0,
}
_THREAD = None


def _set(**kw):
    with _LOCK:
        _STATE.update(kw)
        _STATE["at"] = time.time()


def status() -> dict:
    with _LOCK:
        return dict(_STATE)


def ensure_schema(conn: sqlite3.Connection) -> None:
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════
# NUMERIC HELPERS — numpy only, no scikit dependency on the critical path
# ══════════════════════════════════════════════════════════════════════════
def _pca2(np, mat):
    """Two principal components via SVD on the centred matrix.

    The always-available tier. `full_matrices=False` keeps the decomposition
    at (n, d) rather than (n, n), which is the difference between a few
    hundred megabytes and an allocation failure.
    """
    centred = mat - mat.mean(axis=0, keepdims=True)
    # Randomised range-finding: a 384-dim matrix does not need an exact SVD to
    # give an exact-enough top-2, and the exact one over 180k rows is slow.
    try:
        rng = np.random.default_rng(0)
        probe = rng.standard_normal((centred.shape[1], 8), dtype=np.float32)
        sketch = centred @ probe
        q, _ = np.linalg.qr(sketch)
        small = q.T @ centred
        _, _, vt = np.linalg.svd(small, full_matrices=False)
        return (centred @ vt[:2].T).astype(np.float32)
    except Exception:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        return (centred @ vt[:2].T).astype(np.float32)


def _extend(np, fit_vecs, fit_xy, rest_vecs, k=EXTEND_K, block=2048):
    """Place unfitted points at the weighted centroid of their k nearest fitted
    neighbours.

    The honest out-of-sample rule for a projection that has no `transform()`.
    A point lands among the points it is most similar to, which is the only
    claim the map makes anyway. Done in blocks because the full similarity
    matrix would be (rest x fit) — tens of gigabytes at archive scale.
    """
    out = np.zeros((len(rest_vecs), 2), dtype=np.float32)
    fit_n = fit_vecs / (np.linalg.norm(fit_vecs, axis=1, keepdims=True) + 1e-9)
    for i in range(0, len(rest_vecs), block):
        chunk = rest_vecs[i:i + block]
        chunk_n = chunk / (np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-9)
        sim = chunk_n @ fit_n.T
        kk = min(k, sim.shape[1])
        idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        rows = np.arange(len(chunk))[:, None]
        w = np.clip(sim[rows, idx], 0.0, None) ** 3      # sharpen: near wins
        w_sum = w.sum(axis=1, keepdims=True)
        w = np.where(w_sum > 1e-9, w / np.maximum(w_sum, 1e-9), 1.0 / kk)
        out[i:i + block] = (fit_xy[idx] * w[:, :, None]).sum(axis=1)
    return out


def _project(np, mat, want: str = "auto"):
    """(xy, method_name). Best available of UMAP → t-SNE → PCA."""
    n = len(mat)
    fit_idx = None
    if n > FIT_SAMPLE:
        rng = np.random.default_rng(0)        # seeded: a rebuild is reproducible
        fit_idx = np.sort(rng.choice(n, FIT_SAMPLE, replace=False))
    sub = mat if fit_idx is None else mat[fit_idx]

    if want in ("auto", "umap"):
        try:
            import umap                                  # noqa: PLC0415
            _set(detail=f"UMAP over {len(sub)} vector(s)")
            red = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.12,
                            metric="cosine", random_state=0, verbose=False)
            xy_sub = red.fit_transform(sub).astype("float32")
            if fit_idx is None:
                return xy_sub, "umap"
            _set(detail=f"UMAP transform for the remaining "
                        f"{n - len(sub)} vector(s)")
            xy = np.zeros((n, 2), dtype=np.float32)
            xy[fit_idx] = xy_sub
            mask = np.ones(n, dtype=bool)
            mask[fit_idx] = False
            # UMAP can transform out-of-sample directly, which beats the
            # neighbour extension — use it and fall back only if it throws.
            try:
                xy[mask] = red.transform(mat[mask]).astype("float32")
            except Exception as e:
                log(f"map: umap.transform failed ({type(e).__name__}) — "
                    f"placing the rest by nearest neighbours")
                xy[mask] = _extend(np, sub, xy_sub, mat[mask])
            return xy, "umap"
        except ImportError:
            pass
        except Exception as e:
            log(f"map: UMAP failed ({type(e).__name__}: {e}) — trying t-SNE")

    if want in ("auto", "tsne"):
        try:
            from sklearn.manifold import TSNE            # noqa: PLC0415
            _set(detail=f"t-SNE over {len(sub)} vector(s)")
            per = max(5.0, min(40.0, len(sub) / 100.0))
            red = TSNE(n_components=2, perplexity=per, metric="cosine",
                       init="pca", random_state=0, n_iter=750)
            xy_sub = red.fit_transform(sub).astype("float32")
            if fit_idx is None:
                return xy_sub, "tsne"
            _set(detail=f"placing the remaining {n - len(sub)} vector(s)")
            xy = np.zeros((n, 2), dtype=np.float32)
            xy[fit_idx] = xy_sub
            mask = np.ones(n, dtype=bool)
            mask[fit_idx] = False
            xy[mask] = _extend(np, sub, xy_sub, mat[mask])
            return xy, "tsne"
        except ImportError:
            pass
        except Exception as e:
            log(f"map: t-SNE failed ({type(e).__name__}: {e}) — using PCA")

    _set(detail=f"PCA over {n} vector(s)")
    return _pca2(np, mat), "pca"


def _kmeans(np, xy, vecs, k):
    """k-means++ over the *embeddings*, not the projection.

    Clustering the 2D positions would cluster the projection's artefacts. The
    projection is a lossy view; the vectors are the evidence, so the grouping
    is computed where the meaning actually lives and only *displayed* on the
    map. Where a cluster looks scattered on screen, that is the projection
    being honest about what it could not flatten.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans      # noqa: PLC0415
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                             batch_size=2048, max_iter=200)
        return km.fit_predict(vecs).astype("int32")
    except ImportError:
        pass
    except Exception as e:
        log(f"map: MiniBatchKMeans failed ({type(e).__name__}) — using numpy")

    n = len(vecs)
    rng = np.random.default_rng(0)
    unit = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    # k-means++ seeding, on a sample when the archive is large: the seeding
    # pass is O(n·k) and dominates everything else at 180k points.
    seed_pool = unit if n <= 20000 else unit[rng.choice(n, 20000, replace=False)]
    centres = [seed_pool[rng.integers(len(seed_pool))]]
    d2 = np.full(len(seed_pool), np.inf, dtype=np.float32)
    for _ in range(k - 1):
        d2 = np.minimum(d2, 1.0 - seed_pool @ centres[-1])
        total = float(d2.sum())
        if not (total > 0) or not np.isfinite(total):
            centres.append(seed_pool[rng.integers(len(seed_pool))])
            continue
        centres.append(seed_pool[int(np.searchsorted(
            np.cumsum(d2 / total), rng.random()))])
    cent = np.asarray(centres, dtype=np.float32)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(KMEANS_ITERS):
        moved = False
        for i in range(0, n, 8192):
            block = unit[i:i + 8192]
            new = np.argmax(block @ cent.T, axis=1).astype(np.int32)
            if not moved and not np.array_equal(new, labels[i:i + 8192]):
                moved = True
            labels[i:i + 8192] = new
        if not moved:
            break
        for c in range(k):
            members = unit[labels == c]
            if len(members):
                v = members.mean(axis=0)
                cent[c] = v / (np.linalg.norm(v) + 1e-9)
    return labels


def _spread(np, xy):
    """Rescale to [0,1]² on percentiles, not min/max.

    Every projection produces a handful of far-flung outliers, and scaling to
    the true extremes compresses 99% of the archive into the middle 5% of the
    canvas — the single most common way a map like this looks broken. Clipping
    at the 1st and 99th percentile spends the canvas on the body of the data;
    the outliers are still drawn, pinned at the edge where they belong.
    """
    out = np.zeros_like(xy)
    for a in (0, 1):
        col = xy[:, a]
        lo, hi = np.percentile(col, 1.0), np.percentile(col, 99.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-9:
            out[:, a] = 0.5
        else:
            out[:, a] = np.clip((col - lo) / (hi - lo), -0.04, 1.04)
    return out


# ══════════════════════════════════════════════════════════════════════════
# NAMING THE CLUSTERS
# ══════════════════════════════════════════════════════════════════════════
def _label_clusters(texts: list, labels, k: int) -> list:
    """The words that make each cluster *different*, not the words it contains.

    Plain frequency names every cluster "the", and dropping stopwords only
    moves the problem to "video" and "shows". The score used here is a term's
    share inside the cluster against its share across the whole archive, so a
    word only surfaces when this cluster is where it lives. That is what makes
    the legend read like a table of contents rather than a word cloud.
    """
    from collections import Counter
    per = [Counter() for _ in range(k)]
    overall = Counter()
    sizes = [0] * k

    for text, c in zip(texts, labels):
        c = int(c)
        if not (0 <= c < k):
            continue
        sizes[c] += 1
        seen = set()
        for w in _WORD.findall((text or "").lower()):
            if w in _STOP or w in seen:
                continue
            seen.add(w)
            per[c][w] += 1
            overall[w] += 1

    total_docs = max(1, sum(sizes))
    out = []
    for c in range(k):
        n = max(1, sizes[c])
        scored = []
        for w, count in per[c].most_common(400):
            if count < 2:
                continue
            inside = count / n
            outside = max(1e-9, (overall[w] - count) / max(1, total_docs - n))
            # Lift, damped by how much evidence there is for it. Without the
            # log a word appearing twice in a small cluster outranks a word
            # appearing four hundred times in a large one.
            scored.append((inside / (inside + outside) * math.log1p(count), w))
        scored.sort(reverse=True)
        terms = [w for _, w in scored[:8]]
        out.append({"terms": terms,
                    "label": " · ".join(terms[:3]) if terms else f"group {c + 1}",
                    "size": sizes[c]})
    return out


# ══════════════════════════════════════════════════════════════════════════
# THE BUILD
# ══════════════════════════════════════════════════════════════════════════
def start_build(db_path: str = None, method: str = "auto") -> bool:
    """Kick the build off in the background. False if one is already running."""
    global _THREAD
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return False
        _THREAD = threading.Thread(
            target=_build, args=(db_path or config.DB_PATH, method),
            name="atlas-map", daemon=True)
        _THREAD.start()
    return True


def _build(db_path: str, method: str = "auto") -> None:
    _set(phase="reading", running=True, error="", started_at=time.time(),
         finished_at=0.0, points=0, clusters=0, method="",
         detail="loading the dense index")
    conn = None
    try:
        import numpy as np
    except ImportError:
        _set(phase="unavailable", running=False, finished_at=time.time(),
             detail="numpy is not installed — the scatter plot still works")
        return

    try:
        from . import index as index_mod
        meta = index_mod.vector_state()
        if not meta or not meta.get("count"):
            _set(phase="unavailable", running=False, finished_at=time.time(),
                 detail="no dense index yet — the semantic and cluster maps "
                        "need the encoder to have run. The scatter plot works "
                        "without it.")
            return

        dim = int(meta.get("dim") or config.EMBED_DIM)
        vecs = np.fromfile(config.VECTOR_PATH, dtype=np.float32)
        ids = np.fromfile(config.VECTOR_PATH + ".ids", dtype=np.int64)
        if dim <= 0 or vecs.size % dim:
            raise ValueError(f"vector file is not a multiple of {dim} floats")
        vecs = vecs.reshape(-1, dim)
        if len(ids) != len(vecs):
            raise ValueError(f"vector/id mismatch: {len(vecs)} vs {len(ids)}")

        conn = sqlite3.connect(db_path, timeout=60.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)

        # Only the moments that still exist. A reindex renumbers `moments`,
        # and projecting an id that has been deleted would put a point on the
        # map that no click could ever resolve.
        rows = conn.execute(
            "SELECT id, video_key, t_start, source, text FROM moments"
        ).fetchall()
        by_id = {int(r["id"]): r for r in rows}
        keep = np.array([i for i, m in enumerate(ids) if int(m) in by_id],
                        dtype=np.int64)
        if len(keep) < 4:
            _set(phase="unavailable", running=False, finished_at=time.time(),
                 detail="not enough indexed passages to draw a map")
            return
        if len(keep) < len(ids):
            log(f"map: {len(ids) - len(keep)} vector(s) belong to passages that "
                f"no longer exist — rebuilding the dense index would clear them")
        vecs = vecs[keep]
        ids = ids[keep]
        picked = [by_id[int(m)] for m in ids]

        _set(phase="fitting",
             detail=f"projecting {len(vecs)} passage(s) to two dimensions")
        xy, used = _project(np, vecs, method)
        xy = _spread(np, xy)
        _set(method=used)

        k = int(max(MIN_CLUSTERS,
                    min(MAX_CLUSTERS, round(math.sqrt(len(vecs) / 2.0)))))
        _set(phase="clustering", detail=f"finding {k} group(s) of meaning")
        labels = _kmeans(np, xy, vecs, k)
        named = _label_clusters([r["text"] for r in picked], labels, k)

        # ── video level: the mean position of a video's passages ──────────
        # Computed from the same fit rather than a second projection, so the
        # two levels are the same map at two resolutions. A video sits at the
        # centre of gravity of everything it says.
        _set(phase="writing", detail="summarising videos onto the map")
        agg = {}
        for r, (px, py), c in zip(picked, xy, labels):
            slot = agg.setdefault(r["video_key"], {"x": 0.0, "y": 0.0, "n": 0,
                                                   "c": {}})
            slot["x"] += float(px)
            slot["y"] += float(py)
            slot["n"] += 1
            slot["c"][int(c)] = slot["c"].get(int(c), 0) + 1

        video_rows = []
        for key, slot in agg.items():
            n = max(1, slot["n"])
            # A video's cluster is the one most of its passages fell into —
            # a plurality vote, not an average, because averaging cluster
            # numbers is meaningless.
            top = max(slot["c"].items(), key=lambda kv: kv[1])[0]
            video_rows.append(("video", key, key, slot["x"] / n, slot["y"] / n,
                               top, None, None))

        moment_rows = [
            ("moment", str(int(m)), r["video_key"], float(px), float(py),
             int(c), r["t_start"], r["source"])
            for m, r, (px, py), c in zip(ids, picked, xy, labels)]

        conn.execute("DELETE FROM map_point")
        conn.execute("DELETE FROM map_cluster")
        conn.executemany(
            "INSERT OR REPLACE INTO map_point"
            "(level, ref, video_key, x, y, cluster, t_start, source) "
            "VALUES (?,?,?,?,?,?,?,?)", moment_rows + video_rows)

        cl_rows = []
        for level in ("moment", "video"):
            centres = conn.execute(
                "SELECT cluster, COUNT(*) n, COUNT(DISTINCT video_key) v,"
                "       AVG(x) cx, AVG(y) cy FROM map_point "
                "WHERE level=? GROUP BY cluster", (level,)).fetchall()
            for row in centres:
                c = int(row["cluster"])
                info = named[c] if 0 <= c < len(named) else {"terms": [],
                                                             "label": ""}
                cl_rows.append((level, c, info["label"] or f"group {c + 1}",
                                json.dumps(info["terms"]), int(row["n"]),
                                int(row["v"]), float(row["cx"]),
                                float(row["cy"])))
        conn.executemany(
            "INSERT OR REPLACE INTO map_cluster"
            "(level, cluster, label, terms, size, videos, cx, cy) "
            "VALUES (?,?,?,?,?,?,?,?)", cl_rows)

        from .ingest import meta_set
        meta_set(conn, "map_built_at", time.time())
        meta_set(conn, "map_method", used)
        meta_set(conn, "map_points", len(moment_rows))
        meta_set(conn, "map_videos", len(video_rows))
        meta_set(conn, "map_clusters", k)
        conn.commit()

        _set(phase="done", running=False, finished_at=time.time(),
             points=len(moment_rows), clusters=k,
             detail=f"{len(moment_rows)} passage(s) and {len(video_rows)} "
                    f"video(s) mapped in {k} group(s) · {used}")
        log(f"map built — {len(moment_rows)} points, {len(video_rows)} videos, "
            f"{k} clusters, method={used}")
    except Exception as e:
        _set(phase="error", running=False, finished_at=time.time(),
             error=f"{type(e).__name__}: {e}", detail="map build failed")
        log(f"map build failed — {type(e).__name__}: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════
# READING THE MAP
# ══════════════════════════════════════════════════════════════════════════
def built(conn: sqlite3.Connection) -> bool:
    try:
        return bool(conn.execute(
            "SELECT 1 FROM map_point LIMIT 1").fetchone())
    except sqlite3.Error:
        return False


def meta(conn: sqlite3.Connection, level: str = "video") -> dict:
    """Everything needed to draw the legend and size the canvas."""
    ensure_schema(conn)
    level = "moment" if str(level) == "moment" else "video"
    from .ingest import meta_get
    clusters = [
        {"cluster": int(r["cluster"]), "label": r["label"],
         "terms": json.loads(r["terms"] or "[]"), "size": int(r["size"] or 0),
         "videos": int(r["videos"] or 0),
         "cx": float(r["cx"] or 0), "cy": float(r["cy"] or 0)}
        for r in conn.execute(
            "SELECT * FROM map_cluster WHERE level=? ORDER BY size DESC",
            (level,)).fetchall()]
    n = conn.execute("SELECT COUNT(*) FROM map_point WHERE level=?",
                     (level,)).fetchone()[0]
    st = status()
    return {
        "ok": True,
        "level": level,
        "count": int(n),
        "clusters": clusters,
        "method": meta_get(conn, "map_method", "") or st.get("method", ""),
        "built_at": float(meta_get(conn, "map_built_at", 0) or 0),
        "status": st,
    }


def points_binary(conn: sqlite3.Connection, level: str = "video") -> bytes:
    """The point cloud as a packed buffer.

    Twelve bytes a point — two float32 coordinates and one int32 cluster —
    against roughly forty for the same thing as JSON. At archive scale that is
    the difference between a 2 MB response the canvas can draw immediately and
    a 7 MB one the browser spends half a second parsing. Row order is fixed
    and returned by `refs()`, so index *i* in this buffer is ref *i* there;
    that pairing is what makes a click resolvable without a round trip.
    """
    import struct
    level = "moment" if str(level) == "moment" else "video"
    rows = conn.execute(
        "SELECT x, y, cluster FROM map_point WHERE level=? ORDER BY rowid",
        (level,)).fetchall()
    out = bytearray()
    pack = struct.Struct("<ffi").pack
    for r in rows:
        out += pack(float(r["x"]), float(r["y"]), int(r["cluster"] or 0))
    return bytes(out)


def refs(conn: sqlite3.Connection, level: str = "video") -> dict:
    """The ref for every point, in the same order as `points_binary`."""
    level = "moment" if str(level) == "moment" else "video"
    rows = conn.execute(
        "SELECT ref, video_key, t_start FROM map_point WHERE level=? "
        "ORDER BY rowid", (level,)).fetchall()
    return {"ok": True, "level": level, "count": len(rows),
            "refs": [r["ref"] for r in rows],
            "keys": [r["video_key"] for r in rows],
            "t": [r["t_start"] for r in rows]}


def region(conn: sqlite3.Connection, level: str, x0: float, y0: float,
           x1: float, y1: float, limit: int = 500) -> dict:
    """What is inside a dragged box — the selection that feeds the other tabs."""
    level = "moment" if str(level) == "moment" else "video"
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    rows = conn.execute(
        "SELECT p.ref, p.video_key, p.t_start, p.cluster, p.source,"
        "       v.title, v.creator, v.duration, v.moment_count "
        "FROM map_point p LEFT JOIN video_index v ON v.video_key = p.video_key "
        "WHERE p.level=? AND p.x BETWEEN ? AND ? AND p.y BETWEEN ? AND ? "
        "ORDER BY v.moment_count DESC LIMIT ?",
        (level, lo_x, hi_x, lo_y, hi_y, max(1, min(2000, int(limit))))
    ).fetchall()
    keys, seen = [], set()
    for r in rows:
        if r["video_key"] not in seen:
            seen.add(r["video_key"])
            keys.append(r["video_key"])
    return {"ok": True, "level": level, "count": len(rows),
            "videos": len(keys), "keys": keys,
            "items": [dict(r) for r in rows]}


def point(conn: sqlite3.Connection, level: str, ref: str) -> dict:
    """Everything behind one dot — the drill-down the maps exist to reach.

    A dot on a scatter plot is worthless if it cannot say what it is. This is
    the same contract the Data tab makes: a click lands on the row, the passage
    text, the model that produced it, and the second it happens at.
    """
    level = "moment" if str(level) == "moment" else "video"
    row = conn.execute(
        "SELECT * FROM map_point WHERE level=? AND ref=?",
        (level, str(ref))).fetchone()
    if not row:
        return {"ok": False, "note": "no such point"}

    key = row["video_key"]
    video = conn.execute("SELECT * FROM video_index WHERE video_key=?",
                         (key,)).fetchone()
    out = {"ok": True, "level": level, "ref": str(ref), "video_key": key,
           "x": float(row["x"]), "y": float(row["y"]),
           "cluster": int(row["cluster"] or 0),
           "t_start": row["t_start"], "source": row["source"],
           "video": dict(video) if video else {}}

    cl = conn.execute(
        "SELECT * FROM map_cluster WHERE level=? AND cluster=?",
        (level, int(row["cluster"] or 0))).fetchone()
    if cl:
        out["cluster_info"] = {"label": cl["label"],
                               "terms": json.loads(cl["terms"] or "[]"),
                               "size": int(cl["size"] or 0),
                               "videos": int(cl["videos"] or 0)}

    if level == "moment":
        m = conn.execute(
            "SELECT id, video_key, t_start, t_end, source, src_table, weight,"
            "       text FROM moments WHERE id=?", (int(ref),)).fetchone()
        if m:
            out["moment"] = dict(m)
            # The exact query that produced the row, so the claim is
            # verifiable rather than merely displayed.
            out["sql"] = f"SELECT * FROM moments WHERE id = {int(ref)};"
    else:
        out["moments"] = [dict(r) for r in conn.execute(
            "SELECT id, t_start, t_end, source, text FROM moments "
            "WHERE video_key=? ORDER BY COALESCE(t_start, 0) LIMIT 40",
            (key,)).fetchall()]
        out["sql"] = ("SELECT * FROM moments WHERE video_key = "
                      f"'{str(key)[:80]}';")
    return out


def cluster_detail(conn: sqlite3.Connection, level: str, cluster: int,
                   limit: int = 30) -> dict:
    """One cluster: what names it, and the videos most central to it."""
    level = "moment" if str(level) == "moment" else "video"
    cl = conn.execute("SELECT * FROM map_cluster WHERE level=? AND cluster=?",
                      (level, int(cluster))).fetchone()
    if not cl:
        return {"ok": False, "note": "no such cluster"}
    # Ranked by distance from the cluster centre: the most typical members
    # first, which is what "show me this group" should mean.
    rows = conn.execute(
        "SELECT p.video_key, p.ref, p.t_start,"
        "       ((p.x-?)*(p.x-?) + (p.y-?)*(p.y-?)) d,"
        "       v.title, v.creator, v.duration, v.moment_count "
        "FROM map_point p LEFT JOIN video_index v ON v.video_key=p.video_key "
        "WHERE p.level=? AND p.cluster=? ORDER BY d ASC LIMIT ?",
        (cl["cx"], cl["cx"], cl["cy"], cl["cy"], level, int(cluster),
         max(1, min(200, int(limit))))).fetchall()
    return {"ok": True, "level": level, "cluster": int(cluster),
            "label": cl["label"], "terms": json.loads(cl["terms"] or "[]"),
            "size": int(cl["size"] or 0), "videos": int(cl["videos"] or 0),
            "items": [dict(r) for r in rows]}


# ══════════════════════════════════════════════════════════════════════════
# THE SCATTER PLOT — schema-derived, no embedding required
# ══════════════════════════════════════════════════════════════════════════
# Columns that are numeric but plotting them means nothing: an id is an
# arbitrary integer and a hash is noise. Everything else numeric is offered.
_AXIS_SKIP = frozenset(("id", "rowid", "msg_id", "record_msg_id", "text_hash",
                        "seq", "cluster"))

_NICE = {
    "duration": "duration (s)", "size_mb": "file size (MB)",
    "moment_count": "indexed moments", "text_len": "text indexed (chars)",
    "created_at": "captured", "likes": "likes", "fps": "frames per second",
    "width": "width (px)", "height": "height (px)",
    "has_speech": "has speech", "has_narrative": "has narrative",
}


def axes(conn: sqlite3.Connection) -> dict:
    """Every numeric column worth plotting, discovered from the schema.

    Nothing here is a hardcoded list of fields — a bundle that arrives with a
    `cut_rate` or an `engagement` column gets those axes automatically, which
    is the same rule the rest of the server follows.
    """
    out = []
    for col in reflect.columns(conn, "video_index"):
        name = col["name"]
        if name in _AXIS_SKIP or not reflect._is_numeric(col):
            continue
        row = conn.execute(
            f"SELECT COUNT({reflect._q(name)}) n, MIN({reflect._q(name)}) lo,"
            f"       MAX({reflect._q(name)}) hi FROM video_index").fetchone()
        if not row or not row["n"] or row["lo"] is None:
            continue
        if float(row["hi"]) - float(row["lo"]) < 1e-12:
            continue                      # a constant column plots as a line
        out.append({"name": name, "label": _NICE.get(name, name.replace("_", " ")),
                    "count": int(row["n"]), "min": float(row["lo"]),
                    "max": float(row["hi"])})
    out.sort(key=lambda a: -a["count"])
    return {"ok": True, "axes": out,
            "colour_by": ["cluster", "creator", "category", "source"]}


def scatter(conn: sqlite3.Connection, x: str, y: str, colour: str = "cluster",
            limit: int = 6000, log_x: bool = False,
            log_y: bool = False) -> dict:
    """Two numeric columns against each other, every point clickable.

    Validated against the live schema rather than a whitelist, because the
    whole point is that a bundle with new columns becomes plottable without
    anyone editing this file. `reflect._q` quotes the identifier and the
    membership check below is what makes that safe — the name has to already
    exist as a column before it can reach the SQL.
    """
    cols = {c["name"]: c for c in reflect.columns(conn, "video_index")}
    if x not in cols or y not in cols:
        return {"ok": False, "note": f"unknown axis: {x if x not in cols else y}"}
    if not (reflect._is_numeric(cols[x]) and reflect._is_numeric(cols[y])):
        return {"ok": False, "note": "both axes must be numeric"}

    qx, qy = reflect._q(x), reflect._q(y)
    join, pick = "", ""
    if colour == "cluster" and built(conn):
        join = ("LEFT JOIN map_point m ON m.video_key = v.video_key "
                "AND m.level='video' ")
        pick = ", m.cluster AS _c"
    elif colour in cols:
        pick = f", v.{reflect._q(colour)} AS _c"

    rows = conn.execute(
        f"SELECT v.video_key, v.title, v.creator, v.category, v.duration,"
        f"       v.moment_count, {qx} AS _x, {qy} AS _y{pick} "
        f"FROM video_index v {join}"
        f"WHERE {qx} IS NOT NULL AND {qy} IS NOT NULL "
        f"ORDER BY v.moment_count DESC LIMIT ?",
        (max(1, min(20000, int(limit))),)).fetchall()

    pts, groups = [], {}
    for r in rows:
        vx, vy = float(r["_x"]), float(r["_y"])
        if log_x:
            vx = math.log10(vx) if vx > 0 else None
        if log_y:
            vy = math.log10(vy) if vy > 0 else None
        if vx is None or vy is None:
            continue            # a log axis cannot show zero or a negative
        g = r["_c"] if "_c" in r.keys() else None
        g = "—" if g is None or g == "" else str(g)
        if g not in groups:
            groups[g] = len(groups)
        pts.append({"key": r["video_key"], "x": vx, "y": vy, "g": groups[g],
                    "title": r["title"] or r["video_key"],
                    "creator": r["creator"], "n": r["moment_count"]})

    return {"ok": True, "x": x, "y": y, "colour": colour,
            "log_x": bool(log_x), "log_y": bool(log_y),
            "x_label": _NICE.get(x, x.replace("_", " ")),
            "y_label": _NICE.get(y, y.replace("_", " ")),
            "count": len(pts), "points": pts,
            "groups": [k for k, _ in sorted(groups.items(), key=lambda kv: kv[1])],
            "note": ("" if len(pts) == len(rows)
                     else f"{len(rows) - len(pts)} video(s) dropped — a log "
                          f"axis cannot plot zero or negative values")}
