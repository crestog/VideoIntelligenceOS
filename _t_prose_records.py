"""Does the record test refuse the containers and keep what they carried?

`prose_columns` decides what search reads. The fifth test refuses a column of
serialized records, and the line it has to walk is narrow: `video.meta` is a
mapping and must go, `frame_notes.objects` is a list of detections and must stay,
and nothing a person wrote may move either way. Run against a copy of the real
archive, because the whole rule was written from what that archive actually holds.

    ATLAS_DB=<a copy of atlas.db> python _t_prose_records.py
"""
import os, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas import reflect, index

DB = os.environ.get("ATLAS_DB") or os.path.join(
    os.environ.get("LOCALAPPDATA", "/tmp"), "Temp", "viosmeas", "atlas.db")

# ── the test on its own, before any database ──────────────────────────────
CASES = [
    # a record: the container itself
    ('{"capture": {"msg_id": 30, "file_id": "BAACAgUAAyEGAAM", "likes": 134272}}', True),
    ('{"bytes": 5000000, "w": 1080}', True),
    ('{}', True),
    # a list: content that arrived in a container
    ('[{"label": "dog", "conf": 0.75}, {"label": "bed", "conf": 0.69}]', False),
    ('["desserts", "dinners"]', False),
    ('[0.013, 0.981, 0.44]', False),
    # prose, whatever it starts with
    ('{this is not json, it just opens with a brace}', False),
    ('today we are making a garlic butter steak', False),
    ('42', False),
    ('"a bare quoted string"', False),
    ('', False),
    (None, False),
]
for value, want in CASES:
    got = reflect._is_record(value)
    assert got is want, (value, got, want)
print(f"unit: {len(CASES)} cases OK")

if not os.path.exists(DB):
    print(f"no archive at {DB} — unit cases only")
    raise SystemExit(0)

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' "
    "AND name NOT LIKE 'sqlite_%' ORDER BY name")]

print("\n== what search would read ==")
sources, dropped = [], []
for t in tables:
    cols = reflect.columns(conn, t)
    names = [c["name"] for c in cols]
    content = reflect.content_columns(cols)
    prose = reflect.prose_columns(conn, t, cols)
    for c in content:
        rows = conn.execute(
            f'SELECT COUNT(*) FROM "{t}" WHERE "{c}" IS NOT NULL '
            f'AND TRIM("{c}") <> \'\'').fetchone()[0]
        (sources if c in prose else dropped).append((t, c, rows))
    del names
for t, c, n in sources:
    print(f"   keep {t:16s} {c:18s} {n:6d} rows")
print(f"\n   {len(sources)} text source(s) kept, {len(dropped)} refused")

kept = {(t, c) for t, c, _ in sources}
refused = {(t, c): n for t, c, n in dropped}

# ── the two the test exists for ───────────────────────────────────────────
print("\n== the containers ==")
for pair in (("video", "meta"), ("artifact", "meta")):
    n = refused.get(pair)
    print(f"   refused {pair[0]}.{pair[1]:6s} {n} rows")
    assert pair not in kept, f"{pair} is a mapping and is still indexed"

# ── and the one that must survive it ──────────────────────────────────────
print("\n== the content that arrived in a container ==")
if ("frame_notes", "objects") in kept or ("frame_notes", "objects") in refused:
    assert ("frame_notes", "objects") in kept, \
        "frame_notes.objects was dropped — 2,399 rows of object detections"
    raw = conn.execute("SELECT objects FROM frame_notes WHERE objects IS NOT NULL "
                       "AND TRIM(objects) <> '' LIMIT 3").fetchall()
    for r in raw:
        print(f"   raw     {r[0][:70]}")
        print(f"   indexed {index.clean_text(r[0])!r}")
        assert index.clean_text(r[0]), "unwrapped to nothing"
    print(f"   kept    frame_notes.objects "
          f"{[n for t, c, n in sources if (t, c) == ('frame_notes', 'objects')][0]} rows")
else:
    print("   this archive has no frame_notes.objects — skipped")

# ── nothing a person wrote moved ──────────────────────────────────────────
print("\n== the human columns ==")
for t, c in (("video", "title"), ("video", "uploader"), ("video", "caption"),
             ("claim", "value"), ("frame_notes", "description"),
             ("frame_notes", "ocr_text")):
    if (t, c) not in kept and (t, c) not in refused:
        continue
    print(f"   {'keep' if (t, c) in kept else 'REFUSED':7s} {t}.{c}")
    assert (t, c) in kept, f"{t}.{c} is prose and was refused"

print("\nRECORD TEST OK — containers refused, contents kept")
