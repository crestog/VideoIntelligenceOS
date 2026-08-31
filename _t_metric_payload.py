"""Does a packed measurement survive the trip, and does the log name what was lost?

Two defects, both measured in this laptop's atlas.db — 151 MB, the output of the
Kaggle sessions under investigation:

  594 of 1,909 `frame_metric` rows have `values_ IS NULL`, `noise` and `exposure`
  at 30 of 30. `frame_metric` is not in `_PAYLOAD_KINDS`, so nothing rehomed its
  bytes and the generic opaque-column drop was the whole story for it. Which rows
  survived was decided by shard composition rather than by anything about the
  metric: `_OPAQUE` needs 256 characters, a float32 array of 32 frames is exactly
  256 hex characters, and `_is_opaque` drops a column only when *every* value in
  it is long — so one short metric sharing a shard saved every long one beside it.

  The line reporting it read `dropped N opaque column(s)` for all five causes
  that write to that list, counting list entries rather than rows. Twelve
  thousand vectors lost to a SQLite error and one harmless reserved table name
  both printed `dropped 1 opaque column(s)`.

Seven groups. The engine writes real shards; from the import on, only Atlas.
"""
import os, struct, tempfile, shutil

BASE = tempfile.mkdtemp()
os.environ["VIOS_BASE_DIR"] = BASE
os.environ["ATLAS_HOME"] = os.path.join(BASE, "atlas")
os.environ["ATLAS_CACHE_DIR"] = os.path.join(BASE, "cache")
os.environ["VIOS_OMNI"] = "0"
os.makedirs(os.environ["ATLAS_HOME"], exist_ok=True)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"   ok   {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL += 1
        print(f"   FAIL {name}  [{detail}]")


def f32(vals):
    """The bytes the engine packs, so the test compares numbers to numbers."""
    return struct.pack(f"<{len(vals)}f", *vals)


# ── the engine side ───────────────────────────────────────────────────────
from vios.process import store as S, intake

st = S.Store(os.path.join(BASE, "ev.db"))
st.add_video(video_key="REEL1", url="https://instagram.com/p/REEL1/",
             duration=30.0, width=1080, height=1920, meta={})
st.set_shots("REEL1", [{"t0": 0.0, "t1": 15.0}, {"t0": 15.0, "t1": 30.0}],
             "pyscenedetect")
oid = st.observer("signal", "opencv", "v1", {})
st.add_claims("REEL1", oid, [
    {"channel": "visual", "kind": "note", "shot_idx": 0,
     "value": "a hand turns a steak in a black pan"}])
st.add_vector("REEL1", "clip", [0.01 * i for i in range(384)], oid, shot_idx=0)

# 64 frames of float32 is 512 hex characters — over `_OPAQUE`'s 256 — so on its
# own this column is unanimously long and the old code dropped it outright.
LONG = {"sharpness": [round(0.4 + 0.001 * i, 4) for i in range(64)],
        "noise": [round(0.02 * (i % 7), 4) for i in range(64)]}
# 8 frames is 64 hex characters. Not opaque, and under the old rule its presence
# in a shard is what saved the two above.
SHORT = {"hue": [round(30.0 + i, 3) for i in range(8)]}

for name, vals in LONG.items():
    st.add_frame_metric("REEL1", name, list(range(len(vals))), vals, oid)

sid1 = f"{intake.site_id(st)}-0001"
shard1 = os.path.join(BASE, intake.shard_name(sid1))
s1 = st.export_shard(shard1, 0, st.max_claim_id(), 0, st.max_vector_id(),
                     "gpu", 0, st.max_frame_vector_id(), 0,
                     st.max_frame_metric_id())
print(f"\nengine wrote {intake.shard_name(sid1)} — {s1['claims']} claims, "
      f"{s1['vectors']} vectors, {s1['frame_metrics']} metrics, {s1['bytes']}b")

# ── from here on, only Atlas ──────────────────────────────────────────────
from atlas import ingest, reflect, tgchannel

MSG = {"message_id": 6100, "date": 1754700000,
       "caption": f"vios evidence · {sid1}\n1 claims, 1 vectors",
       "document": {"file_name": intake.shard_name(sid1), "file_id": "FAKE",
                    "file_size": s1["bytes"], "mime_type": "application/gzip"}}
info1 = tgchannel.message_document(MSG)
tgchannel.fetch_document = lambda i, dest, _p=[shard1]: \
    bool(shutil.copyfile(_p[0], dest)) or True

conn = ingest.connect(os.path.join(os.environ["ATLAS_HOME"], "atlas.sqlite"))
ingest.ensure_meta(conn)


def cols_of(table):
    return {r[1]: r[2] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def one(sql, *a):
    r = conn.execute(sql, a).fetchone()
    return r[0] if r else None


print("\n== 1. the column the old build dropped is really opaque ==")
hexed = [f32(v).hex() for v in LONG.values()]
check("a 64-frame float32 metric hex-encodes past _OPAQUE's 256 characters",
      all(len(h) >= 256 for h in hexed), f"{len(hexed[0])} chars")
check("so _is_opaque calls the column a payload — the guarded path is the one "
      "under test", ingest._is_opaque(hexed), "unanimously long")
check("and a short metric is not opaque, which is what used to save its "
      "neighbours", not ingest._is_opaque([f32(v).hex() for v in SHORT.values()]),
      f"{len(f32(SHORT['hue']).hex())} chars")
check("frame_metric is not a payload kind, so nothing rehomes its bytes",
      "frame_metric" not in ingest._PAYLOAD_KINDS, str(ingest._PAYLOAD_KINDS))

print("\n== 2. the numbers arrive, and they are the numbers ==")
r1 = ingest.import_shard(info1, conn, BASE)
check("the shard imported", r1.get("ok"), str(r1)[:90])
mc = cols_of("frame_metric")
check("frame_metric.values_ exists in Atlas", "values_" in mc, str(sorted(mc)))
check("and is TEXT, which is what reflect.py's type tests read",
      mc.get("values_") == "TEXT", str(mc.get("values_")))
nulls = one("SELECT COUNT(*) FROM frame_metric WHERE values_ IS NULL")
check("no metric row asserts a measurement it does not carry", nulls == 0,
      f"{nulls} null of "
      f"{one('SELECT COUNT(*) FROM frame_metric')}")
back = {}
for row in conn.execute("SELECT name, values_ FROM frame_metric"):
    raw = bytes.fromhex(row[1])
    back[row[0]] = [round(x, 4) for x in
                    struct.unpack(f"<{len(raw) // 4}f", raw)]
check("and the floats round-trip through hex unchanged", back == LONG,
      f"{sorted(back)} · {len(back.get('sharpness') or [])} values")

print("\n== 3. the two tables that ARE rehomed still lose their columns ==")
vc = cols_of("vector")
check("vector.data is still dropped from the reflected table",
      "data" not in vc, str(sorted(vc)))
kept = one("SELECT COUNT(*) FROM vec_payload WHERE kind='vector'")
check("because the bytes are in vec_payload instead, exactly once", kept == 1,
      f"{kept} row(s)")
check("a second hex copy would have cost this archive 240 MB to say nothing new",
      one("SELECT SUM(LENGTH(data)) FROM vec_payload") == 384 * 4,
      f"{one('SELECT SUM(LENGTH(data)) FROM vec_payload')} bytes")

print("\n== 4. keeping the column does not put numbers into search ==")
# The reason keeping it is safe rather than merely honest: reflect.py was already
# written for this state. `prose_columns` names `frame_metric.values_` and
# `frame_metric.frames` as hex buffers to keep out of the index, having found
# 1,315 of them indexed as prose — the 1,315 being exactly the rows that
# survived the old drop by accident of shard composition.
names = reflect.columns(conn, "frame_metric")
prose = reflect.prose_columns(conn, "frame_metric", names)
check("values_ is not indexed as prose", "values_" not in prose, str(prose))
check("frames is not indexed as prose", "frames" not in prose, str(prose))
check("and neither is the label half of the pair", "name" not in prose,
      str(prose))
claim_prose = reflect.prose_columns(conn, "claim",
                                    reflect.columns(conn, "claim"))
check("while a claim's text still is", "value" in claim_prose,
      str(claim_prose))

print("\n== 5. survival no longer depends on what shared the shard ==")
for name, vals in SHORT.items():
    st.add_frame_metric("REEL1", name, list(range(len(vals))), vals, oid)
sid2 = f"{intake.site_id(st)}-0002"
shard2 = os.path.join(BASE, intake.shard_name(sid2))
s2 = st.export_shard(shard2, s1["hi_id"], st.max_claim_id(),
                     s1["hi_vec"], st.max_vector_id(), "gpu",
                     s1["hi_fvec"], st.max_frame_vector_id(),
                     s1["hi_fmet"], st.max_frame_metric_id())
tgchannel.fetch_document = lambda i, dest, _p=[shard2]: \
    bool(shutil.copyfile(_p[0], dest)) or True
info2 = dict(info1, file_name=intake.shard_name(sid2), message_id=6101,
             caption=f"vios evidence · {sid2}")
r2 = ingest.import_shard(info2, conn, BASE)
check("the mixed shard imported too", r2.get("ok"), str(r2)[:70])
check("a short metric lands beside the long ones",
      one("SELECT COUNT(*) FROM frame_metric") == 3,
      f"{one('SELECT COUNT(*) FROM frame_metric')} rows")
check("and still nothing is asserting a measurement it does not carry",
      one("SELECT COUNT(*) FROM frame_metric WHERE values_ IS NULL") == 0)
hue = one("SELECT values_ FROM frame_metric WHERE name='hue'")
check("the short one round-trips as well",
      [round(x, 3) for x in struct.unpack("<8f", bytes.fromhex(hue))]
      == SHORT["hue"], f"{len(hue)} hex chars")

print("\n== 6. the report counts rows, and names the cause ==")
note = ingest._drop_note
check("nothing dropped says nothing", note([]) == "", repr(note([])))
big = [("lost", 12000, "vector payload ×12000 (OperationalError: disk full)"),
       ("skip", 1, "moments (reserved name)")]
txt = note(big)
check("twelve thousand lost vectors are reported as twelve thousand",
      "lost 12000 row(s)" in txt, txt.strip())
check("and no longer share a word with a harmless reserved table",
      "skipped 1 row(s)" in txt and txt.index("lost") < txt.index("skipped"),
      txt.strip())
check("the old line called both of these `dropped 1 opaque column(s)`",
      f"dropped {len(big)} opaque column(s)" not in txt, txt.strip())
wide = [("skip", n, f"t{n}.c (opaque)") for n in (3, 90, 7, 40, 1, 500)]
txt = note(wide)
check("the widest cause is shown, not the first one the loop noticed",
      "t500.c" in txt and "t1.c" not in txt, txt.strip())
check("and what does not fit is counted rather than cut silently",
      "and 2 more cause(s)" in txt and "skipped 641 row(s)" in txt, txt.strip())
check("a lossy cause survives the bundles note's 400 characters",
      "lost" in (note(big + wide))[:400], note(big + wide)[:120])

print("\n== 7. the 594 already-null rows are recoverable, no reprocessing ==")
conn.execute("UPDATE frame_metric SET values_=NULL WHERE name='sharpness'")
conn.execute("DELETE FROM bundles WHERE seq=?", (f"shard:{sid1}",))
conn.commit()
check("an archive damaged by the old build has a row asserting nothing",
      one("SELECT COUNT(*) FROM frame_metric WHERE values_ IS NULL") == 1)
check("and imported_seqs no longer matches, so the shard can arrive again",
      f"shard:{sid1}" not in ingest.imported_seqs(conn))
tgchannel.fetch_document = lambda i, dest, _p=[shard1]: \
    bool(shutil.copyfile(_p[0], dest)) or True
ingest.import_shard(info1, conn, BASE)
check("_enrich fills the blank from the shard still sitting in the channel",
      one("SELECT COUNT(*) FROM frame_metric WHERE values_ IS NULL") == 0)
again = one("SELECT values_ FROM frame_metric WHERE name='sharpness'")
check("with the same numbers it should have had all along",
      [round(x, 4) for x in
       struct.unpack("<64f", bytes.fromhex(again))] == LONG["sharpness"])
check("and nothing was duplicated doing it",
      one("SELECT COUNT(*) FROM frame_metric") == 3,
      f"{one('SELECT COUNT(*) FROM frame_metric')} rows")

print(f"\n{PASS} passed, {FAIL} failed")
if FAIL:
    raise SystemExit(1)
print("METRIC PAYLOAD OK — a packed measurement lands whole however its shard "
      "was composed, the two tables whose bytes are rehomed still shed theirs, "
      "and a lost row is reported as a lost row.")
