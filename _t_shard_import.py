"""Does Atlas's own importer read a shard the engine actually wrote?

_t_seam.py replayed a shard by hand to prove the reflection layer. This proves
the thing that ships: tgchannel.looks_like_shard on a real message shape, then
ingest.import_shard against a real gzipped JSONL, with no help from vios.
"""
import os, tempfile, sqlite3, json

BASE = tempfile.mkdtemp()
os.environ["VIOS_BASE_DIR"] = BASE
os.environ["ATLAS_HOME"] = os.path.join(BASE, "atlas")
os.environ["ATLAS_CACHE_DIR"] = os.path.join(BASE, "cache")
os.environ["VIOS_OMNI"] = "0"
os.makedirs(os.environ["ATLAS_HOME"], exist_ok=True)

from vios.process import store as S           # writer only — Atlas never imports it

st = S.Store(os.path.join(BASE, "ev.db"))
st.add_video(video_key="REEL1", url="https://instagram.com/p/REEL1/",
             uploader="chef", duration=None, width=1080, height=1920,
             bytes=5_000_000, taken_at=None, meta={})
st.set_shots("REEL1", [{"t0": 0.0, "t1": 4.0}, {"t0": 4.0, "t1": 9.5}],
             "pyscenedetect")
oid = st.observer("transcript", "whisper", "large-v3", {})
st.add_claims("REEL1", oid, [
    {"channel": "speech", "kind": "transcript", "shot_idx": 0,
     "value": "today we are making a garlic butter steak"},
    {"channel": "speech", "kind": "transcript", "shot_idx": 1,
     "value": "get the pan properly screaming hot"}])
oid2 = st.observer("ocr", "paddle", "v4", {})
st.add_claims("REEL1", oid2, [
    {"channel": "ocr", "kind": "text", "shot_idx": 0, "value": "MEDIUM RARE 54C"}])
# A vector, so the opaque-payload rule has something real to reject.
st.add_vector("REEL1", "clip", [0.01 * i for i in range(384)], oid, shot_idx=0)

from vios.process import intake
sid = f"{intake.site_id(st)}-0001"
shard = os.path.join(BASE, intake.shard_name(sid))
stats = st.export_shard(shard, 0, st.max_claim_id(), 0, st.max_vector_id(), "gpu")
print(f"engine wrote {intake.shard_name(sid)} — {stats['claims']} claims, "
      f"{stats['vectors']} vectors, {stats['bytes']}b")

# ── from here on, only Atlas ──────────────────────────────────────────────
from atlas import identity, index, ingest, reflect, search, tgchannel

MSG = {"message_id": 5150, "date": 1754700000,
       "caption": f"vios evidence · {sid}\n3 claims, 1 vectors",
       "document": {"file_name": intake.shard_name(sid),
                    "file_id": "FAKE", "file_size": stats["bytes"],
                    "mime_type": "application/gzip"}}
info = tgchannel.message_document(MSG)

print("\n== recognition ==")
print("   looks_like_shard   ", tgchannel.looks_like_shard(info))
print("   looks_like_manifest", tgchannel.looks_like_manifest(info))
print("   shard_seq          ", repr(tgchannel.shard_seq(info)))
assert tgchannel.looks_like_shard(info)
assert not tgchannel.looks_like_manifest(info), "a shard must not read as a bundle"
assert tgchannel.shard_seq(info) == sid

# Caption-only (a future engine renames the file) and name-only (caption lost).
assert tgchannel.looks_like_shard({"file_name": "", "caption": f"vios evidence · {sid}"})
assert tgchannel.looks_like_shard({"file_name": intake.shard_name(sid), "caption": ""})
# A reel is not a shard.
assert not tgchannel.looks_like_shard(
    {"file_name": "REEL1.mp4", "caption": "Creator: @chef\nLikes: 4200"})
# A manifest is not a shard.
assert not tgchannel.looks_like_shard(
    {"file_name": "manifest-0007.json", "caption": "✅ VIOS bundle 0007"})

# The download the importer would do, without a network.
import shutil
tgchannel.fetch_document = lambda i, dest: bool(shutil.copyfile(shard, dest)) or True

conn = ingest.connect(os.path.join(os.environ["ATLAS_HOME"], "atlas.sqlite"))
ingest.ensure_meta(conn)

print("\n== import ==")
res = ingest.import_shard(info, conn, BASE)
print("  ", res)
assert res["ok"], res

built = {r[0]: r[1] for r in conn.execute(
    "SELECT name, sql FROM sqlite_master WHERE type='table' "
    "AND name NOT LIKE 'sqlite_%' ORDER BY name")}
print("\n== what Atlas built for itself ==")
for name, sql in built.items():
    if name in ("bundles", "atlas_meta"):
        continue
    print("  ", " ".join(sql.split())[:120])

cols = {r[1]: r[2] for r in conn.execute('PRAGMA table_info("claim")')}
print("\n   claim column types:", cols)
assert cols.get("t0") == "REAL", cols          # a timestamp, not a string
assert cols.get("shot_idx") == "INTEGER", cols
assert cols.get("value") == "TEXT", cols
assert "data" not in {r[1] for r in conn.execute('PRAGMA table_info("vector")')}, \
    "the embedding payload should have been dropped, not stored"

idx = [r[0] for r in conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")]
print("\n   unique indexes Atlas inferred:")
for s in idx:
    print("     ", " ".join(s.split()))

# ── idempotency: the same shard twice must change nothing ────────────────
before = {t: conn.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0]
          for t in built if t not in ("bundles", "atlas_meta")}
ingest.import_shard(info, conn, BASE)
after = {t: conn.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0]
         for t in before}
print("\n== re-import ==")
print("   before", before)
print("   after ", after)
assert before == after, (before, after)

# ── enrichment: a later shard that measured the duration fills the blank ──
print("\n== enrichment ==")
vcols = {r[1] for r in conn.execute('PRAGMA table_info("video")')}
print("   video columns after shard 1:", sorted(vcols))
assert "duration" not in vcols, \
    "a column that was null in every row should not exist yet — typed on that " \
    "evidence it becomes TEXT and stores 30.0 as a string forever"
st.conn.execute("UPDATE video SET duration=30.0 WHERE video_key='REEL1'")
st.conn.commit()
st.add_claims("REEL1", oid, [{"channel": "speech", "kind": "transcript",
                              "shot_idx": 1, "value": "butter at the very end"}])
sid2 = f"{intake.site_id(st)}-0002"
shard2 = os.path.join(BASE, intake.shard_name(sid2))
st.export_shard(shard2, stats["hi_id"], st.max_claim_id(), 0, 0, "gpu")
tgchannel.fetch_document = lambda i, dest: bool(shutil.copyfile(shard2, dest)) or True
info2 = dict(info, file_name=intake.shard_name(sid2),
             caption=f"vios evidence · {sid2}", message_id=5151)
r2 = ingest.import_shard(info2, conn, BASE)
print("  ", r2)
dur = conn.execute("SELECT duration FROM video WHERE video_key='REEL1'").fetchone()[0]
print("   duration after :", repr(dur),
      conn.execute("SELECT typeof(duration) FROM video "
                   "WHERE video_key='REEL1'").fetchone()[0])
assert dur == 30.0, "a later shard should fill the blank, not be ignored"
assert isinstance(dur, float), f"duration came back as {type(dur).__name__}"
assert conn.execute("SELECT COUNT(*) FROM video").fetchone()[0] == 1

# ── and the whole point: is it searchable? ────────────────────────────────
print("\n== searchable ==")
r = index.rebuild(conn, embed=False)
print("  ", {k: r.get(k) for k in ("ok", "moments", "videos")})
for q, want in (("garlic butter steak", "speech"), ("medium rare", "ocr"),
                ("butter at the very end", "speech")):
    hits = search.search(conn, q, limit=3).get("results") or []
    best = (hits[0].get("best") or {}) if hits else {}
    print(f'   "{q}" -> {hits[0]["video_key"] if hits else None} '
          f'@ {best.get("t_start")}-{best.get("t_end")}s [{best.get("source")}]')
    assert hits and hits[0]["video_key"] == "REEL1", q
    assert best.get("source") == want, (q, best.get("source"))
    assert best.get("t_start") is not None, q

rows = ingest.bundle_rows(conn)
print("\n== Sources tab ==")
for b in rows:
    print(f"   {b['seq']:20s} msg {b['manifest_id']}  {b['status']:7s} "
          f"{b['counts']}\n{'':26s}{b['note'][:80]}")
assert {b["seq"] for b in rows} == {f"shard:{sid}", f"shard:{sid2}"}
# Re-importing shard 1 must not have rewritten its history to say it held
# nothing — the Sources tab answers "what arrived", not "what this run added".
first = [b for b in rows if b["seq"] == f"shard:{sid}"][0]
assert first["counts"].get("claim") == 3, first["counts"]

# ── nothing silently dropped between the claims and the passages ─────────
texts = {r[0] for r in conn.execute("SELECT value FROM claim")}
moments = {r[0] for r in conn.execute("SELECT text FROM moments")}
missing = [t for t in texts if not any(t in m for m in moments)]
print("\n   4 claim value(s) ->",
      conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0], "moment(s)")
for m in conn.execute("SELECT video_key,t_start,t_end,source,text FROM moments "
                      "ORDER BY t_start"):
    print("     ", m)
assert not missing, f"claim text that reached no moment: {missing}"

# ── a shard an older build wrote, naming videos by message id ────────────
# Hand-written, because the engine can no longer produce one: `add_video` and
# `_replay` refuse a non-identity key. The channel still holds shards from
# before that, so this is the rebuild door, tested from the outside.
print("\n== legacy spellings ==")
import gzip
sid3 = f"{intake.site_id(st)}-0003"
shard3 = os.path.join(BASE, intake.shard_name(sid3))
legacy = [
    {"_": "vios-evidence-shard", "schema": 3, "component": "gpu", "at": 1.0},
    # the permalink is in the row, so the message id is not ambiguous
    {"t": "video", "video_key": "9001", "url": "https://instagram.com/p/REEL1/",
     "msg_id": 9001, "bytes": 5_000_000, "added_at": 1.0, "meta": "{}"},
    {"t": "claim", "uid": "legacy-1", "video_key": "9001", "shot_idx": 1,
     "t0": 4.0, "t1": 9.5, "channel": "speech", "kind": "transcript",
     "value": "rest it under foil for five minutes", "observer_id": oid,
     "created_at": 1.0},
    # nothing in the row and nothing in the alias map can place this one
    {"t": "video", "video_key": "4242", "url": "", "msg_id": 4242,
     "bytes": 1_000, "added_at": 1.0, "meta": "{}"},
]
with gzip.open(shard3, "wt", encoding="utf-8") as fh:
    for r in legacy:
        fh.write(json.dumps(r) + "\n")
tgchannel.fetch_document = lambda i, dest: bool(shutil.copyfile(shard3, dest)) or True
info3 = dict(info, file_name=intake.shard_name(sid3),
             caption=f"vios evidence · {sid3}", message_id=5152)
r3 = ingest.import_shard(info3, conn, BASE)
print("  ", r3)
vids = [r[0] for r in conn.execute("SELECT video_key FROM video ORDER BY video_key")]
print("   videos after the legacy shard:", vids)
assert vids == ["REEL1"], f"a message id became a video: {vids}"
assert r3["rows"].get("identity:rehomed") == 1, r3["rows"]
note = [b["note"] for b in ingest.bundle_rows(conn) if b["seq"] == f"shard:{sid3}"][0]
print("   note:", note)
assert "not an identity" in note, note
alias = conn.execute(
    "SELECT video_key FROM video_alias WHERE alias='9001'").fetchone()
print("   alias 9001 ->", alias[0] if alias else None)
assert alias and alias[0] == "REEL1", alias
# and the evidence that arrived under the old spelling reads as the video's own
index.rebuild(conn, embed=False)
hits = search.search(conn, "rest it under foil", limit=3).get("results") or []
print(f'   "rest it under foil" -> {hits[0]["video_key"] if hits else None}')
assert hits and hits[0]["video_key"] == "REEL1", hits
assert conn.execute("SELECT COUNT(*) FROM video").fetchone()[0] == 1

# ── and it survives the *next* rebuild, which is where it used to die ──────
# `identity.rebuild` empties `video_alias` and re-derives it, on purpose — a
# stale weak alias must not outrank a new strong one forever. But the row that
# taught it "9001 is REEL1" was the legacy shard's `video` row, and that row is
# deliberately never written, so there was nothing left to re-derive it from.
#
# It passed above only by accident: REEL1 arrived with no `msg_id`, so `_enrich`
# filled the blank with the legacy shard's 9001 and the fact survived inside the
# `video` row. Give REEL1 the message id the capture plane would have given it —
# a different number, which `_enrich` would never overwrite — and the accident is
# gone. What holds it now is `video_message`, a record rather than a derivation.
print("\n== the alias survives a rebuild ==")
conn.execute("UPDATE video SET msg_id=? WHERE video_key=?", (5150, "REEL1"))
conn.commit()
msgs = identity.messages_for(conn, "REEL1")
print("   video_message:", msgs)
assert 9001 in [m["msg_id"] for m in msgs], msgs

index.rebuild(conn, embed=False)
again = conn.execute(
    "SELECT video_key FROM video_alias WHERE alias='9001'").fetchone()
print("   after rebuild, alias 9001 ->", again[0] if again else None)
assert again and again[0] == "REEL1", "the rehomed alias did not survive"
hits = search.search(conn, "rest it under foil", limit=3).get("results") or []
print(f'   "rest it under foil" -> {hits[0]["video_key"] if hits else None}')
assert hits and hits[0]["video_key"] == "REEL1", hits

rep = identity.audit(conn)
print("   audit:", {k: rep[k] for k in ("ok", "videos", "aliases", "messages",
                                        "reuploads", "unresolved")})
assert rep["messages"] >= 2, rep          # 5150 from the row, 9001 from the shard
assert rep["reuploads"] == 1, rep         # one reel, two messages — not a duplicate
card = conn.execute(
    "SELECT msg_id, messages FROM video_index WHERE video_key='REEL1'").fetchone()
print("   card msg_id:", card[0], " messages:", card[1])
assert json.loads(card[1]) == [5150, 9001], card[1]

# ── collections: one reel on several shelves, and a filter that says so ────
# The requirement this guards is a sentence: one video might be in two or more
# collections I have saved, and that must not make it two videos. So the shelves
# arrive the way production sends them — inside `video.meta.capture.collections`,
# in the same `video` row every shard already carries, with no new record type —
# and a second video exists so that a filter has something to exclude.
print("\n== collections: one reel, several shelves ==")
sid4 = f"{intake.site_id(st)}-0004"
shard4 = os.path.join(BASE, intake.shard_name(sid4))
second = [
    {"_": "vios-evidence-shard", "schema": 3, "component": "gpu", "at": 1.0},
    {"t": "video", "video_key": "REEL2", "url": "https://instagram.com/p/REEL2/",
     "msg_id": 5160, "bytes": 2_000_000, "added_at": 1.0,
     "meta": json.dumps({"capture": {"collections": ["desserts", "dinners"]}})},
    {"t": "claim", "uid": "c-tart", "video_key": "REEL2", "shot_idx": 0,
     "t0": 0.0, "t1": 6.0, "channel": "speech", "kind": "transcript",
     "value": "today we are making a chocolate tart", "observer_id": oid,
     "created_at": 1.0},
]
with gzip.open(shard4, "wt", encoding="utf-8") as fh:
    for r in second:
        fh.write(json.dumps(r) + "\n")
tgchannel.fetch_document = lambda i, dest: bool(shutil.copyfile(shard4, dest)) or True
info4 = dict(info, file_name=intake.shard_name(sid4),
             caption=f"vios evidence · {sid4}", message_id=5153)
r4 = ingest.import_shard(info4, conn, BASE)
print("  ", r4)
assert r4["ok"], r4
# REEL1's shelves the other way in: the ledger's API, which is what a capture
# machine calls. Additive on purpose — a label is never removed by a refresh.
identity.set_collections(conn, "REEL1", ["steak", "dinners"], "test")
conn.commit()
index.rebuild(conn, embed=False)

shelves = {v: identity.collections_for(conn, v) for v in ("REEL1", "REEL2")}
print("   shelves:", shelves)
assert shelves == {"REEL1": ["dinners", "steak"],
                   "REEL2": ["desserts", "dinners"]}, shelves
counts = {c["name"]: c["videos"] for c in identity.collection_counts(conn)}
print("   facet  :", counts)
assert counts == {"dinners": 2, "desserts": 1, "steak": 1}, counts
s = search.stats(conn)
print("   stats  :", {k: s[k] for k in ("videos", "collections", "memberships")})
# Two videos and four filings: the numbers differing *is* the squeeze working.
assert (s["videos"], s["collections"], s["memberships"]) == (2, 3, 4), s

wide = search.search(conn, "today we are making", limit=10)
assert {r["video_key"] for r in wide["results"]} == {"REEL1", "REEL2"}, wide
narrow = search.search(conn, "today we are making", collection="steak", limit=10)
print(f'   "today we are making" -> {[r["video_key"] for r in wide["results"]]}'
      f'   in steak -> {[r["video_key"] for r in narrow["results"]]}')
assert [r["video_key"] for r in narrow["results"]] == ["REEL1"], narrow
# Filtered after grouping, so the row keeps the score it had unfiltered and the
# UI can say "1 of 2, narrowed by your filters" without lying about the query.
assert (narrow["matched"], narrow["total"]) == (2, 1), narrow
w1 = [r for r in wide["results"] if r["video_key"] == "REEL1"][0]
assert narrow["results"][0]["score"] == w1["score"], (narrow, wide)
# One video answers to both of its shelves, and to none of the others.
for shelf in ("steak", "dinners"):
    got = search.search(conn, "today we are making", collection=shelf, limit=10)
    assert any(r["video_key"] == "REEL1" for r in got["results"]), shelf
off = search.search(conn, "today we are making", collection="desserts", limit=10)
assert [r["video_key"] for r in off["results"]] == ["REEL2"], off
# The chips count filings over the matched pool, and the active one keeps its
# own count so it can be switched back off.
fac = {f["value"]: f["count"] for f in wide["facets"]["collections"]}
print("   chips  :", fac)
assert fac == counts, (fac, counts)
assert {f["value"]: f["count"]
        for f in narrow["facets"]["collections"]} == fac, narrow["facets"]
# And every result carries its own identity, so a card never asks a second time.
assert narrow["results"][0]["collections"] == ["dinners", "steak"], narrow
assert narrow["results"][0]["messages"] == [5150, 9001], narrow

print("\nIMPORTER OK — Atlas reads the GPU plane's shards on its own")
