"""Process shard -> Atlas. Does the evidence survive the trip and become
searchable, seekable moments?"""
import os, tempfile, gzip, json, sqlite3
BASE = tempfile.mkdtemp(); os.environ["VIOS_BASE_DIR"] = BASE
os.environ["ATLAS_HOME"] = os.path.join(BASE, "atlas")
os.environ["VIOS_OMNI"] = "0"
os.makedirs(os.environ["ATLAS_HOME"], exist_ok=True)

from vios.process import store as S

st = S.Store(os.path.join(BASE, "ev.db"))
st.add_video(video_key="REEL1", url="https://instagram.com/p/REEL1/",
             uploader="chef", duration=30.0, width=1080, height=1920,
             bytes=5_000_000, taken_at=None, meta={})
st.set_shots("REEL1", [{"t0": 0.0, "t1": 4.0}, {"t0": 4.0, "t1": 9.5},
                       {"t0": 9.5, "t1": 14.0}], "pyscenedetect")

oid = st.observer("transcript", "whisper", "large-v3", {})
st.add_claims("REEL1", oid, [
    {"channel": "speech", "kind": "transcript", "shot_idx": 0,
     "value": "today we are making a garlic butter steak"},
    {"channel": "speech", "kind": "transcript", "shot_idx": 1,
     "value": "get the pan properly screaming hot before anything"},
    {"channel": "speech", "kind": "transcript", "shot_idx": 2,
     "value": "butter goes in at the very end or it burns"}])

oid2 = st.observer("ocr", "paddle", "v4", {})
st.add_claims("REEL1", oid2, [
    {"channel": "ocr", "kind": "text", "shot_idx": 0,
     "value": "MEDIUM RARE 54C"}])

oid3 = st.observer("caption", "qwen", "2.5-vl", {})
st.add_claims("REEL1", oid3, [
    {"channel": "visual", "kind": "caption", "shot_idx": 1,
     "value": "a cast iron pan smoking over a gas flame"}])

shard = os.path.join(BASE, "vios-evidence-site-0001.jsonl.gz")
stats = st.export_shard(shard, 0, st.max_claim_id(), 0, st.max_vector_id(), "test")
print(f"shard: {stats['claims']} claims, {stats['bytes']}b")

# ---- replay into a fresh Atlas database, the way an importer would ----
adb = os.path.join(os.environ["ATLAS_HOME"], "atlas.sqlite")
conn = sqlite3.connect(adb)
tables = {}
with gzip.open(shard, "rt", encoding="utf-8") as fh:
    head = json.loads(fh.readline())
    for line in fh:
        row = json.loads(line); t = row.pop("t")
        tables.setdefault(t, []).append(row)
print("shard tables:", {k: len(v) for k, v in tables.items()})

for t, rows in tables.items():
    cols = sorted({c for r in rows for c in r})
    quoted = ", ".join('"%s"' % c for c in cols)
    marks = ", ".join("?" for _ in cols)
    conn.executescript(S.SCHEMA)          # the real schema, with real types
    conn.executemany('INSERT OR IGNORE INTO "%s" (%s) VALUES (%s)'
                     % (t, quoted, marks),
                     [[r.get(c) for c in cols] for r in rows])
conn.commit()

from atlas import reflect, index
print("\n-- what Atlas sees --")
for sp in reflect.text_sources(conn):
    print(f"   {sp['table']}.{sp['text']:12s} key={sp['key']:10s} "
          f"start={sp['start'] or '-':6s} end={sp['end'] or '-':6s} src={sp['source']}")

res = index.rebuild(conn, embed=False)
print("\nrebuild:", {k: res.get(k) for k in ("ok", "moments", "videos", "note")})
rows = conn.execute("SELECT video_key,t_start,t_end,source,substr(text,1,44) "
                    "FROM moments ORDER BY t_start").fetchall()
print(f"\nmoments ({len(rows)}):")
for r in rows:
    print("   ", r)

bad = [r for r in rows if r[1] is None]
print(f"\n{len(bad)} of {len(rows)} moment(s) have NO timestamp "
      f"(expected 1: the creator name, which is not a point in time)")

# ── the collision regression: two shortcodes whose digits are identical ──
print("\n== two reels that used to collide ==")
for k in ("REEL1", "DBd2xyz", "C_abc-123", "tg4242", "4242", "vios:REEL1"):
    print(f"   {k:12s} -> {reflect.normalize_key(k)!r}")
assert reflect.normalize_key("REEL1") == "REEL1"
assert reflect.normalize_key("DBd2xyz") == "DBd2xyz"
assert reflect.normalize_key("REEL1") != reflect.normalize_key("DBd2xyz")
assert reflect.normalize_key("tg4242") == "4242" == reflect.normalize_key("4242")
assert reflect.normalize_key("vios:REEL1") == "REEL1"

st.add_video(video_key="REEL2", url="", uploader="baker", duration=20.0,
             width=1080, height=1920, bytes=4_000_000, taken_at=None, meta={})
st.set_shots("REEL2", [{"t0": 0.0, "t1": 6.0}], "pyscenedetect")
st.add_claims("REEL2", oid, [
    {"channel": "speech", "kind": "transcript", "shot_idx": 0,
     "value": "we are proofing sourdough overnight in the fridge"}])
sh2 = os.path.join(BASE, "vios-evidence-site-0002.jsonl.gz")
st.export_shard(sh2, stats["hi_id"], st.max_claim_id(), 0, 0, "second")
tabs = {}
with gzip.open(sh2, "rt", encoding="utf-8") as fh:
    fh.readline()
    for line in fh:
        r = json.loads(line); tabs.setdefault(r.pop("t"), []).append(r)
for t, rws in tabs.items():
    cs = sorted({c for r in rws for c in r})
    conn.executemany('INSERT OR IGNORE INTO "%s" (%s) VALUES (%s)'
                     % (t, ", ".join('"%s"' % c for c in cs),
                        ", ".join("?" for _ in cs)),
                     [[r.get(c) for c in cs] for r in rws])
conn.commit()
index.rebuild(conn, embed=False)

keys = [r[0] for r in conn.execute("SELECT DISTINCT video_key FROM moments")]
print("   distinct video keys after both shards:", sorted(keys))
assert set(keys) == {"REEL1", "REEL2"}, keys

# ── search: does a question find the right second of the right reel? ──
print("\n== search ==")
from atlas import search
EXPECT = {"garlic butter steak": ("REEL1", "speech"),
          "sourdough":           ("REEL2", "speech"),
          "medium rare":         ("REEL1", "ocr")}
for q, (want_key, want_src) in EXPECT.items():
    res = search.search(conn, q, limit=3)
    got = res.get("results") or []
    print(f'   "{q}"  ({res.get("mode")}, {res.get("took_ms")}ms)')
    for v in got:
        b = v.get("best") or {}
        print(f"      {v['video_key']} @ {b.get('t_start')}-{b.get('t_end')}s "
              f"[{b.get('source')}] {str(b.get('text'))[:44]}")
    assert got, f"no hit for {q!r}"
    top, best = got[0], (got[0].get("best") or {})
    assert top["video_key"] == want_key, (q, top["video_key"])
    assert best.get("source") == want_src, (q, best.get("source"))
    assert best.get("t_start") is not None, f"{q}: hit is not seekable"

print("\nSEAM OK — shards become searchable, seekable, per-reel moments")
