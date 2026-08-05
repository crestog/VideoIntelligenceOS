"""
vios.process.store — the evidence store. Claims, not facts.

The v1 database recorded conclusions: one transcript, one description, one
answer. That is fine right up until two models disagree, at which point the
schema has nowhere to put the disagreement and the second answer overwrites the
first. Disagreement is the most informative thing this system can observe — it
is precisely where a single model would have quietly been wrong — so it needs a
place to live.

Hence: **every row is a claim, attributed to an observer.**

    "a person is holding a knife"     ← claimed by qwen3vl-8b-awq@a1b2, conf 0.81
    "a person is holding a phone"     ← claimed by internvl3-8b@c3d4,   conf 0.74

Both are kept. Nothing merges them. The reader — the interface, the search
index, the report writer — decides what to do with two claims about the same
shot, and it can do that because it can see who said what and how sure they
were. A `fact` table would have thrown the second one away at write time,
before anyone could look.

Three structural decisions follow from that:

**Append-only.** Claims are inserted, never updated. Re-running a model produces
a new observer id and a new set of rows; the old ones stay. This is what makes
"process it again with a better model" safe.

**Deterministic uid.** Every claim's uid is a hash of
(video, observer, channel, kind, shot, ordinal). Re-inserting the same claim is
a no-op via `INSERT OR IGNORE`, which is what makes shard replay idempotent —
and shard replay is how a fresh Kaggle session rebuilds this database from
Telegram in a couple of minutes.

**Time is derived, never claimed.** A claim carries a `shot_idx`; `t0`/`t1` are
filled in from the shot table by this module. A model that emits "at 4.2
seconds" is making a number up — MLLMs hallucinate temporal localisation badly
and confidently — so the schema does not offer it the opportunity. Models emit
shot indices; arithmetic turns those into seconds.
"""

from __future__ import annotations

import array
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time

from . import CHANNELS, SCHEMA_VERSION

_CHANNEL_SET = frozenset(CHANNELS)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=OFF;

-- One row per video. Everything here is measured, not inferred: ffprobe and
-- the capture record are the only writers.
CREATE TABLE IF NOT EXISTS video (
    video_key   TEXT PRIMARY KEY,
    url         TEXT,
    uploader    TEXT,
    duration    REAL,
    width       INTEGER,
    height      INTEGER,
    fps         REAL,
    has_audio   INTEGER DEFAULT 0,
    bytes       INTEGER,
    sha256      TEXT,
    msg_id      INTEGER,          -- where the original lives in Telegram
    taken_at    REAL,
    shots       INTEGER DEFAULT 0,
    partition   INTEGER DEFAULT 0, -- video_key hash % 64, for static sharding
    added_at    REAL,
    meta        TEXT               -- JSON: the capture record's flattened head
);

-- Shot boundaries. THE atomic unit. Every temporal claim keys to one of these.
CREATE TABLE IF NOT EXISTS shot (
    video_key   TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    t0          REAL NOT NULL,
    t1          REAL NOT NULL,
    score       REAL,             -- detector confidence in the boundary
    detector    TEXT,             -- 'pyscenedetect+transnetv2'
    keyframe    REAL,             -- the timestamp we sampled to represent it
    PRIMARY KEY (video_key, idx)
);

-- Who said it. One row per (model, version, parameters) combination that has
-- ever run. The params hash is in the id, so changing the prompt or the frame
-- count creates a new observer rather than silently contaminating the old one.
CREATE TABLE IF NOT EXISTS observer (
    observer_id TEXT PRIMARY KEY,
    component   TEXT NOT NULL,    -- registry component id
    model       TEXT NOT NULL,    -- HF repo id or tool name
    revision    TEXT,             -- pinned commit / version string
    params      TEXT,             -- JSON of everything that affects output
    device      TEXT,
    first_seen  REAL,
    runs        INTEGER DEFAULT 0
);

-- The heart. Append-only.
CREATE TABLE IF NOT EXISTS claim (
    id          INTEGER PRIMARY KEY,
    uid         TEXT NOT NULL UNIQUE,
    video_key   TEXT NOT NULL,
    shot_idx    INTEGER,          -- NULL = a claim about the whole video
    t0          REAL,             -- derived from shot; never model-supplied
    t1          REAL,
    channel     TEXT NOT NULL,
    kind        TEXT NOT NULL,    -- 'transcript','object','summary','palette'…
    value       TEXT,             -- text payload, or JSON for structured kinds
    num         REAL,             -- numeric payload, when the claim is a number
    confidence  REAL DEFAULT 1.0,
    observer_id TEXT NOT NULL,
    ordinal     INTEGER DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_claim_video   ON claim(video_key, channel);
CREATE INDEX IF NOT EXISTS ix_claim_shot    ON claim(video_key, shot_idx);
CREATE INDEX IF NOT EXISTS ix_claim_kind    ON claim(kind, video_key);
CREATE INDEX IF NOT EXISTS ix_claim_obs     ON claim(observer_id, id);
CREATE INDEX IF NOT EXISTS ix_claim_time    ON claim(created_at);

-- Embeddings live apart from claims because they are bytes, not text, and
-- because the publish step streams them out as one contiguous f32 file.
CREATE TABLE IF NOT EXISTS vector (
    id          INTEGER PRIMARY KEY,
    uid         TEXT NOT NULL UNIQUE,
    video_key   TEXT NOT NULL,
    shot_idx    INTEGER,
    space       TEXT NOT NULL,    -- 'siglip2','bge-m3','clap' — never mix spaces
    dim         INTEGER NOT NULL,
    data        BLOB NOT NULL,    -- float32 little-endian, len == dim*4
    observer_id TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_vector_space ON vector(space, video_key);

-- Derived media: poster, proxy, sprite sheet, waveform. Small, regenerable,
-- but expensive enough in aggregate to be worth putting in Telegram once.
CREATE TABLE IF NOT EXISTS artifact (
    video_key   TEXT NOT NULL,
    kind        TEXT NOT NULL,    -- 'poster','proxy','sprite','waveform','loop'
    msg_id      INTEGER,
    file_id     TEXT,
    bytes       INTEGER,
    meta        TEXT,
    created_at  REAL,
    PRIMARY KEY (video_key, kind)
);

-- Every shard ever pushed to permanent storage, so a rebuild knows what to
-- replay and in what order.
CREATE TABLE IF NOT EXISTS shard (
    shard_id    TEXT PRIMARY KEY,
    component   TEXT,
    msg_id      INTEGER,
    claims      INTEGER,
    vectors     INTEGER,
    bytes       INTEGER,
    lo_id       INTEGER,          -- claim.id range covered, for the watermark
    hi_id       INTEGER,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""

# FTS is created separately: on a machine whose SQLite was built without FTS5
# the whole store must still work, just without full-text search. That happens
# often enough on stripped-down container Pythons to be worth handling.
#
# The index is kept in step by a trigger rather than by the writer. Hand-syncing
# an external-content FTS table means every insert path has to remember to do
# it, and the one that forgets produces rows that exist but cannot be found —
# the worst kind of bug, because the data looks fine. Claims are append-only, so
# an AFTER INSERT trigger is the entire contract; there is no update or delete
# path to mirror.
FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS claim_fts USING fts5(
    value, video_key UNINDEXED, channel UNINDEXED, kind UNINDEXED,
    content='claim', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS claim_fts_ai AFTER INSERT ON claim
WHEN new.value IS NOT NULL BEGIN
    INSERT INTO claim_fts(rowid, value, video_key, channel, kind)
    VALUES (new.id, new.value, new.video_key, new.channel, new.kind);
END;
"""


def _uid(*parts) -> str:
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def partition_of(video_key: str, buckets: int = 64) -> int:
    """Stable bucket for a video, so ten workers can split the archive with no
    coordination at all: worker N takes the buckets where `p % workers == N`.

    A hash, not `rowid % n`: rowids shift when rows are inserted out of order,
    and a video silently changing partition mid-run means two workers do it
    twice or neither does it once.
    """
    h = hashlib.blake2b(video_key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") % buckets


class Store:
    """The evidence database. One file, opened once, written from one thread.

    SQLite is the right choice here and not a compromise: the whole database is
    a single file, which is what makes "snapshot it to Telegram" a copy rather
    than a dump, and the read path in the browser is sqlite-wasm over HTTP range
    requests against this exact schema.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        try:
            self.conn.executescript(FTS)
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        cur = self.conn.execute("SELECT v FROM meta WHERE k='schema'")
        row = cur.fetchone()
        if row is None:
            self.conn.execute("INSERT INTO meta(k,v) VALUES('schema',?)",
                              (str(SCHEMA_VERSION),))
        elif int(row["v"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"evidence store at {path} is schema v{row['v']}, this code "
                f"speaks v{SCHEMA_VERSION}. Migrate or start a new file — do "
                f"not let two schemas share one database.")
        self.conn.commit()

    # ── videos ──────────────────────────────────────────────────────────
    def add_video(self, video_key: str, **fields) -> None:
        """Register a video. Idempotent; later calls fill in blanks only.

        Blanks-only matters: `probe` learns the duration, the capture record
        knows the uploader, and neither should clobber the other's column
        because it happened to run second.
        """
        cols = ("url", "uploader", "duration", "width", "height", "fps",
                "has_audio", "bytes", "sha256", "msg_id", "taken_at", "shots",
                "meta")
        row = self.conn.execute(
            "SELECT * FROM video WHERE video_key=?", (video_key,)).fetchone()
        if row is None:
            vals = {c: fields.get(c) for c in cols}
            if isinstance(vals.get("meta"), (dict, list)):
                vals["meta"] = json.dumps(vals["meta"], ensure_ascii=False)
            self.conn.execute(
                f"INSERT INTO video(video_key,partition,added_at,"
                f"{','.join(cols)}) VALUES(?,?,?,{','.join('?' * len(cols))})",
                (video_key, partition_of(video_key), time.time(),
                 *[vals[c] for c in cols]))
        else:
            sets, args = [], []
            for c in cols:
                v = fields.get(c)
                if v is None or row[c] not in (None, "", 0):
                    continue
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                sets.append(f"{c}=?")
                args.append(v)
            if sets:
                args.append(video_key)
                self.conn.execute(
                    f"UPDATE video SET {','.join(sets)} WHERE video_key=?", args)
        self.conn.commit()

    def update_video(self, video_key: str, **fields) -> int:
        """Overwrite columns on an existing video. The one caller is `probe`.

        `add_video` fills blanks only, which is correct when two sources each
        know part of the truth. It is wrong for ffprobe: if a capture record
        guessed a duration from Instagram's metadata and the container says
        otherwise, the container wins — every shot boundary and every claim
        timestamp is derived from that number, so a stale value quietly skews
        the whole video's evidence rather than failing.
        """
        cols = ("url", "uploader", "duration", "width", "height", "fps",
                "has_audio", "bytes", "sha256", "msg_id", "taken_at", "meta")
        sets, args = [], []
        for c in cols:
            if c not in fields or fields[c] is None:
                continue
            v = fields[c]
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            sets.append(f"{c}=?")
            args.append(v)
        if not sets:
            return 0
        args.append(video_key)
        cur = self.conn.execute(
            f"UPDATE video SET {','.join(sets)} WHERE video_key=?", args)
        self.conn.commit()
        return max(cur.rowcount, 0)

    def video(self, video_key: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM video WHERE video_key=?",
                              (video_key,)).fetchone()
        return dict(r) if r else None

    def videos(self, partition_mod: int = 0, partition_idx: int = 0,
               limit: int = 0) -> list:
        sql = "SELECT * FROM video"
        args: list = []
        if partition_mod > 1:
            sql += " WHERE (partition % ?) = ?"
            args += [partition_mod, partition_idx]
        sql += " ORDER BY video_key"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def video_keys(self) -> list:
        return [r[0] for r in
                self.conn.execute("SELECT video_key FROM video ORDER BY video_key")]

    # ── shots ───────────────────────────────────────────────────────────
    def set_shots(self, video_key: str, shots: list, detector: str) -> int:
        """Replace the shot list for a video.

        The one place in this module that deletes. Shots are structure, not
        evidence: two detectors disagreeing about a boundary is not an insight
        worth carrying, and every claim in the database keys to a shot index, so
        two competing shot lists would make `shot_idx` ambiguous. One detector
        wins, its name is recorded, and re-running it re-derives everything.
        """
        rows = []
        for i, s in enumerate(shots):
            t0, t1 = float(s["t0"]), float(s["t1"])
            rows.append((video_key, i, t0, t1, s.get("score"), detector,
                         s.get("keyframe", (t0 + t1) / 2.0)))
        self.conn.execute("DELETE FROM shot WHERE video_key=?", (video_key,))
        self.conn.executemany(
            "INSERT INTO shot(video_key,idx,t0,t1,score,detector,keyframe) "
            "VALUES(?,?,?,?,?,?,?)", rows)
        self.conn.execute("UPDATE video SET shots=? WHERE video_key=?",
                          (len(rows), video_key))
        self.conn.commit()
        return len(rows)

    def shots(self, video_key: str) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM shot WHERE video_key=? ORDER BY idx", (video_key,))]

    def _shot_times(self, video_key: str) -> dict:
        return {r["idx"]: (r["t0"], r["t1"]) for r in self.conn.execute(
            "SELECT idx,t0,t1 FROM shot WHERE video_key=?", (video_key,))}

    # ── observers ───────────────────────────────────────────────────────
    def observer(self, component: str, model: str, revision: str = "",
                 params: dict | None = None, device: str = "") -> str:
        """Get or create an observer id.

        The id is derived from everything that can change the output — model,
        revision, and the parameters — so a prompt edit does not contaminate the
        rows written before it. That is the difference between a database you
        can still trust in a year and one you cannot.
        """
        blob = json.dumps(params or {}, sort_keys=True, ensure_ascii=False)
        oid = f"{component}@{_uid(model, revision, blob)[:12]}"
        self.conn.execute(
            "INSERT OR IGNORE INTO observer(observer_id,component,model,"
            "revision,params,device,first_seen,runs) VALUES(?,?,?,?,?,?,?,0)",
            (oid, component, model, revision, blob, device, time.time()))
        self.conn.execute(
            "UPDATE observer SET runs=runs+1 WHERE observer_id=?", (oid,))
        self.conn.commit()
        return oid

    def observers(self) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM observer ORDER BY component, first_seen")]

    # ── claims ──────────────────────────────────────────────────────────
    def add_claims(self, video_key: str, observer_id: str, claims: list) -> int:
        """Write a batch of claims. Returns how many were new.

        `claims` is a list of dicts: channel, kind, value, and optionally
        shot_idx, num, confidence, ordinal. Times are ignored if supplied —
        they are looked up from the shot table, because a model's opinion about
        when something happened is not evidence.
        """
        times = self._shot_times(video_key)
        dur = self.conn.execute("SELECT duration FROM video WHERE video_key=?",
                                (video_key,)).fetchone()
        whole = (0.0, float(dur["duration"] or 0.0) if dur else 0.0)
        rows, now = [], time.time()
        for n, c in enumerate(claims):
            ch = c.get("channel", "")
            if ch not in _CHANNEL_SET:
                # Loud, not silent. A typo'd channel is invisible in the
                # interface — it renders in no colour and matches no filter —
                # and it would be found months later by its absence.
                raise ValueError(
                    f"unknown channel {ch!r}; expected one of {sorted(_CHANNEL_SET)}")
            si = c.get("shot_idx")
            si = None if si is None else int(si)
            if si is not None and si not in times:
                raise ValueError(
                    f"{video_key}: claim references shot {si}, which does not "
                    f"exist ({len(times)} shots). Run the shots pass first.")
            t0, t1 = times.get(si, whole) if si is not None else whole
            val = c.get("value")
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            ordinal = int(c.get("ordinal", n))
            rows.append((
                _uid(video_key, observer_id, ch, c.get("kind", ""), si, ordinal),
                video_key, si, t0, t1, ch, str(c.get("kind", "")),
                None if val is None else str(val),
                c.get("num"), float(c.get("confidence", 1.0)),
                observer_id, ordinal, now))
        if not rows:
            return 0
        # `cursor.rowcount`, not `conn.total_changes`. The FTS trigger writes to
        # three shadow tables per claim, and total_changes counts every one of
        # them — it reported 17 for 3 claims. rowcount counts only the rows the
        # statement itself inserted, which is what "how many were new" means.
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO claim(uid,video_key,shot_idx,t0,t1,channel,"
            "kind,value,num,confidence,observer_id,ordinal,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        written = max(cur.rowcount, 0)
        self.conn.commit()
        return written

    def claims(self, video_key: str, channel: str = "", kind: str = "",
               observer_id: str = "", limit: int = 2000) -> list:
        sql = "SELECT * FROM claim WHERE video_key=?"
        args: list = [video_key]
        for col, val in (("channel", channel), ("kind", kind),
                         ("observer_id", observer_id)):
            if val:
                sql += f" AND {col}=?"
                args.append(val)
        sql += " ORDER BY shot_idx, ordinal, id LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def search(self, query: str, limit: int = 50) -> list:
        """Full-text over every claim. Present for the engine tab's spot checks;
        the real search runs in the browser over the published bundle."""
        if not self.fts or not query.strip():
            return []
        return [dict(r) for r in self.conn.execute(
            "SELECT c.* FROM claim_fts f JOIN claim c ON c.id=f.rowid "
            "WHERE claim_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit))]

    # ── vectors ─────────────────────────────────────────────────────────
    def add_vector(self, video_key: str, space: str, values, observer_id: str,
                   shot_idx: int | None = None) -> bool:
        buf = array.array("f", [float(v) for v in values])
        # The publish step mmaps these as a flat Float32Array in the browser,
        # which assumes little-endian. Every platform we run on is, but the
        # assumption is cheap to make explicit and expensive to discover later.
        if sys.byteorder != "little":
            buf.byteswap()
        uid = _uid(video_key, observer_id, space, shot_idx)
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO vector(uid,video_key,shot_idx,space,dim,"
            "data,observer_id,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (uid, video_key, shot_idx, space, len(buf), buf.tobytes(),
             observer_id, time.time()))
        self.conn.commit()
        return cur.rowcount > 0

    def vectors_for(self, video_key: str, space: str) -> list:
        """One video's vectors in one space, ordered by shot.

        The whole-archive `vectors()` below is what the publish step reads. This
        is what a runner reads, and the distinction is not cosmetic: the tagger
        runs inside a per-video loop, and pulling five thousand videos' worth of
        1152-dimensional floats out of SQLite to use one video's worth would
        cost about a gigabyte of Python lists per call.
        """
        out = []
        for r in self.conn.execute(
                "SELECT shot_idx,dim,data FROM vector WHERE video_key=? AND "
                "space=? ORDER BY shot_idx", (video_key, space)):
            arr = array.array("f")
            arr.frombytes(r["data"])
            if sys.byteorder != "little":
                arr.byteswap()
            out.append({"shot_idx": r["shot_idx"], "values": list(arr)})
        return out

    def vectors(self, space: str, limit: int = 0) -> list:
        sql = ("SELECT video_key,shot_idx,dim,data FROM vector WHERE space=? "
               "ORDER BY video_key, shot_idx")
        if limit:
            sql += f" LIMIT {int(limit)}"
        out = []
        for r in self.conn.execute(sql, (space,)):
            arr = array.array("f")
            arr.frombytes(r["data"])
            if sys.byteorder != "little":
                arr.byteswap()
            out.append({"video_key": r["video_key"], "shot_idx": r["shot_idx"],
                        "values": list(arr)})
        return out

    # ── artifacts ───────────────────────────────────────────────────────
    def set_artifact(self, video_key: str, kind: str, msg_id: int | None,
                     file_id: str = "", nbytes: int = 0,
                     meta: dict | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO artifact(video_key,kind,msg_id,file_id,"
            "bytes,meta,created_at) VALUES(?,?,?,?,?,?,?)",
            (video_key, kind, msg_id, file_id, nbytes,
             json.dumps(meta or {}, ensure_ascii=False), time.time()))
        self.conn.commit()

    def artifacts(self, video_key: str) -> dict:
        return {r["kind"]: dict(r) for r in self.conn.execute(
            "SELECT * FROM artifact WHERE video_key=?", (video_key,))}

    # ── shards: how this database survives a session ────────────────────
    def export_shard(self, path: str, lo_id: int = 0, hi_id: int = 0,
                     lo_vec: int = 0, hi_vec: int = 0,
                     component: str = "") -> dict:
        """Write claims and vectors in an id range to a gzipped JSONL file.

        Why a range and not "everything since a timestamp": ids are assigned by
        SQLite in insertion order and never move, so a range is exact and
        replayable. Wall-clock is not — a session whose clock is wrong, or that
        runs across a DST boundary, would skip or duplicate rows.

        Why JSONL and not a copy of the .db: shards are *merged* on restore,
        from ten different accounts that each processed a different partition.
        Merging ten SQLite files means attaching and inserting anyway; JSONL
        skips the step and compresses better.

        Claims and vectors carry **separate** id ranges. An embedding pass
        writes vectors and no claims at all, so keying the vector export off the
        claim range would silently drop every embedding the cohort produced —
        and it would look like a successful upload.
        """
        hi_id = hi_id or self.max_claim_id()
        hi_vec = hi_vec or self.max_vector_id()

        # The set of videos this shard touches, from either side.
        keys = "(SELECT video_key FROM claim WHERE id>? AND id<=? " \
               "UNION SELECT video_key FROM vector WHERE id>? AND id<=?)"
        span = (lo_id, hi_id, lo_vec, hi_vec)

        n_claims = n_vectors = 0
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
            fh.write(json.dumps({
                "_": "vios-evidence-shard", "schema": SCHEMA_VERSION,
                "component": component, "lo_id": lo_id, "hi_id": hi_id,
                "lo_vec": lo_vec, "hi_vec": hi_vec, "at": time.time()}) + "\n")

            for r in self.conn.execute(
                    "SELECT * FROM observer WHERE observer_id IN ("
                    " SELECT observer_id FROM claim WHERE id>? AND id<=?"
                    " UNION SELECT observer_id FROM vector WHERE id>? AND id<=?)",
                    span):
                fh.write(json.dumps({"t": "observer", **dict(r)},
                                    ensure_ascii=False) + "\n")
            for table in ("video", "shot", "artifact"):
                for r in self.conn.execute(
                        f"SELECT * FROM {table} WHERE video_key IN {keys}", span):
                    fh.write(json.dumps({"t": table, **dict(r)},
                                        ensure_ascii=False) + "\n")
            for r in self.conn.execute(
                    "SELECT * FROM claim WHERE id>? AND id<=? ORDER BY id",
                    (lo_id, hi_id)):
                d = dict(r)
                d.pop("id", None)
                fh.write(json.dumps({"t": "claim", **d}, ensure_ascii=False) + "\n")
                n_claims += 1
            for r in self.conn.execute(
                    "SELECT * FROM vector WHERE id>? AND id<=? ORDER BY id",
                    (lo_vec, hi_vec)):
                d = dict(r)
                d.pop("id", None)
                d["data"] = d["data"].hex()
                fh.write(json.dumps({"t": "vector", **d}) + "\n")
                n_vectors += 1

        return {"path": path, "claims": n_claims, "vectors": n_vectors,
                "lo_id": lo_id, "hi_id": hi_id,
                "lo_vec": lo_vec, "hi_vec": hi_vec,
                "bytes": os.path.getsize(path)}

    def import_shard(self, path: str) -> dict:
        """Replay a shard into this database. Idempotent by uid.

        This is the restore path: ten accounts each push shards to the channel,
        any one of them can pull all of them and end up with the union. Order
        does not matter, duplicates do not matter, and a partial shard from a
        session that died mid-upload is simply a shorter file.
        """
        counts = {"claim": 0, "vector": 0, "video": 0, "shot": 0,
                  "observer": 0, "artifact": 0, "skipped": 0}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            header = json.loads(fh.readline() or "{}")
            if header.get("_") != "vios-evidence-shard":
                raise ValueError(f"{path} is not an evidence shard")
            if int(header.get("schema", 0)) != SCHEMA_VERSION:
                raise ValueError(
                    f"{path} is schema v{header.get('schema')}, this code "
                    f"speaks v{SCHEMA_VERSION}")
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A shard truncated by a session dying mid-write. Everything
                    # before the tear is still good; stop and keep it.
                    counts["skipped"] += 1
                    break
                t = rec.pop("t", "")
                if t not in counts:
                    continue
                try:
                    self._replay(t, rec)
                    counts[t] += 1
                except (sqlite3.Error, ValueError, KeyError):
                    counts["skipped"] += 1
        self.conn.commit()
        return counts

    def _replay(self, t: str, rec: dict) -> None:
        if t == "observer":
            self.conn.execute(
                "INSERT OR IGNORE INTO observer(observer_id,component,model,"
                "revision,params,device,first_seen,runs) VALUES(?,?,?,?,?,?,?,?)",
                (rec["observer_id"], rec.get("component", ""), rec.get("model", ""),
                 rec.get("revision", ""), rec.get("params", "{}"),
                 rec.get("device", ""), rec.get("first_seen", 0), rec.get("runs", 0)))
        elif t == "video":
            cols = ["video_key", "url", "uploader", "duration", "width", "height",
                    "fps", "has_audio", "bytes", "sha256", "msg_id", "taken_at",
                    "shots", "partition", "added_at", "meta"]
            self.conn.execute(
                f"INSERT OR IGNORE INTO video({','.join(cols)}) "
                f"VALUES({','.join('?' * len(cols))})",
                [rec.get(c) for c in cols])
        elif t == "shot":
            self.conn.execute(
                "INSERT OR REPLACE INTO shot(video_key,idx,t0,t1,score,detector,"
                "keyframe) VALUES(?,?,?,?,?,?,?)",
                (rec["video_key"], rec["idx"], rec["t0"], rec["t1"],
                 rec.get("score"), rec.get("detector"), rec.get("keyframe")))
        elif t == "claim":
            self.conn.execute(
                "INSERT OR IGNORE INTO claim(uid,video_key,shot_idx,t0,t1,"
                "channel,kind,value,num,confidence,observer_id,ordinal,"
                "created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["uid"], rec["video_key"], rec.get("shot_idx"),
                 rec.get("t0"), rec.get("t1"), rec["channel"], rec["kind"],
                 rec.get("value"), rec.get("num"), rec.get("confidence", 1.0),
                 rec["observer_id"], rec.get("ordinal", 0),
                 rec.get("created_at", time.time())))
        elif t == "vector":
            self.conn.execute(
                "INSERT OR IGNORE INTO vector(uid,video_key,shot_idx,space,dim,"
                "data,observer_id,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (rec["uid"], rec["video_key"], rec.get("shot_idx"), rec["space"],
                 rec["dim"], bytes.fromhex(rec["data"]), rec["observer_id"],
                 rec.get("created_at", time.time())))
        elif t == "artifact":
            self.conn.execute(
                "INSERT OR REPLACE INTO artifact(video_key,kind,msg_id,file_id,"
                "bytes,meta,created_at) VALUES(?,?,?,?,?,?,?)",
                (rec["video_key"], rec["kind"], rec.get("msg_id"),
                 rec.get("file_id", ""), rec.get("bytes", 0),
                 rec.get("meta", "{}"), rec.get("created_at", time.time())))

    def note_shard(self, shard_id: str, component: str, msg_id: int | None,
                   stats: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO shard(shard_id,component,msg_id,claims,"
            "vectors,bytes,lo_id,hi_id,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (shard_id, component, msg_id, stats.get("claims", 0),
             stats.get("vectors", 0), stats.get("bytes", 0),
             stats.get("lo_id", 0), stats.get("hi_id", 0), time.time()))
        self.conn.commit()

    def shards(self) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM shard ORDER BY created_at")]

    # ── housekeeping ────────────────────────────────────────────────────
    def max_claim_id(self) -> int:
        return self.conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM claim").fetchone()[0]

    def max_vector_id(self) -> int:
        return self.conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM vector").fetchone()[0]

    def get_meta(self, k: str, default: str = "") -> str:
        r = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r["v"] if r else default

    def set_meta(self, k: str, v: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)",
                          (k, str(v)))
        self.conn.commit()

    def checkpoint(self) -> None:
        """Fold the WAL back into the main file.

        Mandatory before copying or uploading the .db. In WAL mode recent
        commits live in `<db>-wal`, so a snapshot taken without this is missing
        exactly the work done since the last automatic checkpoint — the newest
        and least reproducible rows. The capture ledger had this bug; it is not
        being repeated here.
        """
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass

    def stats(self) -> dict:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        by_channel = {r["channel"]: r["n"] for r in self.conn.execute(
            "SELECT channel, COUNT(*) n FROM claim GROUP BY channel")}
        by_observer = [dict(r) for r in self.conn.execute(
            "SELECT observer_id, COUNT(*) n, COUNT(DISTINCT video_key) videos "
            "FROM claim GROUP BY observer_id ORDER BY n DESC LIMIT 40")]
        return {
            "videos": q("SELECT COUNT(*) FROM video"),
            "shots": q("SELECT COUNT(*) FROM shot"),
            "claims": q("SELECT COUNT(*) FROM claim"),
            "vectors": q("SELECT COUNT(*) FROM vector"),
            "observers": q("SELECT COUNT(*) FROM observer"),
            "artifacts": q("SELECT COUNT(*) FROM artifact"),
            "shards": q("SELECT COUNT(*) FROM shard"),
            "by_channel": by_channel,
            "by_observer": by_observer,
            "fts": self.fts,
            "bytes": os.path.getsize(self.path) if os.path.exists(self.path) else 0,
        }

    def close(self) -> None:
        try:
            self.checkpoint()
            self.conn.close()
        except sqlite3.Error:
            pass
