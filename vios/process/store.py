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

**A frame is addressable, and a run of frames is one row.** A shot is the right
unit for "what is this scene about"; it is far too coarse for "find the moment
the price appeared on screen". So a claim may also carry `frame_idx` — and
`frame_hi`, which turns *frames 100 through 142 all read "SUBSCRIBE"* into a
single row that is still answerable per frame with
`frame_idx <= ? AND frame_hi >= ?`. Coverage stays total; storage is
run-length. The frame timestamp still comes from the extractor's manifest,
which reads the container's presentation timestamps, so time remains measured
rather than claimed.

Per-frame embeddings and per-frame scalars do not go in `claim` or `vector` at
all. 900 frames × 1152 dimensions as 900 rows is a row-store answering a
column-store question; instead `frame_vector` and `frame_metric` hold one row
per (video, space, observer) with the frame indices and the values as packed
blobs. One reel of SigLIP is one row of about 1.4 MB.
"""

from __future__ import annotations

import array
import bisect
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time

from . import CHANNELS, SCHEMA_VERSION

# The identity law, imported rather than restated. `atlas/identity.py` is where
# the three legal shapes of a video key are written down, and it is pure stdlib
# with no config and no side effects — importing it costs two module loads.
#
# The direction looks backwards (the engine importing the reader's package) and
# is deliberate: the alternative is a second copy of the rule inside this plane,
# and two copies of "what is a video key" drifting apart is precisely the defect
# this import exists to end. `atlas` is importable exactly when `vios` is, since
# they are sibling packages in one repository, so this adds no new way to fail.
from atlas import identity

_CHANNEL_SET = frozenset(CHANNELS)


class NotAnIdentity(ValueError):
    """A row named a video by something that is not a video's identity.

    A `ValueError` subclass on purpose: the shard replay already treats those as
    a skippable bad record, so an older `except` clause keeps working while the
    counter below can tell a refused identity apart from a torn line.
    """

# Shard schema versions this build can replay. The reason the list grows rather
# than moves: the channel holds the only copy of every earlier session's work, so
# a shard written last month must still import. A v1 shard replays with NULL
# frame columns, which is exactly what "this claim is about a shot" already
# means; a v1 or v2 shard carries no `coverage` section, which is exactly what
# "we do not know what this session had already finished" already means — the
# restore falls back to re-deriving it, which is what it does today.
SHARD_SCHEMAS = (1, 2, 3)

# How advanced a coverage state is, for the shard replay in `_replay`. Ten
# workers push shards into one channel and any of them can pull all of them, so
# two shards can carry the same (video, component) in different states and the
# import must be order-independent. Ranking makes it so: the higher state wins
# whichever arrives second.
#
# `failed` above `queued`/`running` because it records attempts actually spent,
# and `skipped` above `failed` because a skip is a decision and a failure is a
# pending question. `done` is the top and is never overwritten.
_STATE_RANK = {"queued": 0, "running": 1, "failed": 2, "skipped": 3, "done": 4}

# The coverage columns a shard carries. Everything left out is live-session
# state that means nothing anywhere else: `worker` and `token` name a lease
# holder, `lease_until`/`next_try_at` are clocks on a machine that has stopped,
# and `checkpoint` is a resume cursor into files that are gone.
_COVERAGE_COLS = ("video_key", "component", "state", "attempts", "revivals",
                  "seconds", "claims", "vectors", "observer_id", "last_error",
                  "started_at", "done_at")

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
    created_at  REAL NOT NULL,
    frame_idx   INTEGER,          -- v2. NULL = about a shot or the whole video
    frame_hi    INTEGER           -- v2. inclusive end of a run; NULL = one frame
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

-- v2. Per-frame embeddings, packed. One row per (video, space, observer) run:
-- `frames` is int32×n of frame indices, `data` is n×dim in `dtype`. A 900-frame
-- reel of SigLIP is one row of ~1.4 MB rather than 900 rows of base64, and a
-- similarity search reads it as one contiguous buffer instead of 900 seeks.
CREATE TABLE IF NOT EXISTS frame_vector (
    id          INTEGER PRIMARY KEY,
    uid         TEXT NOT NULL UNIQUE,
    video_key   TEXT NOT NULL,
    space       TEXT NOT NULL,    -- 'siglip','clip','clap' — never mix spaces
    dim         INTEGER NOT NULL,
    n           INTEGER NOT NULL, -- how many frames this row covers
    dtype       TEXT NOT NULL,    -- 'f16' normally, 'f32' where numpy is absent
    frames      BLOB NOT NULL,    -- int32 little-endian, len == n*4
    data        BLOB NOT NULL,    -- dtype little-endian, len == n*dim*itemsize
    observer_id TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fvec_space ON frame_vector(space, video_key);

-- v2. Per-frame scalars, packed the same way: brightness, sharpness, motion
-- magnitude, depth mean, aesthetic score, one audio class's activation. The
-- registry's own note on `perframe` — 27 million rows that would only ever be
-- queried one at a time — is why this is columnar rather than a claim per frame.
CREATE TABLE IF NOT EXISTS frame_metric (
    id          INTEGER PRIMARY KEY,
    uid         TEXT NOT NULL UNIQUE,
    video_key   TEXT NOT NULL,
    name        TEXT NOT NULL,    -- 'brightness','motion','depth_mean'…
    n           INTEGER NOT NULL,
    frames      BLOB NOT NULL,    -- int32 little-endian
    values_     BLOB NOT NULL,    -- float32 little-endian
    observer_id TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fmet_name ON frame_metric(name, video_key);

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

# Columns added after v1. Applied to an existing database in place, never by
# recreating it: this file holds the evidence of every video processed so far,
# and re-earning that costs the GPU hours it originally took. The ordering used
# by `coverage.py` is followed here too — table, then columns, then indexes —
# because an index over a column that does not exist yet is the one statement
# that turns a migration into a broken startup.
MIGRATIONS = (
    ("claim", "frame_idx", "INTEGER"),
    ("claim", "frame_hi", "INTEGER"),
)

# Built after the migration, for the same reason.
LATE_INDEXES = """
CREATE INDEX IF NOT EXISTS ix_claim_frame ON claim(video_key, frame_idx);
CREATE INDEX IF NOT EXISTS ix_claim_frun  ON claim(video_key, frame_idx, frame_hi);
"""


def _uid(*parts) -> str:
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def observer_id(component: str, model: str, revision: str = "",
                params: dict | None = None) -> str:
    """The observer id a pass would write, without registering a run.

    The reconcile step that marks coverage done for evidence that already came
    back in a shard needs to know which observer produced that evidence, and
    it must not do so by calling `Store.observer` — that would bump `runs` on
    the same row the restore just imported, claiming credit for a pass that ran
    elsewhere. Computing the id here, from the same bytes `observer()` hashes,
    keeps the two derivations from ever drifting apart: one function is the
    definition, the other is the side effect that uses it.
    """
    return observer_id_from(model, revision, params, component)


def observer_id_from(model: str, revision: str = "",
                     params: dict | None = None,
                     component: str = "") -> str:
    """Pure derivation, split out so both the writer and the reconciler use it.

    `component` is only a prefix in the id, not part of the hash: a component
    edits its prompt and the id changes, and the same model read by two
    components produces two ids because the prefix differs.
    """
    blob = json.dumps(params or {}, sort_keys=True, ensure_ascii=False)
    return f"{component}@{_uid(model, revision, blob)[:12]}"


def _numpy():
    """numpy if this interpreter has it, else None.

    Every machine that runs a model has numpy. A machine that only *reads* this
    database — a restore, a stats call, the schema check — may not, and the
    store refusing to import there would make the evidence unreadable on the one
    kind of host most likely to be inspecting it. So numpy is an optimisation
    for the packing path, not a dependency of the module.
    """
    try:
        import numpy as np  # noqa: PLC0415
        return np
    except Exception:
        return None


def _pack_i32(values) -> bytes:
    buf = array.array("i", [int(v) for v in values])
    if buf.itemsize != 4:                      # 'i' is 4 bytes everywhere we run
        buf = array.array("l", [int(v) for v in values])
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def _unpack_i32(blob: bytes) -> list:
    buf = array.array("i")
    buf.frombytes(blob)
    if sys.byteorder != "little":
        buf.byteswap()
    return list(buf)


def _pack_f32(values) -> bytes:
    buf = array.array("f", [float(v) for v in values])
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def _unpack_f32(blob: bytes) -> list:
    buf = array.array("f")
    buf.frombytes(blob)
    if sys.byteorder != "little":
        buf.byteswap()
    return list(buf)


def _pack_matrix(rows) -> tuple:
    """(dtype, blob) for an n×dim matrix.

    fp16 halves the file for a loss that is below the noise floor of cosine
    similarity over normalised embeddings — the values are already in [-1, 1]
    and the retrieval order does not change. `array` has no half type, so this
    is the one place numpy earns its keep; without it the store falls back to
    fp32 and records that in the `dtype` column rather than writing bytes a
    reader would misinterpret.
    """
    np = _numpy()
    if np is not None:
        m = np.asarray(rows, dtype=np.float16)
        if sys.byteorder != "little":
            m = m.byteswap()
        return "f16", m.tobytes()
    flat = []
    for row in rows:
        flat.extend(float(v) for v in row)
    return "f32", _pack_f32(flat)


def _unpack_matrix(blob: bytes, n: int, dim: int, dtype: str) -> list:
    """The matrix back as a list of `n` lists of `dim` floats."""
    if dtype == "f16":
        np = _numpy()
        if np is None:
            raise RuntimeError(
                "this frame_vector row is fp16 and numpy is not installed; "
                "install numpy to read per-frame embeddings")
        m = np.frombuffer(blob, dtype=np.float16)
        if sys.byteorder != "little":
            m = m.byteswap()
        return m.astype("float32").reshape(n, dim).tolist()
    flat = _unpack_f32(blob)
    return [flat[i * dim:(i + 1) * dim] for i in range(n)]


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
        self._migrate()
        self.conn.executescript(LATE_INDEXES)
        try:
            self.conn.executescript(FTS)
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        cur = self.conn.execute("SELECT v FROM meta WHERE k='schema'")
        row = cur.fetchone()
        found = SCHEMA_VERSION if row is None else int(row["v"])
        if found > SCHEMA_VERSION:
            # Forward is the one direction that cannot be recovered from. A
            # newer build may have written columns this code will silently drop
            # on the next export, so the shard would be quietly lossy.
            raise RuntimeError(
                f"evidence store at {path} is schema v{found}, this code "
                f"speaks v{SCHEMA_VERSION}. Update the code — do not let an "
                f"older build write to a newer database.")
        if row is None or found != SCHEMA_VERSION:
            # The columns are already in place by here; `_migrate` is additive
            # and idempotent, so recording the new version is the last step and
            # not the first. A crash between the two leaves a database that
            # migrates again harmlessly on the next open.
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(k,v) VALUES('schema',?)",
                (str(SCHEMA_VERSION),))
            self.migrated_from = found if row is not None else 0
        else:
            self.migrated_from = 0
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after v1, in place.

        SQLite makes an added column NULL in every existing row, which is
        exactly the meaning v2 assigns to a NULL `frame_idx`: *this claim is
        about a shot, not a frame*. So the migration needs no backfill and no
        rewrite — every v1 row is already correct under the v2 schema.
        """
        seen: dict = {}
        for table, name, spec in MIGRATIONS:
            have = seen.get(table)
            if have is None:
                have = {r["name"] for r in
                        self.conn.execute(f"PRAGMA table_info({table})")}
                seen[table] = have
            if name not in have:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {spec}")
                have.add(name)

    def secondary(self) -> "Store":
        """A second handle on the same file, for a second thread.

        One `Store` is one connection, and a connection carries one implicit
        transaction. Two threads sharing it therefore share a `commit()`: the
        publisher folding its watermark writes would commit whatever pass was
        halfway through writing its claims, and a `SELECT` cursor still open on
        one thread can be invalidated by the other thread's commit outright.
        WAL exists precisely so that a reader and a writer can each have their
        own connection, so the publisher gets one.

        Deliberately does **not** create schema, run migrations, mint a
        `site_id` or record a schema version. Those must happen exactly once,
        from the handle that opened the file first; a second copy of them is a
        second chance to disagree. This is a plain connection with the same
        pragmas, over a file another `Store` has already prepared.
        """
        other = Store.__new__(Store)
        other.path = self.path
        other.conn = sqlite3.connect(self.path, check_same_thread=False,
                                     timeout=30.0)
        other.conn.row_factory = sqlite3.Row
        # journal_mode is a property of the file, not the connection, so it is
        # already WAL; synchronous and busy_timeout are per connection and have
        # to be asked for again.
        other.conn.execute("PRAGMA synchronous=NORMAL")
        other.conn.execute("PRAGMA foreign_keys=OFF")
        other.conn.execute("PRAGMA busy_timeout=30000")
        other.fts = self.fts
        other.migrated_from = 0
        return other

    # ── videos ──────────────────────────────────────────────────────────
    def add_video(self, video_key: str, **fields) -> bool:
        """Register a video. Idempotent; later calls fill in blanks only.

        Blanks-only matters: `probe` learns the duration, the capture record
        knows the uploader, and neither should clobber the other's column
        because it happened to run second.

        Returns False, having written nothing, if the key is not shaped like an
        identity. This table is the identity table — a key exists because there
        is a row here — so this is the one door through which a new video enters
        the archive, and it is the right place to refuse a name that is not one.
        A bare `10` is a Telegram message id, `frames_8` is a working directory,
        and adopting either as an identity is exactly how an archive of thirty
        videos came to report sixty-two. The caller counts the refusal and says
        so; raising would abort a sweep over one bad row, which is worse.
        """
        if not identity.looks_canonical(video_key):
            return False
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
        return True

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

    def video_key_for_msg(self, msg_id) -> str:
        """The video that lives at this Telegram message id, or "".

        The store already knows the answer to "what is message 38?" — `msg_id`
        is filled in for every row the capture ledger synced — so a producer
        holding nothing but a number can resolve it here instead of adopting the
        number as a name. Ambiguity resolves to nothing rather than to a guess:
        two rows on one message id means the archive is wrong about something
        more important than this lookup.
        """
        try:
            n = int(msg_id)
        except (TypeError, ValueError):
            return ""
        if n <= 0:
            return ""
        rows = self.conn.execute(
            "SELECT video_key FROM video WHERE msg_id=? LIMIT 2", (n,)).fetchall()
        return str(rows[0][0]) if len(rows) == 1 else ""

    def video_key_for_sha(self, sha256: str) -> str:
        """The video whose bytes hash to this, or "".

        Same shape of question as `video_key_for_msg` and the same refusal to
        guess. Two videos sharing a digest is a twin pair — two permalinks over
        one file — and picking either one of them here would be arbitrary.
        """
        s = str(sha256 or "").strip().lower()
        if len(s) < 32:
            return ""
        rows = self.conn.execute(
            "SELECT video_key FROM video WHERE lower(sha256)=? LIMIT 2",
            (s,)).fetchall()
        return str(rows[0][0]) if len(rows) == 1 else ""

    def rehome_key(self, value, msg_aliases: dict | None = None) -> str:
        """The identity a legacy spelling meant, or "" when nothing can say.

        Refusing a bad name is right for a *new* write — nothing is lost, because
        the caller still holds the file and can resolve it properly. It is wrong
        for a shard already in the channel: those are permanent, an older build
        wrote some of them naming videos by Telegram message id, and dropping
        their claims would throw away transcripts that cost GPU hours to earn
        while the video they describe sits right there under its shortcode.

        So the restore path resolves before it refuses, the same way
        `atlas.identity` rehomes evidence rows onto the key its alias map knows.
        Two witnesses, strongest first: this database's own `video.msg_id`, then
        the capture ledger's view of the channel passed in as `{msg_id: key}`.
        Both refuse to guess — `video_key_for_msg` drops a message id claimed by
        two rows, and `_ledger_index` drops one claimed by two keys.

        Never a mint. A spelling this cannot resolve is refused, because minting
        an identity for a message id is the defect itself.
        """
        n = identity.msg_id_in(value)
        if n <= 0:
            return ""
        key = self.video_key_for_msg(n)
        if not key and msg_aliases:
            key = str(msg_aliases.get(n) or msg_aliases.get(str(n)) or "")
        return key if identity.looks_canonical(key) else ""


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

    def _shot_spans(self, video_key: str) -> tuple:
        """(starts, indices) sorted by t0, for turning a frame time into a shot.

        A per-frame pass emits hundreds of claims per video and each one needs
        its containing shot, so this is a binary search over a prepared list
        rather than `Job.shot_at`'s linear scan: at 900 frames × 40 shots the
        scan is 36,000 comparisons per pass per video, which is pure waste when
        the boundaries are already sorted.
        """
        rows = [(r["t0"], r["t1"], r["idx"]) for r in self.conn.execute(
            "SELECT idx,t0,t1 FROM shot WHERE video_key=? ORDER BY t0",
            (video_key,))]
        return [r[0] for r in rows], rows

    @staticmethod
    def _shot_of(t: float, starts: list, rows: list):
        """Which shot contains this timestamp, or None if there are no shots.

        Clamped rather than rejected at both ends: the extractor's last frame
        can land a few microseconds past the final shot boundary because the
        boundary came from a detector and the timestamp came from the container.
        Dropping that frame's evidence over a rounding difference would put a
        hole in the coverage this whole change exists to close.
        """
        if not rows:
            return None
        i = bisect.bisect_right(starts, t) - 1
        if i < 0:
            return int(rows[0][2])
        return int(rows[i][2])

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
        oid = observer_id(component, model, revision, params)
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

    def evidence_by_observer(self, observer_ids=None) -> dict:
        """What each observer has actually written, per video.

        `{video_key: {observer_id: {"claims": n, "vectors": n}}}` — the question
        the coverage reconcile asks after a restore, and the reason a session
        that dies at hour twelve does not repeat hour one. A shard replay puts
        rows back into this database without touching the coverage table, so
        without this read the two disagree: the evidence is present and the work
        table still says `queued`.

        All four evidence tables are counted, not just `claim`. `visual-embed`
        and `clip-embed` write nothing but vectors, `depth` writes nothing but
        packed metrics — asking only about claims would send every embedding
        pass in the archive round again for rows it already holds.

        `observer_ids`, when given, restricts the scan to the observers this
        build would produce. That restriction is the whole correctness argument
        for reconciling at all: evidence from a superseded revision must *not*
        satisfy the current one, or bumping a revision to get better output
        would silently mark the better output as already done.
        """
        want = set(observer_ids) if observer_ids is not None else None
        if want is not None and not want:
            return {}

        out: dict = {}
        # Chunked so a large selection cannot overrun SQLite's variable limit
        # (999 by default). Thirty-four components is well inside it today; a
        # future build that reconciles several revisions at once is not.
        def _chunks(ids, n=400):
            ids = list(ids)
            for i in range(0, len(ids), n):
                yield ids[i:i + n]

        plans = (("claim", "claims"), ("vector", "vectors"),
                 ("frame_vector", "vectors"), ("frame_metric", "vectors"))
        for table, bucket in plans:
            if want is None:
                sql = (f"SELECT video_key, observer_id, COUNT(*) n "
                       f"FROM {table} GROUP BY video_key, observer_id")
                batches = [((), sql)]
            else:
                batches = []
                for part in _chunks(want):
                    marks = ",".join("?" * len(part))
                    batches.append((tuple(part), (
                        f"SELECT video_key, observer_id, COUNT(*) n "
                        f"FROM {table} WHERE observer_id IN ({marks}) "
                        f"GROUP BY video_key, observer_id")))
            for params, sql in batches:
                for r in self.conn.execute(sql, params):
                    per = out.setdefault(r["video_key"], {})
                    e = per.setdefault(r["observer_id"],
                                       {"claims": 0, "vectors": 0})
                    e[bucket] += int(r["n"] or 0)
        return out

    def observed_components(self) -> dict:
        """`{component: [observer_id, …]}` for everything this database holds.

        Used to report what a restore brought back that the current build no
        longer recognises — a pass whose revision moved on. Those rows are the
        audit trail and are never deleted; naming them is how a "reconciled 0"
        becomes a legible sentence instead of a suspicion that nothing worked.
        """
        out: dict = {}
        for r in self.conn.execute(
                "SELECT observer_id, component FROM observer "
                "ORDER BY component, observer_id"):
            out.setdefault(r["component"], []).append(r["observer_id"])
        return out

    # ── claims ──────────────────────────────────────────────────────────
    def add_claims(self, video_key: str, observer_id: str, claims: list) -> int:
        """Write a batch of claims. Returns how many were new.

        `claims` is a list of dicts: channel, kind, value, and optionally
        shot_idx, num, confidence, ordinal. Times are ignored if supplied —
        they are looked up from the shot table, because a model's opinion about
        when something happened is not evidence.

        A per-frame claim carries `frame_idx` and `frame_t` instead of
        `shot_idx`, and optionally `frame_hi`/`frame_t1` to make it a run. The
        shot index is then derived from the frame's timestamp, and `t0`/`t1`
        come from the frame times — which the extractor read out of the
        container's presentation timestamps, so time is still measured. A run
        that spans a shot boundary keeps the shot it started in; the frame range
        is the precise answer and `shot_idx` is the coarse one.
        """
        times = self._shot_times(video_key)
        starts, spans = self._shot_spans(video_key)
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

            fi = c.get("frame_idx")
            fi = None if fi is None else int(fi)
            fhi = c.get("frame_hi")
            fhi = None if fhi is None else int(fhi)
            if fhi is not None and fi is not None and fhi < fi:
                raise ValueError(
                    f"{video_key}: frame run {fi}..{fhi} ends before it starts")

            si = c.get("shot_idx")
            si = None if si is None else int(si)
            if fi is not None and si is None:
                ft = c.get("frame_t")
                si = None if ft is None else self._shot_of(float(ft), starts, spans)
            if si is not None and si not in times:
                raise ValueError(
                    f"{video_key}: claim references shot {si}, which does not "
                    f"exist ({len(times)} shots). Run the shots pass first.")

            if fi is not None and c.get("frame_t") is not None:
                t0 = float(c["frame_t"])
                t1 = float(c.get("frame_t1", t0))
            elif si is not None:
                t0, t1 = times[si]
            else:
                t0, t1 = whole

            val = c.get("value")
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            ordinal = int(c.get("ordinal", n))
            # The uid drops the frame terms entirely when there is no frame, so
            # every claim written by v1 hashes to exactly the same value under
            # v2 and shard replay stays idempotent across the version boundary.
            if fi is None:
                uid = _uid(video_key, observer_id, ch, c.get("kind", ""), si,
                           ordinal)
            else:
                uid = _uid(video_key, observer_id, ch, c.get("kind", ""), si,
                           ordinal, "f", fi, fhi)
            rows.append((
                uid, video_key, si, t0, t1, ch, str(c.get("kind", "")),
                None if val is None else str(val),
                c.get("num"), float(c.get("confidence", 1.0)),
                observer_id, ordinal, now, fi, fhi))
        if not rows:
            return 0
        # `cursor.rowcount`, not `conn.total_changes`. The FTS trigger writes to
        # three shadow tables per claim, and total_changes counts every one of
        # them — it reported 17 for 3 claims. rowcount counts only the rows the
        # statement itself inserted, which is what "how many were new" means.
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO claim(uid,video_key,shot_idx,t0,t1,channel,"
            "kind,value,num,confidence,observer_id,ordinal,created_at,"
            "frame_idx,frame_hi) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        written = max(cur.rowcount, 0)
        self.conn.commit()
        return written

    def frame_claims(self, video_key: str, frame_idx: int,
                     channel: str = "") -> list:
        """Every claim covering one frame, runs included.

        This is the query the run-length encoding exists to serve: a claim that
        collapsed frames 100–142 into one row is still the answer to "what was
        on screen at frame 117".
        """
        sql = ("SELECT * FROM claim WHERE video_key=? AND frame_idx IS NOT NULL "
               "AND frame_idx <= ? AND COALESCE(frame_hi, frame_idx) >= ?")
        args: list = [video_key, frame_idx, frame_idx]
        if channel:
            sql += " AND channel=?"
            args.append(channel)
        sql += " ORDER BY channel, kind, ordinal"
        return [dict(r) for r in self.conn.execute(sql, args)]

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

    # ── per-frame vectors and metrics ───────────────────────────────────
    def add_frame_vectors(self, video_key: str, space: str, frames,
                          matrix, observer_id: str) -> bool:
        """Store one video's per-frame embeddings as a single row.

        `frames` and `matrix` must be parallel: `frames[i]` is the frame index
        whose embedding is `matrix[i]`. The pairing is stored explicitly rather
        than assumed contiguous, because a pass that fails on one unreadable
        JPEG should record the 899 frames it did read with their real indices,
        not shift every subsequent frame by one.
        """
        frames = [int(f) for f in frames]
        rows = [list(r) for r in matrix]
        if len(frames) != len(rows):
            raise ValueError(
                f"{video_key}: {len(frames)} frame indices for {len(rows)} "
                f"embeddings — these must be parallel")
        if not rows:
            return False
        dim = len(rows[0])
        if any(len(r) != dim for r in rows):
            raise ValueError(
                f"{video_key}: ragged embedding matrix in space {space!r}; "
                f"every frame must have {dim} dimensions")
        dtype, blob = _pack_matrix(rows)
        uid = _uid(video_key, observer_id, space, "frames")
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO frame_vector(uid,video_key,space,dim,n,"
            "dtype,frames,data,observer_id,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (uid, video_key, space, dim, len(rows), dtype,
             _pack_i32(frames), blob, observer_id, time.time()))
        self.conn.commit()
        return cur.rowcount > 0

    def frame_vectors(self, video_key: str, space: str = "",
                      observer_id: str = "") -> list:
        """Unpacked per-frame embeddings: [{space, observer_id, frames, values}].

        `values` is a list of lists, parallel to `frames`. The caller that wants
        speed rather than convenience should read the blobs directly — see
        `frame_vector_blobs`.
        """
        sql = ("SELECT space,observer_id,dim,n,dtype,frames,data "
               "FROM frame_vector WHERE video_key=?")
        args: list = [video_key]
        for col, val in (("space", space), ("observer_id", observer_id)):
            if val:
                sql += f" AND {col}=?"
                args.append(val)
        out = []
        for r in self.conn.execute(sql, args):
            out.append({
                "space": r["space"], "observer_id": r["observer_id"],
                "dim": r["dim"], "frames": _unpack_i32(r["frames"]),
                "values": _unpack_matrix(r["data"], r["n"], r["dim"],
                                        r["dtype"])})
        return out

    def frame_vector_blobs(self, space: str, limit: int = 0) -> list:
        """Every video's packed rows in one space, still packed.

        The search path in Atlas wants to run cosine over the whole archive
        without turning 4 million embeddings into Python floats first, so this
        hands back the raw buffers and lets numpy do the work.
        """
        sql = ("SELECT video_key,observer_id,dim,n,dtype,frames,data "
               "FROM frame_vector WHERE space=? ORDER BY video_key")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(sql, (space,))]

    def add_frame_metric(self, video_key: str, name: str, frames, values,
                         observer_id: str) -> bool:
        """Store one per-frame scalar series as a single row."""
        frames = [int(f) for f in frames]
        values = [float(v) for v in values]
        if len(frames) != len(values):
            raise ValueError(
                f"{video_key}: {len(frames)} frame indices for {len(values)} "
                f"values of {name!r} — these must be parallel")
        if not values:
            return False
        uid = _uid(video_key, observer_id, "metric", name)
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO frame_metric(uid,video_key,name,n,frames,"
            "values_,observer_id,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (uid, video_key, name, len(values), _pack_i32(frames),
             _pack_f32(values), observer_id, time.time()))
        self.conn.commit()
        return cur.rowcount > 0

    def frame_metrics(self, video_key: str, name: str = "") -> list:
        """Unpacked per-frame scalars: [{name, observer_id, frames, values}]."""
        sql = ("SELECT name,observer_id,n,frames,values_ FROM frame_metric "
               "WHERE video_key=?")
        args: list = [video_key]
        if name:
            sql += " AND name=?"
            args.append(name)
        sql += " ORDER BY name"
        return [{"name": r["name"], "observer_id": r["observer_id"],
                 "frames": _unpack_i32(r["frames"]),
                 "values": _unpack_f32(r["values_"])}
                for r in self.conn.execute(sql, args)]

    def max_frame_vector_id(self) -> int:
        return self.conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM frame_vector").fetchone()[0]

    def max_frame_metric_id(self) -> int:
        return self.conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM frame_metric").fetchone()[0]

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
                     component: str = "",
                     lo_fvec: int = 0, hi_fvec: int = 0,
                     lo_fmet: int = 0, hi_fmet: int = 0) -> dict:
        """Write claims and vectors in an id range to a gzipped JSONL file.

        Why a range and not "everything since a timestamp": ids are assigned by
        SQLite in insertion order and never move, so a range is exact and
        replayable. Wall-clock is not — a session whose clock is wrong, or that
        runs across a DST boundary, would skip or duplicate rows.

        Why JSONL and not a copy of the .db: shards are *merged* on restore,
        from ten different accounts that each processed a different partition.
        Merging ten SQLite files means attaching and inserting anyway; JSONL
        skips the step and compresses better.

        Every table carries its **own** id range. An embedding pass writes
        vectors and no claims at all, so keying the vector export off the claim
        range would silently drop every embedding the cohort produced — and it
        would look like a successful upload. The same trap applies twice more
        now that per-frame vectors and per-frame metrics are their own tables:
        a pass that writes only `frame_metric` rows must still publish them.

        `coverage` is the exception to "id range": it is keyed by
        `(video_key, component)` and has no id, so it travels for the videos this
        shard touches, in whatever state they are in now. That is the point of
        carrying it at all — a database rebuilt from shards alone knew everything
        that had been *learned* and nothing about what had been *done*, so it
        re-attempted three hours of work whose evidence it was already holding.
        Only settled rows go: see `_settled_coverage`.
        """
        hi_id = hi_id or self.max_claim_id()
        hi_vec = hi_vec or self.max_vector_id()
        hi_fvec = hi_fvec or self.max_frame_vector_id()
        hi_fmet = hi_fmet or self.max_frame_metric_id()

        # The set of videos this shard touches, from any side.
        keys = ("(SELECT video_key FROM claim WHERE id>? AND id<=? "
                "UNION SELECT video_key FROM vector WHERE id>? AND id<=? "
                "UNION SELECT video_key FROM frame_vector WHERE id>? AND id<=? "
                "UNION SELECT video_key FROM frame_metric WHERE id>? AND id<=?)")
        span = (lo_id, hi_id, lo_vec, hi_vec, lo_fvec, hi_fvec,
                lo_fmet, hi_fmet)

        n_claims = n_vectors = n_fvec = n_fmet = n_cov = 0
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
            fh.write(json.dumps({
                "_": "vios-evidence-shard", "schema": SCHEMA_VERSION,
                "component": component, "lo_id": lo_id, "hi_id": hi_id,
                "lo_vec": lo_vec, "hi_vec": hi_vec,
                "lo_fvec": lo_fvec, "hi_fvec": hi_fvec,
                "lo_fmet": lo_fmet, "hi_fmet": hi_fmet,
                "at": time.time()}) + "\n")

            for r in self.conn.execute(
                    "SELECT * FROM observer WHERE observer_id IN ("
                    " SELECT observer_id FROM claim WHERE id>? AND id<=?"
                    " UNION SELECT observer_id FROM vector WHERE id>? AND id<=?"
                    " UNION SELECT observer_id FROM frame_vector WHERE id>? AND id<=?"
                    " UNION SELECT observer_id FROM frame_metric WHERE id>? AND id<=?)",
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
            for r in self.conn.execute(
                    "SELECT * FROM frame_vector WHERE id>? AND id<=? ORDER BY id",
                    (lo_fvec, hi_fvec)):
                d = dict(r)
                d.pop("id", None)
                d["frames"] = d["frames"].hex()
                d["data"] = d["data"].hex()
                fh.write(json.dumps({"t": "frame_vector", **d}) + "\n")
                n_fvec += 1
            for r in self.conn.execute(
                    "SELECT * FROM frame_metric WHERE id>? AND id<=? ORDER BY id",
                    (lo_fmet, hi_fmet)):
                d = dict(r)
                d.pop("id", None)
                d["frames"] = d["frames"].hex()
                d["values_"] = d["values_"].hex()
                fh.write(json.dumps({"t": "frame_metric", **d}) + "\n")
                n_fmet += 1
            for d in self._settled_coverage(keys, span):
                fh.write(json.dumps({"t": "coverage", **d},
                                    ensure_ascii=False) + "\n")
                n_cov += 1

        return {"path": path, "claims": n_claims, "vectors": n_vectors,
                "frame_vectors": n_fvec, "frame_metrics": n_fmet,
                "coverage": n_cov,
                "lo_id": lo_id, "hi_id": hi_id,
                "lo_vec": lo_vec, "hi_vec": hi_vec,
                "lo_fvec": lo_fvec, "hi_fvec": hi_fvec,
                "lo_fmet": lo_fmet, "hi_fmet": hi_fmet,
                "bytes": os.path.getsize(path)}

    def _settled_coverage(self, keys: str, span: tuple) -> list:
        """The coverage rows worth shipping, for the videos in this shard.

        Only `done`, `skipped` and `failed`. A `queued` row carries no
        information — the target derives it from the registry the moment it
        reconciles — and a `running` row is a live lease on a machine the reader
        cannot see, so importing it would hand a video to a worker that no longer
        exists and wait `LEASE_SECONDS` to find out.

        `last_error` is truncated here rather than at the reader. A shard is
        gzipped JSONL and one 8 KB traceback repeated across sixty videos is
        larger than the evidence it travels with; 500 characters is the whole of
        every message this codebase raises, because `SkipPass` and the runners
        already clip theirs to 200.
        """
        cols = ",".join(_COVERAGE_COLS)
        rows = []
        try:
            cur = self.conn.execute(
                f"SELECT {cols} FROM coverage "
                f"WHERE state IN ('done','skipped','failed') "
                f"AND video_key IN {keys} ORDER BY component, video_key", span)
        except sqlite3.OperationalError:
            # No `coverage` table: this database has never had a `Coverage` on
            # it, so nothing has been *done* and there is nothing to ship. Not
            # created here — an export is a read, and a store that only ever
            # exported should not grow a table because it did.
            return rows
        for r in cur:
            d = dict(r)
            if d.get("last_error"):
                d["last_error"] = str(d["last_error"])[:500]
            rows.append(d)
        return rows

    def import_shard(self, path: str, msg_aliases: dict | None = None) -> dict:
        """Replay a shard into this database. Idempotent by uid.

        This is the restore path: ten accounts each push shards to the channel,
        any one of them can pull all of them and end up with the union. Order
        does not matter, duplicates do not matter, and a partial shard from a
        session that died mid-upload is simply a shorter file.

        Every schema in `SHARD_SCHEMAS` replays, not only the current one. A v1
        shard has no frame columns and no per-frame tables; its claims land with
        NULL `frame_idx`, which is precisely what they meant when they were
        written. Refusing them would strand every hour of work done before this
        version, since the channel is the only place that work still exists.

        One caveat worth stating because it is the only way to misuse this: a
        shard's `coverage` section describes the videos it touches as they stood
        when it was written, while their claims may have gone up in an *earlier*
        shard. So importing a hand-picked subset of shards can land a `done` state
        whose evidence is in a file you did not import. A restore walks the whole
        channel, which is why it is correct there; "import this one shard into an
        empty database and expect a complete database" was never a supported
        reading and is less true now than it was.

        Records naming a video by something that is not an identity are resolved
        if the archive can say what they meant and counted under `refused` if it
        cannot. The channel is permanent, so a shard an older build wrote under a
        Telegram message id will still be replayed years from now — and importing
        it *as a video* is how the duplicate would come back after being fixed.
        So the message id is looked up (`rehome_key`) and its evidence lands on
        the shortcode it always belonged to, counted under `rehomed`; only a
        spelling nothing can resolve is dropped. `msg_aliases` is the capture
        ledger's `{msg_id: video_key}` view, optional and only consulted when
        this database's own `video` table cannot answer.
        """
        counts = {"claim": 0, "vector": 0, "video": 0, "shot": 0,
                  "observer": 0, "artifact": 0, "frame_vector": 0,
                  "frame_metric": 0, "coverage": 0, "skipped": 0}
        refused = rehomed = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            header = json.loads(fh.readline() or "{}")
            if header.get("_") != "vios-evidence-shard":
                raise ValueError(f"{path} is not an evidence shard")
            if int(header.get("schema", 0)) not in SHARD_SCHEMAS:
                raise ValueError(
                    f"{path} is schema v{header.get('schema')}, this code "
                    f"replays {'/'.join('v%d' % v for v in SHARD_SCHEMAS)}")
            if int(header.get("schema", 0)) >= 3:
                # Before the loop, not on the first coverage line: `executescript`
                # issues an implicit COMMIT, and doing that mid-replay would
                # commit half an import for no reason.
                self._ensure_coverage_table()
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
                    if self._replay(t, rec, msg_aliases):
                        rehomed += 1
                    counts[t] += 1
                except NotAnIdentity:
                    refused += 1
                except (sqlite3.Error, ValueError, KeyError):
                    counts["skipped"] += 1
        self.conn.commit()
        if refused:
            # Added after the loop, never before: the loop uses `counts` as the
            # set of record types it knows, and neither of these is one.
            counts["refused"] = refused
        if rehomed:
            counts["rehomed"] = rehomed
        return counts

    def _replay(self, t: str, rec: dict, msg_aliases: dict | None = None) -> bool:
        """Land one shard record. True when its key had to be rehomed first.

        One guard for every record type that names a video, rather than one per
        branch below. `video` is the identity table and the others reference it,
        so a spelling refused here cannot arrive by a side door — and a claim
        replayed under a name with no `video` row is an orphan, which is the same
        defect wearing a different table's clothes.

        A rehomed `video` record lands as a no-op: `INSERT OR IGNORE` against a
        row that already exists, which is exactly right. The archive already has
        that video under its identity — that is *how* the spelling resolved — and
        the record's own fields describe the same video, so there is nothing to
        add and nothing to overwrite.
        """
        rehomed = False
        if "video_key" in rec and not identity.looks_canonical(rec["video_key"]):
            spelling = str(rec["video_key"])
            true_key = self.rehome_key(spelling, msg_aliases)
            if not true_key:
                raise NotAnIdentity(
                    f"{t} record names {spelling[:40]!r}, "
                    f"which is not a video identity")
            rec["video_key"] = true_key
            rehomed = True
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
            # `frame_idx`/`frame_hi` are absent from a v1 shard, and `.get`
            # returning None is the correct reading of that: the claim was
            # about a shot. No branch on the shard's schema is needed.
            self.conn.execute(
                "INSERT OR IGNORE INTO claim(uid,video_key,shot_idx,t0,t1,"
                "channel,kind,value,num,confidence,observer_id,ordinal,"
                "created_at,frame_idx,frame_hi) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["uid"], rec["video_key"], rec.get("shot_idx"),
                 rec.get("t0"), rec.get("t1"), rec["channel"], rec["kind"],
                 rec.get("value"), rec.get("num"), rec.get("confidence", 1.0),
                 rec["observer_id"], rec.get("ordinal", 0),
                 rec.get("created_at", time.time()),
                 rec.get("frame_idx"), rec.get("frame_hi")))
        elif t == "frame_vector":
            frames = bytes.fromhex(rec["frames"])
            data = bytes.fromhex(rec["data"])
            n = int(rec["n"])
            dim = int(rec["dim"])
            width = 2 if rec.get("dtype", "f16") == "f16" else 4
            # A shard that lost bytes in transit would otherwise land as a row
            # that unpacks into garbage vectors, and a wrong embedding is worse
            # than a missing one: it returns confident nonsense from search.
            if len(frames) != n * 4 or len(data) != n * dim * width:
                raise ValueError(
                    f"frame_vector {rec.get('uid','?')}: blob sizes disagree "
                    f"with n={n} dim={dim} dtype={rec.get('dtype')}")
            self.conn.execute(
                "INSERT OR IGNORE INTO frame_vector(uid,video_key,space,dim,n,"
                "dtype,frames,data,observer_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (rec["uid"], rec["video_key"], rec["space"], dim, n,
                 rec.get("dtype", "f16"), frames, data, rec["observer_id"],
                 rec.get("created_at", time.time())))
        elif t == "frame_metric":
            frames = bytes.fromhex(rec["frames"])
            vals = bytes.fromhex(rec["values_"])
            n = int(rec["n"])
            if len(frames) != n * 4 or len(vals) != n * 4:
                raise ValueError(
                    f"frame_metric {rec.get('uid','?')}: blob sizes disagree "
                    f"with n={n}")
            self.conn.execute(
                "INSERT OR IGNORE INTO frame_metric(uid,video_key,name,n,"
                "frames,values_,observer_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (rec["uid"], rec["video_key"], rec["name"], n, frames, vals,
                 rec["observer_id"], rec.get("created_at", time.time())))
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
        elif t == "coverage":
            self._replay_coverage(rec)
        return rehomed

    def _replay_coverage(self, rec: dict) -> None:
        """Land one coverage row, more-advanced-state-wins.

        The rule has to be commutative, because shards arrive in whatever order
        the channel hands them over and ten workers write them independently. So
        the comparison is on `_STATE_RANK` and nothing else: whichever of the two
        states is further along survives, and importing the same shard twice — or
        two shards in either order — reaches the same row.

        What it must never do is walk a state backwards. A local `done` is a
        finished pass with its claims in this file; overwriting it with a remote
        `queued` would schedule work that is already complete, which is the exact
        failure this section exists to prevent, in the opposite direction.
        `INSERT … ON CONFLICT DO UPDATE … WHERE` puts that test in the statement
        rather than in a read-then-write that two threads could interleave.

        `lease_until`, `next_try_at`, `worker`, `token` and `checkpoint` are left
        at their defaults on insert and untouched on update. A lease belongs to
        the machine holding it.
        """
        state = str(rec.get("state") or "")
        rank = _STATE_RANK.get(state)
        if rank is None or rank <= 0:
            # Not a state worth importing: unknown, or `queued`, which the target
            # derives for itself. Counted as a replayed row either way — the line
            # was understood, and it asked for nothing.
            return
        self._ensure_coverage_table()
        cols = ",".join(_COVERAGE_COLS)
        marks = ",".join("?" * len(_COVERAGE_COLS))
        self.conn.execute(
            f"INSERT INTO coverage({cols}) VALUES({marks}) "
            f"ON CONFLICT(video_key, component) DO UPDATE SET "
            f"  state       = excluded.state,"
            f"  attempts    = MAX(COALESCE(coverage.attempts,0),"
            f"                    COALESCE(excluded.attempts,0)),"
            f"  revivals    = MAX(COALESCE(coverage.revivals,0),"
            f"                    COALESCE(excluded.revivals,0)),"
            f"  seconds     = COALESCE(excluded.seconds, coverage.seconds),"
            f"  claims      = COALESCE(excluded.claims, coverage.claims),"
            f"  vectors     = COALESCE(excluded.vectors, coverage.vectors),"
            f"  observer_id = COALESCE(excluded.observer_id, coverage.observer_id),"
            f"  last_error  = excluded.last_error,"
            f"  started_at  = COALESCE(coverage.started_at, excluded.started_at),"
            f"  done_at     = COALESCE(excluded.done_at, coverage.done_at) "
            f"WHERE ? > (CASE coverage.state"
            f"             WHEN 'done' THEN 4 WHEN 'skipped' THEN 3"
            f"             WHEN 'failed' THEN 2 WHEN 'running' THEN 1"
            f"             ELSE 0 END)",
            [rec.get(c) for c in _COVERAGE_COLS] + [rank])

    def _ensure_coverage_table(self) -> None:
        """Create `coverage` if this database has never had a `Coverage` on it.

        `coverage` is owned by `coverage.py` and normally created when a
        `Coverage` is constructed over the store's connection — but the restore
        path opens a `Store` and replays shards, and on a genuinely fresh
        database that happens before any engine has built a `Coverage`. Without
        this the coverage section of every shard would land in the
        `except sqlite3.Error` arm as `skipped`, which is exactly the silent loss
        the section was added to prevent.

        Idempotent, memoised, and using coverage.py's own DDL rather than a copy
        of it — a second definition of the table is a second chance for the two to
        drift.
        """
        if getattr(self, "_cov_table", False):
            return
        from .coverage import SCHEMA as COVERAGE_SCHEMA  # noqa: PLC0415
        self.conn.executescript(COVERAGE_SCHEMA)
        self._cov_table = True

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
            "frame_claims": q("SELECT COUNT(*) FROM claim "
                              "WHERE frame_idx IS NOT NULL"),
            # Rows are run-length, so the honest measure of per-frame coverage
            # is frames spanned, not rows written. A pass that collapsed 900
            # frames into 12 runs has still read all 900.
            "frames_claimed": q(
                "SELECT COALESCE(SUM(COALESCE(frame_hi,frame_idx)-frame_idx+1),0)"
                " FROM claim WHERE frame_idx IS NOT NULL"),
            "vectors": q("SELECT COUNT(*) FROM vector"),
            "frame_vectors": q("SELECT COUNT(*) FROM frame_vector"),
            "frame_vector_frames": q(
                "SELECT COALESCE(SUM(n),0) FROM frame_vector"),
            "frame_metrics": q("SELECT COUNT(*) FROM frame_metric"),
            "frame_metric_frames": q(
                "SELECT COALESCE(SUM(n),0) FROM frame_metric"),
            "observers": q("SELECT COUNT(*) FROM observer"),
            "artifacts": q("SELECT COUNT(*) FROM artifact"),
            "shards": q("SELECT COUNT(*) FROM shard"),
            "by_channel": by_channel,
            "by_observer": by_observer,
            "fts": self.fts,
            "schema": SCHEMA_VERSION,
            "bytes": os.path.getsize(self.path) if os.path.exists(self.path) else 0,
        }

    def close(self) -> None:
        try:
            self.checkpoint()
            self.conn.close()
        except sqlite3.Error:
            pass
