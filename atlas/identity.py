"""
atlas.identity — one video, one name.

THE LAW
    A video's identity is the Instagram shortcode in its permalink. A video that
    was never on Instagram — dropped into the channel by hand — is `up_<msg_id>`.
    A video that was never on Instagram *or* in the channel, adopted from a
    folder on disk, is `file_<sha256[:16]>`. Nothing else is ever a video's
    identity. Every other spelling some table happens to hold (a Telegram
    message id, `tg93`, a folder named `frames_8`, a filename, a scratch path)
    is an **alias**, and lives in `video_alias` pointing at the one true key.

WHY THIS FILE EXISTS
    Home said 62 videos over an archive of 30. That was not a display bug. Two
    generations of the pipeline named the same reel two different ways — the
    capture plane by shortcode, the older harvest by Telegram message id — and
    the two read models, `moments` and `video_index`, unioned them without ever
    resolving the two spellings. So a reel showing 19 findings had another 186
    filed under its other name, and dense search ranked the halves against each
    other.

    The map needed to fix it was already in the database: `video.msg_id` is
    populated for every row, and the capture ledger holds the same fact
    independently. Nothing was missing. What was missing was a *place* — no
    single module owned the question "are these two strings the same video?", so
    eleven callers each answered it with their own string munging.

THE RULES, which are what stop it recurring
    1. `video` is the identity table. A key exists if and only if it has a
       `video` row. Every other table references it.
    2. Nothing writes a video key it did not resolve. `resolve()` is the door,
       `Resolver` is the fast version for hot loops, and `reflect.normalize_key`
       is wired to it — so the existing callers are covered without knowing this
       module is here.
    3. A collection is not an identity. One reel saved into three collections is
       one row in `video` and three in `video_collection`.
    4. A message id is not an identity either, and one video has more than one.
       The channel holds the old harvest's upload *and* the capture plane's, so
       "which message held this reel" is a set, and it lives in `video_message`.
       It is a **record**, not a derivation: the fact that message 10 held this
       reel arrives once, inside a shard whose `video` row is then deliberately
       discarded, and nothing can re-derive it afterwards. Learned only as an
       alias it died at the next rebuild, and every claim spelled `10` left the
       index with it.
    5. Ambiguity is recorded, never guessed. Two keys claiming one alias lands in
       `identity_conflict`, and both keys are left exactly as they were.
    6. Byte-identical content under two different permalinks is two videos, not
       one. The permalink is the identity; the shared bytes are a fact about
       them, recorded in `video_twin` so the interface can say "you saved this
       twice" and the engine can copy evidence instead of recomputing it.
    7. `audit()` runs at every index build and its violations are shown in the
       interface, not written to a log. An invariant nobody looks at is a
       comment.

WHAT THIS MODULE MAY NOT DO
    It may not delete, merge or rewrite a `video` row. Identity is additive: it
    learns aliases and records conflicts. The destructive half of a merge — one
    poster file instead of two, one proxy instead of two — is a separate,
    reversible sweep that reads this table and is asked for explicitly.
"""

import hashlib
import json
import os
import re
import sqlite3
import time

# ══════════════════════════════════════════════════════════════════════════
# THE LAW, in code
# ══════════════════════════════════════════════════════════════════════════
# Every shape Instagram uses for a single-post permalink. Identical to the
# capture ledger's own regex on purpose: the ledger decides what gets fetched
# and this decides what it is called afterwards, and the two disagreeing is
# exactly the class of bug this file exists to end. The trailing group is not
# anchored to `/` so a bare shortcode pasted into a note still matches.
PERMALINK = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/"
    r"(?:[A-Za-z0-9_.]+/)?"
    r"(reel|reels|p|tv)/"
    r"([A-Za-z0-9_-]{5,})",
    re.IGNORECASE,
)

UPLOAD_PREFIX = "up_"

_ALL_DIGITS = re.compile(r"^\d+$")
_UPLOAD_KEY = re.compile(r"^up_(\d+)$")
# Producers that spell a numeric Telegram message id with a prefix. An explicit
# short list, never a generic "letters then digits" rule — that would also match
# the shortcode `Cx1234` and throw away the letters that make it unique.
_PREFIXED_NUMBER = re.compile(r"^(?:tg|msg|id|up|video|frames)[_:-]?(\d+)$", re.I)
# The spellings a *witness* wears, read by `msg_id_in`. Wider than the list
# above on purpose, and the asymmetry is the point: refusing a key is
# destructive, so `looks_canonical` uses the short conservative list — a video
# whose shortcode happened to be `reel12` must not be thrown out. Reading one is
# not, because every caller checks the answer against the `video` table or the
# capture ledger before acting on it, so a false `reel12 → message 12` resolves
# to nothing and falls through. `38`, `tg38` and `frames_38` are all message 38
# under different producers' habits.
_MSG_SPELLING = re.compile(
    r"^(?:tg|msg|id|up|video|vid|frames|reel)?[_:\-]?(\d{1,12})$", re.I)
_KEY_NAMESPACE = re.compile(r"^vios[:=]", re.I)
# A shortcode is base64-ish and at least five characters. Case matters: `Abc`
# and `aBc` are different posts.
_SHORTCODE = re.compile(r"^[A-Za-z0-9_-]{5,32}$")

# Alias kinds, for the evidence column. Only descriptive — nothing branches on
# these — but a conflict is impossible to diagnose without knowing which rule
# produced the claim.
KIND_SELF = "self"
KIND_MSG = "msg_id"
KIND_RECORD = "record_msg_id"
KIND_PREFIXED = "prefixed"
KIND_URL = "url"
KIND_UPLOAD = "upload"
KIND_CONTENT = "content"
KIND_FOLDER = "folder"

_DDL = (
    # Any spelling ever seen → the one true key. `alias` is the primary key
    # because an alias means exactly one video; a video has many aliases.
    "CREATE TABLE IF NOT EXISTS video_alias ("
    "  alias TEXT PRIMARY KEY,"
    "  video_key TEXT NOT NULL,"
    "  kind TEXT,"
    "  evidence TEXT,"
    "  first_seen REAL)",
    "CREATE INDEX IF NOT EXISTS video_alias_key ON video_alias(video_key)",

    # Every channel message this video has ever occupied, one row each.
    # `video.msg_id` is one column and the channel is not: reel `DZDNyKgv70R` was
    # uploaded twice, so the old harvest recorded message 10 and the capture
    # ledger recorded 40, and a single column can only hold one of them. The same
    # squeeze `video_collection` exists to undo, one field over.
    #
    # This is a *record*, which is why `rebuild` does not clear it — the same
    # reason `video_twin` and `identity_conflict` survive. It has to be, because
    # of the one case that has no other witness: a shard an older build wrote
    # names its video by message id and carries the permalink beside it, so
    # `ingest._canonical_video_rows` rehomes the row onto the shortcode and then
    # that row is deliberately *not* written. Learned as an alias, "10 is this
    # reel" lasted until the next index build and was then dropped, taking every
    # claim keyed `10` out of the index with it. Recorded here, it is a seed for
    # every rebuild after it.
    "CREATE TABLE IF NOT EXISTS video_message ("
    "  video_key TEXT NOT NULL,"
    "  msg_id INTEGER NOT NULL,"
    "  source TEXT,"
    "  first_seen REAL,"
    "  PRIMARY KEY (video_key, msg_id))",
    "CREATE INDEX IF NOT EXISTS video_message_msg ON video_message(msg_id)",

    # One reel, many collections. Separate table rather than a column so a
    # second import adds memberships without rewriting the video, and so
    # "in three collections" can never be mistaken for "three videos".
    "CREATE TABLE IF NOT EXISTS video_collection ("
    "  video_key TEXT NOT NULL,"
    "  collection TEXT NOT NULL,"
    "  source TEXT,"
    "  added_at REAL,"
    "  PRIMARY KEY (video_key, collection))",
    "CREATE INDEX IF NOT EXISTS video_collection_name "
    "  ON video_collection(collection)",

    # Where refusals go. A row here means two canonical keys claimed one alias
    # and this module declined to pick; both keys are untouched and the
    # interface has something to show instead of a silently wrong merge.
    "CREATE TABLE IF NOT EXISTS identity_conflict ("
    "  alias TEXT NOT NULL,"
    "  video_key TEXT NOT NULL,"
    "  other_key TEXT NOT NULL,"
    "  evidence TEXT,"
    "  at REAL,"
    "  PRIMARY KEY (alias, video_key, other_key))",

    # Byte-identical content under two permalinks. Not a merge — see rule 6.
    "CREATE TABLE IF NOT EXISTS video_twin ("
    "  video_key TEXT NOT NULL,"
    "  twin_key TEXT NOT NULL,"
    "  sha256 TEXT,"
    "  bytes INTEGER,"
    "  found_at REAL,"
    "  PRIMARY KEY (video_key, twin_key))",

    # A hash cache, so the content pass is free after the first run. Keyed by
    # the three things that change when a file changes; a media directory is
    # append-only in practice, so this makes rebuild cost nothing on a machine
    # that has already hashed its videos once.
    "CREATE TABLE IF NOT EXISTS media_hash ("
    "  path TEXT PRIMARY KEY,"
    "  size INTEGER,"
    "  mtime REAL,"
    "  sha256 TEXT,"
    "  at REAL)",
)

# Tables Atlas owns and a shard may never land on. Kept here as well as in
# reflect so this module is the one thing to read when adding an identity table.
OWNED = ("video_alias", "video_message", "video_collection",
         "identity_conflict", "video_twin", "media_hash")


def ensure(conn: sqlite3.Connection) -> None:
    """Create the identity tables. Idempotent, cheap, safe to call per request."""
    for stmt in _DDL:
        conn.execute(stmt)


# ══════════════════════════════════════════════════════════════════════════
# READING A SPELLING
# ══════════════════════════════════════════════════════════════════════════
def key_from_url(url) -> str:
    """The shortcode in an Instagram permalink, or "" if it is not one."""
    m = PERMALINK.search(str(url or ""))
    return m.group(2) if m else ""


def canonical_url(key: str) -> str:
    """The permalink for a shortcode. Empty for an upload key or a loose file —
    deliberately: nothing downstream should be able to look at a hand-uploaded
    video, or one adopted from a folder, and conclude Instagram has a copy."""
    k = str(key or "")
    if not k or is_upload(k) or is_local(k) or _ALL_DIGITS.match(k):
        return ""
    return f"https://www.instagram.com/reel/{k}/"


def upload_key(msg_id) -> str:
    """The canonical key for a bare video sitting at `msg_id` in the channel."""
    return f"{UPLOAD_PREFIX}{int(msg_id)}"


def is_upload(key) -> bool:
    return bool(_UPLOAD_KEY.match(str(key or "")))


def upload_msg_id(key) -> int:
    m = _UPLOAD_KEY.match(str(key or ""))
    return int(m.group(1)) if m else 0


# A video that is neither on Instagram nor in the channel: a loose file on a
# disk the engine can see, adopted from a folder. It needs an identity for the
# same reason an upload does, and gets one by the same argument — derived from
# something permanent about the video itself, so that a rename cannot mint a
# second identity and a store rebuilt from the same folder lands on the same
# key. For an upload that permanent thing is the message id; for a loose file
# the bytes are all there is, so it is the head of its sha256.
#
# This is the namespace that replaces naming a video after its filename. A
# folder of files called `10.mp4`, `11.mp4` is how thirty-two numeric keys got
# into an archive of thirty videos: the stem was adopted as an identity, and a
# Telegram message id is not one. `file_` cannot collide with a shortcode for
# the same reason `up_` cannot — a shortcode is 11 characters of base64 and is
# never `file_` followed by exactly 16 hex digits.
LOCAL_PREFIX = "file_"
_LOCAL_KEY = re.compile(r"^file_[0-9a-f]{16}$")
_HEX16 = re.compile(r"^[0-9a-f]{16,}$")


def local_key(digest) -> str:
    """The key for a loose file: no permalink, no channel message, just bytes."""
    d = str(digest or "").strip().lower()
    if not _HEX16.match(d):
        raise ValueError("a local key is derived from a hex digest")
    return f"{LOCAL_PREFIX}{d[:16]}"


def is_local(key) -> bool:
    return bool(_LOCAL_KEY.match(str(key or "")))


def looks_canonical(value) -> bool:
    """True if this string is *shaped* like a canonical key.

    Three forms and no others: an Instagram shortcode, `up_<msg_id>` for a video
    handed straight to the channel, and `file_<digest16>` for a loose file that
    was never on Instagram at all.

    Shape only — it says nothing about whether the video exists. A bare number
    is never canonical: a Telegram message id is scoped to one channel and gets
    reused, which is how "could not download message 38" happened after a
    channel change.
    """
    s = str(value or "").strip()
    if not s:
        return False
    if is_upload(s) or is_local(s):
        return True
    if _ALL_DIGITS.match(s):
        return False
    if _PREFIXED_NUMBER.match(s):
        return False
    return bool(_SHORTCODE.match(s))


def msg_id_in(value) -> int:
    """The Telegram message id a non-identity spelling is naming, or 0.

    The counterpart to `looks_canonical`: that function says "this is not an
    identity", and this one says "…but here is what it is evidence *of*". A
    filename stem of `38`, a legacy `video_key` of `tg38`, a working directory
    called `frames_38` — all of them are message 38, and message 38 in the
    capture ledger or in the `video` table is one particular reel.

    Deliberately not the inverse of `looks_canonical`, in both directions:
    `up_4471` answers 4471 here and is *also* canonical, because a video handed
    straight to the channel has nothing else to be named after; and `reel12`
    answers 12 here while staying canonical, because refusing a key is
    destructive and reading one is not. Callers check identity first and only ask
    this about a spelling they are already looking to place — see
    `Store.rehome_key`, which then requires the `video` table or the capture
    ledger to agree before anything moves.
    """
    m = _MSG_SPELLING.match(str(value or "").strip())
    return int(m.group(1)) if m else 0


def spellings(video_key: str, msg_id=None, record_msg_id=None, url=None) -> list:
    """Every alias a video is known by, as (alias, kind) pairs.

    One function so the writers and the repair agree by construction. A reader
    that invents a twelfth spelling later adds it here, once, and every caller
    starts resolving it.
    """
    out = [(str(video_key), KIND_SELF)]
    if url:
        sc = key_from_url(url)
        if sc and sc != video_key:
            out.append((sc, KIND_URL))
    for value, kind in ((msg_id, KIND_MSG), (record_msg_id, KIND_RECORD)):
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        out.append((str(n), kind))
        for form in (f"tg{n}", f"msg{n}", f"msg_{n}", f"tg_{n}", f"id{n}",
                     f"up_{n}", f"video_{n}", f"frames_{n}"):
            out.append((form, KIND_PREFIXED))
    seen, uniq = set(), []
    for alias, kind in out:
        if alias and alias not in seen:
            seen.add(alias)
            uniq.append((alias, kind))
    return uniq


# ══════════════════════════════════════════════════════════════════════════
# LEARNING AN ALIAS
# ══════════════════════════════════════════════════════════════════════════
def learn(conn: sqlite3.Connection, alias, video_key, kind: str = "",
          evidence: str = "") -> str:
    """Record that `alias` means `video_key`. Returns what happened.

    "ok"        — written.
    "known"     — already recorded, pointing the same way.
    "conflict"  — already recorded pointing somewhere else. Nothing is changed
                  and a row lands in `identity_conflict`.
    "rejected"  — the target is not shaped like a canonical key.

    First writer wins, and the loser is kept rather than dropped. That ordering
    is why `rebuild` seeds from `video` before it touches anything weaker: the
    strongest evidence goes in first and everything after it is either
    corroboration or a conflict worth a human's attention.
    """
    a = str(alias or "").strip()
    k = str(video_key or "").strip()
    if not a or not k or not looks_canonical(k):
        return "rejected"
    row = conn.execute("SELECT video_key FROM video_alias WHERE alias=?",
                       (a,)).fetchone()
    if row is not None:
        if str(row[0]) == k:
            return "known"
        conn.execute(
            "INSERT OR REPLACE INTO identity_conflict"
            "(alias, video_key, other_key, evidence, at) VALUES (?,?,?,?,?)",
            (a, str(row[0]), k, evidence or kind, time.time()))
        return "conflict"
    conn.execute(
        "INSERT INTO video_alias(alias, video_key, kind, evidence, first_seen) "
        "VALUES (?,?,?,?,?)", (a, k, kind or "", evidence or "", time.time()))
    return "ok"


def forget(conn: sqlite3.Connection, alias) -> int:
    """Drop one alias. The escape hatch for a genuinely wrong learned mapping;
    the repair is to delete it and re-run `rebuild`, never to edit it in place."""
    cur = conn.execute("DELETE FROM video_alias WHERE alias=? AND kind<>?",
                       (str(alias), KIND_SELF))
    return cur.rowcount or 0


def note_message(conn: sqlite3.Connection, video_key, msg_id,
                 source: str = "") -> int:
    """Record that this video occupies a channel message. Returns 1 if new.

    Every witness that knows a message id calls this, so the set is complete
    rather than whatever the `video` row's one column happened to keep. Idempotent
    on `(video_key, msg_id)`, and deliberately not on `msg_id` alone: two videos
    claiming one message is a real conflict, but it is not this function's to
    judge — `learn` sees the same pair through `_seed_from_messages` and records
    it in `identity_conflict` there, once, in the one place that already does it.

    Refuses a non-canonical `video_key` for the same reason `learn` does. A table
    of message ids keyed by a message id would be the defect writing down its own
    excuse.
    """
    k = str(video_key or "").strip()
    if not looks_canonical(k):
        return 0
    try:
        n = int(msg_id)
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0
    cur = conn.execute(
        "INSERT OR IGNORE INTO video_message(video_key, msg_id, source, "
        "first_seen) VALUES (?,?,?,?)", (k, n, source or "", time.time()))
    return cur.rowcount or 0


def messages_for(conn: sqlite3.Connection, video_key: str) -> list:
    """Every message this video sits at, newest record first. For the drill-down.

    Two entries is not a duplicate and the interface should say so: it means the
    same reel was uploaded to the channel twice, and both copies play.
    """
    try:
        return [{"msg_id": int(r[0]), "source": str(r[1] or "")}
                for r in conn.execute(
                    "SELECT msg_id, source FROM video_message "
                    "WHERE video_key=? ORDER BY msg_id", (video_key,))]
    except sqlite3.Error:
        return []


# ══════════════════════════════════════════════════════════════════════════
# RESOLVING
# ══════════════════════════════════════════════════════════════════════════
def _string_rules(value) -> str:
    """The fallback, for a spelling the table has never seen.

    Deliberately timid. It strips a namespace marker and unwraps a prefixed
    number, and otherwise hands the string back untouched. It must never invent
    a key: a number this database has no video for stays that number, so it
    shows up in `audit()` as an unresolved key rather than quietly becoming a
    video that does not exist.
    """
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    s = str(value).strip()
    if not s:
        return ""
    s = _KEY_NAMESPACE.sub("", s, count=1).strip()
    if _ALL_DIGITS.match(s):
        return s
    if is_upload(s):
        return s
    m = _PREFIXED_NUMBER.match(s)
    return m.group(1) if m else s


def load_map(conn: sqlite3.Connection) -> dict:
    """The whole alias table as a dict. 30 videos is a few hundred entries."""
    try:
        return {str(a): str(k) for a, k in conn.execute(
            "SELECT alias, video_key FROM video_alias")}
    except sqlite3.Error:
        return {}


def resolve(conn: sqlite3.Connection, value) -> str:
    """One spelling → the canonical key. A single lookup, then the string rules."""
    s = _string_rules(value)
    if not s:
        return ""
    try:
        row = conn.execute("SELECT video_key FROM video_alias WHERE alias=?",
                           (s,)).fetchone()
    except sqlite3.Error:
        return s
    if row is not None:
        return str(row[0])
    raw = str(value or "").strip()
    if raw and raw != s:
        try:
            row = conn.execute("SELECT video_key FROM video_alias WHERE alias=?",
                               (raw,)).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None:
            return str(row[0])
    return s


class Resolver:
    """A resolver backed by one snapshot of the table, for hot loops.

    `_collect` walks 27,000 claims and every text row in the archive. One SQL
    round trip per row to answer a question whose whole answer set is a few
    hundred pairs is the kind of cost that gets a correctness fix reverted for
    being slow, so the map is read once.

    Snapshot semantics are the point, not a compromise: an index build must see
    one consistent identity map from first row to last. A build that learned a
    new alias halfway would file the first half of a video's evidence under one
    key and the second half under another — the exact defect, reintroduced by
    the fix for it.
    """

    __slots__ = ("map", "misses")

    def __init__(self, conn: sqlite3.Connection | None = None, mapping=None):
        self.map = dict(mapping) if mapping is not None else (
            load_map(conn) if conn is not None else {})
        self.misses = {}

    def __call__(self, value) -> str:
        raw = str(value or "").strip() if not isinstance(value, (int, float)) \
            else str(int(value))
        hit = self.map.get(raw)
        if hit:
            return hit
        s = _string_rules(value)
        if not s:
            return ""
        hit = self.map.get(s)
        if hit:
            return hit
        self.misses[s] = self.misses.get(s, 0) + 1
        return s

    def unresolved(self) -> dict:
        """Spellings this resolver was asked about and had no row for.

        Read after a build. Every entry is either a video the archive has no
        `video` row for — real, and worth showing — or a namespace nobody has
        taught this module about yet. Both are findings; neither is silent.
        """
        return dict(sorted(self.misses.items(), key=lambda kv: -kv[1]))


def install(conn: sqlite3.Connection) -> dict:
    """Wire the alias map into `reflect.normalize_key`.

    This is the seam that makes the fix reach code that predates it. Eleven call
    sites already normalise a key through reflect; none of them needs to change,
    and none of them can accidentally opt out.
    """
    from . import reflect                                  # noqa: PLC0415
    mapping = load_map(conn)
    reflect.set_aliases(mapping)
    return mapping


# ══════════════════════════════════════════════════════════════════════════
# BUILDING THE MAP
# ══════════════════════════════════════════════════════════════════════════
# Where a video key can appear, outside `video` itself. Used by the orphan pass
# and by `audit`. Table and column only — no join, because half of these tables
# arrived from a shard and their shape is whatever that shard held.
KEY_COLUMNS = (
    ("videos", "msg_id"),
    ("posts", "video_id"),
    ("frame_notes", "msg_id"),
    ("transcripts", "msg_id"),
    ("omni_frames", "video_uuid"),
    ("omni_chunks", "video_uuid"),
    ("scanned_ids", "video_id"),
    ("claim", "video_key"),
    ("shot", "video_key"),
    ("artifact", "video_key"),
    ("frame_metric", "video_key"),
    ("frame_vector", "video_key"),
    ("vector", "video_key"),
    ("vec_payload", "video_key"),
    ("parts", "video_key"),
    ("coverage", "video_key"),
    ("moments", "video_key"),
    ("video_index", "video_key"),
)

# Columns whose unresolved spellings are worth counting but must not fail the
# audit, because not every value in them is supposed to be a video.
#
# `scanned_ids` is the legacy harvester's "I have already looked at message N"
# ledger. It holds 133 message ids, and the channel contains far more than
# videos: text posts, shard bundles, the epoch marker. Roughly 30 of those
# messages are reels; the rest resolving to nothing is the correct answer, not a
# gap. Demanding that every scanned message name a video would either fail the
# audit forever or — much worse — tempt someone to invent a video for each one.
_ADVISORY = {("scanned_ids", "video_id")}


def _has(conn: sqlite3.Connection, table: str) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') "
            "AND name=?", (table,)).fetchone() is not None
    except sqlite3.Error:
        return False


def _cols(conn: sqlite3.Connection, table: str) -> set:
    try:
        return {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
    except sqlite3.Error:
        return set()


def _capture_meta(raw) -> dict:
    """`video.meta` → the capture block, or {}. Never raises on bad JSON."""
    if not raw:
        return {}
    try:
        obj = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return {}
    cap = obj.get("capture")
    return cap if isinstance(cap, dict) else {}


def _seed_from_video(conn: sqlite3.Connection, stats: dict) -> set:
    """Pass 1 — `video` is the identity table, so it goes in first.

    Two sweeps, and the order matters. Every `video_key` claims itself before any
    derived spelling is offered, because `up_4471` is a *canonical key* when a
    video row exists for it and merely an alias of a shortcode when one does not.
    Interleaving the two sweeps would let whichever row happened to be read first
    decide, which is a coin flip dressed as a rule.
    """
    if not _has(conn, "video"):
        return set()
    cols = _cols(conn, "video")
    have = [c for c in ("video_key", "url", "msg_id", "meta") if c in cols]
    if "video_key" not in have:
        return set()
    rows = conn.execute(f'SELECT {", ".join(have)} FROM video').fetchall()
    keys = set()

    for r in rows:
        d = dict(zip(have, r))
        vk = str(d.get("video_key") or "").strip()
        if not vk:
            continue
        if not looks_canonical(vk):
            # A `video` row keyed by something that is not an identity. Recorded
            # as a violation rather than adopted: adopting it is how the numeric
            # keys got in.
            stats["video_rows_not_canonical"].append(vk)
            continue
        keys.add(vk)
        learn(conn, vk, vk, KIND_SELF, "video.video_key")

    for r in rows:
        d = dict(zip(have, r))
        vk = str(d.get("video_key") or "").strip()
        if vk not in keys:
            continue
        cap = _capture_meta(d.get("meta"))
        # Recorded before the aliases are derived from it, so the set of messages
        # this video occupies is never smaller than what the column can hold.
        for value, why in ((d.get("msg_id"), "video.msg_id"),
                           (cap.get("msg_id"), "meta.capture.msg_id"),
                           (cap.get("record_msg_id"),
                            "meta.capture.record_msg_id")):
            note_message(conn, vk, value, why)
        for alias, kind in spellings(
                vk, msg_id=d.get("msg_id") or cap.get("msg_id"),
                record_msg_id=cap.get("record_msg_id"), url=d.get("url")):
            if alias in keys and alias != vk:
                continue          # another video's identity; never an alias
            got = learn(conn, alias, vk, kind, "video")
            stats[got] = stats.get(got, 0) + 1
    stats["canonical_keys"] = len(keys)
    return keys


def _seed_from_messages(conn: sqlite3.Connection, keys: set,
                        stats: dict) -> int:
    """Pass 1b — the messages a video was recorded at, including the lost ones.

    Runs immediately after `video` and before the ledger because it is the same
    strength of evidence: it *is* `video.msg_id`, plus the message ids that
    column could not hold. Pass 1 wrote most of these rows itself a moment ago,
    so most of this pass is corroboration; the rows that matter are the ones no
    other witness still has, which is why the table is not cleared by `rebuild`.

    A message whose video no longer has a `video` row is skipped rather than
    stubbed. `_adopt_orphans` is the pass that decides whether an unidentifiable
    thing deserves a stub, and it has the evidence to weigh; this one does not.
    """
    if not keys:
        return 0
    learned = 0
    try:
        rows = conn.execute(
            "SELECT video_key, msg_id, source FROM video_message "
            "ORDER BY video_key, msg_id").fetchall()
    except sqlite3.Error:
        return 0
    for vk, msg_id, source in rows:
        k = str(vk or "").strip()
        if k not in keys:
            continue
        for alias, kind in spellings(k, msg_id=msg_id):
            if alias in keys and alias != k:
                continue
            got = learn(conn, alias, k, kind,
                        f"video_message ({source})" if source
                        else "video_message")
            stats[got] = stats.get(got, 0) + 1
            learned += (got == "ok")
    stats["from_messages"] = learned
    return learned


def _seed_from_ledger(conn: sqlite3.Connection, keys: set, ledger_path: str,
                      stats: dict) -> dict:
    """Pass 2 — the capture ledger, the one witness that predates the archive.

    Its `item` table is keyed by shortcode and holds the message id for every
    reel it ever uploaded, so it answers "which shortcode is message 38?" even
    for a video the processing plane never touched. Returns {msg_id: shortcode}
    for the ledger's whole view, which the orphan pass then uses.

    Read-only, and optional. On a machine with no ledger — a fresh restore, a
    Kaggle session that only imports — every other pass still works.
    """
    seen = {}
    if not ledger_path or not os.path.exists(ledger_path):
        return seen
    try:
        led = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True,
                              timeout=5)
    except sqlite3.Error:
        return seen
    try:
        rows = led.execute(
            "SELECT key, url, msg_id, record_msg_id FROM item "
            "WHERE msg_id IS NOT NULL").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        try:
            led.close()
        except sqlite3.Error:
            pass

    for key, url, msg_id, record_id in rows:
        k = str(key or "").strip()
        if not k or not looks_canonical(k):
            continue
        try:
            seen[str(int(msg_id))] = k
        except (TypeError, ValueError):
            pass
        if k not in keys:
            continue              # a reel the archive holds no evidence for
        note_message(conn, k, msg_id, "capture_ledger")
        note_message(conn, k, record_id, "capture_ledger record")
        for alias, kind in spellings(k, msg_id=msg_id, record_msg_id=record_id,
                                     url=url):
            if alias in keys and alias != k:
                continue
            got = learn(conn, alias, k, kind, "capture_ledger")
            stats[got] = stats.get(got, 0) + 1
    stats["ledger_msg_ids"] = len(seen)
    return seen


def _digest(conn: sqlite3.Connection, path: str) -> str:
    """SHA-256 of a file, cached on (path, size, mtime)."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    row = conn.execute(
        "SELECT sha256 FROM media_hash WHERE path=? AND size=? AND mtime=?",
        (path, st.st_size, round(st.st_mtime, 3))).fetchone()
    if row and row[0]:
        return str(row[0])
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    dig = h.hexdigest()
    conn.execute(
        "INSERT OR REPLACE INTO media_hash(path, size, mtime, sha256, at) "
        "VALUES (?,?,?,?,?)",
        (path, st.st_size, round(st.st_mtime, 3), dig, time.time()))
    return dig


def note_twin(conn: sqlite3.Connection, a: str, b: str, sha: str = "",
              nbytes: int = 0) -> None:
    """Record that two canonical keys hold identical bytes. Symmetric.

    Both directions are written because both are true and because a reader
    asking "does this video have a twin?" should not have to know which of the
    two was discovered first.
    """
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO video_twin(video_key, twin_key, sha256, bytes, "
        "found_at) VALUES (?,?,?,?,?)",
        [(a, b, sha, int(nbytes or 0), now), (b, a, sha, int(nbytes or 0), now)])


def _seed_from_content(conn: sqlite3.Connection, keys: set, media_dir: str,
                       stats: dict) -> dict:
    """Pass 3 — the bytes, which cannot be spelled two ways.

    Everything above this reads a string somebody wrote down. This reads the
    file. It is the only pass that can resolve a spelling nothing recorded a map
    for, and it is how `8` was shown to be a second upload of message `38`
    rather than a thirty-first video.

    Two outcomes, and keeping them apart is rule 6:

      one canonical key in the group → every other stem is an alias of it. The
          same reel was written to disk twice under two names.
      two or more canonical keys    → two permalinks, identical bytes. Two
          videos. Recorded in `video_twin` and left alone, because the permalink
          is the identity and a repost is a different post.

    Returns {stem: digest} for the whole directory, which the caller reports.
    """
    root = os.path.join(media_dir or "", "video")
    out = {}
    if not media_dir or not os.path.isdir(root):
        return out
    groups = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        dig = _digest(conn, path)
        if not dig:
            continue
        stem = os.path.splitext(name)[0]
        out[stem] = dig
        groups.setdefault(dig, []).append((stem, os.path.getsize(path)))

    for dig, members in groups.items():
        canon = {}
        for stem, _ in members:
            k = resolve(conn, stem)
            if k in keys:
                canon.setdefault(k, stem)
        if len(canon) == 1:
            k = next(iter(canon))
            for stem, _ in members:
                if resolve(conn, stem) == k or stem in keys:
                    continue
                got = learn(conn, stem, k, KIND_CONTENT, f"sha256 {dig[:12]}")
                stats[got] = stats.get(got, 0) + 1
                stats["by_content"] = stats.get("by_content", 0) + (got == "ok")
            learn(conn, f"sha256:{dig}", k, KIND_CONTENT, "media/video")
        elif len(canon) > 1:
            ks = sorted(canon)
            size = members[0][1] if members else 0
            for i, a in enumerate(ks):
                for b in ks[i + 1:]:
                    note_twin(conn, a, b, dig, size)
                    stats["twins"] = stats.get("twins", 0) + 1
    return out


def _stub_video(conn: sqlite3.Connection, video_key: str, msg_id=None,
                url: str = "", why: str = "") -> bool:
    """Register a video the archive holds evidence about but has no row for.

    The narrow exception to "identity never writes to `video`", and it earns its
    place: without it, a reel the older harvest transcribed and the new plane
    never touched has evidence in four tables and no addressable identity, so
    every reader either drops it or invents a numeric key for it — which is the
    defect.

    The row claims nothing it has not measured. No duration, no dimensions, no
    hash; `meta.identity.stub` is true so every count can separate "videos this
    archive has processed" from "videos this archive knows the name of".
    """
    if not looks_canonical(video_key):
        return False
    cols = _cols(conn, "video")
    if not cols:
        return False
    payload = {"video_key": video_key,
               "url": url or canonical_url(video_key),
               "msg_id": int(msg_id) if str(msg_id or "").isdigit() else None,
               "added_at": time.time(),
               "meta": json.dumps({"identity": {
                   "stub": True, "why": why,
                   "at": time.time()}}, ensure_ascii=False)}
    use = {k: v for k, v in payload.items() if k in cols}
    names = ", ".join(f'"{k}"' for k in use)
    marks = ", ".join("?" * len(use))
    try:
        cur = conn.execute(
            f"INSERT OR IGNORE INTO video ({names}) VALUES ({marks})",
            list(use.values()))
    except sqlite3.Error:
        return False
    return bool(cur.rowcount)


def is_stub(conn: sqlite3.Connection, video_key: str) -> bool:
    row = conn.execute("SELECT meta FROM video WHERE video_key=?",
                       (video_key,)).fetchone()
    if not row or not row[0]:
        return False
    try:
        return bool((json.loads(row[0]).get("identity") or {}).get("stub"))
    except (ValueError, AttributeError):
        return False


def unresolved_spellings(conn: sqlite3.Connection, keys: set = None) -> dict:
    """Every spelling in the archive that does not resolve to a known video.

    Returns {spelling: [tables it appears in]}. This is the measurement the whole
    module is judged by: an archive whose identity is sound returns {} once the
    advisory columns in `_ADVISORY` are set aside.
    """
    known = keys if keys is not None else canonical_keys(conn)
    res = Resolver(conn)
    out = {}
    for table, col in KEY_COLUMNS:
        if table in OWNED or not _has(conn, table) or col not in _cols(conn, table):
            continue
        try:
            rows = conn.execute(
                f'SELECT DISTINCT "{col}" FROM "{table}" '
                f'WHERE "{col}" IS NOT NULL').fetchall()
        except sqlite3.Error:
            continue
        for (raw,) in rows:
            k = res(raw)
            if k and k not in known:
                out.setdefault(k, []).append(f"{table}.{col}")
    return out


def _advisory_only(where) -> bool:
    """True when a spelling appears nowhere except advisory columns.

    A message id that only `scanned_ids` has ever seen is a message, not a
    missing video. The same id appearing in `claim` as well is a real gap, so
    the test is over every place it was found, not the first.
    """
    pairs = {("scanned_ids", "video_id") if w == "scanned_ids.video_id"
             else tuple(w.split(".", 1)) for w in where}
    return bool(pairs) and pairs <= _ADVISORY


def _adopt_orphans(conn: sqlite3.Connection, keys: set, ledger_map: dict,
                   stats: dict) -> set:
    """Pass 4 — evidence whose video has no row.

    A spelling that survives passes 1–3 unresolved is one of two things: a reel
    only the older harvest ever saw, or a namespace nobody has taught this module
    about. The first is fixable and the second must not be guessed at, so the
    only bridge accepted here is a *recorded* one — the ledger's message-id map,
    or a permalink sitting in the archive next to the evidence.
    """
    orphans = unresolved_spellings(conn, keys)
    adopted = set()
    for spelling, where in sorted(orphans.items()):
        shortcode = ledger_map.get(spelling) or _permalink_near(conn, spelling)
        if not shortcode:
            # Recorded nowhere. If the only witness is an advisory column it is
            # a channel message that was never a video, which is not a defect.
            bucket = "not_video" if _advisory_only(where) else "unresolved"
            stats.setdefault(bucket, []).append(
                f"{spelling} ({', '.join(where[:3])})")
            continue
        if shortcode not in keys:
            if not _stub_video(conn, shortcode, msg_id=spelling,
                               why=f"evidence in {', '.join(where[:3])}"):
                stats["unresolved"].append(f"{spelling} (stub refused)")
                continue
            keys.add(shortcode)
            adopted.add(shortcode)
            stats["stubs"] = stats.get("stubs", 0) + 1
        got = learn(conn, spelling, shortcode, KIND_MSG, "orphan adoption")
        stats[got] = stats.get(got, 0) + 1
    return adopted


# Tables whose rows may carry a permalink next to a numeric key. Every text
# column of the matching row is searched, because which column holds the link
# has changed twice: the first harvest put it in `caption`, the second in a
# `title`, and a hand-written import in the path.
_PERMALINK_TABLES = (("posts", "video_id"), ("videos", "msg_id"),
                     ("scan_seen", "msg_id"))


def _permalink_near(conn: sqlite3.Connection, spelling: str) -> str:
    """A shortcode from a permalink stored beside this spelling, or "".

    Recorded evidence, not inference: the row itself says which post it is.
    """
    for table, col in _PERMALINK_TABLES:
        if not _has(conn, table) or col not in _cols(conn, table):
            continue
        try:
            cur = conn.execute(f'SELECT * FROM "{table}" WHERE "{col}"=? LIMIT 4',
                               (spelling,))
            rows = cur.fetchall()
        except sqlite3.Error:
            continue
        for row in rows:
            for val in row:
                if isinstance(val, str) and "instagram.com" in val:
                    sc = key_from_url(val)
                    if sc:
                        return sc
    return ""


def canonical_keys(conn: sqlite3.Connection) -> set:
    """Every key with a `video` row. The definition of "a video exists"."""
    try:
        return {str(r[0]) for r in conn.execute("SELECT video_key FROM video")}
    except sqlite3.Error:
        return set()


def rebuild(conn: sqlite3.Connection, media_dir: str = "",
            ledger_path: str = "", commit: bool = True) -> dict:
    """Build the alias map from scratch. Idempotent, and safe to run often.

    Learned aliases are dropped first and rebuilt, because the passes are ordered
    by strength of evidence and a stale weak alias would otherwise outrank a new
    strong one forever. `identity_conflict`, `video_twin` and `video_message` are
    *not* cleared — the first two are findings and the third is a record of what
    arrived, and none of the three is derivable from the tables this rebuilds
    from. `video_message` is what makes dropping the alias table safe: a message
    id learned from a shard survives in it, so the pass order can stay strict
    without the strict order costing evidence.
    """
    ensure(conn)
    stats = {"ok": 0, "known": 0, "conflict": 0, "rejected": 0,
             "unresolved": [], "not_video": [],
             "video_rows_not_canonical": []}
    conn.execute("DELETE FROM video_alias")
    keys = _seed_from_video(conn, stats)
    _seed_from_messages(conn, keys, stats)
    ledger_map = _seed_from_ledger(conn, keys, ledger_path, stats)
    _seed_from_content(conn, keys, media_dir, stats)
    _adopt_orphans(conn, keys, ledger_map, stats)
    stats["aliases"] = conn.execute(
        "SELECT COUNT(*) FROM video_alias").fetchone()[0]
    stats["videos"] = len(canonical_keys(conn))
    stats["messages"] = conn.execute(
        "SELECT COUNT(*) FROM video_message").fetchone()[0]
    stats["conflicts_open"] = conn.execute(
        "SELECT COUNT(*) FROM identity_conflict").fetchone()[0]
    if commit:
        conn.commit()
    return stats


def absorb(conn: sqlite3.Connection, rows) -> dict:
    """Learn identity from `video` rows that just arrived. For the import path.

    `rebuild` recomputes everything from the whole table and is what the index
    build calls; this is the incremental half, so a shard that lands at 3am
    leaves the archive with correct identity without waiting for the next build.
    Same two-sweep order and the same rules — a row keyed by something that is
    not an identity is refused here too, because an importer that adopts a
    numeric key is precisely how the numeric keys got in.

    Takes dicts, not a cursor: the importer has the rows in hand as JSON and
    re-reading them from SQLite would only be slower and less complete.
    """
    ensure(conn)
    out = {"videos": 0, "aliases": 0, "collections": 0, "messages": 0,
           "refused": 0}
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    fresh = set()
    for r in rows:
        vk = str(r.get("video_key") or "").strip()
        if not vk:
            continue
        if not looks_canonical(vk):
            out["refused"] += 1
            continue
        fresh.add(vk)
        learn(conn, vk, vk, KIND_SELF, "shard video.video_key")
    known = canonical_keys(conn) | fresh

    for r in rows:
        vk = str(r.get("video_key") or "").strip()
        if vk not in fresh:
            continue
        out["videos"] += 1
        cap = _capture_meta(r.get("meta"))
        for value, why in ((r.get("msg_id"), "shard video.msg_id"),
                           (cap.get("msg_id"), "shard meta.capture.msg_id"),
                           (cap.get("record_msg_id"),
                            "shard meta.capture.record_msg_id")):
            out["messages"] += note_message(conn, vk, value, why)
        for alias, kind in spellings(
                vk, msg_id=r.get("msg_id") or cap.get("msg_id"),
                record_msg_id=cap.get("record_msg_id"), url=r.get("url")):
            if alias in known and alias != vk:
                continue
            if learn(conn, alias, vk, kind, "shard video") == "ok":
                out["aliases"] += 1
        names = cap.get("collections")
        if names:
            out["collections"] += set_collections(conn, vk, names,
                                                  source="shard")
    conn.commit()
    if out["aliases"] or out["collections"]:
        install(conn)             # the map changed; the resolver must see it
    return out


# ══════════════════════════════════════════════════════════════════════════
# COLLECTIONS — one reel, many shelves
# ══════════════════════════════════════════════════════════════════════════
# A collection is a *label you chose*, and the reason it gets its own table
# rather than a column is the sentence that started this work: one video may be
# in two or more saved collections. As a column it would be a comma-joined
# string that nothing can filter on; as a second `video` row it would be the
# duplication defect with a friendlier cause. As a join table it is neither, and
# "show me everything in Recipes" is one index seek.
#
# The membership already travels: the capture plane writes the collections it
# found into `video.meta.capture.collections`, and every shard carries the
# `video` row. So no new record type is needed in the shard format, no schema
# version bump, and the 76 shards already sitting in the channel yield their
# memberships the next time they are read.
def set_collections(conn: sqlite3.Connection, video_key: str, names,
                    source: str = "") -> int:
    """Add memberships for one video. Additive — never removes a label."""
    now = time.time()
    rows = []
    for raw in names or ():
        name = str(raw or "").strip()
        if name:
            rows.append((video_key, name, source, now))
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO video_collection(video_key, collection, source, "
        "added_at) VALUES (?,?,?,?)", rows)
    return conn.total_changes - before


def rebuild_collections(conn: sqlite3.Connection, ledger_path: str = "",
                        commit: bool = True) -> dict:
    """Promote memberships out of JSON and into a table anything can query.

    Two witnesses, both read-only: `video.meta.capture.collections`, which every
    shard carries, and the capture ledger's `membership` table when the machine
    has one. The ledger is the richer of the two — it knows about reels that have
    not been processed yet — but it is only present where capture runs, which is
    why the JSON path is not merely a fallback.
    """
    ensure(conn)
    out = {"from_meta": 0, "from_ledger": 0, "videos": 0, "collections": 0,
           "skipped_unknown": 0}
    keys = canonical_keys(conn)
    res = Resolver(conn)

    if _has(conn, "video") and "meta" in _cols(conn, "video"):
        for vk, meta in conn.execute("SELECT video_key, meta FROM video"):
            names = _capture_meta(meta).get("collections")
            if names:
                out["from_meta"] += set_collections(
                    conn, str(vk), names, "capture.meta")

    if ledger_path and os.path.exists(ledger_path):
        try:
            led = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True,
                                  timeout=5)
            rows = led.execute(
                "SELECT key, collection FROM membership").fetchall()
            led.close()
        except sqlite3.Error:
            rows = []
        for key, name in rows:
            vk = res(key)
            if vk not in keys:
                out["skipped_unknown"] += 1
                continue
            out["from_ledger"] += set_collections(conn, vk, [name], "ledger")

    row = conn.execute(
        "SELECT COUNT(DISTINCT video_key), COUNT(DISTINCT collection) "
        "FROM video_collection").fetchone()
    out["videos"], out["collections"] = int(row[0] or 0), int(row[1] or 0)
    if commit:
        conn.commit()
    return out


def collections_for(conn: sqlite3.Connection, video_key: str) -> list:
    try:
        return [str(r[0]) for r in conn.execute(
            "SELECT collection FROM video_collection WHERE video_key=? "
            "ORDER BY collection", (video_key,))]
    except sqlite3.Error:
        return []


def collection_counts(conn: sqlite3.Connection) -> list:
    """[{name, videos}] for the facet list, largest first."""
    try:
        return [{"name": str(r[0]), "videos": int(r[1])} for r in conn.execute(
            "SELECT collection, COUNT(*) n FROM video_collection "
            "GROUP BY collection ORDER BY n DESC, collection ASC")]
    except sqlite3.Error:
        return []


def aliases_for(conn: sqlite3.Connection, video_key: str) -> list:
    """Other names this video answers to, without the self-alias.

    Shown in the interface on purpose. "Also message 38" is the sentence that
    makes a merged card believable — the alternative is a card that silently
    holds twice as much evidence as the last time you looked.
    """
    try:
        return [{"alias": str(r[0]), "kind": str(r[1] or ""),
                 "evidence": str(r[2] or "")} for r in conn.execute(
            "SELECT alias, kind, evidence FROM video_alias "
            "WHERE video_key=? AND kind<>? ORDER BY kind, alias",
            (video_key, KIND_SELF))]
    except sqlite3.Error:
        return []


def twins_for(conn: sqlite3.Connection, video_key: str) -> list:
    try:
        return [{"video_key": str(r[0]), "sha256": str(r[1] or ""),
                 "bytes": int(r[2] or 0)} for r in conn.execute(
            "SELECT twin_key, sha256, bytes FROM video_twin WHERE video_key=?",
            (video_key,))]
    except sqlite3.Error:
        return []


def bulk(conn: sqlite3.Connection) -> dict:
    """{video_key: {aliases, messages, collections, twins}} in four queries.

    For the index build, which needs all of it for every video and must not do
    ninety round trips to get it.

    `messages` is kept apart from `aliases` because they answer different
    questions. `aliases` is every string that has ever meant this video —
    `10`, `tg10`, `msg_10`, `frames_10` — which is what a resolver needs and what
    no person wants to read. `messages` is `[10, 40]`: the two places in the
    channel this reel actually sits, which is the sentence a card can show.
    """
    out = {}

    def slot(k):
        return out.setdefault(str(k), {"aliases": [], "messages": [],
                                       "collections": [], "twins": []})

    try:
        for a, k, kind in conn.execute(
                "SELECT alias, video_key, kind FROM video_alias WHERE kind<>?",
                (KIND_SELF,)):
            slot(k)["aliases"].append(str(a))
    except sqlite3.Error:
        pass
    try:
        for k, n in conn.execute(
                "SELECT video_key, msg_id FROM video_message ORDER BY msg_id"):
            slot(k)["messages"].append(int(n))
    except sqlite3.Error:
        pass
    try:
        for k, name in conn.execute(
                "SELECT video_key, collection FROM video_collection "
                "ORDER BY collection"):
            slot(k)["collections"].append(str(name))
    except sqlite3.Error:
        pass
    try:
        for k, t in conn.execute(
                "SELECT video_key, twin_key FROM video_twin"):
            slot(k)["twins"].append(str(t))
    except sqlite3.Error:
        pass
    return out


# ══════════════════════════════════════════════════════════════════════════
# THE AUDIT — rule 7
# ══════════════════════════════════════════════════════════════════════════
def audit(conn: sqlite3.Connection) -> dict:
    """Measure the invariant: every key in every table resolves to a video.

    Run after every index build and shown in the interface. The number that
    matters is `ok`: true means no table names a video this archive cannot
    identify, which is the property that was false when Home said 62.

    Cheap — one `SELECT DISTINCT` per key column, over columns that are indexed
    in every table that matters, against an archive with tens of videos.
    """
    ensure(conn)
    keys = canonical_keys(conn)
    stubs = 0
    for vk in keys:
        if is_stub(conn, vk):
            stubs += 1
    res = Resolver(conn)
    tables, bad_total, advisory_total = [], 0, 0
    for table, col in KEY_COLUMNS:
        if not _has(conn, table) or col not in _cols(conn, table):
            continue
        try:
            raw = [r[0] for r in conn.execute(
                f'SELECT DISTINCT "{col}" FROM "{table}" '
                f'WHERE "{col}" IS NOT NULL')]
        except sqlite3.Error:
            continue
        if not raw:
            continue
        mapped = {res(v) for v in raw}
        bad = sorted(k for k in mapped if k and k not in keys)
        advisory = (table, col) in _ADVISORY
        if advisory:
            advisory_total += len(bad)
        else:
            bad_total += len(bad)
        tables.append({"table": table, "column": col,
                       "spellings": len(raw), "videos": len(mapped),
                       "unresolved": len(bad), "examples": bad[:5],
                       "advisory": advisory,
                       "ok": advisory or not bad})

    conflicts = []
    try:
        conflicts = [{"alias": str(r[0]), "kept": str(r[1]),
                      "refused": str(r[2]), "evidence": str(r[3] or "")}
                     for r in conn.execute(
                "SELECT alias, video_key, other_key, evidence "
                "FROM identity_conflict ORDER BY at DESC LIMIT 50")]
    except sqlite3.Error:
        pass

    twins = 0
    try:
        twins = conn.execute(
            "SELECT COUNT(*)/2 FROM video_twin").fetchone()[0] or 0
    except sqlite3.Error:
        pass

    # Videos the channel holds more than one copy of. Not a fault and not a
    # duplicate — it is one reel uploaded twice — but it is the number that used
    # to *become* a duplicate, so it is worth showing rather than inferring.
    reuploads = 0
    try:
        reuploads = conn.execute(
            "SELECT COUNT(*) FROM (SELECT video_key FROM video_message "
            "GROUP BY video_key HAVING COUNT(*) > 1)").fetchone()[0] or 0
    except sqlite3.Error:
        pass

    return {"ok": not bad_total and not conflicts,
            "videos": len(keys), "stubs": stubs,
            "real": len(keys) - stubs,
            "aliases": conn.execute(
                "SELECT COUNT(*) FROM video_alias").fetchone()[0],
            "messages": conn.execute(
                "SELECT COUNT(*) FROM video_message").fetchone()[0],
            "reuploads": int(reuploads),
            "unresolved": bad_total,
            "not_video": advisory_total,
            "conflicts": conflicts,
            "twins": int(twins),
            "collections": collection_counts(conn),
            "tables": tables}


def refresh(conn: sqlite3.Connection, media_dir: str = "",
            ledger_path: str = "") -> dict:
    """Everything, in the right order, as one call.

    The entry point for callers who should not have to know the order: build the
    map, promote the memberships, wire the resolver into `reflect`, then measure.
    `atlas.index.rebuild` calls exactly this before it reads a single row.
    """
    stats = rebuild(conn, media_dir=media_dir, ledger_path=ledger_path)
    cols = rebuild_collections(conn, ledger_path=ledger_path)
    install(conn)
    report = audit(conn)
    return {"identity": stats, "collections": cols, "audit": report}














