"""
Read a plain pg_dump into SQLite.

The bundle carries `omnidb.sql.zst` — the Omniscient store as `pg_dump`
produced it. Atlas has no PostgreSQL: installing a server, initialising a
cluster and replaying a dump costs minutes of startup and a permanently
running service, to hold three tables that are read-only for the lifetime of
this program. So the dump is parsed directly.

That is only reasonable because of what pg_dump's plain format actually is:
DDL, then `COPY table (cols) FROM stdin;` followed by tab-separated rows
terminated by a lone `\\.`. There is no expression evaluation and no procedural
code in a data dump, so a parser needs to handle exactly two things — the CREATE
TABLE column lists, and the COPY blocks. Everything else (SET, ALTER, indexes,
constraints, ownership) is deliberately skipped: it either does not apply to
SQLite or is regenerated locally.

The escapes are the part worth getting right. Inside a COPY block Postgres
writes `\\N` for NULL and backslash escapes for tab, newline, carriage return
and backslash itself. Treating those literally is the difference between a
narrative that reads correctly and one with a stray `\\n` in the middle of every
sentence — and, worse, a row that splits across two lines because an embedded
newline was not unescaped.
"""

import io
import re
import sqlite3

_COPY_RE = re.compile(
    r"^COPY\s+(?:(?P<schema>[\w\"]+)\.)?(?P<table>[\w\"]+)\s*"
    r"\((?P<cols>[^)]*)\)\s+FROM\s+stdin;", re.IGNORECASE)

_CREATE_RE = re.compile(
    r"^CREATE\s+TABLE\s+(?:(?:[\w\"]+)\.)?(?P<table>[\w\"]+)\s*\(",
    re.IGNORECASE)

_PK_RE = re.compile(r"PRIMARY\s+KEY\s*\(([^)]*)\)", re.IGNORECASE)

# Postgres type → SQLite affinity. SQLite is dynamically typed, so this only
# decides storage class hints; getting it approximately right is enough, and
# anything unrecognised becomes TEXT, which never loses data.
_TYPE_MAP = (
    ("bool", "INTEGER"), ("smallint", "INTEGER"), ("bigint", "INTEGER"),
    ("integer", "INTEGER"), ("int", "INTEGER"), ("serial", "INTEGER"),
    ("double", "REAL"), ("real", "REAL"), ("numeric", "REAL"),
    ("decimal", "REAL"), ("float", "REAL"),
    ("json", "TEXT"), ("text", "TEXT"), ("char", "TEXT"),
    ("timestamp", "TEXT"), ("date", "TEXT"), ("time", "TEXT"),
    ("uuid", "TEXT"), ("bytea", "BLOB"),
)


def _unquote(name: str) -> str:
    return name.strip().strip('"')


def _affinity(pg_type: str) -> str:
    t = pg_type.lower()
    for needle, sql in _TYPE_MAP:
        if needle in t:
            return sql
    return "TEXT"


def _unescape(field: str):
    r"""Decode one COPY field. Returns None for \N.

    Written as an explicit scan rather than a chain of str.replace calls: the
    replace approach turns a literal backslash-n in the source text (which
    pg_dump writes as \\n) into a newline, because the second pass sees the
    backslash the first pass left behind.
    """
    if field == r"\N":
        return None
    if "\\" not in field:
        return field
    out = []
    i, n = 0, len(field)
    while i < n:
        c = field[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        nxt = field[i + 1]
        mapped = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
                  "b": "\b", "f": "\f", "v": "\v"}.get(nxt)
        if mapped is not None:
            out.append(mapped)
            i += 2
        else:
            out.append(nxt)
            i += 2
    return "".join(out)


def _parse_create(stream, first_line: str) -> tuple:
    """Consume a CREATE TABLE block, returning (table, [(col, affinity)], pk).

    `pk` matters more than it looks. Atlas merges every bundle in the channel
    into one database, and a table with no primary key cannot be merged — a
    second import of an overlapping snapshot doubles every row. pg_dump writes
    the key as a table-level CONSTRAINT inside the column list, so it is picked
    up here and re-applied on the SQLite side, which turns re-import into
    INSERT OR REPLACE instead of INSERT-again.
    """
    m = _CREATE_RE.match(first_line)
    table = _unquote(m.group("table"))
    body = first_line[m.end():]
    depth = 1 + body.count("(") - body.count(")")
    chunks = [body]
    while depth > 0:
        line = stream.readline()
        if not line:
            break
        depth += line.count("(") - line.count(")")
        chunks.append(line)

    text = "".join(chunks)
    text = text.rsplit(")", 1)[0]

    cols = []
    depth = 0
    current = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            cols.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        cols.append("".join(current))

    out, pk = [], []
    for raw in cols:
        stripped = raw.strip()
        parts = stripped.split()
        if len(parts) < 2:
            continue
        head = parts[0].upper()
        # Table-level constraints share the column list; they are not columns.
        if head in ("CONSTRAINT", "PRIMARY", "UNIQUE", "FOREIGN", "CHECK",
                    "EXCLUDE"):
            pkm = _PK_RE.search(stripped)
            if pkm:
                pk = [_unquote(c) for c in pkm.group(1).split(",") if c.strip()]
            continue
        name = _unquote(parts[0])
        rest = " ".join(parts[1:])
        # A column can also carry the key inline: `id integer PRIMARY KEY`.
        if "PRIMARY KEY" in rest.upper():
            pk = [name]
        out.append((name, _affinity(rest)))
    return table, out, pk



def load_dump(sql_path: str, conn: sqlite3.Connection, prefix: str = "omni_",
              progress=None, replace: bool = False) -> dict:
    """Replay a plain pg_dump into `conn`. Returns {table: row_count}.

    Tables are prefixed so the Omniscient side cannot collide with the harvest
    index — `videos` means something different in each, and silently merging
    them would be a data bug that only shows up as wrong search results.

    Default is merge, not replace. Atlas holds every bundle the channel has
    ever carried, and each bundle is a full snapshot of the machine at one
    moment: importing an older one must not delete rows a newer one contributed,
    and importing the same one twice must not double anything. A primary key
    makes both true, so rows go in with INSERT OR REPLACE when the dump declared
    one. Without a key the only safe option is append, and re-importing an
    unkeyed table would duplicate it — so an unkeyed table is replaced instead,
    which at least converges to the newest snapshot.
    """
    counts = {}
    schemas = {}
    keys = {}

    with open(sql_path, "r", encoding="utf-8", errors="replace",
              newline="") as fh:
        while True:
            line = fh.readline()
            if not line:
                break

            if _CREATE_RE.match(line):
                table, cols, pk = _parse_create(fh, line)
                if cols:
                    schemas[table] = cols
                    keys[table] = pk
                continue

            m = _COPY_RE.match(line)
            if not m:
                continue

            table = _unquote(m.group("table"))
            cols = [_unquote(c) for c in m.group("cols").split(",") if c.strip()]
            dest = prefix + table
            pk = [c for c in keys.get(table, []) if c in cols]

            # The COPY header is authoritative for column order; the CREATE
            # gives types. A dump can COPY a subset of columns, so build the
            # table from the intersection rather than assuming they match.
            declared = dict(schemas.get(table, []))
            existed = _table_exists(conn, dest)
            if existed and (replace or not pk):
                conn.execute(f'DROP TABLE IF EXISTS "{dest}"')
                existed = False
            if not existed:
                ddl_cols = ", ".join(
                    f'"{c}" {declared.get(c, "TEXT")}' for c in cols)
                if pk:
                    ddl_cols += (", PRIMARY KEY (" +
                                 ", ".join(f'"{c}"' for c in pk) + ")")
                conn.execute(f'CREATE TABLE "{dest}" ({ddl_cols})')
            else:
                # A later bundle can carry columns the earlier one did not.
                # Adding them keeps both snapshots readable in one table
                # instead of forcing a full re-import on every schema change.
                _add_missing_columns(conn, dest, cols, declared)

            verb = "INSERT OR REPLACE" if pk else "INSERT"
            placeholders = ", ".join("?" * len(cols))
            insert = (f'{verb} INTO "{dest}" '
                      f'({", ".join(chr(34) + c + chr(34) for c in cols)}) '
                      f'VALUES ({placeholders})')

            n = _copy_rows(fh, conn, insert, len(cols))
            counts[dest] = counts.get(dest, 0) + n
            if progress:
                progress(dest, n)

    conn.commit()
    return counts


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone()
    return row is not None


def _add_missing_columns(conn: sqlite3.Connection, dest: str, cols: list,
                         declared: dict) -> None:
    have = {r[1] for r in conn.execute(f'PRAGMA table_info("{dest}")')}
    for c in cols:
        if c in have:
            continue
        try:
            conn.execute(f'ALTER TABLE "{dest}" '
                         f'ADD COLUMN "{c}" {declared.get(c, "TEXT")}')
        except sqlite3.Error:
            pass



def _copy_rows(fh: io.TextIOBase, conn: sqlite3.Connection, insert: str,
               width: int) -> int:
    """Consume one COPY block's rows, batching inserts.

    Batched at 2000 because a per-row executemany on a 300k-row narrative table
    spends more time in Python call overhead than in SQLite, and an unbatched
    single transaction risks the container's memory on a large dump.
    """
    batch, total = [], 0
    while True:
        line = fh.readline()
        if not line:
            break
        if line.rstrip("\r\n") == r"\.":
            break
        fields = line.rstrip("\r\n").split("\t")
        # A short row means the split found fewer tabs than columns, which only
        # happens on a malformed dump. Padding keeps the load going rather than
        # aborting a 300 MB import over one bad line.
        if len(fields) < width:
            fields += [r"\N"] * (width - len(fields))
        elif len(fields) > width:
            fields = fields[:width]
        batch.append(tuple(_unescape(f) for f in fields))
        total += 1
        if len(batch) >= 2000:
            conn.executemany(insert, batch)
            batch.clear()
    if batch:
        conn.executemany(insert, batch)
    return total
