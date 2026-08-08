"""
vios.capture.inputs — turn whatever the user has into a permalink queue.

Two front doors, because the user has both:

  1. The Instagram data export ZIP. Requested from Accounts Center, arrives as
     a ZIP of JSON (or HTML) files. `saved_posts.json` /
     `saved_collections.json` hold the saved reels, and the collection name is
     the thing worth preserving — it is the only human categorisation in the
     entire pipeline and it is free.

  2. A markdown file of the shape the old Colab script read: `## Category`
     headers followed by permalinks. Hand-maintained, so it must tolerate
     anything — links inline in prose, links in bullet lists, duplicate
     headers, links before the first header.

Both produce the same thing: an ordered list of (url, collection) pairs. The
ledger deduplicates, so the two inputs can be imported over each other.

The export's JSON layout has changed shape several times and Meta does not
document it. Rather than pattern-match one version, this walks the decoded
JSON generically and pulls every Instagram permalink it finds, attributing it
to the nearest enclosing name-ish key. That is robust to a rename of
`saved_saved_media` (which has happened) and to the HTML export, which is
parsed by the same permalink regex over raw text.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile

from .ledger import PERMALINK, canonical

# Keys that have carried the collection/title in some export version.
_NAME_KEYS = ("title", "name", "collection_name", "saved_collection_name",
              "string_map_data", "media_owner", "value")

# Files inside the export that are worth reading. Everything else is photos,
# messages and ads data — hundreds of megabytes we have no use for.
_INTERESTING = re.compile(
    r"(saved|collection|bookmark|liked)", re.IGNORECASE)

# ...except these, which match the filter by accident. A *comment* you liked is
# not a reel you saved, and importing the posts they hang off would queue
# thousands of videos the user never asked for.
_BORING = re.compile(r"(comment|music|audio|hashtag)", re.IGNORECASE)

MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")

# Export members whose name describes a container rather than a category.
# `liked_posts` is deliberately absent: it is a real distinction worth keeping,
# so those arrive labelled "liked posts" and can be skipped with one entry in
# the skip-collections box.
_GENERIC = re.compile(
    r"^(saved_?posts?|saved_?saved_?media|saved|bookmarks?|"
    r"your_?saved|posts?)$", re.IGNORECASE)


def _clean_collection(name: str) -> str:
    """Normalise a collection label so the same one from two inputs agrees."""
    name = re.sub(r"\s+", " ", (name or "").strip())
    name = name.strip("#*_-–—:· ").strip()
    # Export headers sometimes arrive mojibaked (latin-1 over utf-8). Try to
    # undo it; leave the string alone if it was already correct.
    try:
        fixed = name.encode("latin-1").decode("utf-8")
        if fixed.count("�") == 0 and any(ord(c) > 127 for c in name):
            name = fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return name[:120]


# ═══════════════════════════════════════════════════════════════════════
# Markdown
# ═══════════════════════════════════════════════════════════════════════
def parse_markdown(text: str, default_collection: str = "uncategorised") -> list:
    """`## Category` headers plus permalinks -> [(url, collection), ...].

    Order is preserved, which matters: capture runs FIFO, so the top of the
    file is fetched first and the user's own ordering becomes the priority.
    """
    out: list = []
    current = default_collection
    for line in (text or "").splitlines():
        head = MD_HEADER.match(line)
        if head:
            label = _clean_collection(head.group(1))
            if label:
                current = label
            continue
        for m in PERMALINK.finditer(line):
            out.append((m.group(0), current))
    return out


def parse_markdown_file(path: str, **kw) -> list:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_markdown(f.read(), **kw)


# ═══════════════════════════════════════════════════════════════════════
# Instagram export ZIP
# ═══════════════════════════════════════════════════════════════════════
def _links_in(node) -> list:
    """Permalinks reachable from this dict without descending into children.

    Covers both places the export puts them: a plain string value, and the
    `string_map_data` envelope's `href`.
    """
    found = []
    if not isinstance(node, dict):
        return found
    for v in node.values():
        if isinstance(v, str) and "instagram.com" in v:
            found += [m.group(0) for m in PERMALINK.finditer(v)]
    smd = node.get("string_map_data")
    if isinstance(smd, dict):
        for v in smd.values():
            if isinstance(v, dict):
                for s in (v.get("href"), v.get("value")):
                    if isinstance(s, str) and "instagram.com" in s:
                        found += [m.group(0) for m in PERMALINK.finditer(s)]
    return found


_COLLECTION_KEYS = ("name", "collection", "collection name", "collection_name",
                    "saved collection", "folder")


def _explicit_label(node) -> str:
    """A collection name the export states outright, in `string_map_data`.

    This is read *even when the node also carries a link*, and that is the
    whole reason it is a separate function. `saved_collections.json` puts both
    in one entry:

        {"title": "",
         "string_map_data": {"Name": {"value": "workout"},
                             "Post":  {"href": "https://instagram.com/reel/…"}}}

    The older rule — a node that has a link cannot name a collection — was
    written to stop `saved_posts.json` attributing every reel to its creator's
    username, and it does stop that. But it also threw away the one place the
    real collection names live, so every import came back labelled after the
    file it was read from. `Name` inside `string_map_data` is unambiguous:
    saved_posts puts the username in `title`, never here.
    """
    if not isinstance(node, dict):
        return ""
    smd = node.get("string_map_data")
    if not isinstance(smd, dict):
        return ""
    for k, v in smd.items():
        if k.strip().lower() in _COLLECTION_KEYS and isinstance(v, dict):
            val = v.get("value")
            if isinstance(val, str) and not val.startswith("http"):
                got = _clean_collection(val)
                if got:
                    return got
    return ""


def _label_of(node) -> str:
    """The collection name this node announces, or "".

    Two tiers. An explicit `string_map_data` name always wins. Failing that, a
    node that carries a permalink is a *saved item*, and in `saved_posts.json`
    its `title` is the creator's username, not a collection — attributing 4,000
    reels to 900 one-reel "collections" named after their creators is worse
    than attributing them to the file they came from. So the loose `title`/
    `name` fallback only applies to nodes with no link of their own.
    """
    if not isinstance(node, dict):
        return ""
    explicit = _explicit_label(node)
    if explicit:
        return explicit
    if _links_in(node):
        return ""
    for k in _NAME_KEYS:
        v = node.get(k)
        if isinstance(v, str) and v.strip() and not v.startswith("http"):
            got = _clean_collection(v)
            if got and len(got) > 1:
                return got
    return ""


# ── the export's current envelope: label_values ──────────────────────────
#
# Meta's 2025+ JSON export does not use `string_map_data` for saved content at
# all. Every record is a list of `{"label": …, "value": …}` fields, and related
# entities hang off it as `{"dict": [...], "title": "Owner"}` groups:
#
#   saved_collections.json          saved_posts.json
#   ─────────────────────           ────────────────
#   label_values:                   label_values:
#     {label: Name,  value: brain}    {label: URL,   value: …/reel/ABC/}
#     {label: Type,  value: Default}  {label: Caption, value: …}
#     {dict: [ …33 posts… ],          {dict: […], title: Owner}
#      title: Media}                     └─ {label: Username, value: someone}
#
# Walked generically, the Owner group's `Username` looks exactly like a name and
# becomes the "collection" for every reel that follows it — which is how 82 real
# collections turned into 9,353 imaginary ones named after creators. So the
# shape is parsed structurally instead: a record's name comes only from its own
# direct `Name` field, and entity groups are never descended into.
_ENTITY_GROUPS = {"owner", "brand partner", "brand partners", "hashtags",
                  "co-author", "co-authors", "coauthor", "tagged",
                  "collaborators", "participants", "sharer"}


def _field_list(node):
    """The record's field list, under whichever key this export version used."""
    if not isinstance(node, dict):
        return None
    for key in ("label_values", "dict"):
        v = node.get(key)
        if isinstance(v, list):
            return v
    return None


def _field(fields, label: str) -> str:
    """The direct string value of one label. Groups are not searched."""
    for e in fields:
        if not isinstance(e, dict) or "dict" in e:
            continue
        if str(e.get("label") or "").strip().lower() == label:
            v = e.get("value")
            if isinstance(v, str):
                return v
    return ""


def _walk_fields(fields, inherited: str, out: list, depth: int):
    """One `label_values` record: emit its link, or name a collection.

    A record that carries a URL *is* a saved item, so it keeps the collection it
    was found in and its own `Name`-ish fields are ignored. A record with no URL
    but a `Name` is a collection, and that name is passed down to the group of
    posts nested inside it.
    """
    own = ""
    for lab in ("url", "link", "permalink", "media url"):
        own = _field(fields, lab)
        if own:
            break

    label = inherited
    if not own:
        for lab in ("name", "collection", "collection name", "folder", "title"):
            named = _clean_collection(_field(fields, lab))
            if named:
                label = named
                break

    if own and "instagram.com" in own:
        for m in PERMALINK.finditer(own):
            out.append((m.group(0), inherited))

    for e in fields:
        if not isinstance(e, dict):
            continue
        group = e.get("dict")
        if not isinstance(group, list):
            continue
        if str(e.get("title") or "").strip().lower() in _ENTITY_GROUPS:
            continue
        for member in group:
            _walk_json(member, label, out, depth + 1)


def _walk_json(node, inherited: str, out: list, depth: int = 0):
    """Depth-first walk collecting permalinks with their nearest label.

    Two attribution rules, because the export uses two layouts:

      * **Nested.** The collection title sits one or two levels above the media
        list, so `inherited` carries it down.
      * **Flat.** `saved_collections.json` is a single list where a
        `{"Name": "fitness"}` entry is followed by the entries saved into it —
        siblings, not children. So walking a list keeps a running label that
        each name-marker updates, exactly like the `## Header` rule in the
        markdown parser.

    Meta has shipped both shapes; handling only the first is how an import
    silently loses every collection name.
    """
    if depth > 24:
        return
    if isinstance(node, dict):
        # The current export shape is handled structurally — see _walk_fields.
        # Falling through to the generic walk for these records is what
        # attributed every reel to its creator instead of its collection.
        fields = _field_list(node)
        if fields is not None:
            _walk_fields(fields, inherited, out, depth)
            return
        label = _label_of(node) or inherited
        for url in _links_in(node):
            out.append((url, label))
        for v in node.values():
            if not isinstance(v, str):
                _walk_json(v, label, out, depth + 1)
    elif isinstance(node, list):
        running = inherited
        for v in node:
            marker = _label_of(v)
            if marker:
                running = marker
            _walk_json(v, running, out, depth + 1)
    elif isinstance(node, str) and "instagram.com" in node:
        for m in PERMALINK.finditer(node):
            out.append((m.group(0), inherited))


def _from_html(text: str, base: str) -> list:
    """Permalinks out of the HTML export.

    The links live in `href` attributes, so tags cannot simply be stripped —
    doing that deletes every URL in the file and the import silently returns
    nothing. Anchors are unwrapped to their target first, `<h*>` becomes a
    markdown header so the section titles survive as collections, and only
    then is the remaining markup removed.
    """
    text = re.sub(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>", r" \1 ", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"<h[1-6][^>]*>", "\n## ", text, flags=re.IGNORECASE)
    text = re.sub(r"</h[1-6]>", "\n", text, flags=re.IGNORECASE)
    return parse_markdown(re.sub(r"<[^>]+>", " ", text), base)


def _read_zip(source, all_files: bool = False) -> list:
    """Read an Instagram export -> [(url, collection), ...].

    `source` is a path or any seekable file object, so the admin tab can hand
    over an upload without writing a multi-hundred-megabyte export to disk.

    `all_files=False` reads only the members whose path looks saved-related,
    which is the difference between parsing four files and parsing the whole
    export. If that finds nothing — because Meta renamed the directory again,
    which it has — the caller retries with `all_files=True`.
    """
    out: list = []
    with zipfile.ZipFile(source) as z:
        for name in z.namelist():
            low = name.lower()
            if name.endswith("/"):
                continue
            if not (low.endswith(".json") or low.endswith(".html")
                    or low.endswith(".htm")):
                continue
            if not all_files and not _INTERESTING.search(low):
                continue
            if not all_files and _BORING.search(low):
                continue
            try:
                raw = z.read(name)
            except (zipfile.BadZipFile, OSError, RuntimeError):
                continue
            # The file name is the fallback label, used only for reels no
            # collection claims. `saved_posts.json` is the container of
            # *everything* saved, so "saved posts" is not a category — it is
            # the absence of one, and saying so keeps the tally readable.
            stem = os.path.splitext(os.path.basename(name))[0]
            base = ("uncategorised" if _GENERIC.match(stem)
                    else _clean_collection(stem.replace("_", " ")))
            text = raw.decode("utf-8", "replace")
            if low.endswith(".json"):
                try:
                    _walk_json(json.loads(text), base, out)
                except json.JSONDecodeError:
                    continue
            else:
                out.extend(_from_html(text, base))
    if not out and not all_files:
        if hasattr(source, "seek"):
            source.seek(0)
        return _read_zip(source, all_files=True)
    return out


def parse_export_zip(path: str, all_files: bool = False) -> list:
    return _read_zip(path, all_files)


# ═══════════════════════════════════════════════════════════════════════
# One entry point for the admin tab
# ═══════════════════════════════════════════════════════════════════════
def parse_any(path: str, data: bytes | None = None) -> dict:
    """Detect the format and parse it.

    Accepts a path, or bytes plus a path used only for its extension — the
    admin tab uploads a file object and never touches disk for the .md case.
    Returns {"items": [...], "format": str, "collections": {name: n}}.
    """
    low = (path or "").lower()
    fmt = "unknown"
    items: list = []

    if data is not None and (low.endswith(".zip") or _looks_zip(data)):
        # An in-memory buffer satisfies zipfile's need for a seekable object
        # and avoids writing a multi-hundred-megabyte export to the Kaggle
        # output quota just to read four files out of it.
        items = _read_zip(io.BytesIO(data))
        fmt = "instagram-export-zip"
    elif low.endswith(".zip"):
        items = parse_export_zip(path)
        fmt = "instagram-export-zip"
    elif low.endswith(".json"):
        text = (data.decode("utf-8", "replace") if data is not None
                else open(path, "r", encoding="utf-8", errors="replace").read())
        try:
            node = json.loads(text)
            out = []
            _walk_json(node, "uncategorised", out)
            items = out
            fmt = "instagram-json"
        except Exception:
            # Malformed JSON; fall back to text scraping.
            items = parse_markdown(text)
            fmt = "markdown" if items else "text"
    else:
        text = (data.decode("utf-8", "replace") if data is not None
                else open(path, "r", encoding="utf-8", errors="replace").read())
        items = parse_markdown(text)
        fmt = "markdown" if items else "text"

    # Dedupe while preserving first-seen order, but keep every collection a
    # reel appeared under — a reel saved in two collections is two memberships.
    #
    # The membership test is a set, not `in ordered`. A linear scan of a list
    # that grows to five figures, run once per entry, is quadratic: an export
    # with 18,000 entries spent minutes here while the tab said "Reading…".
    seen: dict = {}
    emitted: set = set()
    ordered: list = []
    for url, col in items:
        can = canonical(url)
        if not can:
            continue
        key, clean = can[0], can[1]
        if key not in seen:
            seen[key] = set()
            ordered.append((clean, col))
            emitted.add((clean, col))
        if col and col not in seen[key]:
            seen[key].add(col)
            if (clean, col) not in emitted:
                emitted.add((clean, col))
                ordered.append((clean, col))

    tally: dict = {}
    for _u, c in ordered:
        tally[c] = tally.get(c, 0) + 1
    return {"items": ordered, "format": fmt, "unique": len(seen),
            "collections": dict(sorted(tally.items(), key=lambda kv: -kv[1]))}


def _looks_zip(data: bytes) -> bool:
    return data[:2] == b"PK"
