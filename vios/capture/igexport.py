"""
Merge the Instagram export's record for one reel into its capture metadata.

The capture pipeline knows a reel through yt-dlp: uploader, caption, engagement
counts, and a recording of what was on the page the day it was fetched. The
export knows it through Meta: the saved collection, the "save later" timestamps,
the account's own labels. Neither sees what the other sees, and a good record
for a saved reel wants both.

This module is deliberately tiny and read-only. It walks the export's JSON once,
building `permalink -> slice` where a slice is every field the export recorded
about that reel (saved collections it belongs to, owner, taken-at dates,
captions, and the raw node for the rare case where the schema moved on). The
capture engine merges the slice into `<key>.json`; Atlas renders it in the
detail pane so a search result can say *which collection the person saved it
into* — a label that never existed on Instagram itself.

Reuses `vios/capture/inputs.py` where the walk overlaps, but keeps its own
builder: inputs.py produces *collections* (a name per reel) while this produces
*metadata* (everything the export said about a reel).
"""

import json
import os
import re
from collections import OrderedDict

from .inputs import PERMALINK as _PERMALINK

# Saved collections are where a person curates; saved posts are the undirected
# "later" pile. Both are the labels Atlas can search by, and the generic walk
# in inputs.py already proved the export's shape.
_INTERESTING = re.compile(
    r"/(?:saved_collections|saved_posts|collections|saved)(?:/|\.json|$)",
    re.IGNORECASE)

# Collection names are URLs, empty, or the same five words the exporter uses as
# placeholders. A label that collapses to one of these carries no signal.
_BLANK = re.compile(r"^\s*(default|saved|saved posts|all posts|unsaved|"
                    r"untitled|collection|none|—)\s*$", re.IGNORECASE)

_SIZE_CAP = 200 * 1024 * 1024          # refuse to slurp a giant file twice
_MAX_ENTRIES = 400_000                 # sanity bound on one JSON array


def _permaliinks_in(value, out: list):
    """Every permalink anywhere in one export value (strings and nested lists)."""
    if isinstance(value, str):
        out.extend(m.group(0) for m in _PERMALINK.finditer(value))
    elif isinstance(value, list):
        for item in value:
            _permaliinks_in(item, out)


def _clean_collection(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    if len(name) > 60:
        name = name[:60]
    if not name or _BLANK.match(name):
        return ""
    return name


def _extract_name(fields) -> str:
    """The direct `Name` field of a record, if it is a real collection name."""
    if not isinstance(fields, list):
        return ""
    for e in fields:
        if not isinstance(e, dict) or "dict" in e:
            continue
        if str(e.get("label") or "").strip().lower() != "name":
            continue
        val = e.get("value")
        if isinstance(val, str):
            return _clean_collection(val)
    return ""


def _link_of(fields) -> str:
    if not isinstance(fields, list):
        return ""
    for e in fields:
        if not isinstance(e, dict) or "dict" in e:
            continue
        for k in ("href", "value", "url"):
            val = e.get(k)
            if isinstance(val, str) and "instagram.com" in val:
                m = _PERMALINK.search(val)
                if m:
                    return m.group(0)
    return ""


def _walk_records(node, url: str, out: dict):
    """One export file's records → per-permalink slices.

    A record is any dict with a permalink reachable from it. Records are the
    leaves of the file: descending into `dict` groups once is enough, because
    the current export keeps entity groups (Owner, Hashtags) inside `dict`
    values and the records themselves are shallow. Slices *merge* across files,
    so a reel saved in three collections accumulates all three names.
    """
    if not isinstance(node, dict):
        return
    links = []
    _permaliinks_in(node, links)
    if links:
        # Normalise to the same key the ledger uses (shortcode).
        can = None
        for link in links:
            m = _PERMALINK.search(link)
            if m:
                can = m.group(2)
                break
        if can:
            fields = node.get("label_values") or node.get("dict")
            name = _extract_name(fields) if isinstance(fields, list) else ""
            entry = out.setdefault(can, OrderedDict())
            entry.setdefault("collections", [])
            if name and name not in entry["collections"]:
                entry["collections"].append(name)
            entry["url"] = url
            for group in ("owner", "taken_at", "media", "caption",
                          "description", "title"):
                v = node.get(group)
                if v is not None and group not in entry:
                    entry[group] = v
            entry.setdefault("raw", node)
        return
    # No permalink here — this dict is a container; keep looking.
    for v in node.values():
        if isinstance(v, list):
            for item in v:
                _walk_records(item, url, out)


def _process_file(archive, name: str, out: dict):
    if archive.getinfo(name).file_size > _SIZE_CAP:
        return
    low = name.lower()
    if not low.endswith(".json"):
        return
    if not _INTERESTING.search(low):
        return
    try:
        raw = archive.read(name)
    except Exception:
        return
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return
    if isinstance(data, list):
        data = {"list": data}
    if not isinstance(data, dict):
        return
    _walk_records(data, name, out)


def _file_cap() -> int:
    try:
        return int(os.environ.get("VIOS_IG_EXPORT_MB", "0")) * 1024 * 1024
    except ValueError:
        return 0


def slices_from_export(path: str) -> dict:
    """permalink → slice for every reel in an Instagram export zip.

    Returns {} on any failure — the capture must not die because a zip Meta
    produced changed shape. The one thing the caller is told is whether the
    export was read at all, so an empty result can be distinguished from an
    unreadable one.
    """
    cap = _file_cap()
    if cap and os.path.getsize(path) > cap:
        return {}
    try:
        import zipfile
        archive = zipfile.ZipFile(path)
    except Exception:
        return {}
    out = OrderedDict()
    try:
        for name in archive.namelist():
            _process_file(archive, name, out)
    except Exception:
        return {}
    finally:
        try:
            archive.close()
        except Exception:
            pass
    return out


def slice_for(slices: dict, key: str) -> dict:
    """The slice for one reel, or {}."""
    entry = slices.get(str(key))
    if not entry:
        return {}
    return {"collections": entry.get("collections", []),
            "owner": entry.get("owner"),
            "taken_at": entry.get("taken_at"),
            "media": entry.get("media"),
            "caption": entry.get("caption"),
            "raw": entry.get("raw")}


def merge_into(slices: dict, key: str, record: dict) -> dict:
    """Merge the export slice into a record dict, in place.

    The record is the capture's; the slice is the export's. Where both carry a
    value the record wins — it was recorded on the day, the export only when
    the person exported. Collection names are the one thing the export owns
    outright, so they are appended rather than clobbered.
    """
    entry = slice_for(slices, key)
    if not entry:
        return record
    record["instagram_export"] = {
        "collections": entry["collections"],
        "owner": entry.get("owner"),
        "taken_at": entry.get("taken_at"),
        "caption": entry.get("caption"),
        "media": entry.get("media"),
    }
    return record
