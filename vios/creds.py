"""
vios.creds — say the credentials once, not once per session.

The complaint this module answers is exact: "it should only take input from me
only once and should function for months and years, until I revoke or change
things." The previous arrangement held credentials in the engine's memory for
the life of the process, which on Kaggle means twelve hours, and then asked
again. That is the right *security* answer and the wrong *product* answer, and
it does not have to be a trade.

Four places are consulted, in this order, first hit wins:

  1. **What you typed this session.** An explicit value always wins, so a
     paste is still how you override or test something without touching
     anything permanent.

  2. **Kaggle Secrets.** This is the one that makes the promise true. Add-ons →
     Secrets in the notebook editor, one row per credential, attached to your
     Kaggle account rather than to a notebook or a session. Set it once and
     every future session of every future notebook has it, for as long as you
     leave it there. Revoking is deleting the row. Nothing is written to the
     repo, the notebook, or the output quota — Kaggle hands the value to the
     process and it never touches disk.

  3. **Environment variables.** How the same code runs on a laptop, and how a
     contributor runs a pass on their own GPU without being given anything
     permanent.

  4. **A local file**, `~/.vios/credentials.json`, mode 0600, outside the
     repository. Laptop convenience only. Deliberately *not* inside the project
     directory: this repo is public, and a credential file one `git add -A`
     away from being committed is a credential that will eventually be
     committed.

The names, which are the same in Kaggle Secrets and in the environment:

    VIOS_BOT_TOKEN      the bot token from @BotFather
    VIOS_CHANNEL_ID     the channel id, -100…
    VIOS_API_ID         from my.telegram.org
    VIOS_API_HASH       from my.telegram.org
    VIOS_HF_TOKEN       Hugging Face, for the diarisation pass
    VIOS_IG_COOKIES     the Instagram cookie jar, Netscape format

What this module will not do is write a credential anywhere. `save_local` is
the single exception, it is opt-in, it refuses to run on Kaggle, and it writes
outside the repo. Everything else is read-only by construction.
"""

from __future__ import annotations

import json
import os

# name → (env var / secret label, human description)
FIELDS = {
    "bot_token":  ("VIOS_BOT_TOKEN", "Telegram bot token"),
    "channel_id": ("VIOS_CHANNEL_ID", "Telegram channel id"),
    "api_id":     ("VIOS_API_ID", "Telegram API id"),
    "api_hash":   ("VIOS_API_HASH", "Telegram API hash"),
    "hf_token":   ("VIOS_HF_TOKEN", "Hugging Face token"),
    "ig_cookies": ("VIOS_IG_COOKIES", "Instagram cookie jar"),
}

# Other names the same credential is known by, tried after the canonical one.
#
# This exists because a stored secret that is never asked for is
# indistinguishable from a missing one. A session with all four Telegram
# secrets saved correctly still printed "Telegram disabled", because they had
# been stored as TELEGRAM_BOT_TOKEN and VIOS_TELEGRAM_BOT_TOKEN while this
# module only ever called get_secret("VIOS_BOT_TOKEN"). Nothing was wrong with
# the secrets, the bridge, or the engine — the two halves just disagreed about
# spelling, and the log blamed the user for not doing the thing they had done.
#
# The three-way pattern is not arbitrary: root config.py already accepts
# TELEGRAM_* as an environment alias and atlas/config.py already accepts
# ATLAS_*, so these names are what the rest of the tree reads. The only piece
# missing was asking Kaggle for them.
ALIASES = {
    "bot_token":  ("VIOS_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN",
                   "ATLAS_BOT_TOKEN"),
    "channel_id": ("VIOS_TELEGRAM_CHANNEL_ID", "TELEGRAM_CHANNEL_ID",
                   "ATLAS_CHANNEL_ID"),
    "api_id":     ("VIOS_TELEGRAM_API_ID", "TELEGRAM_API_ID",
                   "ATLAS_API_ID"),
    "api_hash":   ("VIOS_TELEGRAM_API_HASH", "TELEGRAM_API_HASH",
                   "ATLAS_API_HASH"),
    "hf_token":   ("VIOS_HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"),
    "ig_cookies": ("VIOS_INSTAGRAM_COOKIES", "IG_COOKIES"),
}

# Secrets this module does not resolve as credentials, but which the rest of
# the system reads straight from os.environ under exactly this name. Bridged
# verbatim so storing one in Kaggle Secrets is enough — VIOS_NIM_API_KEY is
# read by config.py and gates GraphRAG entity extraction.
PASSTHROUGH = ("VIOS_NIM_API_KEY",)

# Names a credential must also appear under because third-party code reads
# them and will never learn ours.
#
# ALIASES is the inbound direction — where a value may already be sitting. This
# is the outbound one, and the two are not symmetric: normalising to a canonical
# name is the right rule for code we own, and useless for code we do not.
# `huggingface_hub` reads HF_TOKEN out of the environment by itself, deep inside
# `from_pretrained`, and nothing in this repository is in a position to hand it
# one. So a session with VIOS_HF_TOKEN stored correctly still declined the
# diarisation pass with "no Hugging Face token in the environment" — the secret
# was present under the only name pyannote could not see. The engine did mirror
# it, but only when the token arrived through the settings form, which is the
# path a Kaggle session never takes.
#
# Mirrors are written only into names that are empty, so an explicit export of
# HF_TOKEN still wins, and mirroring happens even when the canonical name was
# already set — otherwise a hand-exported VIOS_HF_TOKEN would skip the bridge it
# most needs.
MIRROR = {
    "hf_token": ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
}

SECRET = "kaggle-secrets"
ENV = "environment"
FILE = "local file"
TYPED = "typed this session"


def labels(name: str) -> tuple:
    """Every name a credential may be stored under, canonical first.

    The canonical name is the one written back into the environment, so which
    alias a value arrived under never leaks into the rest of the system.
    """
    return (FIELDS[name][0],) + tuple(ALIASES.get(name, ()))

_local_path_override = ""


def local_path() -> str:
    """Where the optional laptop credential file lives.

    `~/.vios/`, never the project directory. See the module docstring for why
    that distinction is load-bearing rather than tidy.
    """
    if _local_path_override:
        return _local_path_override
    return os.path.join(os.path.expanduser("~"), ".vios", "credentials.json")


def on_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                or os.path.isdir("/kaggle/input"))


# ── the four sources ──────────────────────────────────────────────────────
def _from_kaggle() -> dict:
    """Kaggle Secrets, or {} anywhere else.

    Every read is individually guarded: a secret that has not been added
    raises, and one missing secret must not hide the five that are present.
    Each field is tried under all of its names and stops at the first hit, so
    the cost is one call per credential when the canonical name was used and a
    handful of failed calls when it was not. That is paid once, at boot.

    Passthrough secrets come back keyed by their own env-var name, which is
    never a key in FIELDS — so `resolve` and `describe`, which filter on
    FIELDS, ignore them, and only `export_to_env` passes them on.
    """
    try:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415
    except Exception:
        return {}
    try:
        client = UserSecretsClient()
    except Exception:
        return {}

    def _get(label):
        try:
            val = client.get_secret(label)
        except Exception:
            return ""
        return str(val).strip() if val else ""

    out = {}
    for name in FIELDS:
        for label in labels(name):
            val = _get(label)
            if val:
                out[name] = val
                break
    for label in PASSTHROUGH:
        val = _get(label)
        if val:
            out[label] = val
    return out


def _from_env() -> dict:
    out = {}
    for name in FIELDS:
        for label in labels(name):
            val = os.environ.get(label, "")
            if val and val.strip():
                out[name] = val.strip()
                break
    return out


def _from_file() -> dict:
    path = local_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: str(v).strip() for k, v in data.items()
            if k in FIELDS and v and str(v).strip()}


def export_to_env() -> dict:
    """Make Kaggle Secrets visible to code that reads only the environment.

    Returns `{field: env var}` for the variables this call actually set — the
    names, never the values, so a launcher can print the result into a notebook
    log that may be shared.

    Kaggle Secrets are not environment variables. They are an API you have to
    call, and this module is the only thing in the repository that calls it.
    Every other program here reads `os.environ` and has no fallback value,
    because a literal default once put a live bot token in a public repo. Those
    two facts together produced a session that had all four secrets stored
    correctly and still printed "Telegram disabled" — the harvester, the upload
    bot and Atlas had simply never asked. The advice in the boot log was to
    export them by hand in the launch cell, which works and which puts the
    burden in the one place a credential should never end up: a notebook.

    So the launcher asks once, and every process it spawns inherits the answer,
    because `subprocess.Popen` passes this environment on. Nothing is written
    to disk — `save_local` remains the single exception in this module, and it
    refuses to run on Kaggle at all.

    A variable that is already set always wins, so an explicit export is still
    how you override a stored secret for one session without deleting it.

    A value found under an alias is written back under the canonical name as
    well as left where it was, so a credential stored as TELEGRAM_BOT_TOKEN
    reaches code that only ever looks up VIOS_BOT_TOKEN. Normalising here means
    no other file needs an alias list.

    The reverse also happens, for the few names third-party code insists on:
    see MIRROR. Those entries come back keyed `field:ENV_NAME` so a launcher can
    show that HF_TOKEN was set without implying a second secret was found.
    """
    from_kaggle = _from_kaggle()
    exported = {}

    for name in FIELDS:
        canonical = FIELDS[name][0]
        val = os.environ.get(canonical, "").strip()
        if not val:                        # an explicit export outranks a store
            for label in labels(name)[1:]:  # an alias already in the environment
                if os.environ.get(label, "").strip():
                    val = os.environ[label].strip()
                    break
            val = (val or str(from_kaggle.get(name, "") or "")).strip()
            if val:
                os.environ[canonical] = val
                exported[name] = canonical
        if not val:
            continue
        # Outbound mirrors, for libraries that read their own name and cannot be
        # told ours. Deliberately outside the `if not val` above: a token
        # exported by hand under the canonical name needs the mirror just as
        # much as one that came from Kaggle Secrets, and the early `continue`
        # this replaced is exactly why diarisation declined on a session that
        # had the secret stored.
        for label in MIRROR.get(name, ()):
            if os.environ.get(label, "").strip():
                continue
            os.environ[label] = val
            exported[f"{name}:{label}"] = label

    for label in PASSTHROUGH:              # bridged under their own name
        if os.environ.get(label, "").strip():
            continue
        val = from_kaggle.get(label, "")
        if val:
            os.environ[label] = str(val)
            exported[label] = label

    return exported


# ── the resolver ──────────────────────────────────────────────────────────
def resolve(typed: dict | None = None) -> dict:
    """Merge the four sources. Returns {"values": {...}, "sources": {...}}.

    `sources` is what the interface shows. It names where each credential came
    from and never the credential itself, which is what lets the Setup page say
    "bot token: from Kaggle Secrets" — enough to debug a wrong value without
    printing one into a notebook log that may be shared.
    """
    layers = [(FILE, _from_file()), (ENV, _from_env()),
              (SECRET, _from_kaggle())]
    if typed:
        layers.append((TYPED, {k: str(v).strip() for k, v in typed.items()
                               if k in FIELDS and v and str(v).strip()}))

    values: dict = {}
    sources: dict = {}
    for origin, layer in layers:      # later layers win
        for k, v in layer.items():
            values[k] = v
            sources[k] = origin

    if "api_id" in values:
        try:
            values["api_id"] = int(str(values["api_id"]).strip())
        except (TypeError, ValueError):
            values.pop("api_id", None)
            sources.pop("api_id", None)
    return {"values": values, "sources": sources}


def describe(typed: dict | None = None) -> dict:
    """A safe report for the Setup page: presence and origin, never a value."""
    got = resolve(typed)
    values, sources = got["values"], got["sources"]
    rows = []
    for name, (label, desc) in FIELDS.items():
        rows.append({
            "name": name, "label": label, "description": desc,
            "aliases": list(labels(name)[1:]),
            "present": bool(values.get(name)),
            "source": sources.get(name, ""),
        })
    return {
        "fields": rows,
        "on_kaggle": on_kaggle(),
        "kaggle_secrets_available": bool(_from_kaggle()),
        "local_file": local_path(),
        "local_file_present": os.path.isfile(local_path()),
        "complete": all(values.get(k) for k in
                        ("bot_token", "channel_id", "api_id", "api_hash")),
    }


def save_local(values: dict) -> dict:
    """Write the laptop credential file. Opt-in, and never on Kaggle.

    Refused on Kaggle for a reason that is not paranoia: the notebook's
    filesystem is either wiped or published, and there is no third option.
    Kaggle Secrets is the durable store there, and it is a better one.
    """
    if on_kaggle():
        raise RuntimeError(
            "Not on Kaggle. Use Add-ons → Secrets instead — it survives the "
            "session, and the notebook filesystem does not.")
    keep = {k: str(v).strip() for k, v in (values or {}).items()
            if k in FIELDS and v and str(v).strip()}
    if not keep:
        raise RuntimeError("Nothing to save.")
    path = local_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = _from_file()
    existing.update(keep)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    return {"path": path, "fields": sorted(existing)}


def forget_local() -> dict:
    """Delete the laptop credential file. The revoke half of "set it once"."""
    path = local_path()
    if os.path.isfile(path):
        os.remove(path)
        return {"removed": True, "path": path}
    return {"removed": False, "path": path}
