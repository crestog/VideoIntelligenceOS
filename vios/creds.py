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

SECRET = "kaggle-secrets"
ENV = "environment"
FILE = "local file"
TYPED = "typed this session"

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
    """
    try:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415
    except Exception:
        return {}
    try:
        client = UserSecretsClient()
    except Exception:
        return {}
    out = {}
    for name, (label, _desc) in FIELDS.items():
        try:
            val = client.get_secret(label)
        except Exception:
            continue
        if val and str(val).strip():
            out[name] = str(val).strip()
    return out


def _from_env() -> dict:
    out = {}
    for name, (label, _desc) in FIELDS.items():
        val = os.environ.get(label, "")
        if val and val.strip():
            out[name] = val.strip()
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
    """
    exported = {}
    for name, val in _from_kaggle().items():
        label = FIELDS[name][0]
        if os.environ.get(label, "").strip():
            continue
        os.environ[label] = str(val)
        exported[name] = label
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
