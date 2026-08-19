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
import socket
import threading
import time
from urllib.error import HTTPError

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

# ── why a Kaggle sweep found nothing ──────────────────────────────────────
# The variable Kaggle injects into a notebook session, and which `kaggle_secrets`
# requires before it will talk to the secrets proxy at all. Named here because
# its absence has to be told apart from a *rejected* token: the client raises
# CredentialError for both, and the only way to know which is to ask whether the
# variable is there. The two have opposite remedies.
KAGGLE_TOKEN_VAR = "KAGGLE_USER_SECRETS_TOKEN"

# Set by the launcher, to the outcome of the sweep it did, and inherited by every
# process it starts. Names a fact, never a value. Written only by `mark_swept`.
#
# One boot is not one sweep. Phase 0 sweeps, and then ui_server.py starts and
# both v2 engines call `resolve()` in *their* process, where this module's cache
# is empty — so a boot that found nothing asked the store for everything twice.
# Kaggle counts secret reads per account, and that multiplication is how a store
# holding thirteen good rows starts answering "Too many requests".
#
# The saving is free rather than a trade: attaching a secret does not reach a
# running kernel, so nothing the store holds can change during a session, and
# every value the sweep found is already in this environment — `Popen` hands it
# over. A second sweep can only hear the same answer, more slowly, and closer to
# the limit.
SWEPT_VAR = "VIOS_KAGGLE_SWEPT"

# Which fields the value in this environment originally came out of the store
# for. Field *names*, never values — the companion to `SWEPT_VAR`, and the reason
# the Setup page in a child process can still say "from Kaggle Secrets" about a
# value it received as an inherited environment variable. Both statements are
# true; this is the more useful one, because it answers "did my stored row get
# used" rather than "which mechanism carried it across a process boundary".
ORIGIN_VAR = "VIOS_KAGGLE_ORIGIN"


# The ways a sweep can fail, none of which used to be reported.
#
# `_from_kaggle` returned a bare `{}` for every one of them, so the boot log
# printed a single sentence — "No Kaggle Secrets to add (already set, or none
# stored)" — that is false for four of the five. A session with thirteen secrets
# attached correctly read as a session with none, and the only remedy the log
# offered was to store the secrets that were already stored.
#
# This module's docstring already says a stored secret that is never asked for
# is indistinguishable from a missing one. A stored secret whose *lookup* failed
# is the same bug one layer down, and it was in this file the whole time.
NO_MODULE = "no-module"        # not a Kaggle session — nothing to ask
NO_TOKEN = "no-token"          # the token variable is absent from THIS process
NO_ACCESS = "no-access"        # 401/403 — a token is present and was refused
UNREACHABLE = "unreachable"    # URLError, or the client's own 40 s timeout
RATE_LIMIT = "rate-limited"    # 429 — too many reads, per account, clears itself
BACKEND = "backend"            # anything else the proxy said

# The one failure where every remaining label would fail identically *and* the
# fact is provable without asking: the session token is not in this process, so
# the store was never reachable in the first place.
#
# UNREACHABLE used to be in here, and that single entry cost this module the very
# thing it exists for. Kaggle's own client reports every HTTP status — 401, 403,
# 404, 500 — with the sentence "Connection error trying to communicate with
# service", because in `kaggle_web_client.make_post_request` the
# `except (URLError, socket.timeout)` clause is written *above* the `except
# HTTPError` one, and HTTPError is a subclass of URLError. So the HTTPError
# branch is unreachable code and "this row does not exist" arrives wearing the
# clothes of "the network is down". One strike was fatal, the first label asked
# is VIOS_BOT_TOKEN, and a user who had stored thirteen secrets under the
# TELEGRAM_* and VIOS_TELEGRAM_* spellings had their sweep end after that one
# label — twelve rows that were sitting right there were never asked for, and
# the boot log told them to turn on the internet they were plainly using.
#
# `_http_status` now digs the status code out of `__cause__`, so a 404 is read as
# what it is. What remains genuinely worth abandoning a sweep for is a *run* of
# transport failures with no status code behind them, because those are the only
# ones that cost 40 seconds each.
_FATAL = (NO_TOKEN,)

# How many consecutive BACKEND failures, with nothing found yet, before the
# sweep stops believing they are about individual rows. Three costs two minutes
# at worst; twenty-four costs sixteen.
_BACKEND_RUN = 3

# How many consecutive genuine transport failures — no HTTP status anywhere in
# the chain — before the sweep gives up. Two, because each one is two 40-second
# waits and the second says nothing the first did not; but two rather than one,
# because a single blip must never again cost a twelve-hour session its
# credentials.
_UNREACHABLE_RUN = 2

# How long to wait after a single transport failure before asking the same label
# once more. Small on purpose: this is for a reset connection, not for a throttle.
_BLIP_WAIT = 1.5

# ── the throttle ──────────────────────────────────────────────────────────
# HTTP 429, `{"errors":["Too many requests"],"error":{"code":8}}`. The store is
# not broken, the rows are not missing, the network is fine, and the token is
# good: Kaggle is counting how many secrets this *account* has read lately and
# asking VIOS to come back later. It was landing in BACKEND — "anything else the
# proxy said" — which stopped after three and offered no remedy, so a limit that
# clears itself in about a minute disabled Telegram for a twelve-hour session.
#
# Three things follow from that, and all three are here rather than in the
# classifier, because a 429 is about the endpoint and not about the row:
#
#   * It is retried on the *same* label, not counted against it. Waiting is the
#     documented remedy for a 429 and the only one.
#   * The waiting is bounded, because Phase 0 blocks the boot — but the bound has
#     to be larger than the number Kaggle asks for, or honouring `Retry-After` is
#     theatre. It asks for twenty seconds. A thirty-second budget therefore waited
#     the twenty, was refused once more, and gave up with the second wait already
#     over budget: two and a half minutes of patience is the difference between a
#     boot that starts a minute late and a twelve-hour session with no Telegram.
#   * Every call after the first 429 is spaced further apart, since continuing to
#     hammer a limiter is what extends the window.
_RATE_BACKOFF = (5.0, 15.0, 30.0)  # per successive 429 on one label
_RATE_MAX_WAIT = 60.0              # cap on a Retry-After the server asks for
_RATE_BUDGET = 150.0               # total seconds this sweep will spend waiting
_RATE_TRIES = 6                    # refused calls on one label before giving up

# Called with a progress line while the sweep is waiting out a throttle, if the
# launcher set it. Phase 0 can now legitimately block for two minutes, and a boot
# log that goes silent for two minutes is indistinguishable from a hung one.
on_wait = None

# A small gap between store calls, always. The limiter's shape is not published,
# so this is a hedge and not a proof: a dozen sequential lookups that each return
# in 200 ms is a burst, and a burst is the shape most rate limiters are built to
# catch. Eleven labels cost about two seconds of boot for it. After a 429 the gap
# widens, where it is no longer a guess.
_PACE = 0.2
_PACE_SLOW = 1.0

# A 401 or 403 is not fatal either, though it looks session-wide. Kaggle answers
# a row that exists but is not switched on for *this* notebook the same way it
# answers a stale token, and the toggle case is per-label — so the sweep keeps
# going (an HTTP error costs no timeout) and `kaggle_advice` decides afterwards,
# from how many labels were refused versus answered, which of the two it was.
_REFUSED = (401, 403)

# Statuses whose meaning is settled by the number, so the error body is never
# read: 401/403 is "refused", 404 is "no such row". Anything else — a 400, a 500 —
# is a status VIOS has no rule for, and there the body is the only thing that
# might say which row or what went wrong.
_CODE_IS_ENOUGH = _REFUSED + (404,)

# How far to walk `__cause__`/`__context__` looking for the real exception.
# Kaggle wraps once; this allows for a wrapper of a wrapper without ever risking
# a cycle.
_CHAIN = 6

# The credentials whose absence stops something whole rather than one feature.
# These four gate the channel, and the channel is where every bundle lives — so
# without them there is no harvest, no upload bot and no restore. `hf_token` and
# `ig_cookies` each disable exactly one pass, which is worth one quiet line
# rather than the same alarm. config.py's `missing_telegram_secrets` is the
# authority at run time; this list exists only so the advice can rank itself,
# and creds must not import config (config imports the environment this fills).
_REQUIRED = ("bot_token", "channel_id", "api_id", "api_hash")


def labels(name: str) -> tuple:
    """Every name a credential may be stored under, canonical first.

    The canonical name is the one written back into the environment, so which
    alias a value arrived under never leaks into the rest of the system.
    """
    return (FIELDS[name][0],) + tuple(ALIASES.get(name, ()))


def _prefix_of(name: str, label: str) -> str:
    """The spelling `label` used, as a prefix — "" if it was the canonical name.

    Derived by walking back from the end of both names until they differ, so the
    shared tail (`_BOT_TOKEN`) is discarded and what is left is the family
    (`VIOS_TELEGRAM`, `TELEGRAM`, `ATLAS`). Computed rather than listed because a
    list of prefixes would be a fourth place that has to agree with ALIASES.
    """
    canon = label_of = FIELDS[name][0]
    shared = 0
    while (shared < min(len(canon), len(label))
           and canon[-1 - shared] == label[-1 - shared]):
        shared += 1
    return "" if label == label_of else label[:len(label) - shared]


def _ordered(name: str, prefix: str) -> tuple:
    """`labels(name)`, with the spelling that already answered moved first.

    Someone who stores secrets stores a *set* of them under one spelling, so the
    prefix that produced a value for the first credential is the one most likely
    to produce the next, and trying it first is free.

    It is worth more than tidiness: every label tried and missed is an HTTPS call
    against a store that rate-limits, and this turns the sweep for a
    `VIOS_TELEGRAM_*` store from sixteen calls into eleven. It also means the four
    Telegram values come from one family rather than a mix, which is what someone
    holding two sets of rows meant.

    By prefix and not by position, which is what it used to be. The alias lists
    only line up for the four Telegram credentials; `hf_token`'s second entry is
    `VIOS_HUGGINGFACE_TOKEN`, so a store answering under `VIOS_TELEGRAM_*` taught
    the sweep to ask for that before `VIOS_HF_TOKEN` — a wasted call, on the field
    most likely to be stored under its canonical name. A prefix that matches
    nothing here simply leaves the order alone, which is the correct answer.

    Ordering only — nothing here decides *whether* a label is asked for, so no
    credential becomes unreachable because of it.
    """
    labs = list(labels(name))
    if prefix:
        for i, lab in enumerate(labs):
            if lab.startswith(prefix):
                labs.insert(0, labs.pop(i))
                break
    return tuple(labs)


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
# Every sweep this process has done, keyed by the `skip` it was given. Values
# live here in memory only — the same place `os.environ` keeps them — and never
# reach disk. Kept because secrets cannot change during a Kaggle session
# (attaching one requires a restart), so re-sweeping is thirteen HTTPS calls for
# an answer that could not have moved, and `describe` runs on every status poll.
_KAGGLE_CACHE: dict = {}

# The `skip` keys this process actually put on the wire, as opposed to answered
# from the cache or from a parent's verdict. Separate from `_KAGGLE_CACHE`
# because "we have an answer" and "we are the ones who paid for it" are different
# questions, and only the second one decides whether `SWEPT_VAR` still applies.
_SWEPT_HERE: set = set()


def _retry_after(exc) -> float:
    """The server's own answer to "how long should I wait", or 0.0.

    A 429 usually carries `Retry-After` in seconds. It may instead carry an HTTP
    date, which is ignored rather than parsed: guessing wrong there means either
    hammering the limiter or holding the boot for a minute, and the backoff
    schedule is a safe default for both.
    """
    hdrs = getattr(exc, "headers", None) or getattr(exc, "hdrs", None)
    raw = None
    try:
        raw = hdrs.get("Retry-After") if hdrs is not None else None
    except Exception:                                        # noqa: BLE001
        raw = None
    if not raw:
        return 0.0
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return 0.0


def _http_status(exc) -> tuple:
    """(code, body, retry_after) for the HTTPError under Kaggle's ConnectionError.

    The load-bearing function in this file. `kaggle_secrets` cannot tell a
    missing row from a dead network — see the note above `_FATAL` for why — but
    it does re-raise `from e`, so the HTTPError is still hanging on `__cause__`
    with its status code intact. Reading it back is the difference between "no
    such secret, ask about the next one" and "the internet is off, stop".

    `(0, "", 0.0)` when there is no HTTPError in the chain, which is the one case
    that really is a transport failure.
    """
    cur, seen = exc, 0
    while cur is not None and seen < _CHAIN:
        if isinstance(cur, HTTPError):
            code = 0
            try:
                code = int(getattr(cur, "code", 0) or 0)
            except (TypeError, ValueError):
                code = 0
            wait = _retry_after(cur)
            # The body is only consulted for statuses the code alone does not
            # settle. Reading it means reading from the socket urlopen left open,
            # and while its 40-second timeout bounds that, spending it is
            # pointless for a 404 — which already says everything a 404 can say.
            # A 429 is not in that set on purpose: "Too many requests" in the body
            # is what finally identified this failure, and it costs nothing next
            # to the wait that follows it.
            if code and code in _CODE_IS_ENOUGH:
                return code, str(getattr(cur, "reason", "") or "").strip(), wait
            body = ""
            try:
                # Kaggle never reads the error body, so it is usually still
                # there — and it is where "No user secrets exist for kernel id
                # …" is written. Capped, and never trusted to exist.
                raw = cur.read(2000)
                body = (raw.decode("utf-8", "replace")
                        if isinstance(raw, (bytes, bytearray)) else str(raw))
            except Exception:                                # noqa: BLE001
                body = str(getattr(cur, "reason", "") or "")
            return code, body.strip(), wait
        seen += 1
        cur = cur.__cause__ or cur.__context__
    return 0, "", 0.0


def _is_timeout(exc) -> bool:
    """Whether a real socket timeout is anywhere under this exception."""
    cur, seen = exc, 0
    while cur is not None and seen < _CHAIN:
        if isinstance(cur, (socket.timeout, TimeoutError)):
            return True
        if isinstance(getattr(cur, "reason", None),
                      (socket.timeout, TimeoutError)):
            return True
        seen += 1
        cur = cur.__cause__ or cur.__context__
    return False


def _code_tally(report: dict) -> dict:
    """{http status: how many labels got it}, for the log and the Setup page."""
    tally: dict = {}
    for code in (report.get("codes") or {}).values():
        tally[code] = tally.get(code, 0) + 1
    return tally


def _classify(exc) -> tuple:
    """(reason, text, http_code, retry_after) for one failed lookup.

    A reason of `""` means the store answered cleanly that this label is not
    attached, which is not a failure and must not stop the sweep. Kaggle has two
    ways of saying it and neither is an exception type of its own: on some
    backends `get_secret` raises `BackendError` whose args carry "No user secrets
    exist", and on others the proxy returns an HTTP 404 that arrives here
    disguised as a connection error. Both are read as "not stored".

    The status code is returned as well as consumed, so the report can say
    "HTTP 404 ×24" — which is the sentence that would have ended this bug in a
    minute rather than a session.
    """
    name = type(exc).__name__
    text = (str(exc) or str(getattr(exc, "args", "")) or name).strip()
    low = (text + " " + str(getattr(exc, "args", ""))).lower()

    code, body, wait = _http_status(exc)
    if code:
        blow = body.lower()
        detail = f"HTTP {code} — {body[:120] if body else text}"
        if code in _REFUSED:
            return NO_ACCESS, detail, code, wait
        if code == 429 or "too many requests" in blow:
            # Not this row, not this notebook, not the network. A count, kept per
            # account, that goes back down on its own — so it is the one failure
            # whose remedy is to wait, and the only one worth spending boot time
            # on rather than reporting.
            return RATE_LIMIT, detail, code, wait
        if code == 404 or "no user secret" in blow or "not found" in blow:
            return "", detail, code, wait
        return BACKEND, detail, code, wait

    if name == "NotFoundError" or "no user secret" in low or "not found" in low:
        return "", text, 0, 0.0
    if name == "CredentialError":
        # Present and refused, or never there at all. Same exception, opposite
        # remedy: restart the session, versus fix how boot.py was started.
        return (NO_ACCESS if os.environ.get(KAGGLE_TOKEN_VAR)
                else NO_TOKEN), text, 0, 0.0
    if "too many requests" in low or "rate limit" in low:
        # The same throttle with no HTTPError under it — a wrapper that kept only
        # the sentence. Rare, and cheap to honour.
        return RATE_LIMIT, text, 0, 0.0
    if name == "ConnectionError" or "timeout" in low or _is_timeout(exc):
        return UNREACHABLE, text, 0, 0.0
    return BACKEND, text, 0, 0.0


def _blank() -> dict:
    """An empty sweep report. Shape documented on `read_kaggle`."""
    return {"values": {}, "reason": "", "detail": "",
            "asked": [], "found": [], "absent": [], "broken": [],
            "codes": {},
            # Labels a throttle talked the sweep out of trying. Reported, because
            # "not asked for" and "not stored" reading the same is the original
            # bug in this module and must not come back through a side door.
            "skipped": [],
            # How often the store said "too many requests", and how long this
            # sweep spent waiting for it. Counted separately from `codes`
            # because a label that was throttled and then answered is not a
            # failed label — it is a slow one, and the difference is the whole
            # reason the boot no longer gives up on it.
            "throttled": 0, "waited": 0.0,
            # Presence only, never the value: it is a bearer JWT. Recorded even
            # on the paths that never reach the proxy, because it is the single
            # bit that separates "no token here" from "token refused", and
            # those two have opposite remedies.
            "token": bool(os.environ.get(KAGGLE_TOKEN_VAR, "").strip()),
            "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")}


def read_kaggle(skip: tuple = (), force: bool = False) -> dict:
    """Ask Kaggle Secrets for every credential, and say what happened.

    Returns

        {"values":  {field or passthrough label: value},
         "reason":  "" or one of NO_MODULE / NO_TOKEN / NO_ACCESS /
                    UNREACHABLE / RATE_LIMIT / BACKEND,
         "detail":  the exception text, when there was one,
         "asked":   [label, …]  every name the proxy was asked for,
         "found":   [label, …]  the ones it answered with a value,
         "absent":  [label, …]  the ones it said are not attached,
         "broken":  [(label, text), …]  the ones that failed some other way,
         "codes":   {label: http status}  every status code seen, error or not,
         "skipped": [label, …]  aliases a throttle talked the sweep out of,
         "throttled": how many replies were HTTP 429,
         "waited":  seconds this sweep spent waiting one out,
         "token":   whether the session token variable is set — presence only,
         "run_type": KAGGLE_KERNEL_RUN_TYPE, for the log}

    `reason` is `""` whenever the proxy answered at all, however it answered. A
    session with nothing attached is a *working* sweep that found nothing, and it
    needs a different sentence from a sweep that could not run — the two used to
    share one, and the shared one described neither.

    Every read is still individually guarded, because one missing secret must not
    hide the five that are present. What is no longer guarded away is the reason:
    a lookup that failed on a timeout was indistinguishable from a row that was
    never created, so one blip on the first of two dozen sequential HTTPS calls
    disabled Telegram for a twelve-hour session and blamed the user for not
    storing what they had stored.

    Nor does one failed label end the sweep any more. It did, and because the
    first label asked is `VIOS_BOT_TOKEN` — a name nobody has to use, since three
    aliases are accepted — a store holding thirteen secrets under the other
    spellings was declared unreadable after a single question. See `_FATAL`.

    `skip` names fields already satisfied from a higher-priority source. An
    explicit export outranks a stored secret anyway, so asking about one is a
    call with a 40-second timeout whose answer is discarded.

    If `SWEPT_VAR` is in the environment and this process has not swept, the
    sweep is inherited rather than repeated: a parent already asked, this session,
    and put what it found here. `force=True` overrides that — the `__main__`
    probe means the question literally.
    """
    key = tuple(sorted(skip))
    if not force and key in _KAGGLE_CACHE:
        return _KAGGLE_CACHE[key]

    # Somebody already asked, in this session, on our behalf. `SWEPT_VAR` says
    # so and says how it went. Ask nothing: the store cannot have changed since
    # (attaching a secret does not reach a running kernel), whatever was found is
    # in this environment already, and the store is counting.
    swept = os.environ.get(SWEPT_VAR, "").strip()
    if swept and not force and not _SWEPT_HERE:
        report = _blank()
        report["reason"] = "" if swept == "ok" else swept
        report["detail"] = ("An earlier process in this session already read the "
                            "store; not asking again.")
        _KAGGLE_CACHE[key] = report
        return report

    report = _blank()
    stop: set = set()          # a reason that makes the remaining labels moot
    runs = {UNREACHABLE: 0, BACKEND: 0}   # consecutive failures, by kind
    gap = [_PACE]              # seconds between calls; widens after a throttle

    def done() -> dict:
        _KAGGLE_CACHE[key] = report
        _SWEPT_HERE.add(key)
        return report

    try:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415
    except Exception as exc:                                # noqa: BLE001
        report["reason"] = NO_MODULE
        report["detail"] = f"{type(exc).__name__}: {exc}"
        return done()
    try:
        client = UserSecretsClient()
    except Exception as exc:                                # noqa: BLE001
        report["reason"] = NO_ACCESS if report["token"] else NO_TOKEN
        report["detail"] = f"{type(exc).__name__}: {exc}"
        return done()

    def ask(label: str) -> tuple:
        """(value, reason, text). Retries the transport once, a throttle longer.

        Two failures are retried here rather than reported, because for both of
        them the answer is not about the label being asked about: a reset
        connection, and a 429. Everything else is an answer, including "no such
        row", and asking twice would only cost a call against a store that
        counts them.
        """
        report["asked"].append(label)
        throttles, blipped = 0, False
        while True:
            if gap[0]:
                time.sleep(gap[0])
            try:
                val = client.get_secret(label)
            except Exception as exc:                         # noqa: BLE001
                reason, text, code, wait = _classify(exc)
                if reason == RATE_LIMIT:
                    report["throttled"] += 1
                    # The server's own Retry-After if it sent one, capped, else
                    # the schedule. Widen the gap for every later call too: the
                    # limiter has just said the current rate is too high, and
                    # the rest of the sweep is what would prove it right.
                    delay = min(wait or _RATE_BACKOFF[
                        min(throttles, len(_RATE_BACKOFF) - 1)], _RATE_MAX_WAIT)
                    throttles += 1
                    gap[0] = max(gap[0], _PACE_SLOW)
                    if (throttles < _RATE_TRIES
                            and report["waited"] + delay <= _RATE_BUDGET):
                        report["waited"] = round(report["waited"] + delay, 1)
                        if on_wait:
                            # A name and a number of seconds. The launcher prints
                            # it so two minutes of waiting reads as waiting.
                            try:
                                on_wait(
                                    f"Kaggle is rate-limiting secret reads "
                                    f"(HTTP 429). Waiting {delay:g}s and asking "
                                    f"for {label} again — attempt "
                                    f"{throttles + 1} of {_RATE_TRIES}, "
                                    f"{report['waited']:g}s of "
                                    f"{_RATE_BUDGET:g}s spent.")
                            except Exception:               # noqa: BLE001
                                pass
                        time.sleep(delay)
                        continue
                    # Out of budget, or out of tries. Both bounds matter and for
                    # different reasons: Phase 0 blocks the boot, so the remedy
                    # for a long window is a sentence in the log rather than a
                    # longer wait — and a limiter counts *requests*, so a server
                    # that asks for three seconds twenty times over must not be
                    # answered with twenty more calls.
                    report["codes"][label] = code
                    return "", RATE_LIMIT, text
                if reason == UNREACHABLE and not blipped:
                    # The proxy is genuinely flaky, and the cost of believing it
                    # the first time is a whole session with no Telegram. One
                    # retry, then take the answer. Only reached when there was no
                    # HTTP status behind the error — a 404 is an answer, and
                    # asking it twice is 1.5 s spent to hear it again.
                    blipped = True
                    time.sleep(_BLIP_WAIT)
                    continue
                if code:
                    report["codes"][label] = code
                return "", reason, text
            return (str(val).strip() if val else ""), "", ""

    def sweep(label: str) -> str:
        val, reason, text = ask(label)
        if reason:
            report["reason"] = report["reason"] or reason
            report["detail"] = report["detail"] or text
            report["broken"].append((label, text[:200]))
            if reason in _FATAL:
                stop.add(reason)
            elif reason == RATE_LIMIT:
                # It already waited as long as it is allowed to, so every further
                # label is a call that will be refused and will push the window
                # out. Nothing here is wrong with the store or the rows.
                stop.add(reason)
            elif reason == UNREACHABLE:
                # No status code anywhere in the chain, so this one really is the
                # transport — and the only failure that costs two 40-second waits
                # per label. A run of them is worth abandoning the sweep for; the
                # first of them never was.
                runs[UNREACHABLE] += 1
                runs[BACKEND] = 0
                if runs[UNREACHABLE] >= _UNREACHABLE_RUN:
                    stop.add(UNREACHABLE)
            elif reason == BACKEND and not report["found"]:
                # A bad row, or a bad connection wearing a row's clothing. The
                # run length is what tells them apart, and it is cheaper to
                # decide after three than to pay for twenty-four.
                runs[BACKEND] += 1
                runs[UNREACHABLE] = 0
                if runs[BACKEND] >= _BACKEND_RUN:
                    stop.add(BACKEND)
            # NO_ACCESS deliberately falls through: an HTTP 401/403 costs no
            # timeout, and it is per-label whenever the cause is a row that
            # exists but is not switched on for this notebook. Stopping here
            # would turn one un-toggled row into a session with no credentials.
            if stop:
                # What ended the sweep outranks whatever merely happened during
                # it, because that is the line someone has to act on.
                report["reason"] = reason
                report["detail"] = text or report["detail"]
        elif val:
            report["found"].append(label)
            runs[UNREACHABLE] = runs[BACKEND] = 0
        else:
            report["absent"].append(label)
            runs[UNREACHABLE] = runs[BACKEND] = 0
        return val

    prefix = ""                # the spelling that last produced a value
    for name in FIELDS:
        if name in skip or stop:
            continue
        # Once the store has throttled us even once, the long tail of aliases is
        # no longer free: it is more calls into a limiter that has already said
        # there have been too many. So an *optional* credential gets its canonical
        # name and the spelling this store has been answering under, and no more —
        # two calls rather than four, on the fields whose absence disables one
        # pass. The four that gate the channel keep every spelling: a session
        # without them has no harvest, no upload bot and no restore.
        tail = 2 if (report["throttled"] and name not in _REQUIRED) else None
        tried = _ordered(name, prefix)
        for label in tried[:tail]:
            if stop:
                break
            val = sweep(label)
            if val:
                # Keyed by the canonical field name, never by the alias it was
                # found under — which alias Kaggle happened to hold must not
                # leak past this function.
                report["values"][name] = val
                prefix = _prefix_of(name, label) or prefix
                break
        else:
            # Named, so a credential that was never asked for cannot be reported
            # as one the store does not hold. That confusion is this module's
            # original bug, and a throttle must not reintroduce it.
            report["skipped"] += [lb for lb in tried[tail or len(tried):]]
    for label in PASSTHROUGH:
        if stop or label in skip:
            continue
        val = sweep(label)
        if val:
            # Passthroughs come back under their own env-var name, which is
            # never a key in FIELDS — so `resolve` and `describe`, which filter
            # on FIELDS, ignore them, and only `export_to_env` passes them on.
            report["values"][label] = val
    return done()


def _from_kaggle() -> dict:
    """Just the values, for `resolve`. See `read_kaggle` for the diagnosis.

    Reuses what this process has already swept rather than re-asking. After
    `export_to_env` every value found is in `os.environ`, so a second sweep can
    only re-ask about labels that were *absent* — two dozen HTTPS calls, on a
    status poll, for an answer a Kaggle session cannot change.
    """
    if _KAGGLE_CACHE:
        merged = {}
        for rep in _KAGGLE_CACHE.values():
            merged.update(rep["values"])
        return merged
    return dict(read_kaggle()["values"])


def kaggle_report() -> dict:
    """The sweep already done in this process, or a fresh one.

    Prefers a report that carries a `reason`: a failure is the thing a caller
    needs to show, and a later skip-limited sweep that asked nothing would
    otherwise look clean.
    """
    for rep in _KAGGLE_CACHE.values():
        if rep["reason"]:
            return rep
    if _KAGGLE_CACHE:
        return max(_KAGGLE_CACHE.values(), key=lambda r: len(r["asked"]))
    return read_kaggle()


def mark_swept(report: dict | None = None) -> str:
    """Record this sweep's verdict for the processes this one is about to start.

    The launcher calls this, never the sweep itself, and that is the whole safety
    of it. `boot.py` marks the answer for the children it spawns, and its
    environment dies with it — so re-running the cell asks the store again, which
    is the documented remedy for a rate limit and has to keep working. A notebook
    kernel that imports this module marks nothing: a verdict left in a kernel
    that lives twelve hours would outlast the minute a limit takes to clear and
    would make every later boot inherit a failure that had already passed.

    Returns the verdict written, which is a reason or "ok" — never a value.
    """
    rep = report if report is not None else kaggle_report()
    verdict = (rep.get("reason") or "ok") if rep else "ok"
    os.environ[SWEPT_VAR] = verdict
    return verdict


# ── asking again, later ───────────────────────────────────────────────────
# Phase 0 is bounded on purpose: it blocks the boot, and a boot that waits
# forever for a store is worse than one that starts without Telegram. But
# "bounded" was implemented as "final", and those are different claims. A rate
# limit clears in about a minute; the session it landed in lasts twelve hours.
# Giving up at the 150-second mark and never asking again is how a one-minute
# throttle cost a whole session its channel, its harvester and its restore.
#
# So the launcher's failure is not the last word. If the reason it failed is one
# that time fixes — throttled, unreachable, a backend having a bad minute — a
# daemon thread in the long-lived process asks again on a widening ladder, and
# the moment the answer changes it writes the values into `os.environ` where
# every late-binding reader in this process (see the note in config.py) picks
# them up without a restart.
#
# What does *not* get retried is a settled answer. NO_MODULE means this is not
# Kaggle. NO_TOKEN and NO_ACCESS mean the session cannot read the store at all,
# which no amount of waiting changes. And a sweep that worked and found nothing
# is an answer: the rows are not there, and a Kaggle session cannot see a row
# attached after it started. Retrying any of those is a call against a store
# that counts calls, in exchange for the same reply.
_RECOVER_LADDER = (45.0, 90.0, 180.0, 360.0, 600.0)
_RECOVER_STARTED = threading.Lock()
_recovering = False


def recoverable(report: dict | None) -> bool:
    """Is this a failure that waiting could fix?

    True only for the three reasons that describe the endpoint rather than the
    rows: RATE_LIMIT, UNREACHABLE, BACKEND. A sweep with no reason at all
    succeeded, however little it found, and there is nothing to recover.
    """
    if not report:
        return False
    return report.get("reason") in (RATE_LIMIT, UNREACHABLE, BACKEND)


def recover_later(on_ready=None, on_note=None, report: dict | None = None) -> bool:
    """Re-ask the store on a widening ladder, in the background. Returns whether
    a thread was started.

    `on_note(text)` gets one line per attempt, for the log. `on_ready(exported)`
    is called once, with `export_to_env`'s `{field: env var}` — names only, never
    values — the first time an attempt actually sets something. Both are called
    from the recovery thread, so a caller that touches an event loop from them
    has to hand the work over itself.

    Bounded five ways, because the failure mode to avoid here is a background
    thread quietly hammering a rate limiter for twelve hours:

      * only for `recoverable` reasons;
      * at most one thread per process, ever (`_recovering`);
      * at most `len(_RECOVER_LADDER)` attempts, ~20 minutes end to end;
      * it stops the moment the four required credentials are present, whoever
        supplied them — the Setup page counts;
      * daemon, so it can never hold the process open at shutdown.
    """
    global _recovering

    rep = report if report is not None else kaggle_report()
    if not recoverable(rep):
        return False

    with _RECOVER_STARTED:
        if _recovering:
            return False
        _recovering = True

    def note(text: str):
        if on_note:
            try:
                on_note(text)
            except Exception:                             # noqa: BLE001
                pass

    def satisfied() -> bool:
        return all(
            any(os.environ.get(lbl, "").strip() for lbl in labels(name))
            for name in _REQUIRED
        )

    def loop():
        for attempt, gap in enumerate(_RECOVER_LADDER, start=1):
            time.sleep(gap)
            if satisfied():
                note("credentials arrived from elsewhere — recovery stands down")
                return
            try:
                # force, or the SWEPT_VAR this process inherited from the boot
                # would have it answer from the very failure it is recovering
                # from. `mark_swept` after, so anything this process spawns
                # later inherits the new verdict rather than the old one.
                exported = export_to_env(force=True)
                verdict = mark_swept(kaggle_report())
            except Exception as exc:                      # noqa: BLE001
                note(f"retry {attempt}/{len(_RECOVER_LADDER)} failed: "
                     f"{type(exc).__name__}: {str(exc)[:120]}")
                continue
            if exported:
                note(f"retry {attempt} recovered: "
                     f"{', '.join(sorted(exported.values()))}")
                if on_ready:
                    try:
                        on_ready(exported)
                    except Exception:                     # noqa: BLE001
                        pass
                return
            note(f"retry {attempt}/{len(_RECOVER_LADDER)}: store still says "
                 f"{verdict}")
        note("gave up re-asking Kaggle Secrets; type them on the Setup page "
             "or re-run the launch cell")

    threading.Thread(target=loop, name="vios-creds-recover",
                     daemon=True).start()
    return True


def adopt(values: dict) -> dict:
    """Bridge credentials typed into the interface to the whole process.

    Returns `{field: env var}` for what it set — names, never values.

    `configure()` on either engine used to keep typed credentials in that
    engine's own state, which is enough for the uploader that reads them from
    there and nothing else. `db_restore`, `tg_transport`, the harvester and
    Atlas all read `os.environ`, so a user who typed four correct credentials
    into the Setup page after a throttled boot still got "Telegram is not
    configured" from restore. This is the missing half: typed values go where
    every reader already looks.

    Typed wins, so unlike `export_to_env` this overwrites what is there — that
    is the point of typing it. It never *blanks* a variable, though: a form
    submitted with three fields filled must not delete the fourth.

    Not recorded in ORIGIN_VAR. That variable answers "did this come out of
    Kaggle Secrets", the Setup page shows it as the source, and a typed value
    claiming to be stored would make the Setup page lie about what survives a
    restart.
    """
    exported = {}
    for name, val in (values or {}).items():
        if name not in FIELDS:
            continue
        val = str(val or "").strip()
        if not val:
            continue
        canonical = FIELDS[name][0]
        os.environ[canonical] = val
        exported[name] = canonical
        for label in MIRROR.get(name, ()):
            os.environ[label] = val
            exported[f"{name}:{label}"] = label
    return exported


def kaggle_advice(report: dict) -> list:
    """What to do about a sweep that came back empty, as printable lines.

    Shared by the boot log and the Setup page so the two cannot drift, and it
    names remedies rather than restating the failure — the previous advice was
    "add these as Kaggle Secrets", which is useless to the only person who ever
    reads it: someone who has already added them.

    Never contains a value. That is the whole premise of this module.
    """
    reason = report.get("reason") or ""
    absent = list(report.get("absent") or ())
    broken = list(report.get("broken") or ())
    codes = dict(report.get("codes") or {})
    asked = list(report.get("asked") or ())
    out = []
    if reason == NO_MODULE:
        out.append("Not a Kaggle session — `kaggle_secrets` will not import "
                   "here, so Add-ons → Secrets is not the store in use. Set the "
                   "VIOS_* environment variables instead.")
    elif reason == NO_TOKEN:
        out += [
            f"{KAGGLE_TOKEN_VAR} is not in this process's environment, so no "
            "secret can be read at all — the store was never reached.",
            "Kaggle sets it in a notebook session and child processes inherit "
            "it. Restart the session (Run → Restart session) and re-run the "
            "launch cell.",
            "If boot.py was started from a Kaggle Terminal, or detached with "
            "nohup, run it from a notebook cell instead: the variable is "
            "inherited, not global.",
        ]
    elif reason == NO_ACCESS:
        refused = sorted(lbl for lbl, c in codes.items() if c in _REFUSED)
        answered = len(report.get("found") or ()) + len(absent)
        if refused and answered:
            # Some labels answered and others were refused, which a stale token
            # cannot do — a token is either good for the session or good for
            # none of it. What is per-row is the notebook toggle.
            out += [
                f"Kaggle answered for {answered} label(s) and refused "
                f"{len(refused)}: " + ", ".join(refused[:6])
                + ". A refusal for some rows and not others is the per-notebook "
                "switch, not the session token.",
                "In Add-ons → Secrets, switch those rows on for THIS notebook, "
                "then restart the session.",
            ]
        else:
            out += [
                "Kaggle refused this session's token (401/403). A token is "
                "present, so the secrets are not missing — the token is stale.",
                "That is exactly what a session started *before* the secrets were "
                "attached looks like: attaching does not reach a kernel that is "
                "already running. Restart the session and re-run the launch cell.",
            ]
    elif reason == UNREACHABLE:
        out += [
            "The Kaggle secrets proxy did not answer, twice, and there was no "
            "HTTP status behind it — so this one is the transport. Secrets are "
            "read over HTTPS, so Internet must be on for the notebook "
            "(Settings → Internet).",
        ]
    elif reason == RATE_LIMIT:
        # The remedy is a clock, so the advice has to say so plainly. Everything
        # a person would reach for first — check the rows, check the toggle,
        # check the internet, restart the session — is wasted here, and a restart
        # actively makes it worse: the new session starts by sweeping again.
        waited = report.get("waited") or 0
        hits = report.get("throttled") or 0
        out.append(
            "Kaggle is rate-limiting secret reads for this account (HTTP 429). "
            "The rows are fine, the token is fine, the network is fine — the "
            "store is counting how many secrets have been read lately and is "
            "asking VIOS to come back later.")
        if hits:
            # Only when this process is the one that waited. A child process
            # inherits the verdict without the counters, and "it waited 0s
            # across 0 replies" would be a lie about a number nobody needs.
            out.append(
                f"It waited {waited:g}s across {hits} throttled repl"
                f"{'y' if hits == 1 else 'ies'} — up to {_RATE_BUDGET:g}s is "
                "allowed — and then stopped rather than hold the boot open. The "
                "count is per Kaggle account and it falls back down on its own, "
                "usually within a minute or two.")
        out += [
            "So: wait a minute, then re-run the cell. Do not restart the "
            "session — a fresh session starts by sweeping the store again, "
            "which is the one thing that keeps the window open.",
            "What runs the count up is repeated sweeps in a short window: every "
            "boot.py asks for up to a dozen labels, and `python -m vios.creds` "
            "is a whole sweep of its own. If you are re-running boot.py a lot, "
            "run the bridge cell in RUNNING.md once instead — it puts the values "
            "in the notebook kernel's environment, and every later boot inherits "
            "them and asks the store for nothing.",
        ]
    elif reason == BACKEND:
        if report.get("found"):
            out.append(
                f"Kaggle answered for {len(report['found'])} label(s) and "
                "errored on at least one other — those credentials are "
                "present, and only the failed labels below are unresolved.")
        else:
            out.append("Kaggle answered with an error instead of a value — the "
                       "detail above is its own wording. The sweep stopped "
                       f"after {_BACKEND_RUN} in a row rather than spend "
                       "forty seconds per label proving it.")
    elif absent:
        # Only credentials that ended up with no value at all, named by their
        # canonical label rather than by every alias tried. A field found under
        # its third alias leaves the first two in `absent` quite legitimately,
        # and reporting those as "not attached" after a sweep that resolved
        # everything reads as a failure — the same confusion this function
        # exists to end, pointed the other way.
        have = set(report.get("values") or ())
        left = set(report.get("skipped") or ())

        def _short(names) -> list:
            # A credential with an unasked spelling is not reported here at all.
            # "Not stored" and "not looked for" are different sentences, the
            # second one is printed below, and a field cannot honestly get both.
            return [FIELDS[n][0] for n in names if n not in have
                    and any(lbl in absent for lbl in labels(n))
                    and not any(lbl in left for lbl in labels(n))]

        missing = _short(_REQUIRED)
        optional = _short([n for n in FIELDS if n not in _REQUIRED])
        optional += [lbl for lbl in PASSTHROUGH
                     if lbl in absent and lbl not in have]
        if missing:
            out += [
                "Kaggle answered, and holds no row for: " + ", ".join(missing)
                + f" — nor for any alias of them ({len(absent)} labels tried).",
                "In Add-ons → Secrets check the row exists, check its toggle is "
                "on for this notebook, and then restart the session — attaching "
                "a secret does not reach a kernel that is already running.",
            ]
        elif optional:
            out.append("Kaggle answered. Not stored, each disabling only its "
                       "own feature: " + ", ".join(optional) + ".")
    if (report.get("throttled") or 0) and reason != RATE_LIMIT:
        # The sweep was throttled and rode it out. Worth one line, because
        # otherwise twenty extra seconds of silence in Phase 0 has no
        # explanation — and because it is the warning shot before the version of
        # this that does not recover.
        out.append(
            f"Kaggle throttled {report['throttled']} read(s) with HTTP 429 and "
            f"the sweep waited {report.get('waited') or 0:g}s for the limit to "
            "clear. Nothing was lost; if it starts happening every boot, see the "
            "bridge cell in RUNNING.md, which asks the store for nothing.")
    if report.get("skipped"):
        # Not the same sentence as "absent", and never merged into it: these
        # labels were never asked about. The credential may be stored under one of
        # them, and the reason it was not looked for is that looking costs a call.
        skipped = list(report["skipped"])
        out.append(
            f"To stay under the limit it stopped trying alternative spellings for "
            f"optional credentials: {', '.join(skipped[:6])}"
            f"{f' and {len(skipped) - 6} more' if len(skipped) > 6 else ''}. "
            "These were not asked about, which is not the same as not stored — "
            "the four that gate the channel were tried under every spelling. A "
            "later boot with no throttle asks for all of them.")
    if codes:
        # The line that would have ended the worst version of this bug in a
        # minute. `kaggle_secrets` prints "Connection error trying to communicate
        # with service" for a 404 and for a dead network alike, so the status
        # codes are the only honest evidence about which happened — and until
        # they were dug out of `__cause__` nothing in the log had them.
        tally: dict = _code_tally(report)
        out.append(
            "Status from the store: "
            + ", ".join(f"HTTP {c} ×{n}" for c, n in sorted(tally.items()))
            + f" over {len(asked)} label(s). Kaggle's client reports every one "
            "of these as \"Connection error trying to communicate with "
            "service\", so that wording is not evidence about your network.")
    if broken and reason != NO_MODULE:
        out.append("Failed for another reason: "
                   + "; ".join(f"{lbl} — {txt}" for lbl, txt in broken[:3]))
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


def export_to_env(force: bool = False) -> dict:
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

    Nothing is asked about a credential that is already in the environment. Each
    Kaggle lookup carries a 40-second timeout, so a sweep for values that would
    be discarded anyway is minutes of boot spent to lose an argument it already
    lost.

    `force=True` re-asks the store even though something in this session already
    swept it — for `recover_later`, whose entire job is to ask again after a
    throttle has had time to clear. It does not re-ask about credentials already
    in the environment; those are answered, not throttled.
    """
    # What is already satisfied, decided before the sweep so the sweep can skip
    # it. `labels()` covers the canonical name and every alias, which is the same
    # rule the per-field loop below applies — kept as one expression so the two
    # cannot disagree about what "already set" means.
    satisfied = tuple(
        name for name in FIELDS
        if any(os.environ.get(lbl, "").strip() for lbl in labels(name))
    ) + tuple(
        label for label in PASSTHROUGH if os.environ.get(label, "").strip()
    )
    from_kaggle = read_kaggle(skip=satisfied, force=force)["values"]
    exported = {}
    origin = {n for n in os.environ.get(ORIGIN_VAR, "").split(",") if n}

    for name in FIELDS:
        canonical = FIELDS[name][0]
        val = os.environ.get(canonical, "").strip()
        if not val:                        # an explicit export outranks a store
            for label in labels(name)[1:]:  # an alias already in the environment
                if os.environ.get(label, "").strip():
                    val = os.environ[label].strip()
                    break
            if not val:
                val = str(from_kaggle.get(name, "") or "").strip()
                if val:
                    origin.add(name)       # so every later process can say so
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

    if origin:
        os.environ[ORIGIN_VAR] = ",".join(sorted(origin))
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

    # A value that reached this process as an environment variable because a
    # parent process read it out of the store is still, to the person reading the
    # Setup page, "from Kaggle Secrets" — and saying "environment" there would
    # send them looking for an export they never wrote. Only ever narrows ENV to
    # SECRET, so a hand-set override keeps its own label.
    for name in (n for n in os.environ.get(ORIGIN_VAR, "").split(",") if n):
        if sources.get(name) == ENV:
            sources[name] = SECRET

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
    # The diagnosis, not a re-sweep. `bool(_from_kaggle())` used to mean thirteen
    # HTTPS calls on every status poll, and it answered the wrong question: False
    # said "no secrets" when it often meant "could not ask", which is the failure
    # a Setup page exists to explain.
    report = kaggle_report()
    return {
        "fields": rows,
        "on_kaggle": on_kaggle(),
        "kaggle_secrets_available": bool(report["found"]),
        "kaggle_reason": report["reason"],
        "kaggle_detail": report["detail"][:300],
        "kaggle_token": report["token"],
        "kaggle_asked": len(report["asked"]),
        "kaggle_found": len(report["found"]),
        "kaggle_codes": _code_tally(report),
        "kaggle_throttled": report.get("throttled", 0),
        "kaggle_waited": report.get("waited", 0.0),
        "kaggle_advice": kaggle_advice(report),
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


# ── the probe ─────────────────────────────────────────────────────────────
# `python -m vios.creds`, from the same notebook cell that runs boot.py.
#
# Phase 0 prints a summary, and a summary is the wrong instrument when the
# question is "which of my rows did the store actually answer for". This prints
# one line per label with the HTTP status beside it, which is the fact that was
# missing while a masked 404 was being read as a dead network.
#
# It prints names and statuses. It never prints a value, and it never prints the
# session token — the same rule as everything else in this file, because a
# notebook log is a thing people paste into issues.
if __name__ == "__main__":
    print("(this is a full sweep of its own — Kaggle counts secret reads per "
          "account, so run it instead of boot.py, not seconds before it)")
    _rep = read_kaggle(force=True)
    print(f"on_kaggle={on_kaggle()}  {KAGGLE_TOKEN_VAR}="
          f"{'present' if _rep['token'] else 'ABSENT'}  "
          f"run_type={_rep['run_type'] or 'unset'}")
    print(f"asked={len(_rep['asked'])}  found={len(_rep['found'])}  "
          f"not-stored={len(_rep['absent'])}  failed={len(_rep['broken'])}  "
          f"reason={_rep['reason'] or '(none — the store answered)'}")
    if _rep["throttled"]:
        print(f"throttled={_rep['throttled']} replies  "
              f"waited={_rep['waited']:g}s")
    _tally = _code_tally(_rep)
    if _tally:
        print("status codes: "
              + ", ".join(f"HTTP {c} ×{n}" for c, n in sorted(_tally.items())))
    _why = dict(_rep["broken"])
    for _label in _rep["asked"]:
        if _label in _rep["found"]:
            _state = "STORED (value not printed)"
        elif _label in _rep["absent"]:
            _state = "not stored"
        else:
            _state = "FAILED — " + _why.get(_label, "")
        _code = _rep["codes"].get(_label)
        print(f"   {_label:<26} {_state}"
              + (f"   [HTTP {_code}]" if _code else ""))
    for _line in kaggle_advice(_rep):
        print(f"   {_line}")
