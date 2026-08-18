"""Does the bridge find secrets stored under the names the user actually used?

The store in this test is a verbatim copy of the user's Kaggle Secrets list:
TELEGRAM_*, VIOS_TELEGRAM_* and VIOS_NIM_API_KEY, and not one of the six
VIOS_* names creds.FIELDS used to ask for. Before the alias change every one
of these was invisible and the boot log said "Telegram disabled".

The second half is about the *other* way that sentence gets printed: a sweep
that could not run at all. Thirteen secrets attached correctly still read as
none, because `_from_kaggle` returned a bare `{}` for five different causes and
the log described all five as "already set, or none stored". Each cause is
exercised here, because the only thing that distinguishes them is code that has
to keep working.

The third half — `_raise_masked` and everything below it — is the one that
actually bit. Kaggle's own client reports every HTTP status as
"Connection error trying to communicate with service", because in
`kaggle_web_client.make_post_request` the `except (URLError, socket.timeout)`
clause sits above the `except HTTPError` one and HTTPError subclasses URLError.
So "there is no row called VIOS_BOT_TOKEN" arrives looking exactly like "the
internet is off" — and since VIOS_BOT_TOKEN is the first label asked and one
unreachable used to be fatal, a store holding thirteen secrets was declared
unreadable after a single question, with twelve of those rows never asked for.

The fourth is the session after that one, where the store answered HTTP 429
"Too many requests" instead. Kaggle counts secret reads per account, a sweep is
a dozen of them, and re-running the boot cell a few times is enough to run the
count out. That is a wait, not an error, and it was being reported as an error
with no remedy — so the cases below pin down that a throttle is retried on the
same label, bounded in both seconds and calls, and that the advice says "wait"
rather than anything about the network, the rows or a restart.

Values here are obvious fakes. Nothing real is ever hardcoded in this repo.
"""
import io, os, sys, types
from urllib.error import HTTPError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STORE = {
    "TELEGRAM_API_HASH":       "fake-hash-lower",
    "TELEGRAM_API_ID":         "111111",
    "TELEGRAM_BOT_TOKEN":      "111:fake-lower",
    "TELEGRAM_CHANNEL_ID":     "-1009999999999",
    "VIOS_NIM_API_KEY":        "nvapi-fake",
    "VIOS_TELEGRAM_API_HASH":  "fake-hash-vios",
    "VIOS_TELEGRAM_API_ID":    "222222",
    "VIOS_TELEGRAM_BOT_TOKEN": "222:fake-vios",
    "VIOS_TELEGRAM_CHANNEL_ID": "-1008888888888",
}
ASKED = []


# The four exceptions kaggle_web_client defines. `ConnectionError` shadows the
# builtin there too, which is why creds classifies on the type *name* rather
# than with isinstance — an isinstance check against the builtin would catch
# the wrong thing, or nothing.
class CredentialError(Exception): pass
class BackendError(Exception): pass
class ValidationError(Exception): pass
class ConnectionError(Exception): pass  # noqa: A001 — mirrors Kaggle exactly


_ENDPOINT = "https://www.kaggle.com/requests/GetUserSecretByLabelRequest"


def _raise_masked(code, body=b"", msg="error", hdrs=None):
    """Fail the way Kaggle really fails: an HTTP status wearing a network error.

    `make_post_request` catches URLError before HTTPError, so it re-raises every
    status — 401, 403, 404, 429, 500 — as this one ConnectionError sentence.
    `from` is what saves it: the HTTPError stays on `__cause__`, which is where
    `creds._http_status` goes looking for the code, the body and Retry-After.
    """
    http = HTTPError(_ENDPOINT, code, msg, hdrs or {}, io.BytesIO(body))
    raise ConnectionError(
        "Connection error trying to communicate with service.") from http


_NO_ROW = b'{"wasSuccessful":false,"errors":[{"message":' \
          b'"No user secrets exist for kernel id 1234567"}]}'

# Verbatim from the boot log that prompted this half of the file.
_TOO_MANY = b'{"errors":["Too many requests"],"error":{"code":8},' \
            b'"wasSuccessful":false}'


class _Client:
    def get_secret(self, label):
        ASKED.append(label)
        if label not in STORE:
            # Verbatim Kaggle behaviour: `get_secret` does no translation, so an
            # unattached label arrives as a plain BackendError whose args carry
            # this sentence. Kaggle's own code string-matches it, and so must
            # this — it is the only signal that separates "not attached" from
            # "the store is broken", and getting it wrong means a working sweep
            # reads as a failed one.
            raise BackendError("No user secrets exist for kernel id 1234567")
        return STORE[label]


mod = types.ModuleType("kaggle_secrets")
mod.UserSecretsClient = _Client
sys.modules["kaggle_secrets"] = mod

# A clean environment, so nothing but the bridge can put a value there.
for k in list(os.environ):
    if k.startswith(("VIOS_", "TELEGRAM_", "ATLAS_", "HF_TOKEN")):
        del os.environ[k]

from vios import creds

# Every wait creds asks for, recorded instead of taken. There are three of them
# now — the pacing gap between calls, the retry after a dropped connection, and
# the backoff for a 429 — and a test that actually slept through them would take
# minutes to prove something about seconds.
SLEPT = []


def _no_sleep(mod):
    mod.time = types.SimpleNamespace(sleep=SLEPT.append)
    return mod


_no_sleep(creds)

print("== what the bridge asks Kaggle for ==")
got = creds._from_kaggle()
print("   asked   :", ", ".join(ASKED))
print("   found   :", ", ".join(sorted(got)))
assert set(got) == {"bot_token", "channel_id", "api_id", "api_hash",
                    "VIOS_NIM_API_KEY"}, got

# Canonical first: VIOS_TELEGRAM_BOT_TOKEN is tried before TELEGRAM_BOT_TOKEN,
# so the VIOS_-prefixed row wins when both exist. Which one wins matters less
# than it being deterministic — a user with two rows must not get a coin flip.
assert got["bot_token"] == STORE["VIOS_TELEGRAM_BOT_TOKEN"], got["bot_token"]
assert got["api_id"] == STORE["VIOS_TELEGRAM_API_ID"]

print("\n== export_to_env ==")
exported = creds.export_to_env()
print("   set     :", ", ".join(sorted(exported.values())))
assert exported == {"bot_token": "VIOS_BOT_TOKEN",
                    "channel_id": "VIOS_CHANNEL_ID",
                    "api_id": "VIOS_API_ID",
                    "api_hash": "VIOS_API_HASH",
                    "VIOS_NIM_API_KEY": "VIOS_NIM_API_KEY"}, exported

# The whole point: config.py reads only canonical names and has no alias list.
for label in ("VIOS_BOT_TOKEN", "VIOS_CHANNEL_ID", "VIOS_API_ID",
              "VIOS_API_HASH", "VIOS_NIM_API_KEY"):
    assert os.environ.get(label), label

print("\n== config.py, imported after the bridge ==")
import config
print("   missing_telegram_secrets():", config.missing_telegram_secrets())
print("   NIM_API_KEY present       :", bool(config.NIM_API_KEY))
assert config.telegram_ready(), config.missing_telegram_secrets()
assert config.CHANNEL_ID == int(STORE["VIOS_TELEGRAM_CHANNEL_ID"])
assert config.API_ID == int(STORE["VIOS_TELEGRAM_API_ID"])
assert config.NIM_API_KEY == STORE["VIOS_NIM_API_KEY"]

print("\n== atlas/config.py, same environment ==")
import importlib
import atlas.config as ac
importlib.reload(ac)
print("   missing_secrets():", ac.missing_secrets())
assert ac.telegram_ready(), ac.missing_secrets()
assert ac.CHANNEL_ID == int(STORE["VIOS_TELEGRAM_CHANNEL_ID"])

print("\n== an explicit export still wins ==")
os.environ["VIOS_BOT_TOKEN"] = "333:typed-by-hand"
again = creds.export_to_env()
assert "bot_token" not in again, again
assert os.environ["VIOS_BOT_TOKEN"] == "333:typed-by-hand"
print("   VIOS_BOT_TOKEN kept the hand-set value, not the stored one")

print("\n== an alias already in the environment, with no Kaggle at all ==")
del sys.modules["kaggle_secrets"]
for k in list(os.environ):
    if k.startswith(("VIOS_", "TELEGRAM_", "ATLAS_")):
        del os.environ[k]
os.environ["TELEGRAM_BOT_TOKEN"] = "444:from-env-alias"
importlib.reload(creds)
env_exported = creds.export_to_env()
print("   set     :", env_exported)
assert env_exported == {"bot_token": "VIOS_BOT_TOKEN"}, env_exported
assert os.environ["VIOS_BOT_TOKEN"] == "444:from-env-alias"

print("\n== describe() still reports presence, never a value ==")
d = creds.describe()
blob = repr(d)
for v in STORE.values():
    assert v not in blob, "describe() leaked a stored value"
assert "444:from-env-alias" not in blob, "describe() leaked the env value"
for row in d["fields"]:
    print(f"   {row['label']:<18} present={row['present']!s:<5} "
          f"src={row['source'] or '-':<12} aliases={len(row['aliases'])}")


# ── why a sweep came back empty ───────────────────────────────────────────
# One sentence used to cover all of these: "No Kaggle Secrets to add (already
# set, or none stored)". It is true for exactly one of them.
TOKEN_VAR = creds.KAGGLE_TOKEN_VAR


def _fresh(client_factory, token=True):
    """A creds module with a given fake client and a controlled token bit."""
    for k in list(os.environ):
        if k.startswith(("VIOS_", "TELEGRAM_", "ATLAS_", "HF_TOKEN",
                         "HUGGING")):
            del os.environ[k]
    if token:
        os.environ[TOKEN_VAR] = "fake.jwt.value"
    else:
        os.environ.pop(TOKEN_VAR, None)
    if client_factory is None:
        sys.modules.pop("kaggle_secrets", None)
    else:
        m = types.ModuleType("kaggle_secrets")
        m.UserSecretsClient = client_factory
        sys.modules["kaggle_secrets"] = m
    importlib.reload(creds)
    SLEPT.clear()
    return _no_sleep(creds)


def _check(title, client_factory, expect, token=True, asked=None, values=None):
    c = _fresh(client_factory, token=token)
    rep = c.read_kaggle()
    advice = c.kaggle_advice(rep)
    print(f"\n== {title} ==")
    print(f"   reason  : {rep['reason'] or '(none — the proxy answered)'}")
    print(f"   detail  : {rep['detail'][:70]}")
    print(f"   asked   : {len(rep['asked'])} labels   "
          f"found={len(rep['found'])} absent={len(rep['absent'])} "
          f"broken={len(rep['broken'])}   token={rep['token']}")
    tally = creds._code_tally(rep)
    if tally:
        print("   status  : "
              + ", ".join(f"HTTP {c} ×{n}" for c, n in sorted(tally.items())))
    if rep["throttled"]:
        print(f"   throttle: {rep['throttled']} × HTTP 429, "
              f"waited {rep['waited']:g}s of a {creds._RATE_BUDGET:g}s budget")
    for line in advice:
        print(f"     - {line[:100]}")
    assert rep["reason"] == expect, (title, rep["reason"], expect)
    assert advice, f"{title}: a failure with no advice is the old behaviour"
    for v in STORE.values():                  # the invariant, on every path
        assert v not in repr(advice), f"{title}: advice leaked a value"
    assert "fake.jwt.value" not in repr(rep) + repr(advice), \
        f"{title}: the session token must never be printed"
    if asked is not None:
        assert len(rep["asked"]) == asked, (title, rep["asked"])
    if values is not None:
        assert set(rep["values"]) == values, (title, sorted(rep["values"]))
    return rep


# Kaggle answered, and answered "nothing is attached". The only case the old
# sentence described correctly — and it must not abort the sweep, so every label
# gets asked.
_empty = _check("nothing attached — a working sweep that found nothing",
                lambda: type("C", (), {
                    "get_secret": lambda self, label: (_ for _ in ()).throw(
                        BackendError("No user secrets exist for kernel 1"))
                })(),
                expect="", values=set())
assert len(_empty["absent"]) == len(_empty["asked"]) > 12, _empty["absent"]
assert not _empty["broken"], _empty["broken"]

# The token is not in this process at all, so nothing could have been read.
# Remedy: restart the session, or stop launching from a detached shell.
_check("no token in the environment",
       lambda: (_ for _ in ()).throw(CredentialError("no token")),
       expect=creds.NO_TOKEN, token=False, asked=0)

# A token is present and Kaggle refused it — what a kernel started *before* the
# secrets were attached looks like. Same exception as above, opposite remedy,
# and telling them apart is the only reason `token` is in the report.
_nak = _check("token present and refused (401/403)",
              lambda: (_ for _ in ()).throw(
                  CredentialError("HTTP 403 from /RequestUserSecret")),
              expect=creds.NO_ACCESS, token=True, asked=0)
assert any("restart" in line.lower() for line in creds.kaggle_advice(_nak))

# The proxy did not answer, and no HTTP status is hiding underneath — the one
# case that really is the transport. Retried once per label, then abandoned after
# a run of two: at 40 s per call, believing a dozen of these costs most of nine
# minutes of silent boot. Two rather than one, because one used to be fatal and
# that is what cost a session with thirteen secrets all of them.
_calls = []


class _Dead:
    def get_secret(self, label):
        _calls.append(label)
        raise ConnectionError("timed out talking to the secrets proxy")


_unr = _check("proxy unreachable", _Dead, expect=creds.UNREACHABLE,
              asked=creds._UNREACHABLE_RUN)
assert len(_calls) == 2 * creds._UNREACHABLE_RUN, \
    f"expected one retry per label, got {len(_calls)} calls"
assert len(_unr["broken"]) == creds._UNREACHABLE_RUN, _unr["broken"]
assert not creds._code_tally(_unr), "a transport failure has no status code"

# And the reason that retry exists: one blip used to disable Telegram for a
# twelve-hour session, because a transient failure and an absent secret were
# the same empty dict.
_flaky = {"n": 0}


class _Flaky:
    def get_secret(self, label):
        _flaky["n"] += 1
        if _flaky["n"] == 1:
            raise ConnectionError("connection reset by peer")
        if label not in STORE:
            raise BackendError("No user secrets exist for kernel 1")
        return STORE[label]


_ok = _check("one transient blip, then the proxy answers", _Flaky, expect="")
assert set(_ok["values"]) == {"bot_token", "channel_id", "api_id", "api_hash",
                             "VIOS_NIM_API_KEY"}, sorted(_ok["values"])
print("   recovered: the blip cost a retry, not the session")


# ── the session in the bug report ─────────────────────────────────────────
# Thirteen rows stored, none of them under a name creds asks for *first*, and a
# store that answers a missing label with an HTTP 404 masked as a connection
# error. Every credential is there; the old sweep saw one "connection error" on
# VIOS_BOT_TOKEN, called the store unreachable, and stopped — so the log advised
# turning on the internet that had just finished installing 200 MB of wheels.
class _MaskedNotFound:
    def get_secret(self, label):
        if label not in STORE:
            _raise_masked(404, _NO_ROW, msg="Not Found")
        return STORE[label]


_mask = _check("a missing row arrives as \"Connection error\" (HTTP 404)",
               _MaskedNotFound, expect="")
assert set(_mask["values"]) == {"bot_token", "channel_id", "api_id", "api_hash",
                                "VIOS_NIM_API_KEY"}, sorted(_mask["values"])
assert len(_mask["asked"]) > 8, \
    f"the sweep stopped early again: {_mask['asked']}"
assert not _mask["broken"], \
    f"a 404 is an answer, not a failure: {_mask['broken']}"
assert creds._code_tally(_mask).get(404), "the status code was not recovered"
print("   404 read as \"no such row\": the sweep carried on and found all four")

# The same masking over a stale token. Nothing can be read, and the remedy is a
# restart — but it must not be described as a network fault, and every label is
# still asked because an HTTP error costs no timeout.
class _MaskedRefused:
    def get_secret(self, label):
        _raise_masked(401, b"", msg="Unauthorized")


_ref = _check("a stale token arrives the same way (HTTP 401)", _MaskedRefused,
              expect=creds.NO_ACCESS)
assert len(_ref["asked"]) > 12, _ref["asked"]
assert creds._code_tally(_ref) == {401: len(_ref["asked"])}, \
    creds._code_tally(_ref)
_adv = " ".join(creds.kaggle_advice(_ref)).lower()
assert "restart" in _adv, _adv
assert "internet" not in _adv, f"blamed the network for a 401: {_adv}"

# And a 403 on some rows while others answer, which a stale token cannot do: a
# row that exists but is not switched on for this notebook. Per-label, so the
# sweep must keep going — the lower aliases still resolve every credential.
class _NotSharedWithNotebook:
    def get_secret(self, label):
        if label.startswith("VIOS_TELEGRAM_"):
            _raise_masked(403, b"", msg="Forbidden")
        if label not in STORE:
            _raise_masked(404, _NO_ROW, msg="Not Found")
        return STORE[label]


_tog = _check("some rows not shared with this notebook (HTTP 403)",
              _NotSharedWithNotebook, expect=creds.NO_ACCESS)
assert set(_tog["values"]) == {"bot_token", "channel_id", "api_id", "api_hash",
                               "VIOS_NIM_API_KEY"}, sorted(_tog["values"])
assert _tog["values"]["bot_token"] == STORE["TELEGRAM_BOT_TOKEN"], \
    "the un-shared alias should have fallen through to the shared one"
_adv = " ".join(creds.kaggle_advice(_tog)).lower()
assert "this notebook" in _adv, _adv
print("   the four credentials still came back, from the aliases that answered")


# ── the throttle ──────────────────────────────────────────────────────────
# The next session in the same bug report. Internet on, token present, thirteen
# rows attached, and the store answered every call with HTTP 429 "Too many
# requests" — a count Kaggle keeps per *account*, run up by sweeping the store
# once per boot and then re-running the boot. It landed in BACKEND, which stops
# after three and offers no remedy, so a limit that clears itself in a minute
# disabled Telegram for a twelve-hour session and the log called it an error
# instead of a wait.
class _Throttled:
    """Throttles the first two calls, then behaves. The common case."""

    def __init__(self):
        self.n = 0

    def get_secret(self, label):
        self.n += 1
        if self.n <= 2:
            _raise_masked(429, _TOO_MANY, msg="Too Many Requests")
        if label not in STORE:
            _raise_masked(404, _NO_ROW, msg="Not Found")
        return STORE[label]


_thr = _check("throttled, then the store answers (HTTP 429 ×2)", _Throttled,
              expect="")
assert set(_thr["values"]) == {"bot_token", "channel_id", "api_id", "api_hash",
                              "VIOS_NIM_API_KEY"}, sorted(_thr["values"])
assert _thr["throttled"] == 2, _thr["throttled"]
# 5 s then 15 s: the schedule, on the same label, not counted against it.
assert _thr["waited"] == sum(creds._RATE_BACKOFF[:2]), _thr["waited"]
assert not creds._code_tally(_thr).get(429), \
    "a label that was throttled and then answered is not a failed label"
# The optional credentials keep their canonical spelling and the one this store
# answers under, and lose the rest — those are calls into a limiter that has
# just complained. Named in the advice, never counted as absent.
assert _thr["skipped"], "a throttled sweep still tried every alias"
assert not set(_thr["skipped"]) & set(_thr["asked"]), \
    f"reported as skipped and asked for: {_thr['skipped']}"
for _lbl in _thr["skipped"]:
    assert not any(_lbl in creds.labels(n) for n in creds._REQUIRED), \
        f"skipped a spelling of a credential that gates the channel: {_lbl}"
_adv = " ".join(creds.kaggle_advice(_thr)).lower()
assert "429" in _adv and "waited" in _adv, _adv
assert "not the same as not stored" in _adv, _adv
# And it must not say both. A field with an unasked spelling cannot be reported
# as one the store does not hold — that is this module's original bug verbatim.
assert "vios_hf_token" not in _adv.split("optional credentials")[0], _adv
print(f"   waited it out on the first label and finished the sweep, "
      f"{len(_thr['skipped'])} optional spellings left unasked")


class _HardThrottle:
    """Throttles everything. The remedy is a clock, and the budget is the point."""

    def get_secret(self, label):
        _raise_masked(429, _TOO_MANY, msg="Too Many Requests")


_hard = _check("throttled all the way through", _HardThrottle,
               expect=creds.RATE_LIMIT, asked=1)
assert _hard["waited"] <= creds._RATE_BUDGET, _hard["waited"]
assert _hard["waited"] >= creds._RATE_BACKOFF[0], _hard["waited"]
assert _hard["throttled"] <= creds._RATE_TRIES, _hard["throttled"]
# One label, because every further call is refused anyway and each one pushes
# the window out. This is the only reason that stops the sweep on its first row.
assert creds._code_tally(_hard) == {429: 1}, creds._code_tally(_hard)
_adv = " ".join(creds.kaggle_advice(_hard)).lower()
assert "wait a minute" in _adv, _adv
assert "internet" not in _adv, f"blamed the network for a 429: {_adv}"
assert "too many requests" in _hard["detail"].lower(), _hard["detail"]
print(f"   gave up after {_hard['waited']:g}s of waiting, not after a restart")


class _RetryAfter:
    """Same, but the server says how long. Its number wins over the schedule."""

    def get_secret(self, label):
        _raise_masked(429, _TOO_MANY, msg="Too Many Requests",
                      hdrs={"Retry-After": "3"})


_ra = _check("the store says Retry-After: 3", _RetryAfter,
             expect=creds.RATE_LIMIT, asked=1)
assert 3.0 in SLEPT, SLEPT
assert creds._RATE_BACKOFF[0] not in SLEPT, \
    f"ignored Retry-After and used the schedule: {SLEPT}"
assert _ra["waited"] <= creds._RATE_BUDGET, _ra["waited"]
# A small Retry-After must not turn into a long series of refused calls: the
# limiter is counting requests, so the number of tries is bounded as well as the
# time. Eighteen seconds of waiting, six calls, then a sentence in the log.
assert _ra["throttled"] == creds._RATE_TRIES, _ra["throttled"]
print(f"   honoured the server's own number for {_ra['throttled']} tries "
      f"({_ra['waited']:g}s), then stopped instead of hammering it")


# The number Kaggle actually sends is twenty seconds, and the budget has to be
# bigger than it or honouring Retry-After is theatre: the previous 30 s budget
# waited the twenty, was refused once, and quit with the second wait already over
# budget. That is the boot log the user sent — "waited 20s across 2 throttled
# replies", one label asked, twelve rows never reached.
class _RetryAfter20:
    def get_secret(self, label):
        _raise_masked(429, _TOO_MANY, msg="Too Many Requests",
                      hdrs={"Retry-After": "20"})


_ra20 = _check("the store asks for 20s, repeatedly", _RetryAfter20,
               expect=creds.RATE_LIMIT, asked=1)
assert _ra20["throttled"] > 2, \
    f"gave up after {_ra20['throttled']} replies again: {_ra20['waited']}s"
assert _ra20["waited"] >= 60, _ra20["waited"]
assert SLEPT.count(20.0) >= 3, SLEPT
print(f"   honoured 20s {_ra20['throttled'] - 1}× ({_ra20['waited']:g}s) "
      f"instead of quitting on the second")


# And the same store, throttled once and then answering — the case the budget
# exists for. Everything resolves; the boot is a minute late instead of blind.
class _ThrottledOnce:
    def __init__(self):
        self.n = 0

    def get_secret(self, label):
        self.n += 1
        if self.n == 1:
            _raise_masked(429, _TOO_MANY, msg="Too Many Requests",
                          hdrs={"Retry-After": "20"})
        if label not in STORE:
            _raise_masked(404, _NO_ROW, msg="Not Found")
        return STORE[label]


_once = _check("throttled once for 20s, then the store answers", _ThrottledOnce,
               expect="")
assert set(_once["values"]) == {"bot_token", "channel_id", "api_id", "api_hash",
                               "VIOS_NIM_API_KEY"}, sorted(_once["values"])
assert _once["waited"] == 20, _once["waited"]
print("   one 20s wait bought back every credential in the store")


# The launcher gets told about a wait that long, because a boot log that says
# nothing for two minutes reads as a hung one.
print("\n== a long wait is announced, not silent ==")
_said = []
_c = _fresh(_RetryAfter20)
_c.on_wait = _said.append
_c.read_kaggle()
assert _said, "waited without a word in the log"
assert any("429" in line and "20s" in line for line in _said), _said
for v in STORE.values():
    assert v not in " ".join(_said), "the wait notice leaked a value"
print(f"   {len(_said)} progress line(s), e.g. {_said[0][:88]}")


# Fewer calls is the other half of not being throttled, and it costs nothing:
# the alias lists line up by position, so the spelling that answered for the
# first credential is tried first for the rest.
print("\n== the sweep learns which spelling the store uses ==")
_c = _fresh(_Client)
ASKED.clear()
_learn = _c.read_kaggle()
print(f"   asked   : {len(_learn['asked'])} labels")
print("     " + ", ".join(_learn["asked"]))
assert _learn["asked"][:2] == ["VIOS_BOT_TOKEN", "VIOS_TELEGRAM_BOT_TOKEN"], \
    _learn["asked"][:2]
# Position 1 answered for bot_token, so position 1 is asked first for the next
# credential — one call instead of two, and the same store, and the same answer.
assert _learn["asked"][2] == "VIOS_TELEGRAM_CHANNEL_ID", _learn["asked"][2]
# But only where the spelling exists. hf_token has no VIOS_TELEGRAM_ form, and
# the positional version of this learned "position 1" and duly asked for
# VIOS_HUGGINGFACE_TOKEN before VIOS_HF_TOKEN — a wasted call on the field most
# likely to be stored under its canonical name.
assert (_learn["asked"].index("VIOS_HF_TOKEN")
        < _learn["asked"].index("VIOS_HUGGINGFACE_TOKEN")), _learn["asked"]
assert set(_learn["values"]) == {"bot_token", "channel_id", "api_id", "api_hash",
                                 "VIOS_NIM_API_KEY"}, sorted(_learn["values"])
assert len(_learn["asked"]) <= 13, len(_learn["asked"])
assert not _learn["skipped"], "nothing was throttled, so nothing may be skipped"
print(f"   {len(_learn['asked'])} calls where asking every alias in order is 16")


# Anything else the backend said, carried through verbatim rather than reduced
# to an empty dict — and abandoned after three consecutive failures, because at
# that point it is plainly not about one row. Twenty-four × 40 s is sixteen
# minutes of boot to learn what the third call already said.
_check("backend error",
       lambda: type("C", (), {
           "get_secret": lambda self, label: (_ for _ in ()).throw(
               BackendError("500 Internal Server Error"))
       })(),
       expect=creds.BACKEND, asked=creds._BACKEND_RUN)

# But one bad row among good ones must not stop anything — that is the whole
# reason BACKEND is not simply fatal.


class _OneBadRow:
    def get_secret(self, label):
        if label == "VIOS_BOT_TOKEN":
            raise BackendError("row is corrupt")
        if label not in STORE:
            raise BackendError("No user secrets exist for kernel 1")
        return STORE[label]


_one = _check("one broken row, the rest fine", _OneBadRow,
              expect=creds.BACKEND)
assert set(_one["values"]) == {"bot_token", "channel_id", "api_id", "api_hash",
                              "VIOS_NIM_API_KEY"}, sorted(_one["values"])
assert len(_one["broken"]) == 1, _one["broken"]
print("   the other four credentials still came back")

# Not Kaggle at all. The one case where the boot log must stay silent.
_nm = _check("not a Kaggle session", None, expect=creds.NO_MODULE, asked=0)
assert "kaggle_secrets" in _nm["detail"], _nm["detail"]

print("\n== a child process does not ask again ==")
# One boot is several processes. Phase 0 sweeps and bridges, then ui_server.py
# starts and both v2 engines call resolve() in *their* process, where this
# module's cache is empty — so a store that answered was asked for everything
# two or three times per boot, and that is what runs an account into the 429.
# The marker parent → child carries the verdict; the values are already in the
# environment the child inherited.
parent = _fresh(_Client)
ASKED.clear()
parent.export_to_env()
_paid = len(ASKED)
assert _paid, "the parent has to do the actual reading"
# Sweeping does not mark. Only a launcher does, because its environment dies
# with it — a verdict left in a twelve-hour notebook kernel would outlast the
# minute a rate limit takes to clear, and re-running the cell is that remedy.
assert creds.SWEPT_VAR not in os.environ, "the sweep marked the environment"
assert parent.mark_swept() == "ok", os.environ.get(creds.SWEPT_VAR)

importlib.reload(creds)              # a child: fresh module, inherited environ
_no_sleep(creds)
ASKED.clear()
child = creds.read_kaggle()
assert ASKED == [], f"the child re-swept the store: {ASKED}"
assert child["reason"] == "", child["reason"]
child_env = creds.export_to_env()    # the second call in that process, too
assert ASKED == [], f"export_to_env re-swept the store: {ASKED}"
assert not child_env, f"nothing left to bridge: {child_env}"
print(f"   parent paid {_paid} calls; the child's own resolve() and "
      f"export_to_env() cost {len(ASKED)}")

# And the Setup page in that child still says where the value came from. It
# arrived as an environment variable, but it *originated* in the store, and the
# question the page is answering is "did my stored row get used".
child_src = creds.resolve()["sources"]
assert child_src["bot_token"] == creds.SECRET, child_src
print("   and it still reports bot_token as", child_src["bot_token"])

# A failure is inherited too — with its remedy, and without inventing counters
# the child never had.
os.environ[creds.SWEPT_VAR] = creds.RATE_LIMIT
importlib.reload(creds)
_no_sleep(creds)
ASKED.clear()
inherited = creds.read_kaggle()
assert ASKED == [], ASKED
assert inherited["reason"] == creds.RATE_LIMIT, inherited["reason"]
_adv = " ".join(creds.kaggle_advice(inherited)).lower()
assert "wait a minute" in _adv, _adv
assert "waited 0s" not in _adv, _adv
print("   a throttled Phase 0 reaches the Setup page as a throttle, not a blank")

print("\n== the sweep is not repeated ==")
c = _fresh(_Client)
ASKED.clear()
first = len(c.read_kaggle()["asked"])
c.read_kaggle(); c.read_kaggle(); c.describe()
assert len(ASKED) == first, \
    f"re-swept: {len(ASKED)} calls for {first} labels — describe() polls often"
print(f"   {first} labels asked once; three further reads cost 0 calls")

print("\n== export_to_env asks nothing about what is already set ==")
c = _fresh(_Client)
for label in ("VIOS_BOT_TOKEN", "VIOS_CHANNEL_ID", "VIOS_API_ID",
              "VIOS_API_HASH", "VIOS_NIM_API_KEY", "VIOS_HF_TOKEN",
              "VIOS_IG_COOKIES"):
    os.environ[label] = "already-set-by-hand"
ASKED.clear()
assert c.export_to_env() == {
    "hf_token:HF_TOKEN": "HF_TOKEN",
    "hf_token:HUGGINGFACE_TOKEN": "HUGGINGFACE_TOKEN",
    "hf_token:HUGGING_FACE_HUB_TOKEN": "HUGGING_FACE_HUB_TOKEN",
}, c.export_to_env()
assert ASKED == [], f"asked about values it would discard: {ASKED}"
print("   0 calls, 0 × 40 s timeouts — mirrors still written")

print("\nCREDS OK — the names the user stored are the names the bridge asks "
      "for, and a sweep that fails now says which way it failed")
