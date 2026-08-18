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


def _raise_masked(code, body=b"", msg="error"):
    """Fail the way Kaggle really fails: an HTTP status wearing a network error.

    `make_post_request` catches URLError before HTTPError, so it re-raises every
    status — 401, 403, 404, 500 — as this one ConnectionError sentence. `from`
    is what saves it: the HTTPError stays on `__cause__`, which is where
    `creds._http_status` goes looking for the code.
    """
    http = HTTPError(_ENDPOINT, code, msg, {}, io.BytesIO(body))
    raise ConnectionError(
        "Connection error trying to communicate with service.") from http


_NO_ROW = b'{"wasSuccessful":false,"errors":[{"message":' \
          b'"No user secrets exist for kernel id 1234567"}]}'


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
    return creds


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
assert len(_mask["asked"]) > 12, \
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
