"""Does the bridge find secrets stored under the names the user actually used?

The store in this test is a verbatim copy of the user's Kaggle Secrets list:
TELEGRAM_*, VIOS_TELEGRAM_* and VIOS_NIM_API_KEY, and not one of the six
VIOS_* names creds.FIELDS used to ask for. Before the alias change every one
of these was invisible and the boot log said "Telegram disabled".

Values here are obvious fakes. Nothing real is ever hardcoded in this repo.
"""
import os, sys, types

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


class _Client:
    def get_secret(self, label):
        ASKED.append(label)
        if label not in STORE:                 # what Kaggle does for a missing row
            raise Exception(f"Secret {label} not found")
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

print("\nCREDS OK — the names the user stored are the names the bridge asks for")
