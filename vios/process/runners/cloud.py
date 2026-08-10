"""
vios.process.runners.cloud — the passes that run on someone else's GPUs.

One model in the catalogue does not belong on a T4. `narrate-deep` is
InternVL3-38B sharded across both cards over PCIe, ninety-five seconds a video
with nothing else resident, and on a two-card Kaggle session it is the pass most
likely to fall over. The answer is not to fight it and not to drop the quality
tier: it is to run that reading through NVIDIA's hosted API, where the model is
larger than anything that would fit locally and costs no VRAM at all.

Costing no VRAM is what makes this a gain rather than a consolation. A cloud
pass is `device="cpu"`, `vram_mb=0`, so the packer schedules it *alongside* a
full GPU cohort instead of displacing one. The frontier-quality reading happens
in the gaps.

**The constraint that shapes everything here.** The free tier is roughly forty
requests per minute, applied account-wide across every model, and metered in
credits that deplete faster on larger models. The v1 plane already learned this
the hard way: `omni_engine.extract_and_store_graphrag` calls a 550B model once
per narrative chunk, which produced `ResourceExhausted (33/32)` in the log and
then — worse — latched a warn-once flag on the first failure and silently
disabled entity extraction for the rest of the session.

So this module holds exactly one client for the whole process, with one token
bucket, one concurrency bound and one backoff policy. Every NIM caller in the
engine shares them, which is the only way the narrative pass and GraphRAG
cannot starve each other. Two rules follow from the quota rather than from
taste:

- **One call per video.** Not one per chunk, not one per shot. The assembled
  evidence goes up in a single request and the deep reading comes back as one
  structured response.
- **A rate-limited video is deferred, never failed.** Coverage is keyed on
  `(video, component)` and shard replay is idempotent, so a pass that gets
  through two hundred videos today resumes tomorrow exactly where it stopped.
  That is the same bargain as the rest of the system: time is not a constraint,
  quality is the reward for waiting.

Which model is configuration, not code. The catalogue changes month to month,
so `VIOS_NIM_MODEL` selects it and `preflight()` asks `GET /v1/models` what the
key can actually reach today — a wrong model name should read as one clear line
naming the models that exist, not as a 404 buried in a traceback.
"""

from __future__ import annotations

import json
import os
import threading
import time

from .base import DeferPass, Emission, Job, SkipPass


# ══════════════════════════════════════════════════════════════════════════
# Rate limiting
# ══════════════════════════════════════════════════════════════════════════

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _sayer(log):
    """A two-argument logger, whatever was handed in.

    `Job.log` takes one argument and the engine's own takes two. A cloud
    failure is exactly the moment a `TypeError` in the logging call would
    replace the message that explains what went wrong, so the difference is
    absorbed here rather than at every call site.
    """
    if log is None:
        return lambda message, level="info": (
            print(f"[nim] {level}: {message}", flush=True)
            if level in ("warn", "error") else None)

    def say(message: str, level: str = "info") -> None:
        try:
            log(message, level)
        except TypeError:
            try:
                log(f"{message}" if level == "info" else f"[{level}] {message}")
            except Exception:
                pass
        except Exception:
            pass
    return say


# Requests per minute. The observed account-wide ceiling is ~40; the default
# sits under it because the ceiling is shared with the v1 plane's GraphRAG
# calls and with anything else using the same key.
RPM = _env_float("VIOS_NIM_RPM", 30.0)

# Simultaneous in-flight requests. A large model answering a long prompt can
# take a minute, and four of those queued behind each other is enough to keep
# throughput at the rate limit without ever bursting past it.
CONCURRENCY = int(_env_float("VIOS_NIM_CONCURRENCY", 4))

# How long a caller waits for a token before giving up and deferring. Waiting
# longer than this inside a pass would hold a coverage lease open doing
# nothing, which is worse than coming back later.
MAX_WAIT = _env_float("VIOS_NIM_MAX_WAIT", 45.0)

REQUEST_TIMEOUT = _env_float("VIOS_NIM_TIMEOUT", 180.0)

# Retries on a transient refusal, and the waits between them. Beyond this the
# pass defers rather than retrying — the quota is a clock, and the right way to
# wait out a clock is to release the lease.
RETRIES = int(_env_float("VIOS_NIM_RETRIES", 3))
BACKOFF = (4.0, 12.0, 30.0)


class _Bucket:
    """A token bucket shared by every NIM caller in this process.

    Sized at one minute of capacity so a burst after an idle stretch is
    allowed — which is what a rate limit of "40 per minute" actually permits —
    while the refill rate holds the long-run average under the ceiling.
    """

    def __init__(self, rpm: float):
        self.rate = max(0.1, float(rpm)) / 60.0
        self.capacity = max(1.0, float(rpm))
        self._tokens = self.capacity
        self._at = time.monotonic()
        self._lock = threading.Lock()

    def take(self, timeout: float) -> float:
        """Spend one token. Returns seconds waited, or -1 on timeout."""
        deadline = time.monotonic() + max(0.0, timeout)
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity,
                                   self._tokens + (now - self._at) * self.rate)
                self._at = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                need = (1.0 - self._tokens) / self.rate
            if time.monotonic() + need > deadline:
                return -1.0
            time.sleep(min(need, 1.0))
            waited += min(need, 1.0)

    def available(self) -> float:
        with self._lock:
            now = time.monotonic()
            return min(self.capacity,
                       self._tokens + (now - self._at) * self.rate)


# ══════════════════════════════════════════════════════════════════════════
# Classifying a refusal
# ══════════════════════════════════════════════════════════════════════════

class RateLimited(Exception):
    """The account is out of requests. Try again later; nothing is wrong."""

    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message)
        self.retry_after = float(retry_after)


class NotConfigured(Exception):
    """No key, a rejected key, or a model this key cannot reach.

    Permanent for this session. Distinguished from `RateLimited` because the
    two demand opposite responses — one waits, the other must stop retrying and
    say so once. Conflating them is precisely the v1 bug: a transient burst
    latched the warn-once flag and disabled extraction for the whole session.
    """


_TRANSIENT = ("429", "resourceexhausted", "resource exhausted", "rate limit",
              "too many requests", "503", "502", "504", "overloaded",
              "service unavailable", "timeout", "timed out", "connection")
_PERMANENT = ("401", "403", "unauthorized", "invalid api key",
              "authentication", "does not exist", "not found", "404",
              "model_not_found")


def _retry_after(exc: Exception) -> float:
    """Seconds the server asked us to wait, when it said."""
    for attr in ("response", "http_response"):
        resp = getattr(exc, attr, None)
        headers = getattr(resp, "headers", None)
        if not headers:
            continue
        for name in ("retry-after", "Retry-After", "x-ratelimit-reset"):
            try:
                value = headers.get(name)
            except Exception:
                value = None
            if value:
                try:
                    return max(1.0, float(value))
                except (TypeError, ValueError):
                    pass
    return 0.0


def classify(exc: Exception) -> Exception:
    """Turn a provider exception into one of ours, or return it unchanged.

    Text matching rather than exception classes because the OpenAI SDK, httpx
    and the gRPC layer underneath NIM all surface the same conditions as
    different types, and the strings are the only thing common to all three.
    Permanent is checked first: a 404 whose body happens to mention a timeout
    is still a 404.
    """
    blob = f"{type(exc).__name__}: {exc}".lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        if status in (401, 403, 404):
            return NotConfigured(str(exc))
        if status in (429, 500, 502, 503, 504):
            return RateLimited(str(exc), _retry_after(exc) or 60.0)
    if any(t in blob for t in _PERMANENT):
        return NotConfigured(str(exc))
    if any(t in blob for t in _TRANSIENT):
        return RateLimited(str(exc), _retry_after(exc) or 60.0)
    return exc


# ══════════════════════════════════════════════════════════════════════════
# The client
# ══════════════════════════════════════════════════════════════════════════

class NimClient:
    """One process-wide NIM client: one bucket, one semaphore, one policy.

    Deliberately a singleton via `client()`. Two clients would mean two token
    buckets against one account-wide quota, which is the same as having none —
    the narrative pass and GraphRAG would each stay under the limit
    individually and blow through it together.

    Credentials are read from the environment at construction and never stored,
    logged, or written anywhere. This repository is public and once shipped a
    live token as a default value; there are no fallback literals here.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._client = None
        self._bucket = _Bucket(RPM)
        self._gate = threading.Semaphore(max(1, CONCURRENCY))
        self._models: list = []
        self._preflighted = False
        self._permanent = ""          # set once, stops further attempts
        self.stats = {"calls": 0, "ok": 0, "deferred": 0, "retries": 0,
                      "failed": 0, "waited": 0.0, "tokens_in": 0,
                      "tokens_out": 0, "last_error": "", "last_limit_at": 0.0}

    # ── configuration ────────────────────────────────────────────────────
    @staticmethod
    def _key() -> str:
        return (os.environ.get("VIOS_NIM_API_KEY", "")
                or os.environ.get("NVIDIA_API_KEY", "")).strip()

    @staticmethod
    def base_url() -> str:
        return (os.environ.get("VIOS_NIM_BASE_URL", "").strip()
                or "https://integrate.api.nvidia.com/v1")

    @staticmethod
    def model() -> str:
        """The model to call.

        The default is a mid-size model rather than the 550B the v1 config
        points at. On a forty-per-minute, credit-metered account the largest
        model in the catalogue is affordable for a curated subset and ruinous
        for an archive, and this pass runs on every video.
        """
        return (os.environ.get("VIOS_NIM_MODEL", "").strip()
                or "nvidia/llama-3.1-nemotron-70b-instruct")

    def configured(self) -> bool:
        return bool(self._key())

    def _openai(self):
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                key = self._key()
                if not key:
                    raise NotConfigured(
                        "VIOS_NIM_API_KEY is not set — add it to Kaggle "
                        "Secrets; it is never read from a file or a notebook")
                try:
                    from openai import OpenAI  # noqa: PLC0415
                except Exception as exc:       # noqa: BLE001
                    raise NotConfigured(
                        f"the openai package is required for NIM: {exc}"
                    ) from None
                self._client = OpenAI(base_url=self.base_url(), api_key=key,
                                      timeout=REQUEST_TIMEOUT, max_retries=0)
        return self._client

    # ── preflight ────────────────────────────────────────────────────────
    def preflight(self, log=None) -> dict:
        """Ask the key what it can actually reach, once per session.

        Worth a request because the alternative is discovering a renamed model
        as a 404 on video one of five thousand, indistinguishable in the log
        from a network blip. Failure to list is never fatal: the model may be
        reachable even when listing is not.
        """
        say = _sayer(log)
        if self._preflighted:
            return {"ok": not self._permanent, "models": self._models,
                    "model": self.model(), "reason": self._permanent}
        self._preflighted = True
        want = self.model()
        if not self.configured():
            self._permanent = "no NIM key"
            return {"ok": False, "models": [], "model": want,
                    "reason": self._permanent}
        try:
            listing = self._openai().models.list()
            self._models = sorted(
                str(getattr(m, "id", "")) for m in getattr(listing, "data", [])
                if getattr(m, "id", ""))
        except Exception as exc:                # noqa: BLE001
            say(f"NIM preflight could not list models "
                f"({type(exc).__name__}: {str(exc)[:160]}) — trying {want} "
                f"anyway", "warn")
            return {"ok": True, "models": [], "model": want, "reason": ""}

        say(f"NIM reachable: {len(self._models)} models, calling {want} "
            f"at {RPM:.0f} req/min")
        if self._models and want not in self._models:
            stem = want.split("/")[-1].split("-")[0]
            near = [m for m in self._models if stem in m][:6]
            say(f"NIM model {want} is not in this key's catalogue. "
                f"Set VIOS_NIM_MODEL to one of: "
                f"{', '.join(near or self._models[:8])}", "error")
            self._permanent = f"{want} is not reachable with this key"
            return {"ok": False, "models": self._models, "model": want,
                    "reason": self._permanent}
        return {"ok": True, "models": self._models, "model": want,
                "reason": ""}

    # ── the one call everything goes through ─────────────────────────────
    def chat(self, messages: list, *, temperature: float = 0.2,
             max_tokens: int = 2048, model: str = "", log=None) -> str:
        """One completion, rate-limited and retried. Raises ours, not theirs.

        Raises `RateLimited` when the quota is the obstacle — the caller is
        expected to defer, not to fail — and `NotConfigured` when retrying can
        never help.
        """
        say = _sayer(log)
        if self._permanent:
            raise NotConfigured(self._permanent)

        waited = self._bucket.take(MAX_WAIT)
        if waited < 0:
            self.stats["deferred"] += 1
            self.stats["last_limit_at"] = time.time()
            raise RateLimited(
                f"no request budget within {MAX_WAIT:.0f}s at "
                f"{RPM:.0f} req/min", retry_after=90.0)
        self.stats["waited"] = round(self.stats["waited"] + waited, 1)

        last: Exception | None = None
        for attempt in range(max(1, RETRIES)):
            with self._gate:
                try:
                    self.stats["calls"] += 1
                    resp = self._openai().chat.completions.create(
                        model=model or self.model(), messages=messages,
                        temperature=temperature, max_tokens=max_tokens)
                except Exception as exc:        # noqa: BLE001
                    last = classify(exc)
                else:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        self.stats["tokens_in"] += int(
                            getattr(usage, "prompt_tokens", 0) or 0)
                        self.stats["tokens_out"] += int(
                            getattr(usage, "completion_tokens", 0) or 0)
                    choices = getattr(resp, "choices", None) or []
                    if not choices:
                        self.stats["failed"] += 1
                        raise RuntimeError("NIM returned no choices")
                    self.stats["ok"] += 1
                    return choices[0].message.content or ""

            self.stats["last_error"] = f"{type(last).__name__}: {last}"[:300]
            if isinstance(last, NotConfigured):
                # Permanent for the session — latch it so five thousand videos
                # do not each rediscover the same dead key. This is the *only*
                # thing that latches; a transient never does.
                self._permanent = str(last)[:200]
                say(f"NIM is unusable for this session: {str(last)[:200]}",
                    "error")
                raise last
            if not isinstance(last, RateLimited):
                self.stats["failed"] += 1
                raise last
            self.stats["last_limit_at"] = time.time()
            if attempt + 1 >= max(1, RETRIES):
                break
            pause = max(getattr(last, "retry_after", 0.0),
                        BACKOFF[min(attempt, len(BACKOFF) - 1)])
            self.stats["retries"] += 1
            say(f"NIM refused ({str(last)[:120]}) — retrying in {pause:.0f}s",
                "warn")
            time.sleep(pause)

        self.stats["deferred"] += 1
        raise RateLimited(str(last or "rate limited"),
                          retry_after=max(getattr(last, "retry_after", 0.0),
                                          120.0))

    def status(self) -> dict:
        """What the Process tab shows: budget, usage, and the last refusal."""
        out = {
            "configured": self.configured(),
            "model": self.model(),
            "base_url": self.base_url(),
            "rpm": RPM,
            "concurrency": CONCURRENCY,
            "budget_now": round(self._bucket.available(), 1),
            "models_seen": len(self._models),
            "permanent_error": self._permanent,
        }
        out.update(self.stats)
        return out


_CLIENT: "NimClient | None" = None
_CLIENT_LOCK = threading.Lock()


def client() -> NimClient:
    """The one client. See `NimClient` for why there is exactly one."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = NimClient()
    return _CLIENT


# ══════════════════════════════════════════════════════════════════════════
# narrate-cloud
# ══════════════════════════════════════════════════════════════════════════

_PROMPT = """You are analysing one short vertical video for a research archive \
that studies why short-form video holds attention.

Everything below was measured by other systems from the video itself — the \
speech by two independent transcribers, the on-screen text by two independent \
readers, the shot boundaries and the editing statistics by signal analysis. \
Treat all of it as ground truth. Your job is to explain what these \
measurements add up to, not to guess at what the video might contain.

DURATION: {duration:.1f}s across {n_shots} shots
SHOTS (index, length, what is visible):
{shots}

WHAT IS SAID:
{transcript}

WHAT IS WRITTEN ON SCREEN (in order of appearance):
{on_screen}

THE CREATOR'S CAPTION:
{caption}

WHAT OBJECTS AND SOUNDS WERE MEASURED:
{signals}

EDITING: average shot {asl}s, {cut_rate} cuts per minute, rhythm {rhythm}

Rules you must follow:
- Answer in shot numbers only. Never write a timestamp: you cannot see a clock \
and any time you wrote would be invented. Every time in this archive comes from \
the video container, not from a model.
- Only use shot indices that appear in the list above.
- If the evidence does not support a claim, say so rather than filling the \
field. An empty field is data; a plausible invention is contamination.
- Be specific to THIS video. A sentence that would be true of any video in the \
genre is worthless here.

Return this JSON and nothing else:
{{"premise": "<what this video is, one sentence>",
 "hook": {{"shots": [first, last], "what": "<what the opening does>", \
"why": "<why it stops a scroll, tied to something measured above>"}},
 "beats": [{{"shots": [first, last], "what": "<what happens in this stretch>"}}],
 "turn": {{"shots": [first, last], "what": "<the moment the video changes \
direction, or null if it does not>"}},
 "payoff": {{"shots": [first, last], "what": "<what the viewer is left with>"}},
 "why_it_works": ["<a specific mechanism, each tied to evidence above>"],
 "audience": "<who this is for, and what they already believe>",
 "subject": "<the topic in three words or fewer>",
 "claims_made": ["<a factual assertion the video makes, verbatim where \
possible>"],
 "questions_answered": ["<a question a viewer could search for that this \
video answers>"],
 "weakness": "<the weakest part, honestly>"}}"""


def _signal_summary(job: Job) -> str:
    """The measured non-narrative evidence, compacted into a few lines.

    Objects, sounds and dominant colours come from `detect`, `audio-tag` and
    `colour` as per-frame runs. Feeding the model what the other passes already
    established is the difference between an interpretation grounded in the
    video and a plausible essay about the genre — and it is nearly free, since
    the rows exist.
    """
    parts = []
    for channel, kind, label in (("visual", "object", "objects seen"),
                                 ("audio", "sound_event", "sounds heard"),
                                 ("style", "dominant_colour", "colour"),
                                 ("style", "camera_move", "camera"),
                                 ("visual", "face_scale", "faces")):
        seen: dict = {}
        for row in job.claims(channel, kind):
            value = row.get("value")
            if not value:
                continue
            lo = int(row.get("frame_idx") or 0)
            hi = int(row.get("frame_hi") or lo)
            seen[str(value)] = seen.get(str(value), 0) + max(1, hi - lo + 1)
        if seen:
            top = sorted(seen, key=lambda v: -seen[v])[:8]
            parts.append(f"{label}: {', '.join(top)}")
    return "\n".join(parts) or "(no signal-level measurements)"


def narrate_cloud(job: Job) -> Emission:
    """The deep reading, on NVIDIA's hardware instead of Kaggle's.

    Same job as `narrate-deep`, one API call per video. It writes under its own
    observer id, so its claims sit *beside* the local 8B model's rather than
    replacing them — two independent readings of the same evidence, and where
    they disagree that disagreement is itself recorded.

    Failure modes are deliberately distinct. No key or a wrong model name is a
    `SkipPass`: the coverage matrix should say "declined, here is why" rather
    than accumulating retries against a condition no retry fixes. A rate limit
    is a `DeferPass`: the row returns to the queue with a clock on it and no
    attempt spent, because being out of requests this minute is not a failure.
    """
    from .language import _evidence, _parse_json, _as_list, _shot_span  # noqa: PLC0415

    nim = client()
    if not nim.configured():
        raise SkipPass("no NIM key — set VIOS_NIM_API_KEY in Kaggle Secrets")
    flight = nim.preflight(job.log)
    if not flight.get("ok"):
        raise SkipPass(flight.get("reason") or "NIM is not reachable")

    ev = _evidence(job)
    if not ev["shots"]:
        raise SkipPass("no shots")
    if not (ev["transcript"] or ev["on_screen"] or ev["shots"]):
        raise SkipPass("nothing measured to interpret")

    prompt = _PROMPT.format(
        duration=ev["duration"] or 0.0,
        n_shots=len(ev["shot_ids"]),
        shots=ev["shots"],
        transcript=ev["transcript"] or "(nothing said)",
        on_screen=ev["on_screen"] or "(no on-screen text)",
        caption=ev["caption"] or "(no caption)",
        signals=_signal_summary(job),
        asl=round(ev["asl"] or 0, 2),
        cut_rate=round(ev["cut_rate"] or 0, 1),
        rhythm=ev["rhythm"] or "unknown")

    job.heartbeat(f"asking {nim.model()}")
    t0 = time.time()
    try:
        reply = nim.chat(
            [{"role": "system",
              "content": "You return only valid JSON. No prose, no fences."},
             {"role": "user", "content": prompt}],
            temperature=float(job.params.get("temperature", 0.2)),
            max_tokens=int(job.params.get("max_tokens", 2048)),
            log=job.log)
    except RateLimited as exc:
        raise DeferPass(f"NIM rate limit: {str(exc)[:160]}",
                        retry_after=getattr(exc, "retry_after", 120.0)) from None
    except NotConfigured as exc:
        raise SkipPass(f"NIM unusable: {str(exc)[:160]}") from None
    except Exception as exc:
        # Deliberately not a SkipPass. Skip is terminal, and this branch is
        # reached only by something `classify` did not recognise — which makes
        # it far more likely to be a transient nobody has named yet than a
        # permanent condition. Letting it out gives the row the engine's normal
        # treatment: an attempt spent, a backoff, and a revival later. The
        # video is retried; it is not written off on one unexplained error.
        raise RuntimeError(f"NIM call failed: {type(exc).__name__}: "
                           f"{str(exc)[:200]}") from exc

    data = _parse_json(reply)
    if not isinstance(data, dict):
        raise SkipPass("the model did not return usable JSON")

    allowed = ev["shot_ids"]
    em, wrote = Emission(), 0

    def text_claim(kind: str, value, *, shot_idx=None, confidence=0.75,
                   ordinal=None) -> int:
        if not isinstance(value, str) or not value.strip():
            return 0
        em.claim("narrative", kind, value.strip(), shot_idx=shot_idx,
                 confidence=confidence, ordinal=ordinal)
        return 1

    wrote += text_claim("premise", data.get("premise"), confidence=0.8)
    wrote += text_claim("audience", data.get("audience"), confidence=0.7)
    wrote += text_claim("subject", data.get("subject"), confidence=0.75)
    wrote += text_claim("critique", data.get("weakness"), confidence=0.65)

    for key, kind in (("hook", "hook"), ("turn", "turn"),
                      ("payoff", "payoff")):
        entry = data.get(key)
        if not isinstance(entry, dict):
            continue
        first, _last = _shot_span(entry, allowed)
        wrote += text_claim(kind, entry.get("what"), shot_idx=first,
                            confidence=0.75)
        if kind == "hook":
            wrote += text_claim("hook_why", entry.get("why"), shot_idx=first,
                                confidence=0.7)

    for n, entry in enumerate(_as_list(data.get("beats"))):
        if not isinstance(entry, dict):
            continue
        first, _last = _shot_span(entry, allowed)
        wrote += text_claim("beat", entry.get("what"), shot_idx=first,
                            confidence=0.7, ordinal=n)

    # The three list fields that future features read directly: mechanisms for
    # pattern identification, assertions for fact-checking across the library,
    # and questions for the "find the answer in my videos" path.
    for kind, field, base in (("why_it_works", "why_it_works", 100),
                              ("assertion", "claims_made", 200),
                              ("answers", "questions_answered", 300)):
        for n, item in enumerate(_as_list(data.get(field))):
            wrote += text_claim(kind, item, confidence=0.65, ordinal=base + n)

    if not wrote:
        raise SkipPass("the reply contained no usable claims")

    em.notes = {"claims": wrote, "model": nim.model(),
                "seconds": round(time.time() - t0, 1),
                "calls": 1, "shots": len(allowed)}
    return em
