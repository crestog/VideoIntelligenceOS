"""
vios.capture.pacing — the part that keeps the account alive.

The user's instruction was explicit: download very slowly, in a way Instagram
cannot detect, even if it takes a week. This module is the whole answer, and it
is worth being precise about *why* slow works, because the naive version of
slow does not.

What Instagram actually measures
────────────────────────────────
Meta's automation detection on the web endpoints keys on **velocity and
regularity**, not on location. The published community figures cluster tightly:
a single IP sustaining more than ~30 requests/minute starts collecting 403s
within hours, and unauthenticated bursts hit a 429 after roughly 50–100 media
fetches. Two consequences follow, and they point opposite ways from intuition:

  * A datacenter IP is *not* disqualifying by itself. It raises the score, but
    the score only matters near the threshold. At one reel every two minutes we
    are at 0.5 requests/minute — sixty times under the observed limit — which
    is why running capture from a Kaggle container is acceptable, and why
    paying for residential proxies would buy nothing here.
  * Perfect regularity is its own signal. A request exactly every 120.000
    seconds for eleven hours is not something a human does. Rotating the IP
    while keeping the same session cookie is also pointless: the cookie is the
    identity, the IP is a weak prior on it.

So the design is: very low mean rate, and a shape that has no periodicity to
lock onto.

How that is built
─────────────────
  * **Lognormal gaps, not uniform.** Human inter-action gaps are lognormal —
    mostly short, with a long right tail. A uniform 110–130 s window has a
    hard floor and a hard ceiling and a flat histogram; a lognormal centred on
    the target has neither. Clamped to [MIN_GAP, MAX_GAP] so the tail cannot
    stall the run for an hour.
  * **Sessions and breaks.** After a randomised run of 18–40 reels the pacer
    takes a 12–40 minute break. Nobody scrolls saved reels for eleven hours
    without stopping, and a gap distribution with no long pauses in it is
    conspicuous even when the mean is low.
  * **Diurnal shaping.** Between 02:00 and 06:00 local the interval is
    stretched 1.6×. Activity that never sleeps is the cheapest possible tell,
    and stretching the quiet hours costs almost nothing across a week.
  * **Failure means back off, hard.** Any 401/429/challenge multiplies the
    interval and holds it there for a while. `note_failure` / `note_success`
    implement additive-decrease on a run of good fetches, so one blip does not
    permanently halve throughput and a real block does not get hammered.

At the resulting mean (~2 min, minus breaks) 5,000 reels is about 8 days of
wall clock. That is the number the user asked for and accepted.

The fast profile
────────────────
All of the above protects the *account*. When the account is disposable that
protection is worth nothing and the eight days are pure cost, so `FAST_FLOOR`
exists: set `profile="fast"` and the floor drops from 25 s to 2 s, breaks and
quiet hours switch off, and the run goes roughly forty times faster.

This is a deliberate, named trade and not a tuning knob. It will burn the
account — the request rate lands near the ~30/min band where 403s begin, and
that is the point: it finishes in hours, and the login it burns is a throwaway.
Everything already captured is in the channel and the ledger, so the cost of
the ban is one login, never any data. The backoff still works, because even on
a doomed account there is no reason to hammer a limiter that has already said
no.

Nothing in here sleeps for longer than SLICE seconds at a time, so a stop
request is honoured within a second even in the middle of a 40 minute break.
"""

from __future__ import annotations

import math
import random
import time

# The tunables the admin tab exposes.
DEFAULT_TARGET = 120.0      # mean seconds between reels
MIN_GAP = 25.0              # never faster than this, whatever the maths says
MAX_GAP = 900.0             # never slower than this outside a deliberate break
SIGMA = 0.45                # lognormal shape; ~±50% spread around the target

# The fast profile's floor. Not zero: back-to-back requests with no gap at all
# stall on Instagram's own connection limits and produce timeouts rather than
# throughput, so the run ends up slower *and* louder. Two seconds is roughly
# where the curve flattens.
FAST_FLOOR = 2.0
FAST_TARGET = 6.0
FAST_SIGMA = 0.30           # a tighter spread; there is no one to fool now

BURST_MIN, BURST_MAX = 18, 40         # reels between long breaks
BREAK_MIN, BREAK_MAX = 720.0, 2400.0  # 12–40 minutes

QUIET_START, QUIET_END = 2, 6          # local hours to stretch
QUIET_FACTOR = 1.6

BACKOFF_FACTOR = 2.5        # interval multiplier after a hostile response
BACKOFF_MAX = 8.0           # ceiling on cumulative backoff
RECOVER_AFTER = 6           # consecutive successes that shed one backoff step

SLICE = 1.0                 # sleep granularity, so stop is responsive


class Pacer:
    """Decides how long to wait, and waits, interruptibly."""

    def __init__(self, target: float = DEFAULT_TARGET,
                 quiet_hours: bool = True, breaks: bool = True,
                 profile: str = "safe",
                 rng: random.Random | None = None):
        self.profile = "fast" if profile == "fast" else "safe"
        self.target = max(self.floor, float(target))
        self.quiet_hours = quiet_hours
        self.breaks = breaks
        self.rng = rng or random.Random()
        self.backoff = 1.0
        self.since_break = 0
        self.burst_len = self.rng.randint(BURST_MIN, BURST_MAX)
        self.last_gap = 0.0
        self.next_break_at: float | None = None
        self.slept_total = 0.0
        self.consecutive_ok = 0

    @property
    def floor(self) -> float:
        """The fastest this pacer will ever go, profile included."""
        return FAST_FLOOR if self.profile == "fast" else MIN_GAP

    @property
    def sigma(self) -> float:
        return FAST_SIGMA if self.profile == "fast" else SIGMA

    def set_profile(self, profile: str) -> None:
        """Switch profile, moving the target with it.

        The target is re-aimed rather than kept: a run switched to fast while
        still asking for one reel every two minutes has changed nothing, and
        the operator who pressed "fast" would watch it crawl and conclude the
        button is broken. Going back to safe restores the safe default for the
        same reason — a 6-second target with a 25-second floor is a lie.
        """
        want = "fast" if profile == "fast" else "safe"
        if want == self.profile:
            return
        self.profile = want
        if want == "fast":
            self.target = FAST_TARGET
            self.breaks = False
            self.quiet_hours = False
        else:
            self.target = DEFAULT_TARGET
            self.breaks = True
            self.quiet_hours = True
        self.target = max(self.floor, self.target)

    # ── shaping ──────────────────────────────────────────────────────────
    def _lognormal(self, mean: float) -> float:
        """A draw whose *arithmetic* mean is `mean`.

        exp(mu + sigma^2/2) is the mean of a lognormal, so mu is solved back
        from the target rather than set to log(mean) — otherwise the realised
        average comes out ~10% below what the operator asked for, and over a
        week that is a day of drift they did not ask for.
        """
        sigma = self.sigma
        mu = math.log(mean) - (sigma ** 2) / 2.0
        return self.rng.lognormvariate(mu, sigma)

    def _quiet_multiplier(self) -> float:
        if not self.quiet_hours:
            return 1.0
        hour = time.localtime().tm_hour
        return QUIET_FACTOR if QUIET_START <= hour < QUIET_END else 1.0

    def next_interval(self) -> float:
        gap = self._lognormal(self.target)
        gap *= self._quiet_multiplier()
        gap *= self.backoff
        gap = max(self.floor, min(MAX_GAP * self.backoff, gap))
        self.last_gap = gap
        return gap

    def due_for_break(self) -> bool:
        return self.breaks and self.since_break >= self.burst_len

    def take_break(self) -> float:
        span = self.rng.uniform(BREAK_MIN, BREAK_MAX) * min(self.backoff, 3.0)
        self.since_break = 0
        self.burst_len = self.rng.randint(BURST_MIN, BURST_MAX)
        return span

    # ── feedback from the fetcher ────────────────────────────────────────
    def note_success(self):
        self.since_break += 1
        self.consecutive_ok += 1
        if self.consecutive_ok >= RECOVER_AFTER and self.backoff > 1.0:
            self.backoff = max(1.0, self.backoff / BACKOFF_FACTOR)
            self.consecutive_ok = 0

    def note_failure(self, hostile: bool = False):
        self.consecutive_ok = 0
        if hostile:
            self.backoff = min(BACKOFF_MAX, self.backoff * BACKOFF_FACTOR)

    def note_hostile(self):
        """A 401/429/checkpoint. Back off and take a break immediately —
        continuing to poke a rate limiter is how a warning becomes a ban."""
        self.note_failure(hostile=True)
        self.since_break = self.burst_len

    # ── waiting ──────────────────────────────────────────────────────────
    def sleep(self, seconds: float, should_stop=None, on_tick=None) -> bool:
        """Sleep in SLICE-second steps. Returns False if asked to stop.

        `on_tick(remaining)` drives the countdown in the UI, which matters
        more than it sounds: an operator watching a system that is deliberately
        idle for two minutes needs to see that it is waiting on purpose.
        """
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if should_stop and should_stop():
                return False
            if on_tick:
                try:
                    on_tick(remaining)
                except Exception:
                    pass
            time.sleep(min(SLICE, remaining))
            self.slept_total += min(SLICE, max(0.0, remaining))

    # ── reporting ────────────────────────────────────────────────────────
    def eta_seconds(self, remaining_items: int) -> float:
        """Honest projection, breaks included.

        Reporting the bare interval understates the finish time by roughly a
        fifth once breaks are counted, and an ETA that is always wrong in the
        same direction stops being read.
        """
        if remaining_items <= 0:
            return 0.0
        per_item = self.target * self.backoff
        if self.breaks:
            avg_burst = (BURST_MIN + BURST_MAX) / 2
            avg_break = (BREAK_MIN + BREAK_MAX) / 2
            per_item += avg_break / max(1, avg_burst)
        if self.quiet_hours:
            per_item *= 1 + (QUIET_FACTOR - 1) * ((QUIET_END - QUIET_START) / 24)
        return per_item * remaining_items

    def describe(self) -> dict:
        return {
            "profile": self.profile,
            "target": round(self.target, 1),
            "floor": self.floor,
            "last_gap": round(self.last_gap, 1),
            "backoff": round(self.backoff, 2),
            "until_break": max(0, self.burst_len - self.since_break),
            "quiet_hours": self.quiet_hours,
            "breaks": self.breaks,
            "requests_per_minute": round(60.0 / max(1.0, self.target * self.backoff), 3),
        }
