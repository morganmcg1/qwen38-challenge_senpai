#!/usr/bin/env python3
"""E209 stage 0: the PRE-REGISTERED online depth-controller family.

This module holds the controller definitions only. It is committed BEFORE the
replay is run, so the family is registered rather than selected. The replay and
the statistics live in `research/e209_desk.py`.

WHAT E209 ASKS. FINDING 540 measured a hindsight ceiling for the within-prompt
depth-policy family on the E168 `p7` corpus: the per-prompt best FIXED depth
beats the shipped live-EMA rule by +2.589% prose-population median, and holding
the shipped rule's OWN average level while removing only its round-to-round
variation captures +2.460 of that +2.589. So ~95% of the accessible headroom is
variance removal, not level selection. FINDING 538 separately showed the level
tracking is worth ~0.93%, so a controller may not simply freeze at the prior.

The family below therefore keeps the level ONLINE and stops the level MOVING.

CAUSALITY. Every controller here sees only what a live session sees:

  * the EMA vector as updated by `recordAcceptOutcome` from the rounds it
    actually played, at the depths it actually played;
  * the current round's `pendingTop2` margin, if it chooses to use it;
  * its own history of played depths and observed accept counts.

No controller reads the realized accept length of the current round, and no
controller reads a position deeper than the depth it played. A round played at
depth d observes `accepted = min(k, d)` and nothing more, so a controller that
holds a shallow level pays a real information cost. The replay prices that.

LEGALITY. Every signal is within-request, the same legality class as the
shipped EMA. No prompt detection, no benchmark-phase behaviour, no
cross-request state, no reference-row access.

--------------------------------------------------------------------- FAMILY

Three registered families from the assignment, plus ONE controller of my own
design, plus one clearly-labelled DIAGNOSTIC that is NOT promotion-eligible.

  A1-W  settle-then-freeze, EMA-implied level
        Play the shipped rule for the first W rounds. At round W read the level
        the settled EMA implies with the round-level margin removed,
        `costModelDepth(ema, margin=nil, cap)`, and hold it for the rest of the
        leg. The EMA keeps updating; the held level does not. W in {16, 32, 64}.

  A2-W  settle-then-freeze, modal level
        As A1, but the held level is the mode of the last 8 depths played in
        the settle window. Ties break toward the SMALLER depth. W in {16,32,64}.

  B-h   hysteresis / deadband on the level preference
        No settle window. Hold a current level C, initialised from the seed
        prior with no margin. Each round compute the level the EMA now prefers,
        `costModelDepth(ema, margin=nil, cap)`. Move C to that preference only
        after it persists for h consecutive rounds. h in {4, 8, 16}.

  C-p   sticky value estimator with an explicit switching penalty
        Each round score every depth with the shipped ABSTRACT price,
        `V(d) = (1 + 0.18 d) / (1 + E[min(k,d)])`, where the expected accepted
        run uses the current EMA as independent per-position accept rates,
        `E[min(k,d)] = sum_{j<d} prod_{i<=j} ema[i]`. Switch to `argmin V` only
        when it beats the held level by a relative margin p. p in {0.005, 0.02}.
        This is the assignment's tau-margin deadband variant.

  OWN   running-mean level lock (my one own-design controller)
        Motivated directly by the FINDING 540 decomposition. The variance-only
        row of that decomposition holds the shipped rule's OWN AVERAGE level,
        so the estimator that targets the measured headroom is the running mean
        of the shipped rule's own depth choices, not a re-derived level. Play
        the shipped rule for W rounds, lock at `floor(mean(depths) + 0.5)`,
        hold it for the rest of the leg. W in {16, 32, 64}.
        This differs from A2 in the estimator (mean of the whole window against
        mode of the last 8) and from A1 in the source (the rule's realised
        choices against the EMA's re-derived preference).

  DIAG-p  measured-cost-table value estimator     NOT PROMOTION-ELIGIBLE
        C-p with the shipped abstract price replaced by the MEASURED E168
        round-cost table. It is scored against that same table, so it is
        CIRCULAR and cannot support a promotion claim. It is reported only to
        separate "how much of the headroom is level selection given a correct
        cost model" from "how much is variance removal". p in {0.02}.

--------------------------------------------------------------- HYPERPARAMETERS

15 configurations over 8 prose prompts. That is enough freedom to overfit, so
`e209_desk.py` reports the WHOLE table, never a selected row alone, plus the
minimum per-prompt delta and a leave-one-prompt-out check in which the
configuration is chosen on 7 prompts and scored on the 8th.

All numbers are harness=local desk replay at cap 7 (RULE 392: the corpus pins
depth 7, so `accepted = min(k, d)` is exact for every d <= 7). Cap 8 is not
evaluated; no leg ever measured a width-9 cost cell.
"""

from __future__ import annotations

import collections
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e207_desk as D  # noqa: E402


def expected_accepted(ema, depth):
    """E[min(k, d)] under the EMA read as independent per-position rates."""
    total, reach = 0.0, 1.0
    for position in range(depth):
        reach *= ema[position]
        total += reach
    return total


class Shipped:
    """The live rule on the candidate surface today: the comparison baseline."""

    name = "shipped"
    kind = "baseline"

    def choose(self, ema, margin, cap, _round_index):
        return D.depth_walk(ema, margin, cap)


class SettleThenFreeze:
    """A1 / A2 / OWN. One settle window, one estimator, then a held level."""

    kind = "settle-then-freeze"

    def __init__(self, window, estimator):
        self.window = window
        self.estimator = estimator
        self.name = "%s-W%d" % (estimator, window)
        self.history = []
        self.locked = None

    def choose(self, ema, margin, cap, round_index):
        if self.locked is not None:
            return self.locked
        if round_index < self.window:
            depth = D.depth_walk(ema, margin, cap)
            self.history.append(depth)
            return depth
        self.locked = self._lock(ema)
        return self.locked

    def _lock(self, ema):
        if self.estimator == "A1-ema":
            return D.depth_walk(ema, None, D.MAX_DEPTH - 1)
        if self.estimator == "A2-mode":
            tail = self.history[-8:]
            counts = collections.Counter(tail)
            best = max(counts.values())
            return min(d for d, n in counts.items() if n == best)
        if self.estimator == "OWN-mean":
            mean = sum(self.history) / len(self.history)
            return int(mean + 0.5)
        raise ValueError(self.estimator)


class Hysteresis:
    """B-h. Move the level only after the preference persists h rounds."""

    kind = "hysteresis"

    def __init__(self, persistence, cap):
        self.persistence = persistence
        self.cap = cap
        self.name = "B-h%d" % persistence
        self.level = D.depth_walk(D.PRIOR, None, cap)
        self.candidate = None
        self.streak = 0

    def choose(self, ema, _margin, cap, _round_index):
        preferred = D.depth_walk(ema, None, self.cap)
        if preferred == self.level:
            self.candidate, self.streak = None, 0
        elif preferred == self.candidate:
            self.streak += 1
            if self.streak >= self.persistence:
                self.level = preferred
                self.candidate, self.streak = None, 0
        else:
            self.candidate, self.streak = preferred, 1
        return min(self.level, cap)


class StickyValue:
    """C-p and DIAG-p. Switch only on a relative value margin."""

    kind = "sticky-value"

    def __init__(self, penalty, price, label, cap):
        self.penalty = penalty
        self.price = price
        self.cap = cap
        self.name = "%s-p%g" % (label, penalty)
        self.level = D.depth_walk(D.PRIOR, None, cap)

    def choose(self, ema, _margin, cap, _round_index):
        values = {d: self.price(d) / (1.0 + expected_accepted(ema, d))
                  for d in range(self.cap + 1)}
        best = min(values, key=values.get)
        if values[best] < (1.0 - self.penalty) * values[self.level]:
            self.level = best
        return min(self.level, cap)


def abstract_price(depth):
    """The shipped cost model's own price: 1 + 0.18 d."""
    return D.PRICE_CUMULATIVE[depth]


def measured_price(depth):
    """The measured E168 round-cost table. Circular against this evaluation."""
    return D.COST_MS[depth + 1]


def registry(cap):
    """Every pre-registered controller, in registration order."""
    specs = []
    for window in (16, 32, 64):
        specs.append(lambda w=window: SettleThenFreeze(w, "A1-ema"))
    for window in (16, 32, 64):
        specs.append(lambda w=window: SettleThenFreeze(w, "A2-mode"))
    for window in (16, 32, 64):
        specs.append(lambda w=window: SettleThenFreeze(w, "OWN-mean"))
    for persistence in (4, 8, 16):
        specs.append(lambda h=persistence: Hysteresis(h, cap))
    for penalty in (0.005, 0.02):
        specs.append(lambda p=penalty: StickyValue(p, abstract_price, "C", cap))
    specs.append(lambda: StickyValue(0.02, measured_price, "DIAG", cap))
    return specs


PROMOTION_INELIGIBLE = {"DIAG-p0.02"}
