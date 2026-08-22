#!/usr/bin/env python3
"""E128 rung 2 -- price depth-policy counterfactuals on the RANKED cost curve.

`harness=ranked` for every number this file prints. Nothing here is a local
timing measurement and nothing here may be compared with a local ratio.

THE MODEL, in one place so it can be attacked:

  1. Round cost. The board-fitted ranked round cost at verify width `M = d + 1`
     is two linear tiers, `G = 1` for M in 1..4 and `G = 2` for M in 5..8
     (finding F97). A round that drafts `d` tokens and has `a` of them accepted
     costs `round_us(d + 1)` and emits `1 + a` tokens.

  2. Acceptance. Rung 1 measures the UNCENSORED per-position conditional
     acceptance `p_j` on a forced-depth leg, so it is not selected by the very
     estimator under test. Each recorded round contributes one (margin,
     accepted-run) pair.

  3. Transfer. A local fixture is not a ranked prompt. Each ranked prompt is
     given the shape of its nearest fixture and a single level parameter
     `delta`, applied in logit space, fitted so that the SHIPPED policy
     reproduces that prompt's realised mean draft depth (F92). The realised
     accept rate is then a held-out check on the fit, not an input to it.
     Sensitivity to this choice is reported beside the headline.

  4. Counterfactual. Each round's acceptance capability is resampled through
     its own quantile band, so a round that accepted a long run in the fixture
     stays a strong round after transfer, and its margin travels with it. A
     policy that drafts `d` on that round accepts `min(capability, d)`.

  5. Score. Arm cost per token is divided by the shipped arm's cost per token
     to give a per-prompt ratio, that ratio is applied to the receipt's
     measured candidate seconds per token, `raw_p` is recomputed against the
     receipt's own serial numerator, and the published median is recomputed
     exactly over all eight prompts (Rule 67). The F83 weights are reported
     for interpretation; they are never used as a linear substitute for the
     median.

  usage: research/e128_price.py --accept ACCEPT.json [--board /tmp/...] \
             [--json OUT] [--replicates N]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

from e128_replay import (
    EMA_PRIOR, MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP,
    cost_model_depth, record_accept_outcome,
)

# ---------------------------------------------------------------- ranked data

# The ranked round cost `round_us(M)` at verify width `M = d + 1`.
#
# The default is F97, fitted over 147 official runs of other solvers. Route B
# changed our candidate and did not change theirs, so that curve is the board's
# and no longer ours. `--curve-json` swaps in the curve `e128_ourcurve.py` fits
# from our own receipts, which moves the tier step from M >= 5 to M >= 6 and
# lowers both marginal slopes.
CURVE = {
    "name": "board_f97",
    "breakpoint": 5,
    "lo": (27215.4, 3966.4),
    "hi": (17020.7, 7154.2),
}


def load_curve(path: Path, key: str | None) -> None:
    data = json.loads(path.read_text())
    if key is not None:
        data = data["curves"][key]
    CURVE.update({
        "name": data.get("name", "%s:%s" % (path.name, key)),
        "breakpoint": int(data["breakpoint"]),
        "lo": tuple(data["lo"]),
        "hi": tuple(data["hi"]),
    })

# F92 realised behaviour per ranked prompt, and F83 marginal weights.
#
# `fixture` names the E124 prose fixtures written to imitate that ranked
# prompt. The two-fixture prompts are POOLED rather than picked, because
# picking the fixture whose realised depth is nearest the ranked prompt would
# fit the transfer on the quantity the transfer is supposed to predict. Single
# fixtures are priced again as a sensitivity variant.
RANKED_PROMPTS = {
    "beagle":   {"weight": 0.4862, "depth": 4.382, "accept": 0.834,
                 "fixture": ["beagle_a", "beagle_b"]},
    "medicine": {"weight": 0.2508, "depth": 5.256, "accept": 0.892,
                 "fixture": ["medicine_hist", "medicine_hippoc"]},
    "essays":   {"weight": 0.1598, "depth": 5.087, "accept": 0.897,
                 "fixture": ["essays_bacon", "essays_montaigne"]},
    "botany":   {"weight": 0.0124, "depth": 6.148, "accept": 0.865,
                 "fixture": ["botany_andrews"]},
    "republic": {"weight": 0.0100, "depth": 4.989, "accept": 0.903,
                 "fixture": ["republic_jowett"]},
    "drama":    {"weight": 0.0000, "depth": 2.298, "accept": 0.449,
                 "fixture": ["drama_dollhouse"]},
    "travel":   {"weight": 0.0000, "depth": 2.656, "accept": 0.533,
                 "fixture": ["travel_eothen"]},
    "plutarch": {"weight": 0.0000, "depth": 0.154, "accept": 0.333,
                 "fixture": ["plutarch_lives"]},
}

PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

DECODE_TOKENS = 512


def ranked_round_us(rows: int) -> float:
    intercept, slope = CURVE["lo"] if rows < CURVE["breakpoint"] else CURVE["hi"]
    return intercept + slope * rows


def ranked_price_table() -> tuple[list[float], list[float]]:
    """The shipped uniform 0.18 price replaced by the measured ranked curve.

    The walk compares `reach` with `marginal[d] * (1 + expected) / cumulative[d]`,
    so `cumulative[d]` only has to be the cost of a round that drafts `d`
    tokens up to a common factor. Normalising by the `d = 0` cost keeps the
    same units as the shipped table, which makes this a pure two-vector
    substitution rather than a change of the walk.
    """
    base = ranked_round_us(1)
    cumulative = [ranked_round_us(d + 1) / base for d in range(MAX_DEPTH + 1)]
    marginal = [cumulative[d + 1] - cumulative[d] for d in range(MAX_DEPTH)]
    return marginal, cumulative


def expected_accepted(p_vec: list[float], depth: int) -> float:
    """E[accepted] for a chain of independent per-position conditionals."""
    total = 0.0
    running = 1.0
    for j in range(depth):
        running *= p_vec[j]
        total += running
    return total


def static_cost_per_token(p_vec: list[float], depth: int) -> float:
    return ranked_round_us(depth + 1) / (1.0 + expected_accepted(p_vec, depth))


# --------------------------------------------------------------- acceptance

def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def shift_p_vector(p_vec: list[float], delta: float) -> list[float]:
    return [sigmoid(logit(p) + delta) for p in p_vec]


def pooled_positions(legs: list[dict]) -> list[float]:
    """Pools per-position acceptance counts across fixtures for one prompt.

    Counts are pooled, not the ratios, so a fixture that reached position `j`
    on more rounds carries proportionally more weight there.

    A Jeffreys prior (half a success and half a failure) keeps every entry
    strictly inside `(0, 1)`. A raw `0` or `1` has infinite logit, which would
    make the one-parameter transfer either impossible or unbounded, and both
    endpoints occur at the deep positions that only a few rounds reach.
    """
    p_vec = []
    for j in range(MAX_DEPTH):
        reached = accepted = 0
        for leg in legs:
            if j < len(leg["positions"]):
                reached += leg["positions"][j]["reached"]
                accepted += leg["positions"][j]["accepted"]
        if reached == 0:
            p_vec.append(p_vec[-1] if p_vec else 0.5)
        else:
            p_vec.append((accepted + 0.5) / (reached + 1.0))
    return p_vec


def survival(p_vec: list[float]) -> list[float]:
    """`S[k] = P(capability >= k)` for k = 0..len(p_vec)."""
    out = [1.0]
    running = 1.0
    for p in p_vec:
        running *= p
        out.append(running)
    return out


class RoundSampler:
    """Resamples measured rounds and retargets their acceptance capability.

    A recorded round accepted `a` of `d` proposed drafts. That pins the round's
    capability to a band of the fixture's own survival curve: exactly `a` when
    the round rejected at `a`, and at least `d` when it accepted everything
    offered. The round is placed uniformly inside its band, and the same
    quantile is read off the target prompt's survival curve. A round that was
    strong for its fixture stays strong for the prompt, and the margin recorded
    with it travels unchanged.
    """

    def __init__(self, rounds: list[dict], p_fixture: list[float],
                 p_target: list[float], seed: int = 128,
                 shuffle_margins: bool = False, serial: bool = True):
        self.rounds = rounds
        self.s_fix = survival(p_fixture)
        self.s_target = survival(p_target)
        self.rng = random.Random(seed)
        self.margins = [r["margin"] for r in rounds]
        self.shuffle_margins = shuffle_margins
        # Decoding is strongly serially correlated: a leg enters an easy
        # stretch and accepts everything for dozens of consecutive rounds, then
        # leaves it. The EMA is a seven-round window, so it tracks that
        # structure and the shipped depth inherits it. Drawing rounds
        # independently destroys it and pins the EMA near the global mean, so
        # the default walks the recorded order from a random offset and wraps.
        self.serial = serial
        self.cursor = self.rng.randrange(len(rounds))

    def draw(self) -> tuple:
        if self.serial:
            record = self.rounds[self.cursor]
            self.cursor = (self.cursor + 1) % len(self.rounds)
        else:
            record = self.rounds[self.rng.randrange(len(self.rounds))]
        a, d = record["accepted"], record["depth"]
        high = self.s_fix[min(a, len(self.s_fix) - 1)]
        low = 0.0 if a >= d else self.s_fix[min(a + 1, len(self.s_fix) - 1)]
        u = self.rng.uniform(low, high)
        capability = 0
        for k in range(1, len(self.s_target)):
            if u <= self.s_target[k]:
                capability = k
            else:
                break
        margin = record["margin"]
        if self.shuffle_margins:
            margin = self.margins[self.rng.randrange(len(self.margins))]
        return capability, margin


# ------------------------------------------------------------------- policies

def make_policy(arm: str, recal: tuple = (2.0, 3.0), level: dict | None = None):
    """Returns `policy(ema, margin, offer, capability) -> depth`."""
    if arm in LEVEL_ARMS or arm in LEVEL_GRID_ARMS:
        if level is None:
            raise SystemExit("arm %r needs a measured level bias" % arm)
        gamma = level["gamma"]
        gains = level["jensen_gain"]
        if arm.startswith("levelfix") and len(arm) > len("levelfix"):
            gamma = float(arm[len("levelfix"):])
        kwargs = {
            "reachonly": {"reach_gain": gamma},
            "expectedonly": {"expected_gain": gamma},
            "jensen": {"reach_gain_by_step": gains},
            "jensen_both": {"reach_gain_by_step": gains,
                            "expected_gain": sum(gains[:4]) / 4.0},
        }.get(arm, {"reach_gain": gamma, "expected_gain": gamma})
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer, **kwargs)[0]
    if arm == "ship":
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer)[0]
    if arm == "nomargin":
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer,
            margin_scale_0=None, margin_scale_1=None)[0]
    if arm == "nomargin0":
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer, margin_scale_0=None)[0]
    if arm == "nomargin1":
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer, margin_scale_1=None)[0]
    if arm == "recal":
        s0, s1 = recal
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer,
            margin_scale_0=s0, margin_scale_1=s1)[0]
    if arm in ("marginup", "marginfull"):
        # The assignment asks whether the strictly-downward override is what
        # holds depth below the ranked optimum. These two lift that
        # restriction while keeping everything else shipped.
        mode = "max" if arm == "marginup" else "replace"
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer, margin_mode=mode)[0]
    if arm.startswith("rankedprice"):
        marginal, cumulative = ranked_price_table()
        suffix = arm[len("rankedprice"):].lstrip("_")
        s0, s1, mode = 2.0, 3.0, "min"
        if suffix == "nomargin":
            s0 = s1 = None
        elif suffix == "recal":
            s0, s1 = recal
        elif suffix == "marginup":
            mode = "max"
        elif suffix:
            raise SystemExit("unknown rankedprice arm %r" % arm)
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer,
            margin_scale_0=s0, margin_scale_1=s1, margin_mode=mode,
            marginal=marginal, cumulative=cumulative)[0]
    if arm.startswith("price"):
        constant = float(arm[5:])
        marginal = [constant] * MAX_DEPTH
        cumulative = [1.0 + i * constant for i in range(MAX_DEPTH + 1)]
        return lambda ema, m, offer, cap: cost_model_depth(
            ema, m, offered_depth=offer,
            marginal=marginal, cumulative=cumulative)[0]
    if arm == "static7":
        return lambda ema, m, offer, cap: min(
            offer, MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)
    if arm == "oracle":
        def oracle(ema, m, offer, cap):
            cap_depth = min(offer, MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)
            best, best_cost = 0, ranked_round_us(1)
            for depth in range(1, cap_depth + 1):
                cost = ranked_round_us(depth + 1) / (1.0 + min(cap, depth))
                if cost < best_cost:
                    best, best_cost = depth, cost
            return best
        return oracle
    raise SystemExit("unknown arm %r" % arm)


def simulate(policy, sampler: RoundSampler, windows: int,
             tokens: int = DECODE_TOKENS, seed: int = 7) -> dict:
    """Runs whole 512-token decode windows, including the parent's tail offer."""
    total_us = 0.0
    total_tokens = 0
    total_depth = 0
    total_rounds = 0
    total_accepted = 0
    total_drafted = 0
    per_window = []
    for _ in range(windows):
        ema = list(EMA_PRIOR)
        emitted = 0
        window_us = 0.0
        while emitted < tokens:
            remaining = tokens - emitted
            # QwenRuntimeMTPDriver.swift:141-150.
            offer = max(1, min(MAX_DEPTH, remaining - 1))
            capability, margin = sampler.draw()
            depth = policy(ema, margin, offer, capability)
            accepted = min(capability, depth)
            window_us += ranked_round_us(depth + 1)
            emitted += min(1 + accepted, remaining)
            total_depth += depth
            total_rounds += 1
            total_accepted += accepted
            total_drafted += depth
            ema = record_accept_outcome(ema, accepted, depth)
        total_us += window_us
        total_tokens += tokens
        per_window.append(window_us / tokens)
    return {
        "us_per_token": total_us / total_tokens,
        "us_per_token_sem": (
            statistics.stdev(per_window) / math.sqrt(len(per_window))
            if len(per_window) > 1 else 0.0),
        "mean_depth": total_depth / total_rounds,
        "accept_rate": (
            total_accepted / total_drafted if total_drafted else 0.0),
        "rounds_per_window": total_rounds / windows,
    }


# ------------------------------------------------------------------- transfer

def fit_delta(sampler_factory, p_fixture: list[float], target: float,
              windows: int, key: str = "mean_depth") -> tuple:
    """One level parameter, fitted so the SHIPPED arm reproduces one target.

    `key = "mean_depth"` targets the published per-prompt
    `effective_mean_draft_len`, which the board reports directly and which
    therefore carries NO assumption about the ranked round count `R`. That is
    the default and the anchor of the headline result. `key = "accept_rate"`
    targets the R-derived accept rate instead and is used only to report how
    the answer moves across the pinned `R` band.

    The reachable range is bounded from above by the shipped estimator itself:
    the margin override can hold depth down however good acceptance becomes. A
    target outside the reachable range is CLAMPED to the nearest endpoint and
    the residual is reported, because silently dropping the prompt would
    quietly change which eight values the median is taken over.
    """
    policy = make_policy("ship")
    low, high = -8.0, 12.0

    def value_at(delta: float) -> float:
        p_target = shift_p_vector(p_fixture, delta)
        return simulate(policy, sampler_factory(p_target), windows)[key]

    d_low, d_high = value_at(low), value_at(high)
    if target <= d_low:
        return low, d_low, d_high
    if target >= d_high:
        return high, d_low, d_high
    for _ in range(40):
        mid = 0.5 * (low + high)
        if value_at(mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high), d_low, d_high


def validate_simulator(forced: dict, shipped: dict, windows: int,
                       seed: int = 128) -> list[dict]:
    """Held-out check with NO fitted parameter anywhere.

    Feed the simulator the uncensored acceptance of a fixture and run the
    shipped policy on it at `delta = 0`. The answer must reproduce the same
    fixture's separately measured shipped depth and accept rate. Neither
    measured value is used to build the input, so this can fail. If it does,
    no counterfactual arm on the same machinery is believable.
    """
    rows = []
    for name, leg in sorted(forced.items()):
        if name not in shipped:
            continue
        p_vec = pooled_positions([leg])
        sampler = RoundSampler(leg["rounds_detail"], p_vec, p_vec, seed=seed)
        out = simulate(make_policy("ship"), sampler, windows)
        rows.append({
            "fixture": name,
            "measured_depth": shipped[name]["mean_depth"],
            "simulated_depth": out["mean_depth"],
            "measured_accept": shipped[name]["accept_rate"],
            "simulated_accept": out["accept_rate"],
            "forced_uncensored_p": p_vec,
        })
    return rows


def fit_margin_scale(rounds: list[dict], position: int) -> dict:
    """MLE of the shipped one-parameter logistic `sigmoid(margin / s)`.

    The shipped override reads `conf` as a probability of acceptance at this
    position, so the honest recalibration is the scale that makes it one.
    """
    pairs = []
    for record in rounds:
        if math.isnan(record["margin"]) or record["depth"] <= position:
            continue
        if record["accepted"] < position:
            continue
        pairs.append((record["margin"], 1 if record["accepted"] > position else 0))
    if not pairs:
        return {"position": position, "samples": 0, "scale": None}
    best_scale, best_ll = None, -float("inf")
    scale = 0.05
    while scale <= 40.0:
        ll = 0.0
        for margin, hit in pairs:
            q = sigmoid(margin / scale)
            q = min(max(q, 1e-12), 1.0 - 1e-12)
            ll += math.log(q) if hit else math.log(1.0 - q)
        if ll > best_ll:
            best_scale, best_ll = scale, ll
        scale *= 1.02
    base = 2.0 if position == 0 else 3.0
    base_ll = 0.0
    for margin, hit in pairs:
        q = min(max(sigmoid(margin / base), 1e-12), 1.0 - 1e-12)
        base_ll += math.log(q) if hit else math.log(1.0 - q)
    return {
        "position": position, "samples": len(pairs),
        "positives": sum(hit for _, hit in pairs),
        "scale": best_scale, "log_likelihood": best_ll,
        "shipped_scale": base, "shipped_log_likelihood": base_ll,
    }


# ---------------------------------------------------------------------- board

def load_board_receipt(path: Path, prefix: str) -> dict:
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):
        rows = rows["submissions"]
    for row in rows:
        if row["id"].startswith(prefix):
            metrics = row["officialMetrics"]
            per_prompt = {}
            for entry in metrics["per_prompt"]:
                name = PROMPT_NAMES.get(entry["prompt_sha256"][:8])
                per_prompt[name] = {
                    "candidate": entry["mtp_seconds_per_token_mean"],
                    "serial": entry["serial_seconds_per_token_mean"],
                    "raw": entry["raw_ratio_of_means"],
                    "draft_len": entry["effective_mean_draft_len"],
                }
            return {
                "id": row["id"], "score": row["officialScore"],
                "status": row["status"], "solver": row["solverUsername"],
                "commit": metrics.get("commit"), "mode": metrics.get("mode"),
                "per_prompt": per_prompt,
            }
    raise SystemExit("no board row with id prefix %r" % prefix)


def median_of(values: list[float]) -> float:
    ordered = sorted(values)
    return 0.5 * (ordered[3] + ordered[4])


# ----------------------------------------------------------------------- main

ARMS = ["ship", "nomargin", "nomargin0", "nomargin1", "recal",
        "marginup", "marginfull",
        "rankedprice", "rankedprice_nomargin", "rankedprice_recal",
        "rankedprice_marginup", "static7", "oracle"]

# The three halves of the level correction, with the EMA feedback path that the
# myopic replay in `e128_jensen.py` deliberately leaves out. `reachonly` and
# `expectedonly` are the two opposing consumers of the same biased estimate;
# `levelfix` corrects both, which is the corrected estimator. `jensen` and
# `jensen_both` use the measured per-round heterogeneity gain instead of a
# fitted scalar and have no free parameter.
LEVEL_ARMS = ["reachonly", "expectedonly", "levelfix", "jensen", "jensen_both"]

# One global scalar swept across a grid that brackets the measured -9 to -24
# percent level bias, so a single best global gamma can be reported alongside
# the per-prompt fitted one.
LEVEL_GRID = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40]
LEVEL_GRID_ARMS = ["levelfix%0.2f" % g for g in LEVEL_GRID]

# Uniform price constants swept as a one-number implementable alternative to
# the shipped 0.18. The winner has to win on the median, not per prompt, so the
# grid is priced whole and the constant is chosen once for all eight prompts.
PRICE_GRID = [0.06, 0.09, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.26, 0.30, 0.36]
PRICE_ARMS = ["price%0.2f" % c for c in PRICE_GRID]
HEADLINE_ARMS = list(ARMS) + LEVEL_ARMS
ARMS += LEVEL_ARMS + LEVEL_GRID_ARMS + PRICE_ARMS


def pooled_level(rows: dict, fixture: list[str]) -> dict:
    """Pools the measured level bias and Jensen gain over a prompt's fixtures.

    `gamma` is pooled as a ratio of round-weighted totals rather than a mean of
    per-fixture ratios, so a fixture with more rounds carries proportionally
    more weight and the pooled value is the level bias of the pooled leg.
    """
    chosen = [rows[name] for name in fixture if name in rows]
    if len(chosen) != len(fixture):
        raise SystemExit("no measured level bias for %r" % fixture)
    weight = [row["rounds"] for row in chosen]
    total = sum(weight)
    accepted = sum(w * row["mean_accepted"] for w, row in zip(weight, chosen))
    expected = sum(w * row["mean_expected"] for w, row in zip(weight, chosen))
    gains = [sum(w * row["jensen_gain"][k] for w, row in zip(weight, chosen))
             / total for k in range(1, MAX_DEPTH + 1)]
    return {"gamma": accepted / expected, "jensen_gain": gains,
            "fixtures": len(chosen)}


def price(legs: dict, receipt: dict, windows: int, fit_windows: int,
          fixture_override: dict | None = None,
          constant_p: bool = False,
          shuffle_margins: bool = False,
          hold_zero_weight: bool = False,
          accept_targets: dict | None = None,
          level_rows: dict | None = None,
          seed: int = 128) -> dict:
    """One complete pricing pass. Every variant of the model comes through here.

    `accept_targets` replaces the depth anchor with an R-derived accept-rate
    anchor for the named prompts. It exists so the headline can be reported as
    a function of `R` across the pinned band; the default path never reads it
    and is therefore R-free.
    """
    results = {}
    transfer = {}
    for prompt, spec in RANKED_PROMPTS.items():
        if hold_zero_weight and spec["weight"] == 0.0:
            continue
        fixture = (fixture_override or {}).get(prompt, spec["fixture"])
        if isinstance(fixture, str):
            fixture = [fixture]
        chosen = [legs[name] for name in fixture if name in legs]
        if len(chosen) != len(fixture):
            print("skip %s: missing fixture legs %r" % (
                prompt, [n for n in fixture if n not in legs]))
            continue
        p_fixture = pooled_positions(chosen)
        rounds = [r for leg in chosen for r in leg["rounds_detail"]]

        def factory(p_target, rounds=rounds, p_fixture=p_fixture):
            return RoundSampler(rounds, p_fixture, p_target, seed=seed,
                                shuffle_margins=shuffle_margins)

        if constant_p:
            # The advisor's headline assumption: one per-step conditional for
            # the whole chain, inverted from the prompt's realised accept rate
            # and depth. No shape, no decline with position.
            p_const = invert_accept_rate(spec["accept"], spec["depth"])
            p_target = [p_const] * MAX_DEPTH
            delta = float("nan")
        elif accept_targets is not None:
            delta, d_low, d_high = fit_delta(
                factory, p_fixture, accept_targets[prompt], fit_windows,
                key="accept_rate")
            p_target = shift_p_vector(p_fixture, delta)
        else:
            delta, d_low, d_high = fit_delta(
                factory, p_fixture, spec["depth"], fit_windows)
            p_target = shift_p_vector(p_fixture, delta)

        recal_fit = (
            fit_margin_scale(rounds, 0)["scale"] or 2.0,
            fit_margin_scale(rounds, 1)["scale"] or 3.0,
        )
        level = pooled_level(level_rows, fixture) if level_rows else None
        prompt_arms = {}
        for arm in ARMS:
            prompt_arms[arm] = simulate(
                make_policy(arm, recal=recal_fit, level=level),
                factory(p_target), windows)
        transfer[prompt] = {
            "fixture": "+".join(fixture),
            "delta": delta,
            "level_gamma": level["gamma"] if level else None,
            "jensen_gain": level["jensen_gain"] if level else None,
            "p_fixture": p_fixture,
            "p_target": p_target,
            "target_depth_f92": spec["depth"],
            "simulated_depth_ship": prompt_arms["ship"]["mean_depth"],
            "target_accept_f92": spec["accept"],
            "simulated_accept_ship": prompt_arms["ship"]["accept_rate"],
            "recal_scales": recal_fit,
        }
        results[prompt] = prompt_arms

    board_prompts = receipt["per_prompt"]
    medians = {}
    per_prompt_delta = {}
    for arm in ARMS:
        raws = []
        deltas = {}
        for prompt, entry in board_prompts.items():
            ratio = 1.0
            if prompt in results:
                ratio = (results[prompt][arm]["us_per_token"]
                         / results[prompt]["ship"]["us_per_token"])
            raws.append(entry["serial"] / (entry["candidate"] * ratio))
            deltas[prompt] = (1.0 / ratio - 1.0) * 100.0
        medians[arm] = median_of(raws)
        per_prompt_delta[arm] = deltas
    base_median = median_of(
        [e["serial"] / e["candidate"] for e in board_prompts.values()])
    ship_median = medians["ship"]
    return {
        "harness": "ranked",
        "receipt": receipt,
        "base_median": base_median,
        "medians": medians,
        "median_gain_pct_vs_ship": {
            arm: (medians[arm] / ship_median - 1.0) * 100.0 for arm in ARMS},
        "per_prompt_candidate_gain_pct": per_prompt_delta,
        "transfer": transfer,
        "arms": results,
        "windows": windows,
        "curve": dict(CURVE),
    }


def invert_accept_rate(rate: float, depth: float) -> float:
    """Rule 80. Invert `E[accepted] = p (1 - p^d) / (1 - p)` for `p`."""
    target = rate * depth
    low, high = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (low + high)
        value = mid * (1.0 - mid ** depth) / (1.0 - mid)
        if value < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def accept_rate_at_R(eff: float, rounds: float) -> float:
    """Rule 12 identity, solved for the accept rate at a stated round count.

    `R + accepted = 512` and `accepted = R * eff * accept_rate`, so
    `accept_rate = (512 - R) / (R * eff)`. This is the ONLY place `R` enters
    the model.
    """
    return min(max((DECODE_TOKENS - rounds) / (rounds * eff), 0.01), 0.999)


def r_band_scenarios(identity: dict) -> dict:
    """Named `R` points per prompt, taken from the local identity manifold."""
    pred = identity["manifold"]["predictions"]
    scenarios = {}
    for factor, label in ((0.90, "predicted_x0.90"), (0.95, "predicted_x0.95"),
                          (1.00, "predicted"), (1.05, "predicted_x1.05"),
                          (1.10, "predicted_x1.10")):
        row = {}
        for prompt, entry in pred.items():
            base = entry["R_predicted"]
            if base != base:  # out of local eff range, no point estimate
                base = entry["R_assumed"]
            row[prompt] = max(entry["R_floor"],
                              min(DECODE_TOKENS, base * factor))
        scenarios[label] = row
    scenarios["band_low"] = {
        p: (e["R_band"][0] if e["R_band"][0] == e["R_band"][0]
            else e["R_assumed"]) for p, e in pred.items()}
    scenarios["band_high"] = {
        p: (e["R_band"][1] if e["R_band"][1] == e["R_band"][1]
            else e["R_assumed"]) for p, e in pred.items()}
    scenarios["assumed"] = {p: float(e["R_assumed"]) for p, e in pred.items()}
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", type=Path, required=True,
                        help="rung 1 acceptance JSON from e128_accept.py")
    parser.add_argument("--extra-accept", type=Path, action="append",
                        default=[], help="additional acceptance JSON files")
    parser.add_argument("--board", type=Path,
                        default=Path("/tmp/yukon-board/full.json"))
    parser.add_argument("--receipt", default="44559d02")
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--fit-windows", type=int, default=60)
    parser.add_argument("--shipped", type=Path,
                        help="shipped-policy acceptance JSON, held out for "
                             "the zero-parameter simulator validation")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--sensitivity-json", type=Path)
    parser.add_argument("--identity", type=Path,
                        help="rung0-identity.json; enables the R-band report")
    parser.add_argument("--jensen", type=Path, required=True,
                        help="jensen-and-sign.json; supplies the measured "
                             "level bias and Jensen gain per fixture")
    parser.add_argument("--r-band-json", type=Path)
    parser.add_argument("--curve-json", type=Path,
                        help="ranked cost curve fitted by e128_ourcurve.py; "
                             "the F97 board curve is used when omitted")
    parser.add_argument("--curve-key",
                        help="named curve inside a multi-curve --curve-json")
    parser.add_argument("--curve-sweep", nargs="*", default=[],
                        help="curve names to reprice every arm against")
    parser.add_argument("--curve-sweep-json", type=Path)
    args = parser.parse_args()
    if args.curve_json:
        load_curve(args.curve_json, args.curve_key)
    print("ranked cost curve %s: break M>=%d  %.1f + %.1f M  |  %.1f + %.1f M"
          % (CURVE["name"], CURVE["breakpoint"], CURVE["lo"][0], CURVE["lo"][1],
             CURVE["hi"][0], CURVE["hi"][1]))
    print("round us by M: %s" % "  ".join(
        "%d:%.0f" % (m, ranked_round_us(m)) for m in range(1, 10)))

    legs = {}
    for path in [args.accept] + args.extra_accept:
        for leg in json.loads(path.read_text())["legs"]:
            legs[leg["prompt_id"]] = leg
    level_rows = {row["prompt_id"]: row for row in
                  json.loads(args.jensen.read_text())["hypothesis_j"]}

    validation = []
    validation_gate = None
    if args.shipped:
        shipped = {leg["prompt_id"]: leg
                   for leg in json.loads(args.shipped.read_text())["legs"]}
        validation = validate_simulator(legs, shipped, args.windows)
        print("\nzero-parameter simulator validation "
              "(uncensored acceptance in, measured shipped leg out):")
        print("%-20s %9s %9s %9s %9s %9s" % (
            "fixture", "depth*", "depth~", "d err", "accept*", "accept~"))
        for row in validation:
            print("%-20s %9.3f %9.3f %+9.3f %9.4f %9.4f" % (
                row["fixture"], row["measured_depth"], row["simulated_depth"],
                row["simulated_depth"] - row["measured_depth"],
                row["measured_accept"], row["simulated_accept"]))
        errs = [r["simulated_depth"] - r["measured_depth"] for r in validation]
        aerrs = [r["simulated_accept"] - r["measured_accept"]
                 for r in validation]
        if errs:
            print("depth error:  mean %+0.3f, max |err| %0.3f over %d fixtures"
                  % (sum(errs) / len(errs), max(abs(e) for e in errs),
                     len(errs)))
            print("accept error: mean %+0.4f, max |err| %0.4f"
                  % (sum(aerrs) / len(aerrs), max(abs(e) for e in aerrs)))
            # Pre-registered gate. The simulator has one free level parameter
            # per prompt and it is fitted on depth, so a depth bias here is a
            # bias the fit absorbs, while an accept bias is not corrected
            # anywhere. Both have to be small for a counterfactual to mean
            # anything.
            validation_gate = {
                "mean_depth_error": sum(errs) / len(errs),
                "max_abs_depth_error": max(abs(e) for e in errs),
                "mean_accept_error": sum(aerrs) / len(aerrs),
                "max_abs_accept_error": max(abs(e) for e in aerrs),
                "fixtures": len(errs),
                "depth_tolerance": 0.25,
                "accept_tolerance": 0.05,
                "passed": (abs(sum(errs) / len(errs)) <= 0.25
                           and abs(sum(aerrs) / len(aerrs)) <= 0.05),
            }
            print("zero-parameter validation gate: %s "
                  "(|mean depth err| <= 0.25 and |mean accept err| <= 0.05)"
                  % ("PASS" if validation_gate["passed"] else "FAIL"))

    receipt = load_board_receipt(args.board, args.receipt)
    data = price(legs, receipt, args.windows, args.fit_windows,
                 level_rows=level_rows)

    arms = ARMS
    medians = data["medians"]
    transfer = data["transfer"]
    per_prompt_delta = data["per_prompt_candidate_gain_pct"]
    base_median = data["base_median"]
    ship_median = medians["ship"]
    board_prompts = receipt["per_prompt"]

    print("\nreceipt %s (%s, %s) official %.8f  model-reconstructed %.8f" % (
        receipt["id"][:8], receipt["solver"], receipt["status"],
        receipt["score"], base_median))
    print("\ntransfer fit (one level parameter per prompt, fitted on depth):")
    print("%-10s %-33s %7s %9s %9s %9s %9s" % (
        "prompt", "fixture", "delta", "depth*", "depth~", "accept*", "accept~"))
    for prompt, entry in transfer.items():
        print("%-10s %-33s %7.3f %9.3f %9.3f %9.4f %9.4f" % (
            prompt, entry["fixture"], entry["delta"],
            entry["target_depth_f92"], entry["simulated_depth_ship"],
            entry["target_accept_f92"], entry["simulated_accept_ship"]))

    print("\nranked median per arm (Rule 67, median recomputed over 8 prompts):")
    print("%-20s %14s %12s %12s" % ("arm", "median", "vs ship %", "vs base %"))
    for arm in HEADLINE_ARMS:
        print("%-20s %14.8f %12.4f %12.4f" % (
            arm, medians[arm],
            (medians[arm] / ship_median - 1.0) * 100.0,
            (medians[arm] / base_median - 1.0) * 100.0))

    print("\nper-prompt candidate speedup (%) by arm:")
    print("%-10s" % "prompt" + "".join("%11s" % a for a in HEADLINE_ARMS))
    for prompt in board_prompts:
        print("%-10s" % prompt + "".join(
            "%11.3f" % per_prompt_delta[arm][prompt] for arm in HEADLINE_ARMS))

    print("\nuniform price constant sweep (shipped is 0.18):")
    print("%-10s %14s %12s" % ("constant", "median", "vs ship %"))
    for arm in PRICE_ARMS:
        print("%-10s %14.8f %12.4f" % (
            arm[5:], medians[arm],
            (medians[arm] / ship_median - 1.0) * 100.0))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        data["validation"] = validation
        data["validation_gate"] = validation_gate
        args.json.write_text(json.dumps(data, indent=2) + "\n")

    if args.sensitivity_json:
        variants = {
            "headline": {},
            "constant_p": {"constant_p": True},
            "shuffle_margins": {"shuffle_margins": True},
            # Holds the three zero-weight prompts at the shipped ratio 1.0
            # instead of repricing them; it does not remove them from the
            # eight-prompt median.
            "hold_zero_weight_at_shipped": {"hold_zero_weight": True},
            "seed_777": {"seed": 777},
            "single_fixture_a": {"fixture_override": {
                "beagle": ["beagle_a"], "medicine": ["medicine_hist"],
                "essays": ["essays_bacon"]}},
            "single_fixture_b": {"fixture_override": {
                "beagle": ["beagle_b"], "medicine": ["medicine_hippoc"],
                "essays": ["essays_montaigne"]}},
            "benchfixture_only": {"fixture_override": {
                p: ["benchfixture"] for p in RANKED_PROMPTS}},
            "receipt_crown": {"receipt": "d3c491b5"},
            "receipt_prev_crown": {"receipt": "bc070b7b"},
            "receipt_ec778a91": {"receipt": "ec778a91"},
        }
        rows = {}
        for name, kwargs in variants.items():
            kwargs = dict(kwargs)
            rec = receipt
            rid = kwargs.pop("receipt", None)
            if rid is not None:
                rec = load_board_receipt(args.board, rid)
            if "fixture_override" in kwargs:
                missing = [f for names in kwargs["fixture_override"].values()
                           for f in names if f not in legs]
                if missing:
                    print("skip sensitivity %s: no legs %s" % (name, missing))
                    continue
            out = price(legs, rec, args.windows, args.fit_windows,
                        level_rows=level_rows, **kwargs)
            rows[name] = {
                "receipt": rec["id"][:8],
                "kwargs": {k: v for k, v in kwargs.items()},
                "medians": out["medians"],
                "median_gain_pct_vs_ship": out["median_gain_pct_vs_ship"],
                "simulated_depth_ship": {
                    p: e["simulated_depth_ship"]
                    for p, e in out["transfer"].items()},
            }

        print("\nsensitivity of median gain vs ship (%):")
        print("%-18s" % "variant" + "".join("%11s" % a for a in arms))
        for name, row in rows.items():
            print("%-18s" % name + "".join(
                "%11.4f" % row["median_gain_pct_vs_ship"][a] for a in arms))

        args.sensitivity_json.parent.mkdir(parents=True, exist_ok=True)
        args.sensitivity_json.write_text(json.dumps({
            "harness": "ranked",
            "variants": rows,
        }, indent=2) + "\n")

    if args.curve_sweep:
        # The tier step and the R vector are the two structural choices in the
        # cost curve, so every arm is repriced on each candidate curve. A
        # conclusion that changes sign between curves is a conclusion about the
        # fit, not about the scheduler.
        arms = [a for a in ARMS if not a.startswith("price")]
        saved = dict(CURVE)
        curves = {}
        for key in args.curve_sweep:
            load_curve(args.curve_json, key)
            out = price(legs, receipt, args.windows, args.fit_windows,
                        level_rows=level_rows)
            curves[key] = {
                "curve": dict(CURVE),
                "round_us": {m: ranked_round_us(m) for m in range(1, 10)},
                "medians": out["medians"],
                "median_gain_pct_vs_ship": out["median_gain_pct_vs_ship"],
                "simulated_depth_ship": {
                    p: e["simulated_depth_ship"]
                    for p, e in out["transfer"].items()},
            }
        CURVE.update(saved)

        print("\nmedian gain vs ship (%) by ranked cost curve:")
        print("%-14s %6s" % ("curve", "break")
              + "".join("%11s" % a for a in arms))
        for key, row in curves.items():
            print("%-14s %6d" % (key, row["curve"]["breakpoint"]) + "".join(
                "%11.4f" % row["median_gain_pct_vs_ship"][a] for a in arms))

        if args.curve_sweep_json:
            args.curve_sweep_json.parent.mkdir(parents=True, exist_ok=True)
            args.curve_sweep_json.write_text(json.dumps({
                "harness": "ranked", "curves": curves,
            }, indent=2) + "\n")

    if args.identity:
        identity = json.loads(args.identity.read_text())
        scenarios = r_band_scenarios(identity)
        band = {}
        for name, rounds in scenarios.items():
            targets = {p: accept_rate_at_R(RANKED_PROMPTS[p]["depth"],
                                           rounds[p])
                       for p in RANKED_PROMPTS if p in rounds}
            out = price(legs, receipt, args.windows, args.fit_windows,
                        accept_targets=targets, level_rows=level_rows)
            band[name] = {
                "R": rounds,
                "accept_targets": targets,
                "medians": out["medians"],
                "median_gain_pct_vs_ship": out["median_gain_pct_vs_ship"],
                "simulated_depth_ship": {
                    p: e["simulated_depth_ship"]
                    for p, e in out["transfer"].items()},
                "simulated_accept_ship": {
                    p: e["simulated_accept_ship"]
                    for p, e in out["transfer"].items()},
            }

        print("\nheadline as a function of R (accept-rate anchor; the depth "
              "anchor above is R-free):")
        print("%-18s %9s" % ("R scenario", "R beagle")
              + "".join("%11s" % a for a in HEADLINE_ARMS))
        for name, row in band.items():
            print("%-18s %9.1f" % (name, row["R"]["beagle"]) + "".join(
                "%11.4f" % row["median_gain_pct_vs_ship"][a]
                for a in HEADLINE_ARMS))
        implementable = [a for a in HEADLINE_ARMS if a != "oracle"]
        # Advisor F2 rule: an arm whose sign flips anywhere inside the pinned
        # band is sign indeterminate. It is never reported as a gain, however
        # large the favourable end of its range is.
        ordered = sorted(band.items(), key=lambda kv: kv[1]["R"]["beagle"])
        spans = {}
        for arm in HEADLINE_ARMS:
            values = [row["median_gain_pct_vs_ship"][arm]
                      for row in band.values()]
            series = [(row["R"]["beagle"], row["median_gain_pct_vs_ship"][arm])
                      for _, row in ordered]
            flips = []
            for (r0, v0), (r1, v1) in zip(series, series[1:]):
                if (v0 > 0) != (v1 > 0) and v1 != v0:
                    flips.append(r0 + (r1 - r0) * (0.0 - v0) / (v1 - v0))
            spans[arm] = {
                "min": min(values), "max": max(values),
                "span": max(values) - min(values),
                "sign_indeterminate": bool(flips),
                "sign_flip_R_beagle": flips,
            }
        print("\nR-band span of the median gain (%%): oracle %.4f, "
              "widest implementable %.4f" % (
                  spans["oracle"]["span"],
                  max(spans[a]["span"] for a in implementable)))
        print("sign of the median gain across the pinned R band:")
        for arm in HEADLINE_ARMS:
            entry = spans[arm]
            verdict = "SIGN INDETERMINATE, flips at R(beagle) = %s" % (
                ", ".join("%.1f" % r for r in entry["sign_flip_R_beagle"])
            ) if entry["sign_indeterminate"] else (
                "positive throughout" if entry["min"] > 0 else
                "negative throughout" if entry["max"] < 0 else "zero")
            print("  %-22s [%+8.4f, %+8.4f] %s"
                  % (arm, entry["min"], entry["max"], verdict))
        if args.r_band_json:
            args.r_band_json.parent.mkdir(parents=True, exist_ok=True)
            args.r_band_json.write_text(json.dumps({
                "harness": "ranked",
                "anchor": "accept_rate derived from R; depth anchor is R-free",
                "identity_source": str(args.identity),
                "scenarios": band,
                "spans": spans,
                "depth_anchored_headline": data["median_gain_pct_vs_ship"],
            }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
