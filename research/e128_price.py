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

# F97, fitted over 147 official runs. `round_us(M)` at verify width M = d + 1.
RANKED_TIER = {
    1: (27215.4, 3966.4),   # M = 1..4
    2: (17020.7, 7154.2),   # M = 5..8
}

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
    tier = 1 if rows <= 4 else 2
    intercept, slope = RANKED_TIER[tier]
    return intercept + slope * rows


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
            p_vec.append(accepted / reached)
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
                 shuffle_margins: bool = False):
        self.rounds = rounds
        self.s_fix = survival(p_fixture)
        self.s_target = survival(p_target)
        self.rng = random.Random(seed)
        self.margins = [r["margin"] for r in rounds]
        self.shuffle_margins = shuffle_margins

    def draw(self) -> tuple:
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

def make_policy(arm: str, recal: tuple = (2.0, 3.0)):
    """Returns `policy(ema, margin, offer, capability) -> depth`."""
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

def fit_delta(sampler_factory, p_fixture: list[float], target_depth: float,
              windows: int) -> tuple:
    """One level parameter, fitted so the SHIPPED arm reproduces F92 depth."""
    policy = make_policy("ship")
    low, high = -4.0, 6.0

    def depth_at(delta: float) -> float:
        p_target = shift_p_vector(p_fixture, delta)
        return simulate(policy, sampler_factory(p_target), windows)["mean_depth"]

    d_low, d_high = depth_at(low), depth_at(high)
    if not (d_low <= target_depth <= d_high):
        return None, d_low, d_high
    for _ in range(40):
        mid = 0.5 * (low + high)
        if depth_at(mid) < target_depth:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high), d_low, d_high


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
        "static7", "oracle"]


def price(legs: dict, receipt: dict, windows: int, fit_windows: int,
          fixture_override: dict | None = None,
          constant_p: bool = False,
          shuffle_margins: bool = False,
          hold_zero_weight: bool = False,
          seed: int = 128) -> dict:
    """One complete pricing pass. Every variant of the model comes through here."""
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
        else:
            delta, d_low, d_high = fit_delta(
                factory, p_fixture, spec["depth"], fit_windows)
            if delta is None:
                print("skip %s: shipped depth %.3f outside simulator range "
                      "[%.3f, %.3f]" % (prompt, spec["depth"], d_low, d_high))
                continue
            p_target = shift_p_vector(p_fixture, delta)

        recal_fit = (
            fit_margin_scale(rounds, 0)["scale"] or 2.0,
            fit_margin_scale(rounds, 1)["scale"] or 3.0,
        )
        prompt_arms = {}
        for arm in ARMS:
            prompt_arms[arm] = simulate(
                make_policy(arm, recal=recal_fit), factory(p_target), windows)
        transfer[prompt] = {
            "fixture": "+".join(fixture),
            "delta": delta,
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
    parser.add_argument("--json", type=Path)
    parser.add_argument("--sensitivity-json", type=Path)
    args = parser.parse_args()

    legs = {}
    for path in [args.accept] + args.extra_accept:
        for leg in json.loads(path.read_text())["legs"]:
            legs[leg["prompt_id"]] = leg

    receipt = load_board_receipt(args.board, args.receipt)
    data = price(legs, receipt, args.windows, args.fit_windows)
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
    print("%-10s %14s %12s %12s" % ("arm", "median", "vs ship %", "vs base %"))
    for arm in arms:
        print("%-10s %14.8f %12.4f %12.4f" % (
            arm, medians[arm],
            (medians[arm] / ship_median - 1.0) * 100.0,
            (medians[arm] / base_median - 1.0) * 100.0))

    print("\nper-prompt candidate speedup (%) by arm:")
    header = "%-10s" % "prompt" + "".join("%11s" % a for a in arms)
    print(header)
    for prompt in board_prompts:
        print("%-10s" % prompt + "".join(
            "%11.3f" % per_prompt_delta[arm][prompt] for arm in arms))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n")

    if args.sensitivity_json:
        variants = {
            "headline": {},
            "constant_p": {"constant_p": True},
            "shuffle_margins": {"shuffle_margins": True},
            "drop_zero_weight": {"hold_zero_weight": True},
            "seed_777": {"seed": 777},
            "single_fixture_a": {"fixture_override": {
                "beagle": ["beagle_a"], "medicine": ["medicine_hist"],
                "essays": ["essays_bacon"]}},
            "single_fixture_b": {"fixture_override": {
                "beagle": ["beagle_b"], "medicine": ["medicine_hippoc"],
                "essays": ["essays_montaigne"]}},
            "benchfixture_only": {"fixture_override": {
                p: ["benchfixture"] for p in RANKED_PROMPTS}},
            "receipt_crown": {"receipt": "bc070b7b"},
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
            out = price(legs, rec, args.windows, args.fit_windows, **kwargs)
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
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
