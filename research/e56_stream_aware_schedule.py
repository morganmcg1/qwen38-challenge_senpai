#!/usr/bin/env python3
"""E56 Step 0: price the depth walk's scalar `h` against the real cost staircase.

WHAT THIS ASKS. `Qwen36MTPBlockSession.costModelDepth` extends a draft chain
while `reach > h * (1 + expected) / (1 + depth * h)`. That threshold is the
exact algebraic consequence of ONE cost assumption: a round of depth `d` costs
`C(d) = C(0) * (1 + d * h)`, i.e. every extra draft row costs the same constant
fraction `h` of a depth-0 round. This script asks whether that assumption is
true, where it breaks, and what a corrected cost term would be worth.

Derivation of the shipped threshold, so the generalisation below is provably
the same rule with one assumption relaxed. Extending depth `d` -> `d+1`
improves time per emitted token iff

    C(d+1) / (1 + expected + reach)  <  C(d) / (1 + expected)
<=> (C(d+1) - C(d)) * (1 + expected)  <  C(d) * reach
<=> reach  >  marginal(d) * (1 + expected) / C(d).

With `C(d) = 1 + d*h` and `marginal(d) = h` this is byte-for-byte the shipped
line. So a width-dependent cost term needs no new policy shape: only `C` and
`marginal` change.

THREE BOUNDARIES ARE PRICED SEPARATELY, per the advisor's correction:

  1. QMV 4 -> 5   (weight streams 1 -> 2), from the shipped dispatch table.
  2. QMV 8 -> 9   (weight streams 2 -> 3), from the shipped dispatch table.
  3. SDPA 5 -> 6  the WIDE-DECODE EXACTNESS CHUNK in our own
     `AttentionUtils.swift` issues a second SDPA call for qL >= 6. This is a
     property of OUR predicate, not of the kernel family: the trusted host
     dispatch selects vector mode on `qL <= 8` with no GQA term, and only
     splits to the 2-pass route when `kL >= 1024`. It is therefore reported as
     "cost of the current chunk predicate at width 6" and is NEVER compiled
     into a shipped cost term.

EVIDENCE USED, with provenance:
  * E1 (`research/results/qwen38-r1-e1-depth-cost-curve.md`), THIS host class
    (M4 Pro), declared 4-bit head: measured C(0..8) over 1778/129/83/61/60/2/
    36/7/32 rounds. This is the primary truth model. It is a direct
    measurement of the quantity the schedule guesses.
  * E46 (thorfinn, PR #51, M4 Pro): `T(M) = 16.757 + 27.532*ceil(M/IPG) +
    9.624*M`, a QMV-only fit. Used for its RATIO 27.532/9.624 only; its levels
    exceed the measured whole-round cost and cannot be scored time.
  * E53 (this student, PR #56): two-state burst acceptance models fitted to the
    published per-prompt `effective_mean_draft_len` and accept ratio for the
    two scored prompts, replayed through an exact port of the shipped walk.

Reproduce:  python3 research/e56_stream_aware_schedule.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e53_width_mixture as E53
import qmv_score_leverage as Q

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "e56-stream-aware-schedule.json"

SESSION = ROOT / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
ATTENTION = ROOT / "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift"
QUANTIZED_H = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
SDPA_CPP = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp"

# ---- E1 measured round cost, µs, M4 Pro, declared 4-bit head ---------------
# Table "Measured curve", pooled arms d0,d1,d2,d3,d4,d6,base-decl,d8.
E1_C_US = [65009.4, 70482.4, 75519.2, 91287.8, 115690.9,
           134668.0, 154169.1, 172827.0, 198236.5]
E1_N = [1778, 129, 83, 61, 60, 2, 36, 7, 32]
E1_HEAD_STEP_US = 2590.0          # isolated head step, E1 "making the policy
                                  # head-agnostic"; ratio 2590/65009 = 0.039819

# ---- E46 QMV-only fit, M4 Pro. RATIO ONLY; levels are not scored time. -----
E46_INTERCEPT, E46_PER_STREAM, E46_PER_ROW = 16.757, 27.532, 9.624

# ---- leg composition ------------------------------------------------------
# E3 part A (thorfinn, PR #?): the timed 512-token seed prefill is 23.9 % of the
# candidate leg at the ranked window on this host, and is irreducible.
PREFILL_SHARE_OF_LEG = 0.239
DECODE_SHARE_OF_LEG = 1.0 - PREFILL_SHARE_OF_LEG

# askeladd's local end-to-end null floor, the pre-registered stop rule.
NULL_FLOOR_PCT = 0.0629


# =========================================================================
# 1. Live source, parsed rather than restated
# =========================================================================

def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_schedule_constants() -> dict:
    """Every scalar the walk reads, taken from the live Swift source."""
    text = _read(SESSION)
    patterns = {
        "headStepCostRatio": r"static let headStepCostRatio = ([0-9.]+)",
        "sdpaWidthWallDepthCap": r"static let sdpaWidthWallDepthCap = (\d+)",
        "segmentedVerifyDepthCap": r"static let segmentedVerifyDepthCap = (\d+)",
        "segmentedStreakGate": r"static let segmentedStreakGate = (\d+)",
        "acceptEMAAlpha": r"static let acceptEMAAlpha = ([0-9.]+)",
        "emaPriorBase": r"\.map \{ ([0-9.]+) \* pow\(",
        "emaPriorDecay": r"pow\(([0-9.]+), Double\(\$0\)\) \}",
        "optimismCap": r"if positionAcceptEMA\[acceptedCount\] < ([0-9.]+) \{",
    }
    out = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise RuntimeError(f"{name} not found in {SESSION}")
        value = match.group(1)
        out[name] = float(value) if "." in value else int(value)
    return out


def parse_ipg_table() -> dict:
    """{M: IPG} from the shipped `out_vec_size >= 4096` dispatch switch.

    The schedule's staircase must be DERIVED from this table, never restated:
    if someone retunes an IPG, a stale staircase is a silent mis-price.
    """
    text = _read(QUANTIZED_H)
    start = text.index("if (out_vec_size >= 4096)")
    end = text.index("} else {", start)
    body = text[start:end]
    table = {}
    for m, ipg in re.findall(
            r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>", body):
        table[int(m)] = int(ipg)
    for m in re.findall(r"qmv_fast_crossrow_affine4_g64<T, (\d+)>", body):
        table.setdefault(int(m), int(m))     # all rows resident: one stream
    if not table:
        raise RuntimeError("no QMV dispatch cells parsed")
    return table


def parse_chunk_predicate() -> dict:
    """The live WIDE-DECODE EXACTNESS CHUNK predicate in our own attention."""
    text = _read(ATTENTION)
    match = re.search(
        r"if queries\.dim\(0\) == (\d+), qL >= (\d+), qL <= (\d+), kL >= qL,",
        text)
    if match is None:
        raise RuntimeError("chunk predicate not found in AttentionUtils.swift")
    split = re.search(r"let split = (\d+)", text)
    if split is None:
        raise RuntimeError("chunk split constant not found")
    return {"batch": int(match.group(1)), "min_qL": int(match.group(2)),
            "max_qL": int(match.group(3)), "split": int(split.group(1))}


def parse_host_sdpa_route() -> dict:
    """The trusted host dispatch that the chunk predicate claims to serve."""
    text = _read(SDPA_CPP)
    vector = re.search(r"if \(q_pre\.shape\(2\) <= (\d+)\)", text)
    two_pass = re.search(r"k\.shape\(2\) >= (\d+)", text)
    if vector is None or two_pass is None:
        raise RuntimeError("host SDPA route conditions not found")
    return {"vector_mode_max_qL": int(vector.group(1)),
            "two_pass_min_kL": int(two_pass.group(1))}


def sdpa_calls(width: int, key_len: int, predicate: dict) -> int:
    """SDPA calls our attention issues for a decode of `width` rows."""
    if (predicate["min_qL"] <= width <= predicate["max_qL"]
            and key_len >= width):
        return 2
    return 1


# =========================================================================
# 2. Cost models
# =========================================================================

def streams(width: int, ipg: dict) -> int:
    if width in ipg:
        return math.ceil(width / ipg[width])
    if width in (1, 2):
        return 1
    raise ValueError(f"width {width} outside the dispatch table")


def crosses_stream_boundary(depth: int, ipg: dict) -> bool:
    """Does extending depth -> depth+1 (width d+1 -> d+2) add a weight pass?"""
    return streams(depth + 2, ipg) > streams(depth + 1, ipg)


class Price:
    """A normalised round-cost model: C(0) = 1, C(d) = 1 + sum of marginals."""

    def __init__(self, name: str, marginals: list[float], note: str = "") -> None:
        if len(marginals) != E53.MAX_DEPTH:
            raise ValueError("need one marginal per legal extension")
        self.name = name
        self.marginals = list(marginals)
        self.note = note
        self.cum = [1.0]
        for step in marginals:
            self.cum.append(self.cum[-1] + step)

    def cost(self, depth: int) -> float:
        return self.cum[depth]

    def marginal(self, depth: int) -> float:
        return self.marginals[depth]

    def mean_marginal(self) -> float:
        return statistics.fmean(self.marginals)


def shipped_price(h: float) -> Price:
    return Price("shipped_scalar", [h] * E53.MAX_DEPTH,
                 "C(d) = 1 + d*h, the assumption behind the live threshold")


def measured_price() -> Price:
    """E1's measured curve, renormalised to C(0) = 1. Recomputed, not quoted."""
    base = E1_C_US[0]
    marginals = [(E1_C_US[d + 1] - E1_C_US[d]) / base
                 for d in range(E53.MAX_DEPTH)]
    return Price("measured_e1", marginals,
                 "measured round cost, M4 Pro, declared 4-bit head")


def pinned_shape_price(name: str, shape: list[float], head_ratio: float,
                       mean_target: float, note: str) -> Price:
    """A verify-cost SHAPE plus the head step, pinned to the live mean price."""
    verify_budget = mean_target - head_ratio
    if verify_budget <= 0:
        raise ValueError("head ratio exceeds the mean price budget")
    scale = verify_budget / statistics.fmean(shape)
    return Price(name, [head_ratio + scale * value for value in shape], note)


def stream_aware_price(ipg: dict, head_ratio: float, mean_target: float,
                       ratio: float = E46_PER_STREAM / E46_PER_ROW,
                       chunk_surcharge: float = 0.0,
                       chunk_from_depth: int | None = None) -> Price:
    """Head step + QMV staircase, calibrated to preserve the tuned mean price.

    The staircase changes only the SHAPE of the price. Its mean over the eight
    legal extensions is pinned to `mean_target` (the live `h`), so this is not
    a disguised retune of `headStepCostRatio`: the advisor's constraint is one
    hypothesis, and "the price has the wrong shape" is that hypothesis.
    """
    raw = [1.0 + ratio * (1.0 if crosses_stream_boundary(d, ipg) else 0.0)
           for d in range(E53.MAX_DEPTH)]
    verify_budget = mean_target - head_ratio
    if verify_budget <= 0:
        raise ValueError("head ratio exceeds the mean price budget")
    scale = verify_budget / statistics.fmean(raw)
    marginals = [head_ratio + scale * value for value in raw]
    if chunk_surcharge and chunk_from_depth is not None:
        marginals[chunk_from_depth] += chunk_surcharge
    name = "stream_aware" if not chunk_surcharge else "stream_aware_plus_chunk"
    return Price(name, marginals,
                 f"head {head_ratio:.6f} + staircase ratio {ratio:.4f}, "
                 f"mean pinned to {mean_target}")


# =========================================================================
# 3. The walk, generalised over a price
# =========================================================================

class PricedSchedule(E53.Schedule):
    """`costModelDepth` with the cost assumption made explicit."""

    def __init__(self, price: Price) -> None:
        super().__init__()
        self.price = price

    def depth(self, offered: int, margin: float | None) -> int:
        width_cap = (E53.SEGMENTED_VERIFY_DEPTH_CAP
                     if self.streak >= E53.SEGMENTED_STREAK_GATE
                     else E53.SDPA_WIDTH_WALL_DEPTH_CAP)
        cap = min(min(offered, E53.MAX_DEPTH), width_cap)
        if cap <= 0:
            return 0
        reach, expected, depth = 1.0, 0.0, 0
        while depth < cap:
            p = self.ema[depth]
            if margin is not None:
                if depth == 0:
                    p = min(p, E53.sigmoid(margin / 2.0))
                elif depth == 1:
                    p = min(p, E53.sigmoid(margin / 3.0))
            reach *= p
            threshold = (self.price.marginal(depth) * (1.0 + expected)
                         / self.price.cost(depth))
            if not reach > threshold:
                break
            expected += reach
            depth += 1
        return depth


def run_window(model: E53.BurstAcceptance, rng: random.Random, price: Price,
               truth: list[float]) -> dict:
    """One 512-token window. `truth` is round cost in µs indexed by depth.

    Every round consumes exactly 1 + MAX_DEPTH uniforms whatever depth is
    chosen, so two policies driven from the same seed see the same acceptance
    draws until their round boundaries diverge (common random numbers).
    """
    sched = PricedSchedule(price)
    emitted = 0
    widths: dict[int, int] = {}
    stops_below_cap: dict[int, int] = {}
    total_us = 0.0
    drafted_total = accepted_total = rounds = non_drafting = 0
    while emitted < E53.TOTAL_TOKENS:
        remaining = E53.TOTAL_TOKENS - emitted
        offered = max(1, min(E53.OFFERED_DEPTH, E53.MAX_DEPTH, remaining - 1))
        probs, margin = model.step(rng)
        draws = [rng.random() for _ in range(E53.MAX_DEPTH)]
        depth = sched.depth(offered, margin)
        accepted = 0
        for index in range(depth):
            if draws[index] < probs[index]:
                accepted += 1
            else:
                break
        rounds += 1
        drafted_total += depth
        accepted_total += accepted
        if depth == 0:
            non_drafting += 1
        widths[depth + 1] = widths.get(depth + 1, 0) + 1
        cap = min(offered, E53.MAX_DEPTH,
                  E53.SEGMENTED_VERIFY_DEPTH_CAP
                  if sched.streak >= E53.SEGMENTED_STREAK_GATE
                  else E53.SDPA_WIDTH_WALL_DEPTH_CAP)
        if depth < cap:
            stops_below_cap[depth + 1] = stops_below_cap.get(depth + 1, 0) + 1
        total_us += truth[depth]
        emitted += 1 + accepted
        sched.record(accepted, depth)
        if rounds > 4 * E53.TOTAL_TOKENS:
            raise RuntimeError("schedule failed to close the window")
    return {
        "rounds": rounds,
        "drafted": drafted_total,
        "accepted": accepted_total,
        "non_drafting": non_drafting,
        "emitted": emitted,
        "us_per_token": total_us / emitted,
        "mean_draft_len": drafted_total / rounds,
        "accept_ratio": accepted_total / drafted_total if drafted_total else 0.0,
        "widths": widths,
        "stops_below_cap": stops_below_cap,
    }


def replay(fit: dict, price: Price, truth: list[float], windows: int,
           seed: int) -> dict:
    runs = []
    for index in range(windows):
        rng = random.Random(seed + index)
        model = E53.BurstAcceptance(fit["share_easy"], fit["persistence"],
                                    fit["q_easy"], fit["q_hard"])
        model.easy = rng.random() < fit["share_easy"]
        runs.append(run_window(model, rng, price, truth))
    widths: dict[int, int] = {}
    stops: dict[int, int] = {}
    for run in runs:
        for width, count in run["widths"].items():
            widths[width] = widths.get(width, 0) + count
        for width, count in run["stops_below_cap"].items():
            stops[width] = stops.get(width, 0) + count
    total_rounds = sum(widths.values())
    per_window = [run["us_per_token"] for run in runs]
    return {
        "policy": price.name,
        "windows": windows,
        "us_per_token": statistics.fmean(per_window),
        "us_per_token_sem": (statistics.stdev(per_window) / math.sqrt(windows)
                             if windows > 1 else 0.0),
        "per_window": per_window,
        "mean_draft_len": sum(r["drafted"] for r in runs) / total_rounds,
        "accept_ratio": (sum(r["accepted"] for r in runs)
                         / max(1, sum(r["drafted"] for r in runs))),
        "rounds": statistics.fmean([r["rounds"] for r in runs]),
        "non_drafting": statistics.fmean([r["non_drafting"] for r in runs]),
        "mixture": {w: widths[w] / total_rounds for w in sorted(widths)},
        "stops_below_cap": {w: stops.get(w, 0) / total_rounds
                            for w in sorted(widths)},
    }


def paired_delta_pct(base: dict, other: dict) -> dict:
    """Paired % change in µs/token, `other` against `base`. Negative is faster."""
    pairs = list(zip(base["per_window"], other["per_window"]))
    diffs = [100.0 * (b - a) / a for a, b in pairs]
    mean = statistics.fmean(diffs)
    sem = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return {"decode_pct": mean, "decode_pct_sem": sem,
            "leg_pct": mean * DECODE_SHARE_OF_LEG,
            "leg_pct_sem": sem * DECODE_SHARE_OF_LEG}


# =========================================================================
# 4. Fits, from E53
# =========================================================================

def load_burst_fits() -> dict:
    """The feasible E53 burst models for the two scored prompts."""
    with open(ROOT / "research" / "e53-width-mixture.json", encoding="utf-8") as fh:
        report = json.load(fh)
    fits = {}
    for prompt, rows in report["burst"].items():
        feasible = [row for row in rows if row.get("feasible")]
        if not feasible:
            raise RuntimeError(f"no feasible burst fit for {prompt}")
        fits[prompt] = feasible
    return fits


def central_fit(rows: list[dict], target_mean: float) -> dict:
    """The feasible point closest to the published mean draft length."""
    return min(rows, key=lambda row: abs(row["mean_draft_len"] - target_mean))


# =========================================================================
# 4b. Falsification: the campaign has already measured three scalar prices
# =========================================================================

# `Qwen36MTPBlockSession.swift:719-726`, receipts for scalar-h changes made on
# the real machine. Only the h = 0.32 arm carries a decode-time number and a
# per-prompt draft-length change, so it is the one usable quantitative check.
H_RECEIPTS = {
    0.32: {"score": 2.84585, "candidate_decode_time_pct": +0.95,
           "mean_draft_len_before": [4.35, 4.89, 5.78, 5.33, 5.04],
           "mean_draft_len_after": [3.36, 4.01, 4.53, 4.03, 4.76],
           "commit": "fc62d1aa"},
    0.15: {"score": 2.667, "candidate_decode_time_pct": None},
    0.14: {"score": 2.766, "candidate_decode_time_pct": None},
}


def match_scalar_depth(fit: dict, target_mean_draft: float, truth: list[float],
                       windows: int, seed: int) -> tuple[float, dict]:
    """The scalar price whose walk drafts as deeply as a shaped price does.

    THE CONTROL THAT DECIDES THIS EXPERIMENT. A shaped cost term changes two
    things at once: the average depth and the depth-to-depth SHAPE. Only the
    shape is the hypothesis; the average depth is `headStepCostRatio`, which
    is bracketed on both sides by measured ranked receipts and which the
    assignment forbids retuning. Comparing a shaped price against the scalar
    at MATCHED mean draft length removes the level and leaves the shape.
    """
    low, high = 0.05, 0.60
    run = None
    scalar = high
    for _ in range(22):
        scalar = 0.5 * (low + high)
        run = replay(fit, shipped_price(scalar), truth, windows, seed)
        if run["mean_draft_len"] > target_mean_draft:
            low = scalar
        else:
            high = scalar
    return scalar, run


def scalar_validation(fits: dict, truth: list[float], windows: int,
                      seed: int, live_h: float) -> dict:
    """Does this cost model predict the measured sign of a scalar-h change?

    A counterfactual that cannot reproduce an experiment the machine has
    already run must not be used to size an experiment it has not.
    """
    out = {}
    for prompt, rows in fits.items():
        fit = central_fit(rows, 485 / 107 if prompt == "beagle" else 472 / 99)
        base = replay(fit, shipped_price(live_h), truth, windows, seed)
        entry = {"live": {"h": live_h,
                          "mean_draft_len": base["mean_draft_len"],
                          "us_per_token": base["us_per_token"]}}
        for candidate in sorted(H_RECEIPTS):
            other = replay(fit, shipped_price(candidate), truth, windows, seed)
            delta = paired_delta_pct(base, other)
            entry[str(candidate)] = {
                "mean_draft_len": other["mean_draft_len"],
                "mean_draft_len_delta": (other["mean_draft_len"]
                                         - base["mean_draft_len"]),
                "predicted_decode_pct": delta["decode_pct"],
                "predicted_decode_pct_sem": delta["decode_pct_sem"],
                "measured_decode_pct": H_RECEIPTS[candidate][
                    "candidate_decode_time_pct"],
            }
        out[prompt] = entry
    return out


# =========================================================================
# 5. Report
# =========================================================================

def boundary_table(constants: dict, ipg: dict, price_measured: Price,
                   predicate: dict) -> list[dict]:
    h = constants["headStepCostRatio"]
    rows = []
    for depth in range(E53.MAX_DEPTH):
        width_from, width_to = depth + 1, depth + 2
        qmv_step = E46_PER_ROW + (E46_PER_STREAM
                                  if crosses_stream_boundary(depth, ipg) else 0.0)
        rows.append({
            "depth_step": f"{depth}->{depth + 1}",
            "width_step": f"{width_from}->{width_to}",
            "streams": f"{streams(width_from, ipg)}->{streams(width_to, ipg)}",
            "qmv_boundary": crosses_stream_boundary(depth, ipg),
            "chunk_boundary": (sdpa_calls(width_to, 512, predicate)
                               > sdpa_calls(width_from, 512, predicate)),
            "scheduler_price": h,
            "e46_qmv_marginal_ms": qmv_step,
            "measured_marginal_ratio": price_measured.marginal(depth),
            "measured_marginal_us": E1_C_US[depth + 1] - E1_C_US[depth],
            "measured_over_scheduler": price_measured.marginal(depth) / h,
        })
    return rows


def chunk_predicate_price() -> dict:
    """Price the width-6 chunk predicate from the measured curve.

    Two estimators, because the d=5 row has N=2 and cannot carry the answer
    alone. Both are labelled as the cost of OUR predicate at width 6, not the
    cost of width 6.
    """
    m = {d: E1_C_US[d] - E1_C_US[d - 1] for d in range(1, 9)}
    direct = m[5] - statistics.fmean([m[6], m[7]])
    span = (E1_C_US[6] - E1_C_US[4]) - 2.0 * m[7]
    return {
        "measured_marginals_us": m,
        "direct_estimate_us": direct,
        "direct_estimate_note": "m(5) - mean(m(6), m(7)); m(5) rests on N = 2",
        "span_estimate_us": span,
        "span_estimate_note": ("C(6)-C(4) minus two within-tier chunked steps "
                               "priced at m(7); avoids the N = 2 row"),
        "within_tier_chunked_step_us": statistics.fmean([m[6], m[7]]),
        "as_fraction_of_depth0_round": {
            "direct": direct / E1_C_US[0],
            "span": span / E1_C_US[0],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=4000)
    parser.add_argument("--envelope-windows", type=int, default=400)
    parser.add_argument("--control-windows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    constants = parse_schedule_constants()
    ipg = parse_ipg_table()
    predicate = parse_chunk_predicate()
    host_route = parse_host_sdpa_route()

    # The port must agree with the live source or nothing below is valid.
    assert constants["sdpaWidthWallDepthCap"] == E53.SDPA_WIDTH_WALL_DEPTH_CAP
    assert constants["segmentedVerifyDepthCap"] == E53.SEGMENTED_VERIFY_DEPTH_CAP
    assert constants["segmentedStreakGate"] == E53.SEGMENTED_STREAK_GATE
    assert constants["acceptEMAAlpha"] == E53.EMA_ALPHA
    assert constants["headStepCostRatio"] == E53.HEAD_STEP_COST_RATIO
    for width, value in E53.IPG.items():
        if ipg.get(width) != value:
            raise RuntimeError(
                f"IPG table moved at width {width}: live {ipg.get(width)}, "
                f"E53 model {value}. The staircase is stale.")

    h = constants["headStepCostRatio"]
    head_ratio = E1_HEAD_STEP_US / E1_C_US[0]
    price_measured = measured_price()
    prices = {
        "shipped_scalar": shipped_price(h),
        "measured_e1": price_measured,
        "stream_aware": stream_aware_price(ipg, head_ratio, h),
    }
    chunk = chunk_predicate_price()
    prices["stream_aware_plus_chunk"] = stream_aware_price(
        ipg, head_ratio, h,
        chunk_surcharge=chunk["as_fraction_of_depth0_round"]["span"],
        chunk_from_depth=4)
    # Ablations, all pinned to the same mean price so only the shape differs.
    verify_shape = [price_measured.marginal(d) - head_ratio
                    for d in range(E53.MAX_DEPTH)]
    prices["measured_shape_pinned"] = pinned_shape_price(
        "measured_shape_pinned", verify_shape, head_ratio, h,
        "E1 measured verify shape, mean price pinned to the live h")
    for tag, depths in (("boundary45_only", (3,)), ("boundary89_only", (7,)),
                        ("both_boundaries", (3, 7))):
        shape = [1.0 + (E46_PER_STREAM / E46_PER_ROW if d in depths else 0.0)
                 for d in range(E53.MAX_DEPTH)]
        prices[tag] = pinned_shape_price(
            tag, shape, head_ratio, h,
            f"surcharge at depth steps {depths} only, mean pinned")

    truths = {
        "measured_e1": [value for value in E1_C_US],
        "stream_aware_model": [E1_C_US[0] * prices["stream_aware"].cost(d)
                               for d in range(E53.MAX_DEPTH + 1)],
    }
    for tag, ratio in (("ratio_minus_30pct", 0.7), ("ratio_plus_30pct", 1.3)):
        scaled = stream_aware_price(
            ipg, head_ratio, h,
            ratio=ratio * E46_PER_STREAM / E46_PER_ROW)
        prices[f"stream_aware_{tag}"] = scaled
        truths[f"stream_aware_model_{tag}"] = [
            E1_C_US[0] * scaled.cost(d) for d in range(E53.MAX_DEPTH + 1)]

    fits = load_burst_fits()
    report = {
        "live_constants": constants,
        "ipg_table": ipg,
        "chunk_predicate": predicate,
        "host_sdpa_route": host_route,
        "head_step_ratio_e1": head_ratio,
        "prices": {name: {"marginals": price.marginals,
                          "cumulative": price.cum,
                          "mean_marginal": price.mean_marginal(),
                          "note": price.note}
                   for name, price in prices.items()},
        "boundaries": boundary_table(constants, ipg, price_measured, predicate),
        "chunk_predicate_price": chunk,
        "null_floor_pct": NULL_FLOOR_PCT,
        "decode_share_of_leg": DECODE_SHARE_OF_LEG,
        "counterfactual": {},
    }

    report["scalar_validation"] = scalar_validation(
        fits, truths["measured_e1"], args.windows, args.seed, h)
    report["h_receipts"] = H_RECEIPTS

    targets = {"beagle": 485 / 107, "medicine": 472 / 99}
    report["fit_envelope"] = {}
    for prompt, rows in fits.items():
        envelope = []
        for fit in rows:
            base = replay(fit, prices["shipped_scalar"], truths["measured_e1"],
                          args.envelope_windows, args.seed)
            row = {"persistence": fit["persistence"], "q_easy": fit["q_easy"],
                   "share_easy": fit["share_easy"], "q_hard": fit["q_hard"],
                   "shipped_mean_draft_len": base["mean_draft_len"]}
            for policy in ("stream_aware", "measured_e1"):
                other = replay(fit, prices[policy], truths["measured_e1"],
                               args.envelope_windows, args.seed)
                row[policy] = paired_delta_pct(base, other)["decode_pct"]
                row[policy + "_mean_draft_len"] = other["mean_draft_len"]
            envelope.append(row)
        report["fit_envelope"][prompt] = envelope

    report["iso_depth_control"] = {}
    for prompt, rows in fits.items():
        fit = central_fit(rows, targets[prompt])
        block = {}
        for policy in ("stream_aware", "measured_e1", "measured_shape_pinned",
                       "boundary45_only", "boundary89_only", "both_boundaries"):
            shaped = replay(fit, prices[policy], truths["measured_e1"],
                            args.windows, args.seed)
            scalar, matched = match_scalar_depth(
                fit, shaped["mean_draft_len"], truths["measured_e1"],
                args.control_windows, args.seed)
            matched_full = replay(fit, shipped_price(scalar),
                                  truths["measured_e1"], args.windows,
                                  args.seed)
            block[policy] = {
                "shaped_mean_draft_len": shaped["mean_draft_len"],
                "matched_scalar_h": scalar,
                "matched_mean_draft_len": matched_full["mean_draft_len"],
                "shape_only_delta": paired_delta_pct(matched_full, shaped),
                "level_delta_vs_shipped": paired_delta_pct(
                    replay(fit, prices["shipped_scalar"],
                           truths["measured_e1"], args.windows, args.seed),
                    matched_full),
                "shaped_mixture": shaped["mixture"],
                "matched_mixture": matched_full["mixture"],
            }
        report["iso_depth_control"][prompt] = block

    leg_gains = {}
    for prompt, rows in fits.items():
        fit = central_fit(rows, targets[prompt])
        report["counterfactual"][prompt] = {"fit": {
            k: fit[k] for k in ("share_easy", "persistence", "q_easy", "q_hard",
                                "mean_draft_len", "accept_ratio", "rounds")}}
        for truth_name, truth in truths.items():
            base = replay(fit, prices["shipped_scalar"], truth,
                          args.windows, args.seed)
            entry = {"shipped": {k: v for k, v in base.items()
                                 if k != "per_window"}}
            for policy in ("measured_e1", "stream_aware",
                           "stream_aware_plus_chunk",
                           "stream_aware_ratio_minus_30pct",
                           "stream_aware_ratio_plus_30pct"):
                other = replay(fit, prices[policy], truth, args.windows,
                               args.seed)
                delta = paired_delta_pct(base, other)
                entry[policy] = {**{k: v for k, v in other.items()
                                    if k != "per_window"}, **delta}
            report["counterfactual"][prompt][truth_name] = entry
        primary = report["counterfactual"][prompt]["measured_e1"]
        leg_gains[prompt] = {
            policy: -primary[policy]["leg_pct"]
            for policy in ("measured_e1", "stream_aware",
                           "stream_aware_plus_chunk")}

    report["score"] = {}
    for policy in ("measured_e1", "stream_aware", "stream_aware_plus_chunk"):
        gains = {prompt: leg_gains[prompt][policy] for prompt in leg_gains}
        report["score"][policy] = {
            "leg_gain_pct": gains,
            "score_pct": Q.score_pct_from_leg_gains(gains),
        }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print_report(report)


def print_report(report: dict) -> None:
    print("E56 Step 0 — scalar h against the measured cost staircase\n")
    print("live constants:", json.dumps(report["live_constants"]))
    print("IPG table     :", json.dumps(report["ipg_table"]))
    print("chunk pred.   :", json.dumps(report["chunk_predicate"]))
    print("host route    :", json.dumps(report["host_sdpa_route"]))
    print()
    header = ("step  width  streams  qmv?  chunk?   sched     E46 ms   "
              "measured   meas/sched")
    print(header)
    for row in report["boundaries"]:
        print("%-5s %-6s %-8s %-5s %-7s %7.4f %9.3f %10.4f %9.2fx" % (
            row["depth_step"], row["width_step"], row["streams"],
            "yes" if row["qmv_boundary"] else "no",
            "yes" if row["chunk_boundary"] else "no",
            row["scheduler_price"], row["e46_qmv_marginal_ms"],
            row["measured_marginal_ratio"], row["measured_over_scheduler"]))
    print()
    chunk = report["chunk_predicate_price"]
    print("width-6 chunk predicate cost: direct %.0f µs, span %.0f µs "
          "(%.4f / %.4f of a depth-0 round)" % (
              chunk["direct_estimate_us"], chunk["span_estimate_us"],
              chunk["as_fraction_of_depth0_round"]["direct"],
              chunk["as_fraction_of_depth0_round"]["span"]))
    print()
    print("VALIDATION — does this cost model reproduce a measured scalar-h arm?")
    for prompt, entry in report["scalar_validation"].items():
        live = entry["live"]
        print("  %-9s live h=%.2f  mean draft %.3f  %.1f µs/token"
              % (prompt, live["h"], live["mean_draft_len"],
                 live["us_per_token"]))
        for key, values in entry.items():
            if key == "live":
                continue
            measured = values["measured_decode_pct"]
            print("    h=%-5s mean draft %.3f (%+.3f)  predicted decode "
                  "%+7.4f %% ±%.4f  measured %s"
                  % (key, values["mean_draft_len"],
                     values["mean_draft_len_delta"],
                     values["predicted_decode_pct"],
                     values["predicted_decode_pct_sem"],
                     "n/a" if measured is None else "%+.2f %%" % measured))
    print()
    print("ISO-DEPTH CONTROL — shaped price against the scalar at equal depth")
    for prompt, block in report["iso_depth_control"].items():
        for policy, values in block.items():
            print("  %-9s %-12s shaped draft %.3f vs matched scalar h=%.4f "
                  "(draft %.3f)" % (prompt, policy,
                                    values["shaped_mean_draft_len"],
                                    values["matched_scalar_h"],
                                    values["matched_mean_draft_len"]))
            print("            level (scalar retune) %+7.4f %% ±%.4f | "
                  "shape only %+7.4f %% ±%.4f"
                  % (values["level_delta_vs_shipped"]["decode_pct"],
                     values["level_delta_vs_shipped"]["decode_pct_sem"],
                     values["shape_only_delta"]["decode_pct"],
                     values["shape_only_delta"]["decode_pct_sem"]))
    print()
    print("FIT ENVELOPE — decode % under the measured truth, per feasible fit")
    for prompt, rows in report["fit_envelope"].items():
        values = [row["stream_aware"] for row in rows]
        measured = [row["measured_e1"] for row in rows]
        drafts = [row["shipped_mean_draft_len"] for row in rows]
        print("  %-9s n=%d  shipped mean draft %.3f–%.3f  stream_aware "
              "%+.4f..%+.4f %%  measured_e1 %+.4f..%+.4f %%"
              % (prompt, len(rows), min(drafts), max(drafts),
                 min(values), max(values), min(measured), max(measured)))
    print()
    for prompt, block in report["counterfactual"].items():
        print(f"--- {prompt} ---")
        for truth_name, entry in block.items():
            if truth_name == "fit":
                continue
            base = entry["shipped"]
            print("  truth %-32s shipped %.1f µs/token  M̄ %.3f  R %.1f"
                  % (truth_name, base["us_per_token"],
                     base["mean_draft_len"] + 1.0, base["rounds"]))
            for policy, values in entry.items():
                if policy == "shipped":
                    continue
                print("    %-32s decode %+7.4f %% ±%.4f  leg %+7.4f %%  "
                      "M̄ %.3f  A/D %.4f" % (
                          policy, values["decode_pct"],
                          values["decode_pct_sem"], values["leg_pct"],
                          values["mean_draft_len"] + 1.0,
                          values["accept_ratio"]))
    print()
    for policy, block in report["score"].items():
        print("score %-28s %+8.5f %%   leg gains %s" % (
            policy, block["score_pct"],
            {k: round(v, 5) for k, v in block["leg_gain_pct"].items()}))
    print("\nnull floor %.4f %% (leg)" % report["null_floor_pct"])
    print("written:", OUT)


if __name__ == "__main__":
    main()
