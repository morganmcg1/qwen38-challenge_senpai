#!/usr/bin/env python3
"""E53 Part 1: the scored verify-width mixture, from the shipped schedule.

WHAT THIS REPLACES. The assignment proposed a one-parameter truncated geometric
in a per-token acceptance rate `alpha`, with verify width `M = k + 1` where `k`
is the previous round's accepted count. That mapping is refuted by our own
source at base 45b7c6a4: `Qwen36MTPBlockSession.costModelDepth` is a greedy
marginal-cost walk over per-position acceptance EMAs under a streak-gated width
cap, and `effective_mean_draft_len` counts drafts PROPOSED (ledger 153, MT1).
The refuted null is still simulated here, as `truncated_geometric_mixture`, so
the comparison is quantitative rather than rhetorical.

WHAT THIS IS. An exact Python port of the shipped schedule plus a one-parameter
acceptance model. The port's constants all come from
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift` at base 45b7c6a4 and from
`Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift` (the offer and the
tail clamp). The single free parameter is fitted to the ONE published per-prompt
number that constrains it, `effective_mean_draft_len`. Everything else the model
emits -- the width mixture, the accept ratio A/D, the round count R, the
non-drafting round count -- is a prediction, not a fit.

IDENTIFICATION, NOT SAMPLING. Greedy decode is deterministic, so a fixed
(schedule, head, prompt) triple has ONE width sequence and no sampling
distribution; `research/e53_board_facts.py` shows 152 content-distinct board
trees sharing beagle 485/107 and medicine 472/99 to 16 digits. The interval on
every share below is therefore an IDENTIFICATION interval over the
source-faithful model family, not a standard error over board rows.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics

OUT = pathlib.Path(__file__).resolve().parent / "e53-width-mixture.json"

# ---- shipped schedule constants, Qwen36MTPBlockSession.swift @ 45b7c6a4 ----
MAX_DEPTH = 8                      # Qwen36MTPLimits.maxDepth
EMA_PRIOR = [0.85 * 0.98 ** i for i in range(MAX_DEPTH)]   # line 635
EMA_ALPHA = 0.15                   # acceptEMAAlpha, line 637
HEAD_STEP_COST_RATIO = 0.18        # headStepCostRatio, line 668
SDPA_WIDTH_WALL_DEPTH_CAP = 5      # line 700
SEGMENTED_VERIFY_DEPTH_CAP = 8     # line 707
SEGMENTED_STREAK_GATE = 2          # line 735
OPTIMISM_CAP = 0.95                # line 837
OFFERED_DEPTH = 8                  # ranked `mtp_depth`, board officialMetrics
TOTAL_TOKENS = 512                 # ranked decode window

# ---- cost weights: thorfinn E46 refit, PR #51, base 01f69e18, Apple M4 Pro ---
# T(M) = 16.757 + 27.532 * streams(M) + 9.624 * M, max|resid| 0.770 ms.
# streams(M) = ceil(M / IPG(M)) read from the shipped dispatch switch in
# Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h @ 1917.
# M=1 and M=2 do not use the `_m<T,M,IPG>` cells at all (M=1 falls through to
# qmv_fast_impl, M=2 uses qmv_fast_crossrow_affine4_g64<T,2>), so their T is an
# EXTRAPOLATION of the fit and is marked as such wherever it is used.
IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}
E46_INTERCEPT, E46_PER_STREAM, E46_PER_ROW = 16.757, 27.532, 9.624


def streams(width: int) -> int:
    if width in IPG:
        return math.ceil(width / IPG[width])
    if width in (1, 2):
        return 1
    raise ValueError(f"width {width} is outside the dispatch table")


def cost_ms(width: int) -> float:
    return E46_INTERCEPT + E46_PER_STREAM * streams(width) + E46_PER_ROW * width


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Schedule:
    """Exact port of costModelDepth + recordAcceptOutcome."""

    def __init__(self) -> None:
        self.ema = list(EMA_PRIOR)
        self.streak = 0

    def depth(self, offered: int, margin: float | None) -> int:
        width_cap = (SEGMENTED_VERIFY_DEPTH_CAP
                     if self.streak >= SEGMENTED_STREAK_GATE
                     else SDPA_WIDTH_WALL_DEPTH_CAP)
        cap = min(min(offered, MAX_DEPTH), width_cap)
        if cap <= 0:
            return 0
        h = HEAD_STEP_COST_RATIO
        reach, expected, depth = 1.0, 0.0, 0
        while depth < cap:
            p = self.ema[depth]
            if margin is not None:
                if depth == 0:
                    p = min(p, sigmoid(margin / 2.0))
                elif depth == 1:
                    p = min(p, sigmoid(margin / 3.0))
            reach *= p
            threshold = h * (1.0 + expected) / (1.0 + depth * h)
            if not reach > threshold:
                break
            expected += reach
            depth += 1
        return depth

    def record(self, accepted: int, drafted: int) -> None:
        """recordAcceptOutcome; stop tokens are not modelled (see caveats)."""
        for index in range(min(accepted, len(self.ema))):
            self.ema[index] += EMA_ALPHA * (1.0 - self.ema[index])
        if accepted < drafted and accepted < len(self.ema):
            self.ema[accepted] += EMA_ALPHA * (0.0 - self.ema[accepted])
        elif accepted == drafted and drafted > 0 and accepted < len(self.ema):
            if self.ema[accepted] < OPTIMISM_CAP:
                self.ema[accepted] += EMA_ALPHA * (OPTIMISM_CAP - self.ema[accepted])
        self.streak = self.streak + 1 if accepted == drafted else 0


def accept_probs(q: float, decay: float) -> list[float]:
    return [min(1.0, q * decay ** i) for i in range(MAX_DEPTH)]


class BurstAcceptance:
    """Two-state acceptance: prose is locally easy or locally hard.

    The IID family below is refuted by the published pair (n, A/D): see
    `iid_frontier`. The failure direction says acceptance must be positively
    correlated with the depth the schedule chooses, which is what a persistent
    easy/hard state plus an informative top-2 margin produces.

    `share_easy` is the stationary probability of the easy state and
    `persistence` interpolates from independent rounds (0) to sticky runs (1).
    The two margin levels are FIXED, not fitted: an easy round's pending primary
    is assumed unambiguous and a hard round's nearly tied.
    """

    MARGIN_EASY = 10.0
    MARGIN_HARD = 0.5

    def __init__(self, share_easy: float, persistence: float,
                 q_easy: float, q_hard: float) -> None:
        self.share_easy = share_easy
        self.persistence = persistence
        self.q_easy = q_easy
        self.q_hard = q_hard
        self.easy = True

    def step(self, rng: random.Random) -> tuple[list[float], float]:
        stay = self.persistence + (1.0 - self.persistence) * (
            self.share_easy if self.easy else 1.0 - self.share_easy)
        if rng.random() >= stay:
            self.easy = not self.easy
        q = self.q_easy if self.easy else self.q_hard
        margin = self.MARGIN_EASY if self.easy else self.MARGIN_HARD
        return [q] * MAX_DEPTH, margin


def run_burst_window(model: BurstAcceptance, rng: random.Random) -> dict:
    sched = Schedule()
    emitted = 0
    widths: dict[int, int] = {}
    drafted_total = accepted_total = rounds = non_drafting = 0
    while emitted < TOTAL_TOKENS:
        remaining = TOTAL_TOKENS - emitted
        offered = max(1, min(OFFERED_DEPTH, MAX_DEPTH, remaining - 1))
        probs, margin = model.step(rng)
        depth = sched.depth(offered, margin)
        accepted = 0
        for index in range(depth):
            if rng.random() < probs[index]:
                accepted += 1
            else:
                break
        rounds += 1
        drafted_total += depth
        accepted_total += accepted
        if depth == 0:
            non_drafting += 1
        widths[depth + 1] = widths.get(depth + 1, 0) + 1
        emitted += 1 + accepted
        sched.record(accepted, depth)
        if rounds > 4 * TOTAL_TOKENS:
            raise RuntimeError("schedule failed to close the window")
    return {
        "rounds": rounds,
        "drafted": drafted_total,
        "accepted": accepted_total,
        "non_drafting": non_drafting,
        "mean_draft_len": drafted_total / rounds,
        "accept_ratio": accepted_total / drafted_total if drafted_total else 0.0,
        "widths": widths,
    }


def burst_aggregate(share_easy: float, persistence: float, q_easy: float,
                    q_hard: float, windows: int, seed: int) -> dict:
    rng = random.Random(seed)
    runs = []
    for _ in range(windows):
        model = BurstAcceptance(share_easy, persistence, q_easy, q_hard)
        model.easy = rng.random() < share_easy
        runs.append(run_burst_window(model, rng))
    widths: dict[int, int] = {}
    for run in runs:
        for width, count in run["widths"].items():
            widths[width] = widths.get(width, 0) + count
    total_rounds = sum(widths.values())
    drafted = sum(r["drafted"] for r in runs)
    accepted = sum(r["accepted"] for r in runs)
    return {
        "mean_draft_len": drafted / total_rounds,
        "accept_ratio": accepted / drafted if drafted else 0.0,
        "rounds": statistics.mean(r["rounds"] for r in runs),
        "non_drafting": statistics.mean(r["non_drafting"] for r in runs),
        "mixture": {w: widths[w] / total_rounds for w in sorted(widths)},
    }


def run_window(q: float, decay: float, margin_mean: float | None,
               rng: random.Random) -> dict:
    """One 512-token ranked decode window under the shipped schedule."""
    probs = accept_probs(q, decay)
    sched = Schedule()
    emitted = 0
    widths: dict[int, int] = {}
    drafted_total = accepted_total = rounds = non_drafting = 0
    while emitted < TOTAL_TOKENS:
        remaining = TOTAL_TOKENS - emitted
        offered = max(1, min(OFFERED_DEPTH, MAX_DEPTH, remaining - 1))
        margin = rng.expovariate(1.0 / margin_mean) if margin_mean else None
        depth = sched.depth(offered, margin)
        accepted = 0
        for index in range(depth):
            if rng.random() < probs[index]:
                accepted += 1
            else:
                break
        rounds += 1
        drafted_total += depth
        accepted_total += accepted
        if depth == 0:
            non_drafting += 1
        widths[depth + 1] = widths.get(depth + 1, 0) + 1
        emitted += 1 + accepted
        sched.record(accepted, depth)
        if rounds > 4 * TOTAL_TOKENS:
            raise RuntimeError("schedule failed to close the window")
    return {
        "rounds": rounds,
        "drafted": drafted_total,
        "accepted": accepted_total,
        "non_drafting": non_drafting,
        "mean_draft_len": drafted_total / rounds,
        "accept_ratio": accepted_total / drafted_total if drafted_total else 0.0,
        "widths": widths,
    }


def aggregate(q: float, decay: float, margin_mean: float | None,
              windows: int, seed: int) -> dict:
    rng = random.Random(seed)
    runs = [run_window(q, decay, margin_mean, rng) for _ in range(windows)]
    widths: dict[int, int] = {}
    for run in runs:
        for width, count in run["widths"].items():
            widths[width] = widths.get(width, 0) + count
    total_rounds = sum(widths.values())
    mixture = {w: widths[w] / total_rounds for w in sorted(widths)}
    return {
        "mean_draft_len": statistics.mean(r["mean_draft_len"] for r in runs),
        "mean_draft_len_sd": statistics.pstdev(r["mean_draft_len"] for r in runs),
        "accept_ratio": statistics.mean(r["accept_ratio"] for r in runs),
        "rounds": statistics.mean(r["rounds"] for r in runs),
        "rounds_sd": statistics.pstdev(r["rounds"] for r in runs),
        "non_drafting": statistics.mean(r["non_drafting"] for r in runs),
        "mixture": mixture,
        "per_window": runs,
    }


def fit_q(target_mean: float, decay: float, margin_mean: float | None,
          windows: int, seed: int) -> tuple[float, dict]:
    """Bisect the acceptance parameter against the one published constraint."""
    low, high = 0.01, 0.999
    result = None
    for _ in range(40):
        mid = 0.5 * (low + high)
        result = aggregate(mid, decay, margin_mean, windows, seed)
        if result["mean_draft_len"] < target_mean:
            low = mid
        else:
            high = mid
        if abs(result["mean_draft_len"] - target_mean) < 1e-4:
            break
    return 0.5 * (low + high), result


def iid_frontier(windows: int, seed: int) -> list[dict]:
    """(mean draft length, accept ratio, round count) reachable under IID q."""
    curve = []
    for step in range(5, 100, 5):
        q = step / 100.0
        run = aggregate(q, 1.0, None, windows, seed)
        curve.append({
            "q": q,
            "mean_draft_len": run["mean_draft_len"],
            "accept_ratio": run["accept_ratio"],
            "rounds": run["rounds"],
        })
    return curve


def legal_round_counts(numerator: int, denominator: int) -> list[dict]:
    """R is pinned to multiples of the reduced denominator of D/R.

    `effective_mean_draft_len` is an exact rational; with 512 = R + A the pair
    (R, A) is therefore restricted to a short list, and each member implies one
    accept ratio. Ledger 153 picked the smallest under a monotonicity
    assumption; this function states the whole list instead.
    """
    out = []
    multiple = 1
    while True:
        rounds = denominator * multiple
        drafts = numerator * multiple
        accepted = TOTAL_TOKENS - rounds
        if accepted <= 0:
            break
        out.append({
            "rounds": rounds,
            "drafts": drafts,
            "accepted": accepted,
            "accept_ratio": accepted / drafts,
            "accepted_per_round": accepted / rounds,
        })
        multiple += 1
    return out


def solve_burst(target_mean: float, target_ratio: float, persistence: float,
                q_easy: float, windows: int, seed: int) -> dict | None:
    """Match (mean draft length, accept ratio) by nested bisection."""

    def match_mean(share_easy: float) -> tuple[float, dict]:
        low, high = 0.0, q_easy
        run = None
        for _ in range(24):
            mid = 0.5 * (low + high)
            run = burst_aggregate(share_easy, persistence, q_easy, mid,
                                  windows, seed)
            if run["mean_draft_len"] < target_mean:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high), run

    lo_share, hi_share = 0.05, 0.98
    _, lo_run = match_mean(lo_share)
    _, hi_run = match_mean(hi_share)
    if not (min(lo_run["accept_ratio"], hi_run["accept_ratio"]) <= target_ratio
            <= max(lo_run["accept_ratio"], hi_run["accept_ratio"])):
        return None
    rising = hi_run["accept_ratio"] > lo_run["accept_ratio"]
    best = None
    for _ in range(18):
        mid_share = 0.5 * (lo_share + hi_share)
        q_hard, run = match_mean(mid_share)
        best = {"share_easy": mid_share, "q_hard": q_hard, "q_easy": q_easy,
                "persistence": persistence, **run}
        above = run["accept_ratio"] > target_ratio
        if above == rising:
            hi_share = mid_share
        else:
            lo_share = mid_share
    return best


def cost_shares(mixture: dict[int, float]) -> dict[str, float]:
    total = sum(share * cost_ms(width) for width, share in mixture.items())
    blocks = {"f456": (4, 5, 6), "f78": (7, 8), "f9": (9,), "f123": (1, 2, 3)}
    out = {}
    for name, widths in blocks.items():
        out[name] = sum(mixture.get(w, 0.0) * cost_ms(w) for w in widths) / total
    out["mean_cost_ms"] = total
    return out


def truncated_geometric_mixture(alpha: float) -> dict[int, float]:
    """The assignment's refuted null: M = k + 1, k ~ truncated geometric."""
    probs = {}
    for k in range(MAX_DEPTH + 1):
        probs[k + 1] = ((1 - alpha) * alpha ** k if k < MAX_DEPTH
                        else alpha ** MAX_DEPTH)
    return probs


def selftest() -> None:
    """Cheap gates on the port, with a positive control that must fail."""
    sched = Schedule()
    sched.ema = [1.0] * MAX_DEPTH
    assert sched.depth(8, None) == 5, "saturated EMAs must run to the cold cap"
    sched.streak = SEGMENTED_STREAK_GATE
    assert sched.depth(8, None) == 8, "a qualified streak must open cap 8"
    sched.ema = [0.05] * MAX_DEPTH
    assert sched.depth(8, None) == 0, "a hopeless head must stop drafting"
    dead = Schedule()
    dead.ema = [0.10] * MAX_DEPTH
    dead.record(0, 0)
    assert dead.ema == [0.10] * MAX_DEPTH, "a non-drafting round updates nothing"
    lo = aggregate(0.60, 1.0, None, 40, 11)["mean_draft_len"]
    hi = aggregate(0.95, 1.0, None, 40, 11)["mean_draft_len"]
    assert lo < hi, "mean draft length must increase with acceptance"
    assert cost_ms(9) > cost_ms(8) > cost_ms(7), "T(M) must rise with width"
    assert streams(5) == 2 and streams(9) == 3, "stream vector must match source"
    # positive control: a wrong cap must break the first gate.
    broken = Schedule()
    broken.ema = [1.0] * MAX_DEPTH
    global SDPA_WIDTH_WALL_DEPTH_CAP
    keep, SDPA_WIDTH_WALL_DEPTH_CAP = SDPA_WIDTH_WALL_DEPTH_CAP, 4
    fired = broken.depth(8, None) != 5
    SDPA_WIDTH_WALL_DEPTH_CAP = keep
    assert fired, "the cap gate cannot fail, so it proves nothing"
    print("selftest: 8 gates passed, 1 positive control fired")


# Published per-prompt constants. Tree: our promoted row `2b0c36a0`
# (score 3.232508, head 559b24eb), whose beagle/medicine draft lengths are the
# modal behavioural class on the board (152 content-distinct trees).
PROMPTS = {
    "beagle": {"mean_draft_len": 485 / 107, "non_drafting": 0,
               "ledger153": {"R": 107, "D": 485, "A": 405, "alpha": 405 / 485}},
    "medicine": {"mean_draft_len": 472 / 99, "non_drafting": 0,
                 "ledger153": {"R": 99, "D": 472, "A": 413, "alpha": 413 / 472}},
}
# Marginal score weights for OUR tree's central pair (advisor, order statistics).
WEIGHTS = {"beagle": 0.483694, "medicine": 0.516306}
# Model family used as the identification interval.
VARIANTS = [
    {"name": "A_flat", "decay": 1.0, "margin_mean": None},
    {"name": "B_decay_098", "decay": 0.98, "margin_mean": None},
    {"name": "C_decay_095", "decay": 0.95, "margin_mean": None},
    {"name": "D_margin_8", "decay": 1.0, "margin_mean": 8.0},
    {"name": "E_margin_4", "decay": 1.0, "margin_mean": 4.0},
    {"name": "F_margin_2", "decay": 1.0, "margin_mean": 2.0},
]


BURST_GRID = [
    {"persistence": p, "q_easy": q}
    for p in (0.0, 0.5, 0.9)
    for q in (0.94, 0.95, 0.96, 0.97, 0.98)
]


def boundary_sensitivity(solution: dict, windows: int, seed: int) -> list[dict]:
    """How steeply does the M in {7,8} / M = 9 split move off the fit?

    The depth-7 extension test sits close to equality at the fitted acceptance
    level, so the published constraints pin the {7,8} + {9} block far better
    than they pin the split inside it. This table quantifies that.
    """
    rows = []
    for delta in (-0.02, -0.01, 0.0, 0.01, 0.02):
        q_hard = solution["q_hard"] + delta
        if not 0.0 < q_hard < 1.0:
            continue
        run = burst_aggregate(solution["share_easy"], solution["persistence"],
                              solution["q_easy"], q_hard, windows, seed)
        shares = cost_shares(run["mixture"])
        rows.append({
            "q_hard_delta": delta,
            "mean_draft_len": run["mean_draft_len"],
            "accept_ratio": run["accept_ratio"],
            "f78": shares["f78"],
            "f9": shares["f9"],
            "f78_plus_f9": shares["f78"] + shares["f9"],
            "f456": shares["f456"],
        })
    return rows


def burst_section(report: dict, windows: int, seed: int) -> None:
    """Feasible set of burst models that match BOTH published constraints."""
    report["burst"] = {}
    for prompt, published in PROMPTS.items():
        target_mean = published["mean_draft_len"]
        target_ratio = published["ledger153"]["alpha"]
        solutions = []
        for point in BURST_GRID:
            found = solve_burst(target_mean, target_ratio, point["persistence"],
                                point["q_easy"], windows, seed)
            if found is None:
                solutions.append({**point, "feasible": False})
                continue
            shares = cost_shares(found["mixture"])
            solutions.append({
                **point,
                "feasible": True,
                "share_easy": found["share_easy"],
                "q_hard": found["q_hard"],
                "mean_draft_len": found["mean_draft_len"],
                "accept_ratio": found["accept_ratio"],
                "rounds": found["rounds"],
                "non_drafting": found["non_drafting"],
                "mixture": found["mixture"],
                "shares": shares,
            })
        report["burst"][prompt] = solutions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--burst-windows", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--plutarch-mean", type=float, default=None,
                        help="fit plutarch too, to test the absorbing state")
    args = parser.parse_args()
    selftest()

    report: dict = {"variants": {}, "prompts": {}, "weights": WEIGHTS}
    report["iid_frontier"] = iid_frontier(args.windows, args.seed)
    report["legal_round_counts"] = {
        "beagle": legal_round_counts(485, 107),
        "medicine": legal_round_counts(472, 99),
    }
    for variant in VARIANTS:
        report["variants"][variant["name"]] = variant

    for prompt, published in PROMPTS.items():
        report["prompts"][prompt] = {"published": published, "fits": {}}
        for variant in VARIANTS:
            q, fitted = fit_q(published["mean_draft_len"], variant["decay"],
                              variant["margin_mean"], args.windows, args.seed)
            shares = cost_shares(fitted["mixture"])
            report["prompts"][prompt]["fits"][variant["name"]] = {
                "q": q,
                "mean_draft_len": fitted["mean_draft_len"],
                "mean_draft_len_sd": fitted["mean_draft_len_sd"],
                "predicted_accept_ratio": fitted["accept_ratio"],
                "predicted_rounds": fitted["rounds"],
                "predicted_rounds_sd": fitted["rounds_sd"],
                "predicted_non_drafting": fitted["non_drafting"],
                "mixture": fitted["mixture"],
                "shares": shares,
            }

    burst_section(report, args.burst_windows, args.seed)
    feasible_beagle = [s for s in report["burst"]["beagle"] if s["feasible"]]
    if feasible_beagle:
        report["boundary_sensitivity"] = boundary_sensitivity(
            feasible_beagle[len(feasible_beagle) // 2], args.burst_windows,
            args.seed)

    # Composites over the feasible burst set, at both weightings.
    report["burst_composite"] = {}
    for label, weights in (("marginal_483_517", WEIGHTS),
                           ("advisor_wrong_79_21",
                            {"beagle": 0.79, "medicine": 0.21})):
        rows = []
        beagle_solutions = {(s["persistence"], s["q_easy"]): s
                            for s in report["burst"]["beagle"] if s["feasible"]}
        medicine_solutions = {(s["persistence"], s["q_easy"]): s
                              for s in report["burst"]["medicine"] if s["feasible"]}
        for key in sorted(set(beagle_solutions) & set(medicine_solutions)):
            per_prompt = {"beagle": beagle_solutions[key]["shares"],
                          "medicine": medicine_solutions[key]["shares"]}
            total = sum(weights[p] * per_prompt[p]["mean_cost_ms"] for p in PROMPTS)
            rows.append({
                "persistence": key[0],
                "q_easy": key[1],
                **{
                    field: sum(
                        weights[p] * per_prompt[p][field] * per_prompt[p]["mean_cost_ms"]
                        for p in PROMPTS) / total
                    for field in ("f456", "f78", "f9", "f123")
                },
            })
        report["burst_composite"][label] = rows

    # Identification envelope: the nuisance parameters need not be shared across
    # prompts, so every cross-product of feasible per-prompt fits is admissible.
    envelope = {}
    beagle_feasible = [s for s in report["burst"]["beagle"] if s["feasible"]]
    medicine_feasible = [s for s in report["burst"]["medicine"] if s["feasible"]]
    for field in ("f456", "f78", "f9", "f123"):
        values = []
        for b in beagle_feasible:
            for m in medicine_feasible:
                per_prompt = {"beagle": b["shares"], "medicine": m["shares"]}
                total = sum(WEIGHTS[p] * per_prompt[p]["mean_cost_ms"] for p in PROMPTS)
                values.append(sum(
                    WEIGHTS[p] * per_prompt[p][field] * per_prompt[p]["mean_cost_ms"]
                    for p in PROMPTS) / total)
        envelope[field] = {"low": min(values), "high": max(values)}
    envelope["pairs"] = len(beagle_feasible) * len(medicine_feasible)
    report["burst_identification_envelope"] = envelope

    # Score-weighted composites, plus the wrong 79/21 weights for comparison.
    for label, weights in (("marginal_483_517", WEIGHTS),
                           ("advisor_wrong_79_21",
                            {"beagle": 0.79, "medicine": 0.21})):
        composite = {}
        for variant in VARIANTS:
            name = variant["name"]
            per_prompt = {
                p: report["prompts"][p]["fits"][name]["shares"] for p in PROMPTS
            }
            # Cost-weighted composite: blocks share the same denominator only
            # after weighting each prompt's QMV cost by its score weight.
            total = sum(weights[p] * per_prompt[p]["mean_cost_ms"] for p in PROMPTS)
            composite[name] = {
                key: sum(weights[p] * per_prompt[p][key] * per_prompt[p]["mean_cost_ms"]
                         for p in PROMPTS) / total
                for key in ("f456", "f78", "f9", "f123")
            }
        report[f"composite_{label}"] = composite

    if args.plutarch_mean is not None:
        q, fitted = fit_q(args.plutarch_mean, 1.0, None, args.windows, args.seed)
        report["plutarch"] = {
            "published_mean_draft_len": args.plutarch_mean,
            "q": q,
            "predicted_non_drafting": fitted["non_drafting"],
            "predicted_rounds": fitted["rounds"],
            "mixture": fitted["mixture"],
        }

    report["null_truncated_geometric"] = {
        prompt: {
            "alpha": PROMPTS[prompt]["ledger153"]["alpha"],
            "mixture": truncated_geometric_mixture(
                PROMPTS[prompt]["ledger153"]["alpha"]),
            "implied_mean_draft_len": sum(
                (w - 1) * s for w, s in truncated_geometric_mixture(
                    PROMPTS[prompt]["ledger153"]["alpha"]).items()),
            "published_mean_draft_len": PROMPTS[prompt]["mean_draft_len"],
            "shares": cost_shares(truncated_geometric_mixture(
                PROMPTS[prompt]["ledger153"]["alpha"])),
        }
        for prompt in PROMPTS
    }

    with OUT.open("w") as handle:
        json.dump(report, handle, indent=1)

    for prompt in PROMPTS:
        print(f"\n=== {prompt}  published n={PROMPTS[prompt]['mean_draft_len']:.6f} "
              f"alpha(ledger153)={PROMPTS[prompt]['ledger153']['alpha']:.4f} "
              f"R={PROMPTS[prompt]['ledger153']['R']}")
        for variant in VARIANTS:
            fit = report["prompts"][prompt]["fits"][variant["name"]]
            mix = " ".join(f"{w}:{fit['mixture'][w]:.3f}" for w in sorted(fit["mixture"]))
            print(f"  {variant['name']:<13} q={fit['q']:.4f} n={fit['mean_draft_len']:.4f} "
                  f"A/D={fit['predicted_accept_ratio']:.4f} R={fit['predicted_rounds']:.1f} "
                  f"nd={fit['predicted_non_drafting']:.2f}")
            print(f"                f456={fit['shares']['f456']:.4f} "
                  f"f78={fit['shares']['f78']:.4f} f9={fit['shares']['f9']:.4f} "
                  f"f123={fit['shares']['f123']:.4f} | rho {mix}")
    print("\n=== IID frontier: no q reproduces the published pair")
    for point in report["iid_frontier"]:
        print(f"  q={point['q']:.2f} n={point['mean_draft_len']:.4f} "
              f"A/D={point['accept_ratio']:.4f} R={point['rounds']:.1f}")
    for prompt in PROMPTS:
        legal = report["legal_round_counts"][prompt]
        print(f"  {prompt} legal (R, A, A/D): "
              + "; ".join(f"({row['rounds']}, {row['accepted']}, "
                          f"{row['accept_ratio']:.4f})" for row in legal))

    print("\n=== burst models that match BOTH published constraints")
    for prompt in PROMPTS:
        print(f"  -- {prompt}")
        for row in report["burst"][prompt]:
            if not row["feasible"]:
                print(f"     persistence={row['persistence']:.2f} "
                      f"q_easy={row['q_easy']:.2f}  INFEASIBLE")
                continue
            mix = " ".join(f"{w}:{row['mixture'][w]:.3f}" for w in sorted(row["mixture"]))
            print(f"     persistence={row['persistence']:.2f} q_easy={row['q_easy']:.2f} "
                  f"-> share_easy={row['share_easy']:.3f} q_hard={row['q_hard']:.3f} "
                  f"n={row['mean_draft_len']:.4f} A/D={row['accept_ratio']:.4f} "
                  f"R={row['rounds']:.1f} nd={row['non_drafting']:.2f}")
            print(f"        f456={row['shares']['f456']:.4f} f78={row['shares']['f78']:.4f} "
                  f"f9={row['shares']['f9']:.4f} f123={row['shares']['f123']:.4f} | rho {mix}")
    if "boundary_sensitivity" in report:
        print("\n=== how steeply the {7,8} / {9} split moves off the fit (beagle)")
        for row in report["boundary_sensitivity"]:
            print(f"  d q_hard={row['q_hard_delta']:+.2f} n={row['mean_draft_len']:.4f} "
                  f"A/D={row['accept_ratio']:.4f} f456={row['f456']:.4f} "
                  f"f78={row['f78']:.4f} f9={row['f9']:.4f} "
                  f"f78+f9={row['f78_plus_f9']:.4f}")
    if "plutarch" in report:
        plutarch = report["plutarch"]
        mix = " ".join(f"{w}:{plutarch['mixture'][w]:.3f}"
                       for w in sorted(plutarch["mixture"]))
        print("\n=== plutarch, the absorbing-state control")
        print(f"  published n={plutarch['published_mean_draft_len']:.6f} "
              f"fitted q={plutarch['q']:.4f} predicted non-drafting rounds="
              f"{plutarch['predicted_non_drafting']:.1f} "
              f"predicted R={plutarch['predicted_rounds']:.1f}")
        print(f"  rho {mix}")

    print("\n=== burst composite at marginal weights 0.483694 / 0.516306")
    for row in report["burst_composite"]["marginal_483_517"]:
        print(f"  persistence={row['persistence']:.2f} q_easy={row['q_easy']:.2f} "
              f"f456={row['f456']:.4f} f78={row['f78']:.4f} f9={row['f9']:.4f} "
              f"f123={row['f123']:.4f}")
    print("=== burst composite at the wrong 0.79 / 0.21")
    for row in report["burst_composite"]["advisor_wrong_79_21"]:
        print(f"  persistence={row['persistence']:.2f} q_easy={row['q_easy']:.2f} "
              f"f456={row['f456']:.4f} f78={row['f78']:.4f} f9={row['f9']:.4f} "
              f"f123={row['f123']:.4f}")

    print("\n=== IID composite at marginal weights 0.483694 / 0.516306")
    for name, comp in report["composite_marginal_483_517"].items():
        print(f"  {name:<13} f456={comp['f456']:.4f} f78={comp['f78']:.4f} "
              f"f9={comp['f9']:.4f} f123={comp['f123']:.4f}")
    print("=== composite at the advisor's wrong 0.79 / 0.21")
    for name, comp in report["composite_advisor_wrong_79_21"].items():
        print(f"  {name:<13} f456={comp['f456']:.4f} f78={comp['f78']:.4f} "
              f"f9={comp['f9']:.4f} f123={comp['f123']:.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
