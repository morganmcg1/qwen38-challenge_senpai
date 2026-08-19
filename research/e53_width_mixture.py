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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--plutarch-mean", type=float, default=None,
                        help="fit plutarch too, to test the absorbing state")
    args = parser.parse_args()
    selftest()

    report: dict = {"variants": {}, "prompts": {}, "weights": WEIGHTS}
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
    print("\n=== composite at marginal weights 0.483694 / 0.516306")
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
