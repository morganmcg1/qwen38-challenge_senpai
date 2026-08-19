#!/usr/bin/env python3
"""E56 deliverable: the verify-width mixture at the two score-setting prompts.

WHY. The published score is the mean of the 4th and 5th order statistics over
eight prompts, and receipt `ca9251b8` shows those are `beagle` (107 rounds, 485
proposed, 405 accepted) and `medicine` (99, 472, 413). Every roadmap item that
is priced from "M = 9's share of candidate-leg QMV time" has so far been priced
from the local public fixture, whose mean draft length is 6.269. Neither scored
prompt is near that. The advisor's LP bound from the two published moments is
[0, 70.34 %] on beagle and [0, 67.12 %] on medicine, which prices nothing.

WHAT THIS IS. The shipped schedule (`costModelDepth` + `recordAcceptOutcome`,
ported in `e53_width_mixture.py`) driven by the E53 two-state acceptance fits
that reproduce each prompt's published moments, run to a per-round verify-width
histogram and then converted to a QMV *time* share with thorfinn's E46 fit.

IDENTIFICATION, NOT SAMPLING. Greedy decode is deterministic, so each prompt
supplies ONE width sequence. Every interval below is an identification interval
over the feasible model family, not a standard error. The Monte-Carlo error of
each individual fit is reported separately so the two are never confused.

WHAT IS FIXED AND WHAT IS FREE.
  fixed   `segmentedStreakGate` = 2, `sdpaWidthWallDepthCap` = 5,
          `segmentedVerifyDepthCap` = 8, `headStepCostRatio` = 0.18, the EMA
          rates and the top-2 blends -- all parsed from live source, never
          restated, and never retuned here.
  fixed   the published per-prompt moments (rounds, proposed, accepted).
  free    the four acceptance parameters, swept in E53 and carried here as the
          feasible set.

The width cap gate is load-bearing and is implemented faithfully: a width-9
round needs `fullAcceptStreak >= 2`, so the deep tail is rare by construction
rather than by assumption.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e53_width_mixture as E53
import e56_stream_aware_schedule as E56

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "e56-ranked-width-mixture.json"

# ---- receipt ca9251b8, exact per-prompt reconstruction --------------------
RECEIPT = {
    "plutarch": (487, 75, 25),
    "drama": (252, 579, 260),
    "travel": (212, 563, 300),
    "beagle": (107, 485, 405),
    "medicine": (99, 472, 413),
    "republic": (89, 469, 423),
    "essays": (87, 472, 425),
    "botany": (85, 491, 427),
}
SCORE_SETTING = ("beagle", "medicine")

# ---- campaign constants used only to convert a share into a price ---------
PSI_MTP = 0.693391               # QMV share of the candidate MTP leg
PREFILL_TO_SCORE = 0.9125        # timed seed prefill dilutes a decode win
RANKED_MDE_PCT = 0.283           # 2 sd on the published median of eight
ADVISOR_LOCAL_SHARE_PCT = 53.45  # local fixture M=9 share of QMV time
ADVISOR_LOCAL_PRICE_PCT = (3.61, 3.81, 4.5)   # lo, hi, first quote

# ---- measured local anchors, this PR's own ABBA session -------------------
# research/e56-session1 legs, base and sched arms, trace width histograms.
SESSION1_BASE_WIDTHS = {9: 0.576, 6: 0.182, 7: 0.091, 8: 0.091, 5: 0.030,
                        2: 0.030}
SESSION1_SCHED_WIDTHS = {4: 0.985, 2: 0.015}


# =========================================================================
# 1. QMV cost, and the width -> time conversion
# =========================================================================

def streams(width: int, ipg: dict) -> int:
    """`ceil(M / IPG(M))` from the shipped dispatch switch. M<3 extrapolates."""
    if width < 3:
        return 1
    return math.ceil(width / ipg[width])


def qmv_cost(width: int, ipg: dict, per_stream: float, per_row: float,
             intercept: float) -> float:
    return intercept + per_stream * streams(width, ipg) + per_row * width


def time_shares(mixture: dict, ipg: dict, per_stream: float, per_row: float,
                intercept: float) -> dict:
    """Round-count shares -> shares of candidate-leg QMV time."""
    weight = {int(w): share * qmv_cost(int(w), ipg, per_stream, per_row,
                                       intercept)
              for w, share in mixture.items()}
    total = sum(weight.values())
    return {w: weight[w] / total for w in sorted(weight)}


# =========================================================================
# 2. The shipped walk, traced
# =========================================================================

def run_window(model: E53.BurstAcceptance, rng: random.Random) -> dict:
    """`E53.run_burst_window` with the width-cap gate instrumented.

    The random draw order is identical to E53's, so a fit reproduces the
    moments it was accepted on.
    """
    sched = E53.Schedule()
    emitted = 0
    widths: dict[int, int] = {}
    gate_open_rounds = 0
    gate_open_widths: dict[int, int] = {}
    at_cap = 0
    drafted_total = accepted_total = rounds = non_drafting = 0
    while emitted < E53.TOTAL_TOKENS:
        remaining = E53.TOTAL_TOKENS - emitted
        offered = max(1, min(E53.OFFERED_DEPTH, E53.MAX_DEPTH, remaining - 1))
        probs, margin = model.step(rng)
        gate_open = sched.streak >= E53.SEGMENTED_STREAK_GATE
        cap = min(offered, E53.MAX_DEPTH,
                  E53.SEGMENTED_VERIFY_DEPTH_CAP if gate_open
                  else E53.SDPA_WIDTH_WALL_DEPTH_CAP)
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
        if depth == cap:
            at_cap += 1
        widths[depth + 1] = widths.get(depth + 1, 0) + 1
        if gate_open:
            gate_open_rounds += 1
            gate_open_widths[depth + 1] = gate_open_widths.get(depth + 1, 0) + 1
        emitted += 1 + accepted
        sched.record(accepted, depth)
        if rounds > 4 * E53.TOTAL_TOKENS:
            raise RuntimeError("schedule failed to close the window")
    return {
        "rounds": rounds,
        "drafted": drafted_total,
        "accepted": accepted_total,
        "non_drafting": non_drafting,
        "at_cap": at_cap,
        "gate_open_rounds": gate_open_rounds,
        "widths": widths,
        "gate_open_widths": gate_open_widths,
    }


def replay_fit(fit: dict, windows: int, seed: int) -> dict:
    runs = []
    per_window_share9 = []
    for index in range(windows):
        rng = random.Random(seed + index)
        model = E53.BurstAcceptance(fit["share_easy"], fit["persistence"],
                                    fit["q_easy"], fit["q_hard"])
        model.easy = rng.random() < fit["share_easy"]
        run = run_window(model, rng)
        runs.append(run)
        per_window_share9.append(run["widths"].get(9, 0) / run["rounds"])
    widths: dict[int, int] = {}
    gate_widths: dict[int, int] = {}
    for run in runs:
        for width, count in run["widths"].items():
            widths[width] = widths.get(width, 0) + count
        for width, count in run["gate_open_widths"].items():
            gate_widths[width] = gate_widths.get(width, 0) + count
    total = sum(widths.values())
    drafted = sum(r["drafted"] for r in runs)
    accepted = sum(r["accepted"] for r in runs)
    gate_open = sum(r["gate_open_rounds"] for r in runs)
    return {
        "windows": windows,
        "mean_draft_len": drafted / total,
        "accept_ratio": accepted / drafted if drafted else 0.0,
        "rounds": statistics.fmean(r["rounds"] for r in runs),
        "non_drafting": statistics.fmean(r["non_drafting"] for r in runs),
        "at_cap_share": sum(r["at_cap"] for r in runs) / total,
        "gate_open_share": gate_open / total,
        "mixture": {w: widths[w] / total for w in sorted(widths)},
        "gate_open_mixture": {w: gate_widths[w] / max(1, gate_open)
                              for w in sorted(gate_widths)},
        "share9_window_sem": (statistics.stdev(per_window_share9)
                              / math.sqrt(windows) if windows > 1 else 0.0),
    }


# =========================================================================
# 3. Part A -- the deliverable
# =========================================================================

def moments(prompt: str) -> dict:
    rounds, proposed, accepted = RECEIPT[prompt]
    return {
        "rounds": rounds,
        "proposed": proposed,
        "accepted": accepted,
        "mean_depth": proposed / rounds,
        "per_draft_accept": accepted / proposed,
    }


def prompt_mixture(prompt: str, fits: list[dict], ipg: dict, windows: int,
                   seed: int, ratio_scale: float = 1.0) -> dict:
    published = moments(prompt)
    rows = []
    for fit in fits:
        replay = replay_fit(fit, windows, seed)
        shares = time_shares(replay["mixture"], ipg,
                             E56.E46_PER_STREAM * ratio_scale,
                             E56.E46_PER_ROW, E56.E46_INTERCEPT)
        rows.append({
            "share_easy": fit["share_easy"],
            "persistence": fit["persistence"],
            "q_easy": fit["q_easy"],
            "q_hard": fit["q_hard"],
            "mean_draft_len": replay["mean_draft_len"],
            "accept_ratio": replay["accept_ratio"],
            "rounds": replay["rounds"],
            "gate_open_share": replay["gate_open_share"],
            "at_cap_share": replay["at_cap_share"],
            "count_mixture": replay["mixture"],
            "time_mixture": shares,
            "count_share9": replay["mixture"].get(9, 0.0),
            "time_share9": shares.get(9, 0.0),
            "time_share456": sum(shares.get(w, 0.0) for w in (4, 5, 6)),
            "share9_window_sem": replay["share9_window_sem"],
        })
    count9 = [row["count_share9"] for row in rows]
    time9 = [row["time_share9"] for row in rows]
    time456 = [row["time_share456"] for row in rows]
    widths = sorted({w for row in rows for w in row["count_mixture"]})
    return {
        "published": published,
        "fits": rows,
        "count_share9_interval": [min(count9), max(count9)],
        "time_share9_interval": [min(time9), max(time9)],
        "time_share456_interval": [min(time456), max(time456)],
        "count_mixture_interval": {
            w: [min(row["count_mixture"].get(w, 0.0) for row in rows),
                max(row["count_mixture"].get(w, 0.0) for row in rows)]
            for w in widths},
        "time_mixture_interval": {
            w: [min(row["time_mixture"].get(w, 0.0) for row in rows),
                max(row["time_mixture"].get(w, 0.0) for row in rows)]
            for w in widths},
        "moment_error": {
            "mean_draft_len": [min(row["mean_draft_len"] for row in rows)
                               - published["mean_depth"],
                               max(row["mean_draft_len"] for row in rows)
                               - published["mean_depth"]],
            "accept_ratio": [min(row["accept_ratio"] for row in rows)
                             - published["per_draft_accept"],
                             max(row["accept_ratio"] for row in rows)
                             - published["per_draft_accept"]],
        },
    }


def _screen(share_easy: float, persistence: float, q_easy: float,
            q_hard: float, windows: int, seed: int) -> dict:
    fit = {"share_easy": share_easy, "persistence": persistence,
           "q_easy": q_easy, "q_hard": q_hard}
    return replay_fit(fit, windows, seed)


def maximising_fit(prompt: str, ipg: dict, seed: int,
                   windows_screen: int = 16, windows_confirm: int = 160,
                   depth_tol: float = 0.10, accept_tol: float = 0.008) -> dict:
    """The feasible acceptance model that MAXIMISES M=9's QMV time share.

    The E53 feasible set was built to span the moments, not to stress this
    statistic. A price that survives the worst case is worth more than a price
    that rests on a central fit, so this searches the model family directly:
    for each (share_easy, persistence, q_easy) it solves `q_hard` for the
    published mean draft length, keeps the point only if the accept ratio also
    lands, and reports the extremes of `time_share9` over the survivors.
    """
    pub = moments(prompt)
    survivors = []
    for step in range(0, 13):
        share_easy = step * 0.05
        for persistence in (0.0, 0.4, 0.7, 0.92):
            for q_easy in (0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
                lo, hi = 0.60, 0.999
                q_hard = q_easy
                for _ in range(10):
                    q_hard = 0.5 * (lo + hi)
                    got = _screen(share_easy, persistence, q_easy, q_hard,
                                  windows_screen, seed)
                    if got["mean_draft_len"] < pub["mean_depth"]:
                        lo = q_hard
                    else:
                        hi = q_hard
                if (abs(got["mean_draft_len"] - pub["mean_depth"]) <= depth_tol
                        and abs(got["accept_ratio"] - pub["per_draft_accept"])
                        <= accept_tol):
                    survivors.append((share_easy, persistence, q_easy, q_hard))
    rows = []
    for share_easy, persistence, q_easy, q_hard in survivors:
        got = _screen(share_easy, persistence, q_easy, q_hard,
                      windows_confirm, seed)
        if (abs(got["mean_draft_len"] - pub["mean_depth"]) > depth_tol
                or abs(got["accept_ratio"] - pub["per_draft_accept"])
                > accept_tol):
            continue
        shares = time_shares(got["mixture"], ipg, E56.E46_PER_STREAM,
                             E56.E46_PER_ROW, E56.E46_INTERCEPT)
        rows.append({
            "share_easy": share_easy, "persistence": persistence,
            "q_easy": q_easy, "q_hard": q_hard,
            "mean_draft_len": got["mean_draft_len"],
            "accept_ratio": got["accept_ratio"],
            "count_share9": got["mixture"].get(9, 0.0),
            "time_share9": shares.get(9, 0.0),
            "time_share456": sum(shares.get(w, 0.0) for w in (4, 5, 6)),
        })
    if not rows:
        raise RuntimeError(f"no feasible point found for {prompt}")
    best = max(rows, key=lambda row: row["time_share9"])
    worst = min(rows, key=lambda row: row["time_share9"])
    return {
        "screened": len(survivors),
        "confirmed": len(rows),
        "max": best,
        "min": worst,
        "time_share9_interval": [worst["time_share9"], best["time_share9"]],
    }


def price_two_stream_m9(time_share9: float, ipg: dict) -> dict:
    """What a two-stream M=9 QMV path is worth, given M=9's time share.

    Two independent routes, reported together because they rest on different
    assumptions:
      relative  scale the advisor's own local-fixture price by the share ratio.
                Assumption-free about psi and the prefill, because both cancel.
      explicit  removing one of three weight streams at M=9 removes
                `E46_PER_STREAM / T(9)` of M=9's QMV time.
    """
    t9 = qmv_cost(9, ipg, E56.E46_PER_STREAM, E56.E46_PER_ROW,
                  E56.E46_INTERCEPT)
    stream_fraction_of_t9 = E56.E46_PER_STREAM / t9
    qmv_saved = time_share9 * stream_fraction_of_t9
    explicit = qmv_saved * PSI_MTP * PREFILL_TO_SCORE * 100.0
    scale = time_share9 * 100.0 / ADVISOR_LOCAL_SHARE_PCT
    return {
        "t9_us": t9,
        "stream_fraction_of_t9": stream_fraction_of_t9,
        "qmv_time_saved_pct": qmv_saved * 100.0,
        "explicit_score_pct": explicit,
        "relative_score_pct": [quote * scale for quote in
                               ADVISOR_LOCAL_PRICE_PCT],
        "explicit_sd_vs_mde": explicit / RANKED_MDE_PCT,
    }


# =========================================================================
# 4. Part B -- the M4 Pro -> M5 transfer correction
# =========================================================================

# ---- E56 pinned-width sweep, this host, base binary -----------------------
# Mean seconds per token at verify widths 3,4,5,6 and the rounds each leg ran,
# from research/e56-width-sweep.json. The derived per-ROUND marginals give the
# boundary surcharge at the level the walk actually consumes, which the E46
# per-OPERATION ratio 27.532/9.624 = 2.861 over-states.
SWEEP_ROUND_SECONDS = {3: 0.12912, 4: 0.15878, 5: 0.21549, 6: 0.24170}


def measured_round_boundary_ratio() -> float:
    marginal = {width: SWEEP_ROUND_SECONDS[width] - SWEEP_ROUND_SECONDS[width - 1]
                for width in (4, 5, 6)}
    within = statistics.fmean([marginal[4], marginal[6]])
    return marginal[5] / within


def measured_raw_surcharge(head_ratio: float, mean_target: float) -> float:
    """Raw surcharge `r` whose mean-pinned table has the measured round ratio.

    `stream_aware_price` builds `raw = 1 + r` at a boundary and pins the mean
    of `head + scale*raw` to `mean_target`, so `r` is not the round-level
    ratio. Solving the two constraints for the measured ratio `R`:

        within   = head + scale
        boundary = head + scale * (1 + r) = R * within
        scale    = (mean_target - head) / (1 + 2r/8)
    """
    target = measured_round_boundary_ratio()
    budget = mean_target - head_ratio
    numerator = (target - 1.0) * mean_target
    denominator = budget - (target - 1.0) * head_ratio / 4.0
    return numerator / denominator


def g_corrected_ladder(g: float) -> list[float]:
    """`round_M5(d) / c1 = 1 + g * (C_M4(d) - C_M4(0)) / C_M4(0)`.

    `c1` cancels in every ratio below, so this is a shape, not a level.
    """
    base = E56.E1_C_US[0]
    return [1.0 + g * (value - base) / base for value in E56.E1_C_US]


def g_correction_table(g_lo: float, g_hi: float, head_ratio: float) -> dict:
    """Reproduce the advisor's raw and mean-pinned tables independently.

    `h(d) = marginal(d) / C(d)` -- the extra round cost of the d-th extension
    expressed against the cost of the round that takes it, which is the
    convention the advisor's table uses.
    """
    base = E56.E1_C_US[0]
    marg = [(E56.E1_C_US[d + 1] - E56.E1_C_US[d]) for d in range(E53.MAX_DEPTH)]
    h_m4 = [marg[d] / E56.E1_C_US[d + 1] for d in range(E53.MAX_DEPTH)]
    rows = []
    for d in range(E53.MAX_DEPTH):
        num = marg[d] / base
        row = {"depth": d + 1, "marg_m4_ms": marg[d] / 1000.0, "h_m4": h_m4[d]}
        for tag, g in (("lo", g_lo), ("hi", g_hi)):
            ladder = g_corrected_ladder(g)
            row[f"h_m5_{tag}"] = g * num / ladder[d + 1]
            row[f"ratio_{tag}"] = row[f"h_m5_{tag}"] / h_m4[d]
        rows.append(row)
    pinned = {}
    for tag, source in (("shipped", h_m4),
                        ("m5_lo", [row["h_m5_lo"] for row in rows]),
                        ("m5_hi", [row["h_m5_hi"] for row in rows])):
        scale = head_ratio / statistics.fmean(source)
        pinned[tag] = [value * scale for value in source]
    for index, row in enumerate(rows):
        row["pinned_shipped"] = pinned["shipped"][index]
        row["pinned_m5_lo"] = pinned["m5_lo"][index]
        row["pinned_m5_hi"] = pinned["m5_hi"][index]
        row["pinned_delta_lo_pct"] = 100.0 * (row["pinned_m5_lo"]
                                              / row["pinned_shipped"] - 1.0)
        row["pinned_delta_hi_pct"] = 100.0 * (row["pinned_m5_hi"]
                                              / row["pinned_shipped"] - 1.0)
    return {"g_lo": g_lo, "g_hi": g_hi, "rows": rows}


def counterfactual_under_g(fits: dict, ipg: dict, constants: dict, g: float,
                           windows: int, seed: int) -> dict:
    """Re-run the E56 attribution with the ranked-transferred truth ladder."""
    truth = [value * E56.E1_C_US[0] for value in g_corrected_ladder(g)]
    head = E56.E1_HEAD_STEP_US / E56.E1_C_US[0]
    h = constants["headStepCostRatio"]
    shipped = E56.shipped_price(h)
    ratio = E56.E46_PER_STREAM / E56.E46_PER_ROW
    arms = {
        "stream_aware": E56.stream_aware_price(ipg, head, h),
        "stream_aware_measured": E56.stream_aware_price(
            ipg, head, h, ratio=measured_raw_surcharge(head, h)),
    }
    for tag, depths in (("boundary45_only", (3,)), ("boundary89_only", (7,))):
        shape = [1.0 + (ratio if d in depths else 0.0)
                 for d in range(E53.MAX_DEPTH)]
        arms[tag] = E56.pinned_shape_price(tag, shape, head, h,
                                           f"surcharge at {depths} only")
    out = {}
    for prompt, rows in fits.items():
        fit = E56.central_fit(rows, moments(prompt)["mean_depth"])
        base = E56.replay(fit, shipped, truth, windows, seed)
        out[prompt] = {"arms": {}}
        for name, price in arms.items():
            other = E56.replay(fit, price, truth, windows, seed)
            delta = E56.paired_delta_pct(base, other)
            out[prompt]["arms"][name] = {
                "decode_pct": delta["decode_pct"],
                "decode_pct_sem": delta["decode_pct_sem"],
                "mean_draft_len": other["mean_draft_len"],
            }
        out[prompt]["base_mean_draft_len"] = base["mean_draft_len"]
    return out


H_GRID = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28, 0.32)


def h_sweep_under_g(fits: dict, constants: dict, g: float, windows: int,
                    seed: int) -> dict:
    """Which scalar price is best when the truth ladder is the ranked one?

    ANALYSIS ONLY. `headStepCostRatio` is not retuned in this PR; this answers
    the advisor's testable claim that the ranked optimum `h` differs from the
    locally tuned 0.18, and it is the cleanest way to separate the LEVEL of the
    transfer correction from its SHAPE.
    """
    truth = [value * E56.E1_C_US[0] for value in g_corrected_ladder(g)]
    base_price = E56.shipped_price(constants["headStepCostRatio"])
    m4_base = E56.E1_C_US[0]
    unpinned = E56.Price(
        "m5_unpinned",
        [g * (E56.E1_C_US[d + 1] - E56.E1_C_US[d]) / m4_base
         for d in range(E53.MAX_DEPTH)],
        "the transferred M5 marginals used directly, level and shape")
    out = {}
    for prompt, rows in fits.items():
        fit = E56.central_fit(rows, moments(prompt)["mean_depth"])
        base = E56.replay(fit, base_price, truth, windows, seed)
        arms = {}
        for h in H_GRID:
            other = E56.replay(fit, E56.shipped_price(h), truth, windows, seed)
            arms[f"h_{h:.2f}"] = {
                "decode_pct": E56.paired_delta_pct(base, other)["decode_pct"],
                "mean_draft_len": other["mean_draft_len"],
            }
        other = E56.replay(fit, unpinned, truth, windows, seed)
        arms["m5_unpinned"] = {
            "decode_pct": E56.paired_delta_pct(base, other)["decode_pct"],
            "mean_draft_len": other["mean_draft_len"],
        }
        best = min(arms, key=lambda name: arms[name]["decode_pct"])
        out[prompt] = {"arms": arms, "best": best,
                       "base_mean_draft_len": base["mean_draft_len"]}
    return out


# =========================================================================
# 5. Report
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--stress", action="store_true",
                        help="search the model family for the M=9 maximiser")
    args = parser.parse_args()

    constants = E56.parse_schedule_constants()
    ipg = E56.parse_ipg_table()
    fits = E56.load_burst_fits()

    report = {
        "constants": constants,
        "ipg": {str(k): v for k, v in ipg.items()},
        "windows": args.windows,
        "seed": args.seed,
        "receipt": {p: moments(p) for p in RECEIPT},
        "prompts": {},
        "sensitivity": {},
        "local_anchor": {},
        "pricing": {},
    }

    report["stress"] = {}
    for prompt in SCORE_SETTING:
        report["prompts"][prompt] = prompt_mixture(prompt, fits[prompt], ipg,
                                                   args.windows, args.seed)
        if args.stress:
            report["stress"][prompt] = {}
            for tag, depth_tol, accept_tol in (("strict", 0.05, 0.004),
                                               ("loose", 0.10, 0.008)):
                found = maximising_fit(prompt, ipg, args.seed,
                                       depth_tol=depth_tol,
                                       accept_tol=accept_tol)
                found["price_at_max"] = price_two_stream_m9(
                    found["time_share9_interval"][1], ipg)
                found["tolerance"] = {"mean_depth": depth_tol,
                                      "accept_ratio": accept_tol}
                report["stress"][prompt][tag] = found
        report["pricing"][prompt] = {
            "lo": price_two_stream_m9(
                report["prompts"][prompt]["time_share9_interval"][0], ipg),
            "hi": price_two_stream_m9(
                report["prompts"][prompt]["time_share9_interval"][1], ipg),
        }
        report["sensitivity"][prompt] = {}
        for tag, scale in (("ratio_minus_30pct", 0.7), ("ratio_plus_30pct", 1.3)):
            mixed = prompt_mixture(prompt, fits[prompt], ipg, 60, args.seed,
                                   ratio_scale=scale)
            report["sensitivity"][prompt][tag] = {
                "time_share9_interval": mixed["time_share9_interval"],
                "time_share456_interval": mixed["time_share456_interval"],
            }

    for tag, hist in (("session1_base", SESSION1_BASE_WIDTHS),
                      ("session1_sched", SESSION1_SCHED_WIDTHS)):
        shares = time_shares(hist, ipg, E56.E46_PER_STREAM, E56.E46_PER_ROW,
                             E56.E46_INTERCEPT)
        report["local_anchor"][tag] = {
            "count_mixture": hist,
            "time_mixture": shares,
            "time_share9": shares.get(9, 0.0),
            "mean_width": sum(w * s for w, s in hist.items()),
        }

    report["g_correction"] = g_correction_table(0.7388, 0.7778,
                                                constants["headStepCostRatio"])
    report["g_counterfactual"] = {
        "g_lo": counterfactual_under_g(fits, ipg, constants, 0.7388, 60,
                                       args.seed),
        "g_hi": counterfactual_under_g(fits, ipg, constants, 0.7778, 60,
                                       args.seed),
        "g_one": counterfactual_under_g(fits, ipg, constants, 1.0, 60,
                                        args.seed),
    }
    head = E56.E1_HEAD_STEP_US / E56.E1_C_US[0]
    h = constants["headStepCostRatio"]
    raw_surcharge = measured_raw_surcharge(head, h)
    priced = {
        "shipped": E56.shipped_price(h),
        "stream_aware_e46": E56.stream_aware_price(ipg, head, h),
        "stream_aware_measured": E56.stream_aware_price(ipg, head, h,
                                                        ratio=raw_surcharge),
    }
    report["measured_boundary"] = {
        "sweep_round_seconds": SWEEP_ROUND_SECONDS,
        "round_ratio": measured_round_boundary_ratio(),
        "e46_operation_ratio": E56.E46_PER_STREAM / E56.E46_PER_ROW,
        "raw_surcharge": raw_surcharge,
        "within_tier": priced["stream_aware_measured"].marginal(0),
        "boundary": priced["stream_aware_measured"].marginal(3),
        "best_case_threshold": {
            name: [price.marginal(d) * (d + 1) / price.cost(d)
                   for d in range(E53.MAX_DEPTH)]
            for name, price in priced.items()},
    }
    report["h_sweep"] = {
        "g_lo": h_sweep_under_g(fits, constants, 0.7388, 60, args.seed),
        "g_hi": h_sweep_under_g(fits, constants, 0.7778, 60, args.seed),
        "g_one": h_sweep_under_g(fits, constants, 1.0, 60, args.seed),
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print_report(report)


def print_report(report: dict) -> None:
    print("=" * 74)
    print("E56 deliverable: verify-width mixture at the score-setting prompts")
    print("=" * 74)
    print(f"windows per fit {report['windows']}, seed {report['seed']}")
    print("live constants:", {k: report["constants"][k] for k in
                              sorted(report["constants"])})
    print("IPG table:", report["ipg"])

    for prompt in SCORE_SETTING:
        block = report["prompts"][prompt]
        pub = block["published"]
        print()
        print("-" * 74)
        print(f"{prompt}: published rounds {pub['rounds']}, proposed "
              f"{pub['proposed']}, accepted {pub['accepted']}")
        print(f"  mean depth {pub['mean_depth']:.4f}, per-draft accept "
              f"{pub['per_draft_accept']:.4f}")
        err = block["moment_error"]
        print(f"  fit reproduction error: mean depth "
              f"[{err['mean_draft_len'][0]:+.4f}, {err['mean_draft_len'][1]:+.4f}]"
              f", accept [{err['accept_ratio'][0]:+.4f}, "
              f"{err['accept_ratio'][1]:+.4f}]")
        print(f"  {len(block['fits'])} feasible fits (identification set)")
        print()
        print("   M | rounds share (lo..hi)   | QMV time share (lo..hi)")
        for width in sorted(block["count_mixture_interval"], key=int):
            cnt = block["count_mixture_interval"][width]
            tim = block["time_mixture_interval"][width]
            mark = "  <-- M=9" if int(width) == 9 else ""
            print(f"  {int(width):2d} | {cnt[0]*100:6.2f} .. {cnt[1]*100:6.2f} %"
                  f"       | {tim[0]*100:6.2f} .. {tim[1]*100:6.2f} %{mark}")
        t9 = block["time_share9_interval"]
        c9 = block["count_share9_interval"]
        t456 = block["time_share456_interval"]
        print(f"  M=9 round share  {c9[0]*100:.2f} .. {c9[1]*100:.2f} %")
        print(f"  M=9 QMV time     {t9[0]*100:.2f} .. {t9[1]*100:.2f} %"
              f"   (local fixture reference {ADVISOR_LOCAL_SHARE_PCT} %)")
        print(f"  M in 4,5,6 QMV   {t456[0]*100:.2f} .. {t456[1]*100:.2f} %")
        gate = [row["gate_open_share"] for row in block["fits"]]
        cap = [row["at_cap_share"] for row in block["fits"]]
        print(f"  width-cap gate open on {min(gate)*100:.1f} .. "
              f"{max(gate)*100:.1f} % of rounds; walk stops at the cap on "
              f"{min(cap)*100:.1f} .. {max(cap)*100:.1f} %")
        sens = report["sensitivity"][prompt]
        for tag in sorted(sens):
            interval = sens[tag]["time_share9_interval"]
            print(f"  sensitivity {tag}: M=9 QMV time "
                  f"{interval[0]*100:.2f} .. {interval[1]*100:.2f} %")
        price = report["pricing"][prompt]
        print(f"  two-stream M=9 price, explicit route: "
              f"{price['lo']['explicit_score_pct']:.3f} .. "
              f"{price['hi']['explicit_score_pct']:.3f} % of score "
              f"({price['lo']['explicit_sd_vs_mde']:.2f} .. "
              f"{price['hi']['explicit_sd_vs_mde']:.2f} x the "
              f"{RANKED_MDE_PCT} % MDE)")
        rel_lo = price["lo"]["relative_score_pct"]
        rel_hi = price["hi"]["relative_score_pct"]
        print(f"  two-stream M=9 price, relative route: "
              f"{min(rel_lo):.3f} .. {max(rel_hi):.3f} % of score")
        for tag in sorted(report["stress"].get(prompt, {})):
            stress = report["stress"][prompt][tag]
            best = stress["max"]
            tol = stress["tolerance"]
            print(f"  stress search ({tag}: depth +/-{tol['mean_depth']}, "
                  f"accept +/-{tol['accept_ratio']}): {stress['confirmed']} "
                  f"feasible points, M=9 QMV time "
                  f"{stress['time_share9_interval'][0]*100:.2f} .. "
                  f"{stress['time_share9_interval'][1]*100:.2f} %")
            print(f"    maximiser share_easy {best['share_easy']:.2f}, "
                  f"persistence {best['persistence']:.2f}, q_easy "
                  f"{best['q_easy']:.2f}, q_hard {best['q_hard']:.4f} -> "
                  f"mean depth {best['mean_draft_len']:.3f}, accept "
                  f"{best['accept_ratio']:.4f}; price "
                  f"{stress['price_at_max']['explicit_score_pct']:.3f} % of "
                  f"score ({stress['price_at_max']['explicit_sd_vs_mde']:.2f}"
                  f" x MDE)")

    print()
    print("-" * 74)
    print("local anchors, measured, this PR's ABBA session")
    for tag in sorted(report["local_anchor"]):
        block = report["local_anchor"][tag]
        print(f"  {tag}: mean verify width {block['mean_width']:.3f}, "
              f"M=9 QMV time {block['time_share9']*100:.2f} %")

    print()
    print("-" * 74)
    print("M4 Pro -> M5 transfer correction, reproduced independently")
    table = report["g_correction"]
    print(f"  g in [{table['g_lo']}, {table['g_hi']}]")
    print("   d | marg_M4 ms |  h_M4  | ratio lo | ratio hi | pinned delta lo/hi")
    for row in table["rows"]:
        print(f"  {row['depth']:2d} | {row['marg_m4_ms']:10.2f} | "
              f"{row['h_m4']:.4f} | {row['ratio_lo']:8.4f} | "
              f"{row['ratio_hi']:8.4f} | {row['pinned_delta_lo_pct']:+6.2f} / "
              f"{row['pinned_delta_hi_pct']:+6.2f} %")

    print()
    measured = report["measured_boundary"]
    print(f"measured round-level boundary ratio {measured['round_ratio']:.4f} "
          f"(E46 per-operation ratio {measured['e46_operation_ratio']:.4f}); "
          f"raw surcharge {measured['raw_surcharge']:.4f} gives marginals "
          f"{measured['within_tier']:.6f} within tier and "
          f"{measured['boundary']:.6f} at a boundary")
    print("best-case thresholds (a step is closed at any accept rate if >= 1):")
    for name in ("shipped", "stream_aware_e46", "stream_aware_measured"):
        row = ", ".join(f"d{d}:{value:.4f}" for d, value
                        in enumerate(measured["best_case_threshold"][name]))
        print(f"  {name:22s} {row}")

    print()
    print("counterfactual under the transferred truth ladder (decode %)")
    for tag in ("g_one", "g_lo", "g_hi"):
        block = report["g_counterfactual"][tag]
        for prompt in sorted(block):
            arms = block[prompt]["arms"]
            row = ", ".join(f"{name} {arms[name]['decode_pct']:+.3f}"
                            for name in sorted(arms))
            print(f"  {tag:6s} {prompt:9s} {row}")

    print()
    print("scalar-price sweep under the transferred truth ladder "
          "(decode %, analysis only)")
    for tag in ("g_one", "g_lo", "g_hi"):
        block = report["h_sweep"][tag]
        for prompt in sorted(block):
            arms = block[prompt]["arms"]
            row = " ".join(f"{name.split('_')[-1]}:{arms[name]['decode_pct']:+.2f}"
                           for name in arms)
            print(f"  {tag:6s} {prompt:9s} best={block[prompt]['best']}  {row}")


if __name__ == "__main__":
    main()
