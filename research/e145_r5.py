#!/usr/bin/env python3
"""E145 R5: close the pb6 identification gap and solve the RANKED width curve.

Zero GPU. Requested in advisor notes F6 and F7.

R5-a  `harness=ranked`. F7 replaced the replayed round-count ratio with a
      closed form, `R = 512 / (1 + rate * effective_mean_draft_len)`. This
      module re-derives it from the board rather than copying the advisor's
      digits, and then uses the replayer for the ONE thing the closed form
      cannot supply: the inversion holds the accept rate fixed, and pb6
      changes the depth policy, so the rate may move.

R5-b  `harness=ranked`. How many draws of the 879 us wired-residency state
      does the published pb6 receipt contain? Three readings that do not use
      the cross-prompt shape: the `drama` control, whose schedule pb6 provably
      does not move; a free fit of the per-drafting-round coefficient over the
      seven non-plutarch prompts; and the replayer's own zero-parameter
      closure prediction.

R5-c  `harness=ranked`. Solve the ranked per-width round cost directly from
      the board, with the replayed width masses as the design matrix. This is
      the rung that removes the M4-Pro-to-M5 transfer risk from R2's local
      curve, because it never reads a local curve.

R5-d  One source fact the advisor asked to have on the record.

WHY R5-c IS NOT CIRCULAR. The width masses come from `simulate`, whose depth
walk reads only the PRICE table. `flat_price` and `pb6_price` are constant
tables that carry no cost curve, so the mass vectors do not depend on R2's
measured curve or on the replayed curve. Only `us_per_token` reads a cost
curve, and R5-c never uses `us_per_token`. A gate below asserts that.

WHAT R5-c STILL ASSUMES. The design matrix is replayed. The transfer is fitted
to one moment per prompt, the published `effective_mean_draft_len`, so the
width distribution around that mean is a model output. R5-c removes the
cost-transfer risk and keeps the mass-model risk.

ONE CORRECTION TO F6. F6 proposed solving the bar `684821ed` and the pb6 row
`e003a86d` together. Those rows come from two different trees: the bar is
`newjordan`'s and the pb6 row is ours. At identical draft lengths our anchor
`572b2cc4` is about one percent slower than the bar, so the two families do
not share one per-width cost, and stacking them would fit the level difference
into the width coefficients. Each family is solved on its own and the level
difference is reported separately. Only our family carries two different width
mixtures, so only our family is identified by more than one mixture.

TWO CONSTRAINTS FROM THE ADVISOR, BOTH ENFORCED HERE.
  Advisor Error 156: never interpolate the curve at a mean row count. The
  masses are the design matrix. No mean row count appears below.
  Advisor Error 162: subtract the seed prefill before dividing by the round
  count. `prefill_seconds_per_token` is published per row and per prompt, so
  the subtraction always uses that row's own value.

Usage:
  python3 research/e145_r5.py --seeds 6 --windows 200
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import (  # noqa: E402
    DECODE_TOKENS, MAX_DEPTH, PROMPT_NAMES, RANKED_PROMPTS,
)
from e134_rung2 import build_legs, simulate  # noqa: E402
from e140_cells import (  # noqa: E402
    CELLS, greedy_walker, install, price_for, transfer_cache,
)
from e140_lookahead import load_curves  # noqa: E402
from e145_curve import LEGS_JSON  # noqa: E402
from e145_r3 import (  # noqa: E402
    installable, measured_points, serial_round_us, width1_candidates,
)

# F92 pinned ranked round counts under the shipped schedule. R5 does not refit
# them; R5-a checks the closed form against them.
F92_ROUNDS = {
    "beagle": 110.00, "essays": 92.04, "medicine": 90.01, "republic": 93.00,
    "botany": 81.04, "plutarch": 486.76, "drama": 252.01, "travel": 212.33,
}

# F6 section 1. Askeladd fitted the wired-residency step from zero-delta board
# rows only, so the whole fitted coefficient is the step and no mechanism is
# mixed into it. The correction recovered a known zero to 0.02 pp.
STATE_STEP_US = 879.0
STATE_STEP_SD_US = 54.3
STATE_STEP_N = 3

# The board's published within-row coefficient of variation, used only to ask
# how far the solved curve moves under ordinary ranked measurement noise.
BOARD_ROW_CV = 0.00132

SHIP_ROW = "572b2cc4"
PB6_ROW = "e003a86d"
BAR_ROW = "684821ed"

# R2 measured the local cliff at the 5 -> 6 boundary on this M4 Pro.
LOCAL_CLIFF_FROM_WIDTH = 5

SCORING = ("beagle", "medicine", "essays", "botany", "republic")
NON_PLUTARCH = tuple(p for p in RANKED_PROMPTS if p != "plutarch")

SESSION = pathlib.Path("Sources/MLXFastModel/Qwen36MTPBlockSession.swift")


# ------------------------------------------------------------------- the board

def load_row(board: pathlib.Path, prefix: str) -> dict:
    """One board row, per prompt, with the seed prefill kept separate.

    `total_s` is the whole timed candidate leg, which is the frame a
    submission is priced in. `decode_s` is that leg with the row's own
    published seed prefill removed. Both are carried so a later step cannot
    silently mix the two frames, which is Advisor Error 163.
    """
    rows = json.loads(board.read_text())
    if isinstance(rows, dict):
        rows = rows["submissions"]
    for row in rows:
        if not str(row.get("id", "")).startswith(prefix):
            continue
        metrics = row["officialMetrics"]
        per_prompt = {}
        for entry in metrics["per_prompt"]:
            name = PROMPT_NAMES.get(entry["prompt_sha256"][:8])
            mtp = entry["mtp_seconds_per_token_mean"]
            pref = entry.get("prefill_seconds_per_token")
            if pref is None:
                raise SystemExit("row %s prompt %s publishes no prefill rate;"
                                 " Advisor Error 162 cannot be avoided"
                                 % (prefix, name))
            per_prompt[name] = {
                "mtp_spt": mtp,
                "serial_spt": entry["serial_seconds_per_token_mean"],
                "prefill_spt": pref,
                "total_s": mtp * DECODE_TOKENS,
                "prefill_s": pref * DECODE_TOKENS,
                "decode_s": (mtp - pref) * DECODE_TOKENS,
                "prefill_share": pref / mtp,
                "draft_len": entry["effective_mean_draft_len"],
                "non_drafting_round_count": entry.get(
                    "non_drafting_round_count"),
                "raw": entry["raw_ratio_of_means"],
            }
        return {"id": row["id"], "score": row["officialScore"],
                "status": row["status"], "solver": row["solverUsername"],
                "commit": metrics.get("commit"), "per_prompt": per_prompt}
    raise SystemExit("no board row with id prefix %r" % prefix)


def published_median(values: list[float]) -> float:
    ordered = sorted(values)
    return 0.5 * (ordered[3] + ordered[4])


# ------------------------------------------------ R5-0, the local seed prefill

def local_prefill(reports: pathlib.Path) -> dict:
    """Every E145 leg's own seed prefill, which F5 asked for.

    `QwenRuntimeMTP.swift:344-355` writes `seed_prefill_seconds` as
    observability, so no instrumentation was needed.
    """
    values, per_leg = [], []
    for path in sorted(reports.glob("*/*/report.json")):
        blob = json.loads(path.read_text())
        seconds = blob.get("seed_prefill_seconds")
        if seconds is None:
            continue
        values.append(seconds)
        per_leg.append({
            "leg": "%s/%s" % (path.parent.parent.name, path.parent.name),
            "seed_prefill_seconds": seconds,
            "prefill_seconds_per_token": blob.get("prefill_seconds_per_token"),
            "seed_token_count": blob.get("seed_token_count"),
        })
    if not values:
        return {"available": False}
    return {
        "available": True, "legs": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values), "max_seconds": max(values),
        "spread_pct": 100.0 * (max(values) - min(values))
        / statistics.fmean(values),
        "per_leg": per_leg,
    }


# --------------------------------------------------------------------- R5-a

def round_count(rate: float, draft_len: float) -> float:
    """Finding 219's identity. One round emits one primary plus what it keeps."""
    return DECODE_TOKENS / (1.0 + rate * draft_len)


def arm_behaviour(cache, seeds, windows, curve, cell) -> dict:
    """Round counts, accept rate and width masses for one arm.

    The depth walk reads the PRICE table only, and both prices used here are
    constant tables, so the masses and the accept rate do not depend on
    `curve`. `curve` still enters `us_per_token`, which R5-b needs.
    """
    spec = CELLS[cell]
    price = price_for(spec["price"], curve)
    out = {}
    for prompt in RANKED_PROMPTS:
        counts = np.zeros(MAX_DEPTH + 2)
        rounds = 0
        accept, depth, us_run, us_base, per_seed = [], [], [], [], {}
        for seed in seeds:
            entry = cache[(seed, prompt)]
            install(curve)
            base = simulate(None, entry["factory"](entry["p_target"]), windows)
            run = simulate(None, entry["factory"](entry["p_target"]), windows,
                           price=price, walker=greedy_walker())
            seed_counts = np.asarray(run["depth_counts"], dtype=float)
            counts += seed_counts
            seed_mass = seed_counts / seed_counts.sum()
            per_seed[seed] = {w + 1: float(seed_mass[w])
                              for w in range(len(seed_mass))
                              if seed_mass[w] > 0.0}
            rounds += run["rounds"]
            accept.append(run["accept_rate"])
            depth.append(run["mean_depth"])
            us_run.append(run["us_per_token"])
            us_base.append(base["us_per_token"])
        mass = counts / counts.sum()
        out[prompt] = {
            "rounds_per_window": rounds / (len(seeds) * windows),
            "drafting_share": float(counts[1:].sum() / counts.sum()),
            "non_drafting_share": float(counts[0] / counts.sum()),
            "width_mass": {w + 1: float(mass[w]) for w in range(len(mass))
                           if mass[w] > 0.0},
            "width_mass_per_seed": per_seed,
            "accept_rate": statistics.fmean(accept),
            "mean_depth": statistics.fmean(depth),
            "us_per_token": statistics.fmean(us_run),
            "base_us_per_token": statistics.fmean(us_base),
        }
    return out


def rung_a(ship_row: dict, pb6_row: dict, ship: dict, pb6: dict) -> dict:
    """The closed form, its validation, and the accept-rate transfer check.

    The drafting-round count is taken from the receipt, not from the replay.
    Each row publishes `non_drafting_round_count`, so the ranked drafting
    rounds are the modelled round count minus that published constant. The
    replayed drafting share is carried beside it only as a transfer check,
    because the two disagree and the receipt is the measurement.
    """
    rows = {}
    for prompt in RANKED_PROMPTS:
        rate = RANKED_PROMPTS[prompt]["accept"]
        a = ship_row["per_prompt"][prompt]
        b = pb6_row["per_prompt"][prompt]
        dlen_a, dlen_b = a["draft_len"], b["draft_len"]
        model_ship = round_count(rate, dlen_a)
        pinned = F92_ROUNDS[prompt]
        shift = pb6[prompt]["accept_rate"] - ship[prompt]["accept_rate"]
        model_pb6 = round_count(rate, dlen_b)
        model_pb6_shifted = round_count(rate + shift, dlen_b)
        ratio = model_pb6 / model_ship
        ratio_shifted = model_pb6_shifted / model_ship
        rounds_pb6 = pinned * ratio
        ndr_a = float(a["non_drafting_round_count"] or 0.0)
        ndr_b = float(b["non_drafting_round_count"] or 0.0)
        rows[prompt] = {
            "f92_accept_rate": rate,
            "ship_draft_len": dlen_a,
            "pb6_draft_len": dlen_b,
            "f92_pinned_rounds_ship": pinned,
            "model_rounds_ship": model_ship,
            "model_minus_pinned_rounds": model_ship - pinned,
            "model_rounds_pb6": model_pb6,
            "round_ratio": ratio,
            "round_ratio_with_shifted_rate": ratio_shifted,
            "ratio_sign_survives_accept_shift":
                (ratio > 1.0) == (ratio_shifted > 1.0),
            "replay_accept_rate_ship": ship[prompt]["accept_rate"],
            "replay_accept_rate_pb6": pb6[prompt]["accept_rate"],
            "accept_rate_shift_pb6": shift,
            "model_rounds_pb6_with_shifted_rate": model_pb6_shifted,
            "rounds_pb6_shift_pct": 100.0 * (ratio_shifted / ratio - 1.0),
            "ranked_rounds_ship": pinned,
            "ranked_rounds_pb6": rounds_pb6,
            "ranked_non_drafting_rounds_ship": ndr_a,
            "ranked_non_drafting_rounds_pb6": ndr_b,
            "ranked_drafting_rounds_ship": pinned - ndr_a,
            "ranked_drafting_rounds_pb6": rounds_pb6 - ndr_b,
            "replay_drafting_share_ship": ship[prompt]["drafting_share"],
            "replay_drafting_share_pb6": pb6[prompt]["drafting_share"],
            "receipt_drafting_share_ship": (pinned - ndr_a) / pinned,
            "receipt_drafting_share_pb6": (rounds_pb6 - ndr_b) / rounds_pb6,
        }
    return rows


# --------------------------------------------------------------------- R5-b

def one_step_pct(draft_rounds: float, reference_seconds: float) -> float:
    """What one 879 us state draw costs that prompt, in percent of its leg."""
    return 100.0 * STATE_STEP_US * 1e-6 * draft_rounds / reference_seconds


def effects_at(ship_row: dict, pb6_row: dict, a_rows: dict, s: float) -> dict:
    """The pb6 effect after removing `s` per-drafting-round state draws.

    `s` is continuous. Every aggregate the campaign cares about is reported in
    the same call so no caller can quietly read one frame and label it another:
    the seven-prompt and five-prompt decode means, the ranked-weight version of
    the five, the same three on the whole timed leg, and the published median
    pair, which is the only one of them that is the score.
    """
    weights = {p: RANKED_PROMPTS[p]["weight"] for p in SCORING}
    total_weight = sum(weights.values())
    per_prompt, corrected_raw = {}, {}
    for prompt in RANKED_PROMPTS:
        a = ship_row["per_prompt"][prompt]
        b = pb6_row["per_prompt"][prompt]
        draft_rounds = a_rows[prompt]["ranked_drafting_rounds_pb6"]
        removed = s * STATE_STEP_US * 1e-6 * draft_rounds
        per_prompt[prompt] = {
            "removed_s": removed,
            "decode_pct": 100.0 * ((b["decode_s"] - removed)
                                   / a["decode_s"] - 1.0),
            "total_pct": 100.0 * ((b["total_s"] - removed)
                                  / a["total_s"] - 1.0),
        }
        corrected_raw[prompt] = (b["serial_spt"] * DECODE_TOKENS
                                 / (b["total_s"] - removed))
    median_b = published_median(list(corrected_raw.values()))
    median_a = published_median([v["raw"] for v
                                 in ship_row["per_prompt"].values()])
    return {
        "state_steps": s,
        "per_prompt": per_prompt,
        "decode_seven_mean_pct": statistics.fmean(
            [per_prompt[p]["decode_pct"] for p in NON_PLUTARCH]),
        "decode_five_mean_pct": statistics.fmean(
            [per_prompt[p]["decode_pct"] for p in SCORING]),
        "decode_five_weighted_pct": sum(
            weights[p] * per_prompt[p]["decode_pct"]
            for p in SCORING) / total_weight,
        "total_seven_mean_pct": statistics.fmean(
            [per_prompt[p]["total_pct"] for p in NON_PLUTARCH]),
        "total_five_mean_pct": statistics.fmean(
            [per_prompt[p]["total_pct"] for p in SCORING]),
        "total_five_weighted_pct": sum(
            weights[p] * per_prompt[p]["total_pct"]
            for p in SCORING) / total_weight,
        "corrected_raw": corrected_raw,
        "corrected_median": median_b,
        "ship_median": median_a,
        "corrected_median_pct": 100.0 * (median_b / median_a - 1.0),
    }


def rung_b(ship_row: dict, pb6_row: dict, a_rows: dict, steps: tuple,
           closure: dict) -> dict:
    """The state-step lattice, in both the decode frame and the total frame."""
    weights = {p: RANKED_PROMPTS[p]["weight"] for p in SCORING}
    total_weight = sum(weights.values())
    out = {}
    for s in steps:
        cell = effects_at(ship_row, pb6_row, a_rows, s)
        residual = sum(
            weights[p] * (cell["per_prompt"][p]["decode_pct"]
                          - closure["per_prompt"][p]["decode_pct"]) ** 2
            for p in SCORING) / total_weight
        cell["weighted_sq_error_vs_closure"] = residual
        cell["rms_error_vs_closure_pp"] = math.sqrt(residual)
        out[s] = cell
    return out


def zero_crossing(ship_row: dict, pb6_row: dict, a_rows: dict, key: str,
                  lo: float = 0.0, hi: float = 6.0) -> float | None:
    """The state-step count at which one aggregate changes sign.

    Every aggregate here is monotone in `s`, because raising `s` removes time
    from the pb6 leg only, so a bisection is exact to the returned tolerance.
    """
    def f(x: float) -> float:
        return effects_at(ship_row, pb6_row, a_rows, x)[key]

    f_lo, f_hi = f(lo), f(hi)
    if (f_lo > 0.0) == (f_hi > 0.0):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0.0) == (f_lo > 0.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def free_fit_steps(ship_row: dict, pb6_row: dict, a_rows: dict) -> dict:
    """One free coefficient, microseconds per drafting round, over the seven.

    The regression is through the origin, because a nuisance with zero draws
    must cost zero. `k` is therefore the whole per-drafting-round nuisance the
    receipt contains, whatever its source, and `k / 879` is that in units of
    Askeladd's zero-delta step.
    """
    x, y = [], []
    for prompt in NON_PLUTARCH:
        a = ship_row["per_prompt"][prompt]
        b = pb6_row["per_prompt"][prompt]
        draft_rounds = a_rows[prompt]["ranked_drafting_rounds_pb6"]
        x.append(100.0 * 1e-6 * draft_rounds / a["decode_s"])
        y.append(100.0 * (b["decode_s"] / a["decode_s"] - 1.0))
    xa, ya = np.asarray(x), np.asarray(y)
    k = float((xa * ya).sum() / (xa * xa).sum())
    resid = ya - k * xa
    return {
        "us_per_drafting_round": k,
        "steps": k / STATE_STEP_US,
        "sd_from_zero_delta_step": (k - STATE_STEP_US) / STATE_STEP_SD_US,
        "rms_residual_pp": float(np.sqrt((resid ** 2).mean())),
        "per_prompt_residual_pp": {p: float(resid[i])
                                   for i, p in enumerate(NON_PLUTARCH)},
    }


def control_steps(ship_row: dict, pb6_row: dict, a_rows: dict,
                  prompt: str) -> dict:
    """How many state draws one control prompt's decode leg contains."""
    a = ship_row["per_prompt"][prompt]
    b = pb6_row["per_prompt"][prompt]
    draft_rounds = a_rows[prompt]["ranked_drafting_rounds_pb6"]
    observed = 100.0 * (b["decode_s"] / a["decode_s"] - 1.0)
    per_step = one_step_pct(draft_rounds, a["decode_s"])
    return {
        "draft_len_change_pct": 100.0 * (b["draft_len"] / a["draft_len"]
                                         - 1.0),
        "observed_decode_pct": observed,
        "one_step_decode_pct": per_step,
        "steps": observed / per_step,
        "gap_pp": observed - per_step,
        "drafting_rounds_pb6": draft_rounds,
    }


def closure_prediction(ship: dict, pb6: dict, ship_row: dict) -> dict:
    """The replayer's zero-parameter forecast of the pb6 arm, per prompt.

    The replay prices the decode leg. The total leg carries the seed prefill
    as well, so the decode percentage is scaled by that prompt's own decode
    share rather than by a mean factor. That is Advisor Error 163 applied per
    prompt.
    """
    per_prompt = {}
    for prompt in RANKED_PROMPTS:
        decode_pct = 100.0 * (pb6[prompt]["us_per_token"]
                              / ship[prompt]["us_per_token"] - 1.0)
        share = ship_row["per_prompt"][prompt]["prefill_share"]
        per_prompt[prompt] = {
            "decode_pct": decode_pct,
            "decode_share_of_total_leg": 1.0 - share,
            "total_pct": decode_pct * (1.0 - share),
        }
    weights = {p: RANKED_PROMPTS[p]["weight"] for p in SCORING}
    return {
        "per_prompt": per_prompt,
        "decode_five_weighted_pct": sum(
            weights[p] * per_prompt[p]["decode_pct"]
            for p in SCORING) / sum(weights.values()),
        "total_five_weighted_pct": sum(
            weights[p] * per_prompt[p]["total_pct"]
            for p in SCORING) / sum(weights.values()),
    }


# --------------------------------------------------------------------- R5-c

def nnls(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Lawson-Hanson active-set non-negative least squares.

    `scipy` is not installed on this host, so the classical algorithm is
    written out. A clipped unconstrained fit is NOT the NNLS solution and
    would quietly change the solved curve.
    """
    _, n = a.shape
    passive = np.zeros(n, dtype=bool)
    x = np.zeros(n)
    w = a.T @ (b - a @ x)
    for _ in range(3 * n):
        free = ~passive
        if not free.any() or w[free].max() <= tol:
            break
        passive[np.flatnonzero(free)[int(np.argmax(w[free]))]] = True
        for _ in range(3 * n):
            s = np.zeros(n)
            s[passive] = np.linalg.lstsq(a[:, passive], b, rcond=None)[0]
            if s[passive].min() > tol:
                break
            blocking = passive & (s <= tol)
            ratios = x[blocking] / (x[blocking] - s[blocking])
            x = x + ratios.min() * (s - x)
            passive = passive & (x > tol)
            if not passive.any():
                break
        x = s
        w = a.T @ (b - a @ x)
    return x


def tail_design(mass: dict, widths: list[int]) -> np.ndarray:
    """One row of the monotone design matrix.

    The per-width cost is written as `c_w = c_1 + sum_{j<w} step_j`, so the
    per-round cost is `c_1 + sum_j step_j * P(width > j)`. The unknowns are a
    width-1 level and non-negative steps, and NNLS then enforces a
    non-decreasing curve with no extra penalty. Fitting the levels directly
    would allow a curve that falls with width, which no verify path does, and
    would discard the only prior available on a badly conditioned system.
    """
    row = [1.0]
    for j in widths[:-1]:
        row.append(sum(v for w, v in mass.items() if w > j))
    return np.asarray(row, dtype=float)


def solve_curve(design: np.ndarray, target: np.ndarray,
                widths: list[int]) -> dict:
    x = nnls(design, target)
    residual = target - design @ x
    curve, running = {}, x[0]
    curve[widths[0]] = running
    for index, width in enumerate(widths[1:], start=1):
        running += x[index]
        curve[width] = running
    steps = {"%d->%d" % (widths[i], widths[i + 1]): float(x[i + 1])
             for i in range(len(widths) - 1)}
    ranked_steps = sorted(steps.items(), key=lambda kv: -kv[1])
    cliff = ranked_steps[0]
    runner_up = ranked_steps[1]
    return {
        "curve_us": {int(w): float(v) for w, v in curve.items()},
        "steps_us": steps,
        "steps_pinned_at_zero": [k for k, v in steps.items() if v <= 1e-9],
        "cliff_boundary": int(cliff[0].split("->")[0]),
        "cliff_step_us": float(cliff[1]),
        "runner_up_boundary": int(runner_up[0].split("->")[0]),
        "runner_up_step_us": float(runner_up[1]),
        "cliff_margin_us": float(cliff[1] - runner_up[1]),
        "residual_us": residual.tolist(),
        "rms_residual_us": float(np.sqrt((residual ** 2).mean())),
        "max_abs_residual_us": float(np.abs(residual).max()),
        "relative_rms_pct": float(100.0 * np.sqrt((residual ** 2).mean())
                                  / target.mean()),
        "condition_number": float(np.linalg.cond(design)),
        "equations": int(design.shape[0]),
        "unknowns": int(design.shape[1]),
    }


def stack(blocks: list[tuple], widths: list[int], seed: int | None = None):
    """The design matrix, the target and the row labels for one solve.

    `seed` selects one replay seed's own width masses instead of the pooled
    masses. The target never changes: the board rows are the same rows. Only
    the design matrix moves, which is exactly the uncertainty a board-noise
    jitter cannot see.
    """
    design, target, labels = [], [], []
    for label, masses, costs in blocks:
        for prompt in sorted(costs):
            mass = (masses[prompt]["width_mass_per_seed"][seed]
                    if seed is not None else masses[prompt])
            design.append(tail_design(mass, widths))
            target.append(costs[prompt])
            labels.append("%s|%s" % (label, prompt))
    return np.vstack(design), np.asarray(target, dtype=float), labels


def rung_c(blocks: list[tuple], widths: list[int], jitter: int,
           rng: np.random.Generator, seed_blocks: list[tuple] | None = None,
           seeds: list[int] | None = None) -> dict:
    """Solve one family, then ask how far ordinary board noise moves it."""
    a, b, labels = stack(blocks, widths)
    out = solve_curve(a, b, widths)
    out["widths"] = widths
    out["labels"] = labels
    out["residual_us"] = {labels[i]: out["residual_us"][i]
                          for i in range(len(labels))}

    draws = {w: [] for w in widths}
    cliffs = []
    for _ in range(jitter):
        noisy = b * (1.0 + rng.normal(0.0, BOARD_ROW_CV, size=b.shape))
        trial = solve_curve(a, noisy, widths)
        for w in widths:
            draws[w].append(trial["curve_us"][w])
        cliffs.append(trial["cliff_boundary"])
    out["jitter"] = {
        "draws": jitter,
        "row_cv": BOARD_ROW_CV,
        "curve_sd_us": {int(w): float(np.std(draws[w])) for w in widths},
        "cliff_boundary_share": {
            int(v): cliffs.count(v) / len(cliffs)
            for v in sorted(set(cliffs))},
    }

    if seed_blocks is not None and seeds is not None:
        seed_draws = {w: [] for w in widths}
        seed_cliffs, per_seed = [], {}
        for seed in seeds:
            a_s, b_s, _ = stack(seed_blocks, widths, seed)
            trial = solve_curve(a_s, b_s, widths)
            for w in widths:
                seed_draws[w].append(trial["curve_us"][w])
            seed_cliffs.append(trial["cliff_boundary"])
            per_seed[int(seed)] = {
                "cliff_boundary": trial["cliff_boundary"],
                "cliff_step_us": trial["cliff_step_us"],
                "runner_up_boundary": trial["runner_up_boundary"],
                "cliff_margin_us": trial["cliff_margin_us"],
                "curve_us": trial["curve_us"],
                "rms_residual_us": trial["rms_residual_us"],
            }
        out["design_jitter"] = {
            "seeds": list(seeds),
            "per_seed": per_seed,
            "curve_sd_us": {int(w): float(np.std(seed_draws[w]))
                            for w in widths},
            "cliff_boundary_share": {
                int(v): seed_cliffs.count(v) / len(seed_cliffs)
                for v in sorted(set(seed_cliffs))},
        }
    return out


# --------------------------------------------------------------------- R5-d

def base_default_arm(root: pathlib.Path) -> dict:
    """The compiled default depth-price arm on this base, with its line."""
    path = root / SESSION
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = re.search(r"DepthPriceArm\(rawValue:\s*\w+\)\s*\?\?\s*\.(\w+)",
                          line)
        if match:
            return {"file": str(SESSION), "line": number,
                    "arm": match.group(1), "text": line.strip()}
    raise SystemExit("no compiled default depth-price arm found")


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=HERE / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--reports", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/e145")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--fit-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--jitter", type=int, default=400)
    ap.add_argument("--anchor", default="measured",
                    choices=("measured", "extrapolated", "serial"))
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e145-artifacts/r5.json")
    args = ap.parse_args()

    print("=== e145 R5: the pb6 identification gap and the ranked width"
          " curve ===")
    print("zero GPU")

    bar = load_row(args.board, BAR_ROW)
    ship_row = load_row(args.board, SHIP_ROW)
    pb6_row = load_row(args.board, PB6_ROW)
    for label, row in (("bar", bar), ("ship", ship_row), ("pb6", pb6_row)):
        print("  %-5s %s  %-14s  score %s  status %s  tree %s"
              % (label, row["id"][:8], row["solver"], row["score"],
                 row["status"], str(row["commit"])[:8]))

    prefill = local_prefill(args.reports)
    ranked_prefill = statistics.fmean(
        v["prefill_s"] for v in bar["per_prompt"].values())
    print("\n## R5-0  the local seed prefill, which F5 asked for")
    if prefill["available"]:
        prefill["ranked_prefill_seconds"] = ranked_prefill
        prefill["local_to_ranked_factor"] = (prefill["mean_seconds"]
                                             / ranked_prefill)
        print("  %d legs   mean %.6f s   median %.6f s   spread %.3f %%"
              % (prefill["legs"], prefill["mean_seconds"],
                 prefill["median_seconds"], prefill["spread_pct"]))
        print("  ranked seed prefill at %s is %.6f s, so the local to ranked"
              " prefill level factor is %.4f"
              % (BAR_ROW, ranked_prefill, prefill["local_to_ranked_factor"]))
    else:
        print("  no leg reports a seed prefill")

    # The replay panel.
    legs_e128, gate = build_legs(args.accept, args.runs)
    print("\n## attachment gate")
    print("  legs %d ; attached %d ; accept mismatches %d ; margin mismatches"
          " %d ; unmatched %d"
          % (gate["legs"], gate["attached"], gate["accept_mismatch"],
             gate["margin_mismatch"], gate["unmatched"]))
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")

    seeds = [args.seed + i for i in range(args.seeds)]
    cache = transfer_cache(legs_e128, args.windows, args.fit_windows, seeds)

    legs = json.loads(LEGS_JSON.read_text())["legs"]
    points, anchor = measured_points(legs)
    candidates = width1_candidates(points, anchor, serial_round_us(legs))
    points[1] = candidates[args.anchor]
    replayed_curves, best_form = load_curves()
    replayed = replayed_curves[best_form]
    measured = installable(points, best_form)

    ship = arm_behaviour(cache, seeds, args.windows, measured, "A_ship")
    pb6 = arm_behaviour(cache, seeds, args.windows, measured, "E_pb6")
    ship_rep = arm_behaviour(cache, seeds, args.windows, replayed, "A_ship")
    pb6_rep = arm_behaviour(cache, seeds, args.windows, replayed, "E_pb6")

    mass_gate = max(
        abs(ship[p]["width_mass"].get(w, 0.0)
            - ship_rep[p]["width_mass"].get(w, 0.0))
        for p in RANKED_PROMPTS
        for w in set(ship[p]["width_mass"]) | set(ship_rep[p]["width_mass"]))
    print("\n## the design matrix does not read a cost curve")
    print("  worst width-mass difference between the measured-curve replay"
          " and the replayed-curve replay: %.3e" % mass_gate)
    if mass_gate > 1e-12:
        raise SystemExit("the width masses depend on the cost curve; R5-c"
                         " would be circular")

    # R5-a.
    a_rows = rung_a(ship_row, pb6_row, ship, pb6)
    print("\n## R5-a  the closed-form round count and the accept-rate check")
    print("  %-9s %8s %8s %9s %9s %8s %9s %9s"
          % ("prompt", "dlen A", "dlen B", "R model", "R pinned", "ratio",
             "d accept", "R pb6 d%"))
    for prompt in RANKED_PROMPTS:
        r = a_rows[prompt]
        print("  %-9s %8.4f %8.4f %9.2f %9.2f %8.4f %+9.5f %+9.3f"
              % (prompt, r["ship_draft_len"], r["pb6_draft_len"],
                 r["model_rounds_ship"], r["f92_pinned_rounds_ship"],
                 r["round_ratio"], r["accept_rate_shift_pb6"],
                 r["rounds_pb6_shift_pct"]))
    worst_model = max(abs(r["model_minus_pinned_rounds"])
                      for r in a_rows.values())
    worst_shift = max(abs(r["rounds_pb6_shift_pct"]) for r in a_rows.values())
    inverted = [p for p in RANKED_PROMPTS if a_rows[p]["round_ratio"] < 1.0]
    survives = all(r["ratio_sign_survives_accept_shift"]
                   for r in a_rows.values())
    print("  the closed form reproduces every pinned count to %.2f rounds"
          % worst_model)
    print("  moving the accept rate by the replayed pb6 shift moves R_pb6 by"
          " at most %.3f %%" % worst_shift)
    print("  pb6 lowers the round count only for %s; every other prompt pays"
          " more rounds" % ", ".join(inverted))
    print("  every round-ratio sign survives the replayed accept-rate shift:"
          " %s" % survives)
    print("  the receipt says every prompt but plutarch drafts on every"
          " round; the replay disagrees, worst share gap %.4f"
          % max(abs(r["receipt_drafting_share_pb6"]
                    - r["replay_drafting_share_pb6"])
                for r in a_rows.values()))

    # R5-b.
    closure = closure_prediction(ship, pb6, ship_row)
    closure_rep = closure_prediction(ship_rep, pb6_rep, ship_row)
    steps = (0, 1, 2, 3)
    lattice = rung_b(ship_row, pb6_row, a_rows, steps, closure)
    drama = control_steps(ship_row, pb6_row, a_rows, "drama")
    travel = control_steps(ship_row, pb6_row, a_rows, "travel")
    free_fit = free_fit_steps(ship_row, pb6_row, a_rows)

    print("\n## R5-b  the state-step lattice, decode frame")
    print("  %-11s %s" % ("prompt", " ".join("%9s" % ("s=%d" % s)
                                             for s in steps)))
    for prompt in RANKED_PROMPTS:
        print("  %-11s %s" % (prompt, " ".join(
            "%+9.4f" % lattice[s]["per_prompt"][prompt]["decode_pct"]
            for s in steps)))
    for key, label in (("decode_seven_mean_pct", "mean7"),
                       ("decode_five_mean_pct", "mean5"),
                       ("decode_five_weighted_pct", "mean5w"),
                       ("total_five_mean_pct", "total5"),
                       ("total_five_weighted_pct", "total5w"),
                       ("corrected_median_pct", "medianpair")):
        print("  %-11s %s" % (label, " ".join(
            "%+9.4f" % lattice[s][key] for s in steps)))
    print("  mean5 and total5 are unweighted over the five median-eligible"
          " prompts, which is F7's definition; mean5w and total5w apply the"
          " E128 median-influence weights instead")
    print("  medianpair is the only row that is the published score, and it"
          " is positive when pb6 raises the score")

    print("\n## R5-b  three readings of the step count")
    print("  drama control      %.4f steps  (dlen moved %+.3f %%, observed"
          " %+.4f %%, one step %+.4f %%, gap %+.4f pp)"
          % (drama["steps"], drama["draft_len_change_pct"],
             drama["observed_decode_pct"], drama["one_step_decode_pct"],
             drama["gap_pp"]))
    print("  travel control     %.4f steps  (dlen moved %+.3f %%, gap"
          " %+.4f pp)"
          % (travel["steps"], travel["draft_len_change_pct"],
             travel["gap_pp"]))
    print("  free fit on seven  %.4f steps  (%.1f us per drafting round,"
          " %.2f sd from the zero-delta step, residual %.4f pp)"
          % (free_fit["steps"], free_fit["us_per_drafting_round"],
             free_fit["sd_from_zero_delta_step"],
             free_fit["rms_residual_pp"]))

    print("\n## R5-b  the replayer's zero-parameter closure prediction")
    print("  %-9s %10s %10s %10s"
          % ("prompt", "predicted", "s=1 obs", "gap pp"))
    for prompt in RANKED_PROMPTS:
        pred = closure["per_prompt"][prompt]["decode_pct"]
        obs = lattice[1]["per_prompt"][prompt]["decode_pct"]
        print("  %-9s %+10.4f %+10.4f %+10.4f" % (prompt, pred, obs,
                                                  pred - obs))
    print("  five-weighted predicted %+.4f %% on the measured curve, %+.4f %%"
          " on the replayed curve"
          % (closure["decode_five_weighted_pct"],
             closure_rep["decode_five_weighted_pct"]))
    best_s = min(steps,
                 key=lambda s: lattice[s]["weighted_sq_error_vs_closure"])
    print("  the closure prediction is closest to s = %d (rms %.4f pp)"
          % (best_s, lattice[best_s]["rms_error_vs_closure_pp"]))

    # `travel` is held out of the primary interval because its own draft
    # length moved 0.806 %, six times drama's, so part of its gap is a real
    # schedule effect and not a state draw. It is reported as a sensitivity.
    clean = {"drama_control": drama["steps"], "free_fit": free_fit["steps"]}
    readings = dict(clean, travel_control=travel["steps"])
    control_mean = statistics.fmean(readings.values())
    # One standard deviation of the zero-delta step size widens the interval,
    # because s and the step size enter the correction only as their product.
    tol = STATE_STEP_SD_US / STATE_STEP_US
    s_lo = min(clean.values()) * (1.0 - tol)
    s_hi = max(clean.values()) * (1.0 + tol)
    s_hi_travel = max(readings.values()) * (1.0 + tol)
    score_lo = effects_at(ship_row, pb6_row, a_rows,
                          s_lo)["corrected_median_pct"]
    score_hi = effects_at(ship_row, pb6_row, a_rows,
                          s_hi)["corrected_median_pct"]
    score_hi_travel = effects_at(ship_row, pb6_row, a_rows,
                                 s_hi_travel)["corrected_median_pct"]
    crossings = {
        key: zero_crossing(ship_row, pb6_row, a_rows, key)
        for key in ("corrected_median_pct", "total_five_mean_pct",
                    "decode_seven_mean_pct")}
    score_cross = crossings["corrected_median_pct"]

    if (score_lo > 0.0) != (score_hi > 0.0):
        verdict = "undecided"
    elif max(abs(score_lo), abs(score_hi)) < 0.2:
        verdict = "neutral"
    elif score_lo > 0.0:
        verdict = "win_confirmed"
    else:
        verdict = "loss_confirmed"

    print("\n## R5-b  the verdict, read on the published median pair")
    print("  the two clean readings and one step-size sd put s in [%.4f,"
          " %.4f]; adding the travel reading raises the top to %.4f"
          % (s_lo, s_hi, s_hi_travel))
    print("  the score effect is %+.4f %% at s=%.4f and %+.4f %% at s=%.4f;"
          " at the travel top it is %+.4f %%"
          % (score_lo, s_lo, score_hi, s_hi, score_hi_travel))
    print("  the score effect changes sign at s = %s"
          % ("%.4f state steps" % score_cross if score_cross is not None
             else "no crossing below 6 steps"))
    for key, label in (("total_five_mean_pct", "total5"),
                       ("decode_seven_mean_pct", "mean7")):
        print("  %s changes sign at s = %s"
              % (label, "%.4f" % crossings[key] if crossings[key] is not None
                 else "no crossing below 6 steps"))
    print("  verdict: %s" % verdict)
    if (score_hi > 0.0) != (score_hi_travel > 0.0):
        print("  the verdict is not robust to the travel reading; that one"
              " contaminated control is the whole distance to the other sign")

    # R5-c.
    widths = sorted({w for p in RANKED_PROMPTS for w in ship[p]["width_mass"]}
                    | {w for p in RANKED_PROMPTS
                       for w in pb6[p]["width_mass"]})
    unidentified = [w for w in range(1, MAX_DEPTH + 2) if w not in widths]
    print("\n## R5-c  the ranked width cost curve, solved from the board")
    print("  widths carrying mass: %s" % ", ".join(str(w) for w in widths))
    if unidentified:
        print("  widths %s carry no mass under either arm and are NOT"
              " identified; the segmented verify cap holds the walk below them"
              % ", ".join(str(w) for w in unidentified))

    ship_mass = {p: ship[p]["width_mass"] for p in RANKED_PROMPTS}
    pb6_mass = {p: pb6[p]["width_mass"] for p in RANKED_PROMPTS}
    bar_costs = {p: 1e6 * v["decode_s"] / F92_ROUNDS[p]
                 for p, v in bar["per_prompt"].items()}
    ship_costs = {p: 1e6 * v["decode_s"] / F92_ROUNDS[p]
                  for p, v in ship_row["per_prompt"].items()}
    pb6_costs = {p: 1e6 * v["decode_s"] / a_rows[p]["ranked_rounds_pb6"]
                 for p, v in pb6_row["per_prompt"].items()}
    pb6_corrected = {
        p: c - control_mean * STATE_STEP_US
        * a_rows[p]["ranked_drafting_rounds_pb6"]
        / a_rows[p]["ranked_rounds_pb6"]
        for p, c in pb6_costs.items()}
    level = statistics.fmean([ship_costs[p] / bar_costs[p]
                              for p in RANKED_PROMPTS])

    print("\n  prefill-free ranked cost per round, microseconds")
    print("  %-9s %10s %10s %10s %10s"
          % ("prompt", "bar", "ship", "pb6", "pb6 corr"))
    for prompt in RANKED_PROMPTS:
        print("  %-9s %10.1f %10.1f %10.1f %10.1f"
              % (prompt, bar_costs[prompt], ship_costs[prompt],
                 pb6_costs[prompt], pb6_corrected[prompt]))
    print("  our tree runs %+.3f %% against the bar at identical draft"
          " lengths, so the two families are solved separately"
          % (100.0 * (level - 1.0)))

    rng = np.random.default_rng(20260823)
    solves = {
        "bar_only": rung_c(
            [("bar", ship_mass, bar_costs)], widths, args.jitter, rng,
            [("bar", ship, bar_costs)], seeds),
        "ours_ship_only": rung_c(
            [("ship", ship_mass, ship_costs)], widths, args.jitter, rng,
            [("ship", ship, ship_costs)], seeds),
        "ours_ship_plus_pb6_raw": rung_c(
            [("ship", ship_mass, ship_costs), ("pb6", pb6_mass, pb6_costs)],
            widths, args.jitter, rng,
            [("ship", ship, ship_costs), ("pb6", pb6, pb6_costs)], seeds),
        "ours_ship_plus_pb6_state_corrected": rung_c(
            [("ship", ship_mass, ship_costs),
             ("pb6", pb6_mass, pb6_corrected)], widths, args.jitter, rng,
            [("ship", ship, ship_costs), ("pb6", pb6, pb6_corrected)], seeds),
    }

    print("\n  solved per-width ranked round cost, microseconds")
    print("  %-36s %s" % ("variant", " ".join("%9d" % w for w in widths)))
    for name, sol in solves.items():
        print("  %-36s %s" % (name, " ".join(
            "%9.1f" % sol["curve_us"][w] for w in widths)))
    print("\n  %-36s %9s %9s %10s %6s %6s %9s %s"
          % ("variant", "rms us", "rel rms%", "cond", "cliff", "next",
             "margin us", "pinned zero"))
    for name, sol in solves.items():
        print("  %-36s %9.1f %9.4f %10.3e %6d %6d %9.1f  %s"
              % (name, sol["rms_residual_us"], sol["relative_rms_pct"],
                 sol["condition_number"], sol["cliff_boundary"],
                 sol["runner_up_boundary"], sol["cliff_margin_us"],
                 ",".join(sol["steps_pinned_at_zero"]) or "none"))

    headline = solves["ours_ship_plus_pb6_state_corrected"]
    moved = headline["cliff_boundary"] != LOCAL_CLIFF_FROM_WIDTH
    print("\n  the ranked cliff is at %d -> %d; the local R2 cliff is at"
          " %d -> %d; moved: %s"
          % (headline["cliff_boundary"], headline["cliff_boundary"] + 1,
             LOCAL_CLIFF_FROM_WIDTH, LOCAL_CLIFF_FROM_WIDTH + 1, moved))
    print("  the runner-up boundary is %d -> %d, %.1f us behind"
          % (headline["runner_up_boundary"],
             headline["runner_up_boundary"] + 1, headline["cliff_margin_us"]))
    print("  under %d board-noise draws at cv %.5f the cliff boundary lands"
          " at %s"
          % (args.jitter, BOARD_ROW_CV,
             ", ".join("%d in %.1f %% of draws" % (k, 100.0 * v)
                       for k, v in sorted(
                           headline["jitter"]["cliff_boundary_share"].items())
                       )))
    print("  but the design matrix is replayed, and re-solving with each"
          " seed's own width masses gives %s"
          % ", ".join(
              "%d in %.1f %% of seeds" % (k, 100.0 * v)
              for k, v in sorted(
                  headline["design_jitter"]["cliff_boundary_share"].items())))
    print("  per-width sd from board noise vs from the replayed design, us")
    print("  %-14s %s" % ("board noise", " ".join(
        "%9.1f" % headline["jitter"]["curve_sd_us"][w] for w in widths)))
    print("  %-14s %s" % ("design", " ".join(
        "%9.1f" % headline["design_jitter"]["curve_sd_us"][w]
        for w in widths)))
    dominated = [w for w in widths
                 if headline["design_jitter"]["curve_sd_us"][w]
                 >= headline["jitter"]["curve_sd_us"][w]]
    print("  the replayed design matrix carries the larger uncertainty at"
          " %d of %d widths: %s"
          % (len(dominated), len(widths),
             ", ".join(str(w) for w in dominated) or "none"))

    # R5-d.
    default_arm = base_default_arm(HERE.parent)
    print("\n## R5-d  the compiled default depth-price arm on this base")
    print("  %s:%d  %s" % (default_arm["file"], default_arm["line"],
                           default_arm["text"]))

    blob = {
        "harness": {"r5a": "ranked", "r5b": "ranked", "r5c": "ranked",
                    "replay": "local instrument"},
        "gpu_used": False,
        "seeds": seeds,
        "windows": args.windows,
        "width1_anchor_used": args.anchor,
        "state_step_us": STATE_STEP_US,
        "state_step_sd_us": STATE_STEP_SD_US,
        "state_step_n": STATE_STEP_N,
        "local_seed_prefill": prefill,
        "design_matrix_curve_independence_max_diff": mass_gate,
        "r5a": a_rows,
        "r5a_worst_model_minus_pinned_rounds": worst_model,
        "r5a_worst_rounds_pb6_shift_pct": worst_shift,
        "r5a_inversion_prompts": inverted,
        "r5a_inversion_survives_accept_shift": survives,
        "r5a_width_mass": {"ship": ship_mass, "pb6": pb6_mass},
        "r5b_lattice": {str(s): lattice[s] for s in steps},
        "r5b_drama_control": drama,
        "r5b_travel_control": travel,
        "r5b_free_fit": free_fit,
        "r5b_closure_measured_curve": closure,
        "r5b_closure_replayed_curve": closure_rep,
        "r5b_closure_best_state_steps": best_s,
        "r5b_step_readings": readings,
        "r5b_control_mean_steps": control_mean,
        "r5b_step_reading_low": s_lo,
        "r5b_step_reading_high": s_hi,
        "r5b_step_reading_high_with_travel": s_hi_travel,
        "r5b_score_pct_at_low_reading": score_lo,
        "r5b_score_pct_at_high_reading": score_hi,
        "r5b_score_pct_at_travel_reading": score_hi_travel,
        "r5b_sign_change_state_steps": crossings,
        "r5b_verdict": verdict,
        "r5b_verdict_robust_to_travel":
            (score_hi > 0.0) == (score_hi_travel > 0.0),
        "r5c_cost_per_round_us": {"bar": bar_costs, "ship": ship_costs,
                                  "pb6": pb6_costs,
                                  "pb6_corrected": pb6_corrected},
        "r5c_our_tree_vs_bar_level_pct": 100.0 * (level - 1.0),
        "r5c_solves": solves,
        "r5c_headline": "ours_ship_plus_pb6_state_corrected",
        "r5c_unidentified_widths": unidentified,
        "r5c_local_cliff_from_width": LOCAL_CLIFF_FROM_WIDTH,
        "r5c_ranked_cliff_boundary": headline["cliff_boundary"],
        "r5c_ranked_runner_up_boundary": headline["runner_up_boundary"],
        "r5c_ranked_cliff_margin_us": headline["cliff_margin_us"],
        "r5c_ranked_vs_local_cliff_moved": 1.0 if moved else 0.0,
        "r5c_widths_where_design_beats_board_noise": dominated,
        "r5d_base_default_arm": default_arm,
        "r5d_base_ships_pb6": 1.0 if default_arm["arm"] == "pb6" else 0.0,
        "board_rows": {"bar": bar, "ship": ship_row, "pb6": pb6_row},
        "identification_caveat":
            "A per-round scheduler mechanism and the per-drafting-round"
            " wired-residency state share the same basis when nearly every"
            " round is a drafting round, so the per-prompt shape alone cannot"
            " separate them. They are separated here by three readings that"
            " do not use the shape: the drama control, whose schedule pb6"
            " provably does not move; a free fit of the coefficient itself;"
            " and the replayer's prediction of the mechanism from the width"
            " masses without seeing the receipt. R5-c inherits a second"
            " limit: the design matrix is replayed, and the transfer is"
            " fitted to one moment per prompt, the published effective mean"
            " draft length, so the width distribution around that mean is a"
            " model output and not a measurement.",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(blob, indent=2, sort_keys=True,
                                    default=float) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
