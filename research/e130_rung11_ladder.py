#!/usr/bin/env python3
"""E130 rung 11, F16: read the wired-slack ladder.

Four arms, one binary, differing only by
``DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB``::

    s64    the rung 10a anchor, measured there at -0.1968 % against s512
    s512   the currently shipped value
    s1024  untimed before this session
    s2048  untimed before this session, and the bound-C hard ceiling

MODEL. ``y ~ arm + leg_index``. Rung 10a's palindrome made the trend exactly
orthogonal to every arm contrast, so the normal equations separated. The rung
11 order is a palindrome over the first eight legs plus one pre-registered
random permutation over the last four, which buys degrees of freedom at the
cost of that orthogonality. So this reader solves the normal equations
properly and takes every contrast standard error from the inverted
information matrix rather than assuming balance.

THE HEADLINE IS ABSOLUTE CANDIDATE SECONDS PER TOKEN. Finding 160: the local
ratio adds 0.4788 % of serial noise to a 0.0498 % measurement, a 9.6x variance
penalty, on an estimand the ranked score does not use. The serial channel and
the ratio are fitted too, but only as reported controls.

PRE-REGISTERED. The linear-in-admitted-bytes model and the saturating null
were written into PR 130 interim 17 before any leg ran, together with the
decision rule and the seed of the random permutation.

Usage
-----
    python3 research/e130_rung11_ladder.py --prefix e130-r11 \
        --out research/e130-artifacts/rung11-slack-ladder.json --wandb
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import e130_rung11_rounds

ORDER = [
    "s64", "s512", "s1024", "s2048",
    "s2048", "s1024", "s512", "s64",
    "s2048", "s512", "s64", "s1024",
]
ARMS = ("s64", "s512", "s1024", "s2048")

# Leg 13 is a thirteenth s512 leg run with the round trace switched off. It is
# deliberately outside ORDER and outside ARMS, so it never enters the fit. Its
# only job is to price the trace itself, which every fitted leg pays.
UNTRACED_ARM = "s512untraced"
UNTRACED_INDEX = 13
SLACK_MB = {"s64": 64, "s512": 512, "s1024": 1024, "s2048": 2048,
            UNTRACED_ARM: 512}

# Two-sided 95 % Student t. Only the degrees of freedom this design can produce.
T_CRIT_95 = {
    1: 12.7062, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
    6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
    11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
}

# Percent change in candidate seconds per token under each pre-registered
# model. All three were written down before any leg ran.
#
# CONSTANT SLOPE (F16 section 3, renamed after F17 section 1). This is the one
# measured rung 10a slope extrapolated: -0.1968 % over +448 MiB is -0.2249 %
# per 512 MiB. It carries no independent claim about the slope, only about how
# far the slope reaches. A null here refutes the reach, not the slope.
#
# SATURATING NULL. Predicts 0.0 everywhere. F17 section 2 points out that this
# is the wrong alternative and that our own rung 10 data already rejects it:
# unwired remaining at s512 is 2,442.7 / 2,765.4 / 2,815.0 MiB across the three
# roles, so under first-come admission the useful bytes do not run out until
# roughly 2,400 MiB above s512, which is above the 2,048 MiB ceiling. It is
# kept as a reported reference point, not as the live alternative.
#
# DECLINING MARGINAL VALUE (F17 section 2). The physically motivated
# alternative. The s64 -> s512 admissions were dominated by the 17,825,792 B
# per-layer KV class, which is the hottest state in the round. If early
# admissions are hot by accident of size class, later admissions are colder and
# the slope decays without ever reaching zero.
PREREGISTERED = {
    ("s512", "s1024"): {"constant_slope": -0.225, "saturating": 0.0,
                        "declining_value": -0.15},
    ("s1024", "s2048"): {"constant_slope": -0.450, "saturating": 0.0,
                         "declining_value": -0.20},
    ("s512", "s2048"): {"constant_slope": -0.675, "saturating": 0.0,
                        "declining_value": -0.35},
    ("s64", "s2048"): {"constant_slope": -0.872, "saturating": 0.0,
                       "declining_value": -0.55},
}
MODELS = ("constant_slope", "saturating", "declining_value")

# --------------------------------------------------------------------------
# F19 section 2 and 3. Position dependence of the s64 -> s512 admission.
#
# The rung 10a anchor was measured at 64 decode tokens. Campaign rule 105
# forbids carrying that percentage to 512 tokens unless the mechanism is proven
# window invariant, so this ladder re-measures the same contrast at 512 and
# these constants say what each mechanism predicts we should see.
#
# The s64 -> s512 step admits 448.0 MiB (rung 11 admission screen, slope
# 1.000). 136.0 MiB of that is eight buffers of 17,825,792 B in the per-layer
# KV size class; the other 312.0 MiB is not KV. A KV page holds 4,352 tokens
# per layer, so neither window changes the ALLOCATION, only how much of the
# page is TOUCHED. Touched KV bytes go as the mean sequence length over the
# timed window: 512 + 64/2 = 544 tokens against 512 + 512/2 = 768 tokens, at
# 8 * len * 4096 B, which is 17.00 MiB at 64 tokens and 24.00 MiB at 512.
#
# Non-KV admitted bytes are touched by an unknown fraction f that we assume is
# the same at both windows, because GDN recurrent state and scratch are sized
# by the model and not by position. Benefit is taken proportional to touched
# bytes, so the 512-token effect is the anchor scaled by
#
#     ratio(f) = (f * 312.0 + 24.00) / (f * 312.0 + 17.00)
#
# which is 1.0213 at f = 1 and 1.4118 at f = 0. That is the whole content of
# the split: f near 1 means the tax rides on position-invariant state and the
# percentage barely moves, f near 0 means it rides on KV pages and the
# percentage grows by 41 %.
ANCHOR_PCT_64 = -0.1968
ANCHOR_ADMITTED_MIB = 448.0
KV_ADMITTED_MIB = 136.0
NONKV_ADMITTED_MIB = ANCHOR_ADMITTED_MIB - KV_ADMITTED_MIB
KV_TOUCHED_MIB = {64: 17.00, 512: 24.00}

# The third mechanism. If the tax is paid once in round 1 rather than every
# round, it is a fixed 4.033 ms per leg, which is -0.1963 % of a 64-token leg
# and only -0.0245 % of a 512-token leg.
ONE_TIME_MS_PER_LEG = 4.033
ONE_TIME_PCT_512 = -0.0245

# Pre-registered in PR 130 interim 21 before this ladder ran: f = 1 dominates,
# so the 512-token contrast should land near -0.2010 % and not near -0.2778 %.
PREDICTED_PCT_512 = -0.2010
POSITION_GRID = (1.0, 0.75, 0.5, 0.25, 0.0)


def read_meta(path: Path) -> dict:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


def swap_fields(blob: str) -> dict:
    return {k: int(v) for k, v in re.findall(r"([a-z_]+)=([0-9]+)", blob or "")}


def read_leg(root: Path, prefix: str, index: int, arm: str) -> dict:
    tag = f"{prefix}-{index:02d}-{arm}"
    out = root / tag
    meta = read_meta(out / "meta.txt")
    leg = {
        "tag": tag,
        "index": index,
        "arm": arm,
        "slack_mb": SLACK_MB[arm],
        "exit": meta.get("exit"),
        "wired_residency_active": meta.get("wired_residency_active"),
        "wired_outcome_line": meta.get("wired_outcome_line"),
        "wired_clamped_count": int(meta.get("wired_clamped_count", "0") or 0),
        "wired_apply_failures": int(meta.get("wired_apply_failures", "0") or 0),
        "wired_slack_mb": meta.get("wired_slack_mb"),
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"])
        if meta.get("gpu_temp_entry_c") else None,
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"])
        if meta.get("gpu_temp_exit_c") else None,
        "worker_sha256": meta.get("worker_sha256"),
        "base_sha": meta.get("base_sha"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "leg_trace": meta.get("leg_trace"),
        "trace_anchor_lines": int(meta.get("trace_anchor_lines", "0") or 0),
        "trace_round_lines": int(meta.get("trace_round_lines", "0") or 0),
        "trace_row_lines": int(meta.get("trace_row_lines", "0") or 0),
    }
    entry = swap_fields(meta.get("swap_entry", ""))
    exit_ = swap_fields(meta.get("swap_exit", ""))
    leg["swap_delta"] = {
        key: exit_.get(key, 0) - entry.get(key, 0)
        for key in sorted(set(entry) | set(exit_))
    }
    leg["swapped"] = leg["swap_delta"].get("swapouts", 0) > 0

    score_path = out / "score.json"
    if score_path.exists():
        metrics = json.loads(score_path.read_text()).get("metrics", {})
        leg["mtp_seconds_per_token"] = metrics.get("mtp_seconds_per_token")
        leg["serial_seconds_per_token"] = metrics.get("serial_seconds_per_token")
        leg["mtp_decode_speedup"] = metrics.get("mtp_decode_speedup")
        leg["all_tokens_matched"] = metrics.get("all_tokens_matched")
        leg["decode_tokens"] = metrics.get("decode_tokens")

    # F18. Channel B is the steady-state decode rate and channel C is the
    # depth-controlled round-1 excess. B is NOT comparable in level with the
    # reported seconds per token, because the report includes the 512-token
    # seed prefill and B is built only from decode rounds. Only the arm
    # contrasts of the two channels may be compared.
    rounds = e130_rung11_rounds.read_leg(out)
    if "error" not in rounds:
        leg["steady_seconds_per_token"] = rounds["steady_seconds_per_token"]
        leg["c_raw_us"] = rounds["c_raw_us"]
        leg["c_depth_matched_us"] = rounds["c_depth_matched_us"]
        leg["c_regression_us"] = rounds["c_regression_us"]
        leg["depth_slope_us_per_draft"] = rounds["depth_slope_us_per_draft"]
        leg["round_count"] = rounds["rounds"]
        leg["round1_depth"] = rounds["round1_depth"]
        leg["timed_pid"] = rounds["timed_pid"]
    else:
        leg["round_trace_error"] = rounds["error"]
    return leg


def invert(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan with partial pivoting. The design is 5x5, so this is cheap."""
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise SystemExit(f"singular design at column {col}: an arm has no legs")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def fit(legs: list[dict], field: str) -> dict | None:
    """Least squares for ``y = mean(arm) + slope * (leg_index - centre)``."""
    # NaN is excluded as well as None. The F18 round-1 channel returns NaN when
    # a leg's steady rounds carry one draft count, so its slope is unidentified,
    # and a single NaN would otherwise poison the whole normal-equation solve.
    used = [leg for leg in legs
            if leg.get(field) is not None
            and not (isinstance(leg[field], float) and math.isnan(leg[field]))]
    present = [arm for arm in ARMS if any(leg["arm"] == arm for leg in used)]
    if len(present) < 2:
        return None

    centre = sum(leg["index"] for leg in used) / len(used)
    columns = list(present) + ["trend"]

    def design_row(leg: dict) -> list[float]:
        row = [1.0 if leg["arm"] == arm else 0.0 for arm in present]
        row.append(leg["index"] - centre)
        return row

    rows = [design_row(leg) for leg in used]
    values = [float(leg[field]) for leg in used]
    k = len(columns)
    df = len(used) - k
    if df < 1:
        return None

    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * y for r, y in zip(rows, values)) for i in range(k)]
    inv = invert(xtx)
    beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]

    fitted = [sum(b * v for b, v in zip(beta, r)) for r in rows]
    residuals = [y - f for y, f in zip(values, fitted)]
    sse = sum(r * r for r in residuals)
    sigma2 = sse / df
    sigma = sigma2 ** 0.5
    grand = sum(values) / len(values)

    # How much of the arm-means-only residual the single trend column removes.
    means = {
        arm: sum(y for y, leg in zip(values, used) if leg["arm"] == arm)
        / sum(1 for leg in used if leg["arm"] == arm)
        for arm in present
    }
    ss_arm_only = sum((y - means[leg["arm"]]) ** 2 for y, leg in zip(values, used))

    return {
        "n": len(used),
        "df": df,
        "columns": columns,
        "arm_means_adjusted": {arm: beta[i] for i, arm in enumerate(present)},
        "arm_means_raw": means,
        "slope_per_leg": beta[-1],
        "slope_pct_per_leg": 100.0 * beta[-1] / grand,
        "residual_sd": sigma,
        "residual_sd_pct": 100.0 * sigma / grand,
        "trend_share_of_arm_only_residual":
            1.0 - sse / ss_arm_only if ss_arm_only > 0 else float("nan"),
        "grand_mean": grand,
        "centre": centre,
        "residuals": {leg["tag"]: r for leg, r in zip(used, residuals)},
        "_inv": inv,
        "_sigma2": sigma2,
        "_present": present,
    }


def contrast(model: dict, lo: str, hi: str) -> dict:
    """Signed effect of moving from ``lo`` to ``hi``, in percent of ``lo``.

    The standard error comes from the inverted information matrix, so it stays
    correct even though the last four legs break the palindrome's balance.
    """
    present = model["_present"]
    if lo not in present or hi not in present:
        return {"from": lo, "to": hi, "usable": False}
    k = len(model["columns"])
    vector = [0.0] * k
    vector[present.index(hi)] = 1.0
    vector[present.index(lo)] = -1.0
    inv = model["_inv"]
    variance = model["_sigma2"] * sum(
        vector[i] * inv[i][j] * vector[j] for i in range(k) for j in range(k)
    )
    se = variance ** 0.5
    base = model["arm_means_adjusted"][lo]
    delta = model["arm_means_adjusted"][hi] - base
    t_crit = T_CRIT_95.get(model["df"], 1.96)

    # Significance is decided on the absolute scale. Dividing by a base that is
    # negative or near zero flips the sign of t and inverts the interval, which
    # silently reported a null round-1 channel as significant.
    t = delta / se if se else float("nan")
    out = {
        "from": lo,
        "to": hi,
        "usable": True,
        "delta": delta,
        "se_abs": se,
        "t": t,
        "t_crit_95": t_crit,
        "ci95_abs": [delta - t_crit * se, delta + t_crit * se],
        "significant": abs(t) > t_crit if se else False,
    }
    if base > 0:
        pct = 100.0 * delta / base
        se_pct = 100.0 * se / base
        half = t_crit * se_pct
        out.update({
            "pct": pct,
            "se_pct": se_pct,
            "ci95_pct": [pct - half, pct + half],
        })
    else:
        # A percentage of a non-positive base is not interpretable. The channel
        # is read in its own units instead.
        out.update({"pct": None, "se_pct": None, "ci95_pct": None,
                    "pct_undefined_because_base_not_positive": True})
    return out


def decision(pct: float | None) -> str:
    """The F16 decision rule on the s512 -> s2048 contrast, applied verbatim."""
    if pct is None:
        return "undecided: the s512 to s2048 contrast is missing"
    if pct < -0.35:
        return ("ship the ladder argmax; the linear-in-admitted-bytes model is "
                "confirmed")
    if pct < -0.05:
        return "ship the ladder argmax; the response is sub-linear"
    return ("the linear model is refuted; 512 ships unchanged and the ceiling "
            "is now measured rather than guessed")


def safety(legs: list[dict], model: dict | None) -> dict:
    """F16 section 5. Every one of these must stay clean or the ladder stops."""
    clamped = [leg["tag"] for leg in legs if leg["wired_clamped_count"]]
    failures = [leg["tag"] for leg in legs if leg["wired_apply_failures"]]
    inactive = [leg["tag"] for leg in legs
                if leg.get("wired_residency_active") != "true"]
    swapped = [leg["tag"] for leg in legs if leg.get("swapped")]

    # "any leg slower than its s1024 counterpart by more than 2 sigma"
    regressions = []
    if model:
        s1024 = model["arm_means_adjusted"].get("s1024")
        sigma = model["residual_sd"]
        if s1024 and sigma:
            for leg in legs:
                if leg["arm"] != "s2048" or leg.get("mtp_seconds_per_token") is None:
                    continue
                excess = leg["mtp_seconds_per_token"] - s1024
                if excess > 2.0 * sigma:
                    regressions.append({
                        "tag": leg["tag"],
                        "excess_seconds_per_token": excess,
                        "excess_sigma": excess / sigma,
                    })
    return {
        "clamp_bound_on_any_leg": bool(clamped),
        "clamped_legs": clamped,
        "wired_apply_failed_on_any_leg": bool(failures),
        "apply_failure_legs": failures,
        "wiring_inactive_legs": inactive,
        "swapped_legs": swapped,
        "s2048_legs_slower_than_s1024_by_2sigma": regressions,
        "all_clear": not (clamped or failures or inactive or swapped
                          or regressions),
    }


def thermal(legs: list[dict]) -> dict:
    entries = [leg["gpu_temp_entry_c"] for leg in legs
               if leg.get("gpu_temp_entry_c") is not None]
    exits = [leg["gpu_temp_exit_c"] for leg in legs
             if leg.get("gpu_temp_exit_c") is not None]
    return {
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "entry_temp_c": entries,
        "entry_temp_spread_c": max(entries) - min(entries) if entries else None,
        "exit_temp_c": exits,
        "exit_temp_spread_c": max(exits) - min(exits) if exits else None,
    }


def position_prediction(fraction: float) -> dict:
    """Carry the 64-token anchor to 512 tokens for one non-KV touched fraction."""
    at64 = fraction * NONKV_ADMITTED_MIB + KV_TOUCHED_MIB[64]
    at512 = fraction * NONKV_ADMITTED_MIB + KV_TOUCHED_MIB[512]
    ratio = at512 / at64
    return {
        "nonkv_touched_fraction": fraction,
        "touched_mib_at_64": at64,
        "touched_mib_at_512": at512,
        "ratio": ratio,
        "pct": ANCHOR_PCT_64 * ratio,
    }


def implied_touched_fraction(pct: float) -> float | None:
    """Invert ratio(f). None when the measurement sits off this family."""
    ratio = pct / ANCHOR_PCT_64
    denominator = NONKV_ADMITTED_MIB * (1.0 - ratio)
    if abs(denominator) < 1e-9:
        return None
    return (ratio * KV_TOUCHED_MIB[64] - KV_TOUCHED_MIB[512]) / denominator


def position_dependence(models: dict) -> dict:
    """F19 section 3. Decide which mechanism pays the wired-slack tax.

    Three channels are read on the one contrast that already has a 64-token
    anchor, s64 -> s512:

        A   candidate seconds per token over the whole timed leg
        B   steady-state decode seconds per token, round 1 removed
        C   depth-controlled round-1 excess, in milliseconds per leg

    B says WHERE the tax is paid, A says HOW BIG it is at the only window the
    ranked score uses, and C is the cross check, because a one-time tax has to
    appear there in milliseconds and has to leave B alone.

    A IS THE SHIPPING NUMBER EITHER WAY. It is a direct 512-token measurement,
    so the mechanism does not change it; the mechanism only decides whether
    that percentage may be carried to any other window under campaign rule 105.
    """
    primary = models.get("candidate_mtp_seconds_per_token")
    steady = models.get("steady_seconds_per_token")
    round1 = models.get("round1_excess_depth_matched_us")
    regression = models.get("round1_excess_regression_us")

    out: dict = {
        "contrast": "s64_to_s512",
        "decode_tokens": 512,
        "anchor_pct_at_64_tokens": ANCHOR_PCT_64,
        "preregistered_pct_at_512": PREDICTED_PCT_512,
        "preregistered_in": "PR 130 interim 21, before any leg of this ladder ran",
        "touched_byte_family": [position_prediction(f) for f in POSITION_GRID],
    }
    if not primary:
        out["usable"] = False
        out["verdict"] = "undecided: the candidate channel did not fit"
        return out

    a = contrast(primary, "s64", "s512")
    if not a.get("usable"):
        out["usable"] = False
        out["verdict"] = "undecided: the s64 to s512 contrast is missing"
        return out
    out["usable"] = True
    out["measured_pct"] = a["pct"]
    out["measured_se_pct"] = a["se_pct"]
    out["measured_ci95_pct"] = a["ci95_pct"]

    hypotheses = {
        "one_time_round1": ONE_TIME_PCT_512,
        "per_round_position_invariant": position_prediction(1.0)["pct"],
        "per_round_kv_dominated": position_prediction(0.0)["pct"],
        "window_invariant_carry_of_anchor": ANCHOR_PCT_64,
    }
    scored = {}
    for name, predicted in hypotheses.items():
        z = ((a["pct"] - predicted) / a["se_pct"]) if a["se_pct"] else float("nan")
        scored[name] = {
            "predicted_pct": predicted,
            "measured_minus_predicted_pp": a["pct"] - predicted,
            "z": z,
            "inside_ci95": a["ci95_pct"][0] <= predicted <= a["ci95_pct"][1],
        }
    out["hypotheses"] = scored
    out["hypotheses_not_excluded"] = [
        name for name, row in scored.items() if row["inside_ci95"]
    ]
    out["implied_nonkv_touched_fraction"] = implied_touched_fraction(a["pct"])

    # Channel B. The difference of two fits on the SAME legs is positively
    # correlated, so this standard error is an upper bound. That makes a
    # "consistent" reading weak evidence and an "inconsistent" reading strong.
    b = contrast(steady, "s64", "s512") if steady else {"usable": False}
    out["steady_state_contrast"] = b
    out["steady_state_note"] = (
        "channel B is decode rounds only and channel A includes the 512-token "
        "seed prefill, so only their CONTRASTS may be compared, never their "
        "levels (F18)"
    )
    b_versus_a = None
    if b.get("usable"):
        se_diff = (a["se_pct"] ** 2 + b["se_pct"] ** 2) ** 0.5
        t_crit = T_CRIT_95.get(primary["df"], 1.96)
        b_versus_a = {
            "b_minus_a_pp": b["pct"] - a["pct"],
            "conservative_se_pp": se_diff,
            "conservative_se_is_an_upper_bound": True,
            "b_consistent_with_a": abs(b["pct"] - a["pct"]) <= t_crit * se_diff,
            "b_consistent_with_zero": abs(b["pct"]) <= t_crit * b["se_pct"],
        }
    out["b_versus_a"] = b_versus_a

    # Channel C is read in milliseconds per leg, never as a percentage: a
    # one-time cost divided by a token count is not a rate.
    for label, model in (("round1_contrast", round1),
                         ("round1_contrast_regression_control", regression)):
        c = contrast(model, "s64", "s512") if model else {"usable": False}
        if c.get("usable"):
            c = dict(c)
            c["delta_ms_per_leg"] = c["delta"] / 1000.0
            c["se_ms_per_leg"] = c["se_abs"] / 1000.0
            c["one_time_reference_ms_per_leg"] = ONE_TIME_MS_PER_LEG
        out[label] = c

    if not b_versus_a:
        out["verdict"] = "undecided: the steady-state channel did not fit"
        out["mechanism"] = None
        return out

    per_round = (b_versus_a["b_consistent_with_a"]
                 and not b_versus_a["b_consistent_with_zero"])
    one_time = (b_versus_a["b_consistent_with_zero"]
                and not b_versus_a["b_consistent_with_a"])
    if per_round:
        nearest = min(("per_round_position_invariant", "per_round_kv_dominated"),
                      key=lambda n: abs(scored[n]["z"]))
        out["mechanism"] = nearest
        out["window_invariant_percentage"] = True
        out["verdict"] = (
            f"per-round tax; the steady state carries it and the 512-token "
            f"contrast is nearest {nearest}"
        )
    elif one_time:
        out["mechanism"] = "one_time_round1"
        out["window_invariant_percentage"] = False
        out["verdict"] = (
            "one-time round-1 tax; the percentage is not window invariant, so "
            "restate every rung in milliseconds per leg"
        )
    else:
        out["mechanism"] = None
        out["window_invariant_percentage"] = None
        out["verdict"] = (
            "mixture or underpowered: the steady state is consistent with both "
            "the full effect and with zero"
        )
    out["ship_pct_at_512"] = a["pct"]
    out["ship_note"] = (
        "the shipping number is the direct 512-token measurement; the "
        "mechanism only decides whether it may be carried to another window"
    )
    return out


def trace_tax(primary: dict | None, untraced: dict) -> dict:
    """Price the round trace that all twelve fitted legs carry.

    Leg 13 repeats s512 with the trace off. The fitted s512 mean sits at the
    design centre, so the session trend is carried forward to leg 13 before the
    two are compared. That is one step beyond the fitted range, and together
    with the single untraced leg it is the main weakness of this number.
    """
    value = untraced.get("mtp_seconds_per_token")
    out = {
        "tag": untraced.get("tag"),
        "untraced_seconds_per_token": value,
        "leg_trace": untraced.get("leg_trace"),
        "trace_anchor_lines": untraced.get("trace_anchor_lines"),
        "trace_round_lines": untraced.get("trace_round_lines"),
        "trace_row_lines": untraced.get("trace_row_lines"),
        "round_trace_error": untraced.get("round_trace_error"),
        "self_check_note": (
            "an empty round trace on this leg is the expected result and "
            "proves the trace really was off"
        ),
    }
    trace_off = (untraced.get("leg_trace") == "0"
                 and not untraced.get("trace_anchor_lines"))
    out["trace_confirmed_off"] = trace_off
    if not primary or value is None or "s512" not in primary["_present"]:
        out["usable"] = False
        return out

    present = primary["_present"]
    k = len(primary["columns"])
    vector = [0.0] * k
    vector[present.index("s512")] = 1.0
    vector[-1] = UNTRACED_INDEX - primary["centre"]
    inv = primary["_inv"]
    var_pred = primary["_sigma2"] * sum(
        vector[i] * inv[i][j] * vector[j] for i in range(k) for j in range(k)
    )
    predicted = sum(b * v for b, v in zip(
        [primary["arm_means_adjusted"][arm] for arm in present]
        + [primary["slope_per_leg"]], vector))
    se = (var_pred + primary["_sigma2"]) ** 0.5
    delta = predicted - value
    t_crit = T_CRIT_95.get(primary["df"], 1.96)
    half = t_crit * 100.0 * se / value
    out.update({
        "usable": True,
        "traced_s512_predicted_at_leg_13": predicted,
        "trace_cost_seconds_per_token": delta,
        "trace_cost_pct": 100.0 * delta / value,
        "se_pct": 100.0 * se / value,
        "t": (delta / se) if se else float("nan"),
        "ci95_pct": [100.0 * delta / value - half, 100.0 * delta / value + half],
        "significant": abs(100.0 * delta / value) > half,
        "positive_means_the_trace_costs_time": True,
    })
    return out


def selftest() -> int:
    """Recover known coefficients from a synthetic ladder.

    The solver is hand written and the last four legs break the palindrome's
    balance, so an unbalanced design is exactly where a normal-equations bug
    would hide. This plants a known arm effect and a known session trend in the
    real leg order and checks that both come back, then plants pure trend with
    no arm effect and checks that every contrast returns zero.
    """
    truth = {"s64": 1.0000, "s512": 0.9980, "s1024": 0.9965, "s2048": 0.9955}
    slope = -0.0004
    centre = 6.5
    failures = []

    def build(effects: dict[str, float], drift: float) -> list[dict]:
        return [
            {"tag": f"t{i + 1:02d}", "index": i + 1, "arm": arm,
             "slack_mb": SLACK_MB[arm],
             "wired_clamped_count": 0, "wired_apply_failures": 0,
             "mtp_seconds_per_token": effects[arm] + drift * (i + 1 - centre)}
            for i, arm in enumerate(ORDER)
        ]

    model = fit(build(truth, slope), "mtp_seconds_per_token")
    if model["df"] != 7:
        failures.append(f"expected 7 residual df, got {model['df']}")
    if abs(model["slope_per_leg"] - slope) > 1e-9:
        failures.append(f"slope {model['slope_per_leg']} != {slope}")
    for arm, value in truth.items():
        got = model["arm_means_adjusted"][arm]
        if abs(got - value) > 1e-9:
            failures.append(f"arm {arm}: {got} != {value}")
    got = contrast(model, "s512", "s2048")["pct"]
    want = 100.0 * (truth["s2048"] - truth["s512"]) / truth["s512"]
    if abs(got - want) > 1e-9:
        failures.append(f"s512_to_s2048 contrast {got} != {want}")

    # Pure trend, no arm effect. Every contrast must be exactly zero, which is
    # the case a design that confounds trend with arm identity would fail.
    flat = {arm: 1.0 for arm in ARMS}
    trend_only = fit(build(flat, slope), "mtp_seconds_per_token")
    for lo, hi in (("s64", "s512"), ("s512", "s1024"), ("s1024", "s2048")):
        pct = contrast(trend_only, lo, hi)["pct"]
        if abs(pct) > 1e-9:
            failures.append(f"trend leaked into {lo}->{hi}: {pct}")

    # A positive control that proves the check above can fail: drop the trend
    # column from the data but keep it in the model, and the arm means must
    # still be flat; then plant an arm effect and require a non-zero contrast.
    planted = dict(flat)
    planted["s2048"] = 0.99
    control = fit(build(planted, slope), "mtp_seconds_per_token")
    if abs(contrast(control, "s512", "s2048")["pct"]) < 0.5:
        failures.append("positive control did not fire: a planted 1 % arm "
                        "effect was not recovered")

    # F19 touched-byte family. These five numbers were published in PR 130
    # interim 21 before the ladder ran, so a typo in any admitted-byte constant
    # would move the goalposts after the fact.
    published = {1.0: -0.2010, 0.75: -0.2023, 0.5: -0.2048,
                 0.25: -0.2113, 0.0: -0.2778}
    for fraction, want in published.items():
        got = position_prediction(fraction)["pct"]
        if abs(got - want) > 5e-5:
            failures.append(f"position family f={fraction}: {got:.6f} != {want}")
        back = implied_touched_fraction(got)
        if back is None or abs(back - fraction) > 1e-6:
            failures.append(f"implied fraction did not round trip at f={fraction}")

    # Trace tax. Plant a leg 13 that is exactly 1 % faster than a flat, driftless
    # ladder and require the reader to price the trace at +1 %.
    flat_fit = fit(build(flat, 0.0), "mtp_seconds_per_token")
    planted_tax = trace_tax(flat_fit, {
        "tag": "t13", "mtp_seconds_per_token": 1.0 / 1.01,
        "leg_trace": "0", "trace_anchor_lines": 0,
    })
    if not planted_tax.get("trace_confirmed_off"):
        failures.append("trace_confirmed_off did not fire on an untraced leg")
    if abs(planted_tax.get("trace_cost_pct", 0.0) - 1.0) > 1e-9:
        failures.append(f"trace tax {planted_tax.get('trace_cost_pct')} != 1.0")
    still_traced = trace_tax(flat_fit, {
        "tag": "t13", "mtp_seconds_per_token": 1.0,
        "leg_trace": "1", "trace_anchor_lines": 41,
    })
    if still_traced.get("trace_confirmed_off"):
        failures.append("trace_confirmed_off fired on a leg that was traced")

    for line in failures:
        print(f"  FAIL {line}")
    print(f"selftest: {'PASS' if not failures else 'FAIL'}"
          f"  ({len(failures)} problems)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prefix", default="e130-r11")
    ap.add_argument("--root", type=Path, default=Path("research/out"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    legs = [read_leg(args.root, args.prefix, i + 1, arm)
            for i, arm in enumerate(ORDER)]
    # Leg 13 is read but never fitted, so it cannot move any arm mean.
    untraced = read_leg(args.root, args.prefix, UNTRACED_INDEX, UNTRACED_ARM)
    all_legs = legs + [untraced]

    channels = {
        "candidate_mtp_seconds_per_token": "mtp_seconds_per_token",
        "serial_seconds_per_token": "serial_seconds_per_token",
        "local_ratio_mtp_decode_speedup": "mtp_decode_speedup",
        # F18 channel B. Same design, same 7 df, steady-state decode only.
        "steady_seconds_per_token": "steady_seconds_per_token",
        # F18 channel C, depth controlled. Fitted in microseconds, and its
        # contrasts are read in MILLISECONDS PER LEG, never as a percentage:
        # a one-time cost divided by a token count is not a rate.
        #
        # Campaign rule 106 requires the tail to be matched on round 1's own
        # draft count. c_depth_matched_us does that directly against the steady
        # rounds that actually ran at that depth, so it is the headline form.
        # c_regression_us extrapolates a linear depth model to round 1's depth
        # and is kept as a reported control, not as the decision statistic.
        "round1_excess_depth_matched_us": "c_depth_matched_us",
        "round1_excess_regression_us": "c_regression_us",
    }
    models = {name: fit(legs, field) for name, field in channels.items()}
    primary = models["candidate_mtp_seconds_per_token"]

    pairs = [("s64", "s512"), ("s512", "s1024"), ("s1024", "s2048"),
             ("s512", "s2048"), ("s64", "s2048"), ("s64", "s1024")]

    report = {
        "experiment": "e130-rung11-slack-ladder",
        "harness": "local",
        "model": "arm means + one linear session trend",
        "headline_channel": "candidate_mtp_seconds_per_token",
        "headline_note": (
            "absolute candidate seconds per token; the local ratio is a "
            "control only (finding 160)"
        ),
        "order": ORDER,
        "permutation_seed_material": "e130-rung11-slack-ladder-F16",
        "permutation_seed": 14383609076371482244,
        "legs": legs,
        "untraced_leg": untraced,
        "thermal": thermal(all_legs),
        "one_binary_served_every_arm":
            len({leg["worker_sha256"] for leg in all_legs
                 if leg.get("worker_sha256")}) == 1,
        "exactness_all_legs":
            all(leg.get("all_tokens_matched") is True for leg in all_legs
                if "all_tokens_matched" in leg),
        "channels": {},
    }
    report["safety"] = safety(all_legs, primary)

    for name, model in models.items():
        if model is None:
            report["channels"][name] = {"usable": False}
            continue
        block = {k: v for k, v in model.items() if not k.startswith("_")}
        block["contrasts"] = {f"{lo}_to_{hi}": contrast(model, lo, hi)
                              for lo, hi in pairs}
        report["channels"][name] = block

    report["position_dependence"] = position_dependence(models)
    report["trace_tax"] = trace_tax(primary, untraced)

    if primary:
        table = []
        for (lo, hi), predictions in PREREGISTERED.items():
            got = report["channels"]["candidate_mtp_seconds_per_token"][
                "contrasts"].get(f"{lo}_to_{hi}", {})
            ci = got.get("ci95_pct")
            row = {
                "contrast": f"{lo}_to_{hi}",
                "measured_pct": got.get("pct"),
                "ci95_pct": ci,
                "predicts_pct": dict(predictions),
                "inside_ci": {
                    model: (ci is not None and ci[0] <= value <= ci[1])
                    for model, value in predictions.items()
                },
            }
            survivors = [m for m, ok in row["inside_ci"].items() if ok]
            row["models_not_excluded"] = survivors
            table.append(row)
        report["preregistered_test"] = table
        report["preregistered_note"] = (
            "constant_slope is the single measured rung 10a slope "
            "extrapolated, so it tests reach and not slope (F17 section 1). "
            "saturating is reported as a reference point only; rung 10 already "
            "rejects it because unwired remaining at s512 exceeds the ladder "
            "ceiling (F17 section 2)."
        )

        means = primary["arm_means_adjusted"]
        argmax = min(means, key=means.get)
        headline = report["channels"]["candidate_mtp_seconds_per_token"][
            "contrasts"].get("s512_to_s2048", {})
        report["ladder_argmax_arm"] = argmax
        report["ladder_argmax_slack_mb"] = SLACK_MB[argmax]
        report["s512_to_s2048_pct"] = headline.get("pct")
        report["decision"] = decision(headline.get("pct"))

    print(f"=== e130 rung 11 slack ladder: {args.prefix} ===")
    print(f"  legs with a score  "
          f"{sum(1 for leg in legs if leg.get('mtp_seconds_per_token'))}/12")
    print(f"  one binary         {report['one_binary_served_every_arm']}")
    print(f"  exact all legs     {report['exactness_all_legs']}")
    t = report["thermal"]
    print(f"  entry temp spread  {t['entry_temp_spread_c']} C "
          f"(cool_gate_passed_real_gate=false, gate_qualified_for_timing=false)")
    s = report["safety"]
    print(f"  SAFETY all_clear   {s['all_clear']}")
    for key in ("clamped_legs", "apply_failure_legs", "wiring_inactive_legs",
                "swapped_legs"):
        if s[key]:
            print(f"    {key}: {s[key]}")
    if s["s2048_legs_slower_than_s1024_by_2sigma"]:
        print(f"    s2048 regressions: "
              f"{s['s2048_legs_slower_than_s1024_by_2sigma']}")

    for name, block in report["channels"].items():
        if not block.get("df"):
            print(f"\n=== {name}: not usable ===")
            continue
        print(f"\n=== {name} ===")
        print(f"  df {block['df']}  resid sd {block['residual_sd']:.6e}"
              f" = {block['residual_sd_pct']:.4f} %"
              f"  slope {block['slope_pct_per_leg']:+.4f} %/leg"
              f"  (trend removes "
              f"{100 * block['trend_share_of_arm_only_residual']:.2f} %)")
        for arm in ARMS:
            if arm in block["arm_means_adjusted"]:
                print(f"  mean {arm:>6}  {block['arm_means_adjusted'][arm]:.8f}")
        for key, c in block["contrasts"].items():
            if not c.get("usable"):
                continue
            flag = "SIG" if c["significant"] else "ns"
            if c["pct"] is None:
                print(f"  {key:<16} {c['delta']:+.4g} abs  t={c['t']:+.2f}"
                      f"  95% CI [{c['ci95_abs'][0]:+.4g},"
                      f" {c['ci95_abs'][1]:+.4g}]  {flag}"
                      f"   (percent undefined: base not positive)")
            else:
                print(f"  {key:<16} {c['pct']:+.4f} %  t={c['t']:+.2f}"
                      f"  95% CI [{c['ci95_pct'][0]:+.4f},"
                      f" {c['ci95_pct'][1]:+.4f}]  {flag}")

    pos = report["position_dependence"]
    print("\n=== F19 section 3: position dependence of s64 -> s512 ===")
    if not pos.get("usable"):
        print(f"  {pos.get('verdict')}")
    else:
        print(f"  measured at 512 tokens  {pos['measured_pct']:+.4f} %"
              f"  se {pos['measured_se_pct']:.4f}"
              f"  95% CI [{pos['measured_ci95_pct'][0]:+.4f},"
              f" {pos['measured_ci95_pct'][1]:+.4f}]")
        print(f"  64-token anchor         {pos['anchor_pct_at_64_tokens']:+.4f} %"
              f"   pre-registered 512-token call "
              f"{pos['preregistered_pct_at_512']:+.4f} %")
        print(f"  {'hypothesis':<34} {'predicts':>9} {'z':>7}  in CI")
        for name, row in pos["hypotheses"].items():
            print(f"  {name:<34} {row['predicted_pct']:+9.4f}"
                  f" {row['z']:+7.2f}  {'yes' if row['inside_ci95'] else 'no'}")
        frac = pos.get("implied_nonkv_touched_fraction")
        print(f"  implied non-KV touched fraction  "
              f"{'off the family' if frac is None else f'{frac:+.4f}'}")
        b = pos.get("steady_state_contrast") or {}
        if b.get("usable"):
            bv = pos["b_versus_a"]
            print(f"  channel B (steady state)  {b['pct']:+.4f} %"
                  f"  se {b['se_pct']:.4f}"
                  f"   B-A {bv['b_minus_a_pp']:+.4f} pp"
                  f" (conservative se {bv['conservative_se_pp']:.4f})")
            print(f"    B consistent with A  {bv['b_consistent_with_a']}"
                  f"   B consistent with zero  {bv['b_consistent_with_zero']}")
        else:
            print("  channel B (steady state)  not usable")
        for label, title in (("round1_contrast", "C depth-matched (rule 106)"),
                             ("round1_contrast_regression_control",
                              "C regression control")):
            c = pos.get(label) or {}
            if c.get("usable"):
                print(f"  {title:<28} {c['delta_ms_per_leg']:+.3f}"
                      f" +/- {c['se_ms_per_leg']:.3f} ms/leg"
                      f"  t={c['t']:+.2f}"
                      f"   one-time reference "
                      f"{c['one_time_reference_ms_per_leg']:.3f} ms/leg")
            else:
                print(f"  {title:<28} not usable")
        print(f"  MECHANISM {pos.get('mechanism')}")
        print(f"  VERDICT   {pos['verdict']}")
        ship = pos.get("ship_pct_at_512")
        print("  ship n/a" if ship is None
              else f"  ship at 512 tokens  {ship:+.4f} %")

    tt = report["trace_tax"]
    print("\n=== trace tax: leg 13, untraced s512 (F19 section 3) ===")
    print(f"  leg_trace={tt.get('leg_trace')}"
          f"  anchor_lines={tt.get('trace_anchor_lines')}"
          f"  trace_confirmed_off={tt.get('trace_confirmed_off')}")
    if tt.get("round_trace_error"):
        print(f"  round trace error (expected): {tt['round_trace_error']}")
    if tt.get("usable"):
        print(f"  untraced  {tt['untraced_seconds_per_token']:.8f} s/token")
        print(f"  traced s512 carried to leg 13  "
              f"{tt['traced_s512_predicted_at_leg_13']:.8f} s/token")
        print(f"  trace costs {tt['trace_cost_pct']:+.4f} %"
              f"  se {tt['se_pct']:.4f}  t={tt['t']:+.2f}"
              f"  95% CI [{tt['ci95_pct'][0]:+.4f}, {tt['ci95_pct'][1]:+.4f}]"
              f"  {'SIG' if tt['significant'] else 'ns'}")
        print("  NOTE never put this leg's absolute seconds per token in the "
              "same table as the archive receipt")
    else:
        print("  not usable")

    if "preregistered_test" in report:
        print("\n=== pre-registered test (F16 section 3, F17 sections 1-2) ===")
        print(f"  {'contrast':<16} {'slope':>7} {'satur':>7} {'declin':>7}"
              f" {'measured':>9}  models not excluded")
        for row in report["preregistered_test"]:
            measured = row["measured_pct"]
            got = f"{measured:+.4f}" if measured is not None else "    n/a"
            p = row["predicts_pct"]
            print(f"  {row['contrast']:<16} {p['constant_slope']:+7.3f}"
                  f" {p['saturating']:+7.3f} {p['declining_value']:+7.3f}"
                  f" {got:>9}  {','.join(row['models_not_excluded']) or 'NONE'}")
        print(f"\n  argmax   {report['ladder_argmax_arm']} "
              f"({report['ladder_argmax_slack_mb']} MiB)")
        print(f"  DECISION {report['decision']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nwrote {args.out}")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="qwen38-mlx-challenge-senpai",
            id="e130r11",
            name="e130-rung11-slack-ladder",
            resume="allow",
            config={
                "experiment": "e130-rung11-slack-ladder",
                "harness": "local",
                "arms": list(ARMS),
                "order": ORDER,
                "permutation_seed": 14383609076371482244,
                "cool_gate_passed_real_gate": False,
                "gate_qualified_for_timing": False,
            },
        )
        flat = {
            "e130_rung11_argmax_slack_mb": report.get("ladder_argmax_slack_mb"),
            "e130_rung11_s512_to_s2048_pct": report.get("s512_to_s2048_pct"),
            "e130_rung11_safety_all_clear": report["safety"]["all_clear"],
            "e130_rung11_entry_temp_spread_c":
                report["thermal"]["entry_temp_spread_c"],
            "e130_rung11_one_binary": report["one_binary_served_every_arm"],
            "e130_rung11_exact_all_legs": report["exactness_all_legs"],
        }
        pos_block = report["position_dependence"]
        if pos_block.get("usable"):
            flat.update({
                "e130_rung11_pos_measured_pct": pos_block["measured_pct"],
                "e130_rung11_pos_se_pct": pos_block["measured_se_pct"],
                "e130_rung11_pos_anchor_pct_at_64": ANCHOR_PCT_64,
                "e130_rung11_pos_preregistered_pct": PREDICTED_PCT_512,
                "e130_rung11_pos_implied_nonkv_touched_fraction":
                    pos_block.get("implied_nonkv_touched_fraction"),
                "e130_rung11_pos_mechanism": pos_block.get("mechanism"),
                "e130_rung11_pos_window_invariant":
                    pos_block.get("window_invariant_percentage"),
                "e130_rung11_pos_ship_pct_at_512":
                    pos_block.get("ship_pct_at_512"),
            })
            for name, row in pos_block["hypotheses"].items():
                flat[f"e130_rung11_pos_z_{name}"] = row["z"]
        tax_block = report["trace_tax"]
        flat["e130_rung11_trace_confirmed_off"] = tax_block.get(
            "trace_confirmed_off")
        if tax_block.get("usable"):
            flat["e130_rung11_trace_cost_pct"] = tax_block["trace_cost_pct"]
            flat["e130_rung11_trace_cost_se_pct"] = tax_block["se_pct"]
        if primary:
            block = report["channels"]["candidate_mtp_seconds_per_token"]
            flat["e130_rung11_residual_sd_pct"] = block["residual_sd_pct"]
            flat["e130_rung11_df"] = block["df"]
            for arm, mean in block["arm_means_adjusted"].items():
                flat[f"e130_rung11_mean_{arm}_seconds_per_token"] = mean
            for key, c in block["contrasts"].items():
                if c.get("usable"):
                    flat[f"e130_rung11_{key}_pct"] = c["pct"]
                    flat[f"e130_rung11_{key}_t"] = c["t"]
        run.log(flat)
        run.summary.update(flat)
        if args.out:
            artifact = wandb.Artifact("e130-rung11-slack-ladder", type="analysis")
            artifact.add_file(str(args.out))
            run.log_artifact(artifact)
        run.finish()
        print("logged to W&B run e130r11")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
