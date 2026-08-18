#!/usr/bin/env python3
"""Research-only (qwen38-r1-e25-per-row-draft-price).

Phase 0 of E25: price each drafted row against the round it actually extends,
replay the shipped depth walk under three prices, and cost the counterfactual
from the E21 tape with zero GPU.

CREDIT. The two-piece boundary-aware marginal price and arm C are thorfinn's
E22 follow-up #1 (PR #26). E22 supplied the per-width verify cost curve, E23
(alphonse, PR #27) the dispatch inventory that shows the M=4->5 penalty is not
a launch-count artifact, and E21 (PR #25) the 1947-round tape replayed here.

WHY THIS IS EXACT OFFLINE. `costModelDepth` (Qwen36MTPBlockSession.swift:611)
walks depth with a per-depth accept probability `p` that depends only on the
pre-walk `positionAcceptEMA` snapshot and the pending top-2 margin -- never on
the price. The trace records `p`, `reach` and `threshold` for every step it
took, so any price can be re-walked over the recorded `p` sequence and checked
against the recorded thresholds. Verification is prefix/first-failure, so
TRUNCATING a chain leaves the surviving rows byte-identical and acceptance
becomes min(acc, d'). DEEPENING is not evaluable offline -- the tape holds no
verdict for a row that was never proposed -- so any replay that requests more
depth than was recorded is counted and reported, never credited.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e21_trace import parse_trace  # noqa: E402

# Shipped constants (Qwen36MTPBlockSession.swift:541,:573,:588,:594).
SHIPPED_H = 0.18
SDPA_WIDTH_WALL_DEPTH_CAP = 5
SEGMENTED_VERIFY_DEPTH_CAP = 8

RUNS_ROOT = Path(".mlxfast-private/e21/runs")
PROMPTS = ("english", "narrative", "technical", "dramatic", "travel",
           "philosophy", "natural_history", "medicine")


# --------------------------------------------------------------------------
# tape
# --------------------------------------------------------------------------
def load_tape(runs_root: Path = RUNS_ROOT,
              arm: str = "S18I") -> tuple[list[dict], dict]:
    """Pooled rounds of the timed MTP session, plus each prompt's leg reports."""
    rounds: list[dict] = []
    legs: dict[str, dict] = {}
    for prompt in PROMPTS:
        run = runs_root / f"probe-{prompt}-{arm}"
        sessions = parse_trace(run / "trace.txt")
        if len(sessions) != 1:
            raise SystemExit(f"{run}: expected one traced session, got {len(sessions)}")
        for r in sessions[0]:
            r["prompt"] = prompt
            rounds.append(r)
        serial = json.loads((run / "reports/03-mtp-timed.json").read_text())
        mtp = json.loads((run / "reports/04-mtp-timed.json").read_text())
        if not serial.get("is_serial_control") or mtp.get("is_serial_control"):
            raise SystemExit(f"{run}: leg reports are not (serial, mtp)")
        legs[prompt] = {"serial": serial, "mtp": mtp}
    return rounds, legs


def ledger(rounds: list[dict], legs: dict) -> dict:
    """Reconcile the tape against the legs, and decompose prefill (PR #29 §3).

    `decode_seconds` is prefill-INCLUSIVE (calibration fact (c)); the same
    report also carries the measured `seed_prefill_seconds`, so prefill does
    not have to be inferred from the round-timer residual.
    """
    traced_us = sum(r["round_us"] for r in rounds)
    draft_rows = sum(r["d"] for r in rounds)
    accepted = sum(r["acc"] for r in rounds)
    mtp_leg = sum(legs[p]["mtp"]["decode_seconds"] for p in PROMPTS)
    mtp_prefill = sum(legs[p]["mtp"]["seed_prefill_seconds"] for p in PROMPTS)
    ser_leg = sum(legs[p]["serial"]["decode_seconds"] for p in PROMPTS)
    ser_prefill = sum(legs[p]["serial"]["seed_prefill_seconds"] for p in PROMPTS)
    ser_rounds = sum(legs[p]["serial"]["round_count"] for p in PROMPTS)
    mtp_rounds = sum(legs[p]["mtp"]["round_count"] for p in PROMPTS)

    mtp_true = mtp_leg - mtp_prefill
    ser_true = ser_leg - ser_prefill
    a_leg_us = 1e6 * ser_leg / ser_rounds            # prefill-INCLUSIVE anchor
    a_true_us = 1e6 * ser_true / ser_rounds          # prefill-free anchor
    traced_s = traced_us / 1e6

    def price(mtp_decode_s: float, anchor_us: float) -> float:
        return ((1e6 * mtp_decode_s - mtp_rounds * anchor_us) / draft_rows) / anchor_us

    return {
        "rounds": len(rounds),
        "draft_rows_proposed": draft_rows,
        "accepted_draft_rows": accepted,
        "emitted_tokens": len(rounds) + accepted,
        "traced_round_us_total_s": traced_s,
        "mtp_leg_total_s": mtp_leg,
        "mtp_measured_prefill_s": mtp_prefill,
        "mtp_true_decode_s": mtp_true,
        "serial_leg_total_s": ser_leg,
        "serial_measured_prefill_s": ser_prefill,
        "serial_true_decode_s": ser_true,
        "serial_rounds": ser_rounds,
        "mtp_rounds_in_legs": mtp_rounds,
        "implied_prefill_from_timer_residual_s": mtp_leg - traced_s,
        "timer_coverage_of_leg": traced_s / mtp_leg,
        "timer_coverage_of_true_decode": traced_s / mtp_true,
        "anchor_a_leg_ms": a_leg_us / 1000.0,
        "anchor_a_true_ms": a_true_us / 1000.0,
        "h_leg_anchor_leg_numerator": price(mtp_leg, a_leg_us),
        "h_leg_anchor_traced_numerator": price(traced_s, a_leg_us),
        "h_leg_anchor_true_numerator": price(mtp_true, a_leg_us),
        "h_prefill_free_both_sides": price(mtp_true, a_true_us),
        "h_prefill_free_traced_numerator": price(traced_s, a_true_us),
    }


def annotate_positions(rounds: list[dict]) -> None:
    """Attach each round's emitted-token offset within its own prompt.

    T(d) is a mean over rounds the POLICY chose, not a randomised assignment,
    and KV/GDN work grows with the token offset, so depth and position are
    confounded on this tape and the confound has to be measured before the
    measured step ratios can be used as a price.
    """
    pos: Counter = Counter()
    for r in rounds:
        r["pos"] = pos[r["prompt"]]
        pos[r["prompt"]] += 1 + int(r["acc"])


def position_control(rounds: list[dict], strata: int = 4) -> dict:
    """Is the measured T(d) curve depth, or is it position?

    Three views: the position distribution per depth; T(d) inside equal-width
    position strata; and a least-squares fit of round_ms on a depth indicator
    plus a linear token offset, from which position-adjusted step ratios are
    recomputed.
    """
    pos_max = max(r["pos"] for r in rounds) + 1
    edges = [pos_max * i // strata for i in range(strata + 1)]
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for r in rounds:
        by_depth[r["d"]].append(r)

    dist = {
        d: {
            "n": len(v),
            "mean_pos": statistics.mean(x["pos"] for x in v),
            "median_pos": statistics.median(x["pos"] for x in v),
        }
        for d, v in sorted(by_depth.items())
    }
    strat: dict[str, dict[int, dict]] = {}
    for i in range(strata):
        lo, hi = edges[i], edges[i + 1]
        cell: dict[int, dict] = {}
        for d, v in sorted(by_depth.items()):
            ms = [x["round_us"] / 1000.0 for x in v if lo <= x["pos"] < hi]
            if ms:
                cell[d] = {"n": len(ms), "mean_ms": statistics.mean(ms)}
        strat[f"{lo}-{hi}"] = cell

    depths = sorted(by_depth)
    cols = len(depths) + 1
    xtx = [[0.0] * cols for _ in range(cols)]
    xty = [0.0] * cols
    for r in rounds:
        x = [0.0] * cols
        x[depths.index(r["d"])] = 1.0
        x[-1] = r["pos"] / 1000.0
        y = r["round_us"] / 1000.0
        for a in range(cols):
            xty[a] += x[a] * y
            for b in range(cols):
                xtx[a][b] += x[a] * x[b]
    beta = _solve(xtx, xty)
    mean_pos = statistics.mean(r["pos"] for r in rounds) / 1000.0
    adj = {d: beta[i] + beta[-1] * mean_pos for i, d in enumerate(depths)}
    adj_step = {
        d: (adj[d + 1] - adj[d]) / adj[d] for d in depths if d + 1 in adj
    }
    return {
        "position_by_depth": dist,
        "round_ms_by_position_stratum": strat,
        "ols_ms_per_1000_tokens_of_offset": beta[-1],
        "ols_depth_intercepts_ms": {d: beta[i] for i, d in enumerate(depths)},
        "position_adjusted_round_ms": adj,
        "position_adjusted_step_ratio": adj_step,
    }


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise SystemExit("position_control: singular design matrix")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def round_time_table(rounds: list[dict]) -> dict:
    """Measured mean round wall time per proposed depth, and its step ratios."""
    by_depth: dict[int, list[float]] = defaultdict(list)
    for r in rounds:
        by_depth[r["d"]].append(r["round_us"] / 1000.0)
    table = {
        d: {
            "n": len(v),
            "mean_ms": statistics.mean(v),
            "median_ms": statistics.median(v),
            "stdev_ms": statistics.stdev(v) if len(v) > 1 else 0.0,
            "sem_ms": statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0,
        }
        for d, v in sorted(by_depth.items())
    }
    step = {}
    for d in sorted(table):
        if d + 1 in table:
            t0, t1 = table[d]["mean_ms"], table[d + 1]["mean_ms"]
            step[d] = (t1 - t0) / t0
    per_token = {d: v["mean_ms"] / (d + 1) for d, v in table.items()}
    # Truncation changes accepted rows as well as proposed rows, so charging a
    # truncated round mean T(d') is only valid if T is flat in `acc` at fixed d.
    by_acc: dict[str, dict] = {}
    grp: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rounds:
        grp[(r["d"], r["acc"])].append(r["round_us"] / 1000.0)
    for (d, a), v in sorted(grp.items()):
        by_acc[f"d{d}_acc{a}"] = {"n": len(v), "mean_ms": statistics.mean(v)}
    return {"round_ms": table, "step_ratio": step,
            "round_ms_per_verified_row": per_token,
            "round_ms_by_depth_and_accepted": by_acc}


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
def effective_cap(r: dict) -> int:
    """The cap the walk actually ran under.

    One walk entry is written per loop iteration BEFORE the guard, so a walk of
    exactly `d` entries means the loop exited on `depth == cap` and the
    effective cap (which also folds in the parent's shrinking offeredDepth) is
    `d`. A longer walk broke on the price, and `cap=` only bounds from above.
    """
    walk = r.get("walk") or []
    return int(r["d"]) if len(walk) == int(r["d"]) else int(r["cap"])


def replay(r: dict, price, cap_override: int | None = None) -> dict:
    """Re-walk one round's recorded probability sequence under `price`.

    `price(depth, expected)` returns the threshold `reach` must exceed to add
    row `depth + 1`.
    """
    walk = r.get("walk") or []
    ema = r.get("ema") or []
    cap = effective_cap(r)
    if cap_override is not None:
        cap = min(cap, cap_override)
    reach, expected, depth = 1.0, 0.0, 0
    thresholds: list[float] = []
    reaches: list[float] = []
    while depth < cap:
        if depth < len(walk):
            p = float(walk[depth]["p"])
        elif depth < len(ema):
            p = float(ema[depth])
        else:
            break
        reach *= p
        thr = price(depth, expected)
        thresholds.append(thr)
        reaches.append(reach)
        if not reach > thr:
            break
        expected += reach
        depth += 1
    return {"depth": depth, "thresholds": thresholds, "reaches": reaches}


def shipped_price(depth: int, expected: float) -> float:
    return SHIPPED_H * (1.0 + expected) / (1.0 + depth * SHIPPED_H)


def per_row_price(h_vec: list[float]):
    """§2 per-row form: reach > h_{d+1}(1+expected)/(1+H_d), H_d = h_1..h_d."""
    def price(depth: int, expected: float) -> float:
        if depth >= len(h_vec):
            return float("inf")
        return h_vec[depth] * (1.0 + expected) / (1.0 + sum(h_vec[:depth]))
    return price


def anchor_free_price(step: dict[int, float], fallback: float | None = None):
    """§4 anchor-free form: reach > stepRatio[d] * (1 + expected)."""
    def price(depth: int, expected: float) -> float:
        s = step.get(depth, fallback)
        if s is None:
            return float("inf")
        return s * (1.0 + expected)
    return price


def no_deepen_price(step: dict[int, float]):
    """Arm C restricted to the half the tape can evaluate.

    Both forms are proportional to (1 + expected), so taking the dearer
    per-depth coefficient can only truncate a chain, never extend one. That
    keeps every round's surviving rows byte-identical to the tape, so the
    counterfactual is exact on-tape instead of partly unobservable.
    """
    def price(depth: int, expected: float) -> float:
        shipped = SHIPPED_H / (1.0 + depth * SHIPPED_H)
        s = step.get(depth)
        coef = shipped if s is None else max(shipped, s)
        return coef * (1.0 + expected)
    return price


def deepening_breakdown(rounds: list[dict], price) -> dict:
    """Where an arm asks for depth the tape never proposed."""
    pairs: Counter = Counter()
    for r in rounds:
        want = replay(r, price)["depth"]
        if want > int(r["d"]):
            pairs[(int(r["d"]), want)] += 1
    return {
        "rounds_requesting_more_depth": sum(pairs.values()),
        "transitions_taped_to_requested": {f"{a}->{b}": n for (a, b), n in sorted(pairs.items())},
    }


def binding_constraint(rounds: list[dict]) -> dict:
    """Which limit actually chose each round's depth: the price, or a cap?

    If the caps bound most rounds the price is not the live control and this
    whole experiment is mis-aimed, so this is checked before any arm is costed.
    """
    caps: Counter = Counter()
    reason: Counter = Counter()
    for r in rounds:
        caps[int(r["cap"])] += 1
        walk = len(r.get("walk") or [])
        reason["cap_bound" if walk == int(r["d"]) else "price_bound"] += 1
    return {
        "offered_cap_histogram": dict(sorted(caps.items())),
        "exit_reason": dict(reason),
        "price_bound_fraction": reason["price_bound"] / len(rounds),
        "max_taped_depth": max(int(r["d"]) for r in rounds),
    }


def gate(rounds: list[dict]) -> dict:
    """Mandatory instrument gate (§7.2) plus a threshold-level check.

    Depth equality alone would pass a replay that reproduces the outcome for
    the wrong reason, so every traced threshold and reach is also compared with
    the replayed value at 1e-6, the trace's own print precision.
    """
    flat = per_row_price([SHIPPED_H] * 8)
    depth_mismatch, thr_err, reach_err, compared = [], 0.0, 0.0, 0
    scalar_mismatch = []
    for r in rounds:
        s = replay(r, shipped_price)
        v = replay(r, flat)
        if s["depth"] != int(r["d"]):
            scalar_mismatch.append({"prompt": r["prompt"], "round": r["round"],
                                    "traced": int(r["d"]), "replayed": s["depth"]})
        if v["depth"] != int(r["d"]):
            depth_mismatch.append({"prompt": r["prompt"], "round": r["round"],
                                   "traced": int(r["d"]), "replayed": v["depth"]})
        for i, w in enumerate(r.get("walk") or []):
            if i >= len(v["thresholds"]):
                break
            thr_err = max(thr_err, abs(v["thresholds"][i] - float(w["threshold"])))
            reach_err = max(reach_err, abs(v["reaches"][i] - float(w["reach"])))
            compared += 1
    return {
        "rounds": len(rounds),
        "scalar_form_depth_mismatches": len(scalar_mismatch),
        "per_row_0p18x8_depth_mismatches": len(depth_mismatch),
        "per_row_0p18x8_bit_identical": not depth_mismatch,
        "walk_steps_compared": compared,
        "max_abs_threshold_error": thr_err,
        "max_abs_reach_error": reach_err,
        "examples": (scalar_mismatch + depth_mismatch)[:5],
    }


# --------------------------------------------------------------------------
# counterfactual cost
# --------------------------------------------------------------------------
def cost_arm(rounds: list[dict], depths: dict[tuple[str, int], int],
             round_ms: dict[int, dict], acc_ms: dict[str, dict] | None = None,
             min_cell: int = 8) -> dict:
    """Cost a truncation-only counterfactual in measured round-timer units.

    A round whose depth falls from d to d' keeps its own measured wall time and
    pays the MEASURED mean step difference T(d') - T(d); its accepted count
    falls to min(acc, d') because verification is prefix/first-failure. Tokens
    per round therefore fall, so the comparison is per EMITTED TOKEN, which is
    what a fixed 512-token window charges.
    """
    base_us = new_us = 0.0
    base_tok = new_tok = 0
    rows_saved = tokens_lost = truncated = deepened = 0
    hist_base: Counter = Counter()
    hist_new: Counter = Counter()
    loss_rounds = []
    for r in rounds:
        d0, acc, us = int(r["d"]), int(r["acc"]), float(r["round_us"])
        d1 = depths[(r["prompt"], int(r["round"]))]
        if d1 > d0:
            deepened += 1
            d1 = d0
        acc1 = min(acc, d1)
        delta_ms = 0.0
        if d1 != d0:
            delta_ms = round_ms[d1]["mean_ms"] - round_ms[d0]["mean_ms"]
            if acc_ms is not None:
                lo = acc_ms.get(f"d{d1}_acc{acc1}")
                hi = acc_ms.get(f"d{d0}_acc{acc}")
                if lo and hi and lo["n"] >= min_cell and hi["n"] >= min_cell:
                    delta_ms = lo["mean_ms"] - hi["mean_ms"]
        base_us += us
        new_us += us + 1000.0 * delta_ms
        base_tok += 1 + acc
        new_tok += 1 + acc1
        hist_base[d0] += 1
        hist_new[d1] += 1
        if d1 < d0:
            truncated += 1
            rows_saved += d0 - d1
            if acc1 < acc:
                tokens_lost += acc - acc1
                loss_rounds.append({"prompt": r["prompt"], "round": int(r["round"]),
                                    "d": d0, "acc": acc, "capped_to": d1})
    base_upt = base_us / base_tok
    new_upt = new_us / new_tok
    return {
        "rounds": len(rounds),
        "rounds_truncated": truncated,
        "rounds_requesting_more_depth_than_taped": deepened,
        "draft_rows_saved": rows_saved,
        "exact_on_tape_tokens_lost": tokens_lost,
        "token_loss_rounds": loss_rounds,
        "depth_histogram_base": dict(sorted(hist_base.items())),
        "depth_histogram_arm": dict(sorted(hist_new.items())),
        "mean_depth_base": sum(d * n for d, n in hist_base.items()) / len(rounds),
        "mean_depth_arm": sum(d * n for d, n in hist_new.items()) / len(rounds),
        "base_true_decode_s": base_us / 1e6,
        "arm_true_decode_s": new_us / 1e6,
        "gross_time_saved_s": (base_us - new_us) / 1e6,
        "emitted_tokens_base": base_tok,
        "emitted_tokens_arm": new_tok,
        "ms_per_emitted_token_base": base_upt / 1000.0,
        "ms_per_emitted_token_arm": new_upt / 1000.0,
        "true_decode_gain_pct": 100.0 * (base_upt / new_upt - 1.0),
    }


def project_score(per_prompt: dict, legs: dict) -> dict:
    """Per-prompt leg factor and the MEDIAN OF 8 the official score takes.

    A prompt's timed leg is prefill + true decode over a FIXED 512-token
    window, so a decode-only gain moves the leg by the decode share alone.
    `raw_p` is that prompt's own paired serial/MTP ratio from the same run.
    """
    rows = {}
    for p in PROMPTS:
        mtp, ser = legs[p]["mtp"], legs[p]["serial"]
        prefill = mtp["seed_prefill_seconds"]
        decode = mtp["decode_seconds"] - prefill
        gain = per_prompt[p]["true_decode_gain_pct"] / 100.0
        new_decode = decode / (1.0 + gain)
        leg_factor = (prefill + new_decode) / (prefill + decode)
        raw = ser["parent_measured_seconds_per_token"] / mtp["parent_measured_seconds_per_token"]
        rows[p] = {
            "true_decode_gain_pct": per_prompt[p]["true_decode_gain_pct"],
            "measured_prefill_s": prefill,
            "measured_true_decode_s": decode,
            "decode_share_of_leg": decode / (prefill + decode),
            "leg_factor": leg_factor,
            "local_raw_ratio_base": raw,
            "local_raw_ratio_arm": raw / leg_factor,
            "mean_depth_base": per_prompt[p]["mean_depth_base"],
            "mean_depth_arm": per_prompt[p]["mean_depth_arm"],
            "exact_on_tape_tokens_lost": per_prompt[p]["exact_on_tape_tokens_lost"],
        }
    gains = [rows[p]["true_decode_gain_pct"] for p in PROMPTS]
    factors = [rows[p]["leg_factor"] for p in PROMPTS]
    base_raw = [rows[p]["local_raw_ratio_base"] for p in PROMPTS]
    arm_raw = [rows[p]["local_raw_ratio_arm"] for p in PROMPTS]
    return {
        "per_prompt": rows,
        "median_of_8_true_decode_gain_pct": statistics.median(gains),
        "mean_true_decode_gain_pct": statistics.mean(gains),
        "min_true_decode_gain_pct": min(gains),
        "max_true_decode_gain_pct": max(gains),
        "median_of_8_leg_factor": statistics.median(factors),
        "median_of_8_local_raw_base": statistics.median(base_raw),
        "median_of_8_local_raw_arm": statistics.median(arm_raw),
        "median_of_8_local_raw_gain_pct":
            100.0 * (statistics.median(arm_raw) / statistics.median(base_raw) - 1.0),
    }


def run_arm(rounds: list[dict], price, table: dict, cap_override: int | None = None,
            acc_ms: dict | None = None) -> dict:
    depths = {}
    for r in rounds:
        depths[(r["prompt"], int(r["round"]))] = replay(r, price, cap_override)["depth"]
    pooled = cost_arm(rounds, depths, table["round_ms"], acc_ms)
    per_prompt = {}
    for p in PROMPTS:
        sub = [r for r in rounds if r["prompt"] == p]
        per_prompt[p] = cost_arm(sub, depths, table["round_ms"], acc_ms)
    return {"pooled": pooled, "per_prompt": per_prompt}


# --------------------------------------------------------------------------
# §5 adjudication
# --------------------------------------------------------------------------
def unreachability(step: dict[int, float], trials: int = 400000, seed: int = 20260818) -> dict:
    """Can the anchor-free price ever fire at depth 3 (i.e. select depth 4)?

    The policy only reaches depth 3 having accepted rows 1 and 2, so the walk's
    own monotone EMA gives r1 >= r3 and r2 >= r3. Slack is
    reach - stepRatio[3]*(1+expected) with reach = r1*r2*r3 and
    expected = r1 + r2 in the walk's accumulation.
    """
    s = step[3]
    rng = random.Random(seed)
    best = -float("inf")
    fires = 0
    for _ in range(trials):
        r3 = rng.random()
        r1 = r3 + (1.0 - r3) * rng.random()
        r2 = r3 + (1.0 - r3) * rng.random()
        slack = r1 * r2 * r3 - s * (1.0 + r1 + r2)
        best = max(best, slack)
        if slack > 0:
            fires += 1
    corner = 1.0 - s * 3.0
    return {
        "step_ratio_3": s,
        "monte_carlo_trials": trials,
        "monte_carlo_fires": fires,
        "monte_carlo_best_slack": best,
        "corner_r1_r2_r3_all_one_slack": corner,
        "analytic_unreachable": corner <= 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    ap.add_argument("--arm", default="S18I",
                    help="tape label: probe-<prompt>-<arm> under --runs-root")
    args = ap.parse_args()

    rounds, legs = load_tape(Path(args.runs_root), args.arm)
    annotate_positions(rounds)
    led = ledger(rounds, legs)
    table = round_time_table(rounds)
    step = table["step_ratio"]
    pos = position_control(rounds)

    report: dict = {
        "credit": "thorfinn E22 follow-up #1 (two-piece boundary-aware marginal price, arm C)",
        "tape": {"runs_root": args.runs_root, "arm": args.arm,
                 "prompts": list(PROMPTS)},
        "ledger": led,
        "round_time_table": table,
        "position_control": pos,
        "binding_constraint": binding_constraint(rounds),
        "instrument_gate": gate(rounds),
        "unreachability_depth3": unreachability(step),
    }

    arms = {
        "A_shipped_scalar_h0.18": (shipped_price, None),
        "B_cap_depth3": (shipped_price, 3),
        "C_anchor_free_measured_step": (
            anchor_free_price({**step, 0: SHIPPED_H}), None),
        "C_anchor_free_step0_from_anchor": (
            anchor_free_price({**step, 0: step.get(0, SHIPPED_H)}), None),
        "D_no_deepen_max_shipped_measured": (no_deepen_price(step), None),
    }
    report["arms"] = {}
    for name, (price, cap) in arms.items():
        res = run_arm(rounds, price, table, cap)
        res["projection"] = project_score(res["per_prompt"], legs)
        res["deepening"] = deepening_breakdown(rounds, price)
        for p in PROMPTS:
            res["per_prompt"][p].pop("token_loss_rounds", None)
        report["arms"][name] = res

    acc_ms = table["round_ms_by_depth_and_accepted"]
    report["sensitivity_accept_conditioned_cost"] = {
        name: {
            "pooled_gain_pct": s["pooled"]["true_decode_gain_pct"],
            "median_of_8_gain_pct": project_score(
                s["per_prompt"], legs)["median_of_8_true_decode_gain_pct"],
        }
        for name, s in (
            (name, run_arm(rounds, price, table, cap, acc_ms))
            for name, (price, cap) in arms.items() if name != "A_shipped_scalar_h0.18"
        )
    }

    b = report["arms"]["B_cap_depth3"]
    c = report["arms"]["C_anchor_free_measured_step"]
    report["B_vs_C"] = {
        "identical_depth_histogram": b["pooled"]["depth_histogram_arm"] == c["pooled"]["depth_histogram_arm"],
        "B_mean_depth": b["pooled"]["mean_depth_arm"],
        "C_mean_depth": c["pooled"]["mean_depth_arm"],
        "C_selects_depth4_or_more": sum(
            n for d, n in c["pooled"]["depth_histogram_arm"].items() if d >= 4),
        "B_pooled_gain_pct": b["pooled"]["true_decode_gain_pct"],
        "C_pooled_gain_pct": c["pooled"]["true_decode_gain_pct"],
        "B_median_of_8_gain_pct": b["projection"]["median_of_8_true_decode_gain_pct"],
        "C_median_of_8_gain_pct": c["projection"]["median_of_8_true_decode_gain_pct"],
    }

    print(json.dumps(report, indent=2, sort_keys=False))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
