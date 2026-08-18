#!/usr/bin/env python3
"""Research-only (qwen38-r1-e21-depth-preserving-row-declination).

Fit a whole-round declination predicate offline, from the traced probe rounds,
and score every candidate by the pre-registered cost model rather than by
classifier accuracy.

WHY NOT PRECISION/RECALL AS THE OBJECTIVE. A declined round that would have
accepted nothing is pure profit, but a declined round that would have accepted
k tokens costs roughly k extra rounds later. Those two are not symmetric, so a
predicate can be both precise and unprofitable, or imprecise and profitable.
The objective below is the modelled decode cost per token, which is what the
score actually measures.

THE COST MODEL. Writing `a` for one round (one target forward at M=1) and `X`
for the marginal cost of one drafted token (its proposal-head step plus its
extra verify row), a round that drafts `d` and accepts `acc` costs `1 + X*d`
and yields `1 + acc` tokens. `X` is fitted from the E17 aggregates: S18 spent
245 rounds / 580 drafts and CURVE 246 / 497 for a measured +6.821%, so

    245 + 580X = 1.06821 * (246 + 497X)  =>  X = 0.362

Declining a round drops its `X*d` cost and its `acc` yield, leaving the primary
token untouched. This is first-order only: it holds the rest of the trajectory
fixed, while a real declination also perturbs the EMA and the streak. It is
therefore a PREDICTION to be falsified by the timed arm, not a result.

ONLY LEGAL SIGNALS. Every feature here is already computed by the shipped
schedule before the first head step of the round, from state private to the
current request: the pending top-2 margin, the per-depth acceptance EMA, the
full-accept streak, and the reach walk those three imply. Nothing reads the
reference rows, the golden, the prompt pool or any cross-request history.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Callable

# `headStepCostRatio` as shipped in Qwen36MTPBlockSession.costModelDepth.
SHIPPED_H = 0.18

H_GRID = [0.18, 0.22, 0.25, 0.28, 0.30, 0.325, 0.36, 0.40, 0.45, 0.50, 0.60]


def load_rounds(paths: list[str]) -> list[dict]:
    rounds: list[dict] = []
    for path in paths:
        for line in pathlib.Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                rounds.append(json.loads(line))
    rounds = [featurise(r) for r in rounds]
    add_history(rounds)
    return rounds


def add_history(rounds: list[dict]) -> None:
    """Within-request acceptance history the shipped schedule does not use.

    `positionAcceptEMA` mixes every depth of every past round into one number
    per depth, so a short burst of dead rounds barely moves it. If per-round
    acceptance is bursty at all, the run length of immediately preceding
    zero-accept rounds should carry signal the EMA has averaged away.
    """
    by_prompt: dict[str, list[dict]] = {}
    for r in rounds:
        by_prompt.setdefault(r["prompt"], []).append(r)
    for group in by_prompt.values():
        group.sort(key=lambda r: int(r["round"]))
        zero_run = 0
        prev_acc = -1
        for r in group:
            r["prev_zero_run"] = zero_run
            r["prev_acc"] = prev_acc
            zero_run = zero_run + 1 if int(r["acc"]) == 0 else 0
            prev_acc = int(r["acc"])


def featurise(r: dict) -> dict:
    walk = r.get("walk") or []
    d = int(r["d"])
    ema = r.get("ema") or []
    taken = walk[:d]
    r["expected"] = sum(w["reach"] for w in taken)
    r["reach_last"] = taken[-1]["reach"] if taken else 0.0
    r["slack_last"] = (taken[-1]["reach"] - taken[-1]["threshold"]) if taken else 0.0
    r["p_last"] = taken[-1]["p"] if taken else 0.0
    r["p0"] = walk[0]["p"] if walk else 0.0
    r["ema0"] = ema[0] if ema else 0.0
    r["ema1"] = ema[1] if len(ema) > 1 else 0.0
    r["conf0"] = 1.0 / (1.0 + math.exp(-float(r["m"]) / 2.0))
    # The schedule's own yield forecast against its own drafting bill.
    r["margin_over_cost"] = r["expected"] - SHIPPED_H * d
    return r


def cost_per_token(rounds: list[dict], declined: list[bool], x: float) -> tuple[float, float, float]:
    cost = 0.0
    yield_ = 0.0
    for r, dec in zip(rounds, declined):
        d = 0 if dec else int(r["d"])
        acc = 0 if dec else int(r["acc"])
        cost += 1.0 + x * d
        yield_ += 1.0 + acc
    return cost / yield_, cost, yield_


def evaluate(rounds: list[dict], predicate: Callable[[dict], bool], x: float) -> dict:
    declined = [predicate(r) for r in rounds]
    base_cpt, base_cost, base_yield = cost_per_token(rounds, [False] * len(rounds), x)
    cpt, cost, yield_ = cost_per_token(rounds, declined, x)
    fired = [r for r, dec in zip(rounds, declined) if dec]
    tp = [r for r in fired if int(r["acc"]) == 0]
    fp = [r for r in fired if int(r["acc"]) > 0]
    zero_rounds = [r for r in rounds if int(r["acc"]) == 0]
    kept_depths: dict[int, int] = {}
    for r, dec in zip(rounds, declined):
        if not dec:
            kept_depths[int(r["d"])] = kept_depths.get(int(r["d"]), 0) + 1
    return {
        "fired": len(fired),
        "true_positive": len(tp),
        "false_positive": len(fp),
        "precision_pct": 100.0 * len(tp) / len(fired) if fired else float("nan"),
        "recall_pct": 100.0 * len(tp) / len(zero_rounds) if zero_rounds else float("nan"),
        "drafts_saved": sum(int(r["d"]) for r in fired),
        "tokens_lost": sum(int(r["acc"]) for r in fired),
        "proposed_rows_after": sum(int(r["d"]) for r in rounds) - sum(int(r["d"]) for r in fired),
        "depth_histogram_after": dict(sorted(kept_depths.items())),
        "max_depth_after": max(kept_depths) if kept_depths else 0,
        "cost_per_token_base": base_cpt,
        "cost_per_token_after": cpt,
        "gain_pct": 100.0 * (base_cpt / cpt - 1.0),
    }


def leg_anchor(runs_root: str, prompt: str) -> dict | None:
    """Price a round from the two legs of the same run.

    The serial control leg is not a separate generator: its report carries
    `non_drafting_round_count == round_count`, i.e. it is this very session
    driven at depth 0. That makes its seconds-per-token a *measured* d=0 round
    cost rather than an extrapolated regression intercept -- the anchor a
    within-arm fit on d in 1..4 cannot supply.
    """
    base = pathlib.Path(runs_root) / prompt / "reports"
    try:
        serial = json.loads((base / "03-mtp-timed.json").read_text())
        mtp = json.loads((base / "04-mtp-timed.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if serial.get("non_drafting_round_count") != serial.get("round_count"):
        return None
    d0_us = 1e6 * serial["decode_seconds"] / serial["round_count"]
    rounds = mtp["round_count"]
    drafts = int(round(mtp["effective_mean_draft_len"] * rounds))
    mtp_us = 1e6 * mtp["decode_seconds"]
    draft_us = (mtp_us - rounds * d0_us) / drafts if drafts else float("nan")
    return {
        "serial_rounds": serial["round_count"],
        "serial_decode_seconds": serial["decode_seconds"],
        "d0_round_us_measured": d0_us,
        "mtp_rounds": rounds,
        "mtp_proposed_drafts": drafts,
        "mtp_decode_seconds": mtp["decode_seconds"],
        "draft_us_marginal": draft_us,
        "h_measured": draft_us / d0_us,
    }


def evaluate_nonparametric(rounds: list[dict], anchor: dict, declined: list[bool]) -> dict:
    """Score a declination using each round's own observed wall time.

    `round_us` already contains this round's rejection, rollback and replay
    work, so no separate replay term can be mispriced. The traced round timer
    covers only part of decode wall time, so the measured d=0 leg cost is
    rescaled into round_us units by the same-run coverage ratio.
    """
    kept_us = [float(r.get("round_us") or 0.0) for r in rounds]
    traced_us = sum(kept_us)
    coverage = traced_us / (1e6 * anchor["mtp_decode_seconds"])
    d0_us = anchor["d0_round_us_measured"] * coverage
    base_us = traced_us
    base_tokens = len(rounds) + sum(int(r["acc"]) for r in rounds)
    new_us = 0.0
    new_tokens = 0
    saved_drafts = 0
    lost_tokens = 0
    for r, dec, us in zip(rounds, declined, kept_us):
        if dec:
            new_us += d0_us
            new_tokens += 1
            saved_drafts += int(r["d"])
            lost_tokens += int(r["acc"])
        else:
            new_us += us
            new_tokens += 1 + int(r["acc"])
    base_upt = base_us / base_tokens
    new_upt = new_us / new_tokens
    return {
        "round_timer_coverage_of_decode": coverage,
        "d0_round_us_in_timer_units": d0_us,
        "us_per_token_base": base_upt,
        "us_per_token_after": new_upt,
        "drafts_saved": saved_drafts,
        "tokens_lost": lost_tokens,
        "gain_pct": 100.0 * (base_upt / new_upt - 1.0),
    }


def replay_schedule(r: dict, h: float) -> int:
    """Re-run the shipped depth walk for this round at threshold `h`.

    The per-depth accept probability `p` is a function of the EMA vector and
    the pending-top-2 margin only, so it is invariant to `h` and can be
    replayed exactly from the trace. Depths beyond the recorded walk fall back
    to the raw EMA, which is what the shipped rule uses at depth >= 2.
    """
    walk = r.get("walk") or []
    ema = r.get("ema") or []
    # The traced `cap=` field is widthCap; the effective cap also folds in the
    # parent's offeredDepth, which shrinks at the end of the token window. One
    # walk entry is written per loop iteration *before* the guard, so a walk
    # shorter than d+1 means the loop exited on `depth == cap`, pinning the
    # effective cap at d. Otherwise the walk broke on the threshold and the cap
    # only bounds it from above.
    cap = int(r["d"]) if len(walk) == int(r["d"]) else int(r["cap"])
    expected = 0.0
    reach = 1.0
    depth = 0
    while depth < cap:
        if depth < len(walk):
            p = float(walk[depth]["p"])
        elif depth < len(ema):
            p = float(ema[depth])
        else:
            break
        reach *= p
        threshold = h * (1.0 + expected) / (1.0 + depth * h)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def evaluate_h(rounds: list[dict], h: float, x: float) -> dict:
    """Score threshold `h` under the trajectory-frozen prefix-acceptance model.

    Truncating a linear draft chain leaves the surviving rows byte-identical,
    so acceptance becomes min(acc, d'). Deepening is *not* evaluable offline:
    the trace holds no verdict for rows that were never proposed. Rounds that
    request more depth than was recorded are counted and left at the recorded
    depth so no unobserved acceptance is ever credited.
    """
    base_cpt, _, _ = cost_per_token(rounds, [False] * len(rounds), x)
    drafts = 0
    accepted = 0
    deepened = 0
    truncated = 0
    hist: dict[int, int] = {}
    requested: dict[int, int] = {}
    cost = 0.0
    yield_ = 0.0
    for r in rounds:
        want = replay_schedule(r, h)
        requested[want] = requested.get(want, 0) + 1
        d0 = int(r["d"])
        if want > d0:
            deepened += 1
        d = min(want, d0)
        if d < d0:
            truncated += 1
        acc = min(int(r["acc"]), d)
        drafts += d
        accepted += acc
        hist[d] = hist.get(d, 0) + 1
        cost += 1.0 + x * d
        yield_ += 1.0 + acc
    cpt = cost / yield_
    return {
        "h": h,
        "rounds": len(rounds),
        "proposed_rows": drafts,
        "accepted": accepted,
        "accepted_rate_pct": 100.0 * accepted / drafts if drafts else float("nan"),
        "depth_histogram": dict(sorted(hist.items())),
        "max_depth": max(hist) if hist else 0,
        "mean_depth": drafts / len(rounds) if rounds else float("nan"),
        "requested_histogram": dict(sorted(requested.items())),
        "rounds_deepened": deepened,
        "rounds_truncated": truncated,
        "evaluable": deepened == 0,
        "cost_per_token": cpt,
        "gain_pct": 100.0 * (base_cpt / cpt - 1.0),
    }


PREDICATES: dict[str, Callable[[float], Callable[[dict], bool]]] = {
    # Raw pending-top-2 margin. The shipped rule squashes this through
    # sigmoid(m/2), which cannot fall below 0.5 and so can never decline.
    "margin_lt": lambda t: (lambda r: float(r["m"]) < t),
    # Margin, but only where the round is deep enough for declination to pay.
    "margin_lt_and_d_ge_3": lambda t: (lambda r: float(r["m"]) < t and int(r["d"]) >= 3),
    "margin_lt_and_streak_0": lambda t: (lambda r: float(r["m"]) < t and int(r["streak"]) == 0),
    # The schedule's own forecast: decline when the predicted accepted count
    # does not cover the drafting bill it is about to run up.
    "expected_lt_cost_slack": lambda t: (lambda r: r["margin_over_cost"] < t),
    "expected_lt": lambda t: (lambda r: r["expected"] < t),
    # How much headroom the last accepted depth had over its own threshold.
    "slack_last_lt": lambda t: (lambda r: r["slack_last"] < t),
    "reach_last_lt": lambda t: (lambda r: r["reach_last"] < t),
    "ema1_lt": lambda t: (lambda r: r["ema1"] < t),
    "ema1_lt_and_d_ge_3": lambda t: (lambda r: r["ema1"] < t and int(r["d"]) >= 3),
    # Burstiness: does a dead round predict the next one?
    "prev_zero_run_ge": lambda t: (lambda r: r["prev_zero_run"] >= t),
    "prev_acc_le": lambda t: (lambda r: 0 <= r["prev_acc"] <= t),
}


def auc_zero_accept(rounds: list[dict], feature: str) -> float:
    """Probability that a zero-accept round ranks below an accepting round.

    0.5 means the feature carries no information about whether this round is
    about to accept anything; anything far from 0.5 in either direction is
    exploitable. Ties count as half, so a constant feature scores exactly 0.5.
    """
    pos = [r[feature] for r in rounds if int(r["acc"]) == 0]
    neg = [r[feature] for r in rounds if int(r["acc"]) > 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(
        1.0 if a < b else 0.5 if a == b else 0.0
        for a in pos
        for b in neg
    )
    return wins / (len(pos) * len(neg))


def measure_draft_cost(rounds: list[dict]) -> dict:
    """Measure X on this host instead of inheriting it from two E17 aggregates.

    Each traced round reports its own wall time and its own draft count, so the
    marginal cost of one drafted token is the slope of `round_us` on `d`,
    expressed in units of the intercept (a round that drafts nothing). That
    makes the cost model falsifiable from the same trace that feeds the fit,
    and it does not depend on the +6.821% figure the two-point fit assumed.

    The first round of a session carries warm-up and is dropped.
    """
    pts = [(int(r["d"]), float(r["round_us"])) for r in rounds if int(r["round"]) > 1]
    if len(pts) < 3:
        return {"draft_cost_units_measured": float("nan")}
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    if sxx == 0:
        return {"draft_cost_units_measured": float("nan")}
    slope = sxy / sxx
    intercept = my - slope * mx
    syy = sum((p[1] - my) ** 2 for p in pts)
    r2 = (sxy * sxy) / (sxx * syy) if syy else float("nan")
    by_depth: dict[int, list[float]] = {}
    for d, us in pts:
        by_depth.setdefault(d, []).append(us)
    return {
        "rounds_used": n,
        "round_us_intercept": intercept,
        "draft_us_slope": slope,
        "draft_cost_units_measured": slope / intercept if intercept else float("nan"),
        "r_squared": r2,
        "mean_round_us_by_depth": {
            d: {"n": len(v), "mean_us": statistics.mean(v)} for d, v in sorted(by_depth.items())
        },
    }


AUC_FEATURES = [
    "m",
    "streak",
    "cap",
    "d",
    "ema0",
    "ema1",
    "expected",
    "reach_last",
    "slack_last",
    "p_last",
    "p0",
    "margin_over_cost",
    "prev_zero_run",
    "prev_acc",
]

GRIDS: dict[str, list[float]] = {
    "margin_lt": [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
    "margin_lt_and_d_ge_3": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
    "margin_lt_and_streak_0": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    "expected_lt_cost_slack": [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "expected_lt": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4],
    "slack_last_lt": [-0.05, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "reach_last_lt": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
    "ema1_lt": [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
    "ema1_lt_and_d_ge_3": [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    "prev_zero_run_ge": [1, 2, 3, 4],
    "prev_acc_le": [0, 1, 2],
}


def describe(rounds: list[dict]) -> dict:
    zero = [r for r in rounds if int(r["acc"]) == 0]
    full = [r for r in rounds if int(r["acc"]) == int(r["d"])]
    partial = [r for r in rounds if 0 < int(r["acc"]) < int(r["d"])]
    hist: dict[int, int] = {}
    for r in rounds:
        hist[int(r["d"])] = hist.get(int(r["d"]), 0) + 1
    drafts = sum(int(r["d"]) for r in rounds)
    return {
        "round_count": len(rounds),
        "proposed_drafts": drafts,
        "depth_histogram": dict(sorted(hist.items())),
        # One verified row per draft plus the pending primary token, so a
        # depth-d round submits a width M = d+1 verify block to the target.
        "verify_width_histogram": {d + 1: n for d, n in sorted(hist.items())},
        "effective_mean_draft_len": drafts / len(rounds) if rounds else 0.0,
        "effective_max_draft_len": max(hist) if hist else 0,
        "accepted_drafts": sum(int(r["acc"]) for r in rounds),
        "zero_accept_rounds": len(zero),
        "zero_accept_drafts": sum(int(r["d"]) for r in zero),
        "zero_accept_mean_depth": statistics.mean([r["d"] for r in zero]) if zero else 0.0,
        "full_accept_rounds": len(full),
        "full_accept_mean_depth": statistics.mean([r["d"] for r in full]) if full else 0.0,
        "partial_accept_rounds": len(partial),
        "partial_accept_tokens": sum(int(r["acc"]) for r in partial),
        "zero_accept_margin_median": statistics.median([float(r["m"]) for r in zero]) if zero else 0.0,
        "full_accept_margin_median": statistics.median([float(r["m"]) for r in full]) if full else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", nargs="+", help="JSONL emitted by e21_trace.py --dump-rounds")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument(
        "--runs-root",
        default=".mlxfast-private/e21/runs",
        help="where <prompt>/reports/{03,04}-mtp-timed.json live",
    )
    args = ap.parse_args()

    rounds = load_rounds(args.rounds)
    by_prompt: dict[str, list[dict]] = {}
    for r in rounds:
        by_prompt.setdefault(r["prompt"], []).append(r)

    report: dict = {
        "prompts": sorted(by_prompt),
        "pooled_summary": describe(rounds),
        "per_prompt_summary": {k: describe(v) for k, v in sorted(by_prompt.items())},
    }

    report["measured_cost_model"] = measure_draft_cost(rounds)
    report["measured_cost_model_per_prompt"] = {
        k: measure_draft_cost(v) for k, v in sorted(by_prompt.items())
    }
    report["separability_auc"] = {f: auc_zero_accept(rounds, f) for f in AUC_FEATURES}
    report["separability_auc_per_prompt"] = {
        k: {f: auc_zero_accept(v, f) for f in AUC_FEATURES} for k, v in sorted(by_prompt.items())
    }

    anchors = {k: leg_anchor(args.runs_root, k) for k in sorted(by_prompt)}
    report["leg_anchored_cost_model"] = anchors
    hs = [a["h_measured"] for a in anchors.values() if a]
    report["h_measured_pooled"] = statistics.fmean(hs) if hs else None
    report["h_measured_range"] = [min(hs), max(hs)] if hs else None
    report["nonparametric_oracle_per_prompt"] = {
        k: evaluate_nonparametric(
            by_prompt[k], anchors[k], [int(r["acc"]) == 0 for r in by_prompt[k]]
        )
        for k in sorted(by_prompt)
        if anchors[k]
    }
    if hs:
        report["oracle_at_measured_h"] = {
            k: evaluate(by_prompt[k], lambda r: int(r["acc"]) == 0, anchors[k]["h_measured"])
            for k in sorted(by_prompt)
            if anchors[k]
        }

    x_meas = report["measured_cost_model"]["draft_cost_units_measured"]
    # h == SHIPPED_H must reproduce the recorded depth exactly, otherwise the
    # offline replay does not model the shipped rule and every h row is void.
    replay_mismatch = [
        {"round": r["round"], "recorded_d": int(r["d"]), "replayed_d": replay_schedule(r, SHIPPED_H)}
        for r in rounds
        if replay_schedule(r, SHIPPED_H) != int(r["d"])
    ]
    report["replay_fidelity"] = {
        "shipped_h": SHIPPED_H,
        "rounds": len(rounds),
        "mismatches": len(replay_mismatch),
        "exact": not replay_mismatch,
        "examples": replay_mismatch[:8],
    }
    # Score the sweep at the leg-anchored marginal, not the within-arm
    # regression: the latter has no d=0 observation and is biased by the
    # schedule choosing depth endogenously.
    x_score = report["h_measured_pooled"] or x_meas
    report["h_sweep_scored_at_x"] = x_score
    report["h_sweep"] = {
        f"h={h}": {
            "pooled": evaluate_h(rounds, h, x_score),
            "per_prompt_gain_pct": {
                k: evaluate_h(v, h, x_score)["gain_pct"] for k, v in sorted(by_prompt.items())
            },
        }
        for h in H_GRID
    }
    for row in report["h_sweep"].values():
        row["worst_prompt_gain_pct"] = min(row["per_prompt_gain_pct"].values())

    # Every prompt is priced with its own leg-anchored marginal, so a rule is
    # never credited at a cost the host did not actually charge on that prompt.
    prompt_x = {k: (anchors[k] or {}).get("h_measured") or x_score for k in by_prompt}
    report["oracle"] = oracle = evaluate(rounds, lambda r: int(r["acc"]) == 0, x_score)

    scored = []
    for name, factory in PREDICATES.items():
        for t in GRIDS[name]:
            res = evaluate(rounds, factory(t), x_score)
            res["predicate"] = name
            res["threshold"] = t
            # Worst single prompt, so a rule that only pays on one prompt loses.
            res["per_prompt_gain_pct"] = {
                k: evaluate(v, factory(t), prompt_x[k])["gain_pct"]
                for k, v in sorted(by_prompt.items())
            }
            res["worst_prompt_gain_pct"] = min(res["per_prompt_gain_pct"].values())
            scored.append(res)

    scored.sort(key=lambda r: (r["worst_prompt_gain_pct"], r["gain_pct"]), reverse=True)
    report["ranked"] = scored[: args.top]
    report["all_candidates"] = scored

    best = scored[0]
    report["sensitivity"] = {
        f"X={x:.4f}": evaluate(
            rounds, PREDICATES[best["predicate"]](best["threshold"]), x
        )["gain_pct"]
        for x in (x_score, 0.18, 0.25, 0.362, 0.45, 0.5)
    }

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))

    print(f"prompts: {', '.join(report['prompts'])}")
    print()
    print("shipped-default depth histogram (512 decode tokens, MTP leg, own prose)")
    print(
        f"  {'prompt':<24}{'rounds':>7}{'rows':>6}{'acc':>6}{'rate%':>7}"
        f"{'mean':>7}{'max':>5}  depth -> verify width (M=d+1)"
    )
    for name, v in sorted(report["per_prompt_summary"].items()):
        rate = 100.0 * v["accepted_drafts"] / v["proposed_drafts"] if v["proposed_drafts"] else 0.0
        print(
            f"  {name:<24}{v['round_count']:>7}{v['proposed_drafts']:>6}{v['accepted_drafts']:>6}"
            f"{rate:>7.1f}{v['effective_mean_draft_len']:>7.3f}{v['effective_max_draft_len']:>5}"
            f"  {v['depth_histogram']} -> {v['verify_width_histogram']}"
        )
    p = report["pooled_summary"]
    prate = 100.0 * p["accepted_drafts"] / p["proposed_drafts"] if p["proposed_drafts"] else 0.0
    print(
        f"  {'POOLED':<24}{p['round_count']:>7}{p['proposed_drafts']:>6}{p['accepted_drafts']:>6}"
        f"{prate:>7.1f}{p['effective_mean_draft_len']:>7.3f}{p['effective_max_draft_len']:>5}"
        f"  {p['depth_histogram']} -> {p['verify_width_histogram']}"
    )
    tot = p["round_count"]
    share = {w: round(100.0 * n / tot, 2) for w, n in p["verify_width_histogram"].items()}
    print(f"  pooled verify-width share %: {share}")
    spread = {
        k: (
            min(v[k] for v in report["per_prompt_summary"].values()),
            max(v[k] for v in report["per_prompt_summary"].values()),
        )
        for k in ("effective_mean_draft_len", "effective_max_draft_len", "round_count")
    }
    print(f"  spread across prompts: {spread}")
    print()
    s = report["pooled_summary"]
    print(
        f"pooled: {s['round_count']} rounds, {s['proposed_drafts']} drafts, "
        f"{s['accepted_drafts']} accepted; zero-accept rounds {s['zero_accept_rounds']} "
        f"carrying {s['zero_accept_drafts']} drafts at mean depth "
        f"{s['zero_accept_mean_depth']:.3f}; partial-accept rounds "
        f"{s['partial_accept_rounds']} ({s['partial_accept_tokens']} tokens)"
    )
    print(
        f"margin median: zero-accept {s['zero_accept_margin_median']:.3f} vs "
        f"full-accept {s['full_accept_margin_median']:.3f}"
    )
    print(
        f"ORACLE decline-all-zero-accept: fired {oracle['fired']}, "
        f"drafts saved {oracle['drafts_saved']}, tokens lost {oracle['tokens_lost']}, "
        f"gain {oracle['gain_pct']:+.3f}%"
    )
    print()
    m = report["measured_cost_model"]
    print(
        f"measured cost model: round_us = {m['round_us_intercept']:.0f} "
        f"+ {m['draft_us_slope']:.0f}*d  (R2={m['r_squared']:.3f}, "
        f"n={m['rounds_used']})  =>  X = {m['draft_cost_units_measured']:.3f} "
        f"vs shipped {SHIPPED_H} (this estimator is biased up: no d=0 anchor)"
    )
    for d, st in m["mean_round_us_by_depth"].items():
        print(f"    d={d}: n={st['n']:>4}  mean {st['mean_us']:>9.0f} us")
    print()
    print("leg-anchored cost model (serial control leg IS this session at d=0)")
    for k, a in report["leg_anchored_cost_model"].items():
        if not a:
            print(f"  {k:<28} no usable serial leg")
            continue
        print(
            f"  {k:<28} d0_round={a['d0_round_us_measured']:>8.0f} us  "
            f"marginal_draft={a['draft_us_marginal']:>7.0f} us  "
            f"h_measured={a['h_measured']:.4f}"
        )
    if report["h_measured_range"]:
        lo, hi = report["h_measured_range"]
        print(
            f"  => h_measured pooled {report['h_measured_pooled']:.4f} "
            f"(range {lo:.4f}..{hi:.4f}) vs shipped {SHIPPED_H}"
        )
    print()
    print("ORACLE at the honest price (decline exactly the zero-accept rounds)")
    for k, o in report.get("nonparametric_oracle_per_prompt", {}).items():
        par = report["oracle_at_measured_h"][k]
        print(
            f"  {k:<28} nonparametric {o['gain_pct']:+7.3f}%  "
            f"(round-unit model {par['gain_pct']:+7.3f}%)  "
            f"drafts_saved={o['drafts_saved']:>4} tokens_lost={o['tokens_lost']:>3} "
            f"timer_cov={o['round_timer_coverage_of_decode']:.3f}"
        )
    print()
    f = report["replay_fidelity"]
    print(
        f"offline replay of the shipped walk at h={f['shipped_h']}: "
        f"{f['mismatches']}/{f['rounds']} mismatches "
        f"({'EXACT' if f['exact'] else 'NOT EXACT -- h rows are void'})"
    )
    if not f["exact"]:
        print(f"    examples: {f['examples']}")
    print(
        f"threshold sweep, scored at leg-anchored X={report['h_sweep_scored_at_x']:.4f}"
        " (trajectory-frozen)"
    )
    print(
        f"  {'h':>6} {'rows':>5} {'meandep':>8} {'maxd':>5} {'acc':>5} {'rate%':>6} "
        f"{'trunc':>6} {'deep':>5} {'gain%':>8} {'worst%':>8}  histogram"
    )
    for key, row in report["h_sweep"].items():
        p = row["pooled"]
        flag = "" if p["evaluable"] else "  <- NOT EVALUABLE (deepens)"
        print(
            f"  {p['h']:>6.3f} {p['proposed_rows']:>5} {p['mean_depth']:>8.3f} "
            f"{p['max_depth']:>5} {p['accepted']:>5} {p['accepted_rate_pct']:>6.1f} "
            f"{p['rounds_truncated']:>6} {p['rounds_deepened']:>5} "
            f"{p['gain_pct']:>+8.3f} {row['worst_prompt_gain_pct']:>+8.3f}  "
            f"{p['depth_histogram']}{flag}"
        )
    print()
    print("separability (AUC for 'this round accepts nothing'; 0.500 = no signal)")
    for feat, auc in sorted(
        report["separability_auc"].items(), key=lambda kv: -abs(kv[1] - 0.5)
    ):
        per = report["separability_auc_per_prompt"]
        spread = (
            f"  per-prompt {min(p[feat] for p in per.values()):.3f}"
            f"..{max(p[feat] for p in per.values()):.3f}"
            if len(per) > 1
            else ""
        )
        print(f"  {feat:<20} {auc:.3f}{spread}")
    print()
    print(
        f"{'predicate':<24} {'thr':>7} {'fire':>5} {'prec%':>7} {'rec%':>6} "
        f"{'saved':>6} {'lost':>5} {'rows':>5} {'maxd':>5} {'gain%':>8} {'worst%':>8}"
    )
    for r in report["ranked"]:
        print(
            f"{r['predicate']:<24} {r['threshold']:>7.3f} {r['fired']:>5} "
            f"{r['precision_pct']:>7.1f} {r['recall_pct']:>6.1f} {r['drafts_saved']:>6} "
            f"{r['tokens_lost']:>5} {r['proposed_rows_after']:>5} {r['max_depth_after']:>5} "
            f"{r['gain_pct']:>+8.3f} {r['worst_prompt_gain_pct']:>+8.3f}"
        )
    print()
    print(f"best depth histogram after: {report['ranked'][0]['depth_histogram_after']}")
    print(f"sensitivity to X: {report['sensitivity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
