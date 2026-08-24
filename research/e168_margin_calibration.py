#!/usr/bin/env python3
"""E168: is the pending-primary top-2 margin predictive of draft acceptance,
and is `min(EMA, sigmoid(margin/T))` a calibrated way to use it?

Reads `mtp-trace:` round lines written by `MLX_QWEN_MTP_TRACE=1`. Two leg kinds
answer two different questions and must not be pooled:

  pinned  `MLX_E159_FIXED_DRAFT_DEPTH=7` replaces the schedule, so every round
          proposes 7 drafts whatever the margin says. This is the ONLY leg that
          observes acceptance at a position the shipped clamp would have
          refused to draft, so the calibration curve is fitted here.
  adapt   the shipped adaptive schedule. Used only to count how often the clamp
          binds in production and how much offered depth it removes; its
          acceptance observations are censored BY the clamp under test.

Usage:
  research/e168_margin_calibration.py --pinned DIR [DIR ...] \
      [--adapt DIR [DIR ...]] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

# Shipped constants, Qwen36MTPBlockSession.swift.
HEAD_STEP_COST_RATIO = 0.18
MAX_DEPTH = 8
WIDTH_CAP = 7
CLAMP_T = {0: 2.0, 1: 3.0}

ROUND_RE = re.compile(
    r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*? m=(\S+) streak=(\d+) "
    r"cap=(\d+) ema=(\S+)"
)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def parse_rounds(path: str) -> list[dict]:
    rounds = []
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            match = ROUND_RE.match(line)
            if not match:
                continue
            margin = float(match.group(4))
            rounds.append(
                {
                    "round": int(match.group(1)),
                    "d": int(match.group(2)),
                    "acc": int(match.group(3)),
                    "margin": margin,
                    "streak": int(match.group(5)),
                    "cap": int(match.group(6)),
                    "ema": [float(v) for v in match.group(7).split(",")],
                }
            )
    return rounds


def load_legs(dirs: list[str], kind: str) -> list[dict]:
    legs = []
    for directory in dirs:
        trace = os.path.join(directory, "trace.txt")
        if not os.path.exists(trace):
            sys.exit(f"e168: no trace.txt in {directory}")
        meta = {}
        meta_path = os.path.join(directory, "meta.txt")
        if os.path.exists(meta_path):
            for line in open(meta_path):
                if "=" in line:
                    key, _, value = line.partition("=")
                    meta[key.strip()] = value.strip()
        rounds = parse_rounds(trace)
        legs.append(
            {
                "dir": directory,
                "kind": kind,
                "label": meta.get("prompt_id", os.path.basename(directory)),
                "meta": meta,
                "rounds": rounds,
            }
        )
    return legs


def walk(margin: float, ema: list[float], offered: int, clamped: bool) -> int:
    """Replay `costModelDepth` exactly, with or without the margin clamps."""
    cap = min(min(offered, MAX_DEPTH), WIDTH_CAP)
    if cap <= 0:
        return 0
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        p = ema[depth]
        if clamped and depth in CLAMP_T and not math.isnan(margin):
            p = min(p, sigmoid(margin / CLAMP_T[depth]))
        reach *= p
        threshold = (
            HEAD_STEP_COST_RATIO
            * (1.0 + expected)
            / (1.0 + depth * HEAD_STEP_COST_RATIO)
        )
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def observations(rounds: list[dict], position: int) -> list[tuple[float, int, float]]:
    """(margin, accepted, ema_at_position) for rounds that RESOLVED `position`.

    Position k is resolved when the round proposed at least k+1 drafts and the
    accept walk reached it: every position below k was accepted. Positions past
    the first reject were verified but never walked, so they carry no outcome.
    """
    out = []
    for record in rounds:
        if record["d"] <= position or record["acc"] < position:
            continue
        if math.isnan(record["margin"]):
            continue
        accepted = 1 if record["acc"] > position else 0
        out.append((record["margin"], accepted, record["ema"][position]))
    return out


def fit_temperature(samples: list[tuple[float, int, float]]) -> tuple[float, float]:
    """MLE of T in P(accept) = sigmoid(margin / T) by golden-section search."""
    if not samples:
        return float("nan"), float("nan")

    def negative_log_likelihood(temperature: float) -> float:
        total = 0.0
        for margin, accepted, _ in samples:
            p = min(max(sigmoid(margin / temperature), 1e-12), 1 - 1e-12)
            total -= math.log(p) if accepted else math.log(1.0 - p)
        return total

    lo, hi = 1e-3, 400.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = math.log(lo), math.log(hi)
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = negative_log_likelihood(math.exp(c)), negative_log_likelihood(math.exp(d))
    for _ in range(200):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = negative_log_likelihood(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = negative_log_likelihood(math.exp(d))
    best = math.exp((a + b) / 2.0)
    return best, negative_log_likelihood(best)


def score_predictor(samples, predict) -> dict:
    """Log-loss and Brier score of one probability model."""
    if not samples:
        return {"n": 0}
    log_loss = 0.0
    brier = 0.0
    for margin, accepted, ema in samples:
        p = min(max(predict(margin, ema), 1e-12), 1 - 1e-12)
        log_loss -= math.log(p) if accepted else math.log(1.0 - p)
        brier += (p - accepted) ** 2
    n = len(samples)
    return {
        "n": n,
        "log_loss": log_loss / n,
        "brier": brier / n,
        "mean_prediction": sum(predict(m, e) for m, _, e in samples) / n,
    }


def auc(samples: list[tuple[float, int, float]]) -> float:
    """Mann-Whitney AUC of margin as a ranker of acceptance."""
    positives = [m for m, a, _ in samples if a == 1]
    negatives = [m for m, a, _ in samples if a == 0]
    if not positives or not negatives:
        return float("nan")
    ordered = sorted((m, a) for m, a, _ in samples)
    ranks: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    positive_rank_sum = sum(
        ranks[i] for i, (_, a) in enumerate(ordered) if a == 1
    )
    n_pos, n_neg = len(positives), len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963985
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - spread, centre + spread


BIN_EDGES = [-1e9, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 1e9]


def bin_index(margin: float) -> int:
    for index in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[index] <= margin < BIN_EDGES[index + 1]:
            return index
    return len(BIN_EDGES) - 2


def calibration_table(samples, position: int) -> list[dict]:
    buckets = defaultdict(lambda: {"n": 0, "accepted": 0, "margin_sum": 0.0,
                                   "ema_sum": 0.0})
    for margin, accepted, ema in samples:
        bucket = buckets[bin_index(margin)]
        bucket["n"] += 1
        bucket["accepted"] += accepted
        bucket["margin_sum"] += margin
        bucket["ema_sum"] += ema
    rows = []
    for index in sorted(buckets):
        bucket = buckets[index]
        n = bucket["n"]
        mean_margin = bucket["margin_sum"] / n
        low, high = wilson(bucket["accepted"], n)
        rows.append(
            {
                "bin": f"[{BIN_EDGES[index]:g},{BIN_EDGES[index + 1]:g})".replace(
                    "-1e+09", "-inf"
                ).replace("1e+09", "inf"),
                "n": n,
                "mean_margin": mean_margin,
                "empirical": bucket["accepted"] / n,
                "ci_low": low,
                "ci_high": high,
                "clamp_sigmoid": sigmoid(mean_margin / CLAMP_T[position]),
                "mean_ema": bucket["ema_sum"] / n,
            }
        )
    return rows


def analyse_pinned(legs: list[dict]) -> dict:
    result = {"legs": [], "positions": {}}
    pooled: dict[int, list] = {0: [], 1: [], 2: []}
    for leg in legs:
        rounds = leg["rounds"]
        depths = [r["d"] for r in rounds]
        result["legs"].append(
            {
                "label": leg["label"],
                "dir": leg["dir"],
                "rounds": len(rounds),
                "mean_d": sum(depths) / len(depths) if depths else 0.0,
                "mean_acc": (
                    sum(r["acc"] for r in rounds) / len(rounds) if rounds else 0.0
                ),
                "accept_fraction": (
                    sum(r["acc"] for r in rounds) / sum(depths) if sum(depths) else 0.0
                ),
                "margin_mean": (
                    sum(r["margin"] for r in rounds) / len(rounds) if rounds else 0.0
                ),
            }
        )
        for position in pooled:
            pooled[position].extend(observations(rounds, position))

    for position in (0, 1):
        samples = pooled[position]
        temperature, _ = fit_temperature(samples)
        shipped_t = CLAMP_T[position]
        result["positions"][position] = {
            "n": len(samples),
            "base_rate": (
                sum(a for _, a, _ in samples) / len(samples) if samples else 0.0
            ),
            "fitted_temperature": temperature,
            "shipped_temperature": shipped_t,
            "auc_margin": auc(samples),
            "table": calibration_table(samples, position),
            "predictors": {
                "ema_only": score_predictor(samples, lambda m, e: e),
                "shipped_clamp": score_predictor(
                    samples, lambda m, e, t=shipped_t: min(e, sigmoid(m / t))
                ),
                "fitted_clamp": score_predictor(
                    samples, lambda m, e, t=temperature: min(e, sigmoid(m / t))
                ),
                "margin_only_fitted": score_predictor(
                    samples, lambda m, e, t=temperature: sigmoid(m / t)
                ),
                "base_rate": score_predictor(
                    samples,
                    lambda m, e, b=(
                        sum(a for _, a, _ in samples) / len(samples)
                        if samples
                        else 0.5
                    ): b,
                ),
            },
        }
    return result


def analyse_adapt(legs: list[dict], offered: int) -> dict:
    result = {"legs": [], "pooled": {}}
    total = binds0 = binds1 = 0
    depth_loss_rounds = 0
    depth_loss_sum = 0
    pure_loss_rounds = 0
    replay_mismatch = 0
    for leg in legs:
        leg_total = leg_binds = leg_loss = 0
        leg_loss_sum = 0
        # The parent shortens its offer on the last round so the leg lands on
        # the configured token count, so that round's depth is window-bounded
        # rather than policy-bounded and cannot be replayed at offer 8.
        for record in leg["rounds"][:-1]:
            margin, ema = record["margin"], record["ema"]
            if math.isnan(margin):
                continue
            leg_total += 1
            total += 1
            bind0 = sigmoid(margin / 2.0) < ema[0]
            bind1 = sigmoid(margin / 3.0) < ema[1]
            binds0 += bind0
            binds1 += bind1
            if bind0 or bind1:
                leg_binds += 1
            clamped = walk(margin, ema, offered, clamped=True)
            unclamped = walk(margin, ema, offered, clamped=False)
            # The offline walk must reproduce the depth the session actually
            # chose. Any mismatch means the counterfactual below is fiction.
            if clamped != record["d"]:
                replay_mismatch += 1
            if unclamped > clamped:
                depth_loss_rounds += 1
                leg_loss += 1
                depth_loss_sum += unclamped - clamped
                leg_loss_sum += unclamped - clamped
                # The clamp removed positions the round then never resolved.
                # A round that accepted every draft it DID make is a round
                # where the removed position was very likely acceptable too.
                if record["acc"] == record["d"]:
                    pure_loss_rounds += 1
        result["legs"].append(
            {
                "label": leg["label"],
                "rounds": leg_total,
                "bind_fraction": leg_binds / leg_total if leg_total else 0.0,
                "depth_loss_fraction": leg_loss / leg_total if leg_total else 0.0,
                "mean_depth_loss": leg_loss_sum / leg_total if leg_total else 0.0,
                "mean_d": (
                    sum(r["d"] for r in leg["rounds"]) / len(leg["rounds"])
                    if leg["rounds"]
                    else 0.0
                ),
            }
        )
    result["pooled"] = {
        "rounds": total,
        "replay_mismatch": replay_mismatch,
        "bind_pos0_fraction": binds0 / total if total else 0.0,
        "bind_pos1_fraction": binds1 / total if total else 0.0,
        "depth_loss_fraction": depth_loss_rounds / total if total else 0.0,
        "mean_depth_removed": depth_loss_sum / total if total else 0.0,
        "pure_loss_fraction": pure_loss_rounds / total if total else 0.0,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinned", nargs="*", default=[])
    parser.add_argument("--adapt", nargs="*", default=[])
    parser.add_argument("--offered", type=int, default=8)
    parser.add_argument("--json")
    args = parser.parse_args()

    report: dict = {"offered_depth": args.offered}
    if args.pinned:
        report["pinned"] = analyse_pinned(load_legs(args.pinned, "pinned"))
    if args.adapt:
        report["adapt"] = analyse_adapt(load_legs(args.adapt, "adapt"), args.offered)

    if "adapt" in report:
        pooled = report["adapt"]["pooled"]
        print(f"=== shipped adaptive policy, {pooled['rounds']} rounds ===")
        print(f"offline walk replay mismatches: {pooled['replay_mismatch']}")
        print(
            f"clamp binds at pos0 {pooled['bind_pos0_fraction']:.3f}  "
            f"pos1 {pooled['bind_pos1_fraction']:.3f}"
        )
        print(
            f"rounds where the clamp REMOVED depth "
            f"{pooled['depth_loss_fraction']:.3f}, "
            f"mean rows removed/round {pooled['mean_depth_removed']:.3f}"
        )
        print(
            f"of those, rounds that then accepted every draft they made "
            f"{pooled['pure_loss_fraction']:.3f}"
        )
        for leg in report["adapt"]["legs"]:
            print(
                f"  {leg['label']:<24} n={leg['rounds']:<4} "
                f"bind={leg['bind_fraction']:.3f} "
                f"depth_loss={leg['depth_loss_fraction']:.3f} "
                f"rows_removed={leg['mean_depth_loss']:.3f} "
                f"mean_d={leg['mean_d']:.3f}"
            )
        print()

    if "pinned" in report:
        print("=== pinned-depth legs (uncensored) ===")
        for leg in report["pinned"]["legs"]:
            print(
                f"  {leg['label']:<24} rounds={leg['rounds']:<4} "
                f"mean_d={leg['mean_d']:.2f} mean_acc={leg['mean_acc']:.3f} "
                f"accept_frac={leg['accept_fraction']:.3f} "
                f"mean_margin={leg['margin_mean']:.2f}"
            )
        for position in (0, 1):
            block = report["pinned"]["positions"][position]
            print(
                f"\n--- position {position}: n={block['n']} "
                f"base_rate={block['base_rate']:.4f} "
                f"AUC(margin)={block['auc_margin']:.4f} ---"
            )
            print(
                f"shipped T={block['shipped_temperature']:.2f}  "
                f"fitted T*={block['fitted_temperature']:.4g}"
            )
            print(
                f"{'bin':<12}{'n':>6}{'mean_m':>9}{'empirical':>11}"
                f"{'95% CI':>18}{'clamp':>9}{'mean_ema':>10}"
            )
            for row in block["table"]:
                interval = f"[{row['ci_low']:.3f},{row['ci_high']:.3f}]"
                print(
                    f"{row['bin']:<12}{row['n']:>6}{row['mean_margin']:>9.3f}"
                    f"{row['empirical']:>11.4f}{interval:>18}"
                    f"{row['clamp_sigmoid']:>9.4f}{row['mean_ema']:>10.4f}"
                )
            print(f"{'predictor':<20}{'log_loss':>11}{'brier':>10}{'mean_p':>10}")
            for name, scores in block["predictors"].items():
                if scores.get("n"):
                    print(
                        f"{name:<20}{scores['log_loss']:>11.5f}"
                        f"{scores['brier']:>10.5f}{scores['mean_prediction']:>10.4f}"
                    )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
