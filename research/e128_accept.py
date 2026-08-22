#!/usr/bin/env python3
"""E128 rung 1 -- uncensored per-position acceptance from a forced-depth arm.

The shipped schedule is adaptive, so a shipped trace observes position j only
on rounds the estimator already believed would reach j. Every per-position
acceptance vector read from such a trace is survivor biased upward. A
forced-depth leg proposes the same number of drafts on every round, so it
observes position j on every round that reaches it by acceptance alone.

Reported per leg, never pooled across regimes:

  * `p_j`, the conditional probability that draft j is accepted given drafts
    0..j-1 were accepted, with counts and Wilson 95 % intervals.
  * the margin distribution at the rounds that reach positions 0 and 1, and
    the quantization step of the margin itself.
  * the AUC of the round's margin against acceptance AT each position.
  * how often `pendingTop2` was nil or shorter than two entries, which
    silently disables the shipped margin override.

  usage: research/e128_accept.py RUN_DIR [RUN_DIR ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from e128_replay import MAX_DEPTH, read_meta, read_rounds


def wilson(successes: int, total: int, z: float = 1.959963985) -> tuple:
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(
        (phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return (centre - half, centre + half)


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUC with half credit for ties."""
    if not pos or not neg:
        return None
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum = 0.0
    index = 0
    rank = 1
    while index < len(merged):
        stop = index
        while stop + 1 < len(merged) and merged[stop + 1][0] == merged[index][0]:
            stop += 1
        average_rank = (rank + rank + (stop - index)) / 2.0
        for k in range(index, stop + 1):
            if merged[k][1] == 1:
                rank_sum += average_rank
        rank += stop - index + 1
        index = stop + 1
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_ci(pos: list[float], neg: list[float], draws: int = 2000,
           seed: int = 128) -> tuple:
    point = auc(pos, neg)
    if point is None or draws <= 0:
        return (point, None, None)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        rp = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
        rn = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        value = auc(rp, rn)
        if value is not None:
            samples.append(value)
    samples.sort()
    if not samples:
        return (point, None, None)
    return (point,
            samples[int(0.025 * (len(samples) - 1))],
            samples[int(0.975 * (len(samples) - 1))])


def quantization_step(values: list[float]) -> dict:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return {"samples": 0}
    steps = {}
    for power in range(1, 13):
        unit = 2.0 ** -power
        exact = sum(
            1 for v in finite if abs(v / unit - round(v / unit)) < 1e-9)
        steps["2^-%d" % power] = exact / len(finite)
    distinct = sorted(set(finite))
    gaps = [b - a for a, b in zip(distinct, distinct[1:])]
    return {
        "samples": len(finite),
        "distinct": len(distinct),
        "min_gap": min(gaps) if gaps else None,
        "min": min(finite),
        "max": max(finite),
        "multiple_of_fraction": steps,
    }


def analyse_leg(run_dir: Path, boot: int) -> dict:
    meta = read_meta(run_dir)
    rounds = read_rounds(run_dir)
    forced = meta.get("forced_depth", "none")

    reached = [0] * MAX_DEPTH   # rounds that proposed j and accepted 0..j-1
    accepted = [0] * MAX_DEPTH  # of those, rounds that also accepted j
    margins_pos: list[list[float]] = [[] for _ in range(MAX_DEPTH)]
    margins_neg: list[list[float]] = [[] for _ in range(MAX_DEPTH)]
    nil_margin_rounds = 0
    depths = []
    accepts = []
    for record in rounds:
        depth, acc, margin = record["depth"], record["accepted"], record["margin"]
        depths.append(depth)
        accepts.append(acc)
        if math.isnan(margin):
            nil_margin_rounds += 1
        for j in range(min(depth, MAX_DEPTH)):
            if acc < j:
                break  # the chain broke before this position was tested
            reached[j] += 1
            hit = acc > j
            accepted[j] += hit
            if not math.isnan(margin):
                (margins_pos if hit else margins_neg)[j].append(margin)

    positions = []
    for j in range(MAX_DEPTH):
        if reached[j] == 0:
            continue
        low, high = wilson(accepted[j], reached[j])
        point, lo, hi = auc_ci(margins_pos[j], margins_neg[j], boot)
        positions.append({
            "position": j,
            "reached": reached[j],
            "accepted": accepted[j],
            "p": accepted[j] / reached[j],
            "wilson_low": low,
            "wilson_high": high,
            "margin_auc": point,
            "margin_auc_low": lo,
            "margin_auc_high": hi,
        })

    all_margins = [r["margin"] for r in rounds]
    total_accepted = sum(accepts)
    total_drafted = sum(depths)
    return {
        "run_dir": str(run_dir),
        "prompt_id": run_dir.name,
        "forced_depth": forced,
        "rounds": len(rounds),
        "all_tokens_matched": meta.get("all_tokens_matched"),
        "residual_divergence_count": meta.get("residual_divergence_count"),
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "mean_depth": (sum(depths) / len(depths)) if depths else 0.0,
        "mean_accepted": (total_accepted / len(accepts)) if accepts else 0.0,
        "accept_rate": (total_accepted / total_drafted) if total_drafted else 0.0,
        "nil_margin_rounds": nil_margin_rounds,
        "margin_quantization": quantization_step(all_margins),
        "positions": positions,
        # The empirical joint sample the counterfactual simulator resamples:
        # one (margin, accepted-run, proposed depth) triple per round.
        "rounds_detail": [
            {"margin": r["margin"], "accepted": r["accepted"],
             "depth": r["depth"], "round": r["round"]}
            for r in rounds
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    legs = [analyse_leg(d, args.boot) for d in args.run_dirs]
    for leg in legs:
        print("\n=== %s (forced=%s, %d rounds, matched=%s) ===" % (
            leg["prompt_id"], leg["forced_depth"], leg["rounds"],
            leg["all_tokens_matched"]))
        print("mean depth %.3f  mean accepted %.3f  accept rate %.4f  "
              "nil-margin rounds %d" % (
                  leg["mean_depth"], leg["mean_accepted"], leg["accept_rate"],
                  leg["nil_margin_rounds"]))
        print("%3s %8s %8s %8s %-18s %8s %-18s" % (
            "j", "reached", "accept", "p_j", "wilson95", "auc", "auc95"))
        for row in leg["positions"]:
            print("%3d %8d %8d %8.4f [%.4f,%.4f] %8s %s" % (
                row["position"], row["reached"], row["accepted"], row["p"],
                row["wilson_low"], row["wilson_high"],
                "n/a" if row["margin_auc"] is None
                else "%.4f" % row["margin_auc"],
                "" if row["margin_auc_low"] is None
                else "[%.4f,%.4f]" % (row["margin_auc_low"],
                                      row["margin_auc_high"])))
        q = leg["margin_quantization"]
        if q["samples"]:
            best = [k for k, v in q["multiple_of_fraction"].items() if v == 1.0]
            print("margin: %d samples, %d distinct, min gap %s, range [%s, %s], "
                  "exact multiples of %s" % (
                      q["samples"], q["distinct"], q["min_gap"], q["min"],
                      q["max"], ", ".join(best) if best else "nothing tested"))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"legs": legs}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
