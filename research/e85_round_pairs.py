#!/usr/bin/env python3
"""Median of paired per-round deltas for an E85 ABBA session.

    usage: research/e85_round_pairs.py SESSION_DIR [--json OUT] [--dump-segments]

Leg totals sum the rare multi-millisecond OS scheduling spikes that land in
`d_submit1_us`, `d_chain_us` and `commit_us`. A median over paired per-round
differences rejects them. `qwen-alphonse` measured a factor of four between the
two readings on the verify path, so the paired median is the primary statistic
here and the leg total is reported beside it.

`MLX_QWEN_MTP_TRACE_PATH` is opened `O_APPEND`, and the reference, serial and
timed workers all write the same file, so the records are segmented on the
`mtp-trace: begin` marker and only the timed segment is analysed.

Pairing follows the ABBA blocks. Inside `base ab ab base` the two adjacent
base/ab legs form a pair, and the two base legs at the block ends form the
session null: the same contrast computed over unchanged code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path

FIELD = re.compile(r"(\w+)=(-?[\d.]+)")
BOOTSTRAP = 20000
SEED = 20260820
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def parse_rounds(path: Path) -> list[list[dict]]:
    """Split one trace file into per-worker segments of round records."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("mtp-trace: begin"):
            if current:
                segments.append(current)
            current = []
            continue
        if not line.startswith("mtp-trace: round="):
            continue
        rec = {k: float(v) for k, v in FIELD.findall(line)}
        if "round" in rec:
            current.append(rec)
    if current:
        segments.append(current)
    return segments


def timed_segment(segments: list[list[dict]]) -> list[dict]:
    """The timed MTP leg is the longest segment that actually proposes drafts."""
    drafting = [s for s in segments if any(r.get("d", 0) > 0 for r in s)]
    pool = drafting or segments
    return max(pool, key=len) if pool else []


def sequence_key(rounds: list[dict]) -> tuple:
    return tuple((int(r["d"]), int(r["acc"])) for r in rounds)


def median_ci(values: list[float], reps: int = BOOTSTRAP) -> dict:
    if not values:
        return {"median": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan, "n": 0}
    rng = random.Random(SEED)
    n = len(values)
    boots = []
    for _ in range(reps):
        boots.append(statistics.median(
            [values[rng.randrange(n)] for _ in range(n)]))
    boots.sort()
    return {
        "median": statistics.median(values),
        "ci95_lo": boots[int(0.025 * reps)],
        "ci95_hi": boots[int(0.975 * reps) - 1],
        "n": n,
        "mean": statistics.fmean(values),
    }


def median_ci_clustered(clusters: list[list[float]], reps: int = BOOTSTRAP) -> dict:
    """Bootstrap whole ABBA blocks, not rounds.

    Every round inside one block shares that block's machine state, so a
    round-level resample measures only within-block scatter and reports an
    interval that is far too narrow. Resampling blocks keeps the leg-level
    offset in the interval, which is the term the design must average away.
    """
    clusters = [c for c in clusters if c]
    if len(clusters) < 2:
        return {"median": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan,
                "clusters": len(clusters)}
    rng = random.Random(SEED)
    n = len(clusters)
    boots = []
    for _ in range(reps):
        pooled: list[float] = []
        for _ in range(n):
            pooled += clusters[rng.randrange(n)]
        boots.append(statistics.median(pooled))
    boots.sort()
    per_cluster = [statistics.median(c) for c in clusters]
    mean = statistics.fmean(per_cluster)
    sd = statistics.stdev(per_cluster) if n > 1 else math.nan
    half = T_CRIT_95.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else math.nan
    return {
        "median": statistics.median([v for c in clusters for v in c]),
        "ci95_lo": boots[int(0.025 * reps)],
        "ci95_hi": boots[int(0.975 * reps) - 1],
        "clusters": n,
        "cluster_medians": per_cluster,
        "cluster_median_mean": mean,
        "cluster_median_sd": sd,
        "t_ci95_lo": mean - half,
        "t_ci95_hi": mean + half,
    }


def paired_deltas(a: list[dict], b: list[dict], field: str) -> list[float]:
    """b minus a, round by round, over the rounds both legs share."""
    return [b[i][field] - a[i][field] for i in range(min(len(a), len(b)))]


def contrast(block: list[list[dict]], weights: tuple[float, ...],
             field: str) -> list[float]:
    """Apply per-leg weights to one `base ab ab base` block, round by round."""
    width = min(len(leg) for leg in block)
    return [sum(w * leg[i][field] for w, leg in zip(weights, block))
            for i in range(width)]


# Legs sit at positions 0..3 of the palindrome. The treatment contrast loads
# the two inner `ab` legs and cancels linear drift. The cubic contrast carries
# zero treatment loading and zero drift loading, so it is what this estimator
# returns on unchanged code; dividing by sqrt(20) gives it the same variance as
# the treatment contrast under independent per-leg noise.
EFFECT_WEIGHTS = (-0.5, 0.5, 0.5, -0.5)
NULL_WEIGHTS = tuple(w / math.sqrt(20.0) for w in (1.0, -3.0, 3.0, -1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump-segments", action="store_true")
    args = ap.parse_args()

    root = Path(args.session)
    with (root / "legs.tsv").open() as handle:
        legs = list(csv.DictReader(handle, delimiter="\t"))

    rounds_by_leg: dict[int, list[dict]] = {}
    for row in legs:
        leg = int(row["leg"])
        path = root / f"leg{leg:02d}-{row['arm']}" / "rounds.txt"
        if not path.exists():
            raise SystemExit(f"missing {path}; rerun with MLX_QWEN_MTP_TRACE=1")
        segments = parse_rounds(path)
        if args.dump_segments:
            print(f"leg {leg:2d} {row['arm']:<4} segments="
                  f"{[len(s) for s in segments]}")
        rounds_by_leg[leg] = timed_segment(segments)

    keys = {leg: sequence_key(r) for leg, r in rounds_by_leg.items()}
    distinct = {}
    for leg, key in keys.items():
        distinct.setdefault(key, []).append(leg)

    report: dict = {
        "session": str(root),
        "legs": len(legs),
        "distinct_round_sequences": len(distinct),
        "sequence_groups": [
            {"legs": sorted(v), "rounds": len(k)} for k, v in distinct.items()
        ],
        "rounds_per_leg": {str(leg): len(r) for leg, r in rounds_by_leg.items()},
    }

    arm_of = {int(r["leg"]): r["arm"] for r in legs}
    per_arm: dict[str, dict] = {}
    for arm in sorted(set(arm_of.values())):
        sub = [rounds_by_leg[leg] for leg, a in arm_of.items() if a == arm]
        rounds = [r for leg in sub for r in leg]
        emitted = sum(1 + r["acc"] for r in rounds)
        per_arm[arm] = {
            "legs": len(sub),
            "rounds_total": len(rounds),
            "rounds_per_leg_mean": statistics.fmean(len(s) for s in sub),
            "round_us_median": statistics.median(r["round_us"] for r in rounds),
            "round_us_mean": statistics.fmean(r["round_us"] for r in rounds),
            "mean_draft_len": statistics.fmean(r["d"] for r in rounds),
            "accepted_per_round": statistics.fmean(r["acc"] for r in rounds),
            "rows_per_token": sum(r["d"] + 1 for r in rounds) / emitted
            if emitted else math.nan,
        }
    report["per_arm"] = per_arm

    # ABBA blocks of four: base ab ab base.
    order = sorted(arm_of)
    effect: list[float] = []
    null: list[float] = []
    ends: list[float] = []
    effect_blocks: list[list[float]] = []
    null_blocks: list[list[float]] = []
    per_block: list[dict] = []
    for start in range(0, len(order) - 3, 4):
        block = order[start:start + 4]
        arms = [arm_of[leg] for leg in block]
        if arms[0] != arms[3] or arms[1] != arms[2] or arms[0] == arms[1]:
            continue
        legs4 = [rounds_by_leg[leg] for leg in block]
        block_effect = contrast(legs4, EFFECT_WEIGHTS, "round_us")
        block_null = contrast(legs4, NULL_WEIGHTS, "round_us")
        effect += block_effect
        null += block_null
        effect_blocks.append(block_effect)
        null_blocks.append(block_null)
        a1, _, _, a2 = block
        ends += paired_deltas(rounds_by_leg[a1], rounds_by_leg[a2], "round_us")
        per_block.append({
            "legs": block,
            "effect_median_us_per_round": statistics.median(block_effect),
            "null_median_us_per_round": statistics.median(block_null),
            "effect_iqr_us_per_round": (
                statistics.quantiles(block_effect, n=4)[2]
                - statistics.quantiles(block_effect, n=4)[0]),
        })

    report["paired_effect_us_per_round"] = median_ci(effect)
    report["session_null_us_per_round"] = median_ci(null)
    report["end_base_drift_us_per_round"] = median_ci(ends)
    report["paired_effect_clustered"] = median_ci_clustered(effect_blocks)
    report["session_null_clustered"] = median_ci_clustered(null_blocks)
    report["per_block"] = per_block

    # If rounds were independent replicates, the scatter of the block medians
    # would match the standard error that within-block scatter predicts. A
    # ratio far above one means a per-leg offset dominates, and the
    # round-level interval is then far too narrow to use for a decision.
    within = []
    for block_effect in effect_blocks:
        q1, _, q3 = statistics.quantiles(block_effect, n=4)
        sd = (q3 - q1) / 1.349
        within.append(1.253 * sd / math.sqrt(len(block_effect)))
    between = report["paired_effect_clustered"]["cluster_median_sd"]
    predicted = statistics.fmean(within)
    report["cluster_variance_check"] = {
        "between_block_sd_us_per_round": between,
        "within_block_predicted_sem_us_per_round": predicted,
        "ratio": between / predicted if predicted else math.nan,
        "rounds_are_independent_replicates": bool(
            predicted and between / predicted < 2.0),
    }
    report["contrast_weights"] = {
        "effect": list(EFFECT_WEIGHTS),
        "null": list(NULL_WEIGHTS),
        "note": "effect and null both carry zero linear-drift loading; the "
                "null carries zero treatment loading and equal variance",
    }

    # The unchanged arm is named `base`; alphabetical order would pick `ab`
    # and silently flip the sign of every leg-total figure.
    all_arms = set(arm_of.values())
    base_arm = "base" if "base" in all_arms else min(all_arms)
    tokens_per_round = {
        arm: 1 + per_arm[arm]["accepted_per_round"] for arm in per_arm
    }
    tpr = statistics.fmean(tokens_per_round.values())
    report["tokens_per_round"] = tpr
    for name in ("paired_effect_us_per_round", "session_null_us_per_round",
                 "end_base_drift_us_per_round", "paired_effect_clustered",
                 "session_null_clustered"):
        block = report[name]
        for key in ("median", "ci95_lo", "ci95_hi", "cluster_median_mean",
                    "cluster_median_sd", "t_ci95_lo", "t_ci95_hi"):
            if key in block:
                block[f"{key}_us_per_token"] = (
                    block[key] / tpr if tpr else math.nan)

    # The traced round total must track the harness leg total, or `round_us`
    # is not measuring the window the primary metric measures.
    leg_seconds = {int(r["leg"]): float(r["mtp_s_per_tok"]) for r in legs}
    traced = [sum(r["round_us"] for r in rounds_by_leg[leg]) * 1e-6
              for leg in sorted(leg_seconds)]
    harness = [leg_seconds[leg] for leg in sorted(leg_seconds)]
    report["trace_covers_leg"] = {
        "traced_round_seconds_mean": statistics.fmean(traced),
        "harness_leg_seconds_mean": statistics.fmean(
            v * 512 for v in harness),
        "pearson_r": statistics.correlation(traced, harness)
        if len(traced) > 2 else math.nan,
    }

    # Leg totals, the reading the paired median is meant to replace.
    def leg_total(arm: str) -> float:
        return statistics.fmean(
            float(r["mtp_s_per_tok"]) for r in legs if r["arm"] == arm)

    arms = sorted(set(arm_of.values()))
    other = [a for a in arms if a != base_arm]
    if other:
        naive = (leg_total(other[0]) - leg_total(base_arm)) * 1e6
        report["leg_total_effect_us_per_token"] = naive
        report["leg_total_baseline_s_per_token"] = leg_total(base_arm)
        median_tok = report["paired_effect_us_per_round"]["median_us_per_token"]
        report["leg_total_over_paired_ratio"] = (
            naive / median_tok if median_tok else math.nan)
        report["paired_effect_pct_of_candidate"] = (
            100.0 * median_tok / (leg_total(base_arm) * 1e6)
            if leg_total(base_arm) else math.nan)

    def outside(effect_key: str, null_key: str) -> bool:
        eff, nul = report[effect_key], report[null_key]
        return bool(eff["ci95_hi"] < nul["ci95_lo"]
                    or eff["ci95_lo"] > nul["ci95_hi"])

    report["effect_outside_null_round_level"] = outside(
        "paired_effect_us_per_round", "session_null_us_per_round")
    report["effect_outside_null_clustered"] = outside(
        "paired_effect_clustered", "session_null_clustered")
    # The clustered reading is the decision statistic; the round-level one
    # assumes rounds are independent replicates, which they are not.
    report["effect_outside_null"] = report["effect_outside_null_clustered"]
    if report.get("leg_total_baseline_s_per_token"):
        base_us = report["leg_total_baseline_s_per_token"] * 1e6
        report["paired_effect_clustered_pct_of_candidate"] = (
            100.0 * report["paired_effect_clustered"]["median_us_per_token"]
            / base_us)

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
