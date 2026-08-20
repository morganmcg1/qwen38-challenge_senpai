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


def paired_deltas(a: list[dict], b: list[dict], field: str) -> list[float]:
    """b minus a, round by round, over the rounds both legs share."""
    return [b[i][field] - a[i][field] for i in range(min(len(a), len(b)))]


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
    for start in range(0, len(order) - 3, 4):
        block = order[start:start + 4]
        arms = [arm_of[leg] for leg in block]
        if arms[0] != arms[3] or arms[1] != arms[2] or arms[0] == arms[1]:
            continue
        a1, b1, b2, a2 = block
        effect += paired_deltas(rounds_by_leg[a1], rounds_by_leg[b1], "round_us")
        effect += paired_deltas(rounds_by_leg[a2], rounds_by_leg[b2], "round_us")
        null += paired_deltas(rounds_by_leg[a1], rounds_by_leg[a2], "round_us")

    report["paired_effect_us_per_round"] = median_ci(effect)
    report["session_null_us_per_round"] = median_ci(null)

    base_arm = min(set(arm_of.values()))
    tokens_per_round = {
        arm: 1 + per_arm[arm]["accepted_per_round"] for arm in per_arm
    }
    tpr = statistics.fmean(tokens_per_round.values())
    report["tokens_per_round"] = tpr
    for name in ("paired_effect", "session_null"):
        block = report[f"{name}_us_per_round"]
        for key in ("median", "ci95_lo", "ci95_hi"):
            block[f"{key}_us_per_token"] = block[key] / tpr if tpr else math.nan

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

    lo = report["paired_effect_us_per_round"]["ci95_lo"]
    hi = report["paired_effect_us_per_round"]["ci95_hi"]
    nlo = report["session_null_us_per_round"]["ci95_lo"]
    nhi = report["session_null_us_per_round"]["ci95_hi"]
    report["effect_outside_null"] = bool(hi < nlo or lo > nhi)

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
