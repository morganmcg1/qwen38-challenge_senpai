#!/usr/bin/env python3
"""Reduce the E69 rung-1 cell legs to one decision table per NA.

Each arm contributes two legs per rep at maximum separation in the palindrome,
so the same-arm leg pair is a null at the largest drift the session can carry.
The reported effect is the median over reps of the per-rep arm contrast, which
removes any drift shared by the two arms inside one rep.

  python3 research/e69_rung1_analyze.py [--dir research/e69-artifacts]
                                        [--out research/e69-artifacts/rung1-summary.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

BASELINE = "plain"


def per_rep_medians(legs: list[dict]) -> dict[str, dict[int, float]]:
    """arm -> rep -> median seconds per dispatch over that arm's legs in the rep."""
    bucket: dict[str, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for leg in legs:
        bucket[leg["arm"]][leg["rep"]].append(leg["seconds_per_dispatch"])
    return {arm: {rep: statistics.median(values) for rep, values in reps.items()}
            for arm, reps in bucket.items()}


def same_arm_null(legs: list[dict], arm: str) -> float:
    """Largest-separation same-arm split, as a percentage. The noise floor."""
    by_rep: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)
    for leg in legs:
        if leg["arm"] == arm:
            by_rep[leg["rep"]].append((leg["position"], leg["seconds_per_dispatch"]))
    ratios = []
    for pairs in by_rep.values():
        if len(pairs) != 2:
            continue
        pairs.sort()
        ratios.append(100.0 * (pairs[1][1] / pairs[0][1] - 1.0))
    return statistics.median(ratios) if ratios else float("nan")


def summarize(report: dict) -> dict:
    na = report["na"]
    out = {"na": na, "device": report["device"], "reps": report["reps"],
           "order": report["order"],
           "entry_gpu_temp_c": report["entry_gpu_temp_c"],
           "exit_gpu_temp_c": report["exit_gpu_temp_c"],
           "shapes": []}
    for shape in report["shapes"]:
        medians = per_rep_medians(shape["legs"])
        base = medians[BASELINE]
        arms = {}
        for arm, values in medians.items():
            reps = sorted(set(values) & set(base))
            contrast = [100.0 * (values[rep] / base[rep] - 1.0) for rep in reps]
            absolute = [values[rep] for rep in reps]
            arms[arm] = {
                "median_seconds_per_dispatch": statistics.median(absolute),
                "median_pct_vs_plain": statistics.median(contrast),
                "min_pct_vs_plain": min(contrast),
                "max_pct_vs_plain": max(contrast),
                "sd_pct_vs_plain": (statistics.stdev(contrast)
                                    if len(contrast) > 1 else 0.0),
                "reps": len(contrast),
            }
        gbps = [leg["gbps"] for leg in shape["legs"] if leg["arm"] == BASELINE]
        out["shapes"].append({
            "shape": shape["shape"],
            "k": shape["k"],
            "n": shape["n"],
            "inner": shape["inner"],
            "entry_gpu_temp_c": shape["entry_gpu_temp_c"],
            "exit_gpu_temp_c": shape["exit_gpu_temp_c"],
            "parity_differing_vs_plain": shape["parity_differing_vs_plain"],
            "plain_median_gbps": statistics.median(gbps),
            "same_arm_null_pct": same_arm_null(shape["legs"], BASELINE),
            "arms": arms,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path,
                    default=pathlib.Path("research/e69-artifacts"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("research/e69-artifacts/rung1-summary.json"))
    args = ap.parse_args()

    summaries = []
    for path in sorted(args.dir.glob("rung1-na*.json")):
        summaries.append(summarize(json.loads(path.read_text())))
    summaries.sort(key=lambda s: s["na"])

    arms: list[str] = []
    for summary in summaries:
        for shape in summary["shapes"]:
            for arm in shape["arms"]:
                if arm not in arms:
                    arms.append(arm)
    others = [arm for arm in arms if arm != BASELINE]

    parity_clean = True
    failures: list[str] = []
    for summary in summaries:
        for shape in summary["shapes"]:
            for arm, differing in shape["parity_differing_vs_plain"].items():
                if differing:
                    failures.append(
                        f"na={summary['na']} {shape['shape']} {arm} "
                        f"differing={differing} of {summary['na'] * shape['n']}")
    for summary in summaries:
        print(f"\n### NA={summary['na']}  device={summary['device']}  "
              f"reps={summary['reps']}  "
              f"entry={summary['entry_gpu_temp_c']:.1f}C "
              f"exit={summary['exit_gpu_temp_c']:.1f}C")
        print("| shape | plain us | plain GB/s | null % | "
              + " | ".join(f"{arm} %" for arm in others) + " |")
        print("|" + "---|" * (len(others) + 4))
        for shape in summary["shapes"]:
            if any(count != 0 for count
                   in shape["parity_differing_vs_plain"].values()):
                parity_clean = False
            cells = []
            for arm in others:
                stats = shape["arms"].get(arm)
                cells.append("-" if stats is None
                             else f"{stats['median_pct_vs_plain']:+.2f}")
            print(f"| {shape['shape']} | "
                  f"{shape['arms'][BASELINE]['median_seconds_per_dispatch']*1e6:.1f} | "
                  f"{shape['plain_median_gbps']:.1f} | "
                  f"{shape['same_arm_null_pct']:+.2f} | "
                  + " | ".join(cells) + " |")

    print("\n### Weighted over every shape and NA (median of per-cell medians)")
    print("| arm | median % vs plain | worst cell % | best cell % | cells |")
    print("|---|---|---|---|---|")
    for arm in others:
        cells = [shape["arms"][arm]["median_pct_vs_plain"]
                 for summary in summaries for shape in summary["shapes"]
                 if arm in shape["arms"]]
        if not cells:
            continue
        print(f"| {arm} | {statistics.median(cells):+.2f} | "
              f"{max(cells):+.2f} | {min(cells):+.2f} | {len(cells)} |")
    print(f"\nparity_bit_identical_everywhere={parity_clean}")
    for line in failures:
        print(f"  PARITY FAILURE {line}")

    args.out.write_text(json.dumps(
        {"summaries": summaries,
         "parity_bit_identical_everywhere": parity_clean,
         "parity_failures": failures}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
