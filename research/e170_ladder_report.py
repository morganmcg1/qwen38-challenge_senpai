#!/usr/bin/env python3
"""Contrast the gated fixed-depth ladder arms and report every required gate.

    usage: research/e170_ladder_report.py research/out/e170 [--json OUT]

Legs come from `research/e170_depth_ladder.sh`, which runs the real 40 C gate,
never enables the per-round trace, and asserts the worker digest before and
after every leg. Those three facts are what make these wall times admissible,
so this report prints them rather than assuming them.

The primary quantity is end-to-end `mtp_seconds_per_token`, as instructed:
the arms propose different numbers of rows on purpose, so no per-row rate is
comparable between them and the accept ledgers are expected to differ.

The per-leg noise floor is measured here from the replicates of each arm
instead of being quoted from another host. Arms are compared on the median of
their legs, and the reported interval is the half-range of the arm's own legs,
which is the honest resolution of a four-leg arm.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

GROUPS = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}

REQUIRED_TRUE = ("all_tokens_matched", "public_drift_tripwire_passed")


def read_meta(directory: pathlib.Path) -> dict[str, str]:
    meta = directory / "meta.txt"
    if not meta.exists():
        return {}
    out = {}
    for line in meta.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def load_leg(directory: pathlib.Path) -> dict | None:
    score = directory / "score.json"
    timed = directory / "reports" / "04-mtp-timed.json"
    if not score.exists() or not timed.exists():
        return None
    metrics = json.loads(score.read_text())["metrics"]
    report = json.loads(timed.read_text())
    meta = read_meta(directory)
    widths = [1 + w for w in report["effective_draft_lengths"]]
    counts: dict[int, int] = {}
    for width in widths:
        counts[width] = counts.get(width, 0) + 1
    multi = sum(n for m, n in counts.items() if GROUPS.get(m, 1) >= 2)
    return {
        "leg": int(meta.get("e170_leg", 0)),
        "arm": meta.get("e170_arm", directory.parent.name),
        "path": str(directory),
        "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "serial_seconds_per_token": metrics["serial_seconds_per_token"],
        "mtp_decode_speedup": metrics["mtp_decode_speedup"],
        "decode_tokens": metrics["decode_tokens"],
        "accepted_draft_rate": metrics["accepted_draft_rate"],
        "accepted_draft_total": report["accepted_draft_total"],
        "declared_rows_total": report["declared_rows_total"],
        "emitted_token_total": report["emitted_token_total"],
        "rounds": len(widths),
        "mean_m": statistics.fmean(widths) if widths else 0.0,
        "histogram": dict(sorted(counts.items())),
        "multi_pass_fraction": multi / len(widths) if widths else 0.0,
        "all_tokens_matched": metrics["all_tokens_matched"],
        "residual_divergence_count": metrics["residual_divergence_count"],
        "public_drift_tripwire_passed": metrics["public_drift_tripwire_passed"],
        "gpu_temp_entry": meta.get("gpu_temp_entry", "unavailable"),
        "gpu_temp_exit": meta.get("gpu_temp_exit", "unavailable"),
        "worker_digest_stable": meta.get("worker_digest_stable", "unknown"),
        "worker_sha256_before": meta.get("worker_sha256_before", ""),
        "cool_gate_passed_real_gate": meta.get(
            "cool_gate_passed_real_gate", "unknown"
        ),
        "gate_qualified_for_timing": meta.get(
            "gate_qualified_for_timing", "unknown"
        ),
    }


def gate_failures(leg: dict) -> list[str]:
    problems = []
    for field in REQUIRED_TRUE:
        if leg[field] is not True:
            problems.append(f"{field}={leg[field]}")
    if leg["residual_divergence_count"] != 0:
        problems.append(f"residual_divergence_count={leg['residual_divergence_count']}")
    if leg["emitted_token_total"] != leg["decode_tokens"]:
        problems.append(
            f"emitted {leg['emitted_token_total']} != {leg['decode_tokens']}"
        )
    # Row-ledger closure: every declared row is one primary plus the drafts
    # actually proposed, so the ledger must close against the round record.
    expected_rows = leg["rounds"] + sum(
        (m - 1) * n for m, n in leg["histogram"].items()
    )
    if expected_rows != leg["declared_rows_total"]:
        problems.append(
            f"row ledger {leg['declared_rows_total']} != {expected_rows}"
        )
    if leg["worker_digest_stable"] != "true":
        problems.append(f"worker_digest_stable={leg['worker_digest_stable']}")
    if leg["gate_qualified_for_timing"] != "true":
        problems.append(
            f"gate_qualified_for_timing={leg['gate_qualified_for_timing']}"
        )
    return problems


def summarise(legs: list[dict]) -> dict:
    values = [leg["mtp_seconds_per_token"] for leg in legs]
    median = statistics.median(values)
    half_range = (max(values) - min(values)) / 2.0
    return {
        "legs": len(legs),
        "median_ms_per_token": median * 1000.0,
        "min_ms_per_token": min(values) * 1000.0,
        "max_ms_per_token": max(values) * 1000.0,
        "half_range_pct": (half_range / median) * 100.0 if median else 0.0,
        "sd_pct": (
            (statistics.stdev(values) / median) * 100.0
            if len(values) > 1 and median
            else 0.0
        ),
        "mean_m": statistics.fmean([leg["mean_m"] for leg in legs]),
        "multi_pass_fraction": statistics.fmean(
            [leg["multi_pass_fraction"] for leg in legs]
        ),
        "accepted_draft_rate": statistics.fmean(
            [leg["accepted_draft_rate"] for leg in legs]
        ),
        "accepted_draft_total": statistics.fmean(
            [leg["accepted_draft_total"] for leg in legs]
        ),
        "declared_rows_total": statistics.fmean(
            [leg["declared_rows_total"] for leg in legs]
        ),
        "rounds": statistics.fmean([leg["rounds"] for leg in legs]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--baseline", default="adapt")
    parser.add_argument("--json")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    legs = []
    for arm_dir in sorted(root.iterdir()):
        if not arm_dir.is_dir():
            continue
        for leg_dir in sorted(arm_dir.iterdir()):
            leg = load_leg(leg_dir)
            if leg is not None:
                legs.append(leg)
    if not legs:
        raise SystemExit(f"{root}: no complete legs")
    legs.sort(key=lambda leg: leg["leg"])

    print("=== legs in run order ===")
    print(
        f"{'leg':>4}{'arm':<8}{'entryC':>8}{'exitC':>8}{'meanM':>8}"
        f"{'2pass':>7}{'rounds':>8}{'ms/token':>10}{'  gates':>9}"
    )
    all_problems = {}
    for leg in legs:
        problems = gate_failures(leg)
        all_problems[leg["leg"]] = problems
        entry = leg["gpu_temp_entry"]
        exit_temp = leg["gpu_temp_exit"]
        entry_s = f"{float(entry):.1f}" if entry != "unavailable" else "n/a"
        exit_s = f"{float(exit_temp):.1f}" if exit_temp != "unavailable" else "n/a"
        print(
            f"{leg['leg']:>4}{leg['arm']:<8}{entry_s:>8}{exit_s:>8}"
            f"{leg['mean_m']:>8.3f}{leg['multi_pass_fraction']:>7.3f}"
            f"{leg['rounds']:>8}"
            f"{leg['mtp_seconds_per_token'] * 1000:>10.3f}"
            f"{('OK' if not problems else 'FAIL'):>9}"
        )
    for leg_id, problems in all_problems.items():
        if problems:
            print(f"  leg {leg_id} gate failures: {'; '.join(problems)}")

    arms: dict[str, list[dict]] = {}
    for leg in legs:
        arms.setdefault(leg["arm"], []).append(leg)
    summary = {arm: summarise(rows) for arm, rows in arms.items()}

    print("\n=== arms, primary metric is end-to-end mtp ms/token ===")
    print(
        f"{'arm':<8}{'legs':>5}{'median':>10}{'halfrange':>11}{'sd%':>7}"
        f"{'meanM':>8}{'2pass':>7}{'acc rate':>10}{'rows':>8}"
    )
    for arm in sorted(summary):
        block = summary[arm]
        print(
            f"{arm:<8}{block['legs']:>5}{block['median_ms_per_token']:>10.3f}"
            f"{block['half_range_pct']:>10.3f}%{block['sd_pct']:>6.3f}%"
            f"{block['mean_m']:>8.3f}{block['multi_pass_fraction']:>7.3f}"
            f"{block['accepted_draft_rate']:>10.4f}"
            f"{block['declared_rows_total']:>8.0f}"
        )

    contrasts = {}
    if args.baseline in summary:
        base = summary[args.baseline]["median_ms_per_token"]
        print(f"\n=== contrast against '{args.baseline}' ===")
        for arm in sorted(summary):
            if arm == args.baseline:
                continue
            value = summary[arm]["median_ms_per_token"]
            change = (value / base - 1.0) * 100.0
            noise = summary[arm]["half_range_pct"] + summary[
                args.baseline
            ]["half_range_pct"]
            contrasts[arm] = {
                "change_pct": change,
                "combined_half_range_pct": noise,
                "resolved": abs(change) > noise,
            }
            print(
                f"  {arm:<6}{change:+8.2f}%   combined half-range "
                f"{noise:.3f}%   resolved {abs(change) > noise}"
            )
        print("  Negative means the arm is faster than the shipped controller.")

    report = {
        "legs": legs,
        "arms": summary,
        "contrasts": contrasts,
        "gate_failures": {k: v for k, v in all_problems.items() if v},
    }
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
