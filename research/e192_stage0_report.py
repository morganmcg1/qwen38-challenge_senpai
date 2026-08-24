#!/usr/bin/env python3
"""Reduce the E192 stage-0 legs to the account verdicts.

    usage: python3 research/e192_stage0_report.py TAG [TAG ...] [--json OUT]

Each leg alternates its arm by round parity inside one process, so the
contrast is a paired difference between adjacent rounds. Only rounds that
drafted (`d > 0`), accepted every draft (`acc == d`) and share a draft depth
with their partner are paired, because the commit phase this experiment
measures exists only on the full-acceptance branch.

Two verdicts come out of it:

  in-flight vs allocator   `clear_release_cpu_us` against `clear_release_us`
                           in the unbarriered rounds, and `clear_release_us`
                           with and without the barrier.
  relocation vs removal    `round_us` and `host_thread_cpu_ns` with and
                           without the replay tape.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

FIELD_RE = re.compile(r"(\w+)=(-?\d+)")

# Every per-round field this report reads. A leg whose worker predates the
# stage-0 patch simply lacks the e192 keys, and is reported as such.
INT_FIELDS = (
    "round", "d", "acc",
    "commit_us", "round_us", "eval_wall_us", "verify_build_us",
    "readout_us", "upkeep_us", "draft_build_us", "host_thread_cpu_ns",
    "clear_release_us", "clear_release_cpu_us", "clear_release_count",
    "clear_barrier_us", "e192_barrier_arm", "e192_tape_suppressed_arm",
)

PAIRED_FIELDS = (
    "round_us", "host_thread_cpu_ns", "commit_us", "clear_release_us",
    "clear_release_cpu_us", "clear_barrier_us", "eval_wall_us",
    "verify_build_us", "readout_us", "upkeep_us", "draft_build_us",
)


def read_rounds(tag):
    path = Path("research/out") / tag / "trace.txt"
    rounds = []
    if not path.exists():
        return rounds
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        fields = dict(FIELD_RE.findall(line))
        row = {}
        for key in INT_FIELDS:
            if key in fields:
                row[key] = int(fields[key])
        rounds.append(row)
    return rounds


def full_acceptance(rounds):
    return [r for r in rounds
            if r.get("d", 0) > 0 and r.get("acc") == r.get("d")]


def summarize(rows, arm_key):
    """Median of every field, split by the arm witness."""
    out = {}
    for arm in (0, 1):
        subset = [r for r in rows if r.get(arm_key) == arm]
        entry = {"n": len(subset)}
        if subset:
            for field in set().union(*(r.keys() for r in subset)):
                values = [r[field] for r in subset if field in r]
                if values:
                    entry[f"{field}_median"] = statistics.median(values)
        out[f"arm{arm}"] = entry
    return out


def paired(rows, arm_key):
    """Pair adjacent rounds that differ only in the arm and share a depth.

    The arm alternates by round parity, so an arm-1 round and the arm-0 round
    immediately before or after it form a pair. Requiring equal `d` keeps a
    schedule change out of the contrast.
    """
    by_round = {r["round"]: r for r in rows}
    pairs = []
    for r in rows:
        if r.get(arm_key) != 1:
            continue
        for neighbour in (r["round"] - 1, r["round"] + 1):
            other = by_round.get(neighbour)
            if (other is not None
                    and other.get(arm_key) == 0
                    and other.get("d") == r.get("d")):
                pairs.append((other, r))
                break

    result = {"pair_count": len(pairs)}
    for field in PAIRED_FIELDS:
        diffs = [b[field] - a[field]
                 for a, b in pairs if field in a and field in b]
        if len(diffs) < 2:
            continue
        mean = statistics.fmean(diffs)
        sem = statistics.stdev(diffs) / (len(diffs) ** 0.5)
        result[field] = {
            "n": len(diffs),
            "mean_arm1_minus_arm0": mean,
            "two_sigma": 2 * sem,
            "excludes_zero": abs(mean) > 2 * sem,
            "median_arm1_minus_arm0": statistics.median(diffs),
        }
    return result


def leg_report(tag):
    rounds = read_rounds(tag)
    rows = full_acceptance(rounds)
    meta_path = Path("research/out") / tag / "meta.txt"
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key] = value
    arm_key = ("e192_barrier_arm" if any("e192_barrier_arm" in r for r in rows)
               else "e192_tape_suppressed_arm")
    report = {
        "tag": tag,
        "arm_key": arm_key,
        "trace_rounds": len(rounds),
        "full_acceptance_rounds": len(rows),
        "drafting_rounds": len([r for r in rounds if r.get("d", 0) > 0]),
        "rejecting_rounds": len(
            [r for r in rounds if r.get("d", 0) > 0 and r.get("acc") != r.get("d")]),
        "arm_env": meta.get("e192_arm_env", ""),
        "base_sha": meta.get("base_sha", ""),
        "worker_sha256": meta.get("worker_sha256", ""),
        "host": meta.get("host", ""),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c", ""),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c", ""),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate", ""),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing", ""),
        "tokens": meta.get("tokens", ""),
        "per_arm": summarize(rows, arm_key),
        "paired": paired(rows, arm_key),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    reports = [leg_report(tag) for tag in args.tags]
    for report in reports:
        print(f"== {report['tag']}  ({report['arm_env']})")
        print(f"   rounds traced {report['trace_rounds']}, "
              f"drafting {report['drafting_rounds']}, "
              f"full-acceptance {report['full_acceptance_rounds']}, "
              f"rejecting {report['rejecting_rounds']}")
        print(f"   temps {report['gpu_temp_entry_c']} -> "
              f"{report['gpu_temp_exit_c']} C, "
              f"gate_qualified={report['gate_qualified_for_timing']}")
        for arm, entry in report["per_arm"].items():
            keys = [k for k in sorted(entry) if k.endswith("_median")]
            print(f"   {arm} n={entry['n']}")
            for key in keys:
                print(f"      {key:34s} {entry[key]}")
        print(f"   paired pairs={report['paired'].get('pair_count', 0)}")
        for field in PAIRED_FIELDS:
            stats = report["paired"].get(field)
            if stats:
                print(f"      {field:22s} "
                      f"delta {stats['mean_arm1_minus_arm0']:+10.1f} "
                      f"2sigma {stats['two_sigma']:8.1f} "
                      f"n={stats['n']} "
                      f"{'SIGNIFICANT' if stats['excludes_zero'] else 'null'}")
        print()

    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(json.dumps(reports, indent=2) + "\n")
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
