#!/usr/bin/env python3
"""Reduce the E205 stage-A legs to a rejection-round premium.

    usage: python3 research/e205_report.py TAG [TAG ...] [--json OUT]

Each leg alternates its arm by round parity inside one process, so the
contrast is a paired difference between adjacent rounds (RULE 388). A round is
classified from its own trace witnesses:

    serial   d == 0                                 (no verify window)
    trunc    e205_trunc_applied > 0                 (forced rejection)
    nat      acc < d and e205_trunc_applied == 0    (genuine rejection)
    full     acc == d and e205_trunc_applied == 0   (full acceptance)

Only `trunc` rounds paired with an adjacent `full` round OF THE SAME DEPTH are
priced, because depth sets the verify width and would otherwise dominate the
difference.

The decision statistic is `round_us` (RULE 394). Every other field attributes
the cost: `repair_*` says what the repair site itself spent, and the remaining
phases say where displaced work reappeared when the site and the endpoint
disagree (the FINDING 528 pattern).
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

FIELD_RE = re.compile(r"(\w+)=(-?\d+)")

INT_FIELDS = (
    "round", "d", "acc", "acc_true",
    "e205_trunc_req", "e205_trunc_applied", "repair_path",
    "repair_us", "repair_cpu_us", "repair_replay_us", "repair_trim_us",
    "repair_prefetch_us", "repair_generic_us",
    "clear_release_us", "clear_release_cpu_us",
    "draft_build_us", "d_flush_us", "d_head1_us", "d_submit1_us",
    "d_chain_us", "d_submit2_us",
    "verify_build_us", "eval_wall_us", "readout_us", "commit_us",
    "upkeep_us", "round_us", "host_thread_cpu_ns", "serial_body",
)

PAIRED_FIELDS = (
    "round_us", "host_thread_cpu_ns", "commit_us", "repair_us",
    "repair_cpu_us", "repair_replay_us", "repair_trim_us",
    "repair_prefetch_us", "repair_generic_us", "clear_release_us",
    "clear_release_cpu_us", "verify_build_us", "eval_wall_us", "readout_us",
    "upkeep_us", "draft_build_us", "d_flush_us",
)

# FINDING 518 composition bar and FINDING 482 minimum useful effect.
BAR_US = 200.0
MUE_US = 567.0


def read_rounds(tag):
    path = Path("research/out") / tag / "trace.txt"
    rounds = []
    if not path.exists():
        return rounds
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        fields = dict(FIELD_RE.findall(line))
        row = {key: int(fields[key]) for key in INT_FIELDS if key in fields}
        # The repair sub-phase timers are last-value counters inside the
        # session, so a round that never enters the repair site still prints
        # the previous repair's split. The site total repair_us IS reset per
        # round and reads zero there, which proves the sub-phases must also be
        # zero on those rounds; hold them to zero so a paired delta cannot
        # subtract a stale split.
        if row.get("repair_path", 0) == 0:
            for key in ("repair_replay_us", "repair_trim_us",
                        "repair_prefetch_us", "repair_generic_us"):
                row[key] = 0
        rounds.append(row)
    return rounds


def classify(row):
    if row.get("d", 0) == 0:
        return "serial"
    if row.get("e205_trunc_applied", 0) > 0:
        return "trunc"
    if row.get("acc") != row.get("d"):
        return "nat"
    return "full"


def stats(diffs):
    mean = statistics.fmean(diffs)
    sem = statistics.stdev(diffs) / (len(diffs) ** 0.5) if len(diffs) > 1 else 0.0
    return {
        "n": len(diffs),
        "mean": mean,
        "two_sigma": 2 * sem,
        "excludes_zero": abs(mean) > 2 * sem > 0,
        "median": statistics.median(diffs),
    }


def pair_rounds(rows):
    """Pair each forced-rejection round with an adjacent full-acceptance round
    of the same depth. Parity alternation puts the partner at round +-1."""
    by_round = {r["round"]: r for r in rows}
    pairs = []
    for row in rows:
        if classify(row) != "trunc":
            continue
        for neighbour in (row["round"] - 1, row["round"] + 1):
            other = by_round.get(neighbour)
            if (other is not None and classify(other) == "full"
                    and other.get("d") == row.get("d")):
                pairs.append((other, row))
                break
    return pairs


def paired_report(pairs):
    out = {"pair_count": len(pairs)}
    for field in PAIRED_FIELDS:
        diffs = [b[field] - a[field]
                 for a, b in pairs if field in a and field in b]
        if len(diffs) >= 2:
            out[field] = stats(diffs)
    if pairs:
        base = [a["round_us"] for a, _ in pairs if "round_us" in a]
        out["baseline_round_us_mean"] = statistics.fmean(base) if base else 0.0
        out["depths"] = sorted({a["d"] for a, _ in pairs})
        out["applied"] = sorted({b["e205_trunc_applied"] for _, b in pairs})
        out["replayed_rows_mean"] = statistics.fmean(
            [b["acc"] + 1 for _, b in pairs])
    return out


def census(rows):
    kinds = {}
    for row in rows:
        kinds[classify(row)] = kinds.get(classify(row), 0) + 1
    return kinds


def hist(rows, key):
    out = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def read_meta(tag):
    path = Path("research/out") / tag / "meta.txt"
    meta = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key] = value
    return meta


def leg_report(tag):
    rounds = read_rounds(tag)
    meta = read_meta(tag)
    drafting = [r for r in rounds if r.get("d", 0) > 0]
    pairs = pair_rounds(drafting)
    by_j = {}
    for applied in sorted({b["e205_trunc_applied"] for _, b in pairs}):
        subset = [(a, b) for a, b in pairs
                  if b["e205_trunc_applied"] == applied]
        if len(subset) >= 2:
            by_j[str(applied)] = paired_report(subset)
    by_rows = {}
    for kappa in sorted({b["acc"] for _, b in pairs}):
        subset = [(a, b) for a, b in pairs if b["acc"] == kappa]
        if len(subset) >= 2:
            by_rows[str(kappa + 1)] = paired_report(subset)
    committed = sum(1 + r.get("acc", 0) for r in rounds)
    return {
        "tag": tag,
        "arm_env": meta.get("e205_arm_env", ""),
        "base_sha": meta.get("base_sha", ""),
        "worker_sha256": meta.get("worker_sha256", ""),
        "post_run_worker_sha256": meta.get("post_run_worker_sha256", ""),
        "host": meta.get("host", ""),
        "chip": meta.get("chip", ""),
        "tokens": meta.get("tokens", ""),
        "local_mode": meta.get("local_mode", ""),
        "head_dir": meta.get("head_dir", ""),
        "dirty_candidate_paths": meta.get("dirty_candidate_paths", ""),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c", ""),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c", ""),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate", ""),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing", ""),
        "official_or_ranked_score": meta.get("official_or_ranked_score", ""),
        "exit": meta.get("exit", ""),
        "trace_rounds": len(rounds),
        "census": census(rounds),
        "depth_hist": hist(drafting, "d"),
        "applied_hist": hist([r for r in drafting
                              if r.get("e205_trunc_applied", 0) > 0],
                             "e205_trunc_applied"),
        "repair_path_hist": hist([r for r in drafting
                                  if classify(r) in ("trunc", "nat")],
                                 "repair_path"),
        "committed_tokens_from_trace": committed,
        "paired_all": paired_report(pairs),
        "paired_by_applied_j": by_j,
        "paired_by_replayed_rows": by_rows,
    }


def render(report):
    lines = []
    lines.append(f"== {report['tag']}  ({report['arm_env']})")
    lines.append(
        f"   host {report['host']} {report['chip']}  tokens {report['tokens']}"
        f"  exit {report['exit']}")
    lines.append(
        f"   temps {report['gpu_temp_entry_c']} -> {report['gpu_temp_exit_c']} C"
        f"  cool_gate_passed_real_gate={report['cool_gate_passed_real_gate']}"
        f"  gate_qualified_for_timing={report['gate_qualified_for_timing']}"
        f"  official_or_ranked_score={report['official_or_ranked_score']}")
    lines.append(f"   census {report['census']}  depth {report['depth_hist']}")
    lines.append(
        f"   applied j {report['applied_hist']}"
        f"  repair_path {report['repair_path_hist']}")
    for label, block in [("ALL", report["paired_all"])] + [
            (f"j={key}", value)
            for key, value in report["paired_by_applied_j"].items()] + [
            (f"rows={key}", value)
            for key, value in report["paired_by_replayed_rows"].items()]:
        if not block.get("pair_count"):
            continue
        base = block.get("baseline_round_us_mean", 0.0)
        lines.append(
            f"   -- {label}: pairs={block['pair_count']} depths="
            f"{block.get('depths')} baseline_round={base:.0f} us")
        for field in PAIRED_FIELDS:
            entry = block.get(field)
            if not entry:
                continue
            scale = 1000.0 if field == "host_thread_cpu_ns" else 1.0
            mean = entry["mean"] / scale
            two_sigma = entry["two_sigma"] / scale
            verdict = "SIGNIFICANT" if entry["excludes_zero"] else "null"
            extra = ""
            if field == "round_us" and base:
                extra = f"  ({100 * entry['mean'] / base:+.3f}% of round)"
            lines.append(
                f"      {field:22s} delta {mean:+9.1f} us  2sigma {two_sigma:8.1f}"
                f"  n={entry['n']:3d}  {verdict}{extra}")
    return "\n".join(lines)


def verdicts(reports):
    """FINDING 494 verdict per j, decided at the round endpoint."""
    out = {}
    for report in reports:
        for key, block in report["paired_by_applied_j"].items():
            entry = block.get("round_us")
            site = block.get("repair_us")
            if not entry:
                continue
            base = block.get("baseline_round_us_mean", 0.0)
            lower = abs(entry["mean"]) - entry["two_sigma"]
            if not entry["excludes_zero"]:
                verdict = "no measurable endpoint premium"
            elif lower >= BAR_US:
                verdict = "endpoint premium above the 0.2 ms/round bar"
            else:
                verdict = "endpoint premium below the 0.2 ms/round bar"
            out[f"{report['tag']}:j{key}"] = {
                "endpoint_premium_us": entry["mean"],
                "endpoint_two_sigma_us": entry["two_sigma"],
                "endpoint_premium_pct_of_round": (
                    100 * entry["mean"] / base if base else None),
                "site_premium_us": site["mean"] if site else None,
                "site_two_sigma_us": site["two_sigma"] if site else None,
                "site_minus_endpoint_us": (
                    site["mean"] - entry["mean"] if site else None),
                "above_mue": abs(entry["mean"]) > MUE_US,
                "verdict": verdict,
            }
    return out


def premium_by_replayed_rows(reports):
    """Pool the endpoint premium across legs by the number of replayed rows.

    A rejecting round replays ``kappa + 1`` target rows, where ``kappa`` is the
    accepted count, so this is the index the stage-B premium model needs. Each
    point is weighted by its pair count.
    """
    buckets = {}
    for report in reports:
        for rows, block in report["paired_by_replayed_rows"].items():
            entry = block.get("round_us")
            if not entry:
                continue
            bucket = buckets.setdefault(int(rows), {"n": 0, "sum": 0.0})
            bucket["n"] += entry["n"]
            bucket["sum"] += entry["mean"] * entry["n"]
    points = [{"replayed_rows": rows,
               "endpoint_premium_ms": buckets[rows]["sum"] / buckets[rows]["n"]
                                      / 1000.0,
               "weight": buckets[rows]["n"]}
              for rows in sorted(buckets)]
    if points:
        return points
    # Too few pairs to resolve any single row count: fall back to the pooled
    # measurement placed at its own mean replayed-row count, which stage B
    # then uses as a constant.
    for report in reports:
        block = report["paired_all"]
        entry = block.get("round_us")
        if entry and block.get("replayed_rows_mean"):
            points.append({"replayed_rows": block["replayed_rows_mean"],
                           "endpoint_premium_ms": entry["mean"] / 1000.0,
                           "weight": entry["n"]})
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    reports = [leg_report(tag) for tag in args.tags]
    for report in reports:
        print(render(report))
        print()
    verdict = verdicts(reports)
    print("== FINDING 494 endpoint verdicts")
    for key, value in verdict.items():
        pct = value["endpoint_premium_pct_of_round"]
        print(f"   {key:34s} endpoint {value['endpoint_premium_us']:+8.1f}"
              f" +-{value['endpoint_two_sigma_us']:7.1f} us"
              f"  ({pct:+.3f}% of round)" if pct is not None else "")
        print(f"      site {value['site_premium_us']}"
              f"  site-endpoint {value['site_minus_endpoint_us']}"
              f"  {value['verdict']}")

    points = premium_by_replayed_rows(reports)
    print("\n== endpoint premium by replayed rows (stage-B input)")
    for point in points:
        print(f"   rows {point['replayed_rows']:5.2f}  "
              f"premium {point['endpoint_premium_ms']:+7.3f} ms  "
              f"pairs {point['weight']}")

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"legs": reports, "verdicts": verdict,
             "premium_by_replayed_rows": points,
             "source": "e205 stage A forced-truncation paired endpoint, "
                       + ",".join(args.tags)}, indent=2) + "\n")
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
