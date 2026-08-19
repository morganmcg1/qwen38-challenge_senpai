#!/usr/bin/env python3
"""Turn one E58 census dump into the round dispatch table.

The census writes one JSON line per round (and one `gap` line per round for the
dispatches MLX encoded outside it), keyed by pid so the reference, serial and
candidate legs of a single --local-iterate run stay separable.

usage:
  research/e58_census_report.py research/out/TAG/census.jsonl [--json OUT]

Kernel families are declared, not inferred, and anything that matches no rule is
reported in an `unclassified` row rather than folded into a neighbour.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

# Ordered: the first matching rule wins, so a specific rule must precede a
# general one. Names are the Metal function names Metal itself reported.
FAMILY_RULES = [
    ("qmv", re.compile(r"qmv|quantized_matvec|bs_qmv")),
    ("qmm", re.compile(r"qmm|quantized_matmul|qvm|gather_qmm")),
    ("sdpa_fused", re.compile(r"^sdpa_vector")),
    (
        "sdpa_composed",
        re.compile(
            r"steel_gemm_fused|block_softmax|steel_attention|steel_gemm_splitk"
        ),
    ),
    ("gdn", re.compile(r"gated_delta|gdn|ssm_kernel")),
    ("qk_rms_rope", re.compile(r"attention_qk_rms_rope")),
    ("norm_rope", re.compile(r"rms_norm|rms_looped|rms_single|_rope|rope_|layer_norm")),
    ("top2_readout", re.compile(r"top2|top_2|top32|draft_select|draft_rerank")),
    (
        "elementwise_copy",
        re.compile(
            r"^copy|^g\d|^v\d|^s\d|^vv|^sv|^vs|^ss|arange|^pad|^concat|^select|"
            r"^compare|^unary|^binary|^ternary|^reduce|^scan|^softmax|^slice|"
            r"^gather|^scatter|^partition|^sort|^random|^where|^broadcast|^as_type|"
            r"^astype|^contiguous|^strided"
        ),
    ),
]


def family(kernel: str) -> str:
    for name, rule in FAMILY_RULES:
        if rule.search(kernel):
            return name
    return "unclassified"


def load(path: str):
    rounds = []
    gaps = []
    events = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("event")
            if kind == "round":
                rounds.append(record)
            elif kind == "gap":
                gaps.append(record)
            else:
                events.append(record)
    return rounds, gaps, events


def leg_label(pid_rounds) -> str:
    widths = {record["width"] for record in pid_rounds}
    depths = {record["depth"] for record in pid_rounds}
    if depths == {0}:
        return "serial(depth0)"
    return "candidate(mtp)"


def totals(record):
    dispatches = 0
    barriers = 0
    commits = 0
    for phase in record.get("phases", {}).values():
        dispatches += phase.get("dispatches", 0)
        barriers += phase.get("barriers", 0)
        commits += phase.get("commits", 0)
    return dispatches, barriers, commits


def summarize(path: str):
    rounds, gaps, events = load(path)
    by_pid = defaultdict(list)
    for record in rounds:
        by_pid[record["pid"]].append(record)
    gaps_by_pid = defaultdict(list)
    for record in gaps:
        gaps_by_pid[record["pid"]].append(record)

    report = {"source": path, "install_events": events, "legs": []}
    for pid, pid_rounds in sorted(by_pid.items()):
        label = leg_label(pid_rounds)
        leg = {
            "pid": pid,
            "leg": label,
            "rounds": len(pid_rounds),
            "widths": {},
            "phase_totals": {},
            "family_totals": {},
            "family_by_width": {},
            "kernel_totals": {},
            "kernel_by_phase": {},
            "gap_dispatches": sum(totals(record)[0] for record in gaps_by_pid[pid]),
            "gap_commits": sum(totals(record)[2] for record in gaps_by_pid[pid]),
        }
        # The first `gap` line of a leg carries warmup plus the seed prefill,
        # which is inside the scored window but is not a decode round.
        prefill = {}
        for record in gaps_by_pid[pid]:
            for name, phase in record.get("phases", {}).items():
                bucket = prefill.setdefault(
                    name, {"dispatches": 0, "commits": 0, "kernels": defaultdict(int)}
                )
                bucket["dispatches"] += phase.get("dispatches", 0)
                bucket["commits"] += phase.get("commits", 0)
                for kernel, count in phase.get("kernels", {}).items():
                    bucket["kernels"][kernel] += count
        leg["outside_rounds"] = {
            name: {
                "dispatches": bucket["dispatches"],
                "commits": bucket["commits"],
                "families": _family_counts(bucket["kernels"]),
            }
            for name, bucket in prefill.items()
        }

        width_rounds = defaultdict(int)
        family_width = defaultdict(lambda: defaultdict(int))
        for record in pid_rounds:
            width = record["width"]
            width_rounds[width] += 1
            for phase_name, phase in record.get("phases", {}).items():
                phase_bucket = leg["phase_totals"].setdefault(
                    phase_name, {"dispatches": 0, "barriers": 0, "commits": 0}
                )
                phase_bucket["dispatches"] += phase.get("dispatches", 0)
                phase_bucket["barriers"] += phase.get("barriers", 0)
                phase_bucket["commits"] += phase.get("commits", 0)
                for kernel, count in phase.get("kernels", {}).items():
                    leg["kernel_totals"][kernel] = (
                        leg["kernel_totals"].get(kernel, 0) + count
                    )
                    key = f"{phase_name}|{kernel}"
                    leg["kernel_by_phase"][key] = (
                        leg["kernel_by_phase"].get(key, 0) + count
                    )
                    fam = family(kernel)
                    leg["family_totals"][fam] = leg["family_totals"].get(fam, 0) + count
                    family_width[width][fam] += count

        leg["widths"] = {
            str(width): {
                "rounds": count,
                "dispatches_per_round": round(
                    sum(family_width[width].values()) / count, 2
                ),
                "families_per_round": {
                    fam: round(value / count, 3)
                    for fam, value in sorted(family_width[width].items())
                },
            }
            for width, count in sorted(width_rounds.items())
        }
        leg["family_by_width"] = {
            str(width): dict(sorted(values.items()))
            for width, values in sorted(family_width.items())
        }
        round_dispatches = sum(leg["family_totals"].values())
        leg["dispatch_total_in_rounds"] = round_dispatches
        leg["dispatches_per_round_mean"] = round(
            round_dispatches / max(1, len(pid_rounds)), 2
        )
        leg["commits_in_rounds"] = sum(
            bucket["commits"] for bucket in leg["phase_totals"].values()
        )
        leg["barriers_in_rounds"] = sum(
            bucket["barriers"] for bucket in leg["phase_totals"].values()
        )
        leg["dispatches_per_commit"] = round(
            round_dispatches / max(1, leg["commits_in_rounds"]), 2
        )
        report["legs"].append(leg)
    return report


def _family_counts(kernels) -> dict:
    counts = defaultdict(int)
    for kernel, count in kernels.items():
        counts[family(kernel)] += count
    return dict(sorted(counts.items()))


def print_table(report) -> None:
    for leg in report["legs"]:
        print(f"\n=== pid {leg['pid']} : {leg['leg']} ===")
        print(
            f"rounds={leg['rounds']} dispatches_in_rounds={leg['dispatch_total_in_rounds']} "
            f"per_round={leg['dispatches_per_round_mean']} "
            f"command_buffers={leg['commits_in_rounds']} "
            f"dispatches_per_buffer={leg['dispatches_per_commit']} "
            f"barriers={leg['barriers_in_rounds']}"
        )
        print("-- outside rounds (warmup + seed prefill + inter-round) --")
        for name, bucket in sorted(leg["outside_rounds"].items()):
            print(
                f"   {name:<16} dispatches={bucket['dispatches']:>8} "
                f"commits={bucket['commits']:>6}"
            )
            for fam, count in bucket["families"].items():
                print(f"       {fam:<18} {count:>8}")
        print("-- phases inside rounds --")
        for name, bucket in sorted(leg["phase_totals"].items()):
            print(
                f"   {name:<16} dispatches={bucket['dispatches']:>8} "
                f"per_round={bucket['dispatches'] / max(1, leg['rounds']):>9.2f} "
                f"commits={bucket['commits']:>6} barriers={bucket['barriers']:>8}"
            )
        print("-- families, per round --")
        for fam, count in sorted(
            leg["family_totals"].items(), key=lambda item: -item[1]
        ):
            print(
                f"   {fam:<18} total={count:>9} per_round={count / max(1, leg['rounds']):>9.2f}"
            )
        print("-- top kernels, per round --")
        for kernel, count in sorted(
            leg["kernel_totals"].items(), key=lambda item: -item[1]
        )[:28]:
            print(
                f"   {kernel:<52} {count:>9} {count / max(1, leg['rounds']):>9.2f}"
                f"  [{family(kernel)}]"
            )
        print("-- per verify width M --")
        for width, bucket in leg["widths"].items():
            print(
                f"   M={width:<3} rounds={bucket['rounds']:>4} "
                f"dispatches/round={bucket['dispatches_per_round']:>9.2f}"
            )
            for fam, value in bucket["families_per_round"].items():
                print(f"       {fam:<18} {value:>9.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("census")
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()
    report = summarize(args.census)
    print_table(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
