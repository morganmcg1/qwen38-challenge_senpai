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
# general one. Names are the Metal function names Metal itself reported, so these
# rules are written against observed names and not against guesses.
FAMILY_RULES = [
    ("qmm_splitk", re.compile(r"affine_qmm_t_splitk|quantized_matmul_splitk")),
    ("qmv", re.compile(r"affine_qmv|_qmv|quantized_matvec|bs_qmv")),
    ("qmm", re.compile(r"affine_qmm|quantized_matmul|gather_qmm|_qvm")),
    ("quant_dequant", re.compile(r"affine_dequantize|affine_quantize")),
    ("dense_gemv", re.compile(r"^gemv_")),
    ("steel_splitk", re.compile(r"steel_gemm_splitk")),
    ("sdpa_fused", re.compile(r"^sdpa_vector")),
    ("sdpa_composed", re.compile(r"steel_gemm_fused|block_softmax|steel_attention")),
    (
        "gdn_recurrence",
        re.compile(r"gated_delta_step|packed_gdn_prework|ssm_kernel|depthwise_conv_1d"),
    ),
    ("qk_rms_rope", re.compile(r"attention_qk_rms_rope")),
    ("norm", re.compile(r"rms_norm|^rms|rms_looped|rms_single|layer_norm")),
    ("rope", re.compile(r"^rope_")),
    (
        "top2_readout",
        re.compile(r"linear_top2|draft_select|draft_rerank|top32|top_?2"),
    ),
    # MLX compiles fused elementwise graphs into one kernel whose name is the
    # op tape plus a graph hash and a contiguity suffix. Match that signature,
    # not the op letters, so a new fusion is still classified.
    ("compiled_fusion", re.compile(r"_\d{12,}_(contiguous|strided)")),
    ("reduce_scan", re.compile(r"_reduce|col_reduce|row_reduce|^scan_|softmax")),
    ("copy", re.compile(r"copy")),
    ("gather_scatter", re.compile(r"^gather|^scatter|^take|^put_along")),
    ("arange", re.compile(r"^arange")),
    (
        "elementwise",
        re.compile(
            r"_(Multiply|Add|Subtract|Divide|Sigmoid|Exp|Log|LogAddExp|Negative|"
            r"GreaterEqual|Greater|LessEqual|Less|Equal|NotEqual|Select|AsType|"
            r"Maximum|Minimum|Abs|Sqrt|Rsqrt|Tanh|Erf|Power|Remainder|Sign|Floor|"
            r"Ceil|Round|Square|Cos|Sin)"
        ),
    ),
    ("pad_concat", re.compile(r"^pad|^concat|^slice|^broadcast|^contiguous|^strided")),
]

# The assignment asks for eight declared groups. Keep the fine labels above for
# diagnosis and map them onto those groups for the headline table, so a group
# total is never produced by silently lumping two mechanisms together.
ASSIGNMENT_GROUPS = {
    "qmv": "1_quant_matvec",
    "qmm": "2_qmm_splitk",
    "qmm_splitk": "2_qmm_splitk",
    "quant_dequant": "2_qmm_splitk",
    "dense_gemv": "2_qmm_splitk",
    "steel_splitk": "2_qmm_splitk",
    "sdpa_fused": "3_sdpa",
    "sdpa_composed": "3_sdpa",
    "gdn_recurrence": "4_gdn",
    "norm": "5_norm_rope",
    "rope": "5_norm_rope",
    "qk_rms_rope": "5_norm_rope",
    "copy": "6_elementwise",
    "elementwise": "6_elementwise",
    "compiled_fusion": "6_elementwise",
    "reduce_scan": "6_elementwise",
    "gather_scatter": "6_elementwise",
    "arange": "6_elementwise",
    "pad_concat": "6_elementwise",
    "top2_readout": "7_top2_readout",
}

# `sv_Multiply` and `arangeint32` are members of the eight-dispatch composed
# SDPA fallback, but the same kernels also serve gate arithmetic elsewhere, so a
# name cannot attribute them. These members appear only in the fallback, so the
# fallback's invocation count is read from them instead.
SDPA_COMPOSED_WITNESSES = (
    re.compile(r"block_softmax"),
    re.compile(r"steel_gemm_fused_nn"),
    re.compile(r"_GreaterEqual"),
    re.compile(r"_Select"),
)
SDPA_COMPOSED_DISPATCHES_PER_CALL = 8


def family(kernel: str) -> str:
    for name, rule in FAMILY_RULES:
        if rule.search(kernel):
            return name
    return "unclassified"


def assignment_group(kernel_family: str) -> str:
    return ASSIGNMENT_GROUPS.get(kernel_family, "9_unclassified")


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
        group_totals = defaultdict(int)
        group_width = defaultdict(lambda: defaultdict(int))
        unclassified = defaultdict(int)
        composed_witness = defaultdict(int)
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
                phase_bucket["dispatch_ns"] = phase_bucket.get(
                    "dispatch_ns", 0
                ) + phase.get("dispatch_ns", 0)
                phase_bucket["commit_ns"] = phase_bucket.get("commit_ns", 0) + phase.get(
                    "commit_ns", 0
                )
                phase_bucket["clock_bias_ns"] = phase_bucket.get(
                    "clock_bias_ns", 0
                ) + phase.get("clock_bias_ns", 0)
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
                    group = assignment_group(fam)
                    group_totals[group] += count
                    group_width[width][group] += count
                    if fam == "unclassified":
                        unclassified[kernel] += count
                    for witness in SDPA_COMPOSED_WITNESSES:
                        if witness.search(kernel):
                            composed_witness[kernel] += count

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
        rounds_count = max(1, len(pid_rounds))
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
        # Host-side dispatch price, measured directly rather than inferred: the
        # wall time the encoding thread spent inside Metal's own
        # dispatch/commit implementations, against the round's wall time.
        wall_ns = sum(record.get("wall_ns", 0) for record in pid_rounds)
        dispatch_ns = sum(
            bucket.get("dispatch_ns", 0) for bucket in leg["phase_totals"].values()
        )
        commit_ns = sum(
            bucket.get("commit_ns", 0) for bucket in leg["phase_totals"].values()
        )
        bias_ns = sum(
            bucket.get("clock_bias_ns", 0) for bucket in leg["phase_totals"].values()
        )
        leg["host_timing"] = {
            "wall_ms_per_round": round(wall_ns / 1e6 / rounds_count, 3),
            "encode_ns_per_dispatch": round(dispatch_ns / max(1, round_dispatches), 1),
            "encode_ns_per_dispatch_bias_corrected": round(
                (dispatch_ns - bias_ns) / max(1, round_dispatches), 1
            ),
            "commit_ns_per_commit": round(
                commit_ns / max(1, leg["commits_in_rounds"]), 1
            ),
            "encode_ms_per_round": round(dispatch_ns / 1e6 / rounds_count, 3),
            "commit_ms_per_round": round(commit_ns / 1e6 / rounds_count, 3),
            "host_share_of_round_percent": round(
                100.0 * (dispatch_ns + commit_ns) / max(1, wall_ns), 2
            ),
            "clock_read_ns": round(bias_ns / max(1, round_dispatches), 2),
        }
        leg["group_totals"] = dict(sorted(group_totals.items()))
        leg["group_by_width"] = {
            str(width): dict(sorted(values.items()))
            for width, values in sorted(group_width.items())
        }
        leg["unclassified_kernels"] = dict(
            sorted(unclassified.items(), key=lambda item: -item[1])
        )
        witness_calls = {
            kernel: count for kernel, count in sorted(composed_witness.items())
        }
        leg["sdpa_composed_witnesses"] = witness_calls
        leg["sdpa_composed_calls"] = min(witness_calls.values()) if witness_calls else 0
        leg["sdpa_composed_dispatches_estimate"] = (
            leg["sdpa_composed_calls"] * SDPA_COMPOSED_DISPATCHES_PER_CALL
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
        if leg["unclassified_kernels"]:
            print("-- UNCLASSIFIED kernels (must be empty for a defensible table) --")
            for kernel, count in leg["unclassified_kernels"].items():
                print(f"   {count:>9}  {kernel}")
        print("-- composed SDPA fallback --")
        print(
            f"   calls={leg['sdpa_composed_calls']} "
            f"dispatch_estimate={leg['sdpa_composed_dispatches_estimate']} "
            f"(at {SDPA_COMPOSED_DISPATCHES_PER_CALL} dispatches per call)"
        )
        for kernel, count in leg["sdpa_composed_witnesses"].items():
            print(f"       witness {count:>8}  {kernel[:96]}")
        host = leg["host_timing"]
        print("-- host-side dispatch price (census-on, so wall time is inflated) --")
        print(
            f"   wall={host['wall_ms_per_round']} ms/round  "
            f"encode={host['encode_ms_per_round']} ms/round  "
            f"commit={host['commit_ms_per_round']} ms/round  "
            f"host_share={host['host_share_of_round_percent']}%"
        )
        print(
            f"   encode={host['encode_ns_per_dispatch']} ns/dispatch "
            f"(bias-corrected {host['encode_ns_per_dispatch_bias_corrected']}), "
            f"commit={host['commit_ns_per_commit']} ns/commit, "
            f"clock_read={host['clock_read_ns']} ns"
        )
        print("-- assignment groups, per round --")
        rounds = max(1, leg["rounds"])
        for group, count in sorted(leg["group_totals"].items()):
            print(
                f"   {group:<18} total={count:>9} per_round={count / rounds:>9.2f} "
                f"share={100.0 * count / max(1, leg['dispatch_total_in_rounds']):>6.2f}%"
            )
        print("-- per verify width M: assignment groups per round --")
        groups = sorted(leg["group_totals"])
        header = "   M    rounds  total/round " + "".join(
            f"{group.split('_', 1)[1][:11]:>13}" for group in groups
        )
        print(header)
        for width, bucket in leg["widths"].items():
            counts = leg["group_by_width"][width]
            width_rounds = bucket["rounds"]
            row = f"   {width:<4} {width_rounds:>6} {bucket['dispatches_per_round']:>12.1f} "
            row += "".join(
                f"{counts.get(group, 0) / width_rounds:>13.1f}" for group in groups
            )
            print(row)
        print("-- per verify width M: fine families per round --")
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
