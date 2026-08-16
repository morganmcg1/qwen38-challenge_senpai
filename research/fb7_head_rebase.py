#!/usr/bin/env python3
"""Rebase local block timings and the depth cost curve onto the ranked head.

FB7's finding: the local candidate leg loads the organizer-pinned bf16 head
(849,398,784 tensor bytes) while the ranked candidate leg resolves the
declared remote 4-bit/group-64 head (238,934,093 bytes). Every head forward
therefore carries `delta = (849398784 - 238934093) / BW` locally that the
ranked leg does not pay, and a round at draft depth `d` carries `delta * d`.

Two consequences, both computed here.

1. STALL GUARDRAIL. `check_stall_guardrail` reads
   `max_block_request_seconds_after_first / p50_block_request_seconds_after_first`.
   Subtracting the same absolute amount from numerator and denominator RAISES
   a ratio above 1, so the local reading is optimistic, not conservative. The
   subtraction is only equal when the max round and the p50 round have equal
   depth, so both depths are reported explicitly.

   p50 here is the wrapper's lower median, `sorted[(n-1)//2]` over the
   after-first slice, so it is always an actually observed round with a real
   depth -- no interpolation, nothing to attribute.

2. COST CURVE. Cost per accepted token at full acceptance is `C(d)/(d+1)`.
   Rebasing subtracts `delta*d` from `C(d)`, which is linear in `d`, so an
   interior optimum can move. Both the measured curve and the kink-removed
   linear fit are reported because they disagree.

The serial control leg runs no proposal head and is never rebased.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics

PHASE_MARKERS = (
    ("reference", "generating the MTP reference rows"),
    ("serial", "measuring the TRUE serial control"),
    ("mtp", "measuring native-MTP decode"),
)

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) .*?\bround_us=(\d+)")
DEPTH_RE = re.compile(r"\bd=(\d+)")
ACC_RE = re.compile(r"\bacc=(\d+)")

# Resident local head vs declared ranked head, from
# research/fb7_head_provenance.py.
LOCAL_HEAD_BYTES = 849_398_784
RANKED_HEAD_BYTES = 238_934_093
BANDWIDTH_BYTES_PER_S = 227e9

# Measured round cost by draft depth on this host, ms (Run C, round 1 dropped).
MEASURED_C = {2: 79.7, 4: 126.4, 5: 146.5, 6: 168.3, 7: 189.7, 8: 217.4}
# Kink-removed fit in width, C = FIT_FIXED + FIT_SLOPE * (d + 1).
FIT_FIXED_MS = 12.21
FIT_SLOPE_MS = 22.49


def head_delta_ms(bandwidth: float) -> float:
    return (LOCAL_HEAD_BYTES - RANKED_HEAD_BYTES) / bandwidth * 1e3


def phase_of(line: str, current: str) -> str:
    for name, marker in PHASE_MARKERS:
        if marker in line:
            return name
    return current


def parse_trace(path: str) -> list[dict]:
    rounds: list[dict] = []
    current = "reference"
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            current = phase_of(line, current)
            if current != "mtp":
                continue
            match = ROUND_RE.search(line)
            if not match:
                continue
            depth = DEPTH_RE.search(line)
            accepted = ACC_RE.search(line)
            rounds.append(
                {
                    "round": int(match.group(1)),
                    "seconds": int(match.group(2)) / 1e6,
                    "depth": int(depth.group(1)) if depth else None,
                    "accepted": int(accepted.group(1)) if accepted else None,
                }
            )
    return rounds


def lower_median_index(count: int) -> int:
    """The wrapper's lower-median rule, so p50 names a real round."""
    return (count - 1) // 2


def guardrail(rounds: list[dict], delta_ms: float) -> dict:
    if len(rounds) < 2:
        return {"error": "need at least two rounds"}
    first = rounds[0]
    after = rounds[1:]

    order = sorted(range(len(after)), key=lambda i: after[i]["seconds"])
    max_pos = max(range(len(after)), key=lambda i: after[i]["seconds"])
    p50_pos = order[lower_median_index(len(after))]

    max_round = after[max_pos]
    p50_round = after[p50_pos]

    max_ms = max_round["seconds"] * 1e3
    p50_ms = p50_round["seconds"] * 1e3
    d_max = max_round["depth"]
    d_p50 = p50_round["depth"]

    ranked_max = max_ms - delta_ms * d_max
    ranked_p50 = p50_ms - delta_ms * d_p50

    return {
        "rounds_total": len(rounds),
        "rounds_after_first": len(after),
        "first_ms": first["seconds"] * 1e3,
        "first_depth": first["depth"],
        "p50_after_first_ms": p50_ms,
        "p50_round_index": p50_round["round"],
        "p50_round_depth": d_p50,
        "p50_round_accepted": p50_round["accepted"],
        "max_after_first_ms": max_ms,
        "max_round_index": max_round["round"],
        "max_round_depth": d_max,
        "max_round_accepted": max_round["accepted"],
        "max_round_was_rejection": (
            max_round["accepted"] is not None
            and d_max is not None
            and max_round["accepted"] < d_max
        ),
        "ratio_local": max_ms / p50_ms,
        "first_over_p50_local": first["seconds"] * 1e3 / p50_ms,
        "head_delta_ms_per_draft": delta_ms,
        "ranked_max_after_first_ms": ranked_max,
        "ranked_p50_after_first_ms": ranked_p50,
        "ratio_ranked": ranked_max / ranked_p50,
        "ratio_shift": ranked_max / ranked_p50 - max_ms / p50_ms,
        "depths_equal": d_max == d_p50,
        # Exact direction rule. Writing M/P for the local ratio, rebasing
        # gives (M - k*dm)/(P - k*dp) > M/P iff P*dm < M*dp, i.e. iff
        # dm/dp < M/P. So the local reading is optimistic only when the max
        # round is not enough deeper than the p50 round to outweigh the
        # raggedness already present. Equal depths always make it optimistic.
        "depth_ratio_max_over_p50": (
            d_max / d_p50 if d_p50 else None
        ),
        "ranked_ratio_rises": (
            (d_max / d_p50) < (max_ms / p50_ms) if d_p50 else None
        ),
    }


def cost_curve(delta_ms: float) -> dict:
    measured = {}
    for depth, cost in sorted(MEASURED_C.items()):
        ranked = cost - delta_ms * depth
        measured[depth] = {
            "local_round_ms": cost,
            "ranked_round_ms": ranked,
            "local_ms_per_accepted_token": cost / (depth + 1),
            "ranked_ms_per_accepted_token": ranked / (depth + 1),
        }
    best_local = min(
        measured, key=lambda d: measured[d]["local_ms_per_accepted_token"]
    )
    best_ranked = min(
        measured, key=lambda d: measured[d]["ranked_ms_per_accepted_token"]
    )

    fit = {}
    for depth in range(2, 9):
        local = FIT_FIXED_MS + FIT_SLOPE_MS * (depth + 1)
        ranked = local - delta_ms * depth
        fit[depth] = {
            "local_round_ms": local,
            "ranked_round_ms": ranked,
            "local_ms_per_accepted_token": local / (depth + 1),
            "ranked_ms_per_accepted_token": ranked / (depth + 1),
        }
    fit_best_local = min(
        fit, key=lambda d: fit[d]["local_ms_per_accepted_token"]
    )
    fit_best_ranked = min(
        fit, key=lambda d: fit[d]["ranked_ms_per_accepted_token"]
    )

    marginal_local = MEASURED_C[8] - MEASURED_C[7]
    marginal_ranked = (
        measured[8]["ranked_round_ms"] - measured[7]["ranked_round_ms"]
    )
    return {
        "measured": measured,
        "measured_cost_optimal_depth_local": best_local,
        "measured_cost_optimal_depth_ranked": best_ranked,
        "measured_row9_marginal_local_ms": marginal_local,
        "measured_row9_marginal_ranked_ms": marginal_ranked,
        "measured_row9_repays_local": (
            marginal_local
            <= measured[best_local]["local_ms_per_accepted_token"]
        ),
        "measured_row9_repays_ranked": (
            marginal_ranked
            <= measured[best_ranked]["ranked_ms_per_accepted_token"]
        ),
        "linear_fit": fit,
        "linear_fit_optimal_depth_local": fit_best_local,
        "linear_fit_optimal_depth_ranked": fit_best_ranked,
        "linear_fit_ranked_slope_ms": FIT_SLOPE_MS - delta_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="worker-side trace for one arm",
    )
    parser.add_argument(
        "--parent",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="parent-side CLI report holding block_request_seconds",
    )
    parser.add_argument(
        "--parent-depths-from",
        default=None,
        metavar="LABEL",
        help="borrow the per-round depth sequence of this traced arm for the "
        "parent-side arms, which record no depths of their own",
    )
    parser.add_argument("--bandwidth", type=float, default=BANDWIDTH_BYTES_PER_S)
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    delta = head_delta_ms(args.bandwidth)
    arms: dict[str, dict] = {}
    traced: dict[str, list[dict]] = {}

    for spec in args.trace:
        label, path = spec.split("=", 1)
        rounds = parse_trace(path)
        traced[label] = rounds
        arms[label] = {
            "source": "worker_side_round_us",
            "trace": path,
            **guardrail(rounds, delta),
        }

    borrowed = traced.get(args.parent_depths_from) if args.parent_depths_from else None

    for spec in args.parent:
        label, path = spec.split("=", 1)
        with open(path, "r") as handle:
            report = json.load(handle)
        blocks = _find_blocks(report)
        if not blocks:
            arms[label] = {"error": f"no block_request_seconds in {path}"}
            continue
        rounds = [
            {
                "round": i + 1,
                "seconds": value,
                "depth": (
                    borrowed[i]["depth"]
                    if borrowed and i < len(borrowed)
                    else None
                ),
                "accepted": (
                    borrowed[i]["accepted"]
                    if borrowed and i < len(borrowed)
                    else None
                ),
            }
            for i, value in enumerate(blocks)
        ]
        entry = {
            "source": "parent_side_block_request_seconds",
            "report": path,
        }
        if borrowed is not None and len(borrowed) == len(blocks):
            entry["depths_borrowed_from"] = args.parent_depths_from
            entry["depth_borrow_is_exact_length_match"] = True
            entry.update(guardrail(rounds, delta))
        elif borrowed is not None:
            entry["depths_borrowed_from"] = args.parent_depths_from
            entry["depth_borrow_is_exact_length_match"] = False
            entry["borrowed_round_count"] = len(borrowed)
            entry["block_count"] = len(blocks)
            entry.update(guardrail(rounds, delta))
        else:
            # Head-free leg (serial control): no rebasing is defined.
            plain = guardrail(
                [{**r, "depth": 0} for r in rounds], 0.0
            )
            entry.update(plain)
            entry["head_free_leg_not_rebased"] = True
        arms[label] = entry

    result = {
        "local_head_bytes": LOCAL_HEAD_BYTES,
        "ranked_head_bytes": RANKED_HEAD_BYTES,
        "bandwidth_bytes_per_second": args.bandwidth,
        "head_delta_ms_per_draft": delta,
        "arms": arms,
        "cost_curve": cost_curve(delta),
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    if args.summary:
        print_summary(result)
    else:
        print(text)


def print_summary(result: dict) -> None:
    print(f"head delta per draft = {result['head_delta_ms_per_draft']:.4f} ms")
    print()
    print("GUARDRAIL, local head vs rebased onto the ranked head")
    header = (
        "arm",
        "n_af",
        "p50_ms",
        "d_p50",
        "max_ms",
        "d_max",
        "rej",
        "local",
        "ranked",
        "shift",
        "rises",
    )
    print("%-7s %4s %9s %5s %9s %5s %5s %8s %8s %9s %5s" % header)
    for label, arm in result["arms"].items():
        if "error" in arm:
            print(f"{label}: {arm['error']}")
            continue
        print(
            "%-7s %4d %9.3f %5s %9.3f %5s %5s %8.4f %8.4f %+9.4f %5s"
            % (
                label,
                arm["rounds_after_first"],
                arm["p50_after_first_ms"],
                arm["p50_round_depth"],
                arm["max_after_first_ms"],
                arm["max_round_depth"],
                arm["max_round_was_rejection"],
                arm["ratio_local"],
                arm["ratio_ranked"],
                arm["ratio_shift"],
                arm["ranked_ratio_rises"],
            )
        )
    print()
    curve = result["cost_curve"]
    print("MEASURED cost curve (ms)")
    print(
        "%3s %10s %10s %12s %12s"
        % ("d", "local_C", "ranked_C", "local/tok", "ranked/tok")
    )
    for depth, row in sorted(curve["measured"].items(), key=lambda x: int(x[0])):
        print(
            "%3s %10.2f %10.2f %12.3f %12.3f"
            % (
                depth,
                row["local_round_ms"],
                row["ranked_round_ms"],
                row["local_ms_per_accepted_token"],
                row["ranked_ms_per_accepted_token"],
            )
        )
    print(
        "optimal depth: local=%s ranked=%s"
        % (
            curve["measured_cost_optimal_depth_local"],
            curve["measured_cost_optimal_depth_ranked"],
        )
    )
    print(
        "row 9 marginal: local=%.2f ranked=%.2f  repays local=%s ranked=%s"
        % (
            curve["measured_row9_marginal_local_ms"],
            curve["measured_row9_marginal_ranked_ms"],
            curve["measured_row9_repays_local"],
            curve["measured_row9_repays_ranked"],
        )
    )
    print()
    print(
        "KINK-REMOVED linear fit, ranked slope %.2f ms/row"
        % curve["linear_fit_ranked_slope_ms"]
    )
    print(
        "%3s %10s %10s %12s %12s"
        % ("d", "local_C", "ranked_C", "local/tok", "ranked/tok")
    )
    for depth, row in sorted(
        curve["linear_fit"].items(), key=lambda x: int(x[0])
    ):
        print(
            "%3s %10.2f %10.2f %12.3f %12.3f"
            % (
                depth,
                row["local_round_ms"],
                row["ranked_round_ms"],
                row["local_ms_per_accepted_token"],
                row["ranked_ms_per_accepted_token"],
            )
        )
    print(
        "fit optimal depth: local=%s ranked=%s"
        % (
            curve["linear_fit_optimal_depth_local"],
            curve["linear_fit_optimal_depth_ranked"],
        )
    )


def _find_blocks(node) -> list[float] | None:
    if isinstance(node, dict):
        value = node.get("block_request_seconds")
        if isinstance(value, list) and value:
            return [float(v) for v in value]
        for child in node.values():
            found = _find_blocks(child)
            if found:
                return found
    elif isinstance(node, list):
        for child in node:
            found = _find_blocks(child)
            if found:
                return found
    return None


if __name__ == "__main__":
    main()
