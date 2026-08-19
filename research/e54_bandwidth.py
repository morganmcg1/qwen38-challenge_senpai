#!/usr/bin/env python3
"""Achieved memory bandwidth per working group, for every measured E54 leg.

PR #8 refuted NA=5 on bandwidth: one NA=5 group sustained 95.5 GB/s against
165.6 GB/s for NA <= 4, with break-even near 131. E49 Arm 2 closed only the
register objection, so the bandwidth objection is still open. This script turns
it into a number.

Law A' says weight traffic is proportional to the working-group count, because
`out_row` in `qmv_fast_crossrow_affine4_g64_m` depends only on `tid.y` and
`simd_gid`, so every working group re-reads the same weight rows. Under Law A'
the aggregate achieved bandwidth is roughly constant across a pair and the time
falls with the group count. If instead a lone NA=5 group cannot sustain the
NA <= 4 rate, the per-group bandwidth drops visibly and the time does not fall.

  python3 research/e54_bandwidth.py --out research/e54-artifacts/e54-bandwidth.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e46_analyze import load  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
LEGS = REPO / ".mlxfast-private/e54-legs"

# Activations and outputs are the kernel's T, which is float16 on this path.
# `weight_bytes` in the harness is k*n*(0.5 + 2*2/64), the packed nibbles plus
# fp16 scales and biases at group size 64, so it already matches that choice.
T_BYTES = 2

TEMPLATE_M = re.compile(r"_m<T,\s*(\d+),\s*(\d+)")
TEMPLATE_PLAIN = re.compile(r"affine4_g64<T,\s*(\d+)\s*>")

# pair -> (control arm, treated arm, treated widths)
PAIRS = {
    "P1": ("iso_m5_ipg3", "iso_m5_ipg5", (5,)),
    "P2": ("iso_m7_ipg4", "iso_m7_ipg5", (7,)),
    "P3": ("iso_m8_ipg4", "iso_m8_ipg5", (8,)),
    "P4": ("shipped", "e27_full", (5, 9)),
}


def group_split(m: int, ipg: int) -> list[int]:
    """Replicate the NA per working group chosen by the `_m` wrapper.

    `quantized.h:1170-1185`: TAIL = M % IPG, a group takes IPG rows while at
    least IPG remain, and the short tail takes `max(TAIL, 2)`. `M % IPG != 1` is
    asserted, so TAIL is never 1 and the groups sum to M exactly.
    """
    tail = m % ipg
    groups, tid = [], 0
    while tid * ipg < m:
        first_m = tid * ipg
        if tail == 0 or m - first_m >= ipg:
            groups.append(ipg)
        else:
            groups.append(max(tail, 2))
        tid += 1
    return groups


def row_groups(row: dict) -> tuple[list[int], str]:
    """NA per working group for a measured row, and how it was determined."""
    path = row["in_kernel_path"]
    hit = TEMPLATE_M.search(path)
    if hit:
        m_t, ipg = int(hit.group(1)), int(hit.group(2))
        if m_t != row["m"]:
            raise SystemExit(f"template width {m_t} disagrees with measured m {row['m']}")
        return group_split(row["m"], ipg), "wrapper_split_from_template"
    hit = TEMPLATE_PLAIN.search(path)
    if hit:
        return [int(hit.group(1))], "single_group_from_template"
    # qmv_fast_impl has no crossrow template. The harness reports the stream
    # count it dispatches, which is one weight stream per row.
    streams = row.get("weight_streams")
    if streams is None:
        return [], "unknown"
    per = [1] * streams
    return per, "fallback_streams_reported_by_harness"


def bandwidth_rows(data: dict) -> list[dict]:
    out = []
    peak = data["roofline"]["peak_bandwidth_bytes_per_second"]
    for sh in data["shapes"]:
        k, n, wbytes = sh["k"], sh["n"], sh["weight_bytes"]
        for row in sh["rows"]:
            if row["bits"] != 4:
                continue
            groups, how = row_groups(row)
            if not groups:
                continue
            secs = row["seconds_per_call"]
            per_group_bytes = [wbytes + na * (k + n) * T_BYTES for na in groups]
            total = sum(per_group_bytes)
            out.append({
                "shape": sh["name"],
                "calls_per_verify": sh["calls_per_verify"],
                "m": row["m"],
                "kernel": row["in_kernel_path"],
                "k": k,
                "n": n,
                "weight_bytes": wbytes,
                "group_na": groups,
                "working_groups": len(groups),
                "group_split_source": how,
                # qmv_fast_impl dispatches one weight stream per row, but the
                # later streams hit cache, so the modelled traffic overstates
                # DRAM traffic and the derived rate exceeds the stream peak.
                # Those rows are excluded from every headline bandwidth number.
                "bandwidth_reliable": how != "fallback_streams_reported_by_harness",
                "seconds_per_call": secs,
                "aggregate_bytes": total,
                "aggregate_gbps": total / secs / 1e9,
                "aggregate_frac_of_measured_peak": total / secs / peak,
                # Groups run concurrently and share the memory system, so one
                # group's own traffic over the same wall time is its share.
                "per_group_gbps": [b / secs / 1e9 for b in per_group_bytes],
                "lone_group": len(groups) == 1,
            })
    return out


def lone_group_curve(summary: dict) -> dict | None:
    """Achieved device rate of a SINGLE working group as a function of its NA.

    In the shipped table M = 2, 3 and 4 each run one working group with NA = M,
    so those widths trace the lone-group rate directly. Extrapolating the curve
    to NA = 5 predicts what a lone NA=5 group can sustain, which is the quantity
    Law C and Law A' disagree about. The differences are near constant, so a
    linear extrapolation of one step is reported rather than a fitted model.
    """
    obs: dict[int, list[float]] = {}
    for widths in summary.values():
        for m, s in widths.items():
            if s["working_groups"] == [1] and s["group_na"] == [(m,)]:
                obs.setdefault(m, []).append(s["weighted_gbps"])
    rates = {na: sum(v) / len(v) for na, v in obs.items() if na >= 2}
    if not {2, 3, 4} <= set(rates):
        return None
    steps = [rates[3] - rates[2], rates[4] - rates[3]]
    return {
        "observed_lone_group_gbps": rates,
        "steps_per_extra_row": steps,
        "mean_step": sum(steps) / len(steps),
        "extrapolated_na5_gbps": rates[4] + sum(steps) / len(steps),
        "extrapolated_na5_gbps_last_step": rates[4] + steps[-1],
        "method": "linear extrapolation of one step from the measured NA=2,3,4 lone-group rates",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default=str(LEGS))
    ap.add_argument("--out", default="research/e54-artifacts/e54-bandwidth.json")
    args = ap.parse_args()

    per_arm: dict[str, dict] = {}
    peak = None
    for path in sorted(glob.glob(str(pathlib.Path(args.legs) / "*-leg.json"))):
        leg = json.loads(pathlib.Path(path).read_text())
        tag, arm = leg["tag"], leg["arm"]
        try:
            data, _ = load(tag)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {tag}: {exc}", file=sys.stderr)
            continue
        peak = data["roofline"]["peak_bandwidth_bytes_per_second"]
        per_arm.setdefault(arm, {})[tag] = bandwidth_rows(data)

    if not per_arm:
        raise SystemExit("no readable legs; run the timed legs first")

    # Weight each shape by its calls per verify, exactly as `t_of_m` weights the
    # latency, so the reported rate is the device-level rate over a verify step
    # rather than an unweighted mean over shapes of very different sizes.
    summary: dict[str, dict] = {}
    for arm, legs in per_arm.items():
        by_m: dict[int, list[dict]] = {}
        legs_at_m: dict[int, set[str]] = {}
        for tag, rows in legs.items():
            for r in rows:
                if r["bandwidth_reliable"]:
                    by_m.setdefault(r["m"], []).append(r)
                    legs_at_m.setdefault(r["m"], set()).add(tag)
        summary[arm] = {}
        for m, rows in sorted(by_m.items()):
            n_legs = len(legs_at_m[m])
            wbytes = sum(r["calls_per_verify"] * r["aggregate_bytes"] for r in rows)
            wsecs = sum(r["calls_per_verify"] * r["seconds_per_call"] for r in rows)
            lone = [g for r in rows for g in r["per_group_gbps"] if r["lone_group"]]
            na5 = [r["per_group_gbps"][0] for r in rows if r["group_na"][:1] == [5]]
            per_shape = {r["shape"]: r["aggregate_gbps"] for r in rows}
            summary[arm][m] = {
                "kernels": sorted({r["kernel"] for r in rows}),
                "working_groups": sorted({r["working_groups"] for r in rows}),
                "group_na": sorted({tuple(r["group_na"]) for r in rows}),
                # Per verify step, summed over shapes and averaged over legs.
                "weighted_bytes_per_verify": wbytes / n_legs,
                "weighted_seconds_per_verify": wsecs / n_legs,
                "weighted_gbps": wbytes / wsecs / 1e9,
                "weighted_frac_of_measured_peak": wbytes / wsecs / peak,
                "per_shape_gbps": per_shape,
                "per_shape_gbps_min": min(per_shape.values()),
                "per_shape_gbps_max": max(per_shape.values()),
                "lone_group_gbps_mean": (sum(lone) / len(lone)) if lone else None,
                "na5_group_gbps_mean": (sum(na5) / len(na5)) if na5 else None,
                "samples": len(rows),
            }

    verdicts = {}
    for pair, (ctrl, treat, widths) in PAIRS.items():
        if ctrl not in summary or treat not in summary:
            continue
        cells = {}
        for w in widths:
            if w not in summary[ctrl] or w not in summary[treat]:
                continue
            c, t = summary[ctrl][w], summary[treat][w]
            # The treated arm only breaks even when it moves its own traffic
            # inside the control arm's measured time. That threshold is measured
            # on this host, not carried over from another machine.
            breakeven = t["weighted_bytes_per_verify"] / c["weighted_seconds_per_verify"]
            cells[w] = {
                "control_groups": c["working_groups"],
                "treated_groups": t["working_groups"],
                "control_group_na": c["group_na"],
                "treated_group_na": t["group_na"],
                "control_weighted_gbps": c["weighted_gbps"],
                "treated_weighted_gbps": t["weighted_gbps"],
                "control_seconds_per_verify": c["weighted_seconds_per_verify"],
                "treated_seconds_per_verify": t["weighted_seconds_per_verify"],
                "traffic_ratio_treated_over_control":
                    t["weighted_bytes_per_verify"] / c["weighted_bytes_per_verify"],
                "seconds_delta_pct":
                    (t["weighted_seconds_per_verify"] - c["weighted_seconds_per_verify"])
                    / c["weighted_seconds_per_verify"] * 100.0,
                "gbps_delta_pct":
                    (t["weighted_gbps"] - c["weighted_gbps"])
                    / c["weighted_gbps"] * 100.0,
                "treated_na5_group_gbps": t["na5_group_gbps_mean"],
                "treated_lone_group_gbps": t["lone_group_gbps_mean"],
                "breakeven_gbps_for_treated_arm": breakeven / 1e9,
                # PR #8 reference points, for the reader's comparison only.
                "pr8_na5_gbps": 95.5,
                "pr8_na_le4_gbps": 165.6,
            }
        verdicts[pair] = cells

    curve = lone_group_curve(summary)
    out = {
        "measured_peak_bandwidth_bytes_per_second": peak,
        "t_bytes": T_BYTES,
        "lone_group_curve": curve,
        "per_arm_per_width": summary,
        "pairs": verdicts,
        "detail": per_arm,
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")

    print("measured stream peak: %.1f GB/s" % (peak / 1e9))
    print("crossrow rows only; qmv_fast_impl fallback rows are excluded")
    for arm in sorted(summary):
        print(f"\n{arm}")
        for m, s in sorted(summary[arm].items()):
            na = "/".join(str(list(g)) for g in s["group_na"])
            print("  m=%2d groups=%s na=%-12s %7.1f GB/s (%3.0f%% peak) "
                  "shapes %.0f-%.0f  t=%.3f ms"
                  % (m, s["working_groups"], na, s["weighted_gbps"],
                     100 * s["weighted_frac_of_measured_peak"],
                     s["per_shape_gbps_min"], s["per_shape_gbps_max"],
                     s["weighted_seconds_per_verify"] * 1e3))
    for pair, cells in verdicts.items():
        for w, c in cells.items():
            print("\n%s width %d: groups %s -> %s, traffic x%.3f, time %+.2f %%"
                  % (pair, w, c["control_groups"], c["treated_groups"],
                     c["traffic_ratio_treated_over_control"], c["seconds_delta_pct"]))
            print("   device rate %.1f -> %.1f GB/s (%+.2f %%)"
                  % (c["control_weighted_gbps"], c["treated_weighted_gbps"],
                     c["gbps_delta_pct"]))
            print("   treated arm breaks even at %.1f GB/s on this host"
                  % c["breakeven_gbps_for_treated_arm"])
            if c["treated_na5_group_gbps"]:
                print("   its NA=5 group sustains %.1f GB/s; PR #8 measured 95.5 "
                      "for NA=5 and 165.6 for NA<=4"
                      % c["treated_na5_group_gbps"])
    if curve:
        print("\nlone working group, achieved device rate against its NA")
        for na, g in sorted(curve["observed_lone_group_gbps"].items()):
            print("   NA=%d  %6.1f GB/s   (measured, one working group)" % (na, g))
        print("   NA=5  %6.1f GB/s   (extrapolated, %+.1f per extra row)"
              % (curve["extrapolated_na5_gbps"], curve["mean_step"]))
    print("\nwrote %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
