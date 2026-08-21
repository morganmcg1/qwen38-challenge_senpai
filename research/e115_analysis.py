#!/usr/bin/env python3
"""E115 rung 1 -- turn probe cells into the decision table.

    research/e115_analysis.py research/out/TAG/cells.json [--json OUT]

Every arm is timed forward and then in reverse inside one block.

The forward pass is NOT usable. Reading the GPU temperature runs `macmon` as a
subprocess, which leaves the GPU idle for about a second, and the DVFS ramp
back to full clock costs a fixed 30 to 80 ms of wall clock. That fixed cost is
paid entirely by whichever arm is timed first, so it is not monotone drift and
the forward-to-reverse mean does not cancel it: it inflates `a_one` and makes
every other arm look better than it is. On `mlp.gate_up` NA=4 the forward
`a_one` cell reads 774 to 935 us against 685 to 690 us on the reverse pass,
while every other arm agrees between the two passes to better than 1 %.

Position inside a pass does not matter once the GPU is ramped: `f_nsplit4` is
timed last on the forward pass and first on the reverse pass and the two agree
to 0.3 %. The reverse pass therefore measures every arm at full clock, so
`--pass reverse` is the default estimator. `--pass mean` reproduces the
contaminated reading and exists only to show the size of the defect.

Two estimators are reported for every arm, and a verdict is only stated when
both agree:

  raw   the measured cell time, host cost included
  net   the cell time minus the `control.small` cell of the SAME arm structure
        and the SAME width, which is that structure's host cost plus about
        1.5 us of GPU work. `net` is the estimator that transfers to the scored
        path, where MLX encodes the next dispatch while the GPU runs the
        current one and host cost is hidden.

harness=local throughout. No thermal gate, no score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

CONTROL_SHAPE = "control.small"
A_ONE = "a_one"
# Standing local round weights over NA. askeladd's E114 is re-deriving the
# scoring-correct weights, so the per-NA table must survive a reweighting.
ROUND_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}
KILL_RULE_PCT = 3.0
KILL_RULE_SHAPES = ("mlp.gate_up", "lm_head")
KILL_RULE_WIDTH = 4


def load(path: pathlib.Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def cell_key(cell: dict) -> tuple:
    return (cell["shape"], cell["width"], cell["arm"], cell["block"])


PASS = "reverse"


def mean_us(cell: dict) -> float:
    if PASS == "reverse":
        return cell["reverse_us"]
    if PASS == "forward":
        return cell["forward_us"]
    return (cell["forward_us"] + cell["reverse_us"]) / 2


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def main() -> int:
    global PASS
    parser = argparse.ArgumentParser()
    parser.add_argument("cells", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--drop-blocks", type=int, default=1)
    parser.add_argument(
        "--pass", dest="which_pass", default="reverse",
        choices=("reverse", "forward", "mean"))
    args = parser.parse_args()
    PASS = args.which_pass

    payload = load(args.cells)
    cells = payload["cells"]
    kept = [c for c in cells if c["block"] >= args.drop_blocks]
    if not kept:
        print("no cells left after dropping warm-up blocks", file=sys.stderr)
        return 1

    # control[(width, arm)] -> median host cost of that arm structure
    control: dict[tuple[int, str], float] = {}
    for width in sorted({c["width"] for c in kept}):
        for arm in sorted({c["arm"] for c in kept}):
            values = [
                mean_us(c)
                for c in kept
                if c["shape"] == CONTROL_SHAPE
                and c["width"] == width
                and c["arm"] == arm
            ]
            if values:
                control[(width, arm)] = median(values)

    shapes = [
        s
        for s in dict.fromkeys(c["shape"] for c in kept)
        if s != CONTROL_SHAPE
    ]
    widths = sorted({c["width"] for c in kept})
    arms = list(dict.fromkeys(c["arm"] for c in kept))

    summary: dict[str, dict] = {}
    print(f"harness=local  pass={PASS}  blocks_dropped={args.drop_blocks}  "
          f"eval_overhead_us={payload.get('eval_overhead_us', float('nan')):.1f}")
    print(f"cool_gate_passed_real_gate="
          f"{str(payload.get('cool_gate_passed_real_gate')).lower()}  "
          f"gate_qualified_for_timing="
          f"{str(payload.get('gate_qualified_for_timing')).lower()}")
    print()
    print("DVFS ramp contamination: median 100*(forward/reverse - 1) per arm.")
    print("Arms are timed in list order forward and in reverse order after.")
    ramp: dict[str, float] = {}
    for arm in arms:
        gaps = [
            100 * (c["forward_us"] / c["reverse_us"] - 1)
            for c in kept if c["arm"] == arm and c["reverse_us"] > 0
        ]
        ramp[arm] = median(gaps)
    print("  " + "  ".join(f"{arm}={ramp[arm]:+.1f}%" for arm in arms))

    print()
    print("host cost of each arm structure, from control.small (us)")
    for width in widths:
        row = "  NA=%d  " % width + "  ".join(
            f"{arm}={control.get((width, arm), float('nan')):.1f}" for arm in arms)
        print(row)

    for shape in shapes:
        print()
        print("=" * 100)
        print(f"shape={shape}   harness=local")
        print("=" * 100)
        header = (f"  {'NA':>3s} {'arm':16s} {'raw_us':>9s} {'net_us':>9s} "
                  f"{'raw_%_faster':>13s} {'net_%_faster':>13s} "
                  f"{'net_%_range':>18s} {'blocks':>6s}")
        print(header)
        for width in widths:
            blocks = sorted({
                c["block"] for c in kept
                if c["shape"] == shape and c["width"] == width
            })
            per_arm: dict[str, dict[int, float]] = {}
            for arm in arms:
                per_arm[arm] = {
                    c["block"]: mean_us(c)
                    for c in kept
                    if c["shape"] == shape and c["width"] == width
                    and c["arm"] == arm
                }
            if A_ONE not in per_arm or not per_arm[A_ONE]:
                continue
            for arm in arms:
                if not per_arm[arm]:
                    continue
                raw_values, net_values = [], []
                raw_ratios, net_ratios = [], []
                for block in blocks:
                    if block not in per_arm[arm] or block not in per_arm[A_ONE]:
                        continue
                    raw = per_arm[arm][block]
                    base_raw = per_arm[A_ONE][block]
                    raw_values.append(raw)
                    raw_ratios.append(raw / base_raw)
                    host = control.get((width, arm))
                    base_host = control.get((width, A_ONE))
                    if host is not None and base_host is not None:
                        net = raw - host
                        base_net = base_raw - base_host
                        net_values.append(net)
                        if base_net > 0:
                            net_ratios.append(net / base_net)
                raw_pct = 100 * (1 - median(raw_ratios))
                net_pct = 100 * (1 - median(net_ratios)) if net_ratios else float("nan")
                net_lo = 100 * (1 - max(net_ratios)) if net_ratios else float("nan")
                net_hi = 100 * (1 - min(net_ratios)) if net_ratios else float("nan")
                print(f"  {width:3d} {arm:16s} {median(raw_values):9.1f} "
                      f"{median(net_values) if net_values else float('nan'):9.1f} "
                      f"{raw_pct:+13.2f} {net_pct:+13.2f} "
                      f"{net_lo:+8.2f}..{net_hi:+7.2f} {len(raw_ratios):6d}")
                summary[f"{shape}|NA{width}|{arm}"] = {
                    "shape": shape,
                    "width": width,
                    "arm": arm,
                    "raw_us": median(raw_values),
                    "net_us": median(net_values) if net_values else None,
                    "raw_pct_faster_vs_a_one": raw_pct,
                    "net_pct_faster_vs_a_one": net_pct,
                    "net_pct_faster_min": net_lo,
                    "net_pct_faster_max": net_hi,
                    "blocks": len(raw_ratios),
                }

    print()
    print("=" * 100)
    print("round-weighted net % faster than a_one, weights "
          + " ".join(f"NA{k}={v}" for k, v in ROUND_WEIGHTS.items()))
    print("=" * 100)
    weighted: dict[tuple[str, str], float] = {}
    for shape in shapes:
        for arm in arms:
            total, weight_sum = 0.0, 0.0
            for width, weight in ROUND_WEIGHTS.items():
                entry = summary.get(f"{shape}|NA{width}|{arm}")
                if entry is None or entry["net_pct_faster_vs_a_one"] != entry[
                        "net_pct_faster_vs_a_one"]:
                    continue
                total += weight * entry["net_pct_faster_vs_a_one"]
                weight_sum += weight
            if weight_sum > 0:
                weighted[(shape, arm)] = total / weight_sum
        row = "  %-14s " % shape + "  ".join(
            f"{arm}={weighted.get((shape, arm), float('nan')):+.2f}%"
            for arm in arms if arm != A_ONE)
        print(row)

    print()
    print("=" * 100)
    print("H1 / H2 / H3 decomposition, net percentage points faster than a_one")
    print("  total = c_nsplit_pre, the deployable arm: pre-sliced, concurrent")
    print("  H3 slicing     = e_nsplit_serial, the same two half-N dispatches")
    print("                   with concurrency removed")
    print("  H1 concurrency = total - H3, what running them together adds")
    print("  H2 weight share= total - d_indep, what sharing one weight buffer")
    print("                   adds over two separate half-N buffers")
    print("=" * 100)
    print(f"  {'shape':14s} {'NA':>3s} {'total':>8s} {'H3 slice':>9s} "
          f"{'H1 concur':>10s} {'H2 share':>9s}")
    decomposition: dict[str, dict] = {}
    for shape in shapes:
        for width in widths:
            def pct(arm: str) -> float:
                entry = summary.get(f"{shape}|NA{width}|{arm}")
                return entry["net_pct_faster_vs_a_one"] if entry else float("nan")

            total, serial, indep = (
                pct("c_nsplit_pre"), pct("e_nsplit_serial"), pct("d_indep"))
            print(f"  {shape:14s} {width:3d} {total:+8.2f} {serial:+9.2f} "
                  f"{total - serial:+10.2f} {total - indep:+9.2f}")
            decomposition[f"{shape}|NA{width}"] = {
                "total_pct": total,
                "h3_slicing_pct": serial,
                "h1_concurrency_pct": total - serial,
                "h2_weight_sharing_pct": total - indep,
            }

    print()
    print("=" * 100)
    print("group scaling measured directly: b_msplit / a_one net time ratio.")
    print("b_msplit is two concurrent dispatches of NA rows over the FULL N, so")
    print("it is the [w+w] partition against the [w] partition on one tensor.")
    print("=" * 100)
    group_scaling: dict[str, float] = {}
    for shape in shapes:
        cells_row = []
        for width in widths:
            arm = summary.get(f"{shape}|NA{width}|b_msplit")
            base = summary.get(f"{shape}|NA{width}|{A_ONE}")
            if not arm or not base or not base["net_us"]:
                cells_row.append(f"NA{width}=nan")
                continue
            ratio = arm["net_us"] / base["net_us"]
            group_scaling[f"{shape}|NA{width}"] = ratio
            cells_row.append(f"[{width}+{width}]/[{width}]={ratio:.3f}")
        print(f"  {shape:14s} " + "  ".join(cells_row))

    print()
    print("=" * 100)
    print(f"KILL RULE: c_nsplit must be >= {KILL_RULE_PCT:.1f} % faster than "
          f"a_one at NA={KILL_RULE_WIDTH} on {' and '.join(KILL_RULE_SHAPES)}")
    print("=" * 100)
    kill_inputs = {}
    for shape in KILL_RULE_SHAPES:
        for arm in ("c_nsplit", "c_nsplit_pre"):
            entry = summary.get(f"{shape}|NA{KILL_RULE_WIDTH}|{arm}")
            if entry is None:
                continue
            kill_inputs[f"{shape}|{arm}"] = entry["net_pct_faster_vs_a_one"]
            print(f"  {shape:14s} {arm:14s} net {entry['net_pct_faster_vs_a_one']:+.2f} %"
                  f"   raw {entry['raw_pct_faster_vs_a_one']:+.2f} %")
    passes = bool(kill_inputs) and all(
        value >= KILL_RULE_PCT
        for key, value in kill_inputs.items() if key.endswith("c_nsplit_pre"))
    print(f"  KILL RULE PASSED: {passes}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "harness": "local",
                    "pass": PASS,
                    "ramp_forward_over_reverse_pct": ramp,
                    "decomposition": decomposition,
                    "group_scaling_ratio": group_scaling,
                    "blocks_dropped": args.drop_blocks,
                    "eval_overhead_us": payload.get("eval_overhead_us"),
                    "cool_gate_passed_real_gate": payload.get(
                        "cool_gate_passed_real_gate"),
                    "gate_qualified_for_timing": payload.get(
                        "gate_qualified_for_timing"),
                    "control_host_us": {
                        f"NA{width}|{arm}": value
                        for (width, arm), value in control.items()
                    },
                    "cells": summary,
                    "round_weighted_net_pct": {
                        f"{shape}|{arm}": value
                        for (shape, arm), value in weighted.items()
                    },
                    "kill_rule_pct": KILL_RULE_PCT,
                    "kill_rule_inputs": kill_inputs,
                    "kill_rule_passed": passes,
                    "slice_aliasing": payload.get("slice_aliasing"),
                    "exactness": payload.get("exactness"),
                },
                indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
