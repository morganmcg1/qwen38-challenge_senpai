#!/usr/bin/env python3
"""E117 -- turn shipped-width-frame probe cells into the decision table.

    research/e117_analysis.py research/out/TAG/cells.json [--json OUT]

The E115 probe measured `a_one` and `e_nsplit_serial` in the NA frame, where
every cell is one working x-group in one dispatch. The shipped kernel is in the
M frame. `quantized.cpp:251-254` dispatches `grid_dims(M, N/8, B)`, so
`ntg.x == M`, and `qmv_fast_crossrow_affine4_g64_m` at `quantized.h:1157-1186`
early-returns every threadgroup whose `first_m = tid.x * IPG` is at or past M.
The realised partition and its working-group count are therefore fixed by M
alone, and this script prices every cell against that table.

Two estimators are reported for every arm and no verdict is stated unless both
agree:

  raw   the measured cell time, host cost included
  net   the cell time minus the `control.small` cell of the SAME arm structure
        and the SAME width, which is that structure's host cost plus about
        1.5 us of GPU work. `net` is the estimator that transfers to the scored
        path, where MLX encodes the next dispatch while the GPU runs the
        current one and host cost is hidden.

Passes. Every arm is timed forward and then in reverse inside one block. E115
had to fall back to reverse-pass-only because a `macmon` subprocess left the GPU
idle and the whole DVFS ramp landed on whichever arm was timed first. The E117
probe absorbs that ramp with a discarded burst of fixed WALL-CLOCK duration, so
`--pass mean` is the default here and `--pass forward` / `--pass reverse` exist
to publish the residual gap that proves the fix worked.

CAMPAIGN RULE 39. Every percent is a paired within-block contrast, so its
standard error is the standard error of the per-block contrasts. The estimator
and that standard error are printed next to every number.

harness=local throughout. No thermal gate, no score.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

CONTROL_SHAPE = "control.small"
A_ONE = "a_one"
DEFECT19_FACTOR = 1.5

# `quantized.h:1922-1979`. IPG is the second template argument of
# `qmv_fast_crossrow_affine4_g64_m`; M=1 and M=2 do not reach that template.
IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}


def partition(m: int) -> tuple[int, ...]:
    if m <= 2:
        return (m,)
    ipg = IPG[m]
    groups, left = [], m
    while left > 0:
        groups.append(min(ipg, left))
        left -= min(ipg, left)
    return tuple(groups)


def group_count(m: int) -> int:
    return len(partition(m))


def packed_bytes(outputs: int, hidden: int) -> int:
    """affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias."""
    return outputs * hidden // 2 + 4 * (outputs * hidden // 64)


def pick(cell: dict, which: str) -> float:
    if which == "forward":
        return cell["forward_us"]
    if which == "reverse":
        return cell["reverse_us"]
    return (cell["forward_us"] + cell["reverse_us"]) / 2


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values) / math.sqrt(len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cells", type=pathlib.Path)
    parser.add_argument(
        "--pass", dest="which", default="mean",
        choices=("mean", "forward", "reverse"))
    parser.add_argument("--drop-first-block", action="store_true")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    payload = json.loads(args.cells.read_text())
    cells = payload["cells"]
    if args.drop_first_block:
        cells = [c for c in cells if c["block"] > 0]

    # (shape, width, arm) -> {block: us}
    table: dict[tuple[str, int, str], dict[int, float]] = {}
    geometry: dict[str, tuple[int, int]] = {}
    for cell in cells:
        key = (cell["shape"], cell["width"], cell["arm"])
        table.setdefault(key, {})[cell["block"]] = pick(cell, args.which)
        geometry[cell["shape"]] = (cell["outputs"], cell["hidden"])

    shapes = [s for s in dict.fromkeys(c["shape"] for c in cells) if s != CONTROL_SHAPE]
    widths = sorted({c["width"] for c in cells})
    arms = list(dict.fromkeys(c["arm"] for c in cells))

    def control(width: int, arm: str) -> float:
        blocks = table.get((CONTROL_SHAPE, width, arm))
        return statistics.median(blocks.values()) if blocks else 0.0

    # Harness defect 19, the bimodal low-clock tail: a whole timed block reads
    # far above the cell median. Flag it per block instead of pooling it into
    # the cell statistic.
    flagged: set[tuple[str, int, str, int]] = set()
    dispersion: list[dict] = []
    for (shape_k, width_k, arm_k), blocks_k in table.items():
        med = statistics.median(blocks_k.values())
        hits = [b for b, v in blocks_k.items() if v > DEFECT19_FACTOR * med]
        for b in hits:
            flagged.add((shape_k, width_k, arm_k, b))
        dispersion.append({
            "shape": shape_k,
            "width": width_k,
            "arm": arm_k,
            "median_us": med,
            "min_us": min(blocks_k.values()),
            "max_us": max(blocks_k.values()),
            "max_over_median": max(blocks_k.values()) / med if med > 0 else float("nan"),
            "n_blocks": len(blocks_k),
            "n_flagged": len(hits),
            "flagged_blocks": sorted(hits),
        })

    out: dict = {
        "source": str(args.cells),
        "estimator": args.which,
        "drop_first_block": args.drop_first_block,
        "harness": "local",
        "cool_gate_passed_real_gate": payload.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": payload.get("gate_qualified_for_timing"),
        "ramp_seconds": payload.get("ramp_seconds"),
        "shapes": {},
    }

    print("=" * 96)
    print(f"E117 M-frame table   estimator={args.which}   harness=local   "
          f"gate_qualified_for_timing=false")
    print("=" * 96)

    for shape in shapes:
        outputs, hidden = geometry[shape]
        w_bytes = packed_bytes(outputs, hidden)
        print(f"\n## {shape}  N={outputs} K={hidden}  "
              f"weight={w_bytes / 1e6:.1f} MB  grid.y={(outputs + 7) // 8}")
        header = (f"{'M':>2} {'part':>8} {'grp':>3} "
                  f"{'a_one raw us':>13} {'a_one net us':>13} {'GB/s net':>9} "
                  f"{'ctrl us':>8}")
        print(header)
        print("-" * len(header))
        shape_out: dict = {"outputs": outputs, "hidden": hidden,
                           "weight_bytes": w_bytes, "widths": {}}
        for width in widths:
            blocks = table.get((shape, width, A_ONE))
            if not blocks:
                continue
            raw = statistics.median(blocks.values())
            ctrl = control(width, A_ONE)
            net = raw - ctrl
            groups = group_count(width)
            gbs = groups * w_bytes / (net * 1e-6) / 1e9 if net > 0 else float("nan")
            part = "+".join(str(g) for g in partition(width))
            print(f"{width:>2} {part:>8} {groups:>3} {raw:>13.2f} {net:>13.2f} "
                  f"{gbs:>9.1f} {ctrl:>8.2f}")
            shape_out["widths"][width] = {
                "partition": part,
                "groups": groups,
                "a_one_raw_us": raw,
                "a_one_net_us": net,
                "a_one_net_gbs": gbs,
                "control_us": ctrl,
                "arms": {},
            }

        # Per-arm paired contrast against a_one, block by block.
        print(f"\n### percent faster than a_one, paired per block, {shape}")
        header = f"{'M':>2} {'part':>8}"
        for arm in arms:
            if arm == A_ONE:
                continue
            header += f" | {arm + ' net %':>20} {'se':>6}"
        print(header)
        print("-" * len(header))
        for width in widths:
            base = table.get((shape, width, A_ONE))
            if not base:
                continue
            row = f"{width:>2} {'+'.join(str(g) for g in partition(width)):>8}"
            for arm in arms:
                if arm == A_ONE:
                    continue
                other = table.get((shape, width, arm))
                if not other:
                    row += f" | {'-':>20} {'-':>6}"
                    continue
                base_ctrl = control(width, A_ONE)
                arm_ctrl = control(width, arm)
                shared = sorted(set(base) & set(other))
                pcts, trimmed = [], []
                for b in shared:
                    bn = base[b] - base_ctrl
                    an = other[b] - arm_ctrl
                    if bn > 0:
                        pcts.append((bn - an) / bn * 100.0)
                        if ((shape, width, A_ONE, b) not in flagged
                                and (shape, width, arm, b) not in flagged):
                            trimmed.append((bn - an) / bn * 100.0)
                raw_pcts = [
                    (base[b] - other[b]) / base[b] * 100.0 for b in shared
                ]
                mean_pct = statistics.mean(pcts) if pcts else float("nan")
                row += f" | {mean_pct:>20.3f} {sem(pcts):>6.3f}"
                entry = shape_out["widths"].get(width)
                if entry is not None:
                    entry["arms"][arm] = {
                        "net_pct_faster_mean": mean_pct,
                        "net_pct_faster_sem": sem(pcts),
                        "raw_pct_faster_mean": (
                            statistics.mean(raw_pcts) if raw_pcts else float("nan")),
                        "raw_pct_faster_sem": sem(raw_pcts),
                        "n_blocks": len(pcts),
                        "net_pct_faster_trimmed_mean": (
                            statistics.mean(trimmed) if trimmed else float("nan")),
                        "net_pct_faster_trimmed_sem": sem(trimmed),
                        "n_blocks_trimmed": len(trimmed),
                        "arm_raw_us": statistics.median(other.values()),
                        "arm_net_us": statistics.median(other.values()) - arm_ctrl,
                    }
            print(row)
        out["shapes"][shape] = shape_out

        # The rung-0 discriminator.
        w = shape_out["widths"]
        if 4 in w and 8 in w:
            two_of_four = 2 * w[4]["a_one_net_us"]
            eight = w[8]["a_one_net_us"]
            print(f"\n### rung-0 discriminator, {shape}")
            print(f"  2 x M=4 [4]          {two_of_four:>10.2f} us net")
            print(f"  M=8 [4+4]            {eight:>10.2f} us net")
            print(f"  M=8 / (2 x M=4)      {eight / two_of_four:>10.4f}")
            print(f"  M=8 saving vs 2x[4]  {(1 - eight / two_of_four) * 100:>10.2f} %")
            shape_out["discriminator"] = {
                "two_times_m4_net_us": two_of_four,
                "m8_net_us": eight,
                "m8_over_two_m4": eight / two_of_four,
                "m8_saving_vs_two_m4_pct": (1 - eight / two_of_four) * 100,
            }
            for other in (6, 7, 9):
                if other in w:
                    print(f"  rate M={other} {w[other]['partition']:>6} "
                          f"{w[other]['a_one_net_gbs']:>8.1f} GB/s   "
                          f"vs M=8 [4+4] {w[8]['a_one_net_gbs']:.1f} GB/s")

    # Rung 0b: the rate curve against N at fixed K, so the deficit can be read
    # against N, grid.y and weight bytes at the same time.
    by_hidden: dict[int, list[str]] = {}
    for shape in shapes:
        by_hidden.setdefault(geometry[shape][1], []).append(shape)
    out["n_curves"] = {}
    for hidden, group in by_hidden.items():
        if len(group) < 3:
            continue
        ordered = sorted(group, key=lambda s: geometry[s][0])
        print("\n" + "=" * 96)
        print(f"N curve at K={hidden}: a_one net GB/s per M, and the "
              f"e_nsplit_serial net % at M=8")
        print("=" * 96)
        header = (f"{'N':>7} {'grid.y':>7} {'MB':>7}"
                  + "".join(f" {'M=' + str(wd):>8}" for wd in widths)
                  + f" {'split@8 %':>10} {'se':>6}")
        print(header)
        print("-" * len(header))
        curve = []
        for shape in ordered:
            outputs, _ = geometry[shape]
            w = out["shapes"][shape]["widths"]
            row = (f"{outputs:>7} {(outputs + 7) // 8:>7} "
                   f"{packed_bytes(outputs, hidden) / 1e6:>7.1f}")
            rates = {}
            for wd in widths:
                if wd in w:
                    rates[wd] = w[wd]["a_one_net_gbs"]
                    row += f" {w[wd]['a_one_net_gbs']:>8.1f}"
                else:
                    row += f" {'-':>8}"
            split = w.get(8, {}).get("arms", {}).get("e_nsplit_serial")
            if split:
                row += (f" {split['net_pct_faster_mean']:>10.3f} "
                        f"{split['net_pct_faster_sem']:>6.3f}")
            print(row)
            curve.append({
                "shape": shape,
                "outputs": outputs,
                "grid_y": (outputs + 7) // 8,
                "weight_bytes": packed_bytes(outputs, hidden),
                "a_one_net_gbs": rates,
                "m8_split_net_pct": (
                    split["net_pct_faster_mean"] if split else None),
                "m8_split_net_sem": split["net_pct_faster_sem"] if split else None,
            })
        out["n_curves"][hidden] = curve
        print("\nnote: at fixed K, grid.y = N/8 and weight bytes are both "
              "proportional to N, so this sweep cannot separate them.")

    # Rung 1: price each in-graph serialisation route against the blocking-eval
    # ceiling, in microseconds per occurrence rather than in percent.
    routes = [a for a in arms if a not in (A_ONE, "c_nsplit", "e_nsplit_serial")]
    if routes:
        print("\n" + "=" * 96)
        print("rung 1: cost of causing the barrier, microseconds per occurrence")
        print("=" * 96)
        print("price = (ceiling % - route %) / 100 x a_one net us, where the "
              "ceiling is e_nsplit_serial,")
        print("whose blocking host eval is removed by the two-eval control.")
        out["rung1"] = {}
        for shape in shapes:
            w = out["shapes"][shape]["widths"]
            for width in widths:
                entry = w.get(width)
                if not entry:
                    continue
                ceiling = entry["arms"].get("e_nsplit_serial")
                if not ceiling:
                    continue
                base_us = entry["a_one_net_us"]
                ceil_pct = ceiling["net_pct_faster_mean"]
                print(f"\n## {shape} M={width} {entry['partition']}   "
                      f"a_one net {base_us:.2f} us   "
                      f"ceiling {ceil_pct:+.3f} %")
                header = (f"{'route':>18} {'net us':>10} {'net %':>9} {'se':>6} "
                          f"{'price us':>9} {'round us @48':>13}")
                print(header)
                print("-" * len(header))
                rows = {}
                for arm in ["c_nsplit", "e_nsplit_serial"] + routes:
                    got = entry["arms"].get(arm)
                    if not got:
                        continue
                    price = (ceil_pct - got["net_pct_faster_mean"]) / 100.0 * base_us
                    gain_round = got["net_pct_faster_mean"] / 100.0 * base_us * 48
                    print(f"{arm:>18} {got['arm_net_us']:>10.2f} "
                          f"{got['net_pct_faster_mean']:>9.3f} "
                          f"{got['net_pct_faster_sem']:>6.3f} {price:>9.2f} "
                          f"{gain_round:>13.1f}")
                    rows[arm] = {
                        "net_us": got["arm_net_us"],
                        "net_pct_faster": got["net_pct_faster_mean"],
                        "net_pct_faster_sem": got["net_pct_faster_sem"],
                        "price_us_vs_ceiling": price,
                        "round_gain_us_at_48_layers": gain_round,
                    }
                out["rung1"][f"{shape}|M{width}"] = {
                    "a_one_net_us": base_us,
                    "ceiling_pct": ceil_pct,
                    "routes": rows,
                }

    # Harness defect 19: per-block dispersion for every cell.
    print("\n" + "=" * 96)
    print(f"harness defect 19: blocks above {DEFECT19_FACTOR} x the cell median")
    print("=" * 96)
    out["defect19_dispersion"] = dispersion
    hot = sorted((d for d in dispersion if d["n_flagged"]),
                 key=lambda d: -d["max_over_median"])
    print(f"cells={len(dispersion)} cells_with_flagged_block={len(hot)} "
          f"flagged_blocks={sum(d['n_flagged'] for d in dispersion)}")
    print(f"{'shape':>16} {'M':>3} {'arm':>18} {'median us':>10} "
          f"{'max us':>10} {'max/med':>8} {'flagged':>8} {'n':>3}")
    for d in hot:
        print(f"{d['shape']:>16} {d['width']:>3} {d['arm']:>18} "
              f"{d['median_us']:>10.2f} {d['max_us']:>10.2f} "
              f"{d['max_over_median']:>8.2f} "
              f"{str(d['flagged_blocks']):>8} {d['n_blocks']:>3}")

    # Harness-defect-16 proof: residual forward-versus-reverse gap per arm.
    print("\n" + "=" * 96)
    print("harness defect 16 residual: forward-versus-reverse gap per arm, "
          "percent of the reverse reading")
    print("=" * 96)
    # The median is the campaign convention for this statistic. The mean is
    # reported beside it because a handful of whole cells in the two small
    # tensors are interrupted by something external and read three to four times
    # the reverse value, which is a different phenomenon from the systematic
    # first-position ramp bias this fix targets.
    gaps: dict[str, list[float]] = {}
    per_shape: dict[tuple[str, str], list[float]] = {}
    for cell in cells:
        if cell["shape"] == CONTROL_SHAPE:
            continue
        rev = cell["reverse_us"]
        if rev > 0:
            gap = (cell["forward_us"] - rev) / rev * 100.0
            gaps.setdefault(cell["arm"], []).append(gap)
            per_shape.setdefault((cell["shape"], cell["arm"]), []).append(gap)
    print(f"{'arm':>18} {'median %':>9} {'mean %':>9} {'sem':>7} "
          f"{'min %':>9} {'max %':>9} {'n':>4}")
    out["defect16_residual"] = {}
    for arm, values in gaps.items():
        print(f"{arm:>18} {statistics.median(values):>9.3f} "
              f"{statistics.mean(values):>9.3f} {sem(values):>7.3f} "
              f"{min(values):>9.3f} {max(values):>9.3f} {len(values):>4}")
        out["defect16_residual"][arm] = {
            "median_pct": statistics.median(values),
            "mean_pct": statistics.mean(values),
            "sem_pct": sem(values),
            "min_pct": min(values),
            "max_pct": max(values),
            "n": len(values),
        }
    print(f"\n{'shape':>16} {'arm':>18} {'median %':>9} {'max %':>9} {'n':>4}")
    out["defect16_residual_by_shape"] = {}
    for (shape, arm), values in sorted(per_shape.items()):
        print(f"{shape:>16} {arm:>18} {statistics.median(values):>9.3f} "
              f"{max(values):>9.3f} {len(values):>4}")
        out["defect16_residual_by_shape"][f"{shape}|{arm}"] = {
            "median_pct": statistics.median(values),
            "max_pct": max(values),
            "n": len(values),
        }

    exact = payload.get("exactness", [])
    bad = [e for e in exact if not e.get("nsplit_bit_exact")]
    nocontrol = [e for e in exact if not e.get("positive_control_differs")]
    print(f"\nexactness cells={len(exact)} not_bit_exact={len(bad)} "
          f"positive_control_failed={len(nocontrol)}")
    out["exactness"] = {
        "cells": len(exact),
        "not_bit_exact": len(bad),
        "positive_control_failed": len(nocontrol),
    }
    for record in payload.get("slice_aliasing", []):
        print(f"slice_aliasing {record['shape']} delta_bytes={record['delta_bytes']} "
              f"full_tensor_bytes={record['full_tensor_bytes']}")
    out["slice_aliasing"] = payload.get("slice_aliasing", [])

    if args.json:
        args.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0 if not bad and not nocontrol else 1


if __name__ == "__main__":
    sys.exit(main())
