#!/usr/bin/env python3
"""Read the E95 direct qmv width probe and decide what `b` is.

The verify width model is `verify_us = a + b*G + c*M`, with
`G = ceil(M / 4)` the number of input groups the WIDE affine-4 kernel runs
over one weight tensor. The fit gives `b = 27,377 us`. Read as one pass over
the 14,412 MB of affine-4 weights the verify phase touches, that is
526.4 GB/s, which is 1.99x the 265 GB/s DRAM read ceiling of this host. A
model term cannot describe traffic that the memory system cannot carry, so
either `b` is not a weight pass, or the `b`/`c` split is not identified.

This script reads `research/out/e95_qmv_probe.json`, produced by
`Tests/MLXFastTests/E95QmvWidthProbeTests.swift`, and answers three
questions with the same tensor, the same packed bytes and no external
bandwidth figure:

  1. What read rate does each working set actually reach on this host?
  2. What does the G step from 1 to 2 cost, in us and as a fraction of one
     measured pass over that tensor?
  3. Is the G-step cost proportional to the bytes, which would make `b`
     traffic, or roughly flat, which would make it launch and setup work?

Usage: python3 research/e95_qmv_probe_analysis.py [path]
"""

from __future__ import annotations

import json
import pathlib
import sys

INPUTS_PER_GROUP = 4


def groups(width: int) -> int:
    return -(-width // INPUTS_PER_GROUP)


def main() -> int:
    path = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else "research/out/e95_qmv_probe.json")
    payload = json.loads(path.read_text())

    by_tensor: dict[int, dict[int, float]] = {}
    bytes_of: dict[int, int] = {}
    for cell in payload["cells"]:
        by_tensor.setdefault(cell["outputs"], {})[cell["m"]] = cell["microseconds"]
        bytes_of[cell["outputs"]] = cell["packed_bytes"]

    read_rate = {int(k): v for k, v in payload["read_gb_s"].items()}

    summary: dict[int, dict[str, float]] = {}
    for outputs in sorted(by_tensor, reverse=True):
        cells = by_tensor[outputs]
        nbytes = bytes_of[outputs]
        rate = read_rate[outputs]
        one_pass_us = nbytes / rate / 1e3

        print(f"\n=== O={outputs}  packed={nbytes / 1e6:.1f} MB ===")
        print(f"measured read rate        : {rate:.1f} GB/s")
        print(f"one measured pass         : {one_pass_us:.1f} us")
        print(f"{'M':>3} {'G':>2} {'us':>9} {'us/pass':>8} {'GB/s@1pass':>11}")
        for width in sorted(cells):
            us = cells[width]
            print(
                f"{width:>3} {groups(width):>2} {us:>9.2f} "
                f"{us / one_pass_us:>8.3f} {nbytes / us / 1e3:>11.1f}")

        # The G step: mean of the G==2 plateau minus mean of the G==1 plateau.
        g1 = [cells[m] for m in cells if groups(m) == 1]
        g2 = [cells[m] for m in cells if groups(m) == 2]
        step = sum(g2) / len(g2) - sum(g1) / len(g1)
        # Slope inside a plateau isolates the per-input term `c`.
        inside = sorted(m for m in cells if groups(m) == 1)
        slope = (cells[inside[-1]] - cells[inside[0]]) / (inside[-1] - inside[0])
        summary[outputs] = {
            "bytes": nbytes,
            "one_pass_us": one_pass_us,
            "step_us": step,
            "step_over_pass": step / one_pass_us,
            "step_ns_per_mb": step * 1e3 / (nbytes / 1e6),
            "within_group_slope_us": slope,
        }
        print(f"G 1->2 step               : {step:.2f} us")
        print(f"  as a fraction of a pass : {step / one_pass_us:.3f}")
        print(f"  per packed MB           : {step * 1e3 / (nbytes / 1e6):.1f} ns/MB")
        print(f"within-group slope per M  : {slope:.2f} us")

    big, small = sorted(summary, reverse=True)
    ratio_bytes = summary[big]["bytes"] / summary[small]["bytes"]
    ratio_step = summary[big]["step_us"] / summary[small]["step_us"]
    print("\n=== verdict ===")
    print(f"bytes ratio big/small     : {ratio_bytes:.2f}x")
    print(f"G-step ratio big/small    : {ratio_step:.2f}x")
    print(
        "if the step scales with bytes it is traffic; "
        "if it is flat it is launch and setup work")
    print(
        f"big  step is {summary[big]['step_over_pass']:.3f} of one measured pass")
    print(
        f"small step is {summary[small]['step_over_pass']:.3f} of one measured pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
