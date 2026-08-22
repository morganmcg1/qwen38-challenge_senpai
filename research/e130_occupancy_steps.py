#!/usr/bin/env python3
"""Find the discrete occupancy steps in the E130 positive-control measurement.

    usage: research/e130_occupancy_steps.py MEASURED.json [--out PATH]

`research/e130_occupancy_control.py` measured peak concurrent threadgroups
against declared register pressure. The headline reading was that the smooth
register-floor law over-predicts the fall by up to 0.64 in ratio. That is the
wrong shape to fit. Apple GPUs allocate registers in granular blocks, so
occupancy should fall in discrete steps and sit flat between them.

This tool fits the step structure instead of a smooth law, and then asks the
only question the E130 arm depends on: does a step boundary lie between the
arm's two register counts, 101 before and 90 after?

harness=local. This runs on the local Mac, not on the ranked g17s runner, so it
transfers only as a mechanism check and never as a ranked measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st

ARM_REGISTERS_BEFORE = 101
ARM_REGISTERS_AFTER = 90

# A step is declared when the median peak concurrency changes by more than this
# fraction between neighbouring register counts.
STEP_FRACTION = 0.12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("measured", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    doc = json.loads(args.measured.read_text())
    device = doc.get("device")
    print("harness=local")
    print("device %s   threadgroups dispatched %s   threads per group %s   "
          "replicates %s" % (device, doc.get("threadgroups_dispatched"),
                             doc.get("threads_per_threadgroup"),
                             doc.get("replicates")))

    census = doc.get("census") or {}
    by_reg: dict[int, list[float]] = {}
    g17s_of_g16s: dict[int, int] = {}
    for row in doc["floor_law_test"]["rows"]:
        regs = int(row["registers"])
        by_reg.setdefault(regs, []).append(
            float(row["peak_concurrent_threadgroups"]))
        entry = census.get(str(row.get("ballast")))
        if entry and "applegpu_g17s" in entry:
            g17s_of_g16s[regs] = int(entry["applegpu_g17s"]["registers"])

    regs_sorted = sorted(by_reg)
    print("\nregister sweep %d..%d, %d distinct counts"
          % (regs_sorted[0], regs_sorted[-1], len(regs_sorted)))
    print("\n%6s %6s %10s %8s %8s %s"
          % ("g16s", "g17s", "median", "min", "max", "n"))
    print("-" * 52)
    medians = {}
    for regs in regs_sorted:
        vals = by_reg[regs]
        medians[regs] = st.median(vals)
        print("%6d %6s %10.1f %8.1f %8.1f %d"
              % (regs, g17s_of_g16s.get(regs, "-"), medians[regs],
                 min(vals), max(vals), len(vals)))

    print("\nSTEP DETECTION. a step is a change above %.0f %% between "
          "neighbouring counts." % (100.0 * STEP_FRACTION))
    steps = []
    for a, b in zip(regs_sorted, regs_sorted[1:]):
        lo, hi = medians[a], medians[b]
        if lo <= 0:
            continue
        change = (hi - lo) / lo
        if abs(change) > STEP_FRACTION:
            steps.append({"from_registers": a, "to_registers": b,
                          "from_peak": lo, "to_peak": hi,
                          "relative_change": change})
            print("  step between %3d and %3d registers: %.1f -> %.1f "
                  "threadgroups, %+.1f %%"
                  % (a, b, lo, hi, 100.0 * change))
    if not steps:
        print("  none")

    print("\nTHE ARM'S QUESTION. does a step separate %d registers from %d?"
          % (ARM_REGISTERS_BEFORE, ARM_REGISTERS_AFTER))
    covered = [r for r in regs_sorted
               if ARM_REGISTERS_AFTER <= r <= ARM_REGISTERS_BEFORE]
    inside = [s for s in steps
              if ARM_REGISTERS_AFTER <= s["from_registers"]
              and s["to_registers"] <= ARM_REGISTERS_BEFORE]
    print("  measured register counts inside the arm's interval: %s"
          % (covered if covered else "none"))
    print("  the sweep reaches %d registers; the arm's upper point is %d."
          % (regs_sorted[-1], ARM_REGISTERS_BEFORE))
    answered = (regs_sorted[-1] >= ARM_REGISTERS_BEFORE
                and any(r <= ARM_REGISTERS_AFTER for r in regs_sorted))
    print("  interval fully covered by the sweep: %s" % answered)
    print("  steps found inside the covered part of the interval: %d"
          % len(inside))
    for s in inside:
        print("    %d -> %d registers, %+.1f %%"
              % (s["from_registers"], s["to_registers"],
                 100.0 * s["relative_change"]))

    verdict: str
    if not answered:
        verdict = ("UNANSWERED. the sweep does not reach %d registers, so this "
                   "measurement cannot say whether the arm crosses a step."
                   % ARM_REGISTERS_BEFORE)
    elif inside:
        verdict = ("the arm crosses %d measured step(s) on this device."
                   % len(inside))
    else:
        verdict = ("the arm crosses no step on this device: %d and %d "
                   "registers sit on the same occupancy plateau, so the "
                   "predicted residency gain does not appear here."
                   % (ARM_REGISTERS_BEFORE, ARM_REGISTERS_AFTER))
    print("\nVERDICT. %s" % verdict)
    print("\nTRANSFER LIMIT. this device is %s. the ranked runner is g17s on")
    print("M5. register block granularity and the register file per core are")
    print("device properties, so a plateau here does not prove a plateau")
    print("there, and a step here does not prove a step there.")

    report = {
        "harness": "local",
        "device": device,
        "threadgroups_dispatched": doc.get("threadgroups_dispatched"),
        "threads_per_threadgroup": doc.get("threads_per_threadgroup"),
        "replicates": doc.get("replicates"),
        "median_peak_by_registers": {str(k): v for k, v in medians.items()},
        "steps": steps,
        "arm_registers_before": ARM_REGISTERS_BEFORE,
        "arm_registers_after": ARM_REGISTERS_AFTER,
        "arm_interval_covered_by_sweep": answered,
        "steps_inside_arm_interval": inside,
        "verdict": verdict,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
