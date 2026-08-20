#!/usr/bin/env python3
"""E82 rung 4: which bytes law predicted the `master-bf16` head step?

Before the eight-leg session ran, only `declared` and `pinned` had measured
head steps. Every candidate bytes law was therefore calibrated on those two
arms, and `master-bf16` was a genuine out-of-sample point: 1,006,736,384
artifact bytes, no derived readout, and no island bytes.

This script refits each law on the two calibration arms alone and prints the
prediction it would have made, beside the measurement. A law that was fitted on
all three arms cannot be scored here and is reported as such.

  python3 research/e82_prediction_test.py
"""

from __future__ import annotations

import json
from pathlib import Path

COST = json.loads(Path("research/e82-head-cost-session.json").read_text())
BYTES = json.loads(Path("research/e82-head-bytes.json").read_text())

CALIBRATION = ("declared", "pinned")
HELD_OUT = "master-bf16"


def bytes_of(arm: str, key: str) -> int:
    e = BYTES["arms"][arm]
    if key == "artifact":
        return e["tensor_bytes"]
    if key == "traffic":
        return e["traffic_bytes_per_draft"]
    if key == "effective":
        return e["head_stream_bytes"]
    raise KeyError(key)


def main() -> None:
    ms = {a: v["head_phase_ms_per_draft_median"] for a, v in COST["by_arm"].items()}
    rows = []
    for key in ("artifact", "traffic", "effective"):
        b = {a: bytes_of(a, key) for a in (*CALIBRATION, HELD_OUT)}
        lo, hi = CALIBRATION
        slope = (ms[hi] - ms[lo]) / (b[hi] - b[lo])
        intercept = ms[lo] - slope * b[lo]
        rows.append({
            "law": f"{key} bytes, affine two-point",
            "gb_per_s": 1e-6 / slope,
            "fixed_ms": intercept,
            "predicted_ms": intercept + slope * b[HELD_OUT],
        })
        # The advisor's own form: bytes per millisecond taken from one arm and
        # applied proportionally, with no fixed cost.
        for anchor in CALIBRATION:
            rate = b[anchor] / ms[anchor]
            rows.append({
                "law": f"{key} bytes, proportional from {anchor}",
                "gb_per_s": rate * 1e-6,
                "fixed_ms": 0.0,
                "predicted_ms": b[HELD_OUT] / rate,
            })

    measured = ms[HELD_OUT]
    print(f"{HELD_OUT} measured head step: {measured:.3f} ms/draft"
          f"  (legs {COST['by_arm'][HELD_OUT]['head_phase_ms_per_draft_legs']})")
    print(f"calibrated on {CALIBRATION[0]} {ms[CALIBRATION[0]]:.3f} ms"
          f" and {CALIBRATION[1]} {ms[CALIBRATION[1]]:.3f} ms\n")
    print("law".ljust(44) + "GB/s".rjust(8) + "fixed ms".rjust(10)
          + "predicted".rjust(11) + "error".rjust(9))
    for r in sorted(rows, key=lambda r: abs(r["predicted_ms"] - measured)):
        err = 100 * (r["predicted_ms"] - measured) / measured
        print(f"{r['law'].ljust(44)}{r['gb_per_s']:8.1f}{r['fixed_ms']:10.3f}"
              f"{r['predicted_ms']:11.3f}{err:+8.1f}%")

    pf = COST.get("precision_fit")
    if not pf:
        return
    print(f"\nprecision-class model: fitted on all three arms,"
          f" dof={pf['degrees_of_freedom']}, so it makes no out-of-sample"
          " prediction here.")
    for c in pf["classes"]:
        print(f"    {c.ljust(6)}{pf['effective_gb_per_s'][c]:8.1f} GB/s")


if __name__ == "__main__":
    main()
