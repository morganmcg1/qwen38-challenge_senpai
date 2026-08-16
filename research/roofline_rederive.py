#!/usr/bin/env python3
"""Re-derive the aggregate `nominal x M` table recorded in ESTABLISHED_FACTS FACT 1.

Reads a qmv-curve summary.json and reports, for every plausible aggregation rule,
the per-width aggregate of nominal GB/s and of nominal GB/s * M.  The point is to
find which rule (if any) reproduces the advisor's numbers, and to quantify the
per-shape spread that a single aggregate number hides.
"""

import json
import statistics
import sys
from collections import defaultdict


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        ".mlxfast-private/qmv-curve/e7-na4-base/summary.json"
    )
    doc = load(path)
    rows = doc["per_shape_curve"]

    by_m = defaultdict(list)
    by_shape = defaultdict(dict)
    for r in rows:
        m = r["m"]
        by_m[m].append(r)
        by_shape[r["name"]][m] = r

    widths = sorted(by_m)
    print(f"source: {path}")
    print(f"shapes: {len(by_shape)}  widths: {widths}")
    print()

    keys = [k for k in rows[0] if "gbps" in k or "bandwidth" in k]
    print(f"bandwidth-like keys present: {keys}")
    print()

    for key in keys:
        print(f"=== {key} ===")
        hdr = "M   median      mean        wsum        median*M    mean*M      wsum*M"
        print(hdr)
        for m in widths:
            vals = [r[key] for r in by_m[m] if r[key] is not None]
            if not vals:
                continue
            calls = [r.get("calls_per_verify", 1.0) for r in by_m[m]]
            med = statistics.median(vals)
            mean = statistics.fmean(vals)
            wsum = sum(vals)
            print(
                f"{m}   {med:9.2f}   {mean:9.2f}   {wsum:9.2f}   "
                f"{med * m:9.2f}   {mean * m:9.2f}   {wsum * m:9.2f}"
            )
        print()

    key = "gbps_nominal" if "gbps_nominal" in keys else keys[0]
    print(f"=== per-shape spread of ({key} * M) over M=4..9 ===")
    print("shape                              min     max    max/min")
    for shape, per_m in sorted(by_shape.items()):
        vals = [per_m[m][key] * m for m in widths if 4 <= m <= 9 and m in per_m]
        if not vals:
            continue
        print(f"{shape:34s} {min(vals):7.1f} {max(vals):7.1f} {max(vals)/min(vals):8.3f}")
    print()

    print(f"=== aggregate spread of ({key} * M) over M=4..9 ===")
    for label, fn in (("median", statistics.median), ("mean", statistics.fmean)):
        vals = [fn([r[key] for r in by_m[m]]) * m for m in widths if 4 <= m <= 9]
        lo, hi = min(vals), max(vals)
        mid = statistics.fmean(vals)
        print(
            f"{label:7s} min={lo:8.2f} max={hi:8.2f} mean={mid:8.2f} "
            f"half-range=+/-{100 * (hi - lo) / 2 / mid:5.2f}%"
        )


if __name__ == "__main__":
    main()
