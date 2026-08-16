#!/usr/bin/env python3
"""Re-derive the aggregate `nominal x M` table recorded in ESTABLISHED_FACTS FACT 1.

Reads a qmv-curve summary.json and reports, for every plausible aggregation rule,
the per-width aggregate of nominal GB/s and of nominal GB/s * M.  The point is to
find which rule (if any) reproduces the advisor's numbers, and to quantify the
per-shape spread that a single aggregate number hides.
"""

import json
import math
import statistics
import sys
from collections import defaultdict


def load(path):
    with open(path) as fh:
        return json.load(fh)


def rel_sd(vals):
    return 100 * statistics.stdev(vals) / statistics.fmean(vals)


def three_way(label, ms, nominal):
    """Score the three competing invariants, not just the advisor's two.

    H_A  nominal          flat  <=> t independent of M   (perfect cross-row reuse)
    H_C  nominal*ceil(M/4) flat <=> t ~ ceil(M/4)        (integer weight streams)
    H_B  nominal*M        flat  <=> t ~ M                (zero cross-row reuse)

    roofline_regime_check.py compares only H_A and H_B and then declares the
    memory-side lever family dead -- but H_C *is* the memory-side model, and it
    is never on the ballot.
    """
    a = rel_sd(nominal)
    c = rel_sd([g * math.ceil(m / 4) for g, m in zip(nominal, ms)])
    b = rel_sd([g * m for g, m in zip(nominal, ms)])
    best = min((a, "H_A nominal"), (c, "H_C nominal*ceil(M/4)"), (b, "H_B nominal*M"))
    print(f"{label:32s} n={len(ms):2d}  H_A {a:5.1f}%  H_C {c:5.1f}%  H_B {b:5.1f}%"
          f"   -> {best[1]}  (B vs C: {c / b:.1f}x, B vs A: {a / b:.1f}x)")
    return a, c, b


def specificity_check():
    """What does the advisor's A-vs-B rule report when H_C is TRUE by construction?"""
    print("=== specificity of the H_A-vs-H_B decision rule ===")
    print("Synthetic ground truth, no noise: t = ceil(M/4) exactly (pure")
    print("bandwidth-bound integer-stream). nominal := 1/t.")
    for ms in ([4, 5, 8, 9], list(range(4, 10)), list(range(1, 10))):
        nominal = [1.0 / math.ceil(m / 4) for m in ms]
        a, c, b = three_way(f"  synthetic H_C, M={ms}", ms, nominal)
        rule = "ALU-bound" if b < a else "bandwidth-bound"
        print(f"{'':32s}      advisor rule reports: {rule}"
              f"  ({a / b:.1f}x 'tighter')")
    print()


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
    print()

    specificity_check()

    print("=== three-way invariant test on real data ===")
    advisor_ms = [4, 5, 8, 9]
    advisor_corrected = [165.6, 262.1, 183.0, 239.5]
    three_way(
        "advisor 4 points (FACT 1)",
        advisor_ms,
        [c / math.ceil(m / 4) for c, m in zip(advisor_corrected, advisor_ms)],
    )
    for lo in (4, 1):
        ms = [m for m in widths if lo <= m <= 9]
        med = [statistics.median([r[key] for r in by_m[m]]) for m in ms]
        three_way(f"this run, 8-shape median M={lo}..9", ms, med)
    for shape, per_m in sorted(by_shape.items()):
        ms = [m for m in widths if 4 <= m <= 9 and m in per_m]
        three_way(f"  {shape}", ms, [per_m[m][key] for m in ms])


if __name__ == "__main__":
    main()
