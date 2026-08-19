#!/usr/bin/env python3
"""Turn the E32 grid into the three deliverables: the ladder, the model fit, the decision.

  python3 research/e32_analysis.py /tmp/e32_air/e32-rps-grid.json
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys

# quantized.cpp:251-254, NOT editable: bn = 8 output rows per threadgroup,
# group_dims(32, 2, 1) = 64 threads = 2 simdgroups, grid.y = (N + bn - 1) / bn.
BN = 8
NUM_SIMDGROUPS = 2
MLP_N = 17408
# Per-lane, per-k-block op counts for a 4-row tile, from research/crossrow-closure.md
# section 3. The x-side term is per-lane and independent of the row count, so it is
# the only term that repeats when a tile is covered in 4/r sequential row blocks.
W_SIDE = 128
CORE_FMA_PER_NA = 80
X_SIDE_PER_NA = 32
EPILOGUE_PER_NA = 12


def alu_per_tile(na: int, r: int) -> int:
    blocks = 4 // r
    return W_SIDE + (CORE_FMA_PER_NA + EPILOGUE_PER_NA) * na + blocks * X_SIDE_PER_NA * na


def fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    worst = max(abs(y - (intercept + slope * x)) for x, y in points)
    return intercept, slope, worst


def main() -> None:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/e32_air/e32-rps-grid.json")
    data = json.loads(path.read_text())
    cells = {(c["arm"], c["na"], c["r"]): c for c in data["cells"] if c["status"] == "ok"}
    relaxed = {(na, r): c for (arm, na, r), c in cells.items() if arm == "grid_relaxed"}
    blocked = {(na, r): c for (arm, na, r), c in cells.items() if arm == "coverage_preserving"}

    if data["gate_validation_failures"]:
        sys.exit(f"gate validation failed: {data['gate_validation_failures']}")

    print("## (a) Register/spill grid -- one body per simdgroup pass (grid_relaxed)\n")
    print("| NA | r=1 | r=2 | r=3 | r=4 |")
    print("|---|---|---|---|---|")
    for na in sorted({na for na, _ in relaxed}):
        row = [f"| {na} "]
        for r in (1, 2, 3, 4):
            c = relaxed.get((na, r))
            if c is None:
                row.append("| - ")
                continue
            mark = " SPILL" if c["acc_spill"] else ""
            row.append(f"| {c['peak_live_regs']}/{c['allocas']}{mark} ")
        print("".join(row) + "|")

    print("\n## Max spill-free NA per rows_per_simd\n")
    max_na = {}
    for r in (1, 2, 3, 4):
        clean = [na for (na, rr), c in relaxed.items() if rr == r and not c["acc_spill"]]
        spilled = [na for (na, rr), c in relaxed.items() if rr == r and c["acc_spill"]]
        max_na[r] = max(clean) if clean else 0
        bound = f"{min(spilled)} spills" if spilled else f">{max_na[r]} untested"
        print(f"  r={r}: spill-free through NA={max_na[r]}  ({bound})")

    print("\n## (b) Threadgroups to cover the MLP N=17408, and what is legal\n")
    print(f"  frozen host grid: bn={BN}, group_dims(32,{NUM_SIMDGROUPS},1), "
          f"grid.y = (N+{BN - 1})/{BN} = {(MLP_N + BN - 1) // BN} threadgroups")
    print("| r | rows/threadgroup | threadgroups needed | frozen grid launches | legal? |")
    print("|---|---|---|---|---|")
    frozen = (MLP_N + BN - 1) // BN
    for r in (1, 2, 3, 4):
        rows_tg = NUM_SIMDGROUPS * r
        needed = MLP_N / rows_tg
        need_s = str(int(needed)) if needed == int(needed) else f"{needed:.2f}"
        legal = "yes" if rows_tg == BN else f"NO - covers {frozen * rows_tg} of {MLP_N} rows"
        print(f"| {r} | {rows_tg} | {need_s} | {frozen} | {legal} |")

    print("\n## Coverage-preserving form: r rows per block x 4/r blocks, frozen grid intact\n")
    print("| NA | r | regs/allocas | spill | AIR lines vs 1-block | ALU/tile |")
    print("|---|---|---|---|---|---|")
    for na in sorted({na for na, _ in blocked}):
        for r in (1, 2, 4):
            c = blocked.get((na, r))
            if c is None:
                continue
            one = relaxed.get((na, r))
            ratio = f"{c['air_lines'] / one['air_lines']:.2f}x" if one else "-"
            print(f"| {na} | {r} | {c['peak_live_regs']}/{c['allocas']} | "
                  f"{'SPILL' if c['acc_spill'] else 'clean'} | {ratio} | {alu_per_tile(na, r)} |")

    print("\n## (3) Register model: is it affine in the product rows_per_simd x NA?\n")
    product_pts = [(na * r, c["peak_live_regs"]) for (na, r), c in relaxed.items()
                   if not c["acc_spill"]]
    b0, b1, worst = fit(product_pts)
    print(f"  product model  regs = {b0:.1f} + {b1:.2f}*(r*NA)   max |residual| = {worst:.1f}")
    print(f"  advisor predicted regs(NA=9, r=2) ~ 125; product fit says "
          f"{b0 + b1 * 18:.0f}; MEASURED {relaxed[(9, 2)]['peak_live_regs']}")
    print("\n  same product, different verdict:")
    for (na, r) in ((6, 4), (12, 2)):
        c = relaxed.get((na, r))
        if c:
            print(f"    NA={na} r={r}: {na * r} accumulator floats, {c['peak_live_regs']} regs, "
                  f"{'SPILL' if c['acc_spill'] else 'clean'}")

    print("\n  per-r NA slope (registers per unit of NA at fixed r):")
    slopes = {}
    for r in (1, 2, 3, 4):
        pts = [(na, c["peak_live_regs"]) for (na, rr), c in relaxed.items()
               if rr == r and not c["acc_spill"]]
        if len(pts) < 3:
            continue
        i0, s0, w0 = fit(pts)
        slopes[r] = s0
        print(f"    r={r}: regs = {i0:.1f} + {s0:.2f}*NA   max |residual| = {w0:.2f}")
    if len(slopes) >= 3:
        i2, s2, w2 = fit(list(slopes.items()))
        print(f"\n  slope(r) = {i2:.2f} + {s2:.2f}*r  (max |residual| {w2:.2f})")
        print(f"    NA-only registers per NA:      {i2:.2f}  <- a0..a3 + sums, 5 floats/NA")
        print(f"    per-row registers per NA:      {s2:.2f}  <- acc + partial, 2 floats/NA/row")
        print(f"    ratio {i2 / s2:.2f} against the 5:2 live-float ratio = {5 / 2:.2f}")

    print("\n## (c) DECISION: best legal cell per M, frozen grid respected\n")
    print("| M | shipped IPG/passes | best legal (NA=IPG, r) | M%IPG!=1 | passes | "
          "spill | ALU/tile vs shipped |")
    print("|---|---|---|---|---|---|---|")
    shipped_ipg = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}
    decisions = []
    for m in range(3, 10):
        sipg = shipped_ipg[m]
        spasses = math.ceil(m / sipg)
        shipped_alu = sum(alu_per_tile(min(sipg, m - g * sipg), 4)
                          for g in range(spasses))
        best = None
        for ipg, r in itertools.product(range(2, m + 1), (1, 2, 4)):
            if m % ipg == 1 or ipg > max_na[r]:
                continue
            c = blocked.get((ipg, r)) or relaxed.get((ipg, r))
            if c is None or c["acc_spill"]:
                continue
            passes = math.ceil(m / ipg)
            alu = sum(alu_per_tile(min(ipg, m - g * ipg), r) for g in range(passes))
            key = (passes, alu)
            if best is None or key < best[0]:
                best = (key, ipg, r, passes, alu, c)
        (_, ipg, r, passes, alu, c) = best
        decisions.append({"M": m, "ipg": ipg, "r": r, "passes": passes,
                          "shipped_passes": spasses, "alu": alu, "shipped_alu": shipped_alu})
        print(f"| {m} | {sipg}/{spasses} | NA={ipg}, r={r} | {m % ipg != 1} | {passes} | "
              f"{'SPILL' if c['acc_spill'] else 'clean'} | "
              f"{alu} vs {shipped_alu} ({alu / shipped_alu - 1:+.1%} ALU, "
              f"{passes / spasses - 1:+.0%} weight passes) |")

    hist = {1: 1, 3: 5, 4: 5, 5: 23, 6: 4, 7: 6, 8: 34}
    rounds = sum(hist.values())
    improved = sum(n for depth, n in hist.items()
                   if any(d["M"] == depth + 1 and d["passes"] < d["shipped_passes"]
                          for d in decisions))
    print(f"\n  rounds whose weight-pass count falls: {improved}/{rounds} "
          f"({improved / rounds:.0%}) on the advisor-quoted depth histogram")


if __name__ == "__main__":
    main()
