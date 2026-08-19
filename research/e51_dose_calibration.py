#!/usr/bin/env python3
"""E51: turn the leg-parity ladder into a calibrated dose scale.

`research/e51_leg_parity.py` reports, per (shape, width), the fraction of
affine-4/g64 output elements at which the MTP leg stops matching the serial leg
bitwise. The `d:<n>` arms keep R0's BF16 expression tree verbatim and scale its
fp32 result by `1 + 2^-n`, so they trace that fraction against a KNOWN relative
perturbation of the bias-sum term. This script:

  1. prints the dose-response curve, including the zero-dose null controls
     (`1.0f + 0x1p-24f` rounds to exactly 1.0f in fp32, so d:24 and beyond are
     literally no dose and must not fire);
  2. reads the comparator's minimum detectable effect off that curve; and
  3. converts each real arm's observed divergence fraction into an EQUIVALENT
     relative dose by log-log interpolation between the bracketing rungs, which
     is what makes "the dose is far above the MDE" a number instead of a claim.

Research-only.

    research/e51_dose_calibration.py LADDER.json [MORE.json ...]
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

REAL_ARMS = ("r1", "r2")


def load(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()
    for path in paths:
        payload = json.loads(pathlib.Path(path).read_text())
        for entry in payload["entries"]:
            token = (entry["arm"], entry["shape"], entry["m"])
            if token in seen:
                continue
            seen.add(token)
            rows.append(entry)
    return rows


def main(paths: list[str]) -> int:
    rows = load(paths)
    cells = sorted({(e["shape"], e["m"]) for e in rows})
    doses = sorted({int(e["arm"][2:]) for e in rows if e["arm"].startswith("d:")},
                   reverse=True)
    index = {(e["arm"], e["shape"], e["m"]): e for e in rows}

    def frac(arm: str, cell: tuple) -> float | None:
        entry = index.get((arm, cell[0], cell[1]))
        return None if entry is None else entry["legs"]["mismatch_fraction"]

    print(f"cells: {len(cells)}   arms: {sorted({e['arm'] for e in rows})}")
    print("\ndose-response: relative perturbation of the fp32 bias-sum term ->"
          " legs-divergence fraction")
    for n in doses:
        values = [f for f in (frac(f"d:{n}", c) for c in cells) if f is not None]
        fired = sum(1 for f in values if f > 0)
        mean = sum(values) / len(values)
        note = "  ZERO DOSE (multiplier rounds to 1.0f)" if n >= 24 else ""
        print(f"  2^-{n:<3} = {2.0 ** -n:9.3e}   fired {fired:2d}/{len(values)}"
              f"   mean frac {mean:.4g}{note}")

    for arm in ("r0", "r0b"):
        values = [f for f in (frac(arm, c) for c in cells) if f is not None]
        if values:
            print(f"  {arm:<10} control            fired "
                  f"{sum(1 for f in values if f > 0):2d}/{len(values)}")

    print("\nequivalent relative dose of the real arms, by log-log interpolation")
    for arm in REAL_ARMS:
        exponents = []
        for cell in cells:
            observed = frac(arm, cell)
            if not observed:
                continue
            points = sorted(((frac(f"d:{n}", cell), n) for n in doses
                             if frac(f"d:{n}", cell)), key=lambda p: p[0])
            below = [p for p in points if p[0] <= observed]
            above = [p for p in points if p[0] >= observed]
            if below and above and above[0][0] > below[-1][0]:
                lo, hi = below[-1], above[0]
                t = ((math.log(observed) - math.log(lo[0]))
                     / (math.log(hi[0]) - math.log(lo[0])))
                exponents.append(lo[1] + t * (hi[1] - lo[1]))
            elif below:
                exponents.append(float(below[-1][1]))
        if not exponents:
            continue
        exponents.sort()
        median = exponents[len(exponents) // 2]
        print(f"  {arm}: 2^-{max(exponents):.1f} .. 2^-{min(exponents):.1f}"
              f"   median 2^-{median:.1f}   over {len(exponents)} cells")

    print("\nscore-carrying widths M in {7,8}")
    for arm in ("r0", "r0b", *REAL_ARMS):
        sub = [e for e in rows if e["arm"] == arm and e["m"] in (7, 8)]
        if not sub:
            continue
        diverge = sum(1 for e in sub if not e["legs"]["equal"])
        print(f"  {arm:<4} cells {len(sub):>3}  diverging {diverge:>3}"
              f"  frac {min(e['legs']['mismatch_fraction'] for e in sub):.3g}"
              f"..{max(e['legs']['mismatch_fraction'] for e in sub):.3g}"
              f"  max ulp {max(e['legs']['max_ulp_delta'] for e in sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
