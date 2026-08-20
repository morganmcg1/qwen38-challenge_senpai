#!/usr/bin/env python3
"""E68 rung 1b: do `<T,7,7>`, `<T,8,8>` and `<T,9,9>` even compile? No GPU.

The advisor rejected the NA >= 7 one-group merges on an extrapolation. The
E61 single-stream ladder stops at NA=7, so the `<T,8,8>` = 172.21 ms and
`<T,9,9>` = 199.21 ms entries behind that rejection exist nowhere in the
repository. Before spending a timed leg on them, this census answers the two
cheaper questions:

  1. Does the wide kernel instantiate at NA in [7, 9] at all? `vec<float, NA>`
     and the four-deep `rows_per_simd` accumulator both grow with NA, and the
     shipped `static_assert` stops at 6.
  2. What does each cell really cost in registers, against the shipped table's
     entry point?

A compile failure or a register count that closes the cell is a stronger
closure than any timing, and it costs no GPU.

  python3 research/e68_reg_census.py --out research/e68-artifacts/e68-reg-census.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e59_reg_census as census  # noqa: E402
import e68_arms  # noqa: E402

# `census_arm` reads `ARMS` and `apply_arm` from its own module globals at call
# time, so repointing them here runs the E59 census machinery over the E68 arms
# without forking it.
census.ARMS = e68_arms.ARMS
census.apply_arm = e68_arms.apply_arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="research/e68-artifacts/e68-reg-census.json")
    ap.add_argument("--arms", default=",".join(e68_arms.ARMS))
    args = ap.parse_args()

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    with tempfile.TemporaryDirectory(prefix="e68-reg-census-") as tmp:
        workdir = pathlib.Path(tmp)
        results = {name: census.census_arm(name, workdir) for name in names}

    shipped = results.get("shipped", {})
    print("%-14s %-8s %-6s %-6s  %s" % (
        "arm", "status", "entry", "kmax", "per-width peak live registers"))
    for name, res in results.items():
        cells = res.get("width_cells", {})
        widths = " ".join(
            "M%d=%d" % (m, c["peak_live_regs"]) for m, c in sorted(cells.items()))
        print("%-14s %-8s %-6s %-6s  %s" % (
            name, res.get("status", "?"),
            res.get("entry_point_reg_max", "-"),
            res.get("kernel_wide_reg_max", "-"), widths))
        if res.get("status") != "ok":
            print("    error: %s" % json.dumps(res.get("error")))

    if shipped.get("status") == "ok":
        base = {m: c["peak_live_regs"]
                for m, c in shipped["width_cells"].items()}
        print("\nuntreated-width drift against shipped:")
        for name, res in results.items():
            if name == "shipped" or res.get("status") != "ok":
                continue
            drift = {m: c["peak_live_regs"] - base[m]
                     for m, c in res["width_cells"].items()
                     if m in base and c["peak_live_regs"] != base[m]}
            print("  %-14s %s" % (name, drift or "none"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"ceiling": census.CEILING, "pipeline": "metal -O2 -S | metal-opt -passes=default<O3>",
         "arms": results}, indent=2, sort_keys=True))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
