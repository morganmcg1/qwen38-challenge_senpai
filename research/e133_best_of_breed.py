#!/usr/bin/env python3
"""Best bit-exact nibble form per routed cell, and what selecting per cell buys.

`e133_nibble_floor.py residency` refuses to price a cell where the
`shipped_lifted` neutrality control moves, because there a register delta
cannot be attributed to the idiom under test. That is the right rule for
asking "what does this idiom do".

It is the wrong rule for asking "what is the best kernel I can ship". The
control is the shipped arithmetic hoisted into named locals. It is bit-exact
by construction, so where it allocates better than the shipped form, that is
not drift to be excluded. It is a free win to be taken.

The forms are separate template instantiations per cell, so in principle each
cell can carry its own idiom. This script reports that upper bound. It is an
upper bound and not a result: composing several idioms into one source changes
the translation unit, so the composed form must be censused before it is
believed.
"""

import json
import pathlib
import sys

RANKED = "applegpu_g17s"

# Verified EXACT over the complete 65536-word uint16_t domain by
# `e133_nibble_floor.py exactness`. `no_extract` is excluded: it is declared
# not exact and only marks the floor.
EXACT = ("shipped_lifted", "bfe", "bfe_narrow", "rolling", "bytes",
         "magic_f32", "magic_bf16_pair", "magic_bf16_pair_novec",
         "w2", "w1", "a1", "w1_a1", "w1_w2_a1")

# `Table.onePass67.plan`, Qwen35.swift:1833-1835.
WIDTH_TO_IPG = {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 4, 9: 3}

# Finding 83 as stated. Widths below 6 are not resolved by it.
F83_MASS = {6: 0.188, 7: 0.211, 8: 0.1871}

ELASTICITY = {"E121 g17s residency vs ranked": 2.10 / 10.57,
              "F66 entry-point occupancy tax": 2.6 / 7.69}
QMV_SHARE = 0.8735


def main(path: str) -> int:
    result = json.loads(pathlib.Path(path).read_text())
    V = result["variants"]
    print("best bit-exact form per cell, %s" % RANKED)
    print("  every listed form is EXACT over all 65536 packed words")
    print()
    print("  %6s %8s %9s %9s %-24s %6s %8s"
          % ("cell", "widths", "ship reg", "best reg", "form", "d_sg", "d_sg %"))
    gains = {}
    for na, rps in (tuple(c) for c in result["cells"]):
        key = "cell_na%d_rps%d" % (na, rps)
        ship = V["shipped"]["cells"][key][RANKED]
        widths = sorted(w for w, i in WIDTH_TO_IPG.items() if i == na)
        best_reg, best_name = min(
            (V[n]["cells"][key][RANKED]["registers"], n)
            for n in EXACT if V[n].get("compiled"))
        best = V[best_name]["cells"][key][RANKED]
        dsg = best["simdgroups_derived"] - ship["simdgroups_derived"]
        pct = 100.0 * dsg / ship["simdgroups_derived"]
        gains[na] = pct
        print("  %6s %8s %9d %9d %-24s %+6d %+8.3f"
              % ("ipg%d" % na, ",".join(map(str, widths)), ship["registers"],
                 best_reg, best_name, dsg, pct))

    print()
    print("  priced on the Finding 83 widths only")
    total = 0.0
    for width in sorted(F83_MASS):
        pct = gains[WIDTH_TO_IPG[width]]
        total += F83_MASS[width] * pct
        print("    width %d -> ipg%d   mass %.4f x %+.3f %% = %+.4f"
              % (width, WIDTH_TO_IPG[width], F83_MASS[width], pct,
                 F83_MASS[width] * pct))
    print("    mass-weighted resident-simdgroup change   %+.4f %%" % total)
    print()
    for name, e in ELASTICITY.items():
        print("    %-34s elasticity %.3f -> %+.3f %% leg"
              % (name, e, -total * e * QMV_SHARE))

    if "per_cell_best" not in V or not V["per_cell_best"].get("compiled"):
        print()
        print("  no composed row; the total above is an upper bound")
        return 0

    print()
    print("  composed source against the per-cell bound")
    composed_total, held = 0.0, True
    for na, rps in (tuple(c) for c in result["cells"]):
        key = "cell_na%d_rps%d" % (na, rps)
        ship = V["shipped"]["cells"][key][RANKED]
        bound = min(V[n]["cells"][key][RANKED]["registers"]
                    for n in EXACT if V[n].get("compiled"))
        got = V["per_cell_best"]["cells"][key][RANKED]
        held = held and got["registers"] <= bound
        print("    %6s bound %3d  composed %3d  %s"
              % ("ipg%d" % na, bound, got["registers"],
                 "holds" if got["registers"] <= bound else "GIVES BACK"))
    for width in sorted(F83_MASS):
        key = "cell_na%d_rps4" % WIDTH_TO_IPG[width]
        ship = V["shipped"]["cells"][key][RANKED]
        got = V["per_cell_best"]["cells"][key][RANKED]
        composed_total += F83_MASS[width] * 100.0 * (
            got["simdgroups_derived"] - ship["simdgroups_derived"]
        ) / ship["simdgroups_derived"]
    print()
    print("    composition %s" % ("HOLDS at every cell" if held
                                  else "does not hold"))
    print("    composed mass-weighted residency change  %+.4f %%"
          % composed_total)
    for name, e in ELASTICITY.items():
        print("      %-32s elasticity %.3f -> %+.3f %% leg"
              % (name, e, -composed_total * e * QMV_SHARE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
