#!/usr/bin/env python3
"""E128-F7 item 3 follow-through: reprice thorfinn's one-pass tables with the
BOARD-MEASURED pass price instead of the curve's own tier break.

harness=ranked. Zero GPU.

The board stratification measures the level price of one extra QMV dispatch
pass directly: f = 50.4 +- 253.0 microseconds, from 200 submissions with
submission and prompt fixed effects and errors clustered on the submission.
Arm 2 removes `G(M) - 1` passes at each touched width, so its saving is
`f * (G(M) - 1)` per round at that width. Everything else, including the
residency loss and the free coefficient `c`, is unchanged from E128-F4 3b.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from e128_arm_prices import (
    C_GRID,
    ONEPASS_RESIDENCY_LOSS,
    ONEPASS_TABLES,
    TEMPLATING,
    price_arm,
)
from e128_ourcurve import (
    F83_WEIGHT,
    build_points,
    curve_us,
    fixture_histograms,
    load_receipt,
    r_scenarios,
)

# Our base's dispatch table, Qwen35.swift:1565 `let cases`.
OUR_IPG = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
# Board stratification, two-way fixed effects, clustered on the submission.
F_HAT, F_SE = 50.4, 253.0
# Fitted tier break of the headline curve, for the comparison line.
CURVE_UNIT = 10187.2


def passes(m: int) -> int:
    return math.ceil(m / max(OUR_IPG.get(m, m), 1))


def onepass_fixed(widths, curve, f_us, c, residency_everywhere):
    loss = 1.0 + c * ONEPASS_RESIDENCY_LOSS * TEMPLATING["round_share"]

    def multiplier(m: float) -> float:
        mi = int(m)
        touched = mi in widths
        saved_us = f_us * (passes(mi) - 1) if touched else 0.0
        base = curve_us(curve, m)
        applies = touched or residency_everywhere
        return (1.0 - saved_us / base) * (loss if applies else 1.0)

    return multiplier


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--identity", type=Path,
                    default=here / "e128-artifacts/rung0-identity.json")
    ap.add_argument("--shipped", type=Path,
                    default=here / "e128-artifacts/rung1-shipped.json")
    ap.add_argument("--curves", type=Path,
                    default=here / "e128-artifacts/f4-candidate-curves.json")
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    hists = fixture_histograms(args.shipped)
    scenarios = r_scenarios(args.identity)
    receipt = load_receipt(args.board, args.receipt)
    points = {p["prompt"]: p
              for p in build_points(receipt, scenarios["assumed"], hists)}
    for prompt, point in points.items():
        point["raw"] = receipt["per_prompt"][prompt]["raw"]
    curve = json.loads(args.curves.read_text())["curves"]["slopeonly_b6"]

    print("harness=ranked  E128-F7 - arm 2 repriced on the measured pass price")
    print("receipt %s  official %.8f  curve slopeonly_b6  Rule 67 exact median"
          % (receipt["id"][:8], receipt["score"]))
    print("our table G(M) = " + " ".join(
        "%d:%d" % (m, passes(m)) for m in range(3, 10)))
    print("measured f = %.1f +- %.1f us   curve tier unit = %.1f us   ratio %.0fx\n"
          % (F_HAT, F_SE, CURVE_UNIT, CURVE_UNIT / max(F_HAT, 1e-9)))

    variants = {
        "measured f = 50.4": F_HAT,
        "upper 95 % f = 546.2": F_HAT + 1.96 * F_SE,
        "curve tier unit = 10187.2": CURVE_UNIT,
    }
    out = {"f_hat": F_HAT, "f_se": F_SE, "curve_unit": CURVE_UNIT, "tables": {}}
    for table, widths in ONEPASS_TABLES.items():
        share = sum(
            F83_WEIGHT[p] * sum(
                pr for pr, m in zip(points[p]["hist"]["probs"], range(1, 10))
                if m in widths)
            for p in points
        )
        print("%s  touches %.4f of F83-weighted rounds" % (table, share))
        print("%-26s %-8s %14s %14s" % ("pass price", "c", "touched-only",
                                        "residency-everywhere"))
        for label, f_us in variants.items():
            for c in (0.0, 0.445):
                a = price_arm(points, curve,
                              onepass_fixed(widths, curve, f_us, c, False))
                b = price_arm(points, curve,
                              onepass_fixed(widths, curve, f_us, c, True))
                print("%-26s %-8.3f %+14.4f %+14.4f"
                      % (label, c, a["median_delta_pct"], b["median_delta_pct"]))
                out["tables"].setdefault(table, {})[f"{label}|c={c}"] = {
                    "touched_only": a["median_delta_pct"],
                    "residency_everywhere": b["median_delta_pct"],
                    "f83_weighted": a["f83_weighted_delta_pct"],
                }
        print()

    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()
