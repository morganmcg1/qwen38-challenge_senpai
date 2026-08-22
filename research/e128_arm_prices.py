#!/usr/bin/env python3
"""E128-F4 3b - price the three live kernel arms on our own ranked curve.

harness=ranked. Zero GPU.

Three candidates are in flight with no ranked price. This prices each one as
a published-median delta, per prompt, using the per-prompt realised width
histogram, our fitted ranked round-cost curve, and the exact Rule 67 median
recomputation over all eight board prompts.

How a price is computed
-----------------------

Each arm is a per-width multiplier on the round cost, `m(M)`. For prompt `p`:

    old_round_us(p) = sum_M hist_p(M) * round_us(M)
    new_round_us(p) = sum_M hist_p(M) * round_us(M) * m(M)
    new_raw(p)      = old_raw(p) * old_round_us(p) / new_round_us(p)

The ranked serial numerator is fixed by the runner-owned baseline workspace
and no candidate edit can move it, so scaling the candidate round cost scales
`raw` exactly. The eight new `raw` values are then re-sorted and the median is
recomputed, which handles re-ordering as Rule 67 requires.

The free residency coefficient `c`
----------------------------------

`c` converts a fractional gain in resident simdgroups into a fractional
reduction in the time of the affected work:

    time multiplier on the affected share = 1 - c * (residency gain fraction)

`c = 0` means residency buys nothing. `c = 0.445` is the upper edge the
advisor pinned. Every residency-carrying term is reported across that whole
interval rather than at one assumed value.

AMBIGUITIES, FLAGGED RATHER THAN GUESSED
----------------------------------------

1. The definition of `c` above is my reading of "its price is `c` times a
   known number". If `c` is instead defined against a different denominator,
   every residency term here rescales linearly and the reported interval
   endpoints move with it. The pass-count term of arm 2 does not depend on
   `c` at all and is unaffected.
2. Arm 2's residency loss is quoted as a change in WEIGHTED residency, 38.0
   to 32.3. It is not stated whether that loss applies only at the widths the
   new table changes or across the whole round. Both readings are priced.
3. Arm 3's ranked share is given as a range, 5 % to 15 %, so it is swept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e128_ourcurve import (
    F83_WEIGHT,
    MAX_ROWS,
    build_points,
    curve_us,
    fixture_histograms,
    load_receipt,
    r_scenarios,
)

ROWS = np.arange(0, MAX_ROWS, dtype=float) + 1.0
C_GRID = [0.0, 0.1, 0.2, 0.3, 0.445]

# Arm 1, thorfinn. Per-width templating of the Route B entry point.
TEMPLATING = {"residency_gain": 0.1018, "round_share": 0.88}

# Arm 2, thorfinn. One-pass tables. Weighted resident simdgroups fall from
# 38.0 to 32.3 because the per-width bodies are wider in registers.
ONEPASS_TABLES = {"{6:6}": [6], "{6:6,7:7}": [6, 7], "{6:6,7:7,8:8}": [6, 7, 8]}
ONEPASS_RESIDENCY_LOSS = 1.0 - 32.3 / 38.0

# Arm 3, alphonse. One line, pure residency, on an uncertain ranked share.
PRUNE_NA5 = {"residency_gain": 0.1282, "round_share": [0.05, 0.10, 0.15]}


def median_of(values: list[float]) -> float:
    ordered = sorted(values)
    return 0.5 * (ordered[3] + ordered[4])


def price_arm(points: dict, curve: dict, multiplier) -> dict:
    """Published-median delta for a per-width round-cost multiplier."""
    per_prompt = {}
    for prompt, point in points.items():
        probs = np.array(point["hist"]["probs"])
        base = float(sum(pr * curve_us(curve, m)
                         for pr, m in zip(probs, ROWS)))
        new = float(sum(pr * curve_us(curve, m) * multiplier(m)
                        for pr, m in zip(probs, ROWS)))
        per_prompt[prompt] = {
            "old_round_us": base, "new_round_us": new,
            "round_delta_pct": 100.0 * (new / base - 1.0),
            "old_raw": point["raw"],
            "new_raw": point["raw"] * base / new,
            "raw_delta_pct": 100.0 * (base / new - 1.0),
            "f83_weight": F83_WEIGHT[prompt],
        }
    old_med = median_of([r["old_raw"] for r in per_prompt.values()])
    new_med = median_of([r["new_raw"] for r in per_prompt.values()])
    weighted = sum(F83_WEIGHT[p] * r["raw_delta_pct"]
                   for p, r in per_prompt.items())
    return {
        "per_prompt": per_prompt,
        "old_median": old_med, "new_median": new_med,
        "median_delta_pct": 100.0 * (new_med / old_med - 1.0),
        "f83_weighted_delta_pct": weighted,
    }


def uniform(gain: float, share: float, c: float):
    """A width-independent residency arm."""
    factor = 1.0 - c * gain * share
    return lambda m: factor


def onepass(widths: list, curve: dict, single: dict, c: float,
            residency_everywhere: bool):
    """Arm 2: one pass at the named widths, minus the residency loss.

    The pass saving is read off the fitted curve as the difference between
    the two-pass cost actually paid at that width and the one-pass cost the
    same curve assigns below the tier break. That keeps the saving on the
    curve's own terms instead of assuming a separate per-pass constant.
    """
    loss = 1.0 + c * ONEPASS_RESIDENCY_LOSS * TEMPLATING["round_share"]

    def multiplier(m: float) -> float:
        touched = int(m) in widths
        saved = (curve_us(single, m) / curve_us(curve, m)) if touched else 1.0
        applies = touched or residency_everywhere
        return saved * (loss if applies else 1.0)
    return multiplier


def single_pass_curve(curve: dict) -> dict:
    """The same curve with the tier break pushed above every reachable width.

    Evaluating the low segment at a wide width is exactly the counterfactual
    the one-pass table creates: the same rows, without the second pass.
    """
    return {"breakpoint": MAX_ROWS + 2, "lo": curve["lo"], "hi": curve["hi"]}


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path,
                        default=Path("/tmp/yukon-board/full.json"))
    parser.add_argument("--identity", type=Path,
                        default=here / "e128-artifacts/rung0-identity.json")
    parser.add_argument("--shipped", type=Path,
                        default=here / "e128-artifacts/rung1-shipped.json")
    parser.add_argument("--curves", type=Path,
                        default=here / "e128-artifacts/"
                                       "f4-candidate-curves.json")
    parser.add_argument("--receipt", default="d3c491b5")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    hists = fixture_histograms(args.shipped)
    scenarios = r_scenarios(args.identity)
    receipt = load_receipt(args.board, args.receipt)
    points = {p["prompt"]: p
              for p in build_points(receipt, scenarios["assumed"], hists)}
    for prompt, point in points.items():
        point["raw"] = receipt["per_prompt"][prompt]["raw"]

    curves = json.loads(args.curves.read_text())["curves"]
    admissible = "slopeonly_b6"
    order = [admissible] + [k for k in sorted(curves) if k != admissible]

    print("harness=ranked  E128-F4 3b - ranked price of the three live arms")
    print("receipt %s  official %.8f  R assumed  Rule 67 exact median"
          % (receipt["id"][:8], receipt["score"]))
    print("headline curve %s (physically admissible)\n" % admissible)

    out = {}

    print("## arm 1 - per-width templating of the Route B entry point")
    print("pure residency, +%.2f %% on %.0f %% of the round, width-independent"
          % (100 * TEMPLATING["residency_gain"],
             100 * TEMPLATING["round_share"]))
    print("%-8s " % "c" + " ".join("%22s" % k for k in order))
    rows = {}
    for c in C_GRID:
        line = []
        for key in order:
            got = price_arm(points, curves[key],
                            uniform(TEMPLATING["residency_gain"],
                                    TEMPLATING["round_share"], c))
            line.append(got["median_delta_pct"])
            rows.setdefault(key, {})[c] = got
        print("%-8.3f " % c + " ".join("%+22.4f" % v for v in line))
    out["templating"] = {k: {str(c): v for c, v in byc.items()}
                         for k, byc in rows.items()}

    print("\n## arm 2 - one-pass tables")
    print("pass saving from the curve, minus a residency loss of %.2f %% "
          "weighted" % (100 * ONEPASS_RESIDENCY_LOSS))
    out["onepass"] = {}
    for table, widths in ONEPASS_TABLES.items():
        print("\n%s  touches %.4f of F83-weighted rounds" % (
            table,
            sum(F83_WEIGHT[p] * sum(np.array(points[p]["hist"]["probs"])[
                [w - 1 for w in widths]]) for p in points)))
        print("%-8s %-10s " % ("c", "resid") + " ".join("%22s" % k
                                                        for k in order))
        for everywhere in (False, True):
            for c in C_GRID:
                line = []
                for key in order:
                    curve = curves[key]
                    got = price_arm(points, curve,
                                    onepass(widths, curve,
                                            single_pass_curve(curve), c,
                                            everywhere))
                    line.append(got["median_delta_pct"])
                    out["onepass"].setdefault(table, {}).setdefault(
                        "everywhere" if everywhere else "touched_only",
                        {}).setdefault(key, {})[str(c)] = got
                print("%-8.3f %-10s " % (c, "all" if everywhere else "touched")
                      + " ".join("%+22.4f" % v for v in line))

    print("\n## arm 3 - prune_na5_pair")
    print("pure residency, +%.2f %% on a 5 to 15 %% ranked share"
          % (100 * PRUNE_NA5["residency_gain"]))
    print("%-8s %-8s " % ("c", "share") + " ".join("%22s" % k for k in order))
    out["prune_na5_pair"] = {}
    for share in PRUNE_NA5["round_share"]:
        for c in C_GRID:
            line = []
            for key in order:
                got = price_arm(points, curves[key],
                                uniform(PRUNE_NA5["residency_gain"], share, c))
                line.append(got["median_delta_pct"])
                out["prune_na5_pair"].setdefault(str(share), {}).setdefault(
                    key, {})[str(c)] = got
            print("%-8.3f %-8.2f " % (c, share)
                  + " ".join("%+22.4f" % v for v in line))

    print("\n## per-prompt vectors on the headline curve at c = 0.445")
    print("%-28s " % "arm"
          + " ".join("%9s" % p for p in sorted(points,
                                               key=lambda x: -F83_WEIGHT[x]))
          + " %10s" % "median %")
    named = [("templating",
              uniform(TEMPLATING["residency_gain"],
                      TEMPLATING["round_share"], 0.445))]
    for table, widths in ONEPASS_TABLES.items():
        named.append(("onepass %s" % table,
                      onepass(widths, curves[admissible],
                              single_pass_curve(curves[admissible]), 0.445,
                              False)))
    named.append(("prune_na5_pair s=0.10",
                  uniform(PRUNE_NA5["residency_gain"], 0.10, 0.445)))
    per_prompt_table = {}
    for label, mult in named:
        got = price_arm(points, curves[admissible], mult)
        per_prompt_table[label] = got
        print("%-28s " % label
              + " ".join("%+9.3f" % got["per_prompt"][p]["raw_delta_pct"]
                         for p in sorted(points,
                                         key=lambda x: -F83_WEIGHT[x]))
              + " %+10.4f" % got["median_delta_pct"])
    out["per_prompt_at_c_max"] = per_prompt_table

    payload = {
        "harness": "ranked",
        "source": "e128_arm_prices.py",
        "receipt": {"prefix": args.receipt, "id": receipt["id"],
                    "score": receipt["score"]},
        "headline_curve": admissible,
        "c_grid": C_GRID,
        "c_definition": "time multiplier on the affected share = "
                        "1 - c * residency gain fraction",
        "rule67": "acceptance unchanged for all three arms; median "
                  "recomputed exactly over all eight board prompts",
        "arms": out,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                        default=float))
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
