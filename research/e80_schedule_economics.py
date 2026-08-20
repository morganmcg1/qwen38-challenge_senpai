#!/usr/bin/env python3
"""Turn the measured per-width round cost into the economics of draft depth.

A drafting round at width `M` costs a fixed amount of GPU time whatever the
target accepts, because the target verifies all `M` rows before acceptance is
known. The census measures that cost. Each leg also measures its own serial
cost per token in the same session, on the same host, at the same temperature.

Two numbers follow directly and neither needs an acceptance model:

  ceiling   = the speedup at perfect acceptance, `M * serial / round`
  breakeven = the accepted-token count per round that reaches speedup 1.0

Both are reported per leg against that leg's own `F(1)` thermometer, so no
cross-leg or cross-gate-set comparison is involved.

    usage: research/e80_schedule_economics.py --gated DIR... --ungated DIR...
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e80_blocks as B
import e80_census_report as R


def leg_rows(directory, min_rounds):
    leg = R.Leg([pathlib.Path(directory) / "census.jsonl"])
    rules = B.learn_axis_rules(leg)
    serial = None
    if leg.round_count(1, "target_forward") >= min_rounds:
        att, n = B.attribute(leg, "target_forward", 1, rules)
        serial = sum(v["gpu_ns"] for v in att.values()) / n / 1e6
    rows = []
    for width in sorted(leg.widths()):
        if width <= 1 or serial is None:
            continue
        total, head_gemv = 0.0, 0.0
        rounds = 0
        for phase in ("draft_head", "target_verify"):
            if leg.round_count(width, phase) < min_rounds:
                continue
            att, n = B.attribute(leg, phase, width, rules)
            total += sum(v["gpu_ns"] for v in att.values()) / n / 1e6
            rounds = max(rounds, n)
            if phase == "draft_head":
                head_gemv = sum(v["gpu_ns"] for k, v in att.items()
                                if B.family_of_owner(k) == "gemv") / n / 1e6
        if total:
            rows.append((width, rounds, serial, total, head_gemv))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gated", nargs="*", default=[])
    ap.add_argument("--ungated", nargs="*", default=[])
    ap.add_argument("--min-rounds", type=int, default=20)
    args = ap.parse_args()

    ratio = 427_742_600 / 849_398_784

    print("## the economics of draft depth\n")
    print("| width | set | leg | rounds | own `F(1)` ms | round ms | ceiling at "
          "perfect acceptance | break-even accepted tokens | break-even "
          "acceptance rate |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    seen, model = [], []
    for label, dirs in (("gated", args.gated), ("ungated", args.ungated)):
        for d in dirs:
            for width, rounds, serial, total, gemv in leg_rows(d,
                                                               args.min_rounds):
                ceiling = width * serial / total
                # speedup 1.0 needs serial * (1 + accepted) == total
                accepted = total / serial - 1.0
                name = pathlib.Path(d).name
                print(f"| {width} | {label} | `{name}` | {rounds} | "
                      f"{serial:.3f} | {total:.3f} | {ceiling:.3f} | "
                      f"{accepted:.3f} of {width-1} | "
                      f"{100*accepted/(width-1):.1f}% |")
                seen.append(ceiling)
                model.append((width, label, name, serial,
                              total - gemv * (1 - ratio)))
    if seen:
        print(f"\nThe ceiling spans {min(seen):.3f} to {max(seen):.3f} across "
              f"every measured width, a range of "
              f"{100*(max(seen)/min(seen)-1):.1f} %.")

    print("\n### the same table under the narrow declared-head model\n")
    print("This substitutes the declared head artifact for the resident one by "
          "scaling the head's `gemv` cost by its byte ratio. It is a model, not "
          "a measurement.\n")
    print("| width | set | modelled round ms | ceiling | break-even accepted "
          "tokens | break-even acceptance rate |")
    print("|---:|---|---:|---:|---:|---:|")
    ceilings = []
    for width, label, _, serial, total in model:
        ceiling = width * serial / total
        accepted = total / serial - 1.0
        ceilings.append(ceiling)
        print(f"| {width} | {label} | {total:.3f} | {ceiling:.3f} | "
              f"{accepted:.3f} of {width-1} | "
              f"{100*accepted/(width-1):.1f}% |")
    if ceilings:
        print(f"\nThe modelled ceiling spans {min(ceilings):.3f} to "
              f"{max(ceilings):.3f}, a range of "
              f"{100*(max(ceilings)/min(ceilings)-1):.1f} %.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
