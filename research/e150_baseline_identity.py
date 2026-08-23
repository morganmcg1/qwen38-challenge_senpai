#!/usr/bin/env python3
"""What exactly is the published `+0.1338` shipped cell made of?

`harness=local`. Zero GPU. Zero Swift.

R4 has to build the R0.5 mechanism in Swift, so it must know precisely which
part of the replay corresponds to code that already ships.

`simulate` keeps two different tables apart:

  * reality  - `install(curve)` then `window_us += ranked_round_us(depth + 1)`
  * belief   - the `price` table handed to the chooser

`Env.base` runs the shipped `walk` with `price=None`, so the base believes the
uniform table `1 + 0.18 d`, which is exactly Swift's `makeUniformDepthPrice`
and therefore the deployed belief. Every arm in R0.5 and R1 instead believes
`env.measured_price`, the table derived from the E145 R3 measured curve.

So the `+0.1338` cell could be either of two things, and R4 depends on which:

  A. the deployed policy, in which case R4 changes only the decision rule;
  B. the shipped rule already carrying a swapped price table, in which case
     `+0.1338` is itself a proposed change and R4 is a two-part mechanism.

This script decides it by pricing the shipped rule under the uniform belief.
Under reading A that arm must score 0.0000 against the base by construction
only if the base and the arm share a belief; under reading B it scores 0.0000
and the published `+0.1338` is the price-table swap alone.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_replay import PRICE_CUMULATIVE, PRICE_MARGINAL  # noqa: E402
from e150_lib import R7_SHIPPED_PCT, build_env, score_arm  # noqa: E402
from e150_predict import fixed_state_walker  # noqa: E402
from e150_r05 import STATES  # noqa: E402

UNIFORM = (list(PRICE_MARGINAL), list(PRICE_CUMULATIVE))


def arm(env, belief, clamp, rule="greedy"):
    def make(seed, prompt, entry):
        inner = fixed_state_walker(STATES["shipped"](seed, prompt, entry),
                                   rule, clamp, fixed_lam=None)

        def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                    price=None):
            return inner(ema, margin, offer, adjust, ctx, force, belief)
        return chooser
    return score_arm(env, "measured", env.measured, env.measured_price, make)


def main() -> int:
    env = build_env(200, 6)
    print("E150 baseline identity - what is the +0.1338 cell?")
    print("  harness=local  gpu_used=False")
    print("  reality: the measured curve, for every row below")
    print()
    print("  %-46s %9s" % ("shipped rule, belief =", "median %"))
    rows = {}
    for name, belief in (("uniform 1 + 0.18d  (what Swift compiles)", UNIFORM),
                         ("measured curve table (what R0.5 arms use)",
                          env.measured_price)):
        row = arm(env, belief, None)
        rows[name] = row
        print("  %-46s %+9.4f  depth %.4f"
              % (name, row["median_pct_mean"], row["weighted_mean_depth"]))

    uniform_pct = list(rows.values())[0]["median_pct_mean"]
    measured_pct = list(rows.values())[1]["median_pct_mean"]
    print()
    print("  published R7 shipped cell                      %+9.4f"
          % R7_SHIPPED_PCT)
    print("  price-table swap alone                         %+9.4f pp"
          % (measured_pct - uniform_pct))
    print()
    if abs(uniform_pct) < 0.02:
        print("  READING B: the base and the uniform-belief arm agree, so the")
        print("  published +0.1338 is the PRICE TABLE SWAP, not the deployed")
        print("  policy. R4 is then a two-part mechanism and must say so.")
    else:
        print("  READING A: the uniform-belief arm does not sit at the base,")
        print("  so the +0.1338 cell is not a pure price-table swap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
