#!/usr/bin/env python3
"""E150: audit the width-8 cost cell that R0.5's whole gain stands on.

`harness=local instrument`. Zero GPU. Zero Swift. This module measures
nothing new. It re-reads the E145 R2 timed legs and puts an error bar on the
two cost quantities R0.5's result depends on, because R0.5 puts about 15 % of
its rounds at verify width 8 and every one of its inadmissible rounds is a
width-8 round.

Two separate questions, and they have opposite answers
------------------------------------------------------
1. **Is width 8 really inadmissible?** Rule 138 calls a width admissible when
   its measured cost per token sets a new running minimum. Width 8 misses
   width 5 by 0.43 %, so it is excluded. That exclusion is a claim about a
   difference of two noisy means at `n = 2` legs each, and this module reports
   how many sigma it actually holds by.

2. **Is the linearised rule's preference for width 8 robust to that noise?**
   That is a different question, because the linearised rule does not compare
   cost per token at all. It compares the expected extra tokens against the
   normalised cost step `C(8) - C(5)`, so the relevant error bar is on the
   cost step and on the token threshold it implies.

Reporting both is the point. A rule can be robust while the taxonomy that
forbids it is not.

Usage:
  python3 e150_width8_audit.py --json width8.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e145_r3 import LEGS_JSON  # noqa: E402
from e150_lib import write_artifact  # noqa: E402

# R2 swept pins 1..7 inside one palindrome, so widths 2..8 come from one
# session and are directly comparable. Width 1 is anchored from R2b and is
# deliberately excluded here: it is rescaled across sessions, so its spread is
# not comparable with the rest.
SLOT_PREFIX = "r2-"


def legs_by_width() -> dict[int, list[float]]:
    legs = json.loads(LEGS_JSON.read_text())["legs"]
    by: dict[int, list[float]] = {}
    for leg in legs:
        if not leg["slot"].startswith(SLOT_PREFIX):
            continue
        if leg["pin"] in (None, "", "none"):
            continue
        by.setdefault(int(leg["pin"]) + 1, []).append(
            leg["round_us_from_blocks"])
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial-round-us", type=float,
                    default=65778.93576562102)
    ap.add_argument("--mu", type=float, default=0.45642623901367185,
                    help="R0.5's shipped_noclamp fixed point")
    ap.add_argument("--json", type=pathlib.Path,
                    default=pathlib.Path("width8.json"))
    args = ap.parse_args()

    by = legs_by_width()
    mean = {w: statistics.fmean(v) for w, v in by.items()}
    sd = {w: (statistics.stdev(v) if len(v) > 1 else float("nan"))
          for w, v in by.items()}

    print("E150 width-8 audit — harness=local  gpu_used=False")
    print("  serial round %.1f us   mu* %.6f" % (args.serial_round_us,
                                                 args.mu))
    print("\n%5s %4s %12s %10s %9s %12s"
          % ("width", "n", "mean_us", "sd_us", "sd_pct", "us_per_token"))
    rows = []
    for w in sorted(by):
        rows.append({
            "width": w, "legs": len(by[w]), "mean_us": mean[w],
            "sd_us": sd[w], "sd_pct": 100.0 * sd[w] / mean[w],
            "us_per_token": mean[w] / w,
            "us_per_token_sd": sd[w] / w,
        })
        print("%5d %4d %12.1f %10.1f %8.3f%% %12.1f"
              % (w, len(by[w]), mean[w], sd[w], 100.0 * sd[w] / mean[w],
                 mean[w] / w))

    # Question 1: does Rule 138's exclusion of width 8 survive its own noise?
    best = min(rows, key=lambda r: r["us_per_token"])
    w8 = next(r for r in rows if r["width"] == 8)
    gap = w8["us_per_token"] - best["us_per_token"]
    gap_sd = math.hypot(w8["us_per_token_sd"], best["us_per_token_sd"])
    gap_sigma = gap / gap_sd if gap_sd else float("inf")
    resolved = abs(gap) > 2.0 * gap_sd

    print("\n## question 1: is width 8 really inadmissible?")
    print("  running minimum is width %d at %.1f us/token"
          % (best["width"], best["us_per_token"]))
    print("  width 8 is %.1f us/token, a gap of %+.1f us (%.2f %% of the "
          "minimum)" % (w8["us_per_token"], gap,
                        100.0 * gap / best["us_per_token"]))
    print("  gap %+.1f +- %.1f us  ->  %.2f sigma  ->  exclusion is %s"
          % (gap, gap_sd, gap_sigma,
             "RESOLVED" if resolved else "NOT RESOLVED at 2 sigma"))

    # Question 2: is the linearised rule's width-8 step robust?
    c5, c8 = mean[5], mean[8]
    step = (c8 - c5) / args.serial_round_us
    step_sd = math.hypot(sd[5], sd[8]) / args.serial_round_us
    need = step / args.mu
    need_sd = step_sd / args.mu
    marginal = mean[8] - mean[7]
    marginal_sd = math.hypot(sd[7], sd[8])

    print("\n## question 2: is the linearised rule's jump to width 8 robust?")
    print("  marginal 7->8 %.1f +- %.1f us  (%.1f sigma from zero)"
          % (marginal, marginal_sd, marginal / marginal_sd))
    print("  normalised C(8)-C(5) %.4f +- %.4f" % (step, step_sd))
    print("  extra expected tokens needed to prefer width 8 over width 5: "
          "%.4f +- %.4f" % (need, need_sd))
    print("  the threshold moves by %.2f %% of itself across one sigma, so "
          "the decision boundary is %s"
          % (100.0 * need_sd / need,
             "robust to the cost measurement"
             if need_sd / need < 0.05 else "sensitive to the cost measurement"))

    out = {
        "rung": "E150 width-8 audit",
        "harness": "local",
        "gpu_used": False,
        "source": "E145 R2 timed legs, slots %s*" % SLOT_PREFIX,
        "serial_round_us": args.serial_round_us,
        "mu": args.mu,
        "e150_width_cost_table": rows,
        "e150_w8_running_minimum_width": best["width"],
        "e150_w8_cost_per_token_gap_us": gap,
        "e150_w8_cost_per_token_gap_pct": 100.0 * gap / best["us_per_token"],
        "e150_w8_cost_per_token_gap_sd_us": gap_sd,
        "e150_w8_inadmissibility_sigma": gap_sigma,
        "e150_w8_inadmissibility_resolved": resolved,
        "e150_w8_marginal_7_to_8_us": marginal,
        "e150_w8_marginal_7_to_8_sd_us": marginal_sd,
        "e150_w8_step_from_w5_normalised": step,
        "e150_w8_step_from_w5_normalised_sd": step_sd,
        "e150_w8_tokens_needed_over_w5": need,
        "e150_w8_tokens_needed_over_w5_sd": need_sd,
        "e150_w8_decision_boundary_robust": need_sd / need < 0.05,
    }
    path = write_artifact(args.json, out)
    print("\nwrote %s" % path.relative_to(HERE.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
