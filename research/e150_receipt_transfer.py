#!/usr/bin/env python3
"""E150 R0.5c: re-price the selected policy on the CURRENT promoted receipt.

`harness=local`. Zero GPU. Zero Swift. Rule 79 is not engaged: every number is
an offline replay price, not a timing contrast.

Why this module exists
----------------------
Advisor feedback F3 moved the campaign frontier. Our `.ship` flip is now a
promoted ranked receipt, `0cf1637e` at 3.68278758, and the crown is
`684821ed` at 3.71959723. Every E150 rung before this one priced against
receipt `d3c491b5` at 3.49065044, which is the receipt this experiment
pre-registered. `median_pct` weights the replay by the receipt's per-prompt
serial and candidate times, so the receipt decides WHICH TWO PROMPTS sit in
the even median. If the promotion changed the central pair, the pre-registered
price is answering a question the board no longer asks.

What the receipt does and does not change
-----------------------------------------
It does NOT change mu*. mu* is the fixed point of pooled normalised cost per
token on one cost curve against one acceptance trace, and no receipt enters
that solve. So the policy here is the SAME policy R0.5b selected, at the same
mu*, re-weighted only.

The frame
---------
`median_pct` is already the official functional form: the even median of the
eight per-prompt `serial / candidate` raw ratios, with the replayed per-prompt
cost ratio divided into the candidate time. Its denominator is the receipt's
own shipped median, which for the even-median rule IS the receipt's official
score. So

    projected ranked score = receipt score * (1 + median_pct / 100)

is an arithmetic identity of the frame, not a second model. It is still a
LOCAL projection: the acceptance trace under it comes from local forced-depth
legs on prompt families matched to the hidden prompts by sha8, not from the
hidden prompts themselves.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import RANKED_PROMPTS  # noqa: E402
from e150_curve_bracket import price_cell  # noqa: E402
from e150_lib import build_env, write_artifact  # noqa: E402

# R0.5b solved these on each curve, clamp off, `shipped_noclamp` state.
MU_STAR = {"measured": 0.459979248046875, "replayed": 0.513897705078125}

# The receipts to re-weight against, newest campaign state last.
RECEIPTS = [
    ("d3c491b5", "pre-registered E150 receipt"),
    ("0cf1637e", "current promoted campaign receipt, the .ship flip"),
]


def median_of(values):
    ordered = sorted(values)
    return 0.5 * (ordered[3] + ordered[4])


def leverage(receipt: dict) -> dict:
    """Which prompts the even median actually rests on, and how far the rest sit.

    Only the fourth and fifth order statistics enter the median, so a large
    gain on a prompt far below them buys nothing until it crosses. This table
    is what makes a per-prompt result readable in the ranked frame.
    """
    raws = {p: e["serial"] / e["candidate"]
            for p, e in receipt["per_prompt"].items()}
    ordered = sorted(raws.items(), key=lambda kv: kv[1])
    central = [ordered[3][0], ordered[4][0]]
    med = median_of(list(raws.values()))
    rows = {}
    for rank, (prompt, raw) in enumerate(ordered):
        rows[prompt] = {
            "raw": raw,
            "rank": rank,
            "in_central_pair": prompt in central,
            # Local derivative of the median with respect to a relative
            # candidate-time improvement on this prompt alone.
            "d_median_d_relative_gain": 0.5 * raw if prompt in central else 0.0,
            # How much this prompt must improve before it can move the median.
            "relative_gain_to_reach_central": max(
                0.0, ordered[3][1] / raw - 1.0),
        }
    return {"median": med, "central_pair": central, "per_prompt": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--json", type=str, default="receipt_transfer.json")
    args = ap.parse_args()

    print("E150 R0.5c - the selected policy re-weighted on each receipt")
    print("  harness=local  gpu_used=False  rule_79=not_engaged")
    print("  windows %d  seeds %d" % (args.windows, args.seeds))

    out = {"rung": "R0.5c", "harness": "local", "gpu_used": False,
           "frame": ("official even median of eight per-prompt serial over "
                     "candidate raw ratios, replayed"),
           "windows": args.windows, "seeds": args.seeds,
           "mu_star": MU_STAR, "receipts": {}}

    for prefix, why in RECEIPTS:
        env = build_env(args.windows, seeds=args.seeds, receipt_prefix=prefix)
        score = env.receipt["score"]
        lev = leverage(env.receipt)
        print("\n## receipt %s  score %.8f  status %s  (%s)"
              % (env.receipt["id"][:8], score, env.receipt["status"], why))
        print("   central pair %s  median %.8f" % (lev["central_pair"],
                                                   lev["median"]))
        for prompt in RANKED_PROMPTS:
            r = lev["per_prompt"][prompt]
            print("     %-10s raw %.6f rank %d central %-5s "
                  "d_med/d_gain %.4f  needs %+.2f%% to reach central"
                  % (prompt, r["raw"], r["rank"], str(r["in_central_pair"]),
                     r["d_median_d_relative_gain"],
                     100.0 * r["relative_gain_to_reach_central"]))

        cells = {}
        for curve in ("measured", "replayed"):
            shipped = price_cell(env, curve, curve, "shipped", "greedy",
                                 None, None)
            # `clamp={}` is NOCLAMP; `clamp=None` restores the shipped
            # depth-0 and depth-1 sigmoid clamps.
            chosen = price_cell(env, curve, curve, "shipped", "ratio",
                                {}, MU_STAR[curve])
            gain = chosen["median_pct_mean"] - shipped["median_pct_mean"]
            cells[curve] = {
                "shipped_pct": shipped["median_pct_mean"],
                "policy_pct": chosen["median_pct_mean"],
                "policy_pct_sd": chosen["median_pct_sd"],
                "gain_pp": gain,
                "weighted_mean_depth": chosen["weighted_mean_depth"],
                "frac_rounds_inadmissible": chosen["frac_rounds_inadmissible"],
                "projected_ranked_score":
                    score * (1.0 + chosen["median_pct_mean"] / 100.0),
                "projected_ranked_score_at_gain":
                    score * (1.0 + gain / 100.0),
            }
            print("   %-9s shipped %+8.4f  policy %+8.4f (sd %.4f)  "
                  "gain %+8.4f pp  projected score %.8f"
                  % (curve, shipped["median_pct_mean"],
                     chosen["median_pct_mean"], chosen["median_pct_sd"],
                     gain, cells[curve]["projected_ranked_score"]))

        worst = min(cells[c]["gain_pp"] for c in cells)
        out["receipts"][prefix] = {
            "id": env.receipt["id"], "score": score,
            "status": env.receipt["status"], "why": why,
            "leverage": lev, "cells": cells,
            "gain_min_over_curves_pp": worst,
            "projected_ranked_score_min_over_curves":
                score * (1.0 + worst / 100.0),
        }
        print("   min over curves %+8.4f pp  -> projected score %.8f"
              % (worst, out["receipts"][prefix]
                 ["projected_ranked_score_min_over_curves"]))

    current = out["receipts"]["0cf1637e"]
    out["e150_gain_min_over_curves_on_current_receipt_pp"] = (
        current["gain_min_over_curves_pp"])
    out["e150_projected_ranked_score_on_current_receipt"] = (
        current["projected_ranked_score_min_over_curves"])
    out["e150_receipt_reweight_shift_pp"] = (
        current["gain_min_over_curves_pp"]
        - out["receipts"]["d3c491b5"]["gain_min_over_curves_pp"])
    out["e150_central_pair_unchanged_by_promotion"] = (
        current["leverage"]["central_pair"]
        == out["receipts"]["d3c491b5"]["leverage"]["central_pair"])

    print("\n## summary")
    for key in ("e150_gain_min_over_curves_on_current_receipt_pp",
                "e150_projected_ranked_score_on_current_receipt",
                "e150_receipt_reweight_shift_pp",
                "e150_central_pair_unchanged_by_promotion"):
        print("  %-52s %s" % (key, out[key]))

    write_artifact(args.json, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
