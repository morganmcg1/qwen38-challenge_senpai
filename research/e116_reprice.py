#!/usr/bin/env python3
"""E116 rung 4: reprice the standing kernel arms with the MEASURED transfer.

    usage: research/e116_reprice.py --transfer T --transfer-lo LO
               --transfer-hi HI [--baseline-transfer 0.615] [--json OUT]

WHAT A TRANSFER COEFFICIENT IS. Every kernel arm in this campaign is published
as a percent of wide-QMV KERNEL time. To decide whether an arm is worth
shipping we need its percent of ABSOLUTE candidate MTP seconds per token. The
campaign has been converting between the two with

    0.615 = 0.7786 (assumed wide-QMV share of the round)
          x 0.79   (assumed round-to-leg transfer)

Neither factor was ever measured. E116 measures the product end to end by
injecting a known bit-exact GPU dose and reading the leg.

    leg %  =  kernel %  x  transfer

The CI on each repriced arm here carries ONLY the transfer uncertainty. The
arm's own kernel percent has its own error bar, which lives in the experiment
that measured it and is not double counted here; where that error bar is
published it is printed beside the reprice for the reader to combine.

THE PROMOTION BAR is 0.20 % on absolute candidate MTP seconds per token. An
arm crosses it when the whole repriced interval is beyond the bar, and is
`plausible` when only the point estimate is.
"""

from __future__ import annotations

import argparse
import json
import pathlib

# Round-weighted percent of wide-QMV kernel time, with the source that
# measured it. Sign convention: negative is faster.
ARMS = [
    {"arm": "xv4", "kernel_percent": -0.673, "source": "alphonse E110 r2",
     "note": "live arm; alphonse is extending to n=4 replicates on PR #112"},
    {"arm": "xs_stage", "kernel_percent": -1.458, "source": "alphonse E110 r2",
     "note": "dead by construction; priced for the record only"},
    {"arm": "b_barrier", "kernel_percent": +0.274, "source": "alphonse E110 r2",
     "note": "positive kernel percent, so a slowdown if it transfers"},
    {"arm": "g_pack32", "kernel_percent": +0.334,
     "source": "thorfinn E111 rung 1",
     "note": "measured at mlp.gate_up NA=4"},
]

# alphonse's independent end-to-end readings of the same xv4 arm. The advisor
# reports the powered figure as -0.7498 % without a published interval, so the
# n=3 interval below is the only one available for an overlap test and it is
# the wider of the two.
XV4_ABBA = {"pooled_percent": -0.7498,
            "ci95": [-1.720, 0.246],
            "ci95_source": "the earlier n=3 pooled interval, reused because "
                           "the powered reading has no published interval",
            "source": "alphonse PR #112, properly powered ABBA, as reported "
                      "by the advisor on PR #118"}

XV4_ABBA_N3 = {"pooled_percent": -0.737, "sd": 0.396,
               "ci95": [-1.720, 0.246], "replicates": 3,
               "source": "alphonse PR #112, 3-replicate ABBA, pooled; "
                         "superseded by the powered reading"}

PROMOTION_BAR_PERCENT = 0.20


def classify(point: float, lo: float, hi: float) -> str:
    lo, hi = min(lo, hi), max(lo, hi)
    if hi < -PROMOTION_BAR_PERCENT:
        return "crosses the bar"
    if point < -PROMOTION_BAR_PERCENT:
        return "plausible, interval does not exclude the bar"
    if lo > PROMOTION_BAR_PERCENT:
        return "crosses the bar as a REGRESSION"
    return "below the bar"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", type=float, required=True)
    ap.add_argument("--transfer-lo", type=float, required=True)
    ap.add_argument("--transfer-hi", type=float, required=True)
    ap.add_argument("--baseline-transfer", type=float, default=0.615)
    ap.add_argument("--json")
    args = ap.parse_args()

    priced = []
    for arm in ARMS:
        k = arm["kernel_percent"]
        point = k * args.transfer
        ends = sorted((k * args.transfer_lo, k * args.transfer_hi))
        priced.append({
            **arm,
            "leg_percent": point,
            "leg_percent_ci95": ends,
            "leg_percent_at_baseline_transfer": k * args.baseline_transfer,
            "ratio_to_baseline_price": args.transfer / args.baseline_transfer,
            "verdict": classify(point, ends[0], ends[1]),
        })

    xv4 = next(a for a in priced if a["arm"] == "xv4")
    predicted = xv4["leg_percent"]
    measured = XV4_ABBA["pooled_percent"]
    inside = XV4_ABBA["ci95"][0] <= predicted <= XV4_ABBA["ci95"][1]
    overlap = not (xv4["leg_percent_ci95"][1] < XV4_ABBA["ci95"][0]
                   or XV4_ABBA["ci95"][1] < xv4["leg_percent_ci95"][0])

    out = {
        "harness": "local",
        "experiment":
            "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "rung": 4,
        "measured_transfer": args.transfer,
        "measured_transfer_ci95": [args.transfer_lo, args.transfer_hi],
        "baseline_transfer": args.baseline_transfer,
        "transfer_ratio_to_baseline": args.transfer / args.baseline_transfer,
        "promotion_bar_percent": PROMOTION_BAR_PERCENT,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "arms": priced,
        "xv4_cross_check": {
            "e116_predicted_leg_percent": predicted,
            "e116_predicted_ci95": xv4["leg_percent_ci95"],
            "alphonse_measured": XV4_ABBA,
            "alphonse_measured_n3_superseded": XV4_ABBA_N3,
            "advisor_prediction_at_baseline_transfer":
                xv4["leg_percent_at_baseline_transfer"],
            "prediction_inside_alphonse_ci": inside,
            "intervals_overlap": overlap,
            "prediction_over_measurement":
                predicted / measured if measured else float("nan"),
        },
    }

    print("E116 rung 4 -- repricing the standing kernel arms   harness=local")
    print(f"  measured transfer {args.transfer:.4f}"
          f"  95% CI [{args.transfer_lo:.4f}, {args.transfer_hi:.4f}]")
    print(f"  standing assumption {args.baseline_transfer:.4f}"
          f"  ->  every kernel arm is worth"
          f" {100.0 * (args.transfer / args.baseline_transfer - 1.0):+.1f} %"
          " of what the campaign priced it at")
    print(f"  promotion bar {PROMOTION_BAR_PERCENT:.2f} % of absolute"
          " candidate MTP seconds per token")
    print()
    print(f"{'arm':<11} {'kernel %':>9} {'was':>8} {'now':>8}"
          f" {'95% CI':>20}  verdict")
    for arm in priced:
        ci = f"[{arm['leg_percent_ci95'][0]:+.3f}," \
             f" {arm['leg_percent_ci95'][1]:+.3f}]"
        print(f"{arm['arm']:<11} {arm['kernel_percent']:>+9.3f}"
              f" {arm['leg_percent_at_baseline_transfer']:>+8.3f}"
              f" {arm['leg_percent']:>+8.3f} {ci:>20}  {arm['verdict']}")
    print()
    print("  xv4 cross-check, two students and two independent instruments")
    print(f"    E116 prediction   {predicted:+.3f} %"
          f"  CI [{xv4['leg_percent_ci95'][0]:+.3f},"
          f" {xv4['leg_percent_ci95'][1]:+.3f}]")
    print(f"    advisor at 0.615  "
          f"{xv4['leg_percent_at_baseline_transfer']:+.3f} %"
          "  the standing composed prediction")
    print(f"    alphonse ABBA     {measured:+.4f} %"
          f"  CI [{XV4_ABBA['ci95'][0]:+.3f}, {XV4_ABBA['ci95'][1]:+.3f}]"
          "  (powered; interval carried over from his n=3 pooled reading)")
    print(f"    prediction inside his CI: {inside};"
          f" intervals overlap: {overlap};"
          f" measured / predicted ="
          f" {measured / predicted if predicted else float('nan'):.2f}x")

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
