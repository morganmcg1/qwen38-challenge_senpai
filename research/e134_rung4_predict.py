#!/usr/bin/env python3
"""E134 rung 4: the pre-registered ranked prediction for the `pb6` arm.

Rung 3 chose the arm. This script states what the ranked runner should read if
the arm is real, on two instruments, before the receipt exists:

1. The published median score, which is the ranked metric and the noisiest
   thing in the receipt.
2. The FINDING 154 instrument, the unweighted eight-prompt mean of
   `mtp_seconds_per_token_mean`, whose standard error inside one runner state
   is 0.0187 percent against the published median's 0.0967 percent.

Both are stated against the `d3c491b5` receipt this replayer is calibrated on.
The prediction is a percentage change in the candidate leg; the ranked serial
numerator comes from a different workspace and no candidate edit can move it,
so the same percentage applies to every affected `raw_p`.

Run from `research/`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e128_price  # noqa: E402
from e128_price import load_board_receipt, median_of  # noqa: E402
from e128_replay import MAX_DEPTH  # noqa: E402
from e134_rung2 import build_legs, median_pct, prompt_panel  # noqa: E402
from e134_rung3 import (  # noqa: E402
    CURVES, boundary_price, price_ratios,
)

# FINDING 154. Standard errors of one ranked measurement inside one classified
# runner state, in percent.
SE_PUBLISHED_MEDIAN = 0.0967
SE_UNWEIGHTED_MEAN = 0.0187


def unweighted_mean_pct(receipt: dict, ratios: dict) -> float:
    """Percent change in the unweighted eight-prompt mean candidate time."""
    base, arm = [], []
    for prompt, entry in receipt["per_prompt"].items():
        base.append(entry["candidate"])
        arm.append(entry["candidate"] * ratios.get(prompt, {}).get("ratio", 1.0))
    return (statistics.fmean(arm) / statistics.fmean(base) - 1.0) * 100.0


def raw_scores(receipt: dict, ratios: dict) -> dict:
    out = {}
    for prompt, entry in receipt["per_prompt"].items():
        ratio = ratios.get(prompt, {}).get("ratio", 1.0)
        out[prompt] = {
            "ship_raw": entry["serial"] / entry["candidate"],
            "arm_raw": entry["serial"] / (entry["candidate"] * ratio),
            "ratio": ratio}
    return out


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=here.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=here / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--fit-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--curve", choices=tuple(CURVES), default="ours")
    ap.add_argument("--tier", type=float, default=1.45)
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--json", type=pathlib.Path,
                    default=here / "e134-artifacts/rung4-prediction.json")
    args = ap.parse_args()

    e128_price.CURVE = CURVES[args.curve]
    price = boundary_price(args.tier, args.width - 2)[:2]
    print("## pb6 as the scored worker builds it")
    print("curve %s ; verify width %d ; tier %.4f" % (
        args.curve, args.width, args.tier))
    print("marginal  %s" % " ".join("%.6f" % v for v in price[0]))
    print("cumulative %s" % " ".join("%.6f" % v for v in price[1]))
    total = sum(price[0])
    print("total held at %.10f against the shipped %.10f"
          % (total, MAX_DEPTH * 0.18))

    legs, gate = build_legs(args.accept, args.runs)
    print("\n## attachment gate")
    print("legs %d ; rounds attached %d ; accept mismatches %d ; "
          "margin mismatches %d ; unmatched %d" % (
              gate["legs"], gate["attached"], gate["accept_mismatch"],
              gate["margin_mismatch"], gate["unmatched"]))

    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    medians, means, per_prompt = [], [], {}
    for seed in seeds:
        panel = prompt_panel(legs, args.windows, args.fit_windows, seed)
        ratios = price_ratios(panel, price, args.windows)
        medians.append(median_pct(receipt, ratios))
        means.append(unweighted_mean_pct(receipt, ratios))
        for prompt, entry in ratios.items():
            per_prompt.setdefault(prompt, []).append(entry["ratio"])

    median_mean = statistics.fmean(medians)
    median_sd = statistics.stdev(medians) if len(medians) > 1 else 0.0
    mean_mean = statistics.fmean(means)
    mean_sd = statistics.stdev(means) if len(means) > 1 else 0.0

    print("\n## per prompt candidate time multiplier, mean over %d seeds"
          % len(seeds))
    print("%-10s %10s %10s" % ("prompt", "ratio", "sd"))
    ratio_summary = {}
    for prompt in sorted(per_prompt):
        values = per_prompt[prompt]
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        ratio_summary[prompt] = {"ratio": statistics.fmean(values), "sd": sd}
        print("%-10s %10.6f %10.6f" % (prompt, ratio_summary[prompt]["ratio"], sd))

    scores = raw_scores(receipt, ratio_summary)
    ship_median = median_of([v["ship_raw"] for v in scores.values()])
    arm_median = median_of([v["arm_raw"] for v in scores.values()])

    # The published median reads only the two central prompts, so name them and
    # say what happens if any single prompt does not transfer at all.
    ranked = sorted(scores, key=lambda k: scores[k]["ship_raw"])
    armed = sorted(scores, key=lambda k: scores[k]["arm_raw"])
    print("\n## which prompts set the median, and a one-prompt jackknife")
    print("shipped central pair  %s" % ", ".join(ranked[3:5]))
    print("arm central pair      %s" % ", ".join(armed[3:5]))
    jackknife = {}
    for prompt in sorted(scores):
        held = {k: dict(v) for k, v in ratio_summary.items()}
        held[prompt]["ratio"] = 1.0
        jackknife[prompt] = median_pct(receipt, held)
    print("%-10s %12s" % ("prompt held at 1.0", "median %"))
    for prompt in sorted(jackknife, key=lambda k: jackknife[k]):
        print("%-18s %8.4f" % (prompt, jackknife[prompt]))
    print("\n## the two instruments, against receipt %s" % args.receipt)
    print("published median   %.8f -> %.8f   %+.4f %%  sd %.4f  z %+.2f" % (
        ship_median, arm_median, median_mean, median_sd,
        median_mean / SE_PUBLISHED_MEDIAN))
    print("unweighted 8-prompt mean candidate time   %+.4f %%  sd %.4f  "
          "z %+.2f" % (mean_mean, mean_sd, mean_mean / SE_UNWEIGHTED_MEAN))
    print("\nA ranked receipt that reads the published median WITHIN "
          "%.4f %% of zero" % (2.0 * SE_PUBLISHED_MEDIAN))
    print("does not refute the arm. The unweighted mean is the instrument "
          "that can.")

    payload = {
        "curve": args.curve, "tier": args.tier, "width": args.width,
        "receipt": args.receipt, "seeds": seeds, "windows": args.windows,
        "fit_windows": args.fit_windows, "gate": gate,
        "marginal": price[0], "cumulative": price[1],
        "per_prompt": {k: dict(v, **scores[k]) if k in scores else v
                       for k, v in ratio_summary.items()},
        "published_median_pct": {"mean": median_mean, "sd": median_sd,
                                 "per_seed": medians,
                                 "z": median_mean / SE_PUBLISHED_MEDIAN},
        "unweighted_mean_pct": {"mean": mean_mean, "sd": mean_sd,
                                "per_seed": means,
                                "z": mean_mean / SE_UNWEIGHTED_MEAN},
        "ship_median_raw": ship_median, "arm_median_raw": arm_median,
        "central_pair_ship": ranked[3:5], "central_pair_arm": armed[3:5],
        "jackknife_median_pct": jackknife,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
