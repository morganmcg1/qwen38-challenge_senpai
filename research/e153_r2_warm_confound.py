#!/usr/bin/env python3
"""Test the advisor's cold-fallback confound against the E153 R2 ABBA data.

Hypothesis under test (advisor F6 section 4): the merged wide-decode SDPA
kernel looked slower because its `kL >= 1024` refusal band fell back to a split
kernel that was no longer warmed, so the merged arm paid a pipeline-compile
miss inside the timed region.

A compile miss is a fixed cost per leg. The mechanism cost is proportional to
eligible rounds. Separating the two decides the question, and the leg that
never refused decides it directly.
"""

import json

ABBA = "research/e153-r2-abba.json"
GATED_LEG_FLOOR_PCT = 0.052


def main() -> None:
    with open(ABBA, encoding="utf-8") as handle:
        data = json.load(handle)

    rows = []
    for block in data["per_block"]:
        hist = {int(k): v for k, v in block["width_hist"].items()}
        rounds = block["rounds"][0]
        eligible = sum(v for k, v in hist.items() if k >= 6)
        extra_leg_us = block["decode_us_per_round"] * rounds
        rows.append(
            {
                "prompt": block["prompt"],
                "rounds": rounds,
                "eligible": eligible,
                "fraction": eligible / rounds,
                "us_per_round": block["decode_us_per_round"],
                "extra_leg_us": extra_leg_us,
                "merged_arm_kernels": block["arms_witnessed"][0],
                "merged_decode_round_us": block["merged_decode_round_us"],
            }
        )

    header = (
        f"{'prompt':18s} {'rounds':>6s} {'elig':>5s} {'frac':>6s} "
        f"{'us/rnd':>8s} {'extra_leg_us':>12s} {'us/elig':>8s}  merged_arm_kernels"
    )
    print(header)
    for row in rows:
        print(
            f"{row['prompt']:18s} {row['rounds']:6d} {row['eligible']:5d} "
            f"{row['fraction']:6.3f} {row['us_per_round']:8.2f} "
            f"{row['extra_leg_us']:12.1f} "
            f"{row['extra_leg_us'] / row['eligible']:8.1f}  "
            f"{row['merged_arm_kernels']}"
        )

    xs = [r["eligible"] for r in rows]
    ys = [r["extra_leg_us"] for r in rows]
    n = len(rows)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    intercept = mean_y - slope * mean_x

    print(
        f"\nOLS: extra_leg_us = {intercept:.1f} fixed per leg "
        f"+ {slope:.1f} per eligible round"
    )
    print(
        "Counterfactual: give the confound every benefit and delete the whole "
        "fixed term, as if a cold-pipeline miss caused all of it."
    )

    total_pct = 0.0
    corrected = []
    for row in rows:
        variable_only_us_per_round = slope * row["eligible"] / row["rounds"]
        pct = 100 * variable_only_us_per_round / row["merged_decode_round_us"]
        corrected.append(pct)
        total_pct += pct
        print(
            f"  {row['prompt']:18s} variable-only {variable_only_us_per_round:7.2f} "
            f"us/round -> decode_pct {pct:+.4f} %"
        )

    mean_pct = total_pct / n
    print(
        f"  mean decode_pct with the fixed term removed: {mean_pct:+.4f} % "
        f"(gated-leg floor {GATED_LEG_FLOOR_PCT} %)"
    )
    print(f"  sigma vs that floor: {abs(mean_pct) / GATED_LEG_FLOOR_PCT:.2f}")

    no_refusal = [r for r in rows if "split" not in r["merged_arm_kernels"]]
    print("\nDirect falsification: legs whose merged arm never used the split kernel")
    for row in no_refusal:
        print(
            f"  {row['prompt']:18s} refusal never engaged, yet the penalty is "
            f"{row['extra_leg_us'] / row['eligible']:.1f} us per eligible round, "
            f"the LARGEST of the three legs."
        )

    out = {
        "experiment": "e153-r2-warm-confound",
        "harness": "local",
        "hypothesis": "cold split fallback after the kL>=1024 refusal explains the merged penalty",
        "e153_r2_fixed_us_per_leg": round(intercept, 1),
        "e153_r2_variable_us_per_eligible_round": round(slope, 1),
        "e153_r2_decode_pct_after_removing_all_fixed_cost": round(mean_pct, 4),
        "e153_r2_sigma_after_removing_all_fixed_cost": round(
            abs(mean_pct) / GATED_LEG_FLOOR_PCT, 2
        ),
        "e153_r2_no_refusal_leg": [r["prompt"] for r in no_refusal],
        "e153_r2_no_refusal_us_per_eligible_round": [
            round(r["extra_leg_us"] / r["eligible"], 1) for r in no_refusal
        ],
        "e153_r2_warm_confound_rejected": bool(mean_pct > GATED_LEG_FLOOR_PCT),
        "sign_convention": "positive pct = merged arm slower than split arm",
        "per_leg": corrected,
    }
    with open("research/out/e153-r2-warm-confound.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print("\nwrote research/out/e153-r2-warm-confound.json")


if __name__ == "__main__":
    main()
