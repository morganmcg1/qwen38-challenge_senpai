#!/usr/bin/env python3
"""Attach the E55 register census to the headline bracket run's W&B summary.

The 15:02Z retag destroyed a peer's workspace mid-session, so every reading that
matters is pushed off this machine as soon as it exists rather than at the end.

  python3 research/e55_log_census.py --run wxezisvs
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
SENSITIVITY_PCT_PER_F9 = 8.49751
F9_LOCAL = 0.55435
MEASURED_MTP_LEG_PCT = -4.2952
# E49 Arm 2, merged: control-free contrasts bound the shared-register harm.
E49_ARM2_HARM_BOUND_PCT = 0.0876
RANKED_MIXTURES = {"e48": 0.21630, "edward_upper": 0.08900, "edward_lower": 0.04600}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--census", default="research/e55-reg-census.json")
    args = ap.parse_args()

    c = json.loads(pathlib.Path(args.census).read_text())
    wide = c["kernel_wide_reg_max"]
    b0 = c["entry_batch0"]

    predicted = -SENSITIVITY_PCT_PER_F9 * F9_LOCAL
    shortfall = MEASURED_MTP_LEG_PCT - predicted

    summary = {
        "census/kernel_wide_reg_max_base": wide["base_na4_table"],
        "census/kernel_wide_reg_max_candidate": wide["m9two_candidate"],
        "census/kernel_wide_reg_max_e27": wide["e27_both_cells"],
        "census/entry_batch0_base": b0["base_na4_table"],
        "census/entry_batch0_candidate": b0["m9two_candidate"],
        "census/entry_batch0_e27": b0["e27_both_cells"],
        "census/kernel_wide_reg_max_delta": c["candidate_vs_base"]["kernel_wide_reg_max_delta"],
        "census/entry_batch0_delta": c["candidate_vs_base"]["entry_batch0_delta"],
        "census/candidate_rises_above_base": c["candidate_vs_base"]["candidate_rises_above_base"],
        "census/candidate_carries_full_e27_allocation":
            c["candidate_vs_e27"]["candidate_carries_full_e27_allocation"],
        "census/positive_control_fired": c["positive_control_fired"],
        "census/recorded_value_checks_all_pass": all(
            v[2] for res in c["recorded_value_checks"].values()
            for v in res.values()),
        "census/widths_with_unchanged_reg_count": json.dumps(
            c["widths_unchanged_vs_base"]),
        "census/harm_only_arm_blocked_by": "static_assert(M >= 3 && M <= 9)",
        # E27's case-5 cell is a LONE NA=5 group: streams(M=5, IPG=5) = 1. The
        # candidate's M=9 cell is NA=5 plus an NA=4 sibling. PR #8 measured the
        # collapse on a lone group, so the two are not the same configuration.
        "census/e27_m5_streams": 1,
        "census/candidate_m9_streams": 2,
        "census/e27_m5_reg": 125,
        "pricing/local_predicted_mtp_leg_pct": predicted,
        "pricing/local_measured_mtp_leg_pct": MEASURED_MTP_LEG_PCT,
        "pricing/local_shortfall_pp": shortfall,
        "pricing/e49_arm2_harm_bound_pct": E49_ARM2_HARM_BOUND_PCT,
    }
    for name, f9 in RANKED_MIXTURES.items():
        gain = -SENSITIVITY_PCT_PER_F9 * f9
        summary["pricing/ranked_%s_gain_only_pct" % name] = gain
        summary["pricing/ranked_%s_with_e49_harm_pct" % name] = gain + E49_ARM2_HARM_BOUND_PCT
        # `shortfall` is POSITIVE (a slowdown term), so a width-independent harm
        # ADDS to the gain rather than subtracting from it.
        summary["pricing/ranked_%s_with_full_shortfall_pct" % name] = gain + shortfall
        # E27 essays leg: what case 5 plus the shared harm must be worth for
        # E27's +0.4803 % board result to hold at this mixture.
        summary["pricing/e27_essays_case5_plus_harm_%s_pct" % name] = 0.4803 - gain

    api = wandb.Api()
    run = api.run("%s/%s" % (PROJECT, args.run))
    run.summary.update(summary)
    run.update()
    print("updated %s with %d census/pricing keys" % (run.url, len(summary)))
    for k in sorted(summary):
        print("  %-58s %s" % (k, summary[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
