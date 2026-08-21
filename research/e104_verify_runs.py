#!/usr/bin/env python3
"""Verify the published E104 W&B runs and print the keys the advisor asked for."""
import sys

import wandb

PATH = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
KEYS = (
    "a_local_na5", "a_local_na6", "ranked_gain_pct_na5", "ranked_gain_pct_na6",
    "null_control_a_na2", "null_control_a_na3", "last_width_where_collapse_pays",
    "xf_exactfma_bit_identical_cells", "xf_exactfma_change_pct_na5",
    "xf_exactfma_g16s_text_change_pct_na5", "n_nosums_change_pct_na5",
    "f_fmamax_change_pct_na5", "s_splitacc_change_pct_na5",
    "promotion_bar_met", "closure_bar_met_for_exact_arms",
    "h4_fp_issue_saturation_refuted", "air_op_count_predicts_time",
    "isa_text_bytes_predicts_time", "rate_na_axis_recommended_closed",
)


def main():
    api = wandb.Api()
    for rid in sys.argv[1:]:
        run = api.run(f"{PATH}/{rid}")
        print(f"\n{run.name}  id={run.id}  state={run.state}")
        print(f"  {run.url}")
        for k in KEYS:
            if k in run.summary:
                print(f"    {k} = {run.summary[k]}")


if __name__ == "__main__":
    main()
