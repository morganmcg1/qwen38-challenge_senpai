#!/usr/bin/env python3
"""Insert the W&B run reference under every heading that owns a table.

Advisor feedback f2 requires the run URL and id beside every published table.
Doing it by hand invites a stale id, so the mapping from heading to W&B table
key lives here and the lines are regenerated whenever the run changes.

    usage: research/e114_wandb_refs.py <run_id>
"""

from __future__ import annotations

import pathlib
import re
import sys

DOC = pathlib.Path("research/e114-results.md")
URL = ("https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
       "/runs/%s")
MARK = "_W&B "

# heading -> the W&B table keys that publish the numbers under it
TABLES = {
    "### Round counts are independently confirmed":
        ["rung0/round_count_admissibility"],
    "### Ranked mean verify widths":
        ["rung0/ranked_board_inputs"],
    "### The pre-registered validation gate":
        ["rung0/validation_against_traced_truth"],
    "### Why the vector is not identified":
        ["rung0/recovered_na_weights_by_prompt"],
    "### Model A, the advisor's one-parameter prescription, is FALSIFIED":
        ["rung1b/policy_fit_per_prompt"],
    "### Model B adds one parameter and fits both moments":
        ["rung1b/policy_fit_per_prompt"],
    "### But model B FAILS its own pre-registered held-out gate":
        ["rung1b/held_out_width_census_validation",
         "rung1b/policy_shape_against_rung0_band"],
    "### What rung 1b is worth":
        ["rung1b/structural_gates"],
    "### The published operating point weights prompts, not rounds":
        ["rung1/published_score_sensitivity_mix"],
    "### Provenance check":
        ["rung1/provenance_reproduce_ledger_column"],
    "### The weight vectors":
        ["deliverable/na_weight_vectors"],
    "### The deliverable an arm owner can apply without this script":
        ["deliverable/na_weight_vectors"],
    "### The arm table":
        ["deliverable/arm_rerank"],
    "### What IS decisive":
        ["deliverable/sign_invariance_verdicts"],
    "### Reconciliation against the advisor's own reweighting":
        ["rung1/reconciliation_against_advisor"],
    "### The `weighted % \u2192 round %` factor moves too":
        ["deliverable/weighted_to_round_factor"],
    "### Arms that cannot be re-weighted at all":
        ["rung1/arms_without_per_na_cells"],
    "### Share of published-weighted candidate TIME at exactly verify width 5":
        ["item5/m5_share_of_candidate_time"],
    "### The arithmetic":
        ["item5/collapse_net_of_register_tax"],
    "### The fourth price is rejected":
        ["item5/curve_difference_placebo"],
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_id = sys.argv[1]
    lines, out, seen = DOC.read_text().splitlines(), [], set()
    for line in lines:
        if line.startswith(MARK):
            continue
        out.append(line)
        for head, keys in TABLES.items():
            if line.startswith(head):
                seen.add(head)
                out.append("")
                out.append("%srun `%s`, <%s>, table%s %s._"
                           % (MARK, run_id, URL % run_id,
                              "" if len(keys) == 1 else "s",
                              ", ".join("`%s`" % k for k in keys)))
                break
    missing = sorted(set(TABLES) - seen)
    if missing:
        for head in missing:
            print("FAIL heading not found: %s" % head)
        return 1
    DOC.write_text("\n".join(out) + "\n")
    print("annotated %d headings with run %s" % (len(seen), run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
