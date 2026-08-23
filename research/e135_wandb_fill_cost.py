#!/usr/bin/env python3
"""Publish the chunk-sum fill-cost session to W&B, caveats attached.

The session prices one thing: what the `xsums_v1` fill dispatch costs inside
the real command stream, with the draft schedule held fixed so the contrast is
pure cost and Rule 79 does not bind.

Two numbers leave this session and they are not equally solid.
`e135_fill_total_us_per_round` is measured: the arms remove every fill at once.
`e135_fill_cost_us_per_dispatch` divides that total by 257 and therefore
assumes the relation is linear, which these arms cannot test. The names, the
config caveats and `e135_fill_marginal_tested` all say so, so nobody reads the
per-dispatch figure as measured.

  research/e135_wandb_fill_cost.py --label fill [--dry]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_fill_cost as fc  # noqa: E402

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"

LOCAL_ROUNDS = 78
LOCAL_TOKENS = 512

# Rule 135, same host, same campaign: what one launch overhead column costs.
# A whole fill dispatch below this is direct evidence of command-stream overlap.
OVERHEAD_COLUMN_US = 2.6815

# Rule 134 and Finding 255, for the ranked frame. Both constants are stated in
# the total-leg frame the score is computed in: seed prefill is inside the
# timed leg, so dividing by the decode leg alone would inflate them by ~11 %.
# The campaign figure is the carried one; the exact figure comes from the
# decode round counts recovered from our own receipt.
RULE134_US_PER_ROUND_PER_PCT = 515.2
RULE134_US_PER_ROUND_PER_PCT_EXACT = 524.5
GAP_TO_CROWN_TREE_PCT = 0.4006
TARGET_ABOVE_BAR_PCT = 0.9896

# Finding 252. The pooled saving the two rival Idea 3 receipts show, over the
# five prompts where `tablePays` is true. Inverting it against a site count
# gives the M5 cost per fill.
RIVAL_US_PER_ROUND = 78.4
RIVAL_US_PER_ROUND_SD = 17.3
SITE_READINGS = {"in_projection_qkv_only": 16, "in_projection_both": 64,
                 "boundary_fused_all": 128, "every_wide_qmv": 257}


def read_meta(path: str) -> dict[str, str]:
    out = {}
    for line in open(os.path.join(path, "meta.txt")):
        if "=" in line:
            key, _, value = line.strip().partition("=")
            out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="fill")
    ap.add_argument("--name", default="e135-fill-cost")
    ap.add_argument("--resume", help="append to this existing run id")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    legs = fc.load_legs(args.label)
    if not legs:
        raise SystemExit("no timed legs found")
    stats = fc.fit(legs)

    total_us = (stats["fill"] - stats["replica"]) * LOCAL_TOKENS \
        / LOCAL_ROUNDS * 1e6
    per_fill_us = total_us / fc.QMV_PER_FORWARD
    local_round_us = stats["replica"] * LOCAL_TOKENS / LOCAL_ROUNDS * 1e6

    se_fill_us = stats["se1_pct"] / 100.0 * local_round_us \
        / fc.QMV_PER_FORWARD
    session_us = sorted(v / 100.0 * local_round_us / fc.QMV_PER_FORWARD
                        for v in stats["session_effects_pct"].values())

    metrics = {
        # Measured.
        "e135_fill_total_pct_of_candidate_mtp": stats["effect_pct"],
        "e135_fill_total_us_per_round": total_us,
        "e135_fill_se_mean_arm_pp": stats["se1_pct"],
        "e135_fill_se_drift_fitted_pp": stats["se3_pct"],
        "e135_fill_se_no_drift_pp": stats["se2_pct"],
        "e135_fill_dof_mean_arm": stats["dof1"],
        "e135_fill_dof_drift_fitted": stats["dof3"],
        "e135_fill_dof_no_drift": stats["dof2"],
        "e135_fill_t_mean_arm": abs(stats["effect_pct"]) / stats["se1_pct"],
        "e135_fill_t_no_drift": abs(stats["effect_pct"]) / stats["se2_pct"],
        "e135_fill_position_drift_pct_per_leg": stats["drift_pct_per_leg"],
        "e135_fill_replica_s_per_tok": stats["replica"],
        "e135_fill_noconsume_s_per_tok": stats["fill"],
        "e135_fill_local_round_us": local_round_us,
        "e135_fill_legs": len(legs),
        "e135_fill_sessions": len({d["rep"] for d in legs}),
        "e135_fill_distinct_schedules": len({round(d["edl"], 9) for d in legs}),
        # Derived under an untested linearity assumption. See the config.
        "e135_fill_cost_us_per_dispatch": per_fill_us,
        "e135_fill_cost_us_per_dispatch_se": se_fill_us,
        "e135_fill_cost_us_per_dispatch_dof": stats["dof1"],
        "e135_fill_cost_us_per_dispatch_session_lo": session_us[0],
        "e135_fill_cost_us_per_dispatch_session_hi": session_us[-1],
        "e135_fill_marginal_tested": False,
        # Rule 135 cross-check. The T55 claim that one fill is cheaper than one
        # overhead column is withdrawn: the gap is well inside the interval.
        "e135_fill_vs_overhead_column_ratio": per_fill_us / OVERHEAD_COLUMN_US,
        "e135_fill_vs_overhead_column_t":
            (OVERHEAD_COLUMN_US - per_fill_us) / se_fill_us,
        "e135_fill_cheaper_than_overhead_column_resolved":
            (OVERHEAD_COLUMN_US - per_fill_us) / se_fill_us >= 2.0,
        # Idea 3, both frames named so neither can be quoted as the other.
        "e135_idea3_ceiling_pct_harness_local":
            per_fill_us * fc.FUSABLE_PER_FORWARD / local_round_us * 100.0,
        "e135_idea3_fusable_share":
            fc.FUSABLE_PER_FORWARD / fc.QMV_PER_FORWARD,
    }

    # Invert Finding 252 against each reading of what the rivals fused. The M5
    # cost per fill follows from the site count, not the other way round.
    for name, sites in SITE_READINGS.items():
        implied = RIVAL_US_PER_ROUND / sites
        metrics[f"e135_finding252_implied_m5_us_per_fill_{name}"] = implied
        metrics[f"e135_finding252_m5_to_m4pro_ratio_{name}"] = \
            implied / per_fill_us
    for sites in (64, 128, 192, 257):
        for src, us in (("m5fit", RIVAL_US_PER_ROUND / 64),
                        ("m4pro", per_fill_us)):
            for frame, per_pct in (
                    ("", RULE134_US_PER_ROUND_PER_PCT),
                    ("_exact", RULE134_US_PER_ROUND_PER_PCT_EXACT)):
                pct = sites * us / per_pct
                key = f"{sites}sites_{src}{frame}"
                metrics[f"e135_idea3_ranked_pct_{key}"] = pct
                metrics[f"e135_idea3_share_of_crown_gap_{key}"] = \
                    pct / GAP_TO_CROWN_TREE_PCT
                metrics[f"e135_idea3_share_of_bar_target_{key}"] = \
                    pct / TARGET_ABOVE_BAR_PCT

    first = read_meta(os.path.join("research/out", legs[0]["tag"]))
    config = {
        "experiment": "e135-xsums-fill-dispatch-cost",
        "harness": "local",
        "arms": "replica (no fill) vs fill_noconsume (fill runs, never read)",
        "composition_held":
            "tight grid + onePass67 + p15 probe + width2 route + depth arm ship",
        "design": "counterbalanced palindrome repl fill fill repl per session",
        "gated": "every timed leg passed the real 40 C gate",
        "session_commit": first.get("e135_session_commit"),
        "session_worker_sha256": first.get("e135_session_worker_sha256"),
        "cool_gate_passed_real_gate": first.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": first.get("gate_qualified_for_timing"),
        "official_or_ranked_score": first.get("official_or_ranked_score"),
        "host": first.get("host"),
        "chip": first.get("chip"),
        "caveat_total_not_marginal":
            "the arms remove all 257 fills at once, so the total is measured "
            "and the per-dispatch figure assumes linearity these arms cannot "
            "test; the fillHidden arm at 129 sites is what would settle it",
        "caveat_kernel_object":
            "replica runs qmv_wide_na{tier}_v2 and fill_noconsume runs "
            "qmv_sums_na{tier}_v2 with USE_TABLE=false, so the contrast is the "
            "fill dispatch plus any pipeline-object difference, which makes it "
            "an upper bound on the fill alone",
        "caveat_rule83":
            "this is an M4 Pro cost and transfers to the ranked M5 as a "
            "dispatch count, not as microseconds",
        "caveat_interval":
            "the published width is the mean-and-arm fit at 6 dof, which "
            "leaves drift and the between-session step in the residual; the "
            "narrower 4 and 5 dof fits condition those away and are reported "
            "beside it only to show what the conditioning is worth",
        "caveat_session_spread":
            "the two sessions return %.3f and %.3f us per dispatch; the 6 dof "
            "interval covers that spread and the conditioned intervals do not"
            % (session_us[0], session_us[-1]),
    }

    per_leg = [
        {k: d[k] for k in
         ("tag", "rep", "position", "arm", "mtp", "serial", "edl",
          "entry_c", "exit_c", "gated")}
        for d in legs
    ]

    if args.dry:
        print(json.dumps({"config": config, "metrics": metrics,
                          "legs": per_leg}, indent=2, sort_keys=True))
        return

    import wandb

    entity, project = PROJECT.split("/")
    extra = {"id": args.resume, "resume": "must"} if args.resume else {}
    run = wandb.init(entity=entity, project=project, name=args.name,
                     job_type="local-cost-contrast", config=config, **extra)
    run.config.update(config, allow_val_change=True)
    run.log(metrics)
    run.log({"e135_fill_legs_table": wandb.Table(
        columns=list(per_leg[0]), data=[list(r.values()) for r in per_leg])})
    print(f"run {run.id} {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
