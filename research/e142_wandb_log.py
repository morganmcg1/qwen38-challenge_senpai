#!/usr/bin/env python3
"""Publish an E142 result to W&B.

    usage: research/e142_wandb_log.py --rung r0 [--dry]

RUNG 0 IS NOT A TIMED MEASUREMENT. It replays cached 512-token goldens through
an instrumented, untimed `mtp-verify` leg to capture the post-norm hidden rows
that reach the scored verify readout, and then screens the target `lm_head`
offline. It therefore logs `timing_valid`, `cool_gate_passed_real_gate`,
`gate_qualified_for_timing` and `official_or_ranked_score` verbatim as false,
and it produces no score.

The primary assignment metric `e142_net_ranked_pct` is a rung 3 quantity. Rung
0 logs it as null rather than as zero, because a missing measurement and a
measured null are different claims.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e142-certified-verify-readout"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"

# Assignment section 3: the medpair-weighted share of the scored round that the
# target lm_head pass occupies, and the round time RULE 115 prices against.
LM_HEAD_SHARE_PCT = 6.6
LM_HEAD_SHARE_BAND = (4.5, 7.5)
MEDPAIR_ROUND_US = 52860.0
# Assignment section 7: the rung 0 stop rule.
STOP_MEDIAN_BYTES = 0.45
STOP_P95_BYTES = 0.70

SURVIVAL_COLUMNS = [
    "screen", "threshold", "screen_bytes", "row_survival_median",
    "row_survival_p95", "row_survival_worst", "byte_fraction_median",
    "byte_fraction_p95", "byte_fraction_worst", "round_bytes_median",
    "ranked_pct_if_shipped",
]

WIDTH_COLUMNS = [
    "screen", "threshold", "width", "rounds", "row_survival_median",
    "union_rows_median", "byte_fraction_median", "byte_fraction_p95",
]

SEED_COLUMNS = ["screen", "threshold", "seed", "row_survival_median"]

LOWRANK_COLUMNS = [
    "rank", "weight_energy", "h_perp_over_h_median", "rho_mean",
    "slack_median", "screen_bytes", "row_survival_median",
    "row_survival_p95", "byte_fraction_median", "byte_fraction_p95",
    "epsilon", "ranked_pct_if_shipped",
]

# One row per screen family. `epsilon` is the median certified slack divided by
# the median Cauchy-Schwarz scale ||h|| * ||w_v||, so it measures how tight a
# bound is in units that do not depend on the screen's internals. A screen
# prunes only when `epsilon` drops below `budget_over_cauchy_schwarz`.
EPSILON_COLUMNS = [
    "screen", "variant", "slack_median", "epsilon", "slack_over_budget",
    "prunes_anything",
]

CAPTURE_COLUMNS = [
    "seed", "rounds", "accepted_draft_rate", "effective_mean_draft_len",
    "effective_max_draft_len", "parity_all_ok", "all_tokens_matched",
    "residual_divergence_count", "declared_rows_total",
    "reference_checked_row_total", "head_provenance_sha256",
]


def ranked_pct(byte_fraction: float) -> float:
    """Ranked candidate-leg percent this byte fraction would buy, if shipped.

    The readout pass is bandwidth bound (assignment section 3: 715 MB in
    3,642 us is 196 GB/s, and E93 measured 186.7 GB/s independently), so time
    tracks bytes inside this pass. A byte fraction of 1.0 buys nothing and a
    fraction above 1.0 costs time.
    """
    return LM_HEAD_SHARE_PCT * (1.0 - byte_fraction)


def rows_from(report: dict):
    survival, widths, seeds = [], [], []
    for screen, entry in report["screens"].items():
        for threshold, block in entry["thresholds"].items():
            survival.append({
                "screen": screen,
                "threshold": threshold,
                "screen_bytes": entry["screen_bytes"],
                "row_survival_median": block["row_survival_fraction_median"],
                "row_survival_p95": block["row_survival_fraction_p95"],
                "row_survival_worst": block["row_survival_fraction_worst"],
                "byte_fraction_median": block["round_byte_fraction_median"],
                "byte_fraction_p95": block["round_byte_fraction_p95"],
                "byte_fraction_worst": block["round_byte_fraction_worst"],
                "round_bytes_median": block["round_bytes_median"],
                "ranked_pct_if_shipped": ranked_pct(
                    block["round_byte_fraction_median"]),
            })
            for width, cell in block["by_width"].items():
                widths.append({
                    "screen": screen, "threshold": threshold,
                    "width": int(width), "rounds": cell["rounds"],
                    "row_survival_median": cell["row_survival_fraction_median"],
                    "union_rows_median": cell["union_rows_median"],
                    "byte_fraction_median": cell["byte_fraction_median"],
                    "byte_fraction_p95": cell["byte_fraction_p95"],
                })
            for seed, value in block["by_seed"].items():
                seeds.append({
                    "screen": screen, "threshold": threshold, "seed": seed,
                    "row_survival_median": value,
                })
    return survival, widths, seeds


def table(columns, rows):
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def lowrank_rows(lowrank: dict, cauchy_schwarz: float, budget: float):
    rows = []
    for rank, cell in sorted(lowrank["ranks"].items(), key=lambda kv: int(kv[0])):
        rows.append({
            "rank": int(rank),
            "weight_energy": cell["weight_energy"],
            "h_perp_over_h_median": cell["h_perp_over_h_median"],
            "rho_mean": cell["rho_mean"],
            "slack_median": cell["slack_median"],
            "screen_bytes": cell["screen_bytes"],
            "row_survival_median": cell["row_survival_fraction_median"],
            "row_survival_p95": cell["row_survival_fraction_p95"],
            "byte_fraction_median": cell["round_byte_fraction_median"],
            "byte_fraction_p95": cell["round_byte_fraction_p95"],
            "epsilon": cell["slack_median"] / cauchy_schwarz,
            "ranked_pct_if_shipped": ranked_pct(
                cell["round_byte_fraction_median"]),
        })
    return rows


def epsilon_rows(report: dict, lowrank: dict | None):
    geometry = report["logit_geometry"]
    cauchy_schwarz = geometry["cauchy_schwarz_median"]
    budget = report["slack_budget"]["budget_median"]
    entries = [
        ("S-A", "block max-norm", geometry["sa_median_slack"]),
        ("S-B", "affine-2 group-64 copy", geometry["sb_median_slack"]),
        ("S-C", "sampled leaf radius (ceiling)",
         report["S-C_ceiling"]["slack_median"]),
    ]
    if lowrank:
        for rank, cell in sorted(lowrank["ranks"].items(),
                                 key=lambda kv: int(kv[0])):
            entries.append(("S-D", f"rank {rank}", cell["slack_median"]))
    rows = []
    for screen, variant, slack in entries:
        rows.append({
            "screen": screen, "variant": variant, "slack_median": slack,
            "epsilon": slack / cauchy_schwarz,
            "slack_over_budget": slack / budget,
            "prunes_anything": slack < budget,
        })
    return rows


def r0_summary(report: dict, capture: list[dict], lowrank: dict | None = None):
    survival, widths, seeds = rows_from(report)
    # The best screen is the one that reads the fewest bytes at a valid
    # threshold, which is the quantity the stop rule names.
    best = min(survival, key=lambda r: r["byte_fraction_median"])
    ceiling = report.get("S-C_ceiling", {})
    budget = report["slack_budget"]
    geometry = report["logit_geometry"]
    epsilon = epsilon_rows(report, lowrank)
    lowrank_table = lowrank_rows(
        lowrank, geometry["cauchy_schwarz_median"],
        budget["budget_median"]) if lowrank else []
    if lowrank_table:
        best = min([best] + [{**r, "screen": "S-D",
                              "threshold": f"oracle rank {r['rank']}",
                              "byte_fraction_worst": r["byte_fraction_p95"]}
                             for r in lowrank_table],
                   key=lambda r: r["byte_fraction_median"])
    stop = (best["byte_fraction_median"] > STOP_MEDIAN_BYTES
            and best["byte_fraction_p95"] > STOP_P95_BYTES)
    summary = {
        # PRIMARY. Rung 3 measures it; rung 0 does not.
        "e142_net_ranked_pct": None,
        "e142_certified_survival_fraction_median": best["byte_fraction_median"],
        "e142_certified_survival_fraction_p95": best["byte_fraction_p95"],
        "e142_certified_survival_fraction_worst": best["byte_fraction_worst"],
        "e142_best_screen": best["screen"],
        "e142_best_threshold": best["threshold"],
        "e142_best_ranked_pct_if_shipped": best["ranked_pct_if_shipped"],
        # No threshold used here can exceed the true s2, so no certificate can
        # fail by construction. The measured quantity is the number of rows
        # where a constructed threshold rose above the device s2 by more than
        # the offline-to-device arithmetic gap.
        "e142_certificate_failure_rate": 0.0,
        "e142_threshold_validity_violations": sum(
            v["rows_over_tolerance"] for v in report["threshold_check"].values()),
        "e142_exactness_divergences": sum(
            c["residual_divergence_count"] for c in capture),
        "e142_capture_parity_all_ok": all(c["parity_all_ok"] for c in capture),
        "e142_capture_all_tokens_matched": all(
            c["all_tokens_matched"] for c in capture),
        "e142_screen_added_us_per_round": None,
        "e142_top2_bit_identity": None,
        "e142_rows": report["rows"],
        "e142_rounds": report["rounds"],
        "e142_stop_rule_fires": bool(stop),
        # Why the bounds behave as they do, in one ratio each.
        "e142_slack_budget_median": budget["budget_median"],
        "e142_sa_slack_over_budget": budget["sa_slack_over_budget"],
        "e142_sb_slack_over_budget": budget["sb_slack_over_budget"],
        "e142_sb_actual_error_over_budget": budget["sb_actual_error_over_budget"],
        "e142_sb_required_bits_for_certificate":
            budget["sb_required_bits_for_certificate"],
        "e142_sc_ceiling_prune_fraction_oracle_median":
            ceiling.get("prune_fraction_oracle_median"),
        "e142_sc_leaf_radius_median": ceiling.get("leaf_radius_median"),
        "e142_sc_slack_median": ceiling.get("slack_median"),
        "e142_hidden_l1_over_l2_median":
            report["hidden_geometry"]["l1_over_l2_median"],
        "e142_sqrt_hidden": report["hidden_geometry"]["sqrt_hidden"],
        "e142_draft_token_is_argmax_rate":
            report["seed_value"]["draft_token_is_argmax_rate"],
        "e142_offline_argmax_matches_device":
            report["validation"]["offline_argmax_matches_device"],
        "e142_offline_s1_abs_delta_max": report["validation"]["s1_abs_delta_max"],
        "e142_lm_head_share_pct": LM_HEAD_SHARE_PCT,
        "e142_medpair_round_us": MEDPAIR_ROUND_US,
        "e142_cauchy_schwarz_median": geometry["cauchy_schwarz_median"],
        "e142_budget_over_cauchy_schwarz":
            budget["budget_median"] / geometry["cauchy_schwarz_median"],
        "e142_min_measured_epsilon": min(r["epsilon"] for r in epsilon),
        "e142_screen_families_that_prune": sum(
            1 for r in epsilon if r["prunes_anything"]),
        "harness": "ranked",
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }
    if lowrank_table:
        tightest = min(lowrank_table, key=lambda r: r["epsilon"])
        summary.update({
            "e142_sd_ranks": [r["rank"] for r in lowrank_table],
            "e142_sd_tightest_rank": tightest["rank"],
            "e142_sd_tightest_epsilon": tightest["epsilon"],
            "e142_sd_tightest_row_survival_median":
                tightest["row_survival_median"],
            "e142_sd_best_byte_fraction_median": min(
                r["byte_fraction_median"] for r in lowrank_table),
            "e142_sd_weight_energy_at_tightest_rank": tightest["weight_energy"],
            "e142_sd_h_perp_over_h_at_tightest_rank":
                tightest["h_perp_over_h_median"],
        })
    tables = {
        "survival": table(SURVIVAL_COLUMNS, survival),
        "by_width": table(WIDTH_COLUMNS, widths),
        "by_seed": table(SEED_COLUMNS, seeds),
        "capture": table(CAPTURE_COLUMNS, capture),
        "epsilon": table(EPSILON_COLUMNS, epsilon),
    }
    if lowrank_table:
        tables["lowrank"] = table(LOWRANK_COLUMNS, lowrank_table)
    return summary, tables


def load_capture(paths: list[pathlib.Path]) -> list[dict]:
    rows = []
    for path in paths:
        blob = json.loads(path.read_text())
        rows.append({
            "seed": path.stem,
            "rounds": blob["round_count"],
            "accepted_draft_rate": blob["accepted_draft_rate"],
            "effective_mean_draft_len": blob["effective_mean_draft_len"],
            "effective_max_draft_len": blob["effective_max_draft_len"],
            "parity_all_ok": blob["parity_all_ok"],
            "all_tokens_matched": blob["all_tokens_matched"],
            "residual_divergence_count": blob["residual_divergence_count"],
            "declared_rows_total": blob["declared_rows_total"],
            "reference_checked_row_total": blob["reference_checked_row_total"],
            "head_provenance_sha256": blob["head_provenance"]["sha256"],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=["r0"])
    ap.add_argument("--report", default="research/e142-r0.json")
    ap.add_argument("--lowrank", default="research/e142-r0-lowrank.json")
    ap.add_argument("--capture-dir", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    lowrank_path = pathlib.Path(args.lowrank)
    lowrank = json.loads(lowrank_path.read_text()) if lowrank_path.exists() else None
    capture_dir = pathlib.Path(args.capture_dir) if args.capture_dir else (
        pathlib.Path.home()
        / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e142/verifyrows/verify")
    capture = load_capture(
        [capture_dir / f"{seed}.json" for seed in report["seeds"]])
    summary, tables = r0_summary(report, capture, lowrank)

    config = {
        "experiment": "e142",
        "rung": args.rung,
        "question": (
            "how much of the 248,320-row target lm_head can a PROVABLE screen "
            "skip in the scored verify readout while returning bit-identical "
            "top-2 evidence"),
        "reproduce": (
            "research/e142_rung0.sh && "
            "research/e133_job.sh research/e142_r0.py "
            "--out research/e142-r0.json && "
            "research/e133_job.sh research/e142_r0_lowrank.py "
            "--out research/e142-r0-lowrank.json"),
        "host": HOST,
        "commit": report["commit"],
        "seeds": report["seeds"],
        "source_file": args.report,
        "lowrank_file": str(lowrank_path) if lowrank else None,
    }
    if args.dry:
        print(json.dumps({"config": config, "summary": summary}, indent=2,
                         default=str))
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name="e142-r0-certified-survival", job_type="offline-screen",
                     config=config)
    run.log(tables)
    run.summary.update(summary)
    print(f"run_id {run.id}")
    print(f"url    {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
