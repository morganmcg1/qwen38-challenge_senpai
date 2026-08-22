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


def r0_summary(report: dict, capture: list[dict]):
    survival, widths, seeds = rows_from(report)
    # The best screen is the one that reads the fewest bytes at a valid
    # threshold, which is the quantity the stop rule names.
    best = min(survival, key=lambda r: r["byte_fraction_median"])
    ceiling = report.get("S-C_ceiling", {})
    budget = report["slack_budget"]
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
        "harness": "ranked",
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }
    tables = {
        "survival": table(SURVIVAL_COLUMNS, survival),
        "by_width": table(WIDTH_COLUMNS, widths),
        "by_seed": table(SEED_COLUMNS, seeds),
        "capture": table(CAPTURE_COLUMNS, capture),
    }
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
    ap.add_argument("--capture-dir", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    capture_dir = pathlib.Path(args.capture_dir) if args.capture_dir else (
        pathlib.Path.home()
        / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e142/verifyrows/verify")
    capture = load_capture(
        [capture_dir / f"{seed}.json" for seed in report["seeds"]])
    summary, tables = r0_summary(report, capture)

    config = {
        "experiment": "e142",
        "rung": args.rung,
        "question": (
            "how much of the 248,320-row target lm_head can a PROVABLE screen "
            "skip in the scored verify readout while returning bit-identical "
            "top-2 evidence"),
        "reproduce": (
            "research/e142_rung0.sh && "
            "python3 research/e142_r0.py --out research/e142-r0.json"),
        "host": HOST,
        "commit": report["commit"],
        "seeds": report["seeds"],
        "source_file": args.report,
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
