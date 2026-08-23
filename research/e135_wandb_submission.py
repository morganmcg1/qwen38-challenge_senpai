#!/usr/bin/env python3
"""Publish the E135 shipping-arm submission record to W&B.

`e135_wandb_log.py` fits multi-leg ABBA contrasts. This leg is not a contrast:
it is the single gated exactness leg that certifies the archive actually
submitted, plus the ranked evidence that retired the `pb6` depth-price arm.
Logging it through the contrast estimator would invent an effect that was
never measured, so it gets its own run.

Once the ranked receipt returns, `--resume` folds it into the same run rather
than opening a second one. The local leg and the receipt it produced are one
record: the leg certifies what was archived, the receipt says what it scored.

  research/e135_wandb_submission.py --leg research/out/e135shipexact \
      --submission 0cf1637e-b722-4544-9806-94927d28e558 [--dry]
  research/e135_wandb_submission.py --leg research/out/e135shipexact \
      --submission 0cf1637e-... --resume 9ptcijih --board /tmp/yukon-board/read.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_receipt_read as rr  # noqa: E402

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"

# The crown tree, three draws of one bit-identical solver (Finding 251).
CROWN_DRAWS = ("f7d59543", "684821ed", "165d4ba7")
CROWN_TREE_FAIR_MEDIAN = 3.70486350
OUR_PREVIOUS_BEST = ("572b2cc4", 3.66218564)

# Rule 114 arm signatures, from the schedule the run itself produces.
SHIP_EDL = 6.358974358974359
PB6_EDL = 5.853658536585366

# F34 same-host reference for the shipping arm, for a drift read.
F34_SHIP_MTP = 0.029142
F34_SHIP_SERIAL = 0.073574
F34_SHIP_ROUNDS = 78


def read_meta(path: str) -> dict[str, str]:
    out = {}
    for line in open(os.path.join(path, "meta.txt")):
        if "=" in line:
            key, _, value = line.strip().partition("=")
            out[key] = value
    return out


def receipt_record(board_path: str, submission: str) -> tuple[dict, dict]:
    """Ranked metrics and config for the receipt this submission produced."""
    board = rr.load_board(board_path)
    ours = rr.find(board, submission[:8])
    mine = rr.per_prompt(ours)
    lo, hi = rr.median_pair(ours)
    published = ours["officialScore"]
    us_round = rr.candidate_us_per_round(ours)

    metrics: dict[str, float | bool | int] = {
        "e135_ranked_published_median": published,
        "e135_ranked_median_slot_lower_raw": mine[lo]["raw_ratio_of_means"],
        "e135_ranked_median_slot_upper_raw": mine[hi]["raw_ratio_of_means"],
        # Rule 114 read from the receipt's own published schedule.
        "e135_ranked_plutarch_edl":
            mine["plutarch"]["effective_mean_draft_len"],
        "e135_ranked_plutarch_non_drafting_rounds":
            mine["plutarch"]["non_drafting_round_count"],
        "e135_ranked_arm_is_ship":
            mine["plutarch"]["effective_mean_draft_len"] < 1.0,
        # Headroom, both against the published crown and against the tree
        # Finding 251 showed it to be, stripped of its serial draw.
        "e135_ranked_short_of_published_bar_pct":
            (3.71959722580154 / published - 1.0) * 100.0,
        "e135_ranked_short_of_fair_tree_pct":
            (CROWN_TREE_FAIR_MEDIAN / published - 1.0) * 100.0,
        "e135_ranked_gain_on_our_previous_best_pct":
            (published / OUR_PREVIOUS_BEST[1] - 1.0) * 100.0,
    }
    for name, entry in mine.items():
        metrics[f"e135_ranked_raw_{name}"] = entry["raw_ratio_of_means"]
        metrics[f"e135_ranked_cand_spt_{name}"] = \
            entry["mtp_seconds_per_token_mean"]
        metrics[f"e135_ranked_serial_spt_{name}"] = \
            entry["serial_seconds_per_token_mean"]
        metrics[f"e135_ranked_prefill_spt_{name}"] = \
            entry["prefill_seconds_per_token"]
        metrics[f"e135_ranked_edl_{name}"] = \
            entry["effective_mean_draft_len"]
        metrics[f"e135_ranked_cand_us_per_round_{name}"] = us_round[name]

    for ref_id in CROWN_DRAWS + (OUR_PREVIOUS_BEST[0],):
        ref = rr.find(board, ref_id)
        deltas = rr.compare(ours, ref, "mtp_seconds_per_token_mean")
        mean, sd, se = rr.summarise(deltas)
        pair = statistics.fmean(deltas[n] for n in (lo, hi))
        metrics[f"e135_ranked_cand_pair_pct_vs_{ref_id}"] = pair
        metrics[f"e135_ranked_cand_r148_pct_vs_{ref_id}"] = \
            rr.weighted(deltas, rr.RULE148_WEIGHTS)
        metrics[f"e135_ranked_cand_f83_pct_vs_{ref_id}"] = \
            rr.weighted(deltas, rr.F83_WEIGHTS)
        metrics[f"e135_ranked_cand_mean8_pct_vs_{ref_id}"] = mean
        metrics[f"e135_ranked_cand_mean8_sd_vs_{ref_id}"] = sd
        metrics[f"e135_ranked_cand_mean8_se_vs_{ref_id}"] = se
        detected = sum(
            1 for n in deltas
            if abs(deltas[n]) / 100.0 * us_round[n]
            >= rr.RULE147_DETECT_US_PER_ROUND[n])
        metrics[f"e135_ranked_r147_detected_of_8_vs_{ref_id}"] = detected

    prefill = rr.compare(ours, rr.find(board, "684821ed"),
                         "prefill_seconds_per_token")
    pmean, psd, pse = rr.summarise(prefill)
    metrics["e135_ranked_prefill_mean8_pct_vs_684821ed"] = pmean
    metrics["e135_ranked_prefill_sd_vs_684821ed"] = psd
    metrics["e135_ranked_prefill_inside_null_band"] = \
        abs(pmean) < rr.PREFILL_NULL_BAND_PCT

    config = {
        "ranked_status": ours.get("status"),
        "ranked_promotion_status": str(ours.get("promotionStatus")),
        "ranked_receipt_commit": ours.get("submissionCommitSha"),
        "ranked_created_at": ours.get("createdAt"),
        "ranked_median_pair": f"{lo},{hi}",
        "ranked_us_per_round_convention":
            "decode_tokens / (1 + effective_mean_draft_len); every draft "
            "counted accepted, matching the Rule 147 table",
    }
    return metrics, config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", default="research/out/e135shipexact")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--label", default="e135-ship-submission")
    ap.add_argument("--resume", help="fold the receipt into this run id")
    ap.add_argument("--board", default="/tmp/yukon-board/read.json")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    meta = read_meta(args.leg)
    score = json.load(open(os.path.join(args.leg, "score.json")))
    m = score["metrics"]

    edl = m["effective_mean_draft_len"]
    metrics = {
        # The submitted candidate's own exactness leg.
        "e135_ship_leg_mtp_s_per_tok": m["mtp_seconds_per_token"],
        "e135_ship_leg_serial_s_per_tok": m["serial_seconds_per_token"],
        "e135_ship_leg_local_ratio": m["mtp_decode_speedup"],
        "e135_ship_leg_rounds": int(meta["trace_rounds"]),
        "e135_ship_leg_effective_mean_draft_len": edl,
        "e135_ship_leg_accepted_draft_rate": m["accepted_draft_rate"],
        "e135_ship_leg_decode_tokens": m["decode_tokens"],
        "e135_exact_token_divergences": m["residual_divergence_count"],
        "e135_all_tokens_matched": bool(m["all_tokens_matched"]),
        "e135_public_drift_tripwire_passed": bool(
            m["public_drift_tripwire_passed"]),
        # Rule 114. Distance to BOTH arms, so the witness can be seen to fail.
        "e135_arm_distance_to_ship": abs(edl - SHIP_EDL),
        "e135_arm_distance_to_pb6": abs(edl - PB6_EDL),
        "e135_arm_is_ship": abs(edl - SHIP_EDL) < abs(edl - PB6_EDL),
        # Same-host drift against the F34 shipping arm.
        "e135_ship_leg_vs_f34_mtp_pct":
            (F34_SHIP_MTP - m["mtp_seconds_per_token"]) / F34_SHIP_MTP * 100.0,
        "e135_ship_leg_vs_f34_serial_pct":
            (F34_SHIP_SERIAL - m["serial_seconds_per_token"])
            / F34_SHIP_SERIAL * 100.0,
        "e135_ship_leg_rounds_match_f34":
            int(meta["trace_rounds"]) == F34_SHIP_ROUNDS,
        # Thermal record, per the standing local-measurement rules.
        "e135_ship_leg_entry_temp_c": float(meta["gpu_temp_entry_c"]),
        "e135_ship_leg_exit_temp_c": float(meta["gpu_temp_exit_c"]),
        # The retirement this submission carries. The local reading is logged
        # as a measurement OF THE INSTRUMENT, not of the mechanism (Rule 79,
        # now a hard gate), so it is named to prevent reuse as a price.
        "e135_pb6_ranked_confounded_pair_pct": -2.3800,
        "e135_pb6_naive_linear_isolation_pct": -0.6161,
        "e135_pb6_under_tight_pct_INSTRUMENT_ONLY": 2.2987,
        "e135_ranked_width8_mass_weighted": 0.5390,
        "e135_onepass678_local_pct": -12.0004,
    }

    config = {
        "session_label": args.label,
        "harness": "local",
        "experiment": "e135-ship-arm-submission-exactness",
        "submission_id": args.submission,
        "depth_price_arm": "ship",
        "candidate_arms": "tight grid + onePass67 + p15 probe + width2 route",
        "base_sha": meta["base_sha"],
        "worker_sha256": meta["worker_sha256"],
        "post_run_worker_sha256": meta["post_run_worker_sha256"],
        "cli_sha256": meta["cli_sha256"],
        "metallib_source_fingerprint": meta["metallib_source_fingerprint"],
        "head_provenance_sha256": m["head_provenance_sha256"],
        "uses_pinned_mtp_head": m["uses_pinned_mtp_head"],
        "dirty_candidate_paths": meta["dirty_candidate_paths"],
        # Verbatim from the leg, never asserted from a spec.
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
        "official_or_ranked_score": meta["official_or_ranked_score"],
        "not_rankable_reason": m["not_rankable_reason"],
        "host": meta["host"],
        "chip": meta["chip"],
        "memory_bytes": meta["memory_bytes"],
        "started": meta["started"],
        "finished": meta["finished"],
    }

    if args.resume:
        ranked_metrics, ranked_config = receipt_record(
            args.board, args.submission)
        metrics.update(ranked_metrics)
        config.update(ranked_config)

    if args.dry:
        print(json.dumps({"config": config, "metrics": metrics},
                         indent=2, sort_keys=True))
        return

    import wandb

    entity, project = PROJECT.split("/")
    if args.resume:
        run = wandb.init(entity=entity, project=project, id=args.resume,
                         resume="must")
        run.config.update(config, allow_val_change=True)
    else:
        run = wandb.init(entity=entity, project=project, name=args.label,
                         job_type="local-exactness-leg", config=config)
    run.log(metrics)
    print(f"run {run.id} {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
