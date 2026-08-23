#!/usr/bin/env python3
"""Publish the E135 shipping-arm submission record to W&B.

`e135_wandb_log.py` fits multi-leg ABBA contrasts. This leg is not a contrast:
it is the single gated exactness leg that certifies the archive actually
submitted, plus the ranked evidence that retired the `pb6` depth-price arm.
Logging it through the contrast estimator would invent an effect that was
never measured, so it gets its own run.

  research/e135_wandb_submission.py --leg research/out/e135shipexact \
      --submission 0cf1637e-b722-4544-9806-94927d28e558 [--dry]
"""

from __future__ import annotations

import argparse
import json
import os

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", default="research/out/e135shipexact")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--label", default="e135-ship-submission")
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

    if args.dry:
        print(json.dumps({"config": config, "metrics": metrics},
                         indent=2, sort_keys=True))
        return

    import wandb

    entity, project = PROJECT.split("/")
    run = wandb.init(entity=entity, project=project, name=args.label,
                     job_type="local-exactness-leg", config=config)
    run.log(metrics)
    print(f"run {run.id} {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
