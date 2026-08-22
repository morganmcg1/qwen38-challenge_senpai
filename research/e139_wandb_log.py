#!/usr/bin/env python3
"""Publish an E139 result to W&B.

    usage: research/e139_wandb_log.py --rung channel|ladder [--dry]

NEITHER RUNG IS A TIMED MEASUREMENT. The acceptance channel runs with the
per-round phase trace on, which writes a line per round, so every leg records
`timing_valid=false`. The ladder is an offline corpus replay. Both therefore
log `cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false, and neither may be quoted as a
score.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e139-zero-noise-acceptance-channel"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"

LEG_COLUMNS = [
    "dir", "arm", "prompt_id", "rep", "tokens", "offered_depth",
    "round_count", "accepted_draft_total", "rejected_draft_total",
    "effective_mean_draft_len", "effective_max_draft_len",
    "accepted_draft_rate", "declared_rows_total",
    "reference_checked_row_total", "decode_token_count",
    "emitted_token_total", "parity_all_ok", "non_drafting_round_count",
    "all_tokens_matched", "residual_divergence_count",
    "witness_sel_env", "witness_probes", "expected_probes",
    "witness_fp32_gate", "witness_fp32_rerank_drafts",
    "gpu_temp_entry_c", "gpu_temp_exit_c", "started_utc", "finished_utc",
    "base_sha", "worker_sha256", "worker_sha256_after_leg", "cli_sha256",
    "golden_sha256", "prompt_sha256", "head_manifest_tree_sha256",
    "head_provenance_sha256", "metallib_source_fingerprint",
    "dirty_candidate_paths", "host", "chip", "memory_gib",
    "timing_valid", "cool_gate_passed_real_gate", "gate_qualified_for_timing",
    "official_or_ranked_score", "exit",
]

PRICED_COLUMNS = [
    "arm", "prompt_id", "round_count_ship", "round_count_arm", "delta_rounds",
    "accept_rate_ship", "accept_rate_arm", "accept_delta_pp",
    "declared_rows_ship", "declared_rows_arm", "row_cost_pct",
    "quantisation_floor_pct", "acceptance_cost_pct",
    "gross_byte_pct_local", "gross_byte_pct_ranked", "net_ranked_pct",
    "probes", "rerank_drafts",
]

LADDER_COLUMNS = [
    "family", "arm", "p", "anchor_p", "probes", "coarse_rows", "recall_wg",
    "probe_hit_wg", "survivor_hit_wg", "m_absolute_wg", "net_miss_wg",
    "acc_loss_wg", "d_acc_loss_pp", "d_gross_pct", "d_net_pct",
    "bytes_per_row", "n_gating", "passes_t0", "passes_t0b",
]

COST_COLUMNS = ["cost_model", "p", "probes", "coarse_rows", "stage_bytes",
                "removed_vs_anchor_bytes", "gross_pct_vs_anchor"]

RUNGS = {
    "channel": {
        "run_name": "e139-zero-noise-acceptance-channel",
        "job_type": "acceptance-channel",
        "file": "research/e139-acceptance-channel.json",
        "question":
            "what acceptance cost do the two held riders actually carry, "
            "measured on the zero-variance decode channel rather than "
            "predicted by the offline corpus screen",
        "command":
            "research/e139_session.sh ship benchfixture natural_history && "
            "research/e139_session.sh p010 benchfixture natural_history && "
            "research/e139_session.sh p015 benchfixture natural_history && "
            "research/e139_session.sh fp32 benchfixture natural_history && "
            "E139_REP=offer4 E139_DEPTH=4 research/e139_session.sh ship "
            "benchfixture natural_history && "
            "python3 research/e139_analyse.py "
            "--json research/e139-acceptance-channel.json",
    },
    "ladder": {
        "run_name": "e139-probe-fraction-ladder-below-0.10",
        "job_type": "corpus-replay",
        "file": "research/e139-probe-ladder-priced.json",
        "question":
            "where does recall for the shipped derived-cluster readout "
            "finally break as the probe fraction falls below 0.10, and what "
            "is the unscaled net worth of the argmax",
        "command":
            "python3 research/e133_screen.py screen --families exact0 "
            "--widths 4096 --stage-a sketch --probes "
            "0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,"
            "0.125,0.15,0.20,0.25 "
            "--out research/e139-probe-ladder-screen.json && "
            "python3 research/e136_probe_grid.py "
            "--screen research/e139-probe-ladder-screen.json "
            "--json research/e139-probe-ladder-priced.json --self-check",
    },
}

# E139 F2 / FINDING 192. Two independent ranked receipts for the same
# one-constant change, read off the CANDIDATE leg rather than the raw
# serial/candidate ratio.
RIVAL_RANKED_PCT_AT_P015 = 0.3311
RIVAL_RANKED_2SIGMA_AT_P015 = 0.0948
BYTE_MODEL_PCT_AT_P015 = 0.3403


def table(columns, rows):
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def channel_summary(payload: dict) -> tuple[dict, dict]:
    per_arm = {a["arm"]: a for a in payload["per_arm"]}
    det = payload["determinism"]
    positive_ok = bool(det["positive"]) and all(
        g["identical"] for g in det["positive"])
    negative_ok = any(g["distinguished"] for g in det["negative"])
    instrument_ok = positive_ok and negative_ok and not payload["failures"]

    riders = [a for a in ("fp32", "p010", "p015") if a in per_arm]
    composed = sum(max(0.0, per_arm[a]["net_ranked_pct_worst"])
                   for a in riders if a in ("fp32",))
    probe_best = max(
        ((per_arm[a]["net_ranked_pct_worst"], a) for a in ("p010", "p015")
         if a in per_arm), default=(0.0, None))
    composed += max(0.0, probe_best[0])

    summary = {
        # Primary. Sum of each positive rider's gross byte value minus its
        # MEASURED acceptance cost. Only one probe value can ship, so the
        # ladder contributes its best arm, not the sum of its arms.
        "e139_composed_rider_ranked_pct": composed,
        "e139_probe_argmax_p": ({"p010": 0.10, "p015": 0.15}.get(probe_best[1])
                                if probe_best[1] else None),
        "e139_fp32_rider_acceptance_delta_pp":
            (min(per_arm["fp32"]["accept_delta_pp"])
             if "fp32" in per_arm else None),
        "e139_instrument_positive_polarity_ok": positive_ok,
        "e139_instrument_negative_polarity_ok": negative_ok,
        "e139_instrument_ok": instrument_ok,
        "e139_leg_failures": len(payload["failures"]),
        "e139_local_to_ranked_haircut": payload["local_to_ranked_haircut"],
        "e139_acceptance_transfer": payload["acceptance_transfer"],
        # The channel is noiseless but quantised at one round, so a null is a
        # bound, not a proof of a zero population cost.
        "e139_channel_resolution_pct": min(
            (a["tightest_bound_pct"] for a in payload["per_arm"]),
            default=None),
        "harness": "ranked",
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }
    for arm, a in per_arm.items():
        summary[f"e139_{arm}_net_ranked_pct_worst"] = a["net_ranked_pct_worst"]
        summary[f"e139_{arm}_acceptance_null"] = a["acceptance_null"]
        summary[f"e139_{arm}_delta_rounds"] = max(
            abs(d) for d in a["delta_rounds"])

    rival = payload["rival_reconstruction"]
    if rival.get("available"):
        summary.update({
            "e139_rival_ranked_pct": rival["rival_ranked_pct"],
            "e139_predicted_net_pct_at_p015": rival["predicted_net_pct"],
            "e139_measured_over_model": rival["measured_over_model"],
            "e139_agrees_within_rival_2sigma":
                rival["agrees_within_rival_2sigma"],
        })

    tables = {
        "legs": table(LEG_COLUMNS, payload["legs"]),
        "priced": table(PRICED_COLUMNS, payload["priced"]["rows"]),
    }
    return summary, tables


def ladder_summary(payload: dict) -> tuple[dict, dict]:
    ladders = payload.get("ladders", {})
    rows = [r for family in ladders.values() for r in family]
    argmax = payload.get("argmax", {})
    shipped = argmax.get("exact") or argmax.get("shipped") or {}

    broke = [r for r in rows if r.get("recall_wg", 1.0) < 1.0]
    knee = min((r["p"] for r in broke), default=None)
    summary = {
        "e139_ladder_argmax_p": shipped.get("p"),
        "e139_ladder_argmax_d_net_pct": shipped.get("d_net_pct"),
        # The lowest sampled p at which worst-domain recall is still exactly
        # 1.0. If this equals the bottom of the grid, the argmax is again set
        # by where the sampling stopped.
        "e139_ladder_lowest_p_with_full_recall":
            min((r["p"] for r in rows if r.get("recall_wg") == 1.0),
                default=None),
        "e139_ladder_recall_knee_p": knee,
        "e139_ladder_recall_broke": bool(broke),
        "e139_ladder_min_p_sampled": min((r["p"] for r in rows), default=None),
        "e139_ladder_n_gating": rows[0]["n_gating"] if rows else None,
        "e139_byte_model_pct_at_p015": BYTE_MODEL_PCT_AT_P015,
        "e139_rival_ranked_pct_at_p015": RIVAL_RANKED_PCT_AT_P015,
        "e139_byte_model_measured_over_model":
            RIVAL_RANKED_PCT_AT_P015 / BYTE_MODEL_PCT_AT_P015,
        "harness": "ranked",
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }
    tables = {
        "ladder": table(LADDER_COLUMNS, rows),
        "cost_model": table(COST_COLUMNS,
                            [r for m in payload.get("cost_models", {}).values()
                             for r in m]),
    }
    return summary, tables


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    spec = RUNGS[args.rung]
    payload = json.loads(pathlib.Path(spec["file"]).read_text())
    if args.rung == "channel":
        summary, tables = channel_summary(payload)
    else:
        summary, tables = ladder_summary(payload)

    config = {
        "experiment": "e139",
        "rung": args.rung,
        "question": spec["question"],
        "reproduce": spec["command"],
        "host": HOST,
        "source_file": spec["file"],
    }
    if args.dry:
        print(json.dumps({"config": config, "summary": summary}, indent=2,
                         default=str))
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=spec["run_name"], job_type=spec["job_type"],
                     config=config)
    run.log({k: v for k, v in tables.items()})
    run.summary.update(summary)
    print(f"run_id {run.id}")
    print(f"url    {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
