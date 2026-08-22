#!/usr/bin/env python3
"""Publish an E139 result to W&B.

    usage: research/e139_wandb_log.py --rung channel|ladder [--dry]

NEITHER RUNG IS A TIMED MEASUREMENT. The acceptance channel runs with the
per-round phase trace on, which writes a line per round, so every leg records
`timing_valid=false`. The ladder is an offline corpus replay. Both therefore
log `cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false, and neither may be quoted as a
score.

THE LADDER READS THE LEAF TERM, NOT `recall`. `recall` in e133_screen.py is
survivor retention CONDITIONAL on the probed set, so it is identically 1.0
whenever the survivor width covers every probed row. It stays 1.0 at p=0.005,
where the arm never reaches 8.6 % of the true argmax rows. Every recall-like
quantity logged here is `probe_hit_rate`, and the defective field is logged
beside it under a name that says so.
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

# e133_screen.py:160 asserts this value, so a change there fails loudly rather
# than silently repricing the ladder. e133_screen.py:146-147 for the gates.
MISS_TO_SCORE_PCT = 203.0
T0_NET_MISS = 3.0e-3
T0B_RECALL = 0.997
# E139 F4 / FINDING 196 and CAMPAIGN RULE 116. The published score is the mean
# of the 4th and 5th sorted per-prompt raw ratios. A mechanism whose relative
# effect is uniform across prompts does not change the sorted order, so its
# median gain is the local gross gain times one measured constant.
LOCAL_GROSS_TO_MEDIAN_GAIN = 0.95
RIVAL_MEDIAN_PCT_AT_P015 = 0.3235
RIVAL_MEDIAN_2SIGMA_AT_P015 = 0.0948 * (0.1263 / 0.1050)
BYTE_MODEL_PCT_AT_P015 = 0.3403

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
    "gross_byte_pct_local", "gross_byte_pct_median", "net_median_pct",
    "probes", "rerank_drafts",
]

LADDER_COLUMNS = [
    "arm", "p", "bytes_per_row", "n", "n_gating",
    "gross_pct_local", "gross_pct_median",
    "probe_hit_wg", "survivor_hit_wg", "recall_wg_defective",
    "m_absolute_wg", "net_miss_wg", "acc_loss_wg", "acc_loss_pooled_wg",
    "substitutions_live_gating",
    "net_pct_gating", "net_pct_gating_median",
    "net_pct_absolute", "net_pct_pooled", "net_pct_raw_miss",
    "passes_t0", "passes_t0b", "passes_t0b_leaf",
]

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
            "research/e139_session.sh p002 benchfixture natural_history && "
            "E139_REP=offer4 E139_DEPTH=4 research/e139_session.sh ship "
            "benchfixture natural_history && "
            "E139_REP=offer2 E139_DEPTH=2 research/e139_session.sh ship "
            "natural_history && "
            "python3 research/e139_analyse.py "
            "--json research/e139-acceptance-channel.json",
    },
    "ladder": {
        "run_name": "e139-probe-fraction-ladder",
        "job_type": "corpus-replay",
        "file": "research/e139-probe-ladder-screen.json",
        "question":
            "what does the probe fraction cost in leaf recall and in "
            "realised acceptance at every rung from 0.25 down to 0.005, and "
            "which rung maximises the published median gain",
        "command":
            "python3 research/e133_screen.py screen --families exact0 "
            "--widths 24584 --stage-a sketch --probes "
            "0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.075,"
            "0.08,0.09,0.10,0.125,0.15,0.175,0.20,0.25 "
            "--out research/e139-probe-ladder-screen.json",
    },
}


def table(columns, rows):
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def probe_of(arm: str) -> float | None:
    """`p015` -> 0.15, `fp32p002` -> 0.02, `ship` and `fp32` -> None."""
    for prefix in ("fp32p0", "p0"):
        if arm.startswith(prefix):
            return float(f"0.{arm[len(prefix):]}")
    return None


def channel_summary(payload: dict) -> tuple[dict, dict]:
    per_arm = {a["arm"]: a for a in payload["per_arm"]}
    det = payload["determinism"]
    positive_ok = bool(det["positive"]) and all(
        g["identical"] for g in det["positive"])
    negative_ok = any(g["distinguished"] for g in det["negative"])
    instrument_ok = positive_ok and negative_ok and not payload["failures"]

    # Only one probe fraction can ship, so the ladder contributes its best arm
    # rather than the sum of its arms. `fp32` is an independent knob and adds.
    probe_arms = [a for a in per_arm if probe_of(a) is not None]
    probe_best = max(((per_arm[a]["net_median_pct_worst"], a)
                      for a in probe_arms), default=(0.0, None))
    composed = max(0.0, probe_best[0])
    if "fp32" in per_arm:
        composed += max(0.0, per_arm["fp32"]["net_median_pct_worst"])

    summary = {
        # Primary. Each retained rider's gross byte value minus its MEASURED
        # acceptance cost, converted to published median by one constant.
        "e139_composed_rider_median_pct": composed,
        "e139_probe_argmax_p": (probe_of(probe_best[1])
                                if probe_best[1] else None),
        "e139_fp32_rider_acceptance_delta_pp":
            (min(per_arm["fp32"]["accept_delta_pp"])
             if "fp32" in per_arm else None),
        "e139_instrument_positive_polarity_ok": positive_ok,
        "e139_instrument_negative_polarity_ok": negative_ok,
        "e139_instrument_ok": instrument_ok,
        "e139_leg_failures": len(payload["failures"]),
        "e139_local_gross_to_median_gain":
            payload["local_gross_to_median_gain"],
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
        summary[f"e139_{arm}_net_median_pct_worst"] = a["net_median_pct_worst"]
        summary[f"e139_{arm}_acceptance_null"] = a["acceptance_null"]
        summary[f"e139_{arm}_delta_rounds"] = max(
            abs(d) for d in a["delta_rounds"])

    rival = payload["rival_reconstruction"]
    if rival.get("available"):
        summary.update({
            "e139_rival_median_pct": rival["rival_median_pct"],
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


def ladder_rows(payload: dict) -> list[dict]:
    cells = sorted(payload["cells"], key=lambda c: -c["probe_fraction"])
    g0 = max(cells, key=lambda c: c["probe_fraction"])["pct_head_share_7"]
    rows = []
    for c in cells:
        gross = c["pct_head_share_7"] - g0
        # Clamp the realised loss at zero. It is quantised at roughly 2e-4 and
        # changes sign between neighbouring rungs, so a negative value is one
        # substitution event of noise, not a measured improvement.
        loss = max(0.0, c["acceptance_loss_worst_gating"])
        net_gating = gross - MISS_TO_SCORE_PCT * loss
        rows.append({
            "arm": c["arm"],
            "p": c["probe_fraction"],
            "bytes_per_row": c["bytes_per_row"],
            "n": c["n"],
            "n_gating": c["n_gating"],
            "gross_pct_local": gross,
            "gross_pct_median": gross * LOCAL_GROSS_TO_MEDIAN_GAIN,
            "probe_hit_wg": c["probe_hit_rate_worst_gating"],
            "survivor_hit_wg": c["survivor_hit_rate_worst_gating"],
            "recall_wg_defective": c["recall_worst_gating"],
            "m_absolute_wg": c["m_absolute_worst_gating"],
            "net_miss_wg": c["net_miss_worst_gating"],
            "acc_loss_wg": c["acceptance_loss_worst_gating"],
            "acc_loss_pooled_wg": c["acceptance_loss_pooled_worst_gating"],
            "substitutions_live_gating": c["substitutions_live_gating"],
            "net_pct_gating": net_gating,
            "net_pct_gating_median": net_gating * LOCAL_GROSS_TO_MEDIAN_GAIN,
            "net_pct_absolute": c["predicted_pct_absolute"] - g0,
            "net_pct_pooled": c["predicted_pct_pooled"] - g0,
            "net_pct_raw_miss": c["predicted_pct_raw_miss"] - g0,
            "passes_t0": c["passes_t0"],
            "passes_t0b": c["passes_t0b"],
            "passes_t0b_leaf": c["passes_t0b_leaf"],
        })
    return rows


def ladder_summary(payload: dict) -> tuple[dict, dict]:
    rows = ladder_rows(payload)
    clearing = [r for r in rows if r["passes_t0"] and r["passes_t0b_leaf"]]
    best = max(clearing, key=lambda r: r["net_pct_gating_median"],
               default=None)
    best_any = max(rows, key=lambda r: r["net_pct_gating_median"])
    floor = min((r["p"] for r in clearing), default=None)
    at015 = next((r for r in rows if abs(r["p"] - 0.15) < 1e-12), None)

    def transfer(field: str):
        if not at015 or not at015[field]:
            return None
        return RIVAL_MEDIAN_PCT_AT_P015 / at015[field]

    summary = {
        "e139_ladder_argmax_p": best["p"] if best else None,
        "e139_ladder_argmax_median_pct":
            best["net_pct_gating_median"] if best else None,
        "e139_ladder_argmax_p_ignoring_gates": best_any["p"],
        "e139_ladder_argmax_median_pct_ignoring_gates":
            best_any["net_pct_gating_median"],
        "e139_ladder_lowest_p_clearing_gates": floor,
        # The shipped T0b reads `recall`, which cannot fall on this ladder, so
        # it is a gate that cannot fail. The leaf reading of the same 0.997
        # threshold agrees with T0 at every rung here.
        "e139_ladder_t0b_shipped_is_vacuous":
            all(r["passes_t0b"] for r in rows),
        "e139_ladder_t0_and_t0b_leaf_agree":
            all(r["passes_t0"] == r["passes_t0b_leaf"] for r in rows),
        "e139_ladder_min_probe_hit_wg": min(r["probe_hit_wg"] for r in rows),
        "e139_ladder_min_recall_wg_defective":
            min(r["recall_wg_defective"] for r in rows),
        "e139_ladder_min_p_sampled": min(r["p"] for r in rows),
        "e139_ladder_n_gating": rows[0]["n_gating"],
        "e139_miss_to_score_pct": MISS_TO_SCORE_PCT,
        "e139_t0_net_miss": T0_NET_MISS,
        "e139_t0b_recall": T0B_RECALL,
        "e139_byte_model_pct_at_p015": BYTE_MODEL_PCT_AT_P015,
        "e139_rival_median_pct_at_p015": RIVAL_MEDIAN_PCT_AT_P015,
        "e139_rival_median_2sigma_at_p015": RIVAL_MEDIAN_2SIGMA_AT_P015,
        "e139_byte_model_measured_over_model":
            RIVAL_MEDIAN_PCT_AT_P015 / BYTE_MODEL_PCT_AT_P015,
        # Which of the screen's four estimators the ranked receipt selects. A
        # candidate-leg byte saving cannot be amplified by the ranked harness,
        # so an implied transfer above 1.0 falsifies that estimator.
        "e139_transfer_implied_by_gating_estimator":
            transfer("net_pct_gating"),
        "e139_transfer_implied_by_pooled_estimator":
            transfer("net_pct_pooled"),
        "e139_transfer_implied_by_absolute_estimator":
            transfer("net_pct_absolute"),
        "harness": "ranked",
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }
    return summary, {"ladder": table(LADDER_COLUMNS, rows)}


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
