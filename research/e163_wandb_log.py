#!/usr/bin/env python3
"""E163: publish the pinned width-plan session and every gate behind it.

    usage: python3 research/e163_wandb_log.py --run-name e163-r1-pinned-w5

Everything here is `harness=local`. Not one number is an official or ranked
score, and none may be compared with a receipt.

THE QUESTION. `CAMPAIGN LAW 353` says group count is close to free because
concurrent groups reuse the same weight lines, while accumulator width `NA`
drives registers and occupancy. The shipped QMV width plan picks the smallest
group count. The law says it should pick the smallest legal `NA`. Width 5 is
the law's strongest cell: `NA 5 -> 3` is the largest register drop in the whole
table, `102 -> 90` on the ranked chip, and it stays at `G = 2`, inside the only
group count `FINDING 279` ever measured.

FIVE RECORD GROUPS, ONE RUN.

  identity    base, candidate, worker digest, host, chip, head provenance,
              token window, pin, plans, harness.
  legs        one row per timed leg in schedule order, with entry and exit GPU
              temperature, `R` in ms per round, and the accept ledger.
  census      the live g17s and g16s register and spill count of every template
              the arms instantiate. Zero GPU seconds.
  exactness   every cell of the hexfloat row gate and its positive control.
  contrasts   the arm, the width boundary `R(W+1) - R(W)`, the measured noise
              channel, and the routed-matvec share of a round when it exists.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "research" / "e163-artifacts"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# FINDING 352, harness=ranked. Published points per 1 % cut in the ranked
# per-verified-row term, and per 1 % cut in the per-round intercept.
RANKED_PUBLISHED_PER_ROW_PCT = 0.5892
RANKED_PUBLISHED_PER_INTERCEPT_PCT = 0.3104
# Gap from our anchor 5a9f130a to the frontier ec24d591.
CROWN_GAP_PCT = 0.5735
# F1 dilution, share of natural-schedule rounds at each verify width.
NATURAL_WIDTH_SHARE = {2: 0.013, 4: 0.051, 5: 0.064, 6: 0.064, 7: 0.038, 8: 0.769}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def load(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def flat_table(rows: list[dict], columns: list[str]) -> wandb.Table:
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[
            json.dumps(row.get(c)) if isinstance(row.get(c), (dict, list)) else row.get(c)
            for c in columns
        ])
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--label", default="w5")
    args = ap.parse_args()

    session = load(f"e163_pinned_{args.label}.json")
    census = load("e163_na_register_census.json")
    share = load("e163_qmv_share.json")
    exactness = {
        arm: load(f"exactness-{arm}.json") for arm in ("shipped", "minna5", "minna456")
    }
    if session is None:
        raise SystemExit(f"e163_wandb_log: no session artifact for label {args.label}")

    identity = session["identity"]
    config = {
        "experiment": "e163",
        "assignment_id": "e163-width6-one-pass-single-entry",
        "revision_id": "r1",
        "student": "qwen-alphonse",
        "harness": "local",
        "official_score_produced": False,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "candidate_sha": git("rev-parse", "HEAD"),
        "base_sha": identity["base_sha"],
        "session_commit": identity["session_commit"],
        "worker_sha256": identity["worker_sha256"],
        "host": identity["host"],
        "chip": identity["chip"],
        "metallib_source_fingerprint": identity["metallib_source_fingerprint"],
        "head_provenance_sha256": identity["head_provenance_sha256"],
        "decode_tokens": identity["decode_tokens"],
        "local_mode": identity["local_mode"],
        "sandbox": identity["sandbox"],
        "mechanism": "smallest legal accumulator width NA in the routed QMV width plan",
        "candidate_files": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"],
        "selector": "MLX_E163_IPG_PLAN, read once at process start, default shipped",
        "pin": "MLX_E159_FIXED_DRAFT_DEPTH, a physics instrument and not a schedule",
        "pin_bypasses_cost_model_and_width_cap": True,
        "plans": {
            "shipped": {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3},
            "minna5": {2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3},
            "minna456": {2: 2, 3: 3, 4: 2, 5: 3, 6: 2, 7: 4, 8: 4, 9: 3},
        },
        "width_8_owner": "thorfinn, untouched by every arm here",
        "order": session["order"],
        "cool_gate": "real 40C gate on every timed leg",
        "trace_on_every_timed_leg": True,
        "min_useful_percent": session["min_useful_percent"],
        "predeclared_noise_floor_percent": session["noise_channel"][
            "predeclared_floor_percent"
        ],
        "ranked_published_pct_per_1pct_row_term": RANKED_PUBLISHED_PER_ROW_PCT,
        "ranked_published_pct_per_1pct_intercept": RANKED_PUBLISHED_PER_INTERCEPT_PCT,
        "crown_gap_pct": CROWN_GAP_PCT,
        "natural_width_share": NATURAL_WIDTH_SHARE,
        # The local fixture over-samples the cell these arms touch.
        "local_p_width_ge_6": 0.8718,
        "ranked_p_width_ge_6": 0.5861,
    }

    run = wandb.init(
        project=PROJECT, entity=ENTITY, name=args.run_name,
        job_type="local-pinned-width-abba", config=config)

    leg_columns = [
        "position", "arm_key", "plan", "pinned_depth", "verify_width",
        "mtp_seconds_per_token", "serial_seconds_per_token", "mtp_decode_speedup",
        "R_ms_from_leg", "R_ms_from_trace", "seed_prefill_seconds", "rounds",
        "decode_tokens", "effective_mean_draft_len", "accepted_draft_rate",
        "all_tokens_matched", "residual_divergence_count", "reference_checked_rows",
        "trace_width_histogram", "trace_pinned_width_share", "trace_round_us_mean",
        "trace_round_us_sd", "trace_eval_wall_us_mean", "trace_verify_build_us_mean",
        "trace_draft_build_us_mean", "gpu_temp_entry_c", "gpu_temp_exit_c",
        "cool_gate_passed_real_gate", "gate_qualified_for_timing", "worker_sha256",
        "post_run_worker_sha256", "dirty_candidate_paths", "tag",
    ]
    run.log({"legs": flat_table(session["legs"], leg_columns)})

    if census:
        rows = []
        for variant, cells in census["cells"].items():
            for key, cell in cells.items():
                rows.append({
                    "variant": variant,
                    "cell": key,
                    "M": cell["M"],
                    "IPG": cell["IPG"],
                    "NA": cell["NA"],
                    "role": cell["role"],
                    "active_groups": cell["active_groups"],
                    "tail_lanes": cell["tail_lanes"],
                    "g16s_registers": cell["g16s"]["registers"],
                    "g16s_spill_bytes": cell["g16s"]["spill_bytes"],
                    "g17s_registers": cell["g17s"]["registers"],
                    "g17s_spill_bytes": cell["g17s"]["spill_bytes"],
                    "g17s_text_sha8": cell["g17s"]["text_sha8"],
                })
        run.log({"register_census": flat_table(rows, list(rows[0].keys()))})

    exact_rows = []
    for arm, payload in exactness.items():
        if payload is None:
            continue
        for cell in payload["cells"]:
            exact_rows.append({
                "arm": arm,
                "shape": cell["shape"],
                "m": cell["m"],
                "k": cell["k"],
                "n": cell["n"],
                "inputs_per_group": cell["inputs_per_group"],
                "active_groups": cell["active_groups"],
                "tail_lanes": cell["tail_lanes"],
                "elements": cell["elements"],
                "matches_incumbent_bitwise": cell["matches_incumbent_bitwise"],
                "mismatched_elements": cell["mismatched_elements"],
                "max_abs_delta_vs_incumbent": cell["max_abs_delta_vs_incumbent"],
                "candidate_digest": cell["candidate_digest"],
                "incumbent_digest": cell["incumbent_digest"],
            })
    if exact_rows:
        run.log({"exactness": flat_table(exact_rows, list(exact_rows[0].keys()))})

    if share:
        rows = []
        for width, entry in sorted(share["per_width"].items()):
            row = {"verify_width": int(width)}
            row.update({k: v for k, v in entry.items() if not isinstance(v, (dict, list))})
            rows.append(row)
        columns = sorted({k for row in rows for k in row})
        run.log({"routed_matvec_share": flat_table(rows, columns)})

    summary: dict[str, object] = {
        "leg_count": len(session["legs"]),
        "noise_channel_sd_percent": session["noise_channel"]["sd_percent"],
        "noise_channel_half_range_percent": session["noise_channel"][
            "half_range_percent"
        ],
        "predeclared_noise_floor_percent": session["noise_channel"][
            "predeclared_floor_percent"
        ],
        "problem_count": len(session["problems"]),
        "problems": json.dumps(session["problems"]),
        "exactness_all_bitwise": all(
            cell["matches_incumbent_bitwise"] for cell in exact_rows
        )
        if exact_rows
        else None,
        "exactness_cells": len(exact_rows),
    }
    for item in session["contrasts"]:
        key = f"{item['contrast']}_{item['metric']}"
        summary[f"{key}_gain_pct"] = item["gain_percent"]
        summary[f"{key}_half_range_pct"] = item["half_range_percent"]
        summary[f"{key}_gain_minus_half_range_pct"] = item[
            "gain_minus_half_range_percent"
        ]
    arm = next(
        (
            c
            for c in session["contrasts"]
            if c["contrast"] == "arm" and c["metric"] == "mtp_seconds_per_token"
        ),
        None,
    )
    if arm:
        width = session["legs"][0]["verify_width"]
        dilution = NATURAL_WIDTH_SHARE.get(width, 0.0)
        summary["pinned_verify_width"] = width
        summary["natural_schedule_dilution"] = dilution
        # Projection, harness=local -> harness=ranked. Not a measurement.
        summary["projected_natural_schedule_pct"] = arm["gain_percent"] * dilution
        summary["clears_min_useful"] = (
            arm["gain_minus_half_range_percent"] > session["min_useful_percent"]
        )
    if session["boundary"]:
        for key, value in session["boundary"].items():
            if isinstance(value, (int, float)) or value is None:
                summary[f"boundary_{key}"] = value
    if share:
        for width, entry in share["per_width"].items():
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    summary[f"matvec_w{width}_{key}"] = value

    run.summary.update(summary)
    print(f"wandb run {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
