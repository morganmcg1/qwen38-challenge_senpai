"""Log the F24 receipt analysis to W&B and write its artifact.

Covers the three F24 asks: the submitted-archive witness, the dose-response
test against a matched null, and the W1/W2/A1 ISA table priced with the
measured shortfall. Adds the launch-accounting finding that explains the
shortfall.

No GPU, no timing, no model.
"""

import json
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e129-f24-receipt-analysis"
OUT = "research/e129-artifacts/f24-receipt-analysis.json"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True).stdout.strip()


def main():
    payload = {
        "schema_version": 1,
        "harness": "ranked+compile_only",
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "official_or_ranked_score": True,
        "base_sha": sh("git rev-parse HEAD"),
        "receipt": {
            "id": "623e77af",
            "status": "accepted",
            "promotion": "promoted",
            "official_score": 3.52085227003175,
            "promoted_source_ref":
                "60d5b34a5ad62296faa9746dccab71e762e4001c",
            "previous_crown": {"id": "0b8602e1", "score": 3.51925374},
            "reference": {"id": "0c6191b7", "score": 3.51270586},
        },
        "witness_from_submitted_archive": {
            "e120_width_plan":
                "3:3:4,4:4:4,5:5:4,6:6:4,7:7:4,8:4:4,9:3:4",
            "widths_6_and_7": "6:6:4,7:7:4",
            "reading_a_eliminated": True,
            "qwen35_swift_sha256":
                "937f7de026754b5f5796a137d93f420b8efcc1e6ed3683a661e60a5c25a28081",
            "worker_sha256":
                "d2e841875ee9fa62e7e6421fa3ecfba84452c978f008c4585f105bbb221c6a36",
            "width_plan_needle_is_discriminating": False,
            "width_plan_literals_in_binary": 4,
            "route_needle_is_discriminating": True,
            "route_literals_in_binary": 1,
            "runtime_specializations": 7,
            "na6_and_na7_pipelines_created": True,
        },
        "leg_decomposition": {
            "candidate_mean_pct": 0.0235,
            "candidate_sd_pct": 0.2694,
            "candidate_2se_band": [-0.1670, 0.2140],
            "serial_null_mean_pct": -0.0014,
            "serial_null_sd_pct": 0.4173,
            "f83_weighted_candidate_pct": -0.2032,
            "score_delta": 0.00814641,
            "beagle_share_of_score_move": 0.885,
            "beagle_serial_share": 0.652,
        },
        "dose_response": {
            "grouping": "acts=beagle,republic,essays,medicine; "
                        "inert=plutarch,drama,travel; above=botany",
            "candidate_contrast_pp": -0.4506,
            "candidate_exact_perm_p": 0.0143,
            "serial_contrast_pp": 0.4886,
            "serial_exact_perm_p": 0.1357,
            "empirical_null_sd_pp": 0.2219,
            "empirical_null_pairs": 400,
            "candidate_empirical_p": 0.0425,
            "verdict": "real at about the 4 percent level against a "
                       "conservative matched null, from one receipt",
        },
        "three_levers": {
            "issue_change_pct": -46.4,
            "occupancy_change_pct": -15.7,
            "weight_pass_change_pct": -50.0,
            "observed_round_pct": -0.5,
            "shortfall_f151_issue": 21.9,
            "shortfall_rowkeyed_issue": 81.1,
            "shortfall_weight_traffic": 87.4,
            "blend_exponent_issue": 0.0065,
            "blend_exponent_occupancy": -0.0096,
        },
        "launch_accounting": {
            "source": "Qwen35.swift:1959-1961 and 1546-1550",
            "wide_launched_columns_is_independent_of_ipg": True,
            "f83_weighted_launched_wide": 6.9985,
            "f83_weighted_launched_tight": 1.3192,
            "tight_reduction_factor": 5.30,
            "env_switch": "MLX_E120_QMV_GRID",
            "tested_in_ledger": False,
        },
        "isa_table": {
            "arch": "applegpu_g17s",
            "w1": {"reaches_isa": False, "identical_text_digest": True,
                   "air_delta": 3, "predicted_ranked_leg_pct": 0.0},
            "a1": {"reaches_isa": False, "identical_text_digest": True,
                   "air_delta": -3, "predicted_ranked_leg_pct": 0.0},
            "w1_a1": {"reaches_isa": False, "identical_text_digest": True,
                      "air_delta": 0, "predicted_ranked_leg_pct": 0.0},
            "w2": {"reaches_isa": True, "identical_text_digest": False,
                   "air_delta": 1, "channel": "occupancy, not instruction",
                   "weighted_residency_gain_pct": 0.9127,
                   "qmv_scaled_pct": 0.7972},
            "verdict": "none of the three survives; the premise that they "
                       "delete instructions at constant registers is false",
        },
    }

    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     job_type="analysis", name="e129-f24-receipt-analysis",
                     config=payload)
    wandb.log({
        "official_score": 3.52085227003175,
        "candidate_leg_mean_pct": 0.0235,
        "f83_weighted_candidate_pct": -0.2032,
        "dose_response_contrast_pp": -0.4506,
        "dose_response_empirical_p": 0.0425,
        "shortfall_f151": 21.9,
        "shortfall_weight_traffic": 87.4,
        "tight_grid_reduction_factor": 5.30,
    })
    print(f"wandb run {run.id}  {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
