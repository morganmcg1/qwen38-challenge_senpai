#!/usr/bin/env python3
"""E174 steps 1 and 2b: publish the dedup census and the host record probe.

Two untimed instruments and one reconciliation go to W&B:

    step 1   dedup census, 2 arms x 512 tokens x 77 differenced rounds
    step 2b  CPU-side host kernel record probe, 3 widths x 2 batches x 9 repeats
    step 2b  reconciliation of the probe against the E174 screen bracket

Neither instrument is a timed leg. `gate_qualified_for_timing=false` is carried
verbatim into the run config and summary so no number here can later be read as
a gated result. `harness=local` throughout; no ranked claim is made or implied.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import wandb

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT.parent, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    census = json.loads((OUT / "e174-dedup-census-d1.json").read_text())
    probe = json.loads((OUT / "e174-record-probe.json").read_text())
    reconcile = json.loads((OUT / "e174-record-reconcile.json").read_text())

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e174-step1-census-step2b-record-probe",
        job_type="instrument",
        tags=["e174", "untimed", "harness-local", "census", "cpu-probe"],
        config={
            "experiment": "E174",
            "steps": ["step1-dedup-census", "step2b-record-probe"],
            "harness": "local",
            "timed_leg": False,
            "gate_qualified_for_timing": False,
            "base_sha": "2b8037c3e5207fa2f5633e61d013a779492b05c2",
            "candidate_sha": git("rev-parse", "HEAD"),
            "host": "ip-10-231-2-22.ec2.internal",
            "chip": "Apple M4 Pro",
            "tokens": 512,
            "screen_run_for_bracket": "7ueick4f",
            "marginal_bracket_us": reconcile[
                "marginal_bracket_us_per_dispatch_plus_record"
            ],
        },
    )

    for arm, a in sorted(census["arms"].items()):
        prefix = f"census/{arm}"
        for key in (
            "rounds_traced",
            "rounds_differenced",
            "counter_summary_agree",
            "selfcheck_ok",
            "hit_per_round_median",
            "fill_per_round_median",
            "uniq_per_round_median",
            "duplicate_per_round_median",
            "duplicate_fraction_of_fills",
            "round_us_median",
        ):
            run.summary[f"{prefix}/{key}"] = a[key]
        run.summary[f"{prefix}/shape_calls_per_round"] = json.dumps(
            a["shape_calls_per_round"], sort_keys=True
        )
        run.summary[f"{prefix}/histogram"] = json.dumps(
            a["duplicates_per_activation_histogram"], sort_keys=True
        )

    run.summary["census/instrument_selfcheck_ok"] = census["instrument_selfcheck_ok"]
    run.summary["census/arm_switch_control_ok"] = census["arm_switch_control_ok"]
    run.summary["census/decision_threshold"] = census[
        "decision_threshold_duplicate_fraction"
    ]

    width_table = wandb.Table(
        columns=["arm", "k", "m", "calls_per_round", "uniq_per_round", "duplicates"]
    )
    for arm, a in sorted(census["arms"].items()):
        for shape, calls in sorted(a["shape_calls_per_round"].items()):
            k, _, m = shape.partition("x")
            uniq = a["shape_uniq_per_round"][shape]
            width_table.add_data(arm, int(k), int(m), calls, uniq, calls - uniq)
    run.log({"census/per_width": width_table})

    probe_table = wandb.Table(
        columns=[
            "cell",
            "k",
            "m",
            "batch",
            "record_us_per_call_median",
            "eval_us_per_call_median",
        ]
    )
    for name, cell in sorted(probe["cells"].items()):
        for batch in ("batch_32", "batch_256"):
            probe_table.add_data(
                name,
                cell["k"],
                cell["m"],
                int(batch.split("_")[1]),
                cell[batch]["record_us_per_call_median"],
                cell[batch]["eval_us_per_call_median"],
            )
        run.summary[f"probe/{name}/batch_scaling_ratio"] = cell["batch_scaling_ratio"]
        run.summary[f"probe/{name}/batch_scaling_control_ok"] = cell[
            "batch_scaling_control_ok"
        ]
    run.log({"probe/per_cell": probe_table})

    for key, value in reconcile.items():
        if key == "producer_side_removal_price":
            for k2, v2 in value.items():
                run.summary[f"price/{k2}"] = (
                    json.dumps(v2) if isinstance(v2, (list, dict)) else v2
                )
        elif isinstance(value, (list, dict)):
            run.summary[f"reconcile/{key}"] = json.dumps(value)
        else:
            run.summary[f"reconcile/{key}"] = value

    print(f"wandb run: {run.url}")
    print(f"wandb run id: {run.id}")
    run.finish()


if __name__ == "__main__":
    main()
