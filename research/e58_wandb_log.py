#!/usr/bin/env python3
"""Log the E58 round dispatch census to W&B, one stage at a time.

The advisor's standing instruction is to log while measuring, never once at
session end, so this script resumes one run id and is called after every stage.

  research/e58_wandb_log.py --stage storm  --storm research/out/e58-storm-*/storm.json
  research/e58_wandb_log.py --stage census --census research/out/TAG/census.jsonl
  research/e58_wandb_log.py --stage tax    --arms research/out/e58-tax-* ...
  research/e58_wandb_log.py --stage projection --census-json research/e58-artifacts/census.json

The run id lives in research/e58-artifacts/wandb-run-id.txt.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import e58_census_report as CENSUS  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
ARTIFACTS = pathlib.Path("research/e58-artifacts")
RUN_ID_FILE = ARTIFACTS / "wandb-run-id.txt"

# harness=ranked, from the receipt reconstruction in ledger item 186(D). Every
# projection below is labelled with the harness it came from.
RANKED = {
    "beagle": {
        "leg_ms": 6233.1,
        "rounds": 107,
        "ms_per_round": 53.33,
        "dilution": 0.91552,
        "mean_draft_len": 4.5327,
    },
    "medicine": {
        "leg_ms": 5820.7,
        "rounds": 99,
        "ms_per_round": 53.48,
        "dilution": 0.90953,
        "mean_draft_len": 4.7677,
    },
}
MEDIAN_PAIR_DILUTION = 0.9125
# RETRACTED by ledger 193(E): this is 2 sd of the SERIAL leg's jitter applied to the
# score, and the median over eight prompts does not average the candidate-leg common
# mode away. The measured single-pair ranked MDE is 2.10 %, 7.4x larger. The value
# below is kept so this module's published arithmetic stays reproducible; import
# research/ranked_noise.py for any NEW ranked pricing.
RANKED_MDE_PERCENT = 0.283
LOCAL_NULL_FLOOR_PERCENT = 0.0629


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def identity() -> dict:
    return {
        "assignment_id": "qwen38-r1-e58-round-dispatch-census-and-buffer-batching",
        "revision_id": "e58-r2",
        "pr_number": 61,
        "base_sha": "0e64f0b0d585274fd50c4cb6af7235ca4a111303",
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host_chip": "Apple M4 Pro",
        "host_physical_memory_gib": 48,
        "host_gpu_architecture": "applegpu_g16s",
        "host_arch_gen": 16,
        "nax_available": False,
        "harness": "local",
        "local_mode": "--local-iterate",
        "token_window": 512,
        "mlx_max_ops_per_buffer_local": 64,
        "mlx_max_mb_per_buffer_local": 128,
        "mlx_max_ops_per_buffer_ranked": 50,
        "mlx_max_mb_per_buffer_ranked": 512,
        "ranked_mde_percent_2sd": RANKED_MDE_PERCENT,
        "local_null_floor_percent": LOCAL_NULL_FLOOR_PERCENT,
        "live_promoted_frontier": 3.24985583421771,
        "our_best_official": 3.23250848263467,
        "base_candidate_seconds_per_token": 0.035845,
        "base_local_speedup": 2.084087,
    }


def resume_run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run_id = None
    if RUN_ID_FILE.exists():
        run_id = RUN_ID_FILE.read_text().strip() or None
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume="allow",
        name="e58-round-dispatch-census",
        job_type="dispatch-census",
        config=identity(),
        tags=["e58", "dispatch-census", "latency-bound", "qwen-alphonse", "local"],
    )
    RUN_ID_FILE.write_text(run.id + "\n")
    return run


def log_storm(run, paths) -> None:
    for path in paths:
        report = json.loads(pathlib.Path(path).read_text())
        tag = pathlib.Path(path).parent.name
        table = wandb.Table(
            columns=[
                "arm",
                "mode",
                "ops_per_buffer",
                "dispatches",
                "command_buffers",
                "mean_ms",
                "min_ms",
                "us_per_dispatch",
            ]
        )
        for point in report["points"]:
            table.add_data(
                tag,
                point["mode"],
                point["opsPerBuffer"],
                point["dispatches"],
                point["commandBuffers"],
                point["meanSeconds"] * 1e3,
                point["minSeconds"] * 1e3,
                point["microsecondsPerDispatch"],
            )
        payload = {
            f"storm/{tag}/table": table,
            f"storm/{tag}/metal_per_dispatch_us": report.get(
                "metalPerDispatchMicroseconds"
            ),
            f"storm/{tag}/metal_per_command_buffer_us": report.get(
                "metalPerBufferMicroseconds"
            ),
            f"storm/{tag}/mlx_max_ops_per_buffer": report["mlxMaxOpsPerBuffer"],
        }
        for point in report["points"]:
            key = f"storm/{tag}/{point['mode']}/ops{point['opsPerBuffer']}"
            payload[f"{key}/us_per_dispatch"] = point["microsecondsPerDispatch"]
            payload[f"{key}/mean_ms"] = point["meanSeconds"] * 1e3
        run.log(payload)


def log_census(run, path) -> dict:
    report = CENSUS.summarize(path)
    tag = pathlib.Path(path).parent.name
    payload = {}
    table = wandb.Table(
        columns=[
            "leg",
            "pid",
            "rounds",
            "family",
            "total_dispatches",
            "per_round",
        ]
    )
    width_table = wandb.Table(
        columns=["leg", "width_M", "rounds", "dispatches_per_round", "family", "per_round"]
    )
    for leg in report["legs"]:
        name = leg["leg"].replace("(", "_").replace(")", "")
        prefix = f"census/{tag}/{name}"
        payload[f"{prefix}/rounds"] = leg["rounds"]
        payload[f"{prefix}/dispatches_in_rounds"] = leg["dispatch_total_in_rounds"]
        payload[f"{prefix}/dispatches_per_round"] = leg["dispatches_per_round_mean"]
        payload[f"{prefix}/command_buffers_in_rounds"] = leg["commits_in_rounds"]
        payload[f"{prefix}/dispatches_per_command_buffer"] = leg["dispatches_per_commit"]
        payload[f"{prefix}/barriers_in_rounds"] = leg["barriers_in_rounds"]
        for family, count in leg["family_totals"].items():
            payload[f"{prefix}/family/{family}/total"] = count
            payload[f"{prefix}/family/{family}/per_round"] = count / max(
                1, leg["rounds"]
            )
            table.add_data(
                leg["leg"],
                leg["pid"],
                leg["rounds"],
                family,
                count,
                count / max(1, leg["rounds"]),
            )
        for phase, bucket in leg["phase_totals"].items():
            payload[f"{prefix}/phase/{phase}/total"] = bucket["dispatches"]
            payload[f"{prefix}/phase/{phase}/per_round"] = bucket[
                "dispatches"
            ] / max(1, leg["rounds"])
        for name_outside, bucket in leg.get("outside_rounds", {}).items():
            payload[f"{prefix}/outside/{name_outside}/dispatches"] = bucket[
                "dispatches"
            ]
        for width, bucket in leg["widths"].items():
            for family, value in bucket["families_per_round"].items():
                width_table.add_data(
                    leg["leg"],
                    int(width),
                    bucket["rounds"],
                    bucket["dispatches_per_round"],
                    family,
                    value,
                )
    payload[f"census/{tag}/family_table"] = table
    payload[f"census/{tag}/width_table"] = width_table
    run.log(payload)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{tag}-census.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def log_tax(run, arm_dirs) -> None:
    rows = []
    for directory in arm_dirs:
        path = pathlib.Path(directory)
        score_path = path / "score.json"
        meta_path = path / "meta.txt"
        if not score_path.exists():
            print(f"e58: {score_path} missing, skipping", file=sys.stderr)
            continue
        score = json.loads(score_path.read_text())
        meta = dict(
            line.split("=", 1)
            for line in meta_path.read_text().splitlines()
            if "=" in line
        )
        metrics = score.get("metrics", {})
        rows.append(
            {
                "arm": path.name,
                "tax": int(meta.get("tax", 0)),
                "tax_mode": meta.get("tax_mode", ""),
                "tokens": int(meta.get("tokens", 0)),
                "candidate_seconds_per_token": metrics.get("mtp_seconds_per_token"),
                "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
                "local_speedup": metrics.get("mtp_decode_speedup"),
                "all_tokens_matched": metrics.get("all_tokens_matched"),
                "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
                "accepted_draft_rate": metrics.get("accepted_draft_rate"),
                "gpu_temp_entry_c": float(meta.get("gpu_temp_entry_c") or "nan"),
                "gpu_temp_exit_c": float(meta.get("gpu_temp_exit_c") or "nan"),
                "cool_gate_passed_real_gate": meta.get("cool_gate") == "1",
                "gate_qualified_for_timing": meta.get("cool_gate") == "1",
                "started": meta.get("started"),
                "finished": meta.get("finished"),
            }
        )
    table = wandb.Table(columns=list(rows[0].keys())) if rows else None
    for row in rows:
        table.add_data(*row.values())
    payload = {"tax/table": table} if table else {}
    for row in rows:
        prefix = f"tax/{row['arm']}"
        payload[f"{prefix}/candidate_seconds_per_token"] = row[
            "candidate_seconds_per_token"
        ]
        payload[f"{prefix}/serial_seconds_per_token"] = row["serial_seconds_per_token"]
        payload[f"{prefix}/local_speedup"] = row["local_speedup"]
        payload[f"{prefix}/tax_per_round"] = row["tax"]
        payload[f"{prefix}/gpu_temp_entry_c"] = row["gpu_temp_entry_c"]
    run.log(payload)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "tax-arms.json").write_text(json.dumps(rows, indent=2, sort_keys=True))


def log_analysis(run, path) -> None:
    report = json.loads(pathlib.Path(path).read_text())
    table = wandb.Table(
        columns=[
            "method",
            "measures",
            "ns_per_dispatch",
            "candidate_ms_per_round",
            "beagle_percent_of_ranked_round",
            "medicine_percent_of_ranked_round",
        ]
    )
    what = {
        "in_situ_pipelined_tax": "one more dispatch in the stream, in situ",
        "census_host_encode_and_submit": "host time inside Metal, real path",
        "storm_serialised_floor": "trivial dispatch, encode+submit+wait",
        "e57_real_dispatch_regression_contaminated": (
            "real composed-SDPA dispatches, includes their arithmetic"
        ),
    }
    payload = {}
    for method, row in report["projection"].items():
        table.add_data(
            method,
            what.get(method, ""),
            row["ns_per_dispatch"],
            row["candidate_ms_per_round"],
            row["beagle_percent_of_ranked_round"],
            row["medicine_percent_of_ranked_round"],
        )
        payload[f"projection/{method}/ns_per_dispatch"] = row["ns_per_dispatch"]
        payload[f"projection/{method}/candidate_ms_per_round"] = row[
            "candidate_ms_per_round"
        ]
        for leg in RANKED:
            payload[f"projection/{method}/{leg}_percent_of_ranked_round"] = row[
                f"{leg}_percent_of_ranked_round"
            ]
    payload["projection/table"] = table
    for label, slope in report["slopes"].items():
        key = label.split()[0]
        for field, value in slope.items():
            payload[f"slope/{key}/{field}"] = value
    payload["projection/median_pair_dilution"] = MEDIAN_PAIR_DILUTION
    payload["projection/ranked_mde_percent_2sd"] = RANKED_MDE_PERCENT
    payload["projection/local_null_floor_percent"] = LOCAL_NULL_FLOOR_PERCENT
    run.log(payload)
    run.summary.update(
        {
            "verdict": "not_useful",
            "rung2_gate_percent": 2.0,
            "rung2_opened": False,
            "best_estimate_percent_of_ranked_round": report["projection"][
                "in_situ_pipelined_tax"
            ]["beagle_percent_of_ranked_round"],
            "pessimistic_on_path_percent_of_ranked_round": report["projection"][
                "census_host_encode_and_submit"
            ]["beagle_percent_of_ranked_round"],
        }
    )


def log_buffer(run, path) -> None:
    """The counterbalanced MLX_MAX_OPS_PER_BUFFER session at a fixed 512 MiB."""
    report = json.loads(pathlib.Path(path).read_text())
    table = wandb.Table(
        columns=[
            "ops_per_buffer_limit",
            "runs",
            "candidate_dispatches_per_buffer",
            "candidate_buffers_per_round",
            "mtp_seconds_per_token",
            "mtp_within_pair_spread_percent",
            "serial_seconds_per_token",
            "gpu_temp_entry_c_mean",
        ]
    )
    payload = {}
    for ops, entry in sorted(report["arms"].items(), key=lambda kv: int(kv[0])):
        entry_mean = sum(entry["gpu_temp_entry_c"]) / len(entry["gpu_temp_entry_c"])
        table.add_data(
            entry["ops_per_buffer_limit"],
            ",".join(entry["runs"]),
            entry["candidate_dispatches_per_buffer"],
            entry["candidate_buffers_per_round"],
            entry["mtp_seconds_per_token"],
            entry["mtp_within_pair_spread_percent"],
            entry["serial_seconds_per_token"],
            entry_mean,
        )
        for field in (
            "mtp_seconds_per_token",
            "mtp_within_pair_spread_percent",
            "serial_seconds_per_token",
            "candidate_dispatches_per_buffer",
            "candidate_buffers_per_round",
        ):
            payload[f"buffer/ops{ops}/{field}"] = entry[field]
    payload["buffer/table"] = table
    for field, value in report["effect"].items():
        if isinstance(value, (int, float, bool)):
            payload[f"buffer/effect/{field}"] = value
    run.log(payload)
    run.summary.update(
        {
            "buffer_sweep_verdict": "regression",
            "buffer_sweep_candidate_delta_percent": report["effect"][
                "candidate_delta_percent"
            ],
            "buffer_sweep_ns_per_removed_buffer": report["effect"][
                "ns_per_removed_buffer"
            ],
            "buffer_sweep_entry_temp_spread_c": report["effect"]["entry_temp_spread_c"],
            "cool_gate_passed_real_gate": report["cool_gate_passed_real_gate"],
            "gate_qualified_for_timing": report["gate_qualified_for_timing"],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--storm", nargs="*", default=[])
    parser.add_argument("--census")
    parser.add_argument("--arms", nargs="*", default=[])
    parser.add_argument("--analysis")
    parser.add_argument("--buffer")
    args = parser.parse_args()

    run = resume_run()
    if args.stage == "storm":
        log_storm(run, args.storm)
    elif args.stage == "census":
        log_census(run, args.census)
    elif args.stage == "tax":
        log_tax(run, args.arms)
    elif args.stage == "analysis":
        log_analysis(run, args.analysis)
    elif args.stage == "buffer":
        log_buffer(run, args.buffer)
    else:
        print(f"e58: unknown stage {args.stage}", file=sys.stderr)
        return 2
    print(f"e58: logged stage {args.stage} to {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
