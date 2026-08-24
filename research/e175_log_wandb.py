#!/usr/bin/env python3
"""Log the E175 candidate evidence to W&B.

Logs two blocks in one run:

* `local/*`  the 512-token real-gate `--local-submit` confirmation of the
  four-file candidate, harness=local, directional timing only.
* `ranked/*` the paired per-prompt decomposition of receipts A, B and C,
  harness=ranked, from board telemetry.

Run: python3 research/e175_log_wandb.py score.json
"""

from __future__ import annotations

import json
import subprocess
import sys

import wandb

import e175_board_q_decomposition as board


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    score_path = sys.argv[1] if len(sys.argv) > 1 else "score.json"
    with open(score_path) as fh:
        score = json.load(fh)
    metrics = score["metrics"]

    ranked = board.main()

    run = wandb.init(
        project="qwen38-mlx-challenge-senpai",
        entity="wandb-applied-ai-team",
        name="e175-four-file-candidate-512-confirmation",
        job_type="local-submit",
        config={
            "experiment": "E175",
            "hypothesis": (
                "software-pipelined qmm_t weight tile (four quantized files) on "
                "organizer-pure main is worth a positive ranked amount"
            ),
            "candidate_commit": git("rev-parse", "HEAD"),
            "base_sha": "bd58c55c9786691eb1c32375d3b1625415c989b9",
            "organizer_sha": "0863b06ac16e26e48fc06e97444095b00feb66d4",
            "submitted_files": [
                "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp",
                "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp",
                "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
                "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h",
            ],
            "worker_sha256": (
                "5e8291e02163bd3103cb993f1b783496550ff40f24065066f6eefc7be146c09d"
            ),
            "host": "AWS EC2 Mac, Apple Silicon, macOS 26.5.2, no _nax execution",
            "cool_gate_passed_real_gate": True,
            "gate_qualified_for_timing": True,
            "decode_tokens": metrics["decode_tokens"],
            "command": (
                "MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 "
                "./benchmark-qwen-mtp.sh --local-submit"
            ),
        },
    )

    run.log(
        {
            "local/mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
            "local/serial_seconds_per_token": metrics["serial_seconds_per_token"],
            "local/mtp_decode_speedup": metrics["mtp_decode_speedup"],
            "local/accepted_draft_rate": metrics["accepted_draft_rate"],
            "local/effective_mean_draft_len": metrics["effective_mean_draft_len"],
            "local/residual_divergence_count": metrics["residual_divergence_count"],
            "local/all_tokens_matched": int(metrics["all_tokens_matched"]),
            "local/public_drift_tripwire_passed": int(
                metrics["public_drift_tripwire_passed"]
            ),
            "local/mtp_depth": metrics["mtp_depth"],
            **{f"ranked/{k}": v for k, v in ranked.items()},
        }
    )
    run.summary["local_gate_entry_temp_serial_c"] = 38.2
    run.summary["local_gate_entry_temp_mtp_c"] = 38.7
    run.finish()
    print(f"W&B: {run.url}")


if __name__ == "__main__":
    main()
