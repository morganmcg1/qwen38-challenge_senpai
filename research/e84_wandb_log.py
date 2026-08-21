#!/usr/bin/env python3
"""Log one E84 leg to W&B, immediately after it is measured.

One run per leg in one group, so a session that dies on leg 6 still leaves
legs 1-5 on the board. Everything comes from artifacts the leg runner wrote:
`meta.txt` for the identity tuple and `score.json` for the metrics.

  research/e84_wandb_log.py --leg .mlxfast-private/e84/runs/TAG

Every logged number is `harness=local`. The local serial-to-MTP ratio is
recorded beside absolute candidate time, never instead of it: both E84 legs
use the same candidate build, so a change that speeds the shared target path
partly cancels in that ratio.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
REPO = pathlib.Path(__file__).resolve().parent.parent

ASSIGNMENT = {
    "assignment_id":
        "qwen38-r1-e84-delete-two-pieces-of-ranked-measured-dead-work",
    "revision_id": "r1",
    "pr_number": 86,
    "student": "qwen-askeladd",
    "harness": "local",
    "local_mode": "--local-iterate",
    "scored_files": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
}

ARM_MECHANISM = {
    "base": "none",
    "a": "precision-island K/V dead work removed",
    "b": "state-only Gated DeltaNet prefix replay",
    "ab": "both",
}

NUMERIC_META = {
    "tokens", "offered_depth", "physical_memory_gib", "gpu_temp_entry_c",
    "gpu_temp_exit_c", "mlx_max_mb_per_buffer", "mlx_max_ops_per_buffer",
    "stale_metallib_warnings", "status", "wrapper_exit", "dirty",
    "cool_gate_requested", "warmup_discarded", "worker_assert_post_exit",
}


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False, cwd=REPO).stdout.strip()


def read_meta(path):
    meta = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key in NUMERIC_META and value not in ("", "None"):
            try:
                meta[key] = float(value) if "." in value else int(value)
                continue
            except ValueError:
                pass
        meta[key] = value
    return meta


def draft_width_histogram(leg_dir):
    """Realised verify widths, as `draft length + 1` per round."""
    lengths = []
    for path in sorted((leg_dir / "reports").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if (key in ("effective_draft_lengths", "draft_lengths",
                                "accepted_counts")
                            and isinstance(value, list)):
                        lengths.extend(v for v in value
                                       if isinstance(v, (int, float)))
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(v for v in node if isinstance(v, (dict, list)))
    if not lengths:
        return None
    widths = collections.Counter(int(v) + 1 for v in lengths)
    total = sum(widths.values())
    return {
        "rounds": total,
        "mean_verify_width": sum(w * n for w, n in widths.items()) / total,
        "histogram": {str(w): widths[w] for w in sorted(widths)},
        "fraction": {str(w): widths[w] / total for w in sorted(widths)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True)
    ap.add_argument(
        "--group",
        default="e84-delete-two-pieces-of-ranked-measured-dead-work")
    args = ap.parse_args()

    leg_dir = pathlib.Path(args.leg).resolve()
    meta = read_meta(leg_dir / "meta.txt")
    score_path = leg_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    metrics = score.get("metrics", {})

    arm = meta.get("arm")
    config = dict(ASSIGNMENT)
    config.update({
        "leg_tag": meta.get("tag"),
        "arm": arm,
        "arm_mechanism": ARM_MECHANISM.get(arm, "unknown"),
        "mechanism_a_present": arm in ("a", "ab"),
        "mechanism_b_present": arm in ("b", "ab"),
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    })
    config.update({"meta_" + k: v for k, v in meta.items()})

    log = {
        # The primary. Both local legs use the candidate build, so absolute
        # candidate MTP time is the causal quantity for a broad change.
        "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
        "mtp_decode_speedup": metrics.get("mtp_decode_speedup"),
        "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "residual_divergence_count":
            metrics.get("residual_divergence_count"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "decode_tokens": metrics.get("decode_tokens"),
        "score": score.get("score"),
        "passed": score.get("passed"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "stale_metallib_warnings": meta.get("stale_metallib_warnings"),
        "leg_status": meta.get("status"),
    }
    widths = draft_width_histogram(leg_dir)
    if widths:
        log["realised_mean_verify_width"] = widths["mean_verify_width"]
        log["realised_rounds"] = widths["rounds"]
        for width, fraction in widths["fraction"].items():
            log["realised_verify_width_fraction_%s" % width] = fraction
        config["realised_verify_width_histogram"] = widths["histogram"]

    run = wandb.init(entity=ENTITY, project=PROJECT, group=args.group,
                     job_type="e84-leg", name=meta.get("tag"),
                     config=config, reinit=True)
    run.log({k: v for k, v in log.items() if v is not None})
    run.finish()
    print("e84_wandb_log: logged %s as %s (%s)"
          % (meta.get("tag"), run.id, run.url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
