#!/usr/bin/env python3
"""Log one E68 rung-3 leg to W&B, immediately after it is measured.

One W&B run per leg, all in one group, so a session that dies on leg 5 still
leaves legs 1-4 on the board. The rung-1 curve legs are already separate runs
in the same group, logged by the curve driver.

  research/e68_wandb_log.py --leg .mlxfast-private/e68-e2e/runs/TAG

Everything comes from artifacts the leg runner already wrote: `meta.txt` for
the identity tuple, `score.json` for the metrics, and `arm.json` for the
resolved price vector and its pre-registered depth prediction.
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
    "assignment_id": "qwen38-r1-e68-retune-draft-depth-against-new-cost-curve",
    "revision_id": "r1",
    "pr_number": 71,
    "student": "qwen-thorfinn",
    "host_chip": "Apple M4 Pro",
    "harness": "local",
    "local_mode": "--local-iterate",
    "scored_files": "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
}

# meta.txt values that are numbers, not labels.
NUMERIC_META = {
    "tokens", "offered_depth", "physical_memory_gib", "gpu_temp_entry_c",
    "gpu_temp_exit_c", "mlx_max_mb_per_buffer", "mlx_max_ops_per_buffer",
    "stale_metallib_warnings", "status", "wrapper_exit", "dirty",
    "cool_gate_requested", "warmup_discarded",
}


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False, cwd=REPO).stdout.strip()


def read_meta(path):
    meta = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if not _:
            continue
        if key in NUMERIC_META and value not in ("", "None"):
            try:
                meta[key] = float(value) if "." in value else int(value)
                continue
            except ValueError:
                pass
        meta[key] = value
    return meta


def verify_width_histogram(leg_dir):
    """Realised verify widths, as `draft length + 1` per round.

    The capture directory holds whatever the CLI wrote for the leg. Any report
    that carries a per-round draft length answers the question, so search for
    the key rather than for a file name that may change.
    """
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
    ap.add_argument("--group", default="e68-schedule-against-the-new-cost-curve")
    args = ap.parse_args()

    leg_dir = pathlib.Path(args.leg).resolve()
    meta = read_meta(leg_dir / "meta.txt")
    arm = json.loads((leg_dir / "arm.json").read_text())

    score_path = leg_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    metrics = score.get("metrics", {})

    config = dict(ASSIGNMENT)
    config.update({"rung": 3, "leg_tag": meta.get("tag"),
                   "arm": meta.get("arm"), "arm_role": arm.get("role")})
    config.update({"meta_" + k: v for k, v in meta.items()})
    config["marginal"] = arm.get("marginal")
    config["marginal_total"] = arm.get("marginal_total")
    config["measured_raw"] = arm.get("measured_raw")
    config["head_sha"] = git("rev-parse", "HEAD")
    config["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")

    # The prediction goes on the record with the leg that tests it.
    for name, block in (arm.get("predicted_depth") or {}).items():
        config["predicted_%s_p" % name] = block["p"]
        config["predicted_%s_depth_default_cap" % name] = \
            block["default_cap"]["depth"]
        config["predicted_%s_depth_streak_cap" % name] = \
            block["streak_cap"]["depth"]

    log = {
        # The primary. A schedule change is confined to the candidate MTP leg,
        # so absolute candidate time is the causal quantity; the local ratio is
        # reported beside it, never instead of it.
        "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
        "mtp_decode_speedup": metrics.get("mtp_decode_speedup"),
        "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "residual_divergence_count": metrics.get("residual_divergence_count"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "decode_tokens": metrics.get("decode_tokens"),
        "score": score.get("score"),
        "passed": score.get("passed"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "stale_metallib_warnings": meta.get("stale_metallib_warnings"),
        "leg_status": meta.get("status"),
    }
    widths = verify_width_histogram(leg_dir)
    if widths:
        log["realised_mean_verify_width"] = widths["mean_verify_width"]
        log["realised_rounds"] = widths["rounds"]
        for width, fraction in widths["fraction"].items():
            log["realised_verify_width_fraction_%s" % width] = fraction
        config["realised_verify_width_histogram"] = widths["histogram"]

    run = wandb.init(entity=ENTITY, project=PROJECT, group=args.group,
                     job_type="rung3-leg", name=meta.get("tag"),
                     config=config, reinit=True)
    run.log({k: v for k, v in log.items() if v is not None})
    run.finish()
    print("e68_wandb_log: logged %s as %s" % (meta.get("tag"), run.id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
