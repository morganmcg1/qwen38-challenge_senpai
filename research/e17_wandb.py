#!/usr/bin/env python3
"""E17: publish each timed arm, and the pooled headline, to W&B.

One run per (prompt, arm) so every number in the report is reachable from a run
URL, plus one analysis run carrying the ranked-style medians.

`research/log_wandb.py` cannot be reused here: it parses per-round trace rows,
and E17's timed arms deliberately run with every `MLX_QWEN_MTP_*` variable
cleared, so there are no trace rows to parse.

usage:
  research/e17_wandb.py                       log every completed pair + headline
  research/e17_wandb.py --dry-run             print what would be logged
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e17_analyse import ARMS, HELD_OUT, PROMPTS, collect, median  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
GROUP = "qwen38-r1-e17-curve-transfer-and-refit"
BASE_SHA = "e6e6f81767e84cc8c39b48c09a4f5cac597cdbca"
RUNS = Path(".mlxfast-private/e17/runs")
CURVE_OF = {"CURVE": "merged-h-curve", "FLAT18": "scalar-h-0.18"}


def sh(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def host_config() -> dict:
    return {
        "host_model": sh("sysctl", "-n", "hw.model"),
        "host_chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "host_cores": sh("sysctl", "-n", "hw.ncpu"),
        "host_memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "host_os": platform.mac_ver()[0],
        "git_head": sh("git", "rev-parse", "HEAD"),
        "git_base_sha": BASE_SHA,
        "git_dirty_files": len(sh("git", "status", "--porcelain").splitlines()),
    }


def read_meta(run: Path) -> dict:
    out = {}
    for line in (run / "meta.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def thermals(meta: dict) -> dict:
    """`thermal_before=gpu_temp=41.7C cpu_temp=35.1C ...` -> flat floats."""
    out = {}
    for when in ("before", "after"):
        for field in (meta.get(f"thermal_{when}") or "").split():
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            try:
                out[f"thermal_{when}_{key}"] = float(value.rstrip("CW"))
            except ValueError:
                pass
    return out


def leg(run: Path, name: str) -> dict:
    return json.loads((run / "reports" / name).read_text())


def arm_payload(prompt: str, arm: str, summary: dict) -> tuple[dict, dict, list]:
    run = RUNS / f"{prompt}-{arm}"
    meta = read_meta(run)
    serial, mtp = leg(run, "03-mtp-timed.json"), leg(run, "04-mtp-timed.json")
    score = json.loads((run / "score.json").read_text())

    config = dict(host_config())
    config.update({
        "experiment": "qwen38-r1-e17-curve-transfer-and-refit",
        "assignment_revision": "r1",
        "pr_number": 19,
        "prompt": prompt,
        "prompt_index": PROMPTS.index(prompt),
        "prompt_held_out": prompt in HELD_OUT,
        "arm": arm,
        "h_curve": CURVE_OF[arm],
        "arm_position_in_prompt": 1 if PROMPTS.index(prompt) % 2 == (
            0 if arm == "CURVE" else 1) else 2,
        "decode_tokens": mtp["decode_token_count"],
        "seed_tokens": mtp["seed_token_count"],
        "local_mode": "local-iterate",
        "mtp_depth": mtp["mtp_depth"],
        "serial_control_depth": serial["mtp_depth"],
        "worker_sha256": meta.get("worker_sha256"),
        "source_sha256": meta.get("source_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "head_sha256": mtp["head_provenance"]["sha256"],
        "head_origin": mtp["head_provenance"]["origin"],
        "head_bytes": mtp["head_provenance"]["bytes"],
        "uses_pinned_mtp_head": mtp["uses_pinned_mtp_head"],
        "golden": meta.get("golden"),
        "worktree_dirty_at_run": int(meta.get("dirty", "0")),
        "mlx_qwen_env": meta.get("mlx_qwen_env", ""),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
    })

    n = mtp["decode_token_count"]
    raw = summary["raw"]
    metrics = {
        # headline currency: verbatim, prefill-INCLUSIVE, nothing subtracted
        "raw_p": raw,
        "serial_spt": summary["serial_spt"],
        "mtp_spt": summary["mtp_spt"],
        "score_json_speedup": score["score"],
        # decode-only, reported alongside so the dilution is auditable
        "serial_spt_decode": summary["serial_spt"] - summary["serial_prefill_s"] / n,
        "mtp_spt_decode": summary["mtp_spt"] - summary["mtp_prefill_s"] / n,
        "serial_prefill_s": summary["serial_prefill_s"],
        "mtp_prefill_s": summary["mtp_prefill_s"],
        "prefill_share_of_mtp_leg": summary["mtp_prefill_s"] / (summary["mtp_spt"] * n),
        # drafting behaviour
        "rounds": summary["rounds"],
        "mean_depth": summary["mean_depth"],
        "max_depth": summary["max_depth"],
        "accepted_draft_total": summary["accepted"],
        "rejected_draft_total": summary["rejected"],
        "accepted_draft_rate": summary["accept_rate"],
        "verify_block_replayed_round_count": summary["replays"],
        "non_drafting_round_count": mtp["non_drafting_round_count"],
        # correctness gates
        "all_tokens_matched": summary["matched"],
        "parity_all_ok": summary["parity"],
        "residual_divergence_count": summary["divergence"],
        "declared_rows_total": summary["declared_rows"],
        "reference_checked_row_total": summary["checked_rows"],
        "rejected_rows_reference_checked": mtp["rejected_rows_reference_checked"],
        "max_rejected_tail_logit_delta": mtp["max_rejected_tail_logit_delta"],
        "target_cache_offset_final": mtp["target_cache_offset_final"],
        # latency shape
        "first_block_seconds": mtp["first_block_seconds"],
        "p50_block_request_seconds": mtp["p50_block_request_seconds"],
        "max_block_request_seconds_after_first": mtp[
            "max_block_request_seconds_after_first"],
        "decode_seconds": mtp["decode_seconds"],
    }
    metrics.update(thermals(meta))
    for depth, count in summary["depth_hist"].items():
        metrics[f"depth_hist/d{depth}"] = count
    return config, metrics, mtp["block_request_seconds"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data = collect()
    if not data:
        print("e17_wandb: no completed pairs under", RUNS, file=sys.stderr)
        return 1

    if not args.dry_run:
        import wandb

    urls = {}
    for prompt, arms in data.items():
        for arm in ARMS:
            config, metrics, series = arm_payload(prompt, arm, arms[arm])
            name = f"e17-{prompt}-{arm}"
            if args.dry_run:
                print(f"--- {name}\n  config={json.dumps(config, sort_keys=True)}"
                      f"\n  metrics={json.dumps(metrics, sort_keys=True)}")
                continue
            run = wandb.init(entity=ENTITY, project=PROJECT, name=name, group=GROUP,
                             job_type="local-iterate", config=config, reinit=True,
                             tags=["qwen38-r1-e17", "curve-transfer", arm, prompt])
            for i, seconds in enumerate(series):
                run.log({"block_request_seconds": seconds}, step=i)
            run.summary.update(metrics)
            urls[name] = run.url
            print(f"{name}: {run.url}")
            run.finish()

    # Pooled headline: the ranked-style median over prompts, both populations.
    pooled = {}
    for label, ids in (("held_out7", HELD_OUT), ("all8", PROMPTS)):
        sub = [p for p in ids if p in data]
        if len(sub) < 2:
            continue
        rc = [data[p]["CURVE"]["raw"] for p in sub]
        rf = [data[p]["FLAT18"]["raw"] for p in sub]
        gs = [(data[p]["FLAT18"]["mtp_spt"] - data[p]["CURVE"]["mtp_spt"])
              / data[p]["FLAT18"]["mtp_spt"] for p in sub]
        pooled.update({
            f"{label}/n": len(sub),
            f"{label}/median_raw_curve": median(rc),
            f"{label}/median_raw_flat18": median(rf),
            f"{label}/headline_delta": median(rc) - median(rf),
            f"{label}/headline_delta_pct": 100 * (median(rc) - median(rf)) / median(rf),
            f"{label}/g_median_pct": 100 * median(gs),
            f"{label}/g_min_pct": 100 * min(gs),
            f"{label}/g_max_pct": 100 * max(gs),
            f"{label}/curve_wins": sum(1 for x in gs if x > 0),
            f"{label}/raw_curve_min": min(rc),
            f"{label}/raw_curve_max": max(rc),
            f"{label}/raw_flat18_min": min(rf),
            f"{label}/raw_flat18_max": max(rf),
        })
    for prompt, arms in data.items():
        c, f = arms["CURVE"], arms["FLAT18"]
        mean_serial = (c["serial_spt"] + f["serial_spt"]) / 2
        pooled[f"noise_floor/{prompt}_pct"] = (
            100 * abs(c["serial_spt"] - f["serial_spt"]) / mean_serial)

    if args.dry_run:
        print(f"--- e17-headline\n  {json.dumps(pooled, indent=2, sort_keys=True)}")
        return 0

    config = dict(host_config())
    config.update({"experiment": "qwen38-r1-e17-curve-transfer-and-refit",
                   "arm": "headline", "prompts_completed": sorted(data),
                   "metric_convention": "raw_p = serial_spt / mtp_spt, "
                                        "prefill-inclusive, verbatim"})
    run = wandb.init(entity=ENTITY, project=PROJECT, name="e17-headline", group=GROUP,
                     job_type="analysis", config=config, reinit=True,
                     tags=["qwen38-r1-e17", "curve-transfer", "headline"])
    table = wandb.Table(columns=[
        "prompt", "held_out", "serial_curve", "serial_flat18", "noise_floor_pct",
        "mtp_curve", "mtp_flat18", "raw_curve", "raw_flat18", "d_raw", "g_pct",
        "rounds_curve", "rounds_flat18", "mean_d_curve", "mean_d_flat18"])
    for prompt, arms in data.items():
        c, f = arms["CURVE"], arms["FLAT18"]
        mean_serial = (c["serial_spt"] + f["serial_spt"]) / 2
        table.add_data(
            prompt, prompt in HELD_OUT, c["serial_spt"], f["serial_spt"],
            100 * abs(c["serial_spt"] - f["serial_spt"]) / mean_serial,
            c["mtp_spt"], f["mtp_spt"], c["raw"], f["raw"], c["raw"] - f["raw"],
            100 * (f["mtp_spt"] - c["mtp_spt"]) / f["mtp_spt"],
            c["rounds"], f["rounds"], c["mean_depth"], f["mean_depth"])
    run.log({"per_prompt_pairs": table})
    run.summary.update(pooled)
    print(f"e17-headline: {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
