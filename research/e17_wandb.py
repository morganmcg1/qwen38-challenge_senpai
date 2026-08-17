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
import e17_analyse  # noqa: E402
from e17_analyse import (  # noqa: E402
    ARMS, CONTROL, HELD_OUT, PROMPTS, candidates, collect, median, wide_share,
)

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
GROUP = "qwen38-r1-e17-curve-transfer-and-refit"
BASE_SHA = "af80b0fc93cf20e8405631bb53365ace21a1f913"

# h[1] is the swept element; see research/e17-r2-prereg.md sections 1 and 4 for why
# it is the one that both closes depth 2 and opens depth 3+.
ARM_DESC = {
    "S18": ("scalar-h-0.18-untouched-head", 0.18, "control"),
    "S18R": ("scalar-h-0.18-replicate", 0.18, "noise-floor-replicate"),
    "CURVE": ("merged-h-curve", 0.0775, "transfer-candidate"),
    "H1LO": ("flat-0.18-h1-0.0800", 0.0800, "below-measured-marginal"),
    "H1MEAS": ("flat-0.18-h1-0.1152", 0.1152, "measured-marginal"),
    "H1HI": ("flat-0.18-h1-0.3000", 0.3000, "above-shipped"),
    # r1 arms, kept so --runs-root can replay r1 through the same logger
    "FLAT18": ("scalar-h-0.18", 0.18, "r1-control"),
}


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


def session_order(prompt: str, arms: dict[str, dict]) -> dict[str, int]:
    """Rank the prompt's arms by actual wall-clock start, 1-based.

    The runner rotates arm order by prompt index, so a slot cannot be derived
    from the arm name alone. `meta.txt` records when each arm really ran, which
    is what a thermal-position confound would attach to.
    """
    started = {a: arms[a]["meta"].get("started", "") for a in arms}
    return {a: i + 1 for i, a in enumerate(sorted(started, key=lambda k: started[k]))}


def arm_payload(
    prompt: str, arm: str, summary: dict, position: int
) -> tuple[dict, dict, list]:
    run, meta = Path(summary["run"]), summary["meta"]
    serial, mtp = leg(run, "03-mtp-timed.json"), leg(run, "04-mtp-timed.json")
    score = json.loads((run / "score.json").read_text())
    h_curve, h1, role = ARM_DESC[arm]

    config = dict(host_config())
    config.update({
        "experiment": "qwen38-r1-e17-curve-transfer-and-refit",
        "assignment_revision": "r2",
        "pr_number": 19,
        "prompt": prompt,
        "prompt_index": PROMPTS.index(prompt),
        "prompt_held_out": prompt in HELD_OUT,
        "arm": arm,
        "arm_role": role,
        "arm_is_control": arm == CONTROL,
        "h_curve": h_curve,
        "h1": h1,
        "arm_position_in_prompt": position,
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
        # headline currency: verbatim (P + D) / N, prefill-inclusive, which is
        # what the track scores. Nothing is subtracted or added to reach it.
        "raw_p": raw,
        "serial_spt": summary["serial_spt"],
        "mtp_spt": summary["mtp_spt"],
        "score_json_speedup": score["score"],
        # decode-only DIAGNOSTIC: subtract each leg's own measured seed prefill
        # out of the already-inclusive field. Never a headline.
        "serial_spt_decode_only": summary["serial_spt"] - summary["serial_prefill_s"] / n,
        "mtp_spt_decode_only": summary["mtp_spt"] - summary["mtp_prefill_s"] / n,
        "raw_p_decode_only": (summary["serial_spt"] - summary["serial_prefill_s"] / n)
        / (summary["mtp_spt"] - summary["mtp_prefill_s"] / n),
        "serial_prefill_s": summary["serial_prefill_s"],
        "mtp_prefill_s": summary["mtp_prefill_s"],
        "decode_share_of_mtp_leg": (summary["mtp_spt"] - summary["mtp_prefill_s"] / n)
        / summary["mtp_spt"],
        # drafting behaviour
        "rounds": summary["rounds"],
        "mean_depth": summary["mean_depth"],
        "max_depth": summary["max_depth"],
        # M = depth + 1 per the row ledger, so M >= 5 is depth >= 4; this is the
        # share of rounds that reach the sdpaWidthWallDepthCap-relevant widths.
        "wide_round_share_M_ge_5": wide_share(summary),
        "rows_per_round": summary["declared_rows"] / max(summary["rounds"], 1),
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
        print("e17_wandb: no completed pairs under", e17_analyse.RUNS,
              file=sys.stderr)
        return 1

    if not args.dry_run:
        import wandb

    urls = {}
    for prompt, arms in data.items():
        position = session_order(prompt, arms)
        for arm in ARMS:
            if arm not in arms:
                continue
            config, metrics, series = arm_payload(
                prompt, arm, arms[arm], position[arm])
            name = f"e17r2-{prompt}-{arm}"
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

    # Pooled headline: ranked-style median over prompts, per candidate arm.
    # `n` is always logged, so a one- or two-prompt "median" cannot be mistaken
    # for the eight-prompt ranked aggregation.
    pooled = {}
    all_arms = sorted({a for arms in data.values() for a in arms})
    for arm in all_arms:
        if arm == CONTROL:
            continue
        for label, ids in (("held_out", HELD_OUT), ("all", PROMPTS)):
            sub = [p for p in ids if p in data and arm in data[p]]
            if not sub:
                continue
            # `raw` is the scored currency; `g` is the MTP leg's seconds/token
            # reduction. They are different percentages of different things:
            # headline_delta_pct moves the score, g_median_pct moves decode time.
            r_arm = [data[p][arm]["raw"] for p in sub]
            r_ctl = [data[p][CONTROL]["raw"] for p in sub]
            gs = [(data[p][CONTROL]["mtp_spt"] - data[p][arm]["mtp_spt"])
                  / data[p][CONTROL]["mtp_spt"] for p in sub]
            pooled.update({
                f"{arm}/{label}/n": len(sub),
                f"{arm}/{label}/prompts": ",".join(sub),
                f"{arm}/{label}/median_raw_arm": median(r_arm),
                f"{arm}/{label}/median_raw_control": median(r_ctl),
                f"{arm}/{label}/headline_delta": median(r_arm) - median(r_ctl),
                f"{arm}/{label}/headline_delta_pct":
                    100 * (median(r_arm) - median(r_ctl)) / median(r_ctl),
                f"{arm}/{label}/g_median_pct": 100 * median(gs),
                f"{arm}/{label}/g_min_pct": 100 * min(gs),
                f"{arm}/{label}/g_max_pct": 100 * max(gs),
                f"{arm}/{label}/arm_wins": sum(1 for x in gs if x > 0),
            })

    # Two independent noise floors. The serial floor is within-prompt across two
    # runs of a byte-identical depth-0 leg; the S18R floor is a whole arm-level
    # repeat of the control binary and is the threshold a candidate must clear.
    for prompt, arms in data.items():
        legs = [v["serial_spt"] for v in arms.values()]
        pooled[f"noise_floor_serial/{prompt}_pct"] = (
            100 * (max(legs) - min(legs)) / (sum(legs) / len(legs)))
        if "S18R" in arms and CONTROL in arms:
            r, a = arms["S18R"], arms[CONTROL]
            pooled[f"noise_floor_arm/{prompt}_g_pct"] = (
                100 * (a["mtp_spt"] - r["mtp_spt"]) / a["mtp_spt"])
            pooled[f"noise_floor_arm/{prompt}_d_raw"] = r["raw"] - a["raw"]

    if args.dry_run:
        print(f"--- e17-headline\n  {json.dumps(pooled, indent=2, sort_keys=True)}")
        return 0

    config = dict(host_config())
    config.update({"experiment": "qwen38-r1-e17-curve-transfer-and-refit",
                   "arm": "headline", "prompts_completed": sorted(data),
                   "metric_convention": "raw_p = serial_spt / mtp_spt, verbatim "
                                        "(P+D)/N prefill-inclusive = scored currency; "
                                        "raw_p_decode_only is a diagnostic only"})
    run = wandb.init(entity=ENTITY, project=PROJECT, name="e17-headline", group=GROUP,
                     job_type="analysis", config=config, reinit=True,
                     tags=["qwen38-r1-e17", "curve-transfer", "headline"])
    table = wandb.Table(columns=[
        "prompt", "arm", "role", "h1", "is_control", "held_out", "position",
        "serial_spt", "mtp_spt", "raw", "d_raw_vs_control", "g_pct_vs_control",
        "rounds", "mean_depth", "max_depth", "wide_share_M_ge_5",
        "rows_per_round", "accept_rate", "declared_rows", "tokens_matched",
        "divergence"])
    for prompt, arms in data.items():
        position = session_order(prompt, arms)
        ctl = arms.get(CONTROL)
        for arm in sorted(arms, key=lambda a: position[a]):
            s = arms[arm]
            _, h1, role = ARM_DESC.get(arm, ("?", None, "?"))
            table.add_data(
                prompt, arm, role, h1, arm == CONTROL, prompt in HELD_OUT,
                position[arm], s["serial_spt"], s["mtp_spt"], s["raw"],
                None if ctl is None else s["raw"] - ctl["raw"],
                None if ctl is None else
                100 * (ctl["mtp_spt"] - s["mtp_spt"]) / ctl["mtp_spt"],
                s["rounds"], s["mean_depth"], s["max_depth"],
                100 * wide_share(s), s["declared_rows"] / max(s["rounds"], 1),
                s["accept_rate"], s["declared_rows"], s["matched"],
                s["divergence"])
    run.log({"per_prompt_arms": table})
    run.summary.update(pooled)
    print(f"e17-headline: {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
