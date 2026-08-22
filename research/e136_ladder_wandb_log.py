#!/usr/bin/env python3
"""Publish one E136 launch-column ladder session to W&B.

    usage: research/e136_wandb_log.py --label L1 \
               --json research/e136-artifacts/L1-ladder.json [--dry]

The ladder measures how candidate round time scales with the number of
launched QMV threadgroup columns while the work performed and the emitted
token stream stay fixed. Its linear slope is the direct regression estimate of
`e135_launch_cost_us_per_column`, which FINDING 182 could only reach as a
two-point difference.

Every leg runs with `MLXFAST_LOCAL_COOL_GATE=0` under the standing
counterbalanced-arm exception, so this run records
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false. Nothing here is a ranked score.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e136_ladder_report as report  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e135-tight-qmv-launch-grid"

QUESTION = (
    "is the cost of a launched QMV threadgroup column flat, linear or "
    "logarithmic in the number of columns launched, given that every padded "
    "column takes the kernel early return and writes nothing"
)

# The one-pass plans that FINDING 182 put on the stop list. Reopening them is
# only justified if a column still costs something at low column counts.
REOPENER = {8: 8, 9: 9}


def law_key(name: str) -> str:
    return name.split()[0]


def marginal_saving_columns(hist: dict[int, int], plan: dict[int, int],
                            ) -> tuple[float, float]:
    """Columns and ln-columns removed per leg by adopting the REOPENER plans.

    Under `tight`, width `m` launches `ceil(m / ipg)` columns. A one-pass plan
    launches exactly one. Only the launch side is priced here: the register
    and spill cost of the wider accumulator is a separate, opposing term.
    """
    cols = 0.0
    logs = 0.0
    for m, count in hist.items():
        if m not in plan or m not in REOPENER:
            continue
        now = -(-m // plan[m])
        then = -(-m // REOPENER[m])
        cols += count * (now - then)
        logs += count * (math.log(now) - math.log(then))
    return cols, logs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="L1")
    ap.add_argument("--json", required=True,
                    help="summary written by e136_ladder_report.py --json")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    summary = json.loads(pathlib.Path(args.json).read_text())
    rows = report.legs(args.label, "ladder")
    if not rows:
        print("e136_wandb_log: no scored ladder legs")
        return 1

    fits = {law_key(f["name"]): f for f in summary["fits"]}
    linear = fits["linear"]
    noise = summary["pooled_noise_us_per_round"]

    # The law that both fits the ladder best and predicts the held-out `wide`
    # anchor best. Ranking on residual alone would let a law win by bending
    # through the rungs while missing the anchor entirely.
    def penalty(f: dict) -> float:
        err = abs(f.get("wide_out_of_sample_error_us", float("nan")))
        return f["residual_sd"] + (0.0 if math.isnan(err) else err)

    verdict = min(fits.values(), key=penalty)

    hist = {int(k): v for k, v in summary["width_histogram"].items()}
    plan = report.plan_from(
        json.loads((report.OUT / f"e136{args.label}wtight"
                    / "pipelines.json").read_text())["plan"])
    rounds = summary["rounds"]
    save_cols, save_logs = marginal_saving_columns(hist, plan)

    divergences = sum(int(r["metrics"].get("residual_divergence_count", 0) or 0)
                      for r in rows)
    matched = all(r["metrics"].get("all_tokens_matched") is True for r in rows)
    drafts = sorted({r["metrics"].get("effective_mean_draft_len")
                     for r in rows} - {None})
    accepts = sorted({r["metrics"].get("accepted_draft_rate")
                      for r in rows} - {None})

    meta = rows[0]["meta"]
    config = {
        "experiment": "e136-qmv-launch-column-ladder",
        "question": QUESTION,
        "harness": "local",
        "session_label": args.label,
        "decode_tokens": summary["tokens"],
        "rounds_per_leg": rounds,
        "local_mode": meta.get("local_mode"),
        "arms": "wide, tight, tight2, tight4, tight8",
        "design": ("wide tight tight2 tight4 tight8 tight8 tight4 tight2 "
                   "tight wide palindrome x 2, every arm at mean position 5.5"),
        "legs": len(rows),
        "wide_is_held_out_of_sample": True,
        "padded_columns_take_kernel_early_return": True,
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "memory_bytes": meta.get("memory_bytes"),
        "sandbox": meta.get("sandbox"),
        "head_dir": meta.get("head_dir"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
        "qmv_table": "onepass67",
        "width_histogram": summary["width_histogram"],
        "launched_columns_per_leg": {
            a: summary["census"][a]["columns"] for a in report.ARMS},
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "reproduce": f"research/e136_ladder_abba.sh 2 512 {args.label} 1",
    }

    metrics = {
        # Required E135 metric, now a regression slope rather than a
        # two-point difference.
        "e135_launch_cost_us_per_column": linear["slope"],
        "e135_launch_cost_us_per_column_residual_sd": linear["residual_sd"],
        "e135_exact_token_divergences": divergences,
        "e135_all_tokens_matched": matched,

        "e136_law_verdict": law_key(verdict["name"]),
        "e136_pooled_noise_us_per_round": noise,
        "e136_wide_us_per_round": summary.get("wide_us_per_round"),
        "e136_tight_us_per_round": summary.get("tight_us_per_round"),
        "e136_schedule_identical_across_rungs":
            len(drafts) == 1 and len(accepts) == 1,
        "e136_effective_mean_draft_len": drafts[0] if drafts else None,

        # The launch-side half of the FINDING 182 reopener. The register and
        # spill cost of a wider accumulator is not priced here.
        "e136_reopener_columns_saved_per_round": save_cols / rounds,
        "e136_reopener_launch_saving_us_per_round_linear":
            linear["slope"] * save_cols / rounds,
    }
    if "log" in fits:
        metrics["e136_reopener_launch_saving_us_per_round_log"] = (
            fits["log"]["slope"] * save_logs / rounds)

    for key, f in fits.items():
        metrics[f"e136_{key}_slope"] = f["slope"]
        metrics[f"e136_{key}_residual_sd"] = f["residual_sd"]
        metrics[f"e136_{key}_sd_over_noise"] = f["ratio_to_noise"]
        if "wide_out_of_sample_error_us" in f:
            metrics[f"e136_{key}_wide_out_of_sample_error_us"] = (
                f["wide_out_of_sample_error_us"])

    for arm, a in summary["arms"].items():
        metrics[f"e136_us_per_round_{arm}"] = a["us_per_round"]
        metrics[f"e136_pct_vs_tight_{arm}"] = a["pct_vs_tight"]
        metrics[f"e136_entry_temp_spread_c_{arm}"] = (
            a["entry_temp_c_max"] - a["entry_temp_c_min"])

    print(json.dumps({"config": config, "metrics": metrics}, indent=2,
                     default=str))
    if args.dry:
        return 0

    import wandb

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        id=args.run_id, name=args.name or f"e136-{args.label}-launch-column-ladder",
        job_type="local-abba-session", config=config)
    for r in rows:
        m = r["metrics"]
        run.log({
            "leg_index": int(r["meta"]["e136_leg_index"]),
            "leg_arm": r["arm"],
            "leg_launched_columns": summary["census"][r["arm"]]["columns"],
            "leg_mtp_seconds_per_token": m.get("mtp_seconds_per_token"),
            "leg_us_per_round": (
                m["mtp_seconds_per_token"] * summary["tokens"] * 1e6 / rounds
                if m.get("mtp_seconds_per_token") else None),
            "leg_serial_seconds_per_token": m.get("serial_seconds_per_token"),
            "leg_local_ratio": m.get("mtp_decode_speedup"),
            "leg_gpu_temp_entry_c": r["meta"].get("gpu_temp_entry_c"),
            "leg_gpu_temp_exit_c": r["meta"].get("gpu_temp_exit_c"),
            "leg_effective_mean_draft_len": m.get("effective_mean_draft_len"),
            "leg_all_tokens_matched": m.get("all_tokens_matched"),
        })
    for key, value in metrics.items():
        run.summary[key] = value
    run.finish()
    print(f"\ne136_wandb_log: {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
