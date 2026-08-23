#!/usr/bin/env python3
"""E147: log every required metric and the full identity tuple to one W&B run.

HARNESS LABELLING. `program.md` invalidates an unlabelled score model, and this
experiment carries two harnesses that must never be mixed:

  harness=local   the M4 Pro ABBA session. It measures the NON-NAX
                  `affine_qmm_t`, which is the seed-prefill path on a host
                  where `is_nax_available()` is false. It validates the
                  transformation. It is not a ranked estimate and it is not
                  gate qualified.
  harness=ranked  the eight published `prefill_seconds_per_token` values from
                  an official receipt, and the published-median repricing
                  derived from them. Both ranked legs pay the same absolute
                  seed prefill, so `raw' = (S - dP) / (C - dP)` and no
                  `psi_serial` term exists or may be subtracted.

Every metric name below carries its harness, and `measurement_harness` in the
config records which one produced each artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# harness=ranked. The published seed prefill at the current bar, and the modern
# band every recent row except two sits inside.
RANKED_PREFILL_SPT_BAR = 0.001028291
RANKED_BAND_LOW = 0.0010277
RANKED_BAND_HIGH = 0.0010349

# harness=ranked. Advisor value model, published median against prefill cut.
FORECAST_TABLE = {
    0.0: 0.0,
    -1.5: 0.1122,
    -3.0: 0.2251,
    -4.1: 0.3071,
    -5.1: 0.3822,
}

# Threadgroup staging footprint in bytes, derived from the instantiation
# macros and the allocation expressions rather than measured.
THREADGROUP_BYTES = {
    "non_nax_32x32x32_bf16_xs_and_ws_doubled": 10_240,
    "non_nax_32x32x32_float32_xs_and_ws_doubled": 18_432,
    "nax_64x64x64_bf16_ws_doubled": 18_432,
    "nax_64x64x64_float32_ws_doubled_would_be": 34_816,
    "nax_64x64x64_float32_kept_unpipelined": 17_408,
    "limit": 32_768,
}


def load_json(path: str):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def load_kv(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def interpolate_forecast(pct: float) -> float:
    """Linear interpolation of the advisor's ranked value table."""
    xs = sorted(FORECAST_TABLE)
    if pct >= xs[-1]:
        return FORECAST_TABLE[xs[-1]]
    if pct <= xs[0]:
        # Below the tabulated range, extend the local slope.
        lo, hi = xs[0], xs[1]
    else:
        lo = max(x for x in xs if x <= pct)
        hi = min(x for x in xs if x >= pct)
        if lo == hi:
            return FORECAST_TABLE[lo]
    span = hi - lo
    frac = (pct - lo) / span
    return FORECAST_TABLE[lo] + frac * (FORECAST_TABLE[hi] - FORECAST_TABLE[lo])


def rung_a_metrics(rung_a: dict) -> tuple[dict, dict]:
    legs = rung_a["legs"]
    base = [leg["seed_prefill_seconds"] for leg in legs if leg["arm"] == "base"]
    cand = [leg["seed_prefill_seconds"] for leg in legs if leg["arm"] == "cand"]
    mean_base = statistics.fmean(base)
    mean_cand = statistics.fmean(cand)
    sd_base = statistics.stdev(base)
    sd_cand = statistics.stdev(cand)
    stderr = math.sqrt(sd_base**2 / len(base) + sd_cand**2 / len(cand))
    diff = mean_cand - mean_base
    pct = 100.0 * diff / mean_base
    pooled = rung_a["pooled"]

    metrics = {
        "e147_rungA_local_prefill_pct": pct,
        "e147_rungA_local_prefill_pct_block_form": pooled["prefill_gain_pct"],
        "e147_rungA_local_prefill_pct_stderr": 100.0 * stderr / mean_base,
        "e147_rungA_local_prefill_t": diff / stderr,
        "e147_local_seed_prefill_seconds_base": mean_base,
        "e147_local_seed_prefill_seconds_arm": mean_cand,
        "e147_local_seed_prefill_spt_base": mean_base / 512.0,
        "e147_local_to_ranked_level_factor": (mean_base / 512.0)
        / RANKED_PREFILL_SPT_BAR,
        "e147_rungA_exactness_divergences": float(pooled["divergences"]),
        "e147_rungA_unmatched_legs": float(pooled["unmatched_legs"]),
        "e147_rungA_witness_mismatches": float(pooled["witness_mismatches"]),
        "e147_rungA_legs": float(pooled["leg_count"]),
        "e147_rungA_decode_spt_gain_pct": pooled["decode_spt_gain_pct"],
        "e147_rungA_rank_separation_complete": float(max(cand) < min(base)),
    }
    for prompt, row in rung_a["prompts"].items():
        metrics[f"e147_rungA_local_prefill_pct_{prompt}"] = row["prefill_gain_pct"]
        metrics[f"e147_rungA_prefill_base_seconds_{prompt}"] = row["prefill_base_mean"]
        metrics[f"e147_rungA_prefill_cand_seconds_{prompt}"] = row["prefill_cand_mean"]

    config = {
        "rungA_identity": rung_a["identity"],
        "rungA_estimator": rung_a["estimator"],
        "rungA_arm_selector": rung_a["arm_selector"],
        "rungA_entry_temp_c_min": pooled.get("entry_temp_c_min"),
        "rungA_entry_temp_c_max": pooled.get("entry_temp_c_max"),
        "rungA_blocks_pct": pooled["prefill_gain_pct_per_block"],
    }
    return metrics, config


def rung_d_metrics(receipt: dict) -> tuple[dict, dict]:
    """harness=ranked. Read the eight published prefill values off a receipt."""
    values = receipt["prefill_seconds_per_token"]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    pct = 100.0 * (mean / RANKED_PREFILL_SPT_BAR - 1.0)
    forecast = interpolate_forecast(pct)
    metrics = {
        "e147_ranked_prefill_pct": pct,
        "e147_ranked_prefill_spt_mean": mean,
        "e147_ranked_prefill_within_row_cv_pct": 100.0 * sd / mean,
        "e147_ranked_prefill_vs_band_low_pct": 100.0 * (mean / RANKED_BAND_LOW - 1.0),
        "e147_ranked_prefill_vs_band_high_pct": 100.0 * (mean / RANKED_BAND_HIGH - 1.0),
        "e147_forecast_median_pct": forecast,
    }
    realised = receipt.get("realised_median_pct")
    if realised is not None:
        metrics["e147_realised_median_pct"] = realised
        metrics["e147_forecast_error_pp"] = realised - forecast
    config = {
        "ranked_prefill_spt_per_prompt": values,
        "ranked_submission_id": receipt.get("submission_id"),
        "ranked_score": receipt.get("score"),
        "ranked_base_score": receipt.get("base_score"),
    }
    return metrics, config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung-a", default="research/e147-rungA.json")
    ap.add_argument("--control", default="research/e147-rungA-control.txt")
    ap.add_argument("--receipt", default="research/e147-rungD.json")
    ap.add_argument("--name", default="e147-nax-seed-prefill-double-buffer")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    config: dict = {
        "experiment": "E147",
        "pr": 147,
        "commit": commit,
        "base_sha": "bcc11dc6527ea4fa32be22c15b99a3a95695bfaf",
        "assignment_scope_base": "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf",
        "mechanism": (
            "software-pipeline the affine quantized transposed GEMM k-loop: "
            "double the threadgroup staging buffer, stage tile 0 in a "
            "prologue, keep one barrier per iteration, alternate halves"
        ),
        "reference_implementation": "fp_quantized_nax.h shipped pipelined k-loop",
        "reassociation": "none; per-element mma sequence unchanged (Rule 92)",
        "live_source_form": (
            "Metal JIT string compiled into the runtime worker; "
            "nojit_kernels.cpp is excluded from the package"
        ),
        "threadgroup_bytes": THREADGROUP_BYTES,
        "e147_max_threadgroup_bytes": max(
            v for k, v in THREADGROUP_BYTES.items()
            if k not in ("limit", "nax_64x64x64_float32_ws_doubled_would_be")
        ),
        "host": "Mac16,11 Apple M4 Pro 48 GiB applegpu_g16s",
        "nax_available_locally": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "depth_price_arm": "pb6 (compiled default, MLX_E134_DEPTH_PRICE_ARM unset)",
        "ranked_prefill_spt_bar": RANKED_PREFILL_SPT_BAR,
        "ranked_value_model": "raw' = (S - dP)/(C - dP); no psi_serial term",
    }

    metrics: dict = {
        "e147_max_threadgroup_bytes": float(config["e147_max_threadgroup_bytes"]),
    }

    rung_a = load_json(args.rung_a)
    if rung_a:
        rung_a_m, rung_a_c = rung_a_metrics(rung_a)
        metrics.update(rung_a_m)
        config.update(rung_a_c)

    control = load_kv(args.control)
    if control:
        fired = control.get("e147_rungA_positive_control_failed")
        if fired is not None:
            metrics["e147_rungA_positive_control_failed"] = float(fired)
        for mode in ("barrier0", "nobarrier", "wronghalf"):
            key = f"e147_control_{mode}_caught"
            if key in control:
                metrics[f"e147_rungA_control_{mode}_caught"] = float(control[key])
        config["rungA_control_detail"] = control

    receipt = load_json(args.receipt)
    if receipt:
        rung_d_m, rung_d_c = rung_d_metrics(receipt)
        metrics.update(rung_d_m)
        config.update(rung_d_c)

    revert = subprocess.run(
        ["git", "diff", "--stat", "bdba19f66e84e7e2aa1f8162eeaa0a82579edc94",
         "--", "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"],
        capture_output=True, text=True, check=True).stdout.strip()
    metrics["e147_e141_revert_is_byte_exact"] = 1.0 if revert == "" else 0.0

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        config=config,
        mode="offline" if args.offline else "online",
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    print(f"e147 wandb run {run.id} {run.url}")
    for key in sorted(metrics):
        print(f"  {key} = {metrics[key]}")
    run.finish()


if __name__ == "__main__":
    main()
