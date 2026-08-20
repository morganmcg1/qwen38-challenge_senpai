#!/usr/bin/env python3
"""Turn one counterbalanced E60 session into the rung-0 answer.

  research/e60_rung0.py --tokens 300 --json research/e60-artifacts/e60-rung0-t300.json

The headline is the ABSOLUTE candidate seconds per token. The local speedup
ratio is reported only as a diagnostic: Qwen35.swift sits on both local legs, so
a target-side change cancels there while remaining a pure ranked gain.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics

LOCAL_NULL_FLOOR_PERCENT = 0.0629
# Ledger 193 retracted the +0.283 % ranked MDE. The measured 95 % threshold for
# a single ranked A/B pair is 2.10 %, from 37 within-group pairs over 18
# byte-identical submitted surfaces.
RANKED_SINGLE_PAIR_THRESHOLD_PERCENT = 2.10


def read_meta(leg: pathlib.Path) -> dict:
    meta: dict[str, str] = {}
    path = leg / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                meta[key.strip()] = value.strip()
    return meta


def collect(tokens: int) -> dict:
    legs = {}
    for path in sorted(glob.glob(f"research/out/e60-t{tokens}-*")):
        leg = pathlib.Path(path)
        # The declared thermal warm-up leg is discarded before any arm is read.
        if not leg.name.split("-")[2][:-1].isdigit():
            continue
        meta = read_meta(leg)
        record = {
            "tag": leg.name,
            "arm": meta.get("arm"),
            "order": int(leg.name.split("-")[2][:-1]),
            "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"])
            if meta.get("gpu_temp_entry_c") else None,
            "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"])
            if meta.get("gpu_temp_exit_c") else None,
            "exit": int(meta.get("exit", -1) or -1),
            "worker_env": (leg / "worker-env.txt").read_text().split()
            if (leg / "worker-env.txt").exists() else [],
            "worker_sha256": meta.get("staged_worker_sha256"),
        }
        score_path = leg / "score.json"
        if score_path.exists():
            metrics = json.loads(score_path.read_text())["metrics"]
            mdl = metrics["effective_mean_draft_len"]
            rate = metrics["accepted_draft_rate"]
            decode = metrics["decode_tokens"]
            rounds = decode / (1.0 + rate * mdl)
            drafted = rounds * mdl
            accepted = decode - rounds
            record.update(
                {
                    "ok": True,
                    "decode_tokens": decode,
                    "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
                    "serial_seconds_per_token": metrics["serial_seconds_per_token"],
                    "mtp_decode_speedup": metrics["mtp_decode_speedup"],
                    "effective_mean_draft_len": mdl,
                    "accepted_draft_rate": rate,
                    "all_tokens_matched": metrics["all_tokens_matched"],
                    "residual_divergence_count": metrics["residual_divergence_count"],
                    "rounds": rounds,
                    "drafted_rows": drafted,
                    "accepted_rows": accepted,
                    "rejected_rows": drafted - accepted,
                }
            )
        else:
            record["ok"] = False
            err = leg / "wrapper.err"
            record["error_tail"] = (
                err.read_text().splitlines()[-12:] if err.exists() else []
            )
        legs[leg.name] = record
    return legs


def arm_summary(legs: dict) -> dict:
    arms: dict[str, dict] = {}
    for record in legs.values():
        arm = record["arm"]
        arms.setdefault(arm, {"legs": [], "failed_legs": []})
        if record.get("ok"):
            arms[arm]["legs"].append(record)
        else:
            arms[arm]["failed_legs"].append(record["tag"])
    for arm, entry in arms.items():
        good = entry["legs"]
        if not good:
            continue
        mtp = [leg["mtp_seconds_per_token"] for leg in good]
        serial = [leg["serial_seconds_per_token"] for leg in good]
        entry["tags"] = [leg["tag"] for leg in good]
        entry["mtp_seconds_per_token"] = statistics.fmean(mtp)
        entry["serial_seconds_per_token"] = statistics.fmean(serial)
        entry["mtp_within_arm_spread_percent"] = (
            (max(mtp) - min(mtp)) / statistics.fmean(mtp) * 100.0
            if len(mtp) > 1 else None
        )
        entry["serial_within_arm_spread_percent"] = (
            (max(serial) - min(serial)) / statistics.fmean(serial) * 100.0
            if len(serial) > 1 else None
        )
        entry["local_ratio"] = statistics.fmean(
            [leg["mtp_decode_speedup"] for leg in good])
        entry["rounds"] = statistics.fmean([leg["rounds"] for leg in good])
        entry["effective_mean_draft_len"] = statistics.fmean(
            [leg["effective_mean_draft_len"] for leg in good])
        entry["accepted_rows"] = statistics.fmean(
            [leg["accepted_rows"] for leg in good])
        entry["rejected_rows"] = statistics.fmean(
            [leg["rejected_rows"] for leg in good])
        entry["all_tokens_matched"] = all(leg["all_tokens_matched"] for leg in good)
        entry["residual_divergence_count"] = sum(
            leg["residual_divergence_count"] for leg in good)
        entry["gpu_temp_entry_c"] = [leg["gpu_temp_entry_c"] for leg in good]
        entry["gpu_temp_exit_c"] = [leg["gpu_temp_exit_c"] for leg in good]
        entry["worker_env"] = good[0]["worker_env"]
        entry["worker_sha256"] = good[0]["worker_sha256"]
    return arms


def blocks(legs: dict, left: str, right: str, size: int = 4) -> dict:
    """Split the session into counterbalanced blocks and estimate the effect twice.

    Each block is its own palindrome, so monotone drift cancels inside it and the
    two block estimates are independent measurements of one quantity. Their
    disagreement is the null for the effect estimator itself, which is what a
    same-arm pair cannot give: a same-arm pair measures leg repeatability, not
    the repeatability of a two-arm contrast.
    """
    good = sorted(
        (leg for leg in legs.values() if leg.get("ok")), key=lambda leg: leg["order"]
    )
    estimates = []
    for start in range(0, len(good) - size + 1, size):
        chunk = good[start : start + size]
        arms_in_chunk = {leg["arm"] for leg in chunk}
        if arms_in_chunk != {left, right}:
            continue
        left_mean = statistics.fmean(
            leg["mtp_seconds_per_token"] for leg in chunk if leg["arm"] == left
        )
        right_mean = statistics.fmean(
            leg["mtp_seconds_per_token"] for leg in chunk if leg["arm"] == right
        )
        estimates.append(
            {
                "order": [leg["arm"] for leg in chunk],
                "tags": [leg["tag"] for leg in chunk],
                f"{left}_mtp_seconds_per_token": left_mean,
                f"{right}_mtp_seconds_per_token": right_mean,
                "delta_seconds_per_token": left_mean - right_mean,
                "percent": (left_mean - right_mean) / right_mean * 100.0,
            }
        )
    if len(estimates) < 2:
        return {"blocks": estimates, "resolved": False}

    percents = [estimate["percent"] for estimate in estimates]
    pooled = statistics.fmean(percents)
    disagreement = max(percents) - min(percents)
    # sd(single block) = disagreement / sqrt(2), so sd(pooled of two) =
    # disagreement / 2, and a two-sd bar on the pooled effect is exactly the
    # observed disagreement. One degree of freedom, so this is a coarse bar.
    return {
        "blocks": estimates,
        "resolved": True,
        "pooled_percent": pooled,
        "pooled_delta_seconds_per_token": statistics.fmean(
            estimate["delta_seconds_per_token"] for estimate in estimates
        ),
        "block_disagreement_percent": disagreement,
        "two_sd_bar_percent": disagreement,
        "significant_at_two_sd": abs(pooled) > disagreement,
        "degrees_of_freedom": len(estimates) - 1,
    }


def regression_null(legs: dict, left: str, right: str) -> dict:
    """Fit seconds/token on leg position and arm, then read the residual scatter.

    The two-block disagreement carries one degree of freedom, so it is a badly
    over-confident null. Fitting all legs at once spends two degrees of freedom
    on the mean and the drift slope and leaves the rest for the noise estimate.

    This is only valid because the arm order is balanced on position: the C legs
    and the B legs have equal position sums, so drift is orthogonal to the arm
    contrast and the simple difference of arm means is already drift-adjusted.
    The function checks that balance rather than assuming it.
    """
    good = sorted(
        (leg for leg in legs.values() if leg.get("ok")), key=lambda leg: leg["order"]
    )
    chosen = [leg for leg in good if leg["arm"] in (left, right)]
    if len(chosen) < 4:
        return {"resolved": False, "reason": "fewer than four usable legs"}

    positions = [leg["order"] for leg in chosen]
    values = [leg["mtp_seconds_per_token"] for leg in chosen]
    left_positions = [leg["order"] for leg in chosen if leg["arm"] == left]
    right_positions = [leg["order"] for leg in chosen if leg["arm"] == right]
    balanced = (
        len(left_positions) == len(right_positions)
        and sum(left_positions) == sum(right_positions)
    )

    mean_position = statistics.fmean(positions)
    mean_value = statistics.fmean(values)
    sxx = sum((p - mean_position) ** 2 for p in positions)
    sxy = sum(
        (p - mean_position) * (v - mean_value) for p, v in zip(positions, values)
    )
    slope = sxy / sxx
    left_mean = statistics.fmean(
        leg["mtp_seconds_per_token"] for leg in chosen if leg["arm"] == left
    )
    right_mean = statistics.fmean(
        leg["mtp_seconds_per_token"] for leg in chosen if leg["arm"] == right
    )
    effect = left_mean - right_mean

    residuals = []
    for leg, position, value in zip(chosen, positions, values):
        indicator = 0.5 if leg["arm"] == left else -0.5
        fitted = mean_value + slope * (position - mean_position) + effect * indicator
        residuals.append(value - fitted)
    dof = len(chosen) - 3
    residual_sd = (sum(r * r for r in residuals) / dof) ** 0.5
    standard_error = residual_sd * (1.0 / len(left_positions) + 1.0 / len(right_positions)) ** 0.5
    # Student t 97.5th percentile for the small dof this design can reach.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}.get(dof, 2.0)
    return {
        "resolved": True,
        "position_balanced": balanced,
        "left_position_sum": sum(left_positions),
        "right_position_sum": sum(right_positions),
        "drift_slope_seconds_per_token_per_leg": slope,
        "drift_slope_percent_per_leg": slope / mean_value * 100.0,
        "effect_seconds_per_token": effect,
        "effect_percent": effect / right_mean * 100.0,
        "residual_sd_seconds_per_token": residual_sd,
        "residual_sd_percent": residual_sd / mean_value * 100.0,
        "standard_error_percent": standard_error / right_mean * 100.0,
        "t_statistic": effect / standard_error,
        "degrees_of_freedom": dof,
        "significant_at_95_percent": abs(effect / standard_error) > critical,
        "confidence_interval_95_percent": [
            (effect - critical * standard_error) / right_mean * 100.0,
            (effect + critical * standard_error) / right_mean * 100.0,
        ],
    }


def contrasts(arms: dict) -> dict:
    out = {}
    for left, right in (("B", "A"), ("C", "A"), ("C", "B")):
        if left not in arms or right not in arms:
            continue
        if "mtp_seconds_per_token" not in arms[left]:
            continue
        if "mtp_seconds_per_token" not in arms[right]:
            continue
        base = arms[right]["mtp_seconds_per_token"]
        delta = arms[left]["mtp_seconds_per_token"] - base
        percent = delta / base * 100.0
        out[f"{left}-{right}"] = {
            "delta_seconds_per_token": delta,
            "percent": percent,
            "faster_arm": left if delta < 0 else right,
            "exceeds_local_null_floor": abs(percent) > LOCAL_NULL_FLOOR_PERCENT,
            "exceeds_ranked_single_pair_threshold": (
                abs(percent) > RANKED_SINGLE_PAIR_THRESHOLD_PERCENT
            ),
            "serial_percent": (
                arms[left]["serial_seconds_per_token"]
                - arms[right]["serial_seconds_per_token"]
            ) / arms[right]["serial_seconds_per_token"] * 100.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    legs = collect(args.tokens)
    arms = arm_summary(legs)
    temps = [
        leg["gpu_temp_entry_c"] for leg in legs.values()
        if leg.get("gpu_temp_entry_c") is not None
    ]
    report = {
        "harness": "local",
        "local_mode": "--local-iterate",
        "token_window": args.tokens,
        "mlx_max_mb_per_buffer": 512,
        "mlx_max_ops_per_buffer": 50,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "local_null_floor_percent": LOCAL_NULL_FLOOR_PERCENT,
        "ranked_single_pair_threshold_percent": RANKED_SINGLE_PAIR_THRESHOLD_PERCENT,
        "entry_temperature_spread_c": (max(temps) - min(temps)) if temps else None,
        "entry_temperatures_c": temps,
        "legs": legs,
        "arms": arms,
        "contrasts": contrasts(arms),
        "counterbalanced_blocks": blocks(legs, "C", "B"),
        "regression_null": regression_null(legs, "C", "B"),
    }
    text = json.dumps(report, indent=2, default=str)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
