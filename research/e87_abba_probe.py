#!/usr/bin/env python3
"""E87 follow-up: position-balanced ABBA contrast of probe fraction 0.15 vs 0.25.

The probe fraction is a compile-time constant, so the two arms need two
binaries and cannot interleave inside one process. The design instead
counterbalances leg order A B B A so a monotone host drift cancels to first
order, and reads the depth-0 serial leg of each pair as a head-free host-state
control: the serial leg never touches the proposal head, so any movement in it
is drift, not mechanism.

    usage: python3 research/e87_abba_probe.py [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

LEGS = [
    ("e87r2sub-derived25", 0.25, 1),
    ("e87d15g-derived15", 0.15, 2),
    ("e87abba3-derived15", 0.15, 3),
    ("e87abba4-derived25", 0.25, 4),
]

GATE_FIELDS = [
    "all_tokens_matched",
    "residual_divergence_count",
    "public_drift_tripwire_passed",
    "decode_tokens",
    "uses_pinned_mtp_head",
    "head_provenance_sha256",
]


def read_leg(tag: str, probe: float, position: int) -> dict:
    score = json.loads((OUT / tag / "score.json").read_text())
    meta = dict(
        line.split("=", 1)
        for line in (OUT / tag / "meta.txt").read_text().splitlines()
        if "=" in line
    )
    m = score["metrics"]
    leg = {
        "tag": tag,
        "probe_fraction": probe,
        "position": position,
        "passed": score["passed"],
        "mtp_seconds_per_token": m["mtp_seconds_per_token"],
        "serial_seconds_per_token": m["serial_seconds_per_token"],
        "mtp_decode_speedup": m["mtp_decode_speedup"],
        "effective_mean_draft_len": m["effective_mean_draft_len"],
        "accepted_draft_rate": m["accepted_draft_rate"],
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
        "worker_sha256": meta["worker_sha256"],
        "base_sha": meta["base_sha"],
        "meta_probe_fraction": float(meta["e87_probe_fraction"]),
        "sandbox": meta["sandbox"],
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
    }
    for field in GATE_FIELDS:
        leg[field] = m.get(field)
    if leg["meta_probe_fraction"] != probe:
        raise SystemExit(
            f"{tag}: meta records probe {leg['meta_probe_fraction']}, expected {probe}"
        )
    return leg


def arm_stats(legs: list[dict], key: str) -> tuple[float, float]:
    values = [leg[key] for leg in legs]
    return st.mean(values), max(values) - min(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    legs = [read_leg(*spec) for spec in LEGS]
    a = [leg for leg in legs if leg["probe_fraction"] == 0.25]
    b = [leg for leg in legs if leg["probe_fraction"] == 0.15]

    pos_a = sum(leg["position"] for leg in a)
    pos_b = sum(leg["position"] for leg in b)

    mtp_a, range_a = arm_stats(a, "mtp_seconds_per_token")
    mtp_b, range_b = arm_stats(b, "mtp_seconds_per_token")
    ser_a, _ = arm_stats(a, "serial_seconds_per_token")
    ser_b, _ = arm_stats(b, "serial_seconds_per_token")

    mtp_pct = 100.0 * (mtp_b - mtp_a) / mtp_a
    ser_pct = 100.0 * (ser_b - ser_a) / ser_a

    # Range of two observations estimates sigma as range / sqrt(2).
    sd_a, sd_b = range_a / 2**0.5, range_b / 2**0.5
    pooled_sd = ((sd_a**2 + sd_b**2) / 2) ** 0.5
    se_diff = pooled_sd  # sqrt(1/2 + 1/2) * pooled_sd with n = 2 per arm

    result = {
        "experiment": "e87-derived15-vs-derived25-abba",
        "harness": "local",
        "design": "ABBA, one leg per cell, two binaries",
        "position_sum_p025": pos_a,
        "position_sum_p015": pos_b,
        "position_balanced": pos_a == pos_b,
        "mtp_mean_p025": mtp_a,
        "mtp_mean_p015": mtp_b,
        "mtp_delta_seconds": mtp_b - mtp_a,
        "mtp_delta_percent": mtp_pct,
        "serial_control_mean_p025": ser_a,
        "serial_control_mean_p015": ser_b,
        "serial_control_delta_percent": ser_pct,
        "drift_adjusted_percent": mtp_pct - ser_pct,
        "within_arm_range_p025_percent": 100.0 * range_a / mtp_a,
        "within_arm_range_p015_percent": 100.0 * range_b / mtp_b,
        "pooled_sd_seconds": pooled_sd,
        "se_difference_seconds": se_diff,
        "effect_over_se": abs(mtp_b - mtp_a) / se_diff,
        "draft_len_identical_all_legs": len({leg["effective_mean_draft_len"] for leg in legs}) == 1,
        "accept_rate_identical_all_legs": len({leg["accepted_draft_rate"] for leg in legs}) == 1,
        "all_legs_passed": all(leg["passed"] for leg in legs),
        "legs": legs,
    }

    for leg in legs:
        print(
            f"{leg['position']}  {leg['tag']:<22} p={leg['probe_fraction']:.2f}  "
            f"mtp={leg['mtp_seconds_per_token']:.9f}  "
            f"serial={leg['serial_seconds_per_token']:.9f}  "
            f"T={leg['gpu_temp_entry_c']:.2f}->{leg['gpu_temp_exit_c']:.2f}  "
            f"worker={leg['worker_sha256'][:8]}"
        )
    print()
    print(f"position sums     p0.25={pos_a}  p0.15={pos_b}  balanced={pos_a == pos_b}")
    print(f"mtp    p0.25={mtp_a:.9f}  p0.15={mtp_b:.9f}  delta={mtp_b - mtp_a:+.9f}  {mtp_pct:+.4f} %")
    print(f"serial p0.25={ser_a:.9f}  p0.15={ser_b:.9f}  delta={ser_b - ser_a:+.9f}  {ser_pct:+.4f} %")
    print(f"drift-adjusted mtp effect  {mtp_pct - ser_pct:+.4f} %")
    print(
        f"within-arm range  p0.25={100.0 * range_a / mtp_a:.4f} %  "
        f"p0.15={100.0 * range_b / mtp_b:.4f} %"
    )
    print(f"pooled sd={pooled_sd:.3e} s  se(diff)={se_diff:.3e} s  effect/se={result['effect_over_se']:.1f}")
    print(f"draft length identical across legs   {result['draft_len_identical_all_legs']}")
    print(f"accept rate identical across legs    {result['accept_rate_identical_all_legs']}")
    print(f"all legs passed                      {result['all_legs_passed']}")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
