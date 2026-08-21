#!/usr/bin/env python3
"""Readable view of an `e85_round_pairs.py` report.

    usage: research/e85_pairs_print.py research/e85-artifacts/round-pairs.json
"""
from __future__ import annotations

import json
import sys

BLOCKS = [
    ("paired_effect_us_per_round", "paired effect, round bootstrap"),
    ("session_null_us_per_round", "drift-matched null, round bootstrap"),
    ("end_base_drift_us_per_round", "end base drift (NOT drift-matched)"),
    ("paired_effect_clustered", "paired effect, BLOCK bootstrap"),
    ("session_null_clustered", "drift-matched null, BLOCK bootstrap"),
]


def main() -> None:
    report = json.load(open(sys.argv[1]))
    tpr = report["tokens_per_round"]
    base_us = report["leg_total_baseline_s_per_token"] * 1e6

    print(f"legs {report['legs']}  distinct round sequences "
          f"{report['distinct_round_sequences']}  rounds/leg "
          f"{report['sequence_groups'][0]['rounds']}")
    cov = report["trace_covers_leg"]
    print(f"trace coverage: traced {cov['traced_round_seconds_mean']:.3f}s of "
          f"harness {cov['harness_leg_seconds_mean']:.3f}s per leg, "
          f"pearson r={cov['pearson_r']:.4f}")
    print()

    print(f"{'statistic':38s} {'us/token':>10s} {'95% CI':>24s} {'% cand':>9s}")
    for key, label in BLOCKS:
        b = report[key]
        med = b["median_us_per_token"]
        lo, hi = b["ci95_lo_us_per_token"], b["ci95_hi_us_per_token"]
        print(f"{label:38s} {med:+10.2f} [{lo:+9.2f},{hi:+9.2f}] "
              f"{100.0 * med / base_us:+9.4f}")
    print()

    print("per ABBA block (independent replicates)")
    for entry in report["per_block"]:
        legs = entry["legs"]
        print(f"  legs {legs}  effect {entry['effect_median_us_per_round'] / tpr:+9.2f}"
              f"  null {entry['null_median_us_per_round'] / tpr:+9.2f}"
              f"  effect IQR {entry['effect_iqr_us_per_round'] / tpr:9.2f} us/token")
    clustered = report["paired_effect_clustered"]
    print(f"  between-block mean {clustered['cluster_median_mean_us_per_token']:+.2f}"
          f"  sd {clustered['cluster_median_sd_us_per_token']:.2f} us/token")
    print(f"  t interval on 3 block medians "
          f"[{clustered['t_ci95_lo_us_per_token']:+.2f}, "
          f"{clustered['t_ci95_hi_us_per_token']:+.2f}] us/token "
          f"([{100.0 * clustered['t_ci95_lo_us_per_token'] / base_us:+.4f}, "
          f"{100.0 * clustered['t_ci95_hi_us_per_token'] / base_us:+.4f}] %)")
    print()

    chk = report["cluster_variance_check"]
    print(f"cluster variance check: between-block sd "
          f"{chk['between_block_sd_us_per_round']:.1f} vs within-block "
          f"predicted sem {chk['within_block_predicted_sem_us_per_round']:.1f} "
          f"us/round, ratio {chk['ratio']:.1f}x")
    print(f"  rounds are independent replicates: "
          f"{chk['rounds_are_independent_replicates']}")
    print()

    print(f"leg-total effect        {report['leg_total_effect_us_per_token']:+.2f} us/token"
          f"  ({100.0 * report['leg_total_effect_us_per_token'] / base_us:+.4f} %)")
    print(f"leg-total / paired      {report['leg_total_over_paired_ratio']:.3f}")
    print(f"outside null, round     {report['effect_outside_null_round_level']}")
    print(f"outside null, CLUSTERED {report['effect_outside_null_clustered']}  <- decision")


if __name__ == "__main__":
    main()
