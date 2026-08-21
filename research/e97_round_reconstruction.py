#!/usr/bin/env python3
"""Reconstruct the round-level per-row marginal from the E97 isolated fit.

harness=local. CPU only, no GPU and no model.

E97 rung 2 measured the marginal cost of one verify row for a single isolated
`quantizedMM` at two output widths. This script extrapolates that fit over the
quantized-projection geometry of one decode round and compares the sum with the
round-level marginal `c` that E95 rung 2 and E92 fitted independently.

The comparison answers advisor question f2 reading 3: if the isolated sum is
close to the round-level marginal, the marginal row does not overlap other work
and the concurrent encoder cannot inflate the E97 rate. If the isolated sum is
much larger, the marginal row overlaps and its effective rate is not a rate.
"""

import json

HIDDEN = 5120
VOCAB = 248_320
ATTN_LAYERS = 16
GDN_LAYERS = 48
LAYERS = ATTN_LAYERS + GDN_LAYERS
ATTN_V_DIM = 6_144
MLP_INTERMEDIATE = 17_408

# (K, N, dispatches per round), from QMV_CLASSES in research/e95_verify_census.py
ROUND_GEOMETRY = [
    (HIDDEN, 14_336, ATTN_LAYERS, "full-attention fused QKV+gate"),
    (HIDDEN, 16_480, GDN_LAYERS, "GDN in_proj"),
    (HIDDEN, 34_816, LAYERS, "MLP gate_up fused"),
    (ATTN_V_DIM, HIDDEN, LAYERS, "out_proj"),
    (MLP_INTERMEDIATE, HIDDEN, LAYERS, "MLP down_proj"),
    (HIDDEN, VOCAB, 1, "lm_head"),
]

# E97 rung 2, G=2 band, us per row: slope = a_N + b_N * K
RUNG2 = {34_816: (6.43, 11.378e-3), 248_320: (32.94, 84.795e-3)}

# E95 rung 2 / E92 round-level marginal, us per row, harness=local
ROUND_LEVEL_C = 10_268.0


def calibrate():
    """Split a_N and b_N into a per-dispatch constant and an N-proportional rate."""
    (n1, (a1, b1)), (n2, (a2, b2)) = sorted(RUNG2.items())
    q_a = (a2 - a1) / (n2 - n1)
    p_a = a1 - q_a * n1
    q_b = (b2 / n2 + b1 / n1) / 2.0
    return p_a, q_a, q_b, {n: b / n for n, (_a, b) in RUNG2.items()}


def main():
    p_a, q_a, q_b, per_col = calibrate()
    rows = []
    total = 0.0
    macs = 0
    for k, n, count, label in ROUND_GEOMETRY:
        launch = p_a * count
        width = q_a * n * count
        reduce_us = q_b * k * n * count
        each = p_a + q_a * n + q_b * k * n
        total += launch + width + reduce_us
        macs += k * n * count
        rows.append({
            "class": label, "k": k, "n": n, "dispatches": count,
            "us_per_row_each": each,
            "us_per_row_launch": launch,
            "us_per_row_width": width,
            "us_per_row_reduce": reduce_us,
            "us_per_row_total": launch + width + reduce_us,
            "extrapolated_below_calibration": n < 34_816,
        })

    reduce_total = sum(r["us_per_row_reduce"] for r in rows)
    launch_total = sum(r["us_per_row_launch"] for r in rows)
    width_total = sum(r["us_per_row_width"] for r in rows)
    out = {
        "harness": "local",
        "experiment": "e97",
        "gpu_used": False,
        "calibration": {
            "per_dispatch_constant_us_per_row": p_a,
            "per_output_column_us_per_row": q_a,
            "per_mac_us_per_row": q_b,
            "per_mac_us_per_row_by_width": per_col,
        },
        "classes": rows,
        "isolated_sum_us_per_row": total,
        "isolated_reduce_us_per_row": reduce_total,
        "isolated_width_us_per_row": width_total,
        "isolated_launch_us_per_row": launch_total,
        "round_level_c_us_per_row": ROUND_LEVEL_C,
        "isolated_over_round_level": total / ROUND_LEVEL_C,
        "trunk_macs_per_row": macs,
        "implied_round_level_tflops": 2 * macs / (ROUND_LEVEL_C * 1e-6) / 1e12,
        "implied_isolated_tflops": 2 * macs / (total * 1e-6) / 1e12,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
