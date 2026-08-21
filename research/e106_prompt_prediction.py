#!/usr/bin/env python3
"""E106 rung 0 -- turn the measured width ladder into an official-probe forecast.

    usage: research/e106_prompt_prediction.py

The E106 anomaly is gated by decode width, so its size on any official prompt
depends on that prompt's width histogram, not on its round count. Plutarch
draws 38 drafting rounds out of 487 at mean width 1.154, which puts 92 % of its
rounds at M=1 where the census measures no anomaly. This prices the plutarch
TARGET probe and the drafting DRAFT probe against the measured ladder.

Every value marked INFERRED is an interpolation, not a measurement.

A census leg is never a timing leg. Only Metal's GPU clock is valid here.
"""

from __future__ import annotations

# Measured per-dispatch excess against the same-width refitted law, and the
# dispatch count per round. Source: research/out/e106/trace-{d4,w1}.json.
LADDER = {
    1: {"gdn.out_proj": (-3.71, 48), "fa.o_proj": (-4.69, 16),
        "mlp.down": (-10.18, 64)},
    4: {"gdn.out_proj": (18.31, 48), "fa.o_proj": (2.51, 16),
        "mlp.down": (30.63, 64)},
    5: {"gdn.out_proj": (16.68, 48), "fa.o_proj": (2.44, 16),
        "mlp.down": (35.43, 64)},
}
# Census GPU-busy round total, same leg.
ROUND_US = {1: 67405.9, 4: 86090.7, 5: 102864.5}

PLUTARCH_ROUNDS = 487
PLUTARCH_DRAFTING_ROUNDS = 38
PLUTARCH_MEAN_WIDTH = 1.154

RES_TARGET_PCT = 0.0431   # plutarch replicate resolution, same measurement mode
RES_DRAFT_PCT = 0.1139    # five drafting prompts, same measurement mode


def anomaly_us(width: int) -> float:
    return sum(e * n for e, n in LADDER[width].values())


def main() -> None:
    print("measured width ladder (census GPU clock)")
    for width in sorted(LADDER):
        total = anomaly_us(width)
        pct = 100.0 * total / ROUND_US[width]
        print(f"  M={width}: {total:+9.1f} us/round = {pct:+6.2f} % of the "
              f"{ROUND_US[width]:.0f} us round")

    upper_m1 = -3.71 * 48 + -4.69 * 16 + 0.55 * 64
    print(f"  M=1 with the mlp.down upper reading (+0.55): {upper_m1:+.1f} "
          f"us/round = {100.0 * upper_m1 / ROUND_US[1]:+.2f} %")

    serial_rounds = PLUTARCH_ROUNDS - PLUTARCH_DRAFTING_ROUNDS
    width_sum = PLUTARCH_ROUNDS * PLUTARCH_MEAN_WIDTH
    drafting_width = (width_sum - serial_rounds) / PLUTARCH_DRAFTING_ROUNDS
    print(f"\nplutarch: {PLUTARCH_ROUNDS} rounds, "
          f"{PLUTARCH_DRAFTING_ROUNDS} drafting, mean width "
          f"{PLUTARCH_MEAN_WIDTH}")
    print(f"  -> {serial_rounds} rounds at M=1 and "
          f"{PLUTARCH_DRAFTING_ROUNDS} rounds at mean drafting width "
          f"{drafting_width:.2f}")

    m3_round = ROUND_US[1] + (ROUND_US[4] - ROUND_US[1]) * (3 - 1) / (4 - 1)
    print(f"  INFERRED M=3 round anchor, linear from M=1 to M=4: "
          f"{m3_round:.0f} us")
    total_us = serial_rounds * ROUND_US[1] + PLUTARCH_DRAFTING_ROUNDS * m3_round
    print(f"  plutarch total round time: {total_us / 1e6:.2f} s")

    print("\n  recoverable time on the plutarch TARGET probe")
    scenarios = [
        ("M=3 behaves like M=4 (upper bound)", anomaly_us(4)),
        ("M=3 sits halfway between M=1 and M=4", anomaly_us(4) / 2),
        ("M=3 behaves like M=1 (lower bound)", 0.0),
    ]
    for label, per_round in scenarios:
        recovered = PLUTARCH_DRAFTING_ROUNDS * per_round
        pct = 100.0 * recovered / total_us
        print(f"    {label:40s} {pct:6.3f} %  -> {pct / RES_TARGET_PCT:5.1f} "
              f"sigma")
    print(f"    {'advisor F2 prediction':40s} {1.674:6.3f} %  -> "
          f"{1.674 / RES_TARGET_PCT:5.1f} sigma")

    print("\n  the same anomaly on the DRAFT probe")
    for width in (4, 5):
        pct = 100.0 * anomaly_us(width) / ROUND_US[width]
        print(f"    drafting prompts at M={width:<26d} {pct:6.3f} %  -> "
              f"{pct / RES_DRAFT_PCT:5.1f} sigma")


if __name__ == "__main__":
    main()
