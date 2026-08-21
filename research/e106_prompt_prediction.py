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
# dispatch count per round. Source: research/out/e106/trace-{d4,w1,w2,w3}.json
# from legs e106r0b-d4-trace (M=1, 4, 5) and e106r0c-d2-w3 (M=2, 3).
LADDER = {
    1: {"gdn.out_proj": (-3.71, 48), "fa.o_proj": (-4.69, 16),
        "mlp.down": (-10.18, 64)},
    2: {"gdn.out_proj": (6.41, 48), "fa.o_proj": (3.13, 16),
        "mlp.down": (33.46, 64)},
    3: {"gdn.out_proj": (13.57, 48), "fa.o_proj": (3.42, 16),
        "mlp.down": (30.96, 64)},
    4: {"gdn.out_proj": (18.31, 48), "fa.o_proj": (2.51, 16),
        "mlp.down": (30.63, 64)},
    5: {"gdn.out_proj": (16.68, 48), "fa.o_proj": (2.44, 16),
        "mlp.down": (35.43, 64)},
}
# Traced rounds behind each ladder point. M=2 and M=4 are single rounds, so
# their clean-family residuals are the honest error bar on that column.
LADDER_ROUNDS = {1: 64, 2: 1, 3: 21, 4: 1, 5: 12}
# Census GPU-busy round total, same leg.
ROUND_US = {1: 67405.9, 2: 71831.0, 3: 75163.1, 4: 86090.7, 5: 102864.5}

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
              f"{ROUND_US[width]:>6.0f} us round   "
              f"({LADDER_ROUNDS[width]} traced rounds)")

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

    total_us = (serial_rounds * ROUND_US[1]
                + PLUTARCH_DRAFTING_ROUNDS * ROUND_US[3])
    print(f"  M=3 round anchor: {ROUND_US[3]:.0f} us (measured, 21 rounds)")
    print(f"  plutarch total round time: {total_us / 1e6:.2f} s")

    print("\n  recoverable time on the plutarch TARGET probe")
    # The M=1 rounds sit below the same-width law, so a fix leaves them alone
    # and recovers nothing there. Only the drafting rounds carry the anomaly.
    scenarios = [
        ("measured, drafting rounds at M=3", anomaly_us(3)),
        ("advisor F2 prediction", None),
    ]
    for label, per_round in scenarios:
        if per_round is None:
            pct = 1.674
        else:
            pct = 100.0 * PLUTARCH_DRAFTING_ROUNDS * per_round / total_us
        print(f"    {label:40s} {pct:6.3f} %  -> {pct / RES_TARGET_PCT:5.1f} "
              f"sigma")

    print("\n  the same anomaly on the DRAFT probe")
    for width in (3, 4, 5):
        pct = 100.0 * anomaly_us(width) / ROUND_US[width]
        print(f"    drafting prompts at M={width:<26d} {pct:6.3f} %  -> "
              f"{pct / RES_DRAFT_PCT:5.1f} sigma")


if __name__ == "__main__":
    main()
