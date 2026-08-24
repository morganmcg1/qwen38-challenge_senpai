#!/usr/bin/env python3
"""Reduce the E170 row-blocking screen to a per-width verdict.

The screen times the seven scored affine-4/g64 projections at each width, for
2 and 4 output rows per simdgroup, ABBA inside every cell. This weights each
shape by the number of times one verify forward reaches it, so the sum is the
wide-QMV share of that forward, and converts the relative change into the
E169 in-situ `T(M)` the assignment's stop rule is written against.

    T(M) = fixed + sum over input groups of pass(NA)
    fixed   = 11.792 ms
    pass    = {2: 55.008, 3: 53.841, 4: 61.336, 5: 71.758} ms

Usage: research/e170_report.py SCREEN_JSON [--exactness EXACTNESS_JSON]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

E169_FIXED_MS = 11.792
E169_PASS_MS = {2: 55.008, 3: 53.841, 4: 61.336, 5: 71.758}
# `Qwen35CustomQMV.activeInputGroups`: M -> inputs per group.
INPUTS_PER_GROUP = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
MINIMUM_USEFUL_FRACTION_OF_T = 0.04


def group_widths(m: int) -> list[int]:
    ipg = INPUTS_PER_GROUP[m]
    widths, first = [], 0
    while first < m:
        widths.append(min(ipg, m - first))
        first += ipg
    return widths


def modelled_pass_ms(m: int) -> float:
    return sum(E169_PASS_MS[max(2, na)] for na in group_widths(m))


def modelled_t_ms(m: int) -> float:
    return E169_FIXED_MS + modelled_pass_ms(m)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("screen", type=Path)
    parser.add_argument("--exactness", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.screen.read_text())
    widths = payload["widths"]

    # per (width, rows) -> weighted milliseconds of wide QMV per verify forward
    weighted: dict[tuple[int, int], list[float]] = {}
    per_shape: dict[tuple[str, int], dict[int, float]] = {}
    for shape in payload["shapes"]:
        calls = shape["calls_per_verify"]
        blocks: dict[tuple[int, int], list[float]] = {}
        for row in shape["rows"]:
            blocks.setdefault((row["m"], row["rows_per_simd"]), []).append(
                row["seconds_per_call"]
            )
        for (m, rows), samples in blocks.items():
            contribution = statistics.median(samples) * calls * 1e3
            weighted.setdefault((m, rows), []).append(contribution)
            per_shape.setdefault((shape["name"], m), {})[rows] = statistics.median(
                samples
            ) * 1e6

    print(f"host device: {payload['device']['architecture']}")
    print(f"entry point: {payload['entry_point']}  arm: {payload['custom_qmv_arm']}")
    print(
        "gpu temp entry/exit C: "
        f"{payload.get('gpu_temp_entry_c')}/{payload.get('gpu_temp_exit_c')}"
    )
    print(
        f"reps={payload['reps']} inner={payload['inner_calls_per_timed_region']} "
        f"order={payload['block_order']}"
    )
    print()

    print("per-shape microseconds per call")
    header = f"{'shape':34s} {'M':>2s} {'rows4':>9s} {'rows2':>9s} {'delta%':>8s}"
    print(header)
    for (name, m), arms in sorted(per_shape.items()):
        if 4 in arms and 2 in arms:
            delta = 100.0 * (arms[2] - arms[4]) / arms[4]
            print(f"{name:34s} {m:2d} {arms[4]:9.1f} {arms[2]:9.1f} {delta:8.2f}")
    print()

    print("verify-forward wide-QMV total, and the implied in-situ T(M)")
    print(
        f"{'M':>2s} {'qmv4 ms':>9s} {'qmv2 ms':>9s} {'delta%':>8s} "
        f"{'abba spread% 4/2':>18s} {'T(M) ms':>9s} {'dT ms':>8s} {'dT%':>7s} "
        f"{'verdict':>9s}"
    )
    verdicts: dict[int, dict[str, float]] = {}
    for m in widths:
        if (m, 4) not in weighted or (m, 2) not in weighted:
            continue
        qmv4 = sum(weighted[(m, 4)])
        qmv2 = sum(weighted[(m, 2)])
        delta = 100.0 * (qmv2 - qmv4) / qmv4

        spreads = {}
        for rows in (4, 2):
            per_cell = []
            for shape in payload["shapes"]:
                samples = [
                    r["seconds_per_call"]
                    for r in shape["rows"]
                    if r["m"] == m and r["rows_per_simd"] == rows
                ]
                if len(samples) == 2 and max(samples) > 0:
                    per_cell.append(
                        100.0 * abs(samples[0] - samples[1]) / statistics.mean(samples)
                    )
            spreads[rows] = max(per_cell) if per_cell else float("nan")

        t_model = modelled_t_ms(m)
        d_t = (delta / 100.0) * modelled_pass_ms(m)
        d_t_pct = 100.0 * d_t / t_model
        verdict = "promote" if d_t_pct <= -100.0 * MINIMUM_USEFUL_FRACTION_OF_T else "stop"
        verdicts[m] = {
            "qmv_ms_rows4": qmv4,
            "qmv_ms_rows2": qmv2,
            "delta_pct": delta,
            "modelled_t_ms": t_model,
            "delta_t_ms": d_t,
            "delta_t_pct": d_t_pct,
            "abba_spread_pct_rows4": spreads[4],
            "abba_spread_pct_rows2": spreads[2],
            "verdict": verdict,
        }
        print(
            f"{m:2d} {qmv4:9.3f} {qmv2:9.3f} {delta:8.2f} "
            f"{spreads[4]:8.2f}/{spreads[2]:8.2f} "
            f"{t_model:9.3f} {d_t:8.3f} {d_t_pct:7.2f} {verdict:>9s}"
        )
    print()
    print(
        "stop rule: promote a width when the implied dT is at or below "
        f"-{100 * MINIMUM_USEFUL_FRACTION_OF_T:.0f} % of T(M)."
    )

    if args.exactness:
        gate = json.loads(args.exactness.read_text())
        print()
        print("exactness gate")
        print(f"  cells digested                     {len(gate['cells'])}")
        print(f"  rows2 vs rows4 mismatches          {gate['mismatches_rows2_vs_rows4']}")
        print(f"  rows4 vs MLX launcher mismatches   {gate['mismatches_rows4_vs_mlx']}")
        print(
            "  control: one-ulp activation moves the output   "
            f"{gate['control_one_ulp_activation_changes_output']}"
        )
        print(
            "  control: starved grid moves the output         "
            f"{gate['control_starved_grid_changes_output']}"
        )

    print()
    print(json.dumps({"widths": verdicts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
