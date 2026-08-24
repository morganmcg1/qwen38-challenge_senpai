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
    parser.add_argument(
        "--identity", type=Path,
        help="wrapper identity.txt; it holds the cool-gate verdict and the "
             "entry and exit GPU temperature that the in-process test cannot "
             "read for itself",
    )
    args = parser.parse_args()

    identity = {}
    if args.identity and args.identity.exists():
        for line in args.identity.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                identity[key.strip()] = value.strip()

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
        f"{identity.get('gpu_temp_c_before_screen', payload.get('gpu_temp_entry_c'))}"
        f"/{identity.get('gpu_temp_c_after_screen', payload.get('gpu_temp_exit_c'))}"
        f"  real cool gate: {identity.get('cool_gate_screen', 'unrecorded')}"
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

    # The measured per-leg noise floor, which replaces the retired 0.039 %
    # figure. Blocks 1 and 2 are the two `rows2` legs and are adjacent in time,
    # so their spread is short-timescale replicate noise with almost no drift
    # in it. Blocks 0 and 3 are the two `rows4` legs and sit at the two ends of
    # the session, so their spread is drift plus the same noise. ABBA cancels
    # linear drift, and block 0 carries the session warm-up, which biases the
    # `rows4` arm slow and therefore understates any `rows4` win.
    noise: dict[str, object] = {"per_cell": [], "by_width": {}}
    sign_rows2_slower = sign_total = 0
    for shape in payload["shapes"]:
        by_block = {(r["m"], r["block"]): r["seconds_per_call"]
                    for r in shape["rows"]}
        for m in widths:
            legs = [by_block.get((m, b)) for b in range(4)]
            if any(v is None for v in legs):
                continue
            a1, b1, b2, a2 = legs
            replicate = 100.0 * abs(b1 - b2) / statistics.mean((b1, b2))
            drift = 100.0 * (a2 - a1) / statistics.mean((a1, a2))
            effect = 100.0 * (statistics.mean((b1, b2))
                              - statistics.mean((a1, a2))) / statistics.mean((a1, a2))
            noise["per_cell"].append({
                "shape": shape["name"], "m": m,
                "replicate_spread_pct_rows2": replicate,
                "session_drift_pct_rows4": drift,
                "effect_pct_rows2_minus_rows4": effect,
            })
            sign_total += 1
            sign_rows2_slower += 1 if effect > 0 else 0

    print()
    print("measured noise floor and drift, from the timed legs themselves")
    print(
        f"{'M':>2s} {'replicate spread% (B1,B2)':>26s} "
        f"{'session drift% (A1->A2)':>24s} {'effect%':>9s} {'effect/noise':>13s}"
    )
    for m in widths:
        cells = [c for c in noise["per_cell"] if c["m"] == m]
        if not cells:
            continue
        rep_med = statistics.median(c["replicate_spread_pct_rows2"] for c in cells)
        rep_max = max(c["replicate_spread_pct_rows2"] for c in cells)
        drift_med = statistics.median(c["session_drift_pct_rows4"] for c in cells)
        eff = verdicts[m]["delta_pct"]
        noise["by_width"][str(m)] = {
            "replicate_spread_pct_median": rep_med,
            "replicate_spread_pct_max": rep_max,
            "session_drift_pct_median": drift_med,
            "effect_pct": eff,
            "effect_over_max_replicate_spread": eff / rep_max if rep_max else None,
        }
        print(
            f"{m:2d} {rep_med:12.3f} (max {rep_max:6.3f}) "
            f"{drift_med:24.3f} {eff:9.2f} {eff / rep_max:13.1f}"
        )
    noise["sign_test"] = {
        "cells": sign_total,
        "cells_where_rows2_is_slower": sign_rows2_slower,
        "two_sided_p_if_no_effect": 2.0 ** -(sign_total - 1)
        if sign_rows2_slower in (0, sign_total) else None,
    }
    print(
        f"sign test: rows2 is slower in {sign_rows2_slower}/{sign_total} "
        "shape-by-width cells"
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
    print(json.dumps(
        {"widths": verdicts, "noise": {"by_width": noise["by_width"],
                                       "sign_test": noise["sign_test"]}},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
