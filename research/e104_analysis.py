#!/usr/bin/env python3
"""E104: turn the four-arm rate probe into rate(NA) curves and overhead factors.

    python3 research/e104_analysis.py research/out/e104-r1-sweep/rate.json \
        --census research/out/e104-r1-sweep/census.json

The rate identity in the brief is `round_us = G * bytes / rate(partition)`. This
prices the second term per shape: absolute microseconds, achieved GB/s, achieved
TFLOP/s and the overhead factor against `max(DRAM floor, FMA floor)`, for every
arm at every NA, plus the load-count model the rung 0 census predicts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

DRAM_PEAK_GBS = 273.0
# E97 batched affine-4 g64 peak, less the measured fixed 12.2 % dequant haircut.
FMA_PEAK_TFLOPS = 6.568 * (1.0 - 0.122)

ARMS = ("a_base", "l_loadonly", "z_noxload", "xw_widex")
# Device loads per k-block from the rung 0 census: 24 fixed + a per-NA term.
LOADS_PER_KBLOCK = {"a_base": lambda na: 24 + 16 * na,
                    "l_loadonly": lambda na: 24 + 16 * na,
                    "z_noxload": lambda na: 80,
                    "xw_widex": lambda na: 24 + 2 * na}


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def collect(doc: dict, warmup_blocks: int = 1) -> dict:
    """Aggregate timing blocks per (shape, NA), discarding cold-start blocks.

    The ABCDDCBA palindrome cancels monotone drift but not the one-time
    cold-start spike, which always lands on whichever arm holds slot 0.
    """
    cells: dict[tuple[str, int], dict] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < warmup_blocks:
            continue
        key = (row["shape"], row["m"])
        cell = cells.setdefault(key, {"bytes": row["read_bytes"],
                                      "flops": row["flops"],
                                      "temp": row["gpu_temp_entry_c"],
                                      "seconds": {a: [] for a in ARMS}})
        for arm, sec in row["seconds"].items():
            cell["seconds"][arm].append(sec)
    return cells


def slot_bias(doc: dict, warmup_blocks: int) -> list[str]:
    """Report first-slot inflation, separately for dropped and kept blocks."""
    order = doc["order"]
    dropped: list[float] = []
    kept: list[float] = []
    for row in doc["measurements"]:
        if row["kind"] != "timing":
            continue
        slots = row["slots"]
        first, last = slots[0], slots[len(order) - 1]
        # Palindrome: slot 0 and the mirrored slot run the same arm.
        ratio = first / last if last > 0 else float("nan")
        (dropped if row["block"] < warmup_blocks else kept).append(ratio)
    out = []
    for label, vals in (("dropped", dropped), ("kept", kept)):
        if vals:
            out.append("  %s blocks: slot0/mirror ratio median=%.3f max=%.3f (n=%d)"
                       % (label, statistics.median(vals), max(vals), len(vals)))
    return out


def fidelity(doc: dict) -> tuple[list[str], list[str]]:
    failures, controls = [], []
    for row in doc["measurements"]:
        if row["kind"] == "fidelity":
            for arm in row["arms"]:
                if arm["exact_required"] and not arm["bit_identical"]:
                    failures.append(
                        "%s M=%d %s: %d/%d differ (first m=%d n=%d)"
                        % (row["shape"], row["m"], arm["arm"],
                           arm["differing"], arm["total"], arm["first_bad_m"],
                           arm["first_bad_n"]))
        elif row["kind"] == "positive_control":
            controls.append(
                "%s M=%d %s: %d/%d differ, detected=%s"
                % (row["shape"], row["m"], row["arm"], row["differing"],
                   row["total"], row["detected"]))
    return failures, controls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rate_json")
    ap.add_argument("--census")
    ap.add_argument("--warmup-blocks", type=int, default=1)
    args = ap.parse_args()

    doc = load(pathlib.Path(args.rate_json))
    cells = collect(doc, args.warmup_blocks)
    failures, controls = fidelity(doc)

    print("device: %s (%s)" % (doc["device"], doc["architecture"]))
    print("order: %s, weight streams per round: %d, blocks per cell: %d"
          % (doc["order"], doc["weight_streams"], doc["pairs"]))
    print("discarded first %d block(s) per cell as cold start"
          % args.warmup_blocks)
    for line in slot_bias(doc, args.warmup_blocks):
        print(line)
    print()

    print("=== exactness ===")
    for c in controls:
        print("  positive control  %s" % c)
    if failures:
        for f in failures:
            print("  *** EXACTNESS FAILURE  %s" % f)
    else:
        print("  every arm required to be exact is bit-identical to a_base "
              "at every cell")
    print()

    shapes = sorted({s for s, _ in cells})
    widths = sorted({m for _, m in cells})

    for shape in shapes:
        print("=== %s ===" % shape)
        print("%3s %-12s %10s %8s %8s %9s %9s %8s" % (
            "NA", "arm", "us", "GB/s", "TFLOP/s", "dram_floor", "fma_floor",
            "overhead"))
        base_rate: dict[int, float] = {}
        for m in widths:
            cell = cells.get((shape, m))
            if not cell:
                continue
            dram_floor_s = cell["bytes"] / (DRAM_PEAK_GBS * 1e9)
            fma_floor_s = cell["flops"] / (FMA_PEAK_TFLOPS * 1e12)
            floor_s = max(dram_floor_s, fma_floor_s)
            for arm in ARMS:
                samples = cell["seconds"][arm]
                if not samples:
                    continue
                sec = statistics.median(samples)
                gbs = cell["bytes"] / sec / 1e9
                tflops = cell["flops"] / sec / 1e12
                if arm == "a_base":
                    base_rate[m] = gbs
                print("%3d %-12s %10.1f %8.1f %8.3f %9.1f %9.1f %8.2fx" % (
                    m, arm, sec * 1e6, gbs, tflops, dram_floor_s * 1e6,
                    fma_floor_s * 1e6, sec / floor_s))
        print()

        # rate(NA) for the shipped kernel, against the rung 0 load-count model.
        if 2 in base_rate:
            print("  rate(NA) for a_base, normalised at NA=2, against the "
                  "rung 0 device-load count")
            print("  %3s %9s %9s %9s %9s" % (
                "NA", "GB/s", "measured", "loads", "1/loads"))
            l2 = LOADS_PER_KBLOCK["a_base"](2)
            for m in widths:
                if m not in base_rate:
                    continue
                loads = LOADS_PER_KBLOCK["a_base"](m)
                print("  %3d %9.1f %9.3f %9d %9.3f" % (
                    m, base_rate[m], base_rate[m] / base_rate[2], loads,
                    l2 / loads))
            print()

        # The candidate fix, per width.
        print("  xw_widex against a_base")
        print("  %3s %10s %10s %9s %9s" % (
            "NA", "base us", "xw us", "change", "speedup"))
        for m in widths:
            cell = cells.get((shape, m))
            if not cell:
                continue
            b = statistics.median(cell["seconds"]["a_base"])
            x = statistics.median(cell["seconds"]["xw_widex"])
            print("  %3d %10.1f %10.1f %+8.2f%% %8.3fx" % (
                m, b * 1e6, x * 1e6, 100.0 * (x - b) / b, b / x))
        print()

    # Thermal record: an ungated session only supports a relative claim.
    temps = [(r["shape"], r["m"], r["gpu_temp_entry_c"], r["gpu_temp_exit_c"])
             for r in doc["measurements"] if r["kind"] == "thermal"]
    if temps:
        entries = [t[2] for t in temps if t[2] == t[2]]
        exits = [t[3] for t in temps if t[3] == t[3]]
        print("=== thermal (ungated, ABCDDCBA counterbalanced) ===")
        print("  cells=%d  entry min/max=%.1f/%.1f C  exit min/max=%.1f/%.1f C"
              % (len(temps), min(entries), max(entries), min(exits),
                 max(exits)))
        print("  entry spread=%.1f C" % (max(entries) - min(entries)))
        print("  cool_gate_passed_real_gate=false")
        print("  gate_qualified_for_timing=false")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
