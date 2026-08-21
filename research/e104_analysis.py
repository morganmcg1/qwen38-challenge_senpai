#!/usr/bin/env python3
"""E104: turn the rate probe into rate(NA) curves, overhead factors and A(NA).

    python3 research/e104_analysis.py research/out/e104-r1-sweep/rate.json
    python3 research/e104_analysis.py research/out/e104-r05-ladder/rate.json \
        --partitions research/out/e104-r05-ladder/census.json

The rate identity in the brief is `round_us = G * bytes / rate(partition)`. This
prices the second term per shape: absolute microseconds, achieved GB/s, achieved
TFLOP/s and the overhead factor against `max(DRAM floor, FMA floor)`, for every
arm at every NA, plus the load-count model the rung 0 census predicts.

With `--partitions` it also answers rung 0.5. `x_onegroup` reads the weights
once and `y_split` reads them `G` times, so

    r1 = bytes / t1        r2 = (G * weights + activations) / t2
    A  = r2 / r1           collapse gain = 1 - A / 2

and a collapse at that width only survives the local-to-ranked transfer when
`A_local` is below `RANKED_NEUTRAL_A`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

DRAM_PEAK_GBS = 273.0
# E97 batched affine-4 g64 peak, less the measured fixed 12.2 % dequant haircut.
FMA_PEAK_TFLOPS = 6.568 * (1.0 - 0.122)

# The advisor's break-even law. A collapse is ranked-neutral at
# `A_ranked = 2`, and `A_ranked = A_local * 1.244` from the two ranked routes,
# so the local decision boundary is 2 / 1.244.
LOCAL_TO_RANKED_A = 1.244
RANKED_NEUTRAL_A = 2.0 / LOCAL_TO_RANKED_A

# Device loads per k-block from the rung 0 census: 24 fixed + a per-NA term.
LOADS_PER_KBLOCK = {"a_base": lambda na: 24 + 16 * na,
                    "l_loadonly": lambda na: 24 + 16 * na,
                    "z_noxload": lambda na: 80,
                    "xw_widex": lambda na: 24 + 2 * na}


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def collect(doc: dict, arms: tuple[str, ...], warmup_blocks: int = 1) -> dict:
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
                                      "seconds": {a: [] for a in arms}})
        for arm, sec in row["seconds"].items():
            cell["seconds"][arm].append(sec)
    return cells


def slot_bias(doc: dict, warmup_blocks: int) -> list[str]:
    """Report first-slot inflation, separately for dropped and kept blocks."""
    dropped: list[float] = []
    kept: list[float] = []
    for row in doc["measurements"]:
        if row["kind"] != "timing":
            continue
        slots = row["slots"]
        first, last = slots[0], slots[-1]
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


def ladder(cells: dict, shapes: list[str], widths: list[int],
           table: dict, one: str, two: str) -> None:
    """Rung 0.5: the isolated one-group rate against the split it replaces."""
    print("=== rung 0.5: A(NA) = r2 / r1 against the ranked-neutral %.4f ==="
          % RANKED_NEUTRAL_A)
    print("  r1 reads the weights once, r2 reads them G times. A collapse at "
          "this width")
    print("  is predicted to win on the ranked host only when A_local < %.4f."
          % RANKED_NEUTRAL_A)
    per_width: dict[int, list[float]] = {m: [] for m in widths}
    for shape in shapes:
        print("\n  %s" % shape)
        print("  %3s %-7s %9s %9s %9s %9s %7s %9s %9s" % (
            "NA", "split", "t1 us", "t2 us", "r1 GB/s", "r2 GB/s", "A",
            "r1 needed", "gain %"))
        for m in widths:
            cell = cells.get((shape, m))
            if not cell or not cell["seconds"][one] or not cell["seconds"][two]:
                continue
            spec = table[two][str(m)]
            streams = spec["weight_streams"]
            t1 = statistics.median(cell["seconds"][one])
            t2 = statistics.median(cell["seconds"][two])
            # `read_bytes` prices one weight stream plus the activations, which
            # every partition reads exactly once. One stream is n * k nibbles
            # plus a bf16 scale and bias per group of 64, and `flops` fixes
            # n * k without parsing the shape name.
            weights = (cell["flops"] / (2.0 * m)) * (0.5 + 4.0 / 64.0)
            activations = cell["bytes"] - weights
            r1 = cell["bytes"] / t1 / 1e9
            r2 = (streams * weights + activations) / t2 / 1e9
            a_local = r2 / r1
            per_width[m].append(a_local)
            print("  %3d %-7s %9.1f %9.1f %9.1f %9.1f %7.3f %9.1f %+8.1f%%" % (
                m, spec["partition"], t1 * 1e6, t2 * 1e6, r1, r2, a_local,
                r2 / RANKED_NEUTRAL_A, 100.0 * (1.0 - a_local / 2.0)))
    print("\n  A_local across shapes, and the ranked projection "
          "A_ranked = A_local * %.3f" % LOCAL_TO_RANKED_A)
    print("  %3s %9s %9s %9s %11s %s" % (
        "NA", "median", "min", "max", "A_ranked", "verdict"))
    for m in widths:
        vals = per_width[m]
        if not vals:
            continue
        med = statistics.median(vals)
        ranked = med * LOCAL_TO_RANKED_A
        if abs(med - 1.0) < 0.02 and len(vals) > 1:
            verdict = "null control"
        elif med < RANKED_NEUTRAL_A:
            verdict = "collapse predicted to win"
        else:
            verdict = "collapse predicted to lose"
        print("  %3d %9.3f %9.3f %9.3f %11.3f %s" % (
            m, med, min(vals), max(vals), ranked, verdict))
    print()


def registers(doc: dict, widths: list[int], arms: tuple[str, ...]) -> None:
    """The register and spill tax E77 charges against any collapse gain.

    The two hosts have different register budgets, so a width that spills
    locally can still fit on the ranked host. Any local timing at such a width
    understates the ranked one-group rate and must be labelled, not projected.
    """
    rows = {row["arm"]: row for row in doc.get("arms", []) if row.get("compiled")}
    if not rows:
        return
    print("=== register and spill per width, both architectures ===")
    arches = [k for k in next(iter(rows.values())) if k.startswith("applegpu_")]
    for arch in arches:
        print("  %s" % arch)
        print("  %3s %10s %10s %10s %10s" % (
            "NA", arms[0], "spill B", arms[1], "spill B"))
        for m in widths:
            cells = [rows[a][arch].get(str(m)) for a in arms if a in rows]
            if any(c is None for c in cells) or len(cells) < 2:
                continue
            print("  %3d %10d %10d %10d %10d" % (
                m, cells[0]["registers"], cells[0]["spill_bytes"] or 0,
                cells[1]["registers"], cells[1]["spill_bytes"] or 0))
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rate_json")
    ap.add_argument("--census")
    ap.add_argument("--partitions",
                    help="census.json or partitions.json from the ladder set")
    ap.add_argument("--warmup-blocks", type=int, default=1)
    args = ap.parse_args()

    doc = load(pathlib.Path(args.rate_json))
    arms = tuple(doc["arms"])
    cells = collect(doc, arms, args.warmup_blocks)
    failures, controls = fidelity(doc)

    print("device: %s (%s)" % (doc["device"], doc["architecture"]))
    print("arms: %s, order: %s, blocks per cell: %d"
          % (", ".join(arms), doc["order"], doc["pairs"]))
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
        print("  every arm required to be exact is bit-identical to %s "
              "at every cell" % arms[0])
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
            for arm in arms:
                samples = cell["seconds"][arm]
                if not samples:
                    continue
                sec = statistics.median(samples)
                gbs = cell["bytes"] / sec / 1e9
                tflops = cell["flops"] / sec / 1e12
                if arm == arms[0]:
                    base_rate[m] = gbs
                print("%3d %-12s %10.1f %8.1f %8.3f %9.1f %9.1f %8.2fx" % (
                    m, arm, sec * 1e6, gbs, tflops, dram_floor_s * 1e6,
                    fma_floor_s * 1e6, sec / floor_s))
        print()

        # rate(NA) for the base arm, normalised at the narrowest width. With
        # the rate arms this also reads against the rung 0 device-load count.
        if base_rate:
            narrow = min(base_rate)
            model = LOADS_PER_KBLOCK.get(arms[0])
            print("  rate(NA) for %s, normalised at NA=%d%s"
                  % (arms[0], narrow,
                     ", against the rung 0 device-load count" if model else ""))
            print("  %3s %9s %9s%s" % (
                "NA", "GB/s", "measured", "     loads   1/loads" if model else ""))
            for m in widths:
                if m not in base_rate:
                    continue
                tail = ""
                if model:
                    tail = " %9d %9.3f" % (model(m), model(narrow) / model(m))
                print("  %3d %9.1f %9.3f%s" % (
                    m, base_rate[m], base_rate[m] / base_rate[narrow], tail))
            print()

        # Every other arm against the base arm, per width.
        for arm in arms[1:]:
            print("  %s against %s" % (arm, arms[0]))
            print("  %3s %10s %10s %9s %9s" % (
                "NA", "base us", "arm us", "change", "speedup"))
            for m in widths:
                cell = cells.get((shape, m))
                if not cell or not cell["seconds"][arm]:
                    continue
                b = statistics.median(cell["seconds"][arms[0]])
                x = statistics.median(cell["seconds"][arm])
                print("  %3d %10.1f %10.1f %+8.2f%% %8.3fx" % (
                    m, b * 1e6, x * 1e6, 100.0 * (x - b) / b, b / x))
            print()

    if args.partitions:
        doc_p = load(pathlib.Path(args.partitions))
        table = doc_p.get("partitions", doc_p)
        ladder(cells, shapes, widths, table, arms[0], arms[1])
        registers(doc_p, widths, arms)

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
