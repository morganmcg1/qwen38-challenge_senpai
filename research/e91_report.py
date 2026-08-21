#!/usr/bin/env python3
"""Summarise one E91 session.

    usage: research/e91_report.py research/out/TAG [research/out/OTHER_TAG ...]

Reads `ladder.json` and `ceiling.json` from every named directory and prints
the tables the E91 result needs: the ABBA stride effects with their own null,
the untimed kernel and dispatch census, and the quantized GEMM ceiling weighted
by each cell's measured share of the seed leg.
"""

import json
import os
import statistics
import sys

# `begin()` is this fraction of the ranked candidate leg on beagle, the binding
# prompt, and 0 % of the ranked serial numerator.
PREFILL_SHARE_OF_LEG = 0.0859

# Cells that do not saturate the GPU. Thorfinn's E83 rule: an isolated-cell
# roofline over-states recoverable time whenever the cell does not saturate the
# GPU, so these two are reported separately and never pooled into a headroom
# claim.
LATENCY_BOUND = {"gdn.in_proj_a", "gdn.in_proj_b"}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def pct(x):
    return f"{100 * x:+.3f} %"


def report_ladder(doc):
    blocks = doc["blocks"]
    begins = [b for b in blocks if b.get("kind") == "begin"]
    census = [b for b in blocks if b.get("kind") == "boundary_census"]

    print("## identity")
    for key, value in sorted(doc["identity"].items()):
        print(f"  {key} = {value}")
    print(f"  harness = {doc['harness']}")
    print(f"  cool_gate_passed_real_gate = {doc['cool_gate_passed_real_gate']}")
    print(f"  gate_qualified_for_timing = {doc['gate_qualified_for_timing']}")

    prints = doc.get("token_fingerprints", {})
    distinct = sorted({p for values in prints.values() for p in values})
    print(f"\n## bit-exactness: {len(distinct)} distinct tail-row fingerprint(s)")
    for value in distinct:
        holders = sorted(k for k, v in prints.items() if value in v)
        print(f"  {value}   arms: {','.join(holders)}")

    for c in census:
        totals = c.get("totals", {})
        label = c.get("ladder_label", "?")
        print(
            f"\n## untimed boundary census, one begin(), ladder={label}"
            f" rungs={c.get('forced_eval_points')}"
            f"  dispatches={totals.get('dispatches')}"
            f"  command_buffer_commits={totals.get('command_buffer_commits')}")
        kernels = totals.get("kernels", {})
        for name, count in sorted(kernels.items(), key=lambda kv: -kv[1]):
            print(f"  {count:6d}  {name}")
        shapes = totals.get("kernel_shapes", {})
        quant = {k: v for k, v in shapes.items() if k.startswith("affine_")}
        if quant:
            print("  quantized GEMM launches by grid and threadgroup:")
            for key, count in sorted(quant.items(), key=lambda kv: -kv[1]):
                name, grid, tg = key.split("|")
                print(f"    {count:5d}  {grid:24s} {tg:14s} {name}")
        print("  per phase (cumulative):")
        for phase, snap in c.get("per_phase", {}).items():
            print(
                f"    {phase:26s} dispatches={snap.get('dispatches'):6d} "
                f"commits={snap.get('command_buffer_commits'):5d}")

    print("\n## absolute begin() wall time by arm (all blocks pooled)")
    print(f"{'arm':10s} {'rungs':>6s} {'n':>3s} {'median ms':>10s} {'min':>9s} "
          f"{'max':>9s} {'cpu ms':>8s} {'entry C':>8s}")
    by_arm = {}
    for b in begins:
        by_arm.setdefault(b["ladder_label"], []).append(b)
    for arm, rows in sorted(by_arm.items(), key=lambda kv: -len(kv[1])):
        ms = sorted(1e3 * r["begin_seconds"] for r in rows)
        cpu = [r.get("host_thread_cpu_ns", 0) / 1e6 for r in rows]
        temps = [r["gpu_temp_entry_c"] for r in rows if r.get("gpu_temp_entry_c")]
        print(f"{arm:10s} {rows[0]['forced_eval_points']:6d} {len(ms):3d} "
              f"{statistics.median(ms):10.2f} {ms[0]:9.2f} {ms[-1]:9.2f} "
              f"{statistics.median(cpu):8.1f} "
              f"{statistics.median(temps) if temps else float('nan'):8.1f}")

    print("\n## ABBA effect against ship, per quad: mean(B,B) - mean(A,A)")
    print(f"{'arm':10s} {'quads':>6s} {'mean ms':>9s} {'sd ms':>8s} "
          f"{'mean %':>9s} {'ship ms':>9s}")
    quads = {}
    for b in begins:
        key = (b.get("pair_arm"), b.get("rep"))
        if key[0] is None:
            continue
        quads.setdefault(key, []).append(b)
    effects = {}
    ship_reference = []
    for (arm, _rep), rows in sorted(quads.items()):
        rows.sort(key=lambda r: r["pair_position"])
        if len(rows) != 4:
            continue
        outer = [1e3 * rows[0]["begin_seconds"], 1e3 * rows[3]["begin_seconds"]]
        inner = [1e3 * rows[1]["begin_seconds"], 1e3 * rows[2]["begin_seconds"]]
        effects.setdefault(arm, []).append(
            (statistics.mean(inner) - statistics.mean(outer),
             statistics.mean(outer)))
        ship_reference.extend(outer)
    null_sd = None
    for arm, values in sorted(effects.items(), key=lambda kv: statistics.mean(
            v[0] for v in kv[1])):
        deltas = [v[0] for v in values]
        ships = [v[1] for v in values]
        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas) if len(deltas) > 1 else float("nan")
        rel = mean / statistics.mean(ships)
        if arm == "ship_null":
            null_sd = (mean, sd, rel)
        print(f"{arm:10s} {len(deltas):6d} {mean:9.2f} {sd:8.2f} "
              f"{100 * rel:+9.3f} {statistics.mean(ships):9.2f}")
    if null_sd:
        print(f"\n  ship-against-ship null: {null_sd[0]:+.2f} ms "
              f"({100 * null_sd[2]:+.3f} %), sd {null_sd[1]:.2f} ms")
    if ship_reference:
        print(f"  pooled ship reference: median {statistics.median(ship_reference):.2f} ms, "
              f"n={len(ship_reference)}")


def tiled_bandwidth(cell):
    """Bytes the tiling asks the memory system for, over the measured time.

    `qmm_t` tiles at `bm = bn = 32` with no cross-threadgroup cooperation, so
    with zero cache reuse each weight element is fetched once per M-tile and
    each activation element once per N-tile. That is an upper bound: anything
    the system cache absorbs never reaches DRAM. It is reported because the
    unique-bytes figure is a lower bound and the two differ by more than 16
    times, so neither alone decides whether a cell is bandwidth bound.
    """
    bm = bn = 32
    m_tiles = (cell["m"] + bm - 1) // bm
    n_tiles = (cell["n"] + bn - 1) // bn
    weight_bytes = m_tiles * cell["weight_bytes_per_call"]
    activation_bytes = n_tiles * cell["m"] * cell["k"] * 2
    return (weight_bytes + activation_bytes) / cell["seconds_median"] / 1e9


def report_ceiling(doc):
    blocks = doc["blocks"]
    peaks = next((b for b in blocks if b.get("kind") == "e91_machine_peaks"), None)
    cells = [b for b in blocks if b.get("kind") == "e91_ceiling_cell"]
    if peaks:
        print("\n## measured machine limits")
        print(f"  streaming read      {peaks['read_gb_per_second']:8.1f} GB/s")
        print(f"  streaming copy      {peaks['copy_gb_per_second']:8.1f} GB/s")
        print(f"  bf16 GEMM 4096^3    {peaks['bf16_tflop_per_second']:8.3f} TFLOP/s")

    print("\n## per-cell ceiling at M = 512")
    print(f"{'family':26s} {'N':>6s} {'L':>3s} {'ship ms':>8s} {'TF':>5s} "
          f"{'uniq':>6s} {'tiled':>7s} {'bf16 ms':>8s} {'TF':>5s} {'share':>7s} "
          f"{'gap %':>7s} {'gap ms':>8s}")
    total = sum(c["modelled_prefill_seconds"] for c in cells)
    saturating_gap = 0.0
    for c in sorted(cells, key=lambda c: -c["modelled_prefill_seconds"]):
        gap_ms = 1e3 * (c["seconds_median"] - c["dense_bf16_seconds_median"]) * c["layers"]
        gap_frac = 1 - c["dense_bf16_seconds_median"] / c["seconds_median"]
        flag = "  (latency-bound)" if c["family"] in LATENCY_BOUND else ""
        if c["family"] not in LATENCY_BOUND:
            saturating_gap += gap_ms
        print(f"{c['family']:26s} {c['n']:6d} {c['layers']:3d} "
              f"{1e3 * c['seconds_median']:8.3f} {c['tflop_per_second']:5.2f} "
              f"{c['gb_per_second']:6.1f} {tiled_bandwidth(c):7.1f} "
              f"{1e3 * c['dense_bf16_seconds_median']:8.3f} "
              f"{c['dense_bf16_tflop_per_second']:5.2f} "
              f"{100 * c['modelled_prefill_seconds'] / total:6.2f}% "
              f"{100 * gap_frac:+7.2f} {gap_ms:8.1f}{flag}")

    print(f"\n  modelled prefill total          {1e3 * total:10.1f} ms")
    print(f"  gap to dense bf16, saturating   {saturating_gap:10.1f} ms "
          f"({100 * saturating_gap / 1e3 / total:.2f} % of prefill)")
    print(f"  implied candidate-leg gain      "
          f"{100 * PREFILL_SHARE_OF_LEG * saturating_gap / 1e3 / total:10.3f} % "
          f"(upper bound, unreachable: dense bf16 is not a legal candidate)")


def main() -> int:
    for directory in sys.argv[1:]:
        print(f"\n{'=' * 78}\n{directory}\n{'=' * 78}")
        ladder = load(os.path.join(directory, "ladder.json"))
        if ladder:
            report_ladder(ladder)
        ceiling = load(os.path.join(directory, "ceiling.json"))
        if ceiling:
            report_ceiling(ceiling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
