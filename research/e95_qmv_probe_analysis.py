#!/usr/bin/env python3
"""Read the E95 direct qmv width probe and decide what `b` is.

The E95 rung-2 verify width model is `verify_us = a + b*G + c*M`, with
`G = ceil(M / IPG)` the number of input groups the WIDE affine-4 kernel runs
over one weight tensor. The in-model fit gives `b = 27,377 us`. Read as one
pass over the 14,412 MB of affine-4 weights the verify phase touches, that
is 526.4 GB/s, about twice the DRAM read rate this chip reaches. A model
term cannot describe traffic the memory system cannot carry.

`Tests/MLXFastTests/E95QmvWidthProbeTests.swift` measures the same kernel
outside the model, the fixture and the worker. This script turns that
measurement into an answer for three questions:

  1. Does the G step from 1 to 2 cost a second pass over the same bytes?
  2. Do the isolated-kernel coefficients reproduce the in-model ones from
     bytes alone, which would prove the width model is a property of this
     one kernel?
  3. What fraction of the verify phase can a byte reduction reach?

Usage: python3 research/e95_qmv_probe_analysis.py [path]
"""

from __future__ import annotations

import json
import pathlib
import sys

MODEL_WIDTHS = (3, 4, 5, 6, 8, 9)

# E95 rung 2, in-model fit over six censused verify legs on this host.
IN_MODEL = {"a": 10_920.0, "b": 27_377.0, "c": 10_268.0}
VERIFY_STREAM_BYTES = 14_412_349_440
# Ranked mean draft widths and the phase totals the in-model fit gives there.
RANKED = {
    "beagle": {"mean_m": 5.38, "groups": 2},
    "essays": {"mean_m": 6.09, "groups": 2},
}


def inputs_per_group(width: int) -> int:
    """The shipped rule in `kernels/quantized.h`: IPG = ceil(M / ceil(M/4))."""
    return -(-width // -(-width // 4))


def groups(width: int) -> int:
    return -(-width // inputs_per_group(width))


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    aug = [matrix[i][:] + [rhs[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            aug[row] = [
                aug[row][c] - factor * aug[col][c] for c in range(size + 1)]
    return [aug[i][size] / aug[i][i] for i in range(size)]


def fit_width_model(samples: list[tuple[int, float]]) -> tuple[float, float, float]:
    design = [[1.0, float(groups(m)), float(m)] for m, _ in samples]
    target = [value for _, value in samples]
    normal = [
        [sum(row[i] * row[j] for row in design) for j in range(3)]
        for i in range(3)
    ]
    moment = [
        sum(design[k][i] * target[k] for k in range(len(design)))
        for i in range(3)
    ]
    a, b, c = solve(normal, moment)
    return a, b, c


def weight_bytes(packed_bytes: int) -> int:
    """`packed.sum()` reads only the 4-bit weight words.

    `packed_bytes` is the affine-4 group-64 total at 0.5625 bytes per weight:
    0.5 in the packed words plus 0.0625 in the bf16 scale and bias. The
    reduction touches the packed words alone, so the read rate must be
    computed against 0.5/0.5625 of the total.
    """
    return packed_bytes * 8 // 9


def report_reads(reads: list[dict], overhead: float) -> dict[int, float]:
    print("\n=== achieved read rate against working-set size ===")
    print(
        f"{'O':>7} {'total MB':>9} {'read MB':>8} {'raw us':>9} {'net us':>9} "
        f"{'net GB/s':>9}")
    rate: dict[int, float] = {}
    for entry in sorted(reads, key=lambda e: -e["packed_bytes"]):
        touched = weight_bytes(entry["packed_bytes"])
        net = entry["raw_us"] - overhead
        rate[entry["outputs"]] = touched / net / 1e3
        print(
            f"{entry['outputs']:>7} {entry['packed_bytes'] / 1e6:>9.1f} "
            f"{touched / 1e6:>8.1f} {entry['raw_us']:>9.2f} "
            f"{net:>9.2f} {touched / net / 1e3:>9.1f}")
    return rate


def report_tensor(
    outputs: int,
    cells: dict[int, tuple[float, float]],
    nbytes: int,
    overhead: float,
    one_pass: float,
) -> dict[str, float]:
    net = {m: (fwd + rev) / 2 - overhead for m, (fwd, rev) in cells.items()}
    a, b, c = fit_width_model([(m, net[m]) for m in MODEL_WIDTHS])

    print(f"\n=== O={outputs}  packed = {nbytes / 1e6:.1f} MB ===")
    print(f"one measured pass over these bytes : {one_pass:.2f} us")
    print(
        f"{'M':>3} {'IPG':>4} {'G':>2} {'fwd us':>9} {'rev us':>9} "
        f"{'drift':>7} {'net us':>9} {'fit':>9} {'resid':>8} {'net/pass':>9}")
    for width in sorted(cells):
        forward, reverse = cells[width]
        modelled = a + b * groups(width) + c * width
        held = "" if width in MODEL_WIDTHS else "  (not in fit)"
        print(
            f"{width:>3} {inputs_per_group(width):>4} {groups(width):>2} "
            f"{forward:>9.2f} {reverse:>9.2f} "
            f"{100 * (reverse - forward) / forward:>6.2f}% "
            f"{net[width]:>9.2f} {modelled:>9.2f} "
            f"{100 * (net[width] - modelled) / net[width]:>7.2f}% "
            f"{net[width] / one_pass:>9.3f}{held}")

    print(f"fit                       : t = {a:.2f} + {b:.2f}*G + {c:.2f}*M")
    print(f"b as a fraction of a pass : {b / one_pass:.3f}")
    print(f"c as a fraction of a pass : {c / one_pass:.3f}")
    print(f"b per packed MB           : {b * 1e3 / (nbytes / 1e6):.1f} ns/MB")
    print(f"c per packed MB           : {c * 1e3 / (nbytes / 1e6):.1f} ns/MB")
    print(
        f"M=1 qmv net / one pass    : {net[1] / one_pass:.3f}"
        "   (a single-row qmv must read the pack exactly once)")
    return {
        "a": a,
        "b": b,
        "c": c,
        "bytes": float(nbytes),
        "one_pass_us": one_pass,
        "b_over_pass": b / one_pass,
        "b_ns_per_mb": b * 1e3 / (nbytes / 1e6),
        "c_ns_per_mb": c * 1e3 / (nbytes / 1e6),
        "m1_over_pass": net[1] / one_pass,
    }


def traffic_share(
    big: dict[str, float], small: dict[str, float], coefficient: str,
    rate_big: float, rate_small: float,
) -> float:
    """How much of a per-byte coefficient behaves like DRAM traffic.

    Pure DRAM traffic costs `bytes / rate`, so on a cache-resident pack its
    per-byte cost falls by exactly `rate_big / rate_small`. Bandwidth-
    independent work costs the same per byte on both packs. The measured
    ratio interpolates between those two limits.
    """
    measured = small[f"{coefficient}_ns_per_mb"] / big[f"{coefficient}_ns_per_mb"]
    traffic_limit = rate_big / rate_small
    return (1.0 - measured) / (1.0 - traffic_limit)


def transfer(summary: dict[str, float], read_gb_s: float) -> None:
    stream_mb = VERIFY_STREAM_BYTES / 1e6
    predicted_b = summary["b_ns_per_mb"] * stream_mb / 1e3
    predicted_c = summary["c_ns_per_mb"] * stream_mb / 1e3
    print("\n=== transfer to the verify weight stream, from bytes alone ===")
    print(f"verify affine-4 weight stream : {stream_mb:.1f} MB")
    print(
        f"b predicted {predicted_b:>9.0f} us   in-model {IN_MODEL['b']:>9.0f} us"
        f"   error {100 * (predicted_b - IN_MODEL['b']) / IN_MODEL['b']:+.1f}%")
    print(
        f"c predicted {predicted_c:>9.0f} us   in-model {IN_MODEL['c']:>9.0f} us"
        f"   error {100 * (predicted_c - IN_MODEL['c']) / IN_MODEL['c']:+.1f}%")

    one_pass = VERIFY_STREAM_BYTES / (read_gb_s * 1e3)
    print("\n=== what a byte reduction can reach in the verify phase ===")
    print(
        f"mandatory single pass over the stream at {read_gb_s:.1f} GB/s "
        f": {one_pass:.0f} us")
    for prompt, spec in RANKED.items():
        phase = (
            IN_MODEL["a"]
            + IN_MODEL["b"] * spec["groups"]
            + IN_MODEL["c"] * spec["mean_m"])
        qmv = phase - IN_MODEL["a"]
        print(
            f"  {prompt:<7} M={spec['mean_m']:.2f} G={spec['groups']}"
            f"  phase {phase:>8.0f} us"
            f"  mandatory pass {100 * one_pass / phase:>5.1f}%"
            f"  qmv term {100 * qmv / phase:>5.1f}%"
            f"  non-qmv fixed {100 * IN_MODEL['a'] / phase:>4.1f}%")


def main() -> int:
    path = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else "research/out/e95_qmv_probe.json")
    payload = json.loads(path.read_text())
    overhead = payload["eval_overhead_us"]
    print(f"=== measured fixed per-eval overhead : {overhead:.2f} us ===")

    rate = report_reads(payload["reads"], overhead)

    cells: dict[int, dict[int, tuple[float, float]]] = {}
    cell_bytes: dict[int, int] = {}
    for cell in payload["cells"]:
        cells.setdefault(cell["outputs"], {})[cell["m"]] = (
            cell["forward_us"], cell["reverse_us"])
        cell_bytes[cell["outputs"]] = cell["packed_bytes"]

    big, small = sorted(cell_bytes, reverse=True)
    summary = {
        outputs: report_tensor(
            outputs, cells[outputs], cell_bytes[outputs], overhead,
            cell_bytes[outputs] / (rate[outputs] * 1e3))
        for outputs in (big, small)
    }

    print("\n=== verdict ===")
    print(
        f"bytes ratio big/small     : "
        f"{summary[big]['bytes'] / summary[small]['bytes']:.2f}x")
    print(
        f"read rate                 : big {rate[big]:.1f} GB/s, "
        f"small {rate[small]:.1f} GB/s, "
        f"cache speedup {rate[small] / rate[big]:.2f}x")
    print(
        f"b is {summary[big]['b_over_pass']:.3f} of a pass on the big tensor "
        f"and {summary[small]['b_over_pass']:.3f} on the small one")
    print(
        f"traffic share of b        : "
        f"{100 * traffic_share(summary[big], summary[small], 'b', rate[big], rate[small]):.0f}%")
    print(
        f"traffic share of c        : "
        f"{100 * traffic_share(summary[big], summary[small], 'c', rate[big], rate[small]):.0f}%")

    print("\n=== elasticity of time to bytes, removing whole weights ===")
    byte_cut = 1.0 - summary[small]["bytes"] / summary[big]["bytes"]
    print(f"{'M':>3} {'big net us':>11} {'small net us':>13} {'time cut':>9} {'elasticity':>11}")
    for width in MODEL_WIDTHS:
        big_net = (sum(cells[big][width]) / 2) - overhead
        small_net = (sum(cells[small][width]) / 2) - overhead
        time_cut = 1.0 - small_net / big_net
        print(
            f"{width:>3} {big_net:>11.2f} {small_net:>13.2f} "
            f"{time_cut:>8.3f} {time_cut / byte_cut:>11.3f}")
    print(f"byte cut big -> small     : {byte_cut:.3f}")

    transfer(summary[big], rate[big])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
