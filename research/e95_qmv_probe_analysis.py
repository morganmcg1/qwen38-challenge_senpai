#!/usr/bin/env python3
"""Read the E95 direct qmv width probe and decide what `b` is.

The verify width model is `verify_us = a + b*G + c*M`, with
`G = ceil(M / IPG)` the number of input groups the WIDE affine-4 kernel runs
over one weight tensor. The fit gives `b = 27,377 us`. Read as one pass over
the 14,412 MB of affine-4 weights the verify phase touches, that is
526.4 GB/s, which is 1.99x the 265 GB/s DRAM read ceiling reported for this
chip. A model term cannot describe traffic the memory system cannot carry,
so either `b` is not a weight pass, or the `b`/`c` split is not identified.

This script reads `research/out/e95_qmv_probe.json`, produced by
`Tests/MLXFastTests/E95QmvWidthProbeTests.swift`.

Every cell is a separate blocking `eval`, so every cell carries the same
fixed host-plus-launch overhead. Two read measurements over two working sets
that share one bandwidth solve for that overhead and for the achieved read
rate with no external constant:

    read_us(bytes) = overhead + bytes / bandwidth

The M=1 quantized matmul is an independent check on that solve, because a
single-row qmv must read the pack exactly once.

With the overhead removed, the same three-parameter model is fitted to one
tensor, so `b` is measured against one measured pass over that tensor's own
bytes.

Usage: python3 research/e95_qmv_probe_analysis.py [path]
"""

from __future__ import annotations

import json
import pathlib
import sys

MODEL_WIDTHS = (3, 4, 5, 6, 8, 9)


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


def fit_width_model(samples: list[tuple[int, float]]) -> tuple[float, ...]:
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
    return tuple(solve(normal, moment))


def report_tensor(
    label: str,
    cells: dict[int, float],
    nbytes: int,
    overhead: float,
    bandwidth: float,
) -> dict[str, float]:
    one_pass = nbytes / (bandwidth * 1e3)
    corrected = {m: us - overhead for m, us in cells.items()}
    a, b, c = fit_width_model([(m, corrected[m]) for m in MODEL_WIDTHS])

    print(f"\n=== {label}  packed = {nbytes / 1e6:.1f} MB ===")
    print(f"one measured pass over these bytes : {one_pass:.1f} us")
    print(
        f"{'M':>3} {'IPG':>4} {'G':>2} {'raw us':>9} {'net us':>9} "
        f"{'fit':>9} {'resid':>8} {'net/pass':>9}")
    for width in sorted(cells):
        net = corrected[width]
        modelled = a + b * groups(width) + c * width
        held = "" if width in MODEL_WIDTHS else "  (not in fit)"
        print(
            f"{width:>3} {inputs_per_group(width):>4} {groups(width):>2} "
            f"{cells[width]:>9.2f} {net:>9.2f} {modelled:>9.2f} "
            f"{100 * (net - modelled) / net:>7.2f}% {net / one_pass:>9.3f}"
            f"{held}")

    print(f"fit                       : t = {a:.1f} + {b:.1f}*G + {c:.1f}*M")
    print(f"b as a fraction of a pass : {b / one_pass:.3f}")
    print(f"b per packed MB           : {b * 1e3 / (nbytes / 1e6):.1f} ns/MB")
    print(f"c per packed MB           : {c * 1e3 / (nbytes / 1e6):.1f} ns/MB")
    print(
        f"M=1 qmv net / one pass    : {corrected[1] / one_pass:.3f}"
        "   (must be about 1.0)")
    return {
        "a": a,
        "b": b,
        "c": c,
        "one_pass_us": one_pass,
        "b_over_pass": b / one_pass,
        "b_ns_per_mb": b * 1e3 / (nbytes / 1e6),
        "c_ns_per_mb": c * 1e3 / (nbytes / 1e6),
        "bytes": float(nbytes),
    }


def main() -> int:
    path = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else "research/out/e95_qmv_probe.json")
    payload = json.loads(path.read_text())

    cells: dict[int, dict[int, float]] = {}
    bytes_of: dict[int, int] = {}
    for cell in payload["cells"]:
        cells.setdefault(cell["outputs"], {})[cell["m"]] = cell["microseconds"]
        bytes_of[cell["outputs"]] = cell["packed_bytes"]
    reads = {
        int(k): bytes_of[int(k)] / (v * 1e3)
        for k, v in payload["read_gb_s"].items()
    }

    big, small = sorted(bytes_of, reverse=True)
    bandwidth = (bytes_of[big] - bytes_of[small]) / (
        (reads[big] - reads[small]) * 1e3)
    overhead = reads[small] - bytes_of[small] / (bandwidth * 1e3)

    print("=== host constants solved from the two read measurements ===")
    print(f"achieved read bandwidth   : {bandwidth:.1f} GB/s")
    print(f"fixed per-eval overhead   : {overhead:.2f} us")
    print(
        f"read O={big:<6} raw {reads[big]:8.2f} us  "
        f"net {reads[big] - overhead:8.2f} us  "
        f"net GB/s {bytes_of[big] / (reads[big] - overhead) / 1e3:6.1f}")
    print(
        f"read O={small:<6} raw {reads[small]:8.2f} us  "
        f"net {reads[small] - overhead:8.2f} us  "
        f"net GB/s {bytes_of[small] / (reads[small] - overhead) / 1e3:6.1f}")

    summary = {
        outputs: report_tensor(
            f"O={outputs}", cells[outputs], bytes_of[outputs], overhead,
            bandwidth)
        for outputs in (big, small)
    }

    print("\n=== verdict ===")
    print(
        f"bytes ratio big/small     : "
        f"{summary[big]['bytes'] / summary[small]['bytes']:.2f}x")
    print(f"b ratio    big/small      : {summary[big]['b'] / summary[small]['b']:.2f}x")
    print(f"c ratio    big/small      : {summary[big]['c'] / summary[small]['c']:.2f}x")
    print(
        f"b is {summary[big]['b_over_pass']:.3f} of a pass on the big tensor "
        f"and {summary[small]['b_over_pass']:.3f} on the small one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
