#!/usr/bin/env python3
"""E129 rung 2e -- can the wide QMV make ONE pass over the weight matrix?

`qwen_e120_qmv_m<M, IPG>` splits `M` input rows into `ceil(M / IPG)` groups and
every group re-reads the whole weight matrix. The shipped table

    (3,3) (4,4) (5,5) (6,3) (7,4) (8,4) (9,3)

gives one pass at M = 3, 4, 5, two passes at M = 6, 7, 8 and three at M = 9.
The rung 2d probe measured what a pass costs: forcing M = 5 to two passes made
it 18.6 % to 39.4 % slower, while the residency change that came with it moved
nothing. So the lever is the pass count, and the interesting direction is the
opposite of rung 2-lite: raise `IPG` so the wide widths make one pass.

`IPG = M` is legal at every routed width (`M % M == 0` satisfies the
`M % IPG != 1` assert), so the only question this census answers is whether
`qwen_e120_qmv_wide<NA>` is buildable at NA = 6...9 without spilling, and what
it costs the shared entry point.

Zero GPU seconds: `xcrun metal-tt` register reports only. Every resident
simdgroup figure is DERIVED as `budget // registers`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e120_g17s_census as e120  # noqa: E402

ARCHS = e120.ARCHS
SIMDGROUP_BUDGET = e120.SIMDGROUP_BUDGET
SHIPPED_CASES = e120.WIDTH_CASES
ONE_PASS_CASES = tuple((m, m, rps) for m, _, rps in SHIPPED_CASES)
BODY_NA = (2, 3, 4, 5, 6, 7, 8, 9)

# `rows_per_simd` sets how many output rows one simdgroup accumulates. Live
# vector registers scale as roughly `(2 * RPS + 5) * NA`, so lowering it is the
# only lever that buys one-pass width without raising the entry point's
# register ceiling. It costs proportionally more activation re-reads, which the
# cache absorbs, and it needs a matching grid in the entry point.
BODY_RPS = (1, 2, 4)

# Passes over the weight matrix per routed width, shipped against one-pass.
SHIPPED_PASSES = {m: -(-m // ipg) for m, ipg, _ in SHIPPED_CASES}

# Realised verification widths over the 312 rounds of the rung 5e session.
REALISED_HISTOGRAM = {4: 16, 5: 20, 6: 20, 7: 12, 8: 240}


def body_only(table: bool, rps: int) -> str:
    """One inlined `qwen_e120_qmv_wide<NA, RPS>` with no width dispatch above."""
    sums = "xsums" if table else "qmv_null_sums"
    flag = "USE_TABLE" if table else "false"
    null_decl = "" if table else "\n    const device float* qmv_null_sums = nullptr;"
    return """    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const int qmv_stride = 8;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * %d + int(qmv_sgid) * %d;
    const int qmv_first_m = int(qmv_tid.x) * NA;%s
    qwen_e120_qmv_wide<NA, %d, %s>(
        w, scales, biases, x, %s, y,
        qmv_k, qmv_n, qmv_stride,
        qmv_first_m, qmv_out_row, qmv_lid);""" % (
        2 * rps, rps, null_decl, rps, flag, sums)


def arm_source(header: str, table: bool, rps: int) -> str:
    base = ("qwen35_custom_affine4_g64_qmv_wide_sums_v2" if table
            else "qwen35_custom_affine4_g64_qmv_wide_v2")
    inputs = e120.QMV_INPUTS + ([("xsums", "float")] if table else [])
    use_table = [("bool", "USE_TABLE", "true")] if table else []
    parts = [e120.PRELUDE, header, ""]
    if rps == 4:
        parts.append(e120.generate(base, inputs, e120.QMV_OUTPUTS,
                                   e120.qmv_body(table), use_table or None))
        parts.append(e120.generate("%s_onepass" % base, inputs, e120.QMV_OUTPUTS,
                                   e120.qmv_body(table, ONE_PASS_CASES),
                                   use_table or None))
    for na in BODY_NA:
        parts.append(e120.generate(
            "%s_body_na%d" % (base, na), inputs, e120.QMV_OUTPUTS,
            body_only(table, rps), [("int", "NA", str(na))] + use_table))
    return "\n".join(parts) + "\n"


def classify(kernel: str) -> tuple[str, int | None]:
    if "_body_na" in kernel:
        return "body", int(kernel.rsplit("_body_na", 1)[1])
    if kernel.endswith("_onepass"):
        return "switch_onepass", None
    return "switch_shipped", None


def rows(census: dict) -> list[dict]:
    out = []
    for (arm, rps), per_arch in census.items():
        for arch, kernels in per_arch.items():
            budget = SIMDGROUP_BUDGET[arch]
            for kernel, stats in kernels.items():
                variant, na = classify(kernel)
                registers = stats["registers"]
                out.append({
                    "arm": arm,
                    "rows_per_simd": rps,
                    "arch": arch,
                    "variant": variant,
                    "na": na,
                    "registers": registers,
                    "spill_bytes": stats["spill_bytes"],
                    "text_bytes": stats["text_bytes"],
                    "resident_simdgroups": budget // registers if registers else None,
                })
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/out/e129-pass-count-census.json"))
    parser.add_argument("--keep", type=pathlib.Path)
    args = parser.parse_args()

    header = e120.swift_literal("qwen35E120QMVHeader")
    arms = {}
    for rps in BODY_RPS:
        arms[("replica_no_table", rps)] = arm_source(header, False, rps)
        arms[("sumtable", rps)] = arm_source(header, True, rps)

    census: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for (tag, rps), source in arms.items():
            name = "%s-rps%d" % (tag, rps)
            if args.keep:
                args.keep.mkdir(parents=True, exist_ok=True)
                (args.keep / ("%s.metal" % name)).write_text(source)
            census[(tag, rps)] = e120.census(source, name, workdir)

    table = rows(census)
    result = {
        "harness": "local",
        "instrument": "xcrun metal-tt, AGX backend, zero GPU seconds",
        "timing_valid": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "base_sha": git("rev-parse", "HEAD"),
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "shipped_cases": [list(c) for c in SHIPPED_CASES],
        "one_pass_cases": [list(c) for c in ONE_PASS_CASES],
        "shipped_passes": {str(k): v for k, v in sorted(SHIPPED_PASSES.items())},
        "realised_histogram": {str(k): v for k, v in sorted(REALISED_HISTOGRAM.items())},
        "rows": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    header_fmt = "%-17s %4s %-14s %-15s %3s %6s %6s %9s %9s"
    print(header_fmt % ("arm", "RPS", "arch", "variant", "NA", "regs", "spill",
                        "text B", "resident"))
    for row in sorted(table, key=lambda r: (r["arm"], r["rows_per_simd"], r["arch"],
                                            r["variant"], r["na"] or 0)):
        print(header_fmt
              % (row["arm"], row["rows_per_simd"], row["arch"], row["variant"],
                 row["na"] if row["na"] else "-", row["registers"],
                 row["spill_bytes"], row["text_bytes"], row["resident_simdgroups"]))

    print("\npasses over the weight matrix per routed width")
    print("  shipped : " + "  ".join(
        "M=%d:%d" % (m, p) for m, p in sorted(SHIPPED_PASSES.items())))
    print("  onepass : " + "  ".join("M=%d:1" % m for m, *_ in SHIPPED_CASES))
    weighted = sum(rounds * SHIPPED_PASSES[m]
                   for m, rounds in REALISED_HISTOGRAM.items())
    total = sum(REALISED_HISTOGRAM.values())
    print("  realised histogram: %.4f passes per routed round -> 1.0000, "
          "a %.2f %% cut in weight traffic"
          % (weighted / total, 100 * (1 - total / weighted)))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
