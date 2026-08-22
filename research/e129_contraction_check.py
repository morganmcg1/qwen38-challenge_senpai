#!/usr/bin/env python3
"""E129 — is the wide QMV body bit-exact across accumulator widths?

    usage: research/e129_contraction_check.py [--out PATH] [--keep DIR]

THE QUESTION. `qwen_e120_qmv_wide<NA, RPS, USE_TABLE>` accumulates `NA` output
columns in one `vec<float, NA>`. Passes PARTITION the m range and never
combine, so for a fixed output element `(row, m)` the scalar floating-point
operation sequence is the same at every `NA`:

    partial += a0*c0 + a1*c1 + a2*c2 + a3*c3          four times per k block
    acc     += scale*partial + sums*bias              once per k block
    y        = bfloat(simd_sum(acc))                  once per row

Changing `IPG` at fixed `M` is therefore bit-exact BY CONSTRUCTION. The only
ways it can fail are compiler-level, not design-level:

  contraction   the front end may fuse `mul`+`add` into `fma` for one vector
                width and not another, changing the rounding of that element.
                `MTLCompileOptions.fastMathEnabled = false` is set, but that
                is `-fno-fast-math` and does NOT by itself imply
                `-ffp-contract=off`.
  miscompile    E65 saw NA=5 exactness failures traced to a k-loop unroll
                miscompile, which no source-level argument can predict.

This module falsifies the first one cheaply, before any GPU second is spent on
the 512-token row-evidence digest. It cannot falsify the second: only the
digest can.

WHAT IS MEASURED. Two independent layers, per `(NA, RPS, USE_TABLE)` case:

  AIR    `xcrun metal -S -emit-llvm` gives the front end's own choice. The
         ORDERED SEQUENCE of floating-point opcodes is extracted with the
         vector lane count erased, so `fmul <3 x float>` and `fmul <6 x float>`
         both read `fmul.vec`. Two widths round identically when their
         sequences are equal: same operations, same operands' provenance,
         same order, and the same fma/mul/add split.
  ISA    `xcrun metal-tt` runs the real AGX backend for the ranked g17s and
         for this host's g16s, and the register, spill and machine-text record
         is read back. Apple ships NO AGX instruction printer: `metal-objdump
         --disassemble` answers `no instruction printer for target agx3`, so
         the backend's own contraction choice cannot be read as text on any
         Mac. What is available is the machine-text SIZE, which must stay
         affine in `NA` with one slope; a width where the backend expanded
         `fmuladd` into separate multiply and add instructions would leave that
         line.

HOW IT IS READ. A case is CLEAN when its erased opcode sequence equals the
reference case's. Lane counts are reported beside it: the vector opcodes must
scale as `NA` with the SAME per-lane coefficient, and any scalar opcode left in
the rolled `m` loop must be constant in `NA`. Either a changed sequence or a
changed per-lane coefficient is a real rounding risk at that width.

WHAT IT CANNOT SHOW. Instruction counts are not correctness evidence. Equal
per-element opcode mixes are consistent with bit-exactness and cannot prove it.
An unequal mix is decisive in the other direction: it identifies a width whose
rounding the compiler has changed. The backend layer is settled not here but by
the `replicaExactness` runtime test, which compares every routed width against
MLX's own `quantizedMM` on the GPU with positive controls, and finally by the
512-token hex-float row digest.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch  # noqa: E402
import e120_g17s_census as e120  # noqa: E402

# Every accumulator width the shipped and one-pass tables can instantiate,
# against the RPS each table pairs with it. NA=3/RPS=4 is the reference: it is
# the body M=3, M=6 and M=9 run today.
CASES = (
    ("na3rps4", 3, 4),
    ("na4rps4", 4, 4),
    ("na5rps4", 5, 4),
    ("na6rps2", 6, 2),
    ("na7rps2", 7, 2),
    ("na8rps2", 8, 2),
    # Same widths at the shipped RPS, to separate a width effect from an RPS
    # effect if either one moves.
    ("na6rps4", 6, 4),
    ("na3rps2", 3, 2),
)
REFERENCE = "na3rps4"

# LLVM float opcodes worth separating. `fneg`, `fdiv` and `fsub` are absent
# from this body; they are matched so their absence is recorded, not assumed.
FMULADD = re.compile(r"@llvm\.fmuladd\.(?:(f32|bf16)|v(\d+)(f32|bf16))")
IR_LINE = re.compile(
    r"=\s+(?:tail call\s+)?(fmul|fadd|fsub|fneg|fdiv)\b[^=]*?"
    r"(?:<(\d+) x )?(float|bfloat|half)")

# `metal-objdump --disassemble` is attempted anyway: it costs one process and
# a future toolchain that ships an AGX instruction printer would then be used
# without a code change. Today it fails and `disassembled` records that.
ISA_FLOAT = re.compile(r"^\s*[0-9a-f]+:(?:\s+[0-9a-f]{2})+\s+([a-z][a-z0-9_.]*)")


def case_source(na: int, rps: int, table: bool) -> str:
    """One kernel that calls the wide body directly, with no width switch."""
    sums = "xsums" if table else "qmv_null_sums"
    null_decl = "" if table else (
        "\n    const device float* qmv_null_sums = nullptr;")
    body = """    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const int qmv_stride = 8;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;%s
    qwen_e120_qmv_wide<%d, %d, %s>(
        w, scales, biases, x, %s, y, qmv_k, qmv_n, qmv_stride,
        int(qmv_tid.x) * %d,
        int(qmv_tid.y) * %d + int(qmv_sgid) * %d,
        qmv_lid);""" % (null_decl, na, rps,
                        "USE_TABLE" if table else "false", sums,
                        na, 2 * rps, rps)
    inputs = e120.QMV_INPUTS + ([("xsums", "float")] if table else [])
    template = [("bool", "USE_TABLE", "true")] if table else None
    name = "probe_%s_na%d_rps%d" % ("sums" if table else "plain", na, rps)
    return e120.generate(name, inputs, e120.QMV_OUTPUTS, body, template)


def ir_trace(source: str, workdir: pathlib.Path) -> dict:
    """The front end's ordered float opcode sequence, with lanes erased."""
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "probe.metal"
    src.write_text(source)
    done = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", "-emit-llvm", str(src), "-o", "-"],
        capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit("metal -emit-llvm failed:\n%s" % done.stderr)
    (workdir / "probe.ll").write_text(done.stdout)

    sequence: list[str] = []
    lanes: collections.Counter = collections.Counter()
    for line in done.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(("declare", ";")):
            continue
        hit = FMULADD.search(stripped)
        if hit:
            width = int(hit.group(2) or 1)
            scalar = hit.group(1) or hit.group(3)
            sequence.append("fmuladd.%s.%s" % (
                "vec" if hit.group(2) else "scalar", scalar))
            lanes["fmuladd" if width > 1 else "fmuladd_scalar"] += width
            continue
        match = IR_LINE.search(stripped)
        if match:
            op, width_text, scalar = match.groups()
            width = int(width_text or 1)
            sequence.append("%s.%s.%s" % (
                op, "vec" if width_text else "scalar", scalar))
            lanes[op if width_text else "%s_scalar" % op] += width
    # The front end can also express contraction as a flag on a plain fmul or
    # fadd rather than as an intrinsic. Record it: it is the same rounding
    # change under a different spelling.
    contract_flagged = sum(
        1 for line in done.stdout.splitlines()
        if " contract " in line and re.search(r"\b(fmul|fadd)\b", line))
    return {
        "sequence": sequence,
        "lanes": dict(sorted(lanes.items())),
        "contract_flagged": contract_flagged,
    }


def isa_counts(source: str, arch: str, workdir: pathlib.Path) -> dict:
    """Back-end float mnemonic histogram plus the register record."""
    lib = agx_crossarch.build_metallib(source, workdir)
    records = agx_crossarch.translate(lib, arch, workdir)
    if len(records) != 1:
        raise SystemExit("%s: %d kernels in a one-kernel probe" % (arch, len(records)))
    name, record = next(iter(records.items()))
    binary = None
    for candidate in sorted(workdir.glob("%s_*.mtlp" % arch)):
        binary = candidate
    done = subprocess.run(
        ["xcrun", "metal-objdump", "--disassemble", str(binary)],
        capture_output=True, text=True)
    histogram: collections.Counter = collections.Counter()
    if done.returncode == 0:
        for line in done.stdout.splitlines():
            hit = ISA_FLOAT.match(line)
            if hit and hit.group(1).startswith("f"):
                histogram[hit.group(1)] += 1
    return {
        "kernel": name,
        "registers": record.get("registers"),
        "spill_bytes": record.get("spill_bytes", 0),
        "text_bytes": record.get("text_bytes"),
        "disassembled": done.returncode == 0 and bool(histogram),
        "disassembler_error": None if done.returncode == 0 else
                              done.stderr.strip().splitlines()[-1:] or None,
        "float_mnemonics": dict(histogram),
    }


# A width whose machine text sits this far off the line its neighbours define
# is in a different unroll regime, not a different rounding regime.
REGIME_TOLERANCE = 0.02


def text_size_fit(rows: dict, arch: str, suffix: str, rps: int) -> dict:
    """Do the widths at one `RPS` share one loop-unroll regime?

    Machine-text size was the intended contraction proxy and it is NOT one:
    it is dominated by how far the backend unrolls the `k` and `i` loops, a
    choice it abandons once register pressure approaches the cap. That makes it
    a good instrument for the OTHER risk in this experiment. E65 traced NA=5
    exactness failures to a `k` loop unroll miscompile, so a width that jumps to
    a new unroll regime is the width to distrust.

    Least squares over the widths at one `RPS`; the residual is reported as a
    fraction of the fitted size, because a shared regime should leave residuals
    at the level of prologue bookkeeping and a regime change moves the size by
    tens of percent.
    """
    points = sorted(
        (row["na"], row["isa"][arch]["text_bytes"])
        for tag, row in rows.items()
        if tag.endswith(suffix) and row["rps"] == rps and "isa" in row)
    if len(points) < 3:
        return {"testable": False, "points": points}
    n = len(points)
    mean_na = sum(na for na, _ in points) / n
    mean_text = sum(text for _, text in points) / n
    spread = sum((na - mean_na) ** 2 for na, _ in points)
    slope = sum((na - mean_na) * (text - mean_text)
                for na, text in points) / spread
    intercept = mean_text - slope * mean_na
    residuals = {na: round((text - (intercept + slope * na)) / (intercept + slope * na), 5)
                 for na, text in points}
    worst = max(residuals, key=lambda na: abs(residuals[na]))
    one_regime = abs(residuals[worst]) <= REGIME_TOLERANCE
    # A single outlier drags the least-squares line, so the largest residual
    # names the wrong width. The break itself is the step between neighbours
    # that departs most from the typical step.
    steps = [(points[i][0], points[i + 1][0],
              (points[i + 1][1] - points[i][1]) / (points[i + 1][0] - points[i][0]))
             for i in range(len(points) - 1)]
    typical = sorted(step for _, _, step in steps)[len(steps) // 2]
    break_at = max(steps, key=lambda step: abs(step[2] - typical))
    return {
        "testable": True,
        "rps": rps,
        "points": points,
        "bytes_per_na": round(slope, 2),
        "relative_residuals": residuals,
        "worst_width": worst,
        "worst_relative_residual": residuals[worst],
        "one_regime": one_regime,
        "break_between": None if one_regime else [break_at[0], break_at[1]],
        "break_bytes_per_na": None if one_regime else round(break_at[2], 2),
        "typical_bytes_per_na": round(typical, 2),
    }


def per_lane(lanes: dict, na: int) -> dict:
    """Vector opcodes divided by `NA`; scalar opcodes left raw.

    The `m` loop is vectorized for the accumulate chain and rolled for the
    activation load, so a clean width scales the first exactly and leaves the
    second untouched.
    """
    return {key: (round(value / na, 6) if not key.endswith("_scalar") else value)
            for key, value in sorted(lanes.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/out/e129-contraction.json"))
    parser.add_argument("--keep", type=pathlib.Path)
    parser.add_argument("--isa", action="store_true",
                        help="also run the AGX backend and disassemble it")
    args = parser.parse_args()

    header = e120.swift_literal("qwen35E120QMVHeader")
    rows = {}
    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(args.keep) if args.keep else pathlib.Path(raw)
        for label, na, rps in CASES:
            for table in (False, True):
                tag = "%s_%s" % (label, "sums" if table else "plain")
                source = "\n".join(
                    [e120.PRELUDE, header, "", case_source(na, rps, table)])
                trace = ir_trace(source, root / tag / "ir")
                row = {
                    "na": na, "rps": rps, "use_table": table,
                    "ir": trace,
                    "ir_per_lane": per_lane(trace["lanes"], na),
                }
                if args.isa:
                    row["isa"] = {
                        arch: isa_counts(source, arch, root / tag / arch)
                        for arch in e120.ARCHS
                    }
                rows[tag] = row

    verdict = {}
    for table in (False, True):
        suffix = "sums" if table else "plain"
        base = rows["%s_%s" % (REFERENCE, suffix)]
        for label, _, _ in CASES:
            tag = "%s_%s" % (label, suffix)
            row = rows[tag]
            same_sequence = row["ir"]["sequence"] == base["ir"]["sequence"]
            same_scaling = row["ir_per_lane"] == base["ir_per_lane"]
            entry = {
                "reference": "%s_%s" % (REFERENCE, suffix),
                "same_opcode_sequence": same_sequence,
                "same_per_lane_counts": same_scaling,
                "matches_reference": same_sequence and same_scaling,
            }
            if not same_scaling:
                entry["per_lane_delta"] = {
                    key: round(row["ir_per_lane"].get(key, 0)
                               - base["ir_per_lane"].get(key, 0), 6)
                    for key in set(base["ir_per_lane"]) | set(row["ir_per_lane"])
                    if row["ir_per_lane"].get(key, 0) != base["ir_per_lane"].get(key, 0)
                }
            verdict[tag] = entry

    text_fit = {}
    if args.isa:
        for arch in e120.ARCHS:
            for suffix in ("plain", "sums"):
                # RPS 2 holds NA 3, 6, 7 and 8; RPS 4 holds NA 3, 4, 5 and 6.
                for rps in (2, 4):
                    text_fit["%s.%s.rps%d" % (arch, suffix, rps)] = text_size_fit(
                        rows, arch, suffix, rps)

    payload = {
        "experiment": "e129",
        "question": "does the front end or the AGX backend contract "
                    "differently at a different accumulator width?",
        "instrument": "xcrun metal -S -emit-llvm; xcrun metal-tt; "
                      "xcrun metal-objdump",
        "flags": "-std=metal4.0 -O2 -fno-fast-math, matching "
                 "MTLCompileOptions.fastMathEnabled = false",
        "reference_case": REFERENCE,
        "clean": all(v["matches_reference"] for v in verdict.values()),
        "verdict": verdict,
        "machine_text_fit": text_fit,
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    pad = max(len(tag) for tag in rows)
    print("%-*s  %-46s  %s" % (pad, "case", "per lane of the m vector", "verdict"))
    for tag, row in rows.items():
        lanes = row["ir_per_lane"]
        shown = " ".join("%s=%s" % (key, value) for key, value in lanes.items())
        print("%-*s  %-46s  %s" % (
            pad, tag, shown,
            "clean" if verdict[tag]["matches_reference"] else
            "DIFFERS %s" % verdict[tag].get("per_lane_delta", "sequence")))
    if text_fit:
        print("\nAGX machine text against NA, one unroll regime per RPS:")
        for key, fit in text_fit.items():
            if not fit["testable"]:
                continue
            print("  %-28s %-38s %s" % (
                key,
                " ".join("na%d=%d" % point for point in fit["points"]),
                "one regime, %g B per NA" % fit["bytes_per_na"]
                if fit["one_regime"] else
                "REGIME BREAK between NA=%d and NA=%d, %g B per NA against "
                "%g elsewhere" % (*fit["break_between"],
                                  fit["break_bytes_per_na"],
                                  fit["typical_bytes_per_na"])))
    print("\nopcode sequence at %s (lane count erased):" % REFERENCE)
    for suffix in ("plain", "sums"):
        print("  %-5s %s" % (
            suffix, " ".join(rows["%s_%s" % (REFERENCE, suffix)]["ir"]["sequence"])))
    print("\nclean = %s -> %s" % (
        payload["clean"],
        "no width changes the front end's contraction choice"
        if payload["clean"] else "at least one width rounds differently"))
    print("wrote %s" % args.out)
    return 0 if payload["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
