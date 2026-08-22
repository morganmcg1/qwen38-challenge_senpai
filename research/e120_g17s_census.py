#!/usr/bin/env python3
"""Register, spill, text and resident-simdgroup census of the E120 kernels.

    usage: research/e120_g17s_census.py [--out PATH]

E123 censused the shipped wide QMV entry point on both architectures and showed
that deleting the WHOLE activation chunk-sum tree buys one extra resident
simdgroup on `applegpu_g17s`, the ranked architecture, and buys nothing on
`applegpu_g16s`, this host:

    arm        arch    registers  spill  text   resident simdgroups
    a_base     g16s    94         0      24942  32
    n_nosums   g16s    95         0      22710  32
    a_base     g17s    101        0      25898  39
    n_nosums   g17s    99         0      23608  40

Route B is the only mechanism in the campaign that deletes all of it, so the
question is whether the E120 table kernel reaches 99 registers or fewer on
g17s. Residency is `budget // registers` with the budget E123 recorded, so the
threshold is exact:

    3968 // 99  = 40
    3968 // 100 = 39

This host cannot run a g17s kernel, but `xcrun metal-tt` runs the real AGX
backend for a named architecture on any Mac, which is what
`research/agx_crossarch.py` wraps. So this census costs zero GPU seconds.

WHY THE SOURCE IS REBUILT HERE RATHER THAN DUMPED. The E120 kernels are JIT
kernels: `MLXFast.metalKernel` hands MLX a body and MLX generates the
`[[kernel]]` signature around it at dispatch time from the dtypes and rank of
the arrays actually passed. Nothing writes that generated source to disk. This
module reproduces the generation rule from
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/common/metal_kernel.cpp` and
records the reproduced source beside the numbers so the reproduction can be
audited rather than trusted.

THE POSITIVE CONTROL IS THE REPLICA. `USE_TABLE=false` recomputes the chunk
sums exactly as the incumbent does, so it stands in for E123's `a_base`, and
`USE_TABLE=true` stands in for the deletion. A register drop between the two,
in the same entry point built by the same compiler invocation, is the claim.
Comparing my absolute count against E123's absolute count is weaker, because
E123 censused the incumbent entry point and this censuses mine.

WHY EVERY WIDTH IS CENSUSED SEPARATELY AS WELL. E123's `NA=0` row is not a
separate arm: it is the whole width switch in one entry point, and its register
count is the maximum over the inlined branches, because mutually exclusive
branches reuse registers. On `applegpu_g17s`, `a_base` reads 101 = max(93, 89,
90, 101) over NA = 2, 3, 4, 5, and the maximum comes from the widest branch.
The E120 switch runs to M = 9, so its widest branch is wider than anything E123
compiled, and the whole dispatch inherits that branch's residency even when it
runs at M = 4. The per-width rows below price that: they are what each width
would cost if `M` became a `metal_kernel` template parameter, so MLX compiled
one specialization per width instead of one switch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch  # noqa: E402

ARCHS = ("applegpu_g16s", "applegpu_g17s")
# `research/e123_arms.py:819`, SIMDGROUP_BUDGET.
SIMDGROUP_BUDGET = {"applegpu_g16s": 3072, "applegpu_g17s": 3968}
# E123 entry-point census of the shipped wide QMV, for context only.
E123_REFERENCE = {
    "a_base": {"applegpu_g16s": 94, "applegpu_g17s": 101},
    "n_nosums": {"applegpu_g16s": 95, "applegpu_g17s": 99},
}

QWEN35 = pathlib.Path(
    "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
)

# `metal_kernel.cpp:122-158`. Order matters: MLX appends an attribute only when
# its name occurs in the body, and it appends them in this order.
METAL_ATTRIBUTES = (
    ("dispatch_quadgroups_per_threadgroup", "uint"),
    ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"),
    ("grid_origin", "uint3"),
    ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"),
    ("thread_index_in_quadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
)

# MLX prepends `metal::utils()` to every custom-kernel library
# (`backend/metal/custom_kernel.cpp:70`). The E120 header declares everything
# it needs from scratch, so the only name it borrows from that preamble is the
# BF16 scalar type.
PRELUDE = """#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;
"""

# Widths and their input-per-group pairing, from the dispatch switch.
WIDTH_CASES = ((3, 3), (4, 4), (5, 5), (6, 3), (7, 4), (8, 4), (9, 3))


def swift_literal(name: str) -> str:
    """Extract a `private let NAME = \"\"\"...\"\"\"` Metal source literal."""
    text = QWEN35.read_text()
    start = text.index('private let %s = """' % name)
    start = text.index("\n", start) + 1
    end = text.index('    """', start)
    body = text[start:end]
    # The literal is indented to the `"""` terminator's column.
    return "\n".join(line[4:] if line.startswith("    ") else line
                     for line in body.splitlines())


def qmv_body(table: bool, widths: tuple = WIDTH_CASES) -> str:
    """Reproduce `qwen35E120QMVSource(table:)` from Qwen35.swift.

    `widths` restricts the switch to a subset. The full tuple reproduces the
    shipped entry point; a single pair models the specialization MLX would
    compile if `M` were a template parameter.
    """
    sums = "xsums" if table else "qmv_null_sums"
    flag = "USE_TABLE" if table else "false"
    cases = "\n".join(
        """        case %d:
            qwen_e120_qmv_m<%d, %d, %s>(
                w, scales, biases, x, %s, y,
                qmv_k, qmv_n, qmv_stride,
                qmv_gx, qmv_out_row, qmv_lid);
            break;""" % (m, m, ipg, flag, sums)
        for m, ipg in widths
    )
    null_decl = "" if table else (
        "\n    const device float* qmv_null_sums = nullptr;"
    )
    return """    const int qmv_m = x_shape[x_ndim - 2];
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const int qmv_stride = qmv_m <= 8 ? 8 : 16;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    const int qmv_gx = int(qmv_tid.x);%s
    switch (qmv_m) {
%s
        default:
            break;
    }""" % (null_decl, cases)


def generate(name: str, inputs: list[tuple[str, str]],
             outputs: list[tuple[str, str]], body: str,
             template: list[tuple[str, str, str]] | None = None) -> str:
    """Reproduce `metal_kernel.cpp` signature generation.

    `inputs` are `(name, metal scalar type)` pairs in binding order, all rank
    >= 1 so every one binds as a `device` pointer. `template` entries are
    `(parameter type, parameter name, instantiated value)`. The prelude and
    header are not emitted here, so several kernels can share one library.
    """
    out = []
    if template:
        out.append("template <%s>" % ", ".join(
            "%s %s" % (kind, param) for kind, param, _ in template))
    out.append("[[kernel]] void %s(" % name)
    lines = []
    index = 0
    for arg, dtype in inputs:
        lines.append("  const device %s* %s [[buffer(%d)]]" % (dtype, arg, index))
        index += 1
        # `metal_kernel.cpp:114-134`: a shape, strides or ndim buffer is bound
        # only when the body mentions it by name.
        for suffix, decl in (("_shape", "const constant int* %s_shape"),
                             ("_strides", "const constant int64_t* %s_strides"),
                             ("_ndim", "const constant int& %s_ndim")):
            if (arg + suffix) in body:
                lines.append("  %s [[buffer(%d)]]" % (decl % arg, index))
                index += 1
    for arg, dtype in outputs:
        lines.append("  device %s* %s [[buffer(%d)]]" % (dtype, arg, index))
        index += 1
    for attr, dtype in METAL_ATTRIBUTES:
        if attr in body:
            lines.append("  %s %s [[%s]]" % (dtype, attr, attr))
    out.append(",\n".join(lines) + ") {")
    out.append(body)
    out.append("}")
    if template:
        out.append("template [[host_name(\"%s\")]] [[kernel]] decltype(%s<%s>) %s<%s>;"
                   % (name, name, ", ".join(v for _, _, v in template),
                      name, ", ".join(v for _, _, v in template)))
    return "\n".join(out) + "\n"


def census(source: str, tag: str, workdir: pathlib.Path) -> dict:
    (workdir / ("%s.metal" % tag)).write_text(source)
    lib = agx_crossarch.build_metallib(source, workdir / tag)
    row: dict = {}
    for arch in ARCHS:
        for kernel, record in agx_crossarch.translate(
                lib, arch, workdir / tag).items():
            registers = record.get("registers")
            row.setdefault(arch, {})[kernel] = {
                "registers": registers,
                "spill_bytes": record.get("spill_bytes", 0),
                "text_bytes": record.get("text_bytes"),
                "resident_simdgroups": (
                    SIMDGROUP_BUDGET[arch] // registers if registers else None
                ),
            }
    return row


QMV_INPUTS = [("w", "uint32_t"), ("scales", "bfloat16_t"),
              ("biases", "bfloat16_t"), ("x", "bfloat16_t")]
QMV_OUTPUTS = [("y", "bfloat16_t")]


def arm_source(header: str, table: bool) -> str:
    """One library holding the shipped switch plus every per-width kernel."""
    base = ("qwen35_custom_affine4_g64_qmv_wide_sums_v1" if table
            else "qwen35_custom_affine4_g64_qmv_wide_v1")
    inputs = QMV_INPUTS + ([("xsums", "float")] if table else [])
    template = [("bool", "USE_TABLE", "true")] if table else None
    parts = [PRELUDE, header, ""]
    parts.append(generate(base, inputs, QMV_OUTPUTS,
                          qmv_body(table), template))
    for case in WIDTH_CASES:
        parts.append(generate("%s_m%d" % (base, case[0]), inputs, QMV_OUTPUTS,
                              qmv_body(table, (case,)), template))
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/out/e120-g17s-census.json"))
    parser.add_argument("--keep", type=pathlib.Path,
                        help="write the reproduced Metal sources here")
    args = parser.parse_args()

    header = swift_literal("qwen35E120QMVHeader")
    arms = {
        "replica_no_table": arm_source(header, table=False),
        "sumtable": arm_source(header, table=True),
    }

    result: dict = {
        "harness": "local",
        "instrument": "xcrun metal-tt, AGX backend, zero GPU seconds",
        "timing_valid": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "e123_reference_entry_point": E123_REFERENCE,
        "arms": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for tag, source in arms.items():
            if args.keep:
                args.keep.mkdir(parents=True, exist_ok=True)
                (args.keep / ("%s.metal" % tag)).write_text(source)
            try:
                result["arms"][tag] = census(source, tag, workdir)
            except subprocess.CalledProcessError as error:
                result["arms"][tag] = {
                    "error": (error.stderr or b"").decode()[-2000:]
                }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    for tag, row in result["arms"].items():
        if "error" in row:
            print("%-18s BUILD FAILED" % tag)
            print(row["error"])
            continue
        for arch in ARCHS:
            for kernel, cell in sorted(row.get(arch, {}).items()):
                print("%-18s %-6s %-8s %4s regs / spill %-4s / %6s B / %s simdgroups"
                      % (tag, arch.replace("applegpu_", ""),
                         kernel.rsplit("_v1", 1)[-1] or "switch",
                         cell["registers"], cell["spill_bytes"],
                         cell["text_bytes"], cell["resident_simdgroups"]))

    arch = "applegpu_g17s"
    for tag, row in result["arms"].items():
        if "error" in row:
            continue
        cells = row.get(arch, {})
        switch = [c for k, c in cells.items() if not k.rsplit("_v1", 1)[-1]]
        if not switch:
            continue
        print("%-18s g17s dispatch %d regs -> %d resident simdgroups (E123 "
              "a_base 101/39, n_nosums 99/40)"
              % (tag, switch[0]["registers"], switch[0]["resident_simdgroups"]))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
