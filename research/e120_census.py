#!/usr/bin/env python3
"""Register, spill and text census of the E120 candidate-owned QMV kernels.

The E120 kernels are JIT-compiled by `MLXFast.metalKernel`, so they never reach
`mlx.metallib` and `agx_crossarch.py census --metallib` cannot see them. This
script lifts their Metal source straight out of `Qwen35.swift`, so the censused
text cannot drift from the shipped text, wraps each arm in a standalone entry
point with the same signature MLX generates, and hands the result to the same
AGX backend census used everywhere else in the campaign.

What is censused is the candidate's own code. MLX's generated wrapper adds only
shape and stride buffers around it; every accumulator, every live vector and
the whole k loop are in the source below.

  python3 research/e120_census.py --out research/out/TAG/census.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import agx_crossarch as agx  # noqa: E402

SWIFT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
)

# M -> IPG, the width switch at `quantized.h:1922-1979`.
WIDTHS = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}

PREAMBLE = """#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;
"""

QMV_ENTRY = """
[[kernel]] void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
{table_buffer}    device bfloat16_t* y [[buffer({y_slot})]],
    const constant int* x_shape [[buffer({shape_slot})]],
    const constant int* w_shape [[buffer({shape_slot_w})]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {{
  const int x_ndim = 2;
  const int qmv_k = x_shape[x_ndim - 1];
  const int qmv_n = w_shape[0];
  const int qmv_stride = {stride};
  const uint3 qmv_tid = threadgroup_position_in_grid;
  const uint qmv_lid = thread_index_in_simdgroup;
  const uint qmv_sgid = simdgroup_index_in_threadgroup;
  const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
  const int qmv_gx = int(qmv_tid.x);
{null_decl}  qwen_e120_qmv_m<{m}, {ipg}, {use_table}>(
      w, scales, biases, x, {sums}, y, qmv_k, qmv_n, qmv_stride,
      qmv_gx, qmv_out_row, qmv_lid);
}}
"""

FILL_ENTRY = """
[[kernel]] void qwen_e120_xsums(
    const device bfloat16_t* x [[buffer(0)]],
    device float* xsums [[buffer(1)]],
    const constant int* x_shape [[buffer(2)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const int x_ndim = 2;
__BODY__
}
"""


def swift_literal(name: str) -> str:
    """The Metal text of one triple-quoted Swift string literal."""
    text = SWIFT.read_text()
    match = re.search(
        r'^(?:private )?let ' + re.escape(name) + r' = """\n(.*?)\n    """$',
        text,
        re.S | re.M,
    )
    if match is None:
        raise SystemExit(f"{name} not found in {SWIFT}")
    return match.group(1)


def fill_source() -> str:
    text = SWIFT.read_text()
    start = text.index('name: "qwen35_custom_affine4_g64_xsums_v1"')
    match = re.search(r'source: """\n(.*?)\n        """', text[start:], re.S)
    if match is None:
        raise SystemExit("xsums kernel source not found")
    return match.group(1)


def build_source() -> str:
    parts = [PREAMBLE, swift_literal("qwen35E120QMVHeader")]
    for m, ipg in WIDTHS.items():
        for table in (False, True):
            for consume in ((False, True) if table else (False,)):
                if table:
                    arm = "sumtable" if consume else "fillnoconsume"
                    flag = "true" if consume else "false"
                else:
                    arm = "replica"
                    flag = "false"
                parts.append(
                    QMV_ENTRY.format(
                        name=f"qwen_e120_qmv_{arm}_m{m}",
                        m=m,
                        ipg=ipg,
                        use_table=flag,
                        stride=8 if m <= 8 else 16,
                        table_buffer=(
                            "    const device float* xsums [[buffer(4)]],\n"
                            if table
                            else ""
                        ),
                        y_slot=5 if table else 4,
                        shape_slot=6 if table else 5,
                        shape_slot_w=7 if table else 6,
                        null_decl=(
                            ""
                            if table
                            else "  const device float* qmv_null_sums = nullptr;\n"
                        ),
                        sums="xsums" if table else "qmv_null_sums",
                    )
                )
    parts.append(FILL_ENTRY.replace("__BODY__", fill_source()))
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--arch", nargs="+", default=[agx.LOCAL_ARCH, agx.RANKED_ARCH])
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = agx.build_metallib(build_source(), workdir)
        census = {arch: agx.translate(lib, arch, workdir) for arch in args.arch}

    names = sorted(census[args.arch[0]])
    header = f"{'kernel':34s}" + "".join(f"{a:>26s}" for a in args.arch)
    print(header)
    print(f"{'':34s}" + "".join(f"{'regs  spill   text':>26s}" for _ in args.arch))
    for name in names:
        row = f"{name:34s}"
        for arch in args.arch:
            record = census[arch][name]
            row += (
                f"{record['registers']:>10}"
                f"{record['spill_bytes']:>7}"
                f"{record['text_bytes']:>9}"
            )
        print(row)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(census, indent=1, sort_keys=True))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
