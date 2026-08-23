#!/usr/bin/env python3
"""E159 F10 item 2: does NA = 6 fit the register budget on THIS base?

`research/e135_f22_na6_census.py` cannot answer it any more. That probe was
written against a QMV template family with an `RPS` template parameter and
arms named by `(M, IPG, RPS)` triples such as `(6, 3, 4)` and `(6, 6, 4)`. The
crown-parity base move deleted `RPS`: the tree now holds

    template <int NA, bool USE_TABLE> qwen_e120_qmv_wide       Qwen35.swift:1426
    template <int M, int IPG, bool USE_TABLE> qwen_e120_qmv_m  Qwen35.swift:1527

so the old probe fails to compile on 8 of its 9 arms, including both of its
channel-check controls, and the ledger's 94/42 and 105/37 figures describe a
kernel body that no longer exists here.

This census reads the live header out of `Qwen35.swift` and instantiates the
real cells. The control is not a remembered number; it is the shipped `(6, 3)`
cell built from the same source, in the same probe, by the same compiler.

Zero GPU seconds. `xcrun metal-tt` runs the AGX backend for a named
architecture on any Mac.

  python3 research/e159_na_register_census.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agx_crossarch  # noqa: E402

REPO = HERE.parent
QWEN35 = REPO / "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
OUT = HERE / "e159-artifacts" / "e159_na_register_census.json"

ARCHES = ("applegpu_g16s", "applegpu_g17s")

SHIPPED_TABLE = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}

# (M, IPG, role). The shipped cells are the control ladder; the diagonal cells
# are the one-pass arms that `activeInputGroups(M) == 1` would need.
CELLS = [
    (2, 2, "shipped"),
    (3, 3, "shipped"),
    (4, 4, "shipped"),
    (5, 5, "shipped"),
    (6, 3, "shipped"),
    (7, 4, "shipped"),
    (8, 4, "shipped"),
    (9, 3, "shipped"),
    (6, 6, "one_pass_arm"),
    (7, 7, "one_pass_arm"),
    (8, 8, "one_pass_arm"),
]

# MLX injects `bfloat16_t` through its own kernel preamble; a bare
# `metal_stdlib` build spells the same type `bfloat`.
PREAMBLE = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;
"""

ENTRY = """
kernel void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    const device float* xsums [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]],
    constant int& qmv_k [[buffer(6)]],
    constant int& qmv_n [[buffer(7)]],
    constant int& qmv_stride [[buffer(8)]],
    uint3 qmv_tid [[threadgroup_position_in_grid]],
    uint qmv_lid [[thread_index_in_simdgroup]],
    uint qmv_sgid [[simdgroup_index_in_threadgroup]])
{{
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    qwen_e120_qmv_m<{m}, {ipg}, {table}>(
        w, scales, biases, x, xsums, y,
        qmv_k, qmv_n, qmv_stride,
        int(qmv_tid.x), qmv_out_row, qmv_lid);
}}
"""


def live_header() -> str:
    text = QWEN35.read_text()
    match = re.search(
        r'private let qwen35E120QMVHeader = """\n(.*?)\n    """\n',
        text,
        re.S,
    )
    if match is None:
        raise SystemExit("could not find qwen35E120QMVHeader in Qwen35.swift")
    return match.group(1)


def cell_name(m: int, ipg: int, table: bool) -> str:
    return f"qmv_m{m}_ipg{ipg}_{'tbl' if table else 'rec'}"


def census(header: str, table: bool) -> dict:
    entries = "\n".join(
        ENTRY.format(
            name=cell_name(m, ipg, table),
            m=m,
            ipg=ipg,
            table="true" if table else "false",
        )
        for m, ipg, _ in CELLS
    )
    source = PREAMBLE + header + "\n" + entries
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = agx_crossarch.build_metallib(source, workdir)
        out: dict[str, dict] = {}
        for arch in ARCHES:
            for name, record in agx_crossarch.translate(lib, arch, workdir).items():
                out.setdefault(name, {})[arch.replace("applegpu_", "")] = record
    return out


def main() -> None:
    header = live_header()
    result = {
        "experiment": "e159-f10-na-register-census",
        "harness": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "source": "Qwen35.swift qwen35E120QMVHeader, read live",
        "template_signature": "template <int NA, bool USE_TABLE> qwen_e120_qmv_wide",
        "why_the_old_census_is_void": (
            "research/e135_f22_na6_census.py targets a template family with an "
            "RPS parameter that this base does not have. It fails to build on "
            "8 of 9 arms including both channel-check controls, so the ledger "
            "figures 94/42 and 105/37 cannot be reproduced or used as controls."
        ),
        "cells": {},
    }

    for table in (True, False):
        try:
            records = census(header, table)
        except Exception as exc:  # noqa: BLE001
            result["cells"][f"use_table_{table}"] = {"error": str(exc)[:2000]}
            continue
        block = {}
        for m, ipg, role in CELLS:
            name = cell_name(m, ipg, table)
            entry = records.get(name, {})
            block[f"m{m}_ipg{ipg}"] = {
                "M": m,
                "IPG": ipg,
                "NA": ipg,
                "role": role,
                "streams_ceil_m_over_ipg": -(-m // ipg),
                "shipped_ipg_for_this_m": SHIPPED_TABLE[m],
                **{
                    arch: {
                        "registers": rec.get("registers"),
                        "spill_bytes": rec.get("spill_bytes"),
                        "text_bytes": rec.get("text_bytes"),
                        "text_sha8": rec.get("text_sha8"),
                    }
                    for arch, rec in entry.items()
                },
            }
        result["cells"][f"use_table_{table}"] = block

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    for key, block in result["cells"].items():
        print(f"\n=== {key} ===")
        if "error" in block:
            print("  ERROR:", block["error"][:400])
            continue
        print(f"  {'cell':12s} {'role':13s} {'g16s reg/spill':>16s} {'g17s reg/spill':>16s}  text_sha8")
        for name, rec in block.items():
            g16 = rec.get("g16s", {})
            g17 = rec.get("g17s", {})
            print(
                f"  {name:12s} {rec['role']:13s}"
                f" {str(g16.get('registers')) + '/' + str(g16.get('spill_bytes')):>16s}"
                f" {str(g17.get('registers')) + '/' + str(g17.get('spill_bytes')):>16s}"
                f"  {g17.get('text_sha8')}"
            )
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
