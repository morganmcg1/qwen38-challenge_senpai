#!/usr/bin/env python3
"""Read the register count and frame bytes of every E77 arm, on both generations.

The sweep is only interpretable if each timed arm carries a measured register
count rather than an assumed one, so this runs edward's E72 oracle
(`research/agx_crossarch.py`) over the same source string the timing harness
compiles. It also censuses every legal `(M, IPG)` cell, which is what tells us
whether the local register count is a function of IPG alone. If it is, the
local occupancy factor is collinear with `q(IPG)` and cannot be identified from
the E73 surface at all, which is the whole reason the sweep exists.

  python3 research/e77_reg_census.py --out research/e77-artifacts/rung0-regs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import agx_crossarch  # noqa: E402
from e73_pairs import name as cell_name, pairs  # noqa: E402
from e77_probe import PROBE_ARGS, arms, entry  # noqa: E402
from jit_string_compile import PREAMBLES, preamble  # noqa: E402

CELL_ENTRY = """
[[kernel]] void e77_cell_{arm}(
{args}) {{
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, {m}, {ipg}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}}
"""


def sweep_source() -> str:
    parts = [preamble(stem, None) for stem in PREAMBLES]
    parts += [entry(spec) for spec in arms()]
    return "".join(parts)


def cell_source() -> str:
    parts = [preamble(stem, None) for stem in PREAMBLES]
    parts += [CELL_ENTRY.format(arm=cell_name(m, ipg), m=m, ipg=ipg,
                                args=PROBE_ARGS) for m, ipg in pairs()]
    return "".join(parts)


def census(source: str, workdir: pathlib.Path, archs: list[str]) -> dict:
    lib = agx_crossarch.build_metallib(source, workdir)
    return {arch: agx_crossarch.translate(lib, arch, workdir) for arch in archs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/e77-artifacts/rung0-regs.json"))
    parser.add_argument("--arch", nargs="+",
                        default=[agx_crossarch.LOCAL_ARCH, agx_crossarch.RANKED_ARCH])
    parser.add_argument("--skip-cells", action="store_true")
    args = parser.parse_args()

    sweep = sweep_source()
    result = {
        "experiment": "e77",
        "rung": 0,
        "archs": args.arch,
        "sweep_source_sha256": hashlib.sha256(sweep.encode()).hexdigest(),
        "arms": {spec["arm"]: spec for spec in arms()},
    }
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        result["sweep"] = census(sweep, work / "sweep", args.arch)
        if not args.skip_cells:
            result["cells"] = census(cell_source(), work / "cells", args.arch)

    for arch in args.arch:
        print(f"{arch}")
        for spec in arms():
            record = result["sweep"][arch][f"e77_{spec['arm']}"]
            print(f"  {spec['arm']:<18} pressure={spec['pressure']:<3} "
                  f"registers={record['registers']:<4} "
                  f"frame_bytes={record['spill_bytes']:<5} "
                  f"text_bytes={record['text_bytes']}")
        if "cells" in result:
            for m, ipg in pairs():
                record = result["cells"][arch][f"e77_cell_{cell_name(m, ipg)}"]
                print(f"  cell {cell_name(m, ipg):<13} "
                      f"registers={record['registers']:<4} "
                      f"frame_bytes={record['spill_bytes']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
