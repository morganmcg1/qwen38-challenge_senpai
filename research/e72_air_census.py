#!/usr/bin/env python3
"""E72 rung 2, compile half: certify the unspilled NA=6 arms from AIR.

Reports, per arm and per NA:

  * alloca count and alloca types. This is the quantity rung 2 targets: the
    shipped body grows a second alloca `[4 x <6 x float>]` at NA = 6 and only
    at NA = 6.
  * peak live registers over the real CFG, reusing the E64 liveness pass.
  * device loads split by operand, so an `xvec` arm is shown to have widened
    the x loads and nothing else.
  * floating-point operation counts. E72 changes storage and control flow, so
    the arithmetic must be IDENTICAL to `plain` at the same NA. A difference
    here is a generator fault, not a result.

Each structural claim is paired with a check that can fail.

  python3 research/e72_air_census.py --na 2 3 4 5 6 \
      --out research/e72-artifacts/rung2-air.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e64_air_census import (  # noqa: E402
    SCORED_FLAGS,
    counts,
    kernel_bodies,
    live_ranges,
    loop_blocks,
    split_blocks,
)
from e69_air_census import load_census  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE = REPO / "research/e72_wide_probe.metal"

ARMS = {
    "plain": "e72_cell_plain",
    "tailfull": "e72_cell_tailfull",
    "allfull": "e72_cell_allfull",
    "xvec": "e72_cell_xvec",
    "tailfullxvec": "e72_cell_tailfullxvec",
    "allfullxvec": "e72_cell_allfullxvec",
}
VECTORIZES_X = {"xvec", "tailfullxvec", "allfullxvec"}
FLOAT_ALLOCA = re.compile(r"^\[\s*\d+\s+x\s+<\s*\d+\s+x\s+float\s*>\s*\]$")


def arm_stats(body: list[str], arm: str, na: int) -> dict:
    text = "\n".join(body)
    loop = loop_blocks(body)
    blocks = dict(split_blocks(body))
    loop_text = "\n".join(line for name in loop for line in blocks.get(name, []))
    liveness = live_ranges(body)
    rows = 4

    census = load_census(blocks, loop)
    total = counts(loop_text)
    x_lanes = 4 if arm in VECTORIZES_X else 1
    dynamic = {
        "weight_loads": 4 * rows,
        "scale_bias_loads": 2 * rows,
        "device_x_loads": 16 * na // x_lanes,
    }
    dynamic["device_loads_total"] = sum(dynamic.values())
    dynamic["device_loads_per_output_row"] = dynamic["device_loads_total"] / rows

    alloca_types = [t.strip() for t in re.findall(r"=\s*alloca\s+([^,\n]+)", text)]
    return {
        "rows_per_simd": rows,
        "loop_blocks": loop,
        "loop_loads_static": census,
        "per_lane_per_k_block": dynamic,
        "loop_fp": {key: total[key] for key in ("fadd", "fmul", "fma")},
        "total_fp": {key: counts(text)[key] for key in ("fadd", "fmul", "fma")},
        "peak_live_cfg_loop": max(
            (liveness["blocks"][name]["peak_live"] for name in loop), default=0),
        "peak_live_cfg_max": max(
            entry["peak_live"] for entry in liveness["blocks"].values()),
        "allocas": len(alloca_types),
        "alloca_types": sorted(alloca_types),
        "float_allocas": sorted(t for t in alloca_types if FLOAT_ALLOCA.match(t)),
    }


def compile_probe(workdir: pathlib.Path, na: int) -> pathlib.Path:
    raw = workdir / f"na{na}.ll"
    optimized = workdir / f"na{na}.o3.ll"
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", *SCORED_FLAGS, f"-DE72_NA={na}",
         "-I", str(INCLUDE), "-S", str(PROBE), "-o", str(raw)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        raise SystemExit(f"compile failed at NA={na}:\n{emit.stderr}")
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(raw), "-o", str(optimized)],
        capture_output=True, text=True)
    if opt.returncode != 0:
        raise SystemExit(f"metal-opt failed at NA={na}:\n{opt.stderr}")
    return optimized


# `allfull` opens the `m` loop, so its k-loop collapses to one basic block and
# every static per-block count is multiplied by a different trip count than
# `plain`'s. Static counts are comparable only between arms that keep the same
# loop structure; the unrolled arms are certified instead by the generator's
# pragma-only check and by the GPU bit-identity harness.
SAME_STRUCTURE = {"plain", "xvec", "tailfull", "tailfullxvec"}


def checks(cell: dict, na: int) -> dict:
    plain = cell["plain"]
    # The fault this whole rung exists to fix, and its boundary: the float
    # alloca must be present at NA = 6 and absent at every narrower width.
    out = {"plain_float_alloca_only_at_na6":
           bool(plain["float_allocas"]) == (na == 6)}
    for arm, stats in cell.items():
        if arm in SAME_STRUCTURE:
            out[f"{arm}_fp_matches_plain"] = stats["loop_fp"] == plain["loop_fp"]
            if arm in VECTORIZES_X:
                out[f"{arm}_x_loads_are_4_wide"] = (
                    stats["loop_loads_static"]["device_x"]["elements"]
                    == 4 * stats["loop_loads_static"]["device_x"]["loads"])
        if arm.startswith(("tailfull", "allfull")):
            out[f"{arm}_has_no_float_alloca"] = not stats["float_allocas"]
        if arm.startswith("allfull"):
            out[f"{arm}_has_no_alloca_at_all"] = stats["allocas"] == 0
            out[f"{arm}_k_loop_is_one_block"] = len(stats["loop_blocks"]) == 1
    out["allfull_fp_matches_allfullxvec"] = (
        cell["allfull"]["loop_fp"] == cell["allfullxvec"]["loop_fp"])
    out["unrolling_does_not_raise_peak_live"] = (
        cell["tailfull"]["peak_live_cfg_max"] <= plain["peak_live_cfg_max"]
        and cell["allfull"]["peak_live_cfg_max"] <= plain["peak_live_cfg_max"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--na", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    cells: dict[int, dict] = {}
    verdicts: dict[int, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for na in args.na:
            bodies = kernel_bodies(compile_probe(workdir, na))
            cell = {}
            for arm, kernel in ARMS.items():
                if kernel not in bodies:
                    raise SystemExit(f"kernel {kernel} missing at NA={na}")
                cell[arm] = arm_stats(bodies[kernel], arm, na)
            cells[na] = cell
            verdicts[na] = checks(cell, na)

    report = {"cells": cells, "checks": verdicts}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"wrote {args.out}")

    print(f"{'NA':>3} {'arm':<14} {'allocas':>7} {'peak_live':>9}  alloca_types")
    for na in args.na:
        for arm in ARMS:
            stats = cells[na][arm]
            print(f"{na:>3} {arm:<14} {stats['allocas']:>7} "
                  f"{stats['peak_live_cfg_max']:>9}  {stats['alloca_types']}")
    failed = {(na, key) for na, block in verdicts.items()
              for key, ok in block.items() if not ok}
    for na, block in verdicts.items():
        for key, ok in sorted(block.items()):
            if not ok:
                print(f"CHECK FAILED NA={na} {key}")
    print(f"checks: {sum(len(b) for b in verdicts.values()) - len(failed)} pass, "
          f"{len(failed)} fail")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
