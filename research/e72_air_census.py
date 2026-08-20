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
from e72_wide_gen import ARMS as GEN_ARMS, PRAGMA_ONLY as GEN_PRAGMA_ONLY  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE = REPO / "research/e72_wide_probe.metal"

ARMS = {tag.removeprefix("e72"): f"e72_cell_{tag.removeprefix('e72')}"
        for tag in GEN_ARMS}
VECTORIZES_X = {"xvec", "tailfullxvec"}
# Arms that differ from `plain` by unroll pragmas alone. The generator proves
# that reduction textually; the census only has to confirm what it bought.
UNROLLED = {tag.removeprefix("e72") for tag in GEN_PRAGMA_ONLY}
# Arms that must clear the NA = 6 accumulator spill. `plain` and `xvec` keep
# the rolled tail and are the controls that still carry it. `mfull` opens only
# the inner `i`/`m` loop and is a bisect probe, not a proposed fix. `shift`
# touches address setup only and is expected to spill exactly like `plain`.
FIX_ARMS = {"tailfull", "rfull", "allfull", "tailfullxvec"}
# Each arm's own control for register pressure: an `xvec` arm must be compared
# against `xvec`, because widening the x load costs registers on its own.
CONTROL = {arm: ("xvec" if arm in VECTORIZES_X else "plain") for arm in ARMS}
# Any alloca holding float data, at any nesting depth. A shape test on the
# outer array is not enough: `split` spills as `[4 x [2 x <4 x float>]]`, which
# is the same accumulator demoted to memory one level deeper.
FLOAT_ALLOCA = re.compile(r"\bfloat\b")
# A `<6 x float>` value is legal AIR but has no native register class, so it is
# the shape most likely to be demoted to memory. `split` exists to remove it.
NON_NATIVE_VECTOR = re.compile(r"<\s*(?:[35]|[67]|\d\d+)\s+x\s+float\s*>")
INT_DIVIDE = re.compile(r"=\s*(?:s|u)(?:div|rem)\b")


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
        "float_allocas": sorted(t for t in alloca_types if FLOAT_ALLOCA.search(t)),
        "non_native_vectors": len(NON_NATIVE_VECTOR.findall(text)),
        "integer_divides": len(INT_DIVIDE.findall(text)),
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
        # Each fix arm must clear the spill, and must not pay for it with more
        # register pressure than its own control already carries.
        if arm in FIX_ARMS:
            out[f"{arm}_has_no_float_alloca"] = not stats["float_allocas"]
            out[f"{arm}_peak_live_not_above_control"] = (
                stats["peak_live_cfg_max"]
                <= cell[CONTROL[arm]]["peak_live_cfg_max"])
        if arm.startswith("allfull"):
            out[f"{arm}_has_no_alloca_at_all"] = stats["allocas"] == 0
            out[f"{arm}_k_loop_is_one_block"] = len(stats["loop_blocks"]) == 1
    # Bisect: opening the inner `i`/`m` loop alone does not clear the spill, so
    # the rolled reduction tail is the cause, not the inner accumulate loop.
    out["mfull_alone_does_not_clear_the_spill"] = (
        bool(cell["mfull"]["float_allocas"]) == (na == 6))
    # `split` is the only arm that reaches native lanes by changing the type,
    # so it is the only one required to leave no odd-width float vector at all.
    out["split_has_only_native_float_vectors"] = (
        cell["split"]["non_native_vectors"] == 0)
    # Falsified prediction, recorded as a check so it cannot be quietly reread:
    # native lane width was expected to remove the spill. It does not. `split`
    # reaches native `float4` chunks and STILL spills at NA = 6, at higher
    # pressure than the shipped body, because the rolled tail still indexes the
    # accumulator with a variable and now has more lanes to hold.
    out["native_lane_width_alone_does_not_remove_the_spill"] = (
        bool(cell["split"]["float_allocas"]) == (na == 6)
        and cell["split"]["peak_live_cfg_max"] >= plain["peak_live_cfg_max"])
    # `in_vec_size` is signed, so `/ 2` and `/ 64` stay as real `sdiv`. Every
    # other divide in the body already folds, so the count is exactly two.
    out["shipped_body_keeps_two_signed_divides"] = plain["integer_divides"] == 2
    out["shift_arm_removes_both_signed_divides"] = (
        cell["shift"]["integer_divides"] == 0)
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
