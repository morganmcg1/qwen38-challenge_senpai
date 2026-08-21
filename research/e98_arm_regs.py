#!/usr/bin/env python3
"""E98 rung 1b: register and AIR census of the three LUT arms. No GPU.

The indexed dequantisation replaces one streamed `biases` load with an integer
index plus two LUT loads. That trades bandwidth for addressing work, so the
arm can only be a real candidate if it does not push the entry point past the
occupancy ceiling. This runs the E46/E59 pipeline (`metal -O2 -S` then
`metal-opt -passes=default<O3>`) over the whole self-contained arm source and
reports the entry point that MLX actually compiles.

  python3 research/e98_arm_regs.py --dir /tmp/e98-arms \
      --out research/out/e98-lut-r1b/regs.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e46_reg_census import CEILING, air_stats  # noqa: E402

ENTRY = "affine_qmv_fast_bfloat16_t_64_4_false"
ARMS = ("a", "b", "c")
ARM_NAME = {"a": "a_shipped", "b": "b_indexed", "c": "c_constant"}


def census(src: pathlib.Path, entry: str) -> dict:
    ll = src.with_suffix(".ll")
    ll_o3 = src.with_suffix(".o3.ll")
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2", "-S",
         str(src), "-o", str(ll)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        return {"status": "compile_failed",
                "error": emit.stderr.strip().splitlines()[-8:]}
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(ll), "-o", str(ll_o3)],
        capture_output=True, text=True)
    if opt.returncode != 0:
        return {"status": "metal_opt_failed",
                "error": opt.stderr.strip().splitlines()[-8:]}

    body, inside = [], False
    for line in ll_o3.read_text().splitlines():
        if line.startswith("define ") and ("@%s(" % entry) in line:
            inside = True
        elif inside and line == "}":
            inside = False
        elif inside:
            body.append(line)
    if not body:
        return {"status": "entry_not_found", "entry": entry}
    return dict(air_stats(body), status="ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/e98-arms")
    ap.add_argument("--entry", default=ENTRY)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms_dir = pathlib.Path(args.dir)
    results = {}
    for a in ARMS:
        src = arms_dir / ("arm_%s.metal" % a)
        results[ARM_NAME[a]] = dict(census(src, args.entry),
                                    source_bytes=src.stat().st_size)

    base = results["a_shipped"]
    for name, res in results.items():
        if res["status"] == "ok" and base["status"] == "ok":
            res["reg_delta_vs_shipped"] = (
                res["peak_live_regs"] - base["peak_live_regs"])
            res["device_load_delta_vs_shipped"] = (
                res["device_loads"] - base["device_loads"])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"entry": args.entry,
         "scope": "entry_point_whole_function",
         # The entry point inlines every width branch, so its peak is a max
         # over branches and is far above the per-cell CEILING the E46/E59
         # census applies. Only the arm-to-arm delta is meaningful here.
         "per_cell_ceiling_not_applicable": CEILING,
         "pipeline": "metal -O2 -S | metal-opt -passes=default<O3>",
         "arms": results}, indent=2) + "\n")

    print("%-12s %-8s %5s %5s %8s %8s" % (
        "arm", "status", "regs", "d", "air", "dev_ld"))
    for name, res in results.items():
        print("%-12s %-8s %5s %5s %8s %8s" % (
            name, res["status"], res.get("peak_live_regs", "-"),
            res.get("reg_delta_vs_shipped", "-"), res.get("air_lines", "-"),
            res.get("device_loads", "-")))
    print("wrote %s" % out)
    return 0 if all(r["status"] == "ok" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
