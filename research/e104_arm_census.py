#!/usr/bin/env python3
"""E104 rung 0: per-NA AIR and register census of the rate-curve arms. No GPU.

The shipped entry point inlines every width branch, so its register peak is a
max over branches and cannot answer an NA question. `e104_variant_sources.py`
therefore appends one isolated `e104_iso_naN` kernel per NA to every arm, and
this script prices each of those cells on its own.

  python3 research/e104_arm_census.py --dir /tmp/e104-arms \
      --out research/out/e104-r0/census.json

The counts that decide rung 0 are `device_loads` (does the load instruction
count scale with NA?) and the nibble-extraction count (does the dequantisation
repeat per activation row?). Both are read from the AIR the runtime compiles,
not from the C++ source.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from air_kernel_stats import peak_live_registers  # noqa: E402

ARMS = ("a_base", "l_loadonly", "z_noxload", "xw_widex")
PROBE_NA = (2, 3, 4, 5, 6)

DEVICE_LOAD = re.compile(r"^\s*%\S+ = load .*addrspace\(1\)", re.M)
ANY_LOAD = re.compile(r"^\s*%\S+ = load ", re.M)
ALLOCA = re.compile(r"^\s*%\S+ = alloca ", re.M)
FMA = re.compile(r"@llvm\.fma\.")
FMUL = re.compile(r"^\s*%\S+ = fmul ", re.M)
FADD = re.compile(r"^\s*%\S+ = fadd ", re.M)
AND_OP = re.compile(r"^\s*%\S+ = and i", re.M)
LSHR = re.compile(r"^\s*%\S+ = lshr i", re.M)
UITOFP = re.compile(r"^\s*%\S+ = uitofp ", re.M)
FPEXT = re.compile(r"^\s*%\S+ = fpext ", re.M)


def vector_width(line: str) -> int:
    m = re.search(r"<(\d+) x ", line)
    return int(m.group(1)) if m else 1


def weighted(body: list[str], pattern: re.Pattern) -> int:
    """Count operations in lanes, so a <5 x float> fmul counts as five."""
    total = 0
    for line in body:
        if pattern.search(line + "\n"):
            total += vector_width(line)
    return total


def air_for(src: pathlib.Path, entry: str) -> dict:
    ll = src.with_suffix(".ll")
    ll_o3 = src.with_suffix(".o3.ll")
    if not ll_o3.exists():
        emit = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2", "-S",
             str(src), "-o", str(ll)],
            capture_output=True, text=True)
        if emit.returncode != 0:
            return {"status": "compile_failed",
                    "error": emit.stderr.strip().splitlines()[-12:]}
        opt = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>",
             "-S", str(ll), "-o", str(ll_o3)],
            capture_output=True, text=True)
        if opt.returncode != 0:
            return {"status": "metal_opt_failed",
                    "error": opt.stderr.strip().splitlines()[-12:]}

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

    text = "\n".join(body) + "\n"
    live = peak_live_registers(body)
    return {
        "status": "ok",
        "air_lines": len(body),
        "device_loads": len(DEVICE_LOAD.findall(text)),
        "loads": len(ANY_LOAD.findall(text)),
        "allocas": len(ALLOCA.findall(text)),
        "peak_live_regs": live[0] if isinstance(live, tuple) else live,
        "fma_lanes": weighted(body, FMA),
        "fmul_lanes": weighted(body, FMUL),
        "fadd_lanes": weighted(body, FADD),
        "and_ops": len(AND_OP.findall(text)),
        "lshr_ops": len(LSHR.findall(text)),
        "uitofp_lanes": weighted(body, UITOFP),
        "fpext_lanes": weighted(body, FPEXT),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/e104-arms")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms_dir = pathlib.Path(args.dir)
    results: dict[str, dict] = {}
    for arm in ARMS:
        src = arms_dir / ("iso_%s.metal" % arm)
        results[arm] = {
            str(na): air_for(src, "e104_iso_na%d" % na) for na in PROBE_NA
        }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"scope": "one isolated qmv_fast_crossrow_affine4_g64_wide<T, NA, true>",
         "pipeline": "metal -O2 -S | metal-opt -passes=default<O3>",
         "arms": results}, indent=2) + "\n")

    head = ("arm", "NA", "status", "regs", "air", "dev_ld", "fma", "fmul",
            "fadd", "and", "lshr", "u2f", "fpext", "alloca")
    print("%-12s %3s %-8s %6s %6s %7s %7s %7s %7s %6s %6s %6s %6s %6s" % head)
    for arm, per_na in results.items():
        for na, r in per_na.items():
            print("%-12s %3s %-8s %6s %6s %7s %7s %7s %7s %6s %6s %6s %6s %6s" % (
                arm, na, r["status"], r.get("peak_live_regs", "-"),
                r.get("air_lines", "-"), r.get("device_loads", "-"),
                r.get("fma_lanes", "-"), r.get("fmul_lanes", "-"),
                r.get("fadd_lanes", "-"), r.get("and_ops", "-"),
                r.get("lshr_ops", "-"), r.get("uitofp_lanes", "-"),
                r.get("fpext_lanes", "-"), r.get("allocas", "-")))
    print("wrote %s" % out)
    ok = all(r["status"] == "ok" for p in results.values() for r in p.values())
    if not ok:
        for arm, per_na in results.items():
            for na, r in per_na.items():
                if r["status"] != "ok":
                    print("FAIL %s NA=%s: %s" % (arm, na, r))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
