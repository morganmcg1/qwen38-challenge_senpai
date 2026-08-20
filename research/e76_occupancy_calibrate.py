#!/usr/bin/env python3
"""E76 calibration: does anything readable here turn a register count into occupancy?

The advisor's variant ranking assumes resident simdgroups
= floor(F / (128 * registers)) with F = 208 KiB. Nothing in `__GPU_METADATA`
reports occupancy, so this asks the runtime instead: it sweeps kernels of rising
register pressure, reads the register count each backend allocates, and reads
`maxTotalThreadsPerThreadgroup` for the same kernel on this host.

If that field falls as registers rise, its steps give the register file directly
and the arithmetic can be calibrated. If it stays pinned at the API maximum, the
field is not register-derived and occupancy is unreadable from here, which is a
complete answer rather than a missing one.

  python3 research/e76_occupancy_calibrate.py --out research/e76-artifacts/occupancy.json
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
from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH,
    RANKED_ARCH,
    build_metallib,
    scalar_kernel,
    translate,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "research/e76_occupancy_probe.m"
PIPELINE = re.compile(
    r"PIPELINE function=(\S+) max_total_threads_per_threadgroup=(\d+) "
    r"thread_execution_width=(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--low", type=int, default=8)
    parser.add_argument("--high", type=int, default=256)
    parser.add_argument("--step", type=int, default=8)
    args = parser.parse_args()

    widths = list(range(args.low, args.high + 1, args.step))
    source = ("#include <metal_stdlib>\nusing namespace metal;\n"
              + "".join(scalar_kernel(n) for n in widths))
    names = [f"k_s{n}" for n in widths]

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        (workdir / "probe.metal").write_text(source)
        lib = build_metallib(source, workdir / "lib")
        census = {arch: translate(lib, arch, workdir / f"tt_{arch}")
                  for arch in (LOCAL_ARCH, RANKED_ARCH)}

        binary = workdir / "occupancy"
        subprocess.run(
            ["clang", "-fobjc-arc", "-O2", "-framework", "Metal",
             "-framework", "Foundation", "-o", str(binary), str(PROBE)],
            check=True, capture_output=True, text=True)
        done = subprocess.run(
            [str(binary), str(workdir / "probe.metal"), *names],
            check=True, capture_output=True, text=True)

    limits = {}
    for line in done.stdout.splitlines():
        match = PIPELINE.search(line)
        if match:
            limits[match.group(1)] = {
                "max_total_threads_per_threadgroup": int(match.group(2)),
                "thread_execution_width": int(match.group(3)),
            }
    device_line = next(line for line in done.stdout.splitlines()
                       if line.startswith("DEVICE"))

    rows = []
    print(device_line)
    print(f"{'floats':>7}{'g16s regs':>11}{'g16s spill':>12}"
          f"{'g17s regs':>11}{'g17s spill':>12}{'max threads':>13}"
          f"{'simdgroups':>12}")
    for n, name in zip(widths, names):
        local = census[LOCAL_ARCH][name]
        ranked = census[RANKED_ARCH][name]
        limit = limits[name]["max_total_threads_per_threadgroup"]
        rows.append({
            "live_floats": n,
            "g16s_registers": local["registers"],
            "g16s_spill_bytes": local["spill_bytes"],
            "g17s_registers": ranked["registers"],
            "g17s_spill_bytes": ranked["spill_bytes"],
            "max_total_threads_per_threadgroup": limit,
            "thread_execution_width": limits[name]["thread_execution_width"],
        })
        print(f"{n:>7}{local['registers']:>11}{local['spill_bytes']:>12}"
              f"{ranked['registers']:>11}{ranked['spill_bytes']:>12}"
              f"{limit:>13}{limit // 32:>12}")

    observed = sorted({row["max_total_threads_per_threadgroup"] for row in rows})
    register_bound = len(observed) > 1
    print(f"\nmax_total_threads_per_threadgroup values observed: {observed}")
    if register_bound:
        # Each step gives one register-file estimate: the largest thread count
        # the runtime still allows at that register count.
        for row in rows:
            regs = row["g16s_registers"]
            threads = row["max_total_threads_per_threadgroup"]
            print(f"  regs={regs:<4} threads={threads:<5} "
                  f"implied register file = {regs * threads * 4} bytes")
    else:
        print("The field never moves, so it is not register-derived on this "
              "host and cannot calibrate the occupancy arithmetic.")

    result = {
        "device": device_line,
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "sweep": rows,
        "max_total_threads_values_observed": observed,
        "max_total_threads_is_register_bound": register_bound,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
