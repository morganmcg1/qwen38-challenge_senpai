"""E213 Stage-0 desk probe: AIR register pressure per QMV plan and (m, IPG) pair.

`maxTotalThreadsPerThreadgroup` saturates at 1024 for every instantiation of this
kernel on M4 Pro, including the single-pass forms E195 rejected, so it cannot
resolve the IPG cliff. This reads the lane-weighted peak live SSA count from the
optimised AIR text instead. The absolute level is an over-count; only the shape
of the curve across IPG is usable evidence.

    python3 research/e213_regs.py PROBE.metal OUT.json
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from air_kernel_stats import kernels, peak_live_registers  # noqa: E402


def air_text(metal: pathlib.Path, work: pathlib.Path) -> pathlib.Path:
    raw = work / "probe.ll"
    opt = work / "probe.opt.ll"
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2", "-S",
         str(metal), "-o", str(raw)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        raise SystemExit(f"e213: metal -S failed\n{emit.stderr}")
    run = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(raw), "-o", str(opt)],
        capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f"e213: metal-opt failed\n{run.stderr}")
    return opt


def main() -> None:
    metal = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    work = metal.parent
    bodies = kernels(air_text(metal, work))
    rows = {}
    for name, body in sorted(bodies.items()):
        wide = re.search(r"qwen_e120_qmv_wideILi(\d+)ELb(\d)E", name)
        if wide is not None:
            label = f"wide_NA{wide.group(1)}_{'table' if wide.group(2) == '1' else 'plain'}"
        elif name.startswith("plan_") or name.startswith("ipg_"):
            label = name
        else:
            continue
        peak, count = peak_live_registers(body)
        rows[label] = {"peak_live_lane_weighted": peak, "peak_live_values": count,
                       "air_lines": len(body)}
    width = max(len(n) for n in rows)
    print(f"{'kernel':<{width}} {'peak_live':>10} {'values':>8} {'air_lines':>10}")
    for name, row in rows.items():
        print(f"{name:<{width}} {row['peak_live_lane_weighted']:>10}"
              f" {row['peak_live_values']:>8} {row['air_lines']:>10}")
    if out is not None:
        out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"e213: wrote {out}")


if __name__ == "__main__":
    main()
