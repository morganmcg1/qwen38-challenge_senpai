#!/usr/bin/env python3
"""E51 driver: emit one runtime-effective JIT string per arm, then run
research/e51_leg_parity.m over the scored affine-4/g64 shapes.

Arms patch the ONE line the campaign invariants pin, inside
`qmv_fast_crossrow_affine4_g64_wide`:

    r0      control, unchanged
    r0b     control with a comment-only change   NULL CONTROL: must stay equal
    d:<n>   R0's BF16 tree kept verbatim, its fp32 result scaled by 1 + 2^-n
            CALIBRATION: walks the comparator's minimum detectable effect
    r1      `(xm[0] + xm[1]) + (xm[2] + xm[3])`  pure reassociation
    r2      `xc[0] + xc[1] + xc[2] + xc[3]`      the fp32 values already in regs

`d:<n>` arms are calibration, never candidates: they widen no stored or
accumulated width and they are not proposals.

Research-only. Nothing here is on the scored path.

    research/e51_leg_parity.py --arms r0 r0b d:23 d:20 r1 r2
    research/e51_leg_parity.py --arms r0 r1 --shapes mlp.down --widths 7,8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from jit_string_compile import assemble  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTRY_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
HARNESS = ROOT / "research/e51_leg_parity.m"
R0_LINE = "sums[m] += xm[0] + xm[1] + xm[2] + xm[3];"


def arm_line(arm: str) -> str:
    if arm == "r0":
        return R0_LINE
    if arm == "r0b":
        return "/* e51 null control: comment only */ " + R0_LINE
    if arm == "r1":
        return "sums[m] += (xm[0] + xm[1]) + (xm[2] + xm[3]);"
    if arm == "r2":
        return "sums[m] += xc[0] + xc[1] + xc[2] + xc[3];"
    if arm.startswith("d:"):
        n = int(arm[2:])
        return ("sums[m] += static_cast<float>(xm[0] + xm[1] + xm[2] + xm[3]) "
                f"* (1.0f + 0x1p-{n}f);")
    raise SystemExit(f"unknown arm {arm!r}")


def emit_arm(arm: str, directory: pathlib.Path) -> pathlib.Path:
    source = assemble((ENTRY_CELL,), None)
    if source.count(R0_LINE) != 1:
        raise SystemExit(f"expected exactly one pinned bias line, found "
                         f"{source.count(R0_LINE)}")
    patched = source.replace(R0_LINE, arm_line(arm))
    path = directory / f"{arm.replace(':', '_')}.metal"
    path.write_text(patched)
    return path


def build_harness(directory: pathlib.Path) -> pathlib.Path:
    binary = directory / "e51_leg_parity"
    command = ["clang", "-fobjc-arc", "-O2", "-framework", "Metal",
               "-framework", "Foundation", "-o", str(binary), str(HARNESS)]
    subprocess.run(command, check=True)
    return binary


def summarise(payload: dict) -> None:
    print(f"\ndevice: {payload['device']}   repeats: {payload['repeats']}")
    arms = []
    for entry in payload["entries"]:
        if entry["arm"] not in arms:
            arms.append(entry["arm"])
    header = (f"{'arm':<8}{'cells':>7}{'legs equal':>12}{'legs diverge':>14}"
              f"{'max|d| legs':>14}{'max ulp':>9}{'wide!=ref':>11}"
              f"{'serial!=ref':>13}{'unstable':>10}")
    print(header)
    print("-" * len(header))
    for arm in arms:
        rows = [e for e in payload["entries"] if e["arm"] == arm]
        equal = sum(1 for e in rows if e["legs"]["equal"])
        diverge = len(rows) - equal
        max_abs = max(e["legs"]["max_abs_delta"] for e in rows)
        max_ulp = max(e["legs"]["max_ulp_delta"] for e in rows)
        wide_diff = sum(1 for e in rows if not e["wide_vs_ref"]["equal"])
        serial_diff = sum(1 for e in rows if not e["serial_vs_ref"]["equal"])
        unstable = sum(1 for e in rows if not e["self_stable"])
        print(f"{arm:<8}{len(rows):>7}{equal:>12}{diverge:>14}{max_abs:>14.4g}"
              f"{max_ulp:>9}{wide_diff:>11}{serial_diff:>13}{unstable:>10}")
    print("\nper-cell detail for arms that move something:")
    for entry in payload["entries"]:
        if entry["legs"]["equal"] and entry["wide_vs_ref"]["equal"]:
            continue
        legs, wide = entry["legs"], entry["wide_vs_ref"]
        print(f"  {entry['arm']:<8} {entry['shape']:<33} M={entry['m']}  "
              f"legs {'EQUAL' if legs['equal'] else 'DIVERGES'} "
              f"({legs['mismatches']}/{legs['compared']}, "
              f"frac {legs['mismatch_fraction']:.3g}, "
              f"max|d| {legs['max_abs_delta']:.3g}, ulp<={legs['max_ulp_delta']})  "
              f"wide_vs_ref {'equal' if wide['equal'] else 'differs'} "
              f"({wide['mismatches']}/{wide['compared']}, "
              f"frac {wide['mismatch_fraction']:.3g})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["r0", "r0b", "r1", "r2"])
    parser.add_argument("--widths", default="3,4,5,6,7,8,9")
    parser.add_argument("--shapes")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json", type=pathlib.Path,
                        default=ROOT / "research/e51-artifacts/leg-parity.json")
    args = parser.parse_args()

    if args.arms[0] != "r0":
        raise SystemExit("the first arm must be r0: it is the reference every "
                         "wide_vs_ref and serial_vs_ref comparison uses")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e51-legs-") as directory:
        work = pathlib.Path(directory)
        binary = build_harness(work)
        command = [str(binary), "--widths", args.widths,
                   "--repeats", str(args.repeats), "--json", str(args.json)]
        if args.shapes:
            command += ["--shapes", args.shapes]
        for arm in args.arms:
            command += ["--arm", f"{arm}={emit_arm(arm, work)}"]
        subprocess.run(command, check=True)
        shutil.rmtree(work / "__pycache__", ignore_errors=True)

    payload = json.loads(args.json.read_text())
    summarise(payload)
    print(f"\nartifact: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
