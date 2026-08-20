#!/usr/bin/env python3
"""E64 rung 0b, compile half: certify the three NA=5 arms from AIR, not intent.

The GPU half of rung 0b is only interpretable if the arms are what they claim to
be, so this reads the compiled AIR and asserts each claim separately:

  forced   `acc` must have LEFT the registers: an `alloca` of the accumulator
           type must appear and the `phi <NA x float>` promotion of `acc` must be
           gone. The fadd/fmul/fma counts must be UNCHANGED against plain, in the
           k-loop body and in total: this arm moves storage, not arithmetic.
  ballast  the opposite: NO accumulator alloca, unchanged k-loop arithmetic, and
           MORE live registers across the k loop, so a reader can tell "the
           accumulator left the registers" from "the kernel holds more
           registers".

Liveness here is a real backward data-flow over the AIR control-flow graph, not
the text-order scan in air_kernel_stats. E63 could use the text-order proxy
because its arms differed only in NA and the bias cancelled. It cannot be used
here: the ballast consumer sits in a cold block that the block LAYOUT places
before the k loop while the CFG places it after, and the text-order scan reads
that as "the ballast dies before the loop" when it is live across all of it.
Both numbers are reported.

  python3 research/e64_air_census.py --out research/e64-artifacts/rung0b-air.json
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
from air_kernel_stats import peak_live_registers, value_width  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE = REPO / "research/e64_wide_probe.metal"
ARMS = {"plain": "e64_cell_plain", "forced": "e64_cell_forced",
        "ballast": "e64_cell_ballast", "rows2": "e64_cell_rows2"}
# Ledger 196(A): -fno-fast-math is the whole register difference. Same flags E63
# used, so the two censuses are comparable.
SCORED_FLAGS = ["-std=metal4.0", "-O2", "-fno-fast-math"]

LABEL = re.compile(r"^([\w.]+):")
DEF = re.compile(r"^\s*(%[\w.\-]+)\s*=\s*(.*)$")
SSA = re.compile(r"%[\w.\-]+")
PHI = re.compile(r"^\s*(%[\w.\-]+)\s*=\s*phi\s+(.*)$")
PHI_ARM = re.compile(r"\[\s*([^,]+),\s*(%[\w.\-]+)\s*\]")
BR = re.compile(r"^\s*br\s+(?:label\s+(%[\w.\-]+)|i1\s+[^,]+,\s*label\s+(%[\w.\-]+),\s*label\s+(%[\w.\-]+))")
ALLOCA = re.compile(r"alloca\s+([^,]+)")
ACC_ALLOCA = re.compile(r"^<\s*(\d+)\s+x\s+float\s*>$|^\[\s*\d+\s+x\s+<\s*\d+\s+x\s+float\s*>\s*\]$")


def kernel_bodies(path: pathlib.Path) -> dict[str, list[str]]:
    bodies: dict[str, list[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        if line.startswith("define "):
            match = re.search(r"@([A-Za-z0-9_$.]+)\(", line)
            current = match.group(1) if match else None
            if current:
                bodies[current] = []
        elif current is not None:
            if line == "}":
                current = None
            else:
                bodies[current].append(line)
    return bodies


def split_blocks(body: list[str]) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = [("entry", [])]
    for line in body:
        match = LABEL.match(line)
        if match:
            blocks.append((match.group(1), []))
        else:
            blocks[-1][1].append(line)
    return blocks


def live_ranges(body: list[str]) -> dict:
    """Backward liveness over the AIR CFG. Returns per-block peak live width."""
    blocks = split_blocks(body)
    order = [name for name, _ in blocks]
    code = {name: lines for name, lines in blocks}

    succs: dict[str, list[str]] = {}
    phis: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {}
    widths: dict[str, int] = {}
    for name, lines in blocks:
        succs[name] = []
        phis[name] = []
        for line in lines:
            branch = BR.match(line)
            if branch:
                succs[name] += [
                    target.lstrip("%")
                    for target in branch.groups() if target
                ]
            phi = PHI.match(line)
            if phi:
                arms = [(value.strip(), block.lstrip("%"))
                        for value, block in PHI_ARM.findall(phi.group(2))]
                phis[name].append((phi.group(1), arms))
            define = DEF.match(line)
            if define:
                widths[define.group(1)] = value_width(define.group(2))

    live_in: dict[str, set[str]] = {name: set() for name in order}
    live_out: dict[str, set[str]] = {name: set() for name in order}
    changed = True
    while changed:
        changed = False
        for name in reversed(order):
            out: set[str] = set()
            for successor in succs[name]:
                out |= live_in.get(successor, set())
                # A phi's operand is live out of the PREDECESSOR it names.
                for _, arms in phis.get(successor, []):
                    for value, block in arms:
                        if block == name and value.startswith("%"):
                            out.add(value)
            inside = set(out)
            for line in reversed(code[name]):
                define = DEF.match(line)
                target, rhs = (define.group(1), define.group(2)) if define \
                    else (None, line)
                if target:
                    inside.discard(target)
                if PHI.match(line):
                    continue  # phi operands belong to the predecessors
                inside |= {ref for ref in SSA.findall(rhs)
                           if ref in widths or ref in inside}
            if out != live_out[name] or inside != live_in[name]:
                live_out[name], live_in[name] = out, inside
                changed = True

    peaks: dict[str, dict] = {}
    for name, lines in blocks:
        live = set(live_out[name])
        peak = sum(widths.get(value, 1) for value in live)
        peak_count = len(live)
        for line in reversed(lines):
            define = DEF.match(line)
            target, rhs = (define.group(1), define.group(2)) if define \
                else (None, line)
            if target:
                live.discard(target)
            if not PHI.match(line):
                live |= {ref for ref in SSA.findall(rhs) if ref in widths}
            total = sum(widths.get(value, 1) for value in live)
            if total > peak:
                peak, peak_count = total, len(live)
        peaks[name] = {"peak_live": peak, "values": peak_count,
                       "lines": len(lines)}
    return {"blocks": peaks, "order": order,
            "backedges": {name: [s for s in succs[name]
                                 if order.index(s) <= order.index(name)]
                          for name in order}}


def loop_blocks(body: list[str]) -> list[str]:
    """Blocks of the heaviest natural loop, by back-edge and predecessor walk.

    A text-order span cannot be used. The compiler lays the ballast arm's cold
    consumer block out between the loop header and the loop body, so a span
    would count that block's stores as k-loop work.

    "Heaviest" is measured in instructions, not blocks. At NA >= 7 the epilogue
    loop over rows and outputs holds more blocks than the k loop, so a
    block-count rank selects the epilogue and reports a k loop with no weight
    loads in it.
    """
    blocks = split_blocks(body)
    order = [name for name, _ in blocks]
    index = {name: i for i, name in enumerate(order)}
    succs: dict[str, list[str]] = {name: [] for name in order}
    for name, lines in blocks:
        for line in lines:
            branch = BR.match(line)
            if branch:
                succs[name] += [t.lstrip("%") for t in branch.groups() if t]
    preds: dict[str, list[str]] = {name: [] for name in order}
    for name in order:
        for successor in succs[name]:
            if successor in preds:
                preds[successor].append(name)

    weight = {name: len(lines) for name, lines in blocks}
    best: list[str] = []
    best_weight = -1
    for tail in order:
        for head in succs[tail]:
            if index.get(head, len(order)) > index[tail]:
                continue  # not a back edge in this layout
            body_blocks = {head, tail}
            stack = [tail]
            while stack:
                block = stack.pop()
                if block == head:
                    continue  # the walk stops at the header, not through it
                for predecessor in preds[block]:
                    if predecessor not in body_blocks:
                        body_blocks.add(predecessor)
                        stack.append(predecessor)
            if "entry" in body_blocks:
                # The entry block cannot be inside a loop, so this candidate
                # came from a back edge whose head does not dominate its tail.
                continue
            total = sum(weight[name] for name in body_blocks)
            if total > best_weight:
                best_weight = total
                best = sorted(body_blocks, key=lambda name: index[name])
    return best


def counts(text: str) -> dict:
    return {
        "fadd": len(re.findall(r"=\s*fadd\b", text)),
        "fmul": len(re.findall(r"=\s*fmul\b", text)),
        "fma": len(re.findall(r"@(?:llvm|air)\.fma\.", text)),
        "device_loads": len(re.findall(r"=\s*load\b.*addrspace\(1\)", text)),
        "private_loads": len(re.findall(r"=\s*load\b(?!.*addrspace\(1\))", text)),
        "private_stores": len(re.findall(r"^\s*store\b(?!.*addrspace\(1\))",
                                         text, re.M)),
        "device_stores": len(re.findall(r"^\s*store\b.*addrspace\(1\)",
                                        text, re.M)),
        "simd_sum": len(re.findall(r"simd_sum|quad_sum|@air\.simdgroup", text)),
    }


def arm_stats(body: list[str], na: int) -> dict:
    text = "\n".join(body)
    allocas = [t.strip() for t in ALLOCA.findall(text)]
    liveness = live_ranges(body)
    loop = loop_blocks(body)
    loop_text = "\n".join(
        line for name, lines in split_blocks(body) if name in loop
        for line in lines
    )
    text_peak, text_values = peak_live_registers(body)
    return {
        "air_lines": len(body),
        "allocas": len(allocas),
        "alloca_types": allocas,
        "acc_allocas": [t for t in allocas if ACC_ALLOCA.match(t)],
        "phi_acc_width": len(re.findall(rf"phi <{na} x float>", text)),
        "total": counts(text),
        "loop_blocks": loop,
        "loop": counts(loop_text),
        "peak_live_cfg_loop": max(
            (liveness["blocks"][name]["peak_live"] for name in loop),
            default=0),
        "peak_live_cfg_max": max(
            entry["peak_live"] for entry in liveness["blocks"].values()),
        "peak_live_text_order": text_peak,
        "peak_live_text_values": text_values,
    }


def compile_probe(workdir: pathlib.Path, na: int) -> pathlib.Path:
    raw = workdir / f"na{na}.ll"
    optimized = workdir / f"na{na}.o3.ll"
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", *SCORED_FLAGS, f"-DE64_NA={na}",
         "-DE64_ROWS2_CELL=1",
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


def verdict(cells: dict[int, dict], na: int) -> dict:
    plain = cells[na]["plain"]
    forced = cells[na]["forced"]
    ballast = cells[na].get("ballast")
    checks = {
        "forced_has_acc_alloca": bool(forced["acc_allocas"]),
        "forced_acc_phi_removed":
            forced["phi_acc_width"] < plain["phi_acc_width"],
        "forced_loop_fp_unchanged":
            all(forced["loop"][key] == plain["loop"][key]
                for key in ("fadd", "fmul", "fma")),
        "forced_total_fp_unchanged":
            all(forced["total"][key] == plain["total"][key]
                for key in ("fadd", "fmul", "fma")),
        "forced_private_traffic_up":
            forced["loop"]["private_loads"] + forced["loop"]["private_stores"]
            > plain["loop"]["private_loads"] + plain["loop"]["private_stores"],
    }
    if ballast is not None:
        checks |= {
            "ballast_no_acc_alloca": not ballast["acc_allocas"],
            "ballast_loop_fp_unchanged":
                all(ballast["loop"][key] == plain["loop"][key]
                    for key in ("fadd", "fmul", "fma")),
            "ballast_raises_live_registers":
                ballast["peak_live_cfg_loop"] > plain["peak_live_cfg_loop"],
        }
    return checks


def compile_merged(workdir: pathlib.Path, widths: list[int]) -> pathlib.Path:
    """AIR for the arm that mirrors the shipped one-kernel runtime switch.

    The scored `affine_qmv_fast` is a single entry point whose widths are cases
    of a runtime switch, so every width shares one register allocation. Each
    per-NA cell above is its own entry point and cannot show that sharing.
    """
    source = workdir / "merged.metal"
    raw = workdir / "merged.ll"
    optimized = workdir / "merged.o3.ll"
    emit = subprocess.run(
        [sys.executable, str(REPO / "research/e64_emit_arms.py"),
         "--na", str(max(widths)), "--merged-widths", *[str(w) for w in widths],
         "--out", str(source)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        raise SystemExit(f"emit failed:\n{emit.stderr}")
    compile_step = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", *SCORED_FLAGS,
         "-I", str(INCLUDE), "-S", str(source), "-o", str(raw)],
        capture_output=True, text=True)
    if compile_step.returncode != 0:
        raise SystemExit(f"merged compile failed:\n{compile_step.stderr}")
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(raw), "-o", str(optimized)],
        capture_output=True, text=True)
    if opt.returncode != 0:
        raise SystemExit(f"merged metal-opt failed:\n{opt.stderr}")
    return optimized


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--na", type=int, nargs="+", default=[5])
    parser.add_argument("--merged-widths", type=int, nargs="*", default=[])
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    report = {
        "head": head,
        "dirty_paths": len(dirty.splitlines()),
        "flags": SCORED_FLAGS,
        "pipeline": "metal %s -S | metal-opt -passes=default<O3>"
                    % " ".join(SCORED_FLAGS),
        "probe": str(PROBE.relative_to(REPO)),
        "cells": {},
    }
    with tempfile.TemporaryDirectory(prefix="e64-air-") as directory:
        workdir = pathlib.Path(directory)
        for na in args.na:
            bodies = kernel_bodies(compile_probe(workdir, na))
            report["cells"][na] = {
                arm: arm_stats(bodies[kernel], na)
                for arm, kernel in ARMS.items() if kernel in bodies
            }
        if args.merged_widths:
            merged = kernel_bodies(compile_merged(workdir, args.merged_widths))
            text = "\n".join(merged["e64_cell_merged"])
            report["merged"] = {
                "widths": args.merged_widths,
                "stats": arm_stats(merged["e64_cell_merged"],
                                   max(args.merged_widths)),
                "phi_acc_width_by_width": {
                    width: len(re.findall(rf"phi <{width} x float>", text))
                    for width in args.merged_widths},
            }
    report["checks"] = {na: verdict(report["cells"], na) for na in args.na}

    for na in args.na:
        print(f"NA={na}")
        for arm, stats in report["cells"][na].items():
            print(f"  {arm:8s} alloca={stats['allocas']} {stats['alloca_types']}")
            print(f"           phi<{na} x float>={stats['phi_acc_width']:3d}  "
                  f"peak_live cfg_loop={stats['peak_live_cfg_loop']:4d} "
                  f"cfg_max={stats['peak_live_cfg_max']:4d} "
                  f"text={stats['peak_live_text_order']:4d}")
            print(f"           loop  {stats['loop']}")
        for check, passed in report["checks"][na].items():
            print(f"  {'PASS' if passed else 'FAIL'}  {check}")

    if "merged" in report:
        stats = report["merged"]["stats"]
        print(f"merged widths={report['merged']['widths']}")
        print(f"  alloca={stats['allocas']} {stats['alloca_types']}")
        print(f"  peak_live cfg_loop={stats['peak_live_cfg_loop']} "
              f"cfg_max={stats['peak_live_cfg_max']} "
              f"text={stats['peak_live_text_order']}")
        print(f"  phi acc width by width: "
              f"{report['merged']['phi_acc_width_by_width']}")
        print(f"  heaviest loop  {stats['loop']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
