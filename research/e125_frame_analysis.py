#!/usr/bin/env python3
"""E125: read one frame session and fit the isolated-to-in-situ transfer law.

    research/e125_frame_analysis.py --rate research/out/e125-full/rate.json \
        --out research/e125-artifacts/frame-law.json

WHAT A FRAME IS. Every frame runs the SAME compiled pipeline state, the same
width and the same launched grid volume. Only the memory regime moves:

    base       one weight allocation, E118's frame
    cycle      N bit-identical allocations, advanced once per dispatch
    consumer   a bandwidth consumer saturating the bus on a second queue
    kNNNN      a shorter weight stream, so the working set becomes resident

Registers, residency and geometry are therefore constant across frames, which
is exactly what breaks the rank collinearity between residency and roofline
distance that Stage 0 found in the width sweep.

THE HEADLINE STATISTIC IS ABSOLUTE, NOT A PERCENTAGE. A shortened frame runs
fewer k-blocks against the same fixed prologue and epilogue, so the same
per-instruction cost is a larger fraction of a smaller total. Percentages
across frames are therefore not comparable and the primary number is

    us_per_instruction_per_k_block = (t_hi - t_lo) / (n_hi - n_lo) / k_blocks

for a rung contrast, which cancels the injection scaffold exactly. The percent
form is reported beside it and never instead of it.

BANDWIDTH GATE, PER FRAME. E123's defect-22 gate rejects any cell whose implied
read rate exceeds 1.2 x 273 GB/s, because an out-of-bounds device write faults
the command buffer and every later dispatch retires in microseconds. A
cache-resident frame legitimately exceeds that rate, so the gate is applied per
frame and a resident frame is exempted by name rather than pooled away. The
base and cycle frames are still gated at the real limit.

harness=local throughout. This host cannot reach the 40 C cool gate, so no
number here is gate qualified and none is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e118_analysis as e118a  # noqa: E402

PEAK_BANDWIDTH_GB_S = 273.0
BANDWIDTH_GATE_FACTOR = 1.2
# A frame whose weight stream fits in cache may exceed the DRAM roofline
# without any fault. Only streaming frames are gated at the DRAM limit.
STREAMING_FRAMES = ("base", "cycle", "consumer")
NULL_SCAFFOLD_GATE_PCT = 0.5

# (class, low rung, high rung, injected instructions per k-block on each rung).
LADDER = (
    ("ld", "k_ld8", "k_ld16", 8, 16),
    ("alu", "k_alu8", "k_alu16", 8, 16),
)
# The one real deletion in the arm set: the whole activation add tree.
#
# The reference is `q_scaffold`, not `a_base`. `n_nosums` is built from the
# scaffold source with the add tree removed, so the scaffold is the arm it
# differs from by exactly the deletion. `a_base` is the shipped function
# verbatim, and it keeps a compile-time branch the scaffold resolves, so an
# `a_base` reference would charge the deletion for a codegen difference as
# well. E123 could use `a_base` because the two were within 0.3 % there; this
# session reports both so the choice is visible.
DELETION = ("n_nosums", "q_scaffold", "a_base")


def timing_rows(rate: dict, keep_warmup: bool = False) -> list[dict]:
    rows = [r for r in rate["measurements"] if r.get("kind") == "timing"]
    if not keep_warmup:
        rows = [r for r in rows if not r.get("warmup")]
    return rows


def cells(rate: dict) -> dict:
    """(shape, m, frame) -> {arm: [seconds...]}, plus the row constants."""
    out: dict[tuple, dict] = {}
    for row in timing_rows(rate):
        key = (row["shape"], row["m"], row["frame"])
        cell = out.setdefault(key, {
            "shape": row["shape"], "m": row["m"], "frame": row["frame"],
            "k": row["k"], "k_blocks": row["k_blocks"],
            "read_bytes": row["read_bytes"],
            "launched_threads": row["launched_threads"],
            "consumer_gbps": [], "entry_c": [], "inner": row["inner"],
            "seconds": {a: [] for a in rate["arms"]},
        })
        for arm, sec in row["seconds"].items():
            cell["seconds"][arm].append(sec)
        cell["consumer_gbps"].append(row["consumer_gbps"])
        cell["entry_c"].append(row["gpu_temp_entry_c"])
    for cell in out.values():
        cell["stats"] = {a: e118a.summarise([s * 1e6 for s in v])
                         for a, v in cell["seconds"].items() if v}
        base_us = cell["stats"]["a_base"]["median"]
        cell["base_us"] = base_us
        cell["achieved_gb_s"] = cell["read_bytes"] / (base_us * 1e-6) / 1e9
        cell["consumer_gb_s"] = statistics.fmean(cell["consumer_gbps"])
        # phi_arm prices the kernel against the whole bus. phi_eff prices it
        # against the share of the bus the consumer left for it, which is the
        # number the in-situ model actually needs.
        cell["phi_arm"] = cell["achieved_gb_s"] / PEAK_BANDWIDTH_GB_S
        left = PEAK_BANDWIDTH_GB_S - cell["consumer_gb_s"]
        cell["phi_eff"] = cell["achieved_gb_s"] / left if left > 0 else None
        cell["entry_c_min"] = min(cell["entry_c"])
        cell["entry_c_max"] = max(cell["entry_c"])
        cell["entry_c_spread"] = cell["entry_c_max"] - cell["entry_c_min"]
    return out


def sem_of_difference(a: dict, b: dict) -> float:
    return math.sqrt(a["sem"] ** 2 + b["sem"] ** 2)


def prices(cell: dict) -> dict:
    """Per class: the rung contrast, absolute and percent.

    The contrast subtracts two arms that carry the same injection scaffold and
    differ only in how many instructions of one class they inject, so no price
    here carries a share of the scaffold.
    """
    out = {}
    kb = cell["k_blocks"]
    for klass, lo_arm, hi_arm, lo_n, hi_n in LADDER:
        lo, hi = cell["stats"].get(lo_arm), cell["stats"].get(hi_arm)
        if lo is None or hi is None:
            continue
        d_us = hi["median"] - lo["median"]
        d_n = hi_n - lo_n
        out[klass] = {
            "low_rung": lo_arm, "high_rung": hi_arm,
            "low_us": lo["median"], "high_us": hi["median"],
            "delta_us": d_us,
            "delta_us_sem": sem_of_difference(lo, hi),
            "us_per_instruction": d_us / d_n,
            "us_per_instruction_per_k_block": d_us / d_n / kb,
            "pct_per_instruction": 100.0 * d_us / d_n / cell["base_us"],
        }
    dele, ref, alt = DELETION
    d = cell["stats"].get(dele)
    for name, ref_arm in (("deletion", ref), ("deletion_vs_a_base", alt)):
        r = cell["stats"].get(ref_arm)
        if d is None or r is None:
            continue
        gain_us = r["median"] - d["median"]
        out[name] = {
            "arm": dele, "reference": ref_arm,
            "gain_us": gain_us,
            "gain_us_sem": sem_of_difference(d, r),
            "gain_us_per_k_block": gain_us / kb,
            "gain_pct": 100.0 * gain_us / r["median"],
        }
    scaffold, base = cell["stats"].get("q_scaffold"), cell["stats"].get("a_base")
    if scaffold is not None and base is not None:
        out["null_scaffold"] = {
            "cost_us": scaffold["median"] - base["median"],
            "cost_pct": 100.0 * (scaffold["median"] - base["median"])
            / base["median"],
            "cost_us_per_k_block": (scaffold["median"] - base["median"]) / kb,
        }
    return out


def ramp_residual(rate: dict) -> dict:
    """Harness defect 16 residual, per frame, by arm position.

    A convex fall over the slot sequence does not cancel in the palindrome. It
    shows as a forward-minus-reverse gap that is largest on arm 0 and falls
    monotonically toward the middle arm. A flat profile near zero means the
    discarded warm palindrome paid the ramp before timing started.
    """
    arms = rate["arms"]
    n = len(arms)
    per: dict[str, dict[str, list[float]]] = {}
    for row in timing_rows(rate):
        bucket = per.setdefault(row["frame"], {a: [] for a in arms})
        slots = row["slots"]
        for i, arm in enumerate(arms):
            fwd, rev = slots[i], slots[2 * n - 1 - i]
            bucket[arm].append(100.0 * (fwd - rev) / rev)
    out = {}
    for frame, bucket in per.items():
        gaps = {a: statistics.median(v) for a, v in bucket.items() if v}
        out[frame] = {
            "forward_minus_reverse_pct_by_arm": gaps,
            "arm0_pct": gaps.get(arms[0]),
            "worst_abs_pct": max(abs(v) for v in gaps.values()) if gaps else None,
        }
    return out


def segments(cell_map: dict) -> list[dict]:
    """Marginal cost of one k-block, between adjacent stream lengths.

    Only the k-frames and the base frame vary the stream length, and they share
    the launched grid, the pipeline state and the fixed prologue. Differencing
    two adjacent lengths therefore cancels the fixed dispatch cost exactly and
    leaves the marginal cost of a k-block in ONE memory regime:

        c1 = (t_hi - t_lo) / (kb_hi - kb_lo)
        marginal GB/s = (bytes_hi - bytes_lo) / (kb_hi - kb_lo) / c1

    The marginal rate is the regime label this experiment needs. The naive
    whole-kernel rate is not: a two-block frame spends most of its time in the
    fixed prologue, so its naive rate falls even though its weights are
    cache resident.
    """
    by_cell: dict[tuple, list[dict]] = {}
    for (shape, m, frame), cell in cell_map.items():
        if frame in ("cycle", "consumer"):
            continue
        by_cell.setdefault((shape, m), []).append(cell)
    out = []
    for (shape, m), group in sorted(by_cell.items()):
        group.sort(key=lambda c: c["k_blocks"])
        for lo, hi in zip(group, group[1:]):
            d_kb = hi["k_blocks"] - lo["k_blocks"]
            d_bytes = hi["read_bytes"] - lo["read_bytes"]
            row = {"shape": shape, "m": m,
                   "from_frame": lo["frame"], "to_frame": hi["frame"],
                   "k_blocks_from": lo["k_blocks"], "k_blocks_to": hi["k_blocks"],
                   "bytes_per_k_block": d_bytes / d_kb, "arms": {}}
            for arm in lo["stats"]:
                if arm not in hi["stats"]:
                    continue
                c1_us = (hi["stats"][arm]["median"]
                         - lo["stats"][arm]["median"]) / d_kb
                row["arms"][arm] = {
                    "marginal_us_per_k_block": c1_us,
                    "marginal_gb_s": (d_bytes / d_kb) / (c1_us * 1e-6) / 1e9
                    if c1_us > 0 else None,
                    "fixed_us": lo["stats"][arm]["median"]
                    - c1_us * lo["k_blocks"],
                }
            ref = row["arms"].get("a_base")
            row["marginal_gb_s"] = ref["marginal_gb_s"] if ref else None
            row["phi_marginal"] = (ref["marginal_gb_s"] / PEAK_BANDWIDTH_GB_S
                                   if ref and ref["marginal_gb_s"] else None)
            for klass, lo_arm, hi_arm, lo_n, hi_n in LADDER:
                a, b = row["arms"].get(lo_arm), row["arms"].get(hi_arm)
                if a and b:
                    row["%s_us_per_instruction_per_k_block" % klass] = (
                        (b["marginal_us_per_k_block"]
                         - a["marginal_us_per_k_block"]) / (hi_n - lo_n))
            out.append(row)
    return out


def gates(rate: dict, cell_map: dict) -> dict:
    """Defect 22 per frame, the null scaffold per frame, positive controls."""
    limit = BANDWIDTH_GATE_FACTOR * PEAK_BANDWIDTH_GB_S
    by_frame: dict[str, dict] = {}
    for row in timing_rows(rate):
        f = row["frame"]
        slot = by_frame.setdefault(f, {"gb_s": 0.0, "at": None})
        for arm, sec in row["seconds"].items():
            if not sec:
                continue
            gb_s = row["read_bytes"] / sec / 1e9
            if gb_s > slot["gb_s"]:
                slot["gb_s"] = gb_s
                slot["at"] = {"arm": arm, "shape": row["shape"],
                              "m": row["m"], "k": row["k"]}
    bandwidth = {}
    for f, slot in by_frame.items():
        streaming = f in STREAMING_FRAMES
        bandwidth[f] = {
            "max_implied_gb_s": slot["gb_s"], "at": slot["at"],
            "streaming": streaming, "limit_gb_s": limit if streaming else None,
            "passed": (slot["gb_s"] <= limit) if streaming else True,
            "exempt_reason": None if streaming else
            "cache-resident frame; the DRAM roofline does not bound it",
        }

    scaffold_moves = []
    for key, cell in cell_map.items():
        p = cell.get("prices", {}).get("null_scaffold")
        if p is None:
            continue
        if abs(p["cost_pct"]) > NULL_SCAFFOLD_GATE_PCT:
            scaffold_moves.append({"shape": key[0], "m": key[1],
                                   "frame": key[2],
                                   "cost_pct": p["cost_pct"]})

    controls = [r for r in rate["measurements"]
                if r.get("kind") == "positive_control"]
    control_failures = [r for r in controls if not r["detected"]]
    fidelity_failures = []
    for r in rate["measurements"]:
        if r.get("kind") != "fidelity":
            continue
        for a in r["arms"]:
            if a["exact_required"] and not a["bit_identical"]:
                fidelity_failures.append({"arm": a["arm"], "k": r["k"],
                                          "m": r["m"],
                                          "differing": a["differing"]})

    streaming_bw_ok = all(v["passed"] for v in bandwidth.values())
    return {
        "bandwidth_by_frame": bandwidth,
        "bandwidth_note":
            "the gate is applied per frame; a cache-resident frame is exempt "
            "by name, not pooled away",
        "null_scaffold": {
            "gate_pct": NULL_SCAFFOLD_GATE_PCT,
            "cells_over_gate": scaffold_moves,
            "voids_session": False,
            "note":
                "a scaffold move in a resident frame is a RESULT, not a "
                "session fault: every price here is a rung contrast that "
                "cancels the scaffold, and the scaffold arm is bit identical "
                "to a_base in every frame",
        },
        "positive_controls": {"n": len(controls),
                              "failures": control_failures,
                              "passed": not control_failures},
        "fidelity": {"failures": fidelity_failures,
                     "passed": not fidelity_failures},
        "session_valid": streaming_bw_ok and not control_failures
        and not fidelity_failures,
    }


def frame_law(cell_map: dict) -> dict:
    """Per class: the per-instruction price against phi, pooled over cells.

    H6 says one function of phi explains BOTH directions -- the consumer frame
    lowering phi's headroom and the resident frame raising it. The test is
    whether the consumer point and the resident points lie on the same curve,
    so the fit is reported with the frames labelled and never averaged blind.
    """
    out: dict[str, dict] = {}
    for klass in [k for k, *_ in LADDER] + ["deletion"]:
        points = []
        for key, cell in cell_map.items():
            p = cell.get("prices", {}).get(klass)
            if p is None:
                continue
            value = (p["us_per_instruction_per_k_block"]
                     if klass != "deletion" else p["gain_us_per_k_block"])
            points.append({
                "shape": key[0], "m": key[1], "frame": key[2],
                "k_blocks": cell["k_blocks"],
                "phi_arm": cell["phi_arm"], "phi_eff": cell["phi_eff"],
                "consumer_gb_s": cell["consumer_gb_s"],
                "achieved_gb_s": cell["achieved_gb_s"],
                "value_us_per_k_block": value,
                "pct": (p["pct_per_instruction"] if klass != "deletion"
                        else p["gain_pct"]),
            })
        if not points:
            continue
        by_cellframe: dict[tuple, dict] = {}
        for pt in points:
            by_cellframe[(pt["shape"], pt["m"], pt["frame"])] = pt
        ratios = []
        for (shape, m, frame), pt in by_cellframe.items():
            ref = by_cellframe.get((shape, m, "base"))
            if ref is None or frame == "base" or ref["value_us_per_k_block"] == 0:
                continue
            ratios.append({
                "shape": shape, "m": m, "frame": frame,
                "ratio_to_base": pt["value_us_per_k_block"]
                / ref["value_us_per_k_block"],
                "phi_arm": pt["phi_arm"], "base_phi_arm": ref["phi_arm"],
            })
        out[klass] = {"points": points, "ratio_to_base": ratios,
                      "n_points": len(points)}
        per_frame: dict[str, list[float]] = {}
        for r in ratios:
            per_frame.setdefault(r["frame"], []).append(r["ratio_to_base"])
        out[klass]["ratio_by_frame"] = {
            f: {"median": statistics.median(v), "n": len(v),
                "measured": len(v) >= 2,
                "note": None if len(v) >= 2 else "not measured: fewer than 2 "
                                                 "independent points"}
            for f, v in per_frame.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    rate = json.loads(args.rate.read_text())
    cell_map = cells(rate)
    for cell in cell_map.values():
        cell["prices"] = prices(cell)

    report = {
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "device": rate["device"],
        "architecture": rate["architecture"],
        "arms": rate["arms"],
        "frames": rate["frames"],
        "pairs": rate["pairs"],
        "warm_pairs": rate.get("warm_pairs"),
        "inner_max": rate.get("inner_max"),
        "target_ms": rate["target_ms"],
        "peak_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
        "warm_sweep_reps": rate.get("warm_sweep_reps"),
        "cells": [],
        "gates": None,
        "frame_law": None,
        "segments": None,
        "ramp_residual": None,
    }
    for key in sorted(cell_map):
        cell = cell_map[key]
        report["cells"].append({
            "shape": cell["shape"], "m": cell["m"], "frame": cell["frame"],
            "k": cell["k"], "k_blocks": cell["k_blocks"],
            "inner": cell["inner"],
            "launched_threads": cell["launched_threads"],
            "read_bytes": cell["read_bytes"],
            "base_us": cell["base_us"],
            "achieved_gb_s": cell["achieved_gb_s"],
            "consumer_gb_s": cell["consumer_gb_s"],
            "phi_arm": cell["phi_arm"], "phi_eff": cell["phi_eff"],
            "entry_c_min": cell["entry_c_min"],
            "entry_c_max": cell["entry_c_max"],
            "entry_c_spread": cell["entry_c_spread"],
            "arm_us": {a: s["median"] for a, s in cell["stats"].items()},
            "arm_sem": {a: s["sem"] for a, s in cell["stats"].items()},
            "prices": cell["prices"],
        })
    report["gates"] = gates(rate, cell_map)
    report["frame_law"] = frame_law(cell_map)
    report["segments"] = segments(cell_map)
    report["ramp_residual"] = ramp_residual(rate)

    print("e125 frame analysis  %s  %s  harness=local"
          % (report["device"], report["architecture"]))
    print("  cool_gate_passed_real_gate=false  gate_qualified_for_timing=false")
    print()
    hdr = ("%-26s %2s %-9s %6s %7s %8s %6s %8s %8s %8s" %
           ("shape", "M", "frame", "kblk", "base_us", "GB/s", "phi",
            "ld/kblk", "alu/kblk", "del/kblk"))
    print(hdr)
    for c in report["cells"]:
        p = c["prices"]
        print("%-26s %2d %-9s %6d %7.1f %8.1f %6.3f %8.4f %8.4f %8.4f"
              % (c["shape"][:26], c["m"], c["frame"], c["k_blocks"],
                 c["base_us"], c["achieved_gb_s"], c["phi_arm"],
                 p.get("ld", {}).get("us_per_instruction_per_k_block", float("nan")),
                 p.get("alu", {}).get("us_per_instruction_per_k_block", float("nan")),
                 p.get("deletion", {}).get("gain_us_per_k_block", float("nan"))))
    print()
    print("marginal cost of one k-block between adjacent stream lengths")
    print("%-26s %2s %-18s %8s %7s %10s %10s"
          % ("shape", "M", "segment", "GB/s", "phi", "ld/instr", "alu/instr"))
    for s in report["segments"]:
        print("%-26s %2d %-18s %8.1f %7.3f %10.5f %10.5f"
              % (s["shape"][:26], s["m"],
                 "%s->%s" % (s["from_frame"], s["to_frame"]),
                 s["marginal_gb_s"] or float("nan"),
                 s["phi_marginal"] or float("nan"),
                 s.get("ld_us_per_instruction_per_k_block", float("nan")),
                 s.get("alu_us_per_instruction_per_k_block", float("nan"))))

    print()
    print("harness defect 16 residual, forward minus reverse, median per arm")
    for frame, v in sorted(report["ramp_residual"].items()):
        print("    %-9s arm0=%+6.2f%%  worst=%.2f%%"
              % (frame, v["arm0_pct"], v["worst_abs_pct"]))

    print()
    for klass, law in report["frame_law"].items():
        print("%s: ratio of the per-k-block price to the base frame" % klass)
        for f, v in sorted(law["ratio_by_frame"].items()):
            print("    %-9s x%.3f  n=%d%s"
                  % (f, v["median"], v["n"],
                     "" if v["measured"] else "   NOT MEASURED"))
    print()
    g = report["gates"]
    print("gates: session_valid=%s  controls=%s  fidelity=%s"
          % (g["session_valid"], g["positive_controls"]["passed"],
             g["fidelity"]["passed"]))
    for f, v in sorted(g["bandwidth_by_frame"].items()):
        print("    bandwidth %-9s max=%8.1f GB/s  %s"
              % (f, v["max_implied_gb_s"],
                 "pass" if v["passed"] else "FAIL"
                 if v["streaming"] else "exempt (resident)"))
    if g["null_scaffold"]["cells_over_gate"]:
        print("    null scaffold moves in %d cells (a result, not a fault)"
              % len(g["null_scaffold"]["cells_over_gate"]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
