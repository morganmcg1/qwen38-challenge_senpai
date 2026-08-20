#!/usr/bin/env python3
"""E80 rung 1 -- the mandatory hard gate.

Joins one E71 width-tax session (wall clock, `census.json`) with the E80
GPU-time ledger for the same session (`census.jsonl`) and prints the gate table.

The join key is the window label the instrument writes, `arm|w<width>|o<order>`,
which is exactly the `(arm, width, order)` triple of an E71 block.

Three pre-registered sub-gates, each reported row by row:

  G0  harness reproduction. This session's WALL-clock F(1), F(6) and family
      taxes against the published E71 values. G0 asks whether the harness still
      measures what E71 measured. It uses no GPU-time evidence at all, so a G0
      failure means the base or the host moved and G1/G2 cannot be interpreted.

  G1  GPU-time family attribution. Per-family GPU-time tax against the published
      E71 wall-clock tax, tolerance 10 %.

  G2  GPU-time level. In-round GPU time at width 1 and width 6 against the
      published F(1) and F(6), tolerance 5 %.

G0 is not in the assignment. It is added because G1 and G2 compare a GPU-time
quantity with a wall-clock quantity, and without G0 a failure cannot be
attributed between "the instrument is wrong" and "the session is not E71".

usage:
  research/e80_gate_report.py --census research/out/TAG-gpu/census.jsonl \\
      --e71 research/out/TAG/census.json [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import defaultdict

# --- published E71 reference, quoted in the E80 assignment -------------------
# Isolated per-round verify cost, milliseconds, wall clock.
E71_F1_MS = 64.706
E71_F6_MS = 122.110
E71_M6_WIDTH_TAX_MS = 57.404  # F(6) - F(1)

# Family taxes at M = 6, milliseconds, wall clock. `mlp_gate_up` is derived by
# E71 as `mlp_all - mlp_down`, so it has no arm of its own.
E71_FAMILY_TAX_MS = {
    "mlp_gate_up": 20.913,
    "mlp_down": 16.887,
    "gdn_out_proj": 3.313,
    "fa_o_proj": 1.161,
    "lm_head": 2.142,
}
# Arms that measure a disjoint slice of the tax. `mlp_all` contains `mlp_down`,
# so summing both would double-count; E71 excludes `mlp_all` from the sum and
# uses the derived `mlp_gate_up` instead.
E71_DISJOINT = ("mlp_gate_up", "mlp_down", "gdn_out_proj", "fa_o_proj", "lm_head")
E71_ATTRIBUTED_FRACTION = 0.7737

G1_TOLERANCE = 0.10
G2_TOLERANCE = 0.05
G0_TOLERANCE = 0.05


def load_gpu_windows(path: pathlib.Path):
    """Return {(arm, width, order): record} from the GPU-time JSONL."""
    windows = {}
    other = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "gputime":
            continue
        label = rec.get("window") or ""
        parts = label.split("|")
        if len(parts) != 3 or not parts[1].startswith("w"):
            other.append(rec)
            continue
        arm, width_tag, order_tag = parts
        try:
            key = (arm, int(width_tag[1:]), int(order_tag[1:]))
        except ValueError:
            other.append(rec)
            continue
        windows[key] = rec
    return windows, other


def verify_ms_per_rep(rec, width: int, reps: int) -> float:
    """GPU time charged to the timed rounds of one block, per rep.

    Read the `verify_block` PHASE bucket, never the window's `gpu_busy_ns`.
    `gpu_busy_ns` is a union over the whole snapshot delta, and the E71 harness
    runs a 768-token seed prefill plus warmup reps between windows. That work
    carries the `outside` phase, so the phase bucket excludes it and the union
    does not.
    """
    bucket = rec["by_width_phase"].get(f"w{width}|verify_block")
    if bucket is None:
        return None
    return bucket["gpu_ns"] / 1e6 / reps


def outside_ms(rec) -> float:
    """GPU time in the same snapshot that no round claimed."""
    return sum(v["gpu_ns"] for k, v in rec["by_width_phase"].items()
               if k.endswith("|outside")) / 1e6


def block_key(block):
    return (block["arm"], block["width"], block["order"])


def abba_tax(blocks, arm_name, width, value_of):
    """E71's family tax: mean(baseline blocks) - mean(arm blocks) in one quartet.

    Returns None when the session did not run that (arm, width) quartet.
    """
    quartets = []
    ordered = sorted(blocks, key=lambda b: b["order"])
    for i in range(len(ordered) - 3):
        window = ordered[i:i + 4]
        names = [b["arm"] for b in window]
        widths = {b["width"] for b in window}
        if widths != {width}:
            continue
        if names == ["baseline", arm_name, arm_name, "baseline"]:
            base = statistics.mean([value_of(window[0]), value_of(window[3])])
            arm = statistics.mean([value_of(window[1]), value_of(window[2])])
            quartets.append(base - arm)
    if not quartets:
        return None
    return statistics.mean(quartets)


def curve_level(blocks, width, value_of):
    """Mean over the baseline curve blocks at one width."""
    vals = [value_of(b) for b in blocks
            if b["arm"] == "baseline" and b["width"] == width]
    return statistics.mean(vals) if vals else None


def verdict(observed, reference, tolerance):
    if observed is None:
        return {"observed": None, "reference": reference, "relative": None,
                "pass": False, "note": "not measured in this session"}
    rel = (observed - reference) / reference if reference else float("nan")
    return {"observed": observed, "reference": reference, "relative": rel,
            "pass": abs(rel) <= tolerance, "note": ""}


def render(title, rows, tolerance):
    print(f"\n### {title} (tolerance {tolerance * 100:.0f} %)\n")
    print("| row | observed | E71 reference | delta | verdict |")
    print("|---|---:|---:|---:|---|")
    for name, v in rows.items():
        if v["observed"] is None:
            print(f"| {name} | -- | {v['reference']:.3f} | -- | **FAIL** ({v['note']}) |")
            continue
        print(f"| {name} | {v['observed']:.3f} | {v['reference']:.3f} "
              f"| {v['relative'] * 100:+.1f} % "
              f"| {'pass' if v['pass'] else '**FAIL**'} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True, type=pathlib.Path)
    ap.add_argument("--e71", required=True, type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    session = json.loads(args.e71.read_text())
    blocks = session["blocks"]
    reps = session["identity"]["reps_per_block"]
    windows, unlabelled = load_gpu_windows(args.census)

    # --- join -----------------------------------------------------------
    matched, missing, unphased = 0, [], []
    for b in blocks:
        rec = windows.get(block_key(b))
        if rec is None:
            missing.append(block_key(b))
            continue
        gpu_ms = verify_ms_per_rep(rec, b["width"], reps)
        if gpu_ms is None:
            unphased.append(block_key(b))
            continue
        b["_gpu_ms"] = gpu_ms
        b["_outside_ms"] = outside_ms(rec)
        b["_gpu_busy_ms"] = rec["gpu_busy_ns"] / 1e6
        b["_gpu_idle_ms"] = rec["gpu_idle_ns"] / 1e6 / reps
        b["_gpu_span_ms"] = rec["gpu_span_ns"] / 1e6 / reps
        b["_verify_dispatches"] = (
            rec["by_width_phase"][f"w{b['width']}|verify_block"]["dispatches"] / reps)
        b["_verify_buffers"] = (
            rec["by_width_phase"][f"w{b['width']}|verify_block"]["buffers"] / reps)
        b["_unmapped"] = rec.get("unmapped_encoder_dispatches", 0)
        b["_untracked"] = rec.get("untracked_buffers", 0)
        b["_zero_time"] = rec.get("zero_time_buffers", 0)
        b["_committed"] = rec.get("committed_total", 0)
        b["_completed"] = rec.get("completed_total", 0)
        matched += 1

    print(f"E71 blocks = {len(blocks)}  GPU windows = {len(windows)}  "
          f"joined = {matched}  unjoined blocks = {len(missing)}  "
          f"joined without a verify_block phase = {len(unphased)}  "
          f"unlabelled GPU records = {len(unlabelled)}")
    if missing:
        print(f"  first unjoined: {missing[:5]}")
    if unphased:
        print(f"  first unphased: {unphased[:5]}")

    joined = [b for b in blocks if "_gpu_ms" in b]
    if not joined:
        print("\nFATAL: no block joined a GPU-time window. The gate cannot run.")
        return 2

    # --- instrument health ----------------------------------------------
    unmapped = sum(b["_unmapped"] for b in joined)
    untracked = sum(b["_untracked"] for b in joined)
    zero_time = sum(b["_zero_time"] for b in joined)
    leaked = sum(b["_committed"] - b["_completed"] for b in joined)
    print(f"\ninstrument health: unmapped_encoder_dispatches={unmapped} "
          f"untracked_buffers={untracked} zero_time_buffers={zero_time} "
          f"undrained_buffers={leaked}")

    wall = lambda b: b["seconds_median"] * 1e3
    gpu = lambda b: b["_gpu_ms"]

    # --- G0: harness reproduction, wall clock ----------------------------
    g0 = {
        "F(1) wall ms": verdict(curve_level(blocks, 1, wall), E71_F1_MS, G0_TOLERANCE),
        "F(6) wall ms": verdict(curve_level(blocks, 6, wall), E71_F6_MS, G0_TOLERANCE),
    }
    f1w, f6w = curve_level(blocks, 1, wall), curve_level(blocks, 6, wall)
    if f1w is not None and f6w is not None:
        g0["F(6)-F(1) wall ms"] = verdict(f6w - f1w, E71_M6_WIDTH_TAX_MS, G0_TOLERANCE)
    for fam, ref in E71_FAMILY_TAX_MS.items():
        if fam == "mlp_gate_up":
            all_tax = abba_tax(blocks, "mlp_all", 6, wall)
            down_tax = abba_tax(blocks, "mlp_down", 6, wall)
            obs = None if all_tax is None or down_tax is None else all_tax - down_tax
        else:
            obs = abba_tax(blocks, fam, 6, wall)
        g0[f"{fam} wall tax ms"] = verdict(obs, ref, G1_TOLERANCE)
    render("G0 -- harness reproduction, WALL clock", g0, G0_TOLERANCE)

    # --- G1: GPU-time family attribution ---------------------------------
    g1 = {}
    for fam, ref in E71_FAMILY_TAX_MS.items():
        if fam == "mlp_gate_up":
            all_tax = abba_tax(blocks, "mlp_all", 6, gpu)
            down_tax = abba_tax(blocks, "mlp_down", 6, gpu)
            obs = None if all_tax is None or down_tax is None else all_tax - down_tax
        else:
            obs = abba_tax(blocks, fam, 6, gpu)
        g1[f"{fam} GPU tax ms"] = verdict(obs, ref, G1_TOLERANCE)
    render("G1 -- family attribution, GPU time", g1, G1_TOLERANCE)

    attributed = [g1[f"{f} GPU tax ms"]["observed"] for f in E71_DISJOINT]
    if all(v is not None for v in attributed):
        f1g, f6g = curve_level(blocks, 1, gpu), curve_level(blocks, 6, gpu)
        gpu_width_tax = None if f1g is None or f6g is None else f6g - f1g
        total = sum(attributed)
        print(f"\nattributed GPU tax = {total:.3f} ms over the five disjoint arms")
        if gpu_width_tax:
            print(f"GPU F(6)-F(1) = {gpu_width_tax:.3f} ms  ->  attributed fraction "
                  f"= {total / gpu_width_tax:.4f}  (E71 wall clock: "
                  f"{E71_ATTRIBUTED_FRACTION:.4f})")

    # --- G2: GPU-time level ----------------------------------------------
    g2 = {
        "F(1) GPU ms": verdict(curve_level(blocks, 1, gpu), E71_F1_MS, G2_TOLERANCE),
        "F(6) GPU ms": verdict(curve_level(blocks, 6, gpu), E71_F6_MS, G2_TOLERANCE),
    }
    f1g, f6g = curve_level(blocks, 1, gpu), curve_level(blocks, 6, gpu)
    if f1g is not None and f6g is not None:
        g2["F(6)-F(1) GPU ms"] = verdict(f6g - f1g, E71_M6_WIDTH_TAX_MS, G2_TOLERANCE)
    render("G2 -- level, GPU time", g2, G2_TOLERANCE)

    # The GPU-to-wall ratio is the headline reconciliation number: it says how
    # much of a round's wall clock the GPU is actually busy.
    print("\n### GPU time as a share of wall clock, baseline blocks\n")
    print("| width | wall ms/rep | verify GPU ms/rep | GPU/wall | unattributed ms/rep "
          "| dispatches/rep | buffers/rep |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for width in sorted({b["width"] for b in joined if b["arm"] == "baseline"}):
        w = curve_level(blocks, width, wall)
        g = curve_level(blocks, width, gpu)
        disp = curve_level(blocks, width, lambda b: b["_verify_dispatches"])
        bufs = curve_level(blocks, width, lambda b: b["_verify_buffers"])
        if w and g:
            print(f"| {width} | {w:.3f} | {g:.3f} | {g / w:.3f} | {w - g:.3f} "
                  f"| {disp:.0f} | {bufs:.1f} |")

    all_rows = {"G0": g0, "G1": g1, "G2": g2}
    overall = {k: all(v["pass"] for v in rows.values()) for k, rows in all_rows.items()}
    print("\n### verdict\n")
    for k, ok in overall.items():
        print(f"- {k}: {'PASS' if ok else 'FAIL'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "census": str(args.census),
            "e71": str(args.e71),
            "cool_gate_passed_real_gate": session.get("cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": session.get("gate_qualified_for_timing"),
            "official_or_ranked_score": session.get("official_or_ranked_score"),
            "identity": session["identity"],
            "blocks_total": len(blocks),
            "blocks_joined": matched,
            "instrument_health": {
                "unmapped_encoder_dispatches": unmapped,
                "untracked_buffers": untracked,
                "zero_time_buffers": zero_time,
                "undrained_buffers": leaked,
            },
            "gates": all_rows,
            "gate_pass": overall,
        }, indent=2, default=float) + "\n")
        print(f"\nwrote {args.json}")

    return 0 if all(overall.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
