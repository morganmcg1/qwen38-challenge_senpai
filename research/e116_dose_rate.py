#!/usr/bin/env python3
"""E116 rung 1: what does one dose unit cost, and does the census see it?

    usage: research/e116_dose_rate.py --isolated TAG --insitu-null TAG
               --insitu-dosed TAG [--json OUT]

TWO RATES, AND THEY ARE NOT THE SAME NUMBER.

  `isolated_us_per_unit`   From a leg run with `MLX_E58_BUFFER_LIMIT_OPS=0`,
                           where one command buffer holds one dispatch. This is
                           the campaign's standard kernel frame: E107's
                           410.93 us for this exact cell, alphonse's percent of
                           wide-QMV kernel time, and every other published
                           per-kernel rate live in it. Measure it at `k = 1`,
                           because at `k > 1` the dose dispatches are
                           independent and can overlap each other, which
                           inflates every per-dispatch interval without
                           inflating the round.

  `insitu_us_per_unit`     `(round GPU busy at k) - (round GPU busy at 0)`,
                           divided by `k`, from two legs run with the DEFAULT
                           command-buffer geometry. This is what the dose
                           really adds to a round that packs many ops per
                           buffer, and it is the quantity a round can absorb.

The ratio of the two is the dose's own self-overlap, and it is reported. It
matters: if the in-situ rate is well below the isolated rate, then part of any
absorption coefficient built on the isolated rate is the dose overlapping
itself rather than the round having slack.

RULE 37. Every rate here is an M = 1 rate. The dose is M = 1 by design, because
that fixes the byte count exactly at 100,270,080 B and makes the injected work
a pure weight-load. It is NOT a scored-width rate and must never be read as
one.

A census leg is never a timing leg.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

DOSE_PHASE = "e116_dose"


def load(tag: str) -> dict:
    path = pathlib.Path("research/out") / tag / "census.jsonl"
    if not path.exists():
        sys.exit(f"e116_dose_rate: no census at {path}")
    rounds: collections.Counter[str] = collections.Counter()
    phase_us: dict[str, float] = collections.defaultdict(float)
    phase_disp: dict[str, float] = collections.defaultdict(float)
    kernels: dict[str, dict[str, float]] = {}
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime":
            continue
        for width in {k.split("|", 1)[0] for k in rec.get("by_width_phase", {})}:
            rounds[width] += rec.get("rounds", 1)
        for key, value in rec.get("by_width_phase", {}).items():
            phase_us[key] += value.get("gpu_ns", 0) / 1e3
            phase_disp[key] += value.get("dispatches", 0)
        for key, value in rec.get("exclusive_kernels", {}).items():
            entry = kernels.setdefault(
                key, {"buffers": 0, "gpu_ns": 0.0, "min_ns": float("inf"),
                      "max_ns": 0.0})
            entry["buffers"] += value["buffers"]
            entry["gpu_ns"] += value["gpu_ns"]
            entry["min_ns"] = min(entry["min_ns"], value["min_ns"])
            entry["max_ns"] = max(entry["max_ns"], value["max_ns"])
    return {"tag": tag, "rounds": dict(rounds), "phase_us": dict(phase_us),
            "phase_disp": dict(phase_disp), "kernels": kernels}


def width_totals(leg: dict) -> dict[str, float]:
    totals: dict[str, float] = collections.defaultdict(float)
    for key, value in leg["phase_us"].items():
        totals[key.split("|", 1)[0]] += value
    return dict(totals)


def dose_totals(leg: dict) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key, value in leg["phase_us"].items():
        width, _, phase = key.partition("|")
        if phase != DOSE_PHASE:
            continue
        out[width] = (value, leg["phase_disp"][key])
    return out


def decode_widths(rounds: dict[str, int]) -> list[str]:
    return [w for w in rounds if w != "w0"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolated", required=True)
    ap.add_argument("--isolated-units", type=int, default=1)
    ap.add_argument("--insitu-null", required=True)
    ap.add_argument("--insitu-dosed", required=True)
    ap.add_argument("--insitu-units", type=int, default=4)
    ap.add_argument("--e107-reference-us", type=float, default=410.93)
    ap.add_argument("--json")
    args = ap.parse_args()

    iso = load(args.isolated)
    null = load(args.insitu_null)
    dosed = load(args.insitu_dosed)

    iso_kernels = {k: v for k, v in iso["kernels"].items()
                   if f"|{DOSE_PHASE}|" in k}
    if not iso_kernels:
        sys.exit(f"e116_dose_rate: {args.isolated} shows NO dose dispatches; "
                 "the dose does not exist in this build")
    iso_ns = sum(v["gpu_ns"] for v in iso_kernels.values())
    iso_buffers = sum(v["buffers"] for v in iso_kernels.values())
    isolated_us_per_unit = iso_ns / 1e3 / iso_buffers
    names = sorted({k.split("|", 2)[2] for k in iso_kernels})

    null_widths = width_totals(null)
    dosed_widths = width_totals(dosed)
    shared = [w for w in sorted(set(null_widths) & set(dosed_widths),
                                key=lambda x: int(x[1:])) if w != "w0"]
    per_width = []
    weighted_delta, weighted_rounds = 0.0, 0
    for width in shared:
        n0, n4 = null["rounds"][width], dosed["rounds"][width]
        a = null_widths[width] / n0
        b = dosed_widths[width] / n4
        per_width.append({"width": width, "null_rounds": n0,
                          "dosed_rounds": n4, "null_us_per_round": a,
                          "dosed_us_per_round": b, "delta_us_per_round": b - a,
                          "delta_us_per_unit": (b - a) / args.insitu_units})
        # Weight by the DOSED leg's round count: that is the leg the injected
        # work actually happened in.
        weighted_delta += (b - a) * n4
        weighted_rounds += n4
    insitu_us_per_round = (weighted_delta / weighted_rounds
                           if weighted_rounds else float("nan"))
    insitu_us_per_unit = insitu_us_per_round / args.insitu_units

    dosed_phase = dose_totals(dosed)
    dosed_phase_us = sum(v[0] for v in dosed_phase.values())
    dosed_phase_disp = sum(v[1] for v in dosed_phase.values())
    dosed_phase_rounds = sum(dosed["rounds"][w] for w in dosed_phase)

    deviation = 100.0 * (isolated_us_per_unit - args.e107_reference_us) \
        / args.e107_reference_us

    out = {
        "harness": "local",
        "experiment":
            "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "rung": 1,
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "dose_cell": "affine 4-bit group-64 quantizedMM, mlp.gate_up shape "
                     "K=5120 -> N=34816, M=1",
        "dose_resident_bytes": 100270080,
        "dose_kernel_names": names,
        "isolated_leg": args.isolated,
        "isolated_units_per_round": args.isolated_units,
        "isolated_dispatches": iso_buffers,
        "isolated_us_per_unit": isolated_us_per_unit,
        "isolated_min_us": min(v["min_ns"] for v in iso_kernels.values()) / 1e3,
        "isolated_max_us": max(v["max_ns"] for v in iso_kernels.values()) / 1e3,
        "e107_reference_us": args.e107_reference_us,
        "deviation_from_e107_percent": deviation,
        "agrees_with_e107_within_10_percent": abs(deviation) <= 10.0,
        "insitu_null_leg": args.insitu_null,
        "insitu_dosed_leg": args.insitu_dosed,
        "insitu_units_per_round": args.insitu_units,
        "insitu_per_width": per_width,
        "insitu_delta_us_per_dosed_round": insitu_us_per_round,
        "insitu_us_per_unit": insitu_us_per_unit,
        "insitu_over_isolated": insitu_us_per_unit / isolated_us_per_unit,
        "dosed_leg_dose_phase_us_per_round":
            dosed_phase_us / dosed_phase_rounds if dosed_phase_rounds else
            float("nan"),
        "dosed_leg_dose_phase_dispatches_per_round":
            dosed_phase_disp / dosed_phase_rounds if dosed_phase_rounds else
            float("nan"),
    }

    print("E116 rung 1 -- the dose unit rate   harness=local")
    print("  census legs; timing_valid=false, not gate qualified, not a score")
    print(f"  cell: {out['dose_cell']}")
    print(f"  kernel(s) seen: {names}")
    print()
    print(f"  ISOLATED  leg {args.isolated}, one dispatch per command buffer,"
          f" k={args.isolated_units}")
    print(f"            {iso_buffers} dispatches,"
          f" {isolated_us_per_unit:.2f} us per unit"
          f"  (min {out['isolated_min_us']:.2f}, max"
          f" {out['isolated_max_us']:.2f})")
    print(f"            E107 reference {args.e107_reference_us:.2f} us"
          f"  ->  {deviation:+.2f} %"
          f"  {'AGREES within 10 %' if out['agrees_with_e107_within_10_percent'] else 'DISAGREES by more than 10 %'}")
    print()
    print(f"  IN SITU   {args.insitu_dosed} minus {args.insitu_null},"
          f" default command-buffer geometry, k={args.insitu_units}")
    print(f"{'width':>7} {'n0':>4} {'nk':>4} {'k=0 us/rnd':>13}"
          f" {'k=%d us/rnd' % args.insitu_units:>13} {'delta':>11}"
          f" {'per unit':>10}")
    for row in per_width:
        print(f"{row['width']:>7} {row['null_rounds']:>4}"
              f" {row['dosed_rounds']:>4} {row['null_us_per_round']:>13,.1f}"
              f" {row['dosed_us_per_round']:>13,.1f}"
              f" {row['delta_us_per_round']:>+11,.1f}"
              f" {row['delta_us_per_unit']:>+10,.1f}")
    print(f"  round-weighted: {insitu_us_per_round:+,.1f} us per dosed round,"
          f" {insitu_us_per_unit:+,.1f} us per unit")
    print(f"  in-situ / isolated = {out['insitu_over_isolated']:.3f}"
          "   (below 1.0 means the dose overlaps itself or the round)")
    print(f"  dosed leg's own e116_dose phase:"
          f" {out['dosed_leg_dose_phase_us_per_round']:,.1f} us/round over"
          f" {out['dosed_leg_dose_phase_dispatches_per_round']:.2f}"
          " dispatches/round")

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
