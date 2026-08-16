#!/usr/bin/env python3
"""Head-to-head comparison of two crossrow QMV sweeps that differ only in NA_max.

The NA=5 build changes the kernel only at M=5 and M=9; every other width
dispatches byte-identical code. Those unchanged widths are therefore a
per-shape control for session drift (temperature, clocks, driver state), and
the drift-corrected ratio at M=5/M=9 is the real effect of the wider packing.

Usage:
  qmv_na_compare.py --a DIR_NA4 --b DIR_NA5 [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

CHANGED = (5, 9)
CONTROL = (3, 4, 6, 7, 8)


def load(d: Path) -> dict:
    return json.loads((d / "summary.json").read_text())


def curve(summary: dict) -> dict:
    return {(r["name"], r["m"]): r for r in summary["per_shape_curve"]}


def shapes(summary: dict) -> list[str]:
    seen: list[str] = []
    for r in summary["per_shape_curve"]:
        if r["name"] not in seen:
            seen.append(r["name"])
    return seen


def fidelity_rows(summary: dict) -> list[dict]:
    return summary["row0_fidelity"]["by_width"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path, help="NA=4 artifact dir")
    ap.add_argument("--b", required=True, type=Path, help="NA=5 artifact dir")
    ap.add_argument("--a-label", default="NA=4")
    ap.add_argument("--b-label", default="NA=5")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sa, sb = load(args.a), load(args.b)
    ca, cb = curve(sa), curve(sb)
    names = shapes(sa)
    widths = sa["widths"]
    report: dict = {
        "a": {"tag": args.a.name, "label": args.a_label,
              "na_max": sa.get("crossrow_na_max"),
              "boundaries": sa.get("stream_boundaries")},
        "b": {"tag": args.b.name, "label": args.b_label,
              "na_max": sb.get("crossrow_na_max"),
              "boundaries": sb.get("stream_boundaries")},
    }

    print(f"\n=== bit-exactness of {args.b_label} row 0 vs its own M=1 ===")
    print("  any width in 1..9 below 8/8 disqualifies the build")
    fid = []
    for r in fidelity_rows(sb):
        ok = r["shapes_bitwise_identical"] == r["shapes_total"]
        fid.append({"m": r["m"], "identical": r["shapes_bitwise_identical"],
                    "total": r["shapes_total"],
                    "max_abs_delta": r["max_abs_delta"]})
        print(f"  M={r['m']:2d}  {r['shapes_bitwise_identical']}/"
              f"{r['shapes_total']} bitwise  max|d|={r['max_abs_delta']:g}"
              f"  {'OK' if ok else 'DIVERGED'}")
    report["fidelity_b"] = fid
    report["fidelity_b_clean_1_to_9"] = all(
        f["identical"] == f["total"] for f in fid if 1 <= f["m"] <= 9)

    print(f"\n=== seconds/call ratio {args.b_label} / {args.a_label} ===")
    print("  M=3,4,6,7,8 dispatch identical code: that block is session drift")
    hdr = "  ".join(f"M{m}" for m in widths)
    print(f"  {'shape':36s} " + "  ".join(f"{f'M{m}':>6s}" for m in widths)
          + f"  {'drift':>7s} {'M5adj':>7s} {'M9adj':>7s}")
    del hdr
    per_shape = []
    for nm in names:
        ratios = {m: cb[(nm, m)]["seconds_per_call"]
                  / ca[(nm, m)]["seconds_per_call"] for m in widths}
        drift = st.median(ratios[m] for m in CONTROL)
        rec = {"name": nm, "ratio": ratios, "control_drift": drift,
               "m5_drift_adjusted": ratios[5] / drift,
               "m9_drift_adjusted": ratios[9] / drift}
        per_shape.append(rec)
        cells = "  ".join(f"{ratios[m]:6.3f}" for m in widths)
        print(f"  {nm:36s} {cells}  {drift:7.3f} "
              f"{rec['m5_drift_adjusted']:7.3f} "
              f"{rec['m9_drift_adjusted']:7.3f}")
    report["seconds_per_call_ratio"] = per_shape
    report["m5_drift_adjusted_median"] = st.median(
        r["m5_drift_adjusted"] for r in per_shape)
    report["m9_drift_adjusted_median"] = st.median(
        r["m9_drift_adjusted"] for r in per_shape)
    report["control_drift_median"] = st.median(
        r["control_drift"] for r in per_shape)
    print(f"\n  median control drift        {report['control_drift_median']:.4f}"
          "   (1.000 = the two sessions are comparable)")
    print(f"  median M=5 drift-adjusted   "
          f"{report['m5_drift_adjusted_median']:.4f}"
          "   (<1 = wider packing is faster at the old boundary)")
    print(f"  median M=9 drift-adjusted   "
          f"{report['m9_drift_adjusted_median']:.4f}")

    print(f"\n=== achieved GB/s, {args.a_label} -> {args.b_label} ===")
    print(f"  {'shape':36s} " + "  ".join(f"{f'M{m}':>13s}" for m in CHANGED))
    gb = []
    for nm in names:
        cells = []
        rec = {"name": nm}
        for m in CHANGED:
            a = ca[(nm, m)].get("gbps_nominal")
            b = cb[(nm, m)].get("gbps_nominal")
            rec[f"m{m}"] = {"a": a, "b": b}
            cells.append(f"{a:6.1f}->{b:6.1f}" if a and b
                         else f"{'n/a':>13s}")
        gb.append(rec)
        print(f"  {nm:36s} " + "  ".join(cells))
    report["gbps_at_changed_widths"] = gb

    print("\n=== staircase rank test, both builds scored under both laws ===")
    print("  rank1st out of 8 shapes; the law that fits is the one that ranks"
          " its own boundaries first")
    report["rank_test"] = {}
    for lbl, s in ((args.a_label, sa), (args.b_label, sb)):
        stair = s.get("staircase_fit")
        if not stair:
            print(f"  {lbl}: no staircase_fit in this summary")
            continue
        n1 = sum(1 for f in stair if f["boundaries_rank_first"])
        ex = [f["step_excess"] for f in stair]
        key = f"{lbl} build, law NA_max={s.get('crossrow_na_max')}"
        report["rank_test"][key] = {
            "rank1st": n1, "total": len(stair),
            "step_excess_min": min(ex), "step_excess_max": max(ex),
            "boundaries": stair[0]["boundaries"]}
        print(f"  {key:44s} boundaries={stair[0]['boundaries']}  "
              f"rank1st={n1}/{len(stair)}  "
              f"step_excess {min(ex):.2f}..{max(ex):.2f}")

    print("\n=== stock (pip MLX) control: unchanged binary, so a cross-session"
          " drift check ===")
    report["stock_control"] = {}
    for lbl, s in ((args.a_label, sa), (args.b_label, sb)):
        sv = s.get("stock_vs_vendored") or []
        secs = {(r["name"], r["m"]): r["stock_seconds"] for r in sv}
        report["stock_control"][lbl] = secs and st.median(secs.values())
        if secs:
            print(f"  {lbl}: {len(secs)} stock points, "
                  f"median {st.median(secs.values())*1e6:.1f} us/call")
    ka, kb = (report["stock_control"].get(args.a_label),
              report["stock_control"].get(args.b_label))
    if ka and kb:
        sa_pts = {(r["name"], r["m"]): r["stock_seconds"]
                  for r in sa["stock_vs_vendored"]}
        sb_pts = {(r["name"], r["m"]): r["stock_seconds"]
                  for r in sb["stock_vs_vendored"]}
        common = sorted(set(sa_pts) & set(sb_pts))
        ratios = [sb_pts[k] / sa_pts[k] for k in common]
        report["stock_drift_median"] = st.median(ratios)
        print(f"  stock {args.b_label}/{args.a_label} over {len(common)}"
              f" common points: median {st.median(ratios):.4f}, "
              f"range {min(ratios):.3f}..{max(ratios):.3f}")

    for lbl, s in ((args.a_label, sa), (args.b_label, sb)):
        rc = s.get("round_cost_model", {})
        print(f"\n  {lbl} round cost model: d*={rc.get('optimal_depth_q100')}"
              f"  H={rc.get('per_draft_head_seconds')}"
              f"  c={rc.get('per_round_constant_seconds')}"
              f"  s/tok={rc.get('optimal_seconds_per_token_q100')}"
              f"  speedup={rc.get('optimal_speedup_q100')}")
    report["round_cost_model"] = {
        args.a_label: sa.get("round_cost_model"),
        args.b_label: sb.get("round_cost_model"),
    }

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
