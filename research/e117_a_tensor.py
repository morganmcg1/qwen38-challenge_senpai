#!/usr/bin/env python3
"""E117 feedback-2 deliverable: shipped-frame group-scaling ratio A_tensor.

A_tensor(IPG=p, G=g) is the cost of running g dispatch groups of p rows against
one group of p rows, at the same tensor and the same per-group width:

    A_tensor(IPG=4, G=2) = T(M=8, [4+4]) / T(M=4, [4])
    A_tensor(IPG=3, G=2) = T(M=6, [3+3]) / T(M=3, [3])
    A_tensor(IPG=3, G=3) = T(M=9, [3+3+3]) / T(M=3, [3])

Reads only committed artifacts under research/e117-artifacts/, per CAMPAIGN
RULE 40. Every value is labelled with its frame and its IPG, per campaign
rule 58.

Each endpoint also carries its launched grid volume V = M * grid.y, because the
E117 rung-0b trough over V in [16384, 18432] moves the endpoints independently
and is the dominant source of spread in A.
"""

import argparse
import json
import math
import os
import statistics

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e117-artifacts")
TROUGH_LO, TROUGH_HI = 16384, 18432

RATIOS = [("ipg4_g2", 4, 8, 2), ("ipg3_g2", 3, 6, 2), ("ipg3_g3", 3, 9, 3)]


def load(name):
    with open(os.path.join(ART, name)) as fh:
        return json.load(fh)


def flagged_widths(summary):
    """(shape, width) pairs holding a block flagged by the defect-19 rule."""
    return {
        (d["shape"], int(d["width"]))
        for d in summary.get("defect19_dispersion", [])
        if d.get("n_flagged", 0) > 0
    }


def in_trough(v):
    return TROUGH_LO <= v <= TROUGH_HI


def entries(summary):
    """Per-shape A_tensor rows, with endpoint volumes and defect-19 status."""
    bad = flagged_widths(summary)
    rows = []
    for shape, sh in sorted(summary["shapes"].items()):
        widths = sh["widths"]
        grid_y = (sh["outputs"] + 7) // 8
        row = {
            "shape": shape,
            "outputs": sh["outputs"],
            "grid_y": grid_y,
            "weight_bytes": sh["weight_bytes"],
        }
        for label, m_lo, m_hi, groups in RATIOS:
            lo, hi = str(m_lo), str(m_hi)
            if lo not in widths or hi not in widths:
                continue
            t_lo = widths[lo]["a_one_net_us"]
            t_hi = widths[hi]["a_one_net_us"]
            if not (t_lo > 0 and t_hi > 0):
                continue
            dirty = (shape, m_lo) in bad or (shape, m_hi) in bad
            row[label] = {
                "a_tensor": t_hi / t_lo,
                "groups": groups,
                "per_extra_stream": (t_hi / t_lo) / groups,
                "t_one_group_us": t_lo,
                "t_multi_group_us": t_hi,
                "v_one_group": m_lo * grid_y,
                "v_multi_group": m_hi * grid_y,
                "one_group_in_trough": in_trough(m_lo * grid_y),
                "multi_group_in_trough": in_trough(m_hi * grid_y),
                "defect19_flagged": dirty,
            }
        rows.append(row)
    return rows


def spread(values):
    if not values:
        return {}
    out = {
        "n": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }
    out["range_pct_of_mean"] = 100.0 * (out["max"] - out["min"]) / out["mean"]
    if len(values) > 1:
        out["sd"] = statistics.stdev(values)
        out["sem"] = out["sd"] / math.sqrt(len(values))
    return out


def aggregate(rows, label):
    clean = [
        r[label]["a_tensor"]
        for r in rows
        if label in r and not r[label]["defect19_flagged"]
    ]
    both_out = [
        r[label]["a_tensor"]
        for r in rows
        if label in r
        and not r[label]["defect19_flagged"]
        and not r[label]["one_group_in_trough"]
        and not r[label]["multi_group_in_trough"]
    ]
    return {
        "all_clean": spread(clean),
        "both_endpoints_out_of_trough": spread(both_out),
    }


def fmt(rows, label, title):
    have = [r for r in rows if label in r]
    if not have:
        return []
    out = [
        "",
        title,
        f"{'shape':<14}{'N':>8}{'V(1grp)':>9}{'V(Ggrp)':>9}"
        f"{'T(1grp)us':>11}{'T(Ggrp)us':>11}{'A_tensor':>10}  flags",
    ]
    for r in have:
        e = r[label]
        flags = []
        if e["one_group_in_trough"]:
            flags.append("1grp-in-trough")
        if e["multi_group_in_trough"]:
            flags.append("Ggrp-in-trough")
        if e["defect19_flagged"]:
            flags.append("DEFECT19-EXCLUDED")
        out.append(
            f"{r['shape']:<14}{r['outputs']:>8}{e['v_one_group']:>9}"
            f"{e['v_multi_group']:>9}{e['t_one_group_us']:>11.2f}"
            f"{e['t_multi_group_us']:>11.2f}{e['a_tensor']:>10.4f}  "
            + ",".join(flags)
        )
    agg = aggregate(have, label)
    for key, title2 in (
        ("all_clean", "  all defect-19-clean shapes"),
        ("both_endpoints_out_of_trough", "  both endpoints out of the trough"),
    ):
        s = agg[key]
        if s:
            line = (
                f"{title2}: n={s['n']} mean={s['mean']:.4f} "
                f"min={s['min']:.4f} max={s['max']:.4f} "
                f"range={s['range_pct_of_mean']:.1f} % of mean"
            )
            if "sd" in s:
                line += f" sd={s['sd']:.4f}"
            out.append(line)
    return out


def defect19_report(summary, name):
    disp = summary.get("defect19_dispersion", [])
    hot = [d for d in disp if d.get("n_flagged", 0) > 0]
    lines = [
        f"{name}: {len(disp)} cells, {len(hot)} cells with a flagged block, "
        f"{sum(d['n_flagged'] for d in hot)} flagged blocks"
    ]
    for d in sorted(hot, key=lambda x: -x["max_over_median"]):
        lines.append(
            f"    {d['shape']:<14} M={d['width']:<2} {d['arm']:<16} "
            f"median={d['median_us']:>9.2f} us  max={d['max_us']:>9.2f} us  "
            f"max/med={d['max_over_median']:.2f}  blocks={d['flagged_blocks']}"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write",
        action="store_true",
        help="add the a_tensor columns back into the artifact JSON files",
    )
    args = ap.parse_args()

    sources = [
        ("rung0b-nsweep-summary.json", "RUNG 0B: synthetic N sweep, K=5120"),
        ("rung0-mframe-summary.json", "RUNG 0: the four real tensors"),
        ("rung1-routes-summary.json", "RUNG 1: cross-session replicate"),
    ]

    print("=" * 78)
    print("E117 a_tensor_shipped_frame   harness=local   frame=A_tensor")
    print("Apple M4 Pro 20-core, 48 GiB; cool_gate_passed_real_gate=false;")
    print("gate_qualified_for_timing=false; timing_valid=false")
    print(f"grid-volume trough band V in [{TROUGH_LO}, {TROUGH_HI}]")
    print("=" * 78)

    defect_lines = []
    for fname, title in sources:
        summary = load(fname)
        rows = entries(summary)
        print()
        print("-" * 78)
        print(title)
        print("-" * 78)
        for label, lo, hi, g in RATIOS:
            ipg = 4 if label.startswith("ipg4") else 3
            head = (
                f"A_tensor(IPG={ipg}, G={g}) = T(M={hi}) / T(M={lo})"
                f"   [shipped frame]" + ("   <- SHIPPED POINT" if ipg == 4 else "")
            )
            for line in fmt(rows, label, head):
                print(line)
        defect_lines += defect19_report(summary, fname.split("-")[0])

        if args.write:
            by_shape = {r["shape"]: r for r in rows}
            for curves in summary.get("n_curves", {}).values():
                for c in curves:
                    r = by_shape.get(c["shape"])
                    if r and "ipg4_g2" in r:
                        c["a_tensor_ipg4_shipped_frame"] = r["ipg4_g2"]["a_tensor"]
            summary["a_tensor_shipped_frame"] = {
                "definition": "T(M=hi)/T(M=lo) on a_one_net_us, one tensor, "
                "same per-group width",
                "trough_band_launched_grid_volume": [TROUGH_LO, TROUGH_HI],
                "per_shape": rows,
                "aggregate": {
                    label: aggregate(rows, label)
                    for label, _, _, _ in RATIOS
                    if any(label in r for r in rows)
                },
            }
            with open(os.path.join(ART, fname), "w") as fh:
                json.dump(summary, fh, indent=1, sort_keys=True)
                fh.write("\n")

    print()
    print("=" * 78)
    print("DEFECT-19 DISPERSION CENSUS (blocks above 1.5 x the cell median)")
    print("=" * 78)
    for line in defect_lines:
        print(line)


if __name__ == "__main__":
    main()
