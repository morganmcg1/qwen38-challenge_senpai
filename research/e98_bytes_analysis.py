#!/usr/bin/env python3
"""Aggregate the E98 rung-1a metadata byte ladder into per-cell effects.

Reads bytes.json produced by E98MetadataByteProbeTests and reports, for every
(shape, width) cell, the ABBA-averaged net time at each group size, the
achieved read bandwidth, the DRAM-floor check, and the group-size contrasts
that price a metadata byte.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict

DRAM_PEAK_GB_S = 273.0  # M4 Pro theoretical peak

# quantized.h:1929-1974 wide crossrow dispatch: inputs per weight stream.
CROSSROW_IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}


def weight_streams(kernel_family, width):
    """Logical passes over the weight matrix in one dispatch.

    quantized.cpp:254 sets grid.x = M. qmv_fast_impl indexes the input row by
    tid.x (quantized.h:788-789), so every one of the M columns streams the whole
    matrix. qmv_fast_crossrow_affine4_g64_m returns early unless
    tid.x * IPG < M (quantized.h:1173-1175), so only ceil(M / IPG) columns run.
    """
    if kernel_family == "crossrow_m":
        return math.ceil(width / CROSSROW_IPG[width])
    if kernel_family == "crossrow":
        return 1
    return width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="research/out/e98-bytes-r1a/bytes.json")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = json.load(open(args.input))
    cells = d["cells"]
    bits = d["bits"]

    by = defaultdict(list)
    for c in cells:
        by[(c["shape"], c["width"], c["group_size"])].append(c)

    shapes = sorted({c["shape"] for c in cells})
    widths = sorted({c["width"] for c in cells})
    gss = sorted({c["group_size"] for c in cells})

    out = {"dram_peak_gb_per_s": DRAM_PEAK_GB_S, "cells": [], "contrasts": []}

    print(f"eval_overhead_us={d['eval_overhead_us']:.2f} blocks={d['blocks']} bits={bits}")
    print()
    for tag_null in d["nulls"]:
        print(
            f"NULL {tag_null['shape']} gs={tag_null['group_size']} m={tag_null['width']} "
            f"tag={tag_null['tag']} us={tag_null['microseconds']:.2f}"
        )
    # session null: same arm measured at open and close
    opens = {n["shape"]: n for n in d["nulls"] if n["tag"] == "session_open"}
    closes = {n["shape"]: n for n in d["nulls"] if n["tag"] == "session_close"}
    for s in sorted(opens):
        if s in closes:
            o, c = opens[s]["microseconds"], closes[s]["microseconds"]
            print(f"SESSION_NULL {s}: open={o:.2f} close={c:.2f} drift={100*(c-o)/o:+.2f}%")
    print()

    for shape in shapes:
        print(f"### {shape}")
        header = f"{'M':>3} " + " ".join(
            f"{'gs'+str(g)+'_us':>12} {'gs'+str(g)+'_GBs':>10}" for g in gss
        )
        print(header)
        for m in widths:
            row = [f"{m:>3}"]
            for g in gss:
                cs = by.get((shape, m, g), [])
                if not cs:
                    row.append(f"{'-':>12} {'-':>10}")
                    continue
                net = statistics.mean(c["net_us"] for c in cs)
                fams = sorted({c["kernel_family"] for c in cs})
                g = weight_streams(fams[0], m)
                rb = cs[0]["read_bytes"] * g
                wb = cs[0]["write_bytes"]
                gbs = rb / (net * 1e-6) / 1e9
                floor_us = (rb + wb) / (DRAM_PEAK_GB_S * 1e9) * 1e6
                out["cells"].append(
                    {
                        "shape": shape,
                        "width": m,
                        "group_size": g,
                        "net_us_mean": net,
                        "net_us_spread_pct": 100
                        * (max(c["net_us"] for c in cs) - min(c["net_us"] for c in cs))
                        / net,
                        "n_blocks": len(cs),
                        "read_bytes": rb,
                        "weight_streams": g,
                        "write_bytes": wb,
                        "read_gb_per_s": gbs,
                        "pct_dram_peak": 100 * gbs / DRAM_PEAK_GB_S,
                        "dram_floor_us": floor_us,
                        "below_dram_floor": net < floor_us,
                        "kernel_families": fams,
                    }
                )
                flag = "!!BELOW_FLOOR" if net < floor_us else ""
                row.append(f"{net:>12.1f} {gbs:>7.1f}/G{g}{flag}")
            print(" ".join(row))

        print(f"{'M':>3} {'kernel families by gs':>20}")
        for m in widths:
            fams = []
            for g in gss:
                cs = by.get((shape, m, g), [])
                f = sorted({c["kernel_family"] for c in cs})
                fams.append(f"gs{g}:{'/'.join(f) if f else '-'}")
            print(f"{m:>3} " + "  ".join(fams))
        print()

        # contrasts
        print(
            f"{'M':>3} {'g64->g128 %':>12} {'pred %':>8} {'g32->g64 %':>12} {'pred %':>8} "
            f"{'g32->g128 %':>12} {'pred %':>8} {'same_kernel':>26}"
        )
        for m in widths:
            vals = {}
            fam = {}
            for g in gss:
                cs = by.get((shape, m, g), [])
                if cs:
                    vals[g] = statistics.mean(c["net_us"] for c in cs)
                    fam[g] = sorted({c["kernel_family"] for c in cs})
            rb = {
                g: by[(shape, m, g)][0]["read_bytes"]
                * weight_streams(fam[g][0], m)
                for g in vals
            }

            def contrast(a, b):
                if a not in vals or b not in vals:
                    return None, None, None
                meas = 100.0 * (vals[b] - vals[a]) / vals[a]
                pred = 100.0 * (rb[b] - rb[a]) / rb[a]
                same = fam[a] == fam[b] and len(fam[a]) == 1
                return meas, pred, same

            c1 = contrast(64, 128)
            c2 = contrast(32, 64)
            c3 = contrast(32, 128)
            same_str = f"64/128:{c1[2]} 32/128:{c3[2]}"
            out["contrasts"].append(
                {
                    "shape": shape,
                    "width": m,
                    "g64_to_g128_pct": c1[0],
                    "g64_to_g128_pred_pct": c1[1],
                    "g64_to_g128_same_kernel": c1[2],
                    "g32_to_g64_pct": c2[0],
                    "g32_to_g64_pred_pct": c2[1],
                    "g32_to_g64_same_kernel": c2[2],
                    "g32_to_g128_pct": c3[0],
                    "g32_to_g128_pred_pct": c3[1],
                    "g32_to_g128_same_kernel": c3[2],
                }
            )

            def f(x):
                return "-" if x is None else f"{x:+.2f}"

            print(
                f"{m:>3} {f(c1[0]):>12} {f(c1[1]):>8} {f(c2[0]):>12} {f(c2[1]):>8} "
                f"{f(c3[0]):>12} {f(c3[1]):>8} {same_str:>26}"
            )
        print()

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
