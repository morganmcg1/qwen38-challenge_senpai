#!/usr/bin/env python3
"""E100: reduce an ABBA stream-collapse session to a benefit term and a tax term.

The dispatch table is a compile-time template selection, so the two arms are two
BUILDS and the session is counterbalanced across them as A B B A. Arm means are
therefore (leg1 + leg4) / 2 and (leg2 + leg3) / 2, which cancels linear thermal
drift to first order.

Two families of width come out of the same session:

  BENEFIT   M = 5 and M = 9. Their dispatch entry changes from IPG 3 to IPG 5,
            so the number of complete passes over the weight matrix falls from
            2 to 1 and from 3 to 2.
  TAX       M = 1, 2, 3, 4, 6, 7, 8. Their dispatch entry is byte-identical in
            both arms. Any movement is the shared register-allocation cost of
            instantiating NA = 5 in the same kernel entry point, and it is the
            term the ranked board cannot measure.

Exactness is read from the row digests rather than argued: every output row must
carry the same FNV-1a digest in both arms and at every width in one arm.

Usage:
  research/e100_analysis.py A1 B1 B2 A2 [--json OUT]
"""

import argparse
import json
import math
import os
import statistics
import sys

DRAM_PEAK_BYTES_PER_S = 273.0e9
BENEFIT_WIDTHS = (5, 9)

# Streams per round, ceil(M / IPG), for the two arms. Read from the dispatcher
# rather than fitted; assert against the leg witness before trusting a rate.
IPG_BASE = {2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}
IPG_COLLAPSE = dict(IPG_BASE, **{5: 5, 9: 5})


def streams(m, ipg):
    return 1 if m == 1 else math.ceil(m / ipg[m])


def load(tag):
    path = os.path.join("research/out", tag, "cells.json")
    with open(path) as f:
        doc = json.load(f)
    meta = {}
    mpath = os.path.join("research/out", tag, "meta.txt")
    if os.path.exists(mpath):
        for line in open(mpath):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                meta[k] = v
    doc["_meta"] = meta
    doc["_tag"] = tag
    return doc


def cell_us(doc):
    """(shape, m) -> net microseconds, drift-symmetric within the leg."""
    overhead = doc["eval_overhead_us"]
    out = {}
    for c in doc["cells"]:
        mean = (c["forward_us"] + c["reverse_us"]) / 2.0
        out[(c["shape"], c["m"])] = mean - overhead
    return out


def cell_bytes(doc):
    return {(c["shape"], c["m"]): c["packed_bytes"] for c in doc["cells"]}


def digests(doc):
    return {(c["shape"], c["m"]): c["row_digests"] for c in doc["cells"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legs", nargs=4, metavar=("A1", "B1", "B2", "A2"))
    ap.add_argument("--json", dest="out")
    args = ap.parse_args()

    a1, b1, b2, a2 = (load(t) for t in args.legs)
    for doc, want in ((a1, "base"), (b1, "collapse"),
                      (b2, "collapse"), (a2, "base")):
        got = doc["_meta"].get("arm")
        if got != want:
            print("leg %s declares arm=%s, expected %s"
                  % (doc["_tag"], got, want))
            return 2
    for doc in (b1, b2):
        if doc["_meta"].get("twin_m5") != "1" or doc["_meta"].get("twin_m9") != "1":
            print("leg %s is not a collapse build: %s"
                  % (doc["_tag"], doc["_meta"].get("twin_na_bound")))
            return 2
    for doc in (a1, a2):
        if doc["_meta"].get("twin_m5") != "0" or doc["_meta"].get("twin_m9") != "0":
            print("leg %s is not a base build" % doc["_tag"])
            return 2

    ua1, ub1, ub2, ua2 = (cell_us(d) for d in (a1, b1, b2, a2))
    bytes_by = cell_bytes(a1)
    keys = sorted(set(ua1) & set(ub1) & set(ub2) & set(ua2),
                  key=lambda k: (k[0], k[1]))

    print("=" * 92)
    print("IDENTITY")
    print("=" * 92)
    for doc in (a1, b1, b2, a2):
        m = doc["_meta"]
        print("  %-10s arm=%-9s head=%s dirty=%s entry_c=%s exit_c=%s %s"
              % (doc["_tag"], m.get("arm"), (m.get("git_head") or "")[:8],
                 m.get("git_dirty"), m.get("gpu_temp_entry_c") or "n/a",
                 m.get("gpu_temp_exit_c") or "n/a", m.get("twin_na_bound")))
    temps = [float(d["_meta"]["gpu_temp_entry_c"]) for d in (a1, b1, b2, a2)
             if (d["_meta"].get("gpu_temp_entry_c") or "").strip()]
    if temps:
        print("  entry temperature spread: %.1f C (min %.1f, max %.1f)"
              % (max(temps) - min(temps), min(temps), max(temps)))
    print("  cool_gate_passed_real_gate=false  gate_qualified_for_timing=false")

    print()
    print("=" * 92)
    print("EXACTNESS")
    print("=" * 92)
    cross = []
    da, db = digests(a1), digests(b1)
    for k in sorted(set(da) & set(db)):
        if da[k] != db[k]:
            cross.append(k)
    print("  within-arm row-digest mismatches: base %d, collapse %d"
          % (len(a1["row_digest_mismatches"]), len(b1["row_digest_mismatches"])))
    print("  across-arm row-digest mismatches: %d of %d cells"
          % (len(cross), len(set(da) & set(db))))
    if cross:
        for k in cross:
            print("    MISMATCH shape=%s m=%d" % k)
    pc = [c for c in b1["positive_control"] if not c["differs"]]
    print("  positive control (float32 dequantised path must differ): "
          "%d of %d shapes differ"
          % (len(b1["positive_control"]) - len(pc), len(b1["positive_control"])))

    print()
    print("=" * 92)
    print("PER-CELL, ARM MEANS (A1+A2)/2 AND (B1+B2)/2")
    print("=" * 92)
    print("  %-13s %2s %2s %6s %11s %11s %8s %9s %9s %8s"
          % ("shape", "m", "G", "G'", "base_us", "coll_us", "delta%",
             "base_GB/s", "coll_GB/s", "floor%"))
    rows = []
    for k in keys:
        shape, m = k
        base = (ua1[k] + ua2[k]) / 2.0
        coll = (ub1[k] + ub2[k]) / 2.0
        g = streams(m, IPG_BASE)
        gp = streams(m, IPG_COLLAPSE)
        by = bytes_by[k]
        base_gb = by * g / base / 1e3
        coll_gb = by * gp / coll / 1e3
        floor_us = by * gp / DRAM_PEAK_BYTES_PER_S * 1e6
        d = 100.0 * (coll / base - 1.0)
        spread = 100.0 * abs(ua1[k] - ua2[k]) / base
        rows.append(dict(shape=shape, m=m, g=g, gp=gp, base_us=base,
                         coll_us=coll, delta=d, base_gb=base_gb,
                         coll_gb=coll_gb, floor_us=floor_us,
                         base_spread=spread, bytes=by))
        print("  %-13s %2d %2d %6d %11.2f %11.2f %+8.3f %9.1f %9.1f %8.1f"
              % (shape, m, g, gp, base, coll, d, base_gb, coll_gb,
                 100.0 * floor_us / coll))

    print()
    print("=" * 92)
    print("BENEFIT AND TAX")
    print("=" * 92)
    ben = [r for r in rows if r["m"] in BENEFIT_WIDTHS]
    tax = [r for r in rows if r["m"] not in BENEFIT_WIDTHS]
    repeat = [r["base_spread"] for r in rows]
    print("  A1-to-A2 repeatability on the unchanged arm: median %.3f %%, "
          "max %.3f %%" % (statistics.median(repeat), max(repeat)))
    for tag, sel in (("BENEFIT m in %s" % (BENEFIT_WIDTHS,), ben),
                     ("TAX     other m", tax)):
        v = [r["delta"] for r in sel]
        print("  %-22s n %2d  median %+7.3f %%  mean %+7.3f %%  "
              "min %+7.3f  max %+7.3f"
              % (tag, len(v), statistics.median(v), statistics.mean(v),
                 min(v), max(v)))
    for m in sorted({r["m"] for r in rows}):
        v = [r["delta"] for r in rows if r["m"] == m]
        print("    m=%d  n %d  median %+7.3f %%  values %s"
              % (m, len(v), statistics.median(v),
                 " ".join("%+.2f" % x for x in v)))

    doc = dict(
        legs=[d["_tag"] for d in (a1, b1, b2, a2)],
        meta={d["_tag"]: d["_meta"] for d in (a1, b1, b2, a2)},
        across_arm_digest_mismatches=["%s:m%d" % k for k in cross],
        within_arm_digest_mismatches=dict(
            base=a1["row_digest_mismatches"],
            collapse=b1["row_digest_mismatches"]),
        positive_control=b1["positive_control"],
        cells=rows,
        summary=dict(
            benefit_median_pct=statistics.median([r["delta"] for r in ben]),
            tax_median_pct=statistics.median([r["delta"] for r in tax]),
            tax_max_abs_pct=max(abs(r["delta"]) for r in tax),
            repeatability_median_pct=statistics.median(repeat),
            repeatability_max_pct=max(repeat)),
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
