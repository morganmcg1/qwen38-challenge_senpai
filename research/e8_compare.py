#!/usr/bin/env python3
"""Compare qmv-curve runs shape-by-shape at the E8 verify widths.

Usage: python3 research/e8_compare.py BASE_TAG CAND_TAG [CAND_TAG ...]

Reports, per shape and per width, seconds_per_call and gbps_nominal for the
base and each candidate, plus the candidate/base time ratio.  A ratio well
below 1.0 means the arm's change made the kernel faster.
"""

import json
import statistics
import sys

ROOT = ".mlxfast-private/qmv-curve"
WIDTHS = (4, 8)


def curve(tag):
    with open(f"{ROOT}/{tag}/summary.json") as fh:
        doc = json.load(fh)
    out = {}
    for r in doc["per_shape_curve"]:
        out[(r["name"], r["m"])] = r
    return out, doc


def main():
    tags = sys.argv[1:]
    if len(tags) < 2:
        sys.exit(__doc__)
    base_tag, cand_tags = tags[0], tags[1:]
    base, base_doc = curve(base_tag)
    cands = {t: curve(t)[0] for t in cand_tags}

    shapes = sorted({k[0] for k in base})
    print(f"base = {base_tag}   candidates = {', '.join(cand_tags)}")
    print(f"host = {base_doc['host']}  peak_bw = "
          f"{base_doc['roofline']['peak_bandwidth_bytes_per_second'] / 1e9:.1f} GB/s")
    print()

    ratios = {t: {m: [] for m in WIDTHS} for t in cand_tags}
    for m in WIDTHS:
        print(f"===== M = {m} =====")
        head = f"{'shape':34s} {'base s/call':>12s} {'base GB/s':>10s}"
        for t in cand_tags:
            head += f" | {t + ' s/call':>16s} {'GB/s':>8s} {'ratio':>7s}"
        print(head)
        for s in shapes:
            b = base.get((s, m))
            if b is None:
                continue
            line = (f"{s:34s} {b['seconds_per_call'] * 1e6:12.2f} "
                    f"{b['gbps_nominal']:10.1f}")
            for t in cand_tags:
                c = cands[t].get((s, m))
                if c is None:
                    line += f" | {'--':>16s} {'--':>8s} {'--':>7s}"
                    continue
                rr = c["seconds_per_call"] / b["seconds_per_call"]
                ratios[t][m].append(rr)
                line += (f" | {c['seconds_per_call'] * 1e6:16.2f} "
                         f"{c['gbps_nominal']:8.1f} {rr:7.3f}")
            print(line)
        print()

    print("===== median candidate/base time ratio =====")
    for t in cand_tags:
        parts = []
        for m in WIDTHS:
            v = ratios[t][m]
            if v:
                parts.append(f"M={m}: {statistics.median(v):.3f} "
                             f"[{min(v):.3f}, {max(v):.3f}]")
        print(f"  {t:16s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
