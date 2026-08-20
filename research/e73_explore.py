#!/usr/bin/env python3
"""E73 rung-2 exploratory view: does the per-byte rate collapse onto occupancy?

Reads the rung-1 session and prints the achieved-bandwidth surface indexed by
working threadgroups per core, so the respecified model can be chosen from the
data rather than assumed.
"""
import json
import math
import statistics
import sys

CORES = 20  # read from device: ioreg gpu-core-count on this M4 Pro
SG_PER_TG = 2  # group_dims(32, 2, 1)

# derived resident simdgroups per core from the rung-0 register census
SG_REG = {2: 37, 3: 31, 4: 26, 5: 23, 6: 16}


def load(path):
    d = json.load(open(path))
    cells = {}
    for sh in d["shapes"]:
        n, k = sh["n"], sh["k"]
        w = sh["bytes_per_stream"]
        by_arm = {}
        for leg in sh["legs"]:
            by_arm.setdefault(leg["arm"], []).append(leg)
        for arm, legs in by_arm.items():
            m, ipg = legs[0]["m"], legs[0]["ipg"]
            groups = math.ceil(m / ipg)
            t = statistics.median(l["seconds_per_dispatch"] for l in legs)
            pos = {}
            for l in legs:
                pos.setdefault(l["position"], []).append(l["seconds_per_dispatch"])
            cells[(sh["shape"], m, ipg)] = dict(
                shape=sh["shape"], n=n, k=k, w=w, m=m, ipg=ipg, groups=groups,
                t=t, tgs=groups * (n // 8), pos={p: statistics.median(v) for p, v in pos.items()},
            )
    return d, cells


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "research/e73-artifacts/rung1-s1.json"
    d, cells = load(path)

    nulls = []
    for c in cells.values():
        v = sorted(c["pos"].values())
        if len(v) == 2:
            nulls.append(abs(v[1] - v[0]) / v[0])
    print(f"session null over {len(nulls)} cells: median {statistics.median(nulls)*100:.4f}% "
          f"p90 {sorted(nulls)[int(.9*len(nulls))]*100:.4f}% max {max(nulls)*100:.4f}%")

    shapes = sorted({(c["shape"], c["n"], c["k"]) for c in cells.values()}, key=lambda s: s[1])
    print("\n== achieved GB/s (weight stream bytes / t), by shape and (M,IPG) ==")
    print("shape                                    n      k   | " +
          "  ".join(f"m{m}i{i}" for m, i in sorted({(c['m'], c['ipg']) for c in cells.values()})))
    for name, n, k in shapes:
        row = []
        for m, ipg in sorted({(c["m"], c["ipg"]) for c in cells.values()}):
            c = cells.get((name, m, ipg))
            row.append(f"{c['groups']*c['w']/c['t']/1e9:6.1f}" if c else "     -")
        print(f"{name[:38]:38s} {n:6d} {k:6d} | " + " ".join(row))

    print("\n== rate r = t/(groups*W) in ps/byte, vs working TGs per core ==")
    pts = []
    for c in cells.values():
        x = c["tgs"] / CORES
        r = c["t"] / (c["groups"] * c["w"]) * 1e12
        pts.append((x, r, c))
    pts.sort()
    print(" tgs/core     ps/B  M IPG sgreg  shape")
    for x, r, c in pts:
        print(f"{x:9.1f} {r:8.3f}  {c['m']} {c['ipg']}   {SG_REG[c['ipg']]:3d}  {c['shape'][:32]}")


if __name__ == "__main__":
    main()
