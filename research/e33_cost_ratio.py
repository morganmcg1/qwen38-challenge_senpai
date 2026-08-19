#!/usr/bin/env python3
"""C_round(M) = sum_shapes calls_per_verify * seconds_per_call, from vendored.json."""
import json
import sys


def cround(path):
    d = json.load(open(path))
    out = {}
    per_shape = {}
    for sh in d["shapes"]:
        cpv = sh["calls_per_verify"]
        for r in sh["rows"]:
            m = r["m"]
            out[m] = out.get(m, 0.0) + cpv * r["seconds_per_call"]
            per_shape.setdefault(m, {})[sh["name"]] = cpv * r["seconds_per_call"]
    return d, out, per_shape


CURVE = ".mlxfast-private/qmv-curve/%s/vendored.json"


def main():
    a_path = sys.argv[1] if len(sys.argv) > 1 else CURVE % "e33-base-r1"
    b_path = sys.argv[2] if len(sys.argv) > 2 else CURVE % "e33-cand-r1"
    treated = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "6").split(",")]
    da, ca, _ = cround(a_path)
    db, cb, _ = cround(b_path)
    widths = [m for m in sorted(set(ca) & set(cb)) if m <= 9]
    control = [m for m in widths if m not in treated and m != 1]
    print(f"{'M':>3} {'A ms':>10} {'B ms':>10} {'B/A':>8} {'A C/M':>9} {'B C/M':>9}  path(B)")
    kp = {}
    for sh in db["shapes"]:
        for r in sh["rows"]:
            kp.setdefault(r["m"], set()).add(r["in_kernel_path"])
    for m in widths:
        paths = ",".join(sorted(kp.get(m, [])))
        print(f"{m:>3} {ca[m]*1e3:10.3f} {cb[m]*1e3:10.3f} {cb[m]/ca[m]:8.4f} "
              f"{ca[m]/m*1e3:9.3f} {cb[m]/m*1e3:9.3f}  {paths[:70]}")
    import statistics as st
    drift = st.median(cb[m] / ca[m] for m in control)
    print(f"\ncontrol widths {control}: median drift {drift:.4f}  "
          f"spread {min(cb[m]/ca[m] for m in control):.4f}..{max(cb[m]/ca[m] for m in control):.4f}")
    for m in treated:
        print(f"M={m}: raw ratio {cb[m]/ca[m]:.4f}   drift-adjusted {cb[m]/ca[m]/drift:.4f}")


main()
