"""E117 rung 1: separate the GPU-side barrier price from the host-side price.

The M-frame analysis subtracts each arm's own `control.small` cell, which models
the scored path as one where MLX encodes the next dispatch while the GPU runs
the current one. That model is right for encode work and wrong for a blocking
`eval`. This script reports both halves separately so the reader can see which
one a route is paying, and computes the route-versus-ceiling price as a paired
per-block contrast with its own standard error.
"""

import json
import statistics as st
import sys

ARMS = ["a_one", "c_nsplit", "e_nsplit_serial", "d_depends",
        "f_dep_which", "g_dep_add", "h_async_eval"]
WIDTHS = [2, 4, 5, 6, 7, 8]
SHAPES = ["gdn.in_proj", "mlp.gate_up"]
GDN_LAYERS = 48


def main(path: str) -> None:
    cells = json.load(open(path))["cells"]
    tab: dict = {}
    meta: dict = {}
    for c in cells:
        key = (c["shape"], c["width"], c["arm"])
        tab.setdefault(key, {})[c["block"]] = (c["forward_us"] + c["reverse_us"]) / 2.0
        meta[c["arm"]] = (c["dispatches"], c["evals_per_replicate"])

    ctrl = {(a, w): st.median(tab[("control.small", w, a)].values())
            for a in ARMS for w in WIDTHS}

    print("=" * 92)
    print("host cost, from control.small: median over blocks, microseconds")
    print("=" * 92)
    print(f"{'arm':>18} {'disp':>5} {'evals':>6} " +
          "".join(f"{'M' + str(w):>8}" for w in WIDTHS) + f"{'mean':>9}{'vs a_one':>10}")
    for a in ARMS:
        vals = [ctrl[(a, w)] for w in WIDTHS]
        base = st.mean(ctrl[("a_one", w)] for w in WIDTHS)
        print(f"{a:>18} {meta[a][0]:>5} {meta[a][1]:>6} " +
              "".join(f"{v:>8.2f}" for v in vals) +
              f"{st.mean(vals):>9.2f}{st.mean(vals) - base:>10.2f}")

    print()
    print("=" * 92)
    print("route versus the e_nsplit_serial ceiling, paired per block")
    print("=" * 92)
    print("gpu price  = net(route) - net(ceiling), microseconds per occurrence. "
          "positive means slower.")
    print("host price = control(route) - control(a_one), microseconds per "
          "occurrence, hidden only if")
    print("             the host stays ahead of the GPU in the scored path.")
    for shape in SHAPES:
        print(f"\n## {shape}")
        print(f"{'M':>2} {'a_one net':>10} {'ceil %':>8} " +
              f"{'route':>16} {'route %':>8} {'gpu price':>10} {'se':>7} "
              f"{'host price':>11} {'gain us':>9} {'net us':>8}")
        for w in WIDTHS:
            aone = st.median(tab[(shape, w, "a_one")].values()) - ctrl[("a_one", w)]
            ceil_net = (st.median(tab[(shape, w, "e_nsplit_serial")].values())
                        - ctrl[("e_nsplit_serial", w)])
            ceil_pct = (aone - ceil_net) / aone * 100.0
            for arm in ["d_depends", "f_dep_which", "g_dep_add", "h_async_eval"]:
                per = []
                for b in tab[(shape, w, "a_one")]:
                    e = tab[(shape, w, "e_nsplit_serial")][b] - ctrl[("e_nsplit_serial", w)]
                    r = tab[(shape, w, arm)][b] - ctrl[(arm, w)]
                    per.append(r - e)
                gpu_price = st.mean(per)
                se = st.stdev(per) / len(per) ** 0.5
                route_net = st.median(tab[(shape, w, arm)].values()) - ctrl[(arm, w)]
                route_pct = (aone - route_net) / aone * 100.0
                host_price = ctrl[(arm, w)] - ctrl[("a_one", w)]
                gain = aone - route_net
                print(f"{w:>2} {aone:>10.2f} {ceil_pct:>8.3f} {arm:>16} "
                      f"{route_pct:>8.3f} {gpu_price:>10.2f} {se:>7.2f} "
                      f"{host_price:>11.2f} {gain:>9.2f} {gain - host_price:>8.2f}")

    # Launched versus working threadgroups. Rung 0b could not separate them
    # because they are 4:1 collinear inside one IPG family. M=7 and M=8 break
    # that: same IPG, same two working groups, same grid.y, but 7 x grid.y
    # against 8 x grid.y launched. Cost per output row is the fair statistic,
    # because the GB/s column charges both widths the same weight bytes while
    # M=8 produces one more row.
    print()
    print("=" * 92)
    print("cost per output row, microseconds. IPG and launched groups from "
          "quantized.h:1922-1979")
    print("=" * 92)
    ipg = {2: 2, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4}
    working = {2: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}
    print(f"{'shape':>14} {'M':>2} {'IPG':>4} {'work grp':>9} {'launch/gy':>10} "
          f"{'work/gy':>8} {'net us':>9} {'us per row':>11}")
    for shape in SHAPES:
        for w in WIDTHS:
            net = st.median(tab[(shape, w, "a_one")].values()) - ctrl[("a_one", w)]
            print(f"{shape:>14} {w:>2} {ipg[w]:>4} {working[w]:>9} {w:>10} "
                  f"{working[w]:>8} {net:>9.2f} {net / w:>11.2f}")

    print()
    print("=" * 92)
    print(f"gdn.in_proj M=8, per round at {GDN_LAYERS} GDN layers, all at M=8")
    print("=" * 92)
    aone = st.median(tab[("gdn.in_proj", 8, "a_one")].values()) - ctrl[("a_one", 8)]
    for arm in ["c_nsplit", "e_nsplit_serial", "d_depends", "f_dep_which",
                "g_dep_add", "h_async_eval"]:
        route_net = st.median(tab[("gdn.in_proj", 8, arm)].values()) - ctrl[(arm, 8)]
        gain = aone - route_net
        host = ctrl[(arm, 8)] - ctrl[("a_one", 8)]
        print(f"{arm:>18} gpu gain {gain * GDN_LAYERS:>10.1f} us   "
              f"host cost {host * GDN_LAYERS:>10.1f} us   "
              f"net if host is exposed {(gain - host) * GDN_LAYERS:>10.1f} us")


if __name__ == "__main__":
    main(sys.argv[1])
