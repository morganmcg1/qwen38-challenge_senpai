"""Ask whether a LOCAL g16s timing arm can observe the ranked g17s residency effect.

The nibble-axis result is a register/occupancy claim. Register allocation is
per-architecture, so the local host (applegpu_g16s, M4 Pro) and the ranked host
(applegpu_g17s, M5) do not necessarily move together. This prices that transfer
before any GPU time is spent.

Research-only: compile-derived, no GPU, no timing, no model.
"""

import json
import sys

# F139 register budgets per core, and F83 routed-width masses.
BUDGET = {"applegpu_g16s": 3072, "applegpu_g17s": 3968}
WIDTH_TO_IPG = {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 4, 9: 3}
F83_MASS = {3: 0.0, 4: 0.0, 5: 0.0, 6: 0.188, 7: 0.211, 8: 0.1871, 9: 0.0}
QMV_SHARE = 0.8735


def residency(regs, arch):
    return BUDGET[arch] // regs


def cell_rows(census, variant):
    out = {}
    for name, c in census["variants"][variant]["cells"].items():
        out[c["na"]] = c
    return out


def main(path, base="shipped", cand="per_cell_best"):
    census = json.load(open(path))
    b = cell_rows(census, base)
    c = cell_rows(census, cand)

    # ipg -> total F83 mass routed into that cell
    mass = {}
    for w, m in F83_MASS.items():
        mass[WIDTH_TO_IPG[w]] = mass.get(WIDTH_TO_IPG[w], 0.0) + m

    print(f"census base_sha : {census['base_sha']}")
    print(f"base variant    : {base}")
    print(f"cand variant    : {cand}")
    print()

    tot = {}
    for arch in ("applegpu_g16s", "applegpu_g17s"):
        print(f"--- {arch}  (budget {BUDGET[arch]}) ---")
        hdr = f"{'ipg':>4} {'mass':>7} {'reg_b':>6} {'reg_c':>6} {'dreg':>5} "
        hdr += f"{'sg_b':>5} {'sg_c':>5} {'dsg':>4} {'dsg_%':>8} {'w_dsg_%':>8}"
        print(hdr)
        acc = 0.0
        for ipg in sorted(b):
            rb = b[ipg][arch]["registers"]
            rc = c[ipg][arch]["registers"]
            sb = residency(rb, arch)
            sc = residency(rc, arch)
            pct = 100.0 * (sc - sb) / sb
            m = mass.get(ipg, 0.0)
            acc += m * pct
            print(
                f"{ipg:>4} {m:>7.4f} {rb:>6} {rc:>6} {rc - rb:>5} "
                f"{sb:>5} {sc:>5} {sc - sb:>4} {pct:>8.3f} {m * pct:>8.4f}"
            )
        print(f"{'':>4} {'':>7} {'':>6} {'':>6} {'':>5} {'':>5} {'':>5} "
              f"{'':>4} {'weighted':>8} {acc:>8.4f}")
        print(f"     QMV-scaled ({QMV_SHARE}) weighted residency gain: "
              f"{acc * QMV_SHARE:+.4f} %")
        tot[arch] = acc * QMV_SHARE
        print()

    loc, rank = tot["applegpu_g16s"], tot["applegpu_g17s"]
    print("--- transfer verdict ---")
    print(f"local  g16s weighted residency gain : {loc:+.4f} %")
    print(f"ranked g17s weighted residency gain : {rank:+.4f} %")
    if abs(rank) > 1e-9:
        print(f"local / ranked observability ratio  : {loc / rank:.3f}")
    print()
    print("A local timing arm can only see the local column. If that column is")
    print("much smaller than the ranked column, a local null is uninformative")
    print("about the ranked effect and must not be read as a refutation.")


if __name__ == "__main__":
    args = sys.argv[1:]
    p = args[0] if args else "research/e133-artifacts/f23-composed-census.json"
    main(p, *args[1:])
