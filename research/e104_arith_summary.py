#!/usr/bin/env python3
"""Cross-shape summary of the E104 arithmetic-arm session.

Regresses isolated kernel time on the static FP instruction count per arm to
test whether floating-point issue is the binding constraint at NA=5.
"""
import argparse
import collections
import json
import statistics as st

FP_OPS_PER_NA = {
    "a_base": 160,
    "n_nosums": 148,
    "xf_exactfma": 112,
    "f_fmamax": 88,
    "s_splitacc": 160,
}
EXACT_REQUIRED = {"xf_exactfma"}
PROMOTION_BAR_PCT = 10.0
CLOSURE_BAR_PCT = 3.0


def load(path):
    doc = json.load(open(path))
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    fidelity = collections.defaultdict(dict)
    for m in doc["measurements"]:
        if m["kind"] == "timing":
            if m["block"] == 0:  # cold start
                continue
            for arm, sec in m["seconds"].items():
                raw[(m["shape"], m["m"])][arm].append(sec * 1e6)
        elif m["kind"] == "fidelity":
            for a in m["arms"]:
                fidelity[(m["shape"], m["m"])][a["arm"]] = a
    timing = {k: {a: st.median(v) for a, v in d.items()} for k, d in raw.items()}
    return doc, timing, fidelity


def pct(cell, arm):
    return 100.0 * (cell[arm] / cell["a_base"] - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rate_json")
    args = ap.parse_args()
    doc, T, F = load(args.rate_json)
    arms = doc["arms"]
    others = [a for a in arms if a != "a_base"]
    shapes = sorted({k[0] for k in T})
    nas = sorted({k[1] for k in T})

    print(f"device: {doc['device']} ({doc['architecture']})")
    print(f"arms: {', '.join(arms)}  shapes: {len(shapes)}  blocks/cell: {doc['pairs']}")

    print("\n=== bit-exactness vs a_base (35 cells) ===")
    for a in others:
        n = sum(1 for k in F if F[k][a]["bit_identical"])
        worst = max(F[k][a]["differing"] for k in F)
        tag = "EXACT-REQUIRED" if a in EXACT_REQUIRED else "diagnostic"
        print(f"  {a:12s} {tag:15s} bit-identical {n}/{len(F)} cells, max differing {worst}")

    print("\n=== median % change vs a_base over 5 shapes (negative = faster) ===")
    print(f"{'NA':>3}" + "".join(f"{a:>14}" for a in others))
    for na in nas:
        cells = [T[(s, na)] for s in shapes]
        vals = [st.median([pct(c, a) for c in cells]) for a in others]
        print(f"{na:>3}" + "".join(f"{v:>+13.2f}%" for v in vals))

    print("\n=== NA=5 per shape, % change vs a_base ===")
    print(f"{'shape':34s}" + "".join(f"{a:>14}" for a in others))
    for s in shapes:
        c = T[(s, 5)]
        print(f"{s:34s}" + "".join(f"{pct(c, a):>+13.2f}%" for a in others))

    print("\n=== FP instruction count vs measured NA=5 time ===")
    print("  static scalar FP instructions per k-block, per NA (rung 0 AIR census)")
    base_i = FP_OPS_PER_NA["a_base"]
    print(f"  {'arm':12s} {'instr/NA':>9} {'instr cut':>10} {'time chg':>10} {'predicted':>10}")
    for a in arms:
        cut = 100.0 * (FP_OPS_PER_NA[a] / base_i - 1.0)
        chg = st.median([pct(T[(s, 5)], a) for s in shapes]) if a != "a_base" else 0.0
        print(f"  {a:12s} {FP_OPS_PER_NA[a]:>9d} {cut:>+9.1f}% {chg:>+9.2f}% {cut:>+9.1f}%")

    print("\n=== revised stop rule ===")
    exact_moves = []
    for a in others:
        if a not in EXACT_REQUIRED:
            continue
        ok = all(F[k][a]["bit_identical"] for k in F)
        chg = st.median([pct(T[(s, 5)], a) for s in shapes])
        exact_moves.append((a, ok, chg))
        print(f"  {a}: bit-exact={ok}  NA=5 median change={chg:+.2f}%")
        print(f"    promotion (rate lift >= {PROMOTION_BAR_PCT:.0f}%): "
              f"{'MET' if ok and chg <= -PROMOTION_BAR_PCT / (1 + PROMOTION_BAR_PCT / 100) else 'NOT MET'}")
    moved = [a for a, ok, chg in exact_moves if abs(chg) > CLOSURE_BAR_PCT]
    print(f"  bit-exact arms moving NA=5 by >{CLOSURE_BAR_PCT:.0f}%: {moved or 'none'}")


if __name__ == "__main__":
    main()
