"""F24 item 3: the W1 / W2 / A1 ISA compile table, priced with the measured shortfall.

W1, W2 and A1 are meant to delete instructions at constant or lower register
count, so unlike the nibble idioms they do not trade occupancy. This asks two
questions per arm:

1. Does the change reach the machine code at all? The test is one-directional.
   An identical `__TEXT,__text` digest at identical size closes the form. An
   unequal digest proves nothing, because a bit-exact rename also moves it.
2. If it reaches, what ranked leg effect survives the shortfall the `623e77af`
   receipt measured?

harness=ranked for the pricing, compile-only for the census. No GPU, no timing.
"""

import json
import sys

WIDTH_TO_IPG = {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 4, 9: 3}
F83_MASS = {6: 0.188, 7: 0.211, 8: 0.1871}
QMV_SHARE = 0.8735
ARCH = "applegpu_g17s"
ARMS = ["w1", "w2", "a1", "w1_a1", "w1_w2_a1"]


def cells(census, variant):
    return {c["na"]: c for c in census["variants"][variant]["cells"].values()}


def routed_mass():
    m = {}
    for w, mass in F83_MASS.items():
        m[WIDTH_TO_IPG[w]] = m.get(WIDTH_TO_IPG[w], 0.0) + mass
    return m


def main(path, shortfall=22.0):
    census = json.load(open(path))
    base = cells(census, "shipped")
    mass = routed_mass()
    total_mass = sum(mass.values())

    print(f"census {census['base_sha'][:8]}   arch {ARCH}   "
          f"toolchain {census['toolchain']}")
    print(f"routed F83 mass covered: {total_mass:.4f} "
          f"(ipg4 {mass.get(4, 0):.4f}, ipg6 {mass.get(6, 0):.4f}, "
          f"ipg7 {mass.get(7, 0):.4f})")
    print()

    print(f"{'arm':<10}{'ipg':>4}{'mass':>7}{'air':>6}{'dair':>6}"
          f"{'ialu':>6}{'reg':>5}{'dreg':>5}{'sg':>4}{'dsg':>4}"
          f"{'text':>7}{'dtext':>7}  digest   reached")
    summary = {}
    for arm in ARMS:
        try:
            cand = cells(census, arm)
        except KeyError:
            continue
        w_air = 0.0
        any_reach = False
        for ipg in sorted(base):
            b, c = base[ipg], cand[ipg]
            ba, ca = b[ARCH], c[ARCH]
            same = ba["text_sha8"] == ca["text_sha8"]
            reached = "no (identical digest)" if same else "unproven"
            if not same:
                any_reach = True
            m = mass.get(ipg, 0.0)
            dair = c["air"]["total"] - b["air"]["total"]
            if m:
                w_air += m * (dair / b["air"]["total"]) * 100
            print(f"{arm:<10}{ipg:>4}{m:>7.4f}{c['air']['total']:>6}"
                  f"{dair:>+6}{c['air']['int_alu']:>6}"
                  f"{ca['registers']:>5}{ca['registers'] - ba['registers']:>+5}"
                  f"{ca['simdgroups_derived']:>4}"
                  f"{ca['simdgroups_derived'] - ba['simdgroups_derived']:>+4}"
                  f"{ca['text_bytes']:>7}"
                  f"{ca['text_bytes'] - ba['text_bytes']:>+7}"
                  f"  {ca['text_sha8']}  {reached}")
        summary[arm] = (w_air, any_reach)
        print()

    print("--- ranked pricing, mass-weighted over the routed cells ---")
    print(f"{'arm':<10}{'w.AIR d%':>10}{'x QMV':>9}{'/shortfall':>12}"
          f"{'predicted ranked leg %':>24}   survives +0.50 %?")
    for arm, (w_air, reached) in summary.items():
        if not reached:
            print(f"{arm:<10}{'0.0000':>10}{'0.0000':>9}{'0.0000':>12}"
                  f"{'exactly 0':>24}   NO - change never reaches the ISA")
            continue
        qmv = w_air * QMV_SHARE
        pred = qmv / shortfall
        print(f"{arm:<10}{w_air:>10.4f}{qmv:>9.4f}{pred:>12.4f}"
              f"{pred:>24.4f}   {'yes' if abs(pred) >= 0.50 else 'NO'}")
    print()
    print(f"shortfall factor applied: {shortfall:g}x, from the 623e77af receipt.")
    print("AIR total is a proxy for issue count. It is measured before the")
    print("backend runs, so an AIR delta that the backend removes shows up as an")
    print("identical text digest, which is why the digest column gates the row.")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "research/e133-artifacts/f23-composed-census.json",
         float(a[1]) if len(a) > 1 else 22.0)
