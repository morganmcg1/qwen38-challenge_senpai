#!/usr/bin/env python3
"""E100: the ranked price of collapsing an x-group at a width that needs NA >= 5.

WHY THIS EXISTS
---------------
`ranked_stream_ab.py` prices ONE weight-stream removal on the ranked M5 runner
using board trees that are byte-identical outside the two QMV kernel files. It
reports -0.700 % +/- 0.285 % per removal. That estimate is built ONLY from
contrasts where exactly one verify width differs, and every one of those
contrasts moves IPG inside [2, 4]:

    M=4  IPG 4>2      M=6  IPG 3>2      M=8  IPG 4>3, 4>2, 3>2

So the published per-removal price is measured entirely at NA <= 4, where the
wide helper allocates at most 91 g17s registers (E76). E100 proposes IPG 5 at
M = 5 and M = 9, which instantiates NA = 5 at 98 g17s registers. Two things the
one-width instrument cannot see therefore matter:

  1. The M = 5 collapse itself is never contrasted, because every board tree
     that sets IPG 5 at M = 5 also changes M = 6 and M = 9 in the same tree.
     `contrasts()` drops it at `if len(diff) != 1`.
  2. Instantiating NA = 5 raises the register allocation of the SINGLE kernel
     entry point that serves EVERY width, so it can tax widths that never
     execute the changed branch. A one-width contrast has no term for that.

This file adds both. It reuses `ranked_stream_ab.load()` and its empirical null
verbatim so the numbers stay comparable with the published instrument.

WHAT IT REPORTS
---------------
  multi   every fingerprint contrast, including multi-width ones, with the
          number of streams removed and the implied effect per removal.
  narrow  the narrow-prompt control. plutarch's mean verify width is 1.154, so
          it never reaches M >= 3 and never executes a changed branch. Its
          delta across an NA >= 5 contrast is a direct read of the shared
          register-allocation tax, with the same sign convention as the signal.
  pair    the full per-prompt table for one named contrast.

Usage:
  research/e100_na5_board.py report
  research/e100_na5_board.py pair <uuid-prefix-lo> <uuid-prefix-hi>
"""

import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ranked_stream_ab as rsa  # noqa: E402

PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
NARROW = "c1ec5866"  # plutarch, mean verify width 1.154

# Ranked verify-width mass, beagle midpoints (ledger 199/200), reused so the
# per-removal weighting matches the published instrument.
RANKED_QMV_SHARE = rsa.RANKED_QMV_SHARE


def streams(table):
    return {M: math.ceil(M / ipg) for M, ipg in table.items()}


def max_na(table):
    """Widest x-group the table instantiates, i.e. the largest NA compiled."""
    return max(min(ipg, M) for M, ipg in table.items())


def contrast_rows(fps, tables, obs, sd_run, max_widths=4):
    """One row per ordered table pair, LO = fewer total streams."""
    out = []
    for fp, names in fps.items():
        byt = {}
        for n in names:
            byt.setdefault(rsa.tkey(tables[n]), []).append(n)
        ks = list(byt)
        for i in range(len(ks)):
            for j in range(len(ks)):
                if i == j:
                    continue
                ta, tb = dict(ks[i]), dict(ks[j])
                shared_w = sorted(set(ta) & set(tb))
                diff = [M for M in shared_w if ta[M] != tb[M]]
                if not diff or len(diff) > max_widths:
                    continue
                if set(ta) != set(tb):
                    continue  # a missing `case M:` is a different mechanism
                sa, sb = streams(ta), streams(tb)
                removed = sum(sb[M] - sa[M] for M in diff)
                if removed <= 0:
                    continue
                A, B = byt[ks[i]], byt[ks[j]]
                sh = None
                for n in A + B:
                    s = set(obs[n]["mtp"])
                    sh = s if sh is None else (sh & s)
                sh = sorted(sh or [])
                if len(sh) < 4:
                    continue
                d = 100.0 * (statistics.median(rsa.arm_mean(obs, A, sh, "mtp"))
                             - statistics.median(rsa.arm_mean(obs, B, sh, "mtp")))
                ds = 100.0 * (statistics.median(rsa.arm_mean(obs, A, sh, "ser"))
                              - statistics.median(rsa.arm_mean(obs, B, sh, "ser")))
                dl = max(abs(statistics.mean(obs[a]["dl"][k] for k in sh)
                             - statistics.mean(obs[b]["dl"][k] for k in sh))
                         for a in A for b in B)
                narrow = None
                if NARROW in sh:
                    narrow = 100.0 * (
                        statistics.median([obs[n]["mtp"][NARROW] for n in A])
                        - statistics.median([obs[n]["mtp"][NARROW] for n in B]))
                se = sd_run * math.sqrt(1.0 / len(A) + 1.0 / len(B))
                out.append(dict(
                    fp=fp[:12], widths=diff, removed=removed,
                    lo_ipg={M: ta[M] for M in diff},
                    hi_ipg={M: tb[M] for M in diff},
                    lo_na=max_na(ta), hi_na=max_na(tb),
                    nA=len(A), nB=len(B), A=A, B=B,
                    d=d, per=d / removed, ser=ds, dl=dl, se=se,
                    narrow=narrow))
    return out


def pooled(rows, key="d"):
    if not rows:
        return float("nan"), float("nan")
    w = [1.0 / r["se"] ** 2 for r in rows]
    eff = sum(wi * r[key] for wi, r in zip(w, rows)) / sum(w)
    return eff, math.sqrt(1.0 / sum(w))


def report():
    fps, tables, obs = rsa.load()
    if fps is None:
        print("board export not found")
        return 2
    nm, ns = rsa.null_pairs(fps, tables, obs)
    sd_run = rsa.mad_sd(nm) / math.sqrt(2.0)
    rows = contrast_rows(fps, tables, obs, sd_run)

    print("=" * 78)
    print("CORPUS")
    print("=" * 78)
    print("  trees %d   fingerprints %d   null pairs %d   sd(one leg) %.3f %%"
          % (len(tables), len(fps), len(nm), sd_run))
    print("  serial-leg null: mean %+.4f %%  sd %.3f %%"
          % (statistics.mean(ns), statistics.stdev(ns)))

    print()
    print("=" * 78)
    print("ALL CONTRASTS, LO(fewer streams) minus HI, multi-width included")
    print("=" * 78)
    print("  %-13s %-11s %3s %5s %3s %3s %8s %8s %7s %8s %8s"
          % ("fp", "widths", "rm", "maxNA", "nA", "nB", "cand%", "per%",
             "t", "serial%", "narrow%"))
    for r in sorted(rows, key=lambda r: (-r["removed"], r["fp"])):
        print("  %-13s %-11s %3d %2d>%-2d %3d %3d %+8.3f %+8.3f %7.2f %+8.3f %s"
              % (r["fp"], ",".join(str(m) for m in r["widths"]), r["removed"],
                 r["lo_na"], r["hi_na"], r["nA"], r["nB"], r["d"], r["per"],
                 r["d"] / r["se"], r["ser"],
                 "     n/a" if r["narrow"] is None
                 else "%+8.3f" % r["narrow"]))

    lo4 = [r for r in rows if r["lo_na"] <= 4]
    lo5 = [r for r in rows if r["lo_na"] >= 5]
    print()
    print("=" * 78)
    print("POOLED PER STREAM REMOVAL, SPLIT BY THE WIDEST GROUP INSTANTIATED")
    print("=" * 78)
    for tag, sel in (("max NA <= 4", lo4), ("max NA >= 5", lo5)):
        eff, se = pooled(sel, "per")
        tot, tse = pooled(sel, "d")
        n = sum(r["nA"] + r["nB"] for r in sel)
        print("  %-12s groups %2d  runs %3d  per-removal %+7.3f +/- %.3f %%"
              "  (t %+.2f)  whole-contrast %+7.3f %%"
              % (tag, len(sel), n, eff, se, eff / se if se else float("nan"),
                 tot))

    nrow = [r for r in rows if r["narrow"] is not None]
    n4 = [r for r in nrow if r["lo_na"] <= 4]
    n5 = [r for r in nrow if r["lo_na"] >= 5]
    print()
    print("=" * 78)
    print("NARROW-PROMPT CONTROL (plutarch, mean verify width 1.154)")
    print("=" * 78)
    print("  A tree that only changes widths >= 3 cannot change plutarch's work.")
    print("  Any systematic plutarch delta is the shared register-allocation tax.")
    for tag, sel in (("max NA <= 4", n4), ("max NA >= 5", n5)):
        if not sel:
            print("  %-12s no contrast carries plutarch" % tag)
            continue
        v = [r["narrow"] for r in sel]
        print("  %-12s n %2d  median %+7.3f %%  mean %+7.3f %%  min %+7.3f  max %+7.3f"
              % (tag, len(v), statistics.median(v), statistics.mean(v),
                 min(v), max(v)))

    print()
    print("=" * 78)
    print("DRAFT-LENGTH NULL (exactness check: a bit-exact kernel cannot move it)")
    print("=" * 78)
    print("  max |delta effective_mean_draft_len| over all contrasts: %.2e"
          % max(r["dl"] for r in rows))
    return 0


def pair(prefix_lo, prefix_hi):
    fps, tables, obs = rsa.load()
    if fps is None:
        print("board export not found")
        return 2

    def find(p):
        hits = [u for u in obs if u.startswith(p)]
        if len(hits) != 1:
            print("prefix %s matched %d trees" % (p, len(hits)))
            sys.exit(2)
        return hits[0]

    a, b = find(prefix_lo), find(prefix_hi)
    fa = [f for f, ns in fps.items() if a in ns][0]
    fb = [f for f, ns in fps.items() if b in ns][0]
    print("LO %s  table %s  maxNA %d" % (a[:8], rsa.tkey(tables[a]),
                                         max_na(tables[a])))
    print("HI %s  table %s  maxNA %d" % (b[:8], rsa.tkey(tables[b]),
                                         max_na(tables[b])))
    print("non-kernel fingerprint identical: %s" % (fa == fb))
    print()
    print("  %-9s %14s %14s %9s %9s %9s"
          % ("prompt", "mtp LO", "mtp HI", "cand%", "serial%", "d.len"))
    sh = sorted(set(obs[a]["mtp"]) & set(obs[b]["mtp"]),
                key=lambda k: obs[a]["dl"][k])
    for k in sh:
        print("  %-9s %14.9f %14.9f %+9.3f %+9.3f %9.4f"
              % (PROMPT_NAMES.get(k, k),
                 math.exp(obs[a]["mtp"][k]), math.exp(obs[b]["mtp"][k]),
                 100.0 * (obs[a]["mtp"][k] - obs[b]["mtp"][k]),
                 100.0 * (obs[a]["ser"][k] - obs[b]["ser"][k]),
                 obs[a]["dl"][k] - obs[b]["dl"][k]))
    d = [100.0 * (obs[a]["mtp"][k] - obs[b]["mtp"][k]) for k in sh]
    print("  mean %+.3f %%   median %+.3f %%   faster on %d of %d"
          % (statistics.mean(d), statistics.median(d),
             sum(1 for x in d if x < 0), len(d)))
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "report":
        return report()
    if mode == "pair":
        return pair(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
