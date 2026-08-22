"""How tightly is the board round count R pinned?

R is only recoverable up to an integer multiple of the minimal denominator of
`effective_mean_draft_len`. Each legal multiple implies an acceptance rate

    alpha = (512 / R - 1) / dl

so the spread of alpha over the UNIQUELY pinned prompt-rows says how much the
remaining multiples can be pruned on physical grounds.
"""

import statistics
from collections import defaultdict

from e128_rounds import load_rows, per_prompt, recover_rounds

TOKENS = 512


def legal_multiples(dl, n0):
    from fractions import Fraction

    if dl == 0:
        return [n0 if n0 else TOKENS]
    base = Fraction(repr(dl)).limit_denominator(TOKENS).denominator
    lo = max(1, n0, int(TOKENS / (1.0 + dl)))
    return [k * base for k in range(1, TOKENS // base + 1) if lo <= k * base <= TOKENS]


def main():
    rows = load_rows()
    uniq = defaultdict(list)
    amb = defaultdict(list)
    for r in rows:
        e = per_prompt(r)
        if len(e) != 8:
            continue
        for name, entry in e.items():
            dl = entry["effective_mean_draft_len"]
            n0 = entry["non_drafting_round_count"]
            if dl == 0:
                continue
            cands = legal_multiples(dl, n0)
            if not cands:
                continue
            if len(cands) == 1:
                uniq[name].append((TOKENS / cands[0] - 1) / dl)
            else:
                amb[name].append((dl, n0, cands))

    print("acceptance rate implied by UNIQUELY pinned prompt-rows")
    print(f"{'prompt':10s}{'n':>7s}{'p10':>9s}{'p50':>9s}{'p90':>9s}")
    allu = []
    for name in sorted(uniq):
        v = sorted(uniq[name])
        allu.extend(v)
        print(
            f"{name:10s}{len(v):7d}{v[int(0.1*(len(v)-1))]:9.4f}"
            f"{statistics.median(v):9.4f}{v[int(0.9*(len(v)-1))]:9.4f}"
        )
    if allu:
        allu.sort()
        print(
            f"{'ALL':10s}{len(allu):7d}{allu[int(0.1*(len(allu)-1))]:9.4f}"
            f"{statistics.median(allu):9.4f}{allu[int(0.9*(len(allu)-1))]:9.4f}"
        )

    lo, hi = 0.30, 1.00
    print(f"\npruning the ambiguous rows with alpha in [{lo}, {hi}]")
    print(f"{'prompt':10s}{'rows':>7s}{'still >1':>10s}{'now 1':>8s}{'now 0':>8s}")
    for name in sorted(amb):
        n1 = n0c = nm = 0
        for dl, _, cands in amb[name]:
            keep = [c for c in cands if lo <= (TOKENS / c - 1) / dl <= hi]
            if len(keep) == 0:
                n0c += 1
            elif len(keep) == 1:
                n1 += 1
            else:
                nm += 1
        print(f"{name:10s}{len(amb[name]):7d}{nm:10d}{n1:8d}{n0c:8d}")

    print("\nworst-case round-count span left after pruning (max R / min R)")
    for name in sorted(amb):
        spans = []
        for dl, _, cands in amb[name]:
            keep = [c for c in cands if lo <= (TOKENS / c - 1) / dl <= hi] or cands
            spans.append(max(keep) / min(keep))
        spans.sort()
        print(
            f"{name:10s} p50 {statistics.median(spans):6.3f}   "
            f"p90 {spans[int(0.9*(len(spans)-1))]:6.3f}   max {max(spans):6.3f}"
        )


if __name__ == "__main__":
    main()
