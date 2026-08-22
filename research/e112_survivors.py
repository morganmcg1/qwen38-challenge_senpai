"""Which single-mechanism board effects still clear 2 sigma at the corrected floor?

Every mechanism this campaign has mined from the leaderboard was priced against
a plutarch TARGET resolution of 0.0431 %, which Finding 46 showed is the spread
of one narrow replicate class rather than the measurement floor. This re-runs
the mining against the corrected floor and reports the survivors.

Definition of a single-mechanism board effect
---------------------------------------------
A pair of submissions qualifies when all of the following hold:

  * both have a full eight-prompt receipt and a fetched public branch;
  * their eight `effective_mean_draft_len` values are bit-identical, so no
    drafting-policy difference is mixed in;
  * their comment-insensitive code identities differ in EXACTLY ONE path.

Everything else in the binary is then identical, so the plutarch contrast
prices exactly that one file. Pairs are grouped by the ordered pair of
canonical digests of that file, so the same edit found in several pairs pools
together with a consistent sign. The effect is always reported as "how much
faster is the higher-digest side", which is arbitrary but stable.

Sigma
-----
    se = conservative per-pair floor / sqrt(number of pairs)

The conservative floor is the widest single replicate class for the probe, from
`research/board_prompt_instrument.py`. Using the pooled point estimate instead
would inflate every sigma by about 1.36x, which is the error this file exists
to stop repeating.

    python3 research/e112_survivors.py
    python3 research/e112_survivors.py --min-sigma 1.0 --point-estimate
"""

import argparse
import math
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from board_prompt_instrument import (  # noqa: E402
    ANCHOR, CONSERVATIVE, DRAFT_PROBES, MODE_DRAFT_SHIFT, PROMPT_ORDER,
    RESOLUTION, TARGET_PROBE, anchor_canon_lines, blob_text, cand_pct,
    canon_code, canon_digest, changed_blobs, code_identity, collect, git,
    load_rows, probes, ZERO)


def one_path_difference(id_a, id_b):
    """The single path two code identities differ in, or None."""
    paths = {p for p, _ in id_a ^ id_b}
    return paths.pop() if len(paths) == 1 else None


def blob_for(ref, path):
    changed = changed_blobs(ref) or {}
    if path in changed:
        _, new_oid = changed[path]
        return None if new_oid == ZERO else new_oid
    proc = git(["rev-parse", f"{ref}:{path}"])
    return proc.stdout.strip() if proc.returncode == 0 else None


def canon_lines_of(oid, path):
    if oid is None:
        return []
    return canon_code(blob_text(oid), path).split("\n")


def label_for(lines_lo, lines_hi, width=3):
    """Both sides of the canonical line change, lo first then hi."""
    lo, hi = set(lines_lo), set(lines_hi)
    added = [ln.strip() for ln in lines_hi if ln not in lo]
    removed = [ln.strip() for ln in lines_lo if ln not in hi]
    out = []
    for sign, group in (("lo only", removed), ("hi only", added)):
        if not group:
            continue
        shown = " | ".join(s[:70] for s in group[:width])
        more = f" ...+{len(group) - width}" if len(group) > width else ""
        out.append(f"{sign} ({len(group)}): {shown}{more}")
    return "\n    ".join(out) or "no canonical line change"


def mine(recs, same_mode_only=True):
    by_sig = defaultdict(list)
    ids = {}
    for rec in recs:
        key = code_identity(rec["ref"])
        if key is None:
            continue
        ids[rec["id8"]] = key
        by_sig[rec["sig"]].append(rec)

    groups = defaultdict(list)
    for cohort in by_sig.values():
        for i in range(len(cohort)):
            for j in range(i + 1, len(cohort)):
                a, b = cohort[i], cohort[j]
                path = one_path_difference(ids[a["id8"]], ids[b["id8"]])
                if path is None:
                    continue
                oid_a, oid_b = blob_for(a["ref"], path), blob_for(b["ref"], path)
                dig_a = canon_digest(oid_a, path) if oid_a else "DELETED"
                dig_b = canon_digest(oid_b, path) if oid_b else "DELETED"
                if dig_a == dig_b:
                    continue
                lo, hi = (a, b) if dig_a < dig_b else (b, a)
                target, draft = probes(lo["pmap"], hi["pmap"])
                if same_mode_only and abs(draft) > MODE_DRAFT_SHIFT:
                    continue
                groups[(path, min(dig_a, dig_b), max(dig_a, dig_b))].append({
                    "lo": lo, "hi": hi, "target": target, "draft": draft,
                    "oid_lo": oid_a if dig_a < dig_b else oid_b,
                    "oid_hi": oid_b if dig_a < dig_b else oid_a,
                })
    return groups


def in_our_tree(path, lines_lo, lines_hi):
    """Does our base carry the hi side, the lo side, or neither?"""
    ours = set(anchor_canon_lines(path))
    if not ours:
        return "path absent"
    lo, hi = set(lines_lo), set(lines_hi)
    added, removed = hi - lo, lo - hi
    have_added = added and added <= ours
    have_removed = removed and removed <= ours
    if have_added and not have_removed:
        return "hi side"
    if have_removed and not have_added:
        return "lo side"
    if have_added and have_removed:
        return "both"
    return "neither"


def normal_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


def show(rows, limit, header):
    print(f"=== {header}: {len(rows)}\n")
    if not rows:
        print("    empty\n")
    for r in rows[:limit]:
        print(f"{r['sigma']:+7.2f} sigma  pooled {r['pooled']:+.4f} %  "
              f"runs {r['nlo']}v{r['nhi']} in {r['n']} pairs  lines {r['size']}  "
              f"in our tree: {r['ours']}")
        print(f"    {r['path']}")
        print(f"    {r['label']}")
        print("    pairs: " + ", ".join(
            f"{p['lo']['id8']}->{p['hi']['id8']} {p['target']:+.4f}"
            for p in r["pairs"][:6]))
        print(f"    {r['homog']}")
        print()


def price(groups, per_run):
    """Pool every pair of each mechanism and price it against the floor."""
    per_pair = per_run * math.sqrt(2)
    rows = []
    for (path, _, _), pairs in groups.items():
        effects = [p["target"] for p in pairs]
        pooled = statistics.fmean(effects)
        # Pairs inside one group share runs: several trees carrying the lo side
        # are often compared with the SAME tree carrying the hi side. Treating
        # those as independent pairs would claim precision that does not exist.
        # The pair mean is a linear form sum_r c_r x_r over run log-times, so
        # its variance is exactly sigma^2 * sum_r c_r^2 for any incidence
        # structure. Keep the matched pair mean and price it honestly.
        coef = defaultdict(float)
        for p in pairs:
            coef[p["lo"]["id8"]] += 1.0 / len(pairs)
            coef[p["hi"]["id8"]] -= 1.0 / len(pairs)
        lo_runs = {p["lo"]["id8"] for p in pairs}
        hi_runs = {p["hi"]["id8"] for p in pairs}
        se = per_run * math.sqrt(sum(c * c for c in coef.values()))
        lines_lo = canon_lines_of(pairs[0]["oid_lo"], path)
        lines_hi = canon_lines_of(pairs[0]["oid_hi"], path)
        lo, hi = set(lines_lo), set(lines_hi)
        # A mechanism with one transferable effect must scatter no wider than
        # the measurement floor. Wider scatter means the pooled mean describes
        # no tree in particular.
        spread = statistics.stdev(effects) if len(effects) > 1 else 0.0
        ratio = spread / per_pair if len(effects) > 1 else 0.0
        homog = (f"between-pair sd {spread:.4f} % = {ratio:.1f}x the floor"
                 + ("  HETEROGENEOUS, pooled mean not transferable"
                    if ratio > 2.0 else "  homogeneous"))
        rows.append({
            "path": path, "n": len(pairs), "pooled": pooled,
            "nlo": len(lo_runs), "nhi": len(hi_runs),
            "sigma": pooled / se, "size": len(lo ^ hi),
            "spread": spread, "het": ratio > 2.0, "homog": homog,
            "label": label_for(lines_lo, lines_hi),
            "ours": in_our_tree(path, lines_lo, lines_hi),
            "pairs": pairs,
        })
    rows.sort(key=lambda r: -abs(r["sigma"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-sigma", type=float, default=2.0)
    ap.add_argument("--point-estimate", action="store_true",
                    help="price against the pooled floor, not the conservative one")
    ap.add_argument("--max-lines", type=int, default=40,
                    help="a whole-file rewrite is not one mechanism")
    ap.add_argument("--max-report", type=int, default=12)
    args = ap.parse_args(argv)

    per_run = (RESOLUTION["target_same_mode"] if args.point_estimate
               else CONSERVATIVE["target_per_run"])
    per_pair = per_run * math.sqrt(2)
    basis = "pooled point estimate" if args.point_estimate else "conservative"

    recs = collect(load_rows())
    groups = mine(recs)
    npairs = sum(len(v) for v in groups.values())
    print(f"anchor {ANCHOR[:8]}   {len(recs)} scored submissions with a "
          f"fetched tree")
    print(f"{len(groups)} distinct single-file mechanisms over {npairs} "
          f"schedule-matched same-mode pairs")
    print(f"floor basis: {basis}, {per_run:.4f} % per run, "
          f"{per_pair:.4f} % per pair\n")

    rows = price(groups, per_run)

    naive = [r for r in rows if abs(r["sigma"]) >= args.min_sigma]
    expected = len(rows) * normal_two_sided(args.min_sigma)
    print("--- multiplicity calibration, which decides how to read any list "
          "below")
    print(f"mechanisms tested                       {len(rows)}")
    print(f"clear {args.min_sigma} sigma                          "
          f"{len(naive)}")
    print(f"expected by chance under the null       {expected:.1f}")
    print(f"excess over the null                    "
          f"{len(naive) - expected:+.1f} "
          f"({len(naive) / expected:.2f}x)")
    bonf = math.sqrt(2) * _inv_erfc(0.05 / len(rows))
    strong = [r for r in rows if abs(r["sigma"]) >= bonf]
    print(f"Bonferroni 5 % threshold                {bonf:.2f} sigma")
    print(f"mechanisms clearing Bonferroni          {len(strong)}\n")

    single = [r for r in rows if r["size"] <= args.max_lines]
    print(f"--- restricting to a real single mechanism: at most "
          f"{args.max_lines} changed canonical lines")
    print(f"mechanisms that qualify                 {len(single)} of "
          f"{len(rows)}")
    replicated = [r for r in single if r["n"] > 1]
    print(f"of those, measured by 2 or more pairs   {len(replicated)}\n")

    passing = [r for r in replicated if abs(r["sigma"]) >= args.min_sigma]
    show([r for r in passing if not r["het"]], args.max_report,
         f"DEFENSIBLE SURVIVORS: small edit, replicated, homogeneous, >= "
         f"{args.min_sigma} sigma")
    show([r for r in passing if r["het"]], args.max_report,
         "REJECTED for heterogeneity: the pairs disagree by more than the "
         "floor, so\n    the pooled effect is not a property of the edit")

    show([r for r in single
          if abs(r["sigma"]) >= args.min_sigma and r["n"] == 1],
         5, f"single-pair small-edit claims at >= {args.min_sigma} sigma "
            f"(NOT defensible, shown for scale)")

    print("--- strongest replicated small-edit mechanisms, whatever the sigma")
    print(f"{'sigma':>8} {'pooled %':>9} {'n':>3} {'lines':>6} {'in tree':>10}"
          f"  file")
    for r in replicated[:12]:
        print(f"{r['sigma']:+8.2f} {r['pooled']:+9.4f} {r['n']:3d} "
              f"{r['size']:6d} {r['ours']:>10}  "
              f"{r['path'].rsplit('/', 1)[-1]}")

    print(f"\na single pair needs {args.min_sigma * per_pair:.4f} % to reach "
          f"{args.min_sigma} sigma; a two-pair mechanism needs "
          f"{args.min_sigma * per_pair / math.sqrt(2):.4f} %")
    return 0


def _inv_erfc(y, lo=0.0, hi=10.0, iters=200):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if math.erfc(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


if __name__ == "__main__":
    sys.exit(main())
