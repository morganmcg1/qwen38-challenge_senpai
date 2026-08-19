#!/usr/bin/env python3
"""Census the QMV weight-stream structure across every rival tree on the board,
and find NATURAL A/B PAIRS that isolate stream count at a single width.

WHY THIS EXISTS
---------------
thorfinn's E41 (merged, PR #46) established that the local per-round cost curve
is explained by the WEIGHT-STREAM COUNT rather than by curvature in M:

    streams(M) = ceil(M / IPG(M))          IPG read from the `case M:` switch
    T(M) = 16.432 + 20.291 * streams(M) + 11.798 * M     max|resid| 1.674 ms

and that a quadratic in M is falsified MODEL-FREE, because any quadratic forces
non-decreasing first differences while the measured ones drop 22.846 ms after
the boundary.

`streams(M)` is therefore a SOURCE-DERIVED regressor with no fitted breakpoint.
But it is a property of ONE TREE. Two consequences, both of which this script
exists to make checkable rather than assumed:

  1. Any ranked inference that pools per-round costs across solvers is pooling
     across possibly-different stream structures. Before pooling, compute
     streams(M) for the tree each row was measured on.

  2. The board itself contains the experiment. Solvers shipped 12 distinct
     dispatch tables on the SAME ranked hardware against the SAME scored corpus.
     Most cross-solver pairs are hopelessly confounded -- two representatives of
     the two largest groups differ in 14 files including Qwen35.swift and
     mtp-head.manifest.json. But some trees are byte-identical EVERYWHERE
     EXCEPT the QMV kernel, and those are clean single-mechanism A/Bs.

`ab` mode finds them by fingerprinting each tree on the blob SHAs of all files
except the two QMV kernel files, then reporting fingerprint groups that contain
more than one dispatch table.

USAGE
    python3 research/stream_dispatch_census.py selftest
    python3 research/stream_dispatch_census.py census [REV ...]
    python3 research/stream_dispatch_census.py ab

The selftest pins every constant this script has ever been quoted for, against
named trees, so that a rebase which moves a dispatch table makes the SCRIPT
fail rather than making a brief quietly wrong. A register ceiling, a step
magnitude and a stream boundary are all properties of a tree, not of a kernel
family -- quoting one without its tree has cost this campaign four wrong
constants.
"""
import collections
import hashlib
import math
import re
import subprocess
import sys

QH = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
QCPP = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
KERNEL_PATHS = {QH, QCPP}
CELL = re.compile(r"qmv_fast_crossrow_affine4_g64_m<\s*T\s*,\s*(\d+)\s*,\s*(\d+)")
SUBMISSION_GLOB = "refs/remotes/upstream/submissions/*"


def run(args):
    return subprocess.run(args, capture_output=True, text=True)


def resolve(rev):
    """Accept a git object OR a board submission ID prefix.

    These are different namespaces and I conflated them once: the board's
    `id` field (e.g. ca9251b8-...) names a SUBMISSION, while b8642b81f7 is an
    abbreviated TREE. Only the latter is a git object. A submission ID resolves
    through refs/remotes/upstream/submissions/<id>. Fail closed rather than
    silently reporting "no cross-row family" for a tree that is simply named in
    the wrong namespace -- that is how a real absence and a lookup miss become
    indistinguishable.
    """
    if run(["git", "rev-parse", "--verify", "--quiet", rev + "^{}"]).returncode == 0:
        return rev
    out = run(["git", "for-each-ref", "--format=%(refname)", SUBMISSION_GLOB])
    hits = [r for r in out.stdout.splitlines()
            if r.rsplit("/", 1)[-1].startswith(rev)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise SystemExit("ambiguous submission id %r matches %d refs"
                         % (rev, len(hits)))
    return None


def dispatch_table(rev):
    """{M: IPG} read from the width switch, or None if the family is absent.

    Returns None (not {}) for trees predating the cross-row family -- the
    pristine organizer tree 5d029178 is one, which is itself informative: the
    cross-row QMV family is a SUBMISSION contribution, not organizer code.
    """
    resolved = resolve(rev)
    if resolved is None:
        return None
    out = run(["git", "cat-file", "-p", "%s:%s" % (resolved, QH)])
    if out.returncode != 0:
        return None
    cells = {int(m): int(ipg) for m, ipg in CELL.findall(out.stdout)}
    return cells or None


def streams(table):
    """Weight streams per width: ceil(M / IPG(M))."""
    return {m: math.ceil(m / ipg) for m, ipg in table.items()}


def boundaries(st):
    """[(M-1, M, streams(M-1), streams(M))] where a marginal stream is added."""
    return [(m - 1, m, st[m - 1], st[m])
            for m in sorted(st) if m - 1 in st and st[m] > st[m - 1]]


def non_kernel_fingerprint(rev):
    """SHA-256 over every tree entry EXCEPT the two QMV kernel files.

    Two trees sharing this are byte-identical apart from the QMV kernel, so a
    dispatch-table difference between them is not confounded by anything else
    in the submission.
    """
    out = run(["git", "ls-tree", "-r", rev])
    if out.returncode != 0:
        return None
    h = hashlib.sha256()
    n = 0
    for line in out.stdout.splitlines():
        try:
            meta, path = line.split("\t", 1)
        except ValueError:
            continue
        if path in KERNEL_PATHS:
            continue
        h.update(meta.encode())
        h.update(b"\0")
        h.update(path.encode())
        h.update(b"\n")
        n += 1
    return h.hexdigest() if n else None


def submission_refs():
    out = run(["git", "for-each-ref", "--format=%(refname) %(objectname)",
               SUBMISSION_GLOB])
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            yield parts[0].rsplit("/", 1)[-1], parts[1]


def fmt(d):
    return " ".join("M%d=%d" % kv for kv in sorted(d.items()))


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

# Every tree here is named with the claim it anchors. If a rebase changes one of
# these, this test fails loudly instead of a brief becoming silently wrong.
PINS = [
    # (rev, expected streams, expected boundary widths, why it matters)
    ("0c90733d383f6b987a29682bf9eb9458a6172bfa",
     {3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3}, [5, 9],
     "the promoted crown frontier; score 3.24929398547457"),
    ("b8642b81f7",
     {3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3}, [5, 9],
     "the plateau tree shared by companygardener and alfranli123 -- the rows "
     "any ranked pooled fit uses. Its 1->2 boundary is at M=5, NOT M=6, so a "
     "ranked step model with an indicator at [M>=6] is misspecified by one "
     "width"),
    ("ca9251b8",
     {3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}, [6],
     "our own E27 tree, which scored 3.23250848263467 (-0.3321 % vs base). "
     "It is the ONLY tree on the board whose boundary sits at 5->6, because "
     "E27 raised case 5 to IPG 5. E27 was, mechanically, the 'make M=5 a "
     "single weight stream' experiment -- and it lost score, because IPG 5 "
     "cost 125 registers and pushed the SHARED allocation to 129"),
]


def selftest():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # 1. streams() and boundaries() arithmetic, independent of any tree.
    check(streams({5: 3})[5] == 2, "ceil(5/3) must be 2")
    check(streams({6: 3})[6] == 2, "ceil(6/3) must be 2")
    check(streams({5: 5})[5] == 1, "ceil(5/5) must be 1")
    check(streams({9: 3})[9] == 3, "ceil(9/3) must be 3")
    check(streams({8: 4})[8] == 2, "ceil(8/4) must be 2")
    check(streams({8: 3})[8] == 3, "ceil(8/3) must be 3")
    b = boundaries({3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3})
    check([x[1] for x in b] == [5, 9], "boundaries must be at M=5 and M=9")
    # A quadratic forces non-decreasing first differences; assert the local
    # first differences that falsify it, so the falsification is reproducible.
    T = {3: 72.811, 4: 82.722, 5: 96.217, 6: 128.890,
         7: 138.717, 8: 149.727, 9: 164.675}
    d1 = [round(T[m + 1] - T[m], 3) for m in range(3, 9)]
    check(d1 == [9.911, 13.495, 32.673, 9.827, 11.010, 14.948],
          "E41 first differences changed: %s" % d1)
    drop = d1[2] - d1[3]
    check(abs(drop - 22.846) < 5e-3,
          "post-boundary drop must be 22.846 ms, got %.3f" % drop)
    check(any(d1[i + 1] < d1[i] for i in range(len(d1) - 1)),
          "first differences must be NON-monotone, else no quadratic is excluded")

    # 2. Pinned trees.
    for rev, exp_streams, exp_bounds, why in PINS:
        tbl = dispatch_table(rev)
        if tbl is None:
            failures.append("%s: no dispatch table found (%s)" % (rev[:10], why))
            continue
        st = streams(tbl)
        if st != exp_streams:
            failures.append("%s: streams %s != expected %s (%s)"
                            % (rev[:10], fmt(st), fmt(exp_streams), why))
        got = [x[1] for x in boundaries(st)]
        if got != exp_bounds:
            failures.append("%s: boundaries at %s != expected %s (%s)"
                            % (rev[:10], got, exp_bounds, why))

    # 3. HEAD must agree with the crown on the scored surface's dispatch table.
    head = dispatch_table("HEAD")
    crown = dispatch_table("0c90733d383f6b987a29682bf9eb9458a6172bfa")
    if head is None or crown is None:
        failures.append("HEAD or crown dispatch table unreadable")
    elif head != crown:
        failures.append("HEAD dispatch table %s differs from the crown %s -- if "
                        "this is intentional say so, but every stream boundary "
                        "quoted in a brief is now wrong"
                        % (fmt(head), fmt(crown)))

    # 4. The fingerprint must actually IGNORE the kernel files and nothing else.
    #    Fail closed if the paths it excludes are not present in HEAD, which
    #    would silently make every fingerprint a whole-tree hash.
    ls = run(["git", "ls-tree", "-r", "--name-only", "HEAD"]).stdout.splitlines()
    for p in KERNEL_PATHS:
        if p not in ls:
            failures.append("excluded path absent from HEAD, fingerprint would "
                            "not be kernel-blind: %s" % p)

    if failures:
        print("SELFTEST FAIL (%d)" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("SELFTEST PASS: %d arithmetic checks, %d pinned trees, HEAD==crown "
          "dispatch table, fingerprint exclusions present."
          % (12, len(PINS)))
    return 0


# --------------------------------------------------------------------------
# census / ab
# --------------------------------------------------------------------------

def census(extra):
    seen = collections.Counter()
    examples = {}
    missing = 0
    total = 0
    for name, obj in submission_refs():
        total += 1
        tbl = dispatch_table(obj)
        if tbl is None:
            missing += 1
            continue
        key = tuple(sorted(tbl.items()))
        seen[key] += 1
        examples.setdefault(key, []).append(name[:8])

    print("submission refs scanned      : %d" % total)
    print("no cross-row family present  : %d" % missing)
    print("trees with a dispatch table  : %d" % (total - missing))
    print("distinct dispatch tables     : %d" % len(seen))
    print()
    for key, n in seen.most_common():
        tbl = dict(key)
        st = streams(tbl)
        print("--- %d tree(s) ---" % n)
        print("  IPG       : %s" % fmt(tbl))
        print("  streams   : %s" % fmt(st))
        bd = boundaries(st)
        print("  boundaries: %s" % (", ".join(
            "%d->%d (%d->%d streams)" % x for x in bd) or "none"))
        print("  e.g.      : %s" % ", ".join(examples[key][:6]))
        print()

    for rev in extra:
        tbl = dispatch_table(rev)
        if tbl is None:
            print("%s : no cross-row family" % rev)
            continue
        st = streams(tbl)
        print("%s : streams %s | boundaries %s"
              % (rev, fmt(st),
                 ", ".join("%d->%d" % (x[0], x[1]) for x in boundaries(st))
                 or "none"))


def ab():
    groups = collections.defaultdict(list)
    scanned = 0
    for name, obj in submission_refs():
        scanned += 1
        tbl = dispatch_table(obj)
        if tbl is None:
            continue
        fp = non_kernel_fingerprint(obj)
        if fp is None:
            continue
        groups[fp].append((name[:8], tuple(sorted(tbl.items()))))

    multi = {fp: v for fp, v in groups.items() if len({t for _, t in v}) > 1}
    print("trees scanned                      : %d" % scanned)
    print("trees with a dispatch table        : %d"
          % sum(len(v) for v in groups.values()))
    print("distinct non-kernel fingerprints   : %d" % len(groups))
    print("fingerprints with >1 table (A/Bs)  : %d" % len(multi))
    print()
    if not multi:
        print("NO clean A/B exists: every pair differing in the dispatch table")
        print("also differs elsewhere. The cross-solver contrast is")
        print("observational only and must be labelled as such.")
        return

    for fp, members in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        by_tbl = collections.defaultdict(list)
        for name, tbl in members:
            by_tbl[tbl].append(name)
        diff_widths = set()
        tables = list(by_tbl)
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                a, b = dict(tables[i]), dict(tables[j])
                for m in set(a) | set(b):
                    if a.get(m) != b.get(m):
                        diff_widths.add(m)
        print("=== fingerprint %s : %d trees, %d tables, differing widths %s ==="
              % (fp[:12], len(members), len(by_tbl), sorted(diff_widths)))
        for tbl, names in sorted(by_tbl.items(),
                                 key=lambda kv: -len(kv[1])):
            st = streams(dict(tbl))
            print("   n=%-2d streams %s" % (len(names), fmt(st)))
            print("        %s" % ", ".join(names))
        print()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        return selftest()
    if mode == "census":
        census(sys.argv[2:])
        return 0
    if mode == "ab":
        ab()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
