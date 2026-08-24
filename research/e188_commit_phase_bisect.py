#!/usr/bin/env python3
"""E188 Stage 2, step 1. Source bisection of the decode commit-phase growth.

E185 measured the per-round commit phase (the window between `tReadDone` and
`tCommitDone` in `Qwen36MTPBlockSession.swift`) growing about 2.5x over the
campaign: 183.9 us at 59b67f50 (E90), 433.1 us in the E165 era, 455.8 us on the
current base, and 100.00% GPU-idle in both instruments. About 272 us/round of
new host-side seam cost appeared and was never explained.

Timed bisection costs one release build plus one metallib rebuild per point.
This script narrows the candidate set for free first: it walks the maintained
base's first-parent merge history, extracts the commit-phase window body AND
the body of every function that window calls, and hashes each one. A merge that
changes none of those hashes cannot have moved the commit phase through source,
so only the merges that do change a hash need a timed leg.

Usage: python3 research/e188_commit_phase_bisect.py [--json OUT]
"""
import argparse
import hashlib
import json
import re
import subprocess

FILE = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
OTHER = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"

# First-parent merges from the E90 base to the current tip that touch FILE,
# oldest first, plus the E90 base itself as the anchor.
REVS = ["59b67f50", "cedb900b", "b81a43d4", "29a42070", "e2b1ab00", "b129f202",
        "67fedb4a", "4e45517b", "7a427dfa", "601c137c", "5a62150c", "eec2c14b",
        "806181de", "b51f893a", "698f2a89", "bd74819b", "HEAD"]

# Everything the commit-phase window calls, directly or through one hop.
CALLEES = ["hiddenRow", "clearRecurrentRollback", "restoreAfterPrefixReject",
           "rollbackAfterVerify", "runHeadUpkeep", "prefetchHeadStep",
           "undoHeadPrefetch", "normedRows", "snapshotRecurrent"]


def show(rev, path):
    try:
        return subprocess.run(["git", "show", f"{rev}:{path}"],
                              capture_output=True, text=True,
                              check=True).stdout
    except subprocess.CalledProcessError:
        return None


def window(src, start_pat, end_pat):
    lines = src.splitlines()
    a = b = None
    for i, ln in enumerate(lines):
        if a is None and start_pat in ln:
            a = i
        elif a is not None and end_pat in ln:
            b = i
            break
    if a is None or b is None:
        return None
    return "\n".join(lines[a + 1:b])


def func_body(src, name):
    """Body of `func <name>(` by brace matching. Returns None when absent."""
    m = re.search(r"func\s+" + re.escape(name) + r"\s*[(<]", src)
    if not m:
        return None
    i = src.index("{", m.end())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
        j += 1
    return None


def strip(code):
    """Drop comments and blank lines so pure comment edits do not look like
    behaviour changes."""
    out = []
    for ln in code.splitlines():
        s = ln.strip()
        if not s or s.startswith("//"):
            continue
        out.append(s)
    return "\n".join(out)


def h(code):
    return "-" if code is None else hashlib.sha256(
        strip(code).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = []
    for rev in REVS:
        src = show(rev, FILE)
        if src is None:
            continue
        win = window(src, "tReadDone = DispatchTime", "tCommitDone = DispatchTime")
        rec = {"rev": rev,
               "sha": subprocess.run(["git", "rev-parse", "--short=8", rev],
                                     capture_output=True, text=True).stdout.strip(),
               "window": h(win),
               "windowCodeLines": len(strip(win).splitlines()) if win else 0}
        for c in CALLEES:
            rec[c] = h(func_body(src, c))
        other = show(rev, OTHER)
        rec["qwen35"] = h(other) if other else "-"
        rows.append(rec)

    cols = ["window"] + CALLEES + ["qwen35"]
    print("Commit-phase source bisection. '=' means unchanged from the previous")
    print("revision; a hash prefix marks the revision that changed it.\n")
    hdr = f"{'rev':10s} {'sha':9s} " + " ".join(f"{c[:11]:>11s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    prev = None
    changed_at = {c: [] for c in cols}
    for r in rows:
        cells = []
        for c in cols:
            if prev is not None and prev[c] == r[c]:
                cells.append(f"{'=':>11s}")
            else:
                cells.append(f"{r[c]:>11s}")
                if prev is not None:
                    changed_at[c].append(r["sha"])
        print(f"{r['rev']:10s} {r['sha']:9s} " + " ".join(cells))
        prev = r

    print("\nMerges that changed each commit-phase component:")
    for c in cols:
        print(f"  {c:26s} {changed_at[c] if changed_at[c] else 'never changed'}")

    suspects = sorted({s for c in cols if c != "qwen35" for s in changed_at[c]})
    print(f"\nCandidate merges needing a timed leg: {suspects}")
    print(f"Merges provably inert for the commit phase: "
          f"{[r['sha'] for r in rows[1:] if r['sha'] not in suspects]}")

    if args.json:
        json.dump({"rows": rows, "changedAt": changed_at,
                   "suspects": suspects}, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
