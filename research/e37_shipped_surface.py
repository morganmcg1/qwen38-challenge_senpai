#!/usr/bin/env python3
"""Independent re-derivation of the campaign's shipped surface, and of this
branch's contribution to it.

The r2 addendum retracts the "frozen at 4 files, +117/-87" constraint: against
the true campaign baseline the shipped surface is larger, and the gate that
kept reporting the old figure never named the commit it compared against. The
lesson generalises, so this script never has an implicit baseline: both ends of
every comparison are printed, and the exit status depends on them.

  python3 research/e37_shipped_surface.py
  python3 research/e37_shipped_surface.py --baseline 5273067 --base 0491f9e5

Two questions, answered separately:

  A. what IS the shipped surface (baseline -> PR base)?   informational
  B. what does THIS branch add to it (PR base -> HEAD)?   must be empty

(B) is the assignment-scope rule and is the only one that gates. It ships with
a negative control: the same predicate is evaluated on a range that genuinely
touches shipped files, and is required to report a violation there. A
zero-bytes-added check that has never seen a non-zero input is not evidence.

Shipped surface = the paths a submission packages: Sources/ and Vendor/.
Tests/, senpai/ and research/ are campaign scaffolding and are reported
separately rather than silently folded in.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SHIPPED = ["Sources", "Vendor"]
SCAFFOLD = ["Tests", "senpai", "research", ".agents"]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          check=True, capture_output=True, text=True).stdout


def numstat(a: str, b: str, paths: list[str]) -> list[tuple[int, int, str]]:
    out = git("diff", "--numstat", "%s..%s" % (a, b), "--", *paths)
    rows = []
    for line in out.splitlines():
        add, dele, path = line.split("\t", 2)
        rows.append((int(add) if add != "-" else 0,
                     int(dele) if dele != "-" else 0, path))
    return sorted(rows, key=lambda r: r[2])


def totals(rows: list[tuple[int, int, str]]) -> tuple[int, int, int]:
    return len(rows), sum(r[0] for r in rows), sum(r[1] for r in rows)


def show(title: str, a: str, b: str, rows: list[tuple[int, int, str]]) -> None:
    n, add, dele = totals(rows)
    print("\n%s" % title)
    print("  range: %s..%s" % (git("rev-parse", "--short", a).strip(),
                               git("rev-parse", "--short", b).strip()))
    for r in rows:
        print("    +%-5d -%-5d %s" % (r[0], r[1], r[2]))
    if not rows:
        print("    (no files)")
    print("  total: %d files, +%d/-%d" % (n, add, dele))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="5273067",
                    help="true campaign baseline the shipped surface is measured from")
    ap.add_argument("--base", default="0491f9e5", help="this PR's base commit")
    ap.add_argument("--head", default="HEAD", help="this branch head")
    ap.add_argument("--json", default="research/results/e37/r2-shipped-surface.json")
    args = ap.parse_args()

    print("=" * 78)
    print("E37 r2 addendum - shipped surface, with both ends of every range named")
    print("=" * 78)

    surface = numstat(args.baseline, args.base, SHIPPED)
    show("A. campaign shipped surface (baseline -> PR base) [informational]",
         args.baseline, args.base, surface)

    mine_shipped = numstat(args.base, args.head, SHIPPED)
    show("B. THIS branch on the shipped surface (must be empty) [gating]",
         args.base, args.head, mine_shipped)

    mine_scaffold = numstat(args.base, args.head, SCAFFOLD)
    show("C. this branch on campaign scaffolding [informational]",
         args.base, args.head, mine_scaffold)

    non_research = sorted({r[2] for r in mine_scaffold
                           if not r[2].startswith("research/")})

    clean = not mine_shipped and not non_research
    print("\n-- negative control " + "-" * 56)
    # Feed the SAME predicate a range that genuinely edits shipped files. If it
    # reports clean there, the predicate is inert and (B) above proves nothing.
    control = numstat(args.baseline, args.base, SHIPPED)
    control_fires = bool(control)
    print("  predicate on a known-dirty range (%s..%s): %s"
          % (args.baseline, args.base,
             "VIOLATION reported (%d files) -- control OK" % len(control)
             if control_fires else "reported clean -- CONTROL DEAD"))

    print("\n-- verdict " + "-" * 65)
    print("  shipped-surface bytes added by this branch: %d files, +%d/-%d"
          % totals(mine_shipped))
    print("  non-research files touched by this branch : %s"
          % (", ".join(non_research) if non_research else "none"))
    print("  assignment-scope rule (research/ only)    : %s"
          % ("PASS" if clean else "FAIL"))
    print("  negative control                          : %s"
          % ("PASS" if control_fires else "FAIL"))

    payload = {
        "schema": "e37-r2-shipped-surface/1",
        "baseline": git("rev-parse", args.baseline).strip(),
        "base": git("rev-parse", args.base).strip(),
        "head": git("rev-parse", args.head).strip(),
        "shipped_paths": SHIPPED,
        "campaign_shipped_surface": {
            "files": [r[2] for r in surface],
            "n_files": totals(surface)[0],
            "insertions": totals(surface)[1],
            "deletions": totals(surface)[2],
        },
        "branch_on_shipped_surface": {
            "n_files": totals(mine_shipped)[0],
            "insertions": totals(mine_shipped)[1],
            "deletions": totals(mine_shipped)[2],
        },
        "branch_non_research_files": non_research,
        "head_note": ("`head` is the commit this gate ran against. The commit that "
                      "publishes this file adds research/ paths only, which cannot "
                      "alter the gating predicate (shipped surface = Sources + Vendor)."),
        "scope_pass": clean,
        "negative_control_fires": control_fires,
        "gate_pass": clean and control_fires,
    }
    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("  wrote %s" % args.json)
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
