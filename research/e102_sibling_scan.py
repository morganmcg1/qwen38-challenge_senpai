#!/usr/bin/env python3
"""E102 helper: how isolated can a wide-row contrast be made on the board?

The crown draft schedule is shared by hundreds of board runs, so "same
schedule" alone leaves a 268-way choice of sibling and the answer swings with
the pick. This scan reports, for each wide-row target, how many same-schedule
cohort members also match on the ``Sources/MLXFastModel`` subtree and carry the
unmodified BASE QMV kernel. Those are the only members for which the QMV kernel
is close to the single difference.
"""

import json
import subprocess
import sys

sys.path.insert(0, "research")
import e102_wide_row_pricing as e102  # noqa: E402


REFS = {}
for _line in open("/tmp/e102_trees.txt"):
    _ref = _line.split()[0]
    REFS[_ref.rsplit("/", 1)[-1][:8]] = _ref


def diff_files(a, b):
    if a[:8] not in REFS or b[:8] not in REFS:
        return None
    out = subprocess.run(["git", "diff", "--name-only", REFS[a[:8]], REFS[b[:8]]],
                         capture_output=True, text=True, check=True).stdout
    return [x for x in out.splitlines() if x]


def main():
    scored = e102.scored_rows()
    fp = e102.fingerprints()
    for prefix, note in e102.TARGETS:
        hits = [r for r in scored if r["id"].startswith(prefix)]
        if not hits:
            print("\n%s  NOT SCORED (no per-prompt rows)  %s" % (prefix, note))
            continue
        row = hits[0]
        coh = [r for r in e102.cohort_of(scored, row) if r["id"] != row["id"]]
        mine = fp.get(prefix, {})
        same_mtp = [r for r in coh
                    if fp.get(r["id"][:8], {}).get(e102.MTP_DIR) == mine.get(e102.MTP_DIR)]
        base_qmv = [r for r in same_mtp if e102.qmv_label(fp, r["id"]) == "BASE-qmv"]
        same_user = [r for r in coh if r.get("solverUsername") == row.get("solverUsername")]
        print("\n%s %-14s cohort %d | same MTP subtree %d | that AND BASE-qmv %d | same solver %d"
              % (prefix, str(row.get("solverUsername"))[:14], len(coh) + 1,
                 len(same_mtp), len(base_qmv), len(same_user)))
        for r in sorted(base_qmv, key=lambda z: z.get("createdAt") or "")[:6]:
            files = diff_files(prefix, r["id"][:8])
            print("    BASE-qmv sibling %s %-14s %s  files differing %d: %s"
                  % (r["id"][:8], str(r.get("solverUsername"))[:14],
                     (r.get("createdAt") or "")[:10], (len(files) if files else -1),
                     " ".join(f.split("/")[-1] for f in (files or [])[:8])))
        for r in sorted(same_user, key=lambda z: z.get("createdAt") or ""):
            files = diff_files(prefix, r["id"][:8])
            print("    same-solver      %s %s  qmv %-22s files %d: %s"
                  % (r["id"][:8], (r.get("createdAt") or "")[:10],
                     e102.qmv_label(fp, r["id"]), (len(files) if files else -1),
                     " ".join(f.split("/")[-1] for f in (files or [])[:8])))


if __name__ == "__main__":
    main()
