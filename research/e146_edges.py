"""E146 R-A: print every declared-replicate edge with its evidence.

An edge is only as good as the declaration behind it, so this prints the title
that produced it and the k the pair returns. Read this before trusting any
cluster.
"""

import sys

import e146_lib as L
import e146_replicates as R

path, rows = L.load()
built = R.build(rows)
by = {r.id8: r for r in rows}

declared = built["declared"]
print("board %s  scored %d  declared %d" % (path, len(rows), len(declared)))
print()
for row in sorted(declared, key=lambda r: r.created):
    targets = R.resolve_targets(row, rows)
    same = [t for t in targets if t.draft_key() == row.draft_key()]
    target = sorted(same, key=lambda r: r.created)[-1] if same else None
    line = "%s %-14s %s score %.6f" % (
        row.id8, row.solver[:14], row.created[5:19], row.score or float("nan"))
    print(line)
    print("    title: %s" % R.title_of(row)[:120])
    if target is None:
        others = sorted(targets, key=lambda r: r.created)
        print("    target: NONE resolved with the same schedule (%d other refs)"
              % len(others))
        continue
    fit = L.fit_k(target, row, basis_row=target)
    print("    target %s %-14s score %.6f   k=%+8.1f us/dr  cand8=%+.4f%%  resid=%.4f"
          % (target.id8, target.solver[:14], target.score or float("nan"),
             fit["k_us_per_drafting_round"], fit["cand_mean8_pct"],
             fit["residual_sd_pp"]))
sys.exit(0)
