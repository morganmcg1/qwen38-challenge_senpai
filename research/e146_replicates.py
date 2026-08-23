"""E146 R-A rung 1: the declared-replicate census.

A REPLICATE IS A DECLARATION PLUS A FINGERPRINT. Solvers routinely resubmit an
unchanged tree to sample the measurement. Those rows say so in the note. The
declaration alone is not evidence, so every declared pair is checked against the
T32 schedule fingerprint: the eight `effective_mean_draft_len` values must be
digit-identical. A declared resample whose schedule moved is reported separately
and never enters the nuisance sample.

WHY NOT THE F211/F216 CONSTRUCTION. Those pair sets are selected BY TIGHTNESS:
a pair enters the sample when its eight candidate legs already agree. That
conditions the sample on a small difference, so the sd it returns is a lower
bound on nuisance, not an estimate of it. The declared-replicate construction
conditions on the solver's INTENT instead, which is independent of the draw, so
it can observe a large nuisance difference when one occurs.
"""

import collections
import json
import os
import re
import sys

import e146_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-replicates.json")

# A declaration is read from the note TITLE, which is where solvers state the
# purpose of the whole submission. A body mention of "byte-identical" usually
# describes one file inside a real mechanism change, so the body is not used.
TITLE_DECLARES_REPLICATE = re.compile(
    r"(zero[- ]?d(elta|iff)|same[- ]content|pure resample|resample|redraw|"
    r"variance (resample|draw|ticket)|note-only|fresh-marker)", re.I)
# `resample` also appears inside "on first mismatch, resample and continue",
# which is a description of the decode rule and not a declaration.
TITLE_REJECTS = re.compile(r"mismatch, resample", re.I)
HEX = re.compile(r"\b([0-9a-f]{7,40})\b")


def title_of(row):
    for line in row.note.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def declares_replicate(row):
    title = title_of(row)
    if not title or TITLE_REJECTS.search(title):
        return False
    return bool(TITLE_DECLARES_REPLICATE.search(title))


def resolve_targets(row, rows):
    """Rows named by a hex reference in the note title or its first lines."""
    head = "\n".join(row.note.split("\n")[:24])
    refs = set(HEX.findall(head.lower()))
    out = []
    for other in rows:
        if other.id8 == row.id8 or other.created >= row.created:
            continue
        keys = [other.id.lower(), (other.commit or "").lower(),
                (other.source_ref or "").lower()]
        for ref in refs:
            if any(key.startswith(ref) for key in keys if key):
                out.append(other)
                break
    return out


def build(rows):
    by_key = collections.defaultdict(list)
    for row in rows:
        by_key[row.draft_key()].append(row)

    declared = [r for r in rows if declares_replicate(r)]
    edges = []
    unresolved = []
    schedule_moved = []
    for row in declared:
        targets = resolve_targets(row, rows)
        same_schedule = [t for t in targets if t.draft_key() == row.draft_key()]
        if same_schedule:
            # The most recent same-schedule ancestor named in the note.
            target = sorted(same_schedule, key=lambda r: r.created)[-1]
            edges.append((row, target))
        elif targets:
            schedule_moved.append((row, sorted(targets, key=lambda r: r.created)[-1]))
        else:
            unresolved.append(row)

    # Same solver, same declared series title, same schedule: `Variance resample
    # #7` and `#8` of one tree are replicates of each other even when neither
    # note names the other's submission id.
    series = collections.defaultdict(list)
    for row in declared:
        stem = re.sub(r"[#0-9]+", "", title_of(row)).strip().lower()
        series[(row.solver, stem, row.draft_key())].append(row)
    for members in series.values():
        members.sort(key=lambda r: r.created)
        for a, b in zip(members, members[1:]):
            edges.append((b, a))

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row, target in edges:
        union(row.id8, target.id8)

    groups = collections.defaultdict(list)
    by_id = {r.id8: r for r in rows}
    for node in list(parent):
        groups[find(node)].append(by_id[node])
    clusters = []
    for members in groups.values():
        members.sort(key=lambda r: r.created)
        keys = {m.draft_key() for m in members}
        if len(keys) != 1:
            # Union-find can bridge two schedules through a chained declaration.
            # Split on the fingerprint rather than trust the chain.
            by_sched = collections.defaultdict(list)
            for m in members:
                by_sched[m.draft_key()].append(m)
            for part in by_sched.values():
                if len(part) >= 2:
                    clusters.append(sorted(part, key=lambda r: r.created))
        elif len(members) >= 2:
            clusters.append(members)
    clusters.sort(key=lambda c: (-len(c), c[0].created))
    return {
        "clusters": clusters,
        "declared": declared,
        "unresolved": unresolved,
        "schedule_moved": schedule_moved,
        "by_key": by_key,
    }


def main():
    path, rows = L.load()
    built = build(rows)
    clusters = built["clusters"]
    sizes = collections.Counter(len(c) for c in clusters)
    print("board %s scored rows %d" % (path, len(rows)))
    print("declared replicate rows            %d" % len(built["declared"]))
    print("  resolved into clusters           %d rows in %d clusters"
          % (sum(len(c) for c in clusters), len(clusters)))
    print("  declared but schedule moved      %d" % len(built["schedule_moved"]))
    print("  declared but no target resolved  %d" % len(built["unresolved"]))
    print("cluster size histogram             %s" % sorted(sizes.items()))
    print("largest cluster                    %d"
          % (max((len(c) for c in clusters), default=0))) 

    print("\n--- replicate clusters ---")
    for cluster in clusters:
        head = cluster[0]
        print("n=%-2d  %s  %s" % (len(cluster), head.solver[:16].ljust(16),
                                  " ".join(m.id8 for m in cluster)))

    payload = {
        "harness": "ranked",
        "board_path": path,
        "scored_rows": len(rows),
        "declared_rows": [r.id8 for r in built["declared"]],
        "unresolved": [r.id8 for r in built["unresolved"]],
        "schedule_moved": [[r.id8, t.id8] for r, t in built["schedule_moved"]],
        "clusters": [[m.id8 for m in c] for c in clusters],
        "cluster_size_histogram": dict(sorted(sizes.items())),
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
