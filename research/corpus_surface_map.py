#!/usr/bin/env python3
"""Statistical map of the public Qwen 3.8 27B native-MTP leaderboard corpus.

Joins the 712 `upstream/submissions/<uuid>` refs (exact submitted trees; each
commit's first parent is the organizer main of its day) with the public
telemetry in `research/e53-board-facts.json` (408 content-unique scored rows),
then answers:

  A. which scored-surface files the field touches, with score stats;
  B. the unexplored editable surface (editablePaths minus everything ever
     touched by any submission ref);
  C. whether scored-surface diff size predicts score;
  D. the top-five solvers, their trajectories, and repeated files;
  E. which files are most associated with refs that never produced a
     telemetry score (landmines);
  F. per-prompt raw_ratio winners and whether one tree wins everywhere.

Read-only: this script never checks out, stages, or copies rival source.
File lists come from two batched `git log --no-walk` calls, so the whole run
finishes in seconds. Run: python3 research/corpus_surface_map.py
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FACTS = HERE / "e53-board-facts.json"

SCORED_PREFIXES = (
    "Sources/",
    "Vendor/",
    "benchmark.json",
    "mtp-head.manifest.json",
    "mtp-head/",
    "Package.swift",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def submission_refs() -> dict[str, str]:
    out = git(
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/remotes/upstream/submissions",
    )
    refs: dict[str, str] = {}
    for line in out.splitlines():
        refname, sha = line.split()
        refs[refname.rsplit("/", 1)[-1]] = sha
    return refs


def batched_first_parent_log(shas: list[str], numstat: bool) -> dict[str, list]:
    """One `git log --no-walk` call over all shas.

    Every submission commit is a non-merge whose first parent is the organizer
    main of its day, so the per-commit file list is exactly the submitted diff.
    Returns sha -> list of paths (numstat=False) or (adds, dels, path) tuples.
    """
    mode = "--numstat" if numstat else "--name-only"
    out = git("log", "--no-walk=unsorted", "--format=@%H", mode, *shas)
    result: dict[str, list] = {}
    cur: list | None = None
    for line in out.splitlines():
        if line.startswith("@"):
            cur = result.setdefault(line[1:], [])
            continue
        if not line.strip() or cur is None:
            continue
        if numstat:
            parts = line.split("\t")
            if len(parts) == 3:
                a, d, p = parts
                cur.append(
                    (int(a) if a.isdigit() else 0, int(d) if d.isdigit() else 0, p)
                )
        else:
            cur.append(line.strip())
    return result


def expand_editable_paths() -> tuple[dict[str, list[str]], list[str]]:
    """editablePaths from benchmark.json, expanded to files at upstream/main."""
    bench = json.loads((REPO / "benchmark.json").read_text())
    declared = list(bench["editablePaths"])
    expanded: dict[str, list[str]] = {}
    for path in declared:
        listing = git("ls-tree", "-r", "--name-only", "upstream/main", "--", path)
        files = listing.splitlines()
        expanded[path] = files if files else []
    return expanded, declared


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return pearson(ranks(xs), ranks(ys))


def ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0


def main() -> int:
    facts = json.loads(FACTS.read_text())
    tele = facts["telemetry"]
    prompts = sorted(tele)
    first = prompts[0]
    rows = {r["submission"]: r for r in tele[first]}  # one row per submission

    refs = submission_refs()
    scored_uuids = sorted(set(rows) & set(refs))
    unscored_uuids = sorted(set(refs) - set(rows))
    print("# Corpus surface map")
    print(f"- upstream submission refs: {len(refs)}")
    print(f"- telemetry rows (content-unique, scored): {len(rows)}")
    print(f"- joinable (ref + telemetry score): {len(scored_uuids)}")
    print(f"- refs with no telemetry score: {len(unscored_uuids)}")
    print(
        "- note: no-score refs mix true failures with content-duplicate "
        "resubmissions; board summary says 476/717 rows scored, 408 unique."
    )
    print()

    # Two batched git calls cover every ref.
    all_files = batched_first_parent_log([refs[u] for u in refs], numstat=False)
    numstat = batched_first_parent_log([refs[u] for u in scored_uuids], numstat=True)

    def files_of(uuid: str) -> list[str]:
        return all_files.get(refs[uuid], [])

    def scored_files_of(uuid: str) -> list[str]:
        return [f for f in files_of(uuid) if f.startswith(SCORED_PREFIXES)]

    # ---------------- B: unexplored editable surface ----------------
    # One submission (a whole-surface dump) can mask the real map, so detect
    # bulk refs that touch >= BULK_THRESHOLD editable files and report the
    # unexplored surface both with and without them.
    BULK_THRESHOLD = 50
    expanded, declared = expand_editable_paths()
    editable_files = sorted({f for v in expanded.values() for f in v})
    editable_set = set(editable_files)
    bulk_refs = [
        u
        for u in refs
        if sum(1 for f in files_of(u) if f in editable_set) >= BULK_THRESHOLD
    ]
    print(f"## B. Unexplored editable surface")
    print(f"bulk refs touching >= {BULK_THRESHOLD} editable files "
          f"(excluded below): {len(bulk_refs)}")
    for u in bulk_refs:
        n = sum(1 for f in files_of(u) if f in editable_set)
        sc = rows[u]["score"] if u in rows else None
        sol = rows[u]["solver"] if u in rows else "(no telemetry score)"
        print(f"  - {u} touched {n} editable files, {sol}"
              + (f", score {sc:.4f}" if sc is not None else ""))
    touched_ever: set[str] = set()
    touch_count: dict[str, int] = {}
    for u in refs:
        if u in bulk_refs:
            continue
        for f in files_of(u):
            touched_ever.add(f)
            if f in editable_set:
                touch_count[f] = touch_count.get(f, 0) + 1
    unexplored_files = [f for f in editable_files if f not in touched_ever]
    print(f"\n### Never touched by any non-bulk submission "
          f"({len(unexplored_files)} of {len(editable_files)} editable files)")
    for f in unexplored_files:
        print(f"- {f}")
    literal_unexplored = [
        f
        for f in editable_files
        if all(f not in files_of(u) for u in refs)
    ]
    print(f"\nliteral unexplored count including bulk refs: "
          f"{len(literal_unexplored)}")
    for f in literal_unexplored:
        print(f"- {f}")
    print("\n### Barely explored (touched by only 1-2 non-bulk refs)")
    for f in editable_files:
        c = touch_count.get(f, 0)
        if 1 <= c <= 2:
            print(f"- {c}x {f}")
    print("\n### Declared-path coverage excluding bulk refs "
          "(files touched / files total)")
    for decl in declared:
        files = expanded[decl]
        if not files:
            touched_under = sorted(
                t for t in touched_ever if t == decl or t.startswith(decl + "/")
            )
            print(f"- {decl}: absent at upstream/main; "
                  f"{len(touched_under)} touched paths under it")
            continue
        t = sum(1 for f in files if f in touched_ever)
        print(f"- {decl}: {t}/{len(files)}")
    print()

    # ---------------- A: scored-surface files, score stats ----------------
    print("## A. Scored-surface files by best score when touched")
    per_file: dict[str, list[float]] = {}
    for u in scored_uuids:
        s = rows[u]["score"]
        for f in scored_files_of(u):
            per_file.setdefault(f, []).append(s)
    all_scores = [rows[u]["score"] for u in scored_uuids]
    best_overall = max(all_scores)
    stats_rows = []
    for f, scores in per_file.items():
        untouched = [
            rows[u]["score"] for u in scored_uuids if f not in set(scored_files_of(u))
        ]
        stats_rows.append(
            (
                f,
                len(scores),
                max(scores),
                statistics.median(scores),
                max(untouched) if untouched else float("nan"),
            )
        )
    stats_rows.sort(key=lambda r: -r[2])
    print(f"{'touched':>7} {'best':>8} {'median':>8} {'best-untouched':>14}  file")
    for f, n, best, med, bu in stats_rows:
        print(f"{n:7d} {best:8.4f} {med:8.4f} {bu:14.4f}  {f}")
    print(f"\nbest overall score in joined set: {best_overall:.6f}")
    print()

    # ---------------- C: diff size vs score ----------------
    print("## C. Does scored-surface diff size predict score?")
    churns, counts, scores = [], [], []
    for u in scored_uuids:
        entries = [
            (a, d, p)
            for (a, d, p) in numstat.get(refs[u], [])
            if p.startswith(SCORED_PREFIXES)
        ]
        churn = sum(a + d for a, d, _ in entries)
        churns.append(float(churn))
        counts.append(float(len(entries)))
        scores.append(rows[u]["score"])
    lchurns = [math.log1p(c) for c in churns]
    print(f"n = {len(scores)} joined submissions")
    print(f"- Pearson(score, churn lines)      = {pearson(churns, scores):+.3f}")
    print(f"- Pearson(score, log1p churn)      = {pearson(lchurns, scores):+.3f}")
    print(f"- Spearman(score, churn lines)     = {spearman(churns, scores):+.3f}")
    print(f"- Pearson(score, file count)       = {pearson(counts, scores):+.3f}")
    print(f"- Spearman(score, file count)      = {spearman(counts, scores):+.3f}")
    print(f"- OLS slope score per 100 churn    = {100 * ols_slope(churns, scores):+.4f}")
    small = [
        (rows[u]["score"], churns[i], rows[u]["solver"], u)
        for i, u in enumerate(scored_uuids)
        if churns[i] <= 20
    ]
    small.sort(reverse=True)
    print(f"- submissions with churn <= 20 lines: {len(small)}")
    if small:
        s, c, sol, u = small[0]
        print(f"- best score at churn <= 20: {s:.6f} ({sol}, churn {c:.0f}, {u[:8]})")
        for s, c, sol, u in small[1:5]:
            print(f"    next: {s:.6f} ({sol}, churn {c:.0f}, {u[:8]})")
    for cap in (50, 100, 200, 400, 10**9):
        subset = [scores[i] for i in range(len(scores)) if churns[i] <= cap]
        if subset:
            label = f"<= {cap}" if cap < 10**9 else "all"
            print(f"- best score at churn {label:>7}: {max(subset):.6f} "
                  f"(n={len(subset)})")
    print()

    # ---------------- D: serial winners ----------------
    print("## D. Top-five solvers by best score")
    by_solver: dict[str, list[str]] = {}
    for u in scored_uuids:
        by_solver.setdefault(rows[u]["solver"], []).append(u)
    ranked_solvers = sorted(
        by_solver, key=lambda s: -max(rows[u]["score"] for u in by_solver[s])
    )[:5]
    for sol in ranked_solvers:
        subs = sorted(by_solver[sol], key=lambda u: rows[u]["created"])
        scores_t = [rows[u]["score"] for u in subs]
        print(f"\n### {sol}: {len(subs)} scored submissions, "
              f"best {max(scores_t):.4f}")
        traj = [
            f"{rows[u]['created'][5:16]}={rows[u]['score']:.3f}" for u in subs
        ]
        # Print trajectory in bounded chunks.
        for i in range(0, len(traj), 6):
            print("  " + "  ".join(traj[i : i + 6]))
        fcount: dict[str, int] = {}
        for u in subs:
            for f in scored_files_of(u):
                fcount[f] = fcount.get(f, 0) + 1
        top = sorted(fcount.items(), key=lambda kv: -kv[1])[:8]
        print("  most-touched files:")
        for f, n in top:
            print(f"    {n:3d}x {f}")
    print()

    # ---------------- E: what the failures touched ----------------
    print("## E. Files most associated with refs that produced no score")
    print(f"(bulk refs excluded: {len(bulk_refs)})")
    fail_count: dict[str, int] = {}
    ok_count: dict[str, int] = {}
    for u in unscored_uuids:
        if u in bulk_refs:
            continue
        for f in files_of(u):
            if f.startswith(SCORED_PREFIXES):
                fail_count[f] = fail_count.get(f, 0) + 1
    for u in scored_uuids:
        for f in scored_files_of(u):
            ok_count[f] = ok_count.get(f, 0) + 1
    landmines = []
    for f, nf in fail_count.items():
        nk = ok_count.get(f, 0)
        landmines.append((f, nf, nk, nf / (nf + nk)))
    landmines.sort(key=lambda r: (-r[3], -r[1]))
    print(f"{'noscore':>7} {'scored':>7} {'noscore%':>8}  file  (min 3 no-score refs)")
    for f, nf, nk, rate in landmines:
        if nf >= 3:
            print(f"{nf:7d} {nk:7d} {100 * rate:7.1f}%  {f}")
    base_rate = len(unscored_uuids) / len(refs)
    print(f"\nbaseline no-score rate across all refs: {100 * base_rate:.1f}%")
    print()

    # ---------------- F: per-prompt winners ----------------
    print("## F. Per-prompt raw_ratio winners")
    winners = {}
    for p in prompts:
        prow = max(tele[p], key=lambda r: r["raw_ratio"])
        winners[p] = prow
        print(
            f"- {p:9s} best raw_ratio {prow['raw_ratio']:.4f} "
            f"score {prow['score']:.4f} solver {prow['solver']:20s} "
            f"sub {prow['submission'][:8]} commit {prow['commit'][:8]}"
        )
    uniq = {w["submission"] for w in winners.values()}
    print(f"\ndistinct winning submissions across 8 prompts: {len(uniq)}")
    by_sub: dict[str, list[str]] = {}
    for p, w in winners.items():
        by_sub.setdefault(w["submission"], []).append(p)
    for s, ps in by_sub.items():
        w = winners[ps[0]]
        print(f"  {s[:8]} ({w['solver']}) wins: {', '.join(ps)}")
    # For the overall best submission, show its per-prompt profile.
    best_u = max(scored_uuids, key=lambda u: rows[u]["score"])
    print(f"\nper-prompt profile of best overall submission "
          f"{best_u[:8]} ({rows[best_u]['solver']}, "
          f"score {rows[best_u]['score']:.4f}):")
    prof = []
    for p in prompts:
        r = next(r for r in tele[p] if r["submission"] == best_u)
        rank = 1 + sum(1 for x in tele[p] if x["raw_ratio"] > r["raw_ratio"])
        prof.append((p, r["raw_ratio"], rank))
    for p, rr, rank in sorted(prof, key=lambda x: -x[1]):
        print(f"  {p:9s} raw_ratio {rr:.4f} (rank {rank} of {len(tele[p])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
