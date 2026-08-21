#!/usr/bin/env python3
"""Per-prompt board instrument for the Qwen 3.8 27B native-MTP track.

The Yukon list endpoint returns ``officialMetrics.per_prompt`` for every scored
submission: eight rows per run carrying candidate and serial seconds per token,
the raw ratio, the effective mean draft length, the non-drafting round count and
the declared head provenance digest.

Candidate seconds per token are directly comparable between seats. The serial
numerator is not a constant: it is drawn fresh on every run with a per-prompt cv
of 0.21-0.24 %, it comes from the runner's own prebuilt baseline workspace, and
candidate-editable code cannot change it. That draw enters the published score
at full weight, so a run can be uniformly faster and still score lower. Use
``serialfree`` to remove it before attributing a score change to a mechanism.

Usage:

    YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
    python3 research/board_per_prompt.py floors
    python3 research/board_per_prompt.py order
    python3 research/board_per_prompt.py heads
    python3 research/board_per_prompt.py tree <uuid-prefix> [...]
    python3 research/board_per_prompt.py serialfree [<uuid-prefix> ...]

``fetch`` writes the raw payload to ``/tmp/yukon-board/full.json``; it is about
11 MB and is deliberately not committed.
"""

import collections
import json
import os
import statistics as st
import sys
import urllib.request

BASE = "https://api.yukon.org/api"
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
CACHE = "/tmp/yukon-board/full.json"

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
WIDE = ["beagle", "medicine", "essays", "republic", "botany"]
PINNED_HEAD = "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71"


def fetch():
    token = os.environ["YUKON_API_TOKEN"]
    req = urllib.request.Request(
        "%s/benchmarks/%s/submissions?all=true" % (BASE, BENCHMARK_ID),
        headers={"Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode())
    rows = payload["submissions"]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as handle:
        json.dump(rows, handle)
    print("wrote %s with %d rows" % (CACHE, len(rows)))


def load():
    rows = json.load(open(CACHE))
    if isinstance(rows, dict):
        # Tolerate a cache holding the whole endpoint payload rather than the
        # submission list that ``fetch`` writes.
        for key in ("submissions", "rows", "data", "items"):
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
        else:
            raise SystemExit("%s holds no submission list; rerun fetch" % CACHE)
    return [r for r in rows
            if isinstance(r, dict)
            and (r.get("officialMetrics") or {}).get("per_prompt")
            and r.get("officialScore") is not None]


def vec(row):
    return {PROMPT_NAMES.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8]): e
            for e in row["officialMetrics"]["per_prompt"]}


def published_median(raws):
    ordered = sorted(raws)
    return (ordered[3] + ordered[4]) / 2


def serial_means(scored):
    out = {}
    for name in PROMPT_NAMES.values():
        values = [e["serial_seconds_per_token_mean"] for r in scored
                  for e in r["officialMetrics"]["per_prompt"]
                  if PROMPT_NAMES.get(e["prompt_sha256"][:8]) == name]
        out[name] = st.mean(values)
    return out


def cmd_floors(scored):
    serial = serial_means(scored)
    print("prompt     n    floor s/tok  draft  by                 score      median s/tok  serial cv")
    floors = {}
    for name in ["plutarch", "drama", "travel", "beagle",
                 "medicine", "essays", "republic", "botany"]:
        entries = sorted(
            ((e["mtp_seconds_per_token_mean"], e, r) for r in scored
             for e in r["officialMetrics"]["per_prompt"]
             if PROMPT_NAMES.get(e["prompt_sha256"][:8]) == name),
            key=lambda t: t[0])
        low, entry, row = entries[0]
        floors[name] = low
        serials = [t[1]["serial_seconds_per_token_mean"] for t in entries]
        print("%-9s %4d  %.8f  %5.2f  %-18s %.6f  %.8f  %.4f%%" % (
            name, len(entries), low, entry.get("effective_mean_draft_len") or 0,
            str(row.get("solverUsername"))[:18], row["officialScore"],
            entries[len(entries) // 2][0],
            100 * st.pstdev(serials) / st.mean(serials)))
    ceiling = sorted(serial[w] / floors[w] for w in WIDE)
    print("\nobserved-floor ceiling: 4th %.6f  5th %.6f  median %.6f"
          % (ceiling[0], ceiling[1], (ceiling[0] + ceiling[1]) / 2))


def cmd_order(scored):
    strong = [r for r in scored if r["officialScore"] > 3.15]
    for label, subset in (("all scored", scored), ("score > 3.15", strong)):
        c4, c5 = collections.Counter(), collections.Counter()
        for row in subset:
            ordered = sorted(row["officialMetrics"]["per_prompt"],
                             key=lambda e: e["raw_ratio_of_means"])
            c4[PROMPT_NAMES.get(ordered[3]["prompt_sha256"][:8])] += 1
            c5[PROMPT_NAMES.get(ordered[4]["prompt_sha256"][:8])] += 1
        print("\n=== %s (n=%d) ===" % (label, len(subset)))
        for slot, counter in (("4th", c4), ("5th", c5)):
            parts = ["%s %.1f%%" % (k, 100 * v / len(subset))
                     for k, v in counter.most_common(4)]
            print("  %s: %s" % (slot, "  ".join(parts)))


def cmd_heads(scored):
    best = collections.defaultdict(lambda: (9.0, None))
    counts = collections.Counter()
    for row in scored:
        entry = vec(row).get("beagle")
        if not entry:
            continue
        digest = entry.get("head_provenance_sha256") or "NONE"
        counts[digest] += 1
        if entry["mtp_seconds_per_token_mean"] < best[digest][0]:
            best[digest] = (entry["mtp_seconds_per_token_mean"], row)
    print("best beagle seconds per token, grouped by declared head")
    for digest, (value, row) in sorted(best.items(), key=lambda kv: kv[1][0])[:16]:
        label = "PINNED" if digest == PINNED_HEAD else digest[:12]
        print("  %.8f  %-14s n=%3d  %-16s %s  score=%.6f" % (
            value, label, counts[digest], str(row.get("solverUsername"))[:16],
            row["id"][:8], row["officialScore"]))


def cmd_tree(scored, prefixes):
    serial = serial_means(scored)
    floors = {w: min(e["mtp_seconds_per_token_mean"] for r in scored
                     for e in r["officialMetrics"]["per_prompt"]
                     if PROMPT_NAMES.get(e["prompt_sha256"][:8]) == w)
              for w in PROMPT_NAMES.values() if w in serial}
    for prefix in prefixes:
        match = [r for r in scored if r["id"].startswith(prefix)]
        if not match:
            print("no such submission:", prefix)
            continue
        row = match[0]
        values = vec(row)
        print("\n=== %s  %s  official %.8f  %s ===" % (
            row["id"][:8], row.get("solverUsername"), row["officialScore"],
            row.get("promotionStatus")))
        for name in sorted(values, key=lambda n: values[n]["raw_ratio_of_means"]):
            entry = values[name]
            gap = 100 * (entry["mtp_seconds_per_token_mean"] / floors[name] - 1)
            print("  %-9s raw=%.6f  mtp=%.8f  (+%.3f%% over floor)  draft=%.3f  nondraft=%s"
                  % (name, entry["raw_ratio_of_means"],
                     entry["mtp_seconds_per_token_mean"], gap,
                     entry.get("effective_mean_draft_len") or 0,
                     entry.get("non_drafting_round_count")))


def serial_free_score(row, means):
    """Published statistic recomputed with each prompt's board-mean serial draw.

    The runner's prebuilt baseline workspace produces the serial numerator, so
    candidate-editable code cannot change it, yet it enters every raw ratio and
    is redrawn on every run. Substituting the board mean per prompt leaves only
    candidate-side effects in the score.
    """
    entries = vec(row)
    return published_median([means[name] / entry["mtp_seconds_per_token_mean"]
                             for name, entry in entries.items()])


def cmd_serialfree(scored, argv):
    means = serial_means(scored)

    worst = max(abs(published_median([e["raw_ratio_of_means"]
                                      for e in vec(r).values()])
                    / r["officialScore"] - 1) for r in scored)
    print("median-of-8 reproduces every published score to %.2e" % worst)

    print("\nboard-mean serial per prompt")
    for name in sorted(means):
        draws = [vec(r)[name]["serial_seconds_per_token_mean"] for r in scored]
        print("  %-9s %.9f   cv %.4f %%"
              % (name, means[name], 100 * st.pstdev(draws) / means[name]))

    ranked = sorted(((serial_free_score(r, means), r) for r in scored),
                    key=lambda pair: -pair[0])
    wanted = [p.lower() for p in argv]
    print("\n%-9s %14s %13s %10s  %-10s %s"
          % ("id", "serial-free", "published", "delta", "status", "created"))
    for index, (free, row) in enumerate(ranked, 1):
        ident = row["id"]
        if index > 20 and not any(ident.startswith(p) for p in wanted):
            continue
        print("%-9s %14.8f %13.8f %+10.5f  %-10s %s   rank %d/%d"
              % (ident[:8], free, row["officialScore"],
                 free - row["officialScore"], row.get("status"),
                 (row.get("createdAt") or "")[:19], index, len(ranked)))


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "floors"
    if command == "fetch":
        fetch()
        return
    scored = load()
    print("scored submissions with per-prompt rows: %d" % len(scored))
    if command == "floors":
        cmd_floors(scored)
    elif command == "order":
        cmd_order(scored)
    elif command == "heads":
        cmd_heads(scored)
    elif command == "tree":
        cmd_tree(scored, sys.argv[2:])
    elif command == "serialfree":
        cmd_serialfree(scored, sys.argv[2:])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
