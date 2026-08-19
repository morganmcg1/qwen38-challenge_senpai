#!/usr/bin/env python3
"""Pull and decompose the ranked benchmark's per-prompt telemetry.

Why this exists
---------------
`yukon submissions` truncates the `metrics` column at 80 characters, ignores
`COLUMNS`, and has no `--json` flag, so the richest signal on this competition
sat unread for most of the campaign. The REST API returns everything, including
each competitor's public `note`.

    export YUKON_API_TOKEN=...        # already present in the campaign env
    python3 research/ranked_telemetry.py --refresh          # fetch + cache
    python3 research/ranked_telemetry.py --top 14            # leaderboard
    python3 research/ranked_telemetry.py --profile 0cd0a6b4  # one row, 8 prompts
    python3 research/ranked_telemetry.py --price beagle      # R vs n vs hbar

What the numbers mean (see campaign-ledger items 87-93)
-------------------------------------------------------
* `raw_ratio_of_means` is the per-prompt score contribution. The submission
  score is the mean of the 4th and 5th *ranked* per-prompt ratios
  (`median_rule = even_n_mean_of_two_central_order_statistics`, n=8), so
  d(score)/d(prompt) is 0.5 for exactly two prompts and 0 for the other six.
* `serial_seconds_per_token_mean` is the PINNED baseline leg. Across 402 scored
  rows its standard deviation is 0.106 %; it is not a lever, it is a thermal
  control.
* `effective_mean_draft_len` (n) is the realised accepted-row count. The verify
  width is `M = n + 1`. Weight passes are `ceil(M/5)` (`quantized.h:1154`), so
  per-row weight traffic is `ceil(M/5)/M` and M=6 is the worst width.
* `hbar` below inverts `R = (1 + alpha*n) / (1 + hbar*n)` for an assumed accept
  rate alpha. It is a RE-PARAMETERISATION of the observed (R, n), not an
  independent fit: with one equation and two unknowns you cannot get both. Use
  it to compare rows at equal n, and do not read it as a measured price.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
API = "https://api.yukon.org/api/benchmarks/%s/submissions" % BENCHMARK_ID
CACHE = pathlib.Path(".mlxfast-private/ranked-telemetry.json")
CONTRACT = pathlib.Path("fixtures/qwen3_8_27b_mtp_track.json")

# Ascending by ratio on the reference frontier row; the central pair is ranks 4-5.
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]
CENTRAL = ("beagle", "medicine")


def prompt_map() -> dict[str, str]:
    """sha256 -> short prompt name, from the track contract."""
    out: dict[str, str] = {}
    if not CONTRACT.exists():
        return out
    contract = json.loads(CONTRACT.read_text())

    def walk(node):
        if isinstance(node, dict):
            if "sha256" in node:
                path = node.get("r2_path") or node.get("path") or ""
                hit = re.search(r"pool-([a-z_]+)\.json", path)
                if hit:
                    out[node["sha256"]] = hit.group(1)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(contract)
    return out


def fetch(refresh: bool) -> list[dict]:
    if not refresh and CACHE.exists():
        return json.loads(CACHE.read_text())["submissions"]
    token = os.environ.get("YUKON_API_TOKEN")
    if not token:
        sys.exit("YUKON_API_TOKEN is not set and no cache at %s" % CACHE)
    raw = subprocess.run(
        ["curl", "-sS", "-H", "Authorization: Bearer %s" % token, API],
        capture_output=True, check=True).stdout
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(raw)
    return json.loads(raw)["submissions"]


def rows(sub: dict, pmap: dict[str, str]) -> dict[str, dict]:
    metrics = sub.get("officialMetrics") or {}
    out = {}
    for entry in metrics.get("per_prompt") or []:
        name = pmap.get(entry.get("prompt_sha256"))
        if name:
            out[name] = entry
    return out


def hbar(ratio: float, n: float, alpha: float) -> float:
    if n <= 0 or ratio <= 0:
        return float("nan")
    return ((1 + alpha * n) / ratio - 1) / n


def cmd_top(subs, pmap, args):
    acc = [s for s in subs if s["status"] == "accepted" and s.get("officialScore")]
    acc.sort(key=lambda s: -s["officialScore"])
    print("%-16s %-9s %10s | %-22s | %-22s | head" %
          ("solver", "id", "score", "beagle R / n / hbar", "medicine R / n / hbar"))
    for sub in acc[:args.top]:
        r = rows(sub, pmap)
        cells = []
        for want in CENTRAL:
            p = r.get(want)
            if not p:
                cells.append("%-22s" % "?")
                continue
            ratio, n = p["raw_ratio_of_means"], p["effective_mean_draft_len"]
            cells.append("%6.4f / %5.3f / %.4f" % (ratio, n, hbar(ratio, n, args.alpha)))
        head = "?"
        if r:
            head = next(iter(r.values())).get("head_provenance_sha256", "?")[:8]
        print("%-16s %-9s %10.5f | %s | %s | %s" %
              (sub["solverUsername"], sub["id"][:8], sub["officialScore"], cells[0], cells[1], head))


def cmd_profile(subs, pmap, args):
    hits = [s for s in subs if s["id"].startswith(args.profile)]
    if not hits:
        sys.exit("no submission id starting with %r" % args.profile)
    for sub in hits:
        r = rows(sub, pmap)
        print("=== %s / %s  status=%s score=%s" %
              (sub["solverUsername"], sub["id"][:8], sub["status"], sub.get("officialScore")))
        if not r:
            print("   (no per-prompt metrics)")
            continue
        ranked = sorted((p["raw_ratio_of_means"], name, p) for name, p in r.items())
        for rank, (ratio, name, p) in enumerate(ranked, 1):
            mark = "  <== CENTRAL (d score/d ratio = 0.5)" if rank in (4, 5) else ""
            n = p["effective_mean_draft_len"]
            print("  %d %-10s R=%7.4f  n=%6.3f  meanM=%5.2f  nondraft=%4d%s" %
                  (rank, name, ratio, n, n + 1,
                   p.get("non_drafting_round_count", -1), mark))
        print("  meanM is a MEAN: a row at meanM 5.53 straddles the 1->2 weight-pass")
        print("  boundary at M=6, so per-round pass counts are mixed. Do not read")
        print("  ceil(meanM/5) as the pass count of any individual round.")
        print("  implied score = %.5f   (reported %s)" %
              ((ranked[3][0] + ranked[4][0]) / 2, sub.get("officialScore")))


def cmd_price(subs, pmap, args):
    want = args.price
    pool = []
    for sub in subs:
        if not sub.get("officialScore"):
            continue
        r = rows(sub, pmap)
        p = r.get(want)
        if not p:
            continue
        if args.head and not p.get("head_provenance_sha256", "").startswith(args.head):
            continue
        pool.append((p["effective_mean_draft_len"], p["raw_ratio_of_means"], sub))
    pool.sort()
    print("=== %s, head prefix %r, %d rows: does a deeper n buy a better ratio? ===" %
          (want, args.head or "*any*", len(pool)))
    print("%8s %8s %8s %10s  %-16s %s" % ("n", "R", "hbar", "score", "solver", "id"))
    groups = collections.Counter(round(n, 6) for n, _, _ in pool)
    for n, ratio, sub in pool:
        tag = " <-- default cluster (%d rows)" % groups[round(n, 6)] if groups[round(n, 6)] > 5 else ""
        print("%8.4f %8.4f %8.4f %10.5f  %-16s %s%s" %
              (n, ratio, hbar(ratio, n, args.alpha), sub["officialScore"],
               sub["solverUsername"], sub["id"][:8], tag))


def cmd_spread(subs, pmap, args):
    acc = [s for s in subs if s["status"] == "accepted" and s.get("officialScore")]
    acc.sort(key=lambda s: -s["officialScore"])
    top = acc[:args.top]
    print("=== spread across the top %d, per prompt ===" % len(top))
    print("%-10s %8s %8s %7s | %8s %8s %7s" %
          ("prompt", "R min", "R max", "R sd%", "n min", "n max", "n sd%"))
    for name in ORDER:
        ratios, ns = [], []
        for sub in top:
            p = rows(sub, pmap).get(name)
            if p:
                ratios.append(p["raw_ratio_of_means"])
                ns.append(p["effective_mean_draft_len"])
        if len(ratios) < 3:
            continue
        rsd = 100 * statistics.stdev(ratios) / statistics.mean(ratios)
        nsd = 100 * statistics.stdev(ns) / statistics.mean(ns) if statistics.mean(ns) else 0.0
        print("%-10s %8.4f %8.4f %6.3f%% | %8.3f %8.3f %6.3f%%" %
              (name, min(ratios), max(ratios), rsd, min(ns), max(ns), nsd))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-fetch from the API")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=0.99, help="assumed accept rate for hbar")
    ap.add_argument("--profile", metavar="ID_PREFIX", help="print one row's 8 prompts, ranked")
    ap.add_argument("--price", metavar="PROMPT", help="R vs n vs hbar for one prompt")
    ap.add_argument("--head", metavar="SHA_PREFIX", default="559b24eb",
                    help="restrict --price to one MTP head ('' for all)")
    ap.add_argument("--spread", action="store_true", help="per-prompt noise across the top")
    args = ap.parse_args()

    subs = fetch(args.refresh)
    pmap = prompt_map()
    if not pmap:
        print("warning: no prompt map (run from the repo root so %s resolves)" % CONTRACT,
              file=sys.stderr)
    counts = collections.Counter(s["status"] for s in subs)
    print("%d submissions: %s" % (len(subs), dict(counts)))
    print()

    if args.profile:
        cmd_profile(subs, pmap, args)
    elif args.price:
        cmd_price(subs, pmap, args)
    elif args.spread:
        cmd_spread(subs, pmap, args)
    else:
        cmd_top(subs, pmap, args)


if __name__ == "__main__":
    main()
