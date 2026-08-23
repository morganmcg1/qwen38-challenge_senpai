#!/usr/bin/env python3
"""Compare per-prompt ranked evidence across two or more Yukon receipts.

The published median hides where a composition lost. This prints the
per-prompt raw ratio, effective mean draft length and absolute candidate
seconds per token side by side, keyed on ``prompt_sha256`` so rows are
matched by prompt rather than by position.
"""

import argparse
import json
import os
import sys
import urllib.request

BASE = "https://api.yukon.org/api"
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"


def fetch():
    token = os.environ["YUKON_API_TOKEN"]
    request = urllib.request.Request(
        "%s/benchmarks/%s/submissions?all=true" % (BASE, BENCHMARK_ID),
        headers={"Authorization": "Bearer %s" % token},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())["submissions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="submission id prefixes, oldest first")
    args = ap.parse_args()

    subs = fetch()
    picked = []
    for prefix in args.ids:
        hits = [s for s in subs if s["id"].startswith(prefix)]
        if len(hits) != 1:
            sys.exit("id %s matched %d rows" % (prefix, len(hits)))
        picked.append(hits[0])

    for s in picked:
        m = s.get("officialMetrics") or {}
        print(
            "%s  %-16s %-10s score=%s commit=%s"
            % (
                s["id"][:8],
                s["solverUsername"],
                s["status"],
                s.get("officialScore"),
                (m.get("commit") or "n/a")[:8],
            )
        )
    print()

    tables = []
    for s in picked:
        m = s.get("officialMetrics") or {}
        tables.append({p["prompt_sha256"]: p for p in m.get("per_prompt", [])})

    shas = sorted(tables[0], key=lambda k: tables[0][k]["raw_ratio_of_means"])
    head = "prompt    " + "".join(
        "%22s" % s["id"][:8] for s in picked
    )
    for field, label, fmt in (
        ("raw_ratio_of_means", "raw ratio", "%10.4f"),
        ("effective_mean_draft_len", "edl", "%10.4f"),
        ("mtp_seconds_per_token_mean", "mtp s/tok", "%10.7f"),
        ("serial_seconds_per_token_mean", "serial s/tok", "%10.7f"),
    ):
        print("== %s ==" % label)
        print(head)
        for sha in shas:
            row = "%-10s" % sha[:8]
            first = None
            for t in tables:
                p = t.get(sha)
                if p is None:
                    row += "%22s" % "-"
                    continue
                v = p[field]
                if first is None:
                    first = v
                    row += "%22s" % (fmt % v)
                else:
                    pct = 100.0 * (v - first) / first
                    row += "%22s" % ((fmt % v) + " %+7.2f%%" % pct)
            print(row)
        print()

    print("== published median ==")
    for s, t in zip(picked, tables):
        vals = sorted(p["raw_ratio_of_means"] for p in t.values())
        if len(vals) == 8:
            med = (vals[3] + vals[4]) / 2.0
            print(
                "%s median=%.9f  central pair=(%.4f, %.4f)"
                % (s["id"][:8], med, vals[3], vals[4])
            )


if __name__ == "__main__":
    main()
