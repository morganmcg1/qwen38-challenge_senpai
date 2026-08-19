#!/usr/bin/env python3
"""Fetch the live board and report our exact rank plus the top cluster.

Read-only. No GPU. Writes a fresh row cache to /tmp/rows_live.json.

Usage: python3 research/rank_check.py [our_submission_id_prefix]
"""
import json
import os
import sys
import urllib.request

BENCH = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
URL = (
    "https://api.yukon.org/api/benchmarks/"
    + BENCH
    + "/submissions?limit=2000"
)
OURS = sys.argv[1] if len(sys.argv) > 1 else "ca9251b8"


def fetch():
    tok = os.environ.get("YUKON_API_TOKEN", "")
    req = urllib.request.Request(URL, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    d = fetch()
    rows = d if isinstance(d, list) else d.get("submissions", d.get("data", []))
    with open("/tmp/rows_live.json", "w") as f:
        json.dump(rows, f)

    status = {}
    for r in rows:
        s = r.get("status", "?")
        status[s] = status.get(s, 0) + 1

    scored = []
    for r in rows:
        sc = r.get("officialScore")
        if isinstance(sc, (int, float)):
            scored.append(r)
    scored.sort(key=lambda r: -r["officialScore"])

    print("total rows      :", len(rows))
    print("status          :", status)
    print("scored rows     :", len(scored))

    ourrow = None
    ourrank = None
    for i, r in enumerate(scored, 1):
        if str(r.get("id", "")).startswith(OURS):
            ourrow, ourrank = r, i
            break

    top = scored[0]["officialScore"]
    print()
    if ourrow is None:
        print("!! our row", OURS, "not found among scored rows")
    else:
        gap = (top / ourrow["officialScore"] - 1.0) * 100.0
        print("our id          :", ourrow.get("id"))
        print("our score       :", repr(ourrow["officialScore"]))
        print("our RANK        :", ourrank, "of", len(scored))
        print("board top       :", repr(top))
        print("gap to #1       : %.4f %%" % gap)

    print()
    print("rank  id        solver                score          status    head")
    lim = min(20, len(scored))
    for i in range(lim):
        r = scored[i]
        om = r.get("officialMetrics") or {}
        head = str(om.get("qwen_mtp_head_sha256", ""))[:8]
        mark = "  <== US" if str(r.get("id", "")).startswith(OURS) else ""
        print(
            "%4d  %-9s %-20s %-14.8f %-9s %s%s"
            % (
                i + 1,
                str(r.get("id", ""))[:8],
                str(r.get("solverName", r.get("solver", "")))[:20],
                r["officialScore"],
                str(r.get("status", ""))[:9],
                head,
                mark,
            )
        )


if __name__ == "__main__":
    main()
