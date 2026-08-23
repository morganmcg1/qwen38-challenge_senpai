#!/usr/bin/env python3
"""Count the distinct decode schedules on the public board.

Draft length is bit reproducible on the ranked runner. Two receipts of the same
arm six hours apart returned all eight draft lengths identical to the last bit.
The eight-vector draft-length tuple is therefore an exact fingerprint of a
decode schedule, and grouping receipts by that tuple says how many distinct
schedules the board is actually running.

That number decides where our remaining candidate-leg gap can come from. If
every leading receipt shares one schedule, then no part of the gap is a
scheduling deficit and all of it is kernel and runtime work.
"""
import argparse
import json
import os
import urllib.request

BENCH = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
API = f"https://api.yukon.org/api/benchmarks/{BENCH}/submissions?all=true"

NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
ORDER = ["beagle", "botany", "drama", "essays", "medicine", "plutarch",
         "republic", "travel"]
OURS = {"623e77af", "572b2cc4", "e003a86d", "0c6191b7", "d3c491b5", "1db9d63e"}


def fetch():
    request = urllib.request.Request(
        API, headers={"Authorization": "Bearer " + os.environ["YUKON_API_TOKEN"]})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["submissions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-22T00:00:00Z")
    args = ap.parse_args()

    groups = {}
    for row in fetch():
        metrics = row.get("officialMetrics") or {}
        prompts = metrics.get("per_prompt") or []
        if len(prompts) != len(NAMES) or row["createdAt"] < args.since:
            continue
        by_name = {NAMES[p["prompt_sha256"][:8]]: p for p in prompts}
        if not all(p.get("serial_seconds_per_token_mean") for p in prompts):
            continue
        key = tuple(by_name[n]["effective_mean_draft_len"] for n in ORDER)
        nondraft = tuple(by_name[n]["non_drafting_round_count"] for n in ORDER)
        groups.setdefault((key, nondraft), []).append(row)

    print(f"receipts since {args.since} with complete metrics: "
          f"{sum(len(v) for v in groups.values())}")
    print(f"distinct decode schedules: {len(groups)}")
    print()

    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for index, ((key, nondraft), rows) in enumerate(ranked):
        scores = [float(r["officialScore"]) for r in rows
                  if r.get("officialScore")]
        mine = sorted(r["id"][:8] for r in rows if r["id"][:8] in OURS)
        print(f"-- schedule {index}  {len(rows)} receipts"
              + (f"  best published {max(scores):.8f}" if scores else "")
              + (f"  OURS: {' '.join(mine)}" if mine else ""))
        print("     " + "  ".join(f"{n}={v:.4f}" for n, v in zip(ORDER, key)))
        if any(nondraft):
            print("     non-drafting "
                  + "  ".join(f"{n}={v}" for n, v in zip(ORDER, nondraft) if v))
        ids = sorted(r["id"][:8] for r in rows)
        print(f"     ids {' '.join(ids)}")
        print()


main()
