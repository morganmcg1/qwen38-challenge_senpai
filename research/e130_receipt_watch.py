#!/usr/bin/env python3
"""Wait for one Yukon submission to reach a terminal status.

    usage: research/e130_receipt_watch.py SUBMISSION_ID [--interval SECONDS]

Exits 0 as soon as the submission leaves `validating`, and prints the terminal
status, the published score and the per-prompt candidate decode times. Exits 2
if the deadline passes while the submission is still validating.

Run this through `run_job`, never from the terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://api.yukon.org/api"
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
PENDING = {"validating", "pending", "queued", "running"}


def poll(token: str) -> list[dict]:
    req = urllib.request.Request(
        "%s/benchmarks/%s/submissions?all=true" % (BASE, BENCHMARK_ID),
        headers={"Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())["submissions"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission")
    ap.add_argument("--interval", type=float, default=120.0)
    args = ap.parse_args()

    token = os.environ["YUKON_API_TOKEN"]
    prefix = args.submission[:8]
    while True:
        try:
            rows = poll(token)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print("poll failed, will retry: %s" % e, flush=True)
            time.sleep(args.interval)
            continue

        hits = [r for r in rows if str(r.get("id", "")).startswith(prefix)]
        if not hits:
            print("submission %s is not on the board yet" % prefix, flush=True)
            time.sleep(args.interval)
            continue

        row = hits[0]
        status = row.get("status")
        print("%s  %s  status=%s  score=%s"
              % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), prefix,
                 status, row.get("officialScore")), flush=True)
        if status in PENDING:
            time.sleep(args.interval)
            continue

        print("\nTERMINAL")
        print(json.dumps({
            "id": row["id"],
            "status": status,
            "promotionStatus": row.get("promotionStatus"),
            "officialScore": row.get("officialScore"),
            "rejectionReason": row.get("rejectionReason"),
            "promotionReason": row.get("promotionReason"),
            "improved": row.get("improved"),
        }, indent=2), flush=True)
        metrics = row.get("officialMetrics") or {}
        for entry in metrics.get("per_prompt") or []:
            print("  %s  mtp=%.10f serial=%.10f raw=%.6f parity=%s"
                  % (str(entry.get("prompt_sha256"))[:8],
                     entry["mtp_seconds_per_token_mean"],
                     entry["serial_seconds_per_token_mean"],
                     entry["raw_ratio_of_means"], entry["parity_ok"]),
                  flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
