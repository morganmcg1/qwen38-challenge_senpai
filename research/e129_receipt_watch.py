#!/usr/bin/env python3
"""Wait for one Yukon submission to leave ``validating`` and save its receipt.

    YUKON_API_TOKEN=... python3 research/e129_receipt_watch.py <uuid-prefix> \
        [--out PATH] [--interval SECONDS] [--deadline SECONDS]

The process exits 0 as soon as the row reaches a terminal status, writing the
whole row to ``--out``. It exits 3 on the deadline and 4 if the row is absent,
so a supervised job wakes the owning conversation with a distinguishable state
instead of a polling loop inside the agent transcript.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE = "https://api.yukon.org/api"
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
PENDING = {"validating", "pending", "queued", "running"}


def board(token: str) -> list[dict]:
    request = urllib.request.Request(
        "%s/benchmarks/%s/submissions?all=true" % (BASE, BENCHMARK_ID),
        headers={"Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())["submissions"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/out/e129-receipt.json"))
    parser.add_argument("--interval", type=float, default=90.0)
    parser.add_argument("--deadline", type=float, default=10200.0)
    args = parser.parse_args()

    token = os.environ["YUKON_API_TOKEN"]
    started = time.time()
    consecutive_errors = 0

    while time.time() - started < args.deadline:
        try:
            rows = board(token)
            consecutive_errors = 0
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            consecutive_errors += 1
            print("poll error %d: %s" % (consecutive_errors, error), flush=True)
            if consecutive_errors >= 10:
                return 5
            time.sleep(args.interval)
            continue

        hits = [r for r in rows if str(r.get("id", "")).startswith(args.prefix)]
        if not hits:
            print("no row matches %r" % args.prefix, flush=True)
            return 4
        row = hits[0]
        status = str(row.get("status", ""))
        if status not in PENDING:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(row, indent=1))
            print("resolved after %.0fs: %s status=%s score=%s"
                  % (time.time() - started, row["id"], status, row.get("score")),
                  flush=True)
            print("wrote %s" % args.out, flush=True)
            return 0
        print("%.0fs elapsed, status=%s" % (time.time() - started, status), flush=True)
        time.sleep(args.interval)

    print("deadline reached with the row still pending", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
