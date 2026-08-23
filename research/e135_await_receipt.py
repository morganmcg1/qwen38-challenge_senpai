#!/usr/bin/env python3
"""Block until one Yukon submission leaves `validating`, then print it.

A passive watcher. It writes nothing in the workspace and holds no GPU, so it
can run while the conversation is suspended. Exit 0 when the row reaches a
terminal state, 2 on the deadline, 3 if the row never appears.

  research/e135_await_receipt.py 0cf1637e --interval 120 --deadline 14400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BOARD_URL = (
    "https://api.yukon.org/api/benchmarks/"
    "5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?all=true"
)
PENDING = {"validating", "pending", "queued", "running"}


def fetch(token: str) -> list[dict]:
    req = urllib.request.Request(
        BOARD_URL, headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=120) as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        return raw
    return raw.get("submissions", raw.get("data", []))


def say(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{stamp} {message}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="submission id prefix")
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--deadline", type=float, default=14400.0)
    args = ap.parse_args()

    token = os.environ["YUKON_API_TOKEN"]
    started = time.monotonic()
    last_ahead = None

    while True:
        elapsed = time.monotonic() - started
        if elapsed > args.deadline:
            say(f"DEADLINE after {elapsed:.0f}s; row still pending")
            return 2

        try:
            rows = fetch(token)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            say(f"board read failed ({exc}); retrying")
            time.sleep(args.interval)
            continue

        hits = [r for r in rows if r["id"].startswith(args.submission)]
        if not hits:
            say(f"{args.submission}: not on the board")
            return 3
        row = hits[0]
        status = row.get("status")

        if status not in PENDING:
            say(f"TERMINAL status={status} score={row.get('officialScore')} "
                f"promotion={row.get('promotionStatus')}")
            print(json.dumps(row, indent=2, sort_keys=True), flush=True)
            return 0

        ours = row.get("createdAt") or ""
        ahead = sum(
            1
            for r in rows
            if r.get("status") in PENDING and (r.get("createdAt") or "") < ours
        )
        if ahead != last_ahead:
            say(f"status={status} queue_ahead={ahead} elapsed={elapsed:.0f}s")
            last_ahead = ahead

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
