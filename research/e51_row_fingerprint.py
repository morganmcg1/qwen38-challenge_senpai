#!/usr/bin/env python3
"""E51 end-to-end exactness sensor.

The pre-registered PRIMARY signal for the dose ladder is the SHA-256 of the
ordered `^mtp-row:` lines that Qwen36MTPBlockSession.traceRow writes. Each line
carries the position, the top-two token ids and their hexfloat values, so the
digest changes if any emitted row evidence changes anywhere in the window.

The secondary signals come from the score JSON the trusted parent writes.

usage:
    research/e51_row_fingerprint.py research/out/e51-r0-a [research/out/e51-r1 ...]
    research/e51_row_fingerprint.py --json out.json research/out/*
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROW = re.compile(r"^mtp-row: pos=(\d+) ids=(\d+),(\d+) v=(\S+)")

SECONDARY = (
    "all_tokens_matched",
    "parity_all_ok",
    "effective_mean_draft_len",
    "round_count",
    "accepted_draft_total",
    "rejected_draft_total",
    "accepted_draft_rate",
    "declared_rows_total",
    "serial_seconds_per_token_mean",
    "mtp_seconds_per_token_mean",
    "score",
)


def collect(path: pathlib.Path) -> dict:
    trace = path / "trace.txt"
    rows: list[str] = []
    if trace.exists():
        for line in trace.read_text(errors="replace").splitlines():
            if line.startswith("mtp-row:"):
                rows.append(line.strip())

    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest() if rows else None
    out = {
        "arm": path.name,
        "row_count": len(rows),
        "row_fingerprint": digest,
        "first_row": rows[0] if rows else None,
        "last_row": rows[-1] if rows else None,
    }

    score = path / "score.json"
    if score.exists():
        blob = json.loads(score.read_text())
        flat: dict = {}

        def walk(node, prefix=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}{key}.")
            else:
                flat[prefix[:-1]] = node

        walk(blob)
        for key in SECONDARY:
            hits = [k for k in flat if k == key or k.endswith("." + key)]
            for hit in hits:
                out[hit] = flat[hit]

    meta = path / "meta.txt"
    if meta.exists():
        for line in meta.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                out[f"meta.{key}"] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", type=pathlib.Path)
    parser.add_argument("--json")
    args = parser.parse_args()

    records = [collect(p) for p in args.arms]

    width = max(len(r["arm"]) for r in records)
    for record in records:
        print(f"{record['arm']:<{width}}  rows={record['row_count']:<6} "
              f"fp={str(record['row_fingerprint'])[:16]}")
    print()

    reference = records[0]
    for record in records[1:]:
        same = record["row_fingerprint"] == reference["row_fingerprint"]
        verdict = "IDENTICAL" if same else "DIFFERS"
        print(f"{record['arm']} vs {reference['arm']}: row evidence {verdict}")
        if not same:
            print(f"    rows {reference['row_count']} -> {record['row_count']}")

    print()
    keys = [k for k in records[0] if k.startswith(("meta.", "all_", "parity"))
            or any(k.endswith(s) for s in SECONDARY)]
    for key in keys:
        values = [str(r.get(key)) for r in records]
        if len(set(values)) > 1 or key in ("meta.arm",):
            print(f"{key}: " + " | ".join(values))

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
