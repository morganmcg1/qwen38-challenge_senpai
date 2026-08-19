#!/usr/bin/env python3
"""E51 end-to-end exactness sensor.

Qwen36MTPBlockSession.traceRow writes one `^mtp-row:` line per evaluated target
row, carrying the position, the top-two token ids and their hexfloat values.

The originally pre-registered signal was the SHA-256 of those lines in emission
order. That signal FAILED its own A/A control: two runs of identical source
emitted 91 and 89 rows. The cause is the drafting schedule, not arithmetic. Both
runs evaluated the same 64 distinct positions, two positions were re-evaluated a
different number of times, and no shared position reported different row
evidence. See research/e51_row_diff.py.

So the PRIMARY signal is the canonical per-position row-evidence digest: every
evaluated position, each with the set of distinct top-two results reported at it.
That digest is invariant to schedule re-evaluation order and still changes if any
row's arithmetic changes. The ordered digest is kept as a secondary schedule
fingerprint and is expected to move between identical runs.

The remaining secondary signals come from the score JSON the trusted parent
writes.

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

    evidence: dict[int, set] = {}
    for line in rows:
        match = ROW.match(line)
        if match:
            pos = int(match.group(1))
            evidence.setdefault(pos, set()).add(
                (match.group(2), match.group(3), match.group(4)))

    canonical = "\n".join(
        f"{pos}|" + ";".join(",".join(t) for t in sorted(evidence[pos]))
        for pos in sorted(evidence))

    out = {
        "arm": path.name,
        "row_count": len(rows),
        "distinct_positions": len(evidence),
        "row_evidence_fingerprint":
            hashlib.sha256(canonical.encode()).hexdigest() if evidence else None,
        "schedule_fingerprint":
            hashlib.sha256("\n".join(rows).encode()).hexdigest() if rows else None,
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
        print(f"{record['arm']:<{width}}  rows={record['row_count']:<5} "
              f"pos={record['distinct_positions']:<4} "
              f"evidence={str(record['row_evidence_fingerprint'])[:16]} "
              f"schedule={str(record['schedule_fingerprint'])[:16]}")
    print()

    reference = records[0]
    for record in records[1:]:
        same = (record["row_evidence_fingerprint"]
                == reference["row_evidence_fingerprint"])
        verdict = "IDENTICAL" if same else "DIFFERS"
        print(f"{record['arm']} vs {reference['arm']}: "
              f"PRIMARY row evidence {verdict}")
        sched = (record["schedule_fingerprint"]
                 == reference["schedule_fingerprint"])
        print(f"    schedule {'identical' if sched else 'differs'}, "
              f"rows {reference['row_count']} -> {record['row_count']}")

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
