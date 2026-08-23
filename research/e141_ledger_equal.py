#!/usr/bin/env python3
"""E141: decide whether two verify legs produced the SAME decode.

Two arms can report the same round count by accident. The claim that a leaf
repartition is acceptance-neutral is much stronger than that, so it needs the
row ledger itself: the same emitted tokens, the same proposed draft tokens in
the same order, and the same accept or reject decision on every row.

The comparison deliberately ignores the fields that MUST differ between legs
(paths, timings, the arm's own configuration echo) and compares only the
decode. It prints the first divergence when there is one, because "not equal"
is only useful when it says where.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The decode is exactly these per-row facts. `reference_*` is the target's own
# answer, so including it also proves the two legs faced the same target.
ROW_KEYS = (
    "round",
    "row_index",
    "draft_index",
    "kind",
    "token",
    "accepted",
    "reference_token",
)


def ledger(path: Path) -> list[tuple]:
    blob = json.loads(path.read_text())
    rows = blob.get("row_ledger") or []
    return [tuple(r.get(k) for k in ROW_KEYS) for r in rows]


def headline(path: Path) -> dict:
    blob = json.loads(path.read_text())
    return {
        k: blob.get(k)
        for k in (
            "round_count",
            "accepted_draft_total",
            "rejected_draft_total",
            "emitted_token_total",
            "declared_rows_total",
            "all_tokens_matched",
            "residual_divergence_count",
        )
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("left")
    ap.add_argument("right")
    args = ap.parse_args()

    left, right = Path(args.left), Path(args.right)
    a, b = ledger(left), ledger(right)
    ha, hb = headline(left), headline(right)

    print(f"left  {left.name}  rows={len(a)}")
    print(f"right {right.name}  rows={len(b)}")
    for key in ha:
        flag = "" if ha[key] == hb[key] else "   <-- DIFFERS"
        print(f"  {key:28s} {ha[key]!s:>10} {hb[key]!s:>10}{flag}")

    if a == b:
        print("VERDICT identical row ledger")
        return
    print(f"VERDICT ledgers differ (lengths {len(a)} vs {len(b)})")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"  first divergence at row {i}")
            for k, u, v in zip(ROW_KEYS, x, y):
                mark = "" if u == v else "  <--"
                print(f"    {k:18s} {u!s:>10} {v!s:>10}{mark}")
            break
    raise SystemExit(1)


if __name__ == "__main__":
    main()
