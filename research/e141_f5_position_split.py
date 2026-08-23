#!/usr/bin/env python3
"""E141 F5 item 4: the unproposable-token census split by draft position.

The advisor asked for the mirror of askeladd's E143 C-a census. His counts
first divergences; mine counts positions inside the live row ledger. Both
populations are reported here so the disagreement, if any, has somewhere to
land.

`draft_index` in the verify ledger is 0-BASED, so draft position 1 is index 0.

Two denominators are reported for every position, because they answer
different questions:

  reached   rounds that actually produced a draft row at this position. A
            round that ended at position 2 never gave position 5 a chance, so
            dividing by the round count understates the deeper positions.
  rounds    every round with at least one draft row.

The reach-weighted rate is events / reached. It is the probability that a
round which GETS to this position is truncated here by an unproposable token,
which is the quantity a position-gated widened probe would act on.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SHIPPED_PREFIX = 98_304
CONTROL_START = 248_044
CONTROL_END = 248_070

# Effective draft length, from the advisor's F5 note.
EFFECTIVE_DRAFT_LEN = {"beagle_a": 4.3818, "essays_montaigne": 5.0870}


def proposable(token: int) -> bool:
    return token < SHIPPED_PREFIX or CONTROL_START <= token < CONTROL_END


def census(path: Path) -> dict:
    blob = json.loads(path.read_text())
    rounds: dict[int, list[dict]] = {}
    for row in blob.get("row_ledger") or []:
        if row["kind"] == "draft":
            rounds.setdefault(row["round"], []).append(row)

    reached = Counter()
    events = Counter()
    rejects = Counter()
    for _, rows in sorted(rounds.items()):
        ordered = sorted(rows, key=lambda r: r["draft_index"])
        for r in ordered:
            reached[r["draft_index"]] += 1
        rejected = [r for r in ordered if not r["accepted"]]
        if not rejected:
            continue
        first = rejected[0]
        pos = first["draft_index"]
        rejects[pos] += 1
        if not proposable(first["reference_token"]):
            events[pos] += 1

    depth = max(reached) + 1 if reached else 0
    positions = []
    for d in range(depth):
        positions.append(
            {
                "position_1based": d + 1,
                "reached": reached[d],
                "first_rejects": rejects[d],
                "unproposable_events": events[d],
                "rate_per_reached_pct": 100.0 * events[d] / reached[d]
                if reached[d]
                else None,
            }
        )

    total = sum(events.values())
    at1 = events[0]
    return {
        "path": str(path),
        "rounds": len(rounds),
        "draft_rows": sum(reached.values()),
        "positions": positions,
        "unproposable_total": total,
        "unproposable_at_position1": at1,
        "unproposable_beyond_position1": total - at1,
        "position1_share_pct": 100.0 * at1 / total if total else None,
        # Reach weighting. A position-gated probe pays only at the positions
        # it covers, so the useful comparison is the event rate per reached
        # row, not per round.
        "rate_at_position1_per_reached_pct": 100.0 * at1 / reached[0]
        if reached[0]
        else None,
        "rate_beyond_position1_per_reached_pct": 100.0
        * (total - at1)
        / sum(reached[d] for d in reached if d >= 1)
        if sum(reached[d] for d in reached if d >= 1)
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify-dir",
        default=str(
            Path.home()
            / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e141/verify/worker-15996ba2"
        ),
    )
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--arm", default="shipped")
    ap.add_argument("--out", default="research/e141-f5-position-split.json")
    args = ap.parse_args()

    report = {"steps": args.steps, "arm": args.arm, "seeds": {}}
    for seed in EFFECTIVE_DRAFT_LEN:
        path = Path(args.verify_dir) / f"{seed}_{args.arm}_{args.steps}.json"
        if not path.exists():
            print(f"  missing {path}")
            continue
        c = census(path)
        c["effective_draft_len_advisor"] = EFFECTIVE_DRAFT_LEN[seed]
        c["effective_draft_len_measured"] = (
            c["draft_rows"] / c["rounds"] if c["rounds"] else None
        )
        report["seeds"][seed] = c

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for seed, c in report["seeds"].items():
        print(f"\n== {seed}  arm={args.arm}  {args.steps} tokens ==")
        print(
            f"  rounds {c['rounds']}  draft rows {c['draft_rows']}  "
            f"effective draft len measured {c['effective_draft_len_measured']:.4f} "
            f"(advisor {c['effective_draft_len_advisor']:.4f})"
        )
        print(f"  {'pos':>4s} {'reached':>8s} {'1st rej':>8s} {'C-a':>5s} {'rate/reached':>13s}")
        for p in c["positions"]:
            rate = p["rate_per_reached_pct"]
            print(
                f"  {p['position_1based']:4d} {p['reached']:8d} "
                f"{p['first_rejects']:8d} {p['unproposable_events']:5d} "
                + (f"{rate:12.4f} %" if rate is not None else f"{'n/a':>13s}")
            )
        share = c["position1_share_pct"]
        print(
            f"  TOTAL unproposable {c['unproposable_total']}  "
            f"at position 1: {c['unproposable_at_position1']}  "
            f"beyond: {c['unproposable_beyond_position1']}  "
            + (f"share {share:.1f} %" if share is not None else "share n/a")
        )
        r1 = c["rate_at_position1_per_reached_pct"]
        r2 = c["rate_beyond_position1_per_reached_pct"]
        print(
            "  reach weighted: position 1 "
            + (f"{r1:.4f} %" if r1 is not None else "n/a")
            + ", positions 2+ "
            + (f"{r2:.4f} %" if r2 is not None else "n/a")
        )

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
