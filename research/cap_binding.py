#!/usr/bin/env python3
"""Cross-tabulate the width cap actually applied against the realised draft depth.

Answers the advisor's three questions directly from the `cap=` field that
`costModelDepth` emits, rather than inferring the gate state from a
counterfactual replay.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

LINE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?streak_in=(\d+) cap=(\d+)"
)


def parse(path):
    rows = []
    with open(path, "rb") as fh:
        for raw in fh:
            m = LINE.search(raw.decode("utf-8", "replace"))
            if m:
                rows.append(
                    {
                        "round": int(m.group(1)),
                        "d": int(m.group(2)),
                        "acc": int(m.group(3)),
                        "streak_in": int(m.group(4)),
                        "cap": int(m.group(5)),
                    }
                )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gate", type=int, required=True)
    ap.add_argument("--deep-cap", type=int, required=True)
    ap.add_argument("--shallow-cap", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = parse(args.trace)
    n = len(rows)
    depth_hist = Counter(r["d"] for r in rows)
    cap_hist = Counter(r["cap"] for r in rows)

    shallow = [r for r in rows if r["cap"] == args.shallow_cap]
    deep = [r for r in rows if r["cap"] == args.deep_cap]
    at_deep_cap = [r for r in deep if r["d"] == args.deep_cap]
    at_shallow_cap = [r for r in shallow if r["d"] == args.shallow_cap]

    by_cap = defaultdict(Counter)
    for r in rows:
        by_cap[r["cap"]][r["d"]] += 1

    out = {
        "label": args.label,
        "trace": args.trace,
        "gate": args.gate,
        "deep_cap": args.deep_cap,
        "shallow_cap": args.shallow_cap,
        "round_count": n,
        "depth_histogram": {str(d): depth_hist.get(d, 0) for d in range(0, 9)},
        "cap_histogram": {str(c): cap_hist[c] for c in sorted(cap_hist)},
        "rounds_at_deep_cap": len(at_deep_cap),
        "frac_at_deep_cap": len(at_deep_cap) / n if n else None,
        "rounds_gate_closed": len(shallow),
        "frac_gate_closed": len(shallow) / n if n else None,
        "rounds_gate_open": len(deep),
        "frac_gate_open": len(deep) / n if n else None,
        "frac_of_open_rounds_at_deep_cap": (
            len(at_deep_cap) / len(deep) if deep else None
        ),
        "rounds_at_shallow_cap": len(at_shallow_cap),
        "frac_of_closed_rounds_at_shallow_cap": (
            len(at_shallow_cap) / len(shallow) if shallow else None
        ),
        "depth_by_cap": {str(c): dict(sorted(v.items())) for c, v in by_cap.items()},
        "accepted_total": sum(r["acc"] for r in rows),
        "mean_depth": sum(r["d"] for r in rows) / n if n else None,
        "mean_accepted": sum(r["acc"] for r in rows) / n if n else None,
    }

    print(f"--- {args.label} (gate={args.gate}, deep cap={args.deep_cap}) ---")
    print(f"rounds: {n}")
    print("depth histogram 0..8:", [out["depth_histogram"][str(d)] for d in range(9)])
    print("cap histogram:", out["cap_histogram"])
    print(
        f"Q2 rounds realising depth {args.deep_cap}: {out['rounds_at_deep_cap']}"
        f"/{n} = {100 * out['frac_at_deep_cap']:.2f}%"
    )
    print(
        f"Q3 rounds with streak < {args.gate} (ceiling {args.shallow_cap}): "
        f"{out['rounds_gate_closed']}/{n} = {100 * out['frac_gate_closed']:.2f}%"
    )
    if out["frac_of_open_rounds_at_deep_cap"] is not None:
        print(
            f"    of the {out['rounds_gate_open']} gate-open rounds, "
            f"{out['rounds_at_deep_cap']} hit the deep cap = "
            f"{100 * out['frac_of_open_rounds_at_deep_cap']:.2f}%"
        )
    if out["frac_of_closed_rounds_at_shallow_cap"] is not None:
        print(
            f"    of the {out['rounds_gate_closed']} gate-closed rounds, "
            f"{out['rounds_at_shallow_cap']} hit the shallow cap = "
            f"{100 * out['frac_of_closed_rounds_at_shallow_cap']:.2f}%"
        )
    print("depth by cap:", out["depth_by_cap"])

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
