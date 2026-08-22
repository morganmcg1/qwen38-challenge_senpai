#!/usr/bin/env python3
"""Check the launched QMV column count a leg recorded against both grids.

`MLX_E120_QMV_GRID` selects `Qwen35CustomQMV.Grid`, and the pipeline log's
`grid` field only reports the parsed enum, so it witnesses that the environment
reached the worker and nothing more. `columns_by_width` is read back off the
dispatch argument itself, so it also witnesses that `launch(m:n:)` honoured the
setting.

The check is falsifiable by construction: it requires the expected map for the
grid under test AND requires the other grid's map to differ. Run it on a `wide`
leg with `--want tight` to watch it fail before trusting it on a `tight` leg.

  python3 research/e135_columns_check.py LOG.json --want tight
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def plan_from(witness: str) -> dict[int, int]:
    """`e120_width_plan/3:3:4,...` -> {m: ipg}."""
    body = witness.split("/", 1)[1]
    out = {}
    for cell in body.split(","):
        m, ipg, _rps = (int(v) for v in cell.split(":"))
        out[m] = ipg
    return out


def expected(plan: dict[int, int], grid: str) -> dict[int, int]:
    if grid == "wide":
        return {m: m for m in plan}
    return {m: -(-m // ipg) for m, ipg in plan.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--want", required=True, choices=("wide", "tight"))
    args = ap.parse_args()

    path = pathlib.Path(args.log)
    if not path.exists():
        print(f"e135_columns_check: no pipeline log at {path}")
        return 3
    doc = json.loads(path.read_text())

    plan = plan_from(doc["plan"])
    seen = {int(k): int(v) for k, v in doc.get("columns_by_width", {}).items()}
    if not seen:
        print("e135_columns_check: the leg recorded no launched columns")
        return 3

    other = "tight" if args.want == "wide" else "wide"
    want = expected(plan, args.want)
    anti = expected(plan, other)
    want_seen = {m: want[m] for m in seen}
    anti_seen = {m: anti[m] for m in seen}

    print(f"e135_columns_check: {path}")
    print(f"  parsed grid enum       {doc.get('grid')}")
    print(f"  plan                   {doc['plan']}")
    print(f"  launched columns       {dict(sorted(seen.items()))}")
    print(f"  expected under {args.want:5s}   {dict(sorted(want_seen.items()))}")
    print(f"  expected under {other:5s}   {dict(sorted(anti_seen.items()))}")

    if seen != want_seen:
        print(f"  VERDICT: FAIL, the launched grid is not {args.want}")
        return 1
    if want_seen == anti_seen:
        print("  VERDICT: FAIL, the two grids agree on every width this leg "
              "reached, so the check cannot discriminate")
        return 2
    print(f"  VERDICT: pass, and the {other} map differs at "
          f"{sum(1 for m in seen if want_seen[m] != anti_seen[m])} of "
          f"{len(seen)} widths, so the check could have failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
