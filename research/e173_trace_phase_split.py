#!/usr/bin/env python3
"""E173: host phase versus eval wait, per verify width, from in-repo traces.

`verify_build_us` is not pure host time: it blocks on the device when MLX has
to wait for buffers, and the campaign ledger records it as roughly 97 % GPU
wait. This script therefore reports the phase table as an upper bound on host
build time, never as the host share of `F`. The isolated host census
(`E173AdmissionCensusTests`) is the instrument that separates them.

harness=local. Source traces carry `gate_qualified_for_timing=false` and
`trace_perturbs_timing=true`, so only within-run differentials are meaningful.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
FILES = [
    ROOT / "results/e37/medicine-rounds.txt",
    ROOT / "results/e37/natural_history-rounds.txt",
]
FIELDS = (
    "round d acc draft_build_us verify_build_us eval_wall_us "
    "readout_us commit_us upkeep_us round_us"
).split()
DROP_FIRST_ROUNDS = 2


def parse() -> list[dict[str, float]]:
    out = []
    for path in FILES:
        for line in path.read_text().splitlines():
            if not line.startswith("mtp-trace: round="):
                continue
            kv = dict(re.findall(r"(\w+)=(-?[\d.]+)", line))
            row = {k: float(kv[k]) for k in FIELDS if k in kv}
            if len(row) == len(FIELDS) and row["round"] > DROP_FIRST_ROUNDS:
                row["prompt"] = path.stem
                out.append(row)
    return out


def main() -> int:
    rows = parse()
    by_width: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_width[int(r["d"]) + 1].append(r)

    table = []
    print(
        f"{'M':>2} {'n':>4} {'round':>8} {'verify_build':>12} {'eval_wall':>10} "
        f"{'draft_build':>11} {'readout':>8} {'commit':>7} {'upkeep':>7} "
        f"{'unnamed':>8}"
    )
    for m in sorted(by_width):
        g = by_width[m]

        def med(key: str) -> float:
            return statistics.median(x[key] for x in g)

        named = sum(
            med(k)
            for k in (
                "verify_build_us",
                "draft_build_us",
                "readout_us",
                "commit_us",
                "upkeep_us",
            )
        )
        entry = {
            "m": m,
            "n": len(g),
            "round_ms": med("round_us") / 1e3,
            "verify_build_ms": med("verify_build_us") / 1e3,
            "eval_wall_ms": med("eval_wall_us") / 1e3,
            "draft_build_ms": med("draft_build_us") / 1e3,
            "readout_ms": med("readout_us") / 1e3,
            "commit_ms": med("commit_us") / 1e3,
            "upkeep_ms": med("upkeep_us") / 1e3,
            "unnamed_ms": (med("round_us") - named) / 1e3,
        }
        table.append(entry)
        print(
            f"{m:2d} {entry['n']:4d} {entry['round_ms']:8.2f} "
            f"{entry['verify_build_ms']:12.2f} {entry['eval_wall_ms']:10.2f} "
            f"{entry['draft_build_ms']:11.2f} {entry['readout_ms']:8.3f} "
            f"{entry['commit_ms']:7.3f} {entry['upkeep_ms']:7.3f} "
            f"{entry['unnamed_ms']:8.2f}"
        )

    out = ROOT / "e173-artifacts/trace-phase-split.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "e173-trace-phase-split",
                "harness": "local",
                "gate_qualified_for_timing": False,
                "trace_perturbs_timing": True,
                "caveat": "verify_build_us is dominated by device wait and is an "
                "upper bound on host build time, not the host share of F",
                "rounds_used": len(rows),
                "drop_first_rounds": DROP_FIRST_ROUNDS,
                "source_files": [str(p.relative_to(ROOT.parent)) for p in FILES],
                "per_width": table,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
