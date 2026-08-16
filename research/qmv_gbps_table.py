#!/usr/bin/env python3
"""Print a per-(shape, width) curve for one or two qmv-curve runs.

The crossrow width study has to be read as achieved bandwidth per point, not
only as seconds: a change that lowers occupancy shows up as a bandwidth
collapse, while the seconds curve alone just looks "slower" for unclear reasons.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

FIELDS = ("gbps_nominal", "gbps_stream_corrected", "seconds_per_call", "weight_streams", "hw_efficiency")


def load(tag: str, field: str) -> dict[str, dict[int, float]]:
    path = pathlib.Path(".mlxfast-private/qmv-curve") / tag / "summary.json"
    summary = json.loads(path.read_text())
    out: dict[str, dict[int, float]] = {}
    for row in summary["per_shape_curve"]:
        value = row.get(field)
        if value is None:
            continue
        out.setdefault(row["name"], {})[int(row["m"])] = float(value)
    return out


def table(title: str, curve: dict[str, dict[int, float]], widths: list[int],
          shapes: list[str], fmt: str) -> None:
    print(f"== {title} ==")
    print("shape".ljust(34) + "".join(f"M={m}".rjust(9) for m in widths))
    for shape in shapes:
        print(shape.ljust(34) + "".join(format(curve[shape].get(m, float("nan")), fmt) for m in widths))
    print("median".ljust(34) + "".join(
        format(statistics.median([curve[s][m] for s in shapes if m in curve[s]]), fmt) for m in widths))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_tag")
    ap.add_argument("candidate_tag", nargs="?")
    ap.add_argument("--field", default="gbps_nominal", choices=FIELDS)
    args = ap.parse_args()

    fmt = {"gbps_nominal": "9.1f", "gbps_stream_corrected": "9.1f", "weight_streams": "9.0f",
           "hw_efficiency": "9.3f", "seconds_per_call": "9.6f"}[args.field]
    base = load(args.baseline_tag, args.field)
    widths = sorted({m for series in base.values() for m in series})

    if not args.candidate_tag:
        table(f"{args.field} :: {args.baseline_tag}", base, widths, sorted(base), fmt)
        return

    cand = load(args.candidate_tag, args.field)
    shapes = [s for s in base if s in cand]
    table(f"{args.field} :: {args.baseline_tag}", base, widths, shapes, fmt)
    print()
    table(f"{args.field} :: {args.candidate_tag}", cand, widths, shapes, fmt)
    print()
    ratio = {s: {m: cand[s][m] / base[s][m] for m in base[s] if m in cand[s] and base[s][m]} for s in shapes}
    table(f"ratio {args.candidate_tag} / {args.baseline_tag}", ratio, widths, shapes, "9.3f")


if __name__ == "__main__":
    main()
