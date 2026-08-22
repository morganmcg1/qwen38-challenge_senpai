#!/usr/bin/env python3
"""Identify the per-launched-column cost from realised round widths.

    python3 research/e135_per_width.py --label t1 --out research/e135-artifacts

WHY THIS DESIGN. The runtime has no forced-width knob, and adding one would
mean editing `Qwen36MTPBlockSession.swift`, which another student owns this
round. The trace already labels every round with its own dispatch width, so the
realised rounds carry the same information without any source change.

Holding the width fixed and differencing the two grids removes the work term
outright:

    round_us_wide(M) - round_us_tight(M) = a * empty_columns(M)
    empty_columns(M) = M - ceil(M / ipg(M))

Every other cost in the round -- weight traffic, issue, occupancy, head chain,
readout, upkeep -- is identical at a fixed M, because the two arms run the same
bytes on the same schedule and differ only in how many threadgroups return
before their first load. Regressing the within-width difference on
`empty_columns(M)` through the origin therefore identifies `a` with no work
term to separate it from.

RULE 106. The comparison never crosses a width, so the tail is width-matched by
construction. Round 1 of every leg is dropped regardless, because the measured
first-round excess is real, GPU-side and width-dependent.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics

OUT = pathlib.Path("research/out")
ROUND = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) .*?round_us=(\d+)",
                   re.M)

# The compiled default, `Table.onePass67`.
IPG = {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 4, 9: 3}

# F83 ranked prompt weights over verify width, from the assignment.
RANKED_WIDTH_SLOPE_US = 3388.5


def empty_columns(m: int) -> int:
    return m - -(-m // IPG[m])


def read_meta(path: pathlib.Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def rounds(tag: pathlib.Path) -> list[tuple[int, int, int]]:
    """(round index, verify width, round_us) with round 1 dropped."""
    trace = tag / "trace.txt"
    if not trace.exists():
        return []
    found = []
    for m in ROUND.finditer(trace.read_text()):
        index, d, _acc, us = (int(g) for g in m.groups())
        if index <= 1:
            continue
        found.append((index, d + 1, us))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="t1")
    ap.add_argument("--out", default="research/e135-artifacts")
    args = ap.parse_args()

    legs = []
    for d in sorted(OUT.glob(f"e135{args.label}*")):
        meta = read_meta(d / "meta.txt")
        arm = meta.get("e135_grid")
        if arm not in ("wide", "tight"):
            continue
        rs = rounds(d)
        if rs:
            legs.append({"tag": d.name, "arm": arm, "rounds": rs, "meta": meta})
    if not legs:
        print(f"e135_per_width: no traced legs for label {args.label!r}")
        return 1

    print(f"E135 per-width launch cost, label {args.label}: {len(legs)} legs")
    for leg in legs:
        print(f"  {leg['tag']:32s} {leg['arm']:6s} {len(leg['rounds']):4d} rounds")
    print()

    by = {"wide": {}, "tight": {}}
    for leg in legs:
        for _index, m, us in leg["rounds"]:
            by[leg["arm"]].setdefault(m, []).append(us)

    print(f"{'M':>3s} {'empty cols':>10s} {'n wide':>7s} {'n tight':>8s} "
          f"{'wide us':>10s} {'tight us':>10s} {'diff us':>9s} "
          f"{'us/col':>8s} {'pct':>7s}")
    points = []
    per_width = {}
    for m in sorted(set(by["wide"]) & set(by["tight"])):
        w, t = by["wide"][m], by["tight"][m]
        if len(w) < 3 or len(t) < 3:
            continue
        mw, mt = statistics.median(w), statistics.median(t)
        cols = empty_columns(m)
        diff = mw - mt
        per_col = diff / cols if cols else float("nan")
        print(f"{m:3d} {cols:10d} {len(w):7d} {len(t):8d} "
              f"{mw:10.1f} {mt:10.1f} {diff:+9.1f} {per_col:+8.2f} "
              f"{100 * diff / mw:+7.3f}")
        per_width[m] = {"empty_columns": cols, "n_wide": len(w),
                        "n_tight": len(t), "median_wide_us": mw,
                        "median_tight_us": mt, "diff_us": diff,
                        "us_per_column": per_col}
        if cols:
            points.append((cols, diff, len(w), len(t)))
    print()

    result = {"label": args.label, "per_width": per_width}
    if len(points) >= 2:
        # Through the origin: a = sum(x*y) / sum(x*x).
        sxy = sum(x * y for x, y, _, _ in points)
        sxx = sum(x * x for x, _, _, _ in points)
        a = sxy / sxx
        resid = [y - a * x for x, y, _, _ in points]
        dof = len(points) - 1
        sigma = math.sqrt(sum(r * r for r in resid) / dof) if dof else float("nan")
        se = sigma / math.sqrt(sxx) if sxx else float("nan")
        sst = sum(y * y for _, y, _, _ in points)
        r2 = 1 - sum(r * r for r in resid) / sst if sst else float("nan")
        print(f"a = {a:+.2f} +- {se:.2f} us per launched column "
              f"({len(points)} widths, dof {dof}, R2 {r2:.3f})")
        share = 100 * a * 1.0 / RANKED_WIDTH_SLOPE_US
        print(f"one extra launched column is {share:.3f} % of the ranked "
              f"{RANKED_WIDTH_SLOPE_US:.1f} us per unit verify width")
        result |= {
            "e135_launch_cost_us_per_column": a,
            "e135_launch_cost_us_per_column_se": se,
            "e135_launch_cost_r2": r2,
            "e135_ranked_launch_share_pct": share,
        }
    else:
        print("not enough widths with a usable round count to fit a")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{args.label}-per-width.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
