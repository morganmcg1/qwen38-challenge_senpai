#!/usr/bin/env python3
"""Per-draft slope reader for the E85 materialised-intermediate census.

Reads two or more census legs recorded at different forced draft widths and
reports, for the MTP timed worker only, the per-round mean of dispatches,
commits, barriers, device allocations and allocated bytes in each phase, plus
the least-squares slope against the forced draft count.
"""

from __future__ import annotations

import argparse
import collections
import json

FIELDS = (
    "dispatches",
    "commits",
    "barriers",
    "allocations",
    "alloc_bytes",
    "waits",
    "dispatch_ns",
    "commit_ns",
)


def load(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mtp_pid(records: list[dict]) -> int:
    """The MTP timed worker is the process whose rounds carry draft phases."""
    candidates = {
        r["pid"]
        for r in records
        if r["event"] == "round" and "draft_head" in r.get("phases", {})
    }
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one MTP worker pid, found {sorted(candidates)}")
    return candidates.pop()


def phase_means(records: list[dict], pid: int, skip: int) -> dict[str, dict[str, float]]:
    rows = [
        r
        for r in records
        if r["event"] == "round" and r["pid"] == pid and r["round"] > skip
    ]
    out: dict[str, dict[str, float]] = {}
    names = sorted({k for r in rows for k in r["phases"]})
    for name in names:
        agg = collections.defaultdict(list)
        for r in rows:
            phase = r["phases"].get(name)
            if phase is None:
                continue
            for field in FIELDS:
                agg[field].append(phase.get(field, 0))
        if not agg["dispatches"]:
            continue
        out[name] = {f: sum(v) / len(v) for f, v in agg.items()}
        out[name]["n"] = float(len(agg["dispatches"]))
    return out


def kernel_means(records: list[dict], pid: int, skip: int, phase: str) -> dict[str, float]:
    rows = [
        r
        for r in records
        if r["event"] == "round" and r["pid"] == pid and r["round"] > skip
    ]
    total = collections.Counter()
    n = 0
    for r in rows:
        block = r["phases"].get(phase)
        if block is None:
            continue
        n += 1
        for kernel, count in block.get("kernels", {}).items():
            total[kernel] += count
    if n == 0:
        return {}
    return {k: v / n for k, v in total.items()}


def slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", nargs=2, action="append", metavar=("DRAFTS", "PATH"), required=True)
    ap.add_argument("--skip-rounds", type=int, default=7)
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--kernels", metavar="PHASE")
    args = ap.parse_args()

    legs = []
    for drafts, path in args.leg:
        records = load(path)
        pid = mtp_pid(records)
        legs.append(
            {
                "drafts": float(drafts),
                "path": path,
                "pid": pid,
                "rounds": sum(1 for r in records if r["event"] == "round" and r["pid"] == pid),
                "phases": phase_means(records, pid, args.skip_rounds),
                "kernels": kernel_means(records, pid, args.skip_rounds, args.kernels)
                if args.kernels
                else {},
            }
        )

    for leg in legs:
        print(f"leg drafts={leg['drafts']:g} pid={leg['pid']} rounds={leg['rounds']} path={leg['path']}")
        for name, vals in leg["phases"].items():
            print(
                f"  {name:<16} n={vals['n']:.0f} disp={vals['dispatches']:.1f} "
                f"commit={vals['commits']:.2f} barrier={vals['barriers']:.1f} "
                f"alloc={vals['allocations']:.2f} alloc_bytes={vals['alloc_bytes']:.0f} "
                f"disp_ns={vals['dispatch_ns']:.0f} commit_ns={vals['commit_ns']:.0f}"
            )

    if len(legs) >= 2:
        print("\nper-draft slope (round total against forced draft count)")
        names = sorted({n for leg in legs for n in leg["phases"]})
        for name in names:
            xs, ys = [], []
            present = [leg for leg in legs if name in leg["phases"]]
            if len(present) < 2:
                continue
            print(f"  {name}")
            for field in ("dispatches", "commits", "barriers", "allocations", "alloc_bytes"):
                xs = [leg["drafts"] for leg in present]
                ys = [leg["phases"][name][field] for leg in present]
                print(f"    d{field}/ddraft = {slope(xs, ys):+.3f}   values={['%.2f' % y for y in ys]}")

    if args.kernels and len(legs) >= 2:
        print(f"\nper-kernel per-round counts and per-draft slope in phase '{args.kernels}'")
        names = sorted({k for leg in legs for k in leg["kernels"]})
        rows = []
        for name in names:
            xs = [leg["drafts"] for leg in legs]
            ys = [leg["kernels"].get(name, 0.0) for leg in legs]
            rows.append((slope(xs, ys), name, ys))
        for s, name, ys in sorted(rows, key=lambda r: -abs(r[0])):
            short = name if len(name) <= 62 else name[:59] + "..."
            print(f"    {s:+7.3f} /draft   {['%.2f' % y for y in ys]}   {short}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(legs, handle, indent=2)


if __name__ == "__main__":
    main()
