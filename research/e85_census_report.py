#!/usr/bin/env python3
"""Read E85 census JSONL legs and report the per-draft materialised-intermediate slope.

Each leg pins one draft width with MLX_E80_FORCE_DRAFTS, so a per-round count is
a straight line in the width: count(d) = base + slope * d. The slope is the
per-draft quantity the E85 assignment asks for. Regressing round TOTALS rather
than reading the `draft_head` phase alone is deliberate: MLX encodes
asynchronously, so a phase bracket on the host thread can smear work into the
next phase, while the round total cannot lose it.

usage: e85_census_report.py --leg DRAFTS PATH [--leg DRAFTS PATH ...]
                            [--skip-rounds N] [--json OUT]
"""

import argparse
import collections
import json
import pathlib
import sys

PHASE_KEYS = ("dispatches", "commits", "barriers", "waits",
              "allocations", "alloc_bytes", "alloc_ns",
              "dispatch_ns", "commit_ns")


def load_leg(path, skip_rounds):
    rounds = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") != "round":
            continue
        if rec["round"] < skip_rounds:
            continue
        rounds.append(rec)
    if not rounds:
        raise SystemExit(f"{path}: no round records after skip {skip_rounds}")

    totals = collections.Counter()
    per_phase = collections.defaultdict(collections.Counter)
    kernels = collections.defaultdict(collections.Counter)
    alloc_sizes = collections.defaultdict(collections.Counter)
    widths = collections.Counter()
    for rec in rounds:
        widths[rec["width"]] += 1
        for phase, entry in rec["phases"].items():
            for key in PHASE_KEYS:
                value = entry.get(key, 0)
                totals[key] += value
                per_phase[phase][key] += value
            for name, count in entry.get("kernels", {}).items():
                kernels[phase][name] += count
            for size, count in entry.get("alloc_sizes", {}).items():
                alloc_sizes[phase][int(size)] += count
    n = len(rounds)
    return {
        "path": str(path),
        "rounds": n,
        "widths": dict(widths),
        "per_round": {k: totals[k] / n for k in PHASE_KEYS},
        "per_round_by_phase": {
            p: {k: c[k] / n for k in PHASE_KEYS} for p, c in per_phase.items()
        },
        "kernels_per_round_by_phase": {
            p: {name: count / n for name, count in c.most_common()}
            for p, c in kernels.items()
        },
        "alloc_sizes_per_round_by_phase": {
            p: {str(size): count / n for size, count in sorted(c.items())}
            for p, c in alloc_sizes.items()
        },
    }


def slope(legs, selector):
    """Least-squares slope and intercept of selector(leg) against draft count."""
    xs = [d for d, _ in legs]
    ys = [selector(leg) for _, leg in legs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None, my
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return m, my - m * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", nargs=2, action="append", metavar=("DRAFTS", "PATH"),
                    required=True)
    ap.add_argument("--skip-rounds", type=int, default=8)
    ap.add_argument("--json")
    args = ap.parse_args()

    legs = [(int(d), load_leg(p, args.skip_rounds)) for d, p in args.leg]
    legs.sort(key=lambda dl: dl[0])

    report = {"skip_rounds": args.skip_rounds, "legs": {}, "per_draft_slope": {}}
    for drafts, leg in legs:
        report["legs"][str(drafts)] = leg

    for key in PHASE_KEYS:
        m, b = slope(legs, lambda leg, k=key: leg["per_round"][k])
        report["per_draft_slope"][key] = {"slope": m, "intercept": b}

    phases = sorted({p for _, leg in legs for p in leg["per_round_by_phase"]})
    report["per_draft_slope_by_phase"] = {}
    for phase in phases:
        entry = {}
        for key in PHASE_KEYS:
            m, b = slope(
                legs,
                lambda leg, p=phase, k=key: leg["per_round_by_phase"]
                .get(p, {}).get(k, 0.0))
            entry[key] = {"slope": m, "intercept": b}
        report["per_draft_slope_by_phase"][phase] = entry

    print(f"rounds analysed per leg: "
          f"{ {d: leg['rounds'] for d, leg in legs} }")
    print()
    print(f"{'metric':<16}" + "".join(f"  d={d:<10}" for d, _ in legs)
          + "  per-draft slope")
    for key in PHASE_KEYS:
        row = f"{key:<16}"
        for _, leg in legs:
            row += f"  {leg['per_round'][key]:>10.2f}"
        m = report["per_draft_slope"][key]["slope"]
        row += f"  {m:>14.3f}" if m is not None else "  n/a"
        print(row)

    print()
    for phase in phases:
        print(f"-- phase {phase}")
        for key in ("dispatches", "commits", "allocations", "alloc_bytes"):
            row = f"   {key:<13}"
            for _, leg in legs:
                row += (f"  {leg['per_round_by_phase'].get(phase, {}).get(key, 0.0):>10.2f}")
            m = report["per_draft_slope_by_phase"][phase][key]["slope"]
            row += f"  {m:>14.3f}" if m is not None else "  n/a"
            print(row)

    widest = legs[-1][1]
    print()
    print(f"-- kernels per round at d={legs[-1][0]}")
    for phase in sorted(widest["kernels_per_round_by_phase"]):
        print(f"   {phase}")
        for name, count in widest["kernels_per_round_by_phase"][phase].items():
            print(f"      {count:>8.2f}  {name}")
    print()
    print(f"-- device allocation sizes per round at d={legs[-1][0]}")
    for phase in sorted(widest["alloc_sizes_per_round_by_phase"]):
        sizes = widest["alloc_sizes_per_round_by_phase"][phase]
        if not sizes:
            continue
        print(f"   {phase}")
        for size, count in sorted(sizes.items(), key=lambda kv: -kv[1])[:20]:
            print(f"      {count:>8.2f} x {int(size):>12,} B")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
