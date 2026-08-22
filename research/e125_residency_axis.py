#!/usr/bin/env python3
"""E125 - axis 3 of the correction table: resident-simdgroup steps.

Reads the E123 census and reports, per arm and per architecture, registers,
spill bytes, machine text size and resident simdgroups, then flags every arm
whose residency differs from `a_base` on one architecture but not the other.

A change that crosses a residency step on the ranked architecture and not on
the local one has a local-to-ranked transfer above 1, in the opposite direction
to every frame factor in F87.
"""

from __future__ import annotations

import argparse
import json

BASE_ARM = "a_base"


def collect(doc):
    """Rows for every (arm, arch, width).

    Width 0 is the whole-file entry point. Widths 2..5 are the `e118_iso_naN`
    entry points, and those are the kernels that actually run. The published
    `simdgroups` block is derived from the width-0 row only, so residency per
    running kernel has to be recomputed here from the per-width register count.
    """
    budget = doc["simdgroup_budget"]
    rows = []
    for arm, payload in doc["arms"].items():
        for arch, cells in payload.items():
            if arch == "air" or not isinstance(cells, dict):
                continue
            for width, cell in cells.items():
                if not isinstance(cell, dict):
                    continue
                regs = cell.get("registers")
                rows.append(
                    {
                        "arm": arm,
                        "arch": arch,
                        "width": width,
                        "registers": regs,
                        "spill": cell.get("spill_bytes", cell.get("spill")),
                        "text": cell.get("text_bytes", cell.get("text")),
                        "simdgroups": budget[arch] // regs if regs else None,
                        "published_entrypoint_simdgroups": doc["simdgroups"][arch].get(arm),
                    }
                )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="research/e123-artifacts/entrypoint-census.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    doc = json.load(open(args.census))
    rows = collect(doc)
    if not rows:
        print(json.dumps({k: type(v).__name__ for k, v in doc.items()}, indent=1))
        raise SystemExit("census layout not recognised; inspect the keys above")

    print(f"{'arm':18s} {'arch':16s} {'NA':>3s} {'regs':>5s} {'spill':>6s} {'text':>7s} {'simdgrp':>8s}")
    for r in sorted(rows, key=lambda r: (r["arch"], r["arm"], r["width"])):
        print(
            f"{r['arm']:18s} {r['arch']:16s} {str(r['width']):>3s} "
            f"{str(r['registers']):>5s} {str(r['spill']):>6s} {str(r['text']):>7s} "
            f"{str(r['simdgroups']):>8s}"
        )

    base = {(r["arch"], r["width"]): r["simdgroups"] for r in rows if r["arm"] == BASE_ARM}
    steps = []
    for r in rows:
        if r["arm"] == BASE_ARM:
            continue
        b = base.get((r["arch"], r["width"]))
        if b is None or r["simdgroups"] is None:
            continue
        if r["simdgroups"] != b:
            steps.append(
                {
                    "arm": r["arm"],
                    "arch": r["arch"],
                    "width": r["width"],
                    "base_simdgroups": b,
                    "arm_simdgroups": r["simdgroups"],
                    "delta": r["simdgroups"] - b,
                    "relative": (r["simdgroups"] - b) / b,
                }
            )

    print("\nresidency steps against a_base")
    if not steps:
        print("  none")
    for s in steps:
        print(
            f"  {s['arm']:18s} {s['arch']:16s} NA={s['width']:>2s} "
            f"{s['base_simdgroups']} -> {s['arm_simdgroups']} "
            f"({s['relative']:+.2%})"
        )

    by_arm = {}
    for s in steps:
        by_arm.setdefault(s["arm"], set()).add(s["arch"])
    arches = {r["arch"] for r in rows}
    asymmetric = {a: sorted(v) for a, v in by_arm.items() if set(v) != arches}
    print("\narms whose residency step fires on one architecture only")
    print(f"  {asymmetric if asymmetric else 'none'}")

    if args.out:
        json.dump({"rows": rows, "steps": steps, "asymmetric": {k: list(v) for k, v in asymmetric.items()}},
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
