#!/usr/bin/env python3
"""Honest coverage accounting for the E42 bit-exactness parity grid.

The parity suite reports `covering_cells_by_bits`, which counts cells whose
`in_kernel_path` is anything other than `qmv_fast_impl`.  That metric answers
"did this cell reach a crossrow body", which is the right question for the
crossrow arms (p2, p6, m6) and the *wrong* question for m1.

m1's edit is a width-1 dispatch inserted immediately before the generic
fall-through, so every cell m1 treats is a `qmv_fast_impl` cell -- exactly the
cells the crossrow metric labels non-covering.  Reporting m1 as "0 covering
cells" would understate it to zero; reporting it under the same header as the
crossrow arms would overstate what the shared metric measures.  This script
emits both numbers per arm together with the mechanism that explains the gap.

Usage:
  python3 research/e42_covering_cells.py [--groups A B] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PARITY_ROOT = REPO / ".mlxfast-private" / "e42" / "parity"
GENERIC = "qmv_fast_impl"

sys.path.insert(0, str(REPO / "research"))
from e42_perturb import ARMS  # noqa: E402

# How each arm's edit selects the instantiations it perturbs.
#   "crossrow": template argument on the crossrow switch cases, which the
#               dispatch only reaches at bits == 4 and out_vec_size >= 1024.
#   "generic":  an `ntg.x == 1` dispatch inserted before the generic
#               fall-through, which is reached at every bits and every
#               out_vec_size.
MECHANISM = {
    "p2": "crossrow",
    "p6": "crossrow",
    "m6": "crossrow",
    "m1": "generic",
}

ARM_FAMILY = {
    "p2L1": "p2",
    "p2L2": "p2",
    "p6L1": "p6",
    "p6L2": "p6",
    "m6L2": "m6",
    "m1L1": "m1",
}

# `ARMS["m1"]` is empty because m1 edits no crossrow switch case; its treated
# width comes from the `ntg.x == 1` guard that M1_EDITS inserts.
M1_TREATED_WIDTHS = {1}


def load_ref_cells(groups: list[str]) -> tuple[list[dict], dict]:
    """Cell table from the reference arm, plus a cross-group determinism check."""
    refs = {}
    for group in groups:
        path = PARITY_ROOT / group / "ref.json"
        if path.exists():
            refs[group] = json.loads(path.read_text())
    if not refs:
        raise SystemExit(f"no ref.json under {PARITY_ROOT} for groups {groups}")

    names = sorted(refs)
    primary = refs[names[0]]
    check = {
        "groups_with_ref": names,
        "cross_group_ref_digests_identical": None,
        "cross_group_cells_compared": 0,
        "cross_group_differing_cells": 0,
    }
    if len(names) > 1:
        base = {(e["shape"], e["bits"], e["m"]): e["digest"] for e in primary["entries"]}
        differing = 0
        compared = 0
        for other in names[1:]:
            for e in refs[other]["entries"]:
                key = (e["shape"], e["bits"], e["m"])
                if key in base:
                    compared += 1
                    if base[key] != e["digest"]:
                        differing += 1
        check["cross_group_cells_compared"] = compared
        check["cross_group_differing_cells"] = differing
        check["cross_group_ref_digests_identical"] = differing == 0
    return primary["entries"], check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["A", "B"])
    ap.add_argument(
        "--out", default=str(REPO / "research" / "e42-artifacts" / "covering-cells.json")
    )
    args = ap.parse_args()

    entries, cross_group = load_ref_cells(args.groups)
    shapes = sorted({e["shape"] for e in entries})
    widths = sorted({e["m"] for e in entries})
    bits_list = sorted({e["bits"] for e in entries}, reverse=True)
    total = len(entries)

    path_by_cell = {(e["shape"], e["bits"], e["m"]): e["in_kernel_path"] for e in entries}

    # Reproduce the suite's own metric so the two numbers sit side by side.
    covering_by_bits = {}
    cells_by_bits = {}
    for (shape, bits, m), path in path_by_cell.items():
        cells_by_bits[bits] = cells_by_bits.get(bits, 0) + 1
        if path != GENERIC:
            covering_by_bits[bits] = covering_by_bits.get(bits, 0) + 1

    per_arm = {}
    treated_union: set[tuple[str, int, int]] = set()
    for arm, family in ARM_FAMILY.items():
        mechanism = MECHANISM[family]
        treated_widths = M1_TREATED_WIDTHS if family == "m1" else set(ARMS[family])
        crossrow_covering = 0
        treated: set[tuple[str, int, int]] = set()
        for cell, path in path_by_cell.items():
            _, bits, m = cell
            is_crossrow = path != GENERIC
            if m in treated_widths and is_crossrow:
                crossrow_covering += 1
            if mechanism == "crossrow":
                if m in treated_widths and is_crossrow and bits == 4:
                    treated.add(cell)
            else:
                if m in treated_widths and not is_crossrow:
                    treated.add(cell)
        treated_union |= treated
        treated_bits = {}
        for _, bits, _ in treated:
            treated_bits[bits] = treated_bits.get(bits, 0) + 1
        per_arm[arm] = {
            "family": family,
            "mechanism": mechanism,
            "treated_widths": sorted(treated_widths),
            "crossrow_covering_cells": crossrow_covering,
            "treated_cells": len(treated),
            "treated_cells_by_bits": {str(k): v for k, v in sorted(treated_bits.items())},
            "control_cells": total - len(treated),
        }

    untreated = [c for c in path_by_cell if c not in treated_union]
    untreated_split = {
        "bits_4_crossrow_widths_outside_any_arm": sum(
            1 for (s, b, m) in untreated if b == 4 and path_by_cell[(s, b, m)] != GENERIC
        ),
        "bits_4_generic": sum(
            1 for (s, b, m) in untreated if b == 4 and path_by_cell[(s, b, m)] == GENERIC
        ),
        "bits_3_all": sum(1 for (s, b, m) in untreated if b == 3),
    }

    payload = {
        "grid": {
            "shapes": len(shapes),
            "widths": widths,
            "bits": bits_list,
            "cells_total": total,
            "cells_by_bits": {str(k): v for k, v in sorted(cells_by_bits.items())},
        },
        "suite_metric_covering_cells_by_bits": {
            str(k): v for k, v in sorted(covering_by_bits.items())
        },
        "suite_metric_definition": (
            "cells whose in_kernel_path != 'qmv_fast_impl'; a crossrow-reach "
            "metric, not a per-arm treatment metric"
        ),
        "per_arm": per_arm,
        "treated_union_cells": len(treated_union),
        "untreated_cells": len(untreated),
        "untreated_composition": untreated_split,
        "m1_note": (
            "m1 has 0 crossrow-covering cells by construction: its edit is a "
            "width-1 dispatch before the generic fall-through, so the cells it "
            "treats are exactly the qmv_fast_impl cells the crossrow metric "
            "excludes. Both numbers are reported; neither alone is honest."
        ),
        "bits_3_note": (
            "bits=3 cells are pure controls for p2/p6/m6 (their guard requires "
            "bits==4) but are treated by m1 at width 1, because m1's insert sits "
            "downstream of the bit-width gate."
        ),
        "cross_group_reference_check": cross_group,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"grid: {len(shapes)} shapes x {len(widths)} widths x {bits_list} = {total} cells")
    print(f"suite covering_cells_by_bits: {payload['suite_metric_covering_cells_by_bits']}")
    print()
    print(f"{'arm':<6} {'mech':<9} {'crossrow-cov':>12} {'treated':>8} {'by bits':<14} {'controls':>9}")
    for arm in ("p2L1", "p2L2", "p6L1", "p6L2", "m6L2", "m1L1"):
        r = per_arm[arm]
        print(
            f"{arm:<6} {r['mechanism']:<9} {r['crossrow_covering_cells']:>12} "
            f"{r['treated_cells']:>8} {str(r['treated_cells_by_bits']):<14} "
            f"{r['control_cells']:>9}"
        )
    print()
    print(f"treated union: {len(treated_union)} / {total}   untreated: {len(untreated)}")
    print(f"untreated composition: {untreated_split}")
    if cross_group["cross_group_ref_digests_identical"] is not None:
        verdict = "IDENTICAL" if cross_group["cross_group_ref_digests_identical"] else "DIFFER"
        print(
            f"cross-group ref check: {verdict} "
            f"({cross_group['cross_group_cells_compared']} cells compared, "
            f"{cross_group['cross_group_differing_cells']} differing)"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
