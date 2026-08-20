#!/usr/bin/env python3
"""Print the E70 rung-1 kernel names side by side, real arch vs forced arch.

harness=arch-probe. The forced arm makes `is_nax_available()` return true on
gen-16 silicon, so the kernel NAME it selects is evidence and every number it
computes is meaningless.

usage:
  python3 research/e70_rung1_diff.py [--root research/out/e70-rung1] \
      [--forced applegpu_g17s] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib

# The audited families. Everything else in a capture is RNG or elementwise
# noise from building the inputs.
SIGNAL_PREFIXES = ("affine_", "steel_", "sdpa_", "gemv", "block_softmax")

CELL_ORDER = [
    "qmv_m1", "qmv_m5", "qmv_m9",
    "qmm_m10", "qmm_m511", "qmm_m512",
    "sdpa_prefill_512",
    "sdpa_vector_q1_k768", "sdpa_vector_q1_k1030",
    "sdpa_vector_q5_k768", "sdpa_vector_q5_k1030",
    "dense_gemv_m1", "dense_matmul_m511",
]


def load_arm(directory: pathlib.Path) -> tuple[dict, dict]:
    cells: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        for cell in json.loads(path.read_text()).get("cells", []):
            cells[cell["cell"]] = cell
    exits: dict[str, int] = {}
    manifest = directory / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                exits[entry["cell"]] = entry["exit"]
    return cells, exits


def signal(cell: dict) -> list[str]:
    return [n for n in cell["kernel_names"] if n.startswith(SIGNAL_PREFIXES)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research/out/e70-rung1")
    parser.add_argument("--forced", default="applegpu_g17s")
    parser.add_argument("--json")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    real, real_exits = load_arm(root / "real")
    forced, forced_exits = load_arm(root / args.forced)

    order = CELL_ORDER + [c for c in real if c not in CELL_ORDER]
    rows = []
    for cell_id in order:
        real_cell = real.get(cell_id)
        forced_cell = forced.get(cell_id)
        real_names = signal(real_cell) if real_cell else None
        forced_names = signal(forced_cell) if forced_cell else None
        rows.append({
            "cell": cell_id,
            "site": (real_cell or forced_cell or {}).get("site", ""),
            "shape": (real_cell or forced_cell or {}).get("shape", ""),
            "real_applegpu_g16s": real_names,
            f"forced_{args.forced}": forced_names,
            "forced_exit": forced_exits.get(cell_id),
            "real_exit": real_exits.get(cell_id, real_exits.get("all")),
            "kernel_changed": (
                None if real_names is None or forced_names is None
                else real_names != forced_names),
        })

    print(f"E70 rung 1 -- kernel selection, real vs forced   harness=arch-probe")
    print(f"real arm: applegpu_g16s (this host)   forced arm: {args.forced}")
    print()
    for row in rows:
        print(f"### {row['cell']}   forced_exit={row['forced_exit']}")
        print(f"    {row['shape']}")
        print(f"    real   {row['real_applegpu_g16s']}")
        print(f"    forced {row[f'forced_{args.forced}']}")
        print(f"    changed: {row['kernel_changed']}")
        print()

    changed = [r["cell"] for r in rows if r["kernel_changed"]]
    missing = [r["cell"] for r in rows if r["kernel_changed"] is None]
    failed = [r["cell"] for r in rows if r["forced_exit"] not in (0, None)]
    print(f"cells probed          : {len(rows)}")
    print(f"kernel changed        : {len(changed)}  {changed}")
    print(f"kernel identical      : {len(rows) - len(changed) - len(missing)}")
    print(f"not captured          : {len(missing)}  {missing}")
    print(f"forced-arm failures   : {len(failed)}  {failed}")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({
                "harness": "arch-probe",
                "real_arch": "applegpu_g16s",
                "forced_arch": args.forced,
                "rows": rows,
                "cells_changed": changed,
                "cells_not_captured": missing,
                "forced_arm_failures": failed,
            }, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
