#!/usr/bin/env python3
"""E82: does every built head keep `K=all, V=all` as a full permutation?

E84 exploits a property of the pinned head that the manifest's selection rule
implies but does not state: `precision_islands.k.indices` and
`precision_islands.v.indices` are each a complete permutation of 0 ... 1023,
so every K and every V output row has a BF16 island row and the quantized K
and V rows need not be computed at all. That mechanism is worth a
ranked-measured -0.172 %, and it silently disappears if a new head emits a
truncated or reordered index list of a different length.

The E82 builds recompute islands against their own requantized trunk. This
checks the property directly on every artifact rather than trusting the build
script, and reports Q too, which is a genuine top-k subset by design.

  python3 research/e82_island_permutation.py --out research/e82-island-permutation.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from e82_st import SafeTensors

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
ARMS = {
    "declared": CACHE / "mtp-head-declared-run/model.safetensors",
    "soup-q4": CACHE / "e82/built/e82-soup-q4-run/model.safetensors",
    "qat-q4": CACHE / "e82/built/e82-qat-q4-run/model.safetensors",
}
# Output rows per projection: 24 q heads x 256, 4 kv heads x 256.
ROWS = {"q": 6144, "k": 1024, "v": 1024}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e82-island-permutation.json")
    args = ap.parse_args()

    report = {}
    print("arm".ljust(12) + "proj  count  rows  unique  min  max  full permutation  sorted")
    for arm, path in ARMS.items():
        st = SafeTensors(path)
        entry = {}
        for proj, rows in ROWS.items():
            key = f"precision_islands.{proj}.indices"
            if key not in st:
                entry[proj] = {"present": False}
                print(f"{arm.ljust(12)}{proj:>4}  absent")
                continue
            idx = np.asarray(st.array(key)).ravel()
            uniq = np.unique(idx)
            full = bool(idx.size == rows and uniq.size == rows
                        and uniq[0] == 0 and uniq[-1] == rows - 1)
            entry[proj] = {
                "present": True,
                "count": int(idx.size),
                "output_rows": rows,
                "unique": int(uniq.size),
                "min": int(idx.min()),
                "max": int(idx.max()),
                "is_full_permutation": full,
                "is_sorted": bool(np.all(np.diff(idx) > 0)),
                "weight_shape": list(st.entries[f"precision_islands.{proj}.weight"].shape),
            }
            print(f"{arm.ljust(12)}{proj:>4}{idx.size:7d}{rows:6d}{uniq.size:8d}"
                  f"{idx.min():5d}{idx.max():6d}{str(full):>18}{str(entry[proj]['is_sorted']):>8}")
        report[arm] = entry

    same = {
        proj: len({tuple(np.asarray(SafeTensors(p).array(
            f"precision_islands.{proj}.indices")).ravel().tolist()) for p in ARMS.values()})
        for proj in ROWS
    }
    report["distinct_index_lists_across_arms"] = same
    print("\ndistinct index lists across arms (1 = every arm selected the same rows):")
    print(f"  {same}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
