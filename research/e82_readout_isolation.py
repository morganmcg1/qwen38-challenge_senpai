#!/usr/bin/env python3
"""Prove that `master-bf16` differs from `pinned` in the readout and nowhere else.

The row-level result in `e82_arm_identity.py` only means something if the two
arms really are a controlled pair. This script hashes every shared tensor and
asserts that the sole difference is the presence of the two-bit
`draft_lm_head`, which `master-bf16` copies verbatim from the declared head.

If every shared tensor matches, then the identical draft streams isolate one
variable: the readout precision.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from e82_st import SafeTensors

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1"))
ARMS = {
    "pinned": CACHE / "mtp-head/model.safetensors",
    "declared": CACHE / "mtp-head-declared-run/model.safetensors",
    "master-bf16": CACHE / "e82/built/e82-master-bf16-run/model.safetensors",
}
READOUT = ("draft_lm_head.weight", "draft_lm_head.scales", "draft_lm_head.biases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e82-readout-isolation.json")
    args = ap.parse_args()

    st = {arm: SafeTensors(p) for arm, p in ARMS.items()}
    trunk = sorted(set(st["pinned"].names()))
    shared = sorted(set(st["master-bf16"].names()) & set(trunk))

    rows, mismatched = [], []
    for name in shared:
        a, b = st["pinned"].sha256(name), st["master-bf16"].sha256(name)
        rows.append({"tensor": name, "pinned_sha256": a,
                     "master_bf16_sha256": b, "identical": a == b})
        if a != b:
            mismatched.append(name)

    readout = []
    for name in READOUT:
        a = st["declared"].sha256(name)
        b = st["master-bf16"].sha256(name)
        readout.append({"tensor": name, "declared_sha256": a,
                        "master_bf16_sha256": b, "identical": a == b})

    extra = sorted(set(st["master-bf16"].names()) - set(trunk))
    verdict = (not mismatched
               and all(r["identical"] for r in readout)
               and extra == sorted(READOUT))
    report = {
        "shared_tensors": len(shared),
        "shared_tensors_identical": len(shared) - len(mismatched),
        "mismatched": mismatched,
        "master_bf16_extra_tensors": extra,
        "readout_matches_declared": readout,
        "controlled_pair": verdict,
        "trunk_sha256": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"pinned tensors: {len(trunk)}   master-bf16 tensors: "
          f"{len(st['master-bf16'].names())}")
    print(f"shared tensors byte-identical: {len(shared) - len(mismatched)}"
          f" / {len(shared)}")
    print(f"master-bf16 extra tensors: {extra}")
    for r in readout:
        print(f"  {r['tensor']:<26} matches declared: {r['identical']}")
    print(f"\nCONTROLLED PAIR: {'PASS' if verdict else 'FAIL'}"
          "  (only the readout differs)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
