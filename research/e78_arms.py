#!/usr/bin/env python3
"""E78 arm definitions: should the QMV inner-group count depend on `out_vec_size`?

The shipped dispatch picks ONE inner-group count `IPG` per width `M` and applies
it to every quantized linear in the round. E33 measured a sign flip across the
scored shapes at M=6: the three n=5120 families are FASTER at two groups, and
`lm_head` and `mlp.gate_up_fused` are faster at one. E74 located the in-situ
knee at 1558 working threadgroups. A dispatch that conditions `IPG` on
`out_vec_size` can serve both ends of that range.

`out_vec_size` is a uniform scalar kernel argument, so the added branch is
uniform across the threadgroup.

Arms, all applied to BOTH scored twins from the pinned base text:

  a_ship          the base table: case 5 -> <T,5,5>, case 6 -> <T,6,6>,
                  case 9 -> <T,9,5>. No source change at all.
  b_crown         case 5, 6 and 9 -> IPG 3, the crown's table, with the base's
                  `NA <= 6` wide-helper bound LEFT IN PLACE. No NA > 4
                  instantiation survives, so the bound should be inert.
  b_crown_exact   the crown's exact bytes, checked out from the promoted source
                  ref. Identical to `b_crown` except that the wide-helper bound
                  reads `NA <= 4`. Built only to test that inertness claim by
                  comparing built-worker `__TEXT` digests.
  c_hybrid24928   `IPG` conditioned on `out_vec_size` at 24928. Below the cutoff
                  cases 5, 6 and 9 take the crown's IPG 3; at or above it they
                  take the base's IPG. 24928 is E74's predicted 40-core
                  starvation boundary, and it splits n = 5120, 14336 and 16480
                  from n = 34816, 98336 and 248320.
  d_hybrid8192    the same structure with the cutoff at 8192, which splits only
                  the three n = 5120 families.
  c_perturb       positive control for the rung 1 exactness instrument: arm C
                  with input rows 3 and 4 written to each other's accumulator
                  lane in every NA=5 group. The row ledger MUST differ from
                  arm C. Never timed and never submitted.

Arms are STATES, not patches. Every arm is built from the pinned base text read
out of Git, so no arm can stack on another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
SOURCES = (HEADER, TWIN)

BASE_SHA = "8d938c911df52b6a324f259a55dbaa75e508c822"
# The promoted crown, senpai/frontier-state.json: submission 9ad17378, score
# 3.25238228. Present in this checkout, so arm `b_crown_exact` is a checkout,
# never a reconstruction.
CROWN_SHA = "bfab0de58d43453e506523707e1720a3485570f4"

CUTOFF_NAME = "kQmvNarrowOutVecCutoff"
IN_CUTOFF_NAME = "kQmvSplitInVecCutoff"

LANE_WRITE = """        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];"""
LANE_WRITE_PERTURBED = """        const int lane = (NA == 5) ? ((m == 3) ? 4 : ((m == 4) ? 3 : m)) : m;
        a0[lane] = xc[0];
        a1[lane] = xc[1];
        a2[lane] = xc[2];
        a3[lane] = xc[3];"""

# The single wide-branch dispatch for width M. The narrow band below 4096
# outputs calls `qmv_fast_crossrow_affine4_g64<T, M>`, which has no `_m` and no
# IPG argument, so this anchor cannot reach it.
CALL_RE = (
    r"( *)qmv_fast_crossrow_affine4_g64_m<T, %d, \d+, true>\(\n"
    r" *w, scales, biases, x, y, in_vec_size, out_vec_size,\n"
    r" *tid, simd_gid, simd_lid\);"
)

SWITCH_ANCHOR = """      // promoted pair kernel is kept there byte-for-byte.
      switch (ntg.x) {"""

# The patch carries NO comment. `research/twin_audit.py` pins its quantized.h
# waiver to the sha256 of that section's COMMENT text on both sides, so adding
# even one comment line de-pins a pre-existing, unrelated case-8 waiver and
# re-reds the promotion gate. The constant's name carries the meaning instead,
# and the reasoning lives in this module, the E78 report and the ledger.
CUTOFF_BLOCK = """      // promoted pair kernel is kept there byte-for-byte.
      constexpr int %s = %d;
      switch (ntg.x) {"""


def call_block(indent: str, m: int, ipg: int) -> str:
    return (
        "%sqmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>(\n"
        "%s    w, scales, biases, x, y, in_vec_size, out_vec_size,\n"
        "%s    tid, simd_gid, simd_lid);" % (indent, m, ipg, indent, indent)
    )


def swap_ipg(text: str, m: int, ipg: int) -> str:
    pat = re.compile(CALL_RE % m)
    hits = pat.findall(text)
    if len(hits) != 1:
        raise SystemExit("e78_arms: M=%d dispatch anchor matched %d times"
                         % (m, len(hits)))
    return pat.sub(lambda mo: call_block(mo.group(1), m, ipg), text, count=1)


def split_ipg(text: str, m: int, wide_ipg: int, narrow_ipg: int,
              var: str = "out_vec_size", name: str = CUTOFF_NAME) -> str:
    pat = re.compile(CALL_RE % m)
    hits = pat.findall(text)
    if len(hits) != 1:
        raise SystemExit("e78_arms: M=%d dispatch anchor matched %d times"
                         % (m, len(hits)))

    def repl(mo: re.Match) -> str:
        indent = mo.group(1)
        inner = indent + "  "
        return (
            "%sif (%s >= %s) {\n%s\n%s} else {\n%s\n%s}"
            % (indent, var, name,
               call_block(inner, m, wide_ipg), indent,
               call_block(inner, m, narrow_ipg), indent)
        )

    return pat.sub(repl, text, count=1)


def set_cutoff(text: str, cutoff: int, name: str = CUTOFF_NAME) -> str:
    if text.count(SWITCH_ANCHOR) != 1:
        raise SystemExit("e78_arms: wide-branch switch anchor not unique")
    return text.replace(SWITCH_ANCHOR, CUTOFF_BLOCK % (name, cutoff))


def perturb_lanes(text: str) -> str:
    if text.count(LANE_WRITE) != 1:
        raise SystemExit("e78_arms: lane-write anchor not unique")
    return text.replace(LANE_WRITE, LANE_WRITE_PERTURBED)


# Base table for the three cells this experiment moves.
BASE_CELLS = {5: 5, 6: 6, 9: 5}
CROWN_CELLS = {5: 3, 6: 3, 9: 3}

ARMS: dict[str, dict] = {
    "a_ship": {
        "doc": "the base table, unmodified: <T,5,5>, <T,6,6>, <T,9,5>",
        "cells": dict(BASE_CELLS),
        "cutoff": None,
    },
    "b_crown": {
        "doc": "the crown's table by patch: IPG 3 at M=5, 6 and 9, wide-helper "
               "bound left at NA <= 6",
        "cells": dict(CROWN_CELLS),
        "cutoff": None,
    },
    "b_crown_exact": {
        "doc": "the crown's exact bytes from %s; wide-helper bound NA <= 4"
               % CROWN_SHA[:8],
        "cells": dict(CROWN_CELLS),
        "cutoff": None,
        "checkout": CROWN_SHA,
        "never_submit": True,
    },
    "c_hybrid24928": {
        "doc": "IPG conditioned on out_vec_size at 24928: crown IPG below, base "
               "IPG at or above",
        "cells": dict(BASE_CELLS),
        "narrow_cells": dict(CROWN_CELLS),
        "cutoff": 24928,
    },
    "d_hybrid8192": {
        "doc": "IPG conditioned on out_vec_size at 8192: crown IPG below, base "
               "IPG at or above",
        "cells": dict(BASE_CELLS),
        "narrow_cells": dict(CROWN_CELLS),
        "cutoff": 8192,
    },
    "e_kdown": {
        "doc": "IPG conditioned on in_vec_size at 8192, at M=6 only: mlp.down "
               "takes IPG 3, every other shape keeps the base IPG 6. M=5 and "
               "M=9 are untouched.",
        "cells": {6: 3},
        "narrow_cells": {6: 6},
        "cutoff": 8192,
        "cutoff_var": "in_vec_size",
        "cutoff_name": IN_CUTOFF_NAME,
    },
    "c_perturb": {
        "doc": "positive control: arm C with rows 3 and 4 swapped between "
               "accumulator lanes in every NA=5 group",
        "cells": dict(BASE_CELLS),
        "narrow_cells": dict(CROWN_CELLS),
        "cutoff": 24928,
        "perturb": True,
        "never_time": True,
        "never_submit": True,
    },
}


def base_text(path: str, rev: str) -> str:
    return subprocess.run(["git", "show", "%s:%s" % (rev, path)],
                          cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def apply_arm(path: str, name: str) -> str:
    if name not in ARMS:
        raise SystemExit("e78_arms: unknown arm %s" % name)
    arm = ARMS[name]
    if arm.get("checkout"):
        return base_text(path, arm["checkout"])
    text = base_text(path, BASE_SHA)
    if arm["cutoff"] is not None:
        var = arm.get("cutoff_var", "out_vec_size")
        name = arm.get("cutoff_name", CUTOFF_NAME)
        text = set_cutoff(text, arm["cutoff"], name)
        for m, wide in sorted(arm["cells"].items()):
            text = split_ipg(text, m, wide, arm["narrow_cells"][m], var, name)
    else:
        for m, ipg in sorted(arm["cells"].items()):
            text = swap_ipg(text, m, ipg)
    if arm.get("perturb"):
        text = perturb_lanes(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="write one E78 arm into the worktree")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--out", help="write the arm manifest here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    digests, sizes = {}, {}
    for rel in SOURCES:
        text = apply_arm(rel, args.arm)
        if not args.dry_run:
            (REPO / rel).write_text(text)
        digests[rel] = hashlib.sha256(text.encode()).hexdigest()
        sizes[rel] = len(text.encode())
    spec = ARMS[args.arm]
    payload = {
        "arm": args.arm,
        "doc": spec["doc"],
        "base_sha": BASE_SHA,
        "cells": spec["cells"],
        "narrow_cells": spec.get("narrow_cells"),
        "cutoff": spec["cutoff"],
        "checkout": spec.get("checkout"),
        "dry_run": bool(args.dry_run),
        "never_submit": bool(spec.get("never_submit")),
        "never_time": bool(spec.get("never_time")),
        "source_bytes": sizes,
        "sha256": digests,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
