#!/usr/bin/env python3
"""E125: emit the E123 arm set from a PINNED revision of the generated twins.

    research/e125_arms.py --emit /tmp/e125-arms
    research/e125_arms.py --arm-list

E125 measures a frame axis, not a kernel. Every number it produces is a
within-session contrast between two frames running the SAME arm, so the arm set
only has to be internally consistent and comparable with the session E125
extends. That session is E123, and its 20 already-analysed cells, its
per-instruction prices and its entry-point residency census are all on the
generated twins as they stood before E121 rung 3 transplanted the gated
chunk-sum share into them.

So the twins are pinned, and the pin is stated rather than implied.

TWO REASONS, AND THE SECOND ONE IS A DEFECT REPORT.

1. Comparability. E125's Stage 0 evidence -- the per-instruction price at
   NA=2, 3 and 4, and the finding that residency and roofline distance are
   perfectly rank collinear over that width sweep -- is measured on this
   kernel. A frame axis measured on a different kernel could not be joined to
   it.

2. `research/e123_arms.py` DOES NOT BUILD against the current base. E121 rung 3
   added `uint simd_gid` and `threadgroup float* sums_xchg` to the shipped
   `qmv_fast_crossrow_affine4_g64_wide`, so the shipped definition now takes 12
   parameters while the emitter's per-width entry points still call it with 10.
   `a_base`, whose plan is `None` and which is therefore the shipped function
   verbatim, fails to compile. Every arm that substitutes its own prologue
   still compiles, so the failure is silent until the probe runs. That is a
   real defect in a merged research instrument and it is reported here rather
   than worked around quietly. Repairing it means transcribing the new body,
   the new prologue and the new epilogue into a new plan set, and re-proving
   `q_scaffold`'s byte identity against them. That is a separate piece of work:
   `n_nosums` deletes the same activation add tree that E121 rung 3 now shares
   across simdgroups, so the deletion means a different thing on the new body
   and it must be measured, not assumed.

TRANSFER CAVEAT, stated once and carried into the report. A frame effect
measured here is measured on the pre-E121 wide QMV. It transfers to the current
base as a RATIO between frames, which is what the correction needs, and it does
not transfer as an absolute per-instruction price. Any claim that needs the
absolute price on the current base needs a replay on the current base.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import e118_arms as e118  # noqa: E402
import e123_arms as e123  # noqa: E402
from e104_variant_sources import CELL, widen_asserts  # noqa: E402

# The commit before `5d97175c`, "e121 rung 3: transplant the gated chunk-sum
# share into both quantized twins". Pinned as a full ref rather than a relative
# one so the pin cannot move when history is rewritten around it.
DEFAULT_REV = "5d97175c~1"

# Two instruction classes far apart on the E123 price table, plus the real
# deletion the frame axis is about, plus the byte-identical null that sets the
# session floor.
#
#   n_nosums    arithmetic deletion, the whole activation add tree
#   k_ld8/16    device-load injection, two rungs so the price is a contrast
#   k_alu8/16   ALU injection, two rungs, same
#   q_scaffold  byte identical to `a_base`, so its frame response is the
#               instrument's own noise in every frame
ARMS = ("a_base", "q_scaffold", "n_nosums", "k_ld8", "k_ld16", "k_alu8",
        "k_alu16")


def base_source(outdir: pathlib.Path, rev: str) -> str:
    path = outdir / "base_raw.metal"
    subprocess.run(
        [sys.executable, str(ROOT / "research/jit_string_compile.py"),
         "--emit", str(path), "--rev", rev, "--", CELL],
        check=True, cwd=str(ROOT))
    return path.read_text()


def emit(outdir: pathlib.Path, rev: str) -> None:
    e123.install()
    outdir.mkdir(parents=True, exist_ok=True)
    base = widen_asserts(base_source(outdir, rev))
    (outdir / "base_lone.metal").write_text(base)
    print("e125_arms: twins pinned at %s  base=%d bytes  sha=%s"
          % (rev, len(base), hashlib.sha256(base.encode()).hexdigest()[:12]))
    seen: dict[str, str] = {}
    for arm in ARMS:
        text = e118.arm_source(base, arm)
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        if digest in seen:
            raise SystemExit(
                "e125_arms: %s and %s are byte-identical" % (arm, seen[digest]))
        seen[digest] = arm
        (outdir / ("arm_%s.metal" % arm)).write_text(text)
        print("%-15s %8d bytes  sha=%s  exact=%s"
              % (arm, len(text), digest, arm not in e123.DIAGNOSTIC_ARMS))
    print("\n--arms %s" % arm_list())


def arm_list() -> str:
    return ",".join(a + (":diag" if a in e123.DIAGNOSTIC_ARMS else "")
                    for a in ARMS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", type=pathlib.Path)
    parser.add_argument("--rev", default=DEFAULT_REV)
    parser.add_argument("--arm-list", action="store_true")
    args = parser.parse_args()
    if args.arm_list:
        print(arm_list())
        return 0
    if not args.emit:
        parser.error("one of --emit or --arm-list is required")
    emit(args.emit, args.rev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
