#!/usr/bin/env python3
"""E102 rung 2: enumerate every `affine_qmv_fast` instantiation and decide,
from live source, whether the scored worker can reach it.

Two predicates select a case in the shipped dispatcher
(`Vendor/.../kernels/quantized.h`, `affine_qmv_fast`):

  * `out_vec_size`, which picks the wide (>= 4096) or narrow ([1024, 4096))
    switch. `out_vec_size` is the projection width `N`.
  * `ntg.x`, declared `[[threadgroups_per_grid]]` at quantized.h:1886. The
    launcher sets `grid_dims(M, (N + bn - 1) / bn, B)` at
    `backend/metal/quantized.cpp:254`, so `ntg.x == M`, the verify row count.

Both bounds come from source, not from a measurement:

  * `M = draft_count + 1`, and the session caps the draft count at
    `min(offeredDepth, Qwen36MTPLimits.maxDepth, segmentedVerifyDepthCap)`
    (`Qwen36MTPBlockSession.swift:1073`), with
    `segmentedVerifyDepthCap = 7` (line 1007) and
    `qwenMTPMaxDraftDepth = 8` (`MLXFastCore/Constants.swift:331`).
    So `M <= 8` and `case 9:` cannot fire.
  * The scored quantised projection widths are pinned below. The smallest is
    5120, so no scored dispatch enters the narrow switch at all.

This script only reports. `quantized.h` and its generated twin belong to
another student's assignment; E102 must not edit them.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
KERNEL = REPO / ("Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/"
                 "kernels/quantized.h")
SESSION = REPO / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
CONSTANTS = REPO / "Sources/MLXFastCore/Constants.swift"

# Scored 4-bit group-64 projection widths (`out_vec_size`), with the
# experiment that captured each one in situ.
SCORED_WIDTHS = {
    248320: ("head.lm_head", "e70 2.2, e74 rung 3"),
    98336: ("head.compact_draft_vocab (2-bit, separate entry)", "e25, e74"),
    34816: ("mlp.gate_up fused", "e70 2.2, e74 rung 3"),
    16480: ("gdn.in_proj fused", "e70 2.2, e74 rung 3"),
    14336: ("fa.qkv fused", "e70 2.2, e74 rung 3"),
    5120: ("fa.o_proj, gdn.out_proj, mlp.down", "e70 2.2, e74 rung 3"),
}

CASE_RE = re.compile(
    r"case (\d+):\s*\n\s*(?://[^\n]*\n\s*)*"
    r"(qmv_fast_crossrow_affine4_g64(?:_m)?<[^>]*>)",
    re.MULTILINE)


def read_int(path: pathlib.Path, pattern: str) -> int:
    m = re.search(pattern, path.read_text())
    if not m:
        raise SystemExit(f"e102: {pattern!r} not found in {path.name}")
    return int(m.group(1))


def parse_switches(text: str) -> dict[str, dict[int, str]]:
    """Split the two `switch (ntg.x)` bodies of `affine_qmv_fast`."""
    start = text.index("[[kernel]] void affine_qmv_fast(")
    body = text[start:text.index("\ntemplate ", start)]
    heads = [m.start() for m in re.finditer(r"switch \(ntg\.x\)", body)]
    if len(heads) != 2:
        raise SystemExit(f"e102: expected 2 ntg.x switches, found {len(heads)}")
    wide, narrow = body[heads[0]:heads[1]], body[heads[1]:]
    out = {}
    for name, chunk in (("wide_ge_4096", wide), ("narrow_1024_4095", narrow)):
        out[name] = {int(m.group(1)): m.group(2)
                     for m in CASE_RE.finditer(chunk)}
    return out


def main() -> int:
    max_depth = read_int(CONSTANTS,
                         r"qwenMTPMaxDraftDepth = (\d+)")
    verify_cap = read_int(SESSION,
                          r"segmentedVerifyDepthCap = (\d+)")
    max_m = min(max_depth, verify_cap) + 1
    min_scored = min(SCORED_WIDTHS)
    narrow_live = min_scored < 4096

    switches = parse_switches(KERNEL.read_text())
    wide, narrow = switches["wide_ge_4096"], switches["narrow_1024_4095"]
    live_calls = {call for m, call in wide.items() if m <= max_m}

    rows = []
    for branch, cases in switches.items():
        wide_branch = branch == "wide_ge_4096"
        for m in sorted(cases):
            call = cases[m]
            if wide_branch:
                reach = m <= max_m
                why = ("M reachable" if reach else
                       f"M = {m} > max verify rows {max_m}")
            elif not narrow_live:
                # A dead branch can still hold a live instantiation if the
                # reachable wide switch emits the identical specialisation.
                reach = False
                why = (f"branch dead: smallest scored out_vec_size "
                       f"{min_scored} >= 4096")
                if call in live_calls:
                    why += "; instantiation kept alive by the wide switch"
            else:
                reach = m <= max_m
                why = "narrow branch live"
            rows.append({"branch": branch, "M": m, "call": call,
                         "reachable": reach, "reason": why,
                         "instantiation_also_live_elsewhere":
                             (not wide_branch) and call in live_calls})

    dead_inst = sorted({r["call"] for r in rows
                        if not r["reachable"]
                        and not r["instantiation_also_live_elsewhere"]})
    result = {
        "max_draft_depth": max_depth,
        "segmented_verify_depth_cap": verify_cap,
        "max_verify_rows_M": max_m,
        "smallest_scored_out_vec_size": min_scored,
        "narrow_branch_reachable": narrow_live,
        "scored_widths": {str(k): v[0] for k, v in SCORED_WIDTHS.items()},
        "cases": rows,
        "dead_instantiations": dead_inst,
    }

    print(f"max verify rows M = min({max_depth}, {verify_cap}) + 1 = {max_m}")
    print(f"smallest scored out_vec_size = {min_scored} -> narrow branch "
          f"{'LIVE' if narrow_live else 'DEAD'}\n")
    print(f"{'branch':<18}{'M':>3}  {'reach':<6}{'call':<52}reason")
    for r in rows:
        print(f"{r['branch']:<18}{r['M']:>3}  "
              f"{'yes' if r['reachable'] else 'NO':<6}{r['call']:<52}"
              f"{r['reason']}")
    print(f"\ndead instantiations ({len(dead_inst)}):")
    for call in dead_inst:
        print(f"  {call}")

    if len(sys.argv) > 1:
        out = pathlib.Path(sys.argv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
