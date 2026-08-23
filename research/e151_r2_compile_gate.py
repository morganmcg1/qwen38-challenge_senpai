#!/usr/bin/env python3
"""E151 rung R2: offline compile gate for the SHIPPED double-buffered NAX k-loop.

Same constraint as R1, and the same reason this file exists. `is_nax_available()`
is false on every Mac this campaign owns (Finding 250), so `affine_qmm_t_nax`
never executes here, no local run of any length touches one instruction of the
pipelined loop, and `metal-tt` refuses this kernel family at every shape. A
Metal compile of the exact JIT concatenation is the only executable check
available before the ranked receipt.

WHAT THIS GATE DECIDES, WITH ZERO GPU:

  1. The tree ships the R2 arm ON, read from the concatenated twin that MLX
     actually JIT-compiles, not from the readable header.
  2. All four arm combinations compile: base, R1 alone, R2 alone, R1 composed
     with R2. F3 asks for a separate ranked read of each of the last three, so
     each must be a buildable configuration, not a hypothetical.
  3. Each arm is the whole switch for itself: flipping one flag moves the AIR
     digest and flipping it back restores it.
  4. The capacity predicate carries load rather than decorating the code.
     Defeat it and the `float` group-32 cell, which needs 34816 B of a 32768 B
     threadgroup budget, must be REFUSED by name. Rule 101: a guard that cannot
     fail is not a guard.
  5. The capacity predicate is not silently too strict. Shrink the limit and
     the scored cell must be REFUSED by the arm-liveness assert, which proves
     that assert would catch a predicate that disarmed the scored tile.
  6. Containment: `kDoubleBuffer` defaults to false, so every other call site of
     `qmm_t_nax_tgp_impl` must be byte-identical to R1. This is checked by AIR
     digest, not asserted.
  7. The residue. `affine_qmm_t_nax` cells are NOT byte-identical between R1
     and R2-disarmed; they differ by a few tens of AIR bytes. That is measured
     and attributed rather than hidden, with two competing explanations each
     compiled as its own control:
       - a comment-only line-shift control, which FAILED to move the digest and
         therefore refutes the source-position explanation;
       - a template-parameter-only control, which adds `kDoubleBuffer` to R1 and
         changes nothing else, isolating the mangled-name explanation.
     Campaign calibration `transfer_air_overstates_isa_by = 181.3` is the right
     lens on the magnitude: 32 AIR bytes is about 0.18 ISA bytes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e147_rungE1c as e1c  # noqa: E402

ROOT = e1c.ROOT
OUT = ROOT / "research/e151-r2-compile-gate.json"

HEADER = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp"

R1_FLAG_ON = "constexpr bool kE147NaxRetileOn = true;"
R1_FLAG_OFF = "constexpr bool kE147NaxRetileOn = false;"
R2_FLAG_ON = "constexpr bool kE151NaxDoubleBufferOn = true;"
R2_FLAG_OFF = "constexpr bool kE151NaxDoubleBufferOn = false;"

CAPACITY_GUARD = "kE151NaxDoubleBufferOn && kE151DoubleBufferFits;"
CAPACITY_GUARD_DEFEATED = "kE151NaxDoubleBufferOn;"

TGP_LIMIT = "constexpr int kE151MaxTgpBytes = 32768;"
TGP_LIMIT_TOO_SMALL = "constexpr int kE151MaxTgpBytes = 4096;"

CAPACITY_NEEDLE = "does not fit in threadgroup memory"
ARM_LIVENESS_NEEDLE = "must take the double-buffer arm"

# The cell that cannot hold two 64-wide staging halves: float, BK_padded 68, so
# 2 * 64 * 68 * 4 = 34816 B against a 32768 B budget. It is a real ahead-of-time
# cell -- `quantized_nax.metal` instantiates float at group sizes 128, 64, 32
# and bits 2, 3, 4, 5, 6, 8 -- and group 32 is exactly where R1's loader guard
# forces the tile back to 64 wide.
FLOAT_G32_INST = "affine_qmm_t_nax<float, 32, 4, 1, 0, 64, 64, 64, 2, 2>"
BF16_G32_INST = "affine_qmm_t_nax<bfloat16_t, 32, 4, 1, 0, 64, 64, 64, 2, 2>"

UNTOUCHED_SITES = {
    "untouched_qmm_n": "affine_qmm_n_nax<bfloat16_t, 64, 4, 1, 64, 64, 64, 2, 2>",
    "untouched_gather_rhs": (
        "affine_gather_qmm_rhs_nax<bfloat16_t, 64, 4, 64, 64, 64, 2, 2, true>"
    ),
}

HOST_NAME = "e151_r2_scored"


def rev_parse(rev: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", rev], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def line_shift_control(src: str, added_lines: int) -> str:
    """R1 source plus `added_lines` comment-only lines at the same place R2 grew.

    If this moves the AIR digest, the R2-disarmed residue is a source-position
    artifact and not emitted work. The control can fail: if a pure comment block
    leaves the digest alone, the residue has to be explained some other way.
    It DID fail, which is why `template_param_control` below exists.
    """
    filler = "".join(
        f"  // e151 line-shift control {i}\n" for i in range(added_lines)
    )
    return e1c.substitute(src, R1_FLAG_ON, filler + R1_FLAG_ON, "line_shift_control")


TEMPLATE_HEAD = """    const int kHostBM = BM,
    const int kHostBN = BN>
METAL_FUNC void qmm_t_nax_tgp_impl("""

TEMPLATE_HEAD_WITH_PARAM = """    const int kHostBM = BM,
    const int kHostBN = BN,
    const bool kDoubleBuffer = false>
METAL_FUNC void qmm_t_nax_tgp_impl("""


def template_param_control(src: str) -> str:
    """R1 source plus ONE defaulted template parameter, and nothing else.

    No new loop, no new constant, no changed call site: `kDoubleBuffer` is
    declared, defaulted and never read. Every instantiation still selects the
    same code, but its mangled name now carries one more template argument.
    If this reproduces the R2-disarmed residue, the residue is a name-mangling
    artifact of the AIR module and not emitted work.
    """
    return e1c.substitute(
        src, TEMPLATE_HEAD, TEMPLATE_HEAD_WITH_PARAM, "template_param_control"
    )


def tgp_table() -> list[dict]:
    """The threadgroup accounting the arm predicate implements, recomputed here.

    Recorded so the report quotes a machine-computed table rather than my
    arithmetic, and so a later change to BK or the arm BN shows up as a diff.
    """
    rows = []
    for name, size in (("float", 4), ("float16_t", 2), ("bfloat16_t", 2)):
        bk_padded = 64 + 16 // size
        for retiled in (True, False):
            staged_bn = 32 if retiled else 64
            ws_elems = 64 * bk_padded
            db_elems = 2 * staged_bn * bk_padded
            alloc = max(db_elems, ws_elems)
            fits = alloc * size <= 32768
            rows.append(
                {
                    "T": name,
                    "retile": "on" if retiled else "off",
                    "staged_BN": staged_bn,
                    "BK_padded": bk_padded,
                    "single_buffer_bytes": ws_elems * size,
                    "double_buffer_bytes": alloc * size,
                    "delta_bytes": (alloc - ws_elems) * size if fits else 0,
                    "limit_bytes": 32768,
                    "db_armed": fits,
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1-rev", required=True, help="the R1 tip, R2 arm absent")
    ap.add_argument("--base", required=True, help="the assignment base, both arms off")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tree = e1c.preamble(e1c.worktree, e1c.NAX_TWINS)
    r1 = e1c.preamble(lambda p: e1c.show(args.r1_rev, p), e1c.NAX_TWINS)
    base = e1c.preamble(lambda p: e1c.show(args.base, p), e1c.NAX_TWINS)

    if R2_FLAG_ON not in tree:
        raise SystemExit("e151_r2_compile_gate: the tree twin does not ship the R2 arm on")
    if R2_FLAG_OFF in tree:
        raise SystemExit("e151_r2_compile_gate: the tree twin still carries an R2 arm-off flag")
    if R2_FLAG_ON in r1 or R2_FLAG_OFF in r1:
        raise SystemExit("e151_r2_compile_gate: the R1 revision already carries an R2 flag")
    if R1_FLAG_ON not in tree:
        raise SystemExit("e151_r2_compile_gate: the tree twin lost the R1 arm")

    r2_off = e1c.substitute(tree, R2_FLAG_ON, R2_FLAG_OFF, "r2_forced_off")
    r1_off = e1c.substitute(tree, R1_FLAG_ON, R1_FLAG_OFF, "r1_forced_off")
    both_off = e1c.substitute(r1_off, R2_FLAG_ON, R2_FLAG_OFF, "both_forced_off")
    capacity_defeated = e1c.substitute(
        tree, CAPACITY_GUARD, CAPACITY_GUARD_DEFEATED, "capacity_guard_defeated"
    )
    capacity_too_strict = e1c.substitute(
        tree, TGP_LIMIT, TGP_LIMIT_TOO_SMALL, "capacity_too_strict"
    )

    header_added_lines = len(e1c.worktree(HEADER).splitlines()) - len(
        e1c.show(args.r1_rev, HEADER).splitlines()
    )
    shifted_r1 = line_shift_control(r1, header_added_lines)
    param_only_r1 = template_param_control(r1)

    probes = {
        # the four arm combinations, scored cell
        "base_both_off": (base, e1c.SCORED_INST),
        "tree_both_off": (both_off, e1c.SCORED_INST),
        "r1_alone_rev": (r1, e1c.SCORED_INST),
        "r1_alone_tree": (r2_off, e1c.SCORED_INST),
        "r2_alone": (r1_off, e1c.SCORED_INST),
        "composed": (tree, e1c.SCORED_INST),
        # the capacity predicate, on the cell that cannot hold two halves
        "float_g32_shipped": (tree, FLOAT_G32_INST),
        "float_g32_capacity_defeated": (capacity_defeated, FLOAT_G32_INST),
        "float_g32_r1_alone": (r2_off, FLOAT_G32_INST),
        # the arm-liveness assert, exercised where it can still fire
        "scored_capacity_too_strict": (capacity_too_strict, e1c.SCORED_INST),
        # the group-32 cell R1's loader guard forces back to 64 wide
        "bf16_g32_shipped": (tree, BF16_G32_INST),
        "bf16_g32_r1_alone": (r2_off, BF16_G32_INST),
        "bf16_g32_r1_rev": (r1, BF16_G32_INST),
        # attribution for the disarmed residue: two competing explanations,
        # both compiled rather than argued
        "r1_line_shifted": (shifted_r1, e1c.SCORED_INST),
        "r1_template_param_only": (param_only_r1, e1c.SCORED_INST),
    }
    for label, inst in UNTOUCHED_SITES.items():
        probes[f"{label}_r1"] = (r1, inst)
        probes[f"{label}_tree"] = (tree, inst)

    result = {
        "experiment": "E151",
        "rung": "R2",
        "harness": "offline",
        "base": rev_parse(args.base),
        "r1_rev": rev_parse(args.r1_rev),
        "tree": rev_parse("HEAD"),
        "scored_instantiation": e1c.SCORED_INST,
        "jit_concatenation_order": e1c.NAX_TWINS,
        "header_added_lines_vs_r1": header_added_lines,
        "e151_r2_tgp_table": tgp_table(),
        "probes": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp) / "probe"
        for label, (src, inst) in probes.items():
            result["probes"][label] = e1c.compile_probe(src, inst, HOST_NAME, workdir)

    p = result["probes"]

    def digest(label):
        rec = p[label]
        return rec["air_sha256"] if rec["compiled"] else None

    def air(label):
        return p[label].get("air_bytes")

    result["e151_r2_tree_ships_db_on"] = True
    result["e151_r2_r1_rev_has_no_db_arm"] = True

    # 2. every configuration F3 wants a separate receipt for is buildable
    result["e151_r2_base_compiles"] = p["base_both_off"]["compiled"]
    result["e151_r2_r1_alone_compiles"] = p["r1_alone_tree"]["compiled"]
    result["e151_r2_alone_compiles"] = p["r2_alone"]["compiled"]
    result["e151_r2_composed_compiles"] = p["composed"]["compiled"]

    # 3. each flag is the whole switch for itself, and the four cells are distinct
    combos = {
        "both_off": digest("tree_both_off"),
        "r1_alone": digest("r1_alone_tree"),
        "r2_alone": digest("r2_alone"),
        "composed": digest("composed"),
    }
    result["e151_r2_arm_combination_digests"] = combos
    result["e151_r2_four_combinations_distinct"] = (
        all(combos.values()) and len(set(combos.values())) == 4
    )
    result["e151_r2_arm_moves_digest_with_r1_on"] = (
        combos["composed"] is not None
        and combos["r1_alone"] is not None
        and combos["composed"] != combos["r1_alone"]
    )
    result["e151_r2_arm_moves_digest_with_r1_off"] = (
        combos["r2_alone"] is not None
        and combos["both_off"] is not None
        and combos["r2_alone"] != combos["both_off"]
    )

    # 4. the capacity predicate carries load
    result["e151_r2_float_g32_shipped_compiles"] = p["float_g32_shipped"]["compiled"]
    result["e151_r2_capacity_guard_defeated_rejected"] = not p[
        "float_g32_capacity_defeated"
    ]["compiled"]
    result["e151_r2_capacity_assert_named_in_refusal"] = CAPACITY_NEEDLE in p[
        "float_g32_capacity_defeated"
    ].get("error", "")
    # the disarmed float cell must be identical to R1-alone: the predicate has
    # to disarm it completely, not merely make it build
    result["e151_r2_float_g32_matches_r1_alone"] = (
        p["float_g32_shipped"]["compiled"]
        and p["float_g32_r1_alone"]["compiled"]
        and digest("float_g32_shipped") == digest("float_g32_r1_alone")
    )

    # 5. the predicate is not silently too strict
    result["e151_r2_too_strict_limit_rejected"] = not p["scored_capacity_too_strict"][
        "compiled"
    ]
    result["e151_r2_arm_liveness_named_in_refusal"] = ARM_LIVENESS_NEEDLE in p[
        "scored_capacity_too_strict"
    ].get("error", "")

    # 6. containment: default-false keeps every other call site byte-identical
    untouched = {}
    for label in UNTOUCHED_SITES:
        untouched[label] = (
            p[f"{label}_r1"]["compiled"]
            and p[f"{label}_tree"]["compiled"]
            and digest(f"{label}_r1") == digest(f"{label}_tree")
        )
    result["e151_r2_untouched_site_identity"] = untouched
    result["e151_r2_untouched_sites_identical"] = all(untouched.values())

    # 7. the disarmed residue, measured and attributed
    result["e151_r2_armoff_air_delta_bytes"] = {
        "scored": (air("r1_alone_tree") or 0) - (air("r1_alone_rev") or 0),
        "bf16_g32": (air("bf16_g32_r1_alone") or 0) - (air("bf16_g32_r1_rev") or 0),
    }
    result["e151_r2_armoff_matches_r1"] = digest("r1_alone_tree") == digest(
        "r1_alone_rev"
    )
    result["e151_r2_line_shift_moves_digest"] = (
        p["r1_line_shifted"]["compiled"]
        and p["r1_alone_rev"]["compiled"]
        and digest("r1_line_shifted") != digest("r1_alone_rev")
    )
    result["e151_r2_line_shift_air_delta_bytes"] = (
        air("r1_line_shifted") or 0
    ) - (air("r1_alone_rev") or 0)

    # The line-shift control did NOT fire: comment-only lines leave the AIR
    # digest alone, so the residue is not a source-position artifact. The
    # surviving explanation is the mangled name. `kDoubleBuffer` is a template
    # parameter, so every instantiation of `qmm_t_nax_tgp_impl` now carries one
    # more template argument in its symbol even when the value is false and the
    # selected code is identical. This control isolates exactly that: R1's
    # source plus the parameter declaration and nothing else.
    result["e151_r2_template_param_moves_digest"] = (
        p["r1_template_param_only"]["compiled"]
        and p["r1_alone_rev"]["compiled"]
        and digest("r1_template_param_only") != digest("r1_alone_rev")
    )
    result["e151_r2_template_param_air_delta_bytes"] = (
        air("r1_template_param_only") or 0
    ) - (air("r1_alone_rev") or 0)
    # The residue is fully explained if the parameter alone reproduces the
    # disarmed delta exactly. Equality, not a tolerance band: a tolerance would
    # let real emitted work hide inside the allowance.
    result["e151_r2_armoff_residue_is_mangling"] = (
        result["e151_r2_template_param_moves_digest"]
        and result["e151_r2_template_param_air_delta_bytes"]
        == result["e151_r2_armoff_air_delta_bytes"]["scored"]
    )
    result["e151_r2_armoff_residue_unexplained_bytes"] = result[
        "e151_r2_armoff_air_delta_bytes"
    ]["scored"] - result["e151_r2_template_param_air_delta_bytes"]

    result["e151_r2_air_bytes"] = {k: air(k) for k in probes}

    # The twin is the runtime-effective source. Both loader forms must carry the
    # helper: they are separate classes, so an edit to one is invisible to the
    # other and a partial port would compile and then stage into the wrong half.
    twin_text = e1c.worktree(TWIN)
    result["e151_r2_twin_shift_dst_count"] = twin_text.count(
        "void shift_dst(const int delta)"
    )
    result["e151_r2_twin_carries_both_loaders"] = (
        result["e151_r2_twin_shift_dst_count"] == 2
    )

    required = [
        "e151_r2_tree_ships_db_on",
        "e151_r2_r1_rev_has_no_db_arm",
        "e151_r2_base_compiles",
        "e151_r2_r1_alone_compiles",
        "e151_r2_alone_compiles",
        "e151_r2_composed_compiles",
        "e151_r2_four_combinations_distinct",
        "e151_r2_arm_moves_digest_with_r1_on",
        "e151_r2_arm_moves_digest_with_r1_off",
        "e151_r2_float_g32_shipped_compiles",
        "e151_r2_capacity_guard_defeated_rejected",
        "e151_r2_capacity_assert_named_in_refusal",
        "e151_r2_float_g32_matches_r1_alone",
        "e151_r2_too_strict_limit_rejected",
        "e151_r2_arm_liveness_named_in_refusal",
        "e151_r2_untouched_sites_identical",
        "e151_r2_twin_carries_both_loaders",
        # Promoted to required once the control was shown to explain the residue
        # exactly. This is the strongest containment check in the file: if a
        # later edit ever let the R2-disarmed path emit one real instruction,
        # the unexplained remainder goes non-zero and the gate fails. A tolerance
        # band would have made this field unfalsifiable, so it is an equality.
        "e151_r2_armoff_residue_is_mangling",
    ]
    result["e151_r2_compile_gate_pass"] = all(bool(result[k]) for k in required)
    result["e151_r2_compile_gate_required_fields"] = required

    pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for k in required + [
        "e151_r2_armoff_matches_r1",
        "e151_r2_line_shift_moves_digest",
        "e151_r2_line_shift_air_delta_bytes",
        "e151_r2_template_param_moves_digest",
        "e151_r2_template_param_air_delta_bytes",
        "e151_r2_armoff_residue_is_mangling",
        "e151_r2_armoff_residue_unexplained_bytes",
        "e151_r2_compile_gate_pass",
    ]:
        print(f"{k:<52} {result[k]}")
    print(f"{'e151_r2_armoff_air_delta_bytes.scored':<52} "
          f"{result['e151_r2_armoff_air_delta_bytes']['scored']}")
    combo_probe = {
        "both_off": "tree_both_off",
        "r1_alone": "r1_alone_tree",
        "r2_alone": "r2_alone",
        "composed": "composed",
    }
    for label, dg in combos.items():
        print(f"  combo {label:<10} air={air(combo_probe[label])} "
              f"sha={(dg or 'REFUSED')[:16]}")
    if not result["e151_r2_compile_gate_pass"]:
        for label in (
            "composed",
            "r2_alone",
            "float_g32_shipped",
            "float_g32_capacity_defeated",
            "scored_capacity_too_strict",
        ):
            err = p[label].get("error")
            if err:
                print(f"\n--- {label} error ---\n{err}")
    return 0 if result["e151_r2_compile_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
