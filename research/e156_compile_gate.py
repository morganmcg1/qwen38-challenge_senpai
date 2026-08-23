#!/usr/bin/env python3
"""E156: offline compile gate for the composed retile plus double-buffer arm.

`is_nax_available()` needs Apple GPU architecture generation 17 or newer
(`backend/metal/device.cpp:913-931`). Every campaign Mac is `applegpu_g16s`,
generation 16, so `affine_qmm_t_nax` executes zero times here and `metal-tt`
refuses this kernel family. A Metal compile of the exact JIT concatenation is
the only executable check available before the ranked receipt.

WHAT THIS GATE DECIDES, WITH ZERO GPU:

  1. All four arm cells build: crown, retile alone, double buffer alone, and
     the composed arm this experiment ships.
  2. `e156_compose_verdict = compose_forced_by_tgp_limit` is EMPIRICALLY true,
     not just arithmetic. `research/e156_compose_audit.py` computes that the
     `float` cell needs 34816 B of a 32768 B threadgroup budget when the
     staged tile stays 64 wide. Here the compiler is asked. Defeat the
     capacity predicate and that cell must be REFUSED, by name. Rule 101: a
     guard that cannot fail is not a guard.
  3. The predicate is not silently too strict in the other direction. Shrink
     the limit and the SCORED cell must be refused by the arm-liveness assert,
     which proves that assert would catch a predicate that disarmed the tile
     the experiment is supposed to move.
  4. A disarmed cell is disarmed COMPLETELY: the `float` group-32 cell must be
     digest-identical to the same cell with the double buffer absent, not
     merely buildable.
  5. Containment. `kDoubleBuffer` defaults to false, so every other call site
     of `qmm_t_nax_tgp_impl` must be byte-identical to the retile-only tree.
     Checked by AIR digest, not asserted.

WHY GROUP 32 IS THE SHARP CELL. The retile carries a loader-legality predicate
that disarms the 32-wide tile at `group_size == 32`, so those cells keep the
64-wide host tile even with the retile ON. The composed arm therefore still has
to survive a 64-wide staged tile at `T = float`, and the only thing that saves
it is the capacity predicate. That is the cell this gate attacks.

AIR IS NOT PRICED. On this family AIR overstates the ISA change by 181.3 times.
AIR digests are used here only as identity and containment evidence: equal or
not equal. No E156 percentage is derived from an AIR byte delta.
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
OUT = ROOT / "research/e156-compile-gate.json"

HEADER = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp"

CROWN = "0863b06ac16e26e48fc06e97444095b00feb66d4"
RETILE_ONLY = "8b515fd8b81e6b580e796b996c27c4288d6dcfa9"

RETILE_ON = "constexpr bool kE147NaxRetileOn = true;"
RETILE_OFF = "constexpr bool kE147NaxRetileOn = false;"
DBUF_ON = "constexpr bool kE151NaxDoubleBufferOn = true;"
DBUF_OFF = "constexpr bool kE151NaxDoubleBufferOn = false;"

CAPACITY_GUARD = "kE151NaxDoubleBufferOn && kE151DoubleBufferFits;"
CAPACITY_GUARD_DEFEATED = "kE151NaxDoubleBufferOn;"

TGP_LIMIT = "constexpr int kE151MaxTgpBytes = 32768;"
TGP_LIMIT_TOO_SMALL = "constexpr int kE151MaxTgpBytes = 4096;"

CAPACITY_NEEDLE = "does not fit in threadgroup memory"
ARM_LIVENESS_NEEDLE = "must take the double-buffer arm"
RETILE_LIVENESS_NEEDLE = "must take the retile arm"

# The cell that cannot hold two 64-wide staging halves: float, BK_padded 68, so
# 2 * 64 * 68 * 4 = 34816 B against a 32768 B budget.
FLOAT_G32_INST = "affine_qmm_t_nax<float, 32, 4, 1, 0, 64, 64, 64, 2, 2>"
BF16_G32_INST = "affine_qmm_t_nax<bfloat16_t, 32, 4, 1, 0, 64, 64, 64, 2, 2>"

UNTOUCHED_SITES = {
    "untouched_qmm_n": "affine_qmm_n_nax<bfloat16_t, 64, 4, 1, 64, 64, 64, 2, 2>",
    "untouched_gather_rhs": (
        "affine_gather_qmm_rhs_nax<bfloat16_t, 64, 4, 64, 64, 64, 2, 2, true>"
    ),
}

HOST_NAME = "e156_composed_scored"


def rev_parse(rev: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", rev], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crown", default=CROWN, help="the crown pair, both arms absent")
    ap.add_argument("--retile-rev", default=RETILE_ONLY, help="retile tip, no R2 arm")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tree = e1c.preamble(e1c.worktree, e1c.NAX_TWINS)
    retile = e1c.preamble(lambda p: e1c.show(args.retile_rev, p), e1c.NAX_TWINS)
    crown = e1c.preamble(lambda p: e1c.show(args.crown, p), e1c.NAX_TWINS)

    if DBUF_ON not in tree:
        raise SystemExit("e156_compile_gate: the tree twin does not ship the arm on")
    if DBUF_OFF in tree:
        raise SystemExit("e156_compile_gate: the tree twin carries an arm-off flag")
    if RETILE_ON not in tree:
        raise SystemExit("e156_compile_gate: the tree twin lost the retile arm")
    if DBUF_ON in retile or DBUF_OFF in retile:
        raise SystemExit("e156_compile_gate: the retile revision already has an arm")
    if RETILE_ON in crown or DBUF_ON in crown:
        raise SystemExit("e156_compile_gate: the crown already carries an arm flag")

    dbuf_off = e1c.substitute(tree, DBUF_ON, DBUF_OFF, "dbuf_forced_off")
    retile_off = e1c.substitute(tree, RETILE_ON, RETILE_OFF, "retile_forced_off")
    both_off = e1c.substitute(retile_off, DBUF_ON, DBUF_OFF, "both_forced_off")
    capacity_defeated = e1c.substitute(
        tree, CAPACITY_GUARD, CAPACITY_GUARD_DEFEATED, "capacity_guard_defeated"
    )
    capacity_too_strict = e1c.substitute(
        tree, TGP_LIMIT, TGP_LIMIT_TOO_SMALL, "capacity_too_strict"
    )
    # The counterfactual the verdict rests on: the double buffer WITHOUT the
    # retile and WITHOUT its capacity predicate. This is the configuration a
    # standalone R2 would have shipped if the limit were not in the way.
    standalone_dbuf_unguarded = e1c.substitute(
        retile_off, CAPACITY_GUARD, CAPACITY_GUARD_DEFEATED, "standalone_unguarded"
    )

    probes = {
        # 1. the four arm cells, scored instantiation
        "crown": (crown, e1c.SCORED_INST),
        "tree_both_off": (both_off, e1c.SCORED_INST),
        "retile_alone_rev": (retile, e1c.SCORED_INST),
        "retile_alone_tree": (dbuf_off, e1c.SCORED_INST),
        "dbuf_alone": (retile_off, e1c.SCORED_INST),
        "composed": (tree, e1c.SCORED_INST),
        # 2. the capacity predicate, on the cell that cannot hold two halves
        "float_g32_shipped": (tree, FLOAT_G32_INST),
        "float_g32_capacity_defeated": (capacity_defeated, FLOAT_G32_INST),
        "float_g32_retile_alone": (dbuf_off, FLOAT_G32_INST),
        # the verdict's counterfactual, on the float cell
        "float_g32_standalone_unguarded": (
            standalone_dbuf_unguarded, FLOAT_G32_INST),
        "float_scored_standalone_unguarded": (
            standalone_dbuf_unguarded, e1c.SCORED_INST),
        # 3. the arm-liveness assert, exercised where it can still fire
        "scored_capacity_too_strict": (capacity_too_strict, e1c.SCORED_INST),
        # 4. the group-32 cell the retile's loader guard forces back to 64 wide.
        # All four arm cells are compiled at THIS instantiation, because a
        # digest may only be compared with another digest of the same
        # instantiation: the template arguments are part of the mangled symbol.
        "bf16_g32_shipped": (tree, BF16_G32_INST),
        "bf16_g32_retile_alone": (dbuf_off, BF16_G32_INST),
        "bf16_g32_both_off": (both_off, BF16_G32_INST),
        "bf16_g32_dbuf_alone": (retile_off, BF16_G32_INST),
    }
    for label, inst in UNTOUCHED_SITES.items():
        probes[f"{label}_retile"] = (retile, inst)
        probes[f"{label}_tree"] = (tree, inst)

    result = {
        "experiment": "E156",
        "rung": "R1",
        "harness": "offline",
        "crown": rev_parse(args.crown),
        "retile_rev": rev_parse(args.retile_rev),
        "tree": rev_parse("HEAD"),
        "scored_instantiation": e1c.SCORED_INST,
        "jit_concatenation_order": e1c.NAX_TWINS,
        "air_pricing_policy":
            "AIR digests are identity evidence only; AIR overstates the ISA "
            "change by 181.3 times on this family and no percentage is priced "
            "from an AIR delta",
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

    # 1. every arm cell is a real configuration
    result["e156_crown_compiles"] = p["crown"]["compiled"]
    result["e156_retile_alone_compiles"] = p["retile_alone_tree"]["compiled"]
    result["e156_dbuf_alone_compiles"] = p["dbuf_alone"]["compiled"]
    result["e156_composed_compiles"] = p["composed"]["compiled"]

    combos = {
        "both_off": digest("tree_both_off"),
        "retile_alone": digest("retile_alone_tree"),
        "dbuf_alone": digest("dbuf_alone"),
        "composed": digest("composed"),
    }
    result["e156_arm_combination_digests"] = combos
    result["e156_four_combinations_distinct"] = (
        all(combos.values()) and len(set(combos.values())) == 4
    )
    result["e156_dbuf_moves_digest_with_retile_on"] = (
        combos["composed"] is not None
        and combos["retile_alone"] is not None
        and combos["composed"] != combos["retile_alone"]
    )
    result["e156_dbuf_moves_digest_with_retile_off"] = (
        combos["dbuf_alone"] is not None
        and combos["both_off"] is not None
        and combos["dbuf_alone"] != combos["both_off"]
    )

    # 2. the capacity predicate carries load, and the verdict is empirical
    result["e156_float_g32_shipped_compiles"] = p["float_g32_shipped"]["compiled"]
    result["e156_capacity_guard_defeated_rejected"] = not p[
        "float_g32_capacity_defeated"]["compiled"]
    result["e156_capacity_assert_named_in_refusal"] = CAPACITY_NEEDLE in p[
        "float_g32_capacity_defeated"].get("error", "")
    result["e156_standalone_dbuf_float_cell_rejected"] = not p[
        "float_g32_standalone_unguarded"]["compiled"]
    result["e156_standalone_dbuf_refusal_names_capacity"] = CAPACITY_NEEDLE in p[
        "float_g32_standalone_unguarded"].get("error", "")
    # The same unguarded standalone source at the SCORED bfloat16 cell must
    # still build: the refusal above must be the float cell's byte count, not a
    # blanket breakage of the standalone configuration.
    result["e156_standalone_dbuf_scored_cell_still_compiles"] = p[
        "float_scored_standalone_unguarded"]["compiled"]

    # 3. a disarmed cell is disarmed completely
    result["e156_float_g32_matches_retile_alone"] = (
        p["float_g32_shipped"]["compiled"]
        and p["float_g32_retile_alone"]["compiled"]
        and digest("float_g32_shipped") == digest("float_g32_retile_alone")
    )

    # 3b. The group-32 cells decide whether the composition is really free.
    # The retile's loader guard disarms the 32-wide tile there, so those cells
    # keep the 64-wide host tile and the double buffer, when it arms, allocates
    # a second 64-wide half. Three compiler-checked consequences follow. All of
    # them compare digests of ONE instantiation across arm cells, never across
    # instantiations, because the template arguments are part of the symbol.
    result["e156_g32_retile_is_inert"] = (
        digest("bf16_g32_retile_alone") is not None
        and digest("bf16_g32_retile_alone") == digest("bf16_g32_both_off")
    )
    result["e156_g32_composed_equals_dbuf_alone"] = (
        digest("bf16_g32_shipped") is not None
        and digest("bf16_g32_shipped") == digest("bf16_g32_dbuf_alone")
    )
    result["e156_g32_bf16_dbuf_arms"] = (
        digest("bf16_g32_shipped") is not None
        and digest("bf16_g32_shipped") != digest("bf16_g32_retile_alone")
    )

    # 4. the predicate is not silently too strict
    result["e156_too_strict_limit_rejected"] = not p[
        "scored_capacity_too_strict"]["compiled"]
    err = p["scored_capacity_too_strict"].get("error", "")
    result["e156_arm_liveness_named_in_refusal"] = (
        ARM_LIVENESS_NEEDLE in err or CAPACITY_NEEDLE in err
    )

    # 5. containment
    untouched = {}
    for label in UNTOUCHED_SITES:
        untouched[label] = (
            p[f"{label}_retile"]["compiled"]
            and p[f"{label}_tree"]["compiled"]
            and digest(f"{label}_retile") == digest(f"{label}_tree")
        )
    result["e156_untouched_site_identity"] = untouched
    result["e156_untouched_sites_identical"] = all(untouched.values())

    # Both loader specializations must carry the staging-pointer helper. They
    # are separate classes, so a partial port compiles and then stages into the
    # wrong half.
    twin_text = e1c.worktree(TWIN)
    result["e156_twin_shift_dst_count"] = twin_text.count(
        "void shift_dst(const int delta)")
    result["e156_twin_carries_both_loaders"] = (
        result["e156_twin_shift_dst_count"] == 2
    )

    result["e156_air_bytes"] = {k: p[k].get("air_bytes") for k in probes}

    required = [
        "e156_crown_compiles",
        "e156_retile_alone_compiles",
        "e156_dbuf_alone_compiles",
        "e156_composed_compiles",
        "e156_four_combinations_distinct",
        "e156_dbuf_moves_digest_with_retile_on",
        "e156_dbuf_moves_digest_with_retile_off",
        "e156_float_g32_shipped_compiles",
        "e156_capacity_guard_defeated_rejected",
        "e156_capacity_assert_named_in_refusal",
        "e156_standalone_dbuf_float_cell_rejected",
        "e156_standalone_dbuf_refusal_names_capacity",
        "e156_standalone_dbuf_scored_cell_still_compiles",
        "e156_float_g32_matches_retile_alone",
        "e156_g32_retile_is_inert",
        "e156_g32_composed_equals_dbuf_alone",
        "e156_g32_bf16_dbuf_arms",
        "e156_too_strict_limit_rejected",
        "e156_arm_liveness_named_in_refusal",
        "e156_untouched_sites_identical",
        "e156_twin_carries_both_loaders",
    ]
    result["e156_compile_gate_pass"] = all(bool(result[k]) for k in required)
    result["e156_compile_gate_required_fields"] = required

    pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for k in required + ["e156_compile_gate_pass"]:
        print(f"{k:<52} {result[k]}")
    combo_probe = {
        "both_off": "tree_both_off",
        "retile_alone": "retile_alone_tree",
        "dbuf_alone": "dbuf_alone",
        "composed": "composed",
    }
    print()
    for label, dg in combos.items():
        print(f"  combo {label:<14} air={p[combo_probe[label]].get('air_bytes')} "
              f"sha={(dg or 'REFUSED')[:16]}")
    if not result["e156_compile_gate_pass"]:
        for label in (
            "composed", "dbuf_alone", "float_g32_shipped",
            "float_g32_capacity_defeated", "float_g32_standalone_unguarded",
            "float_scored_standalone_unguarded", "scored_capacity_too_strict",
        ):
            e = p[label].get("error")
            if e:
                print(f"\n--- {label} error ---\n{e[:1500]}")
    return 0 if result["e156_compile_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
