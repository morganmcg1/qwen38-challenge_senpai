#!/usr/bin/env python3
"""E151 rung R1: offline compile gate for the SHIPPED arm-on NAX retile.

E147 rung E-1c gated an arm that shipped OFF. R1 ships it ON, so the polarity of
every check inverts and E-1c can no longer speak for the tree: its first
statement refuses a tree whose twin does not carry `= false`. This gate is the
arm-on twin of that rung and it reuses E-1c's probe machinery unchanged.

WHY IT MATTERS MORE NOW. With the arm off, a defect in the retiled path was
unreachable. With the arm on, the retiled path IS the ranked seed prefill, and
`is_nax_available()` is false on every Mac this campaign owns (Finding 250), so
no local run of any length can execute one instruction of it. A Metal compile of
the exact JIT concatenation is the only executable check available before the
ranked receipt.

WHAT IT DECIDES, WITH ZERO GPU:

  1. The tree ships the arm ON, read from the concatenated twin source that MLX
     actually JIT-compiles, not from the readable header. A twin that drifted
     from its header fails here.
  2. The shipped arm-on source compiles at the scored instantiation.
  3. The flag is still the whole switch: forcing it off changes the AIR digest.
  4. R1 also replaced `if constexpr (...) { ...; return; }` plus a trailing
     unconditional call with a real `if constexpr / else`, so the discarded
     branch is no longer instantiated. Compiling the forced-off tree and the
     arm-off base and comparing digests prices that refactor.
  5. The guards still bite. `illegal_bn48` breaks both retile `static_assert`s
     and must be refused.
  6. RULE 145 is still enforced, and the hazard is still real (Rule 101). Note
     that arm-on changes which guard fires first. With the arm on, the tile is
     always (128, 32), so TM = 4 and TN = 1 and RULE 145 can never trip on the
     scored path; a bad HOST shape trips the earlier retile-area assert instead.
     RULE 145 therefore has to be exercised where it can still fire: the
     forced-off tree at host shape (96, 32), which gives TM = 3, TN = 1 and
     matches no `tile_matmad_nax` branch. The unguarded control is revision
     `PREGUARD_REV`, which carries the same mechanism with no guard and must
     COMPILE that shape.
  7. Every other shipped NAX instantiation still compiles.
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
OUT = ROOT / "research/e151-r1-compile-gate.json"

# The commit immediately before E147 F11 ruling 1 landed the RULE 145 guards. It
# already carries the retile arm, so the guarded and unguarded probes differ by
# the guard alone.
PREGUARD_REV = "7e6157cd"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        required=True,
        help="the arm-off commit this submission is measured against",
    )
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    base_src = e1c.preamble(lambda p: e1c.show(args.base, p), e1c.NAX_TWINS)
    tree_src = e1c.preamble(e1c.worktree, e1c.NAX_TWINS)

    if e1c.FLAG_ON not in tree_src:
        raise SystemExit("e151_r1_compile_gate: the tree twin does not ship the arm on")
    if e1c.FLAG_OFF in tree_src:
        raise SystemExit("e151_r1_compile_gate: the tree twin still carries an arm-off flag")
    if e1c.FLAG_OFF not in base_src:
        raise SystemExit("e151_r1_compile_gate: the base twin does not ship the arm off")

    forced_off = e1c.substitute(tree_src, e1c.FLAG_ON, e1c.FLAG_OFF, "forced_off")
    illegal = e1c.substitute(
        tree_src, e1c.ARM_BN, e1c.ARM_BN_ILLEGAL, "illegal_bn48"
    )

    preguard_src = e1c.preamble(
        lambda p: e1c.show(PREGUARD_REV, p), e1c.NAX_TWINS
    )
    if e1c.RULE_145_NEEDLE in preguard_src:
        raise SystemExit(
            "e151_r1_compile_gate: the pre-guard revision already carries RULE 145"
        )

    probes = {
        "base_arm_off": (base_src, e1c.SCORED_INST),
        "tree_arm_on": (tree_src, e1c.SCORED_INST),
        "tree_forced_off": (forced_off, e1c.SCORED_INST),
        "illegal_bn48": (illegal, e1c.SCORED_INST),
        "failopen_unguarded_preguard": (preguard_src, e1c.FAILOPEN_INST),
        "failopen_guarded_forced_off": (forced_off, e1c.FAILOPEN_INST),
        "failopen_arm_on": (tree_src, e1c.FAILOPEN_INST),
    }

    result = {
        "experiment": "E151",
        "rung": "R1",
        "harness": "offline",
        "base": subprocess.run(
            ["git", "rev-parse", args.base],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "tree": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "scored_instantiation": e1c.SCORED_INST,
        "host_grid": {"BM": e1c.HOST_BM, "BN": e1c.HOST_BN, "BK": e1c.HOST_BK},
        "arm_grid": {"BM": e1c.ARM_BM, "BN": e1c.ARM_BN_VAL},
        "jit_concatenation_order": e1c.NAX_TWINS,
        "base_preamble_bytes": len(base_src),
        "tree_preamble_bytes": len(tree_src),
        "probes": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp) / "probe"
        for label, (src, inst) in probes.items():
            result["probes"][label] = e1c.compile_probe(
                src, inst, e1c.HOST_NAME, workdir
            )
        for label, inst in e1c.SHIPPED_OTHER_SITES.items():
            result["probes"][label] = e1c.compile_probe(
                tree_src, inst, e1c.HOST_NAME, workdir
            )

    p = result["probes"]
    on, off, base = p["tree_arm_on"], p["tree_forced_off"], p["base_arm_off"]

    result["e151_tree_ships_arm_on"] = True
    result["e151_base_source_differs_from_tree"] = base_src != tree_src
    result["e151_arm_on_compiles"] = on["compiled"]
    result["e151_forced_off_compiles"] = off["compiled"]
    result["e151_base_arm_off_compiles"] = base["compiled"]
    result["e151_arm_on_differs_from_arm_off"] = (
        on["compiled"] and off["compiled"] and on["air_sha256"] != off["air_sha256"]
    )
    # The R1 if/else refactor is invisible when the arm is off if and only if
    # these two digests agree. They are the same kernel, same instantiation,
    # same preamble order, built in the same fixed directory.
    result["e151_ifelse_refactor_is_armoff_neutral"] = (
        off["compiled"]
        and base["compiled"]
        and off["air_sha256"] == base["air_sha256"]
    )
    result["e151_arm_on_air_bytes"] = on.get("air_bytes")
    result["e151_arm_off_air_bytes"] = off.get("air_bytes")
    result["e151_base_air_bytes"] = base.get("air_bytes")
    result["e151_illegal_shape_rejected"] = not p["illegal_bn48"]["compiled"]
    result["e151_static_assert_named_in_refusal"] = "static_assert" in p[
        "illegal_bn48"
    ].get("error", "")
    result["e151_preguard_revision"] = PREGUARD_REV
    result["e151_failopen_compiles_unguarded"] = p["failopen_unguarded_preguard"][
        "compiled"
    ]
    result["e151_failopen_shape_rejected"] = not p["failopen_guarded_forced_off"][
        "compiled"
    ]
    result["e151_rule145_named_in_refusal"] = e1c.RULE_145_NEEDLE in p[
        "failopen_guarded_forced_off"
    ].get("error", "")
    # Arm-on cannot reach RULE 145 on the scored path. Record which guard does
    # fire, so the report states the ordering rather than implying RULE 145
    # covers a case it cannot see.
    result["e151_failopen_arm_on_rejected"] = not p["failopen_arm_on"]["compiled"]
    result["e151_failopen_arm_on_guard"] = (
        "retile_area"
        if "a retiled tile must cover the host tile"
        in p["failopen_arm_on"].get("error", "")
        else "rule145"
        if e1c.RULE_145_NEEDLE in p["failopen_arm_on"].get("error", "")
        else "none"
    )
    result["e151_shipped_other_sites_compile"] = all(
        p[label]["compiled"] for label in e1c.SHIPPED_OTHER_SITES
    )

    required = [
        "e151_tree_ships_arm_on",
        "e151_base_source_differs_from_tree",
        "e151_arm_on_compiles",
        "e151_forced_off_compiles",
        "e151_base_arm_off_compiles",
        "e151_arm_on_differs_from_arm_off",
        "e151_illegal_shape_rejected",
        "e151_static_assert_named_in_refusal",
        "e151_failopen_compiles_unguarded",
        "e151_failopen_shape_rejected",
        "e151_rule145_named_in_refusal",
        "e151_failopen_arm_on_rejected",
        "e151_shipped_other_sites_compile",
    ]
    result["e151_r1_compile_gate_pass"] = all(result[k] for k in required)
    result["e151_r1_compile_gate_required_fields"] = required

    pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for k in required + [
        "e151_ifelse_refactor_is_armoff_neutral",
        "e151_failopen_arm_on_guard",
        "e151_arm_on_air_bytes",
        "e151_arm_off_air_bytes",
        "e151_base_air_bytes",
        "e151_r1_compile_gate_pass",
    ]:
        print(f"{k:<48} {result[k]}")
    if not result["e151_r1_compile_gate_pass"]:
        for label in ("tree_arm_on", "tree_forced_off", "illegal_bn48",
                      "failopen_guarded_forced_off",
                      "failopen_unguarded_preguard"):
            err = p[label].get("error")
            if err:
                print(f"\n--- {label} error ---\n{err}")
    return 0 if result["e151_r1_compile_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
