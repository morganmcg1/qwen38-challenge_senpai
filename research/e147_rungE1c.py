#!/usr/bin/env python3
"""E147 rung E-1c: build the (128, 32) NAX retile arm and gate it offline.

WHY THIS RUNG NEEDS ITS OWN GATE. The NAX kernel cannot run here. `applegpu_g16s`
has no NAX unit, `is_nax_available()` is false on every Mac this campaign owns,
and the scored prefill on the ranked M5 therefore reaches `affine_qmm_t_nax`
through a dispatch edge no local run can take. Rung E-1b proved the retile
mechanism is bit exact on hardware, but it proved it for the non-NAX twin of
this code. Rung E-2 proved the (128, 32) NAX tile shape compiles, but it did so
by instantiating the template directly, which bypasses the mechanism entirely.
Neither speaks for the source that would actually ship.

WHAT THIS RUNG DECIDES, WITH ZERO GPU:

  1. The arm-on source is legal. Every `static_assert`, every loader bound and
     every tile shape in the retiled path is checked by a real Metal compile of
     the real mechanism, not of a hand-instantiated proxy.

  2. The flag is the whole switch. Arm-on and arm-off must differ, or the flag
     is decoration and every later measurement of it is meaningless.

  3. What the arm-off default costs. The arm ships off, so the mechanism must be
     close to free when it is off. This rung prices that.

PRICING THE ARM-OFF DEFAULT IS NOT A DIGEST COMPARISON. Measured here, the NAX
arm-off AIR is larger than the pre-arm base. That number is not the cost.
`metal -O2` emits the per-tile body as an out-of-line function with a capture
frame, and the AGX backend inlines it again later. AIR therefore prices a
function call that no GPU ever executes.

The only way to see through that is machine code, and the AGX translator
refuses every real NAX GEMM shape (rung E-2, both arches, byte-identical
refusal). So this rung also prices the arm-off default on the one member of the
refactor family the backend will translate: the non-NAX `affine_qmm_t`, which
carries the same mechanism, written the same way, over the same loop. It reports
the AIR delta and the translated-ISA delta side by side for that pair, so a
reader can see how far AIR overstates the cost, and it reports the NAX AIR delta
as what it is: an upper bound of unknown tightness.

THE POSITIVE CONTROLS (Rule 101). Several checks here are equalities or
inequalities between digests, and a digest comparison that has never failed is
not a comparison. So this rung also compiles sources that must be rejected or
must move:

  * `illegal_bn48` sets the arm tile to (128, 48). It breaks both retile
    `static_assert`s at once (unequal area, and the host tile no longer splits
    along N). It must fail to compile, or the guards are inert.

  * `failopen_unguarded` and `failopen_guarded` are a RULE 145 pair. Tile shape
    (96, 32) gives TM = 3, TN = 1, which matches neither `tile_matmad_nax`
    branch. Rung E-2 found that this shape compiles clean, issues no `mma` and
    returns the zeros the destination tile was cleared with. The unguarded probe
    compiles the pre-arm base at that shape and must SUCCEED, which is what
    makes the hazard real rather than hypothetical. The guarded probe compiles
    the same shape against the edited tree and must FAIL naming RULE 145. A
    guard whose hazard was never observed to compile is not a guard.

  * `retile_direct` compiles the pre-arm base at the (128, 32) instantiation
    rung E-2 used. Same preamble, different kernel. It must compile and its
    digest must differ from the base, or the digest is not reading the kernel.

  * The base preamble and the edited preamble must differ as text, and the
    non-NAX transfer pair must differ in AIR. Otherwise those comparisons are
    reading one source twice.

Zero GPU. Offline compilation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import agx_crossarch as agx  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated"
OUT = ROOT / "research/e147-rungE1c.json"

# The concatenation order in `get_qmm_nax_kernel`, mode == "affine"
# (jit_kernels.cpp:1116-1134).
NAX_TWINS = ["utils.cpp", "gemm_nax.cpp", "quantized_utils.cpp",
             "quantized_nax.cpp"]
# The concatenation order in `get_quantized_kernel`, mode == "affine".
NON_NAX_TWINS = ["utils.cpp", "gemm.cpp", "quantized_utils.cpp",
                 "quantized.cpp"]

# The host launch geometry, frozen at backend/metal/quantized.cpp qmm_nax and
# not editable from the submitted surface.
HOST_BM, HOST_BN, HOST_BK = 64, 64, 64
ARM_BM, ARM_BN_VAL = 128, 32
SCORED_INST = (f"affine_qmm_t_nax<bfloat16_t, 64, 4, 1, 0, "
               f"{HOST_BM}, {HOST_BK}, {HOST_BN}, 2, 2>")
DIRECT_INST = (f"affine_qmm_t_nax<bfloat16_t, 64, 4, 1, 0, "
               f"{ARM_BM}, {HOST_BK}, {ARM_BN_VAL}, 2, 2>")
HOST_NAME = "e147_qmm_t_nax_scored"

# The non-NAX transfer pair. Both ends are git revisions, never the working
# tree: F11 ruling 1 removes the non-NAX mechanism from the shipped candidate,
# and this measurement has to stay reproducible after that revert. The two
# commits differ only by the retile mechanism with its arm off, so rung A,
# which both carry, cancels.
NON_NAX_NO_MECHANISM = "01ed58d5309c06b3db27c71d6a6b73ed6a50a341"
NON_NAX_WITH_MECHANISM = "5dc4d1d9184c37b2dcf0d79f61128c414e0a8784"
NON_NAX_HOST = "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0"
NON_NAX_INST = "affine_qmm_t<bfloat16_t, 64, 4, 1, 0>"

# RULE 145. `tile_matmad_nax` has two `if constexpr` branches and no `else`, so
# (BM, BN) = (96, 32) gives TM = 3, TN = 1, matches neither, and multiplies
# nothing. Rung E-2 found that this is the ONLY NAX shape the offline
# translator accepts, which makes it the one shape a careless census would
# report on. The arm now carries a `static_assert` that must reject it.
FAILOPEN_INST = (f"affine_qmm_t_nax<bfloat16_t, 64, 4, 1, 0, "
                 f"96, {HOST_BK}, 32, 2, 2>")
RULE_145_NEEDLE = "matches no tile_matmad_nax branch"

# The RULE 145 guard is a property of `tile_matmad_nax`, not of the one call
# site this experiment retiles, so it is placed at all three tile-constant sites
# in the header. That is only safe if every shipped instantiation still
# compiles, and no local run can prove it: `is_nax_available()` is false on
# every Mac this campaign owns, so MLX never JIT-compiles a NAX kernel here and
# a broken `static_assert` would first appear on the ranked M5. These two probes
# close that hole offline. Both shapes come from `quantized_nax.metal`, which
# instantiates every NAX kernel at 64/64/64/2/2 and nothing else.
SHIPPED_OTHER_SITES = {
    "shipped_qmm_n": "affine_qmm_n_nax<bfloat16_t, 64, 4, 1, 64, 64, 64, 2, 2>",
    "shipped_gather_rhs":
        "affine_gather_qmm_rhs_nax<bfloat16_t, 64, 4, 64, 64, 64, 2, 2, true>",
}

FLAG_OFF = "constexpr bool kE147NaxRetileOn = false;"
FLAG_ON = "constexpr bool kE147NaxRetileOn = true;"
ARM_BN = "constexpr int kE147NaxRetileBN = kE147NaxRetileOn ? 32 : BN;"
ARM_BN_ILLEGAL = "constexpr int kE147NaxRetileBN = kE147NaxRetileOn ? 48 : BN;"

ARCHES = [agx.LOCAL_ARCH, agx.RANKED_ARCH]
RAW = re.compile(r'R"([A-Za-z_]*)\(\n?(.*)\)\1"', re.DOTALL)


def show(rev: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout


def worktree(path: str) -> str:
    return (ROOT / path).read_text()


def literal(text: str, name: str) -> str:
    m = RAW.search(text)
    if not m:
        raise SystemExit(f"e147_rungE1c: no raw string literal in {name}")
    return m.group(2)


def preamble(reader, twins) -> str:
    return "".join(literal(reader(f"{GEN}/{n}"), n) for n in twins)


def substitute(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            f"e147_rungE1c: {what} needs exactly 1 copy of {old!r}, found {n}")
    return text.replace(old, new)


def compile_probe(source: str, inst: str, host: str,
                  workdir: pathlib.Path) -> dict:
    """Compile one probe in a fixed directory so no path leaks into the digest."""
    if workdir.exists():
        shutil.rmtree(workdir)
    body = (source + f'\ntemplate [[host_name("{host}")]] [[kernel]] '
            f"decltype({inst}) {inst};\n")
    try:
        lib = agx.build_metallib(body, workdir)
    except subprocess.CalledProcessError as exc:
        return {"compiled": False,
                "error": exc.stderr.decode(errors="replace")[-1200:]}
    data = lib.read_bytes()
    rec = {"compiled": True, "air_bytes": len(data),
           "air_sha256": hashlib.sha256(data).hexdigest(),
           "source_bytes": len(body)}
    for arch in ARCHES:
        try:
            got = agx.translate(lib, arch, workdir, select=lambda n: n == host)
        except SystemExit as exc:
            rec[arch] = {"translated": False,
                         "error_body": str(exc).split("\n", 1)[-1].strip()}
            continue
        rec[arch] = dict(got[host], translated=True)
    return rec


def transfer_pair(workdir: pathlib.Path) -> dict:
    """Price the same mechanism where the backend will translate it."""
    old = preamble(lambda p: show(NON_NAX_NO_MECHANISM, p), NON_NAX_TWINS)
    new = preamble(lambda p: show(NON_NAX_WITH_MECHANISM, p), NON_NAX_TWINS)
    out = {"no_mechanism_rev": NON_NAX_NO_MECHANISM,
           "with_mechanism_rev": NON_NAX_WITH_MECHANISM,
           "entry_point": NON_NAX_HOST,
           "instantiation": NON_NAX_INST,
           "sources_differ": old != new}
    for label, src in (("no_mechanism", old), ("arm_off", new)):
        out[label] = compile_probe(src, NON_NAX_INST, NON_NAX_HOST, workdir)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="the pre-arm commit the NAX arm is measured against")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    base_src = preamble(lambda p: show(args.base, p), NAX_TWINS)
    tree_src = preamble(worktree, NAX_TWINS)

    # The edited tree must ship the arm off. Read that from the source that is
    # about to be compiled, not from the file on disk, so a twin that drifted
    # from its header cannot pass this rung.
    if FLAG_OFF not in tree_src:
        raise SystemExit("e147_rungE1c: the tree twin does not ship the arm off")

    probes = {
        "base": (base_src, SCORED_INST),
        "arm_off": (tree_src, SCORED_INST),
        "arm_on": (substitute(tree_src, FLAG_OFF, FLAG_ON, "arm_on"),
                   SCORED_INST),
        "illegal_bn48": (substitute(
            substitute(tree_src, FLAG_OFF, FLAG_ON, "illegal_bn48"),
            ARM_BN, ARM_BN_ILLEGAL, "illegal_bn48"), SCORED_INST),
        "retile_direct": (base_src, DIRECT_INST),
        "failopen_unguarded": (base_src, FAILOPEN_INST),
        "failopen_guarded": (tree_src, FAILOPEN_INST),
    }

    result = {"base": subprocess.run(["git", "rev-parse", args.base], cwd=ROOT,
                                     capture_output=True, text=True,
                                     check=True).stdout.strip(),
              "scored_instantiation": SCORED_INST,
              "direct_retile_instantiation": DIRECT_INST,
              "host_grid": {"BM": HOST_BM, "BN": HOST_BN, "BK": HOST_BK},
              "arm_grid": {"BM": ARM_BM, "BN": ARM_BN_VAL},
              "jit_concatenation_order": NAX_TWINS,
              "base_preamble_bytes": len(base_src),
              "tree_preamble_bytes": len(tree_src),
              "probes": {}}

    with tempfile.TemporaryDirectory() as tmp:
        # One fixed path for every probe: `xcrun metal` records the source file
        # name, so two probes built in differently named directories would
        # differ for a reason that has nothing to do with the kernel.
        workdir = pathlib.Path(tmp) / "probe"
        for label, (src, inst) in probes.items():
            result["probes"][label] = compile_probe(src, inst, HOST_NAME,
                                                    workdir)
        for label, inst in SHIPPED_OTHER_SITES.items():
            result["probes"][label] = compile_probe(tree_src, inst, HOST_NAME,
                                                    workdir)
        result["non_nax_transfer"] = transfer_pair(workdir)

    p = result["probes"]
    off, on, base = p["arm_off"], p["arm_on"], p["base"]

    result["e147_rungE1c_base_source_differs_from_tree"] = base_src != tree_src
    result["e147_rungE1c_base_compiles"] = base["compiled"]
    result["e147_rungE1c_arm_off_compiles"] = off["compiled"]
    result["e147_rungE1c_arm_on_compiles"] = on["compiled"]
    result["e147_rungE1c_arm_on_differs_from_arm_off"] = (
        on["compiled"] and off["compiled"]
        and on["air_sha256"] != off["air_sha256"])
    result["e147_rungE1c_illegal_shape_rejected"] = (
        not p["illegal_bn48"]["compiled"])
    result["e147_rungE1c_static_assert_named_in_refusal"] = (
        "static_assert" in p["illegal_bn48"].get("error", ""))
    result["e147_rungE1c_direct_retile_compiles"] = (
        p["retile_direct"]["compiled"])
    result["e147_rungE1c_digest_control_moves"] = (
        p["retile_direct"]["compiled"]
        and p["retile_direct"]["air_sha256"] != base["air_sha256"])

    # RULE 145. The hazard has to be observed before the guard against it means
    # anything, so the unguarded probe must compile and the guarded one must not.
    result["e147_rungE1c_failopen_instantiation"] = FAILOPEN_INST
    result["e147_rungE1c_failopen_compiles_unguarded"] = (
        p["failopen_unguarded"]["compiled"])
    result["e147_rungE1c_failopen_shape_rejected"] = (
        not p["failopen_guarded"]["compiled"])
    result["e147_rungE1c_rule145_named_in_refusal"] = (
        RULE_145_NEEDLE in p["failopen_guarded"].get("error", ""))
    result["e147_rungE1c_rule145_control_observed"] = (
        result["e147_rungE1c_failopen_compiles_unguarded"]
        and result["e147_rungE1c_failopen_shape_rejected"]
        and result["e147_rungE1c_rule145_named_in_refusal"])
    result["e147_rungE1c_guarded_tile_sites"] = 3
    result["e147_rungE1c_shipped_other_sites_compile"] = all(
        p[k]["compiled"] for k in SHIPPED_OTHER_SITES)

    # The NAX arm-off cost, in the only unit this arch will give up.
    result["e147_rungE1c_nax_arm_off_air_delta_bytes"] = (
        off["air_bytes"] - base["air_bytes"])
    result["e147_rungE1c_nax_arm_on_air_delta_bytes"] = (
        on["air_bytes"] - off["air_bytes"])
    result["e147_rungE1c_nax_translatable"] = any(
        base[a].get("translated") for a in ARCHES)

    # The same mechanism, priced where the backend will translate it.
    t = result["non_nax_transfer"]
    result["e147_rungE1c_transfer_sources_differ"] = t["sources_differ"]
    result["e147_rungE1c_transfer_translated"] = all(
        t[k][a].get("translated")
        for k in ("no_mechanism", "arm_off") for a in ARCHES)
    result["e147_rungE1c_transfer_air_delta_bytes"] = (
        t["arm_off"]["air_bytes"] - t["no_mechanism"]["air_bytes"])
    if result["e147_rungE1c_transfer_translated"]:
        for arch in ARCHES:
            a, b = t["no_mechanism"][arch], t["arm_off"][arch]
            result[f"e147_rungE1c_transfer_isa_text_delta_{arch}"] = (
                b["text_bytes"] - a["text_bytes"])
            result[f"e147_rungE1c_transfer_register_delta_{arch}"] = (
                b["registers"] - a["registers"])
            result[f"e147_rungE1c_transfer_spill_delta_{arch}"] = (
                b["spill_bytes"] - a["spill_bytes"])
        isa = result[
            f"e147_rungE1c_transfer_isa_text_delta_{agx.RANKED_ARCH}"]
        result["e147_rungE1c_transfer_air_overstates_isa_by"] = (
            round(result["e147_rungE1c_transfer_air_delta_bytes"] / isa, 1)
            if isa else None)

    reported = "e147_rungE1c_"
    checks = [k for k in result if k.startswith(reported)
              and isinstance(result[k], bool)
              and k != "e147_rungE1c_nax_translatable"]
    failed = sorted(k for k in checks if not result[k])
    result["e147_rungE1c_verdict"] = "PASS" if not failed else "FAIL"
    result["e147_rungE1c_failed_checks"] = failed

    pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    for k in sorted(result):
        if k.startswith(reported) and k != "e147_rungE1c_verdict":
            print(f"{k}={result[k]}")
    print(f"e147_rungE1c_verdict={result['e147_rungE1c_verdict']}")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
