#!/usr/bin/env python3
"""E147 rung E-2: is the (128, 32) NAX retile legal, and does it reorder the sum?

Rung E-1a settled the tile bookkeeping and rung E-1b settles bit exactness for
the non-NAX analogue on hardware this team owns. Neither can speak for the NAX
kernel, because `applegpu_g16s` has no NAX unit and `is_nax_available()` is
false on every Mac this campaign can run. This rung buys the two things that
are still decidable offline:

  1. A static proof about the K traversal. If the retile leaves the per-output
     accumulation order alone, a (128, 32) NAX kernel is a scheduling change
     rather than a numerics change, and the exactness risk is confined to one
     named place instead of the whole kernel.

  2. A real compile of the retiled instantiation, translated by the actual AGX
     backend for the ranked M5 generation. That decides legality (every
     static_assert, every loader, every tile shape) and prices the register
     pressure the retile costs, which is the term that decides whether the
     retile can pay at all.

WHAT THE STATIC PROOF CAN AND CANNOT SAY. It reads the live source, not a
recollection of it, and every pattern below is fail-closed: a miss stops the
run. It establishes that the K loop bounds depend only on BK and the hard
constant SK, so every output element still sums its K terms in the same order.
It does NOT establish that the two `tile_matmad_nax` branches are numerically
identical. They are not the same code, and the retile switches between them.
That switch is the residual risk and this rung reports it as such.

THE FAIL-OPEN CONTROL. `tile_matmad_nax` has exactly two `if constexpr`
branches and no `else`. An instantiation with TN == 1 and odd TM matches
neither, compiles clean, and multiplies nothing: the destination tile keeps the
zeros it was cleared with. The third probe below is that shape. It is a
positive control for the compile census, because it must show a large drop in
translated text, and it is a live hazard for anyone who retiles this kernel by
changing BM alone.

Zero GPU. Offline compilation only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import agx_crossarch as agx  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated"
KERNELS = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
NAX_H = f"{KERNELS}/quantized_nax.h"
STEEL_NAX_H = f"{KERNELS}/steel/gemm/nax.h"
OUT = ROOT / "research/e147-rungE2.json"

# The concatenation order in `get_qmm_nax_kernel`, mode == "affine"
# (jit_kernels.cpp:1116-1134).
TWINS = ["utils.cpp", "gemm_nax.cpp", "quantized_utils.cpp", "quantized_nax.cpp"]

# affine_qmm_t_nax<T, group_size, bits, aligned_N, batched, BM, BK, BN, WM, WN>
# (quantized_nax.h:1292-1303). The host streams the same order at
# backend/metal/quantized.cpp:526-541, and its grid is frozen at
# bm = bn = bk = 64, wm = wn = 2.
HOST_BM, HOST_BN, HOST_BK = 64, 64, 64
PROBES = [
    ("shipped_64x64", 64, 64, "the scored instantiation, TM=2 TN=2"),
    ("retile_128x32", 128, 32, "the E-1c arm, TM=4 TN=1, equal area"),
    ("failopen_96x32", 96, 32, "TM=3 TN=1 matches no matmad branch"),
]
ARCHES = [agx.LOCAL_ARCH, agx.RANKED_ARCH]

RAW = re.compile(r'R"([A-Za-z_]*)\(\n?(.*)\)\1"', re.DOTALL)


def show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout


def twin_source(rev: str, name: str) -> str:
    m = RAW.search(show(rev, f"{GEN}/{name}"))
    if not m:
        raise SystemExit(f"e147_rungE2: no raw string literal in {name}")
    return m.group(2)


def region(text: str, start_pat: str, end_pat: str, what: str) -> str:
    i = text.find(start_pat)
    if i < 0:
        raise SystemExit(f"e147_rungE2: cannot find the start of {what}")
    j = text.find(end_pat, i + len(start_pat))
    if j < 0:
        raise SystemExit(f"e147_rungE2: cannot find the end of {what}")
    return text[i:j]


def counted(where: str, text: str, pattern: str, want: int, facts: dict) -> None:
    n = text.count(pattern)
    facts[f"{where}:{pattern.strip()}"] = n
    if n != want:
        raise SystemExit(
            f"e147_rungE2: expected {want} copies of {pattern.strip()!r} "
            f"in {where}, found {n}")


def static_proof(rev: str) -> dict:
    nax_h = show(rev, NAX_H)
    steel = show(rev, STEEL_NAX_H)
    facts: dict = {}

    body = region(
        nax_h,
        "METAL_FUNC void qmm_t_nax_tgp_impl(",
        "template <\n    typename T,\n    const int group_size,\n"
        "    const int bits,\n    const int BM = 64,",
        "qmm_t_nax_tgp_impl")

    # The tile shape is a pure function of the template parameters, and the
    # inner K step is a hard constant that no template parameter can move.
    for pat in ("constexpr short SM = BM / WM;",
                "constexpr short SN = BN / WN;",
                "constexpr short SK = 32;",
                "constexpr short TM = SM / 16;",
                "constexpr short TN = SN / 16;",
                "constexpr short TK = SK / 16;"):
        counted("qmm_t_nax_tgp_impl", body, pat, 1, facts)

    # One accumulate body, shared by both schedules, with the only K loop whose
    # bounds a retile could touch.
    counted("qmm_t_nax_tgp_impl", body,
            "for (int kk1 = 0; kk1 < BK; kk1 += SK) {", 1, facts)
    counted("qmm_t_nax_tgp_impl", body, "STEEL_PRAGMA_NO_UNROLL", 1, facts)
    counted("qmm_t_nax_tgp_impl", body, "tile_matmad_nax(", 1, facts)
    counted("qmm_t_nax_tgp_impl", body, "for (int k = 0; k < K; k += BK) {", 2,
            facts)
    counted("qmm_t_nax_tgp_impl", body, "Dtile.clear();", 1, facts)
    counted("qmm_t_nax_tgp_impl", body, "const int y_row = tid.y * BM;", 1,
            facts)
    counted("qmm_t_nax_tgp_impl", body, "const int y_col = tid.x * BN;", 1,
            facts)

    matmad = region(steel, "METAL_FUNC void tile_matmad_nax(",
                    "\n} // namespace steel", "tile_matmad_nax")
    counted("tile_matmad_nax", matmad, "if constexpr (TN == 1 && TM % 2 == 0)",
            1, facts)
    counted("tile_matmad_nax", matmad, "} else if constexpr (TN % 2 == 0) {", 1,
            facts)
    counted("tile_matmad_nax", matmad, "for (short kk = 0; kk < TK; ++kk)", 2,
            facts)
    counted("tile_matmad_nax", matmad, "else {", 0, facts)

    descs = re.findall(
        r"matmul2d_descriptor\((.*?)\);", steel, re.DOTALL)
    normalized = {" ".join(d.split()) for d in descs}
    facts["matmul2d_descriptor_sites"] = len(descs)
    facts["matmul2d_descriptor_distinct"] = sorted(normalized)

    return facts


def derived() -> dict:
    rows = {}
    for label, bm, bn, _ in PROBES:
        wm = wn = 2
        sm, sn, sk = bm // wm, bn // wn, 32
        tm, tn, tk = sm // 16, sn // 16, sk // 16
        if tn == 1 and tm % 2 == 0:
            branch = "M-paired (TN == 1 && TM % 2 == 0)"
        elif tn % 2 == 0:
            branch = "N-paired (TN % 2 == 0)"
        else:
            branch = "NONE: tile_matmad_nax is a no-op at this shape"
        bk_padded = HOST_BK + 16 // 2  # bfloat16_t
        ws_tile = bn * bk_padded
        ws_bytes = 2 * ws_tile * 2
        rows[label] = {
            "BM": bm, "BN": bn, "BK": HOST_BK, "WM": wm, "WN": wn,
            "SM": sm, "SN": sn, "SK": sk, "TM": tm, "TN": tn, "TK": tk,
            "matmad_branch": branch,
            "Atile_frags": tm * tk,
            "Btile_frags": tn * tk,
            "Dtile_floats_per_lane": tm * tn * 8,
            "Ws_bytes_bf16_doubled": ws_bytes,
            "Ws_pipelined": ws_bytes <= 32768,
            "kk1_steps_per_BK": HOST_BK // sk,
            "equal_area_with_host": bm * bn == HOST_BM * HOST_BN,
            "host_BN_divisible_by_BN": HOST_BN % bn == 0,
        }
    return rows


def census(rev: str, workdir: pathlib.Path) -> dict:
    parts = [twin_source(rev, n) for n in TWINS]
    preamble = "".join(parts)
    out = {}
    for label, bm, bn, role in PROBES:
        host = f"e147_qmm_t_nax_{label}"
        inst = (f"affine_qmm_t_nax<bfloat16_t, 64, 4, 1, 0, "
                f"{bm}, {HOST_BK}, {bn}, 2, 2>")
        source = (preamble + f'\ntemplate [[host_name("{host}")]] [[kernel]] '
                  f"decltype({inst}) {inst};\n")
        d = workdir / label
        d.mkdir(parents=True, exist_ok=True)
        try:
            lib = agx.build_metallib(source, d)
        except subprocess.CalledProcessError as exc:
            out[label] = {"role": role, "instantiation": inst,
                          "compiled": False,
                          "error": exc.stderr.decode(errors="replace")[-2000:]}
            continue
        rec = {"role": role, "instantiation": inst, "compiled": True,
               "metallib_bytes": lib.stat().st_size}
        for arch in ARCHES:
            # This toolchain's offline translator refuses the cooperative
            # tensor operand layouts every real NAX GEMM is built from. That
            # refusal is a recorded observation, not a tool failure. The first
            # line of the diagnostic names the kernel, so only the body below
            # it is comparable between probes.
            try:
                got = agx.translate(lib, arch, d, select=lambda n: n == host)
            except SystemExit as exc:
                text = str(exc)
                rec[arch] = {"translated": False,
                             "error_body": text.split("\n", 1)[-1].strip()}
                continue
            if host not in got:
                raise SystemExit(f"e147_rungE2: {label} missing on {arch}")
            rec[arch] = dict(got[host], translated=True)
        out[label] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    result = {
        "rev": args.rev,
        "commit": subprocess.run(["git", "rev-parse", args.rev], cwd=ROOT,
                                 capture_output=True, text=True,
                                 check=True).stdout.strip(),
        "host_grid": {"BM": HOST_BM, "BN": HOST_BN, "BK": HOST_BK,
                      "editable": False,
                      "source": "backend/metal/quantized.cpp qmm_nax"},
        "static_proof": static_proof(args.rev),
        "derived": derived(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        result["census"] = census(args.rev, pathlib.Path(tmp))

    d = result["derived"]
    c = result["census"]
    shipped, arm, fail = (c["shipped_64x64"], c["retile_128x32"],
                          c["failopen_96x32"])
    result["e147_rungE2_k_order_preserved"] = True
    result["e147_rungE2_matmad_branch_switches"] = (
        d["shipped_64x64"]["matmad_branch"] != d["retile_128x32"]["matmad_branch"])
    result["e147_rungE2_retile_compiles"] = arm["compiled"]
    result["e147_rungE2_failopen_shape_exists"] = (
        d["failopen_96x32"]["matmad_branch"].startswith("NONE"))

    ranked = agx.RANKED_ARCH
    translated = all(rec["compiled"] and rec[ranked].get("translated")
                     for rec in (shipped, arm, fail))
    if translated:
        a, s = arm[ranked], shipped[ranked]
        result[f"e147_rungE2_registers_{ranked}"] = {
            "shipped": s["registers"], "retile": a["registers"],
            "delta": a["registers"] - s["registers"]}
        result[f"e147_rungE2_spill_bytes_{ranked}"] = {
            "shipped": s["spill_bytes"], "retile": a["spill_bytes"]}
        result[f"e147_rungE2_text_bytes_{ranked}"] = {
            "shipped": s["text_bytes"], "retile": a["text_bytes"]}
        result["e147_rungE2_retile_spills"] = a["spill_bytes"] > 0
        result["e147_rungE2_register_delta"] = a["registers"] - s["registers"]
    result["e147_rungE2_register_census_available"] = translated

    # A translation refusal only means something if the SCORED instantiation
    # translates. It does not, on either arch, so the refusal is a limit of
    # this offline translator and says nothing about the retile. The two error
    # bodies must be identical for that reading to hold, because the mangled
    # names they carry describe the descriptor and not the tile shape.
    same_error = {}
    for arch in ARCHES:
        s_arch, a_arch = shipped.get(arch, {}), arm.get(arch, {})
        same_error[arch] = (
            not s_arch.get("translated")
            and not a_arch.get("translated")
            and s_arch.get("error_body") == a_arch.get("error_body"))
    result["e147_rungE2_scored_shape_also_untranslatable"] = not shipped[
        agx.RANKED_ARCH].get("translated", False)
    result["e147_rungE2_retile_refusal_identical_to_scored"] = all(
        same_error.values())

    # THE CONTROL. The only NAX instantiation this translator accepts is the
    # one whose matmad resolves to nothing, and it accepts it because every
    # cooperative-tensor operand disappeared with the arithmetic. A shape that
    # multiplies nothing is therefore visible without an M5.
    control_ok = (
        fail["compiled"]
        and all(fail[arch].get("translated") for arch in ARCHES)
        and not any(shipped[arch].get("translated") for arch in ARCHES)
        and not any(arm[arch].get("translated") for arch in ARCHES)
        and fail["metallib_bytes"] < arm["metallib_bytes"])
    result["e147_rungE2_failopen_control_observed"] = control_ok
    result["e147_rungE2_metallib_bytes"] = {
        label: c[label]["metallib_bytes"] for label, *_ in PROBES}

    verdict = "PASS" if (result["e147_rungE2_retile_compiles"]
                         and result["e147_rungE2_failopen_shape_exists"]
                         and result["e147_rungE2_retile_refusal_identical_to_scored"]
                         and control_ok) else "FAIL"
    result["verdict"] = verdict

    pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("static_proof", "census")}, indent=2))
    for label, rec in c.items():
        if not rec["compiled"]:
            print(f"{label}: DID NOT COMPILE\n{rec['error']}")
            continue
        for arch in ARCHES:
            r = rec[arch]
            if not r.get("translated"):
                print(f"{label:16s} {arch} NOT TRANSLATED")
                continue
            print(f"{label:16s} {arch} regs={r['registers']:4d} "
                  f"spill={r['spill_bytes']:6d} text={r['text_bytes']:8d} "
                  f"sha8={r['text_sha8']}")
    print(f"e147_rungE2_verdict={verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
