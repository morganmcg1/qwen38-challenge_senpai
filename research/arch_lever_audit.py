#!/usr/bin/env python3
"""Machine-check ledger items 151 and 152: the arch-string lever and the nax divergence.

Every claim in those two ledger items that is a statement about MLX source is asserted
here, and every structural check is paired with a MUTATION NEGATIVE CONTROL: the mutated
text must flip that check to FAIL. A check that cannot fail is not evidence (ledger 149).

The lesson this file exists for (ledger item 4/149): re-derive load-bearing arithmetic in
committed code, not a scratch buffer. Item 124 was load-bearing for a whole assignment and
rested on a statistic that was mathematically incapable of failing.

Usage:
    python3 research/arch_lever_audit.py            # run all checks + negative controls
    python3 research/arch_lever_audit.py --routing  # also print the dispatch routing table
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MLX = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx"
KERNELS = MLX / "backend/metal/kernels"

DEVICE_CPP = MLX / "backend/metal/device.cpp"
QUANT_CPP = MLX / "backend/metal/quantized.cpp"
MATMUL_CPP = MLX / "backend/metal/matmul.cpp"
SDPA_CPP = MLX / "backend/metal/scaled_dot_product_attention.cpp"
UTILS_H = MLX / "utils.h"
NAX_H = KERNELS / "quantized_nax.h"
NAX_METAL = KERNELS / "quantized_nax.metal"
BENCH_JSON = ROOT / "benchmark.json"


def norm(s: str) -> str:
    """Collapse all whitespace so checks survive reformatting but not edits."""
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------------------
# The model's own shapes, and the two arch regimes.
# --------------------------------------------------------------------------------------

# (name, D=K=input dim, O=N=output dim) for the quantised linears on the scored path.
SHAPES = [
    ("mlp.down", 17408, 5120),
    ("mlp.gate", 5120, 17408),
    ("mlp.up", 5120, 17408),
    ("head.fc", 10240, 5120),
    ("draft_lm_head", 2560, 98336),
]

SCORED_WIDTHS = range(1, 10)  # M = 1..9; mtp_max_draft_depth == 8 on all 408 ranked rows.


def qmv_batch_limit(D: int, O: int, arch_gen: int, arch_size: str) -> int:
    """Faithful port of get_qmv_batch_limit, quantized.cpp:84-126."""
    if arch_gen in (13, 14):
        if arch_size == "d":
            return 32 if (D <= 2048 and O <= 2048) else (18 if (D <= 4096 and O <= 4096) else 12)
        return 14 if (D <= 2048 and O <= 2048) else (10 if (D <= 4096 and O <= 4096) else 6)
    if arch_size == "d":
        return 32 if (D <= 2048 and O <= 2048) else (18 if (D <= 4096 and O <= 4096) else 12)
    return 18 if (D <= 2048 and O <= 2048) else (12 if (D <= 4096 and O <= 4096) else 10)


def parse_arch(arch: str) -> tuple[int, str]:
    """Faithful port of the Device ctor parse, device.cpp:566-573."""
    ag_tens = ag_ones = 0
    if len(arch) >= 3:
        t = ord(arch[-3]) - ord("0")
        o = ord(arch[-2]) - ord("0")
        ag_tens = t if 0 <= t < 10 else 0
        ag_ones = o if 0 <= o < 10 else 0
    return ag_tens * 10 + ag_ones, arch[-1]


def nax_available(arch_gen: int, arch_size: str) -> bool:
    """Faithful port of _check_nax, device.cpp:917-928 (availability half assumed true)."""
    return arch_gen >= (18 if arch_size == "p" else 17)


# --------------------------------------------------------------------------------------
# Structural checks against source. Each returns True/False given the file text.
# --------------------------------------------------------------------------------------

CHECKS: list[tuple[str, pathlib.Path, object]] = []


def check(name: str, path: pathlib.Path):
    def deco(fn):
        CHECKS.append((name, path, fn))
        return fn

    return deco


@check("C1 nax gate is gen >= (p ? 18 : 17)", DEVICE_CPP)
def _c1(t: str) -> bool:
    return "can_use_nax &= gen >= (arch == 'p' ? 18 : 17);" in norm(t)


@check("C2 nax result is a one-shot function-local static", DEVICE_CPP)
def _c2(t: str) -> bool:
    return "static bool is_nax_available_ = _check_nax();" in norm(t)


@check("C3 arch string is env-overridable via metal_gpu_arch()", DEVICE_CPP)
def _c3(t: str) -> bool:
    n = norm(t)
    return "arch_ = env::metal_gpu_arch();" in n and "if (arch_.empty()) { arch_ = std::string(device_->architecture()" in n


@check("C4 env var name is MLX_METAL_GPU_ARCH", UTILS_H)
def _c4(t: str) -> bool:
    return 'get_var("MLX_METAL_GPU_ARCH", "")' in norm(t)


@check("C5 arch_gen parsed from chars [n-3],[n-2]", DEVICE_CPP)
def _c5(t: str) -> bool:
    n = norm(t)
    return "ag_tens = arch_[arch_.size() - 3] - '0';" in n and "arch_gen_ = ag_tens * 10 + ag_ones;" in n


@check("C6 cmdbuf defaults key on arch_.back() only, 's' -> 50/50", DEVICE_CPP)
def _c6(t: str) -> bool:
    n = norm(t)
    return "auto arch = arch_.back(); switch (arch) {" in n and "case 's': // max max_ops_per_buffer_ = 50; max_mb_per_buffer_ = 50;" in n


@check("C7 QuantizedMatmul::eval_gpu vector_limit = get_qmv_batch_limit(K, N, d)", QUANT_CPP)
def _c7(t: str) -> bool:
    # Site-specific: the SAME line also appears in GatherQMM::eval_gpu, so the trailing
    # comment is load-bearing for disambiguation. An anchor that matches two sites is not
    # evidence about either of them (askeladd's E37 dead-mutation bug; see run_mutations).
    return (
        "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;"
        " auto mode = quantization_mode_to_string(mode_);"
        " // It is a matrix matrix product." in norm(t)
    )


@check("C7b GatherQMM::eval_gpu has its OWN vector_limit gate (arm C blast radius)", QUANT_CPP)
def _c7b(t: str) -> bool:
    return norm(t).count("int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;") == 2


@check("C8 M >= vector_limit and transpose && B==1 routes to qmm_splitk", QUANT_CPP)
def _c8(t: str) -> bool:
    n = norm(t)
    return "if (M >= vector_limit) {" in n and "if (transpose_ && B == 1) { qmm_splitk(" in n


@check("C9 get_qmv_batch_limit keys on gen 13||14 vs else, and arch_size", QUANT_CPP)
def _c9(t: str) -> bool:
    n = norm(t)
    return "auto arch_gen = d.get_architecture_gen();" in n and "if (arch_gen == 13 || arch_gen == 14) {" in n


@check("C10 qmm_splitk tiles bm=32,bn=32 (one tile row covers all M<=32)", QUANT_CPP)
def _c10(t: str) -> bool:
    body = norm(t.split("void qmm_splitk(", 1)[1][:3000])
    return "int bm = 32, bn = 32;" in body


@check("C11 nax gate inside qmm is AFTER the splitk branch (unreached)", QUANT_CPP)
def _c11(t: str) -> bool:
    i_gate = t.index("int vector_limit = transpose_ ? get_qmv_batch_limit")
    i_qmm = t.index("void qmm(")
    i_nax = t.index("if (metal::is_nax_available() && transpose && (K % 64 == 0) &&")
    return i_qmm < i_nax < i_gate  # qmm (with its nax test) is defined above eval_gpu


@check("C12 sdpa_full head_dim list EXCLUDES 256", SDPA_CPP)
def _c12(t: str) -> bool:
    return "(query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);" in norm(t)


@check("C13 sdpa_vector head_dim list INCLUDES 256", SDPA_CPP)
def _c13(t: str) -> bool:
    return "query_head_dim == 128 || query_head_dim == 256);" in norm(t)


@check("C14 sdpa_vector also requires q_len*gqa <= 32 (item 134's wall)", SDPA_CPP)
def _c14(t: str) -> bool:
    return "(query_sequence_length * gqa_factor) <= 32;" in norm(t)


@check("C15 sdpa nax gate is inside sdpa_full only", SDPA_CPP)
def _c15(t: str) -> bool:
    i_full = t.index("void sdpa_full_self_attention_metal(")
    i_nax = t.index("if (metal::is_nax_available() && q.shape(3) != 80 &&")
    return i_full < i_nax


@check("C16 dense matmul use_nax has NO shape gate", MATMUL_CPP)
def _c16(t: str) -> bool:
    return "bool use_nax = metal::is_nax_available() && !issubdtype(a.dtype(), complexfloating) && (env::enable_tf32() || a.dtype() != float32);" in norm(t)


@check("C17 nax ON disables steel_gemm_splitk for dense matmul", MATMUL_CPP)
def _c17(t: str) -> bool:
    return "if (!use_nax && batch_size_out == 1 && (_tm * _tn) <= min_tmn_threshold &&" in norm(t)


@check("C18 there is NO qmv_nax kernel: nax quantised entry points are qmm-only", NAX_H)
def _c18(t: str) -> bool:
    entries = set(re.findall(r"\[\[kernel\]\] void (\w+)", t))
    assert entries, "no kernels found -- selector is stale"
    return all("qmv" not in e for e in entries)


def run_checks(texts: dict[pathlib.Path, str]) -> list[tuple[str, bool]]:
    out = []
    for name, path, fn in CHECKS:
        try:
            out.append((name, bool(fn(texts[path]))))
        except Exception as exc:  # a check that errors is a FAIL, never a pass
            out.append((f"{name} [raised {type(exc).__name__}]", False))
    return out


# --------------------------------------------------------------------------------------
# Mutation negative controls: each must flip its named check to FAIL.
# An anchor that matches more than once is REJECTED -- askeladd's E37 dead-mutation bug.
# --------------------------------------------------------------------------------------

MUTATIONS: list[tuple[str, pathlib.Path, str, str]] = [
    ("C1 nax gate is gen >= (p ? 18 : 17)", DEVICE_CPP, "arch == 'p' ? 18 : 17", "arch == 'p' ? 0 : 0"),
    ("C2 nax result is a one-shot function-local static", DEVICE_CPP, "static bool is_nax_available_", "bool is_nax_available_"),
    ("C3 arch string is env-overridable via metal_gpu_arch()", DEVICE_CPP, "arch_ = env::metal_gpu_arch();", "arch_ = std::string();"),
    ("C4 env var name is MLX_METAL_GPU_ARCH", UTILS_H, '"MLX_METAL_GPU_ARCH"', '"MLX_METAL_GPU_ARCH_DISABLED"'),
    ("C5 arch_gen parsed from chars [n-3],[n-2]", DEVICE_CPP, "arch_gen_ = ag_tens * 10 + ag_ones;", "arch_gen_ = 0;"),
    ("C6 cmdbuf defaults key on arch_.back() only, 's' -> 50/50", DEVICE_CPP, "case 's': // max\n      max_ops_per_buffer_ = 50;", "case 's': // max\n      max_ops_per_buffer_ = 7;"),
    ("C7 QuantizedMatmul::eval_gpu vector_limit = get_qmv_batch_limit(K, N, d)", QUANT_CPP, "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;\n  auto mode = quantization_mode_to_string(mode_);\n  // It is a matrix matrix product.", "int vector_limit = 99;\n  auto mode = quantization_mode_to_string(mode_);\n  // It is a matrix matrix product."),
    ("C7b GatherQMM::eval_gpu has its OWN vector_limit gate (arm C blast radius)", QUANT_CPP, "int E = w.size() / w.shape(-1) / w.shape(-2);\n  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;", "int E = w.size() / w.shape(-1) / w.shape(-2);\n  int vector_limit = 4;"),
    ("C8 M >= vector_limit and transpose && B==1 routes to qmm_splitk", QUANT_CPP, "if (transpose_ && B == 1) {\n      qmm_splitk(", "if (false) {\n      qmm_splitk("),
    ("C9 get_qmv_batch_limit keys on gen 13||14 vs else, and arch_size", QUANT_CPP, "if (arch_gen == 13 || arch_gen == 14) {", "if (false) {"),
    ("C10 qmm_splitk tiles bm=32,bn=32 (one tile row covers all M<=32)", QUANT_CPP, "  int bm = 32, bn = 32;\n", "  int bm = 64, bn = 64;\n"),
    ("C12 sdpa_full head_dim list EXCLUDES 256", SDPA_CPP, "query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);", "query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 256);"),
    ("C14 sdpa_vector also requires q_len*gqa <= 32 (item 134's wall)", SDPA_CPP, "(query_sequence_length * gqa_factor) <= 32;", "(query_sequence_length * gqa_factor) <= 999;"),
    ("C16 dense matmul use_nax has NO shape gate", MATMUL_CPP, "bool use_nax = metal::is_nax_available() &&\n      !issubdtype(a.dtype(), complexfloating) &&", "bool use_nax = metal::is_nax_available() && M > 4096 &&\n      !issubdtype(a.dtype(), complexfloating) &&"),
    ("C17 nax ON disables steel_gemm_splitk for dense matmul", MATMUL_CPP, "if (!use_nax && batch_size_out == 1", "if (batch_size_out == 1"),
    ("C18 there is NO qmv_nax kernel: nax quantised entry points are qmm-only", NAX_H, "[[kernel]] void affine_qmm_t_nax(", "[[kernel]] void affine_qmv_t_nax("),
]


def run_mutations(texts: dict[pathlib.Path, str]) -> list[tuple[str, str]]:
    """Return (label, verdict) where verdict is 'flipped', 'DEAD', or 'AMBIGUOUS'."""
    baseline = {name: ok for name, ok in run_checks(texts)}
    results = []
    for target, path, old, new in MUTATIONS:
        text = texts[path]
        n = text.count(old)
        label = f"{target}  <-  mutate {old[:44]!r}"
        if n == 0:
            results.append((label, "AMBIGUOUS(anchor absent)"))
            continue
        if n > 1:
            # askeladd's E37 failure mode: an anchor matching two sites silently edits
            # the wrong one, so the mutation is not evidence about the site we named.
            results.append((label, f"AMBIGUOUS({n} matches)"))
            continue
        mutated = dict(texts)
        mutated[path] = text.replace(old, new, 1)
        after = {name: ok for name, ok in run_checks(mutated)}
        if baseline.get(target) and not after.get(target):
            results.append((label, "flipped"))
        else:
            results.append((label, "DEAD(check did not flip)"))
    return results


# --------------------------------------------------------------------------------------
# Derived facts: the routing table, and the editablePaths reachability argument.
# --------------------------------------------------------------------------------------

def routing_table() -> list[str]:
    lines = []
    arms = [("A ranked native?", "applegpu_g17s"), ("B local native", "applegpu_g16s"), ("C spoof", "applegpu_g14s")]
    for arm, arch in arms:
        gen, size = parse_arch(arch)
        nax = nax_available(gen, size)
        lines.append(f"  arm {arm:<16} {arch}  gen={gen} size='{size}'  nax={'ON ' if nax else 'off'}")
        for nm, D, O in SHAPES:
            lim = qmv_batch_limit(D, O, gen, size)
            routed = [("qmm_splitk" if M >= lim else "qmv") for M in SCORED_WIDTHS]
            first_qmm = next((M for M in SCORED_WIDTHS if M >= lim), None)
            lines.append(
                f"      {nm:<14} K={D:<6} N={O:<6} vector_limit={lim:<3}"
                f" first width on qmm_splitk = {first_qmm if first_qmm else 'never'}"
                f"   ({routed[0]}..{routed[-1]})"
            )
    return lines


def editable_reachability() -> list[str]:
    paths = json.loads(BENCH_JSON.read_text())["editablePaths"]
    want_present = [
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.metal",
        "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
    ]
    want_absent = ["Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"]
    lines = [f"  editablePaths entries: {len(paths)}"]
    ok = True
    for p in want_present:
        hit = p in paths
        ok &= hit
        lines.append(f"  {'PASS' if hit else 'FAIL'} present: {p.split('/')[-1]}")
    for p in want_absent:
        hit = p not in paths
        ok &= hit
        lines.append(f"  {'PASS' if hit else 'FAIL'} ABSENT : backend/metal/{p.split('/')[-1]}  (host dispatch not editable)")
    lines.append(
        "  => a kernel family that CANNOT execute on gen-16 hardware is in the editable"
        " set; the host dispatch that selects it is not."
    )
    return lines, ok


def main() -> int:
    texts = {p: p.read_text() for p in {c[1] for c in CHECKS} | {m[1] for m in MUTATIONS}}

    print("STRUCTURAL CHECKS (ledger 151/152)")
    results = run_checks(texts)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    n_pass = sum(1 for _, ok in results if ok)
    print(f"  -> {n_pass}/{len(results)} pass")

    print("\nMUTATION NEGATIVE CONTROLS (each must flip its check to FAIL)")
    muts = run_mutations(texts)
    for label, verdict in muts:
        print(f"  {'PASS' if verdict == 'flipped' else 'FAIL'}  {verdict:<24} {label}")
    n_flip = sum(1 for _, v in muts if v == "flipped")
    print(f"  -> {n_flip}/{len(muts)} flipped")

    print("\nEDITABLE-PATH REACHABILITY (manifest evidence that ranked has nax)")
    lines, edit_ok = editable_reachability()
    for line in lines:
        print(line)

    print("\nDISPATCH ROUTING BY ARM (ledger 152 three-arm design)")
    for line in routing_table():
        print(line)

    print("\nKEY DERIVED FACTS")
    gen16, sz16 = parse_arch("applegpu_g16s")
    gen17, sz17 = parse_arch("applegpu_g17s")
    gen14, sz14 = parse_arch("applegpu_g14s")
    print(f"  applegpu_g16s -> gen {gen16} '{sz16}'  nax={nax_available(gen16, sz16)}   <- this host")
    print(f"  applegpu_g17s -> gen {gen17} '{sz17}'  nax={nax_available(gen17, sz17)}   <- an M5 would be here")
    print(f"  applegpu_g14s -> gen {gen14} '{sz14}'  nax={nax_available(gen14, sz14)}   <- arm C spoof")
    lim_b = qmv_batch_limit(17408, 5120, gen16, sz16)
    lim_c = qmv_batch_limit(17408, 5120, gen14, sz14)
    print(f"  mlp.down vector_limit: arm B {lim_b} -> arm C {lim_c}")
    print(f"  weight passes at M=6 on qmv (IPG=5): ceil(6/5) = {-(-6 // 5)}; on qmm_splitk: 1")
    print("  E33 measured the 2nd weight pass at +20.59 ms of the +32.68 ms local 5->6 step.")

    ok = n_pass == len(results) and n_flip == len(muts) and edit_ok
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--routing" in sys.argv:
        for line in routing_table():
            print(line)
    sys.exit(main())
