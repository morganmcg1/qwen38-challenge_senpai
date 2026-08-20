#!/usr/bin/env python3
"""E70 rung 0: audit every kernel-selection site that reads the GPU architecture.

The campaign times every candidate on gen-16 Apple silicon and scores it on a
gen-17 ranked host. MLX picks kernels from the architecture string and the
architecture generation at runtime, so a site that reads either one can send the
two hosts to different kernels. This file answers, for each site:

  1. is the site reached by the scored decode round?
  2. is the site reached by the 512-token seed prefill?
  3. what does each host select?
  4. DIVERGES or IDENTICAL, and which kernel each host runs?

Every claim that is a statement about source is a structural check with a
MUTATION NEGATIVE CONTROL: the mutated text must flip that check to FAIL. A
check that cannot fail is not evidence (ledger 149). Line numbers in the report
are LOCATED from the source, never copied, so the table cannot silently drift.

harness=local for the source facts; the host columns are derived, not measured.
Rung 1 (`Tests/MLXFastTests/E70ArchDispatchProbeTests.swift`) is the empirical
arm.

Usage:
    python3 research/e70_dispatch_divergence_audit.py            # checks + table
    python3 research/e70_dispatch_divergence_audit.py --json OUT # machine record
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MLX = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx"
LM = ROOT / "Vendor/mlx-swift-lm/Libraries"

DEVICE_CPP = MLX / "backend/metal/device.cpp"
QUANT_CPP = MLX / "backend/metal/quantized.cpp"
MATMUL_CPP = MLX / "backend/metal/matmul.cpp"
SDPA_CPP = MLX / "backend/metal/scaled_dot_product_attention.cpp"
FAST_CPP = MLX / "fast.cpp"
UTILS_H = MLX / "utils.h"
ATTN_UTILS = LM / "MLXLMCommon/AttentionUtils.swift"
QWEN35_SWIFT = LM / "MLXLLM/Models/Qwen35.swift"
BENCH_JSON = ROOT / "benchmark.json"

SOURCES = [
    DEVICE_CPP, QUANT_CPP, MATMUL_CPP, SDPA_CPP, FAST_CPP, UTILS_H,
    ATTN_UTILS, QWEN35_SWIFT,
]

# ---------------------------------------------------------------------------
# The three architecture arms.
# ---------------------------------------------------------------------------

ARMS = {
    "local": "applegpu_g16s",       # this campaign's M4 Pro hosts (ledger 182(D))
    "ranked_pro_max": "applegpu_g17s",  # M5 Pro / M5 Max (E65 item 4)
    "ranked_base": "applegpu_g17g",     # base M5 (E65 item 4) -- tier not confirmed
}


def parse_arch(arch: str) -> tuple[int, str]:
    """Faithful port of the Device ctor parse, device.cpp:566-573."""
    ag_tens = ag_ones = 0
    if len(arch) >= 3:
        t = ord(arch[-3]) - ord("0")
        o = ord(arch[-2]) - ord("0")
        ag_tens = t if 0 <= t < 10 else 0
        ag_ones = o if 0 <= o < 10 else 0
    return ag_tens * 10 + ag_ones, arch[-1]


def nax_available(arch_gen: int, devc: str) -> bool:
    """device.cpp:917-928. The macOS >= 26.2 half is assumed satisfied."""
    return arch_gen >= (18 if devc == "p" else 17)


def qmv_batch_limit(D: int, O: int, arch_gen: int, devc: str) -> int:
    """Faithful port of get_qmv_batch_limit, quantized.cpp:84-126."""
    if arch_gen in (13, 14):
        if devc == "d":
            return 32 if (D <= 2048 and O <= 2048) else (18 if (D <= 4096 and O <= 4096) else 12)
        return 14 if (D <= 2048 and O <= 2048) else (10 if (D <= 4096 and O <= 4096) else 6)
    if devc == "d":
        return 32 if (D <= 2048 and O <= 2048) else (18 if (D <= 4096 and O <= 4096) else 12)
    return 18 if (D <= 2048 and O <= 2048) else (12 if (D <= 4096 and O <= 4096) else 10)


def qmm_splitk_k(M: int, N: int, K: int, group_size: int = 64) -> int:
    """Faithful port of qmm_splitk's split_k selection, quantized.cpp:776-807."""
    bm = bn = 32
    n_tiles = (N + bn - 1) // bn
    m_tiles = (M + bm - 1) // bm
    split_k = max(1, 512 // (n_tiles * m_tiles))
    k_align = max(group_size, 32)
    split_k = min(split_k, K // k_align)
    while split_k > 1 and (K % (split_k * k_align) != 0):
        split_k -= 1
    return split_k


# ---------------------------------------------------------------------------
# Scored-path shapes. Source: weights/config.json geometry + the call census in
# Qwen35.swift / Qwen36MTPBlockSession.swift (see the E70 report).
# ---------------------------------------------------------------------------

HIDDEN, VOCAB, HEAD_DIM, Q_HEADS, KV_HEADS = 5120, 248_320, 256, 24, 4
GQA = Q_HEADS // KV_HEADS
SEED_ROWS = 512
DECODE_WIDTHS = range(1, 10)   # segmentedVerifyDepthCap 8 -> M = 1 + draft <= 9

# (label, K, N) for the affine-4 g64 linears the scored worker actually runs.
QUANT_LINEARS = [
    ("gdn.in_proj fused", 5120, 16480),
    ("gdn.out_proj", 6144, 5120),
    ("fa.qkv packed", 5120, 14336),
    ("fa.o_proj", 6144, 5120),
    ("mlp.gate_up fused", 5120, 34816),
    ("mlp.down", 17408, 5120),
    ("lm_head", 5120, VOCAB),
]

# ---------------------------------------------------------------------------
# Structural checks.
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, pathlib.Path, object]] = []


def check(name: str, path: pathlib.Path):
    def deco(fn):
        CHECKS.append((name, path, fn))
        return fn
    return deco


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def line_of(path: pathlib.Path, anchor: str) -> int | str:
    """1-based line of the first anchor occurrence, or a loud marker."""
    text = path.read_text()
    idx = text.find(anchor)
    if idx < 0:
        return "ANCHOR-ABSENT"
    return text.count("\n", 0, idx) + 1


@check("A1 nax master switch is gen >= (p ? 18 : 17)", DEVICE_CPP)
def _a1(t: str) -> bool:
    return "can_use_nax &= gen >= (arch == 'p' ? 18 : 17);" in norm(t)


@check("A2 nax ALSO requires macOS 26.2 at runtime", DEVICE_CPP)
def _a2(t: str) -> bool:
    return "__builtin_available( macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)" in norm(t)


@check("A3 arch string comes from MLX_METAL_GPU_ARCH before the real device", DEVICE_CPP)
def _a3(t: str) -> bool:
    n = norm(t)
    return ("arch_ = env::metal_gpu_arch();" in n
            and "if (arch_.empty()) { arch_ = std::string(device_->architecture()" in n)


@check("A4 arch_gen_ is parsed from the OVERRIDABLE arch_, not the device", DEVICE_CPP)
def _a4(t: str) -> bool:
    n = norm(t)
    return ("ag_tens = arch_[arch_.size() - 3] - '0';" in n
            and "ag_ones = arch_[arch_.size() - 2] - '0';" in n
            and "arch_gen_ = ag_tens * 10 + ag_ones;" in n)


@check("A5 env var is MLX_METAL_GPU_ARCH", UTILS_H)
def _a5(t: str) -> bool:
    return 'get_var("MLX_METAL_GPU_ARCH", "")' in norm(t)


@check("B1 get_qmv_batch_limit branches on gen 13||14 then on arch_size", QUANT_CPP)
def _b1(t: str) -> bool:
    n = norm(t)
    return ("auto arch_size = d.get_architecture().back(); auto arch_gen = d.get_architecture_gen();" in n
            and "if (arch_gen == 13 || arch_gen == 14) {" in n)


@check("B2 QuantizedMatmul::eval_gpu gates qmv on M >= get_qmv_batch_limit(K, N, d)", QUANT_CPP)
def _b2(t: str) -> bool:
    return ("int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;"
            " auto mode = quantization_mode_to_string(mode_);"
            " // It is a matrix matrix product. if (M >= vector_limit) {" in norm(t))


@check("B3 qmm() takes the nax early return when available", QUANT_CPP)
def _b3(t: str) -> bool:
    i_qmm = t.index("void qmm(")
    i_nax = t.index("if (metal::is_nax_available() && transpose && (K % 64 == 0) &&")
    i_splitk = t.index("void qmm_splitk(")
    return i_qmm < i_nax < i_splitk


@check("B4 qmm_splitk falls back to qmm() when split_k <= 1", QUANT_CPP)
def _b4(t: str) -> bool:
    return "if (split_k <= 1) { return qmm( x, w, scales, biases, out, true," in norm(t)


@check("B5 gather_qmm has its own nax gate", QUANT_CPP)
def _b5(t: str) -> bool:
    body = norm(t.split("void gather_qmm(", 1)[1][:1200])
    return "if (metal::is_nax_available() && transpose && (K % 64 == 0)" in body


@check("B6 gather_qmm_rhs has its own nax gate", QUANT_CPP)
def _b6(t: str) -> bool:
    body = norm(t.split("void gather_qmm_rhs(", 1)[1][:1200])
    return "if (metal::is_nax_available() && transpose &&" in body


@check("C1 matmul.cpp devc read at :208 is inside steel_matmul_regular_axpby_NAX", MATMUL_CPP)
def _c1(t: str) -> bool:
    i_nax_fn = t.index("void steel_matmul_regular_axpby_nax(")
    i_plain_fn = t.index("void steel_matmul_regular_axpby(")
    i_devc = t.index("char devc = d.get_architecture().back();")
    return i_nax_fn < i_devc < i_plain_fn


@check("C2 the nax regular kernel is steel_gemm_fused_nax_*", MATMUL_CPP)
def _c2(t: str) -> bool:
    return 'kname << "steel_gemm_fused_nax_"' in t


@check("C3 the non-nax regular kernel is steel_gemm_fused_*", MATMUL_CPP)
def _c3(t: str) -> bool:
    return 'kname << "steel_gemm_fused_"' in t


@check("C4 GEMM_TPARAM_MACRO keys on devc 'g'|'p' vs 'd' vs default", MATMUL_CPP)
def _c4(t: str) -> bool:
    n = norm(t)
    return ("#define GEMM_TPARAM_MACRO(devc) \\ if (devc == 'g' || devc == 'p')" in n
            and "} else if (devc == 'd') {" in n)


@check("C5 steel_matmul_axpby computes use_nax with NO shape gate", MATMUL_CPP)
def _c5(t: str) -> bool:
    return ("bool use_nax = metal::is_nax_available() &&"
            " !issubdtype(a.dtype(), complexfloating) &&"
            " (env::enable_tf32() || a.dtype() != float32);" in norm(t))


@check("C6 min_tmn_threshold is 2048 for 's'|'d' and 1024 otherwise", MATMUL_CPP)
def _c6(t: str) -> bool:
    return "int min_tmn_threshold = (devc == 's' || devc == 'd') ? 2048 : 1024;" in norm(t)


@check("C7 Matmul::eval_gpu routes min(M, N) == 1 to gemv before any arch read", MATMUL_CPP)
def _c7(t: str) -> bool:
    body = norm(t.split("void Matmul::eval_gpu(", 1)[1][:6000])
    return ("if (std::min(M, N) == 1) { return gemv(" in body
            and "d.get_architecture()" not in body)


@check("C8 gather_mm and GatherMM read the arch (blast radius of a MoE model)", MATMUL_CPP)
def _c8(t: str) -> bool:
    gather_mm = norm(t.split("void gather_mm(", 1)[1][:1500])
    gather_eval = norm(t.split("void GatherMM::eval_gpu(", 1)[1][:1500])
    return ("char devc = d.get_architecture().back(); GEMM_TPARAM_MACRO(devc)" in gather_mm
            and "if (metal::is_nax_available() &&" in gather_eval)


@check("C9 segmented_mm reads devc and is_nax_available", MATMUL_CPP)
def _c9(t: str) -> bool:
    body = norm(t.split("void segmented_mm(", 1)[1][:1500])
    return ("char devc = d.get_architecture().back(); GEMM_TPARAM_MACRO(devc)" in body
            and "bool use_nax = metal::is_nax_available() &&" in body)


@check("D1 the sdpa nax gate lives inside sdpa_full_self_attention_metal only", SDPA_CPP)
def _d1(t: str) -> bool:
    i_full = t.index("void sdpa_full_self_attention_metal(")
    i_nax = t.index("if (metal::is_nax_available() && q.shape(3) != 80 &&")
    i_vec = t.index("void sdpa_vector(")
    return i_full < i_nax < i_vec


@check("D2 sdpa_full supports head_dim 64|80|128 only -- 256 EXCLUDED", SDPA_CPP)
def _d2(t: str) -> bool:
    return ("(query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);"
            in norm(t))


@check("D3 sdpa_vector supports head_dim 256 but caps at qL*gqa <= 32", SDPA_CPP)
def _d3(t: str) -> bool:
    n = norm(t)
    return ("query_head_dim == 128 || query_head_dim == 256);" in n
            and "(query_sequence_length * gqa_factor) <= 32;" in n
            and "const bool supports_sdpa_vector = (query_sequence_length <= 8) &&" in n)


@check("D4 use_fallback returns !(full || vector)", SDPA_CPP)
def _d4(t: str) -> bool:
    return "return !(supports_sdpa_full || supports_sdpa_vector);" in norm(t)


@check("D5 sdpa_vector_2pass block count keys on devc", SDPA_CPP)
def _d5(t: str) -> bool:
    body = norm(t.split("void sdpa_vector_2pass(", 1)[1][:3000])
    return ("char devc = d.get_architecture().back();" in body
            and "if (devc == 's') { blocks = 64;" in body
            and "} else if (devc == 'd') { blocks = 128;" in body)


@check("D6 the 2pass route predicate keys on devc and k.shape(2) >= 1024", SDPA_CPP)
def _d6(t: str) -> bool:
    return ("if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||"
            " (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {" in norm(t))


@check("E1 the SDPA fallback is two dense matmul calls", FAST_CPP)
def _e1(t: str) -> bool:
    body = norm(t.split("auto fallback = [scale,", 1)[1][:4000])
    return ("auto scores = matmul(q, swapaxes(k, -1, -2, s), s);" in body
            and "auto out = matmul(scores, v, s);" in body)


@check("E2 the primitive is built only when use_fallback is false", FAST_CPP)
def _e2(t: str) -> bool:
    n = norm(t)
    return ("if (!ScaledDotProductAttention::use_fallback(" in n
            and "return fallback(std::move(inputs))[0];" in n)


@check("F1 the wide-decode chunk keeps every scored decode SDPA at qL <= 5", ATTN_UTILS)
def _f1(t: str) -> bool:
    n = norm(t)
    return ("if queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL, case .causal = mask" in n
            and "let split = 5" in n)


@check("F2 the MoE block is built only when numExperts > 0", QWEN35_SWIFT)
def _f2(t: str) -> bool:
    return "args.numExperts > 0" in t


@check("F3 the bf16 precision-island patch is a dense matmul on the MTP layer", QWEN35_SWIFT)
def _f3(t: str) -> bool:
    n = norm(t)
    return ("let exact = matmul(input, weight.transposed(1, 0))" in n
            and "layer.selfAttn.installExactQKVRows(" in n)


@check("G1 the host dispatchers are NOT in editablePaths", BENCH_JSON)
def _g1(t: str) -> bool:
    paths = json.loads(t)["editablePaths"]
    host = [
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/matmul.cpp",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp",
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/fast.cpp",
    ]
    return all(p not in paths for p in host)


@check("G2 the callers that CHOOSE the shapes ARE editable", BENCH_JSON)
def _g2(t: str) -> bool:
    paths = json.loads(t)["editablePaths"]
    callers = [
        "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift",
        "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
    ]
    return all(p in paths for p in callers)


def run_checks(texts: dict[pathlib.Path, str]) -> list[tuple[str, bool]]:
    out = []
    for name, path, fn in CHECKS:
        try:
            out.append((name, bool(fn(texts[path]))))
        except Exception as exc:
            out.append((f"{name} [raised {type(exc).__name__}]", False))
    return out


MUTATIONS: list[tuple[str, pathlib.Path, str, str]] = [
    ("A1 nax master switch is gen >= (p ? 18 : 17)", DEVICE_CPP,
     "gen >= (arch == 'p' ? 18 : 17)", "gen >= (arch == 'p' ? 0 : 0)"),
    ("A2 nax ALSO requires macOS 26.2 at runtime", DEVICE_CPP,
     "macOS 26.2, iOS 26.2", "macOS 99.9, iOS 26.2"),
    ("A3 arch string comes from MLX_METAL_GPU_ARCH before the real device", DEVICE_CPP,
     "arch_ = env::metal_gpu_arch();", "arch_ = std::string();"),
    ("A4 arch_gen_ is parsed from the OVERRIDABLE arch_, not the device", DEVICE_CPP,
     "arch_gen_ = ag_tens * 10 + ag_ones;", "arch_gen_ = 16;"),
    ("A5 env var is MLX_METAL_GPU_ARCH", UTILS_H,
     '"MLX_METAL_GPU_ARCH"', '"MLX_METAL_GPU_ARCH_OFF"'),
    ("B1 get_qmv_batch_limit branches on gen 13||14 then on arch_size", QUANT_CPP,
     "if (arch_gen == 13 || arch_gen == 14) {", "if (false) {"),
    ("B2 QuantizedMatmul::eval_gpu gates qmv on M >= get_qmv_batch_limit(K, N, d)", QUANT_CPP,
     "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;\n"
     "  auto mode = quantization_mode_to_string(mode_);\n"
     "  // It is a matrix matrix product.",
     "int vector_limit = 4;\n  auto mode = quantization_mode_to_string(mode_);\n"
     "  // It is a matrix matrix product."),
    ("B4 qmm_splitk falls back to qmm() when split_k <= 1", QUANT_CPP,
     "if (split_k <= 1) {\n    return qmm(", "if (false) {\n    return qmm("),
    ("C2 the nax regular kernel is steel_gemm_fused_nax_*", MATMUL_CPP,
     'kname << "steel_gemm_fused_nax_"', 'kname << "steel_gemm_fused_NOPE_"'),
    ("C5 steel_matmul_axpby computes use_nax with NO shape gate", MATMUL_CPP,
     "bool use_nax = metal::is_nax_available() &&\n"
     "      !issubdtype(a.dtype(), complexfloating) &&",
     "bool use_nax = metal::is_nax_available() && M > 4096 &&\n"
     "      !issubdtype(a.dtype(), complexfloating) &&"),
    ("C6 min_tmn_threshold is 2048 for 's'|'d' and 1024 otherwise", MATMUL_CPP,
     "int min_tmn_threshold = (devc == 's' || devc == 'd') ? 2048 : 1024;",
     "int min_tmn_threshold = 1024;"),
    ("C7 Matmul::eval_gpu routes min(M, N) == 1 to gemv before any arch read", MATMUL_CPP,
     "  if (std::min(M, N) == 1) {\n    return gemv(",
     "  if (false) {\n    return gemv("),
    ("D2 sdpa_full supports head_dim 64|80|128 only -- 256 EXCLUDED", SDPA_CPP,
     "query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);",
     "query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 256);"),
    ("D3 sdpa_vector supports head_dim 256 but caps at qL*gqa <= 32", SDPA_CPP,
     "(query_sequence_length * gqa_factor) <= 32;",
     "(query_sequence_length * gqa_factor) <= 999;"),
    ("D5 sdpa_vector_2pass block count keys on devc", SDPA_CPP,
     "  if (devc == 's') {\n    blocks = 64;", "  if (false) {\n    blocks = 64;"),
    ("D6 the 2pass route predicate keys on devc and k.shape(2) >= 1024", SDPA_CPP,
     "if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||",
     "if (((devc == 'd' || devc == 's') && k.shape(2) >= 999999) ||"),
    ("E1 the SDPA fallback is two dense matmul calls", FAST_CPP,
     "auto out = matmul(scores, v, s);", "auto out = scores;"),
    ("F1 the wide-decode chunk keeps every scored decode SDPA at qL <= 5", ATTN_UTILS,
     "let split = 5", "let split = 7"),
    ("F2 the MoE block is built only when numExperts > 0", QWEN35_SWIFT,
     "args.numExperts > 0", "args.numExperts >= 0"),
    ("F3 the bf16 precision-island patch is a dense matmul on the MTP layer", QWEN35_SWIFT,
     "let exact = matmul(input, weight.transposed(1, 0))", "let exact = input"),
    ("G1 the host dispatchers are NOT in editablePaths", BENCH_JSON,
     '"Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"',
     '"Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/matmul.cpp"'),
    ("G2 the callers that CHOOSE the shapes ARE editable", BENCH_JSON,
     '"Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift"',
     '"Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtilsX.swift"'),
]


def run_mutations(texts: dict[pathlib.Path, str]) -> list[tuple[str, str]]:
    baseline = dict(run_checks(texts))
    results = []
    for target, path, old, new in MUTATIONS:
        text = texts[path]
        n = text.count(old)
        label = f"{target}  <-  mutate {old[:40]!r}"
        if n == 0:
            results.append((label, "AMBIGUOUS(anchor absent)"))
            continue
        if n > 1:
            results.append((label, f"AMBIGUOUS({n} matches)"))
            continue
        mutated = dict(texts)
        mutated[path] = text.replace(old, new, 1)
        after = dict(run_checks(mutated))
        if baseline.get(target) and not after.get(target):
            results.append((label, "flipped"))
        else:
            results.append((label, "DEAD(check did not flip)"))
    return results


# ---------------------------------------------------------------------------
# The site table. `decode` / `prefill` are proofs, not assumptions: each cites
# the operation and the call path that reaches the site, or states why nothing
# reaches it.
# ---------------------------------------------------------------------------

def site_table() -> list[dict]:
    L = line_of
    sites: list[dict] = []

    sites.append(dict(
        id="S1", file="device.cpp",
        line=L(DEVICE_CPP, "bool is_nax_available() {"),
        predicate="can_use_nax &= gen >= (arch == 'p' ? 18 : 17);  // AND __builtin_available(macOS 26.2)",
        decode="YES - the value is read by every nax gate below; first read on the first quantized matmul of round 1.",
        prefill="YES - same switch.",
        local="applegpu_g16s, gen 16, devc 's' -> is_nax_available() == false",
        ranked="applegpu_g17s, gen 17, devc 's' -> true (if the runner's macOS >= 26.2)",
        base_m5="applegpu_g17g, gen 17, devc 'g' -> true (threshold is 17 for every devc except 'p')",
        verdict="DIVERGES",
        kernels="master switch; selects the _nax family wherever a gate below reads it",
    ))
    sites.append(dict(
        id="S2", file="device.cpp",
        line=L(DEVICE_CPP, "  arch_gen_ = ag_tens * 10 + ag_ones;"),
        predicate="arch_ = env::metal_gpu_arch(); ... arch_gen_ = ag_tens * 10 + ag_ones; auto arch = arch_.back();",
        decode="YES - command-buffer budget, set once in the Device ctor.",
        prefill="YES - same.",
        local="'s' -> max_ops_per_buffer 50, max_mb_per_buffer 50",
        ranked="'s' -> 50 / 50",
        base_m5="'g' -> 40 / 40",
        verdict="IDENTICAL (M5 Pro/Max); DIVERGES vs a base M5",
        kernels="no kernel; buffer geometry only (E58 rung 2 measured this lever)",
    ))
    sites.append(dict(
        id="S3", file="quantized.cpp",
        line=L(QUANT_CPP, "inline int get_qmv_batch_limit("),
        predicate="if (arch_gen == 13 || arch_gen == 14) {...} else { switch (arch_size) { case 'd': ...; default: ... } }",
        decode="YES - every affine-4 linear, M = 1..9.",
        prefill="YES - every affine-4 linear, M = 512.",
        local="gen 16 -> else branch, devc 's' -> default arm, D = 5120 > 4096 -> vector_limit 10",
        ranked="gen 17 -> else branch, devc 's' -> default arm -> vector_limit 10",
        base_m5="gen 17, devc 'g' -> default arm -> vector_limit 10",
        verdict="IDENTICAL",
        kernels="M <= 9 -> dispatch_qmv on both hosts; the cliff is at M = 10",
    ))
    sites.append(dict(
        id="S4", file="quantized.cpp",
        line=L(QUANT_CPP, "  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&"),
        predicate="if (metal::is_nax_available() && transpose && (K % 64 == 0) && (env::enable_tf32() || x.dtype() != float32)) return qmm_nax(...)",
        decode="YES, once per leg - the MTP head history flush on the first drafting round runs M = 511 rows through the head K/V pack, and 511 >= vector_limit 10 -> qmm_splitk -> split_k == 1 -> qmm(). Every other decode matmul has M <= 9 and never reaches qmm().",
        prefill="YES - all 7 linear families at M = 512.",
        local="affine_qmm_t_bfloat16_gs_64_b_4_alN_true_batch_0 (bm32 bn32 wm2 wn2)",
        ranked="affine_qmm_t_nax_bfloat16_gs_64_b_4_bm64_bn64_bk64_wm2_wn2",
        base_m5="same as ranked - this gate reads no devc",
        verdict="DIVERGES",
        kernels="qmm_t vs qmm_t_nax",
    ))
    sites.append(dict(
        id="S5", file="quantized.cpp",
        line=L(QUANT_CPP, "void gather_qmm("),
        predicate="if (metal::is_nax_available() && transpose && (K % 64 == 0) && ...) return gather_qmm_nax(...)",
        decode="NO - GatherQMM enters the graph only through SwitchGLU -> Qwen35SparseMoeBlock, built only under `args.numExperts > 0`. weights/config.json has no num_experts key and the decoder default is 0.",
        prefill="NO - same.",
        local="unreached", ranked="unreached", base_m5="unreached",
        verdict="IDENTICAL (unreachable)",
        kernels="none",
    ))
    sites.append(dict(
        id="S6", file="quantized.cpp",
        line=L(QUANT_CPP, "void gather_qmm_rhs("),
        predicate="if (metal::is_nax_available() && transpose && ...) return gather_qmm_rhs_nax(...)",
        decode="NO - same MoE chain as S5.",
        prefill="NO - same.",
        local="unreached", ranked="unreached", base_m5="unreached",
        verdict="IDENTICAL (unreachable)",
        kernels="none",
    ))
    sites.append(dict(
        id="S7", file="matmul.cpp",
        line=L(MATMUL_CPP, "void steel_matmul_regular_axpby_nax("),
        predicate="char devc = d.get_architecture().back(); if (devc == 's' || 'c' || 'd') { bk = (K >= 8192 && K > (M + N)) ? 64 : 256; bm = 64; wm = 2; }",
        decode="Ranked only, once per leg: the same M = 511 head flush also runs the bf16 precision-island patch `matmul(input, weight.T)` (Qwen35.swift), and min(M, N) > 1 -> steel_matmul. Locally this function is unreachable (use_nax false).",
        prefill="Ranked only: the 16 full-attention layers run the SDPA fallback as two dense bf16 GEMMs (fast.cpp), batch 24.",
        local="UNREACHABLE - guarded by use_nax",
        ranked="devc 's' -> bm 64, bn 128, bk 256, wm 2, wn 4 -> steel_gemm_fused_nax_*",
        base_m5="devc 'g' -> keeps bm 128, bn 128, bk 512, wm 4, wn 4 -> a DIFFERENT nax kernel",
        verdict="DIVERGES (ranked-only site)",
        kernels="steel_gemm_fused_nax_{nt,nn}_bfloat16_bfloat16_bm..._bn..._bk..._wm..._wn...",
    ))
    sites.append(dict(
        id="S8", file="matmul.cpp",
        line=L(MATMUL_CPP, "void steel_matmul_regular_axpby(") + 32,
        predicate="char devc = d.get_architecture().back(); GEMM_TPARAM_MACRO(devc)",
        decode="Local only, once per leg (the M = 511 island patch). On the ranked host S7 runs instead.",
        prefill="Local only: the 32 SDPA-fallback GEMMs. On the ranked host S7 runs instead.",
        local="devc 's' -> macro leaves bm 64, bn 64, bk 16, wm 2, wn 2 -> steel_gemm_fused_*",
        ranked="not reached (use_nax true routes to S7)",
        base_m5="not reached (use_nax true routes to S7)",
        verdict="DIVERGES (local-only site)",
        kernels="steel_gemm_fused_{nt,nn}_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2",
    ))
    sites.append(dict(
        id="S9", file="matmul.cpp",
        line=L(MATMUL_CPP, "  bool use_nax = metal::is_nax_available() &&"),
        predicate="bool use_nax = is_nax_available() && !complex && (tf32 || dtype != f32); char devc = ...; int min_tmn_threshold = (devc == 's' || devc == 'd') ? 2048 : 1024; if (!use_nax && batch_size_out == 1 && ...) splitk",
        decode="YES, once per leg - the M = 511 island patch reaches steel_matmul_axpby. Every M = 1 dense head matmul stops earlier at the gemv route (min(M, N) == 1) and never reads the arch.",
        prefill="YES - the 32 SDPA-fallback GEMMs (batch_size_out 24, so both split-K arms are skipped).",
        local="use_nax false -> steel_matmul_regular_axpby (S8)",
        ranked="use_nax true -> steel_matmul_regular_axpby_nax (S7); the !use_nax split-K arm is also disabled",
        base_m5="use_nax true; min_tmn_threshold 1024 instead of 2048 (dead while use_nax is true)",
        verdict="DIVERGES",
        kernels="the dense-GEMM family selector: steel_gemm_fused_* vs steel_gemm_fused_nax_*",
    ))
    sites.append(dict(
        id="S10", file="matmul.cpp",
        line=L(MATMUL_CPP, "void gather_mm(") + 27,
        predicate="char devc = d.get_architecture().back(); GEMM_TPARAM_MACRO(devc)",
        decode="NO - GatherMM enters the graph only through SwitchGLU (MoE). Same chain as S5.",
        prefill="NO.",
        local="unreached", ranked="unreached", base_m5="unreached",
        verdict="IDENTICAL (unreachable)",
        kernels="none",
    ))
    sites.append(dict(
        id="S11", file="matmul.cpp",
        line=L(MATMUL_CPP, "void GatherMM::eval_gpu(") + 28,
        predicate="if (metal::is_nax_available() && (env::enable_tf32() || a.dtype() != float32)) return gather_mm_rhs_nax(...)",
        decode="NO - same MoE chain.",
        prefill="NO.",
        local="unreached", ranked="unreached", base_m5="unreached",
        verdict="IDENTICAL (unreachable)",
        kernels="none",
    ))
    sites.append(dict(
        id="S12", file="matmul.cpp",
        line=L(MATMUL_CPP, "void segmented_mm(") + 49,
        predicate="char devc = ...; GEMM_TPARAM_MACRO(devc); bool use_nax = is_nax_available() && (tf32 || out.dtype() != f32)",
        decode="NO - the SegmentedMM primitive has no caller in the scored graph (no segmented_mm op is built by the Qwen path).",
        prefill="NO.",
        local="unreached", ranked="unreached", base_m5="unreached",
        verdict="IDENTICAL (unreachable)",
        kernels="none",
    ))
    sites.append(dict(
        id="S13", file="scaled_dot_product_attention.cpp",
        line=L(SDPA_CPP, "  if (metal::is_nax_available() && q.shape(3) != 80 &&"),
        predicate="if (metal::is_nax_available() && q.shape(3) != 80 && (tf32 || q.dtype() != f32)) return sdpa_full_self_attention_nax(...)",
        decode="NO. The gate is inside sdpa_full_self_attention_metal, reached only from the full-attention branch of eval_gpu, which requires the fused primitive to exist. use_fallback keeps sdpa_full_supported_head_dim in {64, 80, 128} and this model is head_dim 256, so supports_sdpa_full is false at EVERY width. The primitive is built only via supports_sdpa_vector, which needs qL <= 8, hence the vector branch always.",
        prefill="NO - at qL = 512 both supports_* are false, so no primitive is built at all and MLX runs the composed matmul fallback instead (S9).",
        local="unreached", ranked="unreached", base_m5="unreached",
        verdict="IDENTICAL (unreachable)",
        kernels="none - the 16 full-attention layers never touch the steel attention family on either host",
    ))
    sites.append(dict(
        id="S14", file="scaled_dot_product_attention.cpp",
        line=L(SDPA_CPP, "  char devc = d.get_architecture().back();\n  int N = k.shape(2);"),
        predicate="char devc = ...; if (devc == 's') { blocks = 64; if (N > 1024 && n_simds > 4) {...} } else if (devc == 'd') {...} else { blocks = n_simds >= 4 ? 64 : 32; }",
        decode="YES, in the last rounds only - 2pass is selected once kL >= 1024, and the 512-seed + 512-decode window reaches kL = 1024 exactly at the end.",
        prefill="NO - prefill takes no fused SDPA route at all (S13).",
        local="devc 's', N = 1024 so `N > 1024` is false -> blocks = 64",
        ranked="devc 's' -> blocks = 64",
        base_m5="devc 'g' -> else arm; and S15 would not route to 2pass at all",
        verdict="IDENTICAL (M5 Pro/Max); DIVERGES vs a base M5",
        kernels="sdpa_vector_2pass_1_bfloat16_256_256 / _2_",
    ))
    sites.append(dict(
        id="S15", file="scaled_dot_product_attention.cpp",
        line=L(SDPA_CPP, "    if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||"),
        predicate="if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) || (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) sdpa_vector_2pass else sdpa_vector",
        decode="YES - all 16 full-attention layers, every round, at qL <= 5 after the exactness chunk.",
        prefill="NO (S13).",
        local="devc 's' -> 2pass once kL >= 1024",
        ranked="devc 's' -> 2pass once kL >= 1024",
        base_m5="devc 'g' -> the first arm fails; the gqa arm needs kL >= 4096, never reached -> sdpa_vector for the whole leg",
        verdict="IDENTICAL (M5 Pro/Max); DIVERGES vs a base M5",
        kernels="sdpa_vector_bfloat16_256_256 vs sdpa_vector_2pass_*",
    ))
    return sites


def derived_facts() -> list[str]:
    out = []
    for arm, arch in ARMS.items():
        gen, devc = parse_arch(arch)
        out.append(f"  {arm:<15} {arch}  gen={gen} devc='{devc}'  nax={nax_available(gen, devc)}")
    out.append("")
    out.append("  QMV/QMM route by width (vector_limit from get_qmv_batch_limit):")
    for label, K, N in QUANT_LINEARS:
        lims = {arm: qmv_batch_limit(K, N, *parse_arch(arch)) for arm, arch in ARMS.items()}
        same = len(set(lims.values())) == 1
        out.append(f"    {label:<20} K={K:<6} N={N:<7} limits={lims}  "
                   f"{'IDENTICAL' if same else 'DIVERGES'}")
    out.append("")
    out.append("  Widths that reach qmm() (and therefore the nax gate S4):")
    for label, K, N in QUANT_LINEARS:
        lim = qmv_batch_limit(K, N, 16, "s")
        hits = [M for M in DECODE_WIDTHS if M >= lim]
        out.append(f"    {label:<20} decode M=1..9 -> {hits if hits else 'none (all qmv)'};"
                   f" prefill M=512 -> qmm_splitk split_k="
                   f"{qmm_splitk_k(SEED_ROWS, N, K)} -> "
                   f"{'qmm()' if qmm_splitk_k(SEED_ROWS, N, K) <= 1 else 'qmm_t_splitk'}")
    out.append("")
    out.append("  MTP head history flush (first drafting round), M = 511 rows:")
    for label, K, N in [("head.kv pack", 5120, 2048), ("head.qkv pack", 5120, 14336)]:
        sk = qmm_splitk_k(511, N, K)
        out.append(f"    {label:<20} K={K} N={N} split_k={sk} -> "
                   f"{'qmm() -> nax gate S4' if sk <= 1 else 'qmm_t_splitk'}")
    out.append("")
    out.append(f"  SDPA fused-vector cap: qL * gqa <= 32 with gqa={GQA} -> qL <= {32 // GQA}")
    out.append(f"  head_dim {HEAD_DIM} is excluded from sdpa_full {{64, 80, 128}} -> "
               "supports_sdpa_full is false at every width")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    texts = {p: p.read_text() for p in SOURCES + [BENCH_JSON]}

    print("STRUCTURAL CHECKS (E70 rung 0)")
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

    print("\nDERIVED ARCH FACTS")
    for line in derived_facts():
        print(line)

    sites = site_table()
    print("\nSITE TABLE")
    for s in sites:
        print(f"  {s['id']:<4} {s['file']}:{s['line']:<6} {s['verdict']}")
    absent = [s["id"] for s in sites if s["line"] == "ANCHOR-ABSENT"]
    if absent:
        print(f"  !! anchors not found for: {absent}")

    diverging = [s for s in sites if s["verdict"].startswith("DIVERGES")]
    print(f"\n  {len(diverging)} of {len(sites)} sites diverge on the M5 Pro/Max arm: "
          f"{[s['id'] for s in diverging]}")

    if args.json:
        args.json.write_text(json.dumps({
            "checks": [{"name": n, "pass": ok} for n, ok in results],
            "mutations": [{"label": l, "verdict": v} for l, v in muts],
            "arms": {a: dict(zip(("gen", "devc"), parse_arch(x))) | {
                "arch": x, "nax": nax_available(*parse_arch(x))} for a, x in ARMS.items()},
            "sites": sites,
        }, indent=2) + "\n")
        print(f"\n  wrote {args.json}")

    ok = n_pass == len(results) and n_flip == len(muts) and not absent
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
