#!/usr/bin/env python3
"""e18 Phase 1: reproduce the MLX quantized-matmul dispatcher in Python.

Every function below is a line-for-line port of
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp` at the
campaign base. Line references are to that file. The port exists so the
`split_k` claim in the e18 assignment can be checked arithmetically without a
GPU, and so the per-projection dispatch table is a committed artifact rather
than prose.

Run:  python3 research/e18_dispatch_table.py
      python3 research/e18_dispatch_table.py --json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# Host description
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Host:
    """The two `metal::Device` fields the dispatcher reads.

    `device.cpp:560-573` parses `arch_` as `<...><tens><ones><size>`, e.g.
    `applegpu_g16s` -> gen 16, size 's'.
    """

    name: str
    architecture: str
    macos_at_least_26_2: bool

    @property
    def arch_size(self) -> str:
        return self.architecture[-1]

    @property
    def arch_gen(self) -> int:
        return int(self.architecture[-3:-1])

    def is_nax_available(self) -> bool:
        """`device.cpp:913-932`.

        `MLX_METAL_NO_NAX` is a CMake-only `target_compile_definitions(mlx
        PRIVATE ...)` at `kernels/CMakeLists.txt:176`. SwiftPM compiles
        `device.cpp` in the `Cmlx` target, which does not receive it
        (`Vendor/mlx-swift/Package.swift:218-222,307`), so the runtime branch
        is what executes in every build this campaign measures.
        """
        return self.macos_at_least_26_2 and self.arch_gen >= (
            18 if self.arch_size == "p" else 17
        )


LOCAL_M4_PRO = Host("local AWS M4 Pro 48GB", "applegpu_g16s", True)

# The ranked box is documented only as "Apple M5 Max" on label
# `m5-qwen38-27b-mtp` (docs/qwen-mtp-go-live-runbook.md:65-66). Its
# `applegpu_*` string and OS version are NOT recorded anywhere in this
# repository, so both rows below are labelled hypotheses, not facts.
RANKED_IF_G17S = Host("ranked M5 Max IF applegpu_g17s + macOS>=26.2", "applegpu_g17s", True)
RANKED_IF_G17P = Host("ranked M5 Max IF applegpu_g17p + macOS>=26.2", "applegpu_g17p", True)
RANKED_IF_OLD_OS = Host("ranked M5 Max IF applegpu_g17s + macOS<26.2", "applegpu_g17s", False)


def get_qmv_batch_limit(D: int, O: int, host: Host) -> int:
    """`quantized.cpp:84-124`."""
    if host.arch_gen in (13, 14):
        if host.arch_size == "d":
            return 32 if (D <= 2048 and O <= 2048) else (18 if (D <= 4096 and O <= 4096) else 12)
        return 14 if (D <= 2048 and O <= 2048) else (10 if (D <= 4096 and O <= 4096) else 6)
    if host.arch_size == "d":
        return 32 if (D <= 2048 and O <= 2048) else (18 if (D <= 4096 and O <= 4096) else 12)
    return 18 if (D <= 2048 and O <= 2048) else (12 if (D <= 4096 and O <= 4096) else 10)


# --------------------------------------------------------------------------
# Kernel-name construction
# --------------------------------------------------------------------------


def name_qmm_nax(mode, dtype, gs, bits, N, transpose, batched) -> str:
    """`quantized.cpp:499-522`. bm=bn=bk=64, wm=wn=2 are hard-coded at :491-495."""
    aligned = N % 64 == 0
    core = "_qmm_t_nax_" if transpose else "_qmm_n_nax_"
    tail = ("_alN_true" if aligned else "_alN_false") if transpose else ""
    return (
        f"{mode}{core}{dtype}_gs_{gs}_b_{bits}_bm64_bn64_bk64_wm2_wn2"
        f"{tail}{'_batch_1' if batched else '_batch_0'}"
    )


def name_qmm(mode, dtype, gs, bits, N, transpose, batched) -> str:
    """`quantized.cpp:722-735`. bm=bn=32, wm=wn=2 at :717-720."""
    aligned = N % 32 == 0
    core = "_qmm_t_" if transpose else "_qmm_n_"
    tail = ("_alN_true" if aligned else "_alN_false") if transpose else ""
    return f"{mode}{core}{dtype}_gs_{gs}_b_{bits}{tail}{'_batch_1' if batched else '_batch_0'}"


def name_qmm_t_splitk(mode, dtype, gs, bits, N) -> str:
    """`quantized.cpp:834-844`."""
    aligned = N % 32 == 0
    return (
        f"{mode}_qmm_t_splitk_{dtype}_gs_{gs}_b_{bits}"
        f"{'_alN_true' if aligned else '_alN_false'}"
    )


def name_qmv(mode, dtype, gs, bits, N, K, batched) -> str:
    """`quantized.cpp:259-269`. bn=8 at :252."""
    fast = (N % 8 == 0) and (K % 512 == 0)
    return (
        f"{mode}{'_qmv_fast_' if fast else '_qmv_'}{dtype}_gs_{gs}_b_{bits}"
        f"{'_batch_1' if batched else '_batch_0'}"
    )


def name_qmv_quad(mode, dtype, gs, bits, batched) -> str:
    """`quantized.cpp:200-...`; only reachable when K in {64, 128}."""
    return f"{mode}_qmv_quad_{dtype}_gs_{gs}_b_{bits}{'_batch_1' if batched else '_batch_0'}"


# --------------------------------------------------------------------------
# The dispatcher
# --------------------------------------------------------------------------


@dataclass
class Decision:
    label: str
    M: int
    N: int
    K: int
    transpose: bool
    B: int
    vector_limit: int
    branch: str = ""
    kernel: str = ""
    # split-K bookkeeping (`quantized.cpp:790-810`)
    n_tiles: int | None = None
    m_tiles: int | None = None
    current_tgs: int | None = None
    split_k_target: int | None = None
    k_align: int | None = None
    split_k: int | None = None
    grid: tuple | None = None
    intermediate_elems: int | None = None
    notes: list = field(default_factory=list)


def dispatch(
    label: str,
    M: int,
    N: int,
    K: int,
    host: Host,
    *,
    transpose: bool = True,
    B: int = 1,
    dtype: str = "bfloat16",
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
) -> Decision:
    """`QuantizedMatmul::eval_gpu`, `quantized.cpp:1393-1461`."""
    vector_limit = get_qmv_batch_limit(K, N, host) if transpose else 4
    d = Decision(label, M, N, K, transpose, B, vector_limit)

    if M >= vector_limit:
        if transpose and B == 1:
            return _qmm_splitk(d, host, dtype, group_size, bits, mode)
        d.branch = "qmm"
        return _qmm(d, host, dtype, group_size, bits, mode)

    # vector branches
    if transpose:
        if K in (64, 128) and bits & (bits - 1) == 0:
            d.branch = "dispatch_qmv -> qmv_quad"
            d.kernel = name_qmv_quad(mode, dtype, group_size, bits, B > 1)
        else:
            d.branch = "dispatch_qmv -> qmv"
            d.kernel = name_qmv(mode, dtype, group_size, bits, N, K, B > 1)
        return d
    if K < 1024:
        d.branch = "qvm"
    else:
        d.branch = "qvm_split_k"
    d.kernel = f"{mode}_{d.branch}_{dtype}_gs_{group_size}_b_{bits}"
    return d


def _qmm(d: Decision, host: Host, dtype, group_size, bits, mode) -> Decision:
    """`quantized.cpp:682-771`, including the NAX early-return at :697-698."""
    tf32_or_not_fp32 = dtype != "float32"  # env::enable_tf32() defaults false
    if host.is_nax_available() and d.transpose and (d.K % 64 == 0) and tf32_or_not_fp32:
        d.branch += " -> qmm_nax"
        d.kernel = name_qmm_nax(mode, dtype, group_size, bits, d.N, d.transpose, d.B > 1)
        d.grid = (math.ceil(d.N / 64), math.ceil(d.M / 64), d.B)
        return d
    d.kernel = name_qmm(mode, dtype, group_size, bits, d.N, d.transpose, d.B > 1)
    d.grid = (math.ceil(d.N / 32), math.ceil(d.M / 32), d.B)
    return d


def _qmm_splitk(d: Decision, host: Host, dtype, group_size, bits, mode) -> Decision:
    """`quantized.cpp:776-873`."""
    bm = bn = 32
    d.n_tiles = (d.N + bn - 1) // bn
    d.m_tiles = (d.M + bm - 1) // bm
    d.current_tgs = d.n_tiles * d.m_tiles
    split_k = max(1, 512 // d.current_tgs)
    d.split_k_target = split_k
    d.k_align = max(group_size, 32)
    split_k = min(split_k, d.K // d.k_align)
    while split_k > 1 and (d.K % (split_k * d.k_align) != 0):
        split_k -= 1
    d.split_k = split_k
    if split_k <= 1:
        d.branch = "qmm_splitk -> qmm"
        return _qmm(d, host, dtype, group_size, bits, mode)
    d.branch = "qmm_splitk"
    d.kernel = name_qmm_t_splitk(mode, dtype, group_size, bits, d.N)
    d.grid = (d.n_tiles, d.m_tiles, split_k)
    d.intermediate_elems = split_k * d.M * d.N
    d.notes.append(
        "allocates a split_k*M*N intermediate and runs a strided sum reduction "
        "(quantized.cpp:817-873); NAX is never consulted on this path"
    )
    return d


# --------------------------------------------------------------------------
# Model shapes
# --------------------------------------------------------------------------

H = 5120            # hidden_size
INTERMEDIATE = 17408
VOCAB = 248320
LINEAR_LAYERS = 48
FULL_LAYERS = 16
LAYERS = LINEAR_LAYERS + FULL_LAYERS
LINEAR_KEY_HEADS = 16       # Qwen35Config.swift:245
LINEAR_VALUE_HEADS = 48     # Qwen35Config.swift:244
LINEAR_KEY_HEAD_DIM = 128   # Qwen35Config.swift:247
LINEAR_VALUE_HEAD_DIM = 128  # Qwen35Config.swift:246
ATTN_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256

LINEAR_CONV_SIZE = 2 * LINEAR_KEY_HEADS * LINEAR_KEY_HEAD_DIM + LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM
LINEAR_VALUE_SIZE = LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM

# (name, K, N, layer_count). Each row is one *separate* `Qwen35Ops.linear`
# call in the executed Swift: this model does NOT fuse gate/up, q/k/v, or
# qkv/z/b/a. See Qwen35MLP.swift, Qwen35Attention.swift:143,162,163,211 and
# Qwen35GatedDelta.swift:241,245,254,255,346.
PROJECTIONS = [
    ("linear_attn.in_proj_qkv", H, LINEAR_CONV_SIZE, LINEAR_LAYERS),
    ("linear_attn.in_proj_z", H, LINEAR_VALUE_SIZE, LINEAR_LAYERS),
    ("linear_attn.in_proj_b", H, LINEAR_VALUE_HEADS, LINEAR_LAYERS),
    ("linear_attn.in_proj_a", H, LINEAR_VALUE_HEADS, LINEAR_LAYERS),
    ("linear_attn.out_proj", LINEAR_VALUE_SIZE, H, LINEAR_LAYERS),
    ("full_attn.q_proj", H, ATTN_HEADS * HEAD_DIM * 2, FULL_LAYERS),
    ("full_attn.k_proj", H, KV_HEADS * HEAD_DIM, FULL_LAYERS),
    ("full_attn.v_proj", H, KV_HEADS * HEAD_DIM, FULL_LAYERS),
    ("full_attn.o_proj", ATTN_HEADS * HEAD_DIM, H, FULL_LAYERS),
    ("mlp.gate_proj", H, INTERMEDIATE, LAYERS),
    ("mlp.up_proj", H, INTERMEDIATE, LAYERS),
    ("mlp.down_proj", INTERMEDIATE, H, LAYERS),
]

# lm_head is deliberately NOT run over the 512 seed rows: `begin()` drops the
# full-width seed projection as a dead lazy graph and projects one row from
# the post-norm hidden (Qwen36MTPBlockSession.swift:378-386). So at prefill
# lm_head is an M=1 QMV, not an M=512 GEMM.
LM_HEAD = ("head.lm_head (M=1, see begin())", H, VOCAB, 1)

# The e18 assignment's Section 3 table, quoted for cross-check. These shapes
# assume a fused model that this checkout does not implement.
ADVISOR_TABLE = [
    ("gate_up_proj", 5120, 17408),
    ("down_proj", 8704, 5120),
    ("qkv_proj", 5120, 7168),
    ("o_proj", 3072, 5120),
    ("in_proj (gated delta)", 5120, 8192),
    ("lm_head", 5120, 248320),
]


def weight_bytes(K: int, N: int, bits: int = 4, group_size: int = 64) -> int:
    """Codes + fp16 scales + fp16 biases for an affine-quantized [N, K]."""
    groups = (K // group_size) * N
    return (N * K * bits) // 8 + groups * 2 * 2


def build_rows(host: Host, M: int = 512):
    rows = []
    for name, K, N, count in PROJECTIONS:
        d = dispatch(name, M, N, K, host)
        rows.append((d, count, weight_bytes(K, N)))
    name, K, N, count = LM_HEAD
    d = dispatch(name, 1, N, K, host)
    rows.append((d, count, weight_bytes(K, N)))
    return rows


def fmt_table(host: Host, M: int = 512) -> str:
    out = []
    out.append(f"host: {host.name}")
    out.append(
        f"  architecture={host.architecture} gen={host.arch_gen} size='{host.arch_size}' "
        f"macos>=26.2={host.macos_at_least_26_2} is_nax_available()={host.is_nax_available()}"
    )
    out.append(f"  prefill M={M} (single {M}-token call, no chunking; begin() -> callWithHidden)")
    out.append("")
    hdr = (
        f"{'projection':<28}{'M':>5}{'K':>7}{'N':>8}{'vlim':>6}"
        f"{'ntl':>5}{'mtl':>5}{'tgs':>6}{'spk':>5}  {'branch':<22} kernel"
    )
    out.append(hdr)
    out.append("-" * len(hdr))
    total_w = 0
    total_intermediate = 0
    for d, count, wbytes in build_rows(host, M):
        total_w += wbytes * count
        if d.intermediate_elems:
            total_intermediate += d.intermediate_elems * 2 * count
        out.append(
            f"{d.label:<28}{d.M:>5}{d.K:>7}{d.N:>8}{d.vector_limit:>6}"
            f"{_o(d.n_tiles):>5}{_o(d.m_tiles):>5}{_o(d.current_tgs):>6}{_o(d.split_k):>5}"
            f"  {d.branch:<22} {d.kernel}"
        )
    out.append("")
    out.append(f"  total quantized weight bytes touched once: {total_w/1e9:.3f} GB")
    out.append(
        f"  split-K intermediate traffic (bf16 write + read): "
        f"{2*total_intermediate/1e6:.1f} MB"
    )
    return "\n".join(out)


def _o(v):
    return "-" if v is None else v


def fmt_kernel_census(host: Host, M: int = 512) -> str:
    census: dict[str, list] = {}
    for d, count, _ in build_rows(host, M):
        census.setdefault(d.kernel, []).append((d.label, count))
    lines = [f"kernel census @ M={M} on {host.name}:"]
    for kernel, entries in sorted(census.items(), key=lambda kv: -sum(c for _, c in kv[1])):
        n = sum(c for _, c in entries)
        lines.append(f"  {n:>4} dispatches  {kernel}")
        for label, count in entries:
            lines.append(f"                 {count:>4}x {label}")
    return "\n".join(lines)


def fmt_advisor_crosscheck(host: Host, M: int = 512) -> str:
    lines = [
        "ADVISOR TABLE CROSS-CHECK (e18 assignment Section 3)",
        "",
        "The assignment's shapes assume gate/up, q/k/v and qkv/z/b/a fusions that",
        "this checkout does not implement, and its down_proj K / o_proj K are half",
        "the executed values. Re-running the same arithmetic on the quoted shapes:",
        "",
        f"{'advisor row':<24}{'K':>7}{'N':>8}{'ntl':>5}{'mtl':>5}{'tgs':>6}{'spk':>5}  branch",
    ]
    for name, K, N in ADVISOR_TABLE:
        m = 1 if name == "lm_head" else M
        d = dispatch(name, m, N, K, host)
        lines.append(
            f"{name:<24}{K:>7}{N:>8}{_o(d.n_tiles):>5}{_o(d.m_tiles):>5}"
            f"{_o(d.current_tgs):>6}{_o(d.split_k):>5}  {d.branch}"
        )
    lines += [
        "",
        "Verdict on the assignment's split_k arithmetic:",
        "  * For every shape the assignment actually lists, split_k == 1 is CORRECT,",
        "    and the correction to the real (unfused) shapes does not change that:",
        "    every real N >= 1024 gives n_tiles >= 32, so current_tgs >= 512 and",
        "    512 / current_tgs == 1.",
        "  * The conclusion 'the qmm_t_splitk kernels are dead code on this model'",
        "    is REFUTED. The assignment's shape list omits in_proj_b and in_proj_a",
        "    (N = linear_num_value_heads = 48, K = 5120, quantized per",
        "    Qwen35Weights.swift:507-508). At N=48 n_tiles=2, so current_tgs=32 and",
        "    split_k lands on 16. qmm_t_splitk fires 96 times per prefill",
        "    (2 projections x 48 linear-attention layers).",
        "  * That path is NAX-immune: qmm_splitk never calls is_nax_available().",
        "    The NAX early-return lives only in qmm() (quantized.cpp:697), which",
        "    qmm_splitk reaches solely when split_k collapses to 1.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Phase 2 - where the E16 "dequant_overhead" can physically live
# --------------------------------------------------------------------------

# STREAM triad on this host, quoted as bytes/s exactly as measured.
# research/results/qwen38-r1-e16-prefill-ladder-adjudication.md and the e18
# assignment both require this number, never 273 GB/s.
STREAM_BYTES_PER_SEC = 227_128_791_836.97

# Every figure below is quoted from E16 (PR #18), same host, same base.
E16 = {
    "P": 4.004000009,
    "gemm_seconds_measured": 3.8875040989999885,
    "gemm_tflop_total": 24.93751230464e12,
    "gemm_at_ceiling": 3.3693021188662633,
    "ceiling_tflops": 7.401388009998707e12,
    "gemm_tflops_achieved": 6.414787398180458e12,
    "nongemm": 0.21271352600006566,
    "dequant_overhead": 0.5182019801337252,
    "overlap_credit": 0.09621761600005385,
}


def phase2_budget(M: int = 512) -> dict:
    """FLOP / byte census for the seed prefill, and the roofline consequence.

    `dequant_overhead` in E16 is not an independent measurement: it is exactly
    `gemm_seconds_measured - gemm_at_ceiling`, i.e. the whole gap between the
    achieved quantized-GEMM rate and a *dense bf16* reference rate. This
    function reproduces the identity and then bounds the part of that gap that
    memory traffic could possibly explain.
    """
    flops = 0
    code_bytes = 0
    meta_bytes = 0
    for name, K, N, count in PROJECTIONS:
        flops += 2 * M * K * N * count
        code_bytes += ((N * K * 4) // 8) * count
        meta_bytes += ((K // 64) * N * 2 * 2) * count
    _, K, N, count = LM_HEAD
    flops += 2 * 1 * K * N * count
    code_bytes += ((N * K * 4) // 8) * count
    meta_bytes += ((K // 64) * N * 2 * 2) * count

    weight_total = code_bytes + meta_bytes
    mem_seconds = weight_total / STREAM_BYTES_PER_SEC
    gemm = E16["gemm_seconds_measured"]

    # split-K partial-buffer round trip: only the shapes that actually take the
    # qmm_splitk branch pay it (write + read of an [split_k, M, N] bf16 buffer).
    splitk_bytes = 0
    for name, K, N, count in PROJECTIONS:
        d = dispatch(name, M, N, K, LOCAL_M4_PRO)
        if d.branch == "qmm_splitk" and d.split_k > 1:
            splitk_bytes += 2 * (d.split_k * M * N * 2) * count

    return {
        "gemm_flops": flops,
        "e16_gemm_tflop_total": E16["gemm_tflop_total"],
        "flop_match_rel_error": abs(flops - E16["gemm_tflop_total"]) / E16["gemm_tflop_total"],
        "code_bytes": code_bytes,
        "meta_bytes": meta_bytes,
        "weight_bytes_total": weight_total,
        "arithmetic_intensity_flop_per_byte": flops / weight_total,
        "weight_traffic_seconds": mem_seconds,
        "weight_traffic_frac_of_gemm": mem_seconds / gemm,
        "meta_traffic_seconds": meta_bytes / STREAM_BYTES_PER_SEC,
        "meta_traffic_frac_of_P": (meta_bytes / STREAM_BYTES_PER_SEC) / E16["P"],
        "splitk_partial_bytes": splitk_bytes,
        "splitk_partial_seconds": splitk_bytes / STREAM_BYTES_PER_SEC,
        "splitk_partial_frac_of_P": (splitk_bytes / STREAM_BYTES_PER_SEC) / E16["P"],
        "dequant_overhead_identity": gemm - E16["gemm_at_ceiling"],
        "compute_bound_factor": gemm / mem_seconds,
    }


def fmt_phase2_budget(M: int = 512) -> str:
    b = phase2_budget(M)
    out = [
        f"PHASE 2 BUDGET  (M={M}, roofline {STREAM_BYTES_PER_SEC:,.2f} B/s)",
        "-" * 100,
        f"  GEMM FLOPs (this census)          {b['gemm_flops']:,}",
        f"  GEMM FLOPs (E16 reported)         {b['e16_gemm_tflop_total']:,.0f}",
        f"  relative error                    {b['flop_match_rel_error']:.3e}",
        "",
        f"  4-bit code bytes                  {b['code_bytes']:,}",
        f"  scale+bias bytes (bf16 pairs)     {b['meta_bytes']:,}",
        f"  total quantized weight bytes      {b['weight_bytes_total']:,}",
        f"  arithmetic intensity              {b['arithmetic_intensity_flop_per_byte']:,.1f} FLOP/byte",
        "",
        f"  weight traffic at roofline        {b['weight_traffic_seconds']:.6f} s"
        f"  = {100 * b['weight_traffic_frac_of_gemm']:.3f}% of measured GEMM seconds",
        f"  -> prefill GEMM is compute-bound by {b['compute_bound_factor']:.1f}x",
        "",
        f"  scale/bias traffic alone          {b['meta_traffic_seconds']:.6f} s"
        f"  = {100 * b['meta_traffic_frac_of_P']:.4f}% of P",
        f"  split-K partial round trip        {b['splitk_partial_seconds']:.6f} s"
        f"  = {100 * b['splitk_partial_frac_of_P']:.4f}% of P"
        f"  ({b['splitk_partial_bytes']:,} B)",
        "",
        "  E16 identity check:",
        f"    gemm_seconds_measured - gemm_at_ceiling = {b['dequant_overhead_identity']:.16f}",
        f"    E16 dequant_overhead                    = {E16['dequant_overhead']:.16f}",
        "    => `dequant_overhead` IS the achieved-vs-dense-bf16-ceiling gap,",
        "       not an isolated measurement of dequantization work.",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Phase 2 - unpack ALU per MAC, straight from the kernel source
# --------------------------------------------------------------------------

# Tile geometry is chosen by the *read-only* host dispatcher
# Vendor/.../backend/metal/quantized.cpp: bm=bn=32 for qmm (:717-720) and
# bm=bn=bk=64 wm=wn=2 for qmm_nax (:491-495). Both kernels dequantize the
# weight tile into threadgroup memory with the same
# `dequantize<T, pack_factor, bits>` helper
# (quantized.h:521-527 and quantized_nax.h:524-530 are byte-identical for
# bits==4), then hand it to hardware MMA. So the weight tile is unpacked once
# and reused BM times, and the unpack cost per MAC is exactly ops_per_weight/BM.
TILES = {
    "affine_qmm_t (non-NAX)": dict(BM=32, BK=32, BN=32, tgp=128),
    "affine_qmm_t_nax": dict(BM=64, BK=64, BN=64, tgp=128),
}

# dequantize<U, N, bits=4> body, per packed byte (== 2 weights):
#   U s[2] = {scale, scale / 16.0f};            -> 1 mul, hoisted per call
#   w_local[0] = s[0] * (w[i] & 0x0f) + bias;   -> AND, cvt, FMA, tg-store
#   w_local[1] = s[1] * (w[i] & 0xf0) + bias;   -> AND, cvt, FMA, tg-store
# The high nibble is deliberately left unshifted and compensated by s[1],
# so there is no shift on either lane.
UNPACK_OPS_PER_WEIGHT_MIN = 4.0  # AND + cvt + FMA + threadgroup store
UNPACK_OPS_PER_WEIGHT_MAX = 5.0  # + amortized s[1] mul and the packed-byte load


def phase2_alu_model() -> dict:
    """Unpack ALU issued per MAC for each tile geometry.

    A BM x BN x BK tile dequantizes BN*BK weights and then performs
    BM*BN*BK MACs, so the ratio collapses to ops_per_weight / BM and is
    independent of BN and BK.
    """
    out = {}
    for kernel, g in TILES.items():
        BM, BK, BN, tgp = g["BM"], g["BK"], g["BN"], g["tgp"]
        pack_factor = 2  # get_pack_factor<4, 8>()
        bcols_packed = BK // pack_factor
        n_reads = 1 if bcols_packed * BN < tgp else (bcols_packed * BN) // tgp
        out[kernel] = {
            **g,
            "n_reads_per_thread": n_reads,
            "weights_dequantized_per_tile": BN * BK,
            "macs_per_tile": BM * BN * BK,
            "alu_frac_min": UNPACK_OPS_PER_WEIGHT_MIN / BM,
            "alu_frac_max": UNPACK_OPS_PER_WEIGHT_MAX / BM,
            # scales/biases are `const device T*` (bfloat16) -> 2 B each per
            # group of `group_size` weights, read once per thread per load.
            "distinct_scale_reads_per_tile": BN * max(1, BK // 64),
            "redundant_scale_read_factor": tgp / (BN * max(1, BK // 64)),
        }
    return out


def fmt_phase2_alu_model() -> str:
    phi = E16["dequant_overhead"] / E16["P"]
    m = phase2_alu_model()
    out = [
        "PHASE 2 - UNPACK ALU PER MAC (source-derived; ratio = ops_per_weight / BM)",
        "-" * 100,
        f"{'kernel':<26}{'BMxBNxBK':>14}{'n_reads':>9}"
        f"{'weights/tile':>14}{'MACs/tile':>12}{'ALU/MAC band':>18}",
    ]
    for kernel, d in m.items():
        geom = "%dx%dx%d" % (d["BM"], d["BN"], d["BK"])
        band = "%.2f-%.2f%%" % (d["alu_frac_min"] * 100, d["alu_frac_max"] * 100)
        out.append(
            "%-26s%14s%9d%14d%12d%18s"
            % (
                kernel,
                geom,
                d["n_reads_per_thread"],
                d["weights_dequantized_per_tile"],
                d["macs_per_tile"],
                band,
            )
        )
    out += [
        "",
        f"E16 measured residual phi = dequant_overhead / P = {phi*100:.3f}% of P",
        "",
        "  E16 was measured on the LOCAL host, where is_nax_available() is False,",
        "  so its 400 GEMM dispatches ran affine_qmm_t at BM=32. phi falls INSIDE",
        "  the BM=32 band and OUTSIDE the BM=64 band. That is a consistency check",
        "  on the model, not a resolution of H1 (which is about the ranked box).",
        "",
        "  Consequence: if the ranked host does take the NAX branch, roughly half",
        "  of the locally-measured 12.942% is already structurally collected there",
        "  by the wider tile, and the local prize does not transfer.",
    ]
    return "\n".join(out)


def self_check() -> None:
    """Fail loudly if any claim this script makes in the E18 write-up breaks.

    These are the load-bearing numbers, so a silent drift would be worse than a
    crash: the whole E18 verdict rests on them.
    """
    b = phase2_budget(512)

    # The E16 residual is an algebraic identity, not an independent measurement.
    assert b["dequant_overhead_identity"] == E16["dequant_overhead"], (
        b["dequant_overhead_identity"],
        E16["dequant_overhead"],
    )

    # The corrected unfused census reproduces E16's FLOP total exactly. This is
    # what cross-validates both the unfused shapes and the M=1 lm_head finding.
    assert b["flop_match_rel_error"] == 0.0, b["flop_match_rel_error"]

    # Stopping rule (c): every byte-traffic term is far under ~2% of P.
    assert b["meta_traffic_frac_of_P"] < 0.02
    assert b["splitk_partial_frac_of_P"] < 0.02
    assert b["compute_bound_factor"] > 50

    # The advisor's split_k==1 result holds for every projection with N >= 1024.
    for d, _, _ in build_rows(LOCAL_M4_PRO, 512):
        if d.N >= 1024 and d.M >= d.vector_limit:
            assert d.split_k == 1, (d.label, d.split_k)

    # ...but qmm_t_splitk is not dead code: the two N=48 gated-delta projections
    # take split_k=16, 96 times per prefill.
    splitk_dispatches = 0
    for name, K, N, count in PROJECTIONS:
        d = dispatch(name, 512, N, K, LOCAL_M4_PRO)
        if d.branch == "qmm_splitk" and d.split_k > 1:
            assert d.split_k == 16, (name, d.split_k)
            splitk_dispatches += count
    assert splitk_dispatches == 96, splitk_dispatches

    # phi lands inside the BM=32 band and outside the BM=64 band, which is what
    # makes the tile-height model consistent with E16's local measurement.
    alu = phase2_alu_model()
    phi = E16["dequant_overhead"] / E16["P"]
    nonnax = alu["affine_qmm_t (non-NAX)"]
    nax = alu["affine_qmm_t_nax"]
    assert nonnax["alu_frac_min"] <= phi <= nonnax["alu_frac_max"], (phi, nonnax)
    assert not (nax["alu_frac_min"] <= phi <= nax["alu_frac_max"]), (phi, nax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--M", type=int, default=512)
    args = ap.parse_args()

    self_check()

    hosts = [LOCAL_M4_PRO, RANKED_IF_G17S, RANKED_IF_G17P, RANKED_IF_OLD_OS]

    if args.json:
        print(
            json.dumps(
                {
                    h.name: {
                        "architecture": h.architecture,
                        "arch_gen": h.arch_gen,
                        "is_nax_available": h.is_nax_available(),
                        "rows": [asdict(d) for d, _, _ in build_rows(h, args.M)],
                    }
                    for h in hosts
                }
                | {
                    "phase2_budget": phase2_budget(args.M),
                    "phase2_alu_model": phase2_alu_model(),
                },
                indent=2,
            )
        )
        return

    for h in hosts:
        print(fmt_table(h, args.M))
        print()
        print(fmt_kernel_census(h, args.M))
        print()
        print("=" * 100)
        print()
    print(fmt_advisor_crosscheck(LOCAL_M4_PRO, args.M))
    print()
    print(fmt_phase2_budget(args.M))
    print()
    print(fmt_phase2_alu_model())


if __name__ == "__main__":
    main()
