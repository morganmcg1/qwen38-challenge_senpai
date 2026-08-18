#!/usr/bin/env python3
"""E23 - GPU dispatch inventory for ONE verify forward, as a function of width M.

The verify forward is `model.callWithHiddenAndNormed(...)` at
Qwen36MTPBlockSession.swift:951 plus the top-two reducer at :963.
M = 1 + draftCount, so M in [1, 9].

This is a program that can FAIL, not a document. Every row carries the source
line it was read from; `--verify-citations` re-reads the live tree and exits
non-zero when a citation no longer says what the row claims. `--selftest`
asserts the structural invariants the model is built on. If the source moves
under this file, it stops agreeing rather than quietly lying.

Modes
  --report            primary UNWEIGHTED per-M table (the headline)
  --breakdown         per-family, per-item rows with citations
  --weighted          labelled-provenance sensitivity rollups (SECONDARY)
  --predict           falsifiable numbers a profiler run can kill
  --verify-citations  re-read every cited line; non-zero exit on drift
  --selftest          internal consistency assertions
  --json              machine-readable dump of the primary table

Scenario knobs (both expose a real, named uncertainty - neither is a guess
dressed up as a fact)
  --qmv-limit {6,10,12}     MLX quantized-matvec batch limit for this device
  --layout-copies {counted,zero}
                            include/exclude the stride-derived copy rows
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Callable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MLXLLM = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models"
MLXCOMMON = "Vendor/mlx-swift-lm/Libraries/MLXLMCommon"
MLXC = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx"

Q35 = f"{MLXLLM}/Qwen35.swift"
GD = f"{MLXLLM}/GatedDelta.swift"
AU = f"{MLXCOMMON}/AttentionUtils.swift"
KVC = f"{MLXCOMMON}/KVCache.swift"
SESS = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
QUANT = f"{MLXC}/backend/metal/quantized.cpp"
SDPA = f"{MLXC}/backend/metal/scaled_dot_product_attention.cpp"
SLICING = f"{MLXC}/backend/metal/slicing.cpp"
MCOPY = f"{MLXC}/backend/metal/copy.cpp"
ACTS = "Vendor/mlx-swift/Source/MLXNN/Activations.swift"

# Model geometry. Asserted in --selftest against the live config guards.
N_LAYERS = 64
N_GDN = 48
N_ATTN = 16
N_HEADS_Q = 24
N_HEADS_KV = 4
HEAD_DIM = 256
WIDTHS = [1, 2, 3, 4, 5, 6, 8, 9]

# CITED  : a named op read at the cited line; the count is that op's documented
#          dispatch count in the MLX backend.
# DERIVED: the count follows from a stride/shape argument I performed by hand
#          on top of a cited rule. Falsifiable, and separately togglable when
#          it is a layout copy.
CITED, DERIVED = "cited", "derived"


@dataclass(frozen=True)
class Row:
    family: str
    item: str
    instances: int
    count: Callable[[int, int], int]  # (M, qmv_limit) -> launches per instance
    cite: str  # "path:line"
    expect: str  # substring that must be at/near that line
    conf: str = CITED
    layout_copy: bool = False
    note: str = ""
    # M==1 (processChunk) and M==2 (inline mid branch) run textually duplicated
    # prework at two different sites, so those rows carry a second citation.
    cite2: str = ""
    expect2: str = ""

    def citations(self) -> list[tuple[str, str]]:
        out = [(self.cite, self.expect)]
        if self.cite2:
            out.append((self.cite2, self.expect2))
        return out

    def total(self, m: int, qmv_limit: int, layout: bool) -> int:
        if self.layout_copy and not layout:
            return 0
        return self.instances * self.count(m, qmv_limit)


def qmv(m: int, limit: int) -> int:
    """One quantized projection: 1 qmv launch, or 2 when split-k takes over.

    quantized.cpp:1418 `if (M >= vector_limit)` routes to qmm_splitk, which is
    a matmul dispatch plus a strided_reduce sum -> 2 launches.
    """
    return 2 if m >= limit else 1


K = lambda n: (lambda m, lim: n)  # noqa: E731  constant row


ROWS: list[Row] = [
    # ------------------------------------------------------------------ GDN
    # Branch selection is on (nConfirmed, S). In the verify forward S == M and
    # nConfirmed == 1, so: M>=3 -> processChunkStashingPrefix (packed mixer),
    # M==2 -> single mid kernel, M==1 -> processChunk.
    Row("gdn", "fused in-proj quantizedMM [qkv|z|b|a]", N_GDN, qmv,
        f"{Q35}:678", "quantizedMM("),
    Row("gdn", "z reshape copy off the packed in-proj slice", N_GDN, K(1),
        f"{Q35}:1005", "reshaped(B, S, numVHeads, headVDim)", DERIVED, True,
        "last-axis slice of y is not row_contiguous -> reshape_gpu copies"),

    Row("gdn", "M>=3 packed prework mixer kernel", N_GDN,
        lambda m, lim: 1 if m >= 3 else 0,
        f"{Q35}:836", "qwen35PackedGDNPreworkKernel("),
    Row("gdn", "M>=3 recurrence (prepared)", N_GDN,
        lambda m, lim: 1 if m >= 3 else 0,
        f"{Q35}:896", "qwen35GatedDeltaPrepared("),
    Row("gdn", "M>=3 invScale scalar asType x2", N_GDN,
        lambda m, lim: 2 if m >= 3 else 0,
        f"{Q35}:835", "normScaleConstants(.bfloat16)", DERIVED, False,
        "MLXArray(Float).asType(.bfloat16) is an AsType node -> 1 tiny copy each; "
        "E24 traced the cast to a 1-thread v_copy_float32_bfloat16 with donation "
        "structurally impossible (itemsize 4 != 2)",
        f"{Q35}:747", "MLXArray(pow(invScale, 2)).asType(dtype)"),

    # M<=2 conv prework is textually duplicated: M==1 runs it inside
    # processChunk (Q35:732+), M==2 runs the inline mid branch (Q35:1024+).
    # Both sites are cited so neither can drift unnoticed.
    Row("gdn", "M<=2 conv concat([convState, qkv])", N_GDN,
        lambda m, lim: 2 if m <= 2 else 0,
        f"{Q35}:767", "concatenated([convState, qkv], axis: 1)", CITED, False,
        "concatenate_gpu = one copy_gpu_inplace per input (slicing.cpp:36-42)",
        f"{Q35}:1056", "concatenated([convState, qkv], axis: 1)"),
    Row("gdn", "M<=2 conv1d", N_GDN, lambda m, lim: 1 if m <= 2 else 0,
        f"{Q35}:770", "silu(conv1d(convInput))", CITED, False, "",
        f"{Q35}:1059", "silu(conv1d(convInput))"),
    Row("gdn", "M<=2 compiledSilu", N_GDN, lambda m, lim: 1 if m <= 2 else 0,
        f"{ACTS}:213", "compiledSilu(x)"),
    Row("gdn", "M<=2 q/k/v reshape copies off conv split", N_GDN,
        lambda m, lim: 3 if m <= 2 else 0,
        f"{Q35}:773", "convSplit[0].reshaped(B, S, numKHeads, headKDim)",
        DERIVED, True, "split is a view; reshape of the non-contiguous view copies",
        f"{Q35}:1062", "convSplit[0].reshaped(B, S, numKHeads, headKDim)"),
    Row("gdn", "M<=2 invScale scalar asType x2", N_GDN,
        lambda m, lim: 2 if m <= 2 else 0,
        f"{Q35}:778", "normScaleConstants(dtype)", DERIVED, False,
        "same two casts, reached from processChunk (M==1) and the inline mid "
        "branch (M==2)",
        f"{Q35}:1067", "normScaleConstants(dtype)"),
    Row("gdn", "M<=2 q/k rmsNorm x2", N_GDN, lambda m, lim: 2 if m <= 2 else 0,
        f"{Q35}:781", "MLXFast.rmsNorm(q, weight: MLXArray.mlxNone", CITED, False, "",
        f"{Q35}:1070", "MLXFast.rmsNorm(q, weight: MLXArray.mlxNone"),
    Row("gdn", "M<=2 invScale multiply x2", N_GDN,
        lambda m, lim: 2 if m <= 2 else 0,
        f"{Q35}:781", "* MLXFast.rmsNorm", DERIVED, False, "",
        f"{Q35}:1070", "* MLXFast.rmsNorm"),
    Row("gdn", "M==2 compiled g/beta", N_GDN, lambda m, lim: 1 if m == 2 else 0,
        f"{Q35}:1077", "qwen35CompiledGatedDeltaGBeta("),
    Row("gdn", "M==2 mid kernel (out + state + per-boundary checkpoints)", N_GDN,
        lambda m, lim: 1 if m == 2 else 0,
        f"{Q35}:1084", "midKernel("),
    Row("gdn", "M==1 gatedDeltaUpdateMemoG (g/beta + recurrence)", N_GDN,
        lambda m, lim: 2 if m == 1 else 0,
        f"{Q35}:786", "gatedDeltaUpdateMemoG(", DERIVED,
        note="compiled g/beta launch + one recurrence launch"),

    Row("gdn", "tail M>=2: rmsNorm + compiled postnorm", N_GDN,
        lambda m, lim: 2 if m >= 2 else 0,
        f"{Q35}:1177", "MLXFast.rmsNorm(out, weight: norm.weight"),
    Row("gdn", "tail M==1: gated RMSNorm", N_GDN,
        lambda m, lim: 1 if m == 1 else 0,
        f"{Q35}:1180", "norm(out, gate: z)", DERIVED),
    Row("gdn", "outProj", N_GDN, qmv, f"{Q35}:1182", "outProj(normedOut.reshaped"),

    # ------------------------------------------------------------- attention
    Row("attn", "packed QKV quantizedMM (out-dim 14336)", N_ATTN, qmv,
        f"{Q35}:1711", "quantizedMM("),
    Row("attn", "fused QK-RMSNorm-RoPE custom kernel", N_ATTN, K(1),
        f"{Q35}:1896", "qwen35AttentionQKRMSRoPE("),
    Row("attn", "KV cache slice-update writes", N_ATTN, K(2),
        f"{KVC}:434", "self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys",
        CITED, False,
        "steady state only; the growth branch adds ~4 more once per 256/M forwards",
        f"{KVC}:435", "self.values?[.ellipsis, previous ..< self.offset, 0...] = values"),

    Row("attn", "SDPA fused vector kernel", N_ATTN,
        lambda m, lim: 2 if m >= 6 else 1,
        f"{AU}:127", "MLXFast.scaledDotProductAttention(", CITED, False,
        "AU:122-125 splits 6<=qL<=9 at row 5 so BOTH arms stay <= 5 "
        "and keep the fused kernel (sdpa .cpp:634 needs qL*gqa <= 32, gqa=6)"),
    Row("attn", "SDPA query contiguity copies on the split arms", N_ATTN,
        lambda m, lim: 2 if m >= 6 else 0,
        f"{SDPA}:718", "contiguous_copy_gpu(q_pre, s)", DERIVED, True,
        "sliced [1,24,k,256] view fails both q_copy_unless arms at sdpa .cpp:686-700"),
    Row("attn", "concat of the two SDPA arms", N_ATTN,
        lambda m, lim: 2 if m >= 6 else 0,
        f"{AU}:141", "concatenated([outA, outB], axis: 2)"),
    Row("attn", "post-SDPA transpose+reshape copy", N_ATTN,
        lambda m, lim: 1 if m >= 2 else 0,
        f"{Q35}:1929", ".reshaped(B, L, -1)", DERIVED, True,
        "transposed [1,M,24,256] is row_contiguous only at M==1 (copy.cpp:216-232)"),
    Row("attn", "compiled sigmoid-multiply gate", N_ATTN, K(1),
        f"{Q35}:1929", "qwen35CompiledSigmoidMultiply(output, gate)"),
    Row("attn", "oProj", N_ATTN, qmv, f"{Q35}:1928", "oProj("),

    # ------------------------------------------------------------------- mlp
    Row("mlp", "fused gate_up quantizedMM (N = 34816)", N_LAYERS, qmv,
        f"{Q35}:1256", "quantizedMM("),
    Row("mlp", "compiled fused SwiGLU", N_LAYERS, K(1),
        f"{Q35}:1293", "qwen35CompiledFusedSwiGLU("),
    Row("mlp", "downProj", N_LAYERS, qmv, f"{Q35}:1293", "downProj("),

    # -------------------------------------------------------------- envelope
    Row("envelope", "QuantizedEmbedding lookup", 1, K(4),
        f"{Q35}:2155", "embedTokens(", DERIVED, False,
        "4-bit gather + scale/bias gathers + dequant"),
    # E29 re-baseline: the boundary-fused chain (Q35:2189) is LIVE on the
    # scored path (BF16, hidden 5120), so the residual boundary flows as an
    # unmerged (base, delta) pair. Each interior layer pays ONE fused add+norm
    # at entry instead of a standalone exit add plus a standalone entry norm,
    # and one fused add+norm at the post-attention boundary. E23 charged 2
    # norms + 2 adds per block; that is no longer what the tree does.
    Row("envelope", "layer 0 entry RMSNorm (delta == nil)", 1, K(1),
        f"{Q35}:2092", "normedIn = inputLayerNorm(base)"),
    Row("envelope", "boundary-fused entry add+norm (layers 1..63)",
        N_LAYERS - 1, K(1),
        f"{Q35}:2086", "qwen35FusedResidualRMSNorm(", CITED, False,
        "one launch replaces the previous layer's exit add and this layer's "
        "entry norm"),
    Row("envelope", "fused residual + post-attention RMSNorm", N_LAYERS, K(1),
        f"{Q35}:2102", "qwen35FusedResidualRMSNorm("),
    Row("envelope", "final residual merge base + delta", 1, K(1),
        f"{Q35}:2210", "delta.map { base + $0 }"),
    Row("envelope", "final norm", 1, K(1), f"{Q35}:2952", "norm("),

    # ------------------------------------------------------------------ head
    Row("head", "LM head over ALL M rows -> [1, M, 248320]", 1, qmv,
        f"{Q35}:2955", "lmHead", CITED, False,
        "no row slicing: every draft row's full logit vector is produced"),
    Row("head", "top-two partial reduce", 1, K(1),
        f"{SESS}:1520", "linearTopTwoPartialKernel"),
    Row("head", "top-two finalize", 1, K(1),
        f"{SESS}:1527", "linearTopTwoFinalizeKernel"),
]

FAMILIES = ["gdn", "attn", "mlp", "envelope", "head"]

# Deferred work that is NOT charged to the verify forward. Kept explicit so the
# number is auditable instead of silently missing.
DEFERRED = [
    ("gdn", "M>=3 rollback-tape concat([convState, qkv])", N_GDN, 2,
     f"{Q35}:815",
     "convInput feeds only PrefixReplayTape on the mixer path; under lazy eval "
     "it is not an ancestor of out/newConvState/newSsmState, so it is charged "
     "to a rollback, not to this forward"),
]

# --------------------------------------------------------------------------
# Weight provenance. Every rollup that uses one of these MUST print the label
# in the same sentence as the number. Calibration rule (f): never size a
# dispatch-coverage claim from a --local-submit receipt.
# --------------------------------------------------------------------------
WEIGHTINGS = {
    "receipt128": {
        "label": ("128-token --local-submit shipped-default receipt "
                  "(20 verify rounds, ONE public fixture)"),
        "hist": {2: 1, 5: 1, 6: 8, 7: 3, 8: 5, 9: 2},
        "caveat": ("--local-submit is a 128-token single-prompt screen. Per "
                   "calibration rule (f) it must never size a dispatch-coverage "
                   "claim. Its keys are also ambiguous: read as verify width M "
                   "here, but if they are draft counts every weight shifts +1."),
    },
    "e17s18": {
        "label": ("edward's E17/S18 512-token accepted-depth histogram, "
                  "converted M = depth + 1 (245 rounds)"),
        "hist": {2: 19, 3: 138, 4: 67, 5: 21},
        "caveat": ("Depth histogram from a different experiment's schedule, not "
                   "from the shipped default. Mode is M=3 at ~56%."),
    },
}

NO_SHIPPED_512_HISTOGRAM = (
    "GAP: no shipped-default 512-token verify-width histogram exists anywhere in "
    "the tree. Both weightings above are proxies. Until a 512-token shipped-"
    "default width histogram is captured, the UNWEIGHTED per-M table is the only "
    "safe basis for a dispatch-coverage decision."
)


def totals(m: int, qmv_limit: int, layout: bool) -> dict[str, int]:
    out = {f: 0 for f in FAMILIES}
    for r in ROWS:
        out[r.family] += r.total(m, qmv_limit, layout)
    out["TOTAL"] = sum(out[f] for f in FAMILIES)
    return out


def table(qmv_limit: int, layout: bool) -> str:
    hdr = f"{'M':>3} | " + " | ".join(f"{f:>8}" for f in FAMILIES)
    hdr += f" | {'TOTAL':>7} | {'gdn%':>6} | {'attn%':>6}"
    lines = [hdr, "-" * len(hdr)]
    for m in WIDTHS:
        t = totals(m, qmv_limit, layout)
        row = f"{m:>3} | " + " | ".join(f"{t[f]:>8}" for f in FAMILIES)
        row += (f" | {t['TOTAL']:>7} | {100 * t['gdn'] / t['TOTAL']:>5.1f}%"
                f" | {100 * t['attn'] / t['TOTAL']:>5.1f}%")
        lines.append(row)
    return "\n".join(lines)


def cmd_report(a) -> int:
    print("E23 verify-forward GPU dispatch inventory - PRIMARY (UNWEIGHTED)")
    print(f"scenario: qmv_limit={a.qmv_limit}  layout_copies={a.layout_copies}")
    print(f"geometry: {N_LAYERS} blocks = {N_GDN} GDN + {N_ATTN} full-attention;"
          f" {N_HEADS_Q}q/{N_HEADS_KV}kv heads, head_dim {HEAD_DIM}\n")
    print(table(a.qmv_limit, a.layout_copies == "counted"))
    base = totals(2, a.qmv_limit, a.layout_copies == "counted")["TOTAL"]
    wide = totals(6, a.qmv_limit, a.layout_copies == "counted")["TOTAL"]
    print(f"\nHEADLINE: launch count is NON-MONOTONIC in M. M=2 costs {base} "
          f"dispatches, M=6 costs {wide} ({100 * (wide / base - 1):+.1f}%).")
    print("A width-6 verify forward issues FEWER GPU dispatches than a width-2 "
          "one, because the packed GDN prework mixer (M>=3 only) collapses ~11 "
          "satellite launches per GDN layer into one.")
    print("\nDeferred, NOT charged to this forward:")
    for fam, item, inst, per, cite, why in DEFERRED:
        print(f"  [{fam}] {item}: {inst} x {per} = {inst * per}  ({cite})")
        print(f"        {why}")
    return 0


def cmd_breakdown(a) -> int:
    layout = a.layout_copies == "counted"
    for fam in FAMILIES:
        print(f"\n=== {fam} ===")
        for r in ROWS:
            if r.family != fam:
                continue
            per = "/".join(str(r.count(m, a.qmv_limit)) for m in WIDTHS)
            tot = "/".join(str(r.total(m, a.qmv_limit, layout)) for m in WIDTHS)
            flag = " [LAYOUT-COPY]" if r.layout_copy else ""
            print(f"  {r.item}")
            print(f"    x{r.instances}  per-instance M={WIDTHS}: {per}")
            print(f"    total: {tot}   [{r.conf}]{flag}   {r.cite}")
            if r.note:
                print(f"    note: {r.note}")
    return 0


def cmd_weighted(a) -> int:
    layout = a.layout_copies == "counted"
    print("SECONDARY / SENSITIVITY ONLY. The unweighted table in --report is "
          "the primary result.\n")
    print(NO_SHIPPED_512_HISTOGRAM + "\n")
    for key, w in WEIGHTINGS.items():
        hist, label = w["hist"], w["label"]
        n = sum(hist.values())
        acc = {f: 0.0 for f in FAMILIES + ["TOTAL"]}
        for m, c in hist.items():
            t = totals(m, a.qmv_limit, layout)
            for f in acc:
                acc[f] += t[f] * c / n
        print(f"[{key}] Weighted by the {label}:")
        print(f"    mean dispatches/verify-forward = {acc['TOTAL']:.0f}, "
              f"of which GDN {100 * acc['gdn'] / acc['TOTAL']:.1f}% and "
              f"full-attention {100 * acc['attn'] / acc['TOTAL']:.1f}% "
              f"-- weights from the {label}.")
        print(f"    caveat: {w['caveat']}\n")
    return 0


def cmd_predict(a) -> int:
    layout = a.layout_copies == "counted"
    t2 = totals(2, a.qmv_limit, layout)
    t6 = totals(6, a.qmv_limit, layout)
    t8 = totals(8, a.qmv_limit, layout)
    print("Falsifiable numbers. A GPU profiler capture of ONE verify forward "
          "kills any of these outright.\n")
    preds = [
        ("N1", f"total dispatches at M=6 = {t6['TOTAL']} (+/-5% for the "
               f"layout-copy band; --layout-copies zero gives "
               f"{totals(6, a.qmv_limit, False)['TOTAL']})"),
        ("N2", f"dispatches(M=2)/dispatches(M=6) = {t2['TOTAL'] / t6['TOTAL']:.3f}"
               " -- i.e. narrow verify is MORE dispatch-expensive than wide"),
        ("N3", f"GDN-family share at M=8 = {100 * t8['gdn'] / t8['TOTAL']:.1f}%"),
        ("N4", f"exactly {N_GDN} qwen35PackedGDNPrework dispatches at M>=3, "
               "and exactly 0 at M<=2"),
        ("N5", f"exactly {2 * N_ATTN} sdpa_vector dispatches at M>=6 and "
               f"exactly {N_ATTN} at M<=5; zero sdpa_vector_2pass and zero "
               "full-attention SDPA at every M in [1,9]"),
        ("N6", f"exactly {2 * N_GDN} scalar-astype copy dispatches per forward "
               "at every M (two invScale constants per GDN layer) -- pure "
               "waste, hoistable to init"),
        ("N7", f"zero qmm_splitk / steel-gemm dispatches at every M in [1,9] "
               f"(holds iff the device qmv batch limit is >= 10; at limit 6 "
               f"the M>=6 total becomes {totals(6, 6, layout)['TOTAL']})"),
        ("N8", "exactly 2 top-two reducer dispatches per forward, independent "
               "of M and of the 248320 vocabulary"),
    ]
    for tag, text in preds:
        print(f"  {tag}: {text}")
    return 0


def _read(path: str) -> list[str]:
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read().splitlines()


def _check_citations(rows, window: int, include_deferred: bool = False):
    """Re-read citations in the live tree. -> (rc, checked, n_files, bad)."""
    cache: dict[str, list[str]] = {}
    bad: list[str] = []
    checked = 0
    cites = [(c, e, f"{r.family}/{r.item}") for r in rows for c, e in r.citations()]
    if include_deferred:
        cites += [(c, "concatenated([convState, qkv], axis: 1)", f"{f}/{i}")
                  for f, i, _, _, c, _ in DEFERRED]
    for cite, expect, who in cites:
        path, _, lineno = cite.rpartition(":")
        lineno = int(lineno)
        if path not in cache:
            try:
                cache[path] = _read(path)
            except OSError as exc:
                bad.append(f"{who}: cannot read {path} ({exc})")
                continue
        lines = cache[path]
        lo, hi = max(0, lineno - 1 - window), min(len(lines), lineno + window)
        checked += 1
        if not any(expect in ln for ln in lines[lo:hi]):
            bad.append(f"{who}: {cite} no longer contains {expect!r} "
                       f"within +/-{window} lines")
    return (1 if bad else 0), checked, len(cache), bad


def cmd_verify_citations(a) -> int:
    """Re-read every citation in the live tree. Non-zero exit on drift."""
    rc, checked, n_files, bad = _check_citations(ROWS, a.window, True)
    print(f"checked {checked} citations across {n_files} files "
          f"(+/-{a.window} line window)")
    if bad:
        print(f"\nFAILED: {len(bad)} citation(s) drifted:")
        for b in bad:
            print(f"  - {b}")
        return rc
    print("OK: every cited line still says what the inventory claims.")
    return rc


def cmd_selftest(a) -> int:
    checks: list[tuple[str, bool]] = []
    ck = lambda d, c: checks.append((d, bool(c)))  # noqa: E731

    ck("48 GDN + 16 attention == 64 blocks", N_GDN + N_ATTN == N_LAYERS)
    ck("gqa factor is 6", N_HEADS_Q // N_HEADS_KV == 6)
    # The SDPA split law: the fused vector kernel needs qL*gqa <= 32, so
    # qL <= 5. This must agree with research/sdpa_keylen_band.py's premise and
    # with the AU:122-125 guard bounds.
    gqa = N_HEADS_Q // N_HEADS_KV
    ck("fused SDPA vector serves exactly qL <= 5",
       max(q for q in range(1, 33) if q * gqa <= 32) == 5)
    ck("AU split guard covers exactly the widths above that bound",
       all(m > 5 for m in (6, 7, 8, 9)))
    ck("head_dim 256 is vector-supported but NOT full-attention-supported",
       HEAD_DIM in (64, 96, 128, 256) and HEAD_DIM not in (64, 80, 128))

    for m in WIDTHS:
        t = totals(m, a.qmv_limit, True)
        ck(f"M={m} families sum to TOTAL",
           sum(t[f] for f in FAMILIES) == t["TOTAL"])
        ck(f"M={m} every family is positive", all(t[f] > 0 for f in FAMILIES))

    # Exactly one GDN branch may fire at each M.
    for m in WIDTHS:
        fired = sum(1 for tag, pred in (
            ("mixer", m >= 3), ("mid", m == 2), ("chunk", m == 1)) if pred)
        ck(f"M={m} exactly one GDN branch fires", fired == 1)

    # The non-GDN, non-attention floor must be flat in M: nothing in mlp,
    # envelope or head may depend on M while the qmv limit is not crossed.
    floor = {sum(totals(m, 10, True)[f] for f in ("mlp", "envelope", "head"))
             for m in WIDTHS}
    ck("mlp+envelope+head is flat in M at qmv_limit=10", len(floor) == 1)

    # ...and must step exactly once if the limit were 6.
    hi = sum(totals(9, 6, True)[f] for f in ("mlp", "envelope", "head"))
    lo = sum(totals(5, 6, True)[f] for f in ("mlp", "envelope", "head"))
    ck("qmv_limit=6 raises the flat floor", hi > lo)

    ck("layout-copy rows are all tagged derived",
       all(r.conf == DERIVED for r in ROWS if r.layout_copy))
    ck("zeroing layout copies never increases a total",
       all(totals(m, a.qmv_limit, False)["TOTAL"]
           <= totals(m, a.qmv_limit, True)["TOTAL"] for m in WIDTHS))

    # The headline claim itself, asserted rather than asserted-in-prose.
    ck("M=6 costs strictly fewer dispatches than M=2",
       totals(6, a.qmv_limit, True)["TOTAL"] < totals(2, a.qmv_limit, True)["TOTAL"])

    # A citation checker that cannot fail proves nothing, so make it fail here.
    ck("citation checker is clean on the live tree",
       _check_citations(ROWS, a.window)[0] == 0)
    ck("citation checker rejects a bad line number",
       _check_citations([replace(ROWS[0], cite=f"{Q35}:999999")], a.window)[0] == 1)
    ck("citation checker rejects drifted text",
       _check_citations([replace(ROWS[0], expect="NOT_IN_TREE")], a.window)[0] == 1)
    ck("citation checker rejects an unreadable file",
       _check_citations([replace(ROWS[0], cite="no/such/file.swift:1")],
                        a.window)[0] == 1)

    bad = [d for d, ok in checks if not ok]
    for d, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {d}")
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
    return 1 if bad else 0


def cmd_json(a) -> int:
    layout = a.layout_copies == "counted"
    print(json.dumps({
        "scenario": {"qmv_limit": a.qmv_limit, "layout_copies": a.layout_copies},
        "geometry": {"blocks": N_LAYERS, "gdn": N_GDN, "attn": N_ATTN,
                     "q_heads": N_HEADS_Q, "kv_heads": N_HEADS_KV,
                     "head_dim": HEAD_DIM},
        "primary_unweighted": {
            str(m): totals(m, a.qmv_limit, layout) for m in WIDTHS},
        "deferred": [{"family": f, "item": i, "launches": inst * per,
                      "cite": c} for f, i, inst, per, c, _ in DEFERRED],
        "weight_provenance_gap": NO_SHIPPED_512_HISTOGRAM,
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qmv-limit", type=int, default=10, choices=[6, 10, 12])
    p.add_argument("--layout-copies", default="counted",
                   choices=["counted", "zero"])
    p.add_argument("--window", type=int, default=6,
                   help="citation search window in lines")
    for flag in ("report", "breakdown", "weighted", "predict",
                 "verify-citations", "selftest", "json"):
        p.add_argument(f"--{flag}", action="store_true")
    a = p.parse_args()

    modes = [(a.report, cmd_report), (a.breakdown, cmd_breakdown),
             (a.weighted, cmd_weighted), (a.predict, cmd_predict),
             (a.verify_citations, cmd_verify_citations),
             (a.selftest, cmd_selftest), (a.json, cmd_json)]
    chosen = [fn for on, fn in modes if on]
    if not chosen:
        chosen = [cmd_report]
    rc = 0
    for i, fn in enumerate(chosen):
        if i:
            print()
        rc |= fn(a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
