#!/usr/bin/env python3
"""E136 rung 0: what does the widened survivor selection cost per draft step?

C1 replaces the exact affine-2 pass over every probed row with a sketch, keeps
the top `N` survivors, rescores those exactly and then takes the exact top 32.
The byte model says that removes 37.483 MB per draft step. It cannot say what
the WIDER SELECTION costs, and the advisor made that a hard gate on the build.

This module measures the selection alone. No sketch, no basis, no readout
change, and it never touches a tracked source: every kernel below is a
verbatim port of the shipped Metal in
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift` or a new kernel that
only the candidate arm would use.

  shipped arm   `Qwen35RowTop32(rows: 24_584, rowsPerCluster: 8)`, the two
                dispatches at `:4448-4496`, selecting 32 of 24,584 probed rows.
  candidate arm three added dispatches -- a 256-bin ordinal histogram, a
                single-threadgroup threshold scan and a compaction -- then the
                SAME two dispatches over the `N` survivors.

`added_us = candidate - shipped`, and `ranked cost % = added_us / 174.1` at the
265 GB/s read-only ceiling of FACT 33.

TIMING PROTOCOL. Each arm is a dependent chain, exactly as the real readout is:
every kernel reads `dep[0]` in a guard that never fires, so MLX cannot reorder
or overlap the repetitions and cannot fold two identical calls together. The
per-iteration cost is the SLOPE of eval wall time against repetition count, so
the intercept absorbs submission and synchronisation and never enters the
answer. Host graph construction is timed separately and excluded.

  research/e133_job.sh python3 research/e136_selection_bench.py --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time

import mlx.core as mx

# Shipped shape constants, `Qwen35.swift:4196-4199` and `:4446`.
TG = 256
TOPK = 32
ROW_TOP32_TILES = 32
CLUSTERS = 12_292
ROWS_PER_CLUSTER = 8
PROBE_SHIPPED = 0.25
PROBE_C1 = 0.35
# The bin key is the TOP 16 BITS of the float ordinal. A bf16 score is exactly
# the top 16 bits of its fp32 value, so the key is a lossless, order preserving
# image of the score and the selection is the exact top `N` up to bf16 ties.
# An 8-bit key is not: its low bit is a float exponent bit, so a whole octave
# lands in one bin and the capacity clamp then throws away real winners. That
# is measured, not assumed -- the 8-bit key kept 14 of the true top 32.
HIST_BINS = 65_536
HIST_GROUPS = 256
HIST_TILES = 64

# FACT 33, edward E92: the read-only bandwidth ceiling on this host family.
BYTE_CEILING_GB_S = 265.0
PCT_PER_MB = 0.02167          # ledger 289, the C1 byte price
US_PER_RANKED_PCT = (1.0 / PCT_PER_MB) * 1e6 / (BYTE_CEILING_GB_S * 1e3)

ORDINAL_HEADER = """
    inline uint qwen_top32_ordinal(float v) {
        if (isnan(v))  { return 0xFFFFFFFFu; }
        if (v == 0.0f) { return 0x80000000u; }
        uint u = as_type<uint>(v);
        return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
    }
"""

# Every kernel opens with this. It reads one word the caller can only produce by
# running the previous kernel, so the repetitions serialise. It never fires:
# `dep` always holds selection output, and 0xFFFFFFFF is not a reachable id.
DEP_GUARD = "    if (dep[0] == 0xFFFFFFFFu) { return; }\n"


class Plan:
    """`Qwen35Top32Plan`, ported verbatim from `Qwen35.swift:4204-4220`."""

    def __init__(self, real_count: int, tiles: int):
        self.real_count = real_count
        self.tiles = tiles
        self.stride = tiles * TG
        self.per_thread = (real_count + self.stride - 1) // self.stride
        self.cands = tiles * TOPK
        self.fin_per_thread = self.cands // TG

    def check(self) -> None:
        # `Qwen35.swift:4455`. The selection bitmasks are 32 bits wide.
        assert self.per_thread <= 32 and self.fin_per_thread <= 32, vars(self)

    def as_dict(self) -> dict:
        return {"real_count": self.real_count, "tiles": self.tiles,
                "stride": self.stride, "per_thread": self.per_thread,
                "cands": self.cands, "fin_per_thread": self.fin_per_thread}


def partial_source(plan: Plan, indirect: bool) -> str:
    """`qwen35Top32PartialSource`, `:4236-4319`.

    `indirect` is the only candidate-arm change: the survivors carry their
    original probed-row index, so the tie break stays on that index and the
    selection is independent of the order compaction happened to emit them in.
    """
    load = ("ord[n] = qwen_top32_ordinal(float(logits[i]));\n"
            "            idx[n] = row_id[i];"
            if indirect else
            "ord[n] = qwen_top32_ordinal(float(logits[i]));\n"
            "            idx[n] = i;")
    return DEP_GUARD + f"""
        constexpr uint REAL_COUNT = {plan.real_count};
        constexpr uint TG_SIZE    = {TG};
        constexpr uint STRIDE     = {plan.stride};
        constexpr uint PER_THREAD = {plan.per_thread};
        constexpr uint TOPK       = {TOPK};
        constexpr uint SIMD_SIZE  = 32;
        constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
        constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
        static_assert(PER_THREAD <= 32, "PER_THREAD exceeds taken-bitmask width");
        static_assert(PB <= 32, "PB exceeds tk2-bitmask width");

        uint tile = threadgroup_position_in_grid.x;
        uint tid  = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg   = simdgroup_index_in_threadgroup;

        uint ord[PER_THREAD];
        uint idx[PER_THREAD];
        for (uint t = 0; t < PER_THREAD; ++t) {{ ord[t] = 0u; idx[t] = 0u; }}
        uint n = 0;
        for (uint i = tile * TG_SIZE + tid; i < REAL_COUNT; i += STRIDE) {{
            {load}
            n++;
        }}

        threadgroup uint sc_ord[NSIMD * TOPK];
        threadgroup uint sc_idx[NSIMD * TOPK];

        uint taken = 0u;
        for (uint r = 0; r < TOPK; ++r) {{
            uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
            for (uint t = 0; t < PER_THREAD; ++t) {{
                if ((taken & (1u << t)) != 0u) {{ continue; }}
                if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {{
                    bo = ord[t]; bi = idx[t]; bs = t;
                }}
            }}
            uint mo = simd_max(bo);
            uint mi = simd_max((bo == mo) ? bi : 0u);
            if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {{
                taken |= (1u << bs);
            }}
            if (lane == 0) {{
                sc_ord[sg * TOPK + r] = mo;
                sc_idx[sg * TOPK + r] = mi;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {{
            uint o2[PB];
            uint i2[PB];
            for (uint t = 0; t < PB; ++t) {{
                uint p = t * SIMD_SIZE + lane;
                o2[t] = sc_ord[p];
                i2[t] = sc_idx[p];
            }}
            uint tk2 = 0u;
            for (uint r = 0; r < TOPK; ++r) {{
                uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                for (uint t = 0; t < PB; ++t) {{
                    if ((tk2 & (1u << t)) != 0u) {{ continue; }}
                    if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {{
                        bo = o2[t]; bi = i2[t]; bs = t;
                    }}
                }}
                uint mo = simd_max(bo);
                uint mi = simd_max((bo == mo) ? bi : 0u);
                if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {{
                    tk2 |= (1u << bs);
                }}
                if (lane == 0) {{
                    cand_ord[tile * TOPK + r] = mo;
                    cand_idx[tile * TOPK + r] = mi;
                }}
            }}
        }}
        """


def finalize_source(plan: Plan) -> str:
    """`qwen35Top32FinalizeSource`, `:4339-4419`, with the fused id lookup."""
    emit = (f"uint cluster = probed[mi / {ROWS_PER_CLUSTER}u]; "
            "token_ids[TOPK - 1u - r] = "
            f"uint(perm[cluster * {ROWS_PER_CLUSTER}u "
            f"+ (mi % {ROWS_PER_CLUSTER}u)]);")
    return DEP_GUARD + f"""
        constexpr uint TG_SIZE    = {TG};
        constexpr uint PER_THREAD = {plan.fin_per_thread};
        constexpr uint TOPK       = {TOPK};
        constexpr uint SIMD_SIZE  = 32;
        constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
        constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
        static_assert(PER_THREAD <= 32, "PER_THREAD exceeds taken-bitmask width");
        static_assert(PB <= 32, "PB exceeds tk2-bitmask width");

        uint tid  = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg   = simdgroup_index_in_threadgroup;

        uint ord[PER_THREAD];
        uint idx[PER_THREAD];
        for (uint t = 0; t < PER_THREAD; ++t) {{
            uint p = t * TG_SIZE + tid;
            ord[t] = cand_ord[p];
            idx[t] = cand_idx[p];
        }}

        threadgroup uint sc_ord[NSIMD * TOPK];
        threadgroup uint sc_idx[NSIMD * TOPK];

        uint taken = 0u;
        for (uint r = 0; r < TOPK; ++r) {{
            uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
            for (uint t = 0; t < PER_THREAD; ++t) {{
                if ((taken & (1u << t)) != 0u) {{ continue; }}
                if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {{
                    bo = ord[t]; bi = idx[t]; bs = t;
                }}
            }}
            uint mo = simd_max(bo);
            uint mi = simd_max((bo == mo) ? bi : 0u);
            if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {{
                taken |= (1u << bs);
            }}
            if (lane == 0) {{
                sc_ord[sg * TOPK + r] = mo;
                sc_idx[sg * TOPK + r] = mi;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {{
            uint o2[PB];
            uint i2[PB];
            for (uint t = 0; t < PB; ++t) {{
                uint p = t * SIMD_SIZE + lane;
                o2[t] = sc_ord[p];
                i2[t] = sc_idx[p];
            }}
            uint tk2 = 0u;
            for (uint r = 0; r < TOPK; ++r) {{
                uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                for (uint t = 0; t < PB; ++t) {{
                    if ((tk2 & (1u << t)) != 0u) {{ continue; }}
                    if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {{
                        bo = o2[t]; bi = i2[t]; bs = t;
                    }}
                }}
                uint mo = simd_max(bo);
                uint mi = simd_max((bo == mo) ? bi : 0u);
                if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {{
                    tk2 |= (1u << bs);
                }}
                if (lane == 0) {{ {emit} }}
            }}
        }}
        """


def hist_source(rows: int) -> str:
    """Stage 1(i): a 65,536-bin histogram of the top 16 ordinal bits.

    A threadgroup-local histogram cannot hold 65,536 bins in 32 KB, so the adds
    go straight to device memory. Contention stays low because a 34,424 row
    population spreads over roughly a thousand live bins.
    """
    return DEP_GUARD + f"""
        constexpr uint ROWS  = {rows};
        constexpr uint GRID  = {HIST_TILES * TG};

        uint gid = thread_position_in_grid.x;
        for (uint i = gid; i < ROWS; i += GRID) {{
            uint b = qwen_top32_ordinal(float(score[i])) >> 16;
            atomic_fetch_add_explicit(
                (device atomic_uint *)&bins[b], 1u, memory_order_relaxed);
        }}
        """


def thresh_source(survivors: int) -> str:
    """Stage 1(ii): one threadgroup finds the exact boundary bin.

    `ctl` is [tau, count_strictly_above_tau, capacity_left_in_tau, cursor].
    Each of the 256 threads reduces a contiguous run of 256 bins, then one
    thread walks the 256 group sums down from the top and re-walks the single
    group that straddles `WANT`. That is 512 serial adds, cheaper in barriers
    than a two-level parallel scan at this width.
    """
    return DEP_GUARD + f"""
        constexpr uint NBINS  = {HIST_BINS};
        constexpr uint GROUPS = {HIST_GROUPS};
        constexpr uint PER    = NBINS / GROUPS;
        constexpr uint WANT   = {survivors};

        uint tid = thread_position_in_threadgroup.x;
        uint s = 0u;
        for (uint i = 0; i < PER; ++i) {{ s += bins[tid * PER + i]; }}
        threadgroup uint g[GROUPS];
        g[tid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {{
            uint acc = 0u, above_groups = 0u;
            uint gsel = 0u;
            for (int q = int(GROUPS) - 1; q >= 0; --q) {{
                uint prev = acc;
                acc += g[q];
                if (acc >= WANT) {{ gsel = uint(q); above_groups = prev; break; }}
                if (q == 0) {{ gsel = 0u; above_groups = prev; }}
            }}
            uint acc2 = above_groups;
            uint tau = gsel * PER;
            uint above = above_groups;
            for (int b = int(PER) - 1; b >= 0; --b) {{
                uint prev = acc2;
                acc2 += bins[gsel * PER + uint(b)];
                if (acc2 >= WANT) {{
                    tau = gsel * PER + uint(b); above = prev; break;
                }}
                if (b == 0) {{ tau = gsel * PER; above = prev; }}
            }}
            ctl[0] = tau;
            ctl[1] = above;
            ctl[2] = (WANT > above) ? (WANT - above) : 0u;
            ctl[3] = 0u;
        }}
        """


def compact_source(rows: int, capacity: int) -> str:
    """Stage 1(iii): emit the survivors into a fixed-capacity buffer.

    Rows above the boundary bin are unconditional and cannot overflow, because
    the threshold scan chose `tau` so that fewer than `WANT` rows sit above it.
    Boundary-bin rows compete for what is left under a hard capacity clamp.
    Unfilled slots keep ordinal 0, the minimum, so they never win stage 2.
    """
    return DEP_GUARD + f"""
        constexpr uint ROWS = {rows};
        constexpr uint CAP  = {capacity};
        constexpr uint GRID = {HIST_TILES * TG};

        uint gid = thread_position_in_grid.x;
        uint tau = ctl[0];

        for (uint i = gid; i < ROWS; i += GRID) {{
            uint o = qwen_top32_ordinal(float(score[i]));
            uint b = o >> 16;
            if (b < tau) {{ continue; }}
            uint slot = atomic_fetch_add_explicit(
                (device atomic_uint *)&cursor[0], 1u, memory_order_relaxed);
            if (slot < CAP) {{
                surv_ord[slot] = o;
                surv_row[slot] = i;
            }}
        }}
        """


def rescore_source(capacity: int) -> str:
    """Stands in for the exact affine-2 rescore of the survivors.

    The rescore itself is priced in the byte model and is NOT part of the
    selection cost, so this kernel only does the addressing work the real one
    would add on top: read the survivor row id and write a score into the
    dense stage-2 input. Timing it separately keeps the two prices apart.
    """
    return DEP_GUARD + f"""
        constexpr uint CAP = {capacity};
        uint i = thread_position_in_grid.x;
        if (i >= CAP) {{ return; }}
        surv_score[i] = exact[surv_row[i]];
        """


def kernel(name, inputs, outputs, source, header=""):
    return mx.fast.metal_kernel(
        name=name, input_names=inputs, output_names=outputs,
        source=source, header=header, ensure_row_contiguous=False)


_DEP_SEED = None


def dep_seed():
    """One materialised guard word, shared by every arm.

    Building it once matters: a fresh `mx.zeros` per call would add a real op
    to the graph under test and would be charged to the arm.
    """
    global _DEP_SEED
    if _DEP_SEED is None:
        _DEP_SEED = mx.zeros([TOPK], dtype=mx.uint32)
        mx.eval(_DEP_SEED)
    return _DEP_SEED


def stage2_tiles(survivors: int) -> int:
    """Stage-2 tile count for `survivors` keys, under the shipped plan rules.

    `finPerThread = tiles * 32 / 256` must be at least 1, so `tiles >= 8`, and
    the selection bitmask caps it at the shipped 32. Between those bounds one
    key per thread is the cheapest legal choice. Holding `tiles` at the shipped
    32 for every `N` would make stage 2 cost the same at every width and the
    sweep would measure nothing.
    """
    return max(8, min(ROW_TOP32_TILES, -(-survivors // TG)))


class PartArm:
    """One dispatch in isolation. Every input except the guard is resident."""

    def __init__(self, label, kern, const_inputs, grid, threadgroup,
                 shapes, dtypes, init_value=None, out_index=0):
        self.label = label
        self.k = kern
        self.const = list(const_inputs)
        self.grid = grid
        self.tg = threadgroup
        self.shapes = shapes
        self.dtypes = dtypes
        self.init_value = init_value
        self.out_index = out_index
        self.dispatches = 1
        mx.eval(*self.const)

    def step(self, dep):
        kw = {} if self.init_value is None else {"init_value": self.init_value}
        return self.k(inputs=self.const + [dep], grid=self.grid,
                      threadgroup=self.tg, output_shapes=self.shapes,
                      output_dtypes=self.dtypes, **kw)[self.out_index]

    def seed(self):
        return dep_seed()


class ShippedArm:
    """The two dispatches the scored worker runs today."""

    label = "shipped"

    def __init__(self, rows: int, probes: int):
        self.rows = rows
        self.plan = Plan(rows, ROW_TOP32_TILES)
        self.plan.check()
        self.partial = kernel("e136_row_top32_partial", ["logits", "dep"],
                              ["cand_ord", "cand_idx"],
                              partial_source(self.plan, indirect=False),
                              ORDINAL_HEADER)
        self.finalize = kernel("e136_row_top32_finalize",
                               ["cand_ord", "cand_idx", "probed", "perm",
                                "dep"], ["token_ids"],
                               finalize_source(self.plan))
        mx.random.seed(11)
        self.score = mx.random.normal([rows]).astype(mx.bfloat16)
        self.probed = mx.arange(probes, dtype=mx.uint32)
        self.perm = mx.arange(CLUSTERS * ROWS_PER_CLUSTER, dtype=mx.int32)
        self.dispatches = 2

    def step(self, dep):
        c = self.partial(inputs=[self.score, dep],
                         grid=(self.plan.tiles * TG, 1, 1),
                         threadgroup=(TG, 1, 1),
                         output_shapes=[[self.plan.cands], [self.plan.cands]],
                         output_dtypes=[mx.uint32, mx.uint32])
        return self.finalize(inputs=[c[0], c[1], self.probed, self.perm, dep],
                             grid=(TG, 1, 1), threadgroup=(TG, 1, 1),
                             output_shapes=[[TOPK]],
                             output_dtypes=[mx.uint32])[0]

    def seed(self):
        return dep_seed()

    def parts(self) -> list:
        cand = mx.random.randint(
            0, 2 ** 31 - 1, [self.plan.cands]).astype(mx.uint32)
        cidx = mx.random.randint(
            0, self.rows, [self.plan.cands]).astype(mx.uint32)
        return [
            PartArm("shipped.partial", self.partial, [self.score],
                    (self.plan.tiles * TG, 1, 1), (TG, 1, 1),
                    [[self.plan.cands], [self.plan.cands]],
                    [mx.uint32, mx.uint32]),
            PartArm("shipped.finalize", self.finalize,
                    [cand, cidx, self.probed, self.perm], (TG, 1, 1),
                    (TG, 1, 1), [[TOPK]], [mx.uint32]),
        ]


class TwoStageArm:
    """Three added dispatches, then the same two over `N` survivors."""

    def __init__(self, rows: int, probes: int, survivors: int, tiles: int,
                 with_rescore: bool):
        self.label = f"N{survivors}"
        self.rows = rows
        self.survivors = survivors
        self.plan = Plan(survivors, tiles)
        self.plan.check()
        self.hist = kernel("e136_sketch_hist", ["score", "dep"], ["bins"],
                           hist_source(rows), ORDINAL_HEADER)
        self.thresh = kernel("e136_sketch_thresh", ["bins", "dep"], ["ctl"],
                             thresh_source(survivors))
        self.compact = kernel("e136_sketch_compact", ["score", "ctl", "dep"],
                              ["surv_ord", "surv_row", "cursor"],
                              compact_source(rows, survivors), ORDINAL_HEADER)
        self.rescore = kernel("e136_survivor_rescore",
                              ["exact", "surv_row", "dep"], ["surv_score"],
                              rescore_source(survivors))
        self.partial = kernel("e136_surv_top32_partial",
                              ["logits", "row_id", "dep"],
                              ["cand_ord", "cand_idx"],
                              partial_source(self.plan, indirect=True),
                              ORDINAL_HEADER)
        self.finalize = kernel("e136_surv_top32_finalize",
                               ["cand_ord", "cand_idx", "probed", "perm",
                                "dep"], ["token_ids"],
                               finalize_source(self.plan))
        mx.random.seed(11)
        self.score = mx.random.normal([rows]).astype(mx.bfloat16)
        self.exact = mx.random.normal([rows]).astype(mx.bfloat16)
        self.probed = mx.arange(probes, dtype=mx.uint32)
        self.perm = mx.arange(CLUSTERS * ROWS_PER_CLUSTER, dtype=mx.int32)
        self.with_rescore = with_rescore
        self.dispatches = 5 + (1 if with_rescore else 0)

    def step(self, dep):
        bins = self.hist(inputs=[self.score, dep],
                         grid=(HIST_TILES * TG, 1, 1), threadgroup=(TG, 1, 1),
                         output_shapes=[[HIST_BINS]],
                         output_dtypes=[mx.uint32], init_value=0)[0]
        ctl = self.thresh(inputs=[bins, dep], grid=(HIST_GROUPS, 1, 1),
                          threadgroup=(HIST_GROUPS, 1, 1),
                          output_shapes=[[4]], output_dtypes=[mx.uint32],
                          init_value=0)[0]
        surv = self.compact(inputs=[self.score, ctl, dep],
                            grid=(HIST_TILES * TG, 1, 1),
                            threadgroup=(TG, 1, 1),
                            output_shapes=[[self.survivors],
                                           [self.survivors], [1]],
                            output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
                            init_value=0)
        surv_ord, surv_row = surv[0], surv[1]
        if self.with_rescore:
            score = self.rescore(
                inputs=[self.exact, surv_row, dep],
                grid=(self.survivors, 1, 1), threadgroup=(TG, 1, 1),
                output_shapes=[[self.survivors]],
                output_dtypes=[mx.bfloat16], init_value=0)[0]
        else:
            score = surv_ord
        c = self.partial(inputs=[score, surv_row, dep],
                         grid=(self.plan.tiles * TG, 1, 1),
                         threadgroup=(TG, 1, 1),
                         output_shapes=[[self.plan.cands], [self.plan.cands]],
                         output_dtypes=[mx.uint32, mx.uint32])
        return self.finalize(inputs=[c[0], c[1], self.probed, self.perm, dep],
                             grid=(TG, 1, 1), threadgroup=(TG, 1, 1),
                             output_shapes=[[TOPK]],
                             output_dtypes=[mx.uint32])[0]

    def seed(self):
        return dep_seed()

    def parts(self) -> list:
        """The same dispatches, each priced on its own."""
        surv_score = mx.random.normal([self.survivors]).astype(mx.bfloat16)
        surv_row = mx.random.randint(
            0, self.rows, [self.survivors]).astype(mx.uint32)
        bins = mx.random.randint(0, 4, [HIST_BINS]).astype(mx.uint32)
        ctl = mx.array([0, 0, 0, 0], dtype=mx.uint32)
        cand = mx.random.randint(
            0, 2 ** 31 - 1, [self.plan.cands]).astype(mx.uint32)
        cidx = mx.random.randint(
            0, self.rows, [self.plan.cands]).astype(mx.uint32)
        out = [
            PartArm(f"{self.label}.hist", self.hist, [self.score],
                    (HIST_TILES * TG, 1, 1), (TG, 1, 1), [[HIST_BINS]],
                    [mx.uint32], init_value=0),
            PartArm(f"{self.label}.thresh", self.thresh, [bins],
                    (HIST_GROUPS, 1, 1), (HIST_GROUPS, 1, 1), [[4]],
                    [mx.uint32], init_value=0),
            PartArm(f"{self.label}.compact", self.compact, [self.score, ctl],
                    (HIST_TILES * TG, 1, 1), (TG, 1, 1),
                    [[self.survivors], [self.survivors], [1]],
                    [mx.uint32, mx.uint32, mx.uint32], init_value=0),
            PartArm(f"{self.label}.partial", self.partial,
                    [surv_score, surv_row], (self.plan.tiles * TG, 1, 1),
                    (TG, 1, 1), [[self.plan.cands], [self.plan.cands]],
                    [mx.uint32, mx.uint32]),
            PartArm(f"{self.label}.finalize", self.finalize,
                    [cand, cidx, self.probed, self.perm], (TG, 1, 1),
                    (TG, 1, 1), [[TOPK]], [mx.uint32]),
        ]
        if self.with_rescore:
            out.insert(3, PartArm(
                f"{self.label}.rescore", self.rescore, [self.exact, surv_row],
                (self.survivors, 1, 1), (TG, 1, 1), [[self.survivors]],
                [mx.bfloat16], init_value=0))
        return out


class NullArm:
    """One dispatch that does nothing but the guard, to price a dispatch."""

    label = "null_dispatch"
    dispatches = 1

    def __init__(self):
        self.k = kernel("e136_null", ["dep"], ["out"],
                        DEP_GUARD + """
        uint tid = thread_position_in_grid.x;
        if (tid < 32u) { out[tid] = dep[tid] + 1u; }
        """)

    def step(self, dep):
        return self.k(inputs=[dep], grid=(TG, 1, 1), threadgroup=(TG, 1, 1),
                      output_shapes=[[TOPK]], output_dtypes=[mx.uint32])[0]

    def seed(self):
        return dep_seed()


def time_calls(arm, iters: int, trials: int) -> dict:
    """Per-call cost, measured exactly as `qwen35BenchRowTop32` measures it.

    The Swift entry point times `iters` independent `eval(selector(...))`
    calls and divides. Copying that shape is the point: it makes the shipped
    arm here directly comparable with a number produced by the real runtime,
    so the port can be checked before any candidate number is believed. Each
    call carries one submit-and-sync overhead, which cancels in an arm-to-arm
    difference and is reported separately through the null arm.
    """
    dep = arm.seed()
    for _ in range(4):
        mx.eval(arm.step(dep))
    mx.synchronize()
    per = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(arm.step(dep))
        mx.synchronize()
        per.append((time.perf_counter() - t0) * 1e6 / iters)
    per.sort()
    return {
        "label": arm.label,
        "dispatches_per_call": arm.dispatches,
        "us_per_call": per[len(per) // 2],
        "us_per_call_min": per[0],
        "us_per_call_max": per[-1],
        "us_per_call_trials": per,
        "iters": iters,
        "trials": trials,
    }


def chain(arm, reps: int):
    dep = arm.seed()
    for _ in range(reps):
        dep = arm.step(dep)
    return dep


def time_arm(arm, reps_grid, trials: int) -> dict:
    mx.eval(chain(arm, 2))
    mx.synchronize()
    samples: dict[int, list[float]] = {r: [] for r in reps_grid}
    build: dict[int, list[float]] = {r: [] for r in reps_grid}
    for _ in range(trials):
        for r in reps_grid:
            t0 = time.perf_counter()
            out = chain(arm, r)
            t1 = time.perf_counter()
            mx.eval(out)
            mx.synchronize()
            t2 = time.perf_counter()
            build[r].append((t1 - t0) * 1e6)
            samples[r].append((t2 - t1) * 1e6)
    med = {r: sorted(v)[len(v) // 2] for r, v in samples.items()}
    # Ordinary least squares of eval time against repetition count. The slope
    # is the per-iteration cost; the intercept is submission and sync.
    xs = [float(r) for r in reps_grid]
    ys = [med[r] for r in reps_grid]
    n = len(xs)
    mx_ = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs)
    slope = sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx_
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    top = max(reps_grid)
    low = min(reps_grid)
    return {
        "label": arm.label,
        "dispatches_per_step": arm.dispatches,
        "us_per_step": slope,
        "us_per_dispatch": slope / arm.dispatches,
        "intercept_us": intercept,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "two_point_us_per_step": (med[top] - med[low]) / (top - low),
        "median_eval_us": med,
        "median_build_us": {r: sorted(v)[len(v) // 2] for r, v in build.items()},
        "iqr_eval_us": {
            r: sorted(v)[int(0.75 * len(v))] - sorted(v)[int(0.25 * len(v))]
            for r, v in samples.items()},
        "trials": trials,
    }


def correctness(arm_two, arm_ship) -> dict:
    """The candidate arm must return the exact top 32 of the survivor set.

    Stage 2 is exact by construction over whatever survived, so the check that
    matters is survivor recall: does the compaction keep the true top 32 of the
    sketch scores? A positive control damages one score and requires the
    comparison to notice.
    """
    dep = arm_two.seed()
    ids = arm_two.step(dep)
    mx.eval(ids)
    host = arm_two.score.astype(mx.float32)
    order = mx.argsort(host)
    true_top = set(order[-TOPK:].tolist())
    # Replay stage 1 alone to read the survivor set back.
    bins = arm_two.hist(inputs=[arm_two.score, dep], grid=(HIST_TILES * TG, 1, 1),
                        threadgroup=(TG, 1, 1), output_shapes=[[HIST_BINS]],
                        output_dtypes=[mx.uint32], init_value=0)[0]
    ctl = arm_two.thresh(inputs=[bins, dep], grid=(HIST_GROUPS, 1, 1),
                         threadgroup=(HIST_GROUPS, 1, 1), output_shapes=[[4]],
                         output_dtypes=[mx.uint32], init_value=0)[0]
    surv = arm_two.compact(inputs=[arm_two.score, ctl, dep],
                           grid=(HIST_TILES * TG, 1, 1), threadgroup=(TG, 1, 1),
                           output_shapes=[[arm_two.survivors],
                                          [arm_two.survivors], [1]],
                           output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
                           init_value=0)
    mx.eval(bins, ctl, surv[0], surv[1], surv[2])
    kept = set(surv[1].tolist())
    emitted = int(surv[2].item())
    # Positive control. Raise the single worst row above every other row: it
    # must appear in the survivor set. A gate that cannot fail is not a gate.
    worst = int(mx.argmin(arm_two.score.astype(mx.float32)).item())
    damaged = mx.array(arm_two.score.astype(mx.float32))
    damaged[worst] = float(mx.max(arm_two.score.astype(mx.float32)).item()) + 1.0
    damaged = damaged.astype(mx.bfloat16)
    dbins = arm_two.hist(inputs=[damaged, dep], grid=(HIST_TILES * TG, 1, 1),
                         threadgroup=(TG, 1, 1), output_shapes=[[HIST_BINS]],
                         output_dtypes=[mx.uint32], init_value=0)[0]
    dctl = arm_two.thresh(inputs=[dbins, dep], grid=(HIST_GROUPS, 1, 1),
                          threadgroup=(HIST_GROUPS, 1, 1), output_shapes=[[4]],
                          output_dtypes=[mx.uint32], init_value=0)[0]
    dsurv = arm_two.compact(inputs=[damaged, dctl, dep],
                            grid=(HIST_TILES * TG, 1, 1),
                            threadgroup=(TG, 1, 1),
                            output_shapes=[[arm_two.survivors],
                                           [arm_two.survivors], [1]],
                            output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
                            init_value=0)
    mx.eval(dsurv[1])
    control_ok = worst in set(dsurv[1].tolist()) and worst not in kept
    return {
        "positive_control_worst_row": worst,
        "positive_control_detects_the_change": control_ok,
        "survivors_requested": arm_two.survivors,
        "rows_that_passed_the_threshold": emitted,
        "distinct_rows_kept": len(kept),
        "true_top32_kept": len(true_top & kept),
        "recall_of_true_top32": len(true_top & kept) / TOPK,
        "histogram_total": int(mx.sum(bins).item()),
        "histogram_covers_every_row": int(mx.sum(bins).item()) == arm_two.rows,
        "tau_bin": int(ctl[0].item()),
        "rows_strictly_above_tau": int(ctl[1].item()),
    }


ANCHOR_PATH = "research/e136-anchor/bench.json"


def load_anchor() -> dict:
    """The shipped selection as the real runtime measures it.

    The suite `QwenRowTop32SelectionTests` times the live `Qwen35RowTop32`
    with the same per-call shape used here. Without that number, a Python
    port of the kernels prices a candidate against a baseline nobody checked.
    Regenerate it with:

      MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \\
      MLXFAST_ROW_TOP32_OUT_DIR=$PWD/research/e136-anchor \\
      swift test --force-resolved-versions --filter QwenRowTop32SelectionTests
    """
    with open(ANCHOR_PATH) as f:
        a = json.load(f)
    a["source"] = ANCHOR_PATH
    return a


ANCHOR = None


def host_facts() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return "unknown"
    return {
        "machine": sh("sysctl", "-n", "hw.model"),
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "os": platform.platform(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "commit": sh("git", "rev-parse", "HEAD"),
        "worktree_clean": sh("git", "status", "--porcelain") == "",
    }


def main() -> None:
    global ANCHOR
    ANCHOR = load_anchor()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e136-selection-bench.json")
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--reps", default="1,2,4,8,16,32")
    ap.add_argument("--survivors", default="1024,4096,8192")
    ap.add_argument("--tiles", default="auto")
    ap.add_argument("--chain", action="store_true",
                    help="also run the serialised chain-slope harness")
    args = ap.parse_args()

    def tiles_for(n: int) -> int:
        return stage2_tiles(n) if args.tiles == "auto" else int(args.tiles)

    reps_grid = [int(v) for v in args.reps.split(",")]
    probes_ship = math.ceil(PROBE_SHIPPED * CLUSTERS)
    probes_c1 = math.ceil(PROBE_C1 * CLUSTERS)
    rows_ship = probes_ship * ROWS_PER_CLUSTER
    rows_c1 = probes_c1 * ROWS_PER_CLUSTER

    out = {
        "harness": "local-microbenchmark",
        "timing_valid": True,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "host": host_facts(),
        "shape": {
            "clusters": CLUSTERS, "rows_per_cluster": ROWS_PER_CLUSTER,
            "probe_fraction_shipped": PROBE_SHIPPED,
            "probe_fraction_c1": PROBE_C1,
            "probed_clusters_shipped": probes_ship,
            "probed_clusters_c1": probes_c1,
            "probed_rows_shipped": rows_ship,
            "probed_rows_c1": rows_c1,
            "hist_bins": HIST_BINS, "hist_tiles": HIST_TILES,
            "shipped_top32_tiles": ROW_TOP32_TILES,
            "stage2_tiles_policy": args.tiles,
        },
        # The same quantity measured by the real runtime, `swift test --filter
        # QwenRowTop32SelectionTests` on this host and commit. The shipped arm
        # below must land near it or the port is not the shipped kernel.
        "swift_anchor": ANCHOR,
        "conversion": {
            "pct_per_mb": PCT_PER_MB,
            "byte_ceiling_gb_s": BYTE_CEILING_GB_S,
            "us_per_ranked_pct": US_PER_RANKED_PCT,
        },
        "arms": {}, "correctness": {}, "verdict": {},
    }

    def run(arm) -> dict:
        row = time_calls(arm, args.iters, args.trials)
        if args.chain:
            row["chain"] = time_arm(arm, reps_grid, args.trials)
        out["arms"][arm.label] = row
        print(f"  {arm.label:22s} {row['us_per_call']:9.3f} us/call "
              f"[{row['us_per_call_min']:9.3f}, {row['us_per_call_max']:9.3f}]")
        return row

    null = run(NullArm())
    floor = null["us_per_call"]

    ship = ShippedArm(rows_ship, probes_ship)
    ship_row = run(ship)
    ship_row["plan"] = ship.plan.as_dict()
    for part in ship.parts():
        run(part)
    out["anchor_check"] = {
        "swift_fused_kernel_us": ANCHOR["fused_kernel_us"],
        "python_shipped_us_per_call": ship_row["us_per_call"],
        "ratio_python_over_swift":
            ship_row["us_per_call"] / ANCHOR["fused_kernel_us"],
        "submit_and_sync_floor_us": floor,
        "swift_minus_floor_us": ANCHOR["fused_kernel_us"] - floor,
        "python_minus_floor_us": ship_row["us_per_call"] - floor,
    }

    for n in [int(v) for v in args.survivors.split(",")]:
        arm = TwoStageArm(rows_c1, probes_c1, n, tiles_for(n),
                          with_rescore=True)
        row = run(arm)
        row["plan"] = arm.plan.as_dict()
        row["added_us_per_step"] = (row["us_per_call"]
                                    - ship_row["us_per_call"])
        row["added_ranked_pct"] = row["added_us_per_step"] / US_PER_RANKED_PCT
        parts = {p.label.split(".", 1)[1]: run(p)["us_per_call"] - floor
                 for p in arm.parts()}
        ship_parts = {"partial":
                      out["arms"]["shipped.partial"]["us_per_call"] - floor,
                      "finalize":
                      out["arms"]["shipped.finalize"]["us_per_call"] - floor}
        row["parts_us_above_floor"] = parts
        row["shipped_parts_us_above_floor"] = ship_parts
        # Independent estimate of the same difference: sum the priced
        # dispatches instead of trusting one whole-arm subtraction. Both arms
        # pay exactly one submit-and-sync per call, so the floor cancels and
        # is not added back. What this sum omits is the in-buffer cost of
        # having more dispatches at all, which the whole-arm number carries,
        # so the two estimates bracket the true added cost.
        row["added_us_from_parts"] = (
            sum(parts.values()) - sum(ship_parts.values()))
        out["correctness"][arm.label] = correctness(arm, ship)
        print(f"  -> {arm.label} added {row['added_us_per_step']:8.3f} us "
              f"(parts {row['added_us_from_parts']:8.3f} us) "
              f"= {row['added_ranked_pct']:6.3f} % ranked, "
              f"recall {out['correctness'][arm.label]['recall_of_true_top32']}")

    selected = out["arms"].get("N4096")
    if selected:
        added = selected["added_us_per_step"]
        out["verdict"] = {
            "selected_cell": "qlowrank256-N4096-p0.35",
            "added_us_per_draft_step": added,
            "added_us_per_draft_step_from_parts":
                selected["added_us_from_parts"],
            "added_ranked_pct_at_265gbs": added / US_PER_RANKED_PCT,
            "added_ranked_pct_at_measured_186_7gbs":
                added / ((1.0 / PCT_PER_MB) * 1e6 / (186.7 * 1e3)),
            "predicted_c1_gross_pct": 0.678,
            "net_pct_after_selection_cost": 0.678 - added / US_PER_RANKED_PCT,
            "stop_rule": ("proceed" if added < 25.0 else
                          "fallback" if added <= 70.0 else "stop"),
        }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["verdict"], indent=2))
    print(json.dumps(out["anchor_check"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
