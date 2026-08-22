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
HIST_BINS = 256
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
    """Stage 1(i): a 256-bin histogram of the top ordinal byte.

    Threadgroup atomics keep the device traffic at one merged word per bin per
    threadgroup. The bin key is the top byte of `qwen_top32_ordinal`, which is
    monotone in the score, so bin order IS score order.
    """
    return DEP_GUARD + f"""
        constexpr uint ROWS  = {rows};
        constexpr uint GRID  = {HIST_TILES * TG};
        constexpr uint NBINS = {HIST_BINS};

        uint tid = thread_position_in_threadgroup.x;
        uint gid = thread_position_in_grid.x;

        threadgroup atomic_uint local[NBINS];
        atomic_store_explicit(&local[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = gid; i < ROWS; i += GRID) {{
            uint b = qwen_top32_ordinal(float(score[i])) >> 24;
            atomic_fetch_add_explicit(&local[b], 1u, memory_order_relaxed);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint v = atomic_load_explicit(&local[tid], memory_order_relaxed);
        if (v != 0u) {{
            atomic_fetch_add_explicit(
                (device atomic_uint *)&bins[tid], v, memory_order_relaxed);
        }}
        """


def thresh_source(survivors: int) -> str:
    """Stage 1(ii): one threadgroup finds the boundary bin.

    `ctl` is [tau, count_strictly_above_tau, capacity_left_in_tau, cursor].
    The scan is 256 serial adds in one thread: at this width a parallel scan
    costs more in barriers than it saves in adds.
    """
    return DEP_GUARD + f"""
        constexpr uint NBINS = {HIST_BINS};
        constexpr uint WANT  = {survivors};

        uint tid = thread_position_in_threadgroup.x;
        threadgroup uint c[NBINS];
        c[tid] = bins[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {{
            uint acc = 0u, tau = 0u, above = 0u;
            for (int b = int(NBINS) - 1; b >= 0; --b) {{
                uint prev = acc;
                acc += c[b];
                if (acc >= WANT) {{ tau = uint(b); above = prev; break; }}
                if (b == 0) {{ tau = 0u; above = prev; }}
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
            uint b = o >> 24;
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
        c = self.partial([self.score, dep],
                         grid=(self.plan.tiles * TG, 1, 1),
                         threadgroup=(TG, 1, 1),
                         output_shapes=[[self.plan.cands], [self.plan.cands]],
                         output_dtypes=[mx.uint32, mx.uint32])
        return self.finalize([c[0], c[1], self.probed, self.perm, dep],
                             grid=(TG, 1, 1), threadgroup=(TG, 1, 1),
                             output_shapes=[[TOPK]],
                             output_dtypes=[mx.uint32])[0]

    def seed(self):
        return mx.zeros([TOPK], dtype=mx.uint32)


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
        bins = self.hist([self.score, dep],
                         grid=(HIST_TILES * TG, 1, 1), threadgroup=(TG, 1, 1),
                         output_shapes=[[HIST_BINS]],
                         output_dtypes=[mx.uint32], init_value=0)[0]
        ctl = self.thresh([bins, dep], grid=(HIST_BINS, 1, 1),
                          threadgroup=(HIST_BINS, 1, 1),
                          output_shapes=[[4]], output_dtypes=[mx.uint32],
                          init_value=0)[0]
        surv = self.compact([self.score, ctl, dep],
                            grid=(HIST_TILES * TG, 1, 1),
                            threadgroup=(TG, 1, 1),
                            output_shapes=[[self.survivors],
                                           [self.survivors], [1]],
                            output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
                            init_value=0)
        surv_ord, surv_row = surv[0], surv[1]
        if self.with_rescore:
            score = self.rescore(
                [self.exact, surv_row, dep],
                grid=(self.survivors, 1, 1), threadgroup=(TG, 1, 1),
                output_shapes=[[self.survivors]],
                output_dtypes=[mx.bfloat16], init_value=0)[0]
        else:
            score = surv_ord
        c = self.partial([score, surv_row, dep],
                         grid=(self.plan.tiles * TG, 1, 1),
                         threadgroup=(TG, 1, 1),
                         output_shapes=[[self.plan.cands], [self.plan.cands]],
                         output_dtypes=[mx.uint32, mx.uint32])
        return self.finalize([c[0], c[1], self.probed, self.perm, dep],
                             grid=(TG, 1, 1), threadgroup=(TG, 1, 1),
                             output_shapes=[[TOPK]],
                             output_dtypes=[mx.uint32])[0]

    def seed(self):
        return mx.zeros([TOPK], dtype=mx.uint32)


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
        return self.k([dep], grid=(TG, 1, 1), threadgroup=(TG, 1, 1),
                      output_shapes=[[TOPK]], output_dtypes=[mx.uint32])[0]

    def seed(self):
        return mx.zeros([TOPK], dtype=mx.uint32)


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
    bins = arm_two.hist([arm_two.score, dep], grid=(HIST_TILES * TG, 1, 1),
                        threadgroup=(TG, 1, 1), output_shapes=[[HIST_BINS]],
                        output_dtypes=[mx.uint32], init_value=0)[0]
    ctl = arm_two.thresh([bins, dep], grid=(HIST_BINS, 1, 1),
                         threadgroup=(HIST_BINS, 1, 1), output_shapes=[[4]],
                         output_dtypes=[mx.uint32], init_value=0)[0]
    surv = arm_two.compact([arm_two.score, ctl, dep],
                           grid=(HIST_TILES * TG, 1, 1), threadgroup=(TG, 1, 1),
                           output_shapes=[[arm_two.survivors],
                                          [arm_two.survivors], [1]],
                           output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
                           init_value=0)
    mx.eval(bins, ctl, surv[0], surv[1], surv[2])
    kept = set(surv[1].tolist())
    emitted = int(surv[2].item())
    return {
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e136-selection-bench.json")
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--reps", default="1,2,4,8,16,32")
    ap.add_argument("--survivors", default="1024,4096,8192")
    ap.add_argument("--tiles", type=int, default=ROW_TOP32_TILES)
    args = ap.parse_args()

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
            "top32_tiles": args.tiles,
        },
        "conversion": {
            "pct_per_mb": PCT_PER_MB,
            "byte_ceiling_gb_s": BYTE_CEILING_GB_S,
            "us_per_ranked_pct": US_PER_RANKED_PCT,
        },
        "arms": {}, "correctness": {}, "verdict": {},
    }

    null = NullArm()
    out["arms"]["null_dispatch"] = time_arm(null, reps_grid, args.trials)
    print(f"null dispatch  {out['arms']['null_dispatch']['us_per_step']:8.3f} us")

    ship = ShippedArm(rows_ship, probes_ship)
    out["arms"]["shipped"] = time_arm(ship, reps_grid, args.trials)
    out["arms"]["shipped"]["plan"] = ship.plan.as_dict()
    print(f"shipped N=32   {out['arms']['shipped']['us_per_step']:8.3f} us")

    for n in [int(v) for v in args.survivors.split(",")]:
        arm = TwoStageArm(rows_c1, probes_c1, n, args.tiles, with_rescore=True)
        row = time_arm(arm, reps_grid, args.trials)
        row["plan"] = arm.plan.as_dict()
        row["added_us_per_step"] = (row["us_per_step"]
                                    - out["arms"]["shipped"]["us_per_step"])
        row["added_ranked_pct"] = row["added_us_per_step"] / US_PER_RANKED_PCT
        out["arms"][arm.label] = row
        out["correctness"][arm.label] = correctness(arm, ship)
        print(f"two-stage {arm.label:6s} {row['us_per_step']:8.3f} us  "
              f"added {row['added_us_per_step']:7.3f} us  "
              f"= {row['added_ranked_pct']:6.3f} % ranked  "
              f"recall {out['correctness'][arm.label]['recall_of_true_top32']}")

    selected = out["arms"].get("N4096")
    if selected:
        added = selected["added_us_per_step"]
        out["verdict"] = {
            "selected_cell": "qlowrank256-N4096-p0.35",
            "added_us_per_draft_step": added,
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
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
