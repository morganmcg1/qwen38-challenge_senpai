#!/usr/bin/env python3
"""E93 -- reduce the per-draft proposal-head dispatch census.

    usage:
      research/e93_head_census.py counts LEG [LEG ...]
      research/e93_head_census.py gputime LEG [LEG ...]
      research/e93_head_census.py nnls LEG [LEG ...]

`counts` fits every `draft_head` dispatch count to

    dispatches(d) = A + B * (d - 1)

where `d` is the round's draft count. `A` is the first draft step, which also
carries the round's history flush, and `B` is the marginal draft step. The fit
is exact on integer counts, so a residual means the model is wrong rather than
noisy.

`gputime` reads the E80 Metal command-buffer clock. In the default buffer
geometry `by_width_phase` gives the IN-SITU head pass. With one op per command
buffer `exclusive_kernels` gives each shape's ISOLATED GPU time.

Every byte figure below is derived from the tensor shape and dtype recorded in
`mtp-head-declared-run/model.safetensors.index.json` plus the dispatch grid, and
never from a guess.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

# --- the per-draft class map -------------------------------------------------
#
# Key is the census `shapes` string. Value is
#   (class, tensor, weight_bytes_read, activation_bytes_moved)
#
# `weight_bytes_read` counts persistent head or target weights only.
# `activation_bytes_moved` counts read plus write of transient tensors, and is
# reported separately because it is not part of the 427,738,112 byte head model.

AFFINE4 = lambda k, n: n * k // 2 + 4 * (n * k // 64)   # noqa: E731  4 bits + bf16 scale + bf16 bias
AFFINE2 = lambda k, n: n * k // 4 + 4 * (n * k // 64)   # noqa: E731
BF16 = lambda n: 2 * n                                   # noqa: E731

CLASS_NAMES = {
    1: "weight-streaming GEMV",
    2: "attention over head history",
    3: "norms and elementwise",
    4: "readout and rerank",
    5: "island scatter",
}

# The three O = 5120 affine-4 GEMVs share one grid, so they are one row with a
# multiplicity of three. Their input widths differ (10240, 6144, 17408) but the
# qmv grid encodes the output width only.
FC_BYTES = AFFINE4(10240, 5120)
O_BYTES = AFFINE4(6144, 5120)
DOWN_BYTES = AFFINE4(17408, 5120)

CACHE_CAPACITY = 768        # bf16 [1, 4, capacity, 256]; grows by 256 steps
CACHE_ARRAY_BYTES = 2 * 4 * CACHE_CAPACITY * 256

VN_COPY_PREFIX = "vn_copybfloat16bfloat16 grid="
VN_COPY_CANONICAL = "vn_copybfloat16bfloat16 grid=196608x1x1 tg=1024x1x1"

MAP = {
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x640x1 tg=32x2x1":
        (1, "fc + o_proj + down_proj (3 dispatches, O=5120)",
         FC_BYTES + O_BYTES + DOWN_BYTES, 0),
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x1536x1 tg=32x2x1":
        (1, "q_proj (O=12288)", AFFINE4(5120, 12288), 0),
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x4352x1 tg=32x2x1":
        (1, "mlp gate_up fused (O=34816)", AFFINE4(5120, 34816), 0),
    "gemv_al_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0 grid=64x1x1 tg=32x1x4":
        (1, "Q precision island, dense bf16 [1024,5120]", BF16(1024 * 5120), 0),
    "gemv_al_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0 grid=128x1x1 tg=32x1x4":
        (1, "K/V exact dense bf16 [2048,5120]", BF16(2048 * 5120), 0),

    "custom_kernel_qwen35_attention_qk_rms_rope_bf16_v1_bfloat16_t_bfloat16_t_"
    "bfloat16_t_bfloat16_t_floats_int32_ts_floats_bfloat16_t_bfloat16_t "
    "grid=1792x1x1 tg=64x1x1":
        (2, "q_norm + k_norm + RoPE (28 heads)", BF16(2 * 256), BF16(2 * 28 * 256)),
    "gg2_copybfloat16bfloat16 grid=256x4x1 tg=256x4x1":
        (2, "K and V cache append, 1 position (2 dispatches)", 0, 2 * 2 * BF16(4 * 256)),
    "sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks grid=24x1x1 tg=1024x1x1":
        (2, "SDPA over head history", 0, 2 * BF16(4 * 512 * 256)),
    "vn_copybfloat16bfloat16 grid=196608x1x1 tg=1024x1x1":
        (2, "head KV cache full-array copy (2 dispatches, capacity-sized)",
         0, 2 * 2 * CACHE_ARRAY_BYTES),

    "custom_kernel_qwen35_embed_dual_rms_norm_concat_bf16_v1_int32_tc_uint32_t_"
    "bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_floats_bfloat16_t "
    "grid=2048x1x1 tg=1024x1x1":
        (3, "fused quantized-embed dual RMSNorm concat",
         AFFINE4(5120, 1) + BF16(2 * 5120), BF16(3 * 5120)),
    "custom_kernel_qwen35_fused_residual_rms_norm_bfloat16_t_bfloat16_t_"
    "bfloat16_t_floats_bfloat16_t_bfloat16_t grid=1024x1x1 tg=1024x1x1":
        (3, "post_attention_layernorm, fused residual", BF16(5120), BF16(3 * 5120)),
    "rms_loopedbfloat16 grid=1024x1x1 tg=1024x1x1":
        (3, "input_layernorm and mtp.norm (2 dispatches)",
         2 * BF16(5120), 2 * BF16(2 * 5120)),
    "vv_Addbfloat16 grid=5120x1x1 tg=1024x1x1":
        (3, "attention residual add", 0, BF16(3 * 5120)),
    "CV2ISigmoidADV2IMultiplyACEV2OMultiplyDB_VV_V2V2_"
    "11160318154034397263_contiguous grid=17408x1x1 tg=1024x1x1":
        (3, "SwiGLU silu(gate) * up", 0, BF16(3 * 17408)),
    "CV2ISigmoidBDV2IBroadcastACEV2IBroadcastCAFV2OMultiplyDE_VV_V2V2_"
    "11160318154034397263_strided_2 grid=256x24x1 tg=64x16x1":
        (3, "attention output sigmoid gate", 0, BF16(3 * 6144)),
    "g1_copybfloat16bfloat16 grid=17408x1x1 tg=1024x1x1":
        (3, "gate/up split copy", 0, BF16(2 * 17408)),

    "affine_qmv_fast_bfloat16_t_gs_64_b_2_batch_0 grid=1x12292x1 tg=32x2x1":
        (4, "draft_lm_head coarse readout, affine-2 [98336,5120]",
         AFFINE2(5120, 98336), 0),
    "custom_kernel_qwen_mtp_draft_top32_partial_bfloat16_t_uint32_t_uint32_t "
    "grid=16384x1x1 tg=256x1x1":
        (4, "top-32 partial reduction over 98336 logits", 0, BF16(98336)),
    "custom_kernel_qwen_mtp_draft_top32_finalize_uint32_t_uint32_t_uint32_t "
    "grid=256x1x1 tg=256x1x1":
        (4, "top-32 finalize", 0, 4 * 256 * 64),
    "affine_gather_qmv_bfloat16_t_gs_64_b_4 grid=1x1x32 tg=32x2x1":
        (4, "gatherQuantizedMM, 32 rows of the target compact head",
         AFFINE4(5120, 32), 0),
    "custom_kernel_qwen_mtp_draft_rerank__98304_149740_bfloat16_t_uint32_t_int32_t "
    "grid=32x1x1 tg=32x1x1":
        (4, "exact rerank over the 32 candidates", 0, BF16(32)),

    "scatter_axisbfloat16int32_none_intcc grid=1x1024x1 tg=1x1024x1":
        (5, "replaceExactRows putAlong scatter, 1024 Q island rows",
         4 * 1024, 2 * BF16(1024)),
}

# Dispatches of each shape in ONE marginal draft step. `fit_counts` measures
# these; the table repeats them so the GPU-time reduction can weight an
# isolated per-dispatch cost without re-reading a count leg.
COUNTS = {
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x640x1 tg=32x2x1": 3,
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x1536x1 tg=32x2x1": 1,
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x4352x1 tg=32x2x1": 1,
    "gemv_al_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0 grid=64x1x1 tg=32x1x4": 1,
    "gemv_al_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0 grid=128x1x1 tg=32x1x4": 1,
    "custom_kernel_qwen35_attention_qk_rms_rope_bf16_v1_bfloat16_t_bfloat16_t_"
    "bfloat16_t_bfloat16_t_floats_int32_ts_floats_bfloat16_t_bfloat16_t "
    "grid=1792x1x1 tg=64x1x1": 1,
    "gg2_copybfloat16bfloat16 grid=256x4x1 tg=256x4x1": 2,
    "sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks grid=24x1x1 tg=1024x1x1": 1,
    VN_COPY_CANONICAL: 2,
    "custom_kernel_qwen35_embed_dual_rms_norm_concat_bf16_v1_int32_tc_uint32_t_"
    "bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_floats_bfloat16_t "
    "grid=2048x1x1 tg=1024x1x1": 1,
    "custom_kernel_qwen35_fused_residual_rms_norm_bfloat16_t_bfloat16_t_"
    "bfloat16_t_floats_bfloat16_t_bfloat16_t grid=1024x1x1 tg=1024x1x1": 1,
    "rms_loopedbfloat16 grid=1024x1x1 tg=1024x1x1": 2,
    "vv_Addbfloat16 grid=5120x1x1 tg=1024x1x1": 1,
    "CV2ISigmoidADV2IMultiplyACEV2OMultiplyDB_VV_V2V2_"
    "11160318154034397263_contiguous grid=17408x1x1 tg=1024x1x1": 1,
    "CV2ISigmoidBDV2IBroadcastACEV2IBroadcastCAFV2OMultiplyDE_VV_V2V2_"
    "11160318154034397263_strided_2 grid=256x24x1 tg=64x16x1": 1,
    "g1_copybfloat16bfloat16 grid=17408x1x1 tg=1024x1x1": 1,
    "affine_qmv_fast_bfloat16_t_gs_64_b_2_batch_0 grid=1x12292x1 tg=32x2x1": 1,
    "custom_kernel_qwen_mtp_draft_top32_partial_bfloat16_t_uint32_t_uint32_t "
    "grid=16384x1x1 tg=256x1x1": 1,
    "custom_kernel_qwen_mtp_draft_top32_finalize_uint32_t_uint32_t_uint32_t "
    "grid=256x1x1 tg=256x1x1": 1,
    "affine_gather_qmv_bfloat16_t_gs_64_b_4 grid=1x1x32 tg=32x2x1": 1,
    "custom_kernel_qwen_mtp_draft_rerank__98304_149740_bfloat16_t_uint32_t_int32_t "
    "grid=32x1x1 tg=32x1x1": 1,
    "scatter_axisbfloat16int32_none_intcc grid=1x1024x1 tg=1x1024x1": 1,
}
MARGINAL_DISPATCHES = 27
assert set(COUNTS) == set(MAP)
assert sum(COUNTS.values()) == MARGINAL_DISPATCHES

# Measured on the in-situ leg: the default buffer geometry runs one marginal
# draft step as a 25-dispatch body buffer plus a standalone embed buffer and a
# standalone rerank buffer, both forced by a host synchronisation.
BUFFERS_PER_MARGINAL_DRAFT = 3

# Weights owned by the TARGET, not by the declared head artifact. They are read
# on the head path but must not be charged to the 427,738,112 byte head model.
TARGET_OWNED_BYTES = AFFINE4(5120, 1) + AFFINE4(5120, 32)
HEAD_ARTIFACT_BYTES = 427_738_112


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(shape):
    """Fold the head KV cache copy onto one key.

    Its grid is the cache CAPACITY, which steps by 256 positions, so the same
    logical dispatch appears under a new key every time the cache grows. The
    capacity itself is reported separately.
    """
    if shape.startswith(VN_COPY_PREFIX):
        return VN_COPY_CANONICAL
    return shape


def fit_counts(paths):
    """Fit A + B*(d-1) per shape over the STEADY-STATE `draft_head` rounds.

    Two kinds of round are excluded and counted:

      flush      the first drafting round of a process also flushes the whole
                 512-position seed history into the head cache, so its first
                 step runs prefill-width shapes that no later round runs;
      growth     a round in which the head KV cache crosses a 256-position
                 capacity step pays one extra concatenate.

    Both are real, but neither is the marginal draft step this census prices.
    """
    per_depth = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(list)
    seen_pids = set()
    dropped = {"flush": 0, "growth": 0}
    capacities = defaultdict(int)
    for path in paths:
        rounds = [r for r in load(path)
                  if r.get("event") == "round" and "draft_head" in r.get("phases", {})]
        modes = defaultdict(lambda: defaultdict(int))
        for record in rounds:
            modes[record["width"] - 1][record["phases"]["draft_head"]["dispatches"]] += 1
        for record in rounds:
            phase = record["phases"]["draft_head"]
            depth = record["width"] - 1
            key = (path, record["pid"])
            if key not in seen_pids:
                seen_pids.add(key)
                dropped["flush"] += 1
                continue
            modal = max(modes[depth].items(), key=lambda kv: kv[1])[0]
            if phase["dispatches"] != modal:
                dropped["growth"] += 1
                continue
            totals[depth].append(phase["dispatches"])
            for shape, count in phase.get("shapes", {}).items():
                if shape.startswith(VN_COPY_PREFIX):
                    grid = int(shape.split("grid=")[1].split("x")[0])
                    capacities[grid * 4 // (4 * 256)] += count
                per_depth[depth][canonical(shape)].append(count)
    return per_depth, totals, dropped, capacities


def solve(pairs):
    """Least-squares A, B for count = A + B*(d-1) from {d: [counts]}."""
    xs, ys = [], []
    for depth, counts in pairs.items():
        for count in counts:
            xs.append(depth - 1)
            ys.append(count)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if var == 0 else sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var
    intercept = mean_y - slope * mean_x
    residual = max(abs(y - (intercept + slope * x)) for x, y in zip(xs, ys))
    return intercept, slope, residual


def report_counts(paths):
    per_depth, totals, dropped, capacities = fit_counts(paths)
    depths = sorted(per_depth)
    print(f"legs: {len(paths)}   draft depths observed: {depths}")
    print(f"rounds dropped: seed-history-flush={dropped['flush']} "
          f"cache-growth={dropped['growth']}")
    print("head KV cache capacity observed (positions -> copy dispatches): "
          + ", ".join(f"{c}->{n}" for c, n in sorted(capacities.items())))
    for depth in depths:
        counts = totals[depth]
        print(f"  d={depth:2d}  rounds={len(counts):3d}  "
              f"draft_head dispatches {sorted(set(counts))}")

    shapes = sorted({s for depth in per_depth for s in per_depth[depth]})
    rows = []
    for shape in shapes:
        pairs = {d: per_depth[d].get(shape, [0] * len(totals[d])) for d in depths}
        first, marginal, residual = solve(pairs)
        rows.append((shape, first, marginal, residual))

    # A shape whose count is an exact affine function of the draft count runs on
    # every marginal draft step. A shape with a residual runs only on the round's
    # FIRST head call, whose flush block is 1 + the previous round's acceptance
    # wide, so its grid moves with the flush width and the affine fit fails.
    marginal_rows = [r for r in rows if r[3] <= 0.01 and r[2] > 0.01]
    first_only = [r for r in rows if r[3] > 0.01 or r[2] <= 0.01]

    print()
    print("MARGINAL DRAFT STEP -- shapes whose count is exactly A + B*(d-1)")
    print(f"{'first':>7} {'marg':>6} {'res':>5}  class  shape")
    total_first = total_marginal = 0.0
    unmapped = []
    for shape, first, marginal, residual in marginal_rows:
        entry = MAP.get(shape)
        cls = entry[0] if entry else "?"
        if entry is None:
            unmapped.append(shape)
        total_first += first
        total_marginal += marginal
        print(f"{first:7.2f} {marginal:6.2f} {residual:5.2f}  {cls:>5}  {shape[:96]}")
    print(f"{total_first:7.2f} {total_marginal:6.2f}         TOTAL")
    if unmapped:
        print("\nUNMAPPED marginal shapes:")
        for shape in unmapped:
            print("  " + shape)

    print()
    print(f"FIRST HEAD CALL ONLY -- {len(first_only)} flush-width-dependent shapes, "
          "not part of the marginal draft step")

    print()
    print("per-class marginal dispatches and modelled bytes per draft step")
    by_class = defaultdict(lambda: [0.0, 0, 0])
    for shape, _first, marginal, _res in marginal_rows:
        entry = MAP.get(shape)
        if entry is None:
            continue
        cls, _tensor, weight_bytes, activation_bytes = entry
        by_class[cls][0] += marginal
        by_class[cls][1] += weight_bytes
        by_class[cls][2] += activation_bytes
    weight_total = activation_total = 0
    for cls in sorted(by_class):
        dispatches, weight_bytes, activation_bytes = by_class[cls]
        weight_total += weight_bytes
        activation_total += activation_bytes
        print(f"  class {cls} {CLASS_NAMES[cls]:<30} "
              f"dispatches={dispatches:5.2f} weight_bytes={weight_bytes:12,d} "
              f"activation_bytes={activation_bytes:11,d}")
    print(f"  {'':<38} weight total   = {weight_total:12,d}")
    print(f"  {'':<38} activation tot = {activation_total:12,d}")
    head_only = weight_total - TARGET_OWNED_BYTES
    delta = head_only - HEAD_ARTIFACT_BYTES
    print()
    print(f"  declared-head weight actually read  = {head_only:12,d}")
    print(f"  target-owned weight read on the path= {TARGET_OWNED_BYTES:12,d}")
    print(f"  campaign head artifact byte model   = {HEAD_ARTIFACT_BYTES:12,d}")
    print(f"  difference                          = {delta:12,d} "
          f"({100.0 * delta / HEAD_ARTIFACT_BYTES:+.3f} %)")
    print(f"  activation traffic, stated separately = {activation_total:,d} "
          f"({100.0 * activation_total / head_only:.2f} % of head weight bytes)")


def split_legs(snapshots):
    """The wrapper runs a serial leg and an MTP leg in separate processes, and
    each process restarts its snapshot index at 0."""
    legs, current = [], []
    for snapshot in snapshots:
        if snapshot["snapshot"] == 0 and current:
            legs.append(current)
            current = []
        current.append(snapshot)
    if current:
        legs.append(current)
    return legs


def accumulate(snapshots):
    totals = {
        "rounds": 0, "busy_ns": 0, "idle_ns": 0, "buffers": 0,
        "mixed": 0, "zero": 0, "untracked": 0, "unmapped": 0,
    }
    phase_ns = defaultdict(int)
    phase_dispatch = defaultdict(int)
    exclusive = defaultdict(lambda: [0, 0, 0])
    for snapshot in snapshots:
        totals["rounds"] += snapshot.get("rounds", 0)
        totals["busy_ns"] += snapshot.get("gpu_busy_ns", 0)
        totals["idle_ns"] += snapshot.get("gpu_idle_ns", 0)
        totals["buffers"] += snapshot.get("completed_buffers", 0)
        totals["mixed"] += snapshot.get("mixed_phase_buffers", 0)
        totals["zero"] += snapshot.get("zero_time_buffers", 0)
        totals["untracked"] += snapshot.get("untracked_buffers", 0)
        totals["unmapped"] += snapshot.get("unmapped_encoder_dispatches", 0)
        for key, bucket in snapshot.get("by_width_phase", {}).items():
            phase_ns[key] += bucket["gpu_ns"]
            phase_dispatch[key] += bucket["dispatches"]
        for key, bucket in snapshot.get("exclusive_kernels", {}).items():
            exclusive[key][0] += bucket["buffers"]
            exclusive[key][1] += bucket["gpu_ns"]
            exclusive[key][2] = max(exclusive[key][2], bucket["max_ns"])
    return totals, phase_ns, phase_dispatch, exclusive


# E85's published in-situ head pass and the assignment's validation band.
E85_HEAD_PASS_US = 2285.283
E85_BAND_US = (2261.0, 2381.0)


def report_gputime(paths, drafts=8, marginal=27, first_call=37):
    for path in paths:
        snapshots = [r for r in load(path) if r.get("event") == "gputime"]
        legs = split_legs(snapshots)
        print(f"=== {path}  snapshots={len(snapshots)}  "
              f"legs={[len(leg) for leg in legs]}")
        for index, leg in enumerate(legs):
            # Snapshot 0 carries weight loading, warmup and the seed prefill.
            totals, phase_ns, phase_dispatch, exclusive = accumulate(leg[1:])
            head_key = f"w{drafts + 1}|draft_head"
            name = "MTP" if head_key in phase_ns else "serial"
            print(f"\n--- leg {index} ({name})  steady snapshots="
                  f"{len(leg) - 1}  round counter={totals['rounds']}")
            print(f"    gpu_busy={totals['busy_ns'] / 1e6:9.2f} ms   "
                  f"gpu_idle={totals['idle_ns'] / 1e6:9.2f} ms   "
                  f"duty={100.0 * totals['busy_ns'] / max(totals['busy_ns'] + totals['idle_ns'], 1):5.1f} %")
            print(f"    buffers={totals['buffers']}  "
                  f"mixed_phase={totals['mixed']}  zero_time={totals['zero']}  "
                  f"untracked={totals['untracked']}  "
                  f"unmapped_dispatches={totals['unmapped']}")
            if totals["mixed"] == 0:
                print("    every buffer is single-phase: phase attribution is "
                      "exact, no dispatch-count split was used")
            total_ns = sum(phase_ns.values())
            print(f"    {'width|phase':<24}{'gpu_ms':>10}{'share':>9}"
                  f"{'dispatches':>12}{'us/dispatch':>13}")
            for key in sorted(phase_ns, key=lambda k: -phase_ns[k]):
                print(f"    {key:<24}{phase_ns[key] / 1e6:10.2f}"
                      f"{100.0 * phase_ns[key] / max(total_ns, 1):8.1f} %"
                      f"{phase_dispatch[key]:12d}"
                      f"{phase_ns[key] / 1e3 / max(phase_dispatch[key], 1):13.3f}")
            print(f"    {'TOTAL':<24}{total_ns / 1e6:10.2f}")

            if head_key in phase_ns:
                per_round = first_call + marginal * (drafts - 1)
                head_rounds = phase_dispatch[head_key] / per_round
                pass_us = phase_ns[head_key] / 1e3 / head_rounds / drafts
                print(f"\n    head rounds (dispatches / {per_round}) = "
                      f"{head_rounds:.2f}")
                print(f"    IN-SITU head pass = {pass_us:.1f} us per draft")
                low, high = E85_BAND_US
                verdict = "INSIDE" if low <= pass_us <= high else "OUTSIDE"
                print(f"    E85 published {E85_HEAD_PASS_US:.3f} us, band "
                      f"{low:.0f}-{high:.0f} us: {verdict} "
                      f"({100.0 * (pass_us - E85_HEAD_PASS_US) / E85_HEAD_PASS_US:+.2f} %, "
                      f"closure {100.0 - abs(100.0 * (pass_us - E85_HEAD_PASS_US) / E85_HEAD_PASS_US):.1f} %)")

            head = [(k, v) for k, v in exclusive.items() if "|draft_head|" in k]
            if not head:
                continue
            print("\n    ISOLATED draft_head kernels "
                  "(one dispatch per command buffer):")
            by_class = defaultdict(lambda: [0, 0.0])
            print(f"    {'cls':>4}{'n':>7}{'mean_us':>10}{'max_us':>9}"
                  f"{'x/draft':>9}{'us/draft':>10}  tensor / shape")
            for key, (buffers, gpu_ns, max_ns) in sorted(
                    head, key=lambda kv: -kv[1][1] / max(kv[1][0], 1)):
                shape = key.split("|", 2)[2]
                entry = MAP.get(canonical(shape))
                mean_us = gpu_ns / max(buffers, 1) / 1e3
                if entry is None:
                    print(f"    {'?':>4}{buffers:7d}{mean_us:10.3f}"
                          f"{max_ns / 1e3:9.3f}{'':>9}{'':>10}  "
                          f"UNMAPPED  {shape[:70]}")
                    continue
                cls, tensor, _, _ = entry
                per_draft = COUNTS.get(canonical(shape), 0)
                cost = mean_us * per_draft
                by_class[cls][0] += per_draft
                by_class[cls][1] += cost
                print(f"    {cls:>4}{buffers:7d}{mean_us:10.3f}"
                      f"{max_ns / 1e3:9.3f}{per_draft:9d}{cost:10.2f}  "
                      f"{tensor}")
            isolated = sum(value[1] for value in by_class.values())
            print(f"\n    {'class':>6}{'dispatches':>12}{'us/draft':>11}"
                  f"{'share':>9}  name")
            for cls in sorted(by_class):
                count, cost = by_class[cls]
                print(f"    {cls:>6}{count:12d}{cost:11.2f}"
                      f"{100.0 * cost / max(isolated, 1e-9):8.1f} %  "
                      f"{CLASS_NAMES.get(cls, '')}")
            print(f"    {'TOTAL':>6}{sum(v[0] for v in by_class.values()):12d}"
                  f"{isolated:11.2f}")


def nnls(matrix, target, iterations=500):
    """Lawson-Hanson non-negative least squares. No scipy on this host."""
    columns = matrix.shape[1]
    passive = np.zeros(columns, dtype=bool)
    solution = np.zeros(columns)
    residual = target - matrix @ solution
    for _ in range(iterations):
        gradient = matrix.T @ residual
        gradient[passive] = -np.inf
        best = int(np.argmax(gradient))
        if gradient[best] <= 1e-9:
            break
        passive[best] = True
        for _ in range(iterations):
            trial = np.zeros(columns)
            active = np.where(passive)[0]
            trial[active] = np.linalg.lstsq(
                matrix[:, active], target, rcond=None)[0]
            if trial[active].min() > 0:
                solution = trial
                break
            blocked = active[trial[active] <= 0]
            step = (solution[blocked]
                    / (solution[blocked] - trial[blocked] + 1e-300)).min()
            solution = solution + step * (trial - solution)
            passive[np.where(passive)[0][solution[passive] <= 1e-12]] = False
        residual = target - matrix @ solution
    return solution


def collect_signatures(snapshots, phase="draft_head", width=9):
    """Sum every single-phase buffer bucket for one phase and verify width."""
    buckets = defaultdict(lambda: [0, 0])
    prefix = f"w{width}|{phase}|"
    for snapshot in snapshots:
        for key, bucket in snapshot.get("signatures", {}).items():
            if not key.startswith(prefix):
                continue
            body = key[len(prefix):]
            buckets[body][0] += bucket["buffers"]
            buckets[body][1] += bucket["gpu_ns"]
    return buckets


def parse_signature(body):
    counts = defaultdict(int)
    for part in body.split(","):
        shape, _, multiplicity = part.rpartition("*")
        counts[canonical(shape)] += int(multiplicity)
    return counts


def report_nnls(paths, min_buffers=4, width=9):
    """Price each kernel inside the head pass from the per-buffer signatures.

    One command buffer reports one GPU interval, so a kernel can only be
    priced when the buffer boundaries move across it. `MLX_E58_BUFFER_LIMIT_OPS`
    supplies that variation, and every buffer then becomes one equation

        gpu_ns(buffer) = overhead + sum over shapes of count * cost(shape)

    solved under a non-negativity constraint.
    """
    for path in paths:
        snapshots = [r for r in load(path) if r.get("event") == "gputime"]
        legs = split_legs(snapshots)
        leg = max(legs, key=lambda candidate: sum(
            1 for snapshot in candidate
            if f"w{width}|draft_head" in snapshot.get("by_width_phase", {})))
        buckets = collect_signatures(leg[1:], width=width)
        kept = {body: value for body, value in buckets.items()
                if value[0] >= min_buffers}
        dropped = len(buckets) - len(kept)
        print(f"=== {path}")
        print(f"    steady snapshots={len(leg) - 1}  signatures={len(buckets)}"
              f"  kept(n>={min_buffers})={len(kept)}  dropped={dropped}")
        if not kept:
            print("    no signature reached the buffer threshold")
            continue
        shapes = sorted({shape for body in kept for shape in parse_signature(body)})
        index = {shape: position for position, shape in enumerate(shapes)}
        rows, target, weights = [], [], []
        for body, (buffers, gpu_ns) in kept.items():
            row = np.zeros(len(shapes) + 1)
            row[-1] = 1.0                      # per-buffer overhead
            for shape, count in parse_signature(body).items():
                row[index[shape]] = count
            weight = np.sqrt(buffers)
            rows.append(row * weight)
            target.append(gpu_ns / buffers * weight)
            weights.append(buffers)
        matrix = np.array(rows)
        vector = np.array(target)
        rank = np.linalg.matrix_rank(matrix)
        solution = nnls(matrix, vector)
        predicted = matrix @ solution
        residual = float(np.linalg.norm(vector - predicted))
        denominator = float(np.linalg.norm(vector - vector.mean()))
        print(f"    equations={matrix.shape[0]}  unknowns={matrix.shape[1]}"
              f"  rank={rank}"
              f"{'  RANK DEFICIENT' if rank < matrix.shape[1] else ''}")
        print(f"    weighted residual={residual / max(np.linalg.norm(vector), 1e-9):.4f}"
              f"  R2={1.0 - (residual / max(denominator, 1e-9)) ** 2:.4f}")
        overhead = solution[-1]

        # A coefficient is identifiable only when no null-space direction of the
        # design matrix moves it. Without this test a rank-deficient fit reports
        # an arbitrary split of two kernels that always share a buffer.
        _, singular, right = np.linalg.svd(matrix)
        tolerance = max(matrix.shape) * singular.max() * np.finfo(float).eps
        null_space = right[len(singular):] if len(singular) < right.shape[0] \
            else np.zeros((0, matrix.shape[1]))
        small = right[np.where(singular <= tolerance)[0]] if len(
            np.where(singular <= tolerance)[0]) else np.zeros(
                (0, matrix.shape[1]))
        null_space = np.vstack([null_space, small])
        if null_space.shape[0]:
            leverage = np.linalg.norm(null_space, axis=0)
        else:
            leverage = np.zeros(matrix.shape[1])
        identifiable = leverage <= 1e-8
        print(f"    identifiable coefficients: "
              f"{int(identifiable.sum())} of {matrix.shape[1]}")
        print(f"    fitted per-command-buffer overhead = {overhead / 1e3:.3f} us"
              f"{'' if identifiable[-1] else '   NOT IDENTIFIED'}")

        appearances = defaultdict(int)
        for body in kept:
            for shape in parse_signature(body):
                appearances[shape] += 1

        by_class = defaultdict(lambda: [0, 0.0, 0, 0])
        print(f"\n    {'cls':>4}{'x/draft':>8}{'sigs':>6}{'us/kernel':>11}"
              f"{'us/draft':>10}{'MB/draft':>10}{'GB/s':>9}  tensor")
        unmapped, unidentified = [], []
        for shape in sorted(shapes, key=lambda s: -solution[index[s]]
                            * COUNTS.get(s, 1)):
            position = index[shape]
            cost = solution[position] / 1e3
            entry = MAP.get(shape)
            if entry is None:
                unmapped.append((shape, cost, appearances[shape],
                                 identifiable[position]))
                continue
            cls, tensor, weight_bytes, activation_bytes = entry
            count = COUNTS[shape]
            total_us = cost * count
            # MAP byte fields already cover every dispatch of the shape in one
            # marginal step, so divide by the count to get one kernel's traffic.
            bytes_each = (weight_bytes + activation_bytes) / count
            by_class[cls][0] += count
            by_class[cls][1] += total_us
            by_class[cls][2] += weight_bytes
            by_class[cls][3] += activation_bytes
            if not identifiable[position]:
                unidentified.append(tensor)
            rate = (f"{bytes_each / cost / 1e3:9.1f}" if cost > 1e-3
                    else f"{'-':>9}")
            flag = "" if identifiable[position] else "   [NOT IDENTIFIED]"
            print(f"    {cls:>4}{count:8d}{appearances[shape]:6d}{cost:11.2f}"
                  f"{total_us:10.2f}{bytes_each * count / 1e6:10.2f}"
                  f"{rate}  {tensor}{flag}")
        modelled = sum(value[1] for value in by_class.values())
        overhead_total = overhead / 1e3 * BUFFERS_PER_MARGINAL_DRAFT
        print(f"\n    {'class':>6}{'disp':>6}{'us/draft':>10}{'share':>8}"
              f"{'MB/draft':>10}{'GB/s':>9}  name")
        for cls in sorted(by_class):
            count, total_us, weight_bytes, activation_bytes = by_class[cls]
            traffic = weight_bytes + activation_bytes
            rate = (f"{traffic / total_us / 1e3:9.1f}" if total_us > 1e-3
                    else f"{'-':>9}")
            print(f"    {cls:>6}{count:6d}{total_us:10.2f}"
                  f"{100.0 * total_us / max(modelled, 1e-9):7.1f} %"
                  f"{traffic / 1e6:10.2f}{rate}  {CLASS_NAMES.get(cls, '')}")
        print(f"    {'kernels':>6}{sum(v[0] for v in by_class.values()):6d}"
              f"{modelled:10.2f}")
        print(f"    {'buffer':>6}{BUFFERS_PER_MARGINAL_DRAFT:6d}"
              f"{overhead_total:10.2f}    per-command-buffer overhead")
        print(f"    {'TOTAL':>6}{'':6}{modelled + overhead_total:10.2f}")
        if unidentified:
            print(f"    NOT IDENTIFIED: {len(unidentified)} mapped kernels; "
                  "their split is arbitrary and only their group sum is real")
        for shape, cost, seen, ok in unmapped:
            print(f"    UNMAPPED us={cost:8.2f} sigs={seen:3d} "
                  f"{'id' if ok else '  '}  {shape[:88]}")


def report_buffers(paths, min_buffers=4, width=9, drafts=8):
    """Print the identification-free per-command-buffer cost of the head pass.

    One command buffer is one measured GPU interval, so this table needs no
    solver and no identifiability argument. The NNLS report then tries to split
    the buffers into kernels and is rank deficient; this table is not.
    """
    for path in paths:
        snapshots = [r for r in load(path) if r.get("event") == "gputime"]
        legs = split_legs(snapshots)
        leg = max(legs, key=lambda candidate: sum(
            1 for snapshot in candidate
            if f"w{width}|draft_head" in snapshot.get("by_width_phase", {})))
        steady = leg[1:]
        buckets = collect_signatures(steady, width=width)
        head_rounds = sum(
            snapshot["by_width_phase"][f"w{width}|draft_head"]["dispatches"]
            for snapshot in steady
            if f"w{width}|draft_head" in snapshot["by_width_phase"]
        )
        rounds = (head_rounds - 37 * len(steady)) / (27.0 * (drafts - 1)) if drafts > 1 else 0
        marginal = max(1.0, rounds) * drafts
        print(f"=== {path}")
        print(f"    steady snapshots={len(steady)}  head dispatches={head_rounds}"
              f"  implied rounds={rounds:.2f}  head calls={marginal:.2f}"
              f"  (every figure below is per head call, not per marginal draft)")
        rows = []
        total_us = 0.0
        total_bytes = 0
        for body, (count, gpu_ns) in buckets.items():
            if count < min_buffers:
                continue
            shapes = parse_signature(body)
            weight = sum(MAP[s][2] * n for s, n in shapes.items() if s in MAP)
            us = gpu_ns / 1000.0 / marginal
            total_us += us
            total_bytes += weight * (count / marginal)
            names = ", ".join(
                f"{MAP[s][1]}x{n}" if s in MAP else f"UNMAPPED[{s[:40]}]x{n}"
                for s, n in shapes.items())
            rows.append((us, count, count / marginal, weight, names))
        for us, count, per_draft, weight, names in sorted(rows, reverse=True):
            rate = (weight * per_draft / (us * 1e-6) / 1e9) if us > 0 and weight else 0.0
            print(f"    {us:9.2f} us/call   n={count:4d}  {per_draft:5.2f} buf/call"
                  f"  {weight / 1e6:8.3f} MB  {rate:7.1f} GB/s  {names[:100]}")
        print(f"    TOTAL {total_us:.2f} us/call over {len(rows)} buffer signatures,"
              f" {total_bytes / 1e6:.3f} MB/call")


HOST_PHASE_GATE_US = 1500.0


def report_hoststate(paths, gate_us=HOST_PHASE_GATE_US, warmup_rounds=1):
    """Stratify every leg by host state before any pooled GPU number is quoted.

    The gate is absolute and arm blind: a round is dirty when the host side of
    a phase spends more than `gate_us` outside its own dispatch encoding. Each
    figure carries its instruction counter (encoded dispatches, commits and
    barriers) so a host-state number can never be compared across rounds that
    encoded a different amount of work.
    """
    for path in paths:
        records = load(path)
        rounds = [r for r in records if r.get("event") == "round"]
        by_pid = {}
        for record in rounds:
            by_pid.setdefault(record["pid"], []).append(record)
        print(f"\n=== {path}")
        for pid in sorted(by_pid):
            leg = sorted(by_pid[pid], key=lambda r: r["round"])[warmup_rounds:]
            if not leg:
                continue
            phases = sorted({name for r in leg for name in r["phases"]})
            print(f"  pid {pid}: {len(leg)} rounds after dropping {warmup_rounds} warmup")
            for phase in phases:
                samples = []
                for record in leg:
                    body = record["phases"].get(phase)
                    if not body:
                        continue
                    samples.append(
                        (
                            body["dispatch_ns"] / 1000.0,
                            body["dispatches"],
                            body.get("commits", 0),
                            body.get("barriers", 0),
                            record["wall_ns"] / 1000.0,
                        )
                    )
                if not samples:
                    continue
                host = [s[0] for s in samples]
                counters = {s[1] for s in samples}
                clean = [s for s in samples if s[0] <= gate_us]
                dirty = [s for s in samples if s[0] > gate_us]
                mean_clean = sum(s[0] for s in clean) / len(clean) if clean else float("nan")
                mean_dirty = sum(s[0] for s in dirty) / len(dirty) if dirty else float("nan")
                flag = "" if len(clean) >= 20 else "   <<< SMALL CLEAN SAMPLE"
                print(
                    f"    {phase:<16} host {min(host):8.1f}..{max(host):9.1f} us  "
                    f"clean(<= {gate_us:.0f}us) {len(clean):4d}/{len(samples):4d} "
                    f"mean {mean_clean:9.1f}  dirty {len(dirty):4d} mean {mean_dirty:9.1f}"
                    f"{flag}"
                )
                counter_note = (
                    f"{sorted(counters)[0]}"
                    if len(counters) == 1
                    else f"{min(counters)}..{max(counters)} over {len(counters)} values"
                )
                print(
                    f"      instruction counter: dispatches {counter_note}, "
                    f"commits mean {sum(s[2] for s in samples) / len(samples):.1f}, "
                    f"barriers mean {sum(s[3] for s in samples) / len(samples):.1f}, "
                    f"host us/dispatch {sum(host) / len(host) / max(1, sorted(counters)[0]):.3f}"
                )


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode, paths = sys.argv[1], sys.argv[2:]
    if mode == "counts":
        report_counts(paths)
    elif mode == "gputime":
        report_gputime(paths)
    elif mode == "nnls":
        report_nnls(paths)
    elif mode == "hoststate":
        report_hoststate(paths)
    elif mode == "buffers":
        report_buffers(paths)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
