#!/usr/bin/env python3
"""E95 -- itemise the `target_verify` phase of one MTP round.

    usage:
      research/e95_verify_census.py counts   LEG [LEG ...]
      research/e95_verify_census.py buffers  LEG [LEG ...]
      research/e95_verify_census.py kernels  LEG [LEG ...]
      research/e95_verify_census.py gdn      LEG [LEG ...]

`counts` fits the per-round `target_verify` dispatch count of every shape at
every observed verify width M. It is an exact integer model, so a residual means
the model is wrong rather than noisy.

`buffers` prints the per-command-buffer table. One command buffer is one
measured GPU interval, so this table needs no solver and no identifiability
argument. It is the primary evidence.

`kernels` prints per-kernel ISOLATED GPU time from `exclusive_kernels`, which is
only complete on a leg run with `MLX_E58_BUFFER_LIMIT_OPS=1`. Isolation removes
intra-buffer concurrency, so it OVER-states each kernel; the ratio of its total
to the in-situ total is the concurrency discount.

`gdn` audits the Gated DeltaNet recurrent state, which covers 48 of the 64
layers and is snapshotted and rolled back on every speculative round.

Every byte figure is derived from the model geometry and the affine-4 group-64
packing, never from a guess. The geometry is checked by construction: the
per-layer sums below reproduce the organiser's 14,412,349,440 byte weight
stream exactly.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict

# --- model geometry ----------------------------------------------------------

HIDDEN = 5120
VOCAB = 248_320
ATTN_LAYERS = 16
GDN_LAYERS = 48
LAYERS = ATTN_LAYERS + GDN_LAYERS

ATTN_Q_HEADS = 24
ATTN_KV_HEADS = 4
ATTN_HEAD_DIM = 256
ATTN_V_DIM = ATTN_Q_HEADS * ATTN_HEAD_DIM          # 6144

GDN_K_HEADS = 16
GDN_V_HEADS = 48
GDN_HEAD_DIM = 128
GDN_CONV_DIM = 10_240
GDN_V_DIM = GDN_V_HEADS * GDN_HEAD_DIM             # 6144
GDN_STATE_BYTES = GDN_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM * 4   # fp32

MLP_INTERMEDIATE = 17_408


def affine4(k, o):
    """affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias."""
    return o * k // 2 + 4 * (o * k // 64)


def bf16(n):
    return 2 * n


# out_vec_size -> (name, [input widths that share the grid], per-round count)
QMV_CLASSES = {
    14_336: ("full-attention fused QKV+gate", [HIDDEN], ATTN_LAYERS),
    16_480: ("GDN in_proj (conv qkv + a/beta + gate)", [HIDDEN], GDN_LAYERS),
    34_816: ("MLP gate_up fused", [HIDDEN], LAYERS),
    5_120: ("out_proj and MLP down_proj", [ATTN_V_DIM, MLP_INTERMEDIATE], 2 * LAYERS),
    VOCAB: ("lm_head over the full vocabulary", [HIDDEN], 1),
}

# The stated total weight stream, reproduced from the class table.
WEIGHT_STREAM_BYTES = sum(
    sum(affine4(k, o) for k in ks) * (count // len(ks))
    for o, (_name, ks, count) in QMV_CLASSES.items()
)

# `qmv_fast` dispatches grid.x = M threadgroups (quantized.cpp:254). The wide
# cross-row kernel gives each x-group IPG input rows and returns immediately
# from every group whose first row is past M (quantized.h:1157-1180), so the
# weight stream is read G = ceil(M / IPG) times and M - G x-groups are launched
# for no work.
IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}


def groups(m):
    """Weight-stream passes for one wide qmv_fast dispatch at verify width M."""
    if m <= 2:
        return 1
    return math.ceil(m / IPG[m])


QMV_RE = re.compile(
    r"^affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=(\d+)x(\d+)x1")
GRID_RE = re.compile(r"grid=(\d+)x(\d+)x(\d+)")


def grid_of(shape):
    match = GRID_RE.search(shape)
    if not match:
        return None
    return tuple(int(g) for g in match.groups())


def qmv_bytes(shape, m):
    """(weight bytes, activation bytes) for one qmv_fast dispatch."""
    match = QMV_RE.match(shape)
    if not match:
        return None
    grid_m, grid_n = int(match.group(1)), int(match.group(2))
    out = grid_n * 8
    if out not in QMV_CLASSES:
        return None
    name, widths, per_round = QMV_CLASSES[out]
    passes = groups(grid_m)
    weight = passes * sum(affine4(k, out) for k in widths) / len(widths)
    activation = bf16(grid_m * (sum(widths) / len(widths) + out))
    return name, out, grid_m, passes, weight, activation


# Shapes outside the quantized projections. Value is
#   (name, weight bytes, activation bytes as a function of M, per-round count)
def other_bytes(shape, m):
    grid = grid_of(shape)
    if shape.startswith("custom_kernel_gated_delta_step__"):
        return ("GDN recurrent step, full [48,128,128] fp32 state",
                0, 2 * GDN_STATE_BYTES, GDN_LAYERS)
    if shape.startswith("custom_kernel_qwen35_gated_delta_step_mid__"):
        return ("GDN recurrent step (mid variant), full state",
                0, 2 * GDN_STATE_BYTES, 0)
    if shape.startswith("custom_kernel_qwen35_gated_delta_replay_state__"):
        return ("GDN recurrent state REPLAY, full [48,128,128] fp32 state",
                0, 2 * GDN_STATE_BYTES, 0)
    if shape.startswith("custom_kernel_qwen35_packed_gdn_prework__"):
        return ("GDN prework: causal conv1d, q/k norm, gates",
                bf16(3 * GDN_CONV_DIM), bf16(m * 3 * GDN_CONV_DIM), GDN_LAYERS)
    if shape.startswith("custom_kernel_qwen35_attention_qk_rms_rope"):
        return ("q_norm + k_norm + RoPE (28 heads)",
                bf16(2 * ATTN_HEAD_DIM),
                bf16(m * 28 * ATTN_HEAD_DIM), ATTN_LAYERS)
    if shape.startswith("sdpa_vector_"):
        rows = grid[1] if grid else m
        return (f"SDPA over the full-attention history ({rows} rows)",
                0, 0, 2 * ATTN_LAYERS)
    if shape.startswith("custom_kernel_qwen35_fused_residual_rms_norm"):
        return ("fused residual + RMSNorm", bf16(HIDDEN),
                bf16(3 * m * HIDDEN), 2 * LAYERS)
    if shape.startswith("gg2_copybfloat16bfloat16 grid=") and grid and grid[1] == 4:
        # [1, 4, positions, 256] bf16 slice update on a full-attention KV cache.
        positions = grid[0] // ATTN_HEAD_DIM
        return (f"full-attention KV cache write, {positions} positions",
                0, 2 * bf16(4 * positions * ATTN_HEAD_DIM), 0)
    return None


# --- census reduction --------------------------------------------------------

def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_legs(snapshots):
    legs, current = [], []
    for snapshot in snapshots:
        if snapshot["snapshot"] == 0 and current:
            legs.append(current)
            current = []
        current.append(snapshot)
    if current:
        legs.append(current)
    return legs


def report_counts(paths, phase="target_verify"):
    """Modal per-round dispatch count of every shape, per verify width."""
    per_width = defaultdict(lambda: defaultdict(list))
    rounds_at = defaultdict(int)
    totals = defaultdict(list)
    for path in paths:
        records = [r for r in load(path)
                   if r.get("event") == "round" and phase in r.get("phases", {})]
        for record in records:
            width = record["width"]
            bucket = record["phases"][phase]
            rounds_at[width] += 1
            totals[width].append(bucket["dispatches"])
            for shape, count in bucket.get("shapes", {}).items():
                per_width[width][shape].append(count)
    print(f"legs: {len(paths)}   phase: {phase}")
    for width in sorted(rounds_at):
        counts = totals[width]
        modal = max(set(counts), key=counts.count)
        share = 100.0 * counts.count(modal) / len(counts)
        print(f"  M={width:2d}  rounds={rounds_at[width]:4d}  "
              f"dispatches modal={modal} ({share:.0f}% of rounds)  "
              f"range={min(counts)}..{max(counts)}")

    widths = sorted(per_width)
    shapes = sorted({s for w in widths for s in per_width[w]})
    print()
    print(f"{'shape':<96}" + "".join(f"{'M=%d' % w:>10}" for w in widths))
    for shape in shapes:
        cells = []
        for width in widths:
            values = per_width[width].get(shape, [])
            if not values:
                cells.append(f"{'-':>10}")
                continue
            modal = max(set(values), key=values.count)
            per_round = modal
            cells.append(f"{per_round:10d}")
        print(f"{shape[:96]:<96}" + "".join(cells))


def collect_signatures(snapshots, phase, width):
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
        counts[shape] += int(multiplicity)
    return counts


def pick_leg(paths, phase, width):
    """The MTP leg of one file, dropping snapshot 0 (weights, warmup, prefill)."""
    for path in paths:
        snapshots = [r for r in load(path) if r.get("event") == "gputime"]
        legs = split_legs(snapshots)
        best = max(legs, key=lambda leg: sum(
            1 for s in leg if f"w{width}|{phase}" in s.get("by_width_phase", {})))
        yield path, best[1:]


def phase_rounds(snapshots, phase, width):
    return sum(s.get("rounds", 0) for s in snapshots
               if f"w{width}|{phase}" in s.get("by_width_phase", {}))


def report_buffers(paths, phase="target_verify", width=9, min_buffers=4):
    for path, steady in pick_leg(paths, phase, width):
        rounds = phase_rounds(steady, phase, width)
        if rounds == 0:
            print(f"=== {path}: no w{width}|{phase} rounds")
            continue
        mixed = sum(s.get("mixed_phase_buffers", 0) for s in steady)
        zero = sum(s.get("zero_time_buffers", 0) for s in steady)
        key = f"w{width}|{phase}"
        phase_ns = sum(s["by_width_phase"][key]["gpu_ns"] for s in steady
                       if key in s["by_width_phase"])
        phase_disp = sum(s["by_width_phase"][key]["dispatches"] for s in steady
                         if key in s["by_width_phase"])
        print(f"=== {path}   M={width}   rounds={rounds}")
        print(f"    phase gpu {phase_ns / 1e3 / rounds:10.1f} us/round   "
              f"{phase_disp / rounds:7.1f} dispatches/round   "
              f"mixed_phase_buffers={mixed}  zero_time_buffers={zero}")
        buckets = collect_signatures(steady, phase, width)
        rows = []
        total_us = total_weight = total_activation = 0.0
        for body, (count, gpu_ns) in buckets.items():
            if count < min_buffers:
                continue
            shapes = parse_signature(body)
            weight = activation = 0.0
            named, unnamed = [], 0
            for shape, multiplicity in shapes.items():
                entry = qmv_bytes(shape, width)
                if entry:
                    name, out, grid_m, passes, wbytes, abytes = entry
                    weight += multiplicity * wbytes
                    activation += multiplicity * abytes
                    named.append((multiplicity * wbytes,
                                  f"{name}[O={out},G={passes}]x{multiplicity}"))
                    continue
                entry = other_bytes(shape, width)
                if entry:
                    name, wbytes, abytes, _per_round = entry
                    weight += multiplicity * wbytes
                    activation += multiplicity * abytes
                    named.append((multiplicity * wbytes, f"{name}x{multiplicity}"))
                    continue
                unnamed += multiplicity
            names = [text for _bytes, text in sorted(named, reverse=True)]
            if unnamed:
                names.append(f"+{unnamed} unmapped elementwise")
            us = gpu_ns / 1e3 / rounds
            per_round = count / rounds
            total_us += us
            total_weight += weight * per_round
            total_activation += activation * per_round
            rows.append((us, count, per_round, weight, activation, ", ".join(names)))
        for us, count, per_round, weight, activation, names in sorted(
                rows, reverse=True):
            moved = (weight + activation) * per_round
            rate = moved / (us * 1e-6) / 1e9 if us > 0 else 0.0
            print(f"    {us:10.2f} us/round  n={count:5d}  {per_round:6.2f} buf/round"
                  f"  {moved / 1e6:9.3f} MB  {rate:7.1f} GB/s  {names[:110]}")
        moved = total_weight + total_activation
        print(f"    TOTAL {total_us:.2f} us/round over {len(rows)} buffer signatures, "
              f"{moved / 1e6:.3f} MB/round "
              f"({total_weight / 1e6:.3f} MB weights + "
              f"{total_activation / 1e6:.3f} MB activations), "
              f"{moved / (total_us * 1e-6) / 1e9:.1f} GB/s")


def report_kernels(paths, phase="target_verify", width=9, min_buffers=4):
    """Per-kernel isolated GPU time. Needs MLX_E58_BUFFER_LIMIT_OPS=1."""
    for path, steady in pick_leg(paths, phase, width):
        rounds = phase_rounds(steady, phase, width)
        if rounds == 0:
            print(f"=== {path}: no w{width}|{phase} rounds")
            continue
        prefix = f"w{width}|{phase}|"
        buckets = defaultdict(lambda: [0, 0, 0])
        for snapshot in steady:
            for key, bucket in snapshot.get("exclusive_kernels", {}).items():
                if not key.startswith(prefix):
                    continue
                shape = key[len(prefix):]
                buckets[shape][0] += bucket["buffers"]
                buckets[shape][1] += bucket["gpu_ns"]
                buckets[shape][2] = max(buckets[shape][2], bucket["max_ns"])
        print(f"=== {path}   M={width}   rounds={rounds}   "
              f"isolated kernels={len(buckets)}")
        print(f"    {'us/round':>10} {'n/round':>8} {'us/disp':>9} {'MB/disp':>9} "
              f"{'GB/s':>7}  kernel")
        rows = []
        for shape, (count, gpu_ns, _max_ns) in buckets.items():
            if count < min_buffers:
                continue
            per_round = count / rounds
            us_round = gpu_ns / 1e3 / rounds
            us_disp = gpu_ns / 1e3 / count
            entry = qmv_bytes(shape, width)
            if entry:
                name, out, _grid_m, passes, weight, activation = entry
                label = f"{name} [O={out}, G={passes}]"
            else:
                entry2 = other_bytes(shape, width)
                if entry2:
                    label, weight, activation, _ = entry2
                else:
                    label, weight, activation = f"UNMAPPED {shape[:60]}", 0, 0
            moved = weight + activation
            rate = moved / (us_disp * 1e-6) / 1e9 if us_disp > 0 and moved else 0.0
            rows.append((us_round, per_round, us_disp, moved, rate, label))
        for us_round, per_round, us_disp, moved, rate, label in sorted(
                rows, reverse=True):
            print(f"    {us_round:10.2f} {per_round:8.2f} {us_disp:9.3f} "
                  f"{moved / 1e6:9.3f} {rate:7.1f}  {label[:88]}")
        print(f"    TOTAL {sum(r[0] for r in rows):.2f} us/round isolated")


def isolated_costs(path, phase, width, min_buffers=4):
    """{shape: (us per dispatch, n)} from a MLX_E58_BUFFER_LIMIT_OPS=1 leg."""
    costs = {}
    for _path, steady in pick_leg([path], phase, width):
        prefix = f"w{width}|{phase}|"
        buckets = defaultdict(lambda: [0, 0])
        for snapshot in steady:
            for key, bucket in snapshot.get("exclusive_kernels", {}).items():
                if key.startswith(prefix):
                    shape = key[len(prefix):]
                    buckets[shape][0] += bucket["buffers"]
                    buckets[shape][1] += bucket["gpu_ns"]
        for shape, (count, gpu_ns) in buckets.items():
            if count >= min_buffers:
                costs[shape] = (gpu_ns / 1e3 / count, count)
    return costs


def modal_counts(path, phase, width):
    """{shape: modal dispatches per round} over the steady rounds at one width."""
    per_shape = defaultdict(list)
    rounds = 0
    for record in load(path):
        if record.get("event") != "round" or record.get("width") != width:
            continue
        bucket = record.get("phases", {}).get(phase)
        if bucket is None:
            continue
        rounds += 1
        for shape, count in bucket.get("shapes", {}).items():
            per_shape[shape].append(count)
    modal = {}
    for shape, values in per_shape.items():
        if len(values) < 0.5 * rounds:
            continue                       # growth-round or rollback-only shape
        modal[shape] = max(set(values), key=values.count)
    return modal, rounds


def report_closure(insitu, iso, phase="target_verify", width=9):
    """Exact per-round counts x isolated per-dispatch cost, per dispatch class.

    Counts come from the in-situ leg and are exact integers. Costs come from the
    one-op-per-command-buffer leg and OVER-state each kernel, because isolation
    removes intra-buffer concurrency. The ratio of the modelled total to the
    measured in-situ phase total is the concurrency discount, and it is reported
    rather than hidden.
    """
    counts, rounds = modal_counts(insitu, phase, width)
    costs = isolated_costs(iso, phase, width)
    measured = 0.0
    for _path, steady in pick_leg([insitu], phase, width):
        key = f"w{width}|{phase}"
        span = phase_rounds(steady, phase, width)
        measured = sum(s["by_width_phase"][key]["gpu_ns"] for s in steady
                       if key in s["by_width_phase"]) / 1e3 / max(span, 1)
    print(f"=== in-situ {insitu}   isolated {iso}   M={width}   "
          f"modal rounds={rounds}")
    print(f"    {'us/round':>10} {'n':>5} {'us/disp':>9} {'MB/disp':>9} "
          f"{'GB/s':>7}  class")
    rows, modelled, priced = [], 0.0, 0
    for shape, count in counts.items():
        cost = costs.get(shape)
        if cost is None:
            rows.append((0.0, count, 0.0, 0.0, 0.0,
                         f"NO ISOLATED COST {shape[:70]}"))
            continue
        us_disp = cost[0]
        entry = qmv_bytes(shape, width)
        if entry:
            name, out, _m, passes, weight, activation = entry
            label = f"{name} [O={out}, G={passes}]"
        else:
            entry2 = other_bytes(shape, width)
            if entry2:
                label, weight, activation, _ = entry2
            else:
                label, weight, activation = f"unmapped {shape[:56]}", 0, 0
        moved = weight + activation
        rate = moved / (us_disp * 1e-6) / 1e9 if us_disp > 0 and moved else 0.0
        us_round = us_disp * count
        modelled += us_round
        priced += 1
        rows.append((us_round, count, us_disp, moved, rate, label))
    for us_round, count, us_disp, moved, rate, label in sorted(rows, reverse=True):
        print(f"    {us_round:10.2f} {count:5d} {us_disp:9.3f} {moved / 1e6:9.3f} "
              f"{rate:7.1f}  {label[:84]}")
    print(f"    modelled isolated total {modelled:.1f} us/round over {priced} "
          f"classes;  measured in-situ phase {measured:.1f} us/round;  "
          f"concurrency discount {100.0 * (1 - measured / max(modelled, 1e-9)):.1f} %")


def report_gdn(paths, phase="target_verify"):
    """Audit every Gated DeltaNet recurrent-state dispatch, per round and width."""
    print(f"GDN recurrent state: {GDN_V_HEADS} heads x {GDN_HEAD_DIM} x "
          f"{GDN_HEAD_DIM} fp32 = {GDN_STATE_BYTES:,} B per layer, "
          f"{GDN_STATE_BYTES * GDN_LAYERS:,} B over {GDN_LAYERS} layers")
    marks = ("custom_kernel_gated_delta_step__",
             "custom_kernel_qwen35_gated_delta_step_mid__",
             "custom_kernel_qwen35_gated_delta_replay_state__",
             "custom_kernel_qwen35_packed_gdn_prework__")
    for path in paths:
        per_width = defaultdict(lambda: defaultdict(list))
        rounds_at = defaultdict(int)
        for record in load(path):
            if record.get("event") != "round" or phase not in record.get("phases", {}):
                continue
            width = record["width"]
            rounds_at[width] += 1
            shapes = record["phases"][phase].get("shapes", {})
            for shape, count in shapes.items():
                if shape.startswith(marks):
                    per_width[width][shape].append(count)
        print(f"\n=== {path}")
        for width in sorted(rounds_at):
            print(f"  M={width}  rounds={rounds_at[width]}")
            for shape in sorted(per_width[width]):
                values = per_width[width][shape]
                present = len(values)
                total = sum(values)
                modal = max(set(values), key=values.count)
                name = other_bytes(shape, width)
                bytes_per = name[2] if name else 0
                print(f"    {total / rounds_at[width]:7.2f} /round  "
                      f"in {100.0 * present / rounds_at[width]:5.1f}% of rounds  "
                      f"modal={modal:3d}  "
                      f"{total / rounds_at[width] * bytes_per / 1e6:9.3f} MB/round  "
                      f"{shape[:76]}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode, paths = sys.argv[1], sys.argv[2:]
    width = 9
    kept = []
    for path in paths:
        if path.startswith("--width="):
            width = int(path.split("=", 1)[1])
        else:
            kept.append(path)
    print(f"modelled weight stream = {WEIGHT_STREAM_BYTES:,} B "
          f"(organiser figure 14,412,349,440 B)")
    if mode == "counts":
        report_counts(kept)
    elif mode == "buffers":
        report_buffers(kept, width=width)
    elif mode == "kernels":
        report_kernels(kept, width=width)
    elif mode == "closure":
        report_closure(kept[0], kept[1], width=width)
    elif mode == "gdn":
        report_gdn(kept)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
