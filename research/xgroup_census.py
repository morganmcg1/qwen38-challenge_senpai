#!/usr/bin/env python3
"""Working-x-block / concurrent-weight-stream census for the crossrow QMV gate.

Every number here is arithmetic over two source facts, both verified:

  backend/metal/quantized.cpp:251-254
      MTL::Size group_dims(32, 2, 1);
      MTL::Size grid_dims(M, (N + bn - 1) / bn, B);      bn = 8
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    => grid_dims is in THREADGROUPS, and grid.x == M.

  kernels/quantized.h:1171-1172   (and :879-880 for the pair kernel)
      const int first_m = int(tid.x) * IPG;
      if (first_m >= M) { return; }
    => working x-blocks = ceil(M / IPG); the other M - ceil(M/IPG) threadgroups
       exit immediately.

So a launch's real parallelism is  ceil(M/IPG) * ceil(n/8)  threadgroups, and the
comment at quantized.h:1919 ("below 4096 outputs the reduced x-group count thins
the grid") is that quantity, named.

Usage:  python3 research/xgroup_census.py            # census + arm comparison
        python3 research/xgroup_census.py --self-test
"""
import argparse
import math

CORES = 20  # M4 Pro, applegpu_g16s; ranked box is m5-max (more cores)
PEAK_GBS = 273.0  # M4 Pro unified memory bandwidth

# (name, n, k, calls_per_verify) -- from E33 section 8.2, measured.
SHAPES = [
    ("mlp.gate_up_fused", 34816, 5120, 64),
    ("mlp.down", 5120, 17408, 64),
    ("linear_attn.in_proj_fused", 16480, 5120, 48),
    ("linear_attn.out_proj", 5120, 6144, 48),
    ("full_attn.qkv_proj_fused", 14336, 5120, 16),
    ("full_attn.o_proj", 5120, 6144, 16),
    ("head.lm_head", 248320, 5120, 1),
    ("head.compact_draft_vocab", 98336, 5120, 0),
]

# Shipped >=4096 tier, read off the switch in quantized.h: M -> IPG.
SHIPPED_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}

# E33's measured per-shape ratio at M=6 (candidate/base), section 8.2.
E33_RATIO_M6 = {
    "mlp.down": 1.0592,
    "linear_attn.out_proj": 1.0492,
    "mlp.gate_up_fused": 0.9941,
    "full_attn.o_proj": 1.0414,
    "linear_attn.in_proj_fused": 0.9947,
    "head.lm_head": 0.9830,
    "full_attn.qkv_proj_fused": 1.0148,
    "head.compact_draft_vocab": 0.9868,
}


def weight_bytes_one_pass(n, k):
    """4-bit weights + fp16 scale and bias per group of 64."""
    return n * k * 0.5 + (n * k / 64.0) * 2 * 2


def act_bytes(n, k, m, row_blocks):
    """Exact activation traffic, fp16.

    Each x-group g serves min(IPG, M - g*IPG) inputs and those sum to exactly M
    over all groups, so the tail group is accounted for automatically. Every one
    of the ceil(n/8) row-groups reads the inputs it needs over the full k, and it
    does so once per row block -- whether the row blocks are a sequential loop in
    one threadgroup or spread over distinct x-blocks. Hence the total is
    independent of that choice, which is what makes E33 vs E38 a clean
    parallelism-only contrast.
    """
    return math.ceil(n / 8) * m * k * 2 * row_blocks


def arm(n, k, m, ipg, row_blocks, blocks_in_x):
    """working x-blocks, threadgroups, weight passes, traffic for one arm.

    row_blocks   = 4 / ROWS_PER_SIMD  (1 for the shipped r=4 cells)
    blocks_in_x  = True  -> row blocks occupy distinct x-blocks (E38 mapping)
                   False -> row blocks are a sequential loop (E33 mapping)
    """
    groups = math.ceil(m / ipg)
    working_x = groups * row_blocks if blocks_in_x else groups
    tgs = working_x * math.ceil(n / 8)
    # Each x-group covers all n rows once, so weight passes == number of input
    # groups; splitting rows across x does not re-read a weight byte.
    passes = groups
    per_tg_row_blocks = 1 if blocks_in_x else row_blocks
    return {
        "working_x": working_x,
        "tgs": tgs,
        "passes": passes,
        "weight_mb": passes * weight_bytes_one_pass(n, k) / 1e6,
        "act_mb": act_bytes(n, k, m, row_blocks) / 1e6,
        "streams": tgs,
        "seq_per_tg": per_tg_row_blocks,
    }


def census():
    print("=== working x-blocks by shipped cell (independent of shape) ===")
    print(f"{'M':>2} {'IPG':>4} {'working':>8} {'idle':>5}  cell")
    cells = {2: "<T,2>", 3: "_m<T,3,3>", 4: "_m<T,4,4>", 5: "_m<T,5,5>",
             6: "_m<T,6,3>", 7: "_m<T,7,4>", 8: "_m<T,8,4>", 9: "_m<T,9,5>"}
    for m in sorted(SHIPPED_IPG):
        ipg = SHIPPED_IPG[m]
        w = math.ceil(m / ipg)
        print(f"{m:>2} {ipg:>4} {w:>8} {m - w:>5}  {cells[m]}")

    print("\n=== M=6: three arms, per shape ===")
    print("shipped  = _m<T,6,3>          2 x-blocks, 2 weight passes, no loop")
    print("E33      = _m<T,6,6,true,2>   1 x-block,  1 weight pass, 2 SEQUENTIAL row blocks")
    print("E38      = same, row blocks mapped to distinct x-blocks (2 x-blocks, 1 pass)")
    hdr = (f"{'shape':<26}{'n':>7}{'k':>7} | {'TGs':>6}{'W MB':>8}{'A MB':>8} |"
           f" {'TGs':>6}{'W MB':>8}{'A MB':>8} | {'TGs':>6}{'W MB':>8}{'A MB':>8} |"
           f" {'E33 obs':>8}")
    print(hdr)
    for name, n, k, calls in SHAPES:
        s = arm(n, k, 6, 3, 1, False)
        a33 = arm(n, k, 6, 6, 2, False)
        a38 = arm(n, k, 6, 6, 2, True)
        print(f"{name:<26}{n:>7}{k:>7} | "
              f"{s['tgs']:>6}{s['weight_mb']:>8.1f}{s['act_mb']:>8.1f} | "
              f"{a33['tgs']:>6}{a33['weight_mb']:>8.1f}{a33['act_mb']:>8.1f} | "
              f"{a38['tgs']:>6}{a38['weight_mb']:>8.1f}{a38['act_mb']:>8.1f} | "
              f"{E33_RATIO_M6[name]:>8.4f}")

    print("\n  KEY: E33 and E38 have IDENTICAL weight and activation traffic.")
    print("  They differ only in threadgroup count (E38 = 2x E33 = shipped count)")
    print("  and in whether each threadgroup runs one or two sequential row blocks.")
    print("  So an E38-vs-E33 difference isolates parallelism from traffic.")

    print("\n=== does threadgroup count predict E33's sign flip better than traffic? ===")
    print(f"{'shape':<26}{'E33 TGs':>9}{'TG/core':>9}{'traffic ratio':>15}{'obs ratio':>11}")
    for name, n, k, calls in SHAPES:
        s = arm(n, k, 6, 3, 1, False)
        a33 = arm(n, k, 6, 6, 2, False)
        tr = (a33["weight_mb"] + a33["act_mb"]) / (s["weight_mb"] + s["act_mb"])
        print(f"{name:<26}{a33['tgs']:>9}{a33['tgs'] / CORES:>9.1f}"
              f"{tr:>15.4f}{E33_RATIO_M6[name]:>11.4f}")
    print("\n  The traffic ratio is ~1.36 for EVERY shape (both terms scale with n),")
    print("  so traffic cannot explain a sign flip. Threadgroup count varies 640..31040")
    print("  and orders the observations correctly. That is the case for E38.")

    print("\n=== is the shipped M=6 mlp.down bandwidth-bound? ===")
    n, k = 5120, 17408
    s = arm(n, k, 6, 3, 1, False)
    per_call_ms = 30.4096 / 64
    xt = 6 * k * 2 / 1e6
    print(f"  base per-call            {per_call_ms:.4f} ms   (30.4096 ms / 64 calls)")
    print(f"  weight traffic           {s['weight_mb']:.1f} MB  (2 passes)")
    print(f"  activation tile          {xt:.3f} MB  -- shared by all "
          f"{math.ceil(n / 8)} row-groups, so cache-served")
    rate = s["weight_mb"] / per_call_ms  # MB/ms == GB/s
    both = (s["weight_mb"] + s["act_mb"]) / per_call_ms
    print(f"  implied DRAM rate        {rate:.0f} GB/s"
          f"  = {rate / PEAK_GBS * 100:.0f}% of {PEAK_GBS:.0f} GB/s peak")
    print(f"  if activations were DRAM too: {both:.0f} GB/s = "
          f"{both / PEAK_GBS * 100:.0f}% of peak, i.e. IMPOSSIBLE")
    print("  => the activation tile really is cache-served, and the shipped cell")
    print("     runs at ~77% of peak on weights alone.")
    print("  => weight-bandwidth-bound when the grid is wide enough to keep")
    print("     enough concurrent streams in flight. Halving the streams (E33)")
    print("     drops it out of that regime; E38 keeps the shipped stream count")
    print("     while still halving the bytes.")


def self_test():
    fails = []
    # Shipped M=6 must be 2 working x-blocks; E33 arm must be 1; E38 arm 2.
    if arm(5120, 17408, 6, 3, 1, False)["working_x"] != 2:
        fails.append("shipped M=6 working_x != 2")
    if arm(5120, 17408, 6, 6, 2, False)["working_x"] != 1:
        fails.append("E33 M=6 working_x != 1")
    if arm(5120, 17408, 6, 6, 2, True)["working_x"] != 2:
        fails.append("E38 M=6 working_x != 2")
    # E33 and E38 must agree on every traffic term.
    a33 = arm(5120, 17408, 6, 6, 2, False)
    a38 = arm(5120, 17408, 6, 6, 2, True)
    if abs(a33["weight_mb"] - a38["weight_mb"]) > 1e-9:
        fails.append("E33/E38 weight traffic differs")
    if abs(a33["act_mb"] - a38["act_mb"]) > 1e-9:
        fails.append("E33/E38 activation traffic differs")
    if a38["tgs"] != 2 * a33["tgs"]:
        fails.append("E38 threadgroups != 2x E33")
    # Coverage: rows written must equal n for both mappings.
    for row_blocks, in_x in ((1, False), (2, False), (2, True), (4, True)):
        r = 4 // row_blocks
        n = 17408
        tg_y = math.ceil(n / 8)
        rows = tg_y * 2 * row_blocks * r  # 2 simdgroups, row_blocks blocks of r
        if rows != n:
            fails.append(f"coverage {row_blocks=} {in_x=}: {rows} != {n}")
    # thorfinn's 50.1 MB for mlp.down, one pass.
    wb = weight_bytes_one_pass(5120, 17408) / 1e6
    if not 49.5 < wb < 50.7:
        fails.append(f"mlp.down one-pass weight bytes {wb:.1f} != ~50.1 MB")
    # E27's M=5 transition: 2 passes -> 1, activation traffic UNCHANGED.
    pre = arm(5120, 17408, 5, 3, 1, False)
    post = arm(5120, 17408, 5, 5, 1, False)
    if abs(pre["act_mb"] - post["act_mb"]) > 1e-9:
        fails.append("E27 M=5 activation traffic should be unchanged")
    if post["passes"] != 1 or pre["passes"] != 2:
        fails.append("E27 M=5 pass count wrong")
    for f in fails:
        print("FAIL:", f)
    print(f"self-test: {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    raise SystemExit(self_test() if args.self_test else (census() or 0))
