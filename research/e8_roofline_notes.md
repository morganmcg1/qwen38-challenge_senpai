# E8 crossrow roofline regime — working notes

Base `ed4269c2`, host Apple M4 Pro 48 GiB (`applegpu_g16s`).
Machine roofline measured in-run: stream peak **226.90 GB/s**, GEMM peak **7.50 TFLOP/s**.

## 1. Re-derivation of the advisor's `nominal x M` table

Source: merged PR #8 artifact `.mlxfast-private/qmv-curve/e7-na4-base/summary.json`,
field `per_shape_curve` (8 scored shapes x widths 1..512).
Reproduced with `research/roofline_rederive.py`.

### 1a. The aggregate reproduces exactly — CONFIRMED

Median over the 8 shapes of `gbps_nominal * M`:

| M | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| median `nominal*M` | 662.3 | 655.4 | 689.2 | 711.2 | 732.0 | 718.6 |
| mean `nominal*M`   | 659.6 | 652.4 | 684.6 | 707.7 | 726.6 | 712.9 |

mean 694.8, min 655.4 (M=5), max 732.0 (M=8) => **half-range +/-5.52%**.
Mean-aggregation gives 690.6 +/-5.37%. Either rule lands on the advisor's
`692 +/- 5.6%`. The number is not an artifact of the aggregation rule.

I also identified the FACT 1 GB/s row (165.6 / 262.1 / 183.0 / 239.5) as the
**median of `gbps_stream_corrected`**, not of `gbps_nominal`:
median stream-corrected = 165.57 (M=4), 262.14 (M=5), 183.00 (M=8), 239.53 (M=9).

### 1b. Two qualifications the aggregate hides

**(i) Per-shape drift is ~4x the quoted band.** `nominal*M` max/min over M=4..9,
per shape:

| shape | max/min |
|---|---|
| head.lm_head | 1.078 |
| head.compact_draft_vocab | 1.082 |
| mlp.gate_up_fused | 1.098 |
| full_attn.qkv_proj_fused | 1.118 |
| linear_attn.in_proj_fused_qkvzba | 1.116 |
| mlp.down | 1.145 |
| full_attn.o_proj | 1.238 |
| linear_attn.out_proj | 1.243 |

So the residual reaches **+24%** on the two K>=6144, N=5120 shapes, versus the
+/-5.6% suggested by the aggregate. The drift is monotone in M for every shape
(not noise) and is ordered by aspect ratio, so "`nominal*M` is invariant" is a
good first-order law with a real, structured residual — not a clean ALU verdict.

**(ii) The residual is staircase + slope, not either alone.** Per-row time
increments for `head.lm_head`, in units of t(M=1):
0.408 (M=5), 0.273, 0.261, 0.263, 0.434 (M=9).
There is an extra ~0.15 t(1) step at exactly M=5 and M=9 — the `ceil(M/4)`
stream boundaries — superimposed on an interior slope of ~0.263 t(1)/row.
Neither a pure ALU-linear law nor a pure integer-stream staircase explains both.

### 1c. Candidate REFUTATION: the stream correction over-counts DRAM traffic

`gbps_stream_corrected` multiplies nominal weight bytes by `ceil(M/4)`, i.e. it
assumes every one of the `ceil(M/4)` passes re-reads the weights from DRAM.
At M=5 that implies **262.1 GB/s** (call-weighted 264.4) of sustained traffic.
The measured achievable copy bandwidth on this host is **226.90 GB/s**.
262.1 / 226.9 = **1.16** — 16% above the machine's own measured ceiling.
M=9 is also over (239.5 vs 226.9, +6%).

A metric cannot exceed the measured roofline. So the `ceil(M/4)`-streams model
is wrong at the boundary widths: a substantial part of the second pass is
served from cache, not DRAM. This matters because the "bandwidth-bound at
67-96% of peak" reading of FACT 1 rests entirely on that correction.

Corollary: the true operating point sits between the two idealisations, which
is exactly why the direct arms are needed rather than more curve fitting.

### 1d. Classical FLOP roofline (for reference only)

4-bit g64 => 0.5625 B/weight; 2 FLOP/weight/row => arithmetic intensity
= 3.56*M FLOP/byte. Machine balance 7500/226.9 = 33 FLOP/byte. Naive roofline
therefore says bandwidth-bound for all M < 10. But `qmv_fast` does not use the
matmul pipeline and pays 4-bit unpack ALU that the roofline does not count, so
this is a lower bound on ALU pressure, not a verdict.

## 2. Arm design and DCE evidence

`research/roofline_arm_patch.py` edits the `_wide` body in BOTH twins
(`quantized.h` and `mlx-generated/quantized.cpp`), refusing to run unless each
anchor is unique inside that function (`const int row = out_row + r;` occurs
twice per file: once in the NA<=2 kernel, once in `_wide`).

- `arm1` — arithmetic /~2.7, bytes constant: keep only the `& 0x000f` term of
  the 4-term unpack accumulate.
- `arm2` — unique weight bytes /4, arithmetic constant: `row = out_row + r*row_span`
  with `row_span = in_vec_size >> 30` (a runtime-opaque zero, so addresses stay
  formally distinct and no load/value CSE can fire, but all 4 rows resolve to
  one tile at runtime).
- `arm2-naive` — literal `row = out_row`, AIR inspection only.

AIR counts (`research/air_kernel_stats.py --match crossrow_na4`):

| build | fmul | fadd | flops | device_loads | loads | loop_backedges |
|---|---|---|---|---|---|---|
| control | 9 | 10 | 19 | 7 | 16 | 11 |
| arm1 | 3 | 7 | 10 | 7 | 16 | 11 |
| arm2 | 9 | 10 | 19 | 7 | 16 | 11 |
| arm2-naive | 9 | 10 | 19 | 7 | 16 | 11 |

**Arm 1 DCE check passes:** arithmetic drops 19 -> 10 while device loads (7),
total loads (16) and loop back-edges (11) are all unchanged. Op accounting is
exact: control fmul 9 = 4 unpack terms + 2 (`acc += scale*partial + sums*bias`)
+ 3 (`load_vector` bits=4 scalings /16, /256, /4096); arm1 = 1 + 2 + 0 = 3.
Control fadd 10 = 3 (term sum) + 4 (sums/acc) + 3 (`load_vector` raw sum);
arm1 = 0 + 4 + 3 = 7. `load_vector<T,float,4,4>` returns the sum of the 4 raw
activation values and that return feeds `sums[m]`, so every activation load is
live by construction; `packed[r][i]` stays live through the surviving
`& 0x000f` term, so no weight load can be eliminated.

Estimated per-thread lane-op cut at NA=4: ~688 -> ~256 ops = **2.7x, not 4x**.
Report the measured effect against 2.7x, not 4x.

**Arm 2 holds arithmetic exactly constant** (identical fmul/fadd/loads/back-edges).

Honest caveat: `arm2-naive` did *not* show an arithmetic collapse at AIR level
either, because Metal emits AIR with loops still rolled (verified: -O2 and -O3
give byte-identical rolled output). So AIR cannot resolve whether the backend
would have CSE'd the naive form; the opaque-zero defence is justified insurance,
not a proven necessity.

## 3. Noise budget

Reusable from PR #8 (same host, same harness):
- drift over unchanged widths: 0.999
- stock pip-MLX control median 1.0000 (range 0.954-1.019)
- two independent NA=4 sessions agree within 0.4%

This experiment adds a direct session-to-session repeatability number by
re-running the unchanged base as `e8-control` and comparing with `e7-na4-base`.
