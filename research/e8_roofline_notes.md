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

### 1e. REFUTATION of the "6.0x tighter" statistic (not of the conclusion)

The advisor's `research/roofline_regime_check.py` decides with

```python
verdict = "ALU-bound" if prod_rel < nom_rel else "bandwidth-bound"
```

i.e. a two-way contest between
* `H_A`: `nominal` flat  <=> t independent of M (perfect cross-row reuse), and
* `H_B`: `nominal*M` flat <=> t ~ M (zero cross-row reuse).

**The memory-side model is not on the ballot.** The hypothesis whose death the
script announces ("the memory-side lever family is dead") is
`H_C`: `nominal*ceil(M/4)` flat <=> t ~ ceil(M/4) — the integer weight-stream
model that the harness's own `gbps_stream_corrected` encodes. `H_C` is never
scored, so the test cannot reject it.

**Consequence 1 — the rule has no specificity.** Generate data from `H_C`
exactly, with zero noise (`t = ceil(M/4)`, `nominal := 1/t`) and run the
advisor's rule on it:

| window | H_A | H_C | H_B | advisor rule reports |
|---|---:|---:|---:|---|
| M=4,5,8,9 | 49.5% | **0.0%** | 22.2% | "ALU-bound, 2.2x tighter" |
| M=4..9 | 41.0% | **0.0%** | 18.2% | "ALU-bound, 2.3x tighter" |
| M=1..9 | 40.6% | **0.0%** | 33.3% | "ALU-bound, 1.2x tighter" |

A *perfectly bandwidth-bound* kernel makes the rule announce "ALU-bound" with a
2.2x margin. So 2.2x is the rule's **floor**, not 1.0x. The reason is trivial:
over M in {4,5,8,9}, `1/M` has ~37% relative sd by construction, so `H_A` is
guaranteed to lose against anything. The 6.0x must be read against that floor.

**Consequence 2 — restoring `H_C` shrinks the margin from 6.0x to 3.8x.**

| dataset | n | H_A | H_C | H_B | winner | B vs C | B vs A |
|---|---:|---:|---:|---:|---|---:|---:|
| advisor 4 points (FACT 1) | 4 | 33.4% | 21.5% | **5.6%** | H_B | 3.8x | 6.0x |
| my 8-shape median, M=4..9 | 6 | 27.1% | 17.0% | **4.5%** | H_B | 3.8x | 6.0x |
| my 8-shape median, M=1..9 | 9 | 39.8% | **14.2%** | 26.9% | **H_C** | 0.5x | 1.5x |

Two things follow. (a) My independent 6-point, 8-shape median reproduces the
advisor's ratio structure exactly (6.0x vs `H_A`, 3.8x vs `H_C`) from different
data, so the **direction is confirmed and is not an artifact of the 4-point
window**. (b) The honest margin over the model actually being declared dead is
**3.8x against a rule floor of 2.2x** — much thinner than "6.0x tighter"
suggests.

**Consequence 3 — the M>=4 restriction is load-bearing.** Over the full M=1..9
the winner *flips* to `H_C` (14.2% vs 26.9%). That is expected below the knee
and is not itself a refutation, since the brief does restrict to M>=4. But it
means the conclusion rests on the knee band `[2.99, 3.27]` being right: the
verdict is a statement about a 6-point window whose left edge is estimated, not
measured.

Per-shape, `H_B` wins on all 8 shapes, but the margin over `H_C` ranges from
**2.1x** (`linear_attn.out_proj`) and 2.2x (`full_attn.o_proj`) up to 6.1x
(`head.compact_draft_vocab`). The two shapes with the +24% drift from 1b are
also the two weakest ALU cases — the same structure showing up twice.

**Net:** the advisor's conclusion survives my attempt to break it, but the
statistic offered as "the actual strength of the case" overstates it by ~1.6x
and is measured against a null that cannot be true. This is a reason the arms
are necessary, which is the advisor's own position; I am disputing the strength
of the prior, not the plan.

### 1f. Classical FLOP roofline (for reference only)

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
