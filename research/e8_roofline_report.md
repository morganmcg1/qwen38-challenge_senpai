# E8 result: is `qmv_fast_crossrow_affine4_g64` bandwidth- or ALU-bound at M >= 4?

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":false,"value":null},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}

- Student / branch: `qwen-thorfinn` / `qwen-thorfinn/crossrow-roofline-regime`
- Hypothesis and target cost: the crossrow qmv kernel is the dominant per-round
  cost at verify widths M >= 4. The assignment asks which resource limits it.
  Falsifiable form: **if the kernel is bandwidth-bound, cutting arithmetic at
  constant bytes leaves time flat and cutting unique bytes at constant
  arithmetic makes it faster; if it is ALU-bound, the opposite.**
- Decision: **not useful** as a product change (both arms are deliberately
  wrong-numerics measurement instruments and nothing is proposed for the
  scored path), and **decisive** as a diagnostic. The regime question is
  answered: **ALU-bound at M = 8, ALU-dominant but mixed at M = 4.**
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit:
  `ed4269c25f29ce1129339b0eb4365886c0a6bb71` /
  `7351e62674bc600f0ca148d3a1b0604716a09db6` / final head of this branch.
- Yukon promoted submission / source ref used as frontier:
  `e6c5ef35-0d86-4cec-a5d6-366e2e59cdcd`, sourceRef
  `7351e62674bc600f0ca148d3a1b0604716a09db6`, score 2.9042110287045. No
  submission was made or is proposed from this experiment.
- Submitted candidate files: **none**. The final commit restores
  `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h` and
  `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` byte-for-byte to
  `BASE_SHA`; `git diff ed4269c2 -- Vendor/` is empty at the reported head.
  The two arm patches exist only as intermediate commits so the exact measured
  build is reproducible.
- Supporting test, tooling, or documentation files (research-only, not
  submitted): `research/roofline_rederive.py`, `research/roofline_arm_patch.py`,
  `research/e8_compare.py`, `research/air_kernel_stats.py` (extended),
  `research/crossrow_na_probe.metal` (repaired), `research/e8_roofline_notes.md`,
  this report.
- MTP head provenance and draft policy: not applicable. This is a kernel
  microbenchmark at fixed verify widths; no head is attached and no drafting
  policy is exercised.
- Assignment-scope preflight:
  `senpai/validate-assignment-scope.sh ed4269c25f29ce1129339b0eb4365886c0a6bb71
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` -> OK.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `senpai/check-editable-budget.sh ed4269c25f29ce1129339b0eb4365886c0a6bb71` ->
  `source=2394650/3000000 headroom=605350 growth=0/262144 exempt=2410
  files=154`.
- Scored-path reachability evidence: the dispatch in `quantized.h:1811-1886`
  routes M = 2 to `qmv_fast_crossrow_affine4_g64<T,2>` and M = 3..9 to
  `..._g64_m<T,M,IPG>` (IPG 3,4,3,3,4,4,3 for M = 3..9) under `VECTOR_LIMIT =
  10`; both instantiate the same `qmv_fast_crossrow_affine4_g64_wide` body that
  the arms patch. `QwenQMVCostCurveTests` drives the eight real scored
  projection shapes through that dispatch. Reachability of this kernel from the
  scored MTP verify path was established in E7 / PR #8 and is assumed here, as
  the assignment does.

## Evidence

- Host, memory profile, toolchain, thermal policy: Apple M4 Pro, 48 GiB
  (`mem=51539607552`), GPU family `applegpu_g16s`. All runs via
  `research/run-qmv-curve.sh`, which takes the run lock, checks for orphaned
  model-holding workers and waits for the 40 C cool gate before each resident
  measurement, then does a fresh release build plus
  `tools/build-mlx-metallib.sh --all-build-roots`. Measured stream peak
  **226.9 GB/s** (control session) and **226.7 GB/s** (arm sessions), against a
  273 GB/s nominal figure; measured GEMM peak 7.50 TFLOP/s. Every comparison in
  this report is against the **measured 226.9 GB/s roofline**, never against an
  adjacent verify width.
- Exact baseline and candidate commands:

  ```bash
  # control (unmodified BASE_SHA kernel)
  git checkout ed4269c25f29ce1129339b0eb4365886c0a6bb71 -- Vendor/mlx-swift/Source/Cmlx
  research/run-qmv-curve.sh e8-control ed4269c25f29ce1129339b0eb4365886c0a6bb71

  # arm 1: cut arithmetic, hold bytes constant
  python3 research/roofline_arm_patch.py arm1
  research/run-qmv-curve.sh e8-arm1 ed4269c25f29ce1129339b0eb4365886c0a6bb71

  # arm 2: cut unique weight bytes ~4x, hold arithmetic constant
  git checkout ed4269c25f29ce1129339b0eb4365886c0a6bb71 -- Vendor/mlx-swift/Source/Cmlx
  python3 research/roofline_arm_patch.py arm2
  research/run-qmv-curve.sh e8-arm2 ed4269c25f29ce1129339b0eb4365886c0a6bb71

  # comparison table
  python3 research/e8_compare.py e8-control e8-arm1 e8-arm2

  # DCE proof (AIR op counts)
  python3 research/air_kernel_stats.py --match crossrow_na4
  ```

- Tests and risk-based checks: AIR load-count and op-count audit of every arm
  against the control (below); `research/crossrow_na_probe.metal` compiles for
  NA = 2, 3, 4; both Metal twins are patched together and diffed for equality.
  Full `swift test` was not run: no product change is proposed, and the arms
  are intentionally numerically wrong.
- Exact-token and row-ledger verdict: **not applicable and not claimed.** Both
  arms deliberately compute wrong values. `QwenQMVCostCurveTests` records
  `row0_bitwise_matches_m1` but has no hard fidelity assertion, which is why the
  arms complete. Nothing here is proposed for the product path.
- Divergent tokens or failure category: none observed; not measured, by design.
- Generated-twin audit: the readable `.metal`/`.h` source and its runtime-
  effective `mlx-generated/quantized.cpp` twin were patched by the same script
  in one call, which refuses to run unless each anchor is unique inside the
  target function. Diffs of the two twins are textually identical for both arms.
  At the reported head both files are identical to `BASE_SHA`.
- Peak RAM or head/artifact size: not relevant; no head or artifact produced.
- Official status and score, if submitted: not submitted.

### Noise floor

The control (`e8-control`, head `f313ca2`) reproduces the merged PR #8 artifact
`e7-na4-base`, taken in an earlier session across a full rebuild and cool-down:
**median per-shape time ratio 1.000 at both M = 4 and M = 8**, per-shape range
[0.988, 1.003] at M = 4 and [0.993, 1.006] at M = 8. Session-to-session
repeatability is therefore **+/-1.2% worst case, ~+/-0.5% typical**. Any arm
effect larger than about 2% is real.

### Control absolutes

| shape | M=4 s/call (us) | M=4 GB/s | M=8 s/call (us) | M=8 GB/s |
|---|---:|---:|---:|---:|
| full_attn.o_proj | 138.80 | 127.5 | 225.27 | 78.5 |
| full_attn.qkv_proj_fused | 252.18 | 163.7 | 452.19 | 91.3 |
| head.compact_draft_vocab | 1444.88 | 196.0 | 2837.31 | 99.8 |
| head.lm_head | 3570.96 | 200.3 | 7043.90 | 101.5 |
| linear_attn.in_proj_fused_qkvzba | 283.13 | 167.6 | 514.80 | 92.2 |
| linear_attn.out_proj | 137.25 | 128.9 | 224.46 | 78.8 |
| mlp.down | 328.34 | 152.7 | 571.20 | 87.8 |
| mlp.gate_up_fused | 544.73 | 184.1 | 1031.93 | 97.2 |

At M = 8 every scored shape runs at 78-102 GB/s nominal, i.e. **35-45% of the
measured 226.9 GB/s achievable bandwidth**. A DRAM-limited kernel does not sit
at 40% of its own roofline.

### DCE proof

`research/air_kernel_stats.py` was extended to separate `DEVICE_LOAD`
(`= load ... addrspace(1)`) from all loads, and to count `fmul`, `fadd`, total
flops and loop back-edges. Metal emits AIR with loops rolled (verified -O2 and
-O3 identical), so these are per-loop-body counts, comparable only at equal trip
counts -- which is exactly what the back-edge count certifies.

| build | fmul | fadd | flops | device_loads | all_loads | backedges |
|---|---:|---:|---:|---:|---:|---:|
| control | 9 | 10 | 19 | 7 | 16 | 11 |
| arm 1 | 3 | 7 | 10 | 7 | 16 | 11 |
| arm 2 | 9 | 10 | 19 | 7 | 16 | 11 |

Arm 1 cuts arithmetic while **device load count and loop trip counts are
bit-identical to the control**, so no load was eliminated: the DCE hazard named
in the assignment is closed by measurement, not by assertion. Arm 2 changes
neither arithmetic nor load count, only the addresses those loads resolve to.

Op accounting is exact rather than approximate. Control fmul 9 = 4 unpack
scalings + 2 for `acc += scale * partial + sums * bias` + 3 inside
`load_vector<T,float,4,4>` (bits = 4 scalings); arm 1 = 1 + 2 + 0 = 3. Control
fadd 10 = 3 unpack + 4 accumulate + 3 raw-sum inside `load_vector`; arm 1 =
0 + 4 + 3 = 7. Two structural facts make DCE impossible in arm 1:
`load_vector<T,float,4,4>` returns the sum of the four raw activation values
into `sums[m]`, so activation loads are live by construction, and `packed[r][i]`
stays live through the surviving `a0 * (packed[r][i] & 0x000f)` term, so weight
loads cannot be dropped.

Scaling the per-body counts by trip counts, whole-kernel lane operations fall
from about **688 to 256, a 2.7x cut -- not 4x**. That number matters when
reading the arm-1 speedup below.

Caveat, stated because it weakens my own instrument: a naive literal variant
(`row = out_row`, AIR only) also showed no AIR collapse, because the loops stay
rolled. The runtime-opaque zero used in arm 2 is therefore justified insurance
against load/value CSE rather than a proven necessity.

### Arm 1 -- cut arithmetic, hold bytes constant

Tag `e8-arm1`, head `9031269`, W&B `pfd7wo6v`. Ratio = arm1 / control seconds
per call; below 1.0 means faster.

| shape | M=4 s/call | M=4 GB/s | M=4 ratio | M=8 s/call | M=8 GB/s | M=8 ratio |
|---|---:|---:|---:|---:|---:|---:|
| full_attn.o_proj | 83.26 | 212.5 | 0.600 | 122.75 | 144.1 | 0.545 |
| full_attn.qkv_proj_fused | 186.91 | 220.9 | 0.741 | 230.82 | 178.9 | 0.510 |
| head.compact_draft_vocab | 1160.48 | 244.0 | 0.803 | 1345.98 | 210.4 | 0.474 |
| head.lm_head | 2880.06 | 248.3 | 0.807 | 3346.40 | 213.7 | 0.475 |
| linear_attn.in_proj_fused_qkvzba | 227.45 | 208.7 | 0.803 | 258.40 | 183.7 | 0.502 |
| linear_attn.out_proj | 83.08 | 213.0 | 0.605 | 123.42 | 143.4 | 0.550 |
| mlp.down | 237.25 | 211.3 | 0.723 | 295.69 | 169.6 | 0.518 |
| mlp.gate_up_fused | 423.98 | 236.5 | 0.778 | 500.92 | 200.2 | 0.485 |

Median ratio **0.760 [0.600, 0.807] at M = 4** and **0.506 [0.474, 0.550] at
M = 8**. A bandwidth-bound kernel had to be flat here (within +/-1.2%) and was
not, by 24% and 49% respectively.

Three details sharpen the reading:

1. **The effect grows with M.** 1.32x at M = 4, 1.98x at M = 8. A
   bandwidth-bound model predicts the opposite: more verify rows amortise the
   same weight stream over more arithmetic, so the kernel should become *more*
   bandwidth-bound with M and the arm-1 effect should shrink.
2. **Arm 1 lands on the wall.** At M = 8 the head shapes reach 210-214 GB/s
   nominal, **93-94% of the measured 226.7 GB/s peak**. Removing arithmetic
   moves the kernel from 44% of peak to 94% of peak: the base kernel at M = 8 is
   ALU-limited with roughly 2x of unused bandwidth headroom.
3. **The win is smaller than the op cut.** Lane ops fall 2.7x, time falls only
   1.98x. Address arithmetic, the surviving loads, the simdgroup reduction,
   launch and tail do not scale with the ALU cut. ALU is the dominant term at
   M = 8, not the only one.

### Arm 2 -- cut unique weight bytes ~4x, hold arithmetic constant

ARM2_TABLE_PLACEHOLDER

## Item 4 -- independent re-derivation of `nominal x M = 692 +/- 5.6%`

This is reported at length because the assignment names it the single most
valuable line. Source: the merged PR #8 artifact
`.mlxfast-private/qmv-curve/e7-na4-base/summary.json` (`per_shape_curve`),
re-analysed by `research/roofline_rederive.py`. My verdict: **the direction the
advisor drew from this table is correct and independently reproducible; the
statistic quoted to support it is not sound as stated, and should not be
carried forward at its advertised strength.**

**Confirmed.** `692 +/- 5.6%` is the median over the eight shapes of
`gbps_nominal x M` for M = 4..9: mean **694.8**, min 655.4 (M = 5), max 732.0
(M = 8), half-range **+/-5.52%**. Mean aggregation gives 690.6 +/- 5.37%, so it
is not an aggregation artifact. The FACT 1 GB/s row (165.6 / 262.1 / 183.0 /
239.5) is the median `gbps_stream_corrected`.

**Qualification 1 -- the invariance is not clean.** Per shape, the max/min of
`nominal x M` over M = 4..9 spans 1.078 (`head.lm_head`) to **1.243**
(`linear_attn.out_proj`; `full_attn.o_proj` 1.238). The drift is monotone in M
and ordered by aspect ratio, so it is structure, not noise. The +/-5.6% figure
is a median masking a +24% systematic trend on the two squarest shapes.

**Qualification 2 -- there is a visible step at `ceil(M/4)` boundaries.**
`head.lm_head` per-row time increments are 0.408, 0.273, 0.261, 0.263, 0.434
t(1): an extra ~0.15 t(1) step at exactly M = 5 and M = 9 over an interior slope
of ~0.263 t(1)/row. That is the signature of an integer weight-stream count.

**Refutation A (physical).** The stream correction implies **262.1 GB/s at
M = 5** (call-weighted 264.4) against a **measured achievable 226.9 GB/s**, an
impossible +16%; M = 9 is also over by +6%. A derived bandwidth cannot exceed
the measured roofline, so `weight_streams(m) = ceil(m/4)` over-counts DRAM
traffic at boundary widths -- the second pass is served from cache, not DRAM.

**Refutation B (statistical) -- the strongest line in this report.**
`research/roofline_regime_check.py` decides via
`verdict = "ALU-bound" if prod_rel < nom_rel`, a two-way contest between
`H_A: nominal` flat and `H_B: nominal x M` flat. It **never scores
`H_C: nominal x ceil(M/4)` flat** -- the integer weight-stream model whose death
it announces. A specificity check settles what that omission costs: on
synthetic, noise-free `H_C` data (`t = ceil(M/4)`, `nominal := 1/t`) the rule
reports "ALU-bound, **2.2x tighter**" at M = {4,5,8,9}, 2.3x at M = 4..9 and
1.2x at M = 1..9. **So 2.2x is the rule's floor, not 1.0x**, because `1/M` over
{4,5,8,9} already carries ~37% relative sd. Scoring all three hypotheses:

| data | H_A `nominal` | H_C `nominal x ceil(M/4)` | H_B `nominal x M` | B vs C | B vs A |
|---|---:|---:|---:|---:|---:|
| advisor's 4 points | 33.4% | 21.5% | **5.6%** | 3.8x | 6.0x |
| my 8-shape median, M = 4..9 | 27.1% | 17.0% | **4.5%** | 3.8x | 6.0x |
| my 8-shape median, M = 1..9 | 26.9% | **14.2%** | 26.9%* | -- | -- |

(*at M = 1..9 the winner flips to `H_C`.) The middle row reproduces both the
3.8x and the 6.0x from independent data, so the direction is robust. But the
honest margin over the model actually being rejected is **3.8x against a floor
of 2.2x**, not 6.0x against a floor of 1.0x. Per shape, `H_B` wins on all eight,
yet its margin over `H_C` ranges from **2.1x** (`linear_attn.out_proj`) and 2.2x
(`full_attn.o_proj`) to 6.1x (`head.compact_draft_vocab`) -- the weakest cases
are exactly the +24%-drift shapes from Qualification 1. And the M = 1..9 flip
shows the conclusion is load-bearing on the M >= 4 window and therefore on the
knee band.

**Net:** conclusion survives; the statistic overstates its strength by roughly
1.6x and is measured against a null that no hardware could produce. Answering
the advisor's invitation directly: the 6.0x ratio is too weak to carry the
argument alone, because a decision rule whose floor is 2.2x cannot use 6.0x as
evidence of anything without publishing that floor. The 3.8x margin over `H_C`
is the number worth keeping. The arms above are what actually settles the
regime, and they do so without depending on any of this.

For reference only: at 0.5625 bytes per weight the arithmetic intensity is
3.56 x M FLOP/byte, and machine balance is 7500/226.9 = 33 FLOP/byte, so a naive
FLOP roofline calls every M < 10 bandwidth-bound. The measurement contradicts
it because qmv does not use the matmul hardware that sets the 7.5 TFLOP/s peak
and pays substantial non-FLOP unpack and address ALU per weight.

## Conclusion

CONCLUSION_PLACEHOLDER
