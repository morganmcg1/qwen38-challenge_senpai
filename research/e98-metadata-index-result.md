# E98 result — transform-owned uint16 (scale,bias) index

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"indexed_metadata_saving_at_m5_lm_head_pct","available":true,"value":-0.3985},"test_metric":{"name":"indexed_dequant_bit_identical","available":true,"value":1}}

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e98-transform-metadata-index`
- Hypothesis and target cost: an affine-4 g64 group reads 36 B per 64 elements
  (32 B nibbles, 2 B bf16 scale, 2 B bf16 bias). A lossless uint16
  `(scale, bias)` pair index plus a per-tensor look-up table makes it 34 B, a
  5.56 % cut of every quantized weight byte the target streams. The advisor
  priced that at a 4.92 % local round improvement, using the E96 streaming
  share of 0.886 and a byte-to-time conversion near 1.0.
- Decision: **dead**. The mechanism is lossless and correct, and it is slower
  than the shipped encoding at every scored decode width.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit:
  `4d937ce35854f75db70eabf00f152daf1bca0ad2` /
  organizer contract as recorded at that base / probe commit `c35ef07c`.
  Budget contract base `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.
- Yukon promoted submission / source ref used as frontier: campaign crown
  `8819b108` 3.32794961 (audreyt); our best published `cb8aeefb` 3.32345770.
  No new submission is proposed by this result.
- Candidate build fingerprint: none. No Swift target was rebuilt and no worker
  was produced, because no submitted path changed.
- Submitted-surface / generated-twin / metallib digests: unchanged. The diff
  touches zero paths in `benchmark.json .editablePaths[]`.
- Submitted candidate files: **none**.
- Supporting test, tooling, or documentation files:
  `Tests/MLXFastTests/E98MetadataByteProbeTests.swift` (opt-in behind
  `MLXFAST_RUN_E98_BYTES=1`), `research/e98_bytes_probe.sh`,
  `research/e98_bytes_analysis.py`, `research/e98_census_analysis.py`,
  `research/e98_variant_sources.py`, `research/e98_qmv_ab.m`,
  `research/e98_lut_probe.sh`, `research/e98_arm_regs.py`,
  `research/e98_lut_analysis.py`, `research/e98_sentinel_census.py`,
  `research/e98_wandb_log.py`, and this file.
- MTP head provenance, digest, and draft policy: not applicable. No rollout,
  no session, no head was loaded. Every measurement is an isolated kernel
  dispatch.
- Token window, fixture, reference source, and harness: not applicable /
  not applicable / an in-process exact double-precision dequantize-and-
  accumulate reference computed on the CPU from the same operands /
  `harness=local`.
- Exact cell: `affine_qmv_fast<bfloat16_t, 64, 4, false>`, JIT source form,
  shapes 5120->34816 (mlp fused `gate_up`), 17408->5120 (mlp `down`) and
  5120->248320 (`lm_head`), M in {1,2,3,5,6,7,8}, dispatch
  `grid (M, (N+7)/8, 1)` x `threads (32, 2, 1)`, which reproduces
  `backend/metal/quantized.cpp:254` exactly. Not an M5 `_nax` variant; see the
  NAX gap below.
- Official causal path and score equation: `harness=ranked` value would come
  only from lowering candidate MTP seconds per token, since
  `d ln(ranked baseline serial time) / d(candidate edit) = 0`. This experiment
  never reached a candidate edit, so it contributes no ranked term.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh 4d937ce3 ...`
  reports all eleven changed files outside `editablePaths`, which is the
  correct outcome for research-only tooling and confirms the Yukon submission
  surface is untouched.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `source=2515544/3000000 headroom=484456 growth=60709/262144 exempt=2410
  files=154`, identical to the pre-experiment reading.
  `senpai/verify-ranked-score-boundary.sh` reports `PASS`.
- Scored-path reachability evidence: `qmv_fast_crossrow_affine4_g64_m` is the
  live decode kernel for M in {3,...,9} at `group_size == 64`, selected by the
  WIDE switch at `kernels/quantized.h:1918-1990`. `qmv_fast_impl` serves
  `ntg.x == 1`. Both were measured.

## Evidence

- Host, instance, chip, memory profile, toolchain, and thermal policy:
  `ip-10-231-2-227.ec2.internal`, Apple M4 Pro, 20 GPU cores, 48 GiB,
  `applegpu_g16s` (arch gen 16), Swift 6.3.3 (swiftlang-6.3.3.1.3),
  Metal `Apple metal version 32023.883 (metalfe-32023.883)`. Cool gate off for
  every arm. `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`, `timing_valid=false`,
  `official_or_ranked_score=false`. Rung 1a entry 36.21 C / exit 55.79 C;
  rung 1b entry 36.10 C / exit 55.86 C; rung 1b LUT control entry 37.78 C /
  exit 55.89 C.
- `head_provenance_sha256` for every leg: not applicable. No proposal head was
  loaded by any leg of this experiment.
- Exact baseline and candidate commands:

  ```bash
  research/e98_bytes_probe.sh e98-bytes-r1a
  research/e98_lut_probe.sh e98-lut-r1b \
      --shapes 0,1,2 --widths 1,2,3,5,6,7,8 --pairs 8 --target-ms 40 --lut 2658
  research/e98_lut_probe.sh e98-lut-r1b-small \
      --shapes 0,2 --widths 1,5,8 --pairs 8 --target-ms 40 --lut 64
  python3 research/e98_lut_analysis.py \
      --input research/out/e98-lut-r1b/lut.json \
      --json-out research/out/e98-lut-r1b/analysis.json
  python3 research/e98_wandb_log.py
  ```

  Arm sources are emitted from
  `research/jit_string_compile.py --emit -- 'affine_qmv_fast<bfloat16_t, 64, 4, false>'`
  and patched by `research/e98_variant_sources.py`. Arm SHA-256s are recorded
  in each `meta.txt` and in the W&B config.
- Cheapest real falsification gate and positive-control verdict: arms (a) and
  (b) receive different operand buffers that encode the same values, so they
  must agree bit for bit. They agree in **all 21 cells**, 0 differing outputs
  of up to 1,986,560 per cell. The positive control perturbs `biases[1][0]` by
  1.5x and is detected in all three shapes (1023/34816, 469/5120,
  7215/248320), so the comparison can fail. An independent exact
  double-precision reference gives `a_vs_double_rms_over_signal` of 2.8e-3 to
  6.7e-3, which is the bf16 output quantum.
- Tests and risk-based checks, in execution order: scope preflight, byte
  budget, ranked-score boundary, the rung-1a group-size ladder, the CPU-only
  AIR register census, the rung-1b three-arm fidelity check, the positive
  control, the rung-1b timing sweep, and the small-LUT residency control.
  `swift test` was not run: no submitted path changed and the new test is
  opt-in, so the 42-issue floor across 11 names is unchanged by construction.
- Exact-token and row-ledger verdict: not applicable. No rollout was run,
  because the pre-registered rung-1 stop rule fired before any candidate code
  was written.
- Divergent tokens or failure category: none. The failure category is
  **"not useful"**: a valid, lossless implementation with no useful end-to-end
  gain.
- Generated-twin audit: not applicable. `mlx-generated/quantized.cpp` and
  `kernels/quantized.h` are unchanged.
- Peak RAM or head/artifact size: not applicable.
- Official status and score: not submitted.

### Rung 1b, block-averaged, ABCCBA, 8 blocks per cell

Session null from the same-arm pair is +0.00 % to +0.73 % in every cell.
Byte share is 5.56 % for (a)->(b) and 11.11 % for (a)->(c) in every cell.

| shape | M | kernel | G | a_us | b_us | c_us | (a)-(b) | (a)-(c) | ratio |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| lm_head | 1 | qmv_fast_impl | 1 | 2865.86 | 2709.74 | 2548.70 | +5.45 % | +11.07 % | 0.492 |
| lm_head | 2 | crossrow | 1 | 2835.22 | 2682.62 | 2530.34 | +5.38 % | +10.75 % | 0.501 |
| lm_head | 3 | crossrow_m | 1 | 2905.04 | 2882.94 | 2825.26 | +0.76 % | +2.75 % | 0.277 |
| lm_head | 5 | crossrow_m | 2 | 5300.16 | 5321.29 | 5196.19 | -0.40 % | +1.96 % | -0.203 |
| lm_head | 6 | crossrow_m | 2 | 5715.80 | 5734.69 | 5619.88 | -0.33 % | +1.68 % | -0.197 |
| lm_head | 7 | crossrow_m | 2 | 6195.56 | 6288.76 | 6163.63 | -1.50 % | +0.52 % | -2.919 |
| lm_head | 8 | crossrow_m | 2 | 6705.03 | 6972.33 | 6791.65 | -3.99 % | -1.29 % | 3.086 |
| mlp_down | 1 | qmv_fast_impl | 1 | 200.03 | 188.48 | 177.18 | +5.77 % | +11.42 % | 0.506 |
| mlp_down | 2 | crossrow | 1 | 232.95 | 225.32 | 211.55 | +3.28 % | +9.19 % | 0.357 |
| mlp_down | 3 | crossrow_m | 1 | 252.73 | 252.73 | 247.27 | +0.00 % | +2.16 % | 0.000 |
| mlp_down | 5 | crossrow_m | 2 | 414.85 | 416.85 | 407.12 | -0.48 % | +1.86 % | -0.258 |
| mlp_down | 6 | crossrow_m | 2 | 445.99 | 447.20 | 438.77 | -0.27 % | +1.62 % | -0.168 |
| mlp_down | 7 | crossrow_m | 2 | 481.23 | 492.14 | 480.71 | -2.27 % | +0.11 % | -21.05 |
| mlp_down | 8 | crossrow_m | 2 | 519.20 | 545.82 | 527.51 | -5.13 % | -1.60 % | 3.204 |
| gate_up | 1 | qmv_fast_impl | 1 | 408.49 | 387.33 | 367.75 | +5.18 % | +9.97 % | 0.519 |
| gate_up | 2 | crossrow | 1 | 409.97 | 386.14 | 365.22 | +5.81 % | +10.91 % | 0.532 |
| gate_up | 3 | crossrow_m | 1 | 425.10 | 421.61 | 412.93 | +0.82 % | +2.86 % | 0.286 |
| gate_up | 5 | crossrow_m | 2 | 760.59 | 763.35 | 745.96 | -0.36 % | +1.92 % | -0.189 |
| gate_up | 6 | crossrow_m | 2 | 819.09 | 821.07 | 805.12 | -0.24 % | +1.71 % | -0.142 |
| gate_up | 7 | crossrow_m | 2 | 887.14 | 899.76 | 882.45 | -1.42 % | +0.53 % | -2.693 |
| gate_up | 8 | crossrow_m | 2 | 958.10 | 993.77 | 969.88 | -3.72 % | -1.23 % | 3.029 |

Achieved read rate on logical bytes at M = 5: lm_head 269.9 GB/s (98.9 % of the
273 GB/s peak), gate_up 263.7 GB/s (96.6 %), mlp_down 242.1 GB/s (88.7 %).

### Register census

CPU only, `metal -O2 -S` then `metal-opt -passes='default<O3>'`, entry
`affine_qmv_fast_bfloat16_t_64_4_false`:

| arm | peak_live_regs | delta vs shipped | peak_live_values | device_loads |
|---|--:|--:|--:|--:|
| a_shipped | 163 | 0 | 90 | 858 |
| b_indexed | 163 | 0 | 90 | 931 |
| c_constant | 155 | -8 | 82 | 712 |

This is entry-point scope with every width branch inlined, so the per-cell
ceiling of 108 from E46 and E59 does not apply. Only the arm delta is
meaningful, and it is zero.

### Look-up-table residency control

A 64-entry, 256-byte table is unconditionally L1-resident. It reproduces the
2,658-entry result within 0.25 pp, so table size is not the limiting factor.

| shape | M | (a)-(b), LUT 2658 | (a)-(b), LUT 64 |
|---|--:|--:|--:|
| lm_head | 1 | +5.45 % | +5.63 % |
| lm_head | 5 | -0.40 % | -0.33 % |
| lm_head | 8 | -3.99 % | -3.74 % |
| gate_up | 1 | +5.18 % | +5.13 % |
| gate_up | 5 | -0.36 % | -0.31 % |
| gate_up | 8 | -3.72 % | -3.52 % |

### Primary comparison

| Metric | Baseline (arm a, shipped) | Candidate (arm b, indexed) | Ratio / delta |
| --- | ---: | ---: | ---: |
| lm_head M=5 dispatch, us | 5300.16 | 5321.29 | -0.40 % |
| gate_up M=5 dispatch, us | 760.59 | 763.35 | -0.36 % |
| mlp_down M=5 dispatch, us | 414.85 | 416.85 | -0.48 % |
| metadata bytes per 64 elements | 36 | 34 | -5.56 % |
| dequantized values identical | reference | 0 differing of 1,986,560 | exact |

Serial seconds per token, MTP seconds per token, local serial-relative
speedup, effective mean draft length and accepted draft rate were **not
measured**, because the stop rule fired before any rollout. Do not infer them.

Every compared identity field matched across arms within each session: same
host, chip, memory profile, toolchain, commit, process, device, command queue,
operand values, dispatch shape and thermal policy. The single varied dimension
is the metadata read form. The 0.886 streaming share and the 249.55 GB/s
streaming rate are configuration imported from alphonse's E96 (W&B
`8m8d3mnr`); they are labelled as such and were not measured here. The local
ratio is never used as a ranked term.

## Conclusion

- What happened and why: the indexed encoding is exactly right and buys
  nothing. The instrument converts bytes into time at 1:1 wherever the kernel
  is byte-limited. At M = 1 in `qmv_fast_impl`, and at M = 2 in
  `crossrow<T,2>`, `(a)-(c)` is 9.97 % to 11.42 % against the 11.1 % byte
  prediction, and the `(a-b)/(a-c)` ratio is 0.49 to 0.53, which is what a pure
  bandwidth model predicts for removing half the metadata bytes. At M >= 3 the
  live `crossrow_m` family amortises each metadata word over IPG rows and holds
  it in cache. Removing **all** metadata then buys only 1.6 % to 2.9 %, and the
  index arm loses.
- Evidence for or against the mechanism: against. The pre-registered gate
  `(a) - (b) >= 0.45 * ((a) - (c))` fails at M = 3, 5, 6, 7 and 8, because
  `(a) - (b)` is zero or negative there. The round-level ceiling, using the E96
  streaming share of 0.886 and the measured M = 5 mean `(a)-(c)` of 1.91 %, is
  1.69 % of the round for removing **all** metadata bytes, so 0.85 % for the
  pair index even with a free lookup. The minimum useful effect in the brief is
  1.5 %. The measured effect is about -0.35 %. Arm (c) removes instructions as
  well as bytes, so its 1.9 % is an upper bound and the true byte term is
  smaller, which only strengthens the kill.
- Prompt or M5 transfer risk: the finding is input-independent, so prompt
  transfer risk is nil. Two hardware gaps remain and neither can rescue the
  mechanism. First, `metal::is_nax_available()` at `device.cpp:913-931`
  requires arch gen >= 17, so the ranked 512-token seed prefill through
  `qmm_t_nax` is not executable on this gen-16 host. Decode `qmv` never uses
  NAX, and decode is where the kill was measured. Second, an M5 with more
  bandwidth headroom would make the metadata stream *even less* likely to be
  DRAM-limited, so the ranked effect should be no larger than the local one.
- Smallest useful next action: re-price every remaining byte-reduction idea in
  the ledger against this measurement before building any of them. In the
  `qmv_fast_crossrow_affine4_g64_m` and `_wide` family, logical byte share
  multiplied by achieved bandwidth over-prices byte-removal work by about 5x.
  The cells read at 88 % to 99 % of DRAM peak **on logical bytes** and still
  return only about 0.17 of the predicted time for a byte cut. The correct
  discriminator is not "is the kernel near peak bandwidth"; it is "does the
  removed byte reach DRAM once per read, or is it replayed from cache".
- Recommendation: **close**. Do not reopen without new evidence that the
  scored decode kernel has become metadata-byte-limited, for example a change
  to IPG, to the cross-row amortisation, or to the cache hierarchy on the
  ranked host.

### Suggested follow-ups, not implemented

1. An arm (d) "pinned metadata" control, reading `scales[group_index & 7]`,
   would separate "metadata is cache-served" from "arm (c) is fast because
   instructions were removed". It sharpens the explanation and cannot change
   the decision.
2. Two source discrepancies stand, both recorded in interim 2:
   `kernels/quantized.h:1954-1966` argues for "3+3+2, not 4+4" at M = 8 while
   `:1967` ships `_m<T, 8, 4, true>`, and `research/e22-prereg-q1-addendum.md:58`
   is stale on the M = 8 IPG value.
3. The sentinel-collision census over all 498 tensors
   (`research/e98_sentinel_census.py`) was written and never run, because no
   transform change will be made. It remains available if the axis reopens.
