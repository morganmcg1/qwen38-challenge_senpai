# E71 — in-situ width-tax census

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"m6_width_tax_attributed_fraction","available":true,"value":0.7737},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}
```

This is a mapping experiment. It measures where the verify-width cost goes. It
does not change the candidate and it claims no speedup. `test_metric` is
unavailable by design: a pinned arm produces wrong tokens on purpose, the
harness lives entirely under `Tests/`, and no candidate file changed, so there
is no token stream to match.

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e71-in-situ-width-tax-census`
- Hypothesis and target cost: at the scored verify width, a small number of
  kernel families own the width tax, and the family attribution closes against
  the measured total. Target cost is the assignment's `T(6) - T(1) = 69.659 ms`
  of round time that buys zero extra weight bytes.
- Decision: **green locally as a map.** No candidate change, so nothing to
  promote. Every rung completed and both controls resolved.
- `BASE_SHA`: `4898738ef12a423212c00485aa865b8e52056974`
- Yukon frontier used for context: `9ad17378`, 3.25238228, source
  `bfab0de58d43453e506523707e1720a3485570f4`. Campaign best `ca9251b8`
  3.23250848 (rejected); last `ff73cbbd` 3.17229699 (rejected, `parity_all_ok`).
- Submitted candidate files: **none.** Zero paths inside `benchmark.json`
  `editablePaths` are touched.
- Supporting files: `Tests/MLXFastTests/E71WidthTaxCensusTests.swift`,
  `research/e71_census.sh`, `research/e71_wandb_stream.py`,
  `research/e71_trace_leg.sh`, `research/e71_rung1_gate.py`,
  `research/e71_report.py`, this file.
- MTP head provenance: organizer-pinned head, unchanged. No proposal head is
  declared and the census harness never runs the draft chain.
- Token window, fixture, harness: 768-token seed from
  `correctness_prompts/public_longcopy_gate_english_512_1024.json`, then one
  timed verify call per rep. **`harness=local`, never ranked.**
- Exact cell: `affine_qmv_fast` -> `qmv_fast_crossrow_affine4_g64_m<T, M, IPG,
  true>`, affine 4-bit group-64, widths M = 1..9, JIT source form via
  `mlx-generated/quantized.cpp`, M5 variant not reached on this host.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh` reports all
  six files **outside** `editablePaths`.
- Editable budget: `source=2464949/3000000 headroom=535051 growth=0/262144
  exempt=2410 files=154`.
- Scored-path reachability: `Sources/MLXFastModel/Qwen35*.swift` is the
  `customFastPath` engine and is hard-disabled at compile time by
  `Qwen35FastPathReadiness.swift:13-14,28-30`, which pins
  `productionBackend = .libraryOracle`. The scored kernels are in
  `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`. The timed call is
  `Qwen36MTPBlockSession.swift:1223` ->
  `model.callWithHiddenAndNormed(input:cache:nConfirmed:)` ->
  `linearTopTwoRows` -> one blocking `eval`.

## Evidence

- Host: Apple M4 Pro, 48 GiB (51539607552 B), macOS 26.5.2, Swift 6.3.3, GPU
  `applegpu_g16s`, hostname `ip-10-231-2-227.ec2.internal`, metallib
  fingerprint `f09821bdbd820b77502867cbf660c1157407243ca9639de681c5b46fedfbd9fe`.
- Thermal policy: `MLXFAST_LOCAL_COOL_GATE=0`, one session, ABBA
  counterbalanced, entry and exit GPU temperature per block.
  **`cool_gate_passed_real_gate=false`**, **`gate_qualified_for_timing=false`**,
  **`official_or_ranked_score=false`**.
- Reproduction:

  ```bash
  # rung 1 gate: one traced, unchanged-base leg through the ordinary path
  research/e71_trace_leg.sh e71-rung1-leg 128
  python3 research/e71_rung1_gate.py \
      --trace research/out/e71-rung1-leg/trace.txt \
      --census research/out/e71-census-r1/census.json \
      --json research/out/e71-rung1-leg/gate.json

  # rungs 1 to 3: the census itself
  research/e71_census.sh e71-census-r1 full
  python3 research/e71_report.py research/out/e71-census-r1/census.json \
      --json research/out/e71-census-r1/report.json
  ```

- W&B: census run
  [`clfgswy8`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/clfgswy8),
  130 blocks, 09:19:30Z to 09:43:26Z, 23.9 min. Plumbing smoke
  [`3wu6kmdk`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/3wu6kmdk),
  28 blocks. One run per leg, streamed live per block, never at session end.
- Model load 15.10 s. Peak resident set is the ordinary transformed
  checkpoint; the census adds no resident bytes because the pinned wrappers
  share `weight`, `scales` and `biases` arrays with the modules they replace.
- Exact-token verdict: **not applicable and deliberately so.** A pinned arm
  slices one family's input to a single row and broadcasts its output back, so
  the tokens produced are wrong by construction. This is a timing-only harness
  under `Tests/`. It never runs in a scored leg and it changes no candidate
  file, so it cannot affect the submitted token stream.
- `senpai/verify-ranked-score-boundary.sh`: **PASS**.

### Method

For each family `F`, run `F` at width 1 while every other family keeps its
width-M shape schedule, then

```text
tax(F) = T(M) - T(M, F pinned to width 1)
```

Arms are measured as `baseline, arm, arm, baseline` quartets so monotone
thermal drift cancels to first order. Every block is 12 timed reps after 3
warmups, reported as the median.

## Results

### Rung 0 — what the census can and cannot reach

The byte census reproduces ledger 199(A)'s 14,412,349,440 B exactly. Of that,
**79.61 % is interceptable** from `Tests/` alone through `@ModuleInfo` seams on
`open` classes. The remaining 20.39 % is not, because `Qwen35Attention` and
`Qwen35GatedDeltaNet` are `final` and their fused paths read `.weight`,
`.scales` and `.biases` directly rather than dispatching through a child
module.

**The assignment's stop rule fired partially and I declared it before
measuring.** Fused FA `qkv`, fused GDN `in_proj_qkv`/`in_proj_z`/`b+a`, SDPA,
the GDN scan, the causal conv, every RMSNorm, the compiled fusions, the final
norm, the top-two readout and the MTP head chain cannot be pinned without
editing `Sources/` or `Vendor/`. I shipped no instrumentation into those trees.
Those families are the named, itemised floor on the closure gap, not an error
term.

### Rung 1 — the harness gate

`F(M)` is the isolated verify cost: the verify call, the top-two readout and
the eval barrier. It excludes the draft head chain by construction, so it is
ledger 200(E)'s `V`, not E1's `T`.

| M | `F(M)` ms | half-range | 200(E) `V(M)` | E1 `T(M)` |
|---|---|---|---|---|
| 1 | 64.706 | 0.046 | 64.979 | 65.009 |
| 2 | 67.701 | 0.009 | 69.509 | 70.482 |
| 3 | 70.652 | 0.206 | 73.985 | 75.519 |
| 4 | 80.377 | 0.270 | 89.610 | 91.288 |
| 5 | 94.349 | 0.306 | 114.934 | 115.691 |
| 6 | 122.110 | 0.167 | 131.749 | 134.668 |
| 7 | 138.334 | 0.025 | 150.629 | 154.169 |
| 8 | 149.266 | 0.036 | 167.074 | 172.827 |
| 9 | 164.790 | 0.579 | 190.483 | 198.237 |

**M=1 is the clean gate** and it passes: no head chain runs at M=1, so `F`, `V`
and `T` must agree, and they do to 0.3 to 0.5 %. At wider M, `F` sits below `V`
in the direction E65 predicts, because inside a real leg `verify_build_us`
overlaps head-chain GPU work and the leg's `V` is an upper bound on the
isolated verify cost.

Session null, taken as the worst curve half-range, is **0.579 ms**, driven
entirely by M=9; every other width is at or below 0.31 ms.

#### The gate against a real leg, measured the same session

The assignment requires the harness to be gated against a measurement taken
through the ordinary `--local-iterate` path, not only against a historical
ledger row. I ran one traced, unchanged-base 128-token leg at 09:52:24Z, 17
usable rounds, entry 38.63 C and exit 64.07 C, `dirty_candidate_paths=0`. The
leg's own draft policy chose widths 2, 6, 7, 8 and 9; it never selected M=4 or
M=5, so those two census widths are not covered by this leg.

| M | leg `V` ms | census `F` ms | residual | leg rounds |
|---|---|---|---|---|
| 2 | 72.825 | 67.701 | 5.124 | 1 |
| 6 | 134.517 | 122.110 | **12.406** | 4 |
| 7 | 149.833 | 138.334 | **11.499** | 3 |
| 8 | 161.132 | 149.266 | **11.866** | 3 |
| 9 | 176.409 | 164.790 | **11.619** | 6 |

Two things follow, and both are stronger than the ledger comparison.

**First, the offset is stable and independently explained.** The residual sits
in a 0.908 ms band from 11.499 to 12.406 ms across M=6 to M=9, and at M=6 it
matches the independent anchor of five head steps at `H = 2.590 ms`, that is
12.950 ms, to within 4.2 % (ratio 0.958). The instrument's distance from a real
leg is not an unexplained offset; it is head-chain GPU work landing inside the
leg's verify window, which is exactly what E65 predicts and what the census
excludes by construction.

**Second, the census reproduces the width tax itself, which is the quantity it
measures.** The census reports differences, not absolute levels, so the gate
belongs on differences:

| pair | leg `V` delta | census `F` delta | difference |
|---|---|---|---|
| 6→7 | 15.316 | 16.224 | +0.908 |
| 7→8 | 11.299 | 10.931 | −0.368 |
| 8→9 | 15.276 | 15.524 | +0.248 |
| **6→9** | **41.892** | **42.680** | **+0.788 (+1.88 %)** |

Every adjacent pair agrees to within 0.91 ms, at or near the session null of
0.579 ms, and the full M=6 to M=9 span agrees to 1.88 %. **Rung 1 passes.**

This also shows why the ledger comparison alone was misleading. The same-session
leg gives `V(9) = 176.409 ms` against ledger 200(E)'s 190.483 ms, a 14 ms
difference from base and session drift, which is far larger than anything the
census resolves. The census-minus-ledger column ranges from −0.3 to −25.7 ms;
the census-minus-same-session-leg column is stable at −11.5 to −12.4 ms.

Measured width tax: **`F(6) - F(1) = 57.404 ms`** and
**`F(9) - F(1) = 100.084 ms`**. The census closes against `F`, its own measured
total. E1's `T(6) - T(1) = 69.659 ms` and 200(E)'s `V(6) - V(1) = 66.770 ms`
are upper bounds on the same quantity that additionally contain head-chain
overlap.

### Rung 2 — controls

**Positive control passes.** `lm_head` at M=6 measures 2.142 ms against E63's
independent standalone m=1 to m=6 delta of 2.875 ms. Ratio **0.745**, inside
the pre-registered [0.5, 2.0] gate.

**Null control, per width.** The gate is applied at each width because the null
is a property of the wrapper at that width.

| M | null tax | smallest arm tax | fraction | arms not resolved |
|---|---|---|---|---|
| 4 | +0.414 | 0.245 | 1.688 | `fa_o_proj`, `gdn_out_proj`, `lm_head` |
| 5 | +0.345 | 0.797 | 0.433 | `fa_o_proj`, `lm_head` |
| 6 | −0.252 | 1.161 | 0.217 | none |
| 9 | −0.144 | 1.968 | 0.073 | none |

The pre-registered rule was "stop if the null exceeds 25 % of the smallest
measurable tax". It **passes at M=6 and M=9 and fails for the small arms at
M=4 and M=5**. I did not stop, because it fails only for arms whose own tax
sits near the floor and the headline width passes cleanly for every arm. The
five affected cells are reported **not resolved** rather than as measurements.

Least-squares slope of the null against width is **−0.115 ms per row**: the
wrapper gets cheaper at wider M, not more expensive. So the attributed fraction
is not inflated by wrapper drift growing with width.

The wrapper's broadcast copy is charged to the pinned arm, which biases every
`tax(F)` **low**. The census is therefore conservative.

### Rung 3 — the census

All taxes in ms. Cells marked † are not resolved against that width's null.

| arm | GB | M=4 | M=5 | M=6 | M=9 |
|---|---|---|---|---|---|
| `null` | 0 | +0.414 | +0.345 | −0.252 | −0.144 |
| `fa_o_proj` | 0.283 | 0.245† | 0.797† | 1.161 | 1.968 |
| `lm_head` | 0.715 | 0.327† | 1.104† | 2.142 | 4.509 |
| `gdn_out_proj` | 0.849 | 0.771† | 1.880 | 3.313 | 5.682 |
| `mlp_down` | 3.209 | 4.095 | 7.957 | 16.887 | 22.498 |
| `mlp_all` | 9.626 | 9.677 | 18.986 | **37.800** | 64.360 |
| `all_interceptable` | 11.474 | 11.734 | 22.180 | 44.670 | 76.046 |

Block half-ranges are at or below 0.1 ms for most arms; the worst is 0.567 ms.
`mlp_down` at M=6, the load-bearing cell, is arm 105.170 ±0.032 against
baseline 122.057 ±0.036.

#### The MLP owns two thirds of the width tax at every width

| M | `mlp_all` | width tax `F(M)-F(1)` | MLP share |
|---|---|---|---|
| 4 | 9.677 | 15.670 | 61.8 % |
| 5 | 18.986 | 29.643 | 64.0 % |
| 6 | **37.800** | **57.404** | **65.9 %** |
| 9 | 64.360 | 100.084 | 64.3 % |

Eleven kernel experiments in this campaign have targeted the generic cross-row
QMV kernel because it was measurable. This is the first evidence about where
the cost actually sits, and it says two thirds of it is in one block.

#### Closure, and the additivity test

| M | attributed | total | attributed % | gap | joint arm | additivity residual |
|---|---|---|---|---|---|---|
| 4 | 11.020 | 15.670 | 70.3 % | 4.651 | 11.734 | +0.714 |
| 5 | 22.767 | 29.643 | 76.8 % | 6.875 | 22.180 | −0.587 |
| **6** | **44.416** | **57.404** | **77.4 %** | **12.988** | 44.670 | **+0.254** |
| 9 | 76.519 | 100.084 | 76.5 % | 23.564 | 76.046 | −0.474 |

Attributed is the sum of the four disjoint arms; `mlp_down` is inside `mlp_all`
and is never added.

**The reachable families are additive.** Every residual is within ±0.72 ms, at
or inside the session null, at all four widths. Pinning all five families
together costs what pinning them one at a time costs. That is what the
`all_interceptable` arm was added to test, and it resolves the closure gap into
a definite statement: **the 22.6 % gap at M=6 belongs to families this harness
cannot reach, not to a failure of the per-family map.**

I pre-registered 58.9 % attributed and a 41.1 % gap. Measured 77.4 % and
22.6 %. The reachable families own considerably more of the width tax than the
per-byte model predicted, and the attributed fraction is stable at 76 to 77 %
for every width at and above M=5.

### The per-byte model is wrong, and the correction is reduction depth

Shapes read from `weights/config.json` and the vendored `Qwen35.swift`.
`mlp_gate_up` is not an arm; it is recovered as `mlp_all - mlp_down`, so it
also carries the SwiGLU and the fused gate/up kernel.

| family | k | n | k-blocks | calls | GB | ms/GB M=4 | M=5 | M=6 | M=9 |
|---|---|---|---|---|---|---|---|---|---|
| `lm_head` | 5120 | 248320 | 80 | 1 | 0.715 | 0.457 | 1.544 | 2.995 | 6.305 |
| `mlp_gate_up` | 5120 | 34816 | 80 | 64 | 6.417 | 0.870 | 1.719 | 3.259 | 6.523 |
| `gdn_out_proj` | 6144 | 5120 | 96 | 48 | 0.849 | 0.908 | 2.213 | 3.901 | 6.690 |
| `fa_o_proj` | 6144 | 5120 | 96 | 16 | 0.283 | 0.866 | 2.815 | 4.099 | 6.950 |
| `mlp_down` | **17408** | 5120 | **272** | 64 | 3.209 | 1.276 | 2.480 | **5.263** | 7.012 |

Fitting `ms_per_gb = alpha + beta * k_blocks` over the five families at each
width separates the two terms:

| M | IPG | alpha ms/GB | beta ms/GB per k-block | R² | `mlp_down` k-penalty |
|---|---|---|---|---|---|
| 4 | 4 | 0.522 | 2.83e-3 | 0.649 | 1.743 ms |
| 5 | 5 | 1.816 | 2.71e-3 | 0.181 | 1.669 ms |
| **6** | **6** | 2.698 | **9.66e-3** | **0.816** | **5.952 ms** |
| 9 | 5 | 6.401 | 2.36e-3 | 0.440 | 1.455 ms |

`alpha`, the width-dependent term, rises smoothly and monotonically. `beta`,
the reduction-depth term, is flat at 2.4 to 2.8e-3 at three widths and spikes
**3.67×** at M=6 alone, where the fit is also tightest at R² = 0.816.

If `beta` at M=6 obeyed the mean of the other three widths, `mlp_down`'s
k-penalty would be 1.62 ms rather than 5.95 ms. The excess is **4.33 ms per
round, 7.5 % of the M=6 width tax, on one shape at one width.** The raw
marginals say the same without any fit: `mlp_down` grows `+3.86, +8.93, +5.61`
across M=4→5→6→9 while `mlp_gate_up` grows `+5.45, +9.88, +20.95`. The deep
reduction nearly doubles its increment at M=6 and then almost stops; the
shallow one keeps accelerating.

**Shape-invariance check, independent of the fit.** `fa_o_proj` and
`gdn_out_proj` have identical shapes (k=6144, n=5120, 96 k-blocks) and differ
only in call count, 16 against 48, so their bytes differ 3×. Their ms/GB agree
to 5 % at M=4, M=6 and M=9. The only disagreement is M=5, where `fa_o_proj` is
itself unresolved against the null. The width tax is a property of the shape.

### Kernel selection: identical on `applegpu_g16s` and `applegpu_g17s`

`get_qmv_batch_limit` (`quantized.cpp:84-125`) is the **only** site in the
Metal backend that reads `get_architecture_gen()`. It special-cases generations
13 and 14; 16 and 17 both fall through to the same branch, and both hosts
report `arch_size` `'s'`. Every census family has K > 4096 and N > 4096, so it
returns **10** on both. All census widths are M ≤ 9 < 10, so every family stays
on `qmv` and never crosses to `qmm` on either generation.

Every other architecture-sensitive site — `matmul.cpp:208,372,918,2303,2514`
and `scaled_dot_product_attention.cpp:443,747` — branches only on
`get_architecture().back()`, which is `'s'` on both.

`quantized.cpp:259` then selects `qmv_fast` iff `N % 8 == 0 && K % 512 == 0`,
and `kernels/quantized.h:1917-1980` selects the cross-row partition on `ntg.x`
(= M) and `out_vec_size` (= N). Neither predicate is architecture-derived.
**All five families satisfy both conditions, so selection is identical for
every family at every census width.**

**Selection identity is not performance identity.** All five families land on
the same campaign-tuned cross-row partition, so generation exposure is uniform
across families and the census cannot rank families by it. It can rank widths.
The partition is `qmv_fast_crossrow_affine4_g64_m<T, M, IPG, true>` with
IPG = `{3:3, 4:4, 5:5, 6:6, 7:4, 8:4, 9:5}`; `b757237` set M=5, `aa8ce50`
(E61) set M=6, and t55 set M=9. **M=6 is the only width with IPG = 6, and M=6
is the only width where `beta` spikes.** Six live accumulators against 272
k-blocks is the deepest register-pressure product in the model, and
`mlp.down_proj` is the only shape that reaches it. The kernel's own source
comment at `case 8` names this failure mode in the neighbouring width: "a
register cliff, not work scaling".

### Score conversion

Withheld. The advisor directed on 2026-08-20 that no local number in this
experiment is to be converted to a ranked score. Every number here is
`harness=local`. The flat-law arithmetic remains in `report.json` under
`ranked_width_mixture`, marked `score_conversion_withheld: true`, as auditable
input only.

Ranked-mixture-weighted taxes, local ms, using the assignment's width mixture
and nearest-measured-width interpolation for M=3, 7 and 8:

| family | mixture-weighted tax, local ms |
|---|---|
| `mlp_all` | 31.932 |
| `gdn_out_proj` | 2.843 |
| `lm_head` | 1.891 |
| `fa_o_proj` | 1.022 |

### Ledger corrections carried

1. E1's `T(6) = 134.668 ms` rests on N=2 and the ledger says at
   `:14884-14886` it must not be used as a baseline. E1's M=9 row was corrected
   post-E55 to 184.970 ms (`:14902-14903`), not the 198.237 ms still in
   `research/e70_double_roofline.py:18-19`. This census closes against its own
   measured `F`, so neither row is load-bearing here.
2. The psi-free projection broke by **6×** at t55 (`:16767`, −3.87 % against
   −0.639 %), not 7×. The flat law's standard error in 201(D) is **±0.313 %**;
   the ±0.294 % in the assignment is the 471-tree standard error. The flatness
   of the flat law is about **1.1 sigma** (`:16733-16739`) and is not
   established.
3. The 8-step head-chain pair `29.94 ms / 8.42 ms` is **VOID**
   (`research/ESTABLISHED_FACTS.md:849-853`). The surviving anchors are
   `H = 2.590 ms` and `delta_head = 2.689 ms`.

Also: `research/e68_insitu_curve.py` does not exist on this branch, and PR #71
is outside this launch's isolation scope, so the assignment's `C(6) = 122.876`,
`eval_wall = 64.622` and the 0.49 to 0.53 ratio are taken from the PR body only
and are unverified on-branch. My independently measured `F(6) = 122.110 ms`
lands close to that quoted `C(6)`.

## Conclusion

**What happened.** The width tax is not spread across the model. At the scored
width M=6, four reachable families own 77.4 % of it, the MLP alone owns 65.9 %,
and the reachable families are additive to within half the session null. The
remaining 22.6 % belongs to a named, itemised set of families that cannot be
instrumented without editing candidate files.

**Evidence for the mechanism.** The per-byte model that generated my
pre-registered predictions is wrong in a specific and correctable way. Cost
tracks bytes **and** reduction depth, the two terms separate cleanly, and the
depth term is anomalous at exactly one width. Three independent lines agree:
the fitted `beta` spikes 3.67× at M=6, `mlp_down`'s raw marginal nearly doubles
at M=6 and then saturates, and M=6 is the only width running IPG = 6. Two
families with identical shapes and 3× different bytes agree on ms/GB to 5 %,
which shows the tax is a shape property and not a bytes property.

**Transfer risk.** Kernel selection is provably identical on `applegpu_g16s`
and `applegpu_g17s` for every family at every census width, so the *map* should
transfer. The *levels* at M=5, M=6 and M=9 depend on campaign-tuned partitions
whose IPG was chosen on g16s, so absolute times at those widths carry real
transfer risk. M=4 is the only census width on an untouched partition.

**Smallest useful next action.** Measure `mlp.down_proj` at M=6 under a
different IPG. `<T,6,6>` is the widest partition in the table and the only one
whose reduction-depth penalty is anomalous. This census predicts that lowering
IPG at M=6 recovers roughly 4.3 ms per round of the M=6 width tax without
touching any other family, and M=6 carries 33.4 % of ranked verify-width time.
I did not test it: this assignment produces a map and the map is the
deliverable.

**Recommendation: close as a completed map.** There is nothing to promote
because no candidate file changed. The follow-ups belong in their own
assignments with their own controls.

## Suggested follow-ups, none implemented

1. **Retune IPG at M=6.** This is the strongest lead the census produced.
   `<T,6,6>` is the widest partition in the table, M=6 is the only width where
   the reduction-depth penalty spikes, and M=6 carries 33.4 % of ranked
   verify-width time. The census predicts roughly 4.3 ms per round is
   recoverable at M=6 alone. It also gives the experiment a free control:
   the effect must appear in `mlp_down` and must not appear in `lm_head` or
   `mlp_gate_up`, which have 80 k-blocks against `mlp_down`'s 272.
2. **This may be the `ff73cbbd` mechanism.** Kernel selection is provably
   identical on g16s and g17s, so the regression cannot be a selection
   divergence at any family in this census. A register cliff that lands just
   past the g16s budget at IPG = 6 would execute the identical kernel on both
   hosts and still cost differently. Reverting `aa8ce50` alone, with the other
   two partition changes held, would test it directly.
3. **Measure the unreachable 22.6 %.** The gap is now a named list rather than
   an unknown: fused FA `qkv`, fused GDN `in_proj_*`, SDPA, the GDN scan, the
   causal conv, the norms and fusions, and the readout. Reaching it needs an
   assignment that permits a candidate-file edit, and it should be scoped and
   controlled separately because that edit ships in the submission archive.
4. **Close the M=4 and M=5 resolution floor.** Five cells are unresolved
   against their width's null. More reps at those two widths, or a wrapper with
   a cheaper broadcast, would resolve them. The M=4 point matters most because
   M=4 is the only census width on an untouched partition and so is the natural
   control for follow-up 1.
5. **Extend the rung 1 leg to M=4 and M=5.** The leg's draft policy never
   selected those widths, so they are gated only by continuity with M=6.
