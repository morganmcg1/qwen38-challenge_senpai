# E79 — Head economics census: go/no-go on 3a and 3b

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":true,"value":2.3515},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

W&B run `0hj17swj` —
<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/0hj17swj>
holds every leg's identity, both per-position censuses with their width
histograms, the per-draft phase decomposition, the rung-0 reprice, the rung-3
score arms, and the ranked calibration tables.

- **Student / branch:** `qwen-alphonse` / `qwen-alphonse/e79-head-economics-census`
- **Hypothesis and target cost:** two named head-side proposals were priced
  against the published median. **3a** re-derives or retrains the proposal head
  weights to raise per-position acceptance. **3b** shrinks the compact draft
  vocabulary from 98,336 rows to 32,768 rows to make the head step cheaper.
  The target cost is the proposal-head step, which the shipped scheduler prices
  at `headStepCostRatio = 0.18` of a round.
- **Decision:** **3a NO-GO. 3b NO-GO.** The measured head-cost lever is real
  and is worth more than either proposal.
- **`BASE_SHA` / `UPSTREAM_SHA` / candidate commit:**
  `8d938c911df52b6a324f259a55dbaa75e508c822` /
  `bfab0de58d43453e506523707e1720a3485570f4` / no candidate commit. This
  experiment changed **zero** candidate files.
- **Yukon promoted submission / source ref used as frontier:**
  `9ad17378-848c-4441-b873-8cdc6177bf51`, score 3.25238228, sourceRef
  `bfab0de5…`. Our best ranked run is `9b241879` at 3.23588901.
- **Candidate build fingerprint:**
  `worker_sha256=a46d83e4610c04b08a84936a5fd98236f87786d11deb3096ce25142a155298fe`,
  `cli_sha256=693c7e13d628a7492546a0e370b4f5a036b740ec56610dc37a8f198ac5d8debf`,
  identical before and after every leg.
- **Submitted-surface / generated-twin / metallib digests:** unchanged. The
  metallib source fingerprint stayed
  `f09821bdbd820b77502867cbf660c1157407243ca9639de681c5b46fedfbd9fe` on every
  leg. No `.metal`, `.h`, or `mlx-generated/*.cpp` file was touched, so no twin
  audit was required.
- **Submitted candidate files:** none.
- **Supporting test, tooling, or documentation files:**
  `research/e79_trace_leg.sh`, `research/e79_session.sh`,
  `research/e79_head_economics.py`, `research/e79_wandb.py`,
  `research/e79-census-*.json`, `research/e79-chainfit.json`,
  `research/e79-reprice-measured.json`, `research/e79-reprice-declared.json`,
  `research/e79-price-declared.json`, `research/e79-calibrate.json`,
  `research/e79-results.md`.
- **MTP head provenance, digest, and draft policy:** two real heads were run.
  - *declared* — the head `mtp-head.manifest.json` declares,
    `qwen38-mtp-incumbent-q4-g64-plus-bf16-qkv-islands-v1`, 427,738,112 tensor
    bytes, staged directory sha256
    `559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`, runtime
    `head_provenance_sha256`
    `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`.
  - *pinned* — the organizer-pinned bf16 head
    `EigenLabs/Qwen3.8-27B-MTP-bf16`, 849,398,784 tensor bytes, runtime
    `head_provenance_sha256`
    `62516c6f3799b66c91171ee13aa6816db5af197aa8c527cec0f6bb4026f0c7b7`.
  - Draft policy: the shipped schedule, unmodified, depth cap 8.
- **Token window, fixture, reference source, and harness:** 512 decoded tokens,
  `--local-iterate`, the public single-prompt fixture, candidate-generated
  reference rows, harness `local`. Not ranked and not official.
- **Exact cell:** the proposal-head step inside `Qwen36MTPBlockSession`, plus
  the target verify ladder it feeds. Source form: Swift plus the vendored
  `Qwen35.swift` head path. No Metal source was inspected as a target because
  no kernel edit was proposed.
- **Official causal path and score equation:** the published score is
  `median(raw_1 … raw_8)`, and with eight prompts that median is the mean of
  the 4th and 5th sorted raw values. Every arm below is priced through that
  equation over the eight ranked prompts of ledger 207(A), not through a local
  ratio.
- **Assignment-scope preflight:**
  `senpai/validate-assignment-scope.sh 8d938c91… <15 research paths>` reports
  "outside editablePaths" for every path, which is the expected verdict for
  research-only work.
- **Editable source bytes / headroom / growth / exempt-head bytes:**
  `senpai/check-editable-budget.sh 8d938c91…` →
  `source=2469371/3000000 headroom=530629 growth=0/262144 exempt=2410 files=154`.
  **Zero candidate-surface growth.**
- **Scored-path reachability evidence:** every number below comes from the
  scored worker's own round trace, emitted from
  `Qwen36MTPBlockSession.swift:1484` while the worker served
  `--local-iterate`. No offline re-implementation of the head was timed.

## Evidence

### Host, toolchain, and thermal policy

- Host `ip-10-231-2-22.ec2.internal`, Apple **M4 Pro**, 51,539,607,552 bytes.
  This is **not** the ranked M5 host `m5-qwen38-27b-mtp`.
- Apple Swift 6.3.3 (swiftlang-6.3.3.1.3), target `arm64-apple-macosx26.0`.
- Two thermal policies were used and are labelled separately.
  - **ABBA, ungated** (`MLXFAST_LOCAL_COOL_GATE=0`): eight legs, four per
    head, run A1 B1 B2 A2 so monotone drift cancels to first order. Entry and
    exit GPU temperature are recorded for every leg.
    `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`.
  - **Gated** (the real 40 C gate, unmodified): one A B pair per head. The
    wrapper waited at the gate before each of the three timed phases and
    passed at 39.7–40.0 C. `cool_gate_passed_real_gate=true`,
    `gate_qualified_for_timing=true`.
- `meta.txt gpu_temp_entry_c` is sampled when the leg starts, which is
  **before** the harness runs its own gate. The gate-qualified entry
  temperature is the one in `wrapper.err`.

### Exact baseline and candidate commands

```bash
research/fetch-declared-head.sh                       # stage the declared head
research/e79_session.sh e79-r1  --pinned-head         # ungated ABBA, pinned
research/e79_session.sh e79-r3-decl                   # ungated ABBA, declared
research/e79_session.sh e79-r5-gated-decl --gated     # gated pair, declared
research/e79_session.sh e79-r6-gated-pin --pinned-head --gated

python3 research/e79_head_economics.py census research/out/<TAG>/trace.txt \
    --out research/e79-census-<TAG>.json

python3 research/e79_head_economics.py chainfit --out research/e79-chainfit.json

python3 research/e79_head_economics.py reprice --head-step-ms 4.7161 --reps 64 \
    --out research/e79-reprice-measured.json
python3 research/e79_head_economics.py reprice --head-step-ms 2.3329 --reps 64 \
    --p-vector 0.9474 0.9718 0.9565 0.9365 0.9649 0.9762 0.9459 0.9333 \
    --out research/e79-reprice-declared.json

python3 research/e79_head_economics.py price \
    --round-fixed 43.7432 --round-slope 17.5984 \
    --head-fixed 0.5074 --head-slope 2.3330 \
    --shape 0.9474 0.9718 0.9565 0.9365 0.9649 0.9762 0.9459 0.9333 \
    --out research/e79-price-declared.json

python3 research/e79_head_economics.py calibrate --out research/e79-calibrate.json
```

### A harness defect found before any measurement

`benchmark-qwen-mtp.sh:280` runs `eval "$(./setup-qwen-mtp.sh --print-paths)"`,
and `setup-qwen-mtp.sh` only provisions the **organizer-pinned** head. Nothing
in the local path reads `mtp-head.manifest.json`. The local harness therefore
runs the pinned head by default, while the ranked candidate leg runs the
declared head.

`score.json.metrics.uses_pinned_mtp_head` does not detect this. It is
`report.usesNativeMTPHead` (`Sources/MLXFastCLI/main.swift:1966`) and reports
`true` for both heads. The real discriminator is `head_provenance_sha256`.

Every earlier local measurement in this campaign that did not set
`MLXFAST_QWEN_MTP_HEAD_DIR` timed the pinned head, not the shipped candidate
head. `research/e79_trace_leg.sh` now defaults to the declared head and keeps
`E79_HEAD_DIR` as the override. This defect is reported, not repaired in the
harness, because the harness is outside this assignment's scope.

### Legs, identity, and correctness

Twelve legs, each timing a serial control and a native-MTP decode. Every leg:
exit 0, `all_tokens_matched=true`, `residual_divergence_count=0`, and a worker
and CLI digest identical before and after the run. Within a head, all legs are
trajectory-identical: same round count (76 pinned, 78 declared), same width
histogram, same accept histogram, same per-position counts. Only the timings
move.

| head | policy | legs | speedup | mtp s/token | serial s/token | M | accepted draft rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pinned | ABBA ungated | 4 | 2.2105 ± 0.0057 | 0.0335469 | 0.0741564 | 6.5132 | 0.8808 |
| declared | ABBA ungated | 4 | 2.3667 ± 0.0056 | 0.0313290 | 0.0741460 | 6.2692 | 0.8875 |
| declared | **gated** | 2 | 2.3515 | 0.031409 | 0.073857 | 6.2692 | 0.8875 |
| pinned | **gated** | 2 | 2.2144 | 0.033522 | 0.074232 | 6.5132 | 0.8808 |

Individual gated legs: declared 2.3571 and 2.3458, pinned 2.2167 and 2.2122.
Every gated leg passed the real 40 C gate three times, at 39.8 to 40.0 C, and
carries `cool_gate_passed_real_gate=true` and `gate_qualified_for_timing=true`.

The serial leg never uses the candidate head, so serial seconds per token is
the control that validates the head swap. It moved by **−0.01%** across the two
ungated blocks and by **−0.50%** across the two gated pairs, against a
candidate-leg move of −6.61% and −6.30%. The control is one order of magnitude
smaller than the effect under both policies. The gated control is the noisier
of the two because each gated arm is two legs rather than four.

**The gated pair confirms the ungated block.** Gate-qualified head-variant
contrast: speedup **+6.19%** and candidate seconds per token **−6.30%**,
against +7.06% and −6.61% ungated. The gated declared pair lands 0.6% below the
ungated declared mean and the gated pinned pair 0.2% above the ungated pinned
mean, so the ABBA block was mildly optimistic, not misleading.

### Rung 1 — per-position acceptance census, with the width mix beside it

Declared head, gate-qualified leg `e79-r5-gated-decl-a1-census512`, 76 scored
rounds:

| position | reached | accepted | p | Wilson 95% | rounds at width == position+1 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 76 | 72 | 0.9474 | [0.8723, 0.9793] | 1 |
| 2 | 71 | 69 | 0.9718 | [0.9030, 0.9922] | 0 |
| 3 | 69 | 66 | 0.9565 | [0.8798, 0.9851] | 5 |
| 4 | 63 | 59 | 0.9365 | [0.8478, 0.9750] | 4 |
| 5 | 57 | 55 | 0.9649 | [0.8808, 0.9903] | 22 |
| 6 | 42 | 41 | 0.9762 | [0.8768, 0.9958] | 4 |
| 7 | 37 | 35 | 0.9459 | [0.8230, 0.9850] | 6 |
| 8 | 30 | 28 | 0.9333 | [0.7868, 0.9815] | 34 |

Pooled 425/445 = **0.9551**.

Width histogram for the same leg: M=2:1, M=4:5, M=5:4, M=6:22, M=7:4, M=8:6,
M=9:34. Mean verify width **7.3158**, tokens per round 6.5921, and **43.6% of
rounds run at M=9**.

Pinned head, same fixture: p = 0.9324, 0.9710, 0.9403, 0.9667, 0.9630, 1.0000,
1.0000, 0.9459, pooled 436/453 = **0.9625**, mean width 7.5676, width
histogram M=4:5, M=5:6, M=6:16, M=7:3, M=8:3, M=9:41.

**Two findings.**

1. **p_i is flat.** There is no depth decay to repair. Position 8 is inside the
   Wilson interval of position 1 on both heads. The shipped EMA seed
   (0.8500 … 0.7379) encodes a decay that this fixture does not show; the
   converged EMA rises to 0.9962 … 0.8438.
2. **The census width mix does not match the ranked mix, and the mismatch runs
   the wrong way for 3a.** Local mean verify width is 7.32. Across the eight
   ranked prompts of ledger 207(A) the mean verify width is **4.86**, and the
   two prompts that set the median run at **5.53 and 5.77**. The local fixture
   spends 43.6% of its rounds at M=9; the median-setting ranked prompts never
   reach position 6 on average. A p_i measured mostly at M=9 therefore
   over-samples exactly the positions that cannot move the median.

### Rung 2 — head-step cost decomposition

Method: run the same leg twice, once plain and once with
`MLX_QWEN_MTP_TRACE_SYNC_HEAD=1`. With the flag the head chain is drained
before the verify window, so the head's GPU time moves out of the trailing
async evaluation and into `draft_build_us`, where the trace can see it. The
sync leg's `draft_build` slope is the in-situ head chain cost per draft.

Per-draft slopes, gate-qualified declared pair:

| phase | plain a1 ms/draft | sync b1 ms/draft | sync fixed ms | sync R² |
| --- | ---: | ---: | ---: | ---: |
| round | 17.6145 | 17.6537 | 43.6312 | 0.983 |
| draft_build | 1.6362 | **2.3050** | 0.8067 | 0.971 |
| d_submit2 | 1.4020 | 2.2605 | 0.1336 | 0.989 |
| verify_build | 8.0760 | 7.3266 | 21.4810 | 0.976 |
| eval_wall | 7.9000 | 8.0562 | 20.5997 | 0.980 |

The ungated declared ABBA block gives the same answer: head chain
`0.5074 + 2.3330 · d`, round `43.7432 + 17.5984 · d`. The pinned block gives
head chain `0.3794 + 4.7161 · d`, round `38.5153 + 20.6024 · d`.

**Only the sync leg is a valid head measurement, and the traces say so.** In
every plain leg the `draft_build` regression has a physically impossible
negative intercept — −4.47 to −6.43 ms on the declared head, −11.39 to −12.77
ms on the pinned head — because an asynchronous chain leaks part of the
previous round's work into the current window. In every sync leg the intercept
is positive and small (+0.27 to +0.81 ms) with R² 0.97 to 1.00. The plain leg
is therefore a lower bound on visible head time, and the sync slope is the
measurement. On the pinned head the two slopes agree anyway (4.61–4.86 plain
against 4.70–4.73 sync), which is consistent with a head heavy enough to block
on its own; on the declared head they do not (1.64–1.70 plain against 2.31–2.34
sync), so a plain-leg reading would have understated declared head cost by 28%
and exaggerated the declared head's advantage.

**True head-step cost ratio.**

| head | head ms/draft | round fixed ms | true h | total marginal ratio |
| --- | ---: | ---: | ---: | ---: |
| declared, ungated ABBA | 2.3330 | 43.7432 | **0.0533** | 0.4023 |
| declared, **gated** | 2.3050 | 43.6312 | **0.0528** | 0.4046 |
| pinned, ungated ABBA | 4.7161 | 38.5153 | **0.1224** | 0.5349 |
| pinned, **gated** | 4.7180 | 38.8201 | **0.1215** | 0.5300 |

The shipped constant is 0.18. Both thermal policies agree to within 1% on every
row, so this decomposition is gate-qualified, not an artefact of ABBA.
Gate-qualified, the pinned head costs **2.047×** the declared head per draft.

**Stage split.** No isolated four-stage micro-benchmark was run, and this is an
honest gap. All four stages are one MLX graph inside `d_submit2`, which is 97%
of the head chain, so a trace-level split is not available without changing
candidate code, and this assignment allows zero candidate edits. The split
below is therefore a **byte model**, validated across two structurally
different heads:

| stage | declared bytes | pinned bytes |
| --- | ---: | ---: |
| (i) head block | 270,400,512 | 849,398,784 |
| (ii) coarse readout | 157,337,600 | 283,207,680 (affine-4 select) |
| (iii) top-32 | 786,640 | 786,640 |
| (iv) rerank | 92,160 | — |
| **total** | **428,616,912** | **1,133,393,104** |

The model predicts a declared/pinned head-step ratio of **0.378**. The measured
ratio is **2.333 / 4.716 = 0.495**. The model is directionally right and
over-predicts the saving by 31%.

The residual has a physical explanation: implied effective bandwidth is
**240 GB/s** for the pinned head and **184 GB/s** for the declared head, against
an M4 Pro peak of 273 GB/s. The smaller head carries proportionally more fixed
per-dispatch overhead. Any byte-count argument for making the head smaller —
which is exactly 3b — is therefore optimistic by roughly this factor.

Coarse readout is 36.7% of the declared head's bytes. That is the largest
single stage 3b could attack, and it bounds 3b from above.

### Rung 0 — chain fit and reprice

**Chain fit** (`research/e79-chainfit.json`). Fitting a constant p per ranked
prompt: plutarch 0.3333, drama 0.6030, travel 0.7001, beagle 0.9362, medicine
0.9544, republic 0.9674, essays 0.9677, botany 0.9591. A geometric-decay chain
is **not identified** from the ranked aggregates: beagle admits any p1 in
0.94–1.00, medicine 0.95–1.00, republic 0.97–1.00, essays and botany 0.98–1.00.
The aggregates cannot resolve the shape. The local census resolves it as flat.

**Reprice, declared head** (`research/e79-reprice-declared.json`), 64
repetitions, measured p vector, shipped scheduler:

| arm | head ms | M | tokens/round | ms/token | vs ship |
| --- | ---: | ---: | ---: | ---: | ---: |
| head as measured, shipped flat price | 2.3330 | 7.3838 | 6.4389 | 24.7995 | +0.00% |
| head as measured, true-cost price | 2.3330 | 4.9295 | 4.5225 | 22.9646 | −7.40% |
| head at half, shipped flat price | 1.1665 | 7.3838 | 6.4389 | 23.6414 | −4.67% |
| head free, shipped flat price | 0.0000 | 7.3838 | 6.4389 | 22.4833 | **−9.34%** |
| head free, true-cost price | 0.0000 | 4.9634 | 4.5513 | 20.9003 | −15.72% |

Under the shipped flat price the width histogram is **identical** across all
head-cost arms (M=6 45.8%, M=9 39.1%). The shipped schedule does not react to
the head price at all, because `headStepCostRatio` is a constant, not a
measurement.

**Rung-0 stop rule, read against the median.** The advisor's stop rule is: if
the zero-head-cost schedule moves the **median** by less than 0.5%, the cost
side is dead. The median arm is `head cost x 0.00` in the rung-3 pricing table:
**+9.00%**. The stop rule does **not** trigger. The cost side is alive.
The −9.34% pooled local ms/token above is a secondary observation on one
prompt; no prompt is scored on pooled local time.

The pinned-head reprice (`research/e79-reprice-measured.json`, head 4.716 ms)
gives −17.29% for a free head, again with an unchanged width histogram.

### Rung 3 — pricing every arm through the published median

`research/e79-price-declared.json`. Constants: round `43.7432 + 17.5984 · d`,
head `0.5074 + 2.3330 · d`, measured position shape. The reconstructed baseline
published median is **3.23235** and the identity closes exactly.

Head share of a round at each prompt's ranked working point: plutarch 1.87%,
drama 6.97%, travel 7.41%, beagle 8.97%, medicine 9.11%, republic 9.38%,
essays 9.46%, botany 9.62%.

**3a arms — raise acceptance.**

| arm | median | delta |
| --- | ---: | ---: |
| p × 1.005 | 3.26492 | +1.01% |
| p × 1.010 | 3.29781 | +2.03% |
| p × 1.020 | 3.36460 | +4.09% |
| p × 1.030 | 3.43273 | +6.20% |
| p × 1.050 | 3.54154 | +9.57% |
| p_i → 0.97 everywhere | 3.36596 | +4.13% |
| p_i → 0.98 everywhere | 3.44314 | +6.52% |
| p_i → 0.99 everywhere | 3.52211 | +8.96% |
| p_i → 1.00 everywhere | 3.60289 | **+11.46%** |
| measured position shape | 3.28743 | +1.70% |
| deepest 3 restored to position 1 | 3.28743 | **+1.70%** |

The last two rows are the decisive pair. Restoring the three deepest positions
to position-1 acceptance is worth **exactly 0.00%** beyond the measured shape:
both arms give median 3.28742956. The median is set by beagle (4.533 drafts)
and medicine (4.768 drafts), and neither prompt reaches positions 6 to 8 often
enough for those positions to enter the median. A perfect head is worth
+11.46%, and that is the ceiling on every 3a variant.

**3b arms — shrink the compact draft vocabulary.** Coverage was measured as the
id-prefix coverage of the 512 decoded tokens of
`correctness_prompts/public_longcopy_gate_english_512_1024.json`.

| rows | head scale | coverage measured | break-even coverage | median | delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 98,330 | 1.0000 | 1.0000 | 0.99999 | 3.23236 | +0.00% |
| 90,138 | 0.9694 | 0.9961 | 0.9987 | 3.21526 | −0.53% |
| 65,562 | 0.8777 | 0.9844 | 0.9949 | 3.16473 | −2.09% |
| 40,986 | 0.7859 | 0.9629 | 0.9911 | 3.04841 | −5.69% |
| **32,794** | **0.7553** | **0.9355** | **0.9898** | **2.88201** | **−10.84%** |
| 24,602 | 0.7248 | 0.9082 | 0.9885 | 2.72570 | −15.67% |

**No size pays.** At the requested 32,768-row scale, coverage must be 0.9898 to
break even and the fixture measures 0.9355. Even with *perfect* coverage the
arm is bounded above by the `head cost × 0.75` arm at **+2.11%**, because a
32,794-row head is only 0.755× the cost of the current head. The `corpus`
re-selection estimator is degenerate on this single fixture; it is recorded in
the JSON and is not used for the decision.

**Head-cost arms — the lever that does work.**

| arm | median | delta |
| --- | ---: | ---: |
| head cost × 0.75 | 3.30045 | +2.11% |
| head cost × 0.50 | 3.37147 | +4.30% |
| head cost × 0.25 | 3.44563 | +6.60% |
| head cost × 0.00 | 3.52311 | **+9.00%** |

### Calibration against the advisor's ranked evidence

`research/e79-calibrate.json`. This is a check on whether a local acceptance
chain can predict ranked behaviour at all.

**Ranked round cost.** Regressing round time on drafts per round over the eight
ranked prompts of ledger 207(A):

```text
round_ms = 27.018 + 5.772 * drafts      R^2 0.9712
```

| quantity | ranked M5 | local M4 Pro, declared head |
| --- | ---: | ---: |
| total marginal ratio | **0.2136** | 0.4023 |
| head-only ratio | **0.0390** | 0.0533 |
| head share of one more draft | 18.2% | 13.3% |

The shipped `headStepCostRatio = 0.18` is **misnamed but well-valued**: it is
0.84× the true ranked *total* marginal ratio and 4.6× the true ranked
*head-only* ratio. It prices a whole extra draft, not a head step. That
explains why the ranked h sweep brackets it — 0.14 → 2.766, 0.15 → 2.667,
0.32 → 2.84585, with the 0.18 era near 2.93 — and it warns that any refit
taken from local slopes would land at roughly 0.40, about 1.9× too dear.
Caveat: this is a between-prompt regression over eight points, not a
within-prompt width sweep.

**Out-of-sample test on width — PASSES.** Each prompt's chain was fitted on the
h = 0.18 column alone, which is an exact one-parameter fit, then used to
predict the h = 0.32 column with no free parameter left.

| prompt | p fitted | obs @0.18 | pred @0.32 | obs @0.32 | error |
| --- | ---: | ---: | ---: | ---: | ---: |
| wide1 | 0.7946 | 4.35 | 3.135 | 3.36 | −0.225 |
| wide2 | 0.8462 | 4.89 | 3.710 | 4.01 | −0.300 |
| wide3 | 0.9236 | 5.78 | 4.927 | 4.53 | +0.397 |
| wide4 | 0.8854 | 5.33 | 4.263 | 4.03 | +0.233 |
| wide5 | 0.8597 | 5.04 | 3.877 | 4.76 | −0.883 |
| hard | 0.3204 | 0.17 | 0.055 | 0.06 | −0.005 |

Mean absolute relative error 9.2%, and 6 of 6 predicted the observed direction.

**Out-of-sample test on accepted tokens — FAILS.** The same chain cannot match
both observables at once. Beagle: p = 0.8081 reproduces 4.495 drafts but only
2.567 accepted tokens; p = 0.9362 reproduces 4.702 accepted tokens but 5.980
drafts. Observed is 4.533 / 3.785. Two extensions were tried and rejected: a
margin-clamp scale drove to its upper bound on all eight prompts, because the
clamp moves both observables the same way; and a two-parameter geometric chain
failed identification on 7 of 8 prompts, since at p1 = 1.0 and decay = 0.40 the
walk still reaches only 2.29 drafts.

Because the chain fails on accepted tokens, its arm-3 forecast is reported as a
**failed model, not a forecast**: it predicts median 2.127 (−34.19%) where
submission `2da69933` actually measured 3.21126 (−0.66%). Something in the
ranked path shortens rounds in a way this model does not contain. Named
candidates, none tested here: a parent `offeredDepth` below 8; unmodelled
session or EMA resets; and the `stoppedEarly` stop-token branch at
`Qwen36MTPBlockSession.swift:1020-1021`.

### 3a training cost, against the +11.46% ceiling

Using the assignment's own reference numbers on this host: teacher-forced
prefill 7.83 ms/token, autoregressive decode 67.3 ms/token, cached hidden
states 10.24 KB/token. A full 960-minute allocation caches at most about
**7.35 M teacher tokens**, or roughly **75.3 GB** of hidden states. Public MTP
and EAGLE-class heads are trained on 10^10 to 10^11 tokens. The budget is three
to five orders of magnitude short of a re-derivation, and the entire prize for
a *perfect* head is +11.46%.

### Tests and risk-based checks, in execution order

1. Assignment-scope preflight on all 15 research paths — all outside
   `editablePaths`, as expected.
2. Editable byte budget — `growth=0/262144`.
3. Worker and CLI digests captured before and after every leg — identical.
4. `all_tokens_matched=true` and `residual_divergence_count=0` on all 12 legs.
5. Trajectory identity within each head block — width, accept, and per-position
   histograms match exactly across legs.
6. Serial-leg control across the head swap — −0.01%.
7. Pricing identity closure — the reconstructed baseline median reproduces
   3.23235 exactly.
8. Out-of-sample calibration against two ranked h settings — passes on width,
   fails on accepted tokens; both reported.
9. `research/e79_wandb.py` executed against every artifact in offline mode
   before the online run.

**Exact-token and row-ledger verdict:** every leg matched, every round
accounted, zero divergences. **Divergent tokens:** none.

### Metric table

Baseline is the organizer-pinned head, candidate is the manifest-declared head.
Both are ungated ABBA means over four legs on one host, one base, one schedule,
one fixture.

| Metric | Baseline (pinned) | Candidate (declared) | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.0741564 | 0.0741460 | −0.01% |
| MTP seconds/token | 0.0335469 | 0.0313290 | **−6.61%** |
| local serial-relative speedup | 2.2105 | 2.3667 | **+7.06%** |
| effective mean draft length | 6.5132 | 6.2692 | −3.75% |
| accepted draft rate | 0.8808 | 0.8875 | +0.76% |
| pooled per-position acceptance | 0.9625 | 0.9551 | −0.77% |
| head step, ms per draft | 4.7160 | 2.3330 | **−50.5%** |
| true `headStepCostRatio` | 0.1224 | 0.0533 | — |

The local score is a one-prompt directional measurement. It is not the ranked
median across eight hidden prompts. Every compared identity field matched:
host, chip, memory profile, base commit, toolchain, metallib fingerprint,
worker and CLI digest, token window, fixture, and schedule. The only
intentional difference is the head.

## Conclusion

### What happened and why

**3a is a NO-GO.** Two structurally different heads were run as a natural
experiment on one machine: an 849 MB bf16 head with no `draft_lm_head`, and a
428 MB affine-4/affine-2 head with its own draft readout. They differ by 2.0×
in cost and by a whole quantization scheme, and their acceptance is
statistically indistinguishable — 0.8808 against 0.8875 accepted draft rate,
pooled per-position 0.9625 against 0.9551, in opposite directions. Acceptance
on this workload is limited by the target's own ambiguity, not by head
capacity. The direct evidence for that is the margin contrast: in rounds that
contain a rejection the target's top-2 margin has median 6.25, against 14.875
in fully accepted rounds — the head fails exactly where the target itself is
near-tied, which no amount of head training can fix.

The shape argument fails too. p_i is flat, so there is no depth decay to
repair, and pricing "restore the three deepest positions to position-1
acceptance" through the published median gives exactly **0.00%**, because the
two prompts that set the median never run that deep. The whole prize for a
*perfect* head is +11.46%, a realistic re-derivation reaches a small fraction
of that, and the available training budget is three to five orders of magnitude
short.

**3b is a NO-GO.** At 32,794 rows the head gets 0.755× cheaper, which through
the median is worth at most +2.11%, but the arm must also pay for every draft
token whose id falls outside the kept rows. Those are guaranteed rejections and
they cap p directly. Break-even needs coverage 0.9898; the fixture measures
0.9355; the arm prices at **−10.84%**. No size in the sweep pays, and the
largest attackable stage — coarse readout, 36.7% of declared head bytes — is
not large enough to change that. The bandwidth measurement makes it worse: the
already-small declared head runs at 67% of peak bandwidth against the pinned
head's 88%, so a smaller head keeps a growing share of fixed dispatch cost and
the modelled 0.755 scale is optimistic.

**The real lever is head cost, and it is worth more than either proposal.** A
free head is +9.00% on the median. The declared head has already captured half
of that path locally: 2.02× less head-step cost than pinned, +7.06% local
speedup, with no acceptance penalty.

### Evidence for or against the mechanism

For: the head-variant contrast is a clean natural experiment with an unchanged
serial control, the byte model predicts its direction across a 2.6× byte range,
and the pricing identity closes exactly on the baseline median. Against 3a and
3b specifically: the acceptance ceiling is small, its deep-position component
is worth zero at the median, and the vocabulary arm is negative at every size
tested.

### Prompt or M5 transfer risk

This is the largest caveat in the report and it runs in a known direction.

1. **Host.** M4 Pro, not the ranked M5. Bandwidth, dispatch overhead, and the
   `_nax` kernel variants all differ.
2. **Width mix.** The local fixture runs at mean verify width 7.32 with 43.6%
   of rounds at M=9. The eight ranked prompts average 4.86, and the two that
   set the median run at 5.53 and 5.77. Every local p_i is therefore measured
   at widths above where the median is decided.
3. **Cost level.** The local total marginal ratio is 0.4023 against the ranked
   0.2136. Local drafting is roughly 1.9× dearer relative to a round than
   ranked drafting. Head-cost deltas measured locally will shrink on M5.
4. **Model gap.** The acceptance chain predicts ranked *width* well and ranked
   *accepted tokens* badly. Any claim in this report that depends on accepted
   tokens rather than width should be treated as weaker.
5. **Single fixture.** Coverage for 3b was measured on 512 tokens of one public
   English prompt. It bounds the arm on that prompt only, and the arm is far
   enough below break-even that a better prompt is unlikely to rescue it.

### Smallest useful next action

Not implemented here, offered as follow-ups:

1. Test the ranked total marginal price directly. Set the schedule's depth
   price to the measured ranked value 0.2136 and take one ranked slot. The
   calibration says the shipped 0.18 is 0.84× of it, so this is a small, cheap,
   well-founded move on the constant that the whole schedule turns on.
2. Price depth with the *true* marginal cost locally rather than a flat
   constant. The reprice arm gives −7.40% ms/token at the current head cost,
   and it works by cutting mean width from 7.38 to 4.93 — that is E75 and
   Thorfinn territory and should be reconciled with their results before
   anyone reruns it.
3. Fix the local harness so `benchmark-qwen-mtp.sh` reads
   `mtp-head.manifest.json`. Until then every local measurement that does not
   set `MLXFAST_QWEN_MTP_HEAD_DIR` times the wrong head, and
   `uses_pinned_mtp_head` will not warn anyone.
4. If head cost is pursued, attack the coarse readout stage, which is 36.7% of
   declared head bytes, but budget for the bandwidth finding: a smaller head
   converts bytes to time at a worse rate than a larger one.

### Recommendation

**Close 3a and 3b.** Both are priced, both are negative, and neither needs a
repeat. Keep the head-cost lever open and route it through the depth-price
constant first, because that is where the calibration found a real, cheap,
ranked-verifiable discrepancy.
