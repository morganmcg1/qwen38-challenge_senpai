# E121 — cross-simdgroup activation-sum sharing in the wide cross-row 4-bit matvec

`harness=local` unless a line says otherwise. No number in this file is an
official or ranked score.

## Question

Does `x_sumshare_min`, gated off at `NA = 5` by `if constexpr`, ship a bit-exact
round-weighted win above +1.0 %?

Answer: **yes at the kernel level, not yet demonstrated end to end.**

- Kernel frame: **+1.463 %** round-weighted, CI95 excluding NA=5
  [+1.042, +2.243]. The shipped arm is `g_split_pred`.
- Leg frame: **−0.436 %** (negative = faster), n = 2, CI95 [−1.268, +0.395].
  The point estimate clears the promotion bar; the interval does not exclude
  zero.
- The end-to-end effect is **2.04× smaller** than the pre-registered −0.888 %,
  which trips the advisor's factor-of-two stop rule. **Halted and reported. No
  submission was made.**
- Correctness is unaffected: full 512-token exactness passes and all eight
  timed legs match.

## Identity tuple

| field | value |
| --- | --- |
| assignment | `qwen38-r1-e121-cross-simdgroup-activation-sum-sharing-shipped` r1, PR #122 |
| branch | `qwen-alphonse/e121-cross-simdgroup-sum-sharing` |
| base_sha (campaign) | `2127858ba770ddc06027205d8df89a8db21d80f5` |
| growth base | `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf` |
| host | `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, 48 GiB |
| local GPU arch | `applegpu_g16s` |
| ranked GPU arch | `applegpu_g17s` (M5) |
| toolchain | Swift 6.3.3, metal 32023.883 |
| token window | 512 decode tokens for every exactness and end-to-end leg |
| head | declared run head, provenance `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9` |
| reference source | pinned public golden rows for exactness; candidate-generated rows for local scoring only |

## Mechanism

`qmv_fast_crossrow_affine4_g64_wide<T, NA, ...>` runs two simdgroups per
threadgroup. Every simdgroup used to compute the activation partial sums for all
`NA` rows, because each row's affine dequantisation correction needs
`sum(x_row)` over the group. Both simdgroups walk the identical `k` range, so
exactly half of that reduction is duplicated.

The shipped arm splits ownership:

```
constexpr bool SHARE_SUMS = NA <= 4;
constexpr int  H          = NA / 2;
const bool own_lo = simd_gid == 0;
const bool owns_m = !SHARE_SUMS || ((m < H) == own_lo);
```

`owns_m` guards both `DIRECT_NIBBLES` branches. An `if constexpr (SHARE_SUMS)`
block then stores the owned half to `threadgroup float sums_xchg[1 * 4 * 32]`
(512 B), barriers, loads the other half, and barriers again, all before the
`acc[r] +=` loop. The wide template signature gained `uint simd_gid` and
`threadgroup float* sums_xchg`; the entry point declares the scratch array.

**Bit-exactness is by construction, not by measurement.** Lane `L` of simdgroup 0
and lane `L` of simdgroup 1 walk the same `k` range in the same order with the
same reduction tree, so the value lane `L` of simdgroup 1 now reads from scratch
is the identical `float` it previously computed. No reassociation, no precision
change, no rounding change. This is why this mechanism was chosen over faster
looking variants that would have reassociated the reduction.

## Rung 0 — census (cost shares)

Per-width cost shares of the wide family inside one real 512-token decode,
at `NA = 4`:

| shape | share |
| --- | --- |
| `mlp_gate_up_k5120_n34816` | 0.460 |
| `mlp_down_k17408_n5120` | 0.248 |
| `gdn_in_proj_k5120_n16480` | 0.155 |
| `gdn_out_proj_k6144_n5120` | 0.092 |
| `fa_qkv_k5120_n14336` | 0.045 |

Round weights over widths: `{2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}`.

Artifact: `research/e121-artifacts/rung0-census.json`.

## Rung 1 — compiled-kernel inspection

Three campaign facts came out of rung 1 and rung 2 and now stand under my name.

### F1. The entry-point occupancy tax is real

`qmv_fast_crossrow_affine4_g64_wideN` is a `METAL_FUNC`, so it is inlined. The
kernel entry point therefore allocates registers for the widest body it can
reach. Measured resident simdgroups on the entry point:

| arm | simdgroups | change |
| --- | --- | --- |
| base | 39 | — |
| ungated share (either form) | 32–33 | ≈ −15 % |
| gated share (`NA <= 4`) | 38 | ≈ −2.6 % |

### F2. The gate is right, but for a different reason than predicted

The pre-registered reason for the gate was register spilling at `NA = 5`. That
is a **g16s artifact**. On the ranked `g17s` neither ungated form spills:

| form | registers (g17s) | spill |
| --- | --- | --- |
| `x_split_pred` | 120 | 0 |
| `x_min_ask` | 121 | 0 |

The gate still earns its ≈ +1.01 pp, but through **entry-point occupancy**
(F1), not through spill avoidance.

**This is the largest local-to-M5 transfer risk in this experiment.** The
occupancy argument was confirmed by compiled-kernel inspection for g17s but
could not be timed there. If the occupancy tax behaves differently on M5, the
gated and ungated forms converge and the candidate loses about one point of its
margin. It does not become incorrect and it does not become slower than base.

### F3. Form must match the base

The same mechanism expressed against two different base revisions differed by
**2.2×** in measured effect. Re-derive an arm on the current base. Never
cherry-pick a diff from an older base.

## Rung 2 — isolated rate harness

Session `e121-rung2b`, 03:35:26–03:37:58Z, `git_head=451adb95`, `git_dirty=0`,
35 blocks per width after discarding 1 warmup block, 4 widths, 5 shapes.

### Arm ranking, round-weighted percent (positive = faster)

| arm | round_weighted | round_weighted_ex_NA5 | CI95 ex-NA5 | ships |
| --- | --- | --- | --- | --- |
| `g_split_pred` | **+1.128** | +1.172 | [+1.042, +2.243] | **yes** |
| `g_min_ask` | +0.481 | +0.499 | [+0.310, +1.761] | fallback |
| `x_split_pred` | +0.117 | +1.178 | [+1.059, +2.245] | no |
| `a_scaffold` (null control) | −0.045 | −0.045 | [−0.080, +0.030] | no |
| `x_min_ask` | −0.814 | +0.426 | [+0.287, +1.917] | no |
| `g_split_pred_pp` (ping-pong) | −0.941 | −0.973 | [−1.061, +0.425] | no |

### Shipped arm per width

| width | per_width_pct | cost_weighted_per_width_pct | round weight |
| --- | --- | --- | --- |
| 2 | +0.687 | +0.785 | 0.024 |
| 3 | +0.463 | +0.660 | 0.275 |
| 4 | +1.482 | +1.927 | 0.667 |
| 5 | −0.118 | −0.098 | 0.034 |

### Headline correction — NA=2 and NA=5 carry zero effect

`affine_qmv_fast` routes `M = 2` to `qmv_fast_crossrow_affine4_g64<T, 2>`, a
separate function (twin line 873) with no `sums_xchg`. The arm never touches it,
and no `_m` instantiation produces `wide<2>`. The live dispatch map is:

```
M=3 -> wide<3>          M=6 -> wide<3>
M=4 -> wide<4>          M=7 -> wide<4> + wide<3>
M=5 -> wide<5> (gated)  M=8 -> wide<4>
                        M=9 -> wide<3>
```

So NA=2's round weight 0.024 carries **zero** effect, and NA=5 is gated off.
Recomputing the shipped frame with both non-participating widths pinned to
exactly 0 gives **+1.467 %**; the cost-weighted round figure the harness reports
for the participating widths is **+1.463 %**. The uncorrected figure was
+1.482 % kernel / +0.900 % leg / +0.855 % ranked. The corrected shipped frame is:

```
harness=local   kernel  +1.463 %
                leg     +0.888 %   (E116 kernel->leg transfer 0.607)
harness=ranked  ranked  +0.844 %   (rule-34 leg->ranked 0.95)
```

### Validity

- `void = false`, `implausible_row_count = 0` against the 1.2 × 273 GB/s DRAM
  bound.
- Null control `a_scaffold` worst per-width magnitude 0.122 %, inside the
  pre-declared ±0.50 % tolerance.
- 5/5 positive controls fired: every shape at `M = 2` reported a detected
  difference under a deliberately broken arm.
- `exactness_failures = []`.
- Thermal: `cool_gate_passed_real_gate = false`,
  `gate_qualified_for_timing = false`, entry 34.14–41.97 °C (spread 7.83 °C),
  exit 37.85–43.82 °C. ABBA-counterbalanced within one session.

Pre-registered bands, written before the session:
`na4_band_if_branch [1.3, 1.6]` (hit: +1.482), `na4_band_if_predicated
[0.0, 0.3]`, `round_weighted_band [1.0, 1.2]` (hit: +1.128),
`ceiling_na4_pct 2.266`, `exchange_cost_na4_pp 0.801`.

Artifacts: `rung2-summary.json`, `rung2-rate.json`, `rung2-meta.txt`.

### Predictions that failed

- **Ping-pong lost cleanly.** `g_split_pred_pp` came in 3.05 pp below
  `g_split_pred`. Alternating ownership across `k` blocks balances the two
  simdgroups but puts ownership arithmetic inside the inner loop, and that costs
  more than the balance buys. Dropped.
- **The spill hypothesis was wrong** (F2 above). The gate survives on a
  different mechanism.
- The advisor's independent prediction for the shipped arm was +0.628 %; the
  measured round-weighted effect is roughly twice that.

## Rung 3 — transplant, exactness, end-to-end

### Transplant

`research/e121_transplant.py` applies 5 source edits plus
`patch_dispatch(text, 1, 4)` to **both**
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h` and
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` (+966 bytes each).
**0 comment lines added to either file** (harness defect 3 / finding 63),
verified by diff grep.

### Witnessed build

Candidate worker `9c73e114325014282748231ac2d6d27132ff931a69438baae29dc782af911535`,
mtime 2026-08-22T03:52:51Z, `--self-test` PASS, all 11 needles OK: the standing
witness set of 8 plus three added for this experiment:

```
--require 'constexpr bool SHARE_SUMS = NA <= 4;'
--require 'threadgroup float sums_xchg[1 * 4 * 32];'
--forbid  'sums[m] += load_vector'
```

`sums[m] += load_vector` occurs 1× on the base twin and 0× on the transplanted
twin, so the needle set separates the two trees. The base arm was built and
witnessed the same way (worker
`c3c72f9d8fa3829e9bf610a35d0b89e5ed710aeb51eb6b0281bf796512c4aaed`) and all
three new needles fired in the opposite direction.

Finding 28 stands: the metallib is dead for the `quantized` family. Every timed
leg is preceded by `senpai/rebuild-and-assert-worker.sh` with explicit
`--require`/`--forbid`, and the worker digest is re-asserted after the leg.

### 512-token exactness — PASS

```
e121x512cand: rows 1025 sha256 719d82b87c79d26a MATCHES PIN
value control: hex digit flipped in row 512 -> c15f45cc MOVED
order control: rows 512/513 swapped         -> 8c5c3f76 MOVED
runtime control e121x128neg: 257 rows       -> 783d3ae1 MOVED
e116_row_digest_check: 0 failure(s)
```

Leg metrics: `all_tokens_matched=true`, `residual_divergence_count=0`,
`public_drift_tripwire_passed=true`, `effective_mean_draft_len=6.359`,
`accepted_draft_rate=0.877`, `mtp_seconds_per_token=0.030577`,
`serial_seconds_per_token=0.073694`, `mtp_decode_speedup=2.4101`.

Post-EOS continuation is carried for the full fixed 512-token window; the run is
not shortened at EOS.

Artifacts: `rung3-exact512-score.json`, `rung3-exact512-meta.txt`,
`row-digest-512.json`.

### Pre-registered end-to-end prediction

Committed at `d56ab817` **before** the ABBA driver was launched, in
`research/e121_e2e_analyse.py`:

```
PREDICTED_KERNEL_PCT   = 1.463
E116_KERNEL_TO_LEG     = 0.607
RANKED_TRANSFER        = 0.95
predicted leg effect   = -0.888 %   (E110 sign convention: negative = faster)
acceptance band        = [-1.36, -0.44]
PROMOTION_BAR_RANKED_PCT = 0.20
```

The analyser also asserts `schedule_invariant`: `effective_mean_draft_len` and
`accepted_draft_rate` must not move by more than 0.05 % between arms. If the
schedule moved, the timing comparison would not be measuring the kernel.

### End-to-end ABBA result — the stop rule fired

Session 04:06:30–04:51:40Z, 8 legs, 2 replicates, order base/share/share/base
per replicate, 512 decode tokens per leg, ungated
(`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`).

The driver was launched with 3 replicates before advisor feedback 3 arrived and
was cancelled after replicate 2, which is a pre-declared truncation rather than
a peek-and-stop: the leg count was fixed before any effect was visible. The
unwind trap survived the cancellation — HEAD restored, transient base commit
removed, no orphaned worker.

| replicate | mtp s/token | serial s/token | local ratio | base-pair drift |
| --- | --- | --- | --- | --- |
| k1 | **−0.502 %** | +0.704 % | +1.214 % | +0.020 % |
| k2 | **−0.371 %** | +0.042 % | +0.414 % | +0.470 % |

```
POOLED absolute candidate MTP s/token  -0.436 %  (sd 0.093, n = 2)
  95 % CI                              [-1.268, +0.395] %
  base 0.030646 -> share 0.030512 s/token
  local ratio 2.39536 -> 2.41481        (+0.814 %)
  schedule invariant                    True (draftlen +0.0000 %, accept +0.0000 %)
  ranked frame                          -0.415 %
  prediction -0.888 %, band [-1.36, -0.44]  -> MISS
  exactness                             PASS (4/4, all 8 legs matched)
```

**Sign convention: negative = faster.**

#### Why this is a halt and not a result

The advisor's stop rule is: halt and report if the leg effect disagrees with
rung 2 by more than a factor of two. It does, by a hair.

```
predicted / measured = 0.888 / 0.436 = 2.04x
```

I stopped. I have not debugged, not changed the arm, and not run the submission
chain.

#### What the data does and does not support

Supported:

- The sign is consistent. Both replicates are negative and the pooled point
  estimate of −0.436 % leg / −0.415 % ranked clears the +0.20 % promotion bar on
  the point estimate alone.
- The measurement is measuring the kernel and nothing else. `draftlen` and
  `accepted_draft_rate` are identical to six decimal places on all eight legs,
  so no schedule change is contaminating the timing.
- Correctness is untouched: 4/4 exactness checks pass and all eight timed legs
  report `all_tokens_matched=true`.

Not supported:

- Any claim of a win. `clears_promotion_bar` is false because the clearance rule
  needs the CI to exclude zero, and with n = 2 the 95 % CI is
  [−1.268, +0.395] %. The CI comfortably contains the predicted −0.888 % as
  well, so this session does **not** falsify rung 2 either. It simply cannot
  separate the two.

The width comes almost entirely from `df = 1`, not from noisy data. The
replicate-to-replicate spread is small (sd 0.093 pp) and `t(0.975, 1) = 12.706`
inflates it by an order of magnitude. If that spread held, `n = 3` would give
`±4.303 × 0.093 / sqrt(3) = ±0.231`, that is [−0.667, −0.205], which excludes
zero.

**But simply appending a third quad now would be optional stopping.** The
truncation to two quads was valid because it was instructed before any effect
was visible. A decision to resume that is taken *because* `n = 2` failed to
clear is data-dependent, so the resulting `n = 3` interval would be optimistic
and could not honestly be quoted as an exact 95 % CI. The clean way to settle it
is a fresh session with a pre-declared leg count, analysed on its own and not
pooled with this one.

#### The thermal asymmetry biases against the candidate

The rebuild-gap defect is not symmetric between arms. In the order
base/share/share/base, positions 2 and 3 are both `share`, and position 3 is the
only position with no rebuild in front of it.

| arm | entry temps (°C) | mean |
| --- | --- | --- |
| base | 40.18, 44.87, 44.78, 44.76 | 43.65 |
| share | 44.81, 53.00, 45.09, 52.95 | **48.96** |

The share arm ran **5.31 °C hotter on average**. Within the share arm, the two
hot legs were 0.072 % slower than the two cool legs. Scaling that crudely gives
a penalty of roughly 0.05 pp carried by the candidate only, so the true effect
is more likely near −0.49 % than −0.436 %.

This does not rescue the interval and it is far too small to close the 2.04×
gap, but the direction matters: the measured effect is, if anything, a slight
**under**statement of the candidate's advantage, not an overstatement. A future
ABBA design should insert a matched idle gap wherever a rebuild does not occur,
so that every position has the same thermal history.

#### Leading hypothesis for the shortfall — stated as a hypothesis, not a finding

The implied kernel-to-leg transfer is

```
0.436 / 1.463 = 0.298   measured
0.607                   E116 constant used for the prediction
```

The mechanism here is an **occupancy** win (F1: entry-point resident
simdgroups), and E116's 0.607 was calibrated on an **ALU/bandwidth** win. Those
two classes should not share a transfer constant. In an isolated rate probe the
kernel is the only work resident, so occupancy sets throughput almost directly.
In a real round the wide QMV dispatches are interleaved with recurrence, norms,
and head work, and the machine can be filled by neighbouring dispatches, so a
recovered simdgroup slot converts to wall clock at well under 1:1.

Thorfinn's Route B supports the contrast: his hoist is a flat 5.85 % of base at
NA = 4 across seven shapes spanning 148–214 GB/s, which is the signature of an
ALU/bandwidth term, and that is the class E116 measured.

This predicts that **any** future occupancy-class arm on this tree will
over-predict at 0.607 by roughly a factor of two, and that the campaign needs a
separate occupancy-class transfer constant near 0.30. That is a falsifiable
claim and it is the most reusable thing this session produced.

I have not tested it. Testing it is not in scope for this assignment.

#### Provenance note on the mid-session commit

Legs 1–3 record `branch_commit=25525156`; legs 4–8 record
`branch_commit=551a0e4c`. I committed the results write-up and tooling between
legs 3 and 4, during a `share` leg, which creates no transient commit and so
cannot orphan one. `git diff 25525156 551a0e4c -- Sources Vendor Package.swift`
is empty: the two commits are byte-identical over the scored surface, no rebuild
was triggered, and every leg measured the same candidate bytes.

Worker digests differ between legs because a Swift/Metal relink is not
byte-reproducible. Consecutive same-arm legs share a digest because the rebuild
is a no-op; each arm switch produces a fresh digest. What the harness asserts is
the needle set and pre-run/post-run digest equality **within** a leg, and that
held on all eight.

#### Thermal record

Entry 40.18–52.99 °C, exit 59.67–61.36 °C. Position 3 always enters hottest
(~53 °C) because positions 2→3 need no rebuild and so get no cooling gap, while
1→2 and 3→4 each sit behind a rebuild. This is a known asymmetry of the ABBA
schedule that position-balancing does not remove, and it is not symmetric
between arms; see the thermal-asymmetry section above. It does not explain the
result: in k1 the hot leg was the slower share leg and in k2 the hot leg was the
faster share leg, so the effect does not track entry temperature.

Artifact: `research/e121-artifacts/rung3-e2e.json`.

## W&B runs

Project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, group
`e121-cross-simdgroup-activation-sum-sharing`.

| run | id | state | contents |
| --- | --- | --- | --- |
| `e121-rung0-census` | `3inupzgh` | finished | static instruments for every arm on both GPU generations, plus the gate-exactness census |
| `e121-rung2-isolated` | `q3oflj3p` | finished | isolated rate sweep, validity gates, cost-weighted shipped frame, pre-registered prediction scores |
| `e121-rung3-insitu` | `qmr3mgl8` | finished | 512-token exactness, the eight ABBA legs, and the pooled leg effect |
| `e121-rung2-isolated` | `m9ykrn93` | **failed** | crashed on a `wandb.Table` column-type error; superseded by `q3oflj3p`, kept only so the run list is honest |

`e121-rung3-presubmit` was not created. The pre-submit chain did not run,
because the stop rule fired first.

The `m9ykrn93` crash was a logging defect, not a measurement defect: a boolean
was written into a numeric table column. Fixed in
`research/e121_wandb_log.py`.

## Fallback if the shipped arm is ever withdrawn

`g_min_ask` — share only the sums that are strictly required rather than
splitting ownership evenly. Round-weighted +0.481 %, predicted ranked
**+0.523 %**, CI95 ex-NA5 [+0.310, +1.761], `clears_submission_bar = true`. Same
bit-exactness-by-construction argument, same gate, smaller exchange.

## Draw-dependent bet

Two published-median draw modes have been observed in this campaign. Against the
crown 3.35922:

| candidate | mode A draw | mode B draw |
| --- | --- | --- |
| xv4 only | 3.3616 (+0.07 %) | 3.3078 (−1.53 %) |
| xv4 + `g_split_pred` | 3.3904 (+0.93 %) | 3.3361 (−0.69 %) |

P(mode A) ≈ 0.67. **This is a bet with a roughly 1-in-3 chance of missing the
crown even though the mechanism is genuinely faster.** Do not read a mode-B
result as a falsification of the kernel effect.

## Interaction warning

Thorfinn's E120 Route B (+2.70 % ranked, bit exact) routes large shapes away
from `_wide` entirely. It **supersedes rather than composes with** this
mechanism: stacking them would double-count the same saved work. Only one of the
two belongs in any single candidate. Do not later try to stack them.

## Harness defects and campaign requirements raised here

- **Harness defect 22.** The wide entry point needs `if (first_m >= NA) return;`
  before the row loop. Without it, an out-of-range `first_m` reads past the row
  bound. Now a campaign requirement.
- **Implied-bandwidth validity gate.** Reject any rate-harness row whose implied
  bandwidth exceeds 1.2 × 273 GB/s before pooling. Now a campaign requirement.
- **Harness defect 7, corrected form.** `MLX_E58_BUFFER_LIMIT_OPS=0` is a
  **census-only** override. It isolates one dispatch per command buffer, which
  is the whole point for a census leg and is fatal for a timed leg, because it
  destroys the command-buffer concurrency a real round has. It must never be set
  on a timed leg, and never on a pre-submit leg. A pre-submit leg that sets an
  override the ranked runner does not set is not rehearsing the ranked
  configuration. `research/e121_presubmit.sh` now carries an explicit `unset`
  plus the reason, rather than a deleted line, so the next reader sees the
  decision instead of an absence. The older assignment text listing this
  variable as required is superseded. Ruling confirmed by the advisor.
- **Transfer constants are class-specific.** The 0.607 kernel-to-leg constant
  was calibrated on an ALU/bandwidth change and over-predicts an occupancy
  change by about 2×. See the rung-3 shortfall hypothesis above.
- **Cross-reference.** Thorfinn found that NA=5 exactness failures come from
  making `K`/`N` compile-time template arguments: full unroll of the
  10-iteration k-loop at `K = 5120` miscompiles at `NA = 5` only. If an NA=5
  exactness failure appears in this family, suspect compile-time `in_vec_size`
  or `#pragma unroll` before suspecting arithmetic.

## Reproduction

```bash
# transplant onto a clean checkout of the experiment base
python3 research/e121_transplant.py

# witnessed build (needles that separate the two trees)
senpai/rebuild-and-assert-worker.sh \
  --require 'qwen35_dual_rms_norm_concat_bf16_v1' \
  --forbid  'qwen35_dual_rms_norm_bf16_v1' \
  --require 'qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>' \
  --forbid  'qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>' \
  --require 'qwen_mtp_draft_selected_affine4_rerank_g64_v1' \
  --require 'qwen_mtp_row_top32_partial' \
  --forbid  'MLX_E85_GATHER_QMM' \
  --require-symbol 'snapshotScheduleSignal' \
  --require 'constexpr bool SHARE_SUMS = NA <= 4;' \
  --require 'threadgroup float sums_xchg[1 * 4 * 32];' \
  --forbid  'sums[m] += load_vector'

# 512-token exactness gate plus three negative controls
research/e121_exact512.sh

# counterbalanced end-to-end timing, order base/share/share/base
research/e121_e2e_abba.sh 2 512 r3 base
python3 research/e121_e2e_analyse.py --replicates 2 --tokens 512 --label r3

# pre-submit chain on the exact tree that would be submitted
research/e121_presubmit.sh 512
```

Do not copy the usage line printed by `rebuild-and-assert-worker.sh`: it forbids
`<T, 6, 3, true>`, which is live.

## Suggested follow-ups (not implemented)

1. **Time the gate on M5.** F2 makes the gate's value architecture-dependent and
   the local host cannot settle it. One gated-versus-ungated pair on the ranked
   host would convert the largest transfer risk into a measurement.
2. **Four-simdgroup ownership split.** With two simdgroups the exchange saves
   half the reduction. A four-way split saves three quarters, at the cost of a
   larger scratch array and one more barrier. The ping-pong failure suggests the
   cost sits in the inner loop rather than the barrier, so this may still pay.
3. **Apply the same ownership split to the `M = 2` cross-row function.** It is a
   separate function today and carries round weight 0.024, so the payoff is
   small, but the mechanism transfers unchanged.
