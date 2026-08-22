# E121 — cross-simdgroup activation-sum sharing in the wide cross-row 4-bit matvec

`harness=local` unless a line says otherwise. No number in this file is an
official or ranked score.

## Question

Does `x_sumshare_min`, gated off at `NA = 5` by `if constexpr`, ship a bit-exact
round-weighted win above +1.0 %?

Answer: **yes at the kernel level, not yet demonstrated end to end.**

Final label: **local winner, not submitted, superseded by Route B.**

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

The advisor closed the experiment without submitting. The submission slot is
worth more to E120 Route B, whose point estimate clears the crown by 1.9 % even
on a slow draw, than to this arm, whose in-situ point estimate lands 0.12 %
*under* the crown. Two independent corrections put our board row `b8b8b860` at a
true 3.3414 rather than 3.2979, so parity with the crown now needs +0.53 %
ranked and this arm measured +0.415 %. The two arms do not compose: Route B
supersedes E121 and they must never be stacked.

After the halt this assignment spent **no GPU time**. The three remaining tasks
were the matched-idle-gap fix, the NA re-weighting, and this report.

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
**under**statement of the candidate's advantage, not an overstatement.

##### Root cause, measured

`swift build` decides what to compile from file stat, so a leg only pays a
rebuild when its sources look newer than the last build. Reading `started` and
`finished` out of the eight leg metadata files gives the prep time in front of
each timed run:

| position | arm | prep before the timed run | entry °C |
| --- | --- | --- | --- |
| 1 | base | 53 s | 40.18, 44.78 |
| 2 | share | 50 s | 44.81, 45.09 |
| 3 | share | **11 s** | **53.00, 52.95** |
| 4 | base | 52 s | 44.87, 44.76 |

Position 3 is the only position whose arm repeats its predecessor's, so it is
the only one that finds both sources untouched and skips the compile. It reached
its timed run after 11 s instead of 50 s and entered 8 °C hotter. A second term
runs the same way: only a rebuilt worker pays the first-use Metal JIT compile,
which is worth another ~48 s of leg wall time (266 s at position 3 against
313 s elsewhere).

Counterbalancing cannot remove this. ABBA removes confounds that are a function
of **position**; under this order the short position is always `share`, so the
confound is a function of the **arm**.

##### Fix, implemented in this branch

`research/e121_e2e_leg.sh` now `touch`es both sources before the build, so every
leg performs the same compile and the same JIT. That reproduces the thermal
history rather than approximating it — an equal-length idle sleep would cool
*further*, because it omits the CPU heat the compile puts into the package.
`E121_PREP_FLOOR_SECONDS`, set to 60 s by the driver, pads any residual, and
each leg records `e121_prep_build_seconds`, `e121_prep_idle_seconds` and
`e121_prep_seconds`.

`research/e121_e2e_abba.sh` then prints a per-arm balance report, because a
matched design has to be checked rather than asserted. Run against the eight
legs of the 04:06Z session it correctly reports the imbalance it was written to
prevent:

```
arm     n   entry_C mean  entry_C range
base    4   43.65         40.18-44.87
share   4   48.96         44.81-53.00
share minus base: entry +5.31 C
IMBALANCED: the arms did not start from the same thermal state
```

That is the positive control: the reporter fires on the known-bad session. It
reports and does not fail the session, because by the time it runs the legs are
already measured and the analysis needs the number to price the imbalance.

Neither change has been exercised on a fresh session. No GPU time was spent on
this assignment after the halt.

#### Why the prediction was 2x too large — a roofline knee in the probe

**Retracted.** My first explanation was that this is an *occupancy-class* arm
and that occupancy wins need their own kernel-to-leg constant near 0.30. The
advisor rejected it and was right. Occupancy is what this arm **pays**, not what
it wins: the ungated form costs ranked occupancy 39 → 32 simdgroups (about
15 %), the gate exists to hold that loss to 39 → 38 (about 2.6 %), and the gain
is arithmetic — a deleted duplicate chunk sum. An arm cannot be in the
occupancy-win class when occupancy is on its cost side.

Re-analysis, zero GPU, from the rung-2 cells already in
`research/e121-artifacts/rung2-rate.json`. Reproduce with
`research/e121_na_reweight.py`; output in `rung3-na-reweight.json`.

**First, the weighting the advisor asked me to check.** Two weights turn the
isolated cells into one leg number, and only the width weight had ever been
looked at. Across widths the analysis already used the realised NA histogram
`{2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}`. Within a width it used each shape's
share of round cost. Re-combining the same cells under each within-width rule,
all in the shipped frame that pins NA = 2 and NA = 5 to zero:

| within-width rule | NA3 | NA4 | kernel % | leg % | vs measured |
| --- | --- | --- | --- | --- | --- |
| shape cost (shipped prediction) | +0.660 | +1.927 | +1.467 | **−0.890** | 2.04x |
| pooled median | +0.463 | +1.482 | +1.116 | **−0.677** | 1.55x |
| uniform mean | +0.762 | +2.093 | +1.606 | −0.975 | 2.24x |
| measured in situ | | | | **−0.436** | 1.00x |

So the answer to the direct question is **it moves, but only part of the way**:
re-weighting closes 0.213 pp of the 0.454 pp shortfall, or 47 %, and cuts the
over-prediction from 2.04x to 1.55x. A 2.6 sd gap remains. The shortfall is
therefore *partly* a weighting error and not *only* a weighting error.

Note also that shape-uniform is **not** the more principled weight. A leg is a
sum over dispatches, so cost weighting is what `program.md` asks for. Dropping
it is not a correction; it is a different arbitrary choice that happens to sit
closer to the measurement.

**Second, what the three rules are actually disagreeing about.** They disagree
because the five cells are bimodal rather than noisy, and they are ordered by
how close each shape runs to the memory roofline. Achieved rate is weight bytes
over base time, at 0.5 B per 4-bit weight plus fp16 scale and bias per group of
64, against the M4 Pro 273 GB/s peak:

| shape | k | n | GB/s | % peak | NA3 | NA4 |
| --- | --- | --- | --- | --- | --- | --- |
| gdn_out_proj | 6144 | 5120 | 162 | 59 | +1.676 | +3.333 |
| mlp_down | 17408 | 5120 | 169 | 62 | +1.389 | +3.399 |
| fa_qkv | 5120 | 14336 | 191 | 70 | +0.427 | +1.391 |
| gdn_in_proj | 5120 | 16480 | 193 | 71 | +0.025 | +1.188 |
| mlp_gate_up | 5120 | 34816 | 203 | 74 | +0.294 | +1.155 |

Pearson r between achieved GB/s and percent gain is **−0.938** over the ten
live cells. The two shapes furthest from the roofline gain 2.7x what the three
nearest it gain, with **no overlap at either width**. That is the expected
signature of deleting arithmetic: it pays where issue, not bandwidth, is the
limit, and it is hidden where the kernel is already streaming near peak.

**Third, the counterfactual that closes the gap.** My isolated probe chains 32
dependent dispatches of one kernel, which is the most issue-bound configuration
on offer. A real decode round interleaves these dispatches with recurrence,
norms and head work, so the same kernels should sit nearer the roofline. Price
every live width at the near-roofline stratum mean instead of its own:

```
near-roofline means      NA3 +0.249   NA4 +1.245
shipped-frame kernel     +0.899  ->  leg -0.545
measured                             leg -0.436  (sd 0.093)
```

That lands **1.2 sd** from the measurement. The shipped prediction is 4.9 sd
away. One assumption — that in situ every shape behaves like the shapes already
near the roofline — takes the prediction from decisively wrong to consistent.

**The reusable rule.** It is not a new transfer constant. It is that a scalar
kernel-to-leg constant is the wrong object when per-cell gains span 2.7x and are
ordered by operating point. Report the achieved bandwidth of every probe cell,
and transfer a gain to a leg only across cells at the same distance from the
roofline. This applies to every isolated probe on this tree, including Route
B's, and it is testable without new hardware.

Route B is the useful contrast and the reason I flag this rather than assert it:
Thorfinn measures a flat 5.85 % of base at NA = 4 across seven shapes spanning
148–214 GB/s. Flat across that range is exactly what my fit says an
arithmetic-deletion gain should **not** be. Either Route B's mechanism is not
bandwidth-sensitive in the way mine is, or one of the two harnesses is not
measuring what it reports. Resolving that disagreement is worth more than either
number alone, and I have not resolved it.

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
| `e121-rung3-reweight` | `5zms9ntd` | finished | post-hoc re-analysis of the rung-2 cells: three within-width weighting rules, and achieved bandwidth against gain for every probe cell. No new measurement, no GPU time |
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

# post-hoc, zero GPU: re-weighting and the roofline fit
python3 research/e121_na_reweight.py
python3 research/e121_wandb_log.py --only reweight
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
4. **Settle the roofline-knee rule against Route B, zero new mechanism.** My ten
   cells put achieved bandwidth against gain at r = −0.938; Thorfinn reports a
   flat 5.85 % across 148–214 GB/s. Both cannot describe the same machine.
   Re-analysing the two existing probe datasets under one bandwidth-stratified
   estimator needs no GPU and would either give the campaign a validated
   correction rule for every isolated probe or identify which harness is wrong.
   This is the cheapest high-value item on the list.
5. **Measure the achieved bandwidth of these five shapes in situ.** The
   counterfactual that reconciles my prediction with my measurement assumes the
   whole-model pipeline moves every shape toward the roofline. That assumption
   is currently untested and one instrumented decode round would test it.
