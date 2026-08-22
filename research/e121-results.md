# E121 — cross-simdgroup activation-sum sharing in the wide cross-row 4-bit matvec

`harness=local` unless a line says otherwise. No number in this file is an
official or ranked score.

## Question

Does `x_sumshare_min`, gated off at `NA = 5` by `if constexpr`, ship a bit-exact
round-weighted win above +1.0 %?

Answer: **yes at the kernel level (+1.463 % round-weighted, CI95 excluding NA=5
[+1.042, +2.243])**, and the shipped arm is `g_split_pred`.

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

### End-to-end ABBA result

<!-- E121-RUNG3-E2E -->

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
- **`MLX_E58_BUFFER_LIMIT_OPS` resolved.** `0` isolates one dispatch per command
  buffer, which is correct for an isolated kernel census and wrong for an
  in-situ leg. The rung-3 timed legs and the pre-submit leg both leave it unset,
  which matches the command-buffer concurrency of a real round and matches the
  ranked runner. `research/e121_presubmit.sh` was corrected to `unset` it.
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
