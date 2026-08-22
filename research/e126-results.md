# E126 — price Route B on the shipped base, and settle the in-situ transfer

`harness=local` for every measurement and every model in this file. Nothing
here is a ranked score.

- Assignment: PR #127, `qwen38-r1-e126-price-route-b-on-the-shipped-base-and-settle-the-in-situ-transfer`, revision `r1`.
- Base: `senpai/qwen38-mtp-r1` at `3f40d9b03dcaffe0a8be7c86904a676937a0d6e6`.
- Host: this student's AWS Mac. Local architecture `applegpu_g16s`; the ranked
  architecture `applegpu_g17s` is reached by cross-compilation only.

## Rung -1 — official submission of the shipped base

Sent before any file in this experiment changed, so the submitted tree is
byte-identical to `BASE_SHA` `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.

| field | value |
| --- | --- |
| submission id | `cf9a9eda-fdb6-4d94-b090-5451db4ff9ea` |
| status at send | `validating` |
| model attribution | `senpai` |
| note | `senpai/submission-note-e121.md`, 13,510 bytes |
| local `--local-submit` speedup | 2.400662 |
| local candidate s/token | 0.030653386609628797 |
| local serial s/token | 0.07358842762187123 |
| draft length / accept | 6.358974 / 0.877016 |
| exactness | `all_tokens_matched=true`, `residual_divergence=0` |
| cool gate | real gate, waits at 38.7 / 38.8 / 39.3 °C |

Crown at send time: `bc070b7` (francip) 3.35922017047244, sourceRef
`fac135f2`, matching `senpai/frontier-state.json`. No sync was needed.

Two operational notes were raised separately on the PR: `submit-official.sh`
reads `frontier-state.json` from `origin/main` and never compares it with live
Yukon, and `yukon submissions` writes ANSI colour codes around the status field
even when piped, which makes `grep ' promoted '` return zero rows and exit 0.

## Rung 0 — zero GPU

Artifacts: `research/e126-artifacts/rung0-census.json`,
`research/e126-artifacts/rung0-model.json`.
Code: `research/e126_arms.py`, `research/e126_rung0_model.py`.

### The five arms

Every arm is a textual transform of the runtime-effective shipped source
emitted by `e104_variant_sources.emit_base`, so the form matches the base
exactly (E121 finding F3).

| arm | what it does | exact |
| --- | --- | :-: |
| `share_off` | forces `SHARE_SUMS = false`; the pre-E121 shape, and E123's `a_base` | yes |
| `share_on` | the shipped base, unmodified | yes |
| `n_sums_free` | deletes the sums tree AND the `sums * bias_local[r]` consumer term | no |
| `n_nosums_e123` | deletes the tree, keeps a scalar `+ bias_local[r]`; E123's `n_nosums` | no |
| `n_sums_loaded` | deletes the tree, keeps the full consumer, reads `sums` from threadgroup memory once per row per k-block | no |

`n_sums_free` is an **upper bound** on Route B's kernel-side prize: it deletes
consumer work that Route B still has to pay, because Route B's published sums
must still be multiplied by the per-row bias. `n_sums_loaded` is the only
mix-faithful arm. `n_nosums_e123` sits between them and exists so this session
can be compared with E123 directly.

Two defects were found and fixed while building `n_sums_loaded`, and both would
have produced a silently wrong price:

1. The first form addressed `sums_xchg[m * SIMD_SIZE + simd_lid]`, which is
   invariant in `k`. The compiler hoisted all NA loads out of the k loop. A ring
   over two slabs, indexed by `(k / block_size) & 1`, restores the dependence.
2. Threadgroup memory starts undefined, so with no producer the compiler folded
   every read to undef and deleted the loads anyway. The arm now seeds both
   slabs once per threadgroup from a runtime value and fences. That seed is
   `2*NA` stores and one barrier per threadgroup, spread over the 10 k-blocks of
   the 5120-wide cell, so the arm is over-charged by about one fifth of a
   k-block of exchange traffic.

The census caught both. AIR threadgroup traffic for `n_sums_loaded` went from
`0 ld + 0 st / 0 barriers` to `1 ld + 2 st / 1 barrier` after the fix, where the
single load is the rolled per-k-block loop and the two stores and one barrier
are the once-per-threadgroup seed.

### A3 answered: the shipped base does NOT reach 40 resident simdgroups

Shipped entry point `affine_qmv_fast`, every width inlined (Rule 56), registers
and resident simdgroups from `agx_crossarch.translate`:

| arm | `applegpu_g16s` R / sg | `applegpu_g17s` R / sg |
| --- | --- | --- |
| `share_off` | 94 / 32 | 101 / **39** |
| `n_sums_free` | 95 / 32 | 98 / **40** |
| `n_nosums_e123` | 95 / 32 | 99 / **40** |
| `n_sums_loaded` | 95 / 32 | 100 / **39** |
| `share_on` (shipped) | 93 / **33** | 102 / **38** |

**The shipped base sits at 38 resident simdgroups on the ranked architecture,
one BELOW the pre-E121 shape's 39.** E123 estimated that Route B would buy
39 → 40. Against the base that actually ships it buys **38 → 40**, two steps and
+5.3 % residency. The stopping rule the assignment tied to "`share_on` already
reaches 40" therefore does **not** fire.

`share_off` at g17s is R=101 with 39 resident simdgroups, which reproduces
E123's `a_base` row exactly (`g17s | 101 | 0 | 25898 | 39`). The two instruments
agree on the reference shape, so E123's prices can be carried into this model.

The residency effect of the shipped exchange has **opposite sign on the two
architectures**: +1 simdgroup on `g16s`, −1 on `g17s`. Any local-to-M5 transfer
claim about E121 or about Route B has to carry that.

### Task 1 — analytic per-lane, per-k-block census

`block_size = 16 * 32 = 512`, so the 5120-wide cell has 10 k-blocks. `alu`
counts scalar float operations; a `vec<float,NA>` operation is NA of them.
Where the two simdgroups do different work the larger is reported, because the
barrier makes the slower simdgroup set block latency.

| arm | NA3 alu/ld/st/bar | NA4 alu/ld/st/bar | NA5 alu/ld/st/bar |
| --- | --- | --- | --- |
| `share_off` | 96/0/0/0 | 128/0/0/0 | 160/0/0/0 |
| `share_on` | 80/2/2/2 | 96/2/2/2 | 160/0/0/0 |
| `n_sums_free` | 24/0/0/0 | 32/0/0/0 | 40/0/0/0 |
| `n_nosums_e123` | 36/0/0/0 | 48/0/0/0 | 60/0/0/0 |
| `n_sums_loaded` | 50/3/0/0 | 66/4/0/0 | 82/5/0/0 |

NA=2 is unreachable: `affine_qmv_fast` routes M=2 to
`qmv_fast_crossrow_affine4_g64<T,2>`, a separate function with no `sums_xchg`,
and no `_m` instantiation produces `wide<2>`. Its round weight 0.024 carries
zero effect. NA=5 folds the shipped gate off, so `share_on` is byte-identical to
`share_off` there — the census confirms it, both at R=101 / 8452 text bytes.

At odd NA the split is unbalanced. At NA=3, `H = 1`, so simdgroup 0 owns one
row and simdgroup 1 owns two. The critical-path saving is 1/3 of the tree, not
1/2, which explains why E121 measured only +0.463 % at NA=3 against +1.482 % at
NA=4.

**AIR cross-check.** Static `fadd` counts in the rolled loop body are
`share_off` 6, `share_on` 6, `n_sums_free` 1, `n_nosums_e123` 2,
`n_sums_loaded` 2. The `share_off → n_sums_free` delta of 5 is exactly the 3
adds of `xv[0]+xv[1]+xv[2]+xv[3]`, the 1 add of `sums[m] +=`, and the 1 consumer
add, which is what the analytic model removes. `n_nosums_e123` keeps one of
those five. The static counts and the analytic dynamic counts agree.

### Task 2 — predicted cost removed, and the overlap O

Two independent price models, both `harness=local` predictions.

**Model A** uses E123's *measured* whole-tree deletion price
(`a_base → n_nosums`, 0.0047 / 0.0182 / 0.0951 / 0.0891 % per instruction per
k-block at NA 2/3/4/5), which already carries the ~2.16× realisation discount
E123 found between its priced census and its own session, plus E123's
`tgld` 0.4044 for the exchange read.

**Model B** anchors on thorfinn's measured Route B grid (PR #121, ledger 269.6):
per-matvec gain 2.20 % at M=3, **5.88 %** at M=4, 6.52 % at M=5, all on a
`share_off` base.

Percent versus `share_off`:

| arm | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: |
| model A `n_nosums_e123` | 0.925 | **6.448** | 7.551 |
| model A `n_sums_free` | 1.144 | 7.969 | 9.333 |
| model A `n_sums_loaded` | 0.226 | 3.118 | 3.569 |
| model B, thorfinn measured | 2.200 | **5.880** | 6.520 |
| `share_on`, E121 measured | 0.463 | 1.482 | 0.000 |

Model A and model B agree to **10 %** at NA=4 (6.448 against 5.880) and to 16 %
at NA=5, which is a genuine cross-instrument replication on the width that
carries 0.667 of the round weight. They disagree by 2.4× at NA=3, where
thorfinn himself flagged M=3 as a special outlier basis.

**Overlap `O = gain(share_on) / gain(sums_free)`:**

| source | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: |
| model A | 0.405 | 0.186 | 0.000 |
| model B | 0.210 | **0.252** | 0.000 |
| model B, on E121's cost-weighted column | 0.300 | 0.328 | 0.000 |

E121 reports two per-width columns and its own corrected headline used the
cost-weighted one. The two disagree by about 30 % at NA=4, and the E126 primary
metric is a difference against that arm, so both are carried.

**Predicted primary, `gain(n_sums_free vs share_on)`:**

| source | NA3 | NA4 | NA5 | round weighted |
| --- | ---: | ---: | ---: | ---: |
| model A | 0.684 | 6.585 | 9.333 | 4.898 |
| model B | 1.745 | **4.464** | 6.520 | 3.679 |
| model B, cost-weighted `share_on` | 1.550 | 4.031 | 6.520 | 3.336 |
| Route B faithful (`n_sums_loaded`) | −0.238 | 1.661 | 3.569 | 1.164 |

### Rung 1 pre-registration

Written before the session. `harness=local`.

| quantity | point | interval | band |
| --- | ---: | --- | --- |
| `gain(n_sums_free vs share_off)` at NA=4 | 6.4 % | [4.5, 8.5] | replication of thorfinn's 5.88 %; a >2× disagreement stops the rung |
| **primary `gain(n_sums_free vs share_on)` at NA=4** | **4.7 %** | **[4.0, 6.6]** | most likely the 4.0–5.0 band |
| `gain(n_nosums_e123 vs share_off)` at NA=4 | 6.4 % | [5.0, 7.5] | E123 replication |
| `gain(n_sums_loaded vs share_on)` at NA=4 | 1.7 % | [0.5, 3.5] | the honest Route B QMV price |
| overlap `O` at NA=4 | 0.25 | [0.15, 0.35] | |
| primary, round weighted | 3.7 % | [3.0, 5.0] | |

The single largest uncertainty is **which arm thorfinn's 5.88 % actually
corresponds to**. Route B's QMV must still apply the published sums, so its
faithful shape is `n_sums_loaded`, and model A prices that at only 3.1 %
against `share_off` — about half of what thorfinn measured. Either model A
over-charges the threadgroup read, or thorfinn's replica kernel absorbs the
`sums * bias` product as well, in which case `n_sums_free` is the faithful arm
and the upper bound is the real number. The three arms bracket the answer, and
this rung settles it.

### Task 4 — thorfinn's rung 5e on an E121-containing base

Chain: wide-QMV round % → leg % at 0.6068 → ranked % at 0.95 (rule 34).
Route B pays one replica dispatch whose absolute cost does not change when E121
is present, so it is subtracted after the marginal gain.

| width | marginal gain % | replica cost pp | marginal net % |
| ---: | ---: | ---: | ---: |
| NA3 | 1.745 | 2.230 | −0.485 |
| NA4 | 4.464 | 2.420 | +2.044 |
| NA5 | 6.520 | 2.120 | +4.400 |

| base | round net % | leg % | ranked % |
| --- | ---: | ---: | ---: |
| `share_off` (thorfinn's own base) | 2.449 | 1.486 | 1.412 |
| E121-containing (the shipped base) | **1.380** | **0.837** | 0.795 |
| E121-containing, fitted φ(3)=3.63 | 1.775 | 1.077 | 1.023 |

**Pre-registered prediction for thorfinn's rung 5e end-to-end leg effect on an
E121-containing base: +0.84 %, interval [+0.45, +1.15] %.**

Assumptions, all falsifiable:

1. Overlap `O(4) = 0.252`, from E121's measured +1.482 % against thorfinn's
   measured 5.88 %.
2. The replica dispatch costs the same absolute time with E121 present, which
   is 2.30 pp of the round when round-weighted.
3. Kernel → leg transfer 0.6068, leg → ranked 0.95.
4. Thorfinn's 5.88 % basis is `n_sums_free`-like, not `n_sums_loaded`-like. If
   rung 1 shows the faithful arm is the right basis, the whole prediction drops
   towards zero and Route B is net-negative at NA=3.

**E121 removes about 44 % of Route B's leg value.** That is the headline of
this rung, and it is the number thorfinn needs before spending a 5e session.

## Rung 1 — isolated, GPU, five arms

Session `research/out/e126-rung1/`, leg commit `10a52275`, worktree clean.
Command:

```
research/e126_probe.sh e126-rung1 --shapes 0,1,2,3,4 --widths 3,4,5
python3 research/e126_analysis.py research/out/e126-rung1/rate.json \
  --model research/e126-artifacts/rung0-model.json \
  --census research/e126-artifacts/rung0-census.json \
  --out research/e126-artifacts/rung1-summary.json
```

Five arms, five scored shapes, widths 3/4/5, palindrome order, four blocks with
block 0 discarded as warmup, so every cell is a pooled median of three blocks.
`harness=local`. `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false`, preserved verbatim. Nothing here is a score.

### Validity

`void=false`, no reasons. 60 exactness checks, zero failures: both exact arms
are bit-identical at every cell and the three `:diag` arms differ, as they must.
Five positive controls fired, so the comparison can detect a difference.

Entry temperature 33.26 to 36.68 C, spread 3.41 / 2.43 / 2.08 C at NA 3/4/5,
exit maximum 38.06 C. Drift is small and the palindrome cancels its monotone
part.

### Headline

Percent faster than `share_off`, the pre-E121 body:

| arm | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: |
| `n_sums_free` | 5.416 | **9.279** | 11.234 |
| `n_nosums_e123` | 1.203 | 7.297 | 8.898 |
| `n_sums_loaded` | 0.772 | 5.311 | 6.843 |
| `share_on` (E121) | 0.304 | 1.662 | 0.144 |

Percent faster than `share_on`, the shipped body. The first row is the primary
metric:

| arm | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: |
| `n_sums_free` (**primary**) | 5.026 | **7.706** | 11.059 |
| `n_nosums_e123` | 0.914 | 5.744 | 8.746 |
| `n_sums_loaded` (faithful) | 0.410 | 3.703 | 6.706 |

Primary round-weighted **6.898 %**, CI95 [6.514, 7.226], weight coverage 1.000.
Overlap `O` = 0.056 / 0.179 / 0.000 at NA 3/4/5.

**The primary lands at 7.706 % at NA=4, inside the advisor's `>= 5.0` band: the
overlap is small and Route B keeps its price.** Every stopping rule was checked
and none fires.

### The crux: which arm is thorfinn's basis

Rung 0 could not decide whether thorfinn's 5.88 % came from a `sums_free`-shaped
body or from the mix-faithful body that still applies the published sums. The
measurement decides it:

| arm | measured vs `share_off` at NA=4 | thorfinn 5.88 | ratio |
| --- | ---: | ---: | ---: |
| `n_sums_free` | 9.279 | 5.88 | 1.58 |
| `n_sums_loaded` (faithful) | 5.311 | 5.88 | **1.11** |

The faithful arm reproduces him to 11 %. The upper-bound arm does not. So his
basis is the faithful one, and the rung-0 worry that the upper bound was being
priced is resolved in his favour.

The cross-instrument replication test uses `n_sums_free` as written, and gives
ratios 2.46 / 1.58 / 1.72 at NA 3/4/5. Only NA=3 exceeds 2x, and NA=3 is the
one width where the isolated arms differ in residency, so that cell is
occupancy confounded and is not evidence that either mechanism was misread.

### Task 4 revised — thorfinn's rung 5e re-priced on the shipped base

His 5e ran against a pre-E121 control and measured +4.249 % leg. The question
this rung answers is what the same arm would read against the base that now
ships. Scale his per-width gross gain by the measured surviving fraction
`gain(loaded vs share_on) / gain(loaded vs share_off)`, then subtract his
replica-dispatch cost, whose absolute value does not change when E121 is
present:

| width | thorfinn gross | surviving fraction | marginal | replica cost | net |
| ---: | ---: | ---: | ---: | ---: | ---: |
| NA3 | +2.20 | 0.531 | +1.17 | 2.23 | **-1.06** |
| NA4 | +5.88 | 0.697 | +4.10 | 2.42 | +1.68 |
| NA5 | +6.52 | 0.980 | +6.39 | 2.12 | +4.27 |

| base | round net | leg | ranked |
| --- | ---: | ---: | ---: |
| `share_off`, his own control | +2.449 % | +1.487 % | +1.412 % |
| shipped, with E121 | **+0.973 %** | **+0.591 %** | +0.561 % |

**E121 removes 60.3 % of Route B's leg value**, against the 44 % predicted at
rung 0. The revised leg point of +0.59 % still falls inside the pre-registered
interval [+0.45, +1.15].

The leg conversion used here is 0.6068 at the standing NA weights. F95 says that
coefficient is width dependent, and inverting 5e implies 0.796 at mean width
7.359. Both figures are stated with their width, and neither is used unlabelled.

### The ranked residency picture inverts against the shipped base

Registers / resident simdgroups for the isolated per-width kernels:

| arm | g16s NA3 | NA4 | NA5 | g17s NA3 | NA4 | NA5 |
| --- | --- | --- | --- | --- | --- | --- |
| `share_off` | 82/37 | 94/32 | 93/33 | 89/**44** | 90/**44** | 101/39 |
| `n_sums_free` | 78/39 | 94/32 | 95/32 | 98/40 | 94/**42** | 98/40 |
| `n_nosums_e123` | 86/35 | 94/32 | 95/32 | 98/40 | 96/**41** | 99/40 |
| `n_sums_loaded` | 87/35 | 94/32 | 95/32 | 93/42 | 94/**42** | 100/39 |
| `share_on` (shipped) | 79/38 | 90/34 | 93/33 | 95/41 | 102/**38** | 101/39 |

Askeladd reports that deleting the whole add tree costs 3 of 44 resident
simdgroups at NA=4 on g17s. My `n_nosums_e123` reads 41 against `share_off`'s
44, which is his number exactly, from an independent emitter built on the
current base. The finding replicates.

**His conclusion does not carry to the base that ships.** Against `share_on` at
38, every sums-deleting arm *gains* residency at NA=4 on g17s: 42, 41, 42. The
ranked-unfavourable reading is true only against a pre-E121 body that no longer
exists. This agrees with the rung-0 entry-point census, where the shipped entry
sits at 38 and the sums-free entry reaches 40.

E121 itself costs 12 registers and 6 of 44 resident simdgroups at NA=4 on g17s,
while on g16s it saves 4 registers and gains 2 simdgroups. The exchange's
residency effect has opposite sign on the two architectures at the scored
width, and the ranked side is the losing one. That is a falsifiable prediction
about `cf9a9eda`: if E121's ranked gain comes in under its local gain, this is
the first place to look.

A register or residency census is a cost observation, never correctness
evidence.

### Bandwidth covariate

Slopes are fitted within width across the five shapes, which is where achieved
rate actually varies. Within-cell slopes are a noise floor at n=3 and are not
used.

| contrast | NA3 | NA4 | NA5 |
| --- | ---: | ---: | ---: |
| `free vs off`, %/(GB/s) | -0.0228 (r -0.76) | -0.0240 (r -0.66) | -0.0032 (r -0.28) |
| `free vs on`, %/(GB/s) | **+0.0305** (r +0.88) | **+0.0450** (r +0.61) | -0.0055 (r -0.60) |
| `on vs off`, %/(GB/s) | -0.0499 (r -0.96) | **-0.0755** (r -0.90) | +0.0078 (r +0.65) |

Two readings, both for the transfer question:

1. Route B against the pre-E121 body is close to bandwidth independent, which
   corroborates thorfinn's flat result over his 49 cells and contradicts the
   strong negative slope I reported from 10 cells in E121.
2. The two mechanisms separate cleanly. E121's own effect *shrinks* as achieved
   bandwidth rises, and Route B's remaining prize *grows*. The ranked M5 streams
   at 542.8 GB/s against the 145 to 236 GB/s measured here, so the sign is a
   transfer risk for E121 and a transfer opportunity for Route B. The measured
   span does not reach the ranked rate, so no numerical extrapolation is made.

### Harness defect 30 does not bias this session

The block is ten slots in palindrome order, so `share_off` holds the two extreme
slots and `share_on` the two middle ones: exactly the shape askeladd reports.
Two in-session bounds:

- Palindrome asymmetry `slot[9-i] - slot[i]` cancels the arm effect and exposes
  monotone drift. The largest median over all arms and widths is 0.214 pp.
- The convex part is confounded with the arm effect at every width except NA=5,
  where the shipped gate folds off and `share_on` is byte-identical to
  `share_off` while sitting in the opposite slot pair. Extreme minus middle
  reads **-0.144 pp**.

Both bounds are far below the 7.706 pp primary effect, so `--warm-sweep-reps`
is not needed for this arm set at this window. Discarding block 0 removes the
ramp: in block 0 slot 0 runs 39 % slow, and by block 1 the asymmetry is gone.

### Occupancy confounding, per contrast cell

17 of the contrast cells differ in isolated residency on the timed g16s. The
important reading is the one that carries the round weight:

- At NA=4 the three diagnostic arms and `share_off` are all 94 registers / 32
  simdgroups, so every `vs share_off` number at NA=4 is occupancy clean.
- At NA=4 `share_on` sits at 34 simdgroups against the diagnostic arms' 32, so
  the primary 7.706 % is if anything understated and the overlap `O(4) = 0.179`
  is an over-estimate.
- NA=3 is badly confounded. `n_sums_free` reads 5.416 % against
  `n_nosums_e123`'s 1.203 % for only 12 deleted adds per lane, and the arms
  stand at 39 against 35 simdgroups. The NA=3 cells price occupancy, not
  instructions.
- Rule 56 means none of these per-width cliffs reaches the shipped kernel,
  which inlines every width into one allocation.

### Pre-registered bands, scored

| prediction | point | band | measured | verdict |
| --- | ---: | --- | ---: | --- |
| `free_vs_off` NA=4 | 6.40 | [4.50, 8.50] | 9.279 | MISS high |
| **primary `free_vs_on`** NA=4 | 4.70 | [4.00, 6.60] | **7.706** | MISS high |
| `e123_vs_off` NA=4 | 6.40 | [5.00, 7.50] | 7.297 | in band |
| `loaded_vs_on` NA=4 | 1.70 | [0.50, 3.50] | 3.703 | MISS high |
| overlap `O` NA=4 | 0.25 | [0.15, 0.35] | 0.179 | in band |
| primary, round weighted | 3.70 | [3.00, 5.00] | 6.898 | MISS high |

Four of six miss, and every miss is in the same direction: the rung-0 model
under-predicted how much a deletion buys. The two predictions that hold are the
ones anchored on E123's measured deletion price rather than on the analytic
instruction count, so the analytic count is the part that is wrong, not the
E123 price ladder.

The primary missing high is a favourable miss, but it is still a miss, and the
honest reading is that the model is not yet good enough to price a mechanism
without measuring it.

## Mode index tool

`research/e126_modeindex.py` carries the F76 weight vector relayed on PR #127
and reads a ranked receipt mode corrected. `--selftest` reproduces every anchor
the advisor published, checks the slow correction against `7bef7d4c`, and runs a
positive control that moves the index by 8.77 units, more than the 1.000 unit
mode flip.

One defect in the relayed vector: the eight weights sum to `-1.0e-4`, not to
zero, because they are published to four decimal places. A uniform 5 % speedup
therefore leaks 0.000513 index units instead of cancelling exactly. That is
0.63 % of the 0.0817 per-run noise, so it changes no classification, but the
selftest bounds it rather than asserting exact cancellation.

