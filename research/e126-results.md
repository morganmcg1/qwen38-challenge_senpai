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
