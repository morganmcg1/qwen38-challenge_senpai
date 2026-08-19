# E45 r2 — Weight-stream cost from the ranked board, and the board's own noise floor

Assignment: PR #50, `qwen38-r1-e45-plateau-pooled-family-separation`, revision `r2`.
Base: `378ecc31c684e2d347e5eaff6b9350418fe0d41d` (= `origin/senpai/qwen38-mtp-r1`).
Analysis provenance commit recorded in the JSON: `67d7c4ae8a421079b715a62bfaea8c62fb37ba01`.
Host: Apple M4 Pro. The analysis is arithmetic over published board metrics; the
one compiler readout in §5 is noted separately. Zero GPU decode work.

Reproduce:

```bash
python3 research/e45_stream_ab.py --self-test      # 34/34
python3 research/e45_stream_ab.py --run           # writes research/e45-stream-ab.json
bash    research/e45_stream_register_probe.sh     # §5, compile-only
```

---

## 0. Headline

**The ranked board's replication noise floor is 0.7353 % — 7.5× the campaign's
recorded `SIGMA_SCORE_PCT` of 0.0978 % — and it is large enough to swallow both
the crown gap and the entire weight-stream effect this experiment set out to
measure.**

The marginal cost of one weight stream at width 8 is **+0.4910 % ± 0.3315 % (se)
of the candidate leg, t = 1.48, 4 of 7 independent groups positive**. That is not
significant. The width-4 contrast (+0.4700 % ± 0.2413 %, t = 1.95) is also null
once you notice that 24 of its 25 pairs live in a single fingerprint group.

**Why this number was wanted, and what it is not.** `delta_T(8)` does not unlock a
lever that exists today: §4 re-derives from the template's own constraints that
the shipped dispatch table is already **stream-minimal at all seven widths** under
the `NA <= 4` bound, so there is no stream left to remove. Its value is as a
*calibration input* for the decision E46 (#51) and E44 (#49) are actually facing —
whether to raise the accumulator bound to `NA = 5`, which is the only change that
would let width 5 (2→1 streams) and width 9 (3→2) each drop one. This experiment's
job was to price that stream before anyone spends a ranked submission on it. The
answer is that **the board cannot price it**: the effect and the noise are the same
size.

Consequences that outlast the null:

- E27's observed `NA = 5` cost was **−0.3321 %**. The stream benefit is
  **+0.4910 % ± 0.3315 %**. Both sit inside a 0.7353 % replication sd, so **a
  single ranked submission cannot resolve either sign.** Anyone planning to
  adjudicate the `NA = 5` bound with one submission should expect a coin flip.
- The crown gap `e43.CROWN_GAP_PCT = 0.5193 %` is **0.71 sd of pure replication
  noise**. Chasing it as a real deficit is chasing measurement error.
- Three instrument defects in `stream_dispatch_census` / `e45_stream_ab`, one of
  them a fail-open that produces *substantive-looking* findings (§6).

---

## 1. What was measured, and the one thing that makes it possible

The ranked board publishes, per submission, an `officialMetrics` block with
per-prompt candidate and serial seconds/token, plus the submitted source. Two
submissions whose git trees differ **only** in the crossrow width→IPG dispatch
table are running identical work with a different number of weight streams at the
affected widths. Their score difference prices a stream — *if* nothing else
differs.

`research/e45_stream_ab.py` builds that comparison in four stages:

1. Census every board tree's dispatch table (`stream_dispatch_census`).
2. Group trees by a **non-kernel fingerprint**: a hash of the tree with both
   kernel files excluded, so trees that agree everywhere except the kernels land
   in one group.
3. Within a group, form pairs and keep only those whose entire kernel diff is
   confined to `qmv_fast_crossrow_affine4_g64_m<...>` dispatch cells and the
   surrounding switch scaffolding (`classify_kernel_diff`, see §6.3).
4. Difference the per-prompt candidate leg, cluster by group.

### Instrument header, quoted verbatim

```
trees scanned                      : 663
trees with a dispatch table        : 486
distinct non-kernel fingerprints   : 377
fingerprints with >1 table (A/Bs)  : 10
```

**This does not match the 653 / 476 / 370 / 10 in the assignment.** It is not the
failure the guard was built for, and it is not being silently ignored:

- +10 trees, and **all 10 carry a dispatch table** (663−653 = 486−476 = 10).
- +7 fingerprints; the remaining 3 joined existing groups.
- **The A/B group count is unchanged at 10**, and all three groups the assignment
  names by hand have exactly the stated composition.

Diagnosis: ten new submissions landed between the assignment's snapshot and this
run. Benign board growth. The reason it needs saying out loud is §6.1 — the same
instrument reports a superficially similar header, with *no error*, when nothing
has been fetched at all.

Provenance: 656 board rows total, **433 with `officialMetrics`**, over **331
distinct trees**.

---

## 2. The replication floor (the result that matters)

Thirteen sets of **byte-identical git trees** — grouped by
`git rev-parse <ref>^{tree}`, so identity is the whole tree rather than a
fingerprint — were submitted by *different solvers* and measured separately.
31 rows. Dedup is by tree, not by row.

| tree | rows | rel sd | rel range | mean score | solvers |
|---|---|---|---|---|---|
| `5bef2ca86b89` | 2 | 1.4369 % | 2.0321 % | 3.114778 | ivanfioravanti, EternaPeptix |
| `715b1c7576a3` | 4 | 0.7428 % | 1.6066 % | 3.217165 | Kamciosz, claudiodekker, scarletbright, xadenryan |
| `6aad7882e169` | 3 | 0.8381 % | 1.4678 % | 3.215191 | paul-hf, Lieisyourlie, bingcheng45 |
| `4cb10f948da1` | 3 | 0.6717 % | 1.3073 % | 2.887909 | DawgZter, machalabs-ai, Amal-David |
| `46f7b00e5e03` | 3 | 0.7285 % | 1.2914 % | 3.128913 | DashiellB, newjordan, mega-dmitriy |
| `3bdf0c0acf2c` | 2 | 0.8968 % | 1.2682 % | 3.096852 | ratacat, welttowelt |
| `888809efd297` | 2 | 0.7917 % | 1.1197 % | 3.166905 | Floofy6, Kamciosz |
| `2804e32b1fc3` | 2 | 0.7888 % | 1.1156 % | 2.859699 | machalabs-ai, ggu77wt |
| `ff2a732f96f4` | 2 | 0.6221 % | 0.8798 % | 3.219934 | vibecodooor, Lieisyourlie |
| `2d65604a66d4` | 2 | 0.4331 % | 0.6125 % | 3.239373 | ofou, jonathan308 |
| `d5a8bcf98d92` | 2 | 0.1099 % | 0.1555 % | 2.927483 | audreyt, ivanfioravanti |
| `dc6c614e79f2` | 2 | 0.0560 % | 0.0792 % | 2.899467 | Hcoder10, Amal-David |
| `b8642b81f72f` | 2 | 0.0057 % | 0.0081 % | 3.243131 | alfranli123, companygardener |

```
pooled within-tree score rel sd : 0.7353 %
median per-set rel sd           : 0.7285 %
widest per-set rel range        : 2.0321 %
e43.SIGMA_SCORE_PCT (recorded)  : 0.0978 %   -> understated 7.5x
e43.CROWN_GAP_PCT               : 0.5193 %   -> 0.71 sd of replication noise
```

`2d65604a66d4` contains the board crown itself (ofou, 3.24929398547457) and a
0.6125 % lower twin. `b8642b81f72f` is the r1 pool tree.

### Nothing but noise is left to explain it

**13 of 13 sets are covariate-invariant.** Within each set, every row agrees on
`head_provenance_sha256` (all 8 prompts), `qwen_mtp_weights_hash`,
`effective_mean_draft_len`, `non_drafting_round_count`, and the full scoring
policy. **Only the commit differs.** There is no behavioural covariate left that
could produce the spread — it is measurement noise plus whatever the harness does
not pin (host instance, thermal history, power state).

### The apparent bimodality is not real

`b8642b81f72f` (range 0.0081 %) next to `5bef2ca86b89` (range 2.0321 %) invites a
two-population story. It is sampling variation of `|difference|` at n = 2: the
expected range for n = 2 at sd 0.75 % is ≈ 0.85 %, and both observations are
ordinary draws from one distribution. Nine of the thirteen sets are n = 2, which is
why the *pooled* figure is the one to quote and the per-set ranges are not.

### The noise is differential, not common-mode host drift

If the spread were the host being globally fast or slow, the pinned serial leg
would absorb it and the ratio would come out clean. It does not:

```
pooled score rel sd                : 0.7353 %   (13 sets, 31 rows)
pooled candidate-leg (mtp) rel sd  : 0.7875 %   (per-prompt, dof 144)
pooled serial-leg rel sd           : 0.2063 %   (per-prompt, dof 144)
pooled per-prompt ratio rel sd     : 0.7647 %
ratio sd / candidate-leg sd        : 0.9711
```

The ratio inherits **97 %** of the candidate-leg noise while the serial leg is
almost four times quieter. **Pinned-serial normalisation does not protect the score
from this** — the noise is differential, not a common-mode host factor that
divides out.

### Per-prompt, and a physical check

Candidate-leg replication rel sd by prompt, with drafting intensity `rho`:

| prompt | rho | replication rel sd |
|---|---|---|
| `3b10cb4d` | 2.6557 | 1.1428 % |
| `4b9e88cd` | 2.2976 | 0.8887 % |
| `919318e1` | 4.5327 | 0.8430 % |
| `192fb621` | 5.7765 | 0.8324 % |
| `a2ea8b60` | 5.4253 | 0.7626 % |
| `00142a44` | 4.7677 | 0.6924 % |
| `ea82dcb5` | 5.2697 | 0.6302 % |
| **`c1ec5866`** | **0.1540** | **0.0615 %** |

The one prompt that barely drafts is **13.5× quieter than the median drafting
prompt** (0.0615 % vs 0.8324 %), and 10.2× quieter than the quietest of them. The
noise lives in drafting rounds. That is physically coherent and it is what makes
`c1ec5866` usable as a negative control in §3.

`c1ec5866` is also the sole exception to the differential-noise finding: its
candidate leg (0.0615 %) is *quieter than its own serial leg* (0.2020 %), giving
ratio/mtp = **3.6973** against 0.9711 pooled. With almost no drafting work to be
noisy about, its candidate leg becomes the stable one and the serial leg dominates
— exactly the inversion the mechanism predicts. Restricting the pooled figures to
the seven drafting prompts sharpens them: mtp 0.8416 %, serial 0.2069 %, ratio
0.8130 %.

---

## 3. Marginal weight-stream cost

### (b) Width 8, 2 → 3 streams

14 dispatch-only, work-identical pairs across **7 independent fingerprint
groups**.

```
cluster mean          : +0.4910 %  of candidate leg per marginal stream
cluster se            :  0.3315 %
cluster t             :  1.48
groups positive       :  4 / 7
```

| group | pairs | mean Δ |
|---|---|---|
| `9a5ecf786cb0` | 6 | +1.9970 % |
| `8bb4dfd03ecb` | 1 | +1.0379 % |
| `cb6151db87fc` | 2 | +0.9627 % |
| `850d910774ab` | 1 | +0.1906 % |
| `c20dd11ef7bb` | 1 | −0.0119 % |
| `e95589cbfdc3` | 2 | −0.3323 % |
| `774949454dfa` | 1 | −0.4068 % |

**Not significant.** The cluster sd across groups (0.8772 %) is larger than the
mean.

#### The direction is consistent, and that is a weaker claim than it looks

All **7 of 7 drafting prompts are positive**, +0.68 % to +1.51 % (sign test
p = 0.0156). But the prompts share the same 14 pairs, and 6 of those 14 share one
arm. This is a statement that *the sign is stable across prompts*, not an
independent p-value. Quoting 0.0156 as the significance of the stream effect would
be wrong.

#### The scatter needs no stream effect to explain it

If the only thing separating two arms of a pair were replication noise, the pair
difference would have sd `√2 ×` the replication sd. Observed against that
prediction:

| prompt | observed pair sd | √2 × replication sd | mean Δ |
|---|---|---|---|
| `192fb621` | 1.4467 % | 1.1772 % | +1.4263 % |
| `a2ea8b60` | 1.0127 % | 1.0785 % | +0.6802 % |
| `ea82dcb5` | 1.1028 % | 0.8913 % | +0.7757 % |
| `00142a44` | 1.7431 % | 0.9793 % | +1.5121 % |
| `919318e1` | 1.2551 % | 1.1922 % | +1.0529 % |
| `3b10cb4d` | 2.3036 % | 1.6162 % | +1.4822 % |
| `4b9e88cd` | 1.7710 % | 1.2568 % | +1.1500 % |
| **`c1ec5866`** | **0.1176 %** | **0.0870 %** | **−0.0493 %** |

Observed ≈ predicted throughout. **The replication floor alone accounts for the
spread.**

#### Negative control: passes 14 / 14

The low-drafting prompt `c1ec5866` (rho 0.1540, 449 non-drafting rounds) moves
**−0.0493 %** while the heavy prompts move +0.68 % … +1.51 %. It ranks smallest of
the eight in most pairs — in one, control `|d| = 0.0255 %` against a heavy-prompt
mean of 3.5466 %, a **139×** ratio. Whatever the drafting prompts are picking up is
localised to drafting rounds, which is where a stream effect must live. The control
passing is the reason this is reported as "underpowered" rather than "absent".

### Width 4, 1 → 2 streams

25 pairs, but only **2** independent groups and 24 of the 25 in one of them.
Cluster +0.4700 % ± 0.2413 %, t = 1.95 — which looks better than width 8 until the
prompt-level check: only **4 of 7** drafting prompts positive, sign test p = 1.0.
**Null.** The tighter se is an artefact of pretending 24 correlated pairs are
independent.

### (c) Dose 2 / 3 / 4: not testable

Only `e95589cbfdc3` reaches three levels. 3→4 streams is a single dispatch-only
pair at **+0.4482 %**, while 2→3 in the *same group* is **−0.3323 %**. Not
monotone, n = 1 per step. **Dose spacing is untestable on the current board.**

### Identifiability limit — the assignment's specified estimator cannot be run

`officialMetrics.per_prompt` contains exactly:

```
effective_mean_draft_len   non_drafting_round_count   mtp_seconds_per_token_mean
serial_seconds_per_token_mean   raw_ratio_of_means   accepted_pair_count
head_provenance_sha256   parity_ok   prompt_sha256
```

**There is no per-width round histogram anywhere in the published metrics.** So:

- `n_8(p)` — how many width-8 rounds a prompt ran — is **unobservable**;
- `delta_T(8)` **in milliseconds is not identified**; only the leg-level product
  `n_8(p) · delta_T(8)` is;
- the assignment's free falsification test (the `delta_leg` ratio across prompts
  should equal the `n_8` ratio) **cannot be evaluated**, because one side of the
  equality is unmeasurable.

Substituted: the low-rho negative control above, which the published data does
support. Leg-relative % is also the decision-relevant currency, since the score is
built from leg ratios rather than from absolute milliseconds.

### Assumptions, and which way each one biases the answer

| assumption | if violated | bias direction |
|---|---|---|
| Byte-identity outside the two kernel files ⇒ identical work | — | holds by construction |
| Dispatch-cell-only kernel diff ⇒ only stream count changed | other kernel edits leak in | **inflates** \|effect\| |
| Fingerprint groups are independent | shared arms correlate them | **shrinks** se, inflates t |
| Replication noise is i.i.d. across submissions | correlated within solver/session | **shrinks** the floor |
| M4 Pro register readout transfers to ranked M5 (§5) | per-arch allocation differs | unknown |

The two that matter most both push *toward* a false positive, and the result is
still null.

**One confound this design cannot remove.** Byte-identity outside the kernel
controls for the rest of the tree; it does **not** control for *why* a solver chose
that cell. A solver who picked the 3-stream width-8 cell may have done so after
their own tuning, on their own hardware, in a session with its own thermal history.
Cell choice is not randomised. This is an observational study on a public
leaderboard, not an experiment.

---

## 4. (a) Streams refit — the ladder belongs to a different tree than HEAD

`e43.LOCAL_LADDER` is annotated "E25/E27 instrument". Censusing the tree it came
from (`04ad6bf1`) gives IPG `M5=5, M9=5` → streams `1,1,1,2,2,2,2`, with a **single
boundary at 5→6**. HEAD's table is different:

```
HEAD     IPG      M3=3 M4=4 M5=3 M6=3 M7=4 M8=4 M9=3
HEAD     streams  1 1 2 2 2 2 3      boundaries 4->5 and 8->9
04ad6bf1 streams  1 1 1 2 2 2 2      boundary  5->6
```

Fits to the ladder (ms):

```
streams_ladder_tree_04ad6bf1   max|resid|  1.7671  rms 1.1119  const=16.3723 stream=20.3258 M=11.7791
streams_HEAD_tree_MISMATCHED   max|resid| 10.0872  rms 5.1299  const=22.4116 stream=-1.6235 M=16.5985
linear                         max|resid|  9.8553  rms 5.1508
quadratic                      max|resid|  8.1794  rms 4.9421  M2=-0.4190
step_at_5                      max|resid|  8.7774  rms 5.0439
step_at_6                      max|resid|  1.7671  rms 1.1119  const=36.6981 step=20.3258 M=11.7791
step_at_7                      max|resid|  8.7111  rms 5.1083
```

Findings:

- The stream model **on its own tree** reproduces thorfinn's
  `16.432 + 20.291·streams + 11.798·M` (max\|resid\| 1.674) to ~0.06 / 0.035 /
  0.019, derived here without importing his coefficients.
- **`step_at_6` ≡ the stream model on `04ad6bf1`**: identical residuals and an
  identical coefficient (`step` = `stream` = 20.3258), because streams = 1 + [M≥6]
  on that tree. `step_at_6_equals_ladder_stream_model: True`. So r1's "step at 6"
  was the stream model in disguise — correct for the ladder's tree, wrong as a
  claim about the shipped tree. The assignment's §2 correction (boundaries 4→5 and
  8→9) applies to the **pooled ranked rows**, not to this ladder.
- Applying HEAD's stream vector to this ladder yields a **negative** stream
  coefficient (−1.6235: more streams = faster). The mismatch announces itself.
- **Deliverable (a) as literally specified is not executable.** The only width
  ladder in the repository belongs to a different dispatch tree, so refitting the
  *shipped* tree requires a new local ladder — which is E46 (#51)'s job.
- Quadratic falsified model-free. First differences:
  `d4=10.267 d5=13.389 d6=32.680 d7=9.851 d8=10.796 d9=14.953`; **max drop
  22.829 ms**, non-monotone, so no quadratic in M can fit.

### Stream-minimality, re-derived rather than quoted

From the template's own constraints (`M ∈ [3,9]`, `M % IPG != 1`, wide helper
`NA ∈ [2,4]`), the legal IPGs and minimum reachable streams per width are:

| M | legal IPGs | min streams | shipped | minimal? |
|---|---|---|---|---|
| 3 | 3, 4 | 1 | 1 | yes |
| 4 | 2, 4 | 1 | 1 | yes |
| 5 | 3 | 2 | 2 | yes |
| 6 | 2, 3, 4 | 2 | 2 | yes |
| 7 | 4 | 2 | 2 | yes |
| 8 | 2, 3, 4 | 2 | 2 | yes |
| 9 | 3 | 3 | 3 | yes |

**The shipped table is stream-minimal at all seven widths.** No stream win exists
under the current `NA <= 4` bound. Raising the bound to `NA = 5` would let M5
(2→1) and M9 (3→2) each drop a stream — that is the only remaining kernel axis,
and §3 is the price sheet for it.

---

## 5. Is the width-8 A/B confounded by occupancy? No.

`affine_qmv_fast` switches on the **runtime** value `ntg.x` inside one
`[[kernel]]`, so all width cells share a single register allocation equal to the
worst branch. If the width-8 cell sets that peak, swapping its IPG moves occupancy
for *every* width and §3 would be an occupancy effect in disguise.

Structurally it should not: `<T,8,4>` (TAIL=0) emits `wide<T,4>`; `<T,8,3>`
(TAIL=2) emits `wide<T,3>` and `wide<T,2>`; and the shipped table already
instantiates all of NA {2,3,4} elsewhere (M4/M7 → 4; M3/M5/M6/M9 → 3; M5/M7 tails
→ 2). Neither arm introduces a body the other lacks. **But the register allocator
is the authority, not that argument** — `_m` is templated on M as well as IPG, so
the two arms are distinct instantiations whose bound checks, branch layout and
inlining decisions the compiler is free to treat differently.

`research/e45_stream_register_probe.sh` compiles the **production entry point**
both ways through a shadow include directory (E40's pattern; the working tree is
never modified — verified 0 modified lines in `quantized.h` at exit).

Readout 1 — AIR at the scored kernel `affine_qmv_fast<bfloat16_t,64,4,false>`:

```
s2 (IPG 4, 2 streams)  peak_live_regs=163  allocas=55  float_ops=1717  backedges=47
s3 (IPG 3, 3 streams)  peak_live_regs=164  allocas=56  float_ops=1769  backedges=49
alloca TYPE SET identical: ['[4 x <2 x float>]', '[4 x [4 x i16]]', '[4 x float]']
```

Readout 2 — register-limited occupancy from the back end (`Apple M4 Pro`):

```
s2  maxTotalThreadsPerThreadgroup = 1024   execWidth 32   tgMem 0
s3  maxTotalThreadsPerThreadgroup = 1024   execWidth 32   tgMem 0
```

Instrument liveness: whole-module optimised AIR differs by **25,345 lines**, so the
swap definitely took effect. Negative controls behaved: the batched twin and the
2-bit draft-readout kernel are **statistic-for-statistic identical** across arms
(peak_live_regs 31 and 57), as they must be — the crossrow block is guarded by
`if (!batched && ...)`, so the batched instantiation eliminates it as dead code.

**Verdict: EQUAL on the authoritative readout. The width-8 stream A/B is not
confounded by kernel-wide occupancy.**

Two honest caveats:

1. The zero-by-construction prediction was *directionally* right but **literally
   wrong**: the compiler does move, by +1 register and +1 alloca (+0.6 % of 163).
   The alloca *type set* is unchanged, so the extra alloca is another instance of
   an existing array, not a new spill class.
2. `maxTotalThreadsPerThreadgroup = 1024` is the **architectural ceiling**, so this
   readout is saturated: it proves neither arm is anywhere near the register cliff,
   but it cannot resolve, say, 40 registers from 60. And it was taken on **M4 Pro,
   not the ranked M5** — register allocation is per-architecture. Transfer risk is
   low given the AIR structures differ by one register, but it is not zero.

---

## 6. Instrument defects found

### 6.1 `selftest` fails open on an unfetched checkout — and does it loudly

Before fetching the submission refs, this checkout had **0** submission refs — the
fail-open state was the *default* here, not something contrived. In that state
`stream_dispatch_census --selftest` **FAILED with 3 errors**: "no dispatch table
found" for `0c90733d38` and `ca9251b8`, plus "HEAD or crown dispatch table
unreadable". Those read as substantive findings about the dispatch tables. They
actually meant *"you have not fetched"*.

After fetching, `selftest` **passes**: 12 arithmetic checks, 3 pinned trees,
HEAD == crown dispatch table, fingerprint exclusions present, 6 `ab_verdict`
fail-closed cases.

The two guards that were checked *do* fail closed: `ab` on an empty ref set exits 1,
and `census deadbeefcafe` exits 1 with `UNRESOLVED`. The residual bug is that
`selftest` itself does not, and dresses a missing prerequisite as a content failure.
**Suggested fix: `selftest` should refuse to run when the submission ref set is
empty, with a message naming the fetch.**

*My own error, recorded for the avoidance of doubt:* my first exit-code check piped
through `head`, so `$?` read `head`'s status rather than the tool's. That produced a
false alarm about a guard that was in fact fine. The tool was not at fault.

### 6.2 Two of the assignment's three "gold" groups are useless, and the largest was missed

- `ae0ff0917146` (13 trees) fails on **two independent counts**. Its minority arm
  `93b58801` has streams `M3=1 M4=1 M5=2 M6=2 M8=3 M9=3` — **M7 is absent
  entirely**, so it is a *width-availability removal*, not a stream-count change;
  and all 12 cross-arm pairs are confounded anyway. Same pattern elsewhere:
  `e617ef07` drops M9 (`7d29a6339532`), `ae97296e` drops M7 (`774949454dfa`).
- The **largest** A/B group, `9a5ecf786cb0` (24 trees, arms 19/4/1), is not
  mentioned in the assignment and supplies clean contrasts at **both** width 4 and
  width 8. It carries 6 of the 14 width-8 pairs.

### 6.3 The census fingerprint does not isolate the dispatch table

The fingerprint excludes **both kernel files** from the hash — and the mechanism
lives in one of them. Consequences found in the data:

- `070f1189` vs `b428c300` share a fingerprint but differ by **101 lines**,
  including an entire `qmv_fast_singlerow_affine2_g64` definition.
- Within a single arm the kernels also differ: `cb6151db87fc`'s streams(8)=2 arm
  has 8 trees but **5 distinct kernel-blob pairs**. So **within-arm spread is not a
  pure noise floor**, and cannot be used as one.

Fixed by adding `classify_kernel_diff`, which keeps a pair only when the entire
kernel diff is confined to `qmv_fast_crossrow_affine4_g64_m<...>` cells and switch
scaffolding. Of 135 candidate pairs this admits 43 and rejects 92:

```
dispatch-only|[8]     14        confounded|[8]      43
dispatch-only|[4]     25        confounded|[4]      32
dispatch-only|[4,8]    4        confounded|[7]       9
                                confounded|[8,9]     3
                                confounded|[9]       2
                                confounded|[6,8]     1
                                confounded|[6]       1
                                confounded|[7,8]     1
```

### 6.4 A stale comment in the shipped dispatch table

`quantized.h` case 8 carries a long comment arguing for **"3+3+2, not 4+4"** — IPG
3, three streams — complete with receipts (`85d5bca3 2.91143`, `yzxoi 2.92675`) and
a register-cliff rationale. The code immediately below it is
`qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>`: IPG **4**, two streams. The
comment documents the opposite of what ships. Flagged, not touched — it is on the
scored surface and outside this assignment's scope.

---

## 7. Preserved r1 findings

r1 asked whether pooling could separate the cost families. It cannot, and the
reason is arithmetic rather than statistical:

- The pooled `x` column is **bit-identical across all five trees** (max gap 0).
- The pooled shape box **equals** the single-row bracket (max gap 0).

So pooling adds no independent constraint — it could never have separated the
families. The offset-only LP is feasible over `λ ∈ [44.159, 129.925]`.

The apparent r1 "separation" was a **clamping artefact**: cone-projected step6
scored 0.256576 with **5 of 6 components clamped**, against quadratic 0.029477 with
**0 clamped**. Comparing a mostly-clamped projection against an unclamped one
measures the clamp, not the model.

### Two record corrections

1. E43's s-bracket upper endpoint is **80.482786128094**. The recorded
   `80.48305253958134` is wrong.
2. The assignment's residual ratio **1.174289** is `rms|resid|` quadratic/step. The
   **max**-relative ratio is **1.485036**. Both are correct numbers for different
   statistics; only the second was mislabelled.

### Withdrawn

The r1 working model `e_p = s · q_p` is **withdrawn**. It presumed a per-prompt
scaling that the identifiability limit in §3 shows is unobservable.

---

## 8. How to read the nulls

Per the campaign's own rule, a null is instrument-suspect before it is physics.
Checked, in order:

1. **Is the instrument alive?** Yes — 25,345 AIR lines move under the §5 swap;
   `--self-test` is 34/34; `selftest` passes post-fetch; the negative control
   discriminates 139×.
2. **Are the pairs real?** Yes — 43 of 135 survive `classify_kernel_diff`, and the
   14 width-8 pairs span 7 independent groups.
3. **Is the effect below the floor?** Yes. This is the answer. Observed pair sd ≈
   √2 × replication sd in 8 of 8 prompts.

So the null is a **power** statement, not an absence claim: an effect of +0.4910 %
cannot be resolved against a 0.7353 % per-submission floor with 7 groups. The
negative control passing means there probably *is* something there, localised to
drafting rounds, of roughly the size estimated — the board just cannot prove it.

---

## 9. Suggested follow-ups (not implemented)

1. **Re-scope `e43.SIGMA_SCORE_PCT`.** 0.0978 % should be documented as
   *within-session* noise. Every cross-solver board comparison — crown-gap chasing
   most of all — should be re-read against the 0.7353 % across-submission figure.
   This is the single highest-value change to campaign bookkeeping in this report.
2. **`selftest` should fail closed on an empty submission ref set** (§6.1).
3. **A local width ladder on HEAD's dispatch tree** is a prerequisite for
   deliverable (a) and belongs to E46 (#51). The existing ladder is `04ad6bf1`'s.
4. **Do not adjudicate the `NA = 5` bound with one ranked submission.** A ±0.33 %
   effect against a 0.7353 % floor needs replication — ABBA-counterbalanced locally
   under the standing ungated measurement mode, not a board row.
5. **Repoint the census fingerprint** to include the dispatch table and exclude only
   the genuinely irrelevant kernel body, so `classify_kernel_diff` becomes a check
   rather than a load-bearing filter.
6. **Reconcile the stale case-8 comment** with the shipped IPG (§6.4) — someone
   should establish which of the two the receipts actually support.
