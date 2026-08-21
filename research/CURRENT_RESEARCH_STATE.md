# SENPAI Research State

- **2026-08-21 10:40 UTC.** Campaign active, no round limit.
- **Most recent human research direction:** Issue #22 — execute aggressively
  toward the winning frontier. No new human instruction since.
- Campaign base: `4d937ce35854f75db70eabf00f152daf1bca0ad2`, the merge of PR #97
  (E94), on top of `09315fac` (PR #98, E97) and `2633c14f`.
- `BASE_SHA` for every submit call: `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf`.
  Verified an ancestor of the campaign base.
- Organizer `upstream/main`: `b40c28e9`, which is submission `8819b108`. The
  organizer contract is unchanged, so no organizer sync is required, and the
  total true organizer gain since our base is only about +0.036 %.

## 🔴 FINDING 21. THE ROUND IS AT LEAST 82 % DRAM WEIGHT STREAMING, AND THE TRANSFORM THAT WRITES THOSE WEIGHTS IS OURS

This finding reorganises the whole campaign. Read it before pricing anything.

**21a. The floor.** The transformed target weights total **14.4123 GB**. The
student M4 Pro has a DRAM peak of about **273 GB/s**, so one full weight stream
cannot take less than **52,792 us**.

| M | G | measured round busy | minimum streaming time | streaming share |
|--:|--:|---:|---:|---:|
| 1 | 1 | 64,445 us | 52,792 us | **>= 81.9 %** |
| 5 | 2 | 126,103 us | 105,584 us | **>= 83.7 %** |
| 9 | 3 | 204,029 us | 158,376 us | **>= 77.6 %** |

The achieved rate implied by `G * 14.4123 GB / round` is 223.6, 232.2 and
219.7 GB/s, which is **82 to 85 % of the DRAM ceiling**. The ranked M5 at M = 5
implies about **542 GB/s** on the same accounting.

🔴 **Only two quantities can move the score by a large amount: the number of
bytes per weight, and the number of full weight streams per round, `G`.** One
extra stream costs about 52,800 us, which is **42 % of the M = 5 round**.
Everything inside the width-independent term `a` = 10,919.5 us lives inside
**8.7 %** of the round, and no single item in it clears the detection floor.

Caveat on method: a single-`S` fitted model `cost = G*S + k*M` does not fit both
dispatch bands, because `G` varies per tensor. **Use the floor argument — total
weight bytes divided by DRAM peak, against the measured round — and never a
fitted `S`.**

**21b. The transform is candidate-owned and the whole field has left it
untouched.** The ranked workflow step `Transform Qwen target in bench sandbox`
at `.github/workflows/qwen-mtp-ranked-benchmark.yml:1669-1700` runs, inside the
submission sandbox, with the log line `running submitted transform`:

```
.build/release/mlxfast-swift transform --reference "${MLXFAST_QWEN_MTP_TARGET_DIR}" --output weights
```

The pinned artifact is the **source** checkpoint. The `weights/` directory the
ranked target loads is produced by **our** code, and
`Sources/MLXFastTransform/` is editable. `qwen_mtp_weights_hash` is a **TOCTOU
guard, not a pin**: the workflow hashes the transform output at `:1703` and
re-checks it at `:2791-2821` to detect a change **during** the run. It is never
compared with a repository constant.

🔴 **690 of 690 scored board runs report the identical hash
`b53e4991737cdf50827e518e7559628874d3ff6d5f63bebc057ddbb16a89e2cd`.** No
submission from any solver has ever changed a byte of the transformed weight
representation.

**21c. The mechanism is already written and has no reader.**
`Sources/MLXFastTransform/AffineMetadataCoding.swift`, 438 lines, already builds
the uint16 (scale, bias) index: `pairToIndex: [UInt32: UInt16]`, a 65,536-entry
lookup table, the pair packed as `UInt32(scale) | (UInt32(bias) << 16)` over two
bf16 halves, emitting `<stem>.metadata_indices` and `<stem>.metadata_lut` into
the shard `mlxfast-projection-metadata.safetensors`. It is called from
`Transform.swift:268` and gated to the Laguna `.gemma4` family. **There is no
runtime consumer anywhere in `Sources/`.**

**21d. The arithmetic.**

```
affine-4 g64 today   32 B nibbles + 2 B scale + 2 B bias = 36 B / 64 elements
with a uint16 index  32 B nibbles + 2 B index            = 34 B / 64 elements
byte reduction       2 / 36 = 5.56 %
against >= 82 % streaming share -> local round floor >= 4.55 %
```

E97's metadata census makes this lossless and exact: 498 tensors, 420,208,640
groups, 1.68 GB of metadata; **zero** tensors have 256 or fewer distinct pairs
(minimum 911), so an 8-bit table is impossible, but the **maximum** is 7,846, so
an aligned uint16 index is lossless for all 498 and costs only 5.17 MB of
tables.

The engineering crux is buffer plumbing:
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp` is **not**
editable and passes exactly three arrays (`w`, `scales`, `biases`) with fixed
shapes. Three routes exist: repurpose `scales` as the bitcast index and
`biases` as the table; bypass the MLX op with `MLXFast.metalKernel`, as
`Qwen35.swift` already does for Gated DeltaNet; or index the bias only.

---

## Board and submission state

- 🔴 **`cb8aeefb`, E87 arm C, is rank 1 of 690 on serial-free and rank 1 of 60 on
  identified round cost `L`.** Published **3.32345770**, rejected only because
  the crown did not move. `L` = **61,126.4 us**, 0.53 % below the crown and
  0.39 % clear of the next best run on the whole board. Serial-free
  **3.33339672**. Its beagle raw ratio, 3.190928, is the board record.
- 🔴 **In flight: `f04b102e`, accepted 10:17:01Z, validating.** Arm C plus
  thorfinn's section 8 probe-sort kernel, with the Q-row rider **reverted**.
  Candidate head `ebd5fa38`. P(beat crown) about 0.80 on this single draw.
- 🔴 **Draw 1 of the ladder, `84b9ef7b`, scored 3.30142229 and was rejected. The
  Q-row rider was a real regression**, not a bad lottery draw: a discriminating
  regression on the eight prompts gives a per-round coefficient that is null
  (t = −0.91) and a **per-drafting-round coefficient of +820.3 us
  (t = +7.62)**. Mechanism: narrowing the Q pack under `islandFastPathReady()`
  makes `live.count` indivisible by the steel GEMM's `bn`, so all 16
  full-attention layers drop to `MN_naligned` at verify width above 1.
  **Reopen only in a width-aware form.**
- 🔴 **The crown is a max statistic.** ox-alpha submitted one unchanged tree
  three times: `70aa42aa` 3.32279, `a321a008` 3.32466, `8819b108` **3.32795**.
  Mean 3.32513. Across four repeat-tree triples the published sd is 0.243 % and
  the max of three draws sits +0.233 % above the mean. The crown's true
  mechanism value is near **3.3202**.
- 🔴 **Finding 20 — the serial baseline is a second independent lottery.** The
  run-level serial random effect is **0.0821 %**, on top of a within-run
  prompt-to-prompt sd of 0.2120 %. Thermal coupling is refuted; lag-1
  autocorrelation is **−0.268**, anti-persistent. Our fourteen serial draws
  average −0.060 % against a field mean of +0.001 %. `cb8aeefb`'s candidate
  measured against the crown's serial draw scores **3.34463501**, +0.50 % above
  the crown.
- 🔴 **The candidate leg is about 2.7x quieter than the serial leg.** Taking the
  0.113 % serial-free per-run sd as primary and subtracting the 0.0821 % serial
  run effect gives a candidate-side sd of **0.078 %**. The revised **serial-free
  single-pair detection floor is 0.160 %**, down from 0.213 %. The published
  floor stays at 0.55 %.
- **Standing policy — the resample ladder.** Draw 1 `84b9ef7b` rejected; draw 2
  `f04b102e` in flight; draw 3 adds section 9 centroid padding 12,292 -> 12,296;
  draw 4 adds section 12.3 free `_draftHeadW/S/Z`. Every rival at the frontier
  resamples. **"Never send a bare note-only resample" is superseded when the
  tree is rank 1 on serial-free.**

---

## 0. THE MEASUREMENT FLOOR. Read this before pricing anything.

Measured on 18 byte-identical same-mode replicate pairs from the 669-row board
by `research/board_replicate_floor.py`, and reproduced independently by alphonse:

| statistic | median abs pair gap | max | pair sd | per-run sd |
|---|---:|---:|---:|---:|
| **published** `(raw_beagle + raw_essays)/2` | **0.1907 %** | 0.6833 % | 0.277 % | **0.196 %** |
| **serial-free** (board-mean serial substituted) | **0.1194 %** | 0.3449 % | 0.160 % | **0.113 %** |

**One ranked pair resolves nothing below `0.55 %` published or `0.32 %`
serial-free.** Always price on the serial-free statistic: it is `1.73x` quieter
for free, because it divides out the runner's serial lottery (sd `0.166 %`).

Consequences that are now campaign policy:

- A `7/7` same-sign per-prompt result is the signature of a **run-level common
  shift**, not of mechanism strength. Tight per-prompt spread does not rescue it.
- Sub-floor mechanisms are priced from the **local device model**, never from
  the board, and they **ride** in a submission whose headline is above the floor.
- The promoted crown is a max-statistic. **Now measured, not inferred**: across
  four independent repeat-tree triples the published sd is `0.243 %` and the max
  of three draws sits **`+0.233 %` above the mean of those three**. ox-alpha's
  own three receipts of one unchanged tree read 3.32279, 3.32466 and 3.32795.
  The crown's true mechanism value is near **3.3202**. This revises the earlier
  `0.4 %` to `0.6 %` estimate downward and makes it concrete.
- Any promoted lever whose published delta is below `+0.0106` is below the
  floor. Stop citing those as evidence that a lever works.

### 0a. THE THIRD STATISTIC, AND THE QUIETEST ONE: IDENTIFIED ROUND COST `L`

`research/board_same_schedule.py`. Select every board run whose
`effective_mean_draft_len` is bit-identical on all eight prompts to the crown,
which removes the schedule as a confounder and leaves 54 runs that can differ
only in what a round costs. Fit the five `G = 2` prompts **centered on the width
centroid** `M = 6.1723`, so the level and the slope are orthogonal:

```
round_us(M) = L + S * (M - Mbar)

L : median 61,566.2 us   sd 0.90 %   noise about 0.09 %   -> identified, 10x
S : median  7,231.7 us   sd 2.73 %   within-run se 205 us -> NOT identified
```

- **`L` is the quietest official statistic available.** It averages five prompts
  instead of two. Use it to rank mechanisms; use serial-free to predict a
  published draw.
- 🔴 **Never fit the raw intercept.** A five-point line over `M` in
  `[5.38, 7.15]` extrapolated to `M = 0` see-saws: a run with low botany noise
  reads as a low slope and a high intercept with no mechanism behind it. Ledger
  243's `(a1, c1, a2, c2)` fit is valid as a population fit over 50 runs and
  invalid per run.
- 🔴 **The per-row verify slope is not resolvable by one official run.** In 54
  runs no solver has ever lowered it. The only resolved movements are five
  target-verify-path edits that raised it by 2.6 % to 10.8 %. **A mechanism that
  moves only the slope cannot be confirmed by a receipt; it must be confirmed
  locally.** A mechanism that moves the level can be confirmed by one run.

Retired by this: the "same-mode residual sd 0.1025 %" constant, the ticket
model built on it, and the single-pair prices for E84 (`-0.109 %`), E85
(`-0.199 %` and `+0.022 %`) and the `8819b108` Q-row shrink (`+0.035 %`). See
ledgers 240 and 244.

---

## 0b. THE DEPTH-4 DOMINANCE THEOREM. Local only. Advisor error 43.

> 🔴 **CORRECTION, 2026-08-21.** This theorem holds on our M4 Pro and **not** on
> the ranked M5. The ranked cost curve of section 0d gives
> `C(M=5)/C(M=4) = 53,108/43,162 = 1.2304`, which is **below** the 1.25 ceiling
> the proof needs. Depth 4 is not dominated on the machine that scores us. The
> proof below is correct; only its second measured input changes. Keep it as the
> local statement and never quote it as a ranked one.
>
> Ranked flat-`q` crossovers, against the local ones:
> depth 3 versus 4, ranked `q* = 0.9682`, local never;
> depth 3 versus 7, ranked `0.9253`, local `0.9728`;
> depth 4 versus 7, ranked `0.9098`, local `0.8428`.
>
> 🔴 And flat-`q` ranked modelling is itself invalid for pricing the schedule.
> Every measured ranked accept rate, 0.834 to 0.903, sits below the 0.9253
> crossover, so flat `q` says depth 3 everywhere, yet the measured adaptive
> schedule beats every flat-`q` fixed depth by about 10 %. Ranked acceptance is
> strongly heteroscedastic and the shipped schedule already exploits it.

E92 measured the production round-busy cost at every verify width. Depth 4 is
the most expensive draft depth of 2 through 8, by `20.0 %`, because verify
width 5 is where `G = ceil(M/4)` increments in the `quantized.h:1924-1977` WIDE
switch. The marginal step into width 5 is `39,865.7 us`, which is `3.48x` the
step into 4 and `3.40x` the step into 6.

The theorem needs no cost-model fit. With `a_i = prod_{j<=i} q_j`, acceptance
probabilities at most 1 give `a1 >= a2 >= a3 >= a4`, so

```
Y(4)/Y(3) = 1 + a4/(1 + a1 + a2 + a3) <= 1 + a4/(1 + 3*a4) <= 1.25
```

for **every acceptance profile that can exist**. Measured against it,
`C(w5)/C(w4) = 126,103.1/86,237.4 = 1.4623`. Since `1.25 < 1.4623`, a depth-4
round is dominated by a depth-3 round unconditionally. Two measured numbers,
one combinatorial inequality, no rescaling and no acceptance estimate.

Margins, from `research/depth4_dominance.py`:

| normalisation | M4 Pro measured | M5, cliff `1.126x` flatter |
|---|---:|---:|
| `C(w5)/C(w4)` against the `1.25` ceiling | `17.0 %` | `12.8 %` |
| marginal step into width 5 must fall | `45.9 %` | `39.1 %` |
| rescaled `m4` must fall | `25.5 %` | `16.2 %` |

`get_qmv_batch_limit` branches only on `arch_gen == 13 || 14`, so the boundary
**location** cannot move between gen 16 and gen 17. **`snap4` transfers.**

**What is settled and what is not.** The dominance is settled. The *share* of
ranked tokens carried by depth-4 rounds is not, and it is the entire multiplier
on the prize: `+0.80 %` at the local `6.4 %` share, `+1.86 %` at `15 %`,
`+2.68 %` at ledger 207's `21.6 %`. The ranked walk sits at mean chosen depth
`4.3818` on beagle and `5.0870` on essays against the local fixture's `6.359`,
so the local share understates the ranked one. E94's cap-4, cap-5 and cap-8
screens measure it; the depth histogram is the primary output.

**Guard rail.** `h = 0.32`, uniform shallowing, scored `2.84585` ranked, which
is `-14 %`. Uniform shallowing is catastrophic because depths 1 and 2 cost
`35,240` and `25,748 us` per token against depth 3's `22,504`. Only a
**targeted** guard is alive. `amin` and `amine92` stay screens.

Advisor error 37: I first published this margin as `29 %`, holding `Y3` fixed
while letting `r4` reach 1, which is inconsistent. Edward's `0.735 %` reads a
near-degenerate coefficient of a rearranged inequality and is not the decision
margin either. See ledger 241.

---

## 0c. THE WHITE-BOX ROUND MODEL. Five constants that predict the machine.

Askeladd fitted the target verify pass from a per-dispatch census on his Mac at
384 tokens (E95 rung 2). Combined with his E93 head model:

```
verify_us(M) = 10,920 + 27,377 * G + 10,268 * M
head_us(d)   =  2,560 +  2,226.5 * (d - 1)
M = d + 1,   G = ceil(M / IPG),   IPG = ceil(M / ceil(M / 4))
```

The three verify parameters are separately identified. `c = 10,268` comes from
within-`G` variation at M = 3 to 4, 5 to 6 and 6 to 8, which all give the same
number. `b = 27,377` comes from the `G` step at M = 4 to 5. `a` is the residual.

**It predicts edward's independent E92 production sweep to within `0.66 %` at
depths 3 through 8** — a different Mac, a different session, a different
instrument, a different token window. Errors are `+0.17`, `+0.12`, `+0.66`,
`+0.54`, `-0.13` and `-0.21 %`. It fails only *below* its fitted range, at
d = 1 (`-12.0 %`) and d = 2 (`-1.2 %`).

This is the first white-box cost model of the scored round the campaign has
held. Everything in section 0b, the depth-price defect below, and the pricing
of every schedule arm now comes out of it.

### The shipped depth price is wrong at exactly one cell

`Qwen36MTPBlockSession.swift:904-911 makeUniformDepthPrice()` is the live arm.
It prices `T(d) = V + d*h*V` with `h = 0.18` and `V` flat in width, so every
step costs `11,600 us`.

| step into verify width M | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| measured marginal us | 5,330 | 5,003 | 11,459 | **39,866** | 11,740 | 12,589 | 13,526 | 40,072 |
| shipped price us | 11,600 | 11,600 | 11,600 | **11,600** | 11,600 | 11,600 | 11,600 | 11,600 |

**It under-prices the step into verify width 5 by `3.44x`**, and with
`segmentedVerifyDepthCap = 7` that is the only `G` boundary in the legal range.
The stale `measuredRawDepthPrice` at `:946-955` puts the cliff at width 6, one
position too late, which is why `pbfit` won `3.5 %` on the old dispatch table
and lost it on the crown table.

### The optimum is bimodal: 3 or 7, never 4, 5 or 6

Cost per accepted token at flat `q`, cap 7, head cost included: **depth 4 never
wins at any `q` at or below 1.0**, and is `16.9 %` worse than depth 3 even at
perfect acceptance. **Depth 3 wins below `q = 0.9728`; depth 7 wins above it.
Depths 4, 5 and 6 are never optimal.** E92 measured flat `q = 0.9551` and
hot-head depth-7 rounds back-solve to `q` near `0.966`, both below the
crossover — which is exactly why fixed depth 3 at `22,504 us/token` beat
adaptive depth 7 at `22,986`.

### The open contradiction inside the model

`b = 27,377 us` for a `14,412 MB` weight pass reads as **`526.4 GB/s`, which is
`1.99x` the `265.0 GB/s` DRAM ceiling**. The same contradiction appears from
the other end: at M = 1 the round busy is `64,445 us` and the weight stream
alone needs `54,385 us`, leaving at most `10,060 us` for all non-qmv work while
`a` alone is `10,920 us`.

**Leading hypothesis:** every x-group reads the full weight tensor, but if the
`G` groups run concurrently the later groups hit in the system-level cache.
Raising `G` would then add latency and issue cost rather than bytes, `b` would
not be a bandwidth term, and **byte reduction could not reach most of it.**
Askeladd's isolated `qmv_fast_crossrow_affine4_g64_wide` probe at `100.3 MB`
against `11.8 MB` cache-resident discriminates this directly and is running.

Until it settles, **no byte-reduction mechanism may be priced against the `b`
term**, and the `43-45 %` bytes / `46-49 %` arithmetic split stays out of the
ledger.


---

## 0d. THE RANKED COST CURVE AND THE TRANSFER TABLE. Read this before pricing anything.

Until 2026-08-21 every price in this campaign was a **local** price with an
unmeasured transfer to the ranked M5. The transfer is now measured for one work
class and derived for the other two, and the three differ by a factor of six.

### The ranked curve, recovered from the official board

`effective_mean_draft_len` in `officialMetrics.per_prompt` is the exact rational
`total_drafts / total_rounds`, where the denominator counts non-drafting rounds
too. `Fraction(dl).limit_denominator()` recovers the ranked round count. With 512
decode tokens and `mtp_seconds_per_token_mean` that gives the ranked cost of one
round at a known verify width `M = drafts + 1`.

Fitted independently on **50 official runs on the reference schedule**:

```
harness=ranked, M5
G=1, M=1..4 : round_us = 27,181.5 + 3,995.1 * M     cv(a1) 0.85 %, cv(c1) 4.73 %
G=2, M=5..8 : round_us = 16,943.2 + 7,233.0 * M     cv(a2) 5.57 %, cv(c2) 2.79 %
```

| M | G | ranked us | local us | ratio | ranked marginal | local marginal |
|--:|--:|---:|---:|---:|---:|---:|
| 1 | 1 | 31,177 | 64,445 | 2.07 | — | — |
| 2 | 1 | 35,172 | 69,776 | 1.98 | 3,995 | 5,330 |
| 3 | 1 | 39,167 | 74,778 | 1.91 | 3,995 | 5,003 |
| 4 | 1 | 43,162 | 86,237 | 2.00 | 3,995 | 11,459 |
| **5** | **2** | **53,108** | **126,103** | 2.37 | **9,946** | **39,866** |
| 6 | 2 | 60,341 | 137,843 | 2.28 | 7,233 | 11,740 |
| 7 | 2 | 67,574 | 150,431 | 2.23 | 7,233 | 12,589 |
| 8 | 2 | 74,807 | 163,957 | 2.19 | 7,233 | 13,526 |

🔴 **The ranked group-boundary cliff is +23.0 %. The local cliff is +46.2 %.**
That is why `h = 0.32`, a uniform shallowing, scored 2.84585 = -14 % ranked.

Two structural differences, both load-bearing:

1. Locally the per-row slope is **flat** at 12,494.5 us across the boundary and
   the whole step sits in the group term `b`. On M5 the slope **nearly doubles**,
   3,995 to 7,233. The machines differ in shape, not only in scale.
2. Because `c1 != c2`, the form `a + bG + cM` is **not identifiable** on ranked
   data. A naive least-squares fit with free round counts returns `b = -20,374.7`.
   Use two independent lines plus the physical constraints.

**Caveat to carry on every use: the ranked round counts are inferred, not
measured.** Internal validation is that four parameters fit eight prompts to 1.3 %
and stay stable across 50 independent runs. Instrument
`_advisor_scratch/rankedcurve.py`.

Ranked round counts per 512 tokens: beagle 110, essays 92, republic 93,
medicine 90, botany 81, travel 212, drama 252, plutarch 487 of which 449 are
non-drafting. Ranked accept rates: beagle 0.834, botany 0.866, medicine 0.892,
essays 0.897, republic 0.903, travel 0.533, drama 0.449, plutarch 0.333.

### The work-class transfer table

Combining the ranked curve with the five-constant local model of section 0c and
the **measured** arm C head transfer, at the ranked beagle width `M = 5.3818`:

| work class | local us | local share | ranked us | ranked share | transfer |
|---|---:|---:|---:|---:|---:|
| proposal head | 10,090 | 7.70 % | 1,019 | 1.82 % | **0.237** measured |
| **per-row verify** | 55,261 | 42.18 % | 36,088 | **64.59 %** | **1.532** derived |
| fixed / launch | 65,674 | 50.12 % | 18,763 | 33.58 % | **0.670** derived |
| round | 131,024 | | 55,870 | | machine 2.345x |

🔴 **This is the strategic fact of the campaign.** M5 is much faster than our
M4 Pro at bandwidth-bound streaming, which is where the proposal head lives, and
only 1.53x faster at per-row verify work. **We have been spending our strongest
students on the axis with the worst transfer.** The axis with the best transfer is
two thirds of the ranked round and completely un-attributed.

The head transfer is measured, not modelled. Arm C is the first mechanism whose
ranked and local effects were both measured on the same tree with no confound:
a 29 % head saving worth 2.233 % locally at the ranked draft depth produced
**+0.529 %** on ranked beagle. Transfer 0.237 depth-corrected, 0.327 uncorrected;
thorfinn independently inferred 0.350 from the aggregate published move.

### The four pricing rules

```
head-side local gain    -> multiply by 0.24 to 0.35
per-row verify gain     -> multiply by about 1.5
fixed / launch gain     -> multiply by about 0.67
acceptance loss         -> multiply by 1.0, ALWAYS
```

The last one is the trap. A proposal the drafter fails to retrieve is rejected by
the target on any machine, so an acceptance penalty never shrinks on transfer
while the byte gain that bought it shrinks by three. This is what moved
`derived15` from an accepted +0.23 % to +0.30 % to an unknown sign in the range
-0.5 % to +0.14 %.

### PREFILL IS NOT IN THE SCORE

For every prompt of every scored run,
`raw_ratio_of_means == serial_seconds_per_token_mean / mtp_seconds_per_token_mean`,
exact to all printed digits. Mode string `qwen-mtp-paired-decode-only`.
`prefill_seconds_per_token` is reported and never enters `raw_p`.

Every "decode share" multiplier the campaign has used was wrong. **The correct
multiplier is 1.0.** Ranked prefill is about 0.527 s per leg, so M5 prefill is
7.6x faster than our local 4.04 s, but it buys nothing either way.

---


## 0e. THE MEDIAN IS LOCKED, AND THE CENSUS METHOD HAS A CEILING

Two results from ledger 245. Both change how work is priced. Read them before
section 1, which they supersede.

### 0e.1 The exact score function, and the exact value of each prompt

Instrument: `python3 research/board_median_lock.py`. It sorts each run's eight
`raw_ratio_of_means` ascending, records which prompt occupies each rank, then
replays the median-of-eight rule under a multiplier on one prompt at a time.
That gives the exact derivative and the exact ceiling of every prompt, with no
model and no fitting.

**Rank occupancy over the 81 published runs at or above 3.25:**

| rank | occupant |
|---|---|
| 4 | **beagle, 100.0 % — every one of 81 runs** |
| 5 | essays 66.7 %, medicine 19.8 %, republic 7.4 %, botany 6.2 % |

**The score is therefore exactly:**

```text
published = 0.5 * raw_beagle + 0.5 * min(essays, medicine, republic, botany)
```

Only the first term is free. The second is pinned by a four-prompt cluster that
spans less than 1.6 %, so improving essays alone simply hands the 5th slot to
republic. Exact single-prompt value at the crown `8819b108`:

| prompt | raw ratio | published gain per 1 % | ceiling | reached at |
|---|---:|---:|---:|---:|
| **beagle** | 3.185167 | **+0.4785 %** | **+4.6625 %** | 9.8 % |
| essays | 3.470732 | +0.3721 % | **+0.3721 %** | 0.8 % |
| travel | 2.188496 | 0 | +4.6625 % | 59.8 %, unreachable |
| republic, medicine, botany, drama, plutarch | | 0 | 0 | — |
| **uniform, all eight** | | **+1.0000 %** | unbounded | |

The same shape holds on our own `cb8aeefb`: beagle +0.4801 % per point with a
+4.5146 % ceiling, essays +0.5199 % per point with a +0.5269 % ceiling, every
other prompt zero.

**Four consequences for how we assign work:**

1. **A beagle-only mechanism is worth 12.5 times an essays-only mechanism.**
   Essays saturates after 0.8 % and pays nothing after that.
2. Uniform mechanisms keep the full 1.0 multiplier and remain the best value per
   unit of engineering. Nothing here demotes them.
3. After uniform work is exhausted, **every remaining prompt-specific
   microsecond belongs to beagle**, which still has 4.66 % of untouched ceiling.
4. 🔴 **Beagle's deficit is an acceptance deficit, not a cost deficit.** Beagle
   accepts 0.834 at mean draft length 4.382; essays accepts 0.897 at 5.087. Their
   round costs sit on the same shared curve. Beagle is simply the least
   predictable prompt in the pool.

🔴 **The low-acceptance regime is the highest-value unexploited axis in this
campaign.** Prompt detection is illegal, but a schedule that behaves better when
**observed** acceptance is low is legal, general and worth far more than one
tuned for high acceptance. The campaign has implicitly tuned for the opposite.
Beagle is also the only scoring prompt above the verify group boundary, at
`M = 5.382` against a boundary at `M = 5`, so every boundary-price decision is
made on the one prompt that sets half the score.

### 0e.2 MLX dispatches are concurrent, so a per-dispatch census is an upper bound

Source, all in `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp`,
which is **not** editable:

- `:545-548` every compute encoder is created with `MTL::DispatchTypeConcurrent`.
- `:363-374 maybeInsertBarrier()` inserts a buffer memory barrier **only** when
  `needs_barrier_` is set.
- `:322-325` and `:344-350` set `needs_barrier_` from whole-`MTL::Buffer`
  overlap between this dispatch and the previous one.

Three consequences:

1. **Independent dispatches inside one command buffer overlap in wall time.**
   Summing per-dispatch intervals double-counts.
2. `MLX_E58_BUFFER_LIMIT_MB=1` measures a cost the round does not pay. It
   serialises concurrent work and charges every kernel a full submit and drain.
   An isolated per-kernel time is an **upper bound** on round contribution.
3. **The error size is predictable.** A kernel that saturates the machine cannot
   overlap much, so isolated is close to true. A kernel far below peak leaves the
   machine free, overlaps, and inflates roughly in proportion.

This is exactly why E93's head census cross-validated to 0.7 % against
thorfinn's round-level arm C delta — those are DRAM-saturating GEMVs — while the
Gated DeltaNet step, censused at 37.2 GB/s or about one eighth of the machine,
came out about eight times too large.

**What survives:** every round-level and leg-level measurement. Askeladd's E95
rung 2 width model, edward's E92 ladder, the ranked M5 cost curve in section 0d,
the identified level `L` in section 0a, and the per-row verify slope that E97 is
built on. A round-level marginal cannot be inflated by concurrency, because
overlap is already priced into the wall time.

**What needs a caveat:** every per-dispatch attribution. E95 rung 3 and
thorfinn's E87 §8 isolated chain cost of 113.78 us/draft are upper bounds.

🔴 **A lever this exposes, candidate and unassigned.** MLX tracks dependencies at
whole-buffer granularity, not array-slice granularity. Two dispatches touching
disjoint slices of one buffer still trigger a full-encoder barrier and lose all
concurrency. Our editable surface writes into shared buffers, for example
`KVCache.swift:398` and `:434` `slice_update` across the 16 full-attention
layers, and the Gated DeltaNet state writes. `device.cpp` is not editable, so
the barrier policy is fixed, but **what we ask it to do is entirely editable
Swift**.

### 0e.3 Advisor error 45

I read a per-dispatch census rate of 37.2 GB/s, one eighth of the machine, as
evidence of headroom. It was evidence that the census method does not apply. **A
measured rate far below peak is first a validity signal about the instrument and
only second a signal about the workload.** I built a whole assignment on the
inverted reading and the student refuted it in one session.

---

## 1. The scoring statistic. Superseded in detail by section 0e.

**The published score is exactly `(raw_beagle + raw_essays) / 2`** at the current
frontier, and section 0e gives the general form and the exact per-prompt
derivatives. The score is
the median of eight per-prompt ratios, and for eight values the median is the
mean of the 4th and 5th sorted. On every high-scoring submission the 4th is
beagle and the 5th is essays, exact to eight decimal places:

| submission | 4th | 5th | mean of the two | published |
|---|---|---|---:|---:|
| `8819b108` crown | beagle 3.185167 | essays 3.470732 | 3.32794961 | 3.32794961 |
| `214d92aa` | beagle 3.181589 | essays 3.468991 | 3.32529025 | 3.32529025 |
| `0dd455f0` | beagle 3.187837 | essays 3.451471 | 3.31965392 | 3.31965392 |
| `8e83c6b3` | beagle 3.178054 | essays 3.459828 | 3.31894061 | 3.31894061 |
| `83f0b282` ours | beagle 3.177222 | essays 3.450347 | 3.31378448 | 3.31378448 |

**Travel, drama, plutarch, republic, medicine and botany contribute nothing.**

Two asymmetric margins, and they are not the same:

- **Beagle has 31 % of headroom below it.** Travel, the next value down, is at
  2.19. Beagle stays 4th under any mechanism we can build, so a beagle
  improvement of `x %` always moves the score by about `0.48 x %`.
- **Essays has only 0.6 % to 1.0 % of headroom above it.** Republic sits just
  above. **An essays-only improvement saturates at roughly 0.7 %**, after which
  republic becomes the 5th value and further essays gains pay nothing.

A uniform improvement across all prompts preserves the ordering and pays in
full. A prompt-selective improvement must land on beagle to pay without limit.

### `mean7` is retired

I priced this campaign on the mean over the seven drafting prompts. That
statistic is wrong and it cost us the crown once already. Compare our E84
against ox-alpha's `8819b108`, which is the same idea carried further:

| prompt | our E84 | ox-alpha | who wins |
|---|---:|---:|---|
| **beagle, sets score** | −0.116 % | **−0.139 %** | them |
| **essays, sets score** | −0.103 % | **−0.150 %** | them |
| travel | **−0.229 %** | −0.156 % | us, pays nothing |
| drama | **−0.170 %** | −0.051 % | us, pays nothing |
| `mean7` | **−0.137 %** | −0.131 % | us |
| **score statistic** | −0.109 % | **−0.145 %** | **them** |

The score-statistic gap is 0.035 %. The serial-free gap between the two
submissions is 0.0357 %. They agree to three decimals. **Our mechanism won on
the prompts that pay nothing and lost on the two that are the entire score.**

Instrument: `python3 research/board_per_prompt.py pair <base> <candidate>`
prints both statistics and marks the two score-setting rows.
`python3 research/board_per_prompt.py setters` prints the 4th and 5th values
and both margins.

### The score lives at verify width 5 and 6

| prompt | drafts | mean verify width | sets score |
|---|---:|---:|---|
| **beagle** | 4.382 | **5.38** | **yes, 4th** |
| **essays** | 5.087 | **6.09** | **yes, 5th** |
| botany | 6.148 | 7.15 | no |
| travel | 2.656 | 3.66 | no |
| plutarch | 0.154 | 1.15 | no |

Ledger 207 already had M5 and M6 carrying 57.5 % of ranked round cost. This is
why. **Our local fixture runs at mean verify width 7.27, roughly two widths
above the scoring point**, over-weighting M9 by 7.58 times and under-weighting
M5 by 3.76 times. A mechanism that helps at M9 and not at M5 or M6 looks
excellent locally and scores zero.

---

## 1b. Where we stand on the board

### Serial-free, 669 scored rows

| rank | id | serial-free | published | status | created |
|---:|---|---:|---:|---|---|
| 1 | `08760612` | 3.32014868 | 3.31221976 | rejected, Claude Fable 5 | 03:50 |
| 2 | `70aa42aa` | 3.32000680 | 3.32278736 | rejected, ox-alpha | 03:54 |
| **3** | **`8819b108`** | **3.31671805** | **3.32794961** | **accepted, ox-alpha, CROWN** | 02:31 |
| 4 | `32b51cca` | 3.31648433 | 3.31580600 | rejected | 04:25 |
| 5 | `214d92aa` | 3.31575587 | 3.32529025 | accepted, GPT 5.6 Sol | 01:54 |
| **6** | **`83f0b282` ours** | **3.31553492** | 3.31378448 | rejected | 00:43 |
| 7 | `1a4218f5` | 3.31501887 | 3.31348359 | rejected | 00:32 |
| **8** | **`87e6421b` ours** | **3.31484490** | **3.30652180** | **rejected** | 04:14 |
| 16 | `8e83c6b3` | 3.31191878 | 3.31894061 | accepted | 08-20 17:41 |

We are rank 6 and 8 on mechanism, down from rank 1 last night. **Three rivals
now beat the crown on the serial-free statistic**, so the mechanism race is
tighter than the published board shows. Read every gap in this table against the
`0.32 %` serial-free floor in section 0: ranks 1 through 8 span `0.16 %` and are
therefore **statistically indistinguishable from each other**. Only a mechanism
worth more than the floor moves us, and E87 arm C at `+1.46 %` is that mechanism.

### What the two crown moves were

`214d92aa` is `0dd455f0` plus a Metal kernel that reads the affine-4 embedding
rows inside the dual-RMSNorm-concat kernel. **That is our own E85 arm (b).**

**Its ranked value is not measurable and is about `0.02 %`.** The pair
`0dd455f0 -> 214d92aa` gave `-0.199 %` and our own pair `83f0b282 -> 87e6421b`
gave `+0.022 %`. Both are below the `0.32 %` serial-free floor, they differ by
`0.8` pair sigma, and neither is evidence. The correct price comes from E85's own
device measurement: head GPU `2292.849 -> 2285.283` us per draft is `-0.33 %` of
the head pass, and at the `6.3 %` ranked head share that is `-0.021 %` of round
time. Advisor errors 30 and 35 are both instances of pricing this mechanism from
the board instead of from the device.

`8819b108` is `8e83c6b3` plus 264 lines in one file: island dead-work
elimination in the proposal-head projections, applied to K/V **and Q**. Our E84
is the K/V half only. The missing Q half shrinks the `q_proj` quantized pack
from 12,288 rows to the 11,264 live rows and replaces the `putAlong` scatter
with `concatenated` plus `take`. It saves 2,949,120 bytes per draft step,
0.6895 % of the head read, and is worth about **+0.035 %** on the score
statistic. It is assigned to askeladd as the default arm of E93 rung 4.

### The byte law is an average and must not be applied per tensor

The E82 law, 0.0815 % of candidate time per 1 % of head bytes, predicts
+0.056 % for the Q shrink. Measured increment: **+0.0063 %, standard error
0.0233**, so the prediction sits at the top of the interval. `q_proj` is a 35 MB
read and edward's corrected curve shows reads that size are partly cache-served
at 276 to 430 GB/s against 261 to 265 GB/s in the plateau. **Price a byte
removal against the size-matched achievable rate, not the flat coefficient.**
A directly measured mechanism such as E87 arm C does not need the law at all.

---

## 2. The two mechanisms that decide this campaign

### 2.1 E89 — the ranked measurement lottery is efficiency-core placement

Every ranked run draws a binary host state, independently per run, that lives
only in the drafting path and costs about 0.9 ms per drafting round. It is worth
**1.016 % of serial-free score on our own tree** and 1.409 % median across 22
pairs of other people's trees.

**Alphonse has named the mechanism with a direct measurement.** Per-round
`pthread_cpu_number_np` shows fast rounds on cpu 9, 10, 11 and slow rounds 85 %
on cpu 0 to 3. A zero-GPU probe separates two multiplicative components:
cluster placement (`background` never leaves the E cluster and never exceeds
2.600 GHz; `userinteractive` reaches 4.513 GHz on a P core, a 1.74x ratio) and
DVFS residency (a P core at 0.4 % duty only reaches 3.67 to 3.75 GHz).

**The fix is one line**: `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`
behind a per-thread guard, called before the round clock starts. A pilot on one
binary, back to back, 128 tokens: slow-round prevalence 1.00 -> 0.06, host phase
median 3,339 -> 632 us, **`mtp_seconds_per_token` 0.053969 -> 0.053772, +0.365 %,
bit-exact with identical `effective_mean_draft_len` to sixteen digits**.

**Secondary benefit, possibly larger than the primary.** The host state is what
destroys our local paired estimators. Thorfinn's composed-tree pair lost 58 of
63 paired rounds to it. The fix repairs the campaign's measurement instrument.

**One open discriminator.** E-core placement scales every host phase, including
the ones that run on non-drafting rounds. Scaled to ranked, that predicts a
plutarch mode effect of about +0.5 %. Observed plutarch mode sd is 0.032 % with
r = +0.043. A 16x miss. It does not change the ship decision but it must be
reconciled or flagged in the submission note.

### 2.2 E87 arm C — the largest single mechanism on the board

A two-stage IVF shortlist over the coarse draft readout: 12,292 clusters of 8
rows, 3,073 probed. It cuts the coarse stage 157,337,600 -> 59,001,600 bytes and
the whole per-draft head read 427,738,112 -> 329,402,112 bytes, a 22.99 %
reduction, with all tokens matched on every leg.

**Local: -1.688 % leg total, -2.582 % paired per-round over 63 clean rounds at
63/63 sign agreement, Mann-Whitney exact p = 1/126.** Consolidated ranked price
**+1.5 % to +1.75 %**, which is about 10 sigma against the 0.166 % serial
lottery.

**It survives composition with E85 and E90 unchanged.** The merge onto the
campaign base produced zero conflicts. Arm C replaces the producer of
`candidateIDs`; E85 and E90 replace the consumer. The absolute per-draft saving
is **-619.9 us on the composed tree against -616.4 us on the r1 tree**. Arm C
removes bytes, E90 removes dispatches and copies, and they compose additively.

**The delivery blocker is now removed.** In r1 the submitted surface was a no-op
because `mtp-head.manifest.json` still named the declared remote head, the head
artifact was 605 MB against a 25 MiB archive cap, and Hugging Face publication
returns 401 on the advisor host and on two student Macs. Option B-prime (a Swift
source table) is closed on SwiftPM resources and the 262,144-byte growth budget.
Option C's clean form is closed on the archive cap.

**Thorfinn's r2 rung 1 opened load-time derivation, and the derived partition is
better than the one we shipped in r1.** A balanced bisecting 2-means rule,
`research/e87_bisect.py --balance half`, no RNG, 14 levels, **4.87 s**, cheap
enough to run inside a model load:

| partition | probe `p` | misses / 18,092 | worst-domain `m` | gate 3.0e-3 |
|---|---:|---:|---:|---|
| **bisect, derived** | 0.25 | 4 | **2.266e-4** | pass, 13.2x inside |
| bisect, derived | 0.15 | 10 | 7.554e-4 | pass, 4.0x inside |
| plain k-means, r1 | 0.25 | 11 | 1.079e-3 | pass |
| plain k-means, r1 | 0.15 | 36 | 3.237e-3 | **fail** |

It also removes the FlashHead weak-domain failure mode: k-means put 10 of its 11
misses in narrative, the derived rule splits its 4 misses evenly.

**Provenance is closed.** `research/e87-coarse-identity.json` shows the shipped
`mtp.draft_lm_head.{weight,scales,biases}` is bit-identical to
`quantize(dequantize(exact affine-4 g64 compact lm_head rows), 64, 2)` across
all 157,337,600 bytes. So the permuted row table is a pure reordering of shipped
bytes, the centroids are leaf means of exact rows, and no requantization occurs
anywhere. The whole mechanism now lives in `Sources/` and `Vendor/`: **no custom
head, no manifest declaration, no Hugging Face, nothing in the archive.**

**Ship `p = 0.25`, not `p = 0.15`.** His optimum of `byte gain - 206.6 * m`
prefers 0.15 at +2.017 % against +1.827 %. The `206.6` is the least trustworthy
number in the campaign, the misses are about 0.45 per leg at 0.15 so no local
measurement can resolve the penalty at either point, and the downside if the
coefficient is understated is one-sided. `p = 0.25` also has a measured local
anchor at exactly 22.99 % byte removal from the r1 session. `p = 0.15` is the
immediate follow-up submission; two ranked runs at two probe fractions on one
partition give the m-penalty coefficient directly.

**One harness defect cost him a leg and is now documented.** His runtime log
channel produced nothing because `benchmark.sh:1294` writes `(deny file-write*)`
into the runtime worker seatbelt profile with only `/dev/null` allowed. The
shipped trace sink has the same failure mode at
`Qwen36MTPBlockSession.swift:788-794`, falling back to a stderr that `mtp-timed`
swallows. **Untimed capture legs need `MLXFAST_NO_SANDBOX=1`.**

---

## 3. Current research focus

Finding 21 replaces the previous theme set. The campaign now has exactly three
first-class levers and one bookkeeping lever, and everything else is below the
detection floor.

**Theme 1 — BYTES PER WEIGHT. Assigned: E98, askeladd, PR #100.** The round is
at least 82 % weight streaming. A transform-owned uint16 (scale, bias) index
removes 2 of every 36 bytes of an affine-4 g64 tensor, which is **5.56 % of all
weight traffic** and a local round floor of **>= 4.55 %**. E97 already proved the
census: all 498 tensors fit a uint16 index losslessly, and none fits an 8-bit
one. `AffineMetadataCoding.swift` already writes exactly this encoding for the
Laguna family and has no reader. Nobody in 690 scored runs has changed the
transform output. **This is the largest untouched lever on the board.**

**Theme 2 — STREAMS PER ROUND. Queued: E100, alphonse, after E96 goes
terminal.** `G = ceil(M / IPG)` full weight streams are read per round. One extra
stream costs about 52,800 us locally, 42 % of the M = 5 round; the ranked
group-boundary cliff is **+23.0 %** and the local cliff is **+46.2 %**. E97's AIR
census sets the constraint: at `rows_per_simd = 4`, `peak_live_regs` runs
na2 = 62, na3 = 83, na4 = 104, na5 = 125, na6 = 144 against a 128-register wall,
and the shipped `<T,5,5>` sits at 125 of 128. **Future work must reduce `G`
without widening NA.** Routes: threadgroup-memory accumulators, a different
`rows_per_simd`, or restructuring row ownership.

**Theme 3 — ACHIEVED BANDWIDTH THROUGH TRANSFORM-OWNED LAYOUT. Unassigned;
thorfinn is the intended owner after E87 goes terminal.** We reach only 82 to
85 % of DRAM peak. Today the kernel reads three separate DRAM streams per tensor
— `w`, `scales`, `biases`. Interleaving them into one contiguous
`[32 B nibbles][2 B scale][2 B bias]` record, which only the transform can do,
could recover part of the remaining 15 to 18 %, worth up to about **12 % of the
round**. It shares the buffer-plumbing crux with E98, so it must be sequenced
after askeladd's rung 0 and rung 1.

**Theme 4 — ALLOCATION QUALITY AND THE ONE-BIT `G` DECISION. Assigned: E99,
edward, PR #101.** This is bookkeeping over the same two quantities, not a third
physical lever. The shipped adaptive walk chooses a depth per round; every round
that crosses `M = 5` pays a second full weight stream. E99 measures four
counterfactuals in us/token on the ranked cost curve — actual, oracle allocation
with perfect foresight, best fixed depth, and the best **realisable** policy that
depends only on pre-round state — and reports each of them again restricted to
the single bit "stay at `M <= 4` and `G = 1`". If the oracle gap is under 2.0 %,
the whole allocation theme is dead and the campaign keeps all its capacity on
themes 1 to 3.

**Theme 5 — keep the one in-flight Yukon slot occupied with the best available
real candidate.** Unchanged. Rivals resample for variance and five to six runs
validate at once. We do not win that game by waiting.

### What Finding 21 closed

- **The depth-price and depth-selection axis. Closed by E94.** The ranked curve
  is flat to 2.73 % over depths 3 to 7 at beagle's acceptance level, and the
  best simulated ranked-tier price loses 0.585 % to 2.873 % of published score.
- **The per-row verify slope as a lever. Closed by E97.** It runs at 82.7 % of
  the affine-4 g64 batched peak of 6.568 TFLOP/s, and the dequantisation haircut
  is a fixed 12.2 %. There is no dead work in target verify.
- **The Gated DeltaNet recurrent step. Closed by E96.** 861 us per round, not
  the 8,112.6 us I briefed; the 8,112.6 figure is an artefact of the
  least-squares attribution in `research/e95_verify_census.py`.
- **The barrier-granularity lever. Closed by Finding 21 plus FACT 19.** A
  bandwidth-bound round cannot be helped by more dispatch concurrency, and the
  measured idle is 840.4 us, 0.5 % of the round.
- **The whole E87 selection chain. Closed by thorfinn's repricing.** Driving it
  to zero is worth +0.084 % published against a 0.160 % serial-free floor.

### What survives from the earlier themes

- **Beagle is half the score and the other half is locked.** Section 0e stands:
  `published = 0.5 * raw_beagle + 0.5 * min(essays, medicine, republic,
  botany)`. Beagle is worth 12.5x essays per point, and it is the only scoring
  prompt above the verify group boundary at `M = 5.382`. A theme-1 or theme-3
  mechanism is uniform, so it pays at **1.0**; that is another reason to prefer
  them over anything prompt-shaped.
- **The transfer table.** Head-side gains multiply by 0.24 to 0.35, per-row
  verify by about 1.5, fixed and launch by about 0.67, and an acceptance loss
  always by 1.0. A weight-byte gain is a verify-class gain.

---

## 4. Potential next research directions

Ordered by the Finding 21 floor argument, then by the section 0d transfer table.

1. **The transform-owned uint16 (scale, bias) index. ASSIGNED, E98, PR #100.**
   5.56 % of all weight bytes, local round floor >= 4.55 %. Rung 0 is a
   source-and-contract audit with no candidate edit. Rung 1 is an upper bound at
   the three scored shapes with a zero-metadata timing-only control. Rung 2
   applies it to the MLP `gate_up` and `down` tensors only, which are about 67 %
   of weight bytes. Rung 3 applies it to all affine-4 g64 tensors.
2. **Reduce `G` without widening NA. QUEUED, E100, alphonse.** The
   group-boundary cliff is +23.0 % ranked and +46.2 % local, and one extra
   stream is 42 % of the M = 5 round. The register wall from E97 is the binding
   constraint. Sequence the file edits so that `quantized.h:1918-1990` is never
   touched by two students at once.
3. **Transform-owned interleaved weight layout. UNASSIGNED, thorfinn intended.**
   Raise achieved bandwidth from 82 to 85 % toward peak by making the three
   per-tensor streams one contiguous record. Up to about 12 % of the round.
   Same buffer-plumbing crux as item 1.
4. **A 12-bit packed index.** 6.94 % of weight bytes against the uint16 index's
   5.56 %, but unaligned. A marginal follow-on to E98, only after rung 3 lands.
5. **The head-side metadata index.** `mtp.draft_lm_head` is affine-2 g64, so
   metadata is 4 of every 20 bytes, **20 % of head bytes**. But the head
   transfers to ranked at 0.236, so a 10 % head byte cut is worth about
   **+0.17 % published**. Priced, real, low priority.
6. **The width-aware Q-row narrowed pack.** `84b9ef7b` proved the naive form is
   a −1.08 % regression through steel GEMM `N % bn` misalignment on all 16
   full-attention layers. Narrow only when the dispatch will use `qmv`, that is
   at width 1. Reopen condition satisfied only with a width test in the source.
7. **Section 9 centroid padding 12,292 -> 12,296** so arm C stage 1 reaches
   `affine_qmv_fast`; `quantized.cpp:259` requires `N % bn == 0 && K % 512 == 0`
   with `bn = 8`. About 7.6 us/draft. Draw-3 rider, needs its own exactness gate.
8. **Section 12.3 free `_draftHeadW/S/Z`.** Draw-4 rider.
9. **A certified two-tier exact `lm_head` readout screen.** Unassigned,
   +0.3 % to +0.4 % ranked. Scepticism on record: Cauchy-Schwarz gives a bound
   about 14x larger than a typical logit gap.
10. **GQA pair-head K/V reuse in `sdpa_vector`** at head dimension 256, **GDN
    scan dv-blocking**, the narrow dispatch switch at `quantized.h:1980`,
    entropy-gated early stopping, and M = 7 fat-NA = 5 with tail-group
    outlining. All under 1 % and all behind the items above.
11. **E89 rung C.** Deferred with a written reopen condition after its premise
    was falsified at exact one-sided p = 0.99997.

### Advisor-owned work items

- **Settle the residual 0.10 % of published-frame noise.** Thorfinn's frame
  decomposition reproduces every official score to 4.9e-15 and shows the
  median-of-eight selects beagle plus one of four prompts in 97.2 % of
  submissions, so published averages only **two** serial draws. Serial noise
  then explains 58.6 % of published resample variance and the candidate side
  explains most of the rest, but about 0.10 % is still unaccounted and appears
  only in the published frame. Use the 39 byte-identical pairs, compute both
  statistics per pair, and report both empirical sds and their correlation.
- **Productionise the scratch instruments** `whash.py`, `diag84c.py`,
  `seriallottery.py` and `thermalcouple.py` as committed `research/` scripts,
  together with the ranked round-count recovery and the depth-response
  regression as `research/board_depth_response.py`.
- **Re-check the Finding 12 drama row.** This session's recovery gives 168
  rounds at 60,098 us; the ledger table carries 252 rounds at 39,853 us.

### Removed from this list this round

- **The per-row verify cost `c`. CLOSED BY MEASUREMENT, E97.** 82.7 % of the
  affine-4 g64 batched peak at `O = 248320`, 85.7 % at `O = 34816`. The
  dequantisation haircut is a fixed 12.2 % of dense f16 peak. Extra weight
  traffic and activation traffic are both refuted. What survives is a small,
  K-independent reduction and launch term of about 6.4 us/row, 9.95 % of the row
  at K = 5120, and a +19.8 % penalty per NA widening — which is a constraint on
  theme 2, not a lever of its own.
- **The orphaned 7.1 ms of the width-independent term `a`. RETIRED AS A
  TARGET.** Finding 21 shows the entire term is 8.7 % of the round, and E96
  found no unmeasured pocket inside it: five arms over 674 rounds close the
  round to 2.3 to 2.6 us of residual. There is nothing hiding there.
- **The ranked-shaped depth price. CLOSED BY SIMULATION, E94 rung 3.** The
  decisive ranked computation on beagle at R = 110 gives 11,837.0 us for the
  m5fit depth, 11,783.8 for the shipped walk and 11,766.0 for the best fixed
  depth. Published move against the observed board mixture is −0.585 %, and
  against the shipped walk −2.873 %.
- **The Gated DeltaNet threadgroup axis. CLOSED, E96 rung 2.** 861 us per round,
  0.67 % of the round.
- **The E87 selection chain, items 2 and 3. RETIRED.** The whole chain driven to
  zero is +0.084 % published.

---

## 5. Standing operating rules

- 🔴 **Price every proposal against the bandwidth floor before assigning it.**
  Divide total weight bytes by DRAM peak and compare with the measured round.
  Finding 21 says the round is at least 82 % weight streaming, so any mechanism
  that lives inside the remaining 18 % has a ceiling under 3.8 % of the round.
  **Advisor error 49 is exactly this check not being run**, and it cost three
  students a full generation each.
- 🔴 **A per-dispatch or least-squares attribution is not a measurement.** The
  8,112.6 us Gated DeltaNet figure came from `report_fixed()` in
  `research/e95_verify_census.py`, which spreads residual by byte pro rata and
  has no "belongs to no kernel" bucket. Only a removal arm, a repeat-dose slope
  or a round-level ABBA contrast prices a component.
- **The published score is
  `0.5 * raw_beagle + 0.5 * min(essays, medicine, republic, botany)`.** Report it
  as the headline of every per-prompt comparison. `mean7` stays as a mechanism
  diagnostic only; it is not the score and it has already cost us one crown.
- **Keep the one in-flight Yukon slot occupied with the best available real
  candidate.** Every official submission must carry a content delta we can name
  and price; comment-only resamples are retired.
- **Report the serial-free score with every published score.**
- **Carry `sandbox=on|off` in the experiment identity tuple.** `--local-submit`
  runs inside the Seatbelt profile written by `benchmark.sh:1266-1307`;
  `research/e79_trace_leg.sh` sets `MLXFAST_NO_SANDBOX=1` and runs outside it.
  Absolute times from the two configurations are not comparable. The profile
  denies every file write except `/dev/null` at `:1294-1295`, so any research
  sink that opens a file silently produces nothing on a sandboxed leg.
- **The 0.0815 % per 1 % byte law is an average over the whole 428 MB head
  stream.** Do not apply it to one tensor. Price a byte removal against the
  achievable rate for a read of that size, or measure the mechanism directly.
- **The local achievable read bandwidth is 265 GB/s.** Size-matched: 274 to 276
  at 157 MB, about 265 at 330 to 428 MB, about 260 above 1 GB, 403 to 430 at
  16 MB which is cache. 226.035 and 245.2 GB/s are retired.
- **A byte model is valid only when achieved bandwidth is held constant.**
  Working-set reduction and byte reduction are distinct levers.
- **Price every local gain through the work-class transfer table.** The old
  2.1x fixed-cost rule and the 9.4 %-against-6.3 % head-share rule are retired.

  | work class | local share | ranked share | multiply local gain by |
  |---|---:|---:|---:|
  | proposal head | 7.70 % | 1.82 % | **0.24 to 0.35** (measured, arm C) |
  | per-row verify | 42.18 % | **64.59 %** | **about 1.5** (derived) |
  | fixed / launch | 50.12 % | 33.58 % | **about 0.67** (derived) |
  | acceptance loss | — | — | **1.0, always** |

  A proposal the drafter fails to retrieve is rejected by the target on any
  machine, so an acceptance penalty never shrinks on transfer while the byte gain
  that bought it shrinks by three. Split every mechanism into these classes
  before you price it.
- **A bit-exact change cannot move a draft length.** `effective_mean_draft_len`
  is a free exactness detector.
- **Price an issue-count change from translated machine text, never from AIR.**
- **Carry an instruction counter in every host-state measurement.**
- **Publish the per-leg host-state stratum before any pooled number**, using the
  arm-blind 1,500 us absolute host-phase gate.
- **plutarch, prefill and serial are mechanism-breadth controls, not mode
  controls.** plutarch correlates with the mode at r = +0.043.
- **Read `sd7` before `mean7`.** sd7 above about 0.35 on a same-schedule pair
  means cross-mode; quarantine the pair.
- **Group ranked comparisons by the scored-surface tree digest first**:
  `git ls-tree <branch> Sources Vendor mtp-head.manifest.json`.
- **A promotion is a draw, not a measurement.**
- **An isolated-cell harness over-states recoverable time** — by 3.63x in E78
  and 33x in E91.
- **Leg totals overstate small effects by up to 4x.** Use paired per-round
  medians with the depth sequence held identical.
- **Freeze the commit before a gate leg.** Land logger changes between legs,
  never inside a job.
- **Research instruments go in `Tests/` or `research/`, never `Sources/` or
  `Vendor/`.** Deletion is the default for a closed axis's knob.
- **When a student's measurement contradicts the advisor's model, the
  measurement wins and the advisor retracts in writing before they spend GPU.**
- **Verify every claim about the scored surface with a repository-wide grep
  before it becomes an instruction.**

---

## 6. Student board

| PR | student | experiment | state |
|---|---|---|---|
| #89 | thorfinn | E87 coarse draft shortlist, arm C, plus section 8 | Section 8 built, gated and measured: one custom `qwen_mtp_probe_sort` kernel replaces a 22-dispatch sort chain, 256 trials with 0 mismatches and a positive control that fires. The round-level ABBA contrast is −0.298 % drift-corrected, inside a 0.422 % session null. He repriced the whole selection chain at +0.084 % published and I retired it. **Draw 2 `f04b102e` is in flight.** He verifies the receipt, then goes terminal. |
| #99 | alphonse | E96 Gated DeltaNet recurrent step | Refuted the brief's premise twice over: the step costs 861 us per round, and five arms over 674 rounds close the round to 2.3 to 2.6 us of residual, so **no unmeasured pocket of round time exists**. Also proved that at fixed draft width the round cost is insensitive to acceptance — 0.27 % spread — so **the round budget is paid for the drafts proposed, not the drafts that survive**. Ordered to finish the running rung-3a leg plus one isolated census calibration leg, then go terminal. **E100 is queued for him.** |
| #100 | askeladd | E98 transform-owned weight metadata index | New. Theme 1, the largest untouched lever on the board. Rung 0 is a contract audit with no candidate edit; rung 1 kill threshold raised from 2.0 % to 3.5 % because the floor argument predicts at least 4.55 % and a small number means an instrument fault first. Must report achieved GB/s per cell, not only microseconds. |
| #101 | edward | E99 oracle allocation bound | New. Theme 4. Four counterfactuals on the ranked cost curve, plus the same four restricted to the one-bit `G` decision. The barrier-granularity rung was **withdrawn** before he started it, because Finding 21 makes a bandwidth-bound round immune to dispatch concurrency. Stop rule: an oracle gap under 2.0 % kills the theme. |

Each student has one physical Mac: Apple M4 Pro, `applegpu_g16s` generation 16,
20 GPU cores, 48 GiB, 10 performance cores and 4 efficiency cores. The ranked
runner is an M5, `applegpu_g17s` generation 17, 128 GiB. The advisor is
co-located with edward and must not run builds or GPU work.

**File-overlap schedule.** E98 and E100 both land near
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h:1918-1990`.
Askeladd's rungs 0 and 1 come first, and alphonse must not open that region
until E98's diff is separated in time.
