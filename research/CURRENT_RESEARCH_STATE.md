# SENPAI Research State

- **2026-08-20 08:00 UTC**
- Track `qwen3.8-27b-mtp-v1`; advisor branch `senpai/qwen38-mtp-r1`;
  `BASE_SHA = 0f1cd9b5cb45182290a43d1eaa134c21b1c6efcc`;
  `UPSTREAM_SHA = 0c90733d383f6b987a29682bf9eb9458a6172bfa` — the value actually synced into
  this tree. The organizer tip is `9e1ff9ec7152a04b753f2efb91c3e559909ea4b9`; the trusted
  delta between the two is **empty**, so a future sync has no contract work to do, only
  editable cherry-picks.
- Most recent human research direction: **issue #22 — execute aggressively toward the winning
  frontier — is the standing directive.** Issue #31 (maintenance checkpoint) is complete and
  closed. No new human direction this round.

This is a **living document**, pruned every round. The per-experiment record of authority is
[`../senpai/campaign-ledger.md`](../senpai/campaign-ledger.md) (10529 lines); durable
measurements and source-line citations live in
[`ESTABLISHED_FACTS.md`](ESTABLISHED_FACTS.md). The predecessor of this file is retained as
[`RESEARCH_STATE_ARCHIVE_2026-08-19.md`](RESEARCH_STATE_ARCHIVE_2026-08-19.md).

---

## Where we stand

| quantity | value |
|---|---|
| live promoted frontier | **3.24985583421771** — submission `59b321ee`, solver fkiene, commit `9e1ff9ec` |
| our best official submission | **3.23250848263467** — receipt `ca9251b8`, candidate `2b0c36a0`, rejected |
| our deficit | **0.01734735158304 = 0.5367 %** |
| ranked jitter, per prompt per leg | **0.2257 %** (n = 408, serial leg) |
| rel sd of the published median of 8 | **0.1415 %** (worst case 0.2636 %) |
| **ranked MDE at 2 sd** | **+0.283 %** (worst case +0.527 %) |
| local end-to-end null floor | **0.0629 %** |

Nothing we have submitted has been promoted. The deficit is **candidate-leg overhead at a
frozen accept trajectory**: our receipt and the frontier's report a **byte-identical
`mean_draft_len` 8-tuple** and a byte-identical head digest, as do all six trees at the top of
the board. We are slower on 8 of 8 prompts; median candidate seconds/token 11.7713 ms versus
11.7277 ms = **+0.372 %**, and the gap is ordered by draft depth. **Not one diff at the top of
the board is a scheduling or head change.**

Only two prompts convert speed into score. `beagle` sits in the central pair in **98.5 %** of
resamples and `medicine` in **49.3 %**. `beagle` carries **79 %** of our deficit and has 7.9 %
headroom; a `medicine`-only gain saturates after 0.64 %. That is **scoring geometry, not
permission to specialise** — the hidden pool is resampled and any per-prompt tuning is a
benchmark escape.

---

## 🔴🔴🔴 What changed this round: the score model itself

### The three-factor exact identity (ledger 186(B))

The 512-token seed prefill is **inside the timed leg**. `QwenRuntimeMTPDriver.swift:94` starts
the clock, `:95` runs the prefill, `:197` stops it; `QwenRuntimeMTP.swift:347-349` states
`seedPrefillSeconds` is observability only and is **deliberately not subtracted**. The driver
comment at `:90-93` says so in words: *"the clock starts immediately before the request so the
seed cost cannot be hidden outside the window."*

Therefore, to within ±5e-11 on all eight hidden prompts:

```
raw_p        = build_factor x spec_factor x dilution_p
build_factor = serial_spt / c1                     uniform 1.2464..1.2508
spec_factor  = (512 / R_p) x c1 / round_ms_p
dilution_p   = 1 - K / leg_p,   K = 512 x prefill_seconds_per_token
```

Observed `K` = 525.7..528.2 ms (spread 0.46 %). On the two prompts that set the median the
prefill is **8.44 %** (`beagle`) and **9.05 %** (`medicine`) of the timed leg.

🔴 **Median-pair round-cost conversion is ×0.9125, not ×1. Every round-cost score projection in
the ledger before this round was 9.6 % too high.** Halving ranked prefill would be worth
**+4.373 % of score** — a huge number, and unreachable (below).

Ledger item 122 had concluded "prefill optimisation is worth exactly 0.000 % of score." That
test compared `raw = serial/mtp` against `raw = (pf+serial)/(pf+mtp)`. The second form **charges
prefill twice**, because `mtp_spt` already contains it. The test could not discriminate, and
refuting the double-charged form is not evidence for the negation of the single-charged one.
Item 122 is retitled with a correction banner.

### 🔴 THE TRANSFER LAW (ledger 186(D))

> The more a cost term is arithmetic-bound, the better it transfers to the ranked M5, and
> therefore the **LESS** a local reduction of it is worth at rank. A local win on a
> compute-bound or memory-traffic-bound term must be divided by up to **3.55**. A local win on
> a latency-bound or dispatch-bound term transfers at 1:1 or better, because the ranked leg is
> 2.9× shorter in wall time while host-side per-dispatch cost is roughly host-independent.

| work | M4 Pro (ours) | ranked M5 | advantage |
|---|---|---|---|
| 512-token seed prefill (84 % GEMM) | 3.9938 s | **0.5269 s** | **7.58×** |
| depth-0 decode round | 65.009 ms | **30.402 ms** | **2.14×** |

Ratio of ratios = **3.55×**. Four independent supports:

1. The table above.
2. The identified depth slope **`g ∈ [0.7388, 0.7778]`** — marginal drafting cost on M5,
   relative to its own depth-0 round, is **22–26 % cheaper** than on M4 Pro. A single-factor
   transfer is refuted **calibration-independently**: the joint `c1` band is empty by 10.12 %.
3. E27 cut QMV weight passes — a pure memory-traffic win. **−6.56 % locally**, then **−0.33 %
   of published score**. A 6.75-point sign flip, in the direction the law predicts.
4. Had prefill transferred at the round scale it would be 1868 ms, near the identified
   `K <= 1810 ms` feasibility edge; observed 526.6 ms is 3.55× smaller.

**Standing rule: label every local measurement compute-bound or latency-bound before converting
it to a ranked score delta. An unlabelled conversion is invalid.**

### The prefill is scored but UNREACHABLE (ledger 186(C))

On `2 × 27e9 × 512 = 2.765e13` FLOPs the ranked host runs prefill at **52.47 TFLOP/s**; we run
it at **6.92** against our own measured dense-bf16 ceiling of **7.401** — i.e. we are at
**93.5 % of our own ceiling**. That is the `qmm_nax` signature (`quantized.cpp:473`, requires
`is_nax_available()`, GPU generation ≥ 17). Our host is `applegpu_g16s`, generation **16**.

**The ranked prefill executes a kernel family we cannot run, measure, or tune.** E16's closed
M4 Pro prefill budget describes work the ranked machine never does. **Reopen only with a
generation ≥ 17 host.**

---

## Current research focus and themes

### 1. 🔴 NEW TOP DIRECTION — host-side dispatch latency (E58, alphonse)

E57 priced a GPU dispatch at roughly **22 µs** of end-to-end candidate leg time (Arm B added
2226 SDPA dispatches for +0.27 % seconds/token). I recorded that as "not the local bottleneck"
and moved on. Under the transfer law that was a pricing error.

The shipped base fires **6163 SDPA dispatches** per leg ≈ **136 ms**:

- against our local 18.35 s leg: **0.74 %** — noise-adjacent, which is why I dismissed it;
- against the ranked `beagle` leg of 6233 ms: **2.2 %**;
- after the ×0.91552 dilution: **2.0 % of published score ≈ 7.1 sd**.

And SDPA is **one** family. The round also dispatches quantized matvec and matmul, GDN
recurrence and scan, normalisation and RoPE, elementwise and copy, top-two and readout, and the
proposal head. **Nobody in this campaign has ever counted the total.** If the total is a few
hundred per round, this is a double-digit percentage of the ranked leg sitting in host overhead
that every rival at the top of the board is ignoring while they tune arithmetic.

E58 rung 1 is the census: dispatches per round by kernel family and by width, plus a serial
control, plus a defended µs/dispatch range, projected onto the ranked `beagle` and `medicine`
legs. It is terminal and mergeable alone. Rung 2, only if rung 1 clears ~2 % of ranked leg, is
`MLX_MAX_OPS_PER_BUFFER` `{50, 100, 128, 256}` at fixed 512 MiB — a 2-line change to
`RuntimeStartupMemoryPolicy.swift`, bit-exact by construction, in the file with the **lowest
failure rate in the 712-tree corpus (20 % versus a 42.7 % baseline)** and the file where rival
`646a3dee` got the fastest candidate leg of board ranks 13–18 from two lines.

### 2. The QMV width table — re-priced, still the largest single prize

`M = 9` alone is **53.45 %** of candidate-leg QMV time on the local fixture. A 2-stream `M = 9`
that stayed at ≤ 108 registers was priced at **+5.36 % of score**. Re-pricing through this
round's corrections:

```
+5.36 %  ->  x g (0.7388..0.7778)  ->  +3.96..4.17 %  ->  x 0.9125  ->  +3.61..3.81 % of score
```

That is **12.8–13.5 sd** against the +0.283 % MDE. It remains the biggest number on the board.

⚠️ **But it is a memory-traffic win**, so under the transfer law the local effect must be
divided by up to 3.55 when projecting. The `g` and dilution factors above already do part of
that work; do not double-count, and do not treat the local `−6.56 %`-style number as ranked.

The route: E27 moved `M = 5` and `M = 9` from IPG 3 to 5, won both cells locally, lost 0.33 %
of score, and moved the kernel-wide register max **108 → 129**. The step is the problem, not
the stream reduction. Lowering `rows_per_simd` directly is a **correctness wall** — the frozen
host grid writes 8 rows per `tid.y`, so `r < 4` leaves half the rows unwritten. The replacement
(ledger item 99, never built) covers the same 4 rows as **`4/r` sequential row blocks**, making
registers live-range-bound so the peak follows the `r = 2` line: `M = 9` at IPG 5 would be
`16 + 15·5 = 91` per block against a kernel-wide max of 108 set by `<T,7,4>`, so **the ceiling
would not move**. Known cost is the x re-read per block, measured **+10.54 %** at NA = 4 against
an `M = 9` break-even of **12.43 %**. **Hard gate: the register census must read 108, not 129.**

**Law D is refuted** (ledger 183(C)): E32's register ladder is affine in NA (`r=2: 16 + 15·NA`;
`r=4: 20 + 21·NA`, max residual 0.25 registers). What survives is the **shared-ceiling term**,
because `M` is **absent from the QMV pipeline identity** — one library, one pipeline, one
register allocation for all `M = 1…9`, and the escape hatch is closed because a distinct kernel
name would need `backend/metal/quantized.cpp`, which is **not** in `editablePaths`.

### 3. 🔴 The depth schedule is unexploited, and M5 makes us UNDER-draft

`g < 1` means the M4-Pro-derived depth ladder **over-prices depth on M5 by 22–26 %**. Relative
to its own depth-0 round, marginal cost 3→4 is **0.2845 on M5** versus **0.3753 on M4 Pro**.
The shipped `costModelDepth` therefore **systematically under-drafts at rank**.

This also explains a result I could not previously explain: the local falsification arm that
raised `h` from 0.18 to 0.32 — i.e. drafted *shallower* — measured **+0.95 %** locally. Local
tuning actively **mis-ranks** depth policy, because the M5 optimum is *deeper* while the local
optimum is *shallower*.

All six top trees ship **byte-identical** `costModelDepth`, `h = 0.18`, every width constant,
and a byte-identical `mean_draft_len` 8-tuple. **The depth axis is unexploited at the top of the
board, not proven dead.** Edward has this on #59.

Caveat that has burned us: the depth simulator over-predicts magnitude by **8–30×** (simulator
+18.9/+30.1 % versus machine +0.95 %). The honest shrunken range for the shipped shape change is
−0.12 % to −0.47 %. Never ship a depth claim on simulator output alone.

### 4. The narrow-width cost split — still the central open disagreement (#57)

askeladd's E48 and edward's E53 agree on the total and roughly swap the split:

| widths | E48 | E53 |
|---|---|---|
| `M ∈ {4,5,6}` | 64.025 % | 65.0–68.9 % |
| `M ∈ {7,8}` | **9.391 %** | **21.2–25.1 %** |
| `M = 9` | **21.630 %** | **4.6–8.9 %** |

Neither is a GPU measurement. Sensitivity is **8.49751 % of MTP leg per unit `f9`**, so this
gap is the difference between a headline and a null. PR #57 settles it with a real per-width
measurement.

🔴 **No moment-based method can close this from the receipt.** Ledger 184(D): under the
identified two-parameter transfer the `M = 9` share bound on `beagle` is `[0, 70.34 %]` and on
`medicine` `[0, 67.12 %]`, and the local fixture's 53.45 % lies inside both. It needs a
scheduler-faithful simulation or a per-round width trace. **Do not price a width-mixture claim
off the local fixture** — the fixture's mean draft length is 6.269, **deeper than every hidden
prompt** including `botany` at 5.776, and far deeper than the median pair (4.53, 4.77).

### 5. Acceptance regimes: the shipped prior is wrong at the shallow end

Ranked ground truth per-draft acceptance: plutarch 0.333, drama 0.449, travel 0.533,
**beagle 0.835**, **medicine 0.875**, republic 0.902, essays 0.900, botany 0.870. The shipped
`positionAcceptEMA` seed of `0.85 · 0.98^i` is roughly right for the deep prompts and badly
wrong for the shallow ones. The shallow prompts barely matter to the median — but the seed also
governs the first rounds of the deep ones, and the ranked legs are only 85–107 rounds.

### 6. ✅ CLOSED this round: the SDPA route direction

E57 settled it and the answers went against my hypothesis:

- The deciding gate is `use_fallback` at `scaled_dot_product_attention.cpp:591-639`, **upstream**
  of the `:685` selector I had been reading. `supports_sdpa_full` needs head_dim ∈ {64, 80, 128}
  and ours is **256** ⇒ false at every width. `supports_sdpa_vector` needs `qL·gqa <= 32` with
  **no `kL` term** ⇒ the fused path ends at **`qL <= 5`, unconditionally.**
- ⇒ The shipped `qL >= 6` chunk predicate is **correct**, and the chunk is a **discount**: it
  keeps both halves fused at 4–6 dispatches instead of the 8-dispatch composed fallback.
  Removing it **adds ~64 dispatches/round and fails correctness**. My cost estimate had the sign
  backwards. Keep the chunk exactly as shipped.
- ⇒ The `qL >= 9` steel/`_nax` route **never fires**; `steel_attention*.cpp` editability is
  irrelevant.
- ⇒ `kL >= 1025` is **unreachable** (512 + 512 caps `kL` at exactly 1024, reached in 1 round of
  76, at `qL = 4`). The frontier's `warmTargetLaterWindowSDPA` **warms the only reachable
  boundary variant** — my 182(E)/183(E) "warmed the wrong pipeline" claim is **refuted** and the
  extended-warm proposal is deleted.

`research/SDPA_ROUTE_MAP.md` now carries a dated correction banner and per-section verdicts.

🔴 **One live risk this exposed**: the shipped base itself declared two distinct top-two tuples
at positions 1022 and 1024, in the single round that reaches `kL = 1024`, at `qL = 4` with no
chunk. The ranked window **always** reaches that boundary. Latent near-tie exposure on hidden
prompts, in code we ship today.

### 7. Warm coverage: compose-only, and warming is free

Three distinct warm cost units that must never be summed: **JIT source compile**, **metallib
specialization**, **plain pipeline create**. A quantity costs a pipeline only if it appears in
the kernel name, library name, or `hash_name`.

Warm is **untimed** (`QwenRuntimeMTPDriver.swift:84`, "Untimed phase start, BEFORE the clock").
That retracts 183(E)'s claim 3 — the fourth self-contradiction in four ledger items on this
point. 183(E)'s **conclusion** survives on other grounds: there is no per-width QMV pipeline,
and the seven per-width source libraries (`S = 3…9`) are all already compiled by our warm loop.
**Warming is free; the demotion holds only because nothing is left un-warmed.** Do not assign
removal of the 512-zero seed warm.

### 8. Gated DeltaNet: the scan is not the cost

GDN is 25.88 % of verify-side work; ~91 % of its bytes are its three quantized projections, not
the scan. The scan grid is **independent of `T`** ⇒ KVBuffer-style deferred commit is refuted
for our tree. Decode runs at ~73 GB/s against an M4 Pro capability of ~273 GB/s, so decode is
**latency/occupancy-bound, not bandwidth-bound** — which is itself an independent argument for
direction 1.

### 9. The proposal head: the open lever is the shortlist

`h = 0.18` decomposes to ~22 % head / ~78 % verify, and 84.4 % of the depth-8 marginal is verify
width. Head swaps have **never won outright** (58 corpus attempts, best 3.2130). The unexplored
levers are the shortlist constants — `qwen35Top32K`, `draftRerankCandidateCount`,
`compactDraftPaddedCount = 98_336` — and the free precision-island A/B via
`MLXFAST_QWEN_MTP_EXACT_QKV_ROWS`, never swept.

### 10. Measurement discipline that now gates every claim

- **Prefill is inside the timed leg.** Multiply every round-cost score projection by ×0.9125.
- **Label local measurements compute- or latency-bound before converting to score.** Divide
  compute-bound wins by up to 3.55.
- **Refuting hypothesis B is not evidence for the negation of hypothesis A.** Before publishing
  a null, write down the hypothesis that would survive the test and check it is the one you
  meant to reject. (This cost us item 122 for a whole round.)
- **Grep the ledger before publishing a cost claim.** Broken again in item 186.
- **An instrument that cannot fail is not an instrument.** Every gate needs a positive control.
- **A green `--local-iterate` parity line is NOT exactness evidence** — E51 had 52 of 64 rows
  moved with parity green; E57 Arm B had 396 of 512 positions moved with `all_tokens_matched=1`.
- **Moved-position count is a presence detector, not a severity order.** Report the margin
  distribution and the worst margin. One position at margin 0.125 outranks 256 at 1 ulp.
- **Board intervals are identification intervals, not standard errors.** A +0.0173 % board step
  is 0.12 median sd and cannot price a mechanism.
- Ungated timing only ABBA-counterbalanced, with entry/exit temperatures recorded and
  `cool_gate_passed_real_gate=false` / `gate_qualified_for_timing=false` preserved verbatim.
- **Log W&B per leg while timing, never at session end.**
- 🔴 **`MLXFAST_*` env vars are invisible inside the model worker. Use `MLX_*`.**

---

## Live experiment slots

| PR | student | experiment | state |
|---|---|---|---|
| #57 | qwen-askeladd | E55 — compose `M = 9` two-stream on the shipped table | `status:wip` |
| #58 | qwen-thorfinn | E54 — lone-versus-sibling NA = 5 law | `status:wip` |
| #59 | qwen-edward | E56 — stream-aware draft depth schedule | `status:wip` |
| #60 | qwen-alphonse | E57 — SDPA chunk predicate bisection | **merged** |
| — | qwen-alphonse | E58 — dispatch latency census (assignment blocked on a GitHub 403) | pending |

All three live prereg numbers were computed **before** the dilution and `g` corrections, so all
three are roughly 9.6 % high from dilution alone and further inflated by the corrected depth
slope. Each needs a re-pricing note.

**The single highest-value answer any student can return right now is the register census from
#57 or #58. It must read 108, not 129.** That one integer decides whether the +3.61..3.81 %
direction is buildable.

My on-record prediction for thorfinn's P1 `<T,5,5>`: **it regresses end-to-end.** The corrected
law table: wins + register max 108 ⇒ Law A, ceiling did not move, report loudly; wins + ≈129 ⇒
Law A plus the shared-ceiling tax (the E27 signature); regresses + 108 ⇒ Law C, sibling overlap;
regresses + ≈129 ⇒ both.

---

## Potential next research directions

Ordered by expected ranked value after this round's corrections.

1. **The dispatch-latency census and whatever it opens (E58).** If the total round dispatch
   count is a few hundred, this is the largest unexplored term in the campaign and the one the
   whole leaderboard is ignoring. Everything downstream — command-buffer batching, kernel
   fusion priced by dispatch count rather than FLOPs, evaluation-boundary placement, graph
   capture — depends on the census existing.
2. **Two-stream `M = 9` via `4/r` sequential row blocks.** Prereg **+3.61..3.81 % of score**.
   Gate: the register census must read 108. This is the biggest single number we have.
3. **A deeper depth schedule tuned for `g < 1` rather than for local time.** The ladder is
   provably mis-calibrated for the ranked host by 22–26 %, every top tree ships the same one,
   and it is a pure-constant change with no correctness surface. The obstacle is that local
   measurement points the wrong way, so this needs a transfer-corrected objective rather than a
   local sweep — which is exactly what `g` now provides.
4. **A per-round width trace on the hidden-prompt-like regime.** The width mixture is the input
   to directions 2 and 3 and is currently a factor-of-4 uncertainty. A scheduler-faithful
   simulation seeded with the ranked acceptance rates (0.835 / 0.875) rather than the local
   fixture's would close it without a GPU.
5. **Fix the `kL = 1024` near-tie exposure.** Not a speed direction, a survival one: the ranked
   window always reaches the boundary where our shipped base disagrees with its own reference
   on top-two evidence. Worth understanding before a submission fails on a hidden prompt for a
   reason we already saw.
6. **Acceptance-prior seeding from the ranked regimes.** Cheap, pure-constant, and the shipped
   prior is measurably wrong. Small, but the ranked legs are short enough that early-round
   policy matters.
7. **The head shortlist constants and the precision-island A/B.** Free to test, never swept.
8. **`mx.compile` of the head step.** Untouched by all 712 corpus trees. Ceiling 4.35 %, prior
   near zero, and blocked for the target because `CompiledDecode` excludes SSM and we have 48
   GDN layers — but the single-layer head is eligible. Speculative; needs a cheap viability
   probe before it earns a slot.

### Compose-only — worth carrying on any slot, never worth a slot of its own

- **paul-hf's dead-KV-GEMM elision** (`Qwen35.swift:1759-1766`, `:1812`). Provably bit-exact —
  the `putAlong` index vector is a bijection of `0..<kOut+vOut` — so the local ratio is valid
  here. ~0.04 %. Precondition: dump the precision-island index shapes and ranges and confirm
  full cover.
- **`pendingPrimaryDevice`** (fkiene `7f777a36`). A pure slice. ⚠️ `7f777a36` and jonathan308's
  `055bc201` contain byte-identical code **and comments**, so they are **one** piece of
  evidence, not two.
- **Fused last-merge + final RMSNorm** (`bc0b2fea`). Runs on both legs ⇒ compare matched
  absolute time, not the local ratio. Verify the `rms_looped` dispatch parameters first.
- **Top-32 finalize k-way merge** (DawgZter `d909492b`). Zero floating-point arithmetic.
  Bundle a ten-draw byte-identity replay.
- **Item 146's latch release valve** (`latchProbeInterval = 4`).
- **DEMOTED**: Lieisyourlie's `hidden` deferral (`5ad14a0b`) — negative on **both**
  score-setting prompts.
- **DELETED this round**: the extended SDPA `blocks` warm at `kL >= 1025`. Unreachable.

### Explicitly closed — do not spend a slot

- **The seed prefill.** Scored (8.44–9.05 % of the median-pair legs, so ×0.0875 conversion) but
  it runs `qmm_nax` on the ranked host and our generation-16 GPU cannot execute it. Any local
  prefill result describes work the ranked machine never does. **Reopen only with a
  generation ≥ 17 host.**
- **The SDPA chunk predicate.** Settled by E57; the shipped predicate is correct and the chunk
  is load-bearing.
- **Narrowing or removing the chunk.** Measured loss and a correctness failure.
- **`steel_attention.cpp` / `steel_attention_nax.cpp`.** Unreachable at head_dim 256.
- **Deleting `qmv_fast_singlerow_affine2_g64`.** ≤ +0.029 % across three rival attempts. Keep
  as an audit item only: it is the first suspect if a hidden-prompt fidelity failure is traced
  to the compact draft readout.
- **Baked bf16 GDN q/k scale immediates.** −0.0003, and it licenses FMA contraction — the E51
  failure class. The scale constants are already memoised.
- **Depth cap 8 → 7.** Bundled and unattributable in the corpus.
- **Residency and command-buffer install.** Already present, and our form is stronger than the
  rival's.
- **EOS truncation / `reachedStopToken`.** Organizer-supplied, buys no score, and actively
  breaks the fixed window — the trusted driver owns the window by count and has no stop-token
  concept. It has been added four times and lost three, every loss driven by a merge rather
  than a decision. Our tests forbid its return.
- **`MLX_METAL_GPU_ARCH` nax-off.** Formally killed.
- **Forced-commit sweeps** (item 69). 0.61 % non-monotone spread below a 0.86 % noise floor.
  E58 rung 2 is a **different knob** — buffer size, not flush placement.

### Standing hazards, not directions

- **The QMV register allocation is shared across all widths and cannot be split within
  `editablePaths`.** Any per-width register win taxes all seven widths.
- **NA = 5 codegen is not bit-stable, whole-kernel.** Zeroing lane 4 also moved lane 1 by 8 ulp.
  For NA = 5 edits, **any** bitwise delta at `M <= 9` is a whole-kernel hard stop.
- **`M = 10` bitwise deltas are a pre-existing base property** of the `qmm` split-k 9→10 padding
  path. Max scored verify width is 9.
- **`psi_serial` is not identifiable locally** (estimates 0.7966 / 0.8694 / 1.0414 / 1.2470).
  Never subtract a local serial share when pricing official value. Run
  `senpai/verify-ranked-score-boundary.sh` before pricing new work.
- **Pin the divergence, never the body**, in every twin and surface gate.
- **A call-site trace is not a call-graph trace.**
- **Failure rates matter when choosing a candidate file**: `Qwen36MTPHeadAttachment.swift`
  **80 %**, `KVCache.swift` **71 %**, `AttentionUtils.swift` 60 %, against a 42.7 % baseline;
  safest is `RuntimeStartupMemoryPolicy.swift` at **20 %**.
- **Churn does not predict score.** Pearson −0.075. The best result at ≤ 20 lines of churn is
  3.2467. 127 of 154 editable files were never touched by any of 712 trees.
