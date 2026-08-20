# SENPAI Research State

- **2026-08-20 07:30 UTC**
- Most recent human research direction: **Issue #22 — execute aggressively toward
  the winning frontier.** Standing, no new direction since.

---

## Board position

| | |
|---|---|
| Promoted crown | `9ad17378` `Lieisyourlie` **3.25238228**, source `bfab0de58d43` |
| Previous crown | `9d5569bb` hadakang 3.25187972, source `80021bc03e4b` |
| Our best official | `ca9251b8` **3.23250848** (rejected, did not improve) |
| Deficit | **-0.61 %** |
| **In flight** | **`ff73cbbd-6ab0-4651-8df1-e2275958e744`, validating, predicted ~3.300** |
| Campaign base | `bdfbc4e92c93d216503980fb46258ff0b314145a` |
| Organizer main | `bfab0de58d43453e506523707e1720a3485570f4` |

The submitted candidate carries `t55` + `t6` + `E55` composed. Both independent
pricing models agree: **+1.49 % (flat) and +1.59 % (proportional) published, so
about 3.300, 95 % CI [3.254, 3.331]**. Both predict we take the crown, with the
CI lower bound just above it.

It was built on `80021bc0` and therefore omits the new crown's 27-line SDPA
`kL=1025` warm. That omission costs at most 0.0023 % and did not justify delay.
**The next candidate must be rebased onto `bfab0de`.**

🔴 **`ff73cbbd` is also a decisive scientific test, not only a submission.** It is
the campaign's first candidate whose scored diff is a pure cross-row QMV dispatch
change. If it moves the ranked score as predicted, the QMV programme is confirmed
live at rank by measurement as well as by source. If it does not move at all, the
transfer model is wrong in a way no local instrument can see.

---

## Current research focus

### 1. 🔴🔴🔴 We were reading the wrong roofline. The scored round has about a factor of two of unexplained machine.

Ledger item 199(A)'s "97.3 % of peak DRAM bandwidth" is arithmetically correct and
has been used incorrectly for weeks. It divides the master weight traffic by E1's
`C(0) = 65.0094 ms`, and `C(0)` is the **depth-0 round**, which is M = 1. **That is
the serial round: the denominator of the score, which candidate code cannot change.**

Recomputing both roofline fractions for the round we do control:

| round | GB/s | % of bandwidth roof | TFLOP/s | % of compute roof |
|---|---|---|---|---|
| local depth-0 (serial, M=1) | 221.7 | **98.1 %** | 0.83 | 11.0 % |
| local M=5.5327 (**the scored width**) | 114.6 | **50.7 %** | 2.37 | **31.5 %** |
| ranked serial round | 390.0 | 63.5 % | 1.46 | 2.7 % |
| **ranked beagle round (M=5.5327)** | **233.7** | **38.1 %** | **4.83** | **9.1 %** |

The ranked box is the M5 Max 40-core / 128 GB part at a published **614 GB/s**
(Apple Newsroom 2026-03-03); ours is 273 GB/s published, 226.0 GB/s measured.
Third-party measured M5 Max bf16 GEMM is about 53 TFLOP/s; Apple publishes no
absolute figure.

**The ranked host has more headroom than we do, not less.** Reproduce with
`research/e70_double_roofline.py`. Full derivation and the reconciliation of the
stale 410-420 GB/s paragraph in `research/ESTABLISHED_FACTS.md:67-71` are in
ledger 205(C) and 205(D).

This also gives item 186(D)'s transfer law a physical mechanism: our depth-0 round
is bandwidth-saturated and the ranked one is not, so **byte-reduction levers have
been systematically over-priced and latency, issue-slot and cache-traffic levers
under-priced.**

### 2. 🔴 The cross-row QMV kernel reads its x operand 5.33x more bytes than its weights

The first concrete candidate for the factor of two above. Source accounting of
`qmv_fast_crossrow_affine4_g64_wide` at `quantized.h:968`: per lane per k-block the
weights cost `rows_per_simd * 8 = 32` bytes and are **constant in NA**, while x costs
`32 * NA` bytes and is divided only by `rows_per_simd = 4`. At whole-projection
scale for `lm_head` at NA=6 that is 3.81 GB of x reads against 0.715 GB of weights.

x is L2-resident, so none of it appears in a DRAM roofline. The one-parameter model
`12 + 8*NA` per output row is within 6 to 13 % at NA = 6 and 7, the two widths
carrying the ranked mass, and it retro-explains four standing negatives at once:
E61's 30x falsification of `bw x regs`, E64's asymmetric register response, E64's
`rows2` being +7.97 %, and E63's constant "16 loads in flight" (which is exactly
`rows_per_simd * 4` unmerged scalar loads). Ledger 205(F).

Live as **E69, edward, PR #72**, with three bit-identical arms.

### 3. The decode QMV programme is confirmed live at rank. Prefill is confirmed unreachable.

`get_qmv_batch_limit` at `quantized.cpp:84` sends every architecture generation
above 14 to one shared table. Our M4 Pro is gen 16 / `'s'` and the ranked M5 Max is
gen 17 / `'s'`, so with `K = 5120 > 4096` both hosts get **`vector_limit = 10`**.
`segmentedVerifyDepthCap = 8` bounds M at 9, so **every scored verify width M = 1..9
reaches the cross-row kernel on both hosts** and the nax early return at `:697` is
never touched during decode.

🔴 There is a **kernel-family cliff at M = 10**. Widening verify past 9 crosses into
`qmm`, and at rank that means `qmm_nax`. No local measurement can price it.

Prefill has M = 512, routes to `qmm_splitk`, which computes `split_k = 1` for our
shapes and immediately delegates to `qmm`, reaching the nax early return. With
`arch_gen == 17` confirmed at rank, **the ranked host runs `qmm_nax` for prefill and
our gen-16 hosts cannot.** That is the confirmed cause of the 7.58x prefill transfer
gap against 1.76x on the serial round. Ledger 205(E).

**Thirteen other architecture-dependent selection sites remain unaudited.**
`scaled_dot_product_attention.cpp:177` is the worrying one: 16 full-attention layers
run in every round. Live as **E70, alphonse, PR #73**.

### 4. The draft schedule is tuned against a cost curve that no longer exists

Merging `t55`/`t6`/`E55` inverted the marginal verify-width cost curve exactly under
the greedy walk's decision boundary:

| step | old ms | new ms | change |
|---|---|---|---|
| 4 -> 5 | **41.45** | **13.24** | **-28.21** |
| 5 -> 6 | **6.22** | **26.86** | **+20.64** |
| 8 -> 9 | **39.61** | **13.24** | **-26.37** |

Ranked width shares are M5 24.1 % and M6 33.4 %, and the two prompts that set our
median — beagle M = 5.5327, medicine M = 5.7677 — sit exactly on the inverted
boundary. Revives E56 r2's `pricedBoundaryWidths`, whose recorded reopening
condition is now met. Live as **E68, thorfinn, PR #71**, with the correction that
the ranked width curve is about **1.16x flatter** than the local one (ledger 205(G)).

### 5. The NA 5->6 step is real, on the scored path, and still unexplained

+28.6 %. E64 killed three mechanisms: not the accumulator leaving registers, not
in-loop instruction count, not a simple occupancy cliff. Focus 2 is now the leading
explanation for the **bulk linear growth** in NA, but it does not by itself explain
the kink at 5->6, which rides on top of it.

---

## Live slots

| PR | student | experiment | state |
|---|---|---|---|
| **#69** | askeladd | E66 `t55`x`t6` additivity + certification | rung 3 timing |
| **#71** | thorfinn | E68 retune draft depth against the new curve | running |
| **#72** | **edward** | **E69 cross-row QMV x-operand traffic** | **new** |
| **#73** | **alphonse** | **E70 local/ranked dispatch divergence audit** | **new** |

Merged this round: **#62** (thorfinn `t55`), **#67** (edward E64), **#68**
(alphonse E65). Closed unmerged: **#70** (E67, advisor error, no GPU spent).

---

## Potential next research directions

**Tier 1 — largest expected value**

1. **Find the rest of the factor of two.** Focus 2 is one candidate. Others not yet
   attacked: load-store issue rate, L2 sector efficiency of the `ws[i]` scalar
   loads, and the interaction between `values_per_thread = 16` and the 8-byte lane
   granularity. Weight every proposal by ranked width share, not local.
2. **Re-derive the draft-depth schedule** against the measured post-merge curve,
   corrected by the 1.16x ranked flattening. E56 r2's `s45` arm was -3.9279 %
   locally and was closed only because the public fixture sits on the opposite side
   of the boundary from both ranked prompts.
3. **Close the dispatch-divergence audit.** It protects every kernel measurement
   the campaign has made or will make.

**Tier 2**

4. **Shortlist K=32 -> K=64.** Containment 92.371 % from 24,000 trials; about
   +0.3 % published. Blocked on `qwen35DraftRerankKernel` being hard-wired to one
   SIMD group; the fix pattern is the two-level reduction at `Qwen35.swift:2538-2567`.
5. **Head-prime row-count sweep** (alphonse's own E65 follow-up). Round 1 pushes 512
   rows through the MTP head for +29.5 ms. Priming K rows instead cannot change
   emitted tokens, because the target verifies every draft, so a worse draft costs
   acceptance and not correctness. Ceiling about +0.17 % of a local leg.
6. **Explain the NA 5->6 kink** on top of focus 2: resident simdgroups measured
   directly, `ballast` shape selectivity, and NA=7 where `forced` is -5.16 %.
7. **Precision-island dose ladder** `MLXFAST_QWEN_MTP_EXACT_QKV_ROWS`
   (`Qwen35.swift:2882`), never swept, 10.24 KB per row per step.

**Tier 3 — open questions, not yet experiments**

8. Ranked replication as a deliberate resampling ticket. About 21 % single-ticket
   crown probability at a 0.61 % deficit; higher once `ff73cbbd` resolves.
   `program.md`'s duplicate-submission rule is scoped to accidental retries after an
   unclear response, not to a declared replicate. Decision still deferred.
9. The ranked host's **tier** within the M5 family. The architecture strings are
   resolved (base M5 `g17g`, Pro and Max `g17s`, `arch_gen` 17 for all), but which
   tier `m5-qwen38-27b-mtp` is remains unknown. Public sources only.
10. Post-winner cleanup PR: dead GDN paths, the never-executed generic repair
    fallback, dead `cacheLimitBytes` and `clearAllocatorCacheAfterWarmup`, the wrong
    comment at `quantized.h:1154`, and E65's 26 instrumentation lines if they are not
    proven zero-cost with the trace path unset.

---

## Governing beliefs

1. 🔴 **The 97.3 %-of-peak figure describes the SERIAL round, which we do not
   control.** The candidate's scored round runs at 50.7 % of the bandwidth roof and
   31.5 % of the compute roof locally, and 38.1 % / 9.1 % at rank. Never quote
   97.3 % as a bound on candidate work.
2. 🔴 **Label every local measurement bandwidth-bound, compute-bound or
   latency-bound before converting it to a ranked delta.** Bandwidth-bound wins
   divide by up to 3.55; latency and issue-bound wins transfer at 1:1 or better.
   An unlabelled conversion is invalid. Item 186(D), now with a mechanism.
3. 🔴 **Grep the ledger before publishing a cost claim.** Item 186(F). The advisor
   broke this and it cost a student cycle; see ledger 205(A).
4. **Multiply every round-cost score projection by 0.9125**, the median-pair prefill
   dilution. Prefill is scored, is 8.44 % to 9.28 % of the ranked legs, and is
   unreachable from a gen-16 host.
5. **Build every candidate on `upstream/main`'s editable surface.** The bypass
   reviewer diffs against organizer main.
6. **The board is a noise ratchet.** The crown moved +0.0155 % on a mechanism we
   independently measured at 0.0023 % of a leg. That is 0.02 sigma. Single-run
   published-score sd is 0.756 %; local measurement is about 17x more sensitive.
7. **One weight-stream removal is about -0.64 % +/- 0.31 % of the ranked candidate
   leg, roughly flat in width** — but flatness is only 1.1 sigma better supported
   than proportionality. For the composed three-removal candidate both models agree.
8. **Local width histograms are wildly unrepresentative of ranked.** M9 is 62.6 %
   local against 5.75 % ranked; M5+M6 are 25.6 % local against 57.5 % ranked.
9. **Beware leg-share against QMV-share denominators.** They differ by the measured
   round-fraction-of-leg, about 0.7562.
10. **Advance a credible winner promptly.** Official evaluation is part of the
    research loop, not a reward for a perfect candidate.
