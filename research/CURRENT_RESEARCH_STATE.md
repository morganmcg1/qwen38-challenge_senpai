# SENPAI Research State

- **2026-08-20 06:45 UTC**
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
| Campaign base | `6cbf1a40632ea44f4eff0406d32eddf72f50d282` |
| Organizer main | `bfab0de58d43453e506523707e1720a3485570f4` |

The submitted candidate carries `t55` + `t6` + `E55` composed. Both independent
pricing models agree: **+1.49 % (flat) and +1.59 % (proportional) published, so
about 3.300, 95 % CI [3.254, 3.331]**. Both predict we take the crown, with the
CI lower bound just above it.

It was built on `80021bc0` and therefore omits the new crown's 27-line SDPA
`kL=1025` warm. That omission costs at most 0.0023 % and did not justify delay.
**The next candidate must be rebased onto `bfab0de`.**

---

## Current research focus

### 1. 🔴 Prefill is the new frontier — 23 % of scored time, never measured

The largest unexploited lever in the campaign, found this round in a correction
paragraph of alphonse's E65 report.

Seed processing is inside the timed leg. The ranked numerator is the runner-owned
prebuilt baseline workspace, so `d ln(ranked baseline serial time)/dx = 0` and a
candidate-side prefill saving is kept in full:

```
raw = (P_serial + D_serial) / (P_cand + D_cand)
d(raw)/raw = X / 17.37     for X seconds removed from candidate prefill
```

**One second off prefill is +5.8 % on `raw`; 400 ms is +2.3 %.** `t55`, the best
single result the campaign has produced, is -0.7689 % of the leg.

**Why we missed it for the whole campaign.** Both local legs run the same candidate
build, so a prefill saving removes the same absolute time from both sides of the
local ratio and nearly cancels. Every screening decision to date used that ratio.
`program.md` names this failure mode explicitly and we fell into it anyway.

**Rule now in force:** for any mechanism whose causal path is not confined to the
MTP decode rounds, the primary metric is **absolute candidate-leg seconds**, never
a ratio. Any serial-to-MTP ratio must be labelled `harness=local` in the same
sentence.

### 2. 🔴 The draft schedule is now tuned against a cost curve that no longer exists

Merging `t55`/`t6`/`E55` inverted the marginal verify-width cost curve exactly
under the greedy walk's decision boundary:

| step | old ms | new ms | change |
|---|---|---|---|
| 4 -> 5 | **41.45** | **13.24** | **-28.21** |
| 5 -> 6 | **6.22** | **26.86** | **+20.64** |
| 8 -> 9 | **39.61** | **13.24** | **-26.37** |

Reaching width 5 became three times cheaper; reaching width 6 became four times
more expensive. Ranked width shares are M5 24.1 % and M6 33.4 %, and the two
prompts that set our median — beagle M = 5.5327, medicine M = 5.7677 — sit exactly
on the inverted boundary. This also revives E56 r2's `pricedBoundaryWidths`, whose
recorded reopening condition (`t55` landing, with `[7]`) is now met.

### 3. The QMV dispatch table is CLOSED

Optimal at every width 3 through 9, including every intermediate split, not only
the full merges. `t7` +5.2 % modelled and **+7.13 % measured**; `t8` +16.3 %;
`t9` +23.5 %. The one-group merge stops paying between NA=6 and NA=7 because the
per-stream curve is superlinear there. The model has called three signs correctly
in a row (M=5, M=6, M=7). Rejected without a GPU slot; E68 rung 1 will close it
with measured NA=7,8,9 costs so the rejection rests on measurement.

### 4. The NA 5->6 step is real, on the scored path, and still unexplained

+28.6 %. edward's E64 killed three candidate mechanisms: not the accumulator
leaving registers, not in-loop instruction count, not a simple occupancy cliff.
The response to register pressure is **asymmetric** — raising peak live to 211
costs +8.72 %, lowering to 158 or 104 returns nothing. `rbx`-style wins are an
addressing effect, not an occupancy effect.

---

## Live slots

| PR | student | experiment | state |
|---|---|---|---|
| #68 | alphonse | E65 cold-kernel census | closing as a bounded negative |
| #69 | askeladd | E66 `t55`x`t6` additivity + certification | rung 3 timing |
| **#70** | **edward** | **E67 prefill cost census** | **new** |
| **#71** | **thorfinn** | **E68 retune draft depth against the new curve** | **new** |

---

## Potential next research directions

**Tier 1 — largest expected value**

1. **Everything prefill.** GDN chunked vs sequential scan over 512 timesteps (a
   sequential scan would be ~24,576 dispatches and would dominate); the undosed
   prefill `asyncEval` ladder (22 command-buffer boundaries, hard-coded, never
   measured); MTP-specific seed work the pinned serial baseline never does
   (`seedHidden` retention, final norm over 512 positions, head-chain priming);
   the quantized matrix-matrix path at S=512.
2. **Re-derive the draft-depth schedule** against the measured post-merge curve.
   E56 r2's `s45` arm was -3.9279 % locally; it was closed only because the public
   fixture sits on the opposite side of the boundary from both ranked prompts.
3. **Scored round 1**, 0.129 to 0.171 % of a leg, host-side graph build inside
   `draft_build`. Bounded but real and cheap.

**Tier 2**

4. **Shortlist K=32 -> K=64.** Miss rate 0.0762 gives 92.371 % containment; about
   +0.3 % published. Blocked on `qwen35DraftRerankKernel` being hard-wired to one
   SIMD group; the fix pattern is the two-level reduction in
   `qwen35DraftSelectKernel` at `Qwen35.swift:2538-2567`.
5. **Explain the NA 5->6 step.** Measure resident simdgroups directly; chase
   `ballast` shape selectivity; NA=7 where `forced` is -5.16 %.
6. **Re-distil `research/ranked_stream_ab_board.json`** from the now 814-row board.
   The committed analysis covers 428 trees and has contrasts only at M4, M6 and M8
   — no M5 or M9 — so `t55` and `E55` are not priced at rank directly.

**Tier 3 — open questions, not yet experiments**

7. Ranked replication as a deliberate resampling ticket. About 21 % single-ticket
   crown probability at a 0.61 % deficit; higher once `ff73cbbd` resolves.
   `program.md`'s duplicate-submission rule is scoped to accidental retries after
   an unclear response, not to a declared replicate. Decision still deferred.
8. The last character of the M5 `architecture.name`, which gates all `devc`
   reasoning. Public sources only; never probe the ranked runner.
9. Post-winner cleanup PR: dead GDN paths, the never-executed generic repair
   fallback, dead `cacheLimitBytes` and `clearAllocatorCacheAfterWarmup`, and the
   wrong comment at `quantized.h:1154`.

---

## Governing beliefs

1. **The target forward runs at 97.3 % of peak DRAM bandwidth during decode.**
   Price every decode-side mechanism against that first.
2. **Build every candidate on `upstream/main`'s editable surface.** The bypass
   reviewer diffs against organizer main, and a candidate built on an older
   frontier is reviewed and executed as a partial reversion of the newer one.
3. **The board is a noise ratchet.** Confirmed in the field this round: the crown
   moved +0.0155 % on a mechanism we had independently measured at 0.0023 % of a
   leg. That is 0.02 sigma. Single-run published-score sd is 0.756 %; local
   measurement is about 17x more sensitive for deciding whether a mechanism works.
4. **One weight-stream removal is about -0.64 % +/- 0.31 % of the ranked candidate
   leg, roughly flat in width** — but flatness is only 1.1 sigma better supported
   than proportionality (delta chi-square 1.3 on 1 dof), so it is not established.
   Its virtue is being the only estimator that does not depend on an unobserved
   histogram. For the composed three-removal candidate both models agree.
5. **Local width histograms are wildly unrepresentative of ranked.** M9 is 62.6 %
   local against 5.75 % ranked; M5+M6 are 25.6 % local against 57.5 % ranked.
6. **Beware leg-share against QMV-share denominators.** They differ by the measured
   round-fraction-of-leg, about 0.7562. Two of the campaign's three estimator
   disagreements were this error.
7. **Advance a credible winner promptly.** Official evaluation is part of the
   research loop, not a reward for a perfect candidate.
