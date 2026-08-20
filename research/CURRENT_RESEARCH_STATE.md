# SENPAI Research State

- **2026-08-20 20:15 UTC**
- Most recent human research direction: **Issue #22 — execute aggressively
  toward the winning frontier.** No new human instruction since.
- Campaign base: advisor branch `f7f356b2834518ced918f3049ca1b88afb6003f3`,
  which adopts organizer commit `8b54ff11c6d686628f6534d7127a261115782757`.
- Official frontier: **`8e83c6b3` at 3.3189406078251036**, promoted
  2026-08-20T17:41:59Z. Our best is `9b241879` at 3.23588901.
- One submission in flight: `32c6dc69`, sent 18:48:46Z, still `validating`.

---

## The three facts that currently govern every decision

### 1. The ranked host runs a latent binary measurement mode worth +1.391 %

`travel` at its locked draft length of 2.656 is bimodal in
`mtp_seconds_per_token_mean`: mode A below 0.01755, mode B at or above. Six of
eleven byte-identical resample groups straddle both modes and price the
difference at **+1.391 % of published score, sd 0.487**.

The serial leg is identical in both modes to eight significant figures, the
cost scales with **drafting rounds** at about 0.85 ms each, and the mode can
flip **between prompts inside one run**. That excludes thermal state, DVFS and
background load. Zero of 126 mode-B runs has ever exceeded 3.29.

**Label every ranked run A or B before comparing it to anything.** A
cross-mode pair measures the mode, not the code.

The lever: `warmMTPDecode()` at `QwenRuntimeMTPDriver.swift:85-92` is untimed,
and the clock starts at `:96-99`. If mode B is any cold or unwarmed state, we
can pay unlimited untimed warmup to force mode A.

### 2. The beagle wall caps all recombinable headroom at +0.69 %

The score is the mean of the two lowest wide raws, exactly, in 581 of 616
scored runs. **beagle is the lowest in 178 of 178 runs scoring 3.2 or above.**
If every prompt simultaneously hit its all-time board record over 616 runs and
859 trees, the score would be 3.332458, which is **+0.69 %**. Restricted to
mode A it is **+0.21 %**.

Every remaining gain must therefore come from a mechanism that has never been
demonstrated on this board. The schedule axis is re-closed on this basis.

### 3. The materialised-intermediate law

Fusing two launches into one without removing a buffer is a null: the
organizer's `ead84bba` → `c6af1e24` measured **+0.05 %**. Making that same
kernel write its output in the consumer's layout, so a materialised
intermediate disappears, is `c6af1e24` → `8e83c6b3`: **mean7 −0.116 %,
sd7 0.021 %, 7 of 7 prompts faster, about 5.5 σ**, on an identical schedule.

That is **13 to 16 µs per draft token per eliminated buffer** — the scale of
one command-buffer commit (13.5 to 17.6 µs) or one allocation, not one
dispatch (0.66 to 1.55 µs). Thorfinn's E83 corroborates independently on the
prefill leg: removing one materialised intermediate per layer saves 5.5 ms in
8 of 8 reps, while merging cells without removing one saves 2.6 ms in 5 of 8
and breaks exactness.

**Minimise the count of materialised intermediates on the per-draft path.**

---

## Operating rules that changed this cycle

- **The ranked board cannot resolve a mechanism below about +0.5 %.** The
  frontier-schedule mode-A cluster has sd 0.15 %. Measure mechanisms locally
  with matched absolute candidate time; spend ranked slots only on stacks that
  can take the frontier.
- 🔴 **Yukon allows exactly one in-flight submission per account.** Our rate is
  bounded by validation latency, currently over an hour. A slot is the
  scarcest resource in the campaign.
- 🔴 **The default local harness head is the wrong head** — 849,400,347 bytes,
  15 tensors, zero precision islands, zero scales, against the declared
  427,742,600 bytes and 40 tensors. Island mechanisms are structurally absent
  under it and `fc` runs bf16 `gemv` instead of `qmv`. Use
  `research/fetch-declared-head.sh`. `gemv` in a head phase is a free
  wrong-head tripwire.
- 🔴 **`benchmark-qwen-mtp.sh --local-submit` does not rebuild the worker.**
  Always run `senpai/rebuild-and-assert-worker.sh` with symbol witnesses first.
- 🔴 **`MLXFAST_*` environment variables never reach the scored worker.** The
  sanitizer at `QwenRuntimeWorker.swift:2565-2590` allows only `DARKBLOOM_`,
  `DYLD_`, `LC_`, `METAL_`, `MLX_`, `MTL_` and an exact-key list. Candidate
  code must not read the process environment in a hot path.
- **An isolated-cell roofline over-states recoverable time whenever the cell
  does not saturate the GPU.** A fusion saving is bounded by removed traffic
  and removed launches, never by the difference of two isolated cell times.

---

## In flight now

| student | PR | experiment | state |
|---|---|---|---|
| edward | #87 | **E85** — eliminate materialised intermediates on the per-draft head path | just assigned |
| alphonse | #84 | **E82** — timed island-drop palindrome | running |
| thorfinn | #85 | **E83 r2** — revert instrument, rebase, resubmit terminal negative | revision requested |
| askeladd | #86 | **E84** — two dead-work eliminations, rebase then submit | rebasing |

**Submission queue order**: `32c6dc69` (in flight) → askeladd's E84 at about
−0.27 % → alphonse's `noislands` at −0.46 to −0.92 % if the timed session
confirms.

---

## Current research focus

**Find and eliminate every materialised intermediate on the per-draft
proposal head path.** This is the one mechanism class with a measured
per-unit price that is currently unmined. Named targets: the three `MLX.take`
gathers plus `quantizedMM` at `Qwen35.swift:3453-3459`, replaceable by one
`gatherQuantizedMM`; the `embedTokens` result at `Qwen35MTP.swift:147` and
`:180`, fusable into the concat kernel; the coarse slice at `:3442-3444`.
Arithmetic prediction −0.35 % to −0.46 %.

**Remove head bytes.** Calibrated by a ranked measurement: askeladd's
mechanism A removes 1.38 % of head bytes for −0.172 % of candidate time,
3.3 σ. Alphonse's `noislands` removes 7.36 % of head bytes and costs no
measurable acceptance (+0.13 pt, paired 4 to 1 in favour). The two are
mutually exclusive; `noislands` dominates on bytes by 5.34 to 1.

**Diagnose and defeat mode B.** Worth +1.391 %, the largest single number
available, and the untimed warm phase is a legal, unlimited lever.

---

## Potential next research directions

1. 🔴 **The mode-B hunt via untimed warmup** — thorfinn's next assignment.
   Twenty identical local legs looking for bimodality in absolute candidate
   seconds per token while the serial leg stays flat. Candidate causes: Metal
   pipeline or JIT cache state, wired-residency eviction, allocator state, GPU
   P-state under the candidate's small-dispatch pattern.
2. 🔴 **A numerically exact head-byte cut** — delete the 4-bit `k_proj` and
   `v_proj` and their islands, ship both as plain BF16. Bit-identical
   acceptance, 5,906,432 bytes saved. Risk is structural: `Load.swift:250-258`
   keys quantization off `.scales`, so the lazy fused `_qkvW` pack may break.
3. 🔴 **Fix the `positionAcceptEMA` prior.** The shipped `0.85 · 0.98^i` is
   materially wrong; alphonse measured per-position acceptance to be **flat**
   at a pooled 0.9551 over eight positions.
4. **Split the fused head `qkv`** so the overwritten K and V rows are never
   computed, priced at +0.096 % and independent of the island decision.
5. **`draft_lm_head` group size g64 → g128 or g256** — 15.7 MB and −3.68 % of
   head cost for g128, about +0.310 % of score. Two files. Behind the island
   work. A worse coarse shortlist cannot break exactness: E79 measured coarse
   recall@32 at exactly 1.0000.
6. **The quantized GEMM path at M=512** — the only prefill target left after
   E83 closed the fusion boundaries. Seed prefill runs at 6.18 TFLOP/s and is
   99.7 % GEMM-bound, so this is a kernel question, not a graph question.
7. **The narrow dispatch switch at `quantized.h:1980`** — head `qkv` at
   n=3072 and the K/V pack at n=2048 are MTP-only shapes.
8. **Entropy-gated early stopping**, AdaEDL, arXiv:2410.18351 — untried, and
   the schedule axis being closed on *depth* does not close it on *gating*.
9. **An IVF or coarse-cluster index over the 98,336 draft rows** — the only
   remaining structural attack on head bytes that is not a quantization
   change.

---

## Closed. Do not reopen without new evidence

Register allocation (bounded at −1.209 %), occupancy (0.52 %), the copy
family (0.016 %), the QMV group-count axis, head fine-tuning and distillation
(six ranked negatives plus a controlled local refutation), non-GEMM prefill
overhead (≤ 32 ms), the prefill fusion gates, post-hoc affine-4 g64
requantization (8 to 11 % error reduction, measures as zero), H-221 host
synchronisation in both decode and prefill forms, the schedule axis
(+0.21 % ceiling), low-rank or truncated draft readouts, and the TG-per-core
knee.
