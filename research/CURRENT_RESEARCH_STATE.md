# SENPAI Research State

- **2026-08-23 14:00 UTC**

## Most recent research direction from the human researcher team

No new human direction since Issue #22 (2026-08-21). The standing direction is
unchanged and is being executed: keep four students productive on independent
current-frontier questions, adopt the promoted editable surface before the next
official submission, push candidate implementations and evidence early, submit
autonomously whenever a clean candidate has credible evidence, and continue
research after every result.

## Where the campaign stands

- **The bar: `ec24d591` newjordan = 3.72911001**, promoted 2026-08-23T11:43Z,
  source `0863b06a`, which is now also `upstream/main`.
- **Campaign best: `0cf1637e` = 3.68278758**, rejected. It remains our
  scientific frontier and the base most current experiments were measured
  against.
- **In flight:** `749da2cf`, Edward's E150 R4 linearised schedule, validating
  since 12:22:01Z. The advisor owns the bounded read-only receipt watcher (job
  `0c80d7de`). The official slot is closed until it resolves.
- Advisor base `c2b601b7`. Growth budget is tight: about 66,648 bytes of shared
  headroom across four students. The import below returns a large block of it.

## Current research focus

**Adopt the promoted frontier surface, then compose independently readable
mechanisms onto it and submit quickly.**

The dominant fact of the day is FINDING 267 combined with FINDING 260: our
editable tree is **+0.7275 % slower on the ranked candidate leg** than
`0863b06a`, for no mechanism we can name. That penalty is larger than every
mechanism in flight, and removing it needs no invention. The diff is five files,
`+694 / -1570`.

FINDING 267 says the penalty is a lineage property, not four unlucky draws:
fitted against the state-exposure shape `s_p = 903 us x drafting_rounds_p /
decode_seconds_p`, all four senpai rows read `k` between +0.28 and +1.16 steps
with `R^2` 0.79 to 0.90, against a rival median of about +0.02 and an empirical
rate of `k >= +0.27` of 5/32. Probability of four independent draws at that
level is about 6e-4. The leading hypothesis is residency pressure from the extra
kernel variants and resident tensors our tree compiles and holds. Thorfinn's
residency probe tests it directly.

FINDING 269 (this cycle) removes the guesswork about what the import costs us.
`derivedClusterRowsPerLeaf`, `buildDerivedClusterIndex`,
`draftTokenIDWithDeclaredRerank`, `segmentedVerifyDepthCap` and `depthPriceArm`
all survive in the frontier. `qwen35ClusterCentroidQMV`, `onePass67`, the
one-pass width table, `passBoundaryTierFactor` and `pb6` do not. The frontier's
NAX header contains zero `kE147` identifiers.

## Four live experiments, one per student, one per physical Mac

1. **#152 thorfinn — import the promoted editable surface.** The campaign
   critical path. Stage A is the import, the test deletions, the gate chain and
   a 512-token exactness leg, pushed before any timing. Stage B is the FINDING
   267 residency probe and matched ABBA timing. His first deliverable is the
   one-line AIR verdict `e152_quantized_nax_air_identical`, which unblocks
   alphonse. A1, the 127-site boundary-fused fill producer, is stopped: the
   organizer's frontier contains it verbatim.
2. **#151 alphonse — the ranked prefill channel.** The 128x32 rectangular NAX
   seed retile (R1) and the affine double-buffered loader (R2). Our prefill
   channel is a measured exact null (FINDING 268, `-0.0084 % +- 0.2132`) because
   `kE147NaxRetileOn = false`, so R1 is the whole mechanism rather than a tweak.
   Two rival receipts price the family: `5cdc9c17` at `-4.9721 %` prefill and
   `43925f29` at `-4.1181 %`. **R1 standalone is the designated next-submission
   mechanism.**
3. **#153 askeladd — leaf16 and the merged SDPA kernel.** leaf16 is
   `per_draft_step` with a ranked discount basis of 0.74453, worth about
   +0.188 % published. The merged SDPA kernel is `width_gated_at_6` and worth
   about +0.29 %, but it is undetectable on plutarch by a factor of 26 and needs
   its own receipt. Both rebase onto the import.
4. **#150 edward — per-round discrimination.** R4 is submitted as `749da2cf`.
   R3 prices the host round trip for a sequential stopping rule against a
   774.95 us/round break-even. The pb6 question is closed by deletion in the
   import.

## Composition plan for the next official submission

Total-leg published %, on top of the imported frontier base.

| step | owner | channel | published % |
| --- | --- | --- | ---: |
| import `0863b06a` editable surface | thorfinn | both | +0.7275 |
| E151 R1 NAX 128x32 seed retile | alphonse | prefill | +0.505 |
| E153 R1 leaf16 | askeladd | decode | +0.188 |
| E151 R2 affine NAX double buffer | alphonse | prefill | +0.419 |
| E151 R1∘R2 composed | alphonse | prefill | +0.663 |
| E153 R2 merged SDPA | askeladd | decode | +0.29 |

Designated candidate: **import ∘ E151 R1**, about **3.748** against the bar
3.72911. Import ∘ leaf16 alone is about 3.736, a margin of +0.007, which sits
inside the nuisance tail and is not worth a slot on its own. Rule 146 keeps the
prefill and decode channels separately readable on one receipt, and Rule 156
gives plutarch as the mechanism-class discriminator, so composing does not cost
us attribution.

## Standing measurement rules confirmed or added this cycle

- **Total-leg frame only** for every published-% claim. Rule 134's 524.5
  us/round is a total-leg constant; the decode-frame equivalent is 468.8, and a
  decode-frame table overstates published effect by about 11 %.
- **RULE 156 amended.** The Plutarch class list is `per_round`,
  `per_drafting_round`, `per_draft_step`, `width_gated_at_<k>`.
- **RULE 157.** Never bucket or key anything on Python's salted `hash()`.
- **Push before you measure again.** No student branch had been pushed;
  `749da2cf`'s submitted tree existed only on one Mac.

## Withdrawn this cycle

FINDING 256, FINDING 259 and RULE 155 are withdrawn. `24fb4012` makes the
byte-identical two-integer width-table edit and reads `-0.0390 % +- 0.0799`,
`z = -0.49`, so the one-pass width-6 rung costs approximately zero. The
`1db9d63e -> 0cf1637e` reading of `-1.5571 %` for the width table is
contaminated by the same state-exposure nuisance. FINDING 230's third pillar is
also refuted by FINDING 268.

## Open questions worth a student when one frees

1. **FINDING 267's mechanism.** If our tree and the frontier sit in different
   residency states, every future mechanism must report its residency footprint
   before its timing. That would be a campaign-wide methodology change.
2. **A head that is cheaper at equal quality**, rather than better at any price.
   119 same-solver head-swap pairs, 16 bought at least +0.01 acceptance, zero
   paid for themselves, best efficiency 0.70 against break-even 1.00.
3. **Prefill beyond `-6.5 %`.** No receipt on the board has ever gone past
   `-4.9721 %`. The channel is worth 10.0438 % of the total leg.
