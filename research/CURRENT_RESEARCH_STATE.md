# SENPAI Research State

- 2026-08-19, after merging E58 (PR #61), assigning E60 (PR #63), and reading
  the organizer submission refs as a ranked experiment matrix (ledger 192).
  Campaign base `38e43f07a0dde9f4abad96d69c1497e2e83db403`.
- Most recent human research direction: issue #22 -- execute aggressively toward
  the winning frontier. Issue #31 is complete and closed. No new human direction
  is outstanding.

## Board

| quantity | value |
|---|---|
| live promoted frontier | **3.24985583421771** (submission `59b321ee`, solver fkiene, source `9e1ff9ec` = `upstream/main`) |
| organizer main `0c90733d` | **3.24929399** (submission `0cd0a6b4`, solver ofou) |
| our best official submission | **3.23250848263467** (receipt `ca9251b8`, candidate `2b0c36a0`, rejected on score) |
| our deficit to the frontier | **0.5366 %** |
| ranked jitter per prompt per leg | **0.2257 %** (n = 408) |
| ranked MDE at 2 sd | **+0.283 %** (worst case +0.527 %) |
| local end-to-end null floor | **0.0629 %** |

## 🔴 The deficit decomposition, and why it changes everything

Every scored rival tree is a local git ref under `refs/remotes/upstream/submissions`,
and a submission commit's first parent is the organizer main of its day. So
`score(S) / score(S^)` is a **ranked A/B measurement of that submission's diff**,
run on the official M5, on all eight hidden prompts, at zero cost to us.

Reading the board that way (ledger 192):

| component | ratio | points of the gap | share |
|---|---|---:|---:|
| **our own E27 scored-surface change** | `x1.00332661` | **0.3327** | **62 %** |
| organizer main advanced after we submitted | `x1.00185982` | 0.1860 | 35 % |
| the frontier's 70-line untimed warm | `x1.00017291` | 0.0173 | 3 % |
| **total deficit to the frontier** | `x1.00536642` | **0.5366** | 100 % |

Sixty-two percent of the gap is a hole we dug ourselves. Thirty-five percent is
base staleness. Only three percent is a mechanism a rival holds and we do not.

**Our current base has never been measured on the ranked M5.** It differs from
organizer main by exactly two files, 144 inserted and 66 deleted lines, and the
E27 quantized table that cost `0.331 %` is no longer in it.

Two retractions follow:

- 🔴 **181(D) is retracted.** The frontier's advantage is not one untimed warm.
  `warmTargetLaterWindowSDPA` is worth `+0.0173 %`, an order of magnitude inside
  ranked jitter. It rides along as arm C of E60; it is not worth a slot.
- 🔴 **Our local pricer's blindness now has an exact reference point.** It said
  E27 was worth `+2.21 %` to `+2.53 %`. The board said `-0.33 %`.

## Current research focus

**Measure and submit our own composite.** That is the only action that can move
97 % of the gap, and it is the only question a local harness cannot answer.
Everything else is downstream of knowing the sign of our own work.

Three secondary themes, all live:

1. **Widen the QMV width table under the 108-register legality floor.** E59
   proved the ceiling dose is zero for both `r=2` row-block mappings. The
   remaining unknown is the `r=2` arithmetic tax, and the route pays at any tax
   below about 14 %.
2. **Spend draft depth where the ranked board can pay for it.** Edward's ranked
   width mixture shows `M = 9` is 3–9 % of ranked QMV time against 53 % on the
   local fixture, and headroom analysis shows medicine saturates after
   `+0.634 %` while beagle pays in full up to `+7.20 %`.
3. **Measure at ranked command-buffer geometry.** No arm this campaign has ever
   measured used the geometry the scored runner uses.

## In flight

| PR | student | experiment | state |
|---|---|---|---|
| #63 | qwen-alphonse | **E60** composite against organizer main, then certify for submission | assigned |
| #62 | qwen-thorfinn | **E59** `<T,5,5>` at `r=2` over row blocks | rungs 1 and 2 passed, rungs 3 and 4 running |
| #59 | qwen-edward | **E56** stream-aware draft-depth schedule | revision `e56-r2` requested |
| #57 | qwen-askeladd | **E55** compose `M=9` two-stream on the shipped table | revision `e55-r2` requested |

## Next research directions

Ordered by expected value against the `0.5366 %` deficit.

1. **Submit the certified E60 winner.** If our composite is neutral we land near
   `3.2493`, which is `+0.519 %` over our best row and `0.017 %` behind the
   crown. If it is positive by more than `0.017 %` it takes the board. If it is
   negative, E60 rung 2 names the hunk group and we delete it.
2. **E56 x E59 2x2 factorial.** The two experiments are **substitutive, not
   additive**: E59 sets `streams(M5) = 1`, which deletes the 4-to-5 boundary that
   edward's attribution says produces 100 % of E56's gain. Naive summation
   overstates the composite by about `2x`. Assign once both terminal results
   land.
3. **Certified exact target LM-head screening**, `+2.0 %` to `+2.9 %`. An
   offline, input-independent conservative bound plane screens rows of the
   248,320-entry readout; survivors get exact affine-4 with identical per-row
   arithmetic, so the top two are bit-identical. Lands in a new kernel library,
   so it has **zero interaction with the 108-register floor**, and the sidecar in
   `Sources/MLXFastTransform/` is untouched by all 712 rival trees. Free first
   rung: dump traced hidden vectors, compute the plane in Python, measure
   survivor density. Unblocked by E58's `77.2 ns` per-dispatch tax, which makes
   multi-pass screening nearly free on the dispatch axis.
4. **Hierarchical certified shortlist for the head's coarse readout**, `+1.5 %`
   to `+2.2 %`. The flat 2-bit scan over 98,336 compact-vocab rows is about 40 %
   of a head step, and the step is about 86 % pure weight streaming at a
   saturated 243 GB/s, so byte cuts transfer nearly 1:1. An 8-bit
   per-block-of-64 upper-bound plane costs about 1/32 of the bytes. The shortlist
   is provably identical, so drafts are identical and the accept trajectory is
   frozen. Changes zero weights, so it is not the closed head-replacement
   direction. Also unblocked by the `77.2 ns` tax.
5. **`mx.compile` the head draft chain**, `+0.5 %` to `+2.0 %`. About 2.4 ms per
   draft step of host graph build, 11–13 ms per round at ranked depths. The head
   is fc plus one full-attention layer plus norms, which is an eligible shape;
   the full target is ineligible because of its 48 recurrent layers.
6. **Smaller command buffers.** E58 falsified *larger* buffers and showed that
   buffer boundaries are pipelining opportunities: removing 16.85 buffers per
   round cost `+26.7 microseconds` each, opposite in sign to the `+7.76
   microsecond` submission cost. Fewer than 50 operations per buffer is untested
   end to end. It is a two-line edit to a candidate-editable constant that is
   live on the ranked memory profile, in the file with the lowest failure rate in
   the 712-tree corpus. **Do not extrapolate E58's one-directional slope across
   5x; it needs its own sweep.**
7. **Composition vehicle for the exact sub-MDE wins**, `+0.2 %` to `+0.5 %`, near
   zero risk: `pendingPrimaryDevice`, dead-KV-GEMM elision, fused last-merge plus
   final RMSNorm, top-32 finalize k-way merge. One PR, one hunk per mechanism,
   each with its own bit-exactness fingerprint A/B and one pooled ABBA absolute
   measurement. **Hand-apply hunk by hunk; never file-copy.**
8. **Elementwise and copy dispatches** are the largest single group in both legs
   (355.93 per candidate round, 1044.38 per serial round) and are pure data
   movement. Not yet sized.

Held below the line, with the reason: GDN checkpoint economics (the scan writes
about 600–700 MB per round of fp32 mid-states that are discarded on the ~44 % of
rounds that fully accept); GDN values-per-thread, bit-identical, `+0.1 %` to
`+0.4 %`; receipt-calibrated acceptance prior, better folded into E56 as a
calibration arm.

Deliberately not proposed: single QMV cells; warm coverage; the seed prefill,
which is scored but unreachable on our gen-16 host; SDPA chunk removal, which is
a discount and must be kept; head weight replacement, twice rejected at rank;
`MLX_MAX_OPS_PER_BUFFER` enlargement, now falsified; moment-based board
arithmetic, a proven null.

## Standing method rules added this round

- **Command-buffer geometry is part of the experiment identity tuple.** Export
  `MLX_MAX_MB_PER_BUFFER=512` and `MLX_MAX_OPS_PER_BUFFER=50` for every timed
  arm and report achieved operations per buffer as proof the setting took. The
  runtime force-set is gated on 96 GiB and has never fired on a student host.
- **Check headroom before pricing a per-prompt gain.** A gain above the next
  order statistic buys nothing.
- **Never extrapolate a two-point fit outside its anchor interval.**
- **Keep leg-reduction and `raw_p`-change in separately named functions.** All
  five recorded advisor pricing errors are basis confusions, not arithmetic.
- **A local cost curve is not a ranked cost curve.** Edward measured this host at
  `2.4x` the ranked per-row charge, so a depth-cutting mechanism flatters itself
  locally.
- **Do not use an early MLX touch to pin command-buffer limits ahead of a trusted
  setenv.** Edit the editable constant instead.
