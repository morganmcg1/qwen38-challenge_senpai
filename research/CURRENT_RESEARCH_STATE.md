# SENPAI Research State

- 2026-08-23 ~11:50 UTC
- Most recent human research direction: none new this cycle. The standing
  direction remains the one in `senpai/program.md`: maximise the official
  decode score on the `qwen3.8-27b-mtp-v1` track, submit the strongest
  legitimate candidate promptly, and never weaken correctness, thermal,
  provenance, scope or submission gates to go faster.

## Where the campaign stands

Our best official row is `0cf1637e` at 3.68278758. The bar is `684821ed` at
3.71959723 and has not moved for about twelve hours. The published gap is
+0.9995 percent, but Finding 255 splits that gap into two parts: a pure
serial lottery term of +0.5989 pp that no candidate edit can touch, and a
real candidate-leg gap of **+0.4006 percent** on the fair median. We plan
against the published +0.9896 percent that a submission must actually clear,
and we price mechanisms against the +0.3990 percent parity point as the
first milestone.

The campaign frontier is a scheduler question again. Edward's E150 shows that
the depth-price belief table that compiles today is not the table that best
explains the measured width-cost curve, and that swapping both the table and
the decision rule is worth between +0.75 and +0.97 percent of the published
median. That is the largest single priced item on the board and it is in the
submission slot now.

## Current research focus

1. **The draft schedule.** E150 R4 replaces the shipped first-break walk with
   a global argmax of `lambda*(1 + E[A_d]) - C_d` over the measured ranked
   width-cost curve, with both `pendingTop2` sigmoid clamps removed. It clears
   the +0.6 pp gate at +0.8543 pp minimum over both curves. Submission is
   authorised with a Rule 148 weighted per-prompt abort at +0.30 pp.
2. **Dispatch count, not fusion.** Thorfinn measured the chunk-sum table fill
   at 1.711 microseconds on M4 Pro, about a quarter of what Finding 252
   assumed. The fill is essentially pure encode overhead, so the lever is the
   number of dispatches removed rather than any arithmetic saved. The
   remaining question is how many of the 257 wide-QMV sites can have their
   table emitted by a producer kernel that already runs.
3. **The ranked prefill channel.** Alphonse owns the 128x32 NAX retile. Rule
   153 now proves the JIT library partition by inspection, so a retile that
   touches only `quantized_nax.h` and its twin cannot reach the decode QMV
   family. Finding 254 raised the value of every prefill saving by 1.37x.
4. **Attention streaming.** Askeladd priced the shipped full-attention split
   at 154.5 microseconds per drafting round in the ranked frame. The kernel
   dispatch path is closed by the vendored C++ guard, but a custom merged
   Metal kernel invoked from editable Swift remains open and is worth about
   +0.30 percent.
5. **Measurement discipline.** Three findings this cycle were corrections to
   our own pricing: Advisor Error 181 withdrew a prefill marginal that was
   confounded with a whole-row slowdown, Advisor Error 182 found that our
   replay baseline was not the code path that compiles, and Advisor Error 183
   found a 4x error in a coverage ladder. Rules 151, 152 and 153 exist to stop
   each of those recurring.

## Potential next research directions

- **Custom merged SDPA kernel (C3).** Replace the two-call causal split in
  `AttentionUtils.swift` with one custom Metal kernel that visits the same key
  set in the same per-row order. The `Qwen35CustomQMV` Route B work is the
  precedent. Prize about +0.30 percent. Unclaimed.
- **Producer-emitted chunk-sum tables.** Enumerate every kernel already on the
  scored path that produces the exact activation a wide QMV consumes, and ask
  which can emit the table as an extra output with no new dispatch. The fused
  residual and RMSNorm path already covers 128 sites; the fused MLP path may
  add 64 more.
- **Composition after the scheduler lands.** Leaf16 on the shipped vocabulary
  is a weak but clean local winner worth about +0.10 percent, and it scales
  with width-8 mass, so it is worth more after a schedule that widens. It is
  held standalone and submission-ready.
- **A host-side readback signal for the scheduler.** E150 R2's demand curve
  says a per-round signal with AUC 0.865 is worth +1.58 to +3.09 percent
  depending on where it is read, and the break-even host readback cost is
  7.75x the stop rule. This is the largest unpriced upside on the board.
- **Pipelined double-buffered K-loop in the NAX GEMM.** Unpriced. Alphonse
  holds it as E151 R2.
- **A cleanup pull request.** Growth budget headroom is down to roughly 66 KiB
  of 262,144. Pruning the retired depth-price arms, the width-pin environment
  gates and the stale table and grid arms would reclaim budget and make the
  winning path the only path. Unassigned.
- **Nibble entropy of our own checkpoint.** A one-hour zero-GPU corrective
  study, available if a student slot opens with no GPU work.

## Standing constraints that shape all of the above

- Only one submission may be in flight at a time, and validation takes 42 to
  130 minutes. The slot is Edward's until his result resolves.
- Rule 138 no longer gates a ranked mechanism. The admissible width set
  `[1,2,3,4,5]` is a property of one local curve on g16s hardware and does not
  transfer to the ranked g17s host at widths 6 through 8.
- Row `7226dc9a` is excluded from all pricing. It carries an unexplained
  whole-row slowdown of about 2.2 pp on prefill and 4.4 pp on decode.
- The general 2 sigma minimum detectable effect for a ranked candidate-leg
  contrast is 0.1547 pp on the realised median pair, or 0.0872 pp when
  neither side is anchored on the bar.
