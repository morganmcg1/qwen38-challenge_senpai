# SENPAI Research State

- **2026-08-23 12:45 UTC**

## Most recent research direction from the human researcher team

No new human direction since the last check. The standing direction is
unchanged: keep four students productive, submit credible candidates promptly
rather than polishing locally, and treat a moved bar as information rather than
as a reason to hold.

## Where the campaign stands

- Campaign best: **`0cf1637e` = 3.68278758**, rejected on submission because the
  bar had already moved. It remains our scientific frontier and the base every
  current experiment is measured against.
- The bar: **`ec24d591` newjordan = 3.72911001**, promoted 2026-08-23T11:43Z,
  source `0863b06a`. Its mechanism is the xsums fill fusion, which is our own
  thorfinn's Idea 3 built and shipped by a rival in about three hours.
- **The real gap is not the published gap.** On the fair median (Finding 255,
  corroborated independently by askeladd's Rule 147 fit at
  `k = 205.4 +- 15.4 us/round`), the previous bar led us by +0.4006 % of the
  candidate leg while the published gap read +0.9995 %; the difference is a pure
  serial lottery draw of +0.5989 pp on essays. Against the new bar the required
  uniform candidate-leg speedup is **+1.0042 %**.
- Advisor base `b27c004a`. Growth budget is tight: **66,648 bytes** of shared
  headroom remain across four students.

## Current research focus

**One theme dominates: composition of small, independently measured,
independently readable candidate-leg mechanisms, submitted quickly.** Nothing
in the queue is worth a full per cent on its own except one line of code. The
gap closes by adding four or five verified fractions, not by finding a single
large idea.

Four live experiments, one per student, one per physical Mac:

1. **#150 edward — E150 per-round discrimination.** The linearised draft-depth
   scheduler: a global argmax of `lambda*(1 + E[A_d]) - C_d` priced on the E145
   measured width-cost curve, with both `pendingTop2` sigmoid clamps removed.
   Built, green, submission authorised and re-confirmed after the bar move. Its
   value is not only the +0.75 to +0.97 % it may publish; it is the campaign's
   **first clean ranked receipt for a schedule-changing mechanism**, which Rule
   122 says no local replay can supply.
2. **#151 alphonse — the ranked prefill channel.** The 128x32 rectangular NAX
   seed retile, armed and gate-green, with a pre-registered -3.5 % prefill
   prediction. Prefill is a separate published channel (Rule 146) worth 1.37x
   more than the campaign assumed (Finding 254), and three rival receipts prove
   the family lands between -4.1 % and -5.0 %.
3. **#152 thorfinn — the width table and the fill producers.** Promoted to A0:
   drop the `onePass6` rung. Then the 127-site boundary-fused fill producer and
   a new custom SwiGLU kernel for `mlp.down`'s 64 sites.
4. **#153 askeladd — merged SDPA and leaf16.** The full-attention SDPA split
   costs 154.5 +- 14.2 us/round and leaf16 is built but never shipped.

## The single most valuable open item

**FINDING 259 and RULE 155.** A rival isolated the width-6 one-pass rung on the
ranked runner (`c47b45be`, single mechanism, schedule-identical, z +4.16) and it
costs **+1.0583 % of the candidate leg**. Our tree ships that rung inside
`onePass67`. Our own isolation of `shipped -> onePass67` reads -1.5571 %, so by
subtraction the width-7 rung alone is worth -2.6154 % and the width-6 rung is
pure loss.

Rule 155, the partition symmetry law, explains why: **M=7 is the only unbalanced
partition in the shipped table** (`[4+3]`), so it is the only width where
collapsing two groups into one removes real critical-path idling. At M=6, M=8
and M=9 the partition is already balanced, so one-pass buys nothing and still
pays the register and occupancy tier. The law retro-predicts every width-table
measurement the campaign holds, including T45's -13.2472 pp forecast miss.

Dropping `onePass6` is one enum case and one changed default. Expected value
**-1.06 % of the candidate leg against a +1.0042 % gap.**

## Potential next research directions

**Ready to submit, in order.**

1. **A0 + E151 R1 on one receipt.** Decode channel and prefill channel are
   published separately, so one row reads both independently. Expected about
   -1.41 % of the candidate leg. This is the submission after Edward's.
2. **The remaining fill ladder.** 130 of 257 wide QMV fills are still
   unproducer-fused: `mlp.down` 64 (+0.144 %), `gdn.out_proj` 48 (+0.108 %),
   `fa.o_proj` 16 (+0.036 %), `lm_head` and the layer-0 entry norm 2 (+0.005 %).
   The ranked dispatch price is 1.18 us and is now confirmed by two independent
   ranked coverage points.
3. **Rung B2, reinstated.** The affine NAX path has no loader helper and no
   double buffer; `43925f29` and `a9dd132a` both bought about -4.13 % of prefill
   by adding them. It composes with the retile only partially and must be
   measured on a separate receipt.

**Open mechanisms with no owner.**

4. **The acceptance axis, framed correctly.** Acceptance leverage is exact:
   `d(raw)/raw / d(alpha) = edl / (1 + alpha*edl)`, giving budgets of 470.7
   us/round on beagle and 502.0 on essays. **+0.0108 of acceptance publishes
   above the old bar — about three times the entire 128-site fill saving.** But
   the board has now run 119 same-solver head swaps: 16 bought at least +0.01
   acceptance and **zero paid for themselves**, best efficiency 0.70 against
   break-even 1.00. The open question is therefore not a better head, it is a
   **cheaper head at equal quality**. We already hold the best head on the
   board (`559b24eb`, 561 of 609 beagle rows and every top score).
5. **The width-7 rung isolated on its own ranked receipt.** Currently known only
   by subtraction from two receipts on two different trees.
6. **The seed prefill outside the GEMM.** The NAX GEMM is the one scored kernel
   family no machine we own can execute, but GDN prework, SDPA and the norms
   inside the 512-token seed leg are all locally executable and are 10.04 % of
   the candidate leg by Rule 148 weighting. Nobody has enumerated that share.
7. **A cleanup PR to reclaim growth budget.** 66,648 bytes remain. `onePass6`,
   `onePass678`, the E87 remnants and the retired pb5/pb6/pb7/pbfit depth arms
   are all dead once A0 lands.

**Themes to escalate to if the composition stalls.**

8. **Verification batching at wider row counts.** `requireStructurallySound`
   forces `M == d + 1` and a single linear chain, so tree drafting is
   structurally blocked; the remaining freedom is in how the fixed chain is
   evaluated, not in what is proposed.
9. **Weight layout and resident metadata.** The round is 88.6 % DRAM weight
   streaming and the QMV family is 88.6 % of decode GPU time. Every per-round
   microsecond we have found so far has come from dispatch overhead rather than
   from moving fewer bytes.
10. **Reading the board as an instrument.** Two of this session's largest
    findings came from rival public notes rather than from our own GPUs:
    Finding 258's ranked coverage point and Finding 259's width-6 isolation. A
    standing, cheap, systematic pricing pass over every resolved rival row is
    worth as much as a GPU slot.

## Standing discipline

- Price every ranked contrast on the **candidate leg**, never the published
  median (Rule 118, Rule 63).
- Name the pricing frame and the sign convention in words (Rule 144). The
  general 2-sigma MDE is **0.1547 pp**; the conditional **0.0872 pp** applies
  only when neither side is anchored on `684821ed`.
- **Rule 154:** a measurement that cannot change the build decision must not
  gate the build. This cost us the fill fusion.
- **Rule 155:** state the balance of the partition a one-pass rung replaces
  before proposing it.
- Integrity boundary unchanged: no prompt-family-matched head training, no
  benchmark-phase detection, no acceptance relaxation. Two rivals crossed it
  and both were rejected.
