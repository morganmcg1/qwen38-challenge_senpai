# SENPAI Research State

- 2026-08-23 10:12 UTC

## Most recent research direction from the human researcher team

No new human direction since the standing instruction to keep the campaign running without waiting for approval. The advisor is deciding experiments, submissions and closures autonomously under `senpai/program.md`.

## Where the campaign stands

The bar is `684821ed` at 3.71959723, promoted at 01:45Z and unmoved for over ten hours. Our best published row is `572b2cc4` at 3.66218564, a real candidate-leg gap of about 0.82 %.

Submission `0cf1637e` went out at 09:23:48Z and is validating. It retires `pb6`, restores `onePass67`, and keeps the tight launch grid, probe 0.15 and the width-2 route. Forecast 3.68 to 3.69.

Seven rival rows are validating alongside it. Two of them, `570e0e35` and `ec24d591`, are independent attempts at the same xsums fusion we are holding for thorfinn, so either one publishing gives us a free ranked read on that mechanism.

## Current research focus

**One. The schedule is the only lever left inside our structural constraint.**

`requireStructurallySound` in the trusted driver forces a single linear chain, so tree drafting, hedge rows and multi-candidate verification are structurally impossible for us. arXiv:2411.00841 Theorem 2 proves that inside the unbiased single-chain class no rejection improvement is available at all. Everything therefore has to come from either the per-round schedule or from making the round itself cheaper.

Edward owns the schedule half as E150. The measured headroom is large and unusually well characterised: a clairvoyant per-round policy gains +6.3508 % against the shipped +0.1338 %, a gap of +6.2170 pp, while a policy with the *perfect marginal distribution* actually loses 0.5144 pp. The acceptance axis has split into two oracles and only the per-round one is worth anything. The literature says methods of this kind recover 25 to 35 % of their own oracle gap, so the honest target is +1.5 to +2.5 pp and the stop rule is written against that number.

**Two. Every price this round rests on a cost curve measured on the wrong host.**

Rule 138's admissible width set, the Rule 143 discount, and the whole E150 scoring all derive from edward's local curve divided by one scalar. The two hosts diverge exactly where the decision lives: our Macs spill at NA=6, the ranked M5 not until NA=8, and the ranked host shows a residency recovery at width 8 that we never see. Askeladd is now solving the ranked curve directly from public receipts, with a synthetic recovery study as the gate. If width 6 or 7 turns out admissible on the scored host, a large part of this round's pricing is mis-framed and the depth axis reopens.

**Three. The prefill is a real, large, unclaimed channel that we cannot screen locally.**

Prefill is 8.45 % of the candidate leg and the best prefill result anyone has published is -4.97 %. On our value model that converts to +0.3822 % of the median, about 46 % of our gap to the crown. Alphonse has just proved the ranked host takes a different GEMM entry point from every machine we own, so there is no local instrument at all: the submission *is* the experiment. His 128x32 NAX retile is the live attempt.

**Four. Detection floors now govern what is worth building.**

Askeladd's per-prompt noise block turned the receipt into a calibrated instrument. Essays resolves 44.5 us per round, beagle 119.5, plutarch 211.9. The median-pair MDE is 0.1547 pp. Several ideas that looked plausible are now provably below the floor and have been killed on arithmetic rather than on GPU time, which is the cheapest possible way to close them.

## Recent closures worth remembering

Board mining is finished. E148 asked whether any rival's unclaimed mechanism was worth taking and the answer is no: the best survivor is under our detection floor and the other is already in our tree. The exercise still paid for itself in instruments — Rule 147, the n=11 noise block, a corrected E85 price and a robust tail-excess result.

`pb6` is retired. It drafts hard on plutarch, the one prompt with exactly zero median weight, and pays for it on beagle. Both local instruments had the sign inverted because local width mass sits at 0.62 to 0.77 against a ranked 0.5390.

Two shippables were killed by a single dispatch table this session: rung A and rung E-1b both edit code paths the ranked host never reaches.

## Potential next research directions

- **Recover the per-round oracle.** The predictor is the bottleneck, not the axis. The strongest published feature is the target's own final-layernorm hidden state, which the scored path already computes, trained as a regression with asymmetric L1 on first-rejection positions only.
- **Fix the policy form.** Our threshold rule compares against the current round's realised rate where renewal-reward theory calls for a global fixed point. Free to test, offline.
- **Settle the ranked cost curve.** Then re-derive the admissible width set, Rule 143 and E150's scoring on it.
- **Prefill.** The NAX seed retile, and after it any other prefill mechanism, now that Rule 146 lets a prefill-only and a decode-only mechanism ride one submission and be read independently.
- **The 6->7 cliff.** 25,861 us on the measured curve, 2.32x the step below it, still unattributed to a mechanism.
- **Round-cost reduction below the QMV family.** The QMV subtotal is 88.6 % of the round and the fused residual, SDPA and GDN prework together are only 1,100 to 1,750 us, so the leverage is concentrated where the kernels are hardest.
- **Corrective, unclaimed, cheap.** Nibble entropy of our own checkpoint; the GDN S=2 mid-state write gate; a cleanup PR to delete the retired depth-price arms and stale table, grid and entry arms now that the winner is decided.
