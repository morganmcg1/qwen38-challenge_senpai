# E120 — own the QMV dispatch: scored pre-registrations

PR #121, assignment `qwen38-r1-e120-own-the-qmv-dispatch-custom-kernel-and-hoisted-activation-sums`,
revision `r1`. Base `2127858ba770ddc06027205d8df89a8db21d80f5`.

Every entry below states the prediction and the falsifier **before** the
measurement it scores. All endpoints are `harness=local`, Apple M4 Pro
`applegpu_g16s`, 48 GiB, macOS 26.5.2, Swift 6.3.3, host
`ip-10-231-2-95.ec2.internal`. No entry here is an official or ranked score.

## A standing caveat on the Metal census

`research/e120_census.py` reads register and spill counts out of the compiled
function. It cannot see what the compiler keeps live across the unrolled copies
of the k-loop, so a census reading is a **cost observation and never
correctness evidence**. Every correctness claim in this file rests on a
bit-exactness sweep with a positive control that is proven able to fail.

## Rung 1 — is a candidate-owned dispatch of the incumbent kernel free?

    Question:        Does dispatching an exact replica of MLX's
                     affine_qmv_fast from candidate code cost measurable time?
    Prediction:      candidate dispatch is at most +0.5% slower than MLX's own
                     launcher, averaged over M=3..9.
    Falsifier:       mean regression worse than +0.5%.
    Measured:        mean -0.188%, worst cell +0.401%, median of 12 per arm.
    Verdict:         PASS. Bit exact on 14 cells, 1,633,536 elements,
                     0 differing.
    Evidence:        W&B `p2513ks1`.

Host dispatch cost measured at the time: MLX 0.363 us against custom 2.038 us,
so +1.674 us per dispatch. That number predates the rung-5c contiguity guard
and is re-measured in rung 5f.

## Rung 2 — can an in-stream chunk-sum fill dispatch pay for itself?

    Question:        What does one extra fill dispatch cost inside the stream?
    Prediction:      25 to 40 us per dispatch.
    Falsifier:       below 22.7 us, the advisor's break-even. Route B viable.
    Measured:        4.85 us median.
    Verdict:         PREDICTION FALSIFIED, and falsified in the useful
                     direction. The fill is 5x cheaper than the break-even, so
                     the chunk-sum table is worth building.
    Evidence:        W&B `wqeuvini` (exactness plus census), `2ajd7gih` (fill).

## Rung 5a — is the routed candidate path bit exact everywhere?

    Question:        Does the shipped path reproduce quantizedMM bit for bit at
                     every routed shape, width and arm?
    Prediction:      0 differing elements on every cell, with all three
                     positive controls able to fail.
    Falsifier:       any differing element, or any control that cannot fail.
    Measured:        210 cells, 43,041,600 elements, differing_elements = 0,
                     max_abs_diff = 0.0, positive_control_can_fail = true on
                     all 210, restored_diff = 0 on all 70 sumtable cells,
                     x_hit = elements everywhere, meta_hit 98.38 to 99.48%.
    Verdict:         PASS, re-run at `31b33371` against the shipped gate.
    Evidence:        W&B `7icvulto` (first run), `e6i354j7` (shipped gate).

## Rung 5d-NA — does the table still pay when the accumulator group is narrow?

    Question:        M=6 and M=9 split their rows into groups of 3. Does the
                     chunk-sum table still pay there?
    Prediction:      net below +2 us per matvec at M=6 and M=9, because the
                     narrow group recomputes less.
    Falsifier:       net above +4 us at either width.
    Measured:        gdn.in_proj M=6 +9.28 us, M=9 +14.26 us.
    Verdict:         PREDICTION FALSIFIED. The table pays at every width from
                     4 to 9, on every shape of the round.
    Evidence:        W&B `qql6zari`, `iyzornb9`, `by2wpwg5`.

## Rung 5d gate — absolute additivity over accumulator groups

    Question:        Does gain(shape, M) equal the sum of per-group gains
                     gain_NA(shape, NA), so a per-(shape, NA) table can price
                     any width?
    Prediction:      (advisor) yes, supported by 6 M=8 cells whose
                     gain(M=8,[4,4]) / gain(M=4,[4]) averaged 2.034.
    Falsifier:       measured / predicted gain outside roughly 0.8 to 1.25 on
                     the multi-group widths.
    Measured:        n=28. mean 1.510, median 1.146, min 0.509, max 4.270.
    Verdict:         FALSIFIED for absolute microseconds. The [4,4] cells that
                     motivated it do hold (0.912 to 1.094), but every
                     NA=3 group fails: lm_head M=6 predicts 45.2 us and
                     measures 193.1 us.
    Replacement:     Fractional additivity holds. Gain as a percentage of the
                     same cell's base time is flat across all seven shapes at
                     every width from 4 to 9, sd 0.10 to 0.77 pp over a 74 to
                     215 GB/s span. Row-weighted per-NA fractions predict every
                     multi-group width within 0.81 pp.
    Consequence:     No table was shipped. All 42 measured cells at M>=4 pay,
                     so a width test is already the oracle there, and the whole
                     M=3 question is worth at most +62 us of a 68,410 us round.

## Rung 5e — does the mechanism move end-to-end decode? (PRE-REGISTERED)

Registered at commit `31b33371`, before the first 512-token leg.

    Question:        Does the candidate-owned QMV dispatch with the chunk-sum
                     table lower ABSOLUTE candidate MTP seconds per token in a
                     full 512-token native-MTP decode at the ranked offered
                     draft ceiling of 8?

    Evidence behind it:
                     The measured 7x7 grid predicts a saving over the 257 wide
                     QMV calls of one round of 4.09% at M=4, 4.82% at M=5,
                     2.28% at M=6, 4.10% at M=7, 5.68% at M=8 and 2.50% at M=9,
                     and 0% at M=3 where the gate declines. The end-to-end
                     effect is that number times the share of the MTP round
                     that the wide QMV calls occupy.

    Prediction:      absolute candidate MTP seconds per token improves by 2.5%
                     against the same-binary `off` arm.

    Falsifier / notify rule:
                     Report to the advisor without waiting if the measured
                     improvement is below 1.25% or above 5.0%, which is the
                     factor-of-two band he named. Otherwise continue to 5f.

    Controls that must hold:
                     - serial seconds per token differs between arms by less
                       than 0.5%. The serial leg decodes one row, `routable`
                       declines M=1, so the arms must agree there.
                     - all_tokens_matched = true on every leg. The wrapper
                       fails closed otherwise.
                     - residual_divergence_count = 0 on every leg.

    Design:          same binary, ABBA counterbalanced inside one session,
                     `off, sumtable, sumtable, off`, 512 decode tokens,
                     MLXFAST_QWEN_MTP_DEPTH=8. The wrapper's real 40C cool gate
                     runs before every timed phase and is not bypassed, so
                     these arms are gate qualified. They remain local: one
                     public fixture, candidate-generated reference rows.
                     Driver: `research/e120_rung5e.sh`.

    Why the same binary rather than two worktrees:
                     `off` makes `Qwen35CustomQMV.matmul` decline every cell,
                     so the routed call sites fall through to `quantizedMM`.
                     That reproduces BASE_SHA behaviour on the QMV path without
                     a build-identity confound. It is conservative: the `off`
                     arm still pays one nil-returning Swift call per matvec.
