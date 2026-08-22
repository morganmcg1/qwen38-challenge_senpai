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

---

# Measured outcomes

Everything above this line was written before its measurement. Everything below
was written after. `harness=local` throughout. Host `ip-10-231-2-95`, Apple M4
Pro (`applegpu_g16s`), 48 GiB, macOS 26.5.2, Swift 6.3.3.

## Rung 5e — the pre-E121 ABBA session

W&B `zkcfcaxr`. Base `2127858b`, worker `f51cb223…`, 8 legs, 512 tokens,
depth 8, gate qualified.

    arm off       median 0.030754   mean 0.030760   sd 2.4e-05   n=4
    arm sumtable  median 0.029447   mean 0.029456   sd 2.8e-05   n=4
    HEADLINE      +4.249 % leg      +4.036 % ranked

Registered prediction was +2.5 % with a notify band of 1.25 % to 5.0 %. The
measurement lands inside the band, above the point estimate.

Serial control: off 0.073824, sumtable 0.073531, spread 0.397 %, inside the
0.5 % limit. Exactness held on every leg.

Two findings the advisor adopted from this session:

- **Harness defect 27.** The same mechanism reads +1.127 % at a 64-token
  window and +4.249 % at 512 tokens, a 3.8x dilution. Short windows
  systematically understate a per-round mechanism because seed prefill is
  amortised over fewer decoded tokens.
- **F95.** The wide-QMV-to-leg transfer is not the constant 0.6070 that E116
  measured. This session implies 4.249 / 5.337 = **0.796** at its own mean
  verify width of 7.359. The coefficient is width dependent.

## Rung 5g — the post-E121 ABBA session, and the headline

W&B `qqjlgtkv`. Base `4c4db8a2` (advisor `3f40d9b0` merged), worker
`0eee61f8…`, 8 legs `off,sumtable,sumtable,off,off,sumtable,sumtable,off`,
06:23:59Z to 06:54:32Z. `cool_gate_passed_real_gate=true`,
`gate_qualified_for_timing=true`.

    arm off       median 0.030629   mean 0.030627   sd 2.50e-05   n=4
    arm sumtable  median 0.029461   mean 0.029463   sd 1.28e-05   n=4
    HEADLINE      +3.813 % leg      +3.622 % ranked

    per leg   01-off 0.0306153  02-sum 0.0294516  03-sum 0.0294523
              04-off 0.0305977  05-off 0.0306425  06-sum 0.0294699
              07-sum 0.0294771  08-off 0.0306522

No overlap between the arms. Within-arm CV 0.061 %, so 2 sigma is 0.087 % and
the effect is 44x that. Entry-temperature spread is -0.089 C, with the
candidate arm the colder one, so thermal drift cannot manufacture the result.

The advisor pre-registered +3.83 % leg for this session. Measured +3.813 %,
0.02 pp apart.

All eight legs report `accepted_draft_rate = 0.8770161290322581` and
`effective_mean_draft_len = 6.358974358974359`, identical to the digit. The
arms therefore did identical work and the contrast is pure time.

**E121 measured in situ.** The `off` arm of 5g against the `off` arm of 5e
isolates the merge, because nothing else changed on the QMV path: **0.433 %**
on means, 0.407 % on medians. Alphonse's independent standalone figure is
0.436 % (W&B `qmr3mgl8`). Two very different mixes, 0.003 pp apart.

## Rung 5f — the `--local-submit` gate and a free metallib control

W&B `idpab9oe`. Commit `1abb7e21`, worker `0eee61f8…` — byte-identical to the
5g worker, so the only variable against 5g is the `mlx.metallib` rebuild.

    5g sumtable arm  mean 0.029462725  sd 1.279e-05  range [0.0294516, 0.0294771]
    5f candidate          0.029451771
    delta                -0.0372 % of the leg,  -0.857 sigma,  inside the range
    stop rule             0.080 %          VERDICT: immaterial

Both 5e (25 warnings) and 5g (3 warnings) ran against a stale metallib. The
staleness predates this branch — 5e ran at `2127858b` — so it was host-wide,
not something the merge introduced. It does not move the answer: 5f reproduces
the candidate leg to within 0.86 standard deviations of the 5g arm, and
measuring 5f against the 5g `off` mean gives -3.837 % leg against 5g's own
-3.801 %.

Gate results: `all_tokens_matched=true`, `residual_divergence_count=0`,
574/574 rows, `public_drift_tripwire_passed=true`, serial 0.073838 s/tok,
local ratio 2.5071. Cool gates 39.8C, 39.9C, 39.5C, all real.

**Head provenance.** The first 5f attempt (job `df62368d`) failed at 123 s in
reference generation with `exit_status=15`. Cause: `MLXFAST_QWEN_MTP_HEAD_DIR`
was unset, so `setup-qwen-mtp.sh:76` supplied the organizer-pinned head. The
shipped two-level index needs `draft_lm_head.*` and `precision_islands.*`,
which exist only in the declared head. Any local run of this candidate must
point at `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run`.
`research/e120_rung5f.sh` now resolves it and fails closed.

## Cross-architecture residency census — a clean negative

`research/e120_g17s_census.py`, output `research/out/e120-g17s-census.json`.
Zero GPU seconds. `xcrun metal-tt`, flags `-std=metal4.0 -O2 -fno-fast-math`,
matching `research/e123_arms.py`. Residency is
`SIMDGROUP_BUDGET // registers`, with the budget 3072 on g16s and 3968 on
g17s.

    g17s              registers  spill  text B  resident simdgroups
    replica_no_table       101      0    52914       39
    sumtable               102      0    49718       38
    E123 a_base            101      0    25898       39
    E123 n_nosums           99      0    23608       40

**Route B does not reach 40 resident simdgroups on g17s.** It reads 102 and 38,
one worse than the incumbent. The residency gift is not available.

The per-width rows say why, and this is the most useful thing the census
produced:

    g17s sumtable   M=3  M=4  M=5  M=6  M=7  M=8  M=9   switch
    registers        90   94  102   90   94   94   90      102
    resident         44   42   38   44   42   42   44       38

A switch entry point's register count is the **maximum over its inlined
branches**, because mutually exclusive branches reuse registers. E123's `NA=0`
row is that maximum: g17s 101 = max(93, 89, 90, 101). So the M=5 branch alone
costs the whole dispatch four resident simdgroups **at every other width**.
M=5 is 6.4 % of these rounds and it penalises the 76.9 % that run at M=8.

Also recorded: the sumtable arm costs **more** registers than the replica at
every width, 102 against 101 at the maximum. That is the same direction as
askeladd's 91 to 92 at NA=4. Hoisting the sums trades arithmetic registers for
a bound buffer and its loaded values. Text falls; registers do not.

## Repricing onto the ranked width regime

`research/e120_ranked_width_models.py`, output
`research/out/e120-ranked-width-models.json`. Against the post-E121 control:

    model                      wide-QMV %   leg %              ranked %
    A  measured local           4.852        3.862 predicted    3.622
                                             3.813 measured
    B  ranked point estimate    3.170        2.019             +1.918
    C  adverse bracket          2.153        1.390             +1.321

B/C is 1.452. A uniform session attenuation of 0.9091 instead of the per-width
model gives B +1.816 % and C +1.286 %, so the brackets are B [1.82, 1.92] and
C [1.29, 1.32].

The local fixture runs at mean verify width 7.359; the ranked prompts run
near 5.4 to 6.3. That gap, not any measurement doubt, is the whole difference
between A and B.

**M=5 is the only width E121 does not touch, and it is the width nearest
beagle's mean.** Beagle carries 48.6 % of the median weight.

Model A is an independent check on the repricing shape rather than a fit to
it: the post-merge model predicts 3.862 % leg and the session measured
3.813 %.

## Follow-ups this experiment did not implement

1. **Template the QMV entry point on `M`.** MLX would compile and cache one
   specialization per width. M=8 would run at 94 registers and 42 resident
   simdgroups on g17s instead of 38, and instruction text would fall from
   about 50 KB to 6-12 KB per pipeline. This is the largest unmeasured item
   found. It changes the kernel, so it invalidates 5g and needs a fresh ABBA.
2. **Measure the ranked width histograms, not their means.** Model C is very
   nearly "assume everything runs at M=6". A measured histogram for beagle
   alone would collapse most of the B-to-C bracket.
3. **Re-price the 5c contiguity guard.** It is still costed against rung 1's
   +1.674 us/dispatch, which was measured before the guard existed.
4. **Probe achieved GB/s against live achieved GB/s per cell**, which would
   settle whether the transfer class really is flat in bandwidth or merely
   flat over the range the probe reaches.

