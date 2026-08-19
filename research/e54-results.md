# E54 — lone versus sibling NA=5: four cells, four laws, one surviving law

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"p1_m5_lone_na5_delta_pct","available":true,"value":-20.253},"test_metric":{"name":"law_a_prime_survives_all_four_cells","available":true,"value":1}}

Label: **local winner at the cell level, and a hard warning at the score level.**

## Answer

`<T,5,5>` is the fastest cell in the table, not the regression the brief expected.
The lone NA=5 group wins **−20.253 %**, and both mixed-sibling cells regress. Law C
predicted the opposite pattern in both places, so its sibling-overlap term has the
wrong sign rather than the wrong size.

The surviving law is **A′, working-group weight traffic**: each working group
re-reads the whole weight matrix, so cost scales with `ceil(M / IPG)` and not with
NA. Everything else in the table is a small correction to it.

| cell | structure | groups | traffic | A′ predicts | measured | reproduced by |
| --- | --- | --- | --- | --- | --- | --- |
| M=5 | **lone NA=5** | 2 → 1 | ×0.501 | −19.00 | **−20.253** | P4 at −20.110 |
| M=7 | sibling 5+2 | 2 → 2 | ×1.000 | 0.00 | **+0.994** | — |
| M=8 | sibling 5+3 | 2 → 2 | ×1.000 | 0.00 | **+1.345** | — |
| M=9 | sibling 5+4 | 3 → 2 | ×0.668 | −12.26 | **−11.548** | E49 at −12.255 |

Two independent reproductions support the table. P1 and P4 measure M=5 through
different arm families and agree to 0.14 points. P4's M=9 and E49's M=9 were
measured in different sessions on different tables and agree to 0.71 points.

## The mechanism, measured rather than inferred

The advisor's standing bandwidth objection is now closed by measurement. The lone
working group's achieved device rate falls almost exactly linearly with its width:

```
NA=2  223.8 GB/s      NA=4  175.2 GB/s
NA=3  199.7 GB/s      NA=5  150.9 GB/s     (-24.3 GB/s per extra row)
```

So an NA=5 group really is bandwidth-inefficient, exactly as PR #8 recorded. It is
simply not inefficient enough. At M=5 the traffic falls by ×0.501, break-even on
this host is **120.4 GB/s**, and the group sustains **148.4 GB/s**. The saving wins
by a wide margin.

The same ladder explains the two regressions without any new term. At M=7 and M=8
the group count does not change, so there is no traffic saving to collect and only
the rate loss remains:

| pair | traffic | achieved rate | time |
| --- | --- | --- | --- |
| P2, M=7 | ×1.000 | 209.0 → 206.9 GB/s, −0.98 % | **+0.994 %** |
| P3, M=8 | ×1.000 | 194.2 → 191.6 GB/s, −1.33 % | **+1.345 %** |

The rate loss and the time cost agree to within 0.01 points in both pairs. The NA=5
penalty is real, it is about 1 %, and it is paid by the cell that carries it.

One prediction is worth recording because it was made before the measurement
existed: I extrapolated the lone-group ladder to **151.7 GB/s** for NA=5 and the
measurement returned **150.9**.

## Law D: mechanism confirmed exactly, consequence not visible

`research/e54_reg_census.py` drives the E46 compile machinery with the real
`research/e54_arms.py` patches, so it measures the source that was timed.

| arm | kernel max | entry `affine_qmv_fast bfloat16_t,64,4,false` |
| --- | --- | --- |
| `shipped` | **108** | **163** |
| `e27_m5_only` | 125 (+17) | 182 (+19) |
| `e27_m9_only` | **129 (+21)** | 181 (+18) |
| `e27_full` | **129 (+21)** | **183 (+20)** |

Both numbers the E27 revert recorded, 108 → 129 and 163 → 183, reproduce to the
digit and untuned. Law D's mechanism is therefore confirmed: adding NA=5 cells does
raise the shared register maximum.

Law D's distinguishing *consequence* is that widths the edited cell never executes
must slow down. P4 is the first experiment able to test that, because it keeps all
seven width cases live. They did not slow down:

```
M=3 +0.123   M=4 -0.207   M=6 +0.056   M=7 -0.069   M=8 +0.364     bar 0.991 %
```

I am deliberately not calling this a refutation. The width sweep is the same
instrument that predicts +2.2 % of score for a composite the board scored at
−0.3321 %, so it is demonstrably blind to whatever costs E27 its points. An
instrument that missed a 2.5-point effect cannot be trusted to exclude a 0.3-point
one. Law D stands as unresolved at the end-to-end level, and the honest statement is
that **the register rise is real and the width sweep cannot see it costing anything.**

**Attribution matters for PR #57.** `<T,9,5>` alone moves the kernel maximum the full
108 → 129. It is not a partial contributor to E27's rise; it accounts for all of it,
while carrying the least score weight of the four cells.

## The result that outranks the law table

P4 measured E27's exact composite on the real shipped table for the first time:
M=5 −20.110 %, M=9 −11.548 %, every other width flat. Priced with
`research/e49_price.py`, `harness=ranked`:

| mixture | predicted score | board observed | gap |
| --- | --- | --- | --- |
| e48 | +2.2118 % | −0.3321 % | −2.5439 **sign differs** |
| e53_low | +2.0447 % | −0.3321 % | −2.3768 **sign differs** |
| e53_mid | +2.2890 % | −0.3321 % | −2.6211 **sign differs** |
| e53_high | +2.5334 % | −0.3321 % | −2.8655 **sign differs** |

Both cells win locally by large, well-resolved margins. Every mixture predicts a
solid board gain. The board lost. The disagreement survives the E48-versus-E53
mixture dispute, so it does not depend on adjudicating that dispute, and it is far
too large to be ranked jitter: the published median's 2 sd threshold is 0.283 %,
worst case 0.527 %, against a 2.4–2.9 point gap.

**No law in this table explains it.** Laws A, B, C and D were all built to explain
cell timing, and cell timing is no longer the unknown — it is measured, reproduced
and consistent. What is unexplained is the step from cell timing to score.

That is the campaign-level finding, and it is a warning rather than a prize: a
QMV cell win of 12–20 %, composed on the real table, produced no board gain. Any
promotion priced from per-width QMV cells inherits this gap.

## Per-width price of each measured cell

| M | win % | e48 share | e48 score | e53_mid share | e53_mid score |
| --- | --- | --- | --- | --- | --- |
| 5 | +20.253 | 12.1744 | **+1.3481** | 22.3363 | **+2.0187** |
| 7 | −0.994 | 4.6307 | −0.0310 | 14.1607 | −0.0948 |
| 8 | −1.345 | 4.7603 | −0.0431 | 8.3949 | −0.0761 |
| 9 | +12.255 | 21.6296 | +1.4084 | 6.7712 | +0.5590 |

M=5 is the most valuable single cell under the E53 mixture and second under E48.
Under both mixtures the two regressions are worth about −0.1 % and are not worth
composing.

Treat these prices as an upper bound, not a forecast. The same pricing model applied
to E27's composite is wrong by 2.4–2.9 points against the only board anchor we have.

## Correctness

Bitwise parity across 192 `(bits, M, shape)` cells per arm, nine arms:

| comparison | result |
| --- | --- |
| P1 `<T,5,3>` vs `<T,5,5>` | BIT-IDENTICAL, 0/192 |
| P2 `<T,7,4>` vs `<T,7,5>` | BIT-IDENTICAL, 0/192 |
| P3 `<T,8,4>` vs `<T,8,5>` | BIT-IDENTICAL, 0/192 |
| P4 `shipped` vs `e27_full` | BIT-IDENTICAL, 0/192 |
| positive control, lanes 3 and 4 swapped | **DIVERGES, 8/8 cells at exactly (bits=4, M=5)** |

No delta at M ≤ 9 in any arm, so the hard stop never fired. The positive control
proves the instrument can fail. `bits=3` never enters a crossrow kernel in any arm
because the family is specialised `affine4_g64`, which is why the positive control
diverges only at `bits=4`.

`vec<float,5>` gate: `sizeof` 32, `alignof` 32, AIR shows
`alloca [4 x <5 x float>], align 32`, all five lanes correct. Honest limit:
`acc_alloca_types` is empty for the real cells because the accumulators are
register-promoted, so this is a front-end and memory fact, not a register-cost
measurement. This independently reproduces askeladd's padding constant on a
different host.

Routing isolation, read out of the built binaries rather than the patch text: P1, P2
and P3 each change exactly one cell and leave 23 of 24 controls byte-identical. P4
changes `(4,5)` and `(4,9)` and leaves 22 of 24.

## Reproduction

```bash
# timing, 16 legs, four ABBA pairs
research/e54_session.sh iso_m5_ipg3:e54-p1-a1 iso_m5_ipg5:e54-p1-b1 \
  iso_m5_ipg5:e54-p1-b2 iso_m5_ipg3:e54-p1-a2 \
  --widths 1,2,3,4,5,6,7,8,9,10 --reps 21 --inner 10
research/e54_session.sh shipped:e54-p4-a1 e27_full:e54-p4-b1 \
  e27_full:e54-p4-b2 shipped:e54-p4-a2 \
  iso_m7_ipg4:e54-p2-a1 iso_m7_ipg5:e54-p2-b1 iso_m7_ipg5:e54-p2-b2 \
  iso_m7_ipg4:e54-p2-a2 \
  iso_m8_ipg4:e54-p3-a1 iso_m8_ipg5:e54-p3-b1 iso_m8_ipg5:e54-p3-b2 \
  iso_m8_ipg4:e54-p3-a2 \
  --widths 1,2,3,4,5,6,7,8,9,10 --reps 21 --inner 10

# zero-GPU analysis
python3 research/e54_analyze.py
python3 research/e54_bandwidth.py
python3 research/e54_price.py
python3 research/e54_reg_census.py
```

Base `a35bb006fd47785dc916241df63ec8780bda8e5c`, host Apple M4 Pro, 48 GiB.
Design: ABBA-counterbalanced, widths 1–10, 21 replicates × 10 inner, one session
per pair, isolated build per leg.

Every arm's timed bytes re-derive exactly from the committed arm definitions:

```
iso_m5_ipg3 b99ff7bd  iso_m5_ipg5 cc259829  iso_m7_ipg4 485f5fad  iso_m7_ipg5 13b17cc6
iso_m8_ipg4 9d73f23d  iso_m8_ipg5 b7ac4b0a  shipped     75d45143  e27_full    e50ca0eb
```

Both replicate legs of every arm carry the same digest, so the two replicates of an
arm timed identical bytes.

## Gates

All 16 legs **passed the real 40 °C cool gate**; none used the permitted ungated
mode. Entry temperature 39.87–40.01 °C, spread 0.14 °C.

```
editable budget OK: source=2458949/3000000 headroom=541051 growth=0/262144 files=154
assignment scope OK: 2 submitted path(s) against BASE_SHA=a35bb00
PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only
TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver
```

The scored surface is byte-identical to base. **This experiment ships no candidate**
— every arm is a probe applied and unwound per leg.

## Predictions against measurement

| source | P1 | P2 | P3 | verdict |
| --- | --- | --- | --- | --- |
| Law A′ | −19.00 | 0.00 | 0.00 | **survives**, off by 1.3 / 1.0 / 1.3 |
| bandwidth θ | −18.38 | +31.62 | +31.62 | right at P1, 30 points wrong at P2/P3 |
| Law C, advisor | regress | −12 ± 3 | −12 ± 3 | **falsified in both directions** |
| Law B | regress | regress | regress | falsified |
| **my blind prediction** | **+25.8** | — | — | **wrong, by the largest margin here** |

I pre-registered a prediction that contradicted my own hypothesis, derived from the
E27 board reconciliation and committed at `80a4218` before any timing existed. It was
the worst prediction in the table. The reconciliation was internally consistent and
converged on 95.5 GB/s from three routes, which is exactly why it is worth recording
that it was still wrong: its shared premise, that E27's board loss is explained by
cell timing, is the premise P4 has now falsified.

## Honest limits

1. The width sweep is a QMV microbenchmark. It does not reproduce board score, and
   P4 quantifies by how much it fails to.
2. Law D is unresolved end-to-end, not refuted. See above.
3. `bandwidth_reliable: false` rows, where modelled traffic exceeds stream peak, are
   excluded from every achieved-rate figure. Those are `qmv_fast_impl` fallback rows.
4. The three cells above 100 % of measured stream peak, M=5 at 240.5 and M=9 at
   233.3, indicate cache reuse across working groups, so absolute rates are effective
   rather than DRAM rates. Comparisons within a pair are unaffected.
5. Host is Apple M4 Pro, not the ranked M5. The ranked-transfer risk is unchanged
   from E49.
6. P2's bar rose to 0.884 % on a single control width, M=1 at −0.884 %, and P2's
   effect of +0.994 % clears it by only 0.11 points. P2 alone is marginal; P3 at
   +1.345 % against a 0.482 % worst control is not, and the two agree in sign, size
   and bandwidth mechanism.

## Suggested follow-ups, not implemented

1. **Measure `e27_full` end-to-end with `--local-iterate`, ABBA.** This is the single
   most valuable next experiment, because it splits the two live explanations for
   the E27 gap: an unrepresentative QMV microbenchmark, or an over-attributing
   leverage model. Serial decode runs at M=1 and never reaches a crossrow case, so
   the edit cannot move the serial leg and the local ratio will not cancel it.
   `research/e33-e2e-run.sh` already runs one such arm with a metallib rebuild and a
   stale-kernel fail-closed check; it needs only arm apply and unwind around it. I did
   not run this because the runtime worker binary and the transformed checkpoint are
   not provisioned in this workspace, so it is a setup task rather than a bolt-on.
2. **Re-derive the leverage model against the P4 anchor.** P4 gives the first matched
   local measurement of a composite with a known board score. That is enough to fit
   or bound ψ_mtp directly instead of assuming it.
3. **Recheck `<T,9,5>` promotion under the register attribution.** `<T,9,5>` alone
   carries the entire +21 register rise and the least score weight of the four cells.
4. **M=6 has no IPG=5 form but does have a group-count question.** M=6 runs `{3,3}`;
   `<T,6,4>` would be `{4,2}`, still two groups, so A′ predicts a null. It is the
   cheapest remaining test of A′ at constant group count, in the 64 % share bucket.
5. **A′ predicts M=5 is not the end.** Any width whose group count can fall by
   raising IPG should win by roughly the traffic ratio. That is a rule for selecting
   cells, which was the stated goal of a good result here.

## W&B runs

Group `e54-lone-vs-sibling-na5`, project
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`. One run per leg, logged while
timing.

| pair | control legs | treated legs |
| --- | --- | --- |
| P1 | `9qt2x4cp`, `vlp3eynm` | `rv1x3qt3`, `zaax7vmy` |
| P2 | `2etarcaw`, `af47v2hf` | `xnhjo9yo`, `05soskch` |
| P3 | `skzyxrmt`, `jiihp8qr` | `vlp828wp`, `x78jqu9n` |
| P4 | `eiswqax5`, `r6limb6r` | `oo4snw5j`, `mnxfzvmi` |

One additional run, `nmvkuwwk`, belongs to an **aborted** session: a first P1 attempt
whose second leg refused to start because I made the worktree dirty by committing
analysis files while the job was running. The leg runner correctly declines to time
an unknown tree. That session's manifests are archived outside the analysed directory
and none of its data is used here.
