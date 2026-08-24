# E174 — Extend the fused-norm xsums epilogue to the 130 unserved producers

`harness=local` everywhere in this document. **No ranked transfer is claimed.**

- PR: #173, branch `qwen-alphonse/e174-xsums-epilogue-extension`, revision `r0`
- Base: `2b8037c3e5207fa2f5633e61d013a779492b05c2`
- Verdict: **not useful.** The mechanism's ceiling is already measured and it
  sits below the assignment's own minimum useful effect.
- Timed legs spent: **0.** The stop rule fired before any timed session,
  because the largest effect consistent with existing matched evidence cannot
  cross the decision threshold.

## 1. What the assignment asked, and what stopped it

The assignment asked me to publish xsums from the producers of the 130 routed
table-paying cells that have no publishing producer, and priced that at
0.650 ms/round of GPU standalone-fill work plus about 0.43 ms/round of host
admission work. Both numbers come from my own E173 inventory.

The GPU number is method-tagged in my own script as
`source-derived (producer/consumer census) x in-source measured fill cost`:
130 cells multiplied by the "measured at 4 to 6 us" comment at
`Qwen35.swift:1732`. RULE 160 requires a 0.21 factor on that class of price
until a matched end-to-end measurement replaces it. Three such measurements
already exist in the campaign ledger on this base. I did not cross-reference
them when I raised this item in the E173 verdict. That is my error.

## 2. The three matched end-to-end measurements

Every row is a thermally gated ABBA session with a real 40 C gate, quoted with
its own published interval. `harness=local`.

| line | lever | measured effect | se |
| --- | --- | ---: | ---: |
| E135 / F41, ledger 313.7 ("THE FILL COSTS 1.711 us, NOT 4.9. ADVISOR ERROR 183") | add a live fill for **all 257** routed cells (`fill_noconsume - replica`) | **+0.2213 %** of candidate MTP time | 0.0942 pp (drift fitted), 0.0681 pp (no drift) |
| E160 replicate 1, FINDING 350, ledger 65434 | **remove 64** fills with a producer epilogue (`fuse - off`) | **+0.0214 %** | 2σ `[-0.1752, +0.2181]` |
| E160 replicate 2, FINDING 359, ledger 65873 | same mechanism, 12 legs | **-0.0015 %** | 2σ `[-0.2059, +0.2029]` |

E135 is the decisive one, and it prices exactly the composite this assignment
proposes to delete. Its `fill_noconsume` arm calls `xsumsTable(x)` for every
routed cell and binds a table the kernel never reads, so the contrast carries
**both** the GPU fill dispatch **and** its host kernel record, with the consumer
held fixed (`consume:` is false on both arms). Its published decomposition is
`+439.7 us/round over 257 fills`, `1.711 us per dispatch`, against a local round
of `198,671 us`. Its four 512-token legs ran the palindrome
`repl fill fill repl` behind the real 40 C gate with `edl = 6.358974` on all
four, so the arms share one schedule and the contrast is a pure cost
measurement. Thorfinn stated three caveats, all of which help this argument:
the arms remove all 257 fills at once so the total is what is measured and the
division by 257 is untested; the two arms run different pipeline objects, so
1.711 us is an **upper bound** on the fill alone; and the cross-session
comparison against the shipped `sumtable` arm is directional.

### The ceiling, two ways

The assignment-free form needs no scaling assumption at all:

> The **entire** 257-cell fill surface costs **+0.2213 %** of candidate MTP
> time. E174 targets a subset of that surface. So E174's payoff is at most
> +0.2213 %, against a predeclared minimum useful effect of **0.30 %**.

With per-cell scaling, on E135's own caveat that the division is untested:

| quantity, 130 cells | value |
| --- | ---: |
| E173 source-derived prediction (0.650 + 0.43 ms/round of a 188.8 ms round) | +0.572 % |
| E135 measured, central | **+0.112 %** |
| E135 measured, 2σ upper bound | **+0.207 %** |
| E160 replicate 1, central, scaled from 64 cells | +0.043 % |
| E160 replicate 2, central, scaled from 64 cells | -0.003 % |
| predeclared minimum useful effect | 0.30 % |

The E173 row overpredicts the measured central value by about **5.1x**, which
is close to the 4.7x overprediction RULE 160 was written to correct.

**And every figure above assumes the replacement epilogue is free.** E160
measured that it is not. Its accepted mechanism: the epilogue re-reads the
chunk from the producer's output after a device barrier with a small fraction of
the threads active, so it converts a well-shaped standalone dispatch into a
badly occupied tail on a larger one. Dispatch count falls, work does not.

## 3. The reopen condition is not met

FINDING 359 closed the producer-fusion axis for `mlp.down` (64 sites),
`gdn.out_proj` (48) and `fa.o_proj` (16) — **128 of the 130 cells** this
assignment targets — and set one reopening condition: a producer whose grid
already matches the fill grid `(32, kBlocks, m)` at full occupancy, so the
epilogue is not a tail. Checked against source on this base:

| site | grid | threadgroup | matches the fill grid? |
| --- | --- | --- | --- |
| `xsumsTable`, the fill itself, `:1803` | `(32, kBlocks, m)` | `(32, 1, 1)` | reference; every thread writes one sum |
| `qwen35FusedResidualRMSNormXSumsKernel`, `:2355` | `(nRows * 1024, 1, 1)` | `(1024, 1, 1)` | no — 320 of 1024 threads per row at K = 5120 |
| `:2156` | `(totalRows * 64, 1, 1)` | row-major | no |
| `:2584`, `:2845`, `:2865` | `(2 * nRows * 1024, 1, 1)` | row-major | no |
| `:654`, `:976`, `:1234` | `(32, ...)` with y and z fixed by head geometry | — | no — y and z are not `kBlocks` and `m` |

No producer of the 130 cells satisfies the condition. The two cells outside the
closed families are `lm_head` (1) and `gdn.in_proj` at layer 0 (1). At E135's
measured 1.711 us per dispatch they are worth at most 0.002 % of a round
together, so they cannot carry the experiment either.

## 4. Why I did not run a timed screen

I planned one, built the arm for it, and then priced the measurement:

- The largest effect consistent with existing evidence is **+0.112 %** central,
  **+0.207 %** at 2σ. The campaign's local per-leg noise floor is **0.120 %**.
- To resolve +0.112 % at 2σ needs `se ~ 0.056 pp`. E135 reached
  `se = 0.068 to 0.094 pp` with 4 gated legs at a **257**-cell lever. A
  130-cell lever needs roughly **11 or more** gated legs to reach 2σ.
- The result would not change the decision. Both the central value and the 2σ
  upper bound for 130 cells already sit below the 0.30 % threshold, so no
  outcome of that session promotes the experiment.

`program.md` rejects an experiment when "the largest plausible gain is below
measurement noise". That is this case, so the timed session was not run. If the
advisor wants the first-hand contrast anyway, it now costs one environment
variable and no new code: see section 5.

## 5. What this branch leaves behind

- `MLX_E174_XSUMS_SIDECAR=off` gates `Qwen35XSumsSidecar.wants`, so the
  producer publishes nothing and all 257 table-paying cells take the standalone
  fill. The shipped default is unchanged. Arithmetic is identical on both arms:
  each reads a table written by the same fill body, and the `off` arm is the
  shipped pre-sidecar dispatch. Witnessed inside the built worker by
  `senpai/rebuild-and-assert-worker.sh --require MLX_E174_XSUMS_SIDECAR`.
- `research/e174_census.sh` and `research/e174_census.py` read the live
  `xs_hit`/`xs_fill` pair that ledger 332.3 asked for and nobody had read. Each
  arm is the other arm's positive control, so a switch that never reached the
  worker cannot pass.
- `research/e174_log_wandb.py` publishes the census and this reconciliation.

**The switch is instrumentation on the scored surface, and nothing here ships,
so I reverted it from the submitted surface before submitting this result.** It
stays recoverable from this branch at commit `b5fd35b1`. That keeps the
campaign's simplification direction intact and matches the instrumentation strip
the advisor has in flight.

## 6. Suggested follow-ups, which I did not implement

1. **Correct the E173 inventory row rather than delete it.** Replace
   `gpu.xsums_standalone_fills` 0.650 ms/round with E135's measured
   `439.7 us/round over 257 cells`, which is 0.222 ms/round for the 130
   unserved cells, and tag it `measured-e2e (E135/F41)`. The same correction
   applies to the host share the assignment took from
   `admission.kernel_record_construction`: E135's contrast already contains
   that host record, so the two terms must not be added.
2. **Apply the same audit to the other three E173 items before assigning
   them.** Item (2), kernel-record caching, rests on the same
   `admission.kernel_record_construction` residual. Item (3), BF16 recurrent
   state, is priced from `gdn.recurrence` 2.0872 ms/round, which is a
   measured-GPU intercept and therefore does not carry the RULE 160 problem.
   Item (3) is the one item whose price survives this audit.
3. **Close the whole chunk-sum axis, not just this experiment.** The two
   remaining levers in this area are both below the campaign's minimum useful
   effect, from measurements already in the tree:
   - Removing fills: at most +0.207 % at 2σ for the 130 unserved cells, central
     +0.112 %, and only if the replacement epilogue were free (E135, E160).
   - Lowering `minimumTableWidth` to 3 per shape: the E120 rung 5d net table at
     `Qwen35.swift:1737-1745` gives -90.1 us for taking every M = 3 cell and at
     most +62.7 us for taking only the four shapes with a positive sign. That
     ceiling is **0.031 %** of a 198,671 us round, and the in-source note
     already declines it because it would hard-code one host's timings.

   The fill is cheap relative to what the table buys (1.711 us against +11 to
   +199 us saved per matvec at M >= 4), so the shipped design is close to
   optimal here. I suggest marking the axis closed in the ledger so it is not
   re-audited a fourth time.
4. **A campaign check worth automating.** Every assignment that cites an E173
   row should carry that row's `method` tag in the assignment body. The two
   method classes in that inventory — `measured-*` and `source-derived` — differ
   by about 5x in realized value, and the tag is already in the artifact.
