# SENPAI Research State

- 2026-08-22 13:05Z

## Most recent research direction from the human researcher team

None received this cycle. The campaign runs autonomously under
`senpai/program.md`.

## Where the campaign actually stands

**We own the second-fastest candidate on the board. We lost the crown to the
pinned serial numerator, not to a rival mechanism.**

Finding 153 (ledger 282) recomputed every board row with complete per-prompt
official metrics (n = 806) on a **common serial denominator**. The seven-row
frontier cluster — every row within `0.30 %` mean absolute candidate-leg
difference, therefore one tree family in one runner state by Rule 98 — ranks
like this on candidate merit:

```
rank  common-denominator   published    id         solver          Yukon
  1        3.525618        3.517689   3b376ba2   Lieisyourlie    rejected
  2        3.521408        3.512706   0c6191b7   morganmcg1      rejected   <- OURS
  3        3.520417        3.516617   cf79f7df   Lieisyourlie    accepted
  4        3.518779        3.512449   dd3c1ff7   Lieisyourlie    rejected
  5        3.518453        3.518453   48423d09   noskillcoding   ACCEPTED = crown
  6        3.518057        3.515941   390ec878   newjordan       rejected
  7        3.517655        3.507747   c63eaa21   newjordan       rejected
```

The crown is rank 5 of 7 on merit. It drew the slowest serial legs.

## The two noise terms that now govern every decision

```
serial numerator lottery, within one state     sd 0.0967 %   Finding 153
candidate-leg run noise, within one state      sd 0.0735 %   Finding 153
runner state term, three levels                up to 2.3 %   Finding 152
```

**Finding 152** identifies the state: one step is `930.9 us` per drafting round,
which is one extra full DRAM traversal of the 427,742,600-byte pinned proposal
head at `459.5 GB/s`, matching Finding 143's ranked `462.2 GB/s` to `0.59 %`.
The cause is in our own source. `Qwen36MTPBlockSession.swift:212-213` sizes the
wired residency ticket at the live post-warm footprint plus a `64 MiB` slack,
and then `143.75` to `215.75 MiB` of KV, GDN recurrent state, GDN conv state and
head history KV allocate on top of it. The head is the last large object loaded,
so it sits at the tail of allocation-ordered wiring and falls out first.

**Finding 153** shows what is underneath the state term. Removing the state term
takes the ranked runner from a `2.3 %`-resolution instrument to a `0.20 %`-
resolution one. That is the largest measurement improvement available and it is
one integer: `wiredZHDefaultSlackMB` `64` to `512`.

Neither finding licenses re-rolling. Resubmitting a tree to chase a serial draw
is a duplicate submission and stays forbidden under Finding 79 and Rule 72. The
answer to a `0.084 %` merit lead inside a `0.19 %` band is to make the lead
`1 %` or `10 %`.

## Which prompts the score actually pays for

At n = 806, the prompts occupying median order statistics 4 and 5:

```
  beagle      789 / 806   97.9 %      mean M 5.38, per-step p 0.9341
  medicine    348 / 806   43.2 %      mean M 6.26
  essays      199 / 806   24.7 %      mean M 6.09
  republic    168 / 806   20.8 %      mean M 5.99
  botany       92 / 806   11.4 %      mean M 7.15
  travel        8 / 806    1.0 %
  plutarch      6 / 806    0.7 %
  drama         2 / 806    0.2 %
```

`published = 0.5 * raw_beagle + 0.5 * min(medicine, essays, republic, botany)`.
beagle alone carries half the score and has the lowest acceptance of the five
drafting prompts. The second half is a `min`, so it is pessimistic: helping
three of four and hurting the fourth may move nothing.

**The ranked mass that pays sits at `M = 5` to `7`.** The local benchfixture
runs at mean width `7.36` with about `77 %` of its mass at `M = 8` and is the
wrong frame for pricing width work.

## Current research focus

1. **Collapse the multi-pass QMV table at the widths that carry the ranked
   median.** Finding 151 shows the adverse reading of the one-pass arm came
   entirely from dropping `rows_per_simd` 4 to 2 to protect registers. At
   `RPS = 4` the arm is `-2.0506` statements per output element at
   `{6:6, 7:7}`, which is `-12.55 %` of QMV issue and `-10.97 %` of the leg.
   Per-width templating is a hard prerequisite and is itself free upside
   bounded below by zero (`+9.18 %` g17s residency, zero statement change).
   Askeladd's D_S code-motion patch clears NA=7 to `118` registers and `0`
   spill. Owner thorfinn, PR #128.

2. **Remove the runner state term.** Measure post-sizing allocation growth, then
   raise the residency slack. Owner alphonse, PR #130.

3. **Decide C1, the sketch-first draft readout.** It removes `53.06 MB` of the
   `323.59 MB` draft step and about `99.3 %` of the readout stage's instruction
   issue, and is priced at `+0.90 %` to `+1.47 %` ranked, central `+1.15 %`. It
   wins under both the byte model and the instruction model. The offline
   falsifier is **not** free: the 18,092-sample hidden-state corpus is not on
   the advisor host and recapture costs one to two hours of exclusive
   resident-model GPU time. Owner askeladd, PR #133.

4. **Get the ranked per-prompt width histograms.** They are the only thing that
   settles the frame conflict between askeladd's `-2.6 %` local price and the
   `-12.55 %` ranked price for `{6:6,7:7}`, and they gate the order in which
   thorfinn ships his two receipts. Owner edward, PR #129.

## Potential next research directions

- **C2, quantize the bf16 precision islands to affine-4 g64.** `22.61 MB` per
  draft step, `+0.38 %` to `+0.45 %`. Reopened: the E82 blocker was a power
  failure, not a result. Unowned.
- **Route C, reach the `qmv_fast_impl` body at a lower allocated register
  count.** Prefer askeladd's C2 form, the shipped `qwen_e120_qmv_wide<1,false>`
  at 56 registers and 70 derived simdgroups, which needs no new kernel text.
  Rider only; Finding 149 prices the residency-only channel at about zero.
- **`{8:8}` as a rider behind `{6:6,7:7}`.** Real, and askeladd proved it beats
  two passes of `wide<4>` by `9.25 %` to `16.62 %` even at the harshest spill
  charge, but it earns most of that on prompts with near-zero ranked weight.
- **The `min` structure of the second median carrier.** Because half the score
  is `min(medicine, essays, republic, botany)`, an arm that lifts the current
  worst of those four is worth more than an arm that lifts the mean. Nobody has
  looked at whether the identity of that argmin is stable across runs.
- **The beagle depth schedule specifically.** beagle carries `97.9 %` of the
  median at the lowest acceptance of the five. E128 found the shipped flat
  `0.18` threshold optimal on the eight-prompt aggregate curve; it has not been
  tested against beagle alone with the `costModelDepth` fixed-round-cost term
  that Finding 152 implies is missing.
- **A stale-suffix recycling trace (O3).** Legal; the reopener is a zero-GPU
  trace showing stale-suffix acceptance above `0.5` at position 1.

## Closed this cycle

The qat-q4 head declaration (advisor error 114: priced from a non-significant
acceptance delta; it is `2.7 %` to `3.4 %` slower per token and has no external
identity). C4, the probe-fraction reduction, retired because C1 inverts its
gradient. C5, centroid-table padding, dead because the imported kernel handles
the tail through `n_valid`. `rows_per_simd = 2` as spill relief, because
candidate B is worse than doing nothing. Deleting `case 9`, because
`Qwen35.swift:1699` routes `3...9` with `default: break`.
