# E78 rung 3 pre-registration

Written after rung 2b and before the rung 3 session starts. Committed so the
prediction cannot be adjusted after the legs run.

`harness=local`. Nothing here is an official or ranked score.

## Why rung 3 exists

Rung 2a measured every `(shape, M, IPG)` cell and found the shipped group count
faster in 23 of 24 contested cells. The single exception is `mlp.down`
(`in_vec_size` 17408, `out_vec_size` 5120) at M = 6, which is 7.97 % faster at
two groups.

That one cell cannot be reached by any cutoff on `out_vec_size`, because
`linear_attn.out_proj` has the identical `out_vec_size` of 5120 at the same
M = 6 and wants the opposite group count. The separating variable is
`in_vec_size`. Arms A to D never read it.

`e_kdown` reads it. At M = 6 only, it sends `in_vec_size >= 8192` to IPG 3 and
everything else to the base IPG 6. M = 5 and M = 9 are untouched. In the scored
shape set only `mlp.down` has `in_vec_size >= 8192`; the next largest is 6144.
The cutoff is therefore robust anywhere in `(6144, 17408]`, and 8192 is not
tuned inside a gap between two scored shapes.

`research/e78_price_arms.py` scores `e_kdown` equal to the `oracle(M, k)` row.
It is not merely a candidate: it is the best table any legal function of M and
the two vector sizes can produce from the measured cells.

## The derived prediction

Round counts come from rung 1: all arms in this family emit byte identical row
ledgers, so every arm runs the same 78 verify rounds and emits 512 tokens. The
conversion is `delta_s_per_token = 78 * delta_ms_per_round / 1000 / 512`.

| arm | derived delta ms per round, local mix | predicted delta s/token | predicted % of a_ship |
|---|---:|---:|---:|
| `a_ship` | +0.0000 | 0 | 0 |
| `e_kdown` | -0.7189 | -1.0951e-4 | -0.348 % |

Rung 2b already validated this conversion. Its three predictions landed at
1.20x, 1.06x and 1.00x of the measured whole-leg deltas, so additivity holds on
this host and this instrument may price an arm.

## Pre-registered pass conditions

The stop rule is the one already coded in `research/e78_analyze.py` and is not
retuned: `e_kdown` must beat `a_ship` by more than twice the largest within-arm
cross-position session null, on absolute `mtp_seconds_per_token`, in the
predicted direction.

1. **Sign.** `e_kdown` is faster than `a_ship`.
2. **Resolvability.** The measured delta exceeds twice the largest session null.
   In rung 2b that threshold was 7.91e-5 s/token, against a predicted effect of
   1.0951e-4. The margin is only 1.38x, so rung 3 runs 3 legs per group instead
   of 2 to shrink the null.
3. **Magnitude.** The measured delta lands within a factor of two of
   -1.0951e-4 s/token.
4. **Exactness.** `all_tokens_matched` on every leg, and the rows-per-round
   histogram is identical to `a_ship`.

## Why exactness is already covered

`e_kdown` selects a strict subset of the `(shape, M, IPG)` cells that rung 1
already proved exact. At M = 6 it sends `mlp.down` to IPG 3, which is exactly
what `d_hybrid8192` did; every other cell is the untouched `a_ship` cell. Rung 1
showed `a_ship`, `c_hybrid24928` and `d_hybrid8192` produce byte identical
220574-byte row ledgers with sha256
`a1044f14dcd6d24d7259357b54eda48170321b8fd7943746a84f6468a04acc98`, 567 declared
rows, 512 tokens and post-EOS continuation. Rung 3 re-checks exactness on every
leg regardless.

## What a failure means

If `e_kdown` is not faster, then the isolated `mlp.down` cell gain does not
survive inside a live block even though the three rung 2b arm predictions did.
That would localise the additivity failure to this one cell rather than to the
instrument, and the most likely cause is that `mlp.down` at M = 6 is not on the
critical path in situ.

## What rung 3 cannot decide

Rung 3 runs on the local M4 Pro with 20 GPU cores. Thorfinn's E75 rung B showed
that a QMV group-count table can invert its sign between this host class and the
ranked 40-core host: the crown table is uniformly far worse here yet was faster
on 8 of 8 ranked prompts. Rung 3 therefore cannot predict the ranked sign of
`e_kdown`. It decides only whether the rung 2a cell instrument correctly
identified the one locally winning table.

## Thermal protocol

Ungated, for the reason recorded in the rung 2b pre-registration: the real 40 C
gate provably aborts on this host, whose measured idle GPU floor is 39.49 C. The
session runs `a e e a` at 3 legs per group after one discarded warm group, which
gives both arms the same mean leg position, and it records entry and exit GPU
temperature for every group. `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` are preserved verbatim in every artifact.
Rung 2b held its 8 timed groups inside a 0.61 C entry-temperature spread.
