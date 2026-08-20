# E69 rung 1: replication session scored

Session 2, job `61fffdb0`, fresh 40 C gate per NA, 21 reps, NA = 4, 5, 6, all
9 arms, `--tag -s2` so it could not overwrite session 1. Entry 38.7-39.7 C.

## Blind predictions: 3 of 3 correct

Scored against `rung1-na5-mechanism.md`, committed while `rung1-na5-s2.json`
and `rung1-na6-s2.json` did not exist. Prediction 3 was explicitly disclaimed
as not blind and is not counted.

| # | prediction | outcome | result |
|---|---|---|---|
| 1 | `xvec` NA=5 within [-5.0, -2.0] % on >= 5 of 7 shapes | 7 of 7 in range | **CORRECT** |
| 2 | `xvec` NA=6 inside +/-1.0 % | 7 of 7, -0.12 to -0.23 % | **CORRECT** |
| 4 | alloca counts identical to session 1 | identical at NA=4, 5, 6 | **CORRECT** |
| 3 | (not blind, disclaimed) | median -0.25 % | not counted |

## The NA=5 spike is real

Per shape, `xvec` at NA=5, the two independent gated sessions:

| shape | session 1 | session 2 | difference |
|---|---|---|---|
| `linear_attn.in_proj_fused_qkvzba` | -3.56 | -3.49 | +0.07 |
| `linear_attn.out_proj` | -3.93 | -4.22 | -0.29 |
| `full_attn.qkv_proj_fused` | -4.13 | -3.92 | +0.21 |
| `mlp.gate_up_fused` | -3.22 | -3.21 | +0.01 |
| `mlp.down` | -4.36 | -4.32 | +0.04 |
| `head.compact_draft_vocab` | -2.70 | -2.74 | -0.04 |
| `head.lm_head` | -2.58 | -2.55 | +0.03 |

14 of 14 shape-sessions negative, agreement within 0.29 percentage points,
against a same-arm null of +/-0.05 %. At NA=6 the same arm gives -0.20 % in
session 1 and -0.22 % in session 2.

The two-condition mechanism holds: `xvec` pays only where the cell is both off
the bandwidth roof and free of the `<6 x float>` spill, and NA=5 is the only
scored width that satisfies both.

## Decision

The assignment's bar is more than 3 % at NA=5 **and** at NA=6.

| arm | NA=5 | NA=6 | clears bar |
|---|---|---|---|
| `xvec` | -3.5 % pass | -0.2 % fail | no |
| `fma` | -0.9 % fail | -1.1 % fail | no |
| `wvec` (arm A) | +0.0 % | +0.1 % | no |
| `tgx` (arm B) | +11.9 % | +16.2 % | no |
| `rows8` (arm C) | +35.8 % | +78.5 % | no |

No arm clears the bar. Terminal negative.

## The `rows8` parity defect, narrowed

`rows8` is not bit-identical at NA=2 and is bit-identical at NA=3, 4, 5 and 6.
The AIR census at NA=2 and NA=3 shows why NA=2 is the special case:

| NA | arm | registers | allocas | alloca types |
|---|---|---|---|---|
| 2 | `plain` | 82 | 1 | `[4 x [4 x i16]]` |
| 2 | `rows8` | 130 | 1 | `[8 x [4 x i16]]` |
| 2 | `rows8wxvec` | 142 | 1 | `[8 x [4 x i16]]` |
| 3 | `rows8` | 184 | 2 | `[8 x <3 x float>]`, `[8 x [4 x i16]]` |
| 3 | `rows8wxvec` | 196 | 2 | `[8 x <3 x float>]`, `[8 x [4 x i16]]` |

**NA=2 is the only width at which the eight-row float accumulators `acc[8]` and
`partial[8]` are fully register-promoted.** From NA=3 upward the compiler gives
up and puts them in memory. The parity failure occurs at exactly, and only, the
width where full promotion happens, which is where the most aggressive
scalarization of `vec<float, 2>` takes place.

That is as far as static evidence goes. `rows8wxvec` is promoted at NA=2 too
and is correct, so promotion alone is not sufficient to trigger the fault; the
vector loads evidently change the scalarization enough to avoid it.

Conclusion: a compiler codegen fault in the fully scalarized
`vec<float,2>` x 8-row path, not an error in the assignment's bit-identity
argument. That argument is sound: nothing in the function reduces across rows,
so row `r` cannot depend on `rows_per_simd`, and the parity harness correctly
reports that the emitted values disagree anyway.

This is a research-only probe arm. It is not on any candidate path, and arm C
is independently dead on time and on submission scope.
