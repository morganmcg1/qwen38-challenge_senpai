# E69 rung 1: scoring the preregistered NA=5 reading

Scored against `rung1-prereg-na5.md`, which was committed in `3b2f716` before
`rung1-na5.json` existed. NA=5 is gated, 21 reps, entry 39.9 C, exit 61.1 C,
same-arm null within +/-0.05 %.

## Score: 2 of 5 predictions correct

| # | prediction | outcome | result |
|---|---|---|---|
| 1 | `plain` GB/s below NA=4, above 117.6 | 141.6-158.5, below NA=4 on 7 of 7 | CORRECT |
| 2 | `fma` is faster at NA=5 | negative on 7 of 7 | CORRECT |
| 3 | `fma` effect between -1.0 % and -3.0 % | median -0.96 %, only 3 of 7 in range | **WRONG** |
| 4 | `wvec`, `xvec`, `wxvec` stay inside +/-1 % | `xvec` -2.58 to -4.36 % on 7 of 7 | **WRONG** |
| 5 | `tgx` and `rows8` stay above +5 % | +7.0 to +34.9 and +23.7 to +39.3 | CORRECT |

Prediction 4 is the important failure, and it fails in the direction that
rescues the assignment's hypothesis rather than burying it.

## What I got wrong, and why it matters

I wrote in the rung-0 report that the x-operand-traffic hypothesis "looks
falsified". That claim was based on NA=2, 3 and 4 only. It is wrong.

`xvec`, which replaces four 2-byte scalar `x` loads with one 8-byte vector
load, is the most direct possible test of the assigned hypothesis. Its effect
across NA is not monotone and not null:

| NA | `xvec` median % vs plain | `plain` GB/s | reading |
|---|---|---|---|
| 2 | +0.03 | 211-225 | at the roof, no effect |
| 3 | -0.54 | 197-221 | at the roof, small effect |
| 4 | -0.13 | 171-190 | leaving the roof, no effect |
| 5 | **-3.56** | 142-159 | **clear effect on 7 of 7 shapes** |
| 6 | pending | pending | decisive for the weighted result |

At NA=5 the gain is consistent in sign and size across all seven scored
shapes, and it is roughly 70x the same-arm null. This is a real effect, not
noise, and it is exactly the mechanism the assignment proposed.

I was also wrong about the mechanism I preregistered. I claimed distance from
the bandwidth roof predicts the payoff of removing work. That predicts a
smooth increase from NA=4 to NA=5 to NA=6 for every work-removing arm. Instead
`xvec` is flat at NA=4 and then jumps at NA=5, while `fma` peaks at NA=4
(-1.55 %) and shrinks at NA=5 (-0.96 %). The two arms peak at different widths,
so a single scalar "distance from the roof" cannot explain both.

## Revised reading

The arms separate by what they remove, and each pays at a different width:

- `wvec` removes weight-load instructions. Null at every NA measured. The
  assignment's arm A is genuinely falsified.
- `xvec` removes x-load instructions. Pays 3.6 % at NA=5.
- `fma` removes arithmetic issue slots. Pays 1.6 % at NA=4.
- `tgx` and `rows8` add threadgroup traffic and register pressure. Both lose
  at every NA. The assignment's arms B and C are genuinely falsified.

So the assignment was right that the `x` operand matters and wrong about the
remedy: staging `x` in threadgroup memory, which is what arm B proposed, costs
7-35 %, while simply widening the `x` load wins.

## What is still open

`xvec` at NA=5 is +/-1 shape-weighted percent away from changing the decision,
and NA=6 carries the largest time share of any width, near 35 %. The 3-rep
ungated NA=6 smoke put `xvec` at -0.27 %, which if confirmed would cut the
weighted gain roughly in half. The gated NA=6 cell is running now and settles
it. I will not choose a rung-2 candidate before it lands.
