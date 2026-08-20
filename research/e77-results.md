# E77 results: the occupancy coefficient is real, reproducible and far too small

**Verdict: not useful.** Register occupancy is a genuine effect on this kernel and
this session measures it to better than 5 % relative precision. It is about
twelve times too small to explain the crown's ranked table, and about 220 times
too small to explain the crown's advantage at M = 5. Rung 3 did not start,
because the pre-registered ranked-ordering gate did not reproduce the correct
sign.

The experiment also produced two results that were not asked for and that matter
more than the occupancy number: the advisor's expected local null does **not**
hold, and the ranked extrapolation is falsified independently of the crown.

## Provenance

| Field | Value |
| --- | --- |
| Base | `41ddc183817979be8d2f0817d79f98b2ddefb984` |
| Head | see the submitted commit |
| Host | `ip-10-231-2-22`, Apple M4 Pro, 20 cores, `applegpu_g16s` |
| Harness | `local` |
| Cool gate | `cool_gate_passed_real_gate=true`, `gate_qualified_for_timing=true` |
| Session entry / exit GPU temp | 36.40 C / 66.04 C |
| Arms, shapes, reps | 52, 4, 15, palindrome leg order, 6240 timed legs |
| Probe source digest | `fae1c6b75f04878364a25edecaecc05caf376f5c61435d198046e6e469fc7bf3`, `census_matches_source=true` |
| Session null | median 0.0193 %, p90 0.0742 %, max 0.2396 %, n = 208 |
| W&B rung 1, timed session | `3ugpmkvo` |
| W&B rung 2, analysis | `jdlqnbwd` |

Full analysis output is committed at `research/e77-artifacts/rung2.txt`, and the
structured result at `research/e77-artifacts/rung2.json`.

Candidate files changed: **zero**. Every change is under `research/`, plus one
`.gitignore` line, and `.gitignore` is not in `editablePaths`.

Reproduce:

```bash
python3 research/e77_reg_census.py --out research/e77-artifacts/rung0-regs.json
research/e77_rung1.sh --reps 15 --target-bytes 12e9 --tag -s1
python3 research/e77_fit.py --cores 20 --ranked-cores 40 --wandb-name e77-rung2
```

## Rung 0: the model, and why only one arm family can identify it

The respecified cost model is

```
t = [ groups*W + beta*M*k*Tn ] * rho0 * c(IPG) * Omega(S_h(R_h(IPG)))
                                     * (1 + lam(IPG)/x_h)
S_h(R) = floor(B_h / (128*R))        c(IPG) = q(IPG) / Omega(S_L(R_L(IPG)))
```

The flip set is the open interval `Omega_R(j)/Omega_R(i) < a < Omega_L(j)/Omega_L(i)`.
It is non-empty exactly when the ranked relative occupancy penalty differs from
the local one, and it collapses to empty when `Omega = 1`, which recovers E73's
host invariance. So the model form *can* flip an `argmin` across hosts.

`Omega_L(IPG)` is a per-IPG constant. The rung-2 refit therefore absorbs it
exactly into `c(IPG)`, the local fit is algebraically identical to E73, and
control 1 cannot fail. I pre-registered that as PR-7 before running anything.
This also makes the ranked flip point invertible in closed form, which is how
the required-exponent table below is produced.

## The register law

The advisor's law holds exactly. Over 19 legal cells on both architectures the
register count is a function of the largest group alone, with no exception.

| largest group | local R | local S | local frame | ranked R | ranked S | ranked frame |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 70 | 43 | 0 | 83 | 47 | 0 |
| 3 | 93 | 33 | 0 | 90 | 44 | 0 |
| 4 | 94 | 32 | 0 | 91 | 43 | 0 |
| 5 | 95 | 32 | 0 | 98 | 40 | 0 |
| 6 | 96 | 32 | **16 B** | 111 | 35 | 0 |

Ranked S uses the extrapolated ranked register file `B_R = 124*128*32 = 496 KiB`.
Local S uses the measured `B_L = 384 KiB`.

Two consequences the advisor named, now confirmed with numbers:

1. Locally IPG 4, 5 and 6 all sit at S = 32. Occupancy does not move across
   them, so the natural family cannot measure occupancy on this host.
2. Our shipped local `<T,6,6>` cell spills 16 B. The same cell on the ranked
   host spills nothing. Our local table pays a penalty at IPG 6 that the ranked
   host does not charge.

## Rung 1a: the natural contrast is not a null

Fixed M, fixed group count, only IPG moves. Pooled geometric mean over the four
shapes, bootstrap 95 % CI per shape.

| M | IPG | local R | ranked R | pooled ratio | spill-free |
| --- | --- | --- | --- | --- | --- |
| 6 | 3 -> 4 | 93 -> 94 | 90 -> 91 | **0.99962** | yes |
| 7 | 4 -> 5 | 94 -> 95 | 91 -> 98 | **1.00822** | yes |
| 8 | 4 -> 5 | 94 -> 95 | 91 -> 98 | **1.01231** | yes |
| 9 | 5 -> 6 | 95 -> 96 | 98 -> 111 | 1.08240 | no, 16 B |
| 8 | 4 -> 6 | 94 -> 96 | 91 -> 111 | 1.09663 | no, 16 B |

Every per-shape 95 % CI excludes 1.000, so the advisor's expected local null is
**rejected** for IPG 4 -> 5 and for IPG 4 -> 6. The prediction was right in
mechanism and wrong in magnitude: with S fixed at 32 across IPG 4, 5 and 6,
none of this movement can be occupancy. It is all `q(IPG)`.

The M = 6 row is null only after pooling, and that hides the most interesting
detail in the session. Its four shapes move in **opposite directions**:
`head.lm_head` 0.99621 and `mlp.gate_up_fused` 0.99739 get faster, while
`linear_attn.out_proj` 1.00174 and `mlp.down` 1.00317 get slower. Each CI
excludes 1.000. The two shapes that speed up are the wide-N shapes; the two that
slow down are the narrow-N, large-K shapes. So IPG 3 -> 4 trades one effect
against another and the pooled mean cancels.

That row is also the only contrast in the session that crosses an S boundary,
33 to 32, which is a 3 % occupancy drop. It costs 0.04 % less on average, not
more. That single contrast already bounds the occupancy exponent near zero
before any ladder is fitted.

### Between-session replication

Every natural arm has a matching cell in the independent E73 114-cell session on
all four shapes. The `p0` probe adds only a dead branch to the shipped body, so
the two sessions must agree.

- absolute time agreement: median 0.085 %, max 0.246 %, n = 44
- natural ratios reproduce: median 0.042 %, max 0.222 %, n = 20

The contrast is a property of the kernel, not of one session.

## Rung 1b: the synthetic ladder

Fixed cell `<T,6,2>`, only inert live state moves. R walks 70 to 96 in unit
steps, S walks 43 down to 32.

At `head.lm_head` the whole range costs `t(96)/t(70) = 1.00504`. The other three
shapes give 1.00628, 1.00218 and 1.00518. So collapsing occupancy by 26 %, from
43 resident simdgroups per core to 32, costs about **half a percent**.

I pre-registered `t(96)/t(70)` in [1.02, 1.35] at `head.lm_head`. The measured
1.00504 is **below** that interval. My prior was wrong by a factor of four at
the low end.

### The load traffic is free; the liveness is what costs

The `q` controls load the same bytes and consume them immediately, so they carry
the traffic without the liveness. At `head.lm_head`:

| pressure | `q` ratio | `p` ratio | liveness cost |
| --- | --- | --- | --- |
| 8 | 0.99916 | 1.00106 | +0.191 % |
| 16 | 0.99965 | 1.00297 | +0.331 % |
| 24 | 0.99986 | 1.00542 | +0.553 % |

Across all 24 `q` arms the largest deviation from the base is 0.302 %, at the
32-float control, and every `q` arm is far below its matched `p` arm. The pad
loads themselves cost little, so the `p` ladder measures register pressure and
not added work. This control passes on all four shapes.

### The response is not a staircase

| statistic | value |
| --- | --- |
| within-tier spreads | 48, median 0.0869 %, max 0.1859 % |
| tier boundaries | 48, median step +0.0218 % |
| steps above 3x the session null | 25 of 48 |

The spread *inside* an S tier is four times the median step *across* a tier
boundary. Time rises smoothly and almost linearly in R, not in steps at
`floor(B/(128R))` boundaries. Pre-registered PR-3 named the specific tier
boundaries R = 72, 74, 75, 77, 79, 81, 84, 86, 88, 91, 94 and expected steps
there. That prediction is **rejected**.

This matters beyond the fitted number: the simdgroups-per-core staircase is the
wrong functional form for this kernel. `Omega(S)` is a smooth summary of a
smooth effect, not a mechanism.

### Spill curve, reported separately and excluded from the fit

Per the advisor's correction, no arm above 96 registers enters the fit. At
`head.lm_head`, `<T,6,3>` with frame bytes:

| frame bytes | ratio to `p0` |
| --- | --- |
| 16 | 1.00095 |
| 16 | 1.00180 |
| 48 | 1.00742 |
| 112 | 1.01672 |

Spill costs roughly 0.015 % per frame byte here. Per float of live state that is
0.052 %, against 0.019 % per float on the spill-free ladder, so spill is about
three times steeper.

## The occupancy exponent

`Omega(S) = (32/S)^gamma`, fitted on spill-free synthetic arms at or below 96
registers only.

| fit | gamma | s.e. | n | log-rms |
| --- | --- | --- | --- | --- |
| pooled | **+0.01346** | 0.00065 | 108 | 0.00125 |
| `head.lm_head` | +0.01477 | 0.00093 | 27 | 0.00087 |
| `linear_attn.out_proj` | +0.01874 | 0.00083 | 27 | 0.00078 |
| `mlp.gate_up_fused` | +0.01524 | 0.00082 | 27 | 0.00077 |
| `mlp.down` | +0.00511 | 0.00093 | 27 | 0.00088 |

The sign is positive and the estimate is 20 standard errors from zero, so the
effect is real. `mlp.down` is three times shallower than the others, so `gamma`
is shape-dependent and a single pooled exponent is already an approximation.

Resulting factors, measured S range 32 to 43:

| IPG | local R/S | `Omega_L` | ranked R/S | `Omega_R` | ranked S extrapolated |
| --- | --- | --- | --- | --- | --- |
| 2 | 70/43 | 0.99603 | 83/47 | 0.99484 | yes |
| 3 | 93/33 | 0.99959 | 90/44 | 0.99572 | yes |
| 4 | 94/32 | 1.00000 | 91/43 | 0.99603 | no |
| 5 | 95/32 | 1.00000 | 98/40 | 0.99700 | no |
| 6 | 96/32 | 1.00000 | 111/35 | 0.99879 | no |

The **entire** occupancy spread available anywhere in this table is 0.52 %.

## Rung 2 refit

With `Omega` fixed from rung 1 and the rest refitted on the 114-cell E73
surface: rel-rms 1.62 %, median 1.06 %, max 6.46 %, which is 68x the E73 session
null. `rho0` 1.5133 ps/byte, `beta` 2.4698.

`c(IPG)` = 2:1.0872 3:1.0000 4:0.9709 5:0.9764 6:1.0456, and
`c*Omega_L` reproduces E73's `q(IPG)` to four decimals, exactly as the algebra
requires.

### Validation 1: local table, control 1

| M | shipped | model | margin over 2nd |
| --- | --- | --- | --- |
| 3 | 3 | 3 | - |
| 4 | 4 | 4 | 39.87 % |
| 5 | 5 | 5 | 25.04 % |
| 6 | 6 | 6 | 6.01 % |
| 7 | 4 | 4 | 0.22 % |
| 8 | 4 | 4 | 0.22 % |
| 9 | 5 | 5 | 9.08 % |

**7/7, pass.** As pre-registered, this control cannot fail, so it is a
consistency check on the code and not evidence for the occupancy hypothesis.

### Validation 2: ranked ordering, at the extrapolated 40 cores

| M | ours | crown | model argmin | crown minus ours, QMV | width share |
| --- | --- | --- | --- | --- | --- |
| 3 | 3 | 3 | 3 | 0.000 % | 3.25 % |
| 4 | 4 | 4 | 4 | 0.000 % | 14.20 % |
| 5 | 5 | 3 | 5 | **+21.024 %** | 24.10 % |
| 6 | 6 | 3 | 4 | **+3.024 %** | 33.40 % |
| 7 | 4 | 4 | **5** | 0.000 % | 12.20 % |
| 8 | 4 | 4 | **5** | 0.000 % | 7.35 % |
| 9 | 5 | 3 | 5 | **+14.866 %** | 5.75 % |

**Fail at M = 5, M = 6 and M = 9.** The model says the crown's table is slower
than ours at every width where the tables differ. Pool-weighted this is
+6.93 % QMV, which converts to **+5.73 %** on the ranked candidate leg. The
measured value is **−0.298 %**. The sign is wrong and the magnitude is about 19
times too large.

### The same failure in exponent units

Because the flip point inverts in closed form, the binary result becomes a
distance.

| M | ours | crown | crown deficit at gamma = 0 | gamma required | sigma from measured |
| --- | --- | --- | --- | --- | --- |
| 5 | 5 | 3 | 1.21129 | +2.9701 | +4519 |
| 6 | 6 | 3 | 1.03299 | +0.1639 | +230 |
| 9 | 5 | 3 | 1.14966 | +2.1610 | +3282 |

Measured `gamma` = +0.01346 ± 0.00065. The smallest exponent any width needs is
12.2 times the measured value and 230 standard errors away. Occupancy is not
merely insufficient; it is excluded as the explanation at every width.

I pre-registered these three thresholds on the PR before the timing data
existed, so this is a genuine out-of-sample test and not a post-hoc fit.

### Validation 2b: the advisor's M = 6 inequality

> `occupancy_penalty(111) − occupancy_penalty(90)` must exceed the cost of one
> extra weight stream at M = 6.

| quantity | value |
| --- | --- |
| `<T,6,3>` / `<T,6,6>` without `Omega` | 1.03342 |
| `<T,6,3>` / `<T,6,6>` with `Omega` | 1.03024 |
| `Omega_R(111)/Omega_R(90)` supplied | **1.00309** |
| required | **> 1.03342** |

**Fail.** Occupancy supplies 0.31 % where 3.34 % is needed, short by a factor of
10.8. At rank, halving weight traffic and paying 21 extra registers remains a
net **win** in this model, not the net loss the crown's behaviour implies. The
measured wide-prompt effect is −0.44 %; the model predicts +3.02 %.

### Validation 3: control 2, E33

Unchanged from E73, as the algebra requires, because `Omega` is a per-IPG
constant that nearly cancels in the E33 contrast: `Omega_L(3)/Omega_L(6)` =
0.9996. Predicted span 1.5048 to 1.7394 against an observed 0.9830 to 1.0592,
so errors of +53 % to +67 %. Rank direction is right, `tau` = −1.000 predicted
and observed. Occupancy does not repair E33 and was never going to.

## Rung 3: not started

The pre-registered stop rule was "do not start rung 3 unless the ranked-ordering
validation reproduces the correct sign". It reproduced the wrong sign at M = 5,
M = 6 and M = 9. Rung 3 did not run. Publishing a ranked-optimal table from a
model that is 19 times wrong on the one ranked quantity we can check would be
misleading.

## Two findings that were not asked for

### 1. The ranked extrapolation is falsified without reference to the crown

At the measured exponent the model puts M = 7 and M = 8 at IPG **5**. Both our
table and the crown's table ship IPG 4 there, and the crown searched that space
independently. So the ranked prediction is wrong at two widths where the two
tables *agree*, which is a crown-free falsification.

The cause is visible in the pre-registration I posted: at `gamma = 0` the pure
core-count term already moves M = 7 and M = 8 from IPG 4 to IPG 5, and an
exponent of about 0.05 or more is needed to restore IPG 4. The measured 0.0135
is below that. The local margin at M = 7 and M = 8 is only 0.22 %, so these
widths are the surface's most fragile predictions and they break first under
extrapolation. That points the blame at the `lam(IPG)/x` core-count term, not at
occupancy.

### 2. The surface's `q(4)` is wrong by about 3 %

The natural contrast is the cleanest possible measurement of `q(IPG)`, and it
disagrees with the fitted surface.

| M | IPG | observed | model | model error |
| --- | --- | --- | --- | --- |
| 6 | 3 -> 4 | 0.99962 | 0.96041 to 0.97103 | **−2.5 % to −4.3 %** |
| 7 | 4 -> 5 | 1.00822 | 0.99807 to 1.00550 | −1.5 % to +0.2 % |
| 8 | 4 -> 5 | 1.01231 | 0.99807 to 1.00550 | −0.6 % to −1.5 % |

Across the 12 spill-free contrasts the median absolute error is 1.50 % and the
maximum is 4.26 %, against a session null of 0.019 %. The surface believes IPG 4
is about 3 % cheaper than IPG 3 at equal M and equal group count. Directly
measured, it is not cheaper at all.

This is not a small bookkeeping error. The model's ranked argmin at M = 6 is
IPG 4, and M = 6 is 33.4 % of the ranked width pool. That prediction rests
entirely on a 3 % discount that this experiment shows does not exist.

Chaining the measured contrasts from `q(3) = 1` gives `q(4)` = 1.000,
`q(5)` = 1.010, `q(6)` = 1.095, against the fitted 0.971, 0.976 and 1.046. The
directly measured `q` is monotone increasing in IPG. The fitted `q` is not.

## What I would do next, and did not do

1. **Refit the surface with `q(IPG)` pinned to the natural contrast.** The 20
   contrast measurements are cleaner than any cell in the 114-cell surface,
   because they hold M, group count, weight traffic, grid and occupancy fixed.
   Pinning them removes the largest known error in the model before any further
   ranked extrapolation. This is cheap: no new GPU time is needed.
2. **Attack `lam(IPG)/x` instead of occupancy.** The M = 7 and M = 8 failure is
   a pure core-count transfer failure. A session that varies the *dispatch
   width* at fixed cell on this host would measure the tail term directly, in
   the same way the ladder measured pressure directly.
3. **Model spill explicitly, not through `q`.** Our local `<T,6,6>` spills 16 B
   and the ranked `<T,6,6>` does not. The spill curve gives about 0.015 % per
   frame byte. Folding the measured frame bytes of each host into the model
   removes a host asymmetry that `q(IPG)` currently hides, and it is the one
   register-file effect in this experiment that is actually large enough to
   matter.
4. **Add the missing `m4_ipg3` arm.** `m4_ipg2` versus `m4_ipg3` is the largest
   natural register contrast available locally, 70 against 93 with the group
   count held at 2. I noticed this after the session had passed the thermal
   gate, and I judged a relaunch more costly than the gap. It would extend the
   `q(IPG)` chain down to IPG 2.

## Honest summary

The experiment did what it was designed to do. It isolated register pressure
from every confound, measured it to 5 % relative precision, and used
pre-registered thresholds to test it. The answer is that register occupancy is
real, smooth, tiny, and not the mechanism behind the crown's ranked table.

The pre-registered thresholds were the right instrument. Without them this would
have read as "occupancy has the right sign", which is true and useless. With
them it reads as "occupancy is short by a factor of 12 at the easiest width and
a factor of 220 at M = 5", which closes the question.

The most valuable output is not the occupancy number. It is the natural-contrast
data, which is the cleanest `q(IPG)` measurement the campaign has and which
shows the current surface is wrong by 3 % at exactly the width that carries a
third of the ranked pool.
