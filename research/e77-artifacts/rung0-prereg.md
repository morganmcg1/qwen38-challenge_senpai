# E77 rung 0 — the occupancy respecification, and what it must beat

Written before the first timed leg. Base `41ddc183817979be8d2f0817d79f98b2ddefb984`.
Host Apple M4 Pro, `applegpu_g16s`, 20 GPU cores. Ranked host M5 Max,
`applegpu_g17s`, 40 GPU cores (**an inference, not a measurement**).

Artifacts: `rung0-regs.json` (register census, both generations),
`rung0.json` (thresholds). Zero candidate files changed.

## 1. The model E73 left, and the hole in it

E73's fitted form was

```
t(M, IPG, shape) = [ groups*W(shape) + beta*M*k*Tn ] * rho0 * q(IPG) * (1 + lam(IPG)/x)
x = groups*Tn/cores,   Tn = ceil(n/8),   groups = ceil(M/IPG)
```

`cores` was the only host term. E73 then proved that `argmin` over IPG is
invariant to any pure rescaling of `rho0`, so no bandwidth difference between
hosts can move a partition, and it showed that `cores` alone cannot reach the
crown's table: M=5 needs 272 cores, M=9 needs 992, and **M=6 is unreachable at
any core count**, because at M=6 the crown's IPG=3 loses to IPG=4 on the level
term `q`, and occupancy cannot overturn a level difference at any grid size.

`q(IPG)` is the hole. It is a per-IPG rate level fitted on this host, and it was
transferred to the ranked host as an identity. That is only correct if nothing
inside `q` is host-dependent.

## 2. Rung 0 result: `q(IPG)` is exactly where the register term hides

`research/e77_reg_census.py` runs edward's E72 oracle over **all 19 legal
`(M, IPG)` cells**, on both generations. E72 covered 6 of them; this covers all.

| IPG | local registers | local frame | local resident SGs/core | ranked registers | ranked frame | ranked resident SGs/core |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 70 | 0 | 43 | 83 | 0 | 47 |
| 3 | 93 | 0 | 33 | 90 | 0 | 44 |
| 4 | 94 | 0 | 32 | 91 | 0 | 43 |
| 5 | 95 | 0 | 32 | 98 | 0 | 40 |
| 6 | 96 | **16** | 32 | 111 | 0 | 35 |

**On both hosts the register count is a function of IPG alone.** Every one of
the 19 cells at a given IPG reports the same register count and the same frame
bytes, tails and second inlined bodies included. Two consequences:

1. **`Omega_L(R_L(IPG))` is perfectly collinear with `q(IPG)` on the local
   surface.** E73's `q(IPG)` is not a code-level term; it is the product of a
   code-level term and the local occupancy penalty, and **no fit on the 114-cell
   surface can separate them.** Rung 1 is not a refinement of E73. It is the
   only way to identify the term at all.
2. The local span is 70 to 96 registers, but the six *shipped* cells span only
   93 to 96, and 3 of those 4 counts sit in the same resident-simdgroup bucket.
   The ranked host spreads the same six cells over 90 to 111 and four buckets.

## 3. The respecified model

```
t(M, IPG, shape, h) = [ groups*W + beta*M*k*Tn ] * rho0 * c(IPG)
                      * Omega( S_h(R_h(IPG)) ) * (1 + lam(IPG)/x_h)

x_h      = groups*Tn/cores_h
S_h(R)   = floor( B_h / (128*R) )        resident simdgroups per core
c(IPG)   = q(IPG) / Omega( S_L(R_L(IPG)) )
```

`128 = 32 threads * 4 bytes` is the register-file cost of one register for one
resident simdgroup. `R_h(IPG)` is **measured** by the oracle, table above.
`Omega` is one host-independent function of resident simdgroups, **measured in
rung 1**. `c(IPG)` is the residual code-level level term — idle x-slots, the
tail body, instruction mix — and is host-independent by construction.

Two host terms now: `cores_h` and `B_h`. `B_L` is the hypothesis rung 1 tests by
locating the steps; `B_R` is an **extrapolation**, derived from the ranked
allocator ceiling of 124 registers under the same "stop at 32 resident
simdgroups" rule the local ceiling of 96 obeys at `B_L = 384 KiB`, giving
`B_R = 124*128*32 = 496 KiB`.

## 4. The new form can flip an `argmin`. The old one could not

Fix `M` and compare two partitions `i` and `j`. Write `t_h(i) = A_h(i)*L_h(i)`,
where `A_h` collects everything except the level term and `L_h(i) = c(IPG_i) *
Omega(S_h(R_h(i)))`. A flip between hosts is

```
t_L(i) < t_L(j)   and   t_R(i) > t_R(j)
```

Divide through and let `a_h = A_h(i)/A_h(j)`:

```
flip  <=>  a_L < L_L(j)/L_L(i)   and   a_R > L_R(j)/L_R(i)
```

`c(IPG)` is host-independent, so it cancels out of the *ratio of ratios*:

```
L_R(j)/L_R(i)     Omega(S_R(R_R(j))) / Omega(S_R(R_R(i)))
-------------  =  ---------------------------------------
L_L(j)/L_L(i)     Omega(S_L(R_L(j))) / Omega(S_L(R_L(i)))
```

At equal core counts `a_L = a_R = a`, and the flip set is the open interval

```
Omega(S_R(R_R(j)))/Omega(S_R(R_R(i)))  <  a  <  Omega(S_L(R_L(j)))/Omega(S_L(R_L(i)))
```

which is non-empty **iff the ranked relative occupancy penalty between the two
cells differs from the local one**. Under E73's form `Omega === 1`, both bounds
collapse to 1, the interval is empty, and no flip exists at equal cores — which
is exactly the invariance E73 proved and exactly why its M=6 row was
unreachable. Under the new form the bounds separate whenever `R_h` maps the two
cells into different resident-simdgroup buckets on one host and the same bucket
on the other. The table in section 2 shows that happening: locally IPG 4, 5 and
6 all sit at 32 resident simdgroups, while on the ranked host they sit at 43, 40
and 35.

**So the respecification passes the test the E73 brief failed. It can represent
the effect, including its sign.** The rest is measuring `Omega`.

## 5. Pre-registered predictions

Equivalently, and this is the form rung 2 uses: replacing `q(IPG)` by
`c(IPG)*Omega(S_h(R_h(IPG)))` multiplies every E73 ranked cost by the transfer
gain `G(IPG) = Omega(S_R(R_R(IPG))) / Omega(S_L(R_L(IPG)))`. The crown's cell
beats ours at width M exactly when `G(ours)/G(crown)` exceeds the E73 ranked cost
ratio. Everything else cancels.

| M | ours | crown | local regs ours/crown | ranked regs ours/crown | required `G(ours)/G(crown)` |
|---:|---:|---:|---|---|---:|
| 5 | 5 | 3 | 95 / 93 | 98 / 90 | **1.2113** |
| 6 | 6 | 3 | 96 / 93 | 111 / 90 | **1.0330** |
| 9 | 5 | 3 | 95 / 93 | 98 / 90 | **1.1497** |

M = 3, 4, 7 and 8 already agree with the crown and must not move.

**PR-1, sign.** The occupancy coefficient is positive: at fixed traffic, fixed
group count and fixed grid, GPU time is non-decreasing in register count over
70 to 96, to within the session null.

**PR-2, magnitude.** On the large-grid shape `head.lm_head`,
`time(96)/time(70)` lands in **[1.02, 1.35]**. The upper bound is the fully
occupancy-limited law `Omega = S_ref/S`, which predicts 43/32 = 1.344; the lower
bound is my floor for calling the mechanism real. On the small-grid shape
`linear_attn.out_proj` the effect is smaller, because 640 working threadgroups
on 20 cores do not fill the residency the register file is limiting.

**PR-3, staircase.** The response is a staircase, not a line. With
`B_L = 384 KiB` the steps in [70, 96] sit at **R = 72, 74, 75, 77, 79, 81, 84,
86, 88, 91, 94**, and the segments between them are flat to within 3x the
session null. If the measured step set is offset, it identifies a different
`B_L`, which is a result rather than a failure. Relative to the cells that
matter, this puts a step between 90 and 91 (IPG 3 vs 4 on the ranked host) and
no step between 94, 95 and 96 (IPG 4, 5, 6 locally).

**PR-4, traffic control.** The `q` arms carry the same P loads with the live
range moved off the loop. They differ from `p0` by **less than 0.2 %**. If they
do not, the sweep is measuring its own loads and the pressure arms are
uninterpretable.

**PR-5, spill.** The frame-bytes arms on the `m6_ipg3` carrier hold R fixed at
96 while frame bytes rise 0, 16, 48, 112. Their cost rises with frame bytes and
exceeds anything the occupancy trend predicts at fixed R. That is spill, not
occupancy, and it is reported separately.

**PR-6, which cells reorder.** Under the pre-registered prior `Omega(S) =
S_ref/S`, with resident simdgroups from section 2 and `B_R` extrapolated:

| M | required | prior `G(ours)/G(crown)` | pre-registered verdict |
|---:|---:|---:|---|
| 5 | 1.2113 | 1.0667 | **does not reorder** |
| 6 | 1.0330 | 1.2190 | **reorders to the crown's IPG=3** |
| 9 | 1.1497 | 1.0667 | **does not reorder** |

So I pre-register **one of the three**: M=6 reorders, M=5 and M=9 do not. This
is a genuine risk. The ranked receipt says the crown's whole table wins by
0.298 % on the scoring prompts and does not decompose per width, so a model that
reorders only M=6 can still reproduce the receipt's sign — M=6 carries 33.4 % of
ranked width time against M=5's 24.1 % and M=9's 5.75 % — but it will predict a
different magnitude, and it names a third table rather than the crown's.

If the measured `Omega` is steeper than `S_ref/S` near the residency knee, M=5
and M=9 flip as well. If it is flatter, none of them do and the mechanism is
refuted.

**PR-7, control 1 cannot fail, and that is a property of the design.** Because
`R_L` is a function of IPG alone (section 2), `Omega(S_L(R_L(IPG)))` is absorbed
exactly by `c(IPG) = q(IPG)/Omega(S_L(R_L(IPG)))`, the product is unchanged, and
the local predicted table is bit-identical to E73's. Control 1 will pass 7 of 7.
I am reporting this now, before running it, because a control that cannot fail
is not evidence, and it should not be presented as if it were. What control 1
does still verify is that the refit is arithmetically consistent and that no
cell was mis-keyed.

## 6. Stop rule

- Stop and report if the sweep cannot raise the register count without changing
  the emitted arithmetic. **Already cleared at rung 0**: the `m6_ipg2` carrier
  moves 70 to 96 in unit steps across 27 pressure levels, with zero frame bytes
  up to 96, wrapping the shipped `qmv_fast_crossrow_affine4_g64_m` call
  unchanged.
- Stop and report if `time(regs)` is flat across the clean range within the
  session null. That falsifies the occupancy mechanism on this host and is a
  complete answer.
- Do not start rung 3 unless the ranked-ordering validation reproduces the
  correct sign.
