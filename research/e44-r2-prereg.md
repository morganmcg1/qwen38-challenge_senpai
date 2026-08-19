# E44 r2 pre-registration — the narrow `M ∈ {7,8}` simdgroup-matrix cell

Written and committed **before the narrow variant was compiled**, on the same
ordering as the r1 Gate 0 pre-registration. Everything below is a claim that can
fail.

## 0. Identity

| field | value |
|---|---|
| base | `9fe0dc5dbdb30af4c807ea71873df99e2da72aa2` (E42, E45, E46 merged) |
| r1 base | `efff400c1b5554be2e8993b01856653d55de7664` |
| shipped-surface drift `efff400 → 9fe0dc5` | **empty** — `git diff efff400 9fe0dc5 -- Sources Vendor benchmark.json mlx-generated` produces no output |
| branch | `qwen-alphonse/simdgroup-qmv-register-gate`, rebased onto `9fe0dc5d` before any r2 measurement |
| host | Apple M4 Pro, `applegpu_g16s`, 48 GiB (ranked box is M5 Max `applegpu_g17s`) |
| Metal toolchain | Apple metal version 32023.883 (metalfe-32023.883) |

**Every register number in this document is attached to a tree.** The r1 base
table below was measured on `efff400`, and it transfers to `9fe0dc5` *only*
because the shipped-surface diff between those two commits is empty — that is
verified above, not assumed.

Base cell table (lane-corrected, tree `efff400` ≡ `9fe0dc5` on the shipped
surface), per width M:

```
M            2    3    4    5    6    7    8    9      mid tier (<4096, non-_m)
regs        89   83  104   87   83  108  104   83                            89
cell     <T,2> <3,3> <4,4> <5,3> <6,3> <7,4> <8,4> <9,3>
```

kernel-wide max **108** at `<T,7,4>`; entry `affine_qmv_fast<bfloat16_t,64,4,false>`
**163**; static threadgroup bytes **0**.

r1 all-widths candidate (tree `c065c62`/rebased `63fdcc5`): kernel-wide max
**89**, entry **143**, sgmm cell naive **344** / lane-corrected **34**, 0 allocas,
0 threadgroup bytes.

## 1. What r2 changes, and why

r1 replaced the scalar cross-row cells at **M ∈ [4,9]** and was refuted: net
**−7.341 %** over the replaced widths, 7/12 resolved regressions. The
weight-streaming mechanism I predicted was **wrong**; the mechanism that actually
fits both shapes is a **fixed 8-row MMA tile** — candidate cost is flat in M
(plateau 139.76 µs / 458.99 µs, cv 2.28 % / 2.00 % across M = 4..8) while the base
rises +71.3 % / +78.6 % from M=4 to M=8. The sign of the effect is set by where
the rising base crosses the flat candidate, bracketed in **M ∈ [6,7]**, and M=9 at
1.6× plateau is the second tile.

That mechanism **predicts** the dispatch set: use the MMA cell only where the base
has already risen past the flat candidate cost, i.e. **M ∈ {7,8}**, and nowhere
else. r2 measures exactly that variant. The cell body is **unchanged from r1** —
only the dispatch range changes — so the comparison isolates the one thing that
differs between r1's measurement and the bankable configuration: the shared
register allocation the cells run under.

**Why r1's M=7,8 numbers do not transfer without re-measurement.** In r1 every
cell at M ∈ [4,9] was the MMA cell, so the kernel-wide max fell to 89. In the
narrow variant the scalar cells at M ∈ {4,5,6,9} survive, `_m<T,4,4>` = 104 is
still instantiated, and the whole kernel — including the MMA cell at M=7,8 — runs
under a **104** allocation instead of 89. `affine_qmv_fast` is one `[[kernel]]`
with a runtime width switch and every helper inlined, so there is exactly one
allocation and it is the max over instantiated cells. r1 measured the win at 89.
The bankable variant runs it at 104.

## 2. Gate A — register readout, compile-only, zero GPU seconds

Same instrument and same five-part verdict as r1 Gate 0
(`research/e44_sgmm_air.sh`, `research/air_kernel_stats.py --simdgroup-distributed`,
opt-in and default-off so no previously published number moves).

**(a) Primary bound — pre-registered now.** Candidate **lane-corrected
kernel-wide max ≤ 108** (the base ceiling on this tree). `> 108` is a fail and I
stop: above the base ceiling every untouched width pays the E27 occupancy tax.

**(b) Spill allocas, independent and first-class.** Any alloca type the base cells
do not carry (`[4 x [4 x i16]]` is the only one they have) fails Gate A on its own
even if (a) passes.

**(c) Entry point vs 163** — corroboration only. The entry also inlines
`qmv_fast_impl` and `adjust_matrix_offsets`, so it moves for reasons the cell
table can attribute and it cannot.

**(d) Static threadgroup bytes, base 0.** One kernel ⇒ a `threadgroup` array
declared for one width cell is allocated on every dispatch of every width.
Non-zero is reported with exact bytes and priced.

**(e)** If the naive and lane-corrected verdicts disagree, the live
`<64 x float>` count must account for the entire gap; if any non-distributed type
contributes, Gate A is reported **inconclusive**, not passed.

### Predictions (Gate A)

1. Kernel-wide max = **104**, binding cell `_m<T,4,4>`. Ceiling movement
   `108 → 104` = **−3.70 %**, *not* the −17.6 % the all-widths variant reached.
2. The **89 floor does not bind here** and the binding cell is not a mid-tier
   cell. I retract 89 as available headroom for the bankable variant — stated in
   the r1 submission as a derivation over the measured cell table, and measured
   directly here.
3. sgmm cell 0 allocas, 0 threadgroup bytes, lane-corrected ≈ 34 (unchanged body).
4. Entry between **143** (r1 all-widths) and **163** (base), inclusive.

## 3. Gate B — exactness of the narrow dispatch

Two-pass integer coverage on the existing harness (`--coverage`), all widths,
both shapes. Inputs are constructed so every value and every partial sum is an
integer ≤ 120 that bf16 holds without loss, which makes the comparison a
**bit-equality** check rather than a tolerance. Required: `bad = 0`,
`worst_abs = 0.0` on every line. M ∈ {4,5,6,9} are byte-identical to base by
construction in this variant and act as a harness control.

**What this does and does not establish, stated up front.** It establishes index
mapping, nibble decode, group/scale/bias association and dispatch routing
exactly. It does **not** establish bit-identity for general bf16 values, because
the MMA fixes its own 8-wide summation order. Neither does the shipped base do
so: the scalar cross-row cells at M ≥ 2 already differ from the M=1 serial
readout, which is why exactness on the scored path is an empirical golden-run
gate rather than an algebraic identity. Clearing the cost bar here does **not**
authorise shipping; the golden run does, and it is the next gate after r2.

## 4. Gate C — timing

`research/run-e44-qmv-ab.sh`, ABBA-counterbalanced inside one harness process at
~100 ms granularity, both shapes (`attn_out n=5120 k=5120`,
`mlp_down n=5120 k=17408`), bf16 affine 4-bit group-64, widths **M ∈ {1..9}**,
`--reps 50 --inner 20`.

**Design change from r1, pre-registered with its reason: `--pairs 9` (df = 8),
and a second session that is a true A/A control.**

r1's floor came from 6 zero-effect cells at M ∈ {1,2,3} only — the three
*cheapest* widths — and the worst |effect| there (0.628 %) exceeded the
pre-registered MDE (0.5040 %), with two guard widths' CIs excluding zero on a
true zero. Two things follow and both are tested here:

* **A/A control session**: identical design, candidate arm = base arm bytes. That
  gives **18 true-zero cells across all nine widths and both shapes**, including
  the expensive widths where r1 had no zero-effect data at all. The honest
  resolution floor for r2 is read from this control, not assumed.
* **pairs 5 → 9** discriminates the floor's nature. If the 0.628 % floor is
  stochastic it shrinks by √(9/5) = 1.34× to ≈ 0.47 %; if it is systematic
  (thermal drift, allocator/clock state) it stays ≈ 0.63 % and **more pairs is not
  a fix**. Reported either way.

Analysis is by paired t on the ABBA pairs with **df = pairs − 1 = 8**, so
MDE(exact, df=8) = (t₀.₉₇₅,₈ + t₀.₈,₈)/√9 × sd = (2.30600 + 0.88890)/3 × sd
= 1.06497 × sd. At r1's guard sd of 0.283 % that is **0.301 %**; the reported
floor will be `max(MDE, worst |effect| in the A/A control)`, per width where
available.

### Pre-registered predictions (Gate C)

1. **Width term.** The four r1 cells reproduce within the empirical floor:
   attn M7 **+10.46 %**, attn M8 **+16.65 %**, mlp M7 **+4.46 %**, mlp M8
   **+13.05 %**. Falsified if any of the four flips sign or falls below **+2 %**.
   Justification: the narrow tree runs those cells at 104 instead of 89, i.e.
   **+16.9 % allocation**, and E40 priced a +19.4 % raise at **+0.186 %** mean
   tax — so the expected transfer loss is ≈ **+0.16 %** of cell time, ~4× below
   the floor.
2. **Ceiling term.** Untouched widths M ∈ {1,2,3,4,5,6,9} run byte-identical code
   in both arms and differ only through the shared allocation (104 vs 108,
   −3.70 %). Predicted effect **+0.0355 %** (faster) — **~18× below the floor**,
   therefore *not measurable*, reported as a **bound**, never as a confirmation.
   Failure condition: any **resolved regression** larger than the A/A floor at an
   untouched width.
3. **Bar.** Net over the **replaced** widths M ∈ {7,8} across both shapes
   **≥ +5 %**, with all four cells positive and resolved, **and** no resolved
   untouched-width regression. Both halves are required; r1's own failure was a
   net-negative replaced set behind a single width that cleared the bar.
4. **Flat-tile check.** The candidate remains flat in M at the two measured
   widths: |cand(M=7) − cand(M=8)| / cand(M=8) < 3 % on both shapes. This is the
   mechanism's own signature and is the reason M=7,8 were chosen; if it fails, the
   mechanism I am now claiming is also wrong.

## 5. Score decomposition — the two halves have opposite signs

From askeladd's E42, measured causally and bit-exactly on the ranked-relevant
path: **ψ_mtp = 0.6736** (QMV share of the candidate leg, verify widths 2..9) and
**ψ_serial = 0.8525** (QMV share of the serial leg, width 1 only). With
`raw_p = serial / mtp` and serial in the numerator:

| channel | where it acts | dScore/dx per 1 % QMV cost removed |
|---|---|---|
| **width term**, M ∈ {7,8} | MTP leg only — the serial leg never dispatches width 7 or 8 | **+0.674 %** |
| **ceiling term**, allocation 108 → 104 | every width **including M = 1** ⇒ uniform | **−0.179 %** |

These are reported **separately, with the sign attached**, and never aggregated.
An aggregate of a `+0.674` term and a `−0.179` term is uninterpretable as score —
that is the E27 trap one level down.

**Priced now, before measuring:**

* Ceiling term: −3.70 % allocation × 0.00959 %cost/%reg (E40's price, +19.4 %
  ↔ +0.186 %) = **0.0355 % uniform speedup** → ΔScore = −0.179 × 0.0355 =
  **−0.0064 %**. Adverse in sign, negligible in size, ~100× below anything this
  instrument can see. **The ceiling channel can neither repay nor ruin this
  mechanism. The width channel decides it.**
* Width term: mean r1 win over the four cells is **11.15 %**. If a fraction `f`
  of MTP-leg QMV cost is dispatched to the wide tier at M ∈ {7,8}, then
  ΔScore ≈ 0.674 × f × 11.15 % = **7.5 f %**.

| f (share of MTP QMV cost at M ∈ {7,8}) | 0 | 0.05 | 0.10 | 0.25 | 0.50 |
|---|---|---|---|---|---|
| ΔScore | 0 % | +0.38 % | +0.75 % | +1.88 % | +3.76 % |

🔴 **`f` is unidentified.** E43 established that the ranked depth mixture cannot
be recovered from the eight per-round costs under any admissible family, so **no
row of that table is a prediction**. It is a sensitivity table, and the campaign
needs `f` from a width census on the scored path, not from me.

**The asymmetry is the argument for the narrow variant.** r1's all-widths variant
had a large downside if the mixture concentrated at M ≤ 6 or M = 9 (measured
−41 % to −52 % at M=4). The narrow variant changes behaviour *only* at widths
where it has already been measured faster; every other width runs the same
instructions. Its downside is bounded by the ceiling term at **≈ −0.01 % of
score** and it is exactly zero if `f = 0`.

## 6. Power caveat, stated before running

The quantity that decides how much the ceiling half costs is predicted at
**0.0355 %**, and the honest resolution floor is **0.628 %** (r1, worst
zero-effect |effect|). If that floor were purely stochastic, resolving 0.0355 %
would need n ≈ 5 × (0.470 / 0.0355)² ≈ **880 pairs**, and only if nothing
systematic survives averaging — which the A/A control exists to test, and which
r1's evidence (worst |effect| above the sd-derived MDE) already argues against.
**So more pairs is not the fix, and I am not claiming the ceiling term as a
measurement in either direction.** It is priced from E40's ledger value and
bounded by the guard. The width term, by contrast, is expected at +4 % to +17 %,
i.e. 6×–27× the floor, and pairs = 9 is comfortably sufficient for it.

## 7. Stop rules

* Gate A `> 108`, or a new alloca type ⇒ **stop**, report, do not time.
* Gate B any `bad ≠ 0` ⇒ **stop**, report as invalid.
* Gate C: bar in §4.3 fails ⇒ report negative and close the narrow arm too.
* Gate C: bar passes ⇒ report **cost bar cleared, exactness unproven**, with the
  golden-run validation named as the next required gate. The scored surface on
  this branch ends at base either way; r2 ships nothing.
