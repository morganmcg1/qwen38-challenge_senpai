# E41 — Resolving E38's R2 into activation re-read (MEM) vs register tile (ILP)

PR #46 · branch `qwen-thorfinn/r2-confound-before-ktiling` · base
`senpai/qwen38-mtp-r1` @ `04ad6bf11437c269df85a47e91faa769c74fe6da`

## Verdict

**ILP / register tile. K-tiled activation staging is dead and deliverable (b)
was not built.**

E38's R2 is the halved register tile and the extra sequential loop, not the
activation re-read. Shrinking the re-read distance does not refund the tax — it
**adds** to it. A future NA=6 single-weight-pass scheme must budget the full
~+11 % at M=6 as unavoidable, because there is no locality left to recover.

## Question

E38 measured a **+10.54 % kernel cost** at M=6 for halving the row tile of the
wide crossrow affine-4 g64 QMV kernel (its "arm (a)"). That number is the price
any future NA=6 single-weight-pass scheme must pay, so it gates deliverable (b),
K-tiled activation staging. But E38's arm (a) changed two things at once:

1. **MEM** — each activation element `x[k]` is now read by two row blocks
   instead of one, and the two reads are separated by a full pass over K.
2. **ILP / register tile** — the register tile halved (4 rows/SIMD → 2), the
   accumulator count halved, and one extra sequential loop appeared.

K-tiled staging can only recover cause 1. If R2 is actually cause 2, building
(b) is wasted work. E41 attributes R2 before any staging is written.

## Why the advisor's suggested arm (a) would have answered the wrong question

The assignment sketched an **unrolled** arm: emit the two row blocks' loads
adjacently in one loop body and see whether the tax disappears. That arm is
unsound here, and it fails in the direction that *falsely kills* K-tiling.

Adjacent row blocks read **identical device addresses** for `x[k]` — the
activation is shared across output rows; only the weight stream differs. With
no intervening store between them, **common-subexpression elimination is legal**,
and the compiler is entitled to delete the second load outright. If the tax then
vanished, the recovery could be attributed either to instruction-level
parallelism or to a deleted load, and those are exactly the two hypotheses under
test. Worse, a CSE'd arm makes the MEM cost unobservable by construction, so a
real re-read cost would read as "no MEM component" → ILP → K-tiling dead.

## Registered alternative: a K-tile distance ladder

Instead of removing a load, hold the **instruction mix, trip counts and
accumulator count fixed** and vary **only the reuse distance** of `x[k]`:

```
for kt in K-tiles:            # k_tile = KT * 512 values
  for b in row blocks:        # BLOCKS_PER_CALL
    for k in this tile:       # the verbatim E38 k-block body
```

Both row blocks stay live, both loads are still issued, and the loop nest is the
same shape at every rung. `KT` alone moves the second block's read of `x[k]`
from "one full pass over K later" (`KT=64`) to "512 values later" (`KT=1`).

| rung | reuse distance | role |
|---|---|---|
| `KT=64` | spans K in one tile (32 768 ≥ widest scored K = 17 408) | no locality |
| `KT=4` | 2 048 values | **discriminator** |
| `KT=1` | 512 values | total-recovery bound |

`KT=1` is *not* the discriminator: both hypotheses predict some recovery there
(it is also the shortest loop). The single-mechanism signal is `KT=64 → KT=4`.

## Pre-registration and the compile-only gate

`research/e41_prereg.py` was committed (`1e8bc95`) **before any kernel compile**
and carries the advisor-corrected constants verbatim: score sensitivity 1.00,
crown 0.5193 %, engineerable gap 0.2586 %, σ_score 0.0978 %, control band
±0.46 %, E38 arm (a) M=6 ratio 1.1054, and ψ·φ = 0.0459 flagged
**BACK-SOLVED, not measured**.

Before spending GPU time, `research/e41_ktile_census.py` compiled every ladder
cell to AIR and checked the design's core claim. Results in
`research/e41-ktile-census.json` (**CENSUS OK**):

- **Non-perturbation.** The shipped kernels are untouched by the refactor:
  `xship_na2/3/4/5` = **62 / 83 / 104 / 125** registers with **32 / 36 / 40 / 44**
  device loads — identical to the E32/E36 census.
- **E32's row-blocked grid reproduces exactly** from the new template
  (`xrb_na3_r1/r2/r4` = 51/66/83; `xrb_na4_*` = 63/83/104; `xrb_na6_*` =
  87/117/144), so the template is a faithful generalisation, not a new kernel.
- **The ladder is one mechanism.** At every (NA, r), `KT ∈ {1, 2, 4, 64}` gives
  **identical** `peak_live_regs`, `device_loads`, `vector_float_ops`,
  `loop_backedges` (=4) and `allocas` (=2). This is the census evidence that the
  `KT=64 → KT=4` step changes reuse distance and nothing else.
- **No timed cell spills.** `xkt_na3_r2` = 77 regs, `xkt_na4_r2` = 97,
  `xkt_na6_r1` = **105** (fits under the 128 wall), `xkt_na6_r2` = 135 (over —
  excluded from timing). Spill-gate controls fired correctly.

`xkt_na6_r1` fitting at 105 registers is the concrete shape deliverable (b)
would use, and it needs **no threadgroup memory and no barrier**.

Registered-vs-measured register predictions were scored honestly: **4 of 7** in
band. Misses: `xkt_na4_*` 97 vs predicted 88–94 (by 3), `xkt_na6_r2` 135 vs
126–132 (by 3), `xkt_na6_r1` 105 vs 116–122 (by 11, in the favourable
direction).

### Two amendments recorded before the GPU ran

1. **"KT = all K" is `KT=64`, not `KT=0`.** `KT=0` sets `k_tile = in_vec_size`,
   which the compiler can prove is trip count 1, so it deletes the loop
   (`loop_backedges` 4 → 3). That would have confounded the top rung with a loop
   removal. `KT=64` spans K in one tile but keeps the loop.
2. **Discriminator M=7 → M=8, bound M=8 → M=7.** M=7 dispatches IPG=4 with a
   TAIL=3 group, mixing NA=4 and NA=3 work. M=4 and M=8 are both **pure NA=4**,
   so `M=4 → M=8` is a one-constant step. M=7 is retained only as the adjacency
   bound, where impurity is tolerable.

## Method

One base build and one arm build, same host, same session, same
`--reps 21 --inner 10` shapes-only configuration; the ratio is taken per M
between the two builds. The arm build carries the whole ladder at once so that
widths **1, 2, 5 and 9 are byte-for-byte the base** and serve as untreated
controls (must land within ±0.46 %).

| M | base | arm | rung |
|---|---|---|---|
| 3 | `<T,3,3,true>` | `<T,3,3,true,2,2,1>` | NA=3, KT=1 |
| 4 | `<T,4,4,true>` | `<T,4,4,true,2,2,64>` | NA=4, no locality |
| 6 | `<T,6,3,true>` | `<T,6,3,true,2,1>` | E38 arm (a) tax anchor |
| 7 | `<T,7,4,true>` | `<T,7,4,true,2,2,1>` | NA=4, adjacency bound |
| 8 | `<T,8,4,true>` | `<T,8,4,true,2,2,4>` | **NA=4, discriminator** |

The M=6 anchor is deliberately `KT=0`, i.e. the K-tile loop folded away, which
makes it the sequential row-blocked kernel E38 actually timed. Replicating
1.1054 there is the check that this harness reproduces E38 before any new claim
is read off it.

**Pre-registered decision rule.** With the M=4 rung defining the tax:

- M=8 recovers **≥ 50 %** of the tax → **MEM** → build deliverable (b).
- M=8 recovers **≤ 10 %**, or the M=4 → M=8 step falls inside the ±0.46 %
  control band → **ILP / register tile** → **K-tiled staging is dead; stop and
  report** without building (b).
- Total recovery already inside the control band at M=4 → informative null.

**No E2E leg was run**, per the assignment: the predicted end-to-end move is
0.07–0.5 %, far inside the n=4 minimum detectable effect (0.417 % / 0.632 %).
All ratios below are **kernel-level only**.

## Provenance and gates

| item | value |
|---|---|
| base commit measured | `5d97fe3` (kernel template landed, dispatch table untouched) |
| arm commit measured | `dfe39af` (dispatch table selects the ladder) |
| host | Apple M4 Pro, 48 GiB |
| scope | `quantized.h` + `mlx-generated/quantized.cpp` only (2 paths) |
| twin lock | `cpp = cpp[0:13] + h + cpp[-6:]`; `twin_audit.py` OK, 29 runtime-effective twins |
| editable budget *as measured* | source 2 467 227 / 3 000 000; growth 11 938 / 262 144 |

That budget row describes the **builds that were timed**, which necessarily
carried the ladder. The branch's final scored surface is reverted to zero diff
and zero growth — see [Scored surface](#scored-surface).

Thermal honesty, as required for an ungated local arm: this host's real 40 °C
cool gate cannot be reached (it stalls at ~43.4 °C), so every curve records
`cool_gate_vendored=stalled_above_40C`. Entry and exit GPU temperatures are
logged per arm, and `cool_gate_passed_real_gate=false` /
`gate_qualified_for_timing=false` are preserved verbatim in the W&B record. E38
measured under the same condition, which is what makes the anchor comparable.
These are **directional causal kernel measurements**, not gate-qualified numbers
and not any kind of score.

Precisely on counterbalancing: these are three sequential runs forming an
**A-B-A bracket**, not one interleaved ABBA session. That cancels monotone drift
to first order and yields a measured per-width floor, but it is weaker than true
interleaving, and the M=9 residual below shows the limit of the design.

## Runs

| tag | role | head | W&B | entry °C | exit °C |
|---|---|---|---|---|---|
| `e41-base-r1` | base (A) | `5d97fe3` | `thrh88b8` | 43.42 | 68.08 |
| `e41-arm-r1` | ladder arm (B) | `dfe39af` | `kw7yrfoy` | 43.24 | 69.29 |
| `e41-base-r2` | base replicate (A) | `5c6693a` (twins = `5d97fe3`) | `ryws3yex` | 43.40 | 86.62 |

All three `dirty=0`, `--reps 21 --inner 10`, shapes-only, widths 1–9, one host.
Ratios below use the per-width **geometric mean of both base runs** as the
reference, so monotone drift across the A-B-A sequence cancels to first order.

## Result

### The session replicates E38 before anything new is read off it

M=6 anchor ρ = **1.1065** against E38's **1.1054**, inside the registered band
[1.0954, 1.1154]. Dispatch readback **PASS** for all 12 treated and control
instantiations; fidelity **PASS**, 0 bitwise failures in both builds.

### The NA=4 ladder: reuse distance buys nothing

| rung | reuse distance | ρ | tax |
|---|---|---|---|
| M=4 | `KT=64`, spans K | 1.2066 | +20.66 % |
| M=8 | `KT=4`, 2 048 values — **discriminator** | **1.2297** | **+22.97 %** |
| M=7 | `KT=1`, 512 values — bound | 1.2222 | +22.22 % |

- **Locality recovery (`KT=64 → KT=4`) = −11.2 % of the tax** (−2.31 pp).
- Total recovery (`KT=64 → KT=1`) = −7.5 % of the tax (−1.56 pp).
- Registered rule: MEM needed **≥ +50 %** (i.e. ≥ +10.33 pp); ILP was ≤ +10 %.

Measured recovery is **negative**, so the rule fires **ILP**. Three independent
observations make this more than a threshold crossing:

1. **Sign consistency, 8/8 scored shapes.** The M=4 → M=8 step is negative at
   every shape (−0.52 pp to −3.57 pp), across K ∈ {5 120, 6 144, 17 408} and
   N from 4 096 to 248 320.
2. **The `KT=1` rung is also worse than `KT=64`**, which closes the "not
   adjacent enough" escape. Even 512-value adjacency refunds nothing.
3. **The confounded NA=3 pair moves the same way**: M=3 (`KT=1`) ρ = 1.1493
   versus the M=6 anchor 1.1065, a −40.2 % "recovery". If MEM were real, the
   NA=3 K-tiled form should have beaten the sequential one. Note M=3 sits
   *earlier* in the sweep than M=6, so accumulated heat would bias this pair the
   other way; it does not rescue MEM.

This is mechanically coherent with the census: `KT` holds `peak_live_regs`,
`device_loads`, `vector_float_ops`, `loop_backedges` and `allocas` invariant, so
the only thing a smaller tile can buy is reuse distance and the only thing it
can cost is loop bookkeeping. We measure the cost and none of the benefit, which
means the re-read was never on the critical path.

### The control gate failed, and the replicate explains why

The pre-registered control gate **FAILED** and is reported as such: with the
bracket applied, untreated widths give M1 = 0.9958, M2 = 1.0081, M5 = 1.0010,
M9 = 1.0090 — worst 0.90 % against my registered ±0.46 % band. Against base-r1
alone it was worse: M1 = 0.9726, worst 2.74 %.

The replicate measures each width's **own** floor, because base-r2 is the same
build timed again — `|base2/base − 1|`:

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| floor % | **4.61** | 0.73 | 0.24 | **0.04** | 0.16 | **0.07** | **0.01** | **0.06** | 0.05 |

Reading this honestly:

- **My registered ±0.46 % band was mis-specified.** It was derived from
  score-level σ, not from measured kernel-level reproducibility, and it is
  simultaneously far too loose for M=4/7/8 and impossibly tight for M=1.
- **M=1 was never interpretable.** Its own floor is 4.61 % — it is the first
  width in the sweep and absorbs warmup/JIT (its mean-vs-min spread is
  22.25 % / 16.43 %, an order of magnitude above every other width). After the
  A-B-A correction its deviation is 0.42 %, *below* its own floor.
- **The widths that carry the verdict are the most reproducible in the sweep.**
  Worst floor across M=4, 6, 7, 8 is **0.07 %**, against a measured step of
  **2.31 pp** — a ratio of about **33×**.
- **Two controls exceed their own floor: M=2 (0.81 % vs 0.73 %, marginal) and
  M=9 (0.90 % vs 0.05 %).** M=9 is the real one, and it is **not thermal**:
  base-r2 exited at 86.62 °C against base-r1's 68.08 °C, yet the two agree at
  M=9 to 0.05 %, so M=9 throughput is insensitive across that range. The
  remaining explanation is a **build-level artifact** — every width is
  JIT-compiled from one `quantized.cpp` source string, and the arm build's extra
  instantiations enlarge that compilation unit. An "untreated width" is
  therefore not a perfectly clean control in this harness.

That artifact is bounded at ≈0.9 pp by M=9 itself. The gap between the measured
step (−2.31 pp) and the MEM threshold (+10.33 pp) is **12.6 pp**, so the
artifact is ~14× too small to change the verdict. Separately, the step is a
difference of two ratios sharing both sessions, so a session-level multiplicative
drift `d` perturbs it only by `(d−1) × step` — visible in the data, since
applying the bracket moved locality recovery from −0.114 to −0.112 and the anchor
from 1.1086 to 1.1065.

### Per-shape at the discriminating step

| shape | base µs | M=4 ρ | M=8 ρ | step |
|---|---|---|---|---|
| `linear_attn.in_proj_fused_qkvzba` | 495.76 | 1.1996 | 1.2243 | −0.0246 |
| `linear_attn.out_proj` | 220.34 | 1.1818 | 1.1985 | −0.0167 |
| `full_attn.qkv_proj_fused` | 435.47 | 1.2021 | 1.2289 | −0.0268 |
| `full_attn.o_proj` | 217.63 | 1.2036 | 1.2088 | −0.0052 |
| `mlp.gate_up_fused` | 987.87 | 1.2126 | 1.2297 | −0.0171 |
| `mlp.down` | 546.34 | 1.2069 | 1.2426 | −0.0357 |
| `head.lm_head` | 6 715.09 | 1.2292 | 1.2433 | −0.0141 |
| `head.compact_draft_vocab` | 2 691.90 | 1.2232 | 1.2410 | −0.0178 |

### Value — kernel-level only, and conditional

ψ·φ = 0.0459 is **back-solved from the crown, not measured**; every score figure
inherits that and I do not claim it. On that basis the measured M=6 tax alone
would be worth **−0.4889 %** of score if paid, and the same tax with this
ladder's "recovery" applied is also **−0.4889 %**, because the recovery is
negative. Neither is a gain over the shipped base, which pays no tax at all: the
tax only matters as the price deliverable (b) would have to pay to buy one weight
pass at NA=6, so (b)'s value would be that weight-pass saving **minus** whatever
tax survives K-tiling. E41 measured the second term and found it survives intact.
Context: crown 0.5193 %, engineerable gap 0.2586 %, σ_score 0.0978 %.

## What this means for the campaign

- **Do not build K-tiled activation staging.** The mechanism it targets does not
  exist at these shapes.
- **The row-tile tax is a hard floor for NA=6 single-weight-pass schemes.** Price
  any future proposal in that direction against ~+11 % at M=6 (NA=3) and ~+21 %
  at NA=4, with no staging discount available.
- **The register tile, not the activation stream, is the binding resource.** The
  census already showed `xkt_na6_r1` fits at 105 registers while `xkt_na6_r2`
  needs 135 against a 128 wall. Work that *raises* rows per SIMD or *reduces*
  accumulator pressure is the direction with headroom; work that subdivides K is
  not.

## Scored surface

The ladder is measurement scaffolding for a mechanism that just died, so the
scored surface is **reverted to the assignment base `04ad6bf`** on this branch:

```text
validate-assignment-scope.sh   OK, 2 submitted paths
check-editable-budget.sh       source=2455289/3000000  growth=0/262144
twin_audit.py                  OK, 29 runtime-effective twins
git diff 04ad6bf -- <h> <cpp>  empty
```

The revert goes all the way to `04ad6bf`, not back to the base build `5d97fe3`.
Keeping `5d97fe3`'s template generalization (`krange` / `wide` / `rowblocked` /
`K_TILE_BLOCKS`) would leave dead parameterization with no consumer, and it is
not free: it is the same enlarged JIT compilation unit that follow-up 1 names as
the leading explanation for the unexplained M=9 residual. Pruning it removes a
measurement hazard rather than removing optionality. Leaving the *arm* dispatch
table would be worse still — a ~21 % kernel regression at M=4/7/8 if merged.

The ladder remains fully reproducible from this branch's history — `5d97fe3` for
the template, `dfe39af` for the arm table, `ae272aa` for the restored-base
bracket — plus the committed census, so nothing measured here is lost.

## Suggested follow-ups (not implemented)

1. **Attribute the M=9 build artifact.** A ~0.9 % cross-kernel effect from
   enlarging the JIT compilation unit, if real and general, is a measurement
   hazard for every future A/B in this harness, and possibly a small free win if
   the shipped source string can be trimmed. The cheap test is a build whose only
   change is adding unreachable instantiations.
2. **Retire the score-derived control band for kernel-level work.** Per-width
   floors from a same-build replicate cost one extra curve and are one to two
   orders of magnitude more informative. M=1 should be excluded from control sets
   or given a warmup pass.
3. **Price the NA=6 weight-pass saving directly.** E41 measured only the cost
   side. Whether *any* NA=6 scheme is viable now depends entirely on whether the
   single-weight-pass saving exceeds ~+11 %, which is a separate measurement and
   the one that decides the whole direction.
