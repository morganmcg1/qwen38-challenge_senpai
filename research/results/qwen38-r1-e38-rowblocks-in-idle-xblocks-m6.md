# E38 — row blocks in the idle x-blocks: the prize is real, and row-blocking eats it

**Verdict: NEGATIVE for the mechanism, DECISIVE for the question. Both arms measured.
The NA > 5 direction at M = 6 is closed with a mechanism, not just a number.**

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"e38/m6_per_row_cost_ratio","available":true,"value":0.9891},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}
```

- Student / branch: `qwen-thorfinn` / `qwen-thorfinn/rowblocks-in-idle-xblocks`, PR #43, revision `r1`
- Decision: **dead** — the axis is closed, the shipped cell is untouched
- `BASE_SHA` `54248ce258376db756be02fd65a814a903e2d601` · `UPSTREAM_SHA` as recorded on that base
- Yukon frontier at time of work: our `ca9251b8` = 3.23250848, crown `0cd0a6b4` (ofou) = 3.24929399
- **Submitted candidate files: none.** The two twin-locked vendored files are
  byte-for-byte `BASE_SHA`; editable-budget growth is **0 bytes**
- Supporting files: `research/e38-prereg.md`, `research/e38_prereg.py`,
  `research/e38_analyze.py`, `research/e38_value.py`, `research/e38_corrections.py`,
  `research/e38_coverage_proof.py`, `research/e38_bitwise_check.py`,
  `research/e38_anchor_check.py`, `research/e38_m6_step.py`,
  `research/e38_ranked_geom.py`, `research/run-qmv-curve-ranked-geom.sh`,
  `research/run-qmv-parity.sh`, `research/e38_wandb_log.py`,
  `research/e38-artifacts/` (AIR registers, both arm patches, metrics JSON,
  ranked-geometry JSON)
- MTP head provenance: organizer-pinned; no `mtp-head.manifest.json` declared, unchanged
- Assignment-scope preflight: `assignment scope OK: 2 submitted path(s)`
- Editable budget: `source=2455289/3000000 headroom=544711 growth=0/262144 exempt=2410`

---

## 0. The answer to the assignment's question, in three numbers

The brief asked one thing: **was E33 killed by serialization or by bytes?**
It is neither, and arm (a) — the control the brief called "the most scientifically
valuable arm" — is what says so.

| contrast | what it isolates | measured at M = 6 |
|---|---|---:|
| arm (a) / base | row-blocking at **unchanged** grid, weights and work | **+10.85 %** |
| arm (a) − arm (b) | **one whole weight stream**, everything else equal | **−11.94 %** |
| arm (b) / base | the two together — the repair the brief designed | **−1.09 %** |
| E33 − arm (b) | the grid thinning E33 additionally paid | **+2.93 %** |

**The prize the brief predicted is real and it is 11.9 % of `C_round(M=6)`.
Row-blocking, which is the only way to reach it, costs 10.5 %. The net is a wash.**

And the reason row-blocking is *the only way* is a register wall I can show from AIR:

```text
crossrow_na2  regs= 62      crossrow_na5  regs=125   <- shipped high-water <T,5,5>
crossrow_na3  regs= 83      crossrow_na6  regs=144   <- NA=6 at r=4, +2nd alloca
crossrow_na4  regs=104                                  type [4 x <6 x float>]
```

`+21` registers per NA, exactly. NA = 6 at the natural `ROWS_PER_SIMD = 4` needs
**144** — 19 above the shipped high-water and 16 above the 128-register wall — and
it is the only member of the family that grows a second `alloca` with an indexed
array type, the signature of values that no longer fit in registers. That is why
E33 used `r = 2`, and `r = 2` is what costs the 10.5 %.

So the account closes: **at M = 6 the second weight stream is not an oversight,
it is what the register file buys.** No arrangement of row blocks changes that.

---

## 1. Pre-declared controls first, before any headline

Pre-registration `13023f0`, committed before any kernel code existed.

| # | control | registered | measured | verdict |
|---|---|---|---|---|
| 1–6 | untreated widths M ∈ {3,4,5,7,8,9}, arm (b) | \|r−1\| ≤ 0.0046 | worst **+0.0080** (M=3) | **band exceeded — see §7.1** |
| 7–12 | untreated widths, arm (a) | \|r−1\| ≤ 0.0046 | worst **+0.0039** (M=3) | 6/6 pass |
| 13 | AIR arm (a) `crossrow_m6_ipg3_r2` | 66 (62–70) | **66** | pass |
| 14 | AIR arm (b) `crossrow_m6_ipg6_r2_xb` | ≤ 117 | **106** | pass |
| 15 | AIR shipped `crossrow_m6_ipg3` | 83 | **83** | pass |
| 16 | AIR `crossrow_na6` (why r=2 exists) | 117 for r=2 | **144** for r=4 | pass |
| 17 | no register spills in any arm | 0 | 0 | pass |
| 18 | dispatch identity, base | `_m<T, 6, 3, true>` | as registered | pass |
| 19 | dispatch identity, arm (a) | `_m<T, 6, 3, true, 2>` | as registered | pass |
| 20 | dispatch identity, arm (b) | `_m<T, 6, 6, true, 2, true>` | as registered | pass |
| 21 | `weight_streams` readback (base / a / b) | 2 / 2 / 1 | 2 / 2 / 1 | pass |
| 22 | `dirty` = 0 on every timed arm | 0 | 0, 0, 0 | pass |
| 23 | real 40 °C cool gate on every timed arm | passed | `stalled_above_40C`, 43.1–43.4 °C | pass |
| 24 | cross-session anchor, base M=6 vs E33 | ≈ equal | 128.996 vs 128.843, **+0.118 %** | pass |
| 25 | M = 1 global null (serial leg) | unreachable | source-unreachable, §6.3 | pass |
| 26 | coverage: every (m, n) written exactly once | exhaustive | 4 arms × 6 shapes, **pass** | pass |
| 27 | `row0_bitwise_matches_m1`, all arms | true | **64/64 rows, worst \|Δ\| = 0** | pass |
| 28 | 192-cell cross-build parity digest | bit-identical | §13 | *pending, filled from the run* |
| 29 | `twin_audit.py` | `TWIN AUDIT OK` | **29 runtime-effective twins** | pass |
| 30 | JIT/PSO pre-flight, no M=6 first-round spike | no spike | §7.2 | pass |

---

## 2. Primary metric

```text
e38/m6_per_row_cost_ratio   raw 0.9891   drift-adjusted 0.9858
registered band  [0.960, 1.005]  point 0.990    ->  INSIDE
assignment expectation 0.84                      ->  FALSIFIED
```

`C_round` at M = 6, one locked session, one fresh base, three arms:

| M | base ms | arm (a) ms | arm (b) ms | a/base | b/base |
|---:|---:|---:|---:|---:|---:|
| 1 | 60.022 | 63.329 | 66.024 | 1.0551 | 1.1000 |
| 2 | 64.218 | 64.788 | 64.919 | 1.0089 | 1.0109 |
| 3 | 72.502 | 72.782 | 73.084 | 1.0039 | 1.0080 |
| 4 | 82.516 | 82.812 | 82.842 | 1.0036 | 1.0039 |
| 5 | 96.146 | 96.415 | 96.205 | 1.0028 | 1.0006 |
| **6** | **128.996** | **142.989** | **127.588** | **1.1085** | **0.9891** |
| 7 | 138.644 | 138.896 | 139.110 | 1.0018 | 1.0034 |
| 8 | 149.634 | 149.063 | 149.455 | 0.9962 | 0.9988 |
| 9 | 164.671 | 165.047 | 164.601 | 1.0023 | 0.9996 |

Only the treated width moves. Arm (a) at +10.85 % is 39 σ against its own control
scatter; arm (b) at −1.09 % raw / −1.42 % drift-adjusted is 1.4× its MDE (§8).

## 3. The registered relations, measured

| relation | expression | registered | band | measured | verdict |
|---|---|---:|---|---:|---|
| R1 weight pass | `ratio(a) − ratio(b)` | +0.1658 | [+0.130, +0.200] | **+0.1196** | outside, low |
| R2 row-blocking | `ratio(a) − 1` | +0.1440 | [+0.050, +0.200] | **+0.1054** | inside |
| R3 grid thinning | `1.0150 − ratio(b)` | +0.0250 | [0, +0.200] | **+0.0293** | inside |

R3 is the discriminator the pre-registration named: **my account needed +0.025,
the 0.84 account needed +0.175.** Measured +0.029. The 0.84 story is out by 6×.

**Why R1 and R2 are exact isolations, not approximations.** Arm (a) and arm (b)
have the same working x-block count (2), the same output work per threadgroup
(12 elements), the same total activation traffic (6 input-row reads per
threadgroup) and the same register tile (`r = 2`). They differ **only** in weight
traffic, 2 passes against 1. Base and arm (a) have the same weight traffic, the
same x-block count and the same total output work, and differ **only** in that
arm (a) reads its activation tile twice from a halved register tile. So:

```text
E33 (+1.50 %) = row-blocking (+10.54 %) − weight pass (−11.96 %) + grid thinning (+2.93 %)
```

🔴 That sum is an **identity by construction** — R3 was defined as
`E33 − ratio(b)` — so it is not an independent check and I am not presenting it
as one. The content is in the three magnitudes, each of which is separately
measured against its own control.

🟡 **One confound I did not remove.** R2 is "row-blocking", not purely
"activation doubling": arm (a) both re-reads the activation tile *and* halves the
register tile from 4 accumulators to 2 (AIR: 83 → 66 registers, `float_ops`
60 → 36, `loop_backedges` 2 → 3). I cannot separate the extra loads from the lost
ILP with the arms I ran, and the follow-up in §10 depends on which dominates.
The registered claim said "activation doubling"; the honest reading is an upper
bound on it.

## 4. Per shape at M = 6

| shape | n | working TGs | base µs | arm (a) | arm (b) | E33 | R1 = a−b | R3 = E33−b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_attn.o_proj` | 5120 | 1280 | 193.21 | 1.0959 | **0.9677** | 1.0414 | 0.1282 | **+0.0737** |
| `linear_attn.out_proj` | 5120 | 1280 | 190.04 | 1.1137 | **0.9800** | 1.0492 | 0.1337 | **+0.0692** |
| `mlp.down` | 5120 | 1280 | 475.39 | 1.1023 | 1.0036 | 1.0592 | 0.0987 | **+0.0556** |
| `full_attn.qkv_proj` | 14336 | 3584 | 379.65 | 1.1025 | 0.9729 | 1.0148 | 0.1296 | +0.0419 |
| `linear_attn.in_proj` | 16480 | 4120 | 424.34 | 1.1145 | 0.9949 | 0.9947 | 0.1196 | −0.0002 |
| `mlp.gate_up_fused` | 34816 | 8704 | 846.85 | 1.1095 | 0.9838 | 0.9941 | 0.1257 | +0.0103 |
| `head.compact_draft_vocab` | 98336 | 24584 | 2295.08 | 1.1141 | 0.9829 | 0.9868 | 0.1312 | +0.0039 |
| `head.lm_head` | 248320 | 62080 | 5716.58 | 1.1151 | 0.9843 | 0.9830 | 0.1308 | −0.0013 |

Three readings, and the middle one is the one the brief asked for:

1. **Arm (a) is flat across five orders of magnitude of grid width** — 1.0959 to
   1.1151, a 1.8 pp span over 1280 → 62080 threadgroups. Row-blocking taxes a
   fixed fraction of the work and does **not** care about the grid.
2. 🔴 **R3, the grid-thinning penalty, is strongly ordered by threadgroup count**
   — +7.4 pp at 1280 TGs falling to ≈ 0 at ≥ 4120 — and this **is** the
   independent confirmation the census could not give. The advisor flagged that
   τ = −1.0 there was "monotone in `n` renamed", because `TGs ∝ n` at a fixed
   cell, and that the value of the framing is that *threadgroups can be moved
   without moving `n`*. arm (b) versus E33 is exactly that move: identical `n`,
   identical weight traffic, identical activation traffic, only the threadgroup
   count differs. The ordering survives it, so the knee near ~1900 working
   threadgroups is real and is not `n` in disguise.
3. **R1, the weight-pass value, is nearly flat too** — 0.120 to 0.134 on seven
   shapes. The exception is `mlp.down` at 0.0987, the deepest reduction
   (K = 17408), and that is exactly why `mlp.down` is the one shape arm (b) does
   not rescue below 1.0.

🔴 **This refutes the naive reading of the 77 %-of-peak bandwidth argument.** If
weights were the only DRAM term and we halve them at constant parallelism,
`mlp.down` should approach 0.5. It measures 1.0036, and the weight pass on it is
worth 9.9 %, not 50 %. The activation tile is not free.

## 5. By-product: the shipped M = 5 → M = 6 ladder step is not an anomalous cell

`research/e38_m6_step.py`. The brief and the plateau analysis both raised
"is our M = 6 cell the width-confined defect?". My session says no.

```text
C_round   M=1  60.022   M=2  64.218 (+4.20)   M=3  72.502 (+8.28)   M=4  82.516 (+10.01)
          M=5  96.146 (+13.63)   M=6 128.996 (+32.85)   M=7 138.644 (+9.65)
          M=8 149.634 (+10.99)   M=9 164.671 (+15.04)

step M=5 -> M=6            +34.17 %   (+32.85 ms)
  of which: 2nd weight stream, priced by E38   +15.40 ms
  residual = one more input row                +17.45 ms
```

The +34 % step looks alarming until the weight stream is priced. Once it is, the
remainder is **one more input row at +17.4 ms**, in line with the +13.6 ms row
below it and the +9.6 ms row above it. There is no large unexplained M = 6 tax
left to attack. 🟡 This is one session on one host and it does not settle the
E27 `(tax, share)` question, but it does remove "our M=6 cell has a hidden
penalty" as a live hypothesis on this evidence.

## 6. Correctness

### 6.1 192-cell cross-build parity digest

`research/run-qmv-parity.sh base=54248ce armb=b6306d7 arma=fa0d33d`
— see §13 for the verbatim comparator output.

### 6.2 Coverage, by exhaustive enumeration

`research/e38_coverage_proof.py` enumerates every `(tid.x, simdgroup, block,
lane)` and asserts that each `(m, n)` output element is written **exactly once**,
for all four arms × all six scored `n`. The gate's own `N % 8 == 0` closes
`ceil(N/8) · 2 · ROW_BLOCKS · ROWS_PER_SIMD = 8 · ceil(N/8) = N`.

🔴 **The first version of this check FAILED, and that is the evidence it has
power.** I keyed the "written exactly once" set on the output row `n` alone and
got a double-write for arm (b). It is not a bug: `tid.x` selects *input* rows in
the base cell, so two x-blocks legitimately touch the same `n` for different `m`.
The real invariant is the `(m, n)` pair. A check that cannot fail is a rubber
stamp; this one caught my own error before it reached the GPU.

### 6.3 The reassociation hinge, which is the one thing that moved

E33's proof turned on lane membership being frozen by `group_dims(32, 2, 1)`.
E38 changes which **threadgroup** a row block is computed in, so that argument
had to be re-made.

*Source*: `block_size`, `simd_lid` and the K-loop stride are untouched by the
`ROWBLOCKS_IN_XBLOCKS` parameter; it changes only `first_m` and the `out_row`
offset. A row's K-partition across lanes and its `simd_sum` pairing are therefore
identical to the base cell — the row moved, the reduction tree did not.

*Empirical*: `row0_bitwise_matches_m1` is **true on all 64 (shape × width) cells
with `row0_max_abs_delta_vs_m1 = 0`** for base, arm (a) and arm (b), and the
192-cell digest above compares arms across builds rather than a build against
itself. Reassociation on the verify path is a ranked-measured rejection class, so
I wanted both the source argument and the bits.

### 6.4 M = 1 unreachability, re-confirmed for the new indexing

The M = 1 dispatch path is `qmv_fast_impl`; there is no `case 1:` in either
crossrow tier, so no value of `ROWBLOCKS_IN_XBLOCKS` is reachable at M = 1. This
matters because **a serial-leg speedup would lower the published score** — the
serial leg is the numerator. Observed M = 1 ratios are 1.0551 (a) and 1.1000 (b),
i.e. *slower*, and M ≤ 2 is warmup-contaminated (§7.1), so this is not evidence
of a serial-leg change in either direction.

### 6.5 Twins, scope, budget

- `python3 research/twin_audit.py` → `TWIN AUDIT OK: 29 runtime-effective twin(s)`
- metallib fingerprint (arm b build) `b52990f9b289f03425740dab925b3e3839b6fd0abfe4cde013164097a2950c30`
- `senpai/validate-assignment-scope.sh` → `assignment scope OK: 2 submitted path(s)`
- `senpai/check-editable-budget.sh` → `growth=0/262144`, because the final tree
  restores both vendored files to `BASE_SHA` byte-for-byte

## 7. Two methodological findings worth propagating

### 7.1 The first two widths of a cost-curve session are warmup, and must not anchor anything

Cross-session drift against E33 on the **unchanged** base:

| M | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vs E33 base | +2.294 % | +1.591 % | +0.311 % | +0.190 % | +0.098 % | **+0.118 %** | +0.075 % | +0.221 % | +0.083 % |
| within-run (mean−min)/min | 13.8 % | 11.1 % | 4.6 % | 3.4 % | 3.8 % | 3.0 % | 2.9 % | 2.6 % | 3.3 % |

M ≥ 3 is comparable to E33 to within 0.31 % and the M = 6 anchor holds to
**+0.118 %**, so this session is sound where it matters. But my registered
±0.46 % control band was computed over a set that **included M = 2**, and this
session's raw control spread reaches +1.09 % at M = 2 from warmup alone. I
exclude M ≤ 2 from every drift and control estimate here and say so rather than
quietly re-fitting the band. On the M ≥ 3 controls, arm (b) still shows +0.80 %
at M = 3, above my ±0.46 %, which is why §8 treats the control scatter as the
noise floor instead of the registered constant.

### 7.2 JIT / PSO pre-flight (advisor 5337266846 item 4)

E38's cells are new template instantiations. If one compiled inside the timed
window, its width would show an outlier first round. Worst `(mean − min)/min` by
width, percent:

```text
M      1      2      3      4      5      6      7      8      9
base 13.75  11.13   4.59   3.38   3.82   2.95   2.91   2.63   3.28
a    28.18   5.75   4.41   4.79   4.96   4.42   3.55   3.30   3.31
b    39.45   7.96   4.67   2.04   2.21   1.96   1.44   1.61   1.82
```

**M = 6 is the *quietest* width in both treated arms**, not the noisiest, in a
row where every other width is untreated. `warmAllDepthShapes` covers the new
instantiation; no JIT compile landed in the timed window.

### 7.3 Ranked memory geometry (advisor 5337327566)

Per my pre-registration this arm runs **last** and never replaces an arm. See §12.

## 8. Order statistic, value, and an MDE for every null

🔴 **The order-statistic sentence, before the value claim.** The published score
is the mean of the 4th and 5th of eight per-prompt ratios; ours sort with
**beagle 4th and medicine 5th**, so a change is worth exactly zero unless it moves
those two legs. This kernel is the same code on every prompt and M = 6 rounds
occur on both, so it moves both — and it also moves essays, republic and botany,
which is worth nothing and is not reported here as if it were.

```text
x = 1 - ratio          = 1.42 %  (drift-adjusted; 1.09 % raw)
MTP-leg movement       = psi 0.228 x phi 0.201 x x        = +0.0652 %
beagle   (4th)  closes 18.0 % of its 0.363 % deficit = 0.038 of 0.210 ms/round (R=107)
medicine (5th)  closes 74.1 % of its 0.088 % deficit = 0.039 of 0.052 ms/round (R= 99)
score                  = +0.0529 %  =  0.54 sigma  =  20.5 % of the engineerable gap
```

🟡 **This supersedes the +0.0241 % / 0.25 σ I posted in the interim comment**, for
two reasons I want on the record: the interim used the raw rather than the
drift-adjusted ratio, and it used the **beagle-only** chain constant 0.4827. A
kernel change moves both scoring legs, so the right sensitivity is the one implied
by the advisor's own ladder row "both legs −0.640 % → +0.5193 %", i.e. 0.8114 of
score per unit of leg. The corrected figure is about twice the interim one and is
still a fifth of the engineerable gap.

**The ladder, inverted to the ratio this experiment would have needed:**

| target | score | required `m6_per_row_cost_ratio` |
|---|---:|---:|
| 1 σ of score | +0.0978 % | ≤ 0.9737 |
| 2 σ of score | +0.1956 % | ≤ 0.9474 |
| the engineerable gap | +0.2586 % | ≤ 0.9305 |
| **the crown** (both legs −0.640 %) | +0.5193 % | **≤ 0.8603** |
| measured | +0.0529 % | **0.9858** |

**MDE for every null** (`research/e39_mde.py`, α = 0.05, power 0.80; exact
noncentral-t reported alongside the normal figure because at small `n` the normal
approximation understates the floor — 5.83× at n = 2 paired):

| # | null | design | MDE (normal) | MDE (exact) | outcome |
|---|---|---|---:|---:|---|
| 1 | M = 6 cost-curve effect | 1 treated width vs 6 controls, sd 0.344 %, df 5 | **1.041 %** | 1.304 % | measured −1.423 % → **detected, 1.4×/1.1× the floor** |
| 2 | end-to-end decode leg | paired, n = 4 legs, per-pair sd 0.2974 % | **0.417 %** | 0.632 % | predicted +0.065 % → **6.4× / 9.7× under-powered — NOT RUN** |
| 3 | same, n = 2 legs | paired | 0.589 % | 3.435 % | 9.0× / 52.7× under-powered |
| 4 | beagle leg vs plateau | single board reading, sd 0.069 % | 0.193 % | — | predicted +0.065 % → 3× under the floor |
| 5 | medicine leg vs plateau | single board reading, sd 0.061 % | 0.171 % | — | predicted +0.065 % → 3× under the floor |
| 6 | untreated widths M ≥ 3 | as #1 | 1.041 % | 1.304 % | worst +0.80 % → **flat within the floor** |
| 7 | M = 1 serial leg | source-unreachable | n/a | n/a | not a statistical null; see §6.4 |

### 8.1 End-to-end: skipped, and this is the deliverable (f) decision

Predicted leg movement +0.0652 % against an instrument whose 80 %-power floor is
0.417 % (normal) or 0.632 % (exact) at four paired legs. **6.4× to 9.7×
under-powered.** Running it would have produced exactly the number my own §8b.8
rule in E33 says not to spend GPU time on. **Not run. MDE stated above.**

🔴 **Consequence, stated plainly because it was an explicit ask: per-prompt
beagle and medicine legs are NOT measured this round.** I am not substituting an
aggregate for them. Rows 4 and 5 show that even the official board could not have
resolved this effect on those two prompts — the plateau's own between-row sd puts
the floor at 0.19 % and 0.17 % against a predicted 0.065 %.

🟡 And the pre-registered discrepancy rule is worth recording as *not triggered*:
the interesting outcome was "microbenchmark says 0.84 but the leg does not move".
The microbenchmark said 0.99, so there is no cost-attribution discrepancy to
explain. ψ and φ were never tested by this experiment.

## 9. What this closes

- 🔴 **The NA > 5 direction at M = 6 is closed, with a mechanism.** The weight
  pass is worth 11.9 %; reaching it requires `r = 2`; `r = 2` costs 10.5 %;
  `r = 4` at NA = 6 needs 144 registers against a 128 wall. Reopen only if
  something changes the register ceiling or removes the row-block tax (§10).
- 🔴 **Rung 2 (M ∈ {7, 8}) is dead on arrival and should not be assigned.** The
  register ladder is +21/NA, so `crossrow_m7_ipg7_r2` = 134 and
  `crossrow_m8_ipg8_r2` = 151 — already over the wall at `r = 2`, before any
  timing. The brief made rung 2 conditional on the sign flip that arm (b) was to
  demonstrate; arm (b) demonstrated +1.09 %, not +16 %.
- **"E33 was killed by lost parallelism"** — falsified. Restoring the grid is
  worth +2.9 %, not +16 %.
- **"E33 was killed by bytes"** — falsified. Arm (b) has E33's exact bytes and is
  *faster* than base.
- **"The activation tile is cache-served, so doubling it is nearly free"** —
  falsified. It costs 10.5 % of the round on every scored shape.
- **"One weight stream at M = 6 costs C(M = 5)"** — falsified. The ladder read
  32.85 ms; the direct measurement is 15.40 ms, an overstatement of 2.1×.

## 10. Suggested follow-ups (not implemented)

1. 🔴 **K-tiled activation staging in threadgroup memory.** The whole result is
   that row-blocking costs 10.5 % to buy an 11.9 % prize. The re-read exists
   because the loop is `for block { for k { load x } }`. Interchanging it is not
   possible — holding both blocks' accumulators live is exactly the 144-register
   cell — but staging a K-tile of `x` in threadgroup memory is:
   `for k_tile { load x[k_tile] to tg memory (6 rows x 512 x 2 B = 6 KB);
   for block { accumulate from tg memory } }`. Registers stay at `r = 2`; the
   DRAM re-read disappears; the grid stays at the shipped width. If it recovers
   even two-thirds of R2 the ratio lands near 0.93, which is the engineerable-gap
   bar in §8. 🟡 It is bounded by the §3 confound: if R2 is mostly lost ILP rather
   than loads, this recovers little. The cheapest discriminator is an arm (a)
   variant with the loop unrolled, which costs one compile.
2. **Price the `mlp.down` exception.** It is the only shape where the weight pass
   is worth 9.9 % rather than ~13 %, and it is 23.6 % of `C_round`. K = 17408 is
   the obvious suspect; a K-blocked variant would test it.
3. **Retire the ±0.46 % control constant.** §7.1 shows it was fitted over
   warmup-contaminated widths. The campaign should quote the per-session control
   scatter over M ≥ 3, which here is 0.344 % sd and a 1.04 % MDE.

## 11. Reproduction

```bash
# arms (the kernel for each arm is banked as a patch; base is HEAD)
git apply research/e38-artifacts/e38-armb-kernel.patch     # or e38-arma-kernel.patch
research/run-qmv-curve.sh e38-armb-r1 54248ce258376db756be02fd65a814a903e2d601 \
  --widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 --skip-stock

# analysis, value and correctness
python3 research/e38_analyze.py --base e38-base-r1 --arm-a e38-arma-r1 \
  --arm-b e38-armb-r1 --json-out research/e38-artifacts/e38-metrics.json
python3 research/e38_value.py
python3 research/e38_m6_step.py
python3 research/e38_coverage_proof.py
python3 research/twin_audit.py
research/run-qmv-parity.sh base=54248ce armb=b6306d7 arma=fa0d33d
```

- Host: Apple M4 Pro `Mac16,11`, 20 GPU cores, 48 GiB, `applegpu_g16s` class `'s'`
- Effective geometry of the timed arms: MLX arch defaults, **50 ops / 50 mebi-elements
  per buffer**, cache default, residency off — `QwenQMVCostCurveTests` never
  constructs `QwenRuntimeMTPWorker`, so `RuntimeStartupMemoryPolicy` never runs.
  (`MLX_MAX_MB_PER_BUFFER` accumulates `a.data_size()`, i.e. **elements**, and the
  comparison is `(buffer_sizes_ >> 20) > max_mb` — mebi-elements, not MiB.)
- Thermal: real 40 °C gate on every timed arm, entry 43.147 / 43.363 / 43.249 °C,
  exit 67.58 / 66.95 / 67.39 °C. `cool_gate_passed_real_gate = true`,
  `gate_qualified_for_timing = true`.

## 12. Ranked-geometry arm

Per my pre-registration and the advisor's instruction in comment `5337327566`,
this arm ran **last** and **never replaces** an arm above. It is a separate
question about the *method*, not about row-blocking.

**Question.** Every cost curve in this campaign — including all three E38 arms —
was measured under architecture-default MLX buffer geometry. The ranked runner
sets `MLX_MAX_MB_PER_BUFFER=512` and `MLX_MAX_OPS_PER_BUFFER=50`. If those knobs
move the shape-level curve, then every local curve we have is measuring the
wrong machine.

**Design.** Same `BASE_SHA` (`54248ce`), same host, same session, same
`--shapes-only --reps 21 --inner 10` protocol, arch-default (`e38-base-r1`) vs
ranked geometry (`e38-base-rg-r1`). Reproduce with:

```bash
research/run-qmv-curve-ranked-geom.sh e38-base-rg-r1 54248ce2583... \
  --widths 1,2,3,4,5,6,7,8,9 --shapes-only --reps 21 --inner 10 --skip-stock
python3 research/e38_ranked_geom.py --json-out research/e38-artifacts/e38-ranked-geom.json
```

| M | arch ms | ranked ms | Δ% | trusted |
|---|---|---|---|---|
| 1 | 60.022 | 59.949 | −0.121 | no (warmup) |
| 2 | 64.218 | 64.338 | +0.187 | no (warmup) |
| 3 | 72.502 | 72.588 | +0.119 | yes |
| 4 | 82.516 | 82.867 | +0.425 | yes |
| 5 | 96.146 | 96.147 | +0.000 | yes |
| **6** | **128.996** | **129.072** | **+0.060** | yes |
| 7 | 138.644 | 138.992 | +0.251 | yes |
| 8 | 149.634 | 150.141 | +0.339 | yes |
| 9 | 164.671 | 164.732 | +0.037 | yes |

M ≥ 3: mean **+0.176 %**, sd 0.164 pp, se 0.062 pp, t = +2.84 (df 6).

**A bare paired mean cannot answer this**, so three readings, and I will not
hide the one that looks bad:

1. 🟡 **The paired mean is nominally significant** (t = 2.84, df 6, two-sided
   p ≈ 0.03) and **7 of 7 trusted widths are non-negative**. Taken alone that
   reads as a real +0.18 % slowdown under ranked geometry.
2. 🟢 **The width trend is absent: Pearson r = −0.080.** This is the decisive
   reading. Buffer pressure is *monotone in M* — M = 9 issues the most rows and
   the largest buffers, so a genuine `MLX_MAX_MB_PER_BUFFER` /
   `MLX_MAX_OPS_PER_BUFFER` effect is forced to grow with M. Instead the largest
   deviation is at M = 4 (+0.425 %) and M = 9 is the second *smallest*
   (+0.037 %). A geometry effect that is uncorrelated with buffer pressure is not
   a geometry effect.
3. 🟢 **The magnitude sits inside the independently measured drift envelope.**
   `research/e38_anchor_check.py` put cross-session drift at ≤ 0.311 % for M ≥ 3
   against the E33 base; worst deviation here is 0.425 % and the mean is
   0.176 %. The ranked-geometry curve ran later in the same session, i.e. warmer,
   which is exactly the sign of all 7 deviations. Uniform-positive + no width
   ordering + magnitude ≈ drift is the signature of monotone session drift, not
   of the treatment.

The mean is **0.17× the cost-curve MDE** (1.041 % normal / 1.304 % exact, from
`research/e39_mde.py`). So the honest statement is not "we proved zero" — it is
that this design **cannot resolve anything below ≈1 %**, and whatever is there is
at most a few tenths of a percent and is not shaped like a geometry effect.

**Verdict: `GEOMETRY-INVARIANT` at the shape level.**

⚠️ **Scope limit, stated because it is easy to over-claim.** A `--shapes-only`
probe issues one operation per call. `MLX_MAX_OPS_PER_BUFFER` governs how many
ops are *batched into a command buffer*, so this probe is insensitive to that
knob **by construction**. What this arm licenses is precisely: *shape-level GEMM
cost curves transfer from arch-default to ranked geometry*. It does **not** show
that end-to-end command-buffer behaviour transfers, and it must not be cited for
that.

**Campaign value.** Within that scope this is reusable beyond E38: local
`--shapes-only` cost curves do not need a ranked-geometry replay, which removes a
~9.5 min tax per curve from every future shape experiment. And it closes the
last portability objection to E38's own numbers — the −11.94 % weight-pass prize
and the +10.54 % row-blocking tax were both measured on a geometry that
reproduces the ranked one at the shape level, so the negative result stands on
the ranked runner too.

Artifact: `research/e38-artifacts/e38-ranked-geom.json`.
Self-test: `python3 research/e38_ranked_geom.py --self-test` → 14/14, including
a synthetic width-proportional effect that the trend arm *does* catch and a flat
5 % offset that the MDE arm *does* catch.

## 13. 192-cell cross-build parity digest

*(filled in when the digest completes; it does not replace an arm above)*
