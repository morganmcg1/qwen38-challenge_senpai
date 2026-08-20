# E73 — the QMV group partition as an occupancy-versus-bandwidth trade

Student: qwen-alphonse. PR #76. Base `8a5f73b55dea4d1c3d995a300e9462c04424a964`.
Host: Apple M4 Pro, **20 GPU cores read from the device** (`ioreg gpu-core-count`,
confirmed by `system_profiler`), macOS 26.5.2. `harness=local`.
Ranked host for the transfer statement: M5 Max, `applegpu_g17s`, **40 GPU cores**
(`senpai/campaign-ledger.md:16923`).

W&B: rung 1 `rwjal2ws`, rung 2 `628e3olc`, project
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`.

**Rungs 0 to 2 changed zero candidate files.** Everything below lives in `research/`.

```
Question:
  What is the exchange rate between weight streams and per-simdgroup live
  state in the QMV cross-row group partition, and does a cost model built
  from it predict the ranked-optimal IPG?

Evidence:
  Our three merged partition changes are exactly the three widths where we cut
  concurrent threadgroups relative to the crown, and those widths carry 63.25 %
  of ranked verify-width time. Ledger item 137 and E71 both order the per-byte
  cost by working threadgroups with Kendall tau = -1.0.

Expected result:
  A model whose residual approaches the session null, which reproduces the
  measured local optimum, and which names a ranked-optimal partition table.

Smallest decisive test:
  One gated session that measures every legal (M, IPG) pair on every scored
  shape, then two positive controls before any prediction.

Stop or promotion rule:
  Rung 3 only if both positive controls pass. If either fails, stop at rung 2
  and report.
```

**Outcome: control 1 passes, control 2 fails. I stopped at rung 2. No rung 3.**

## 1. Rung 0 — static census (complete, reported in the PR)

All five pre-registered predictions hit. The pre-registration commit `f361e68`
landed before the first compile. Summary of what rung 2 uses:

- spill set is exactly IPG=6 (`m6_ipg6`, `m8_ipg6`, `m9_ipg6`), one
  `[4 x <6 x float>]` alloca of 96 bytes;
- `peak_live = 16*IPG + 50` exactly at IPG 2, 3, 4 and 5, then **182** at IPG=6
  where the trend predicts 146 — the spill adds pressure instead of relieving it;
- derived resident simdgroups per core `S(IPG)` = 37, 31, 26, 23, 16 for
  IPG 2..6, assuming a 384 KiB register file. **Derived, not measured.**

## 2. Rung 1 — the measured surface (complete)

One gated session, 19 arms x 6 measured shapes = 114 cells, 11 reps, palindrome
order, frozen grid `dispatchThreadgroups(M, n/8, 1)` with `group_dims(32, 2, 1)`.
`cool_gate_passed_real_gate=true`, `gate_qualified_for_timing=true`.
Parity: 0 differing elements and 0 empty rows on every arm at every shape.
Session temperature 35.9 C to 65.5 C.

`fa.o_proj` and `gdn.out_proj` are the same (n=5120, k=6144) cell, so 6 measured
shapes cover the 7 scored linear shapes.

**Session null** (same arm at its two palindrome positions, 114 cells):
median **0.0238 %**, p90 0.0717 %, max 0.2836 %.

## 3. Rung 2 — the fit

### 3.1 The brief's original additive model is refuted

`t = a*groups*W + b(IPG) + c(shape)`, 11 free parameters:

| model | params | rel-rms | median | max | x session null |
|---|---:|---:|---:|---:|---:|
| A additive (original brief) | 11 | **10.17 %** | 6.77 % | 28.87 % | 426.5 |
| B0 occupancy, S-pinned, q=1 | 3 | 4.94 % | 3.26 % | 22.58 % | 207.2 |
| B occupancy, S-pinned | 7 | 2.29 % | 1.04 % | 13.98 % | 96.2 |
| **B2 occupancy, lam(IPG) free** | 11 | **1.62 %** | 1.06 % | 6.46 % | **68.0** |

A three-parameter occupancy model beats the eleven-parameter additive model by
2.1x. The respecification was correct and it was worth the retraction.

Model A's fitted exchange rate, kept for the record: `a = 3.180 ps/byte`
(1/a = 314.5 GB/s) and `b(IPG)` = 0.00, -2.81, +18.47, +45.29, +81.81 us for
IPG 2, 3, 4, 5, 6. Those two numbers were the requested deliverable, but the
model that produces them is wrong, so they should not be priced.

### 3.2 The model the data supports

```
t(M, IPG, shape) = [ groups*W(shape) + beta*M*k*Tn ] * rho0 * q(IPG) * (1 + lam(IPG)/x)
x  = groups * Tn / cores          working threadgroups per core
Tn = ceil(n/8)
groups = ceil(M/IPG)
```

Fitted on 114 cells at `cores = 20`:

| term | IPG 2 | IPG 3 | IPG 4 | IPG 5 | IPG 6 |
|---|---:|---:|---:|---:|---:|
| `q(IPG)` (rate level, IPG 3 = 1) | 1.0833 | 1.0000 | 0.9713 | 0.9768 | 1.0460 |
| saturated rate `1/(rho0*q)` GB/s | 610.2 | 661.1 | 680.6 | 676.8 | 632.0 |
| `lam(IPG)` working TGs/core | 8.16 | 8.55 | 7.74 | 7.20 | **10.09** |
| 10 %-deficit knee, TGs/core | 81.6 | 85.5 | 77.4 | 72.0 | **100.9** |
| 10 %-deficit knee, TGs at 20 cores | 1632 | 1711 | 1548 | 1440 | **2018** |

`rho0 = 1.5127 ps/byte`, `beta = 2.4698`.

**This is the exchange rate.** Read it as three separate prices:

1. **A weight stream costs `rho0 * W` seconds** at saturation, about
   1.51 ps per byte, so 661 GB/s at IPG=3. That exceeds the M4 Pro DRAM roof, so
   the isolated cell is partly cache-served and this is an effective rate, not a
   DRAM rate.
2. **Live state costs almost nothing at saturation.** `q` moves by only 3 % from
   IPG 3 to IPG 5, and the IPG=6 spill costs 4.6 %. The register cliff that rung 0
   found is real but small when the machine is fed.
3. **Live state is paid through occupancy instead.** `lam` is the number of
   working threadgroups per core at which the per-byte rate doubles. IPG=6 needs
   40 % more grid than IPG=5 to reach the same rate. That is the whole trade.

### 3.3 The rung-0 register census is the wrong occupancy scale

Model B pins `lam(IPG) = l0 * S(IPG)` with the derived resident simdgroups from
rung 0. That form is **refuted**: at fixed M and fixed `groups`, the measured
sensitivity to n *rises* with IPG, while `S(IPG)` falls with IPG.

| contrast, same M and same groups | n-sensitivity ratio lm_head/out_proj |
|---|---|
| M=6, IPG 3 vs 4 | 1.139 vs 1.146 |
| M=7, IPG 4 vs 5 | 1.131 vs 1.140 |
| M=8, IPG 4 vs 5 vs 6 | 1.124 vs 1.127 vs 1.149 |
| M=9, IPG 5 vs 6 | 1.113 vs 1.129 |

Freeing `lam(IPG)` drops the residual from 2.29 % to 1.62 % and fixes the M=7 and
M=8 rows of control 1. The apparent fall in n-sensitivity across the
`m3i3, m4i4, m5i5, m6i6` series is driven by M, not by IPG.

### 3.4 The residual never reaches the session null

The best model sits at **68x the session null**. No model in this family is an
adequate instrument at the 0.02 % scale of the measurement. Median residuals by
IPG for B2: 0.39, 1.24, 0.82, 0.63, 1.79 % for IPG 2, 3, 4, 5, 6. The IPG=6
column is worst, which is where the spill lives.

## 4. Control 1 — reproduce the measured local optimum. PASS

Round-weighted over the seven scored shapes with their calls per verify round.

| M | shipped | measured (margin over 2nd) | A additive | B0 | B | **B2** |
|---|---|---|---|---|---|---|
| 3 | 3 | 3 (only legal) | 3 | 3 | 3 | 3 |
| 4 | 4 | 4 (40.11 %) | 4 | 4 | 4 | 4 |
| 5 | 5 | 5 (27.10 %) | 5 | 5 | 5 | 5 |
| 6 | 6 | 6 (3.10 %) | 6 | 6 | 6 | 6 |
| 7 | 4 | 4 (0.76 %) | 4 | 5 | 5 | 4 |
| 8 | 4 | 4 (1.19 %) | 4 | 6 | 5 | 4 |
| 9 | 5 | 5 (8.14 %) | 5 | 6 | 5 | 5 |

**The measured surface reproduces the shipped table 7 of 7 with no model at all.**
B2 also reproduces it 7 of 7. B and B0 fail at M=7 and M=8.

Note the margins. At M=7 and M=8 the decision is worth 0.76 % and 1.19 %, which
is below B2's own median residual of 1.06 %. B2 gets those rows right, but it
cannot claim to resolve them.

## 5. Control 2 — reproduce the E33 sign flip. FAIL

### 5.1 What rung 1 measured: the same contrast without E33's confound

E33's arm changed two things at once: the group partition `<T,6,3>` to `<T,6,6>`,
**and** a split of the 4 output rows into 2 sequential row blocks that re-read x.
Rung 1 measured the group change alone.

| shape | base working TGs | measured ratio `<T,6,6>` / `<T,6,3>` |
|---|---:|---:|
| gdn.out_proj / fa.o_proj | 1280 | 0.9709 |
| mlp.down | 1280 | **1.0921** |
| fa.qkv | 3584 | 0.9370 |
| gdn.in_proj | 4120 | 0.9340 |
| mlp.gate_up | 8704 | 0.9255 |
| lm_head | 62080 | 0.9186 |

**Kendall tau vs working threadgroups = -1.000**, 0 concordant and 14 discordant.
`mlp.down` is the single loss cell, and it is the same cell E33 found worst
(1.0592). Two instruments, three months apart, agree on the ordering and on the
worst cell.

### 5.2 What the fitted model predicts for E33's arm

Applying the fitted `r` to E33's arm as described, with zero new parameters:

| shape | base TGs | E33 observed | model | error |
|---|---:|---:|---:|---:|
| lm_head | 62080 | 0.9830 | 1.5048 | +53.1 % |
| compact_draft_vocab | 24584 | 0.9868 | 1.5132 | +53.4 % |
| mlp.gate_up | 8704 | 0.9941 | 1.5384 | +54.8 % |
| gdn.in_proj | 4120 | 0.9947 | 1.5804 | +58.9 % |
| fa.qkv | 3584 | 1.0148 | 1.5920 | +56.9 % |
| fa.o_proj | 1280 | 1.0414 | 1.7394 | +67.0 % |
| gdn.out_proj | 1280 | 1.0492 | 1.7394 | +65.8 % |
| mlp.down | 1280 | 1.0592 | 1.7394 | +64.2 % |

The predicted ordering is exact (tau = -1.000, matching the observed tau of
-1.000), but every predicted ratio is a loss, so **the model does not place the
sign flip between 1792 and 2060 working threadgroups. Control 2 fails.**

### 5.3 Why it fails, and it is not a fitting problem

The failure is a traffic-accounting failure and it is robust to interpretation.

- Charging E33's doubled activation re-read at the fitted rate — the ledger's
  1.3571 total-traffic ratio — predicts 1.50 to 1.74. Too high.
- Charging the re-read at zero, which the ledger's own roofline supports because
  the 209 KB x tile is cache-served, reduces the arm to the pure group change.
  The model then predicts 0.85 to 0.98 against measured 0.92 to 1.09
  (errors -9.9 % to +1.4 %, tau = -1.000). Too low.

**E33's observation sits between the two accountings.** Neither the halved weight
pass nor the doubled activation read was realised at the fitted rate. The residual
is E33's row-block serialisation, which rung 1 never ran and which E38 was
designed to isolate. My session cannot separate it.

### 5.4 The physical quantity behind the sign flip *is* reproduced

E33 located the knee at about **95 working threadgroups per core**, near 1900
working threadgroups on 20 cores. My fitted `lam(IPG)` puts the 10 %-deficit knee
at **72 to 101 working threadgroups per core**, 1440 to 2018 on 20 cores.

Two independent instruments, different methods, agree on the knee within 15 %.
So the model captures the mechanism and the knee, and fails only on the level of
one confounded historical arm.

## 6. Cross-check against askeladd's in-situ E71 census

At M=6 on the shipped partition, relative to lm_head:

| family | working TGs | E71 in-situ ms/GB, relative | model, relative |
|---|---:|---:|---:|
| lm_head | 31040 | 1.000 | 1.000 |
| mlp_gate_up | 4352 | 1.088 | 1.040 |
| gdn_out_proj | 640 | 1.303 | 1.307 |
| fa_o_proj | 640 | 1.369 | 1.307 |
| mlp_down | 640 | 1.757 | 1.307 |

**Kendall tau between observed and predicted = +1.000.** The model matches the
`gdn_out_proj` cell to 0.3 %, but it predicts one value for all three 640-TG cells
because the occupancy term depends only on working threadgroups. E71 measures
`mlp_down` 34 % worse than the other two at identical working threadgroups.
That gap is a real k-dependence (k=17408 against k=6144) that my `beta*M*k*Tn`
work term underweights. It is the largest single miss in the model.

## 7. The ranked prediction, and why I do not trust it

### 7.1 One analytical result that constrains H210

`argmin over IPG` is invariant to any pure rescaling of the per-byte rate.
A bandwidth-headroom difference between hosts multiplies every candidate by the
same factor and cannot move a single row. **In this model family, "the ranked host
runs at 44 % of its roof instead of 98 %" cannot by itself flip a partition.**
Only a term that changes the *shape* of the cost over IPG can, and under the
advisor's transfer statement the only such term is `cores`.

### 7.2 The prediction under the cores-only transfer

| M | legal | local argmin | ranked argmin (40 cores) | crown | ranked margin over 2nd |
|---|---|---|---|---|---|
| 3 | 3 | 3 | 3 | 3 | only legal |
| 4 | 2, 4 | 4 | 4 | 4 | 34.44 % |
| 5 | 3, 5 | 5 | **5** | **3** | 21.13 % |
| 6 | 2, 3, 4, 6 | 6 | **4** | **3** | 0.66 % |
| 7 | 4, 5 | 4 | **5** | **4** | 0.10 % |
| 8 | 2..6 | 4 | **5** | **4** | 0.10 % |
| 9 | 3, 5, 6 | 5 | **5** | **3** | 10.90 % |

**Agreement with the crown: 2 of 7.** Three of the five ranked rows that differ
from local carry a margin of 0.10 to 0.66 %, which is below the model's own median
residual of 1.06 %. Those rows are unresolved, not predicted.

### 7.3 How much occupancy pressure the crown's table would need

Smallest core count at which the crown's IPG becomes the round-weighted argmin:

| M | crown IPG | cores required | ranked M5 Max |
|---|---|---|---|
| 3 | 3 | already optimal at 20 | 40 |
| 4 | 4 | already optimal at 20 | 40 |
| 5 | 3 | **272** (13.6x local) | 40 |
| 6 | 3 | **never**, searched to 100000 | 40 |
| 7 | 4 | already optimal at 20 | 40 |
| 8 | 4 | already optimal at 20 | 40 |
| 9 | 3 | **992** (49.6x local) | 40 |

This is the sharpest thing rung 2 produced. Under the advisor's own transfer
statement — `cores` is the only host term — **doubling the core count from 20 to
40 comes nowhere near justifying the crown's table.** At M=6 no core count does,
because IPG=3 loses to IPG=4 on the level term `q`, not on occupancy, and
occupancy cannot overturn a level difference at any grid size.

So one of these must be true, and rung 2 cannot choose between them:

1. `cores` is not the only host term. `S(IPG)`, and therefore `lam(IPG)`, is a
   property of the register file, and the M5 Max register file is not the M4 Pro's.
   A different `lam` profile is the one input that would move M=6.
2. The crown's table is not the ranked optimum. It may be a table tuned on other
   hardware, or tuned against a different objective, and our `ff73cbbd` regression
   may come from a different part of the delta.
3. My isolated-cell surface does not transfer. It compresses the in-situ spread
   (section 6) and it misses the k-dependence that makes `mlp.down` the worst cell,
   and `mlp.down` is where askeladd puts 65.9 % of the M=6 tax.

### 7.4 The two-dimensional deliverable

The kernel already selects on `out_vec_size` at `quantized.h:1917`, so the table
can be a function of both. Optimal IPG per (M, shape band), M = 3..9:

| band | local (20 cores) | ranked (40 cores) |
|---|---|---|
| n=5120, k=6144 gdn.out_proj / fa.o_proj | 3, 4, 5, 4, 5, 5, 5 | 3, 4, 5, 4, 5, 5, 5 |
| n=5120, k=17408 mlp.down | 3, 4, 5, 4, 5, 5, 5 | 3, 4, 5, 4, 5, 5, 5 |
| n=14336 fa.qkv | 3, 4, 5, 6, 4, 4, 5 | 3, 4, 5, 6, 4, 4, 5 |
| n=16480 gdn.in_proj | 3, 4, 5, 6, 4, 4, 5 | 3, 4, 5, 6, 4, 4, 5 |
| n=34816 mlp.gate_up | 3, 4, 5, 6, 4, 4, 5 | 3, 4, 5, 6, 4, 4, 5 |
| n=248320 lm_head | 3, 4, 5, 6, 4, 4, 5 | 3, 4, 5, 6, 4, 4, 5 |

The only band that ever disagrees with the round-weighted table is n=5120 at M=6,
where the model prefers IPG=4 over IPG=6. **I am not proposing this as a candidate.**
Control 2 failed, so rung 3 does not start, and the M=6 margin is inside the
model's residual.

## 8. Did H210 survive?

**Partly, and the part that survived is not the part that predicts.**

- **Survived.** The partition is an occupancy trade, not a traffic trade. A
  three-parameter occupancy model beats an eleven-parameter traffic-plus-state
  model by 2.1x. The fitted knee reproduces E33's independently located knee within
  15 %. The IPG-by-shape interaction the advisor demanded is real and large. Take
  the ratio of the achieved per-byte rate at lm_head to the rate at
  gdn.out_proj — a 48x span in working threadgroups — for each of the 19 arms.
  The four `groups` bands do not overlap: `groups=1` gives 1.204 to 1.291 (4 arms),
  `groups=2` gives 1.112 to 1.150 (11 arms), `groups=3` gives 1.087 to 1.097
  (3 arms), `groups=4` gives 1.074 (1 arm). Working threadgroups, not `n`, set the
  size of the shape effect, and `groups` sets working threadgroups.
- **Did not survive.** The claim that the *same edit inverts sign between hosts*
  because the ranked host has bandwidth headroom. Bandwidth headroom is a pure
  rescaling and cannot flip an argmin. Under the cores-only transfer, 40 cores
  does not reach the crown's table at any width, and M=6 is unreachable at any
  core count.
- **Untested.** Whether `lam(IPG)` differs on `applegpu_g17s`. That is the one
  parameter that could still make H210 right, and it needs an M5 measurement, not
  a local one.

## 9. Reproduction

```bash
# rung 0, compile probe, no GPU
python3 research/e73_air_census.py --out research/e73-artifacts/rung0-air.json

# rung 1, one gated session (already run: W&B rwjal2ws)
research/e73_rung1.sh --reps 11 --target-bytes 12e9 --tag -s1

# rung 2, analysis only, no GPU
python3 research/e73_explore.py
python3 research/e73_fit.py --cores 20 --ranked-cores 40 \
        --wandb-name e73-rung2-cost-model     # -> research/e73-artifacts/rung2.json
```

Gates on this head: `check-editable-budget.sh 8a5f73b5` OK
(`source=2464949/3000000`, `growth=0/262144`); `verify-ranked-score-boundary.sh`
PASS; `validate-assignment-scope.sh` reports every E73 file outside
`editablePaths`, which is the intent.

## 10. Suggested follow-ups, not implemented

1. **Measure `lam(IPG)` on the ranked host.** It is the only untested input that
   could still make H210 right, and rung 1 is a self-contained compile-and-time
   probe that needs no model weights. If a ranked-side arch probe can run the same
   19 arms, the transfer statement becomes a measurement instead of a substitution.
2. **Give the work term a real k-dependence.** The model predicts one value for
   all three 640-TG cells while E71 measures a 34 % spread, and the worst cell is
   `mlp.down` at k=17408, which owns 65.9 % of the M=6 tax. A `k`-aware work term
   is the single largest available residual reduction.
3. **Run E38's arm.** It is still the clean discriminator between grid thinning and
   the activation re-read, and my rung 1 now brackets its expected value: the group
   change alone is worth 0.92 to 1.09 by shape, so E38 should land near 0.93 at
   large n and near 1.05 at n=5120.
4. **Re-examine the `out_vec_size >= 4096` gate rather than the IPG table.**
   `mlp.down` at n=5120 is the smallest scored shape above that gate and is the
   worst cell in three independent instruments. The gate boundary, not the
   partition, may be the cheaper lever.
