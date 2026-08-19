# E48 — Does a uniform QMV speedup lower the score?

Assignment `qwen38-r1-e48-score-weighted-qmv-and-uniform-sign` (PR #52, revision `r1`).
Base `fb0a09d3912477d94ed631bdb90fd04172d7b4cf`. Host: local M4 Pro (**not** the
ranked M5). Nothing here is gate-qualified, ranked-equivalent, or an official score.

> Every timed leg ran with `MLXFAST_LOCAL_COOL_GATE=0`.
> `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
> `official_or_ranked_score=false`.

---

## Part 2 — `psi_mtp` by direct dose injection

### The sign question was withdrawn by the advisor, not answered by me

Part 2 was assigned as "does a uniform QMV speedup lower the score?". The advisor
withdrew that question mid-experiment (PR #52 comment 5342984599) on the grounds
that the ranked baseline leg is a **separately built pinned binary**, so candidate
source changes have zero leverage on `serial`. I record that provenance explicitly:
**the sign question was retired by the advisor's reasoning, not settled by my
measurement.** Arms U and S below are local-only internal consistency checks with
no ranked meaning.

I did independently verify the withdrawal against the enforcing sources rather
than taking it on trust:

| claim | enforcing evidence |
|---|---|
| baseline is a pre-built pinned tree | `.github/workflows/qwen-mtp-ranked-benchmark.yml:224` `MLXFAST_QWEN_MTP_BASELINE_WS: /opt/bench-runner/baseline/qwen3.8-27b-mtp-v1/current` |
| it is never built from candidate source | pre-built existence check `:2926-2928`; no build step writes into that tree anywhere in the workflow |
| the two legs are separate binaries | `--baseline` / `--candidate` `:2970-2971` |
| the score uses those two legs | `baseline_serial_… / candidate_mtp_…` `:3080` |
| candidate source cannot reach the baseline tree | `benchmark.json` `editablePaths` has 89 entries, zero under `.github`, and does not include `benchmark.sh` |

So `d ln(serial)/dx = 0` on the ranked path, uniform ≡ gated ≡ `+psi_mtp`, and the
"free gate" in ledger 173(B) buys nothing. No contradiction with edward's mechanism.

### What remains, and what it is worth

With the sign gone, the one ranked-relevant quantity in the design is the
**candidate-leg elasticity** `psi_mtp = -d ln(mtp_s_per_tok) / d ln(qmv_speed)`:
the fraction of candidate decode time that a QMV kernel win actually converts.
Every dScore in the ledger is implicitly divided by it. Arm G measures it directly
by injecting a *known* slowdown dose into QMV and reading the response.

### Design

`research/e48_perturb.py` injects a calibrated redundant-work dose into
`qmv_fast_impl` with **independent** knobs for the crossrow (`M>=2`) and width-1
paths, so the two elasticity components can be separated instead of confounded:

| arm | crossrow dose | width-1 dose | purpose |
|---|---|---|---|
| `base` | 0 | 0 | reference |
| `base2` | 0 | 0 | **null arm** — pristine byte-identical rebuild, measures run-to-run spread |
| `g1` | 1 | 0 | Arm G dose 1 → `psi_mtp` |
| `g2` | 2 | 0 | Arm G dose 2 → functional-form residual |
| `ulo` | 1 | 2 | dose-matched uniform (local only) |

Doses are **measured, not assumed**: every arm runs the in-process QMV cost curve
in the same session as its timed legs, so the realised dose `x = t_dosed/t_undosed`
is read per width from that arm's own dispatch timings rather than extrapolated.

### Timed results

All arms: 2 legs, 512-token `--local-iterate`, same host, same fixture, same head.

| arm | dose (crossrow/w1) | serial s/tok | mtp s/tok | `raw_p` | ser Δ% | mtp Δ% | `raw_p` Δ% |
|---|---|---|---|---|---|---|---|
| `base` | 0/0 | 0.073460 | 0.033438 | 2.196921 | — | — | — |
| `g1` | 1/0 | 0.073470 | 0.054683 | 1.343544 | **+0.013** | +63.539 | **−38.844** |
| `ulo` | 1/2 | 0.121987 | 0.055473 | 2.199035 | +66.060 | +65.900 | **+0.096** |
| `g2` | 2/0 | *(pending)* | | | | | |
| `base2` | 0/0 | *(pending)* | | | | | |

Per-leg `raw_p` spread: `base` 0.105 %, `g1` 0.144 %, `ulo` 0.116 %.

**Fidelity (advisor Risk 4).** `all_tokens_matched=True` in every arm;
`accepted_draft_rate = 0.88752556` identical to 8 s.f. across all arms;
`rounds = 78`; `residual_divergence_count = 0`. The injection is pure cost and
bit-exact — it does not perturb the trajectory, so the dose response is a clean
cost derivative.

### `psi_mtp`

From `g1`: `mtp_frac = 0.635389` at realised `xbar_X = 0.914900` ⇒

**`psi_mtp = 0.694490`**

Gap-corrected variants (see coverage gap below): **0.711260** (2-bit-scaled) and
**0.728030** (4-bit upper). All three are **lower bounds**.

**Advisor Risk 3 answered: `psi_mtp` transfers across the IPG change.** E42
measured 0.6736 on base `04ad6bf1`; this base gives 0.694490, i.e. **+3.1 %**.
The parameter is stable across the intervening dispatch-shape change, so ledger
dScore corrections built on it survive. Direction of the correction: every dScore
in the ledger — including alphonse's merged +11.421 % — is slightly **under**-priced.

**The advisor's own Arm G prediction validates.** `1/(1+0.6736x)` evaluated at the
realised `xbar_X = 0.91490` predicts −38.130 %; measured **−38.844 %** —
**0.71 pp** agreement on a 39-point effect.

### Structural-churn control

`g1`'s serial leg carries the full dose scaffolding compiled in at dose 0 and moved
**+0.0135 %** — inside the null spread. The scaffolding is free, so the dose
attribution is validated on the scored path rather than assumed.

### The finding I did not expect: `psi_serial` was never identifiable

`ulo` and `g1` carry the identical crossrow dose in fully independent builds, which
turns them into an accidental dosimeter-reproducibility test:

| widths 2..9 | value |
|---|---|
| mean abs diff in realised dose | **0.0039** |
| worst (M=2) | **0.01014** |
| `xbar_X` agreement | **0.084 %** |

The crossrow dosimeter is reproducible to sub-percent. But the **width-1 cell at
TRUE dose zero reads `x = 0.06942`** instead of 0 — seven times the worst crossrow
disagreement — and that offset divides straight into `psi_serial`.

**So `psi_mtp` is identified and `psi_serial` never was.** This single fact
retro-explains E42's three-denominator interval, the unstable
`in_proj_fused_qkvzba` cell, why `rho*` landed *between* the two `rho_unif`
candidates, and my own repeated arithmetic slips on the uniform coefficient. The
unpinnable parameter turned out to be exactly the one the advisor's withdrawal
showed has no ranked leverage — the two findings are independent and agree.

### Local uniform coefficient (the withdrawn quantity), reported as an interval

Because of that width-1 offset there are two defensible treatments and I report
both rather than picking one:

| treatment | `psi_serial` | local uniform coefficient | overstatement vs ledger 173(A) |
|---|---|---|---|
| offset left in | 0.7966 | −0.0769 | 2.33× |
| offset subtracted | 0.8694 | −0.1497 | **1.20×** |

`psi_mtp_TOTAL_local = 0.7197` either way, and the denominator-free crossover
`rho* = 1.9952` is unchanged by the treatment. **Disclosure: I revised this number
three times in this experiment** (−0.0265 → −0.0769 → interval). Root cause each
time was quoting hand arithmetic ahead of the in-arm dosimeter; the fix was to make
every reported number come from analyzer output over measured doses. The interval
above is that output.

### `psi_mtp_w1` by differencing, and why it is a floor

Differencing `ulo` against `g1` isolates the width-1 component:
`mtp_frac` difference 0.023616 / realised `x1 = 0.829258` ⇒
**`psi_mtp_w1 = 0.028478`**, 31 % below E42's arithmetic 0.0415.

It covers **only** the 4-bit `qmv_fast_impl` width-1 path and **excludes the 2-bit
`:1908` readout entirely**, so it is a floor, not an estimate.

### Coverage gap (verified still open)

`qmv_fast_singlerow_affine2_g64` is instantiated only as `<T>` and is therefore
**undosed in every arm**. Untreated candidate-leg share is 5.00 % (4-bit upper) or
2.56 % (2-bit scaled); the additive correction to `psi_mtp` is +0.0353 / +0.0177.
The serial leg has no such gap. This is why every `psi_mtp` figure above is a
lower bound.

Separately: `QwenQMVCostCurveTests.swift:396`'s `in_proj_fused_qkvzba` shape is a
**harness fiction** — GDN `a`/`b` are separate `linear` calls
(`Qwen35GatedDelta.swift:254-255`, `Qwen35FastEngine.swift:495-520`), so no fused
QKVZBA dispatch exists on the scored path. That explains the unstable cell.

### Portability caveat found while checking dispatch

`vector_limit` is **architecture-generation dependent** (`quantized.cpp:84-124`):
gen 13/14 with `D > 4096` gives 6, not 10. This host is `applegpu_g16s` (gen 16,
limit 10). **M5's `arch_gen` is unverified.** If M5 is gen 13/14, widths 7–9 never
reach the crossrow path on the scored host at all, which would change every
per-width share in Part 1 and in ledger 173(C). Flagged, not chased.

---

## Part 1 — score-weighted per-width cost shares

### What was asked, and what is actually identifiable

The brief asked for a per-width dispatch histogram **for beagle and medicine
separately**, then score-weighted cost shares.

**The histogram is not measurable.** `beagle` and `medicine` are R2-only hidden
prompts in `fixtures/qwen3_8_27b_mtp_track.json`; the prompt text never reaches a
solver checkout, and `officialMetrics` exposes no per-round width histogram. E42
already recorded this as `absolute_ranked_share_is_identified: false`. One scalar
per prompt is observable — the mean dispatched width `mean_m`, recoverable from
the published accepted-draft rate.

So the deliverable below is an **extrapolation, not a measurement**, and it is
labelled that way in the artifact's `identification` field.

### Method

`research/e48_score_weighted_shares.py` takes the measured corpus histogram and
applies the **maximum-entropy exponential tilt** that reproduces each hidden
prompt's known `mean_m`:

```text
p_i(M) ∝ p_corpus(M) · exp(−λ_i · M),   λ_i chosen so Σ M·p_i(M) = mean_m_i
```

This is the least-committal distribution consistent with the one statistic that
is actually known: any other reweighting injects an assumption the data does not
support. Dispatch shares are then converted to **cost** shares with thorfinn's
E46 refit and this base's own instructions-per-group table
(`IPG = {2:2, 3:3, 4:4, 5:3, 6:3, 7:4, 8:4, 9:3}`, read off
`kernels/quantized.h:1924-1977`):

```text
T(M) = 16.757 + 27.532·ceil(M / IPG[M]) + 9.624·M
```

Fed the corpus histogram the script reproduces the advisor's published corpus
figures exactly, which is the check that the pricing model was transcribed
correctly.

### Result

| slice | corpus (measured) | beagle (pred.) | medicine (pred.) | **score-weighted 0.79/0.21** |
|---|---|---|---|---|
| M = 9 cost share | 53.45 % | 19.69 % | 23.44 % | **20.48 %** |
| M ∈ {7,8} cost share | 12.25 % | 8.998 % | 9.759 % | **9.158 %** |
| M ∈ {4,5,6} cost share | — | 65.69 % | 62.47 % | **65.01 %** |

beagle's predicted M = 9 **dispatch** share is 12.7 %.

### Why this matters to the ledger

The hidden prompts decode at much lower mean width than the public corpus
(`mean_m` 5.53 / 5.77), so ranked width mass sits in M ∈ {4,5,6}, not at M = 9.
Re-pricing two live ledger items against the score-weighted column rather than
the corpus column:

| ledger item | priced on corpus | priced score-weighted |
|---|---|---|
| 173(C) M = 9 prize | as published | ≈ **+2.06 %**, ≈ 2.7 sd |
| alphonse's M ∈ {7,8} arm | as published | ≈ **+0.69 %**, ≈ 0.90 sd |

The M ∈ {7,8} arm falls **below the 0.7678 % board-visible floor** under this
weighting. That is a prediction, not a measurement, and it should be treated as
a prioritisation signal rather than a reason to cancel work outright — but the
M ∈ {4,5,6} band carrying ~65 % of ranked QMV cost is the more valuable target
on the same reasoning.

Artifact: `research/e48-artifacts/score-weighted-shares.json`.
