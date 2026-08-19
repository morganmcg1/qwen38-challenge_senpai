# E48 — Does a uniform QMV speedup lower the score?

Assignment `qwen38-r1-e48-score-weighted-qmv-and-uniform-sign` (PR #52, revision `r1`).
Base `fb0a09d3912477d94ed631bdb90fd04172d7b4cf`. Host: local M4 Pro (**not** the
ranked M5). Nothing here is gate-qualified, ranked-equivalent, or an official score.

> Every timed leg ran with `MLXFAST_LOCAL_COOL_GATE=0`.
> `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
> `official_or_ranked_score=false`.

## Headline

| result | value |
|---|---|
| **`psi_mtp` (primary)** | **0.693391**, interval [0.692292, 0.694490]; 0.694474 under the second dosimetry |
| ranked dScore per 1 % candidate QMV cost cut | **+0.693391 %**, gated or uniform alike |
| two-dose form test | dose ratio **2.0092** measured; elasticity moves **−0.317 %** |
| null arm (`base2`) | **−0.0629 %** on `raw_p`; Arm G effects are 618× and 891× that |
| `psi_serial` | **not identifiable**; two of four treatments exceed 1.0, which is impossible |
| uniform-sign question | **withdrawn by the advisor**, not answered by me |
| Part 1, score-weighted at 0.4837/0.5163 | M = 9 **21.63 %**, M ∈ {7,8} **9.391 %**, M ∈ {4,5,6} **64.03 %** |
| alphonse's merged M ∈ {7,8} cell | **+0.7437 %** of score — below the 0.7678 % board floor, 1.43× the crown gap |

**Nothing here changes the scored surface.** Every arm's kernel edit was reverted;
the deliverable is measurement, not a candidate.

## Evidence and reproduction

W&B run: **[`e48-psi-mtp-arm-g`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/yd949eze)**
(`yd949eze`, state `finished`), carrying every arm, the per-width dose table, the
width histograms, the Part 1 cost shares and the re-pricing tables.

```bash
# one arm = splice the dose, COMMIT it, then time it. Doses per arm:
#   base 0/0   g1 1/0   g2 2/0   ulo 1/2   base2 0/0 (--revert)
research/e48_perturb.py --crossrow-level 1 --m1-level 0
git commit -am "E48 arm g1"
research/e48-run.sh g1 --curve --legs 2      # 512-token legs + in-arm dose curve

# analysis, no GPU, no lock needed
python3 research/e48_score_weighted_shares.py > research/e48-artifacts/score-weighted-shares.json
python3 research/e48_analyze.py --arms base ulo g1 g2 base2 --wandb \
  > research/e48-artifacts/analysis.json
```

`research/e48-run.sh` pins `E42_BASE_SHA` to this base and defaults
`MLXFAST_LOCAL_RUN_LOCK_DIR` to the shared `/tmp/mlxfast-shared`, so an arm cannot
silently time against the wrong base or take a per-role lock.

Artifacts: `research/e48-artifacts/analysis.json`,
`research/e48-artifacts/score-weighted-shares.json`.
Timing inputs stay out of Git under `.mlxfast-private/e48/`.

---

## Part 2 — `psi_mtp` by direct dose injection

### The sign question was withdrawn by the advisor, not answered by me

Part 2 was assigned as "does a uniform QMV speedup lower the score?". The advisor
withdrew that question mid-experiment (PR #52 comment 5342984599) on the grounds
that the ranked baseline leg is a **separately built pinned binary**, so candidate
source changes have zero leverage on `serial`. I record that provenance explicitly:
**the sign question was retired by the advisor's reasoning, not settled by my
measurement.** The one uniform arm that had already run (`ulo`) is reported below as
a local-only internal consistency check with no ranked meaning. **Arm S was never
run**, and the second uniform dose was not run either: once the sign question was
withdrawn, both arms could only have measured a quantity with no ranked leverage, so
the remaining GPU time went to the null arm the advisor asked for instead. That is a
deliberate substitution, not an incomplete sweep.

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

The advisor branch's own fail-closed gate agrees when pointed at **this** base's
workflow:

```text
$ senpai/verify-ranked-score-boundary.sh          # advisor-branch script, my base's workflow
PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only
```

**Harness labels, as required.** `psi_mtp` and the `+0.693391 %/%` ranked
coefficient are `harness=ranked`. Every `psi_serial`, `psi_mtp_TOTAL`, `rho*`,
uniform-coefficient and `raw_p` figure in this report is `harness=local`, measured
in a same-build two-leg harness where a broad improvement can cancel. No local
quantity is subtracted from a ranked price anywhere in this report.

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
| `g2` | 2/0 | 0.073412 | 0.075989 | 0.966081 | **−0.065** | +127.258 | **−56.026** |
| `ulo` | 1/2 | 0.121987 | 0.055473 | 2.199035 | +66.060 | +65.900 | **+0.096** |
| `base2` | 0/0 | 0.073450 | 0.033454 | 2.195540 | **−0.013** | **+0.050** | **−0.063** |

Per-leg `raw_p` spread: `base` 0.105 %, `g1` 0.144 %, `g2` 0.076 %, `ulo` 0.116 %.

### The null arm, and every effect beside it

`base2` is a pristine byte-identical rebuild of `base` — the scored surface was
proven identical by `git diff` before launch — run in its own session slot at the
far end of the bracket. It is the arm the advisor asked for, and it is the number
that makes every other number in this report interpretable:

| quantity | value |
|---|---|
| null `raw_p` effect | **−0.0629 %** |
| null serial-leg effect | **−0.0133 %** |
| null MTP-leg effect | **+0.0497 %** |

Every effect in the experiment, expressed as a multiple of that floor:

| arm | `raw_p` effect | × null floor |
|---|---|---|
| `g1` | −38.844 % | **618×** |
| `g2` | −56.026 % | **891×** |
| `ulo` | +0.096 % | **1.5×** |

Two conclusions follow, and the second is the one I did not expect.

🟢 **Arm G is not a noise artifact by any margin worth discussing.** The two Arm G
effects are ~600–900× the measured null. The elasticity is real.

🔴 **`ulo`, the local uniform arm, is at the noise floor.** +0.096 % against a
−0.063 % null is a **1.5×** ratio. So even inside the local frame, the honest
statement is that a 66 % QMV slowdown applied to *both* legs moved the local ratio
by an amount this instrument cannot distinguish from zero. That is consistent with
"approximately score-neutral in the local frame", and it is *not* strong enough to
support a signed claim of any particular size — which matters because the signed
claim is the quantity the advisor withdrew for an independent reason.

### The instrument's floor, against the board's floor

Ledger 172 concluded that "local ABBA measurement with a null arm is now the *only*
instrument in the campaign that can establish a per-leg effect". This null arm puts
a number on that instrument for the first time:

| floor | value | source |
|---|---|---|
| this design, `raw_p` | **0.063 %** | `base` vs `base2`, measured here |
| this design, MTP leg | **0.050 %** | same |
| board replication floor (score) | 0.7678 % | ledger 166 |
| board MTP-leg floor | 0.8040 % | ledger 172 |
| smoke-config A/A (advisor's warning) | sd 18.368 %, worst 16.686 % | alphonse's guard arm, relayed on this PR |

The local bracketed design resolves per-leg effects **~12× finer than the board's
score floor and ~16× finer than the board's MTP-leg floor**, and roughly 300×
finer than the smoke-config A/A the advisor warned me about. The difference is
design, not luck: one prompt, one deterministic trajectory, full 512-token legs,
both zero-dose arms bookending the dosed arms in a single session.

🟡 That is a floor for *this* comparison shape only — same host, same session, same
fixture, same head, effects injected rather than engineered. It says nothing about
cross-build or cross-host reproduction, which the dosimeter section below shows is
a materially weaker property.

**Fidelity (advisor Risk 4).** `all_tokens_matched=True` in every arm;
`accepted_draft_rate = 0.88752556` identical to 8 s.f. across all arms;
`rounds = 78`; `residual_divergence_count = 0`. The injection is pure cost and
bit-exact — it does not perturb the trajectory, so the dose response is a clean
cost derivative.

### `psi_mtp` — the headline number

Each Arm G arm gives an independent estimate from its own measured dose:

| arm | measured `xbar_X` | `mtp_frac` | `psi_mtp` |
|---|---|---|---|
| `g1` | 0.914900 | 0.635389 | 0.694490 |
| `g2` | 1.838211 | 1.272578 | 0.692292 |

**`psi_mtp = 0.693391`, interval [0.692292, 0.694490]**

**`base2` also bought a second, independent dosimetry variant.** Before the null
arm existed, `stable_shapes` had nothing to identify *which* curve shapes fail
cross-build reproduction, so it collapsed onto `as_measured` and I reported one
denominator. `base2` is a zero-dose rebuild, so any shape whose per-call cost moves
between `base` and `base2` is irreproducible by construction. Two shapes fail at the
2 % tolerance — `full_attn.qkv_proj_fused` and `mlp.down` — and excluding them gives
a genuinely separate estimate:

| dosimetry | `psi_mtp` | interval | form residual |
|---|---|---|---|
| `as_measured` | **0.693391** | [0.692292, 0.694490] | −0.317 % |
| `stable_shapes` (identified by `base2`) | **0.694474** | [0.693571, 0.695378] | −0.260 % |

**The two dosimetries agree to 0.16 %, and the full envelope over both variants and
both doses is 0.4448 % wide.** `psi_mtp` is therefore not an artifact of a
denominator choice — which is exactly the criticism that E42's three-denominator
interval could not answer, and it is the single most important thing the null arm
delivered beyond a noise floor. Contrast `psi_serial` below, whose variants span
56 %.

**Two-dose functional-form test.** The realised dose ratio between the arms was
**2.0092** (measured, not assumed — the `L=2` injection delivered almost exactly
twice the `L=1` dose). Doubling the dose changes the inferred elasticity by
**−0.317 %**. A single linear-in-dose coefficient describes the candidate leg
across a 39-point and a 56-point effect, so `psi_mtp` is a genuine elasticity over
this range, not a local slope fitted at one operating point.

Gap-corrected variants (see coverage gap below): **0.710161** (2-bit-scaled) and
**0.726931** (4-bit upper). All of these are **lower bounds**.

**Advisor Risk 3 answered: `psi_mtp` transfers across the IPG change.** E42
measured 0.6736 on base `04ad6bf1`; this base gives 0.693391, i.e. **+2.9 %**.
The parameter is stable across the intervening dispatch-shape change, so ledger
dScore corrections built on it survive. Direction of the correction: every dScore
in the ledger — including alphonse's merged +11.421 % — is slightly **under**-priced.

**The advisor's own Arm G prediction validates at both doses.** Evaluating
`1/(1 + 0.6736 x)` at each realised dose:

| arm | realised `xbar_X` | predicted `raw_p` Δ% | measured | miss |
|---|---|---|---|---|
| `g1` | 0.914900 | −38.130 | **−38.844** | 0.71 pp |
| `g2` | 1.838211 | −55.322 | **−56.026** | 0.70 pp |

The miss is the same size and the same sign at both doses — exactly the signature
of a prediction made with a `psi_mtp` that is 2.9 % too small, and not of a
mis-specified functional form.

### Structural-churn control

Each Arm G serial leg carries the full dose scaffolding compiled in at dose 0:

| arm | serial Δ |
|---|---|
| `g1` | +0.0135 % |
| `g2` | −0.0649 % |

Worst absolute deviation **0.065 %**, and the two straddle zero. The scaffolding is
free, so the dose attribution is validated on the scored path rather than assumed.

### The finding I did not expect: `psi_serial` was never identifiable

`ulo` and `g1` carry the identical crossrow dose in fully independent builds, which
turns them into an accidental dosimeter-reproducibility test:

| widths 2..9 | value |
|---|---|
| mean abs diff in realised dose | **0.0039** |
| worst (M=2) | **0.01014** |
| `xbar_X` agreement | **0.084 %** |

Per-width absolute disagreement: M2 0.01014, M3 0.00629, M4 0.00085, M5 0.00465,
M6 0.00031, M7 0.00268, M8 0.00476, M9 0.00124.

The crossrow dosimeter is reproducible to sub-percent. The width-1 dosimeter is not.
`g1` and `g2` both carry **width-1 dose exactly zero**, so both cells must read 0:

| arm | width-1 dose applied | width-1 cell reads |
|---|---|---|
| `g1` | 0 | **0.06942** |
| `g2` | 0 | **0.04230** |

Two independent builds at true zero disagree by **0.0271** — 2.7× the worst crossrow
disagreement, 87× the best — and neither reads zero. That offset divides straight
into `psi_serial`.

**So `psi_mtp` is identified and `psi_serial` never was.** This single fact
retro-explains E42's three-denominator interval, the unstable
`in_proj_fused_qkvzba` cell, why `rho*` landed *between* the two `rho_unif`
candidates, and my own repeated arithmetic slips on the uniform coefficient. The
unpinnable parameter turned out to be exactly the one the advisor's withdrawal
showed has no ranked leverage — the two findings are independent and agree.

**🔴 With `base2` in hand the case is no longer circumstantial: one `psi_serial`
variant is physically impossible.** Across the two dosimetries and the two offset
treatments:

| variant | `psi_serial` |
|---|---|
| `as_measured`, offset left in | 0.7966 |
| `as_measured`, offset subtracted | 0.8694 |
| `stable_shapes`, offset left in | 1.0414 |
| `stable_shapes`, offset subtracted | **1.2470** |

`psi_serial` is a *share of serial-leg time spent in QMV*. It cannot exceed 1.0. Two
of the four defensible treatments return values above 1.0, up to **1.247**. That is
not a wide interval, it is a **proof that the width-1 dosimeter does not measure the
quantity it claims to measure** — so no treatment of it is trustworthy, including the
two that happen to land below 1.0. `psi_mtp`'s envelope over the same four-way
construction stays inside 0.4448 %.

### Second unplanned finding: arithmetic converts to time *differently at M=2*

The dose is a fixed amount of redundant arithmetic per dispatch, so the realised
per-width dose is a direct read of **how much of an injected ALU op actually costs
time at that width**. Normalising each width against the arm's mean crossrow dose:

| width | `g1` (dose 1) | `g2` (dose 2) |
|---|---|---|
| 2 | **0.541** | **0.625** |
| 3 | 0.908 | 0.915 |
| 4 | 0.934 | 0.934 |
| 5 | 1.002 | 0.993 |
| 6 | 0.991 | 0.990 |
| 7 | 0.994 | 0.991 |
| 8 | 0.999 | 0.999 |
| 9 | 1.014 | 1.015 |

The pattern **reproduces at both dose levels**: widths ≥5 convert injected
arithmetic at essentially 100 %, M=4 at ~93 %, M=3 at ~91 %, and **M=2 at only
54–63 %**. The natural reading is that at M=2 the crossrow kernel is weight-traffic
bound and has ALU slack that absorbs roughly 40 % of injected work, while at M≥5 it
is arithmetic-limited and every op costs full time.

**Why this matters beyond this experiment.** `psi_mtp = 0.693` is an average over
*this* decode's realised width mix, which is heavily M≥5 (72 of 78 rounds; M=4 is 5
rounds, M=2 is 1). So the M=2 shortfall barely moves the number here. But it means a
single conversion factor should not be applied per width:

- an **arithmetic-reduction** QMV win converts at ~full rate at M≥5 and at roughly
  half rate at M=2;
- a **weight-traffic** win should convert *better* at M=2 than this dosimeter implies,
  because the dosimeter only probes the ALU axis.

That distinction bears directly on ledger 173(C) (M=9-targeted) versus alphonse's
M∈{7,8} work — both sit in the region where the two axes agree — but it would matter
for any future width-1/2-targeted proposal, and for the hidden prompts, which Part 1
estimates run shallower than this fixture.

### Local uniform coefficient (the withdrawn quantity), reported as an interval

Because of that width-1 offset, and because `base2` added a second dosimetry, there
are four defensible treatments. I report all of them rather than picking one:

| dosimetry | offset | `psi_serial` | `psi_mtp_TOTAL` | local uniform coefficient |
|---|---|---|---|---|
| `as_measured` | left in | 0.7966 | 0.7197 | **−0.0769** |
| `as_measured` | subtracted | 0.8694 | 0.7197 | −0.1497 |
| `stable_shapes` | left in | 1.0414 🔴 | 0.7214 | −0.3201 |
| `stable_shapes` | subtracted | 1.2470 🔴 | 0.7214 | **−0.5257** |

🔴 **Envelope `[−0.5257, −0.0769]`, a 6.8× range that contains ledger 173(A)'s
−0.1789.** So the final honest statement on the withdrawn quantity is not "173(A)
was overstated 2.33×", which is what I said at 13:59Z from the `as_measured` row
alone. It is: **the local uniform coefficient is unidentified, 173(A)'s value lies
inside its envelope, and the envelope is bounded below by treatments whose
`psi_serial` is physically impossible.** My 13:59Z correction was right to move
toward the advisor's number and wrong to present a point.

`psi_mtp_TOTAL_local` is 0.7197–0.7214 across all four treatments, and the
denominator-free crossover `rho* = 1.9952` is unchanged by every treatment —
the two quantities that do not route through the width-1 dosimeter are the two
that stay put.

🟢 **None of this reaches ranked.** The whole table lives inside the frame the
advisor withdrew: `psi_serial` has no ranked leverage because the ranked serial leg
is a pinned separate binary. It is reported so the local frame is on the record, not
because it prices anything.

**Disclosure: I revised this number four times across this experiment**
(−0.0265 → −0.0769 → two-way interval → this four-way envelope). The first three
slips came from quoting hand arithmetic ahead of the in-arm dosimeter. The fourth
change is different in kind: it came from *new evidence* (`base2` identifying the
irreproducible shapes), which is the instrument working as designed rather than a
process failure. Every number above is analyzer output over measured doses.

### `psi_mtp_w1` by differencing, and why it is a floor

Differencing `ulo` against `g1` isolates the width-1 component:
`mtp_frac` difference 0.023616 / realised `x1 = 0.829258` ⇒
**`psi_mtp_w1 = 0.028478`**, 31 % below E42's arithmetic 0.0415.

It covers **only** the 4-bit `qmv_fast_impl` width-1 path and **excludes the 2-bit
`:1908` readout entirely**, so it is a floor, not an estimate.

### Coverage gap (verified still open)

`qmv_fast_singlerow_affine2_g64` is instantiated only as `<T>` and is therefore
**undosed in every arm**. Measured over 489 draft steps against 11.803 s of treated
verify QMV, the untreated candidate-leg share is **4.639 %** (4-bit proxy upper) or
**2.375 %** (2-bit scaled); the additive correction to `psi_mtp` is **+0.03354** /
**+0.01677**. The serial leg has no such gap.

The correction's **sign is certain** — untreated candidate QMV has strictly positive
cost — and only its magnitude is uncertain. That is what makes every `psi_mtp`
figure above a one-sided lower bound rather than a point estimate with unknown bias.

Separately: `QwenQMVCostCurveTests.swift:396`'s `in_proj_fused_qkvzba` shape is a
**harness fiction** — GDN `a`/`b` are separate `linear` calls
(`Qwen35GatedDelta.swift:254-255`, `Qwen35FastEngine.swift:495-520`), so no fused
QKVZBA dispatch exists on the scored path. That explains the unstable cell.

🔴 **Closing this gap is policy-dead, and I am dropping it as my top follow-up.**
I previously called measuring the 2-bit `:1908` path "my top follow-up". The advisor
showed the lever it would justify is prohibited, and I verified the enforcement
myself on this base rather than taking the relay on trust:

| check | evidence, this base |
|---|---|
| the ranked pipeline runs an LLM bypass review | `.github/workflows/qwen-mtp-ranked-benchmark.yml:1203` invokes `.github/scripts/run-submission-static-review.sh` (the advisor's relay said `:1190`; that is the comment block at `:1173`, the invocation is at `:1203` on this base) |
| the quantization rule is appended **after** the per-track `if/else` closes | our track's own branch is `:446`; the dispatch closes at `fi` `:452`; the rule is appended unconditionally at `:453` — so it governs our track on every run |
| the prohibited clause | *"Fail: … **any bit width other than 4 or 8**; any group size other than 16 or 32 … **This holds even when the re-quantized path passes the correctness gates**"* |
| severity | *"Treat re-quantization beyond the accepted envelope as high or critical severity"* |
| our own account's record | two consecutive `morganmcg1` FAILED submissions at exactly that step (2026-08-18T01:30:23, 03:33:22), both a 3-bit compact draft readout |
| `.github` is not editable | `benchmark.json` `editablePaths`, 89 entries, zero under `.github` |

So: **measuring** the 2-bit path's share stays legitimate — it bounds `psi_mtp` and
nothing more. **Changing a readout bit width cannot ship**, whatever it measures. My
three draft-bits branches are unshippable as families for the same reason.

🟢 And the same prompt protects the work in this experiment explicitly: *"Do not fail
ordinary performance work such as quantized matmul or kernel optimization, Metal
kernel tuning (tiling, vectorization, **simdgroup operations**, threadgroup memory,
unrolling, occupancy and dispatch-geometry tuning, dtype or group-size template
specialization…)"* (`:433`). Injection arms, the width census, and every width/IPG
experiment here sit inside that sentence.

🟡 Honest limit on how far to apply the rule. Its stated premise is NVFP4 group-16,
while this track's model is **affine 4-bit group-64**. Read literally, the group-size
clause would condemn the crown tree itself, so the reviewer is demonstrably not
applying it literally and the envelope's edges are unknown. The clause that matches
every observed failure is the bit-width one, and 4 bits is safe by construction
because it ships and it scores. One further guardrail from the same file applies to
my kind of work directly: it fails *"kernel or kernel-dispatch edits that
special-case benchmark-shaped inputs … instead of staying general"* — so a width
slice must be justified by the dispatch table, never by the fixture.

### `arch_gen` portability: I raised this, and it is refuted from source

I flagged that `vector_limit` is architecture-generation dependent
(`quantized.cpp:84-124`), and that gen 13/14 with `D > 4096` returns 6, which would
stop widths 7–9 from ever reaching the crossrow family on the scored host and would
invalidate every per-width share in Part 1. **The conditional is right and the
antecedent is false.** Verified on this base:

```text
device.cpp:564-572   ag_tens = arch_[n-3]-'0';  ag_ones = arch_[n-2]-'0';
                     arch_gen_ = 10*ag_tens + ag_ones
                     => applegpu_g17s -> 17 (M5);  applegpu_g16s -> 16 (this host)
quantized.cpp:86-87  if (arch_gen == 13 || arch_gen == 14) { ...6/10/14 table... }
quantized.cpp:106-123 else, arch_size 's', D,O > 4096 -> return 10
```

Neither host enters the 13/14 branch, so `vector_limit = 10` on both and widths 2..9
all reach the crossrow family on the ranked host. The risk is closed, not deferred.
The M5 architecture string `applegpu_g17s` is the one input I take from campaign
record (alphonse's `research/arch_lever_audit.py`) rather than from this host; the
parse and the branch above are verified here.

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

### Item 1 — what the 78 dispatches were 78 *of*

They are the **MTP verify rounds of ONE 512-token `--local-iterate` decode** of the
single public fixture `public_longcopy_gate_english_512_256.json`, on base
`04ad6bf1`. Not a microbenchmark, but **n = 1 prompt and n = 1 decode**, and that
prompt is a public fixture, not one of the eight ranked prompts. So the histogram
was never corpus-wide in the sense of averaging over the scored prompt pool — it is
one draw from one public prompt. The advisor's worry that it is "weighted toward
prompts that cannot affect the score" is correct and in fact understated: **every**
prompt in it is worth exactly 0.0000 on the board.

### Item 3 — the histogram is deterministic, not a random variable

This is now measured rather than argued. Across **10 independent decodes** (5 arms ×
2 legs, each a separate process, and 4 of the 5 arms carrying different injected
kernel doses):

```text
rounds        = 78         in every draw
mean_m        = 7.269230769230769   in every draw
histogram     = {2:1, 4:5, 5:5, 6:23, 7:4, 8:6, 9:34}   in every draw
mean_m_sd_pct = 0.0
```

`identical_across_all_draws = true`. The repeat-to-repeat spread the advisor asked
for is **exactly zero**, and it stays zero even when the kernel is slowed by 127 %.

Why: decoding is greedy against a fixed target and a pinned head, so the accepted
prefix length at each round is a deterministic function of the prompt. Cost
perturbations change *timing* but cannot change *which* tokens are accepted — which
is the same fact as `accepted_draft_rate` agreeing to 8 significant figures across
all arms. So a single draw *is* the process here, for a fixed prompt.

🔴 **But note the scope of that claim.** Determinism given the prompt does **not**
make the histogram stable *across* prompts — that is precisely the between-prompt
variation the advisor is worried about, and it is the part that remains unidentified
because beagle and medicine are R2-only. So the n = 1 weakness of the 78 dispatches
is entirely "one prompt", and not at all "one draw".

### Item 4 — pre-registration

Pre-registered in the 13:02Z PR comment before the shares were computed:
`psi_mtp_w1 = 0.0415` from the E42 artifact, the three-denominator sign table, and
the prediction that beagle's M = 9 share falls materially below the corpus 43.6 %
of dispatches. The realised prediction is **12.7 % of dispatches / 19.69 % of cost**
— below the advisor's stated expectation, and in the direction he predicted.

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

### Result, at the corrected marginal weights

The weights are **not** re-inlined here: `research/e48_score_weighted_shares.py`
imports `marginal_weights()` from `research/qmv_score_leverage.py`, which derives them
from the pinned crown order statistics. My copy of that helper is byte-identical to
the one on the advisor branch. It returns:

```text
beagle   0.483693625973383      d score% / d beagle leg%
medicine 0.516306374026617      d score% / d medicine leg%
```

| slice | corpus (measured) | **beagle** (pred., unweighted) | **medicine** (pred., unweighted) | **score-weighted 0.4837/0.5163** | superseded 0.79/0.21 |
|---|---|---|---|---|---|
| M = 9 cost share | 53.45 % | **19.692 %** | **23.445 %** | **21.630 %** | 20.480 % |
| M ∈ {7,8} cost share | 12.25 % | **8.998 %** | **9.759 %** | **9.391 %** | 9.158 % |
| M ∈ {4,5,6} cost share | 33.76 % | **65.686 %** | **62.469 %** | **64.025 %** | 65.010 % |

beagle's predicted M = 9 **dispatch** share is 12.71 %.

🟡 Unit convention, because this report quotes both numbers: `mean_m` is the
**dispatched row width**, and a verify round checks the pending primary token plus
its drafts, so `mean_m = effective_mean_draft_len + 1`. beagle 4.5327 → 5.5327,
medicine 4.7677 → 5.7677. The script takes the published draft lengths and adds the
primary row; it does not conflate the two.

The advisor predicted the direction of this correction before I ran it, and he was
right: medicine drafts longer (`mean_m` 5.7677 vs 5.5327), so a medicine-tilted
weighting shifts the mixture **up**. M = 9 rises 20.480 → **21.630 %** and
M ∈ {7,8} rises 9.158 → **9.391 %**. My earlier headline was biased in the
direction that understates both.

🔴 **Why 0.79/0.21 was wrong, retained for the record.** The E48 brief called it
"the marginal value ratio from the order-statistic structure". It is not. It is
E40's per-prompt **leg-effect** split (+0.363 % beagle vs +0.088 % medicine, 4.1×) —
a statement about how much room each prompt has, not about what a unit of gain in
each is worth. Using it as a weight double-counts the heterogeneity. The correct
weights are `0.5 · r_i / score`, which come out near-equal because the two scored
ratios are near-equal. Both columns are kept in the artifact, the superseded one
carrying its `status` and `reason` fields, so the propagation of the mislabelled
constant stays auditable.

### The number the advisor asked for: does alphonse's merged cell clear the floor?

Adopting my own measured `psi_mtp = 0.693391` and his measured 11.421 % QMV
reduction at M ∈ {7,8}:

| `f{7,8}` basis | dScore | vs board floor 0.7678 % | vs crown gap 0.5193 % |
|---|---|---|---|
| 0.09158 (superseded 79/21) | +0.7252 % | 🔴 below | 1.40× |
| **0.09391 (corrected weights)** | **+0.7437 %** | 🔴 **still below, by 0.024 pp** | 1.43× |
| 0.08998 (beagle alone) | +0.7128 % | 🔴 below | 1.37× |
| 0.09759 (medicine alone) | +0.7731 % | 🟢 marginally above | 1.49× |
| 0.1225 (advisor's corpus-wide) | +0.9701 % | 🟢 above | 1.87× |

**Answer: the corrected weights close 43 % of the gap to the floor and do not clear
it.** The mechanism needs `f{7,8} ≥ 0.09696` to reach 0.7678 %; I predict 0.09391,
which is 3.1 % short in relative terms. The verdict flips on medicine alone and does
not flip on beagle alone, so the decision is inside the uncertainty of a
max-entropy extrapolation from one moment per prompt. **I would not cancel his
mechanism on this number, and I would not claim it clears the floor either.**

🟢 One reframing that I think matters more than the point estimate. The 0.7678 %
figure is a *between-submission replication* floor: it says a single ranked
submission cannot confirm a gain that size. It is not a threshold for whether the
mechanism works. This experiment's own null arm measures the local instrument's
floor at **0.063 %**, so a +0.74 % mechanism is ~12× the local floor and is
comfortably measurable *locally* — it is only the board that cannot see it in one
run. Ledger 172 reached the same conclusion from the noise side; the null arm here
gives it a number.

### Re-pricing must re-sort, not multiply a rate

The score is the mean of order statistics 4 and 5, so a constant %/% rate assumes
rank order is preserved. `research/qmv_score_leverage.py` pins the kink at
**+1.0551 %** of scored-pair leg gain and a hard saturation cap at **+4.7156 %** for
any beagle/medicine-only mechanism. The artifact therefore reports
`score_pct_from_leg_gains()` output beside the naive rate:

| mechanism | slice | order-statistic dScore | naive rate | rate error |
|---|---|---|---|---|
| alphonse E44 r2, measured 11.421 % | M ∈ {7,8} | **+0.7437 %** | +0.7437 % | 0.000 pp |
| the same 11.421 %, applied at M = 9 | M = 9 | **+1.2991 %** | +1.7129 % | **+0.414 pp** |

Below the kink the two agree exactly, which is why alphonse's figure is unaffected.
Above it the rate model overstates by **24 %**. So my earlier "173(C) M = 9 prize
≈ +2.06 %" was a rate extrapolation across the kink and should not be used; the M = 9
row above replaces it, and it deliberately reuses alphonse's *measured* 11.421 % so
the two slices are comparable — **it is not a claim about the size of any actual
M = 9 proposal.** I have also dropped the "sd" multiples I quoted earlier, since
ledger 166 retired `SIGMA_SCORE_PCT` as a between-submission yardstick.

### Where the ranked width mass actually sits

The hidden prompts decode at much lower mean width than the public corpus
(`mean_m` 5.53 / 5.77), so ranked QMV cost concentrates in **M ∈ {4,5,6} at
64.0 %**, not at M = 9. On the same reasoning that demotes M = 9, that band is the
most valuable width target in the kernel, and it is where I would point the next
mechanism. That is a prediction from an extrapolation, not a measurement.

### The width-1 conclusion, restated on the order statistics

I previously supported "the candidate never runs width-1 rounds" with
`non_drafting_round_count = 0`. That measurement was for beagle and medicine only;
generalising it to the corpus was wrong — plutarch is non-zero on 320 of 371 healthy
board rows, mode 449, including on the crown tree and ours.

🟢 The conclusion survives on a better reason: **the only prompt that carries
width-1 rounds carries zero marginal weight.** plutarch's ratio is 1.2560 against
rank-4 beagle's 3.1433, so it would need **+150.3 %** to enter the scored pair at all
(`qmv_score_leverage.CROWN_ORDER_STATS`). That rests on the order statistics, which
are pinned in the non-editable workflow, rather than on a property of a binary that
could change under us.

Related count correction, also mine: `mtp_depth = 8` is a **cap, not a firing
count**. The measured `effective_mean_draft_len` is **4.5327** on beagle and
**4.7677** on medicine (`D/R` = 485/107 and 472/99), so the head fires ~4.53 times
per round, not 8. Any share derived from 8-per-round overstates by 1.77×. Two things
remain genuinely unestablished and I flag them rather than assume: whether the
schedule is adaptive or truncated, and whether the field counts proposals or
acceptances.

Artifact: `research/e48-artifacts/score-weighted-shares.json`.

---

## What this experiment does not settle

1. **The hidden per-width histogram.** beagle and medicine are R2-only, so every
   Part 1 share is a max-entropy extrapolation from one moment per prompt. It is
   labelled `PREDICTION ONLY` in the artifact and should stay that way. The
   f{7,8} floor verdict flips between the two prompts taken alone, so this is the
   binding uncertainty in the whole report.
2. **The uniform sign in the local frame.** Envelope `[−0.5257, −0.0769]`, and the
   `ulo` arm that would resolve it sits at 1.5× the null floor. Unidentified, and
   without ranked meaning either way.
3. **Total candidate width-1 QMV exposure.** `psi_mtp_w1 = 0.0285` covers only the
   4-bit `qmv_fast_impl` path; the 2-bit `:1908` readout is undosed in every arm, so
   `psi_mtp` stays a one-sided lower bound (correction `+0.0168` to `+0.0335`).
4. **M5 transfer.** Everything is M4 Pro, ungated. `vector_limit` is now proven equal
   on both hosts, but conversion rates per width are not.
5. **Whether the draft schedule is adaptive or truncated**, and whether
   `effective_mean_draft_len` counts proposals or acceptances.

## Thermal and contention record

Ungated local protocol, permitted for local timed arms: arms are bracketed
`base → ulo → g1 → g2 → base2` in one session, so monotone drift appears as the
`base`/`base2` spread reported above (**0.063 %**). Entry-temperature spread across
arms **5.62 °C**; entry and exit GPU temperature recorded per arm in
`analysis.json.provenance`. `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false`, `official_or_ranked_score=false`, verbatim.

Contention control, as asked: `MLXFAST_LOCAL_RUN_LOCK_DIR=/tmp/mlxfast-shared` was
exported for every arm (`research/e48-run.sh:17` defaults it, so it cannot be
forgotten), every arm ran through `benchmark-qwen-mtp.sh --local-iterate`, which does
take the lock, and `benchmark.sh` was **not** edited. I did not lift the guard
functions, so the `local_run_guard_enabled || return 0` fail-open hole does not apply
here.

## Provenance note: this result was recovered, and the arms were not re-run

The timing arms ran on the previous instance of this host; `base2` finished at
14:48Z. The machine was then re-provisioned, and my branch commits existed only
locally. I recovered them by fetching the branch from the previous workspace
(`470acc1` is an ancestor, so it was a clean fast-forward to `5bb6d0d`) and copying
the gitignored measurement inputs under `.mlxfast-private/e48/`. **No arm was re-run
and no number was re-derived by hand**; every figure in this report comes from
`research/e48_analyze.py` and `research/e48_score_weighted_shares.py` executed after
the recovery, over the original per-arm score files and dose curves. The analyzer
records `head_sha = 5bb6d0d`, the commit the arms were analyzed at; the final result
commit adds only this write-up and the two artifacts.

## Suggested follow-ups, not implemented

1. 🟢 **Measure `f{7,8}` and the M ∈ {4,5,6} share properly instead of extrapolating.**
   The whole Part 1 uncertainty is one missing observable. The public fixture's
   histogram is deterministic and free to collect at any `mean_m`, so a **schedule
   sweep** — cap the offered depth at 3, 4, 5, 6 and record the realised histogram at
   each — would map dispatch mixture against `mean_m` on measured data, and the hidden
   prompts' known `mean_m` could then be read off that curve rather than off a
   max-entropy assumption. This is the highest-value follow-up in this report and it
   needs no new kernel work.
2. 🟢 **Point the next kernel mechanism at M ∈ {4,5,6}** (64.0 % of predicted ranked
   QMV cost), not at M = 9 (21.6 %). Note M = 4 and M = 3 convert injected arithmetic
   at only ~93 % and ~91 %, so an ALU-reduction mechanism there is worth slightly less
   than its kernel benchmark suggests, while a weight-traffic mechanism may be worth
   more.
3. 🟡 **Separate the ALU and weight-traffic axes in the dosimeter.** The current
   injection only adds arithmetic, which is why M=2 reads 54–63 %. A second dose form
   that adds *memory traffic* instead would give the per-width conversion factor for
   bandwidth-bound mechanisms — the missing half of the pricing model.
4. 🟡 **Retire the width-1 dosimeter, or fix it.** Two builds at true dose zero read
   0.069 and 0.042, and offset-corrected `psi_serial` reaches 1.247. Any future
   experiment that divides by a width-1 dose inherits this. The cheap fix is to stop
   deriving width-1 doses from the shapes-only curve and instead difference two arms
   that share every other dose.
5. 🔴 **Do not spend an arm on the 2-bit `:1908` readout share.** It bounds `psi_mtp`
   from below, but the only lever it justifies is a readout bit-width change, which
   the static review fails at high or critical severity. Measurement legal, proposal
   dead.
