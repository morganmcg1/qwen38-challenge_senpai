# E34 — ranked operating point and the optimal depth cap

**PR** #39 · **assignment** `qwen38-r1-e34-ranked-operating-point-depth-cap` r1
**base** `4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549` · **host** aws-mac (M4, analysis only)
**W&B** [`54wbdf3c`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/54wbdf3c)
**GPU timing runs: 0.** thorfinn holds the precision-timing slot; nothing here contends
with it. No `swift build`, no `swift test`, no benchmark wrapper. Every number is
either a replay of already-measured telemetry or an explicitly labelled model
prediction.

Reproduce:

```bash
python3 research/e34_cost_model.py --self-test        # 0 failures
python3 research/e34_ranked_operating_point.py        # ~3 min, writes the JSON
python3 research/e34_wandb_log.py
```

---

## Headline

The shipped constant `sdpaWidthWallDepthCap = 5` was chosen on a dispatch table
that no longer exists. It was raised 4 → 5 by **audreyt's `12b1c699`** (accepted,
2.92520777238747, 2026-08-16) as one arm of a six-way composite. Under the table
in force *then*, M = 5 and M = 6 both cost two weight passes, so the change
**crossed no pass boundary at all** — it was free, and it was justified as
"unlock one more committed token per round".

Our own E27 changed `itemsPerGroup` at width 5 from 3 to 5. The single-pass
region therefore grew from M ≤ 4 to **M ≤ 5**. The inherited constant now sits
**exactly one row past a cliff that did not exist when it was chosen**: a capped
round at M = 6 pays a second full weight pass to buy 0.84 accepted rows.

Cap **both** depth caps to 4 (M ≤ 5, strictly single-pass) and every cost model
I can build says the central pair moves **3.2493 → 3.7786**, `+16.3 %`.

| | value |
|---|---|
| **`e34/predicted_ranked_central_pair_at_best_cap`** | **3.77855631847542** |
| honest interval | **[3.5498, 4.0073]** |
| baseline (`0cd0a6b4`, ofou) | 3.24929398547457 |
| Δ | **+0.5293 (+16.29 %)** |
| detection threshold (your revised 2σ) | 0.16 % |

**This is a model prediction, not a measurement.** Its provenance, its three
independent cost models, its sensitivity to the inferred acceptance profile, and
the ranked evidence that cuts *against* it are all below.

---

## (a) Head artifact — settled with no run. Your prime suspicion is reversed.

`research/fetch-declared-head.sh` fetched the **declared** artifact. The two
digests you flagged as mismatched are digests of *different objects over the same
bytes*.

`~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-q2q4`:

| property | value |
|---|---|
| file count | **1** (`model.safetensors`) |
| bytes | **427,742,600** — exactly the manifest's `bytes` |
| tree digest | **`559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`** — exactly the manifest's `sha256` |
| sha256 of the file itself | `d038fd41…` |

`559b24eb` is the sha256 of the one-line **tree manifest** over that file;
`d038fd41` is the sha256 **of the file**. Same artifact. `head_verified: true`.
E25 r2/r3 ran the correct declared head.

Other trees resolved on this box, for completeness:

| tree | files | bytes | digest |
|---|---|---|---|
| `mtp-head-declared-q2q4` | 1 | 427,742,600 | `559b24eb…` ✅ declared |
| `…-q2q4-run` | 2 | 427,746,170 | hardlink + organizer `config.json` (3,570 B) |
| `mtp-head-declared` (old) | 1 | 238,937,699 | `07293af7…` |
| `mtp-head` (organizer pinned) | 5 | 849,407,066 | `c8889048…` |

**Correction owed to alphonse's E30.** Its cited head `7bbb40de…` at
**270,408,194 B** matches none of these and is not the declared artifact. Its
per-width costs (M = 8: 150.110 ms, M = 9: 164.900 ms), from which you derived
`F ≈ 14.79`, come from a different head and should not be used to size `F`. My
curve replaces it — see (b).

**So what does cause the local-vs-ranked acceptance gap (0.6926 vs 0.82+)?**
Not the head. The remaining candidate is **prompt-pool identity**: E25 used
locally-authored prose (`research/e11_prose_gate_english_512.txt`,
`research/e17_prose_<id>_512.txt`), while the hidden pool lives at R2 under
`correctness_prompts/qwen3.8-27b-mtp-v1/qwen3.8-27b-pool-<name>.json`.
**Quantifying that requires a timed run, which I did not take.** Flagging it per
your instruction rather than spending your slot.

---

## Point 7 — realised depth vs config echo. Settled by exact enumeration.

You asked me to settle it and I did, independently of the fixture read you sent
later. Both routes agree.

`(mtp_depth, mtp_max_draft_depth) == (8, 8)` on **all 408 scored rows** — one
distinct pair in the entire population. That includes:

- **5 zero-draft serial controls** whose realised n = 0 on all 8 prompts
  (`95611e60` davidtai 0.99921, `3147d255` 0xkydo, `4d1123ce` / `1f60b3fe`
  newjordan) — still reporting depth 8;
- rows realising n up to 7.64 (`45718758` uu6f8 2.4046, `00d87f8f`
  scarletbright 2.5769) — also depth 8.

A field that reads 8 for a run that drafts zero tokens is a **configuration echo
of the parent's offered ceiling**. Your fixture read (`"The candidate declares no
depth"`) confirms the same conclusion from the contract side. Both of your
competing inferences — "depth 5 ⇒ p ≈ 0.965" and "depth 8 ⇒ p ≈ 0.871" — are
unsupported, as you concluded.

---

## Identity check — it closes, and that is exactly why it proves nothing

Replay residual **4.44e-16**; beagle h̄ = **0.16452** against your 0.16453. ✅

But the identity `R = (1 + αn)/(1 + h̄n)` has one equation and two unknowns. Fix
α and h̄ is **defined** as whatever makes it close. It cannot fail, so closing it
is not evidence. Three checks show α = 0.99 is the wrong constant:

| prompt | n | h̄ | implied α | `1+αn` assumes | exact tokens/round |
|---|---:|---:|---:|---:|---:|
| plutarch | 0.1540 | **−0.5354** | 0.3333 | 1.1525 | 1.0513 |
| drama | 2.2976 | 0.3059 | 0.4491 | 3.2746 | **2.0317** |
| travel | 2.6557 | 0.2476 | 0.5329 | 3.6291 | 2.4151 |
| beagle | 4.5327 | 0.1645 | 0.8351 | 5.4874 | 4.7850 |
| medicine | 4.7677 | 0.1478 | 0.8750 | 5.7200 | 5.1717 |
| republic | 5.2697 | 0.1558 | 0.9019 | 6.2170 | 5.7528 |
| essays | 5.4253 | 0.1620 | 0.9004 | 6.3710 | 5.8851 |
| botany | 5.7765 | 0.1641 | 0.8697 | 6.7187 | 6.0235 |

1. **plutarch's h̄ is negative.** A per-row cost ratio cannot be < 0. h̄ is a
   residual, not a cost.
2. `1 + αn` overstates committed tokens per round by up to **1.243** (drama).
3. Implied α ranges **0.333 → 0.902** and is never 0.99.

I therefore did **not** build the counterfactual on this identity. I rebuilt the
ranked ledger from exact integers instead (rounds + accepted = 512 per prompt),
which is where the width distribution actually comes from.

### Ranked ledger, board top `0cd0a6b4` (exact integer reconstruction)

| prompt | mult | rounds | proposed | accepted | rate | mean M | R | round ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plutarch | 1 | 487 | 75 | 25 | 0.3333 | 1.154 | 1.2560 | 31.846 |
| drama | 3 | 252 | 579 | 260 | 0.4491 | 3.298 | 1.9231 | 40.184 |
| travel | 1 | 212 | 563 | 300 | 0.5329 | 3.656 | 2.1895 | 42.011 |
| beagle | 1 | 107 | 485 | 405 | 0.8351 | 5.533 | 3.1433 | 58.011 |
| medicine | 1 | 99 | 472 | 413 | 0.8750 | 5.768 | 3.3553 | 58.731 |
| republic | 1 | 89 | 469 | 423 | 0.9019 | 6.270 | 3.4144 | 64.065 |
| essays | 1 | 87 | 472 | 425 | 0.9004 | 6.425 | 3.3907 | 65.916 |
| botany | 1 | 85 | 491 | 427 | 0.8697 | 6.776 | 3.4491 | 66.607 |

---

## (b) The absolute curve: `F`, `S`, and whether the step resolves

Measured forced-depth round times (E25 r3, parent clock, M = depth + 1,
SEM 0.13–0.18 ms). Two separate builds; the acceptance table is bit-identical
across them over 1594 rounds, so E27 changed time and not tokens.

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| post-E27 | 68.072 | 71.368 | 78.829 | 91.721 | **108.346** | 143.872 | 156.549 | 170.321 |
| pre-E27 | 68.616 | 71.605 | 78.916 | 91.885 | **132.257** | 144.103 | 156.788 | 170.134 |

**Only M = 5 moved (−18.08 %); every other width is within ±0.8 %.** That single
column is a *causal* measurement of one weight pass:

> **one weight pass = 23.911 ms**

### Step vs smooth — the structural question you actually asked

`T(M) = b₀ + F·M (+ F₂·M²) + ceil(M/IPG)·S`

| fit | max abs residual | R² | F (ms/row) | F₂ | S (ms/pass) |
|---|---:|---:|---:|---:|---:|
| linear + **step** | **5.630** | 0.99215 | 10.612 | — | 30.797 |
| linear + smooth | 13.329 | 0.94797 | 16.112 | — | — |
| quadratic + **step** | **3.861** | 0.99610 | 6.125 | 0.612 | **25.085** |
| quadratic + smooth | 12.645 | 0.97493 | 3.885 | 1.359 | — |
| pre-E27, linear + step | 6.118 | 0.99271 | 10.172 | — | 32.378 |

**The step model wins by ~3.4× on max residual and the step term resolves far
above the 0.15 ms noise floor.** I am not able to falsify your structural claim;
a smooth model in `M` does *not* fit as well, at either linear or quadratic
order. Two independent confirmations:

- Fitted `S = 25.085 ms` vs the **causally measured** 23.911 ms — agreement
  within 5 %, from completely different information.
- The source comment at `sdpaWidthWallDepthCap` independently prices a second
  full weight pass at "~25 ms".

On your `F` bracket: my instrument puts `F = 6.125 ms/row` with `F₂ = 0.612`
(so marginal per-row cost 7.3 → 15.9 ms across M = 1..8) against `S ≈ 25 ms`.
That is **between** your two bracket rows, nearer the "F proportional to M" end
that predicted +18.8 %. It does not support alphonse's `F ≈ 14.79` — and (a)
explains why: that measurement was on a different head artifact.

### Host transfer

The local `T(M)` *shape* transfers to the ranked box with **one scalar**:
κ = **0.43728**, max residual **2.120 ms** across round times spanning
31.8–66.6 ms, R² = 0.9926. An independent ranked-only fit (design matrix uses
the board's pass structure) agrees on the structure:

| ranked fit | F | F₂ | S | R² | max resid |
|---|---:|---:|---:|---:|---:|
| linear | 3.086 | — | 17.649 | 0.99669 | 1.132 ms |
| quadratic | 2.291 | 0.216 | 13.046 | 0.99710 | 1.325 ms |

The resulting per-width round cost makes the mechanism visible:

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| board kernel | 30.55 | 33.63 | 36.72 | 39.81 | **60.54** | 63.63 | 66.71 | 69.80 | **90.53** |
| our post-E27 kernel | 30.55 | 33.63 | 36.72 | 39.81 | **42.89** | **63.63** | 66.71 | 69.80 | 72.89 |

E27 moved the cliff from M = 5 to M = 6 and deleted the third pass at M = 9.
**The wall relocated; it was not removed.**

---

## (d) Does the wall bind at the ranked operating point? Yes — decisively.

Your prior was that full-accept streaks are common, so the wall is nearly inert.
The ledger says otherwise. Two independent methods, agreeing within ~10 points:

| prompt | policy sim | max-entropy control | mean M (ledger) | mean passes |
|---|---:|---:|---:|---:|
| plutarch | 0.000 | 0.017 | 1.154 | 1.000 |
| drama | 0.003 | 0.128 | 3.298 | 1.003 |
| travel | 0.017 | 0.159 | 3.656 | 1.017 |
| **beagle** | **0.538** | 0.528 | 5.533 | 1.538 |
| **medicine** | **0.593** | 0.561 | 5.768 | 1.593 |
| republic | 0.705 | 0.649 | 6.270 | 1.705 |
| essays | 0.748 | 0.684 | 6.425 | 1.748 |
| botany | 0.969 | 0.770 | 6.776 | 1.969 |

The simplest argument needs no model at all: **botany's mean M is 6.776 from
exact integers.** A distribution with mean 6.776 cannot be supported on M ≤ 5.
The wall binds on both central prompts more than half the time.

`widthCap = fullAcceptStreak >= 2 ? segmentedVerifyDepthCap(8) : sdpaWidthWallDepthCap(5)`
— so it is the **streak path**, not the ordinary walk, that carries M above 5.
That is why **both** constants must move: `w4_s8` (cap the ordinary walk only)
recovers less than half the gain, because streak-qualified rounds still escape to
M = 9.

### On your request for a fresh timed run to read `effective_max_draft_len`

I read the counter **from disk instead of spending a GPU slot**
(`research/e25r2-timed.json`), and it shows why a new local run could not have
answered the question:

| leg | max draft len | ⇒ max M reached | mean draft len | ⇒ mean M |
|---|---|---|---|---|
| base | 4–5 | 6 | 2.204–2.692 | 3.20–3.69 |
| **candidate** | **3 (all 8 prompts)** | **4** | 1.981–2.294 | 2.98–3.29 |
| **ranked (board top)** | — | — | 0.154–5.776 | **1.15–6.78** |

The local candidate leg **never reaches M = 5, let alone the wall at M = 6**. Its
mean width is ~3.1 while the ranked prompts run at 5.5–6.8. A local run would
have measured 0.000 wall-binding fraction on every prompt and told us nothing
about ranked rounds two rows deeper — this is the same operating-point
non-overlap that correctly killed arm D, pointing the same way. The ledger
reconstruction is the better instrument here, and it is free.

---

## (c) The counterfactual, and the primary metric

Arms are `(sdpaWidthWallDepthCap, segmentedVerifyDepthCap)`. Central pair =
(beagle + medicine)/2, on **our post-E27 kernel**:

| arm | scaled local | ranked linear | ranked quadratic |
|---|---:|---:|---:|
| w3_s3 (3,3) — shape only, closed negative | 3.5932 | 3.6056 | 3.5290 |
| **w4_s4 (4,4) — strictly single-pass** | **3.6337** | **3.9302** | **3.7717** |
| w4_s8 (4,8) | 3.4526 | 3.6198 | 3.5257 |
| w5_s5 (5,5) | 3.3177 | 3.3516 | 3.3463 |
| w5_s8 — **shipped** | 3.3314 | 3.4107 | 3.3683 |
| w8_s8 — no wall | 3.3305 | 3.4329 | 3.3710 |

**`w4_s4` wins under all three cost models.** Note `w5_s5` *loses*: depth 5 still
means M = 6, still two passes. The cap must land on 4 or it buys nothing.

**Calibration of the instrument.** Replaying the *shipped* policy on the kernel
the board actually ran must reproduce the measured 3.2493. It gives
3.1859 / 3.1688 / 3.1905 — an under-prediction of **1.8–2.5 %**. That error is
the basis of the interval.

**Decomposition** (do not credit the cap with E27's own gain):

| component | scaled local | ranked linear | ranked quadratic |
|---|---:|---:|---:|
| E27 kernel alone | +0.1455 | +0.2419 | +0.1778 |
| depth cap alone | +0.3024 | +0.5196 | +0.4034 |

### Worked arithmetic, beagle (ranked-linear model)

| | shipped w5_s8 | candidate w4_s4 |
|---|---|---|
| width distribution | {3:.053, 4:.157, 5:.252, 6:.364, 7:.125, 8:.048, 9:.002} | {3:.071, 4:.171, 5:.758} |
| mean M | 5.501 | 4.687 |
| mean weight passes | 1.538 | **1.000** |
| tokens/round | 4.744 | 4.255 |
| round ms | 53.94 | 41.93 |
| R | 3.3518 | **3.8680** |

It gives up **0.489 tokens/round (−10.3 %)** and buys back **12.01 ms/round
(−22.3 %)**. That is the whole trade.

### The median actually re-ranks — I checked rather than assumed

You warned that medicine has only +0.64 % of headroom before essays displaces it.
The published score is the median of **all eight** prompts, so I re-ranked every
arm instead of trusting the central pair:

| arm | median (ranked linear) | central prompts |
|---|---:|---|
| w5_s8 shipped | 3.3795 | beagle, **botany** |
| **w4_s4** | **3.9302** | **beagle, medicine** |

Under the *shipped* arm on our kernel the central pair is already **not**
beagle+medicine in **all three** models (it is beagle+botany) — so "beagle +
medicine and nothing else" is a property of the measured row, not of the metric.
Under `w4_s4` it is, because the arm helps the wide prompts in proportion to how
much they were paying the second pass, and beagle/medicine gain the **least** of
the five wide prompts:

| prompt | measured | w5_s8 | w4_s4 | ratio |
|---|---:|---:|---:|---:|
| plutarch | 1.2560 | 1.3414 | 1.3414 | ×1.000 |
| drama | 1.9231 | 2.0445 | 2.0465 | ×1.001 |
| travel | 2.1895 | 2.3990 | 2.4201 | ×1.009 |
| **beagle** | 3.1433 | 3.3518 | 3.8680 | ×1.154 |
| **medicine** | 3.3553 | 3.4695 | 3.9925 | ×1.151 |
| republic | 3.4144 | 3.7018 | 4.1791 | ×1.129 |
| essays | 3.3907 | 3.7047 | 4.2023 | ×1.134 |
| botany | 3.4491 | 3.4072 | 4.1874 | ×1.229 |

The narrow prompts are untouched (×1.000–1.009) because they never reach M ≥ 6.
`central_pair_minus_median = 0.0` exactly at the best arm. **The metric does not
overstate the score, and the arm does not trade medicine against beagle.**

### Sensitivity to the inferred acceptance profile

`p` is inferred, not measured, so I swept every profile that reproduces the
ledger's n, accept rate and round time within 5 %:

| prompt | admissible profiles | relative gain of w4_s4 | median |
|---|---:|---|---:|
| beagle | 9 | **+5.93 % … +13.14 %** | +10.13 % |
| medicine | 6 | **+5.75 % … +10.77 %** | +8.47 % |

**The sign is robust across every admissible profile.** The magnitude is not:
the low end is roughly half the headline. Against your revised 0.16 % detection
threshold, even the floor is ~36σ.

---

## Evidence that cuts AGAINST this result

I want this on the record before anyone spends a ranked slot.

**1. No ranked row has ever executed our post-E27 dispatch table.** Every board
observation of "shallower is worse" was made on a kernel where M ≤ 5 saves no
pass at all. Under that kernel my model *also* says capping is worthless. The
board's null is exactly what I predict for the board's kernel — which means the
board cannot corroborate the arm either.

**2. Deeper genuinely beat the default on medicine.** Within head `559b24eb`:

| cohort | rows | median accept | best R | vs default 3.3553 |
|---|---:|---:|---:|---|
| default n = 4.7677 | 117 | 0.8750 | 3.3553 | — |
| **deeper** | 202 | 0.9091 | **3.4095** (`7b46db6d` jonathan308, n = 5.679, 11.1493 ms/tok) | **+1.6 % better** |
| shallower | 89 | 0.9037 | 3.3218 | −1.0 % worse |

The shallower cohort is a *valid* control here and it is mildly **unfavourable**
to my arm.

**3. On beagle the shallower cohort is not a valid control.** 264 shallower rows
have median accept 0.8261 < the default's 0.8351 — they are shallow *because
acceptance is worse*, losing tokens without saving a pass (12.2139 vs 12.1233
ms/tok). Reading them as a cap experiment would be a confound.

**4. From the note corpus (629 submissions), mined and spot-verified:**

- **No clean scored A/B of this constant exists.** The only single-variable
  4 → 5 probe, `90c139e5` (ggu77wt), **failed** at the submission-verification
  gate and was never scored. Both cap-containing composites bundle 4–6 other
  changes.
- **`9100a4e7` (Lieisyourlie, 3.0783, rejected) explicitly claims the cap is not
  binding** — *"the first-round walk never reaches that fifth draft … the cap is
  not the binder."* That is right about the *ordinary* walk and wrong about the
  round distribution: the streak path is what carries M to 6–9, and botany's
  mean M of 6.776 is arithmetic, not modelling.
- **`43f655f1` (newjordan) named this exact idea once** — *"cap depth so rounds
  pay 1 weight stream instead of 2"* — and discarded it as already-landed,
  conflating it with the hard depth cap 4 whose rationale is exactness. **It was
  never measured.**
- **`40864c53` (zhangshuibai) falsified a naive traffic model** (2→3 groups costs
  ~1 ms, not 50 %; threadgroups are cache-served). My `S` is not a traffic
  estimate — it is the measured 23.911 ms delta of a single column — so this does
  not bear on it, but it is the right kind of scepticism.

**5. The interval is honest, and it is wide.** 1.8–2.5 % systematic
under-prediction on the one arm with a known truth value, plus a 0.16 → 0.56
spread across cost models at their own best arm, gives **[3.5498, 4.0073]**.

---

## (e) The durable artifact

`research/e34_cost_model.py` — a cost model that **reads `ceil(M/IPG)` from the
dispatch table at runtime** rather than pricing depth smoothly:

- `dispatch_ipg()` parses the live `itemsPerGroup` table out of the MLX source,
  so it is correct under *both* kernels and will follow thorfinn's E33
  automatically;
- `weight_passes(M)` / `rows_per_pass(M)` expose the step;
- `StepCostModel` + `fit_cost_model(quadratic=, use_passes=, table=, passes=)`
  fit `b₀ + F·M + F₂·M² + ceil(M/IPG)·S` with a dependency-free least squares;
- `PolicySim` reimplements `costModelDepth` / `recordAcceptOutcome` exactly,
  including the EMA prior, the optimism cap and the streak ladder;
- `--self-test` asserts the structural claims (step beats smooth; curvature
  improves the step fit; max residual < 0.25·|S|) and currently reports
  **0 failures**.

The tables it produced, which are the reason the headline is a *relocation*
rather than a coefficient:

```
474c750 (last accepted competitor snapshot, pre-E27)
  IPG {3:3, 4:4, 5:3, 6:3, 7:4, 8:4, 9:3}
  passes M=1..9:  1 1 1 1 2 2 2 2 3     → single-pass top M=4 (depth 3)

HEAD (post-E27, commit 0207de6, ours)
  IPG {3:3, 4:4, 5:5, 6:3, 7:4, 8:4, 9:5}
  passes M=1..9:  1 1 1 1 1 2 2 2 2     → single-pass top M=5 (depth 4)
```

`0207de6` is an ancestor of base `4e5dc2b`. (An earlier note of mine inferred the
pre-E27 table from `0207de6^` = `7b5183d`; that is the *discarded probe arm*
`{4:2, 6:4, 8:3}`, not the competitor baseline. `474c750` is the correct
predecessor and it reproduces the measured tape exactly.)

**Your independent step evidence lines up with this.** Our per-prompt deficit vs
#1 is flat at −0.019 ± 0.073 % for n < 3 and +0.380 ± 0.056 % for n > 4.5 — a
step at exactly the width where the second pass switches on. A smooth price in
`M` cannot generate that; `ceil(M/IPG)` does, from a different dataset than mine.

**Diff scope:** `research/` only. I did **not** add the env-gated model to
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`. Landing dead code on the
scored surface that I cannot compile or smoke-test under the zero-GPU constraint
is a worse trade than shipping the Python, and the Swift constant change this
work argues for is a two-literal edit that belongs in its own measured
experiment. Shipped surface `git diff` is empty.

---

## Result label

**Unclear, leaning local winner — and it is not mine to promote.** The mechanism
is measured (23.911 ms, causal, one column), the structure is measured (step
beats smooth 3.4×, two independent datasets), and the direction is robust across
15 admissible acceptance profiles and 3 cost models. But the *size* rests on a
counterfactual whose only calibration point under-predicts by 1.8–2.5 %, and the
one honest ranked control (medicine, shallower cohort) points mildly the other
way. **This is a well-founded hypothesis with a large predicted effect, not a
measured win.**

## Suggested follow-ups (not implemented)

1. **The decisive experiment is two literals.** `sdpaWidthWallDepthCap 5 → 4`
   and `segmentedVerifyDepthCap 8 → 4`, one `--local-submit` matched pair against
   a fresh base, then rank. Both must move; `w4_s8` and `w5_s5` are predicted
   near-nulls and make good negative controls. Predicted +16 %, floor +6 %, and
   the 0.16 % threshold makes even the floor unmissable.
2. **Re-derive `headStepCostRatio` instead of tuning it.** `h = 0.18` is a smooth
   per-row price that provably cannot express a step. With `F`, `F₂` and `S`
   measured, the marginal-depth rule can consult `ceil(M/IPG)` directly and `h`
   becomes a derived quantity. This is the general form of the thing that
   bracketed h at 0.18 by three failed probes.
3. **Re-run this analysis if E33 lands.** If M = 6..9 become single-pass, the
   optimum moves back **up** to 8 and this arm becomes actively harmful. The tool
   reads the table at runtime, so it is one command — but somebody has to run it.
4. **Close the local acceptance gap by pool identity**, not by head. One timed
   run on the R2 hidden-pool prompts would tell us whether any local fixture can
   test a depth policy at ranked `p`. Right now none can.
