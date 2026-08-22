# E122 rung 0 — the target top-2 margin as a draft-depth conditioner

**Verdict: decisive null. The pre-registered gate kills the axis.**

The pooled prompt-stratified concordance is **0.5109 [0.4850, 0.5364]**. The
pre-registered kill threshold is 0.55. The whole confidence interval sits below
0.55 and it contains 0.5. Rung 1 is not licensed.

The hypothesis was that the target's own top-2 margin — a number the host
already reads every round, at zero extra cost — predicts the next round's
draft acceptance well enough to allocate depth per round, and so captures part
of the 2.88–3.95 % gated-oracle pool E99 measured. The oracle pool is real and
large here too (**31.16 %** of modelled decode cost). A margin-only conditioner
captures **0.73 % held out, which is 2.5 % of that pool**, and **0.00 % on the
hard-end prose proxy**. The margin is not the variable that unlocks the pool.

`harness=local`. `timing_valid=false`. `cool_gate_passed_real_gate=false`.
`gate_qualified_for_timing=false`. `official_or_ranked_score=false`. No timing
claim is made anywhere in this result. Rung 0 measures a correlation and prices
it with a cost model; it does not measure seconds.

---

## 1. Identity tuple

| field | value |
| --- | --- |
| campaign `BASE_SHA` | `2127858ba770ddc06027205d8df89a8db21d80f5` (contains `xv4`) |
| branch | `qwen-edward/e122-target-margin-conditioned-depth` |
| host | `ip-10-231-2-12.ec2.internal`, Apple M4 Pro, 48 GiB |
| host profile | `apple-m4-pro-applegpu_g16s-20core-48gib` |
| worker, shipped arm | sha256 `d6dbb3d42ae97da7aa49b3e90d66b81f0ddb7c02c8967053498330e6d4b2367b` |
| worker, forced arm | sha256 `1f792c5131c3ad51733e6a945bc8914b65a5228f30c83793a8719b66540e7c54` |
| proposal head, as the legs report it | `head_provenance_sha256 dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`, 427,746,170 bytes, 2 files — **identical on all 18 legs of both arms**, checked programmatically, not copied forward |
| proposal head, as `mtp-head.manifest.json` pins it | sha256 `559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71`, 427,742,600 bytes |
| the two digests agree on the artifact | both carry origin `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae6282749a52e052496dd5300b4aa441df7301e8` — the same repo at the same revision. They differ because they digest different objects: the manifest hashes a one-file tree, the runtime hashes the two-file directory it loaded. Same convention as E93 and the scored `score.json` rows. |
| token window | 512 decode tokens, every leg, both arms |
| offered depth | 8 |
| reference source | per-prompt golden replay, `.mlxfast-private/e122/goldens/<id>-rows-513.json` |
| leg kind | `mtp-timed`, one leg per process (harness defect 20) |

The candidate surface state at each run commit, checked against `BASE_SHA`:

```
1e681d1d  Sources vs BASE:  (identical)      <- shipped arm, benchfixture
c3a0fd69  Sources vs BASE:  (identical)      <- shipped arm, 8 prose proxies
9b1de4df  Sources vs BASE:  1 file, +31      <- forced arm, the instrument
8ea4b33f  Sources vs BASE:  (identical)      <- this result tree
```

The shipped arm therefore ran the promoted base runtime exactly. The forced arm
ran the base runtime plus a 31-line depth pin and nothing else. The result tree
proposes **no candidate change**: the branch adds files under `research/` only.

`meta.base_sha` inside each run record is the branch HEAD at run time, not the
campaign base. The table above is the mapping.

Every identity claim above was re-derived from the 18 run records at submission
time rather than carried forward in prose:

```
legs                                                  18
worker_sha256 d6dbb3d4… (shipped) / 1f792c51… (forced)  9 / 9
head_verification asserts the manifest digest         18 / 18   <- fetch-declared-head.sh:
                                                                   "declared head verified …
                                                                   427742600 bytes, sha256 559b24eb…"
head_manifest_tree_sha256 == mtp-head.manifest.json   18 / 18
head_provenance_sha256 identical across legs          18 / 18   (dadbfb80…)
timing_valid=false                                    18 / 18
dirty_candidate_paths=0                               18 / 18
one non-empty trace.txt per leg (harness defect 20)   18 / 18
rebuild-and-assert-worker.sh --self-test                 PASS   (needle 1/0/1 as required)
```

Gates, re-verified on the result tree `07ba3869` immediately before submission:

```
verify-ranked-score-boundary.sh   PASS: ranked numerator is pinned baseline;
                                  candidate edits affect the MTP denominator only
check-editable-budget.sh <BASE>   editable budget OK: source=2556068/3000000
                                  headroom=443932  growth=0/262144  files=154
rebuild-and-assert-worker.sh --self-test    PASS (section 8, symbol-table block)
```

`validate-assignment-scope.sh` takes the paths a run would *submit to Yukon*.
This branch submits none: `git diff BASE..HEAD` touches `research/` only, and
`git diff BASE..HEAD -- Sources Vendor Tests docs fixtures benchmark.json
benchmark.qwen-mtp.json mtp-head.manifest.json Package.swift tools` is empty.
Handed the branch diff, the gate correctly reports each `research/…` path as
outside `editablePaths` — research notes are not submission surface. There is
no submittable path here to pass or fail the gate on, and I am not quoting it
as a PASS.

`senpai/run-all-gates.sh` on the result tree: **26 run, 11 failures, 2
did-not-run.** The identical suite on `BASE_SHA` in a clean worktree gives the
**same 11 failures and the same 2 did-not-run**, name for name (diff of the two
outcome lists is empty). The failures are base-level state — `base-drift`
(recorded base advanced by 1 commit), `campaign-invariants` (an overlay-reverted
`sums[m] += …` line), the surface and self-test gates — and none of them is
caused by this branch, which cannot be otherwise since the candidate surface is
byte-identical to the base. I am not quoting a clean bill of health.

---

## 2. What was measured, and why the forced arm is the clean one

Two arms, nine local prompt proxies each, 512 decode tokens each, one leg per
process.

**Shipped-policy arm** (9 prompts, 1956 rounds). The base runtime chooses depth
with its own margin-derived rule. Its per-prompt `effective_max_draft_len` came
out at 4, 5 or 7 depending on the prompt. Depth is therefore *chosen by* the
margin, so acceptance at position *k* is only observed when the policy already
believed the margin was high enough to draft that far. Any concordance measured
on this arm is contaminated by that selection. This arm is reported for
description only, as the advisor directed.

**Forced-depth arm** (9 prompts, 1761 rounds). A 31-line instrument pins the
requested depth to 8 on every round. Every prompt then reports
`effective_max_draft_len = 7` — uniform, and inside the shipped
`segmentedVerifyDepthCap`. Acceptance at position *k* is observed on every
round that reached position *k*, independent of the margin. This is the arm the
gate reads.

The instrument lives in the tree as a reverse-appliable patch,
`research/e122-patches/forced-depth.patch`, plus its commit `7a64da39`. Commit
`8ea4b33f` reverse-applies it, which is why the result tree's `Sources/` is
byte-identical to `BASE_SHA`.

The E116 patch discipline, checked on the result tree at submission time:

```
git apply --check    research/e122-patches/forced-depth.patch   exit 0
git apply --check -R research/e122-patches/forced-depth.patch   exit 1
```

Both directions are the correct answers and the pair is the assertion. Exit 0
forward says the instrument re-applies cleanly, so the arm is reproducible from
the committed tree. Exit 1 reverse says there is nothing to remove, so the
instrument is genuinely absent from what I am handing over. A tree that passed
*both* checks would be one where the patch was a no-op.

### The gate statistic

Pre-registered and then revised by the advisor to the power-pooled form:

1. Take the forced arm, positions 2 to 5.
2. Inside each `(prompt, position)` cell, count concordant, discordant and tied
   (margin, accepted) pairs.
3. Sum the counts over all 36 cells.
4. Form the ratio once, at the end.
5. Cluster-bootstrap over rounds within prompt, 2000 replicates, seed 122.

Thresholds: **≤ 0.55 kill**, **≥ 0.65 license rung 1**, **0.55–0.65 stop and
ask the advisor**.

Pairing inside a cell is what stops between-prompt margin scale from posing as
signal. A prompt whose margins are large *and* whose acceptance is high will
raise an unstratified statistic without any within-prompt predictive power.
Section 5 shows that this is exactly what happens here.

---

## 3. The gate result

```
## PRIMARY GATE -- prompt-stratified concordance, positions 2 to 5
   pooled AUC 0.5109 [0.4850, 0.5364]   pairs 57907  cells 36
   concordant 28113  discordant 26850  tied 2944
   Somers D from the gate pairs      +0.0218   (= 2 x AUC - 1)
   Somers D of margin vs run length  +0.0284   (the agreement check)
   agreement |delta D| 0.0066  OK
   VERDICT  kill: at or below 0.55
```

The agreement check is the advisor's guard against a statistic that measures
the wrong thing. Somers D computed from the gate's per-position pairs and
Somers D of margin against the whole round's accepted run length must agree to
within 0.10. They agree to 0.0066. The gate is measuring the quantity it claims
to measure, and that quantity is near zero.

### Per prompt

```
   prompt              pairs      AUC   D_gate  D_runlen  spearman  verdict
   benchfixture          577   0.7998  +0.5997   +0.2198   +0.4474  licensed
   dramatic             6657   0.5638  +0.1277   +0.0657   +0.0964  ask
   english              7215   0.5244  +0.0488   +0.0085   +0.0101  kill
   medicine             6522   0.5171  +0.0342   +0.0486   +0.0711  kill
   narrative            7677   0.4836  -0.0328   +0.0086   +0.0119  kill
   natural_history      7212   0.5204  +0.0408   -0.0033   -0.0066  kill
   philosophy           7565   0.4916  -0.0168   -0.0090   -0.0124  kill
   technical            6844   0.4977  -0.0047   +0.0277   +0.0377  kill
   travel               7638   0.4744  -0.0512   +0.0242   +0.0362  kill
```

Seven of nine prompts kill. One asks. One licenses, and that one is
`benchfixture` — the public long-copy gate fixture, where the model is copying
text it has already seen. Its accepted/drafted rate is 0.8642 against 0.17–0.24
for the prose proxies and its mean margin is 11.8 against 0.99–1.40. It is a
degenerate regime, not a preview of the hidden prompts.

**`natural_history` does not overrule the pool.** The advisor's rule was that
the hard-end proxy wins if it disagrees with the pool *in direction*. Its gate
AUC is 0.5204, on the same side of 0.5 as the pool's 0.5109, and its verdict is
the same kill. There is no override and no notification trigger. Its Somers D
against run length is very slightly negative (-0.0033) while its gate D is very
slightly positive (+0.0408); both are inside noise and neither crosses into
"the margin predicts acceptance".

Neither of the advisor's two notification triggers fired: the pooled statistic
is not in 0.55–0.65, and `natural_history` does not disagree in direction with
the pool.

---

## 4. Per-position table, forced arm

```
   pos    n  rej  not_drafted  acc_rate   AUC(raw)   [95% CI]         AUC(strat)   [95% CI]         AUC(z)  prompts
     1  1761  541            0    0.6928    0.5404 [0.5114, 0.5695]      0.5166 [0.4850, 0.5466]   0.5243      9
     2  1217  496            4    0.5924    0.5465 [0.5141, 0.5772]      0.5092 [0.4747, 0.5415]   0.5241      9
     3   719  311            6    0.5675    0.5853 [0.5432, 0.6263]      0.5154 [0.4703, 0.5599]   0.5441      9
     4   406  189           12    0.5345    0.6185 [0.5632, 0.6714]      0.4968 [0.4317, 0.5589]   0.5431      9
     5   216   92           15    0.5741    0.7518 [0.6874, 0.8144]      0.5891 [0.4791, 0.6905]   0.6150      9
     6   124   35           18    0.7177    0.7846 [0.6977, 0.8586]      0.5208 [0.3896, 0.6789]   0.5822      9
     7    89   19           20    0.7865    0.8327 [0.7360, 0.9149]      0.6354 [0.2500, 0.9065]   0.5628      7
```

`not_drafted` is the column that motivated the arm, and on this arm it is dead.
The drafted-depth histogram over all 1761 forced rounds is

```
d=1: 4   d=2: 2   d=3: 6   d=4: 3   d=5: 3   d=6: 2   d=7: 1741
```

so 1741 of 1761 rounds (98.9 %) draft the full 7. I checked what the other 20
are instead of assuming: **all 20 sit in the last one to four rounds of their
leg** — `benchfixture` round 73 of 73, `philosophy` rounds 216-219 of 219, and
so on. They are the 512-token budget closing the window, not the margin
selecting the sample. There is no round anywhere in the arm where a *low margin*
shortened the draft, which is exactly the property the shipped arm lacked.

`n` still falls from 1217 to 216 across positions 2 to 5, for the ordinary
reason: a round reaches position *k* only if positions 1 to *k*-1 were accepted.
That attrition is conditioning on the *outcome*, not on the predictor, so it
costs power but does not restrict the margin's range. And the rejection counts
at the four gated positions are now 496, 311, 189 and 92, against the three to
five the shipped arm offered. That is what makes the pooled statistic decidable
where the original per-position gate was not.

within-prompt spearman(margin, accepted) = **0.0460**

This table is the clearest single piece of evidence in the experiment.

Raw AUC climbs from 0.54 at position 1 to 0.83 at position 7 and would look
like a real, strengthening signal. Stratified AUC stays flat near 0.5 at every
depth, and **every stratified confidence interval contains 0.5**. The climb is
entirely between-prompt: deep positions are reached almost only by
`benchfixture`, whose margins are an order of magnitude larger and whose
acceptance is near 1. Pooling raw pairs across prompts lets that one prompt's
scale masquerade as discrimination.

`AUC(z)`, on the within-prompt standardised margin, is a third view. It tracks
`AUC(strat)` and stays in 0.52–0.62. It is reported as a diagnostic only; the
gate does not use it.

---

## 5. Margin distributions, and the scale-free threshold question

Forced arm, per prompt:

```
prompt                mean      sd     p10     p50     p90      cv  acc/drafted
benchfixture        11.815   5.794   1.250  14.250  16.600   0.490      0.8642
dramatic             1.397   1.749   0.125   0.875   2.750   1.253      0.2374
english              1.260   1.365   0.125   0.875   2.769   1.084      0.1937
medicine             1.191   1.206   0.125   0.875   2.637   1.012      0.2243
narrative            1.035   1.293   0.125   0.625   2.500   1.249      0.1948
natural_history      0.992   1.043   0.125   0.625   2.250   1.052      0.1707
philosophy           1.238   1.279   0.125   0.875   2.750   1.033      0.1928
technical            1.213   1.343   0.125   0.750   2.500   1.107      0.2272
travel               1.172   1.264   0.125   0.750   2.325   1.078      0.2109
```

The assignment asked whether one scale-free threshold could work across
prompts. The answer has two halves.

**Across prose, yes, and it does not matter.** The eight prose proxies cluster
tightly: means 0.99–1.40, p50 0.625–0.875, p10 identical at 0.125, coefficient
of variation 1.01–1.25. One fixed threshold in natural units transfers across
all eight without any normalisation. That is a real property of the
distribution. It is also useless, because there is nothing on the other side of
the threshold to gain: the within-prompt discrimination is ~0.5.

**Between prose and `benchfixture`, no.** A 10x scale gap means any threshold
tuned on one side is a constant on the other. That gap is the mechanism behind
the raw-vs-stratified split in section 4.

`natural_history`, the hard-end proxy, has mean margin 0.992, close to E114's
board-fitted beagle μ of 0.902. That supports the proxy mapping the advisor
approved, and it is the reason `natural_history` gets its own row and its own
verdict rather than being folded into a prose average.

### The margin arrives on a 0.0625 grid

New observation, and the most likely mechanical reason the axis is dead:

```
prompt            distinct  distinct below 2.0  min gap  exact grid
benchfixture         53          8 / 10 rounds   0.0625      0.0625
dramatic             40         18 / 150         0.0625      0.0625
english              48         23 / 176         0.0625      0.0625
medicine             36         18 / 163         0.0625      0.0625
narrative            40         22 / 188         0.0625      0.0625
natural_history      41         23 / 205         0.0625      0.0625
philosophy           40         16 / 180         0.1250      0.1250
technical            37         16 / 160         0.1250      0.0625
travel               37         19 / 174         0.0625      0.0625
```

`natural_history` is the sharpest case: 234 rounds carry 41 distinct margin
values, and the 205 rounds below 2.0 — where every depth decision would
actually be made — carry only **23 distinct levels**. Its smallest values are
`0, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.8125,
0.875`. Every observed margin on every prompt is an exact multiple of
**0.0625 = 2^-4**.

Three candidate explanations were ruled out:

1. **Not the trace format.** The trace writes `m=%.6f`, six decimals. It can
   print 0.001-scale differences and never does.
2. **Not the reducer.** `linearTopTwoRows` declares `outputDTypes: [.int32,
   .float32]` and its Metal state struct holds `float first_value` /
   `float second_value`. The reducer widens; it cannot coarsen.
3. **Not the host read.** `top2Values.asArray(Float.self).map { Double($0) }`
   widens float32 to double.

The remaining source is the logits array the reducer consumes.
`Qwen35Config` asserts `expect("dtype", dtype, "bfloat16")`. bf16 carries 8
significand bits, so its ULP is 2^-4 = 0.0625 for values in [8, 16) and 2^-3
for [16, 32). Every observed margin being an exact multiple of 2^-4, with a
minimum observed gap of exactly 2^-4 and nothing finer anywhere in 1761 rounds,
is what a difference of two bf16 logits in that magnitude range produces. I did
not read the dtype off a live array, so I state this as a well-supported
inference rather than a measurement: **the margin is a
catastrophic-cancellation difference of two bf16 numbers, and it reaches the
scheduler carrying roughly 4 to 5 bits in the range where every decision is
made.**

This is a satisfying explanation for the null, and it is worth saying plainly
that it is a *post hoc* explanation. It was found after the gate had already
fired, not before.

---

## 6. Value model — pricing the conditioner, not just its AUC

An AUC near 0.5 is a strong signal but not a decision. A weak predictor can
still pay if the cost asymmetry is steep enough, and a strong predictor can
still fail to pay if the oracle pool is small. `research/e122_value.py` prices
the conditioner directly under the shipped cost model, `T(d) = V * (1 + 0.18d)`
with `headStepCostRatio = 0.18`, and tokens per round `= 1 + min(L, d)` where
`L` is the round's true accepted run length.

Four policies are compared, all in units of verify-forwards per token:

- **const** — the single best fixed depth `d*`, chosen on the data.
- **margin** — depth chosen from the margin quintile, fitted in sample.
- **oracle** — depth chosen with knowledge of the round's true `L`. The
  ceiling.
- **held** — the margin policy fitted on one half and scored on the other.

```
scope                  n   meanL  d*    const   margin   oracle   pool%  margin%   held%  share%
pooled              1761   1.618   2   0.6469   0.6419   0.4932   31.16     0.78    0.73     2.5
benchfixture          73   6.014   7   0.3222   0.3159   0.2969    8.53     1.99    0.98    23.3
dramatic             193   1.653   2   0.6449   0.6363   0.4891   31.86     1.35   -1.72     4.2
english              218   1.349   2   0.6723   0.6723   0.5291   27.05     0.00   -1.14     0.0
medicine             200   1.560   2   0.6602   0.6490   0.5003   31.96     1.73    0.99     5.4
narrative            218   1.353   2   0.6545   0.6545   0.5285   23.85     0.00   -0.33     0.0
natural_history      234   1.188   2   0.6948   0.6948   0.5548   25.25     0.00    0.00     0.0
philosophy           219   1.338   2   0.6461   0.6461   0.5307   21.73     0.00   -1.38     0.0
technical            198   1.586   2   0.6411   0.6320   0.4971   28.97     1.45    1.88     5.0
travel               208   1.466   2   0.6429   0.6429   0.5125   25.45     0.01   -0.71     0.0

pooled margin bin edges  ['0.2500', '0.6250', '1.1250', '2.0000']
pooled bin depths        [2, 2, 2, 2, 3]     held-out bin depths [2, 2, 2, 2, 3]
run length histogram     {0:541, 1:499, 2:313, 3:191, 4:93, 5:35, 6:19, 7:70}
right-censored rounds    78
```

`pool%` is the oracle's improvement over the best constant depth. `margin%` and
`held%` are the margin policy's improvement over the same constant, in sample
and held out. `share%` is `held% / pool%`.

Read it this way:

- **The prize is real.** A perfect per-round depth oracle saves 31.16 % of
  modelled decode cost against the best fixed depth. Nothing here argues
  against adaptive depth as an idea.
- **The margin does not collect it.** 0.73 % held out, 2.5 % of the pool.
- **On five of nine prompts it collects exactly nothing.** `english`,
  `narrative`, `natural_history`, `philosophy` and `travel` all fit
  `margin% = 0.00`: the fit chose a flat depth vector, because no margin bin
  split improved on the constant.
- **Four prompts have negative held-out gains.** The in-sample fit on
  `dramatic`, `english`, `narrative`, `philosophy` and `travel` is
  overfitting; on the other half it loses.
- **The fitted bin depths are almost flat.** `[2,2,2,2,3]` in sample and
  `[2,2,2,2,3]` held out. After fitting five margin quintiles freely, the
  optimiser essentially reproduces the constant policy `d* = 2`.
- **`natural_history` collects 0.00 % of a 25.25 % pool.** On the hard-end
  proxy, the conditioner is exactly worthless.
- The only prompt with a real share is `benchfixture` at 23.3 %, and its own
  pool is the smallest of any prompt at 8.53 %, because at accept-rate 0.86 the
  best constant depth is already near-oracle.

Depths are censored at 7 by `segmentedVerifyDepthCap`, so 78 rounds with `L = 7`
are right-censored. The estimator uses `min(L, d)`, so censoring can only
understate the oracle pool. It cannot inflate the margin policy's share.

### Positive control

The value estimator has a `--self-test` that runs it on two synthetic
populations:

```
null     d*=2 oracle_pool=32.56% margin_in_sample=0.00% margin_held_out=0.00%  bins=[2,2,2,2,2]
perfect  d*=2 oracle_pool=32.66% margin_in_sample=24.11% margin_held_out=23.90% bins=[0,0,1,2,4]
e122_value self-test: PASS
```

On a margin drawn independently of acceptance it reports 0.00 % and a flat bin
vector. On a margin that determines the run length exactly it recovers 23.90 %
of a 32.66 % pool and a monotone bin vector `[0,0,1,2,4]`. The estimator can see
signal when signal is there. On the real data it reports 0.73 % and `[2,2,2,2,3]`.

The gate has its own positive control built in: `benchfixture` returns AUC
0.7998 and D +0.5997 through the same code path that returns 0.5109 pooled. The
pipeline can fire.

---

## 7. Pre-registration, and where I was wrong

`research/e122-preregistration.md` was committed as `f5bd9424` at
**2026-08-22T04:19:34Z**, before any discrimination statistic was computed. Only
validity fields were inspected first: token match, parity, residual divergence
and `effective_max_draft_len`.

| quantity | predicted | 95 % range | observed | inside range? |
| --- | --- | --- | --- | --- |
| pooled stratified concordance | 0.55 | [0.53, 0.58] | **0.5109** | **no, below** |
| Somers D | 0.10 | [0.06, 0.16] | +0.0218 | no, below |
| within-prompt Spearman | 0.15 | [0.05, 0.25] | 0.0460 | no, below |

All three came in below my pre-registered range. I predicted a weak but real
signal sitting right on the kill line, and the truth is weaker than my lower
bound on every statistic. The direction of my error is worth recording: I had
seen the shipped arm's raw per-position AUCs climbing to 0.83 and, despite
knowing the selection confound, I still anchored high. Pre-registering the
number is what makes that anchoring visible instead of invisible.

The gate thresholds themselves were pre-registered and were not moved after the
result.

---

## 8. Correctness evidence

All 18 legs, both arms:

```
reports                     18
all_tokens_matched          true   18 / 18
parity_all_ok               true   18 / 18
residual_divergence_count   0      18 / 18
```

Every leg generated 512 decode tokens and replayed bit-identically against its
own per-prompt golden. The forced arm pins depth to 8 and still matches token
for token, which is the point: forcing depth changes the schedule, never the
output. The instrument touches draft-count selection only.

`effective_max_draft_len` per leg:

```
shipped  benchfixture 7   dramatic 5  english 4  medicine 5  narrative 5
         natural_history 4  philosophy 5  technical 4  travel 4
forced   7 on all nine prompts
```

The forced arm is uniform, as intended, and sits inside the shipped
`segmentedVerifyDepthCap = 7` rather than pushing past it. Commit `756901df`
made that explicit.

### Swift test suite on the result tree

`swift test --force-resolved-versions` on `8ea4b33f`: **687 tests passed, 40
issues in 733 tests across 66 suites.**

The 40 issues are pre-existing on `BASE_SHA` and are not caused by this branch.
The argument is by construction: the branch's complete diff against `BASE_SHA`
adds files under `research/` and changes nothing else — no `Sources/`, no
`Tests/`, no docs, no fixtures, no workflow, no manifest. The failing suites
read `README.md`, `TASK.md`, `AGENTS.md`, `CLAUDE.md`, the fixture pins and the
artifact manifest, and none of them reads `research/`. The compiled test binary
is identical to the one `BASE_SHA` produces.

The failures fall into four groups, all base-level drift:

1. `BenchmarkScriptTests` / `SetupScriptTests` — the campaign fork's
   `<!-- SENPAI-CAMPAIGN-BEGIN -->` documentation block.
2. `QwenMTPTrackNamingTests` — fixture pins now carry real digests where the
   test still expects the `QWEN38-PENDING-RELEASE` placeholder.
3. `Qwen35ArtifactContractTests` — `config.json` digest in the reference
   manifest.
4. `RuntimeStartupMemoryPolicyTests` — asserts
   `maxMegabytesPerCommandBuffer == 320` and
   `maxOperationsPerCommandBuffer == 128`; the promoted base runs 512 and 50.

Group 4 is the one worth the advisor's attention: it is a stale assertion
against a promoted runtime change, not a documentation mismatch. It is listed
as a follow-up and was not touched here.

The exactness suites all passed, including
`E84ReplayStateKernelExactnessTests`, `E84IslandDeadWorkExactnessTests`,
`boundaryStateIsBitIdenticalAtEveryReplayWidth` and
`qOnlyPackIsBitIdenticalToTheFullPackQRows`.

### The instrument is gone from the built worker

Rebuilt on `8ea4b33f` and asserted by symbol table:

```
worker_sha256 cf674507a36443511fb635282027568aa303ad5f6211fdc3352c3e76486894de
ok   require-symbol 'warmAllDepthShapes':      22
ok   require-symbol 'snapshotScheduleSignal':   2
ok   forbid-symbol  'e122ForcedDepth':          0
rebuild-and-assert-worker: PASS
```

The digest differs from the shipped arm's `d6dbb3d4…` even though the source
tree is identical, which is the effect `senpai/rebuild-and-assert-worker.sh`
documents in its own header: a release-build digest tracks link-time layout,
not source content, and is not a valid arm certificate. The symbol-table
assertion is the certificate. `e122ForcedDepth` is absent from a tree that
previously carried 3 occurrences of it.

---

## 9. Reproduction

```bash
# both arms, nine prompts each, 512 decode tokens, one leg per process
research/e122_rung0_session.sh                       # shipped policy arm
E122_FORCE_DEPTH=8 E122_RUNS_DIR=.mlxfast-private/e122/runs-forced \
  research/e122_rung0_session.sh                     # forced depth arm

# the gate and the per-position analysis
python3 research/e122_auc.py .mlxfast-private/e122/runs-forced/* \
  --json research/e122-artifacts/rung0-forced.json
python3 research/e122_auc.py .mlxfast-private/e122/runs/* \
  --json research/e122-artifacts/rung0-shipped.json

# the value model, and its positive control
python3 research/e122_value.py .mlxfast-private/e122/runs-forced/* \
  --json research/e122-artifacts/rung0-value.json
python3 research/e122_value.py --self-test
```

The forced arm needs the instrument applied first:

```bash
git apply research/e122-patches/forced-depth.patch
senpai/rebuild-and-assert-worker.sh --require-symbol e122ForcedDepth
```

Bootstraps use 2000 replicates and seed 122, so the reported intervals are
reproducible exactly.

Artifacts, committed: `research/e122-artifacts/rung0-forced.{json,txt}`,
`rung0-shipped.{json,txt}`, `rung0-value.{json,txt}`. Raw traces live under
`.mlxfast-private/e122/` and are gitignored.

### W&B

Group `e122-target-margin-conditioned-depth`, project
`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`.

| run | id | contents |
| --- | --- | --- |
| `e122-rung0-forced` | [`9pv5vd0k`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/9pv5vd0k) | the primary gate, the per-position and per-prompt tables, the margin resolution report, and the value model |
| `e122-rung0-shipped` | [`oma770wk`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/oma770wk) | the confounded shipped-policy arm, descriptive only |

Both runs carry `harness=local`, `timing_valid=false`,
`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false` and
`official_or_ranked_score=false` in config. The shipped-arm run labels its own
statistics `_contaminated` and sets `arm_is_confounded=true`, so a later reader
cannot mistake 0.5295 for a clean measurement.

---

## 10. What this closes, and what it does not

**Closed.** The target's top-2 margin is not a usable per-round draft-depth
conditioner on this base, on this head, on these nine prompt proxies. Do not
reopen this axis on margin alone. Reopen only if the logits reaching the top-2
reducer change precision, or if the proposal head changes so much that the
target margin starts tracking head confidence.

**Not closed.** Adaptive depth itself. The oracle pool is 31.16 % pooled and
21.7–32.0 % on every prose proxy. This experiment says the margin does not
collect it; it says nothing about whether some other observable does.

---

## 11. Suggested follow-ups — not implemented

1. **Ask the proposal head, not the target.** The margin is the target's
   confidence in the token it just committed. The quantity a depth scheduler
   actually needs is the *head's* confidence in the drafts it is about to
   propose. That is a different number, it is available before the drafts are
   spent, and its per-draft top-2 gap is not a bf16 cancellation of two large
   logits. This is the natural successor and I did not run it.
2. **Test whether the pool is reachable at all.** Before pricing another
   conditioner, measure the *best achievable* held-out share using the round's
   full observable state. If a rich predictor also collects only single-digit
   percentages of the 31 % pool, the pool is an artifact of hindsight and the
   whole adaptive-depth axis should be closed, not just the margin.
3. **Accepted run length has more autocorrelation than the margin has signal.**
   `D_runlen` for the margin is +0.0284 pooled. A one-round lagged run length,
   or a short EMA of it, costs nothing to maintain and is worth a cheap
   concordance check on the traces already recorded. No new GPU time needed.
4. **Fix `RuntimeStartupMemoryPolicyTests`.** It asserts 320/128 against a
   promoted base running 512/50. A stale assertion in the maintained branch is
   a live hazard for anyone reading a red suite as noise.
5. **Consider whether the top-2 margin should be float32 anywhere it is used
   for control flow.** Not for this experiment, which is dead either way, but
   the reducer already produces float32 and the coarseness is inherited from
   the input. Any future scheduler that reads a logit difference will inherit
   the same 4-to-5-bit resolution problem.
