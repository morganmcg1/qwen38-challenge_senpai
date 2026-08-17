# E11 depth-lever showdown — working notes

Assignment `qwen38-r1-e11-depth-lever-showdown` r2, base `8970d775`.

## The arms

`sdpaWidthWallDepthCap` and `segmentedVerifyDepthCap` are not two independent
clamps. `costModelDepth` picks one of them per round:

```swift
let widthCap = fullAcceptStreak >= Self.segmentedStreakGate   // gate = 3
    ? Self.segmentedVerifyDepthCap                            // shipped 7
    : Self.sdpaWidthWallDepthCap                              // shipped 4
```

So the segmented cap only applies after three consecutive full-accept rounds.
Cold or struggling prompts never see it.

| arm | cost model | width cap | seg cap | built from |
| --- | --- | --- | --- | --- |
| `C` (C1/C2) | flat 0.20 | 4 | 7 | base — shipped defaults |
| `C8` | flat 0.20 | 4 | 8 | base — the pre-PR#2 default |
| `F3` | flat 0.20 | 3 | 3 | base — hard depth-3 clamp |
| `H` | measured curve | 4 | 7 | HEAD |
| `H8` | measured curve | 4 | 8 | HEAD |

All five share one `mlxfast-swift` (`d8cb9d74`) and have five distinct
`mlxfast-runtime-worker` hashes, so no two arms can silently run the same bytes.

## Analytic result: the curve is a depth-3 clamp

`research/e11_depth_reach.py` replays the extend walk. The test to go from
depth `d` to `d+1` is `reach > h[d] * (1 + expected) / (1 + cumH)`, and `reach`
is a product of probabilities, so `reach <= 1.0` always.

With the measured curve `h = [0.0842, 0.0775, 0.2426, 0.3754, ...]` the
step 3 -> 4 threshold is **1.0693 > 1.0**, which no `reach` can clear. A 400k
random + grid search over EMA profiles finds no profile reaching depth 4.

Consequences:

- The curve never reaches depth 4, so it never even reaches the width cap of
  4, let alone the segmented cap of 7 or 8. **Both caps are dead code in the
  curve arms.**
- `H8` must therefore be behaviourally identical to `H` — same tokens, same
  round count, same depth histogram. It is a falsification check on this
  analysis plus a second noise replicate, not an independent lever.
- The honest description of the curve is not "a better cost model". It is
  "a depth-3 clamp with slightly eager depth 0->2 behaviour" (h[0] and h[1] are
  *below* 0.20, h[2] is above it).

That is what makes `F3` the interesting arm: it reaches the same depth-3 clamp
with a two-integer change and no new cost vector. Constant-`q` sweep, cap 8:

```
q=0.70  curve=2  flat=3     q=0.90  curve=3  flat=7
q=0.80  curve=3  flat=4     q=1.00  curve=3  flat=8
```

`F3` clamps flat's column to 3, so `F3` and `H` agree everywhere except low `q`,
where `F3` drafts one deeper.

## The local fixture flatters deep drafting

`--local-iterate` defaults to `public_longcopy_gate_english_512_256.json`,
seeded from `public_longcopy_gate_english_512.txt`, whose instruction is
"Copy the passage between the tags exactly." That is a near-maximal acceptance
regime.

The eight hidden pool prompts are prose (beagle, botany, drama, essays,
medicine, plutarch, republic, travel) with calibration raw ratios spanning
0.847 to 1.073. So every local number here is measured in the friendliest
possible regime for deep drafting: `C8` sees its best case and the depth-3
clamp of `H`/`F3` sees its worst case. A tie on longcopy should widen in the
clamp's favour on prose, which is what T3 exists to check.

## Two fidelity layers, both reported

1. **Drift tripwire** — candidate against the M5-generated public 256-row
   golden, run outside the timing window. This is the only *external*
   reference available locally.
2. **Timed leg** — MTP rows against the same build's own serial reference rows
   over all 512 decode tokens (`all_tokens_matched`,
   `residual_divergence_count`). Self-consistency, not external proof.

## Run hygiene

- Timed pass clears every `MLX_QWEN_MTP_*` name and unsets `MLXFAST_NO_SANDBOX`;
  `meta.txt` records the surviving list verbatim so the `H` arms can prove they
  needed no research variable.
- The depth histogram comes free from the trusted parent's
  `effective_draft_lengths` report field, so no trace pass is needed for it.
- `research/e11-build.sh` rewrites the working-tree copy of
  `Qwen36MTPBlockSession.swift` per arm and restores it on EXIT. Read that file
  with `git show` while a build is running.

## C1 measured: the ternary is visible in the parent's own histogram

First timed arm, 512 decode tokens, declared 4-bit head `54930a1d`:

| field | C1 |
| --- | --- |
| serial s/tok | 0.074336291 |
| MTP s/tok | 0.032924979 |
| local ratio | 2.2578 |
| round_count | 88 |
| effective_mean_draft_len | 5.386 |
| accepted_draft_rate | 0.8945 |
| accepted / rejected drafts | 424 / 50 |
| declared rows = checked rows | 562 = 562 |
| all_tokens_matched | true |
| residual_divergence_count | 0 |

Parent depth histogram: `{1:1, 3:2, 4:39, 5:3, 6:5, 7:38}`.

This is direct confirmation of the ternary reading, not an inference. The two
tall bars sit at *exactly* `sdpaWidthWallDepthCap = 4` (39 rounds) and
*exactly* `segmentedVerifyDepthCap = 7` (38 rounds), with only 11 rounds spread
across every other depth. Under the flat `h = 0.20` control the cost model
always wants to go deeper than it is allowed, so each round is decided by
whichever cap the streak gate selected. The histogram is therefore a direct
readout of the gate's hot/cold split, and each arm's structural claim can be
falsified from the trusted parent alone.

Consequences for the arm table:

- **W5** must move the 39-round bar from 4 to 5 and leave the 7 bar in place.
  If the depth-4 bar survives, the sed did not reach the scored path and the
  arm is void regardless of its timing.
- **C8** must move the 38-round bar from 7 to 8.
- **H** must collapse nearly everything to depth 3, *below* both caps. That is
  the same clamp the reach analysis predicts analytically, so the histogram is
  an independent check on it — and it is why `H8 == H`: a cap at 8 cannot bind
  a curve that stops at 3.

The flat control reaching depth 7 is not in tension with the depth-3 clamp
result: the clamp is a property of the measured curve
(`defaultHeadStepCostRatioByDepth`), not of flat `h = 0.20`.

### Magnitude expectation for H, revised upward

C1 emits 512 tokens in 88 rounds. A curve clamped to depth 3 at the same
acceptance would need roughly `512 / (3 * 0.8945 + 1) ~= 140` rounds, i.e.
about 1.6x the target verifications. The original "+1.0% to +1.5%" guess for
H was calibrated before this histogram existed and is very likely far too
small; H should lose by substantially more on this fixture. Recorded here so
the miss is scored honestly rather than quietly rewritten after the fact.

## C2 replicate: the schedule is deterministic, only the clock is noisy

| leg | C1 | C2 | spread |
| --- | --- | --- | --- |
| serial s/tok | 0.074336291 | 0.074519354 | +0.246% |
| MTP s/tok | 0.032924979 | 0.032639906 | -0.866% |
| local ratio | 2.2577926 | 2.2830969 | **+1.121%** |

Every non-timing field is *bit-identical* across the two replicates:
`round_count` 88, `effective_mean_draft_len` 5.386363636363637,
`accepted_draft_rate` 0.894515, accepted 424, rejected 50, and the depth
histogram `{1:1, 3:2, 4:39, 5:3, 6:5, 7:38}`.

So the candidate's schedule is fully deterministic. Structural differences
between arms carry **zero** measurement noise, and the histogram is a clean
signal. All the noise lives in the wall clock.

### The stop rule fires on the ratio and not on the absolute time

The advisor's rule was "stop if the C1/C2 spread exceeds 1.0%". That is true of
the local ratio (1.121%) and false of the absolute MTP time (0.866%). This must
not be resolved by quietly picking the flattering number, so here is the
mechanism.

The two legs are *independently* thermally reset: `benchmark-qwen-mtp.sh` runs
the 40C cool gate before the serial leg and again before the MTP leg. The
replicate shows the serial leg drifting **up** 0.246% while the MTP leg drifted
**down** 0.866% — anti-correlated. A shared thermal factor would move both legs
the same way and cancel in the ratio; independent per-leg noise compounds in
it, which is what happened. The ratio is therefore the *noisier* statistic
here, by construction, and `program.md` independently warns not to lean on the
local ratio alone.

Consequences adopted for the rest of the experiment:

- **Primary timing metric: absolute MTP seconds/token.** Noise floor 0.866%
  spread, i.e. about +/-0.43% around the mean 0.0327824.
- **Secondary: the local ratio**, reported for every arm but never used to
  decide an arm on its own.
- **The serial leg is a free per-arm drift monitor.** The serial control runs
  MTP off at depth 0, so neither cap nor cost model is reachable and every arm
  executes the *same* computation. A serial leg that moves more than ~0.25%
  from C's flags machine drift rather than an arm effect.

This does not rescue every arm. Scored against a 0.87% noise floor:

- **C8** (predicted -2.8% to -3.5%) and **H** (predicted large and positive)
  are resolvable.
- **W5** (predicted -0.5% to -2.0%) is only *partly* resolvable. A W5 result
  near -0.5% is **unresolvable, not a win**, and will be reported as such. W5's
  verdict therefore leans on its structural claim, which is noise-free, with
  timing as support.

### Fixture caveat carried into every conclusion

`accepted_draft_rate = 0.8945` on the copy-task fixture is close to the
best case for deep drafting. The hidden pool is eight prose prompts with
calibration ratios 0.8467 to 1.0726. Any arm that wins here purely by drafting
deeper (C8, W5) is winning in the regime most favourable to it, so a local win
is weak evidence for the ranked pool and must be labelled as such.

## T1 result: both timing predictions were wrong, in opposite directions

512 decode tokens, `--local-iterate`, head `54930a1d`, CLI `d8cb9d74`, every
arm `pass=timed`, `mlx_qwen_env=""`, `all_tokens_matched=true`,
`residual_divergence_count=0`, declared rows == reference-checked rows.

| arm | worker | MTP s/tok | vs ref | ratio | serial drift | rounds | rows | mean D | accR | rej | maxd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | `bb216d46` | 0.032924979 | +0.435% | 2.2577 | -0.123% | 88 | 562 | 5.386 | 0.8945 | 50 | 7 |
| C2 | `bb216d46` | 0.032639906 | -0.435% | 2.2831 | +0.123% | 88 | 562 | 5.386 | 0.8945 | 50 | 7 |
| C8 | `57cd208f` | 0.032963455 | **+0.552%** | 2.2610 | +0.137% | 84 | 567 | 5.750 | 0.8861 | 55 | 8 |
| H  | `79ff4057` | 0.032452607 | **-1.006%** | 2.2939 | +0.022% | 134 | 532 | 2.970 | 0.9523 | 19 | 3 |

Reference = mean(C1, C2) = 0.032782442 MTP s/tok, 0.074427822 serial s/tok.
Noise floor = 0.873% spread, i.e. about +/-0.44% on `vs ref`.

Depth histograms (trusted parent's `effective_draft_lengths`, not the trace):

```
C1  {1:1, 3:2, 4:39, 5:3, 6:5, 7:38}   replay=15
C2  {1:1, 3:2, 4:39, 5:3, 6:5, 7:38}   replay=15   (bit-identical to C1)
C8  {3:2, 4:40, 5:2, 6:3, 7:7, 8:30}   replay=15
H   {1:1, 2:2, 3:131}                  replay=6
```

### Prediction 1 (C8) - WRONG

Predicted `C8` would beat `C1` by 2.8-3.5%. It **lost** by 0.55% against the
reference mean. Raising `segmentedVerifyDepthCap` 7 -> 8 is a null-to-slightly-
negative move on this fixture, not the free win I assumed.

The histogram shows exactly what the cap did: C's 38-round bar at depth 7 split
into 30 rounds at depth 8 and 7 at depth 7. That bought 4 fewer rounds (88 ->
84) and cost 5 more evaluated rows (562 -> 567) and 5 more rejected drafts
(50 -> 55). Accepted-draft rate fell 0.8945 -> 0.8861. The trade was a wash
that landed on the wrong side.

### Prediction 2 (H) - WRONG, and the error was instructive

Predicted `H` would lose by 1.0-1.5%; I then revised the magnitude *upward*
before measuring. `H` **won** by 1.006% (1.435% against C1 alone). That is just
outside the 0.87% noise floor: suggestive, not established, at n=2 for C and
n=1 for H.

My reasoning failed because I treated per-round overhead as dominant, so 134
rounds against 88 had to lose. Round count is the wrong unit. `H` evaluates
**fewer total rows** (532 vs 562) and throws away far less work (19 rejects vs
50), because clamping to depth 3 lifts acceptance 0.8945 -> 0.9523.

`H`'s serial leg drifted +0.022% from the C mean, so the machine was not
quietly faster during that run.

### Prediction 3 (H8 == H) - CONFIRMED, empirically

`H`'s `effective_max_draft_len` is **3** and 131 of 134 rounds sit at depth 3,
below both caps (4 and 7). Under the cost curve neither cap is ever reached, so
`H8` (cap 8) cannot differ from `H`. This reproduces `research/e11_depth_reach.py`
from measurement instead of arithmetic: stepping 3 -> 4 needs head reach 1.0693,
and reach cannot exceed 1.0. **`H8` is not worth a timed slot.**

## A two-parameter cost model that survives an out-of-sample test

Fit `decode_seconds = a * rounds + b * rows` on **C1 and H only** (the two
extremes, and the fit predates the C8 tabulation):

```
a = 0.012979 s/round      b = 0.027963 s/row
```

Because `rows = rounds * (mean_D + 1)` exactly for every arm measured, this is
equivalently a fixed cost of `a + b = 0.040942` s per round plus `b` per draft
slot.

**Out-of-sample check on C8**: predicted 16.945 s, measured 16.877 s, error
+0.40% - inside the 0.87% noise floor. One check is not a validated model, but
it is no longer merely an exactly-determined 2-point fit. (The model is decode-
only and does **not** extrapolate to the serial leg, which it under-predicts by
about 45%; serial runs a different code path.)

### The model gives a single break-even number

With `E` = emitted tokens per round and `D` = mean draft length,
`s/token = [(a + b) + b*D] / E`. Holding that flat while `D` rises by one slot
requires `E` to rise by `E*b / [(a+b) + b*D]`. At C's operating point that is

```
break-even marginal acceptance = 0.849 accepted tokens per extra draft slot
```

Measured marginal rates:

| move | dD | dE | marginal | vs 0.849 |
| --- | --- | --- | --- | --- |
| C -> C8 (slot 8) | +0.364 | +0.277 | **0.762** | below -> C8 loses |
| H -> C (slots 4-7) | +2.416 | +1.997 | **0.827** | below -> H wins |

**One mechanism explains both failed predictions.** Over the whole depth 3-8
band the marginal accepted tokens per extra draft slot (0.76-0.83) sits *below*
the 0.849 break-even, so on this fixture shallower is better everywhere
measured, and it gets worse the deeper you go. That is why C8 lost and why H
won.

### Revised (post-C8) prediction for W5, recorded against the original

The pre-registered prediction 5 was "`W5` beats C1 by 0.5-2.0%". The cost model
now says otherwise, and both go on the record.

`W5` moves the cold width wall 4 -> 5, so C's 39 depth-4 rounds become depth-5
rounds: `dD` is about +0.443. Slot 5 should accept better than slot 8 (0.762)
and better than the 4-7 band average (0.827), but 0.849 is the bar and the band
is trending the wrong way.

- **Original prediction 5**: -0.5% to -2.0% (a win).
- **Revised prediction 5b**: -0.3% to +0.3% - break-even, most likely
  unresolvable against a 0.87% noise floor.

The structural claim is unchanged and noise-free: **the depth-4 bar must
disappear** from the histogram, replaced by roughly 39 rounds at depth 5. If a
depth-4 bar survives, the arm did not do what its name says and is VOID.

### Tension with the public frontier that the advisor should see

The current promoted frontier (`033f622`, score 2.925) carries
`sdpaWidthWallDepthCap = 5`, `segmentedVerifyDepthCap = 8` and
`headStepCostRatio = 0.18` together. My C8 arm isolates one of those three and
finds it slightly **negative** here. Either the copy-task fixture is
unrepresentative of the hidden pool (likely - see the fixture caveat), or the
levers interact, or the frontier's gain comes from elsewhere in that commit.
Isolating one lever at a time is what makes this visible, and it is a reason to
treat any single-lever local win as weak evidence for the ranked pool.

## T2 result: W5 wins, and the two caps are not the same lever

| arm | worker | MTP s/tok | vs ref | ratio | serial drift | rounds | rows | mean D | accR | rej | replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W5 | `7736e290` | 0.032370 | **-1.258%** | 2.2999 | +0.029% | 80 | 564 | 6.050 | 0.8946 | 51 | 12 |
| F3 | `96ba4c43` | 0.032495 | -0.876% | 2.2923 | +0.082% | 134 | 534 | 2.985 | 0.9475 | 21 | 6 |

```
W5  {1:1, 4:3, 5:29, 6:3, 7:44}   replay=12
F3  {1:1, 3:133}                  replay=6
H   {1:1, 2:2, 3:131}             replay=6   (for comparison)
```

Both `pass=timed`, `env=""`, `all_tokens_matched=true`, `resid=0`, rows OK.

### Prediction 5 (W5) - CORRECT, and my own revision of it was WRONG

`W5` beats C1 by **1.686%** and the reference mean by **1.258%**, inside the
pre-registered -0.5% to -2.0% band. It is now the fastest arm measured.

The revised, model-based **prediction 5b (-0.3% to +0.3%, unresolvable) was
wrong**. I talked myself out of a correct pre-registered prediction using a
model whose *direction* was right but whose input estimate I got wrong: I
assumed the round count would hold at 88, so `dD` was only +0.443. The round
count actually fell to 80, making `dD = +0.664` and `dE = +0.582`, i.e. a
marginal acceptance of **0.877 - above the 0.849 break-even**. Worked properly,
the break-even model predicts a W5 win. The lesson is that `dE` and `dD` are
coupled through the round count and cannot be estimated one at a time.

### The structural claim, and an honest correction to my VOID rule

Pre-registered: "the depth-4 bar must disappear, or the arm is VOID." Measured:
the depth-4 population fell **39 -> 3**. Three rounds still land on depth 4.

Read literally my rule would VOID the arm, and that reading is wrong - so I am
recording the correction rather than quietly reinterpreting it. With the wall at
4, all 39 rounds were *pinned* at the wall. With the wall at 5, the cost model
is free to stop at 4 on its own, and only 3 rounds do. What had to disappear was
the **pile-up at the wall**, not every depth-4 round. 39 -> 3 is unambiguous:
the wall moved. The rule was stated too crudely; the arm is valid.

An unpredicted effect also shows up: depth 7 grew **38 -> 44**. Lifting the cold
wall lets some rounds survive long enough to reach the streak gate, so this
lever moves rounds into the *other* cap's population too.

### The real mechanism: the two caps gate different round populations

`C8` and `W5` both add draft depth to C, yet one loses and the other wins,
because they add it to different rounds:

| lever | population | dD | dE | marginal | vs 0.849 | result |
| --- | --- | --- | --- | --- | --- | --- |
| `C8` cap 7 -> 8 | hot, streak-gated | +0.364 | +0.277 | **0.762** | below | loses |
| `W5` wall 4 -> 5 | cold, width-gated | +0.664 | +0.582 | **0.877** | above | **wins** |

**Cold width-wall rounds have much higher marginal acceptance than hot
streak-capped rounds.** This corrects the over-general story I told after T1
("shallower is better everywhere"). C is not on a slope, it is on a *ridge*:
both drafting deeper (W5) and shallower (H) beat it, because C's mean of 5.386
is a blend of a too-shallow cold population and a too-deep hot population.
Averaged depth is the wrong control variable; the two populations want opposite
treatment.

This also explains the frontier tension above: `033f622` ships the wall at 5
(which my W5 confirms is the good lever) bundled with cap 8 (which my C8 shows
is not, at least here).

### Prediction 4 (F3 within noise of H) - CONFIRMED

`F3` and `H` differ by 0.13%, far inside the +/-0.44% floor, with near-identical
histograms (`{1:1, 3:133}` vs `{1:1, 2:2, 3:131}`) and identical replay counts.

This matters for what would ship: **H's win never needed the cost curve.** Two
constants clamped to 3 reproduce it. Since `W5` beats both, the depth-3 branch
is not the ship candidate anyway - but had it been, the simple version was
available and the curve would have been unjustified complexity.

### Where the 2-parameter cost model breaks

Predicted vs measured decode seconds, model fit on C1/H only:

| arm | pred | act | err |
| --- | --- | --- | --- |
| C2 | 16.857 | 16.712 | +0.87% (= noise floor) |
| C8 | 16.945 | 16.877 | +0.40% |
| F3 | 16.671 | 16.638 | +0.20% |
| **W5** | 16.809 | 16.574 | **+1.42%** |

Five of six arms land inside the noise floor; `W5` is over-predicted by 1.42%,
i.e. it is *faster* than rounds and rows alone can explain. The obvious
candidate for the missing term is `verify_block_replayed_round_count`, which
tracks the win ordering across arms (C1/C2/C8 = 15, W5 = 12, H/F3 = 6) and is
not in the model. Replay is real work the model currently charges nothing for.
I am reporting this as an open discrepancy rather than adding a third parameter
to six points and calling it a fit.

`W6` was added to the table after it ran and lands at pred 17.227 / act 16.921,
+1.8%. That is the same sign and roughly the same size as `W5`'s miss, which is
consistent with the missing replay term rather than with a second effect.

## The noise floor I have been quoting is not a floor

`W5b` re-ran the `W5` binary. The two replicate pairs now available disagree by
an order of magnitude:

| pair | arms | spread on `vs ref` |
| --- | --- | --- |
| control | C1 / C2 | **0.873%** |
| width-5 | W5 / W5b | **0.092%** |

Both pairs are the same binary run twice under the same gate, so a single pair
is evidently a bad estimator of run-to-run spread — 0.873% is one draw, not a
measured floor, and I quoted it as though it were a property of the host. I am
keeping 0.873% as the CONSERVATIVE gate for every claim in this experiment,
because it is the larger of the two and because being wrong in that direction
only costs me claims. But the reasoning I recorded after T1 — that C8 and H
were resolvable and W5 was only partly resolvable — rested on treating one pair
as a calibrated instrument. That reasoning was unsound even where its
conclusion survived. Anything that needs a real floor needs n >= 5 on one arm,
which E11 did not buy.

What is NOT noisy is the schedule. `W5` and `W5b` are bit-identical on every
structural counter and on the full depth histogram, exactly as `C1`/`C2` were.
Only the clock moves.

## W5 vs W6: the cold lever has an interior optimum at 5

`W6` opens the same cold wall one step further, to depth 6.

**Its timing is inconclusive and I am not quoting it as a headline.** The
serial leg drifted +0.873% against the C1/C2 reference, which trips the
~0.25% serial-drift monitor I pre-registered before T1. When the denominator
moves that far, the ratio and the absolute number are both suspect.

The structural comparison needs no clock, and it is decisive. Against `W5` at
*identical* realised mean draft length — 6.049 vs 6.050 — `W6` is worse on
every axis:

| counter | W5 | W6 | direction |
| --- | --- | --- | --- |
| rounds | 80 | 82 | worse |
| target rows | 564 | 578 | worse |
| rejected drafts | 51 | 66 | worse |
| replayed rounds | 12 | 16 | worse |
| accepted-draft rate | 0.8946 | 0.8669 | worse |
| mean draft length | 6.050 | 6.049 | tie |

There is no dimension on which `W6` is better. It buys the same average depth
by a worse route: more speculative rows, a quarter more rejects, a third more
replays. `W6` loses to `W5` whatever the clock says, and the 2-parameter model
puts it behind `C` as well (+2.2% predicted decode).

### The marginal-acceptance cliff, and why the wall belongs at exactly 5

The break-even from the cost model is **0.849** accepted tokens per extra draft
slot. Measured marginals per lever:

| lever | population it gates | dD | dE | marginal | verdict |
| --- | --- | --- | --- | --- | --- |
| C -> C8 (hot cap 7->8) | streak-qualified | +0.364 | +0.277 | 0.762 | loses |
| C -> W5 (cold wall 4->5) | everything else | +0.664 | +0.582 | **0.877** | **wins** |
| W5 -> W6 (cold wall 5->6) | everything else | -0.001 | -0.156 | ~0.4 | loses hard |

Slot 5 clears break-even by 0.028; slot 6 misses it by roughly half. That is a
cliff, not a slope, and it puts the cold wall at exactly 5 — one integer,
bracketed on both sides by measurement rather than by taste.

This independently reproduces `sdpaWidthWallDepthCap = 5` in the public
promoted frontier `033f622` (score 2.92520777238747). That receipt ships wall 5
together with cap 8 and h 0.18. E11 says the wall is the part that pays: cap 8
is a measured loss here (+0.55%), and the advisor already showed h 0.18 is
bit-identical to 0.20 after the cap-7 clamp. Useful as corroboration in one
direction only — that receipt was scored on the real hidden 8-prompt pool,
which is exactly the generalisation my own evidence cannot reach.

## Shipped: one integer

`sdpaWidthWallDepthCap = 4 -> 5`, unconditional, no env var, no new constant.

The committed source at 52be39e hashes to
`341ddc5ad5e153280ec7c4f3d78a93da0ab838c5ed358bd8b8b288eed50e2916`, which is
byte-identical to `.mlxfast-private/e11/runs/W5/meta.txt` `source_sha256`. The
shipped bytes and the measured bytes are the same bytes. This falls out of the
harness design: `e11-build.sh` materialises C/C8/F3/W5/W6 from `git show
BASE:src`, i.e. from the campaign base with the *scalar* `headStepCostRatio`,
so `W5` never was a vector-arithmetic variant and there is no floating-point
equivalence argument to make.

The earlier plan of record — ship the measured per-depth h curve, committed at
7c85b4f — is **reverted**. Three reasons, in order of weight:

1. It lost. `H` (curve) is -1.006%; `W5` (one integer) is -1.303%, at n=2 with
   a 0.09% spread against `H`'s n=1. The gap is inside the conservative floor,
   so this reason alone would not settle it.
2. `F3` reproduces `H` to within 0.13% using two clamped integers and no curve
   at all. Whatever the 8 fitted constants were buying, "stop at depth 3" buys
   the same thing. The curve is over-parameterised for its measured effect.
3. The curve is head-dependent by construction — its own doc says re-fit after
   any head change — and it was fitted on this one copy fixture. One integer
   that a promoted receipt independently landed on is the better bet off-fixture.

They are also not composable, which is worth stating because it is not obvious:
under the curve the realised max depth is 3, strictly below the wall at 4, so
the wall never binds and raising it to 5 changes nothing. Curve-plus-wall is
exactly `H`. This is an either/or, and I took the measured winner.

## T3/T4/T5 — the prose fixture, and the retraction of everything above

Everything in the section above was measured on the `--local-iterate` default,
`public_longcopy_gate_english_512_256.json`. That is a copy task. It runs at
acceptance 0.89-0.95 and mean draft depth 5.4, which is close to the best case
for deep drafting. The hidden pool is eight prose prompts with calibration
ratios 0.8467-1.0726. So I built a held-out prose golden and re-ran the arms.

`research/e11_prose_gate_english_512.txt` is original expository English about
railway time standardisation, deliberately outside every hidden-pool subject
(beagle, botany, drama, essays, medicine, plutarch, republic, travel).
`research/e11-golden.sh` generates the 512/256 golden from it through the
serial reference path with no E11 arm resident. Nothing under `fixtures/` or
`correctness_prompts/` is touched; the golden is selected through
`MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE`, which `benchmark-qwen-mtp.sh:103`
already publishes, and Yukon submits none of it. Because it was generated on
this host, the drift tripwire is self-consistent.

Six timed arms, three jobs, all `pass=timed`, `env=""`, `all_tokens_matched`,
`resid=0`, row-check OK, head `54930a1d`:

| arm  | MTP s/tok | vs ref  | ratio  | rnds | rows | meanD | accR   | rej | maxd | replay | depth hist |
|------|-----------|---------|--------|------|------|-------|--------|-----|------|--------|------------|
| Cp   | 0.050640  | +0.010% | 1.4705 | 253  | 823  | 2.253 | 0.4561 | 310 | 4    | 100    | 1:39 2:129 3:67 4:18 |
| Cp2  | 0.050630  | -0.010% | 1.4735 | 253  | 823  | 2.253 | 0.4561 | 310 | 4    | 100    | 1:39 2:129 3:67 4:18 |
| W5p  | 0.050606  | -0.056% | 1.4717 | 253  | 823  | 2.253 | 0.4561 | 310 | 4    | 100    | 1:39 2:129 3:67 4:18 |
| Hp   | 0.046757  | -7.658% | 1.5925 | 246  | 749  | 2.045 | 0.5288 | 237 | 3    | 76     | 1:2 2:231 3:13 |
| Hp2  | 0.046774  | -7.624% | 1.5955 | 246  | 749  | 2.045 | 0.5288 | 237 | 3    | 76     | 1:2 2:231 3:13 |
| F3p  | 0.050025  | -1.204% | 1.4925 | 254  | 806  | 2.173 | 0.4692 | 293 | 3    | 100    | 1:43 2:124 3:87 |

### The wall is a no-op on prose, provably

`Cp` and `W5p` differ only in `sdpaWidthWallDepthCap` (4 vs 5) and produced
identical round count, identical depth histogram, identical accepted and
rejected totals, and identical replayed-round count. The scheduler computes
`cap = min(offeredDepth, maxDepth, widthCap)` and then extends greedily while
`reach > threshold`. Eighteen prose rounds stopped at depth 4 under a wall of
4; under a wall of 5 the same eighteen still stopped at 4, which proves they
exited on the reach test and not on the cap. The wall never binds on prose.

So the -1.303% I measured for `W5` on the copy fixture predicts approximately
zero on the scored pool. The lever is real, but its reach is a property of the
prompt, not of the code.

### The curve wins on prose, by a lot, and F3 does not stand in for it

`H` is -7.64% on prose against -1.006% on copy, and the replicate pair agrees
to 0.036%. This is roughly 200x the noise. The mechanism is legible: rows
823 -> 749, rejected rows 310 -> 237 (-24%), replayed rounds 100 -> 76 (-24%),
acceptance 0.4561 -> 0.5288. Prose wastes 310 of 823 rows on rejected drafts;
the curve is the only arm that meaningfully attacks that.

All three reasons I gave above for reverting the curve were artifacts of the
copy fixture:

1. "It lost" was true only on copy. On prose the ranking inverts, and it
   inverts by 7.6 points against a no-op.
2. "`F3` reproduces it" was true only on copy. On prose `F3` is -1.204% and
   `H` is -7.658%. A fixed integer cap can only shave the depth-4 tail
   (67 -> 87 at depth 3, 18 -> 0 at depth 4). The curve re-prices every round
   and collapses the distribution onto depth 2 (231 of 246 rounds). The extra
   parameters are buying per-round adaptivity, not slack.
3. "It is fixture-fit" was simply wrong about provenance, and I should have
   checked before asserting it. The vector comes from E1's forced-depth
   marginal-cost arms on the declared head (N = 1778 pooled depth-0 rounds;
   61/60/36/32 full-accept rounds at d3/d4/d6/d8), and the knee at d = 3..4 is
   explained by the affine-4 g64 crossrow kernel adding a weight pass. It is a
   mechanistic measurement of the head, not an end-to-end fit to a fixture.
   That is exactly why it survives transfer to a held-out prompt, and survives
   it with a larger gain.

Composition is unchanged and still analytic: the curve's realised max depth is
3 on both fixtures, strictly below the wall at 4, so curve-plus-wall is exactly
the curve. Shipping the curve makes the wall change moot, so the wall goes back
to 4 and the shipped diff is the curve alone.

### Noise floor, third and final revision

Five replicate pairs now exist, four of them at or below 0.1%:

| pair    | fixture | spread |
|---------|---------|--------|
| Cp/Cp2  | prose   | 0.019% |
| Hp/Hp2  | prose   | 0.036% |
| Cp/W5p  | prose   | 0.066% (provably identical work) |
| W5/W5b  | copy    | 0.092% |
| C1/C2   | copy    | 0.873% |

`Cp`/`Cp2` are from different jobs and still agree to 0.019%, so the host is
stable and the 0.873% on `C1`/`C2` was a thermal excursion rather than the
floor. Typical spread is about 0.1% with occasional ~0.9% outliers. I am still
not claiming a floor from n=5, but the -7.64% headline does not depend on where
in that range the truth sits.

### What this says about the method, not just the result

The copy fixture ranked these levers exactly backwards. It is the default for
`--local-iterate`, it is the cheapest thing to measure, and on the question the
campaign actually cares about it was actively misleading: it promoted a no-op
over a -7.6% change. Any depth or acceptance lever screened only on the default
fixture should be treated as unmeasured until it is run on prose.


---

## r3: replay on the promoted-frontier base

### Provenance, before any number

The r2 measurements were taken on base `8970d775`. The advisor's r3 question is
whether the curve's -7.658% survives the base move to
`fe38ecc21e4084e4d17dac3aa76264bb5897a614` (promoted frontier `32b94cb`, Yukon
submission `03dedda8-fc70-4e3e-881f-5384a17af405`, score 2.94661597308114).

The branch takes the new base by **merge** (`d85d22d`), not rebase, so the r2
result commit `2062fea` stays reachable and the r2 evidence remains auditable.
`git diff --stat fe38ecc d85d22d -- Sources Vendor mtp-head.manifest.json` is
empty: the merge contributes nothing to the scored surface beyond the base.

Four things changed under us, and each one had to be checked rather than assumed.

**1. The E1 research hooks are gone.** The frontier deleted
`overrideHeadStepCostRatioByDepth`, `MLX_QWEN_MTP_H_VECTOR`, `forcedDepth`, and
the trace `h=` field. The r2 report flagged a hazard here -- that a future h
experiment could read the scalar through an instrument that no longer exists --
and it is now moot: after the deletion there were exactly two readers of
`headStepCostRatio`, both inside `costModelDepth`. The r3 curve is therefore a
source-level edit of those two readers with no env plumbing at all, and the
`H`-type arms run with **no** `MLX_QWEN_MTP_*` variable set.

**2. `costModelDepth` gained a depth-1 confidence clamp.** The frontier scales
the depth-1 expected-acceptance term by `conf2 = 1/(1+exp(-margin/3.0))`, which
did not exist in r2. It is preserved verbatim in the curve implementation; the
only thing the curve replaces is the constant `0.18` in the cost term.

**3. The width wall moved from 4 to 5** (`sdpaWidthWallDepthCap`), with
`segmentedVerifyDepthCap = 8` and `segmentedStreakGate = 3`. Q3 is whether the
curve and the wall are still substitutes at wall 5, which is decided by the
curve's realised max depth, not by argument.

**4. The declared MTP head repo changed.** r2 ran
`lowskillcoding/qwen38-mtp-head-4bit-g64@0966ddaf` (sha256 `cc209e30...`,
238934093 bytes). The frontier manifest declares
`dwsdubey/qwen3.8-27b-mtp-4bit@34ee76f6c87a438caa28f975c1cea9b0b005bc71`
(sha256 `7d62702795865b9036afe4bddcd16a2a8eb973c0caced15e5243139dda067f47`,
238934129 bytes). This is the one change that bears directly on the h vector,
because h is a property of the head's per-step cost. Both are affine 4-bit
group-64 requantisations of the same pinned bf16 head with identical geometry
(31 tensors, same names and shapes), so the cost profile should be close and the
acceptance profile is the part free to move. The doc block's own "re-fit after
any head change" trigger did fire; Q2 measures the marginals so the advisor can
see by how much, without re-fitting (out of scope for r3).

The head was re-provisioned from scratch for r3: the stale r2 cache directory
was removed and `research/fetch-declared-head.sh` re-downloaded the declared
revision. Verified tree digest `7d627027...` == manifest, 238934129 bytes.
Standalone `model.safetensors` sha256
`c934b40f1254858425cc0b5fdfe62b6ae13d1a4aff74da9d81606e92fdcf41ee`.

### One staging detail worth writing down

`benchmark-qwen-mtp.sh:215` refuses to run unless
`${MLXFAST_QWEN_MTP_HEAD_DIR}/config.json` is non-empty. The declared head
publishes only `model.safetensors` -- which is also exactly what the ranked
workflow stages -- and the loader does not need a config: the DECLARED-HEAD
STAGING branch in `Qwen36MTPHeadAttachment.verifyHeadTree`
(`Sources/MLXFastModel/Qwen36MTPHeadAttachment.swift:215`) reads structure out
of the safetensors header when `model.safetensors.index.json` is absent. So the
wrapper precondition is satisfied by copying the pinned head's `config.json`
into the declared directory and leaving the index absent, which keeps the loader
on the declared-head branch. The wrapper is trusted and was not modified. r2 ran
the same way (its cache directory carried the same 3570-byte config), so this is
a continuation of the r2 protocol, not a new degree of freedom. Because the
manifest byte count covers `model.safetensors` alone, the standalone file digest
above is the witness to report, not a whole-directory digest.

### Two honesty corrections to r2, found while rebuilding the fixture

**r2's exactness claim covered a prefix.** The r2 golden carried 256 expected
tokens while the decode window was 512, and `all_tokens_matched` compares only
as many tokens as the golden carries. So r2's `all_tokens_matched: true` proved
an exact match over the first 256 of 512 decode tokens, not over the window. r3
regenerates the golden at 512 steps, puts the step count in the filename, and
`research/e11-run.sh` now refuses any golden shorter than the decode window, so
the gap is structurally closed rather than remembered.

**The r2 golden was not stale, though.** The new 512-token golden on the new
base has the r2 256-token golden as an exact prefix: the frontier's 172-line
`Vendor/.../Qwen35.swift` change does not move the serial target trajectory on
this prompt. The r2 -> r3 comparison is therefore apples-to-apples on the
reference side, and the only moving parts are the schedule, the wall, and the
head.

### Retraction: my r2 "trace unreachable" disclosure was wrong

I reported in r2 that `MLX_QWEN_MTP_TRACE` is unreachable because the worker
sandbox denies `file-write*` and `MLXFAST_NO_SANDBOX=1` is refused in benchmark
contexts. The Seatbelt half of that is correct. The conclusion is not, and the
advisor caught it. Verified by symbol on this base:

- `Qwen36MTPBlockSession.traceWrite` (`:463`) writes to
  `FileHandle.standardError`, not to a file, and its own comment says it does so
  *because* the sandbox denies file-write. I went looking for the file the
  doc-comment above `traceRounds` (`:458-460`) still describes -- that stale
  comment is what misled me, and reading the function body instead of the
  comment above it would have settled it in one step.
- `forwardsWorkerStderr` is organizer infrastructure, declared in `QwenRuntime`
  in both `MLXFastHarness` and `MLXFastTrustedHarness` and consumed in each
  `QwenRuntimeWorker` drain (`emit: options.forwardsWorkerStderr ? nil : { _ in }`).
- `Sources/MLXFastCLI/main.swift:1805` sets `forwardsWorkerStderr` from
  `MLX_QWEN_MTP_TRACE` on the local `mtp-timed` verb, and `:2319` ANDs it with
  `!officialRun`.

So `MLX_QWEN_MTP_TRACE=1` on a local `mtp-timed` run emits the trace on stderr
with no sandbox fight. The instrument existed the whole time.

**One thing the correction itself is stale on, though: there is no `h=` field on
this base.** The advisor's note suggests reading `h=` to tell a reach-test stop
from a cap stop, but `h=` was part of the E1 instrumentation block the frontier
deleted. The round line here is
`round= d= acc= draft_build_us= verify_build_us= eval_wall_us= readout_us= commit_us= upkeep_us= round_us=`.
That turns out to be *more* useful for Q2 than `h=` would have been: grouping
`round_us` by `d=` gives per-depth round cost directly, and the five-way split
says **which segment** grows with depth, which no timing delta can show. Q3 does
not need `h=` either -- reachability of depth 4 under the curve is decidable from
the source without running anything.

The obligation attached to the instrument is real and I am keeping it: a traced
run does per-round stderr writes with the parent drain in the loop, so it can
never be a timing arm. Every arm reported with a time below ran with no
`MLX_QWEN_MTP_*` variable set at all, and any trace run is labelled untimed with
its own SHA.

Depth histograms for the timed arms therefore still come from the trusted
parent's `effective_draft_lengths` -- which is the better source regardless,
since it is the parent's accounting rather than the candidate's self-report. The
same array, aligned with `block_request_seconds`, is what
`research/e11_marginal.py` uses to recover per-depth round cost for Q2, so the
forced-depth arms E1 needed are replaced by post-hoc analysis of the ordinary
timed arms at zero extra GPU cost and zero instrumentation overhead.

### RETRACTED 2026-08-17 — the section below is wrong; prefill IS inside the score

**Correction, filed during E17 (`qwen38-r1-e17-curve-transfer-and-refit`, PR
#19). The section that follows this block is retained verbatim for the record and
must not be relied on. Its conclusion — "the score is decode-only, so prefill
does not dilute any E11 delta" — is false. The r3 assignment was right and my
retraction of it was the error.**

#### What I got wrong, and why the evidence looked convincing

Every enforcing citation in the retracted section is individually accurate:
`/scoring/mode` really is `qwen-mtp-paired-decode-only`, the workflow jq really is
`(.serial_seconds_per_token_mean / .mtp_seconds_per_token_mean)`, there really is
no additive prefill operand, and `Score.swift`'s prefill-weighted geometric mean
really is off this track's path. The inference from them was invalid, because I
conflated two different senses of "decode-only":

1. **the score is not a decode+prefill weighted blend** — true, and that is what
   the mode string and the absent additive term are telling us; it distinguishes
   this track from `Score.swift`'s `0.75 · decode ⊕ 0.25 · prefill`;
2. **`seconds_per_token` excludes seed-prefill time** — false.

There is no additive prefill term precisely *because* prefill is already inside
`seconds_per_token`. I read the absence of a second prefill term as evidence that
prefill was absent altogether, and never checked how
`seconds_per_token` is actually produced. That check was the one step missing,
and it is one file away:

| # | source | what it establishes |
| --- | --- | --- |
| 1 | `Sources/MLXFastRuntime/QwenRuntimeMTPDriver.swift:94` — `let started = Date()` | the clock starts **before** any decode work |
| 2 | `…:95` — `beginMTPDecode(...)` | seed prefill runs **after** the clock started, so it is inside the span |
| 3 | `…:197` — `decodeSeconds = Date().timeIntervalSince(started)` | the span runs to the last emitted token and contains prefill |
| 4 | `Sources/MLXFastRuntime/QwenRuntimeMTP.swift:442` — `decodeSecondsPerToken = decodeSeconds / max(decodeTokenCount, 1)` | the whole span, prefill included, is divided by decode tokens only |
| 5 | `Sources/mlxfast-swift/main.swift:~2013-2028` | emits `parent_measured_seconds_per_token` from that value, and emits `seed_prefill_seconds` / `prefill_seconds_per_token` beside it marked *"Observability only — nothing above subtracts it — and omitted (never zero)"* |
| 6 | `benchmark-qwen-mtp.sh:651-652,726` | the reported ratio is built from `parent_measured_seconds_per_token`, unmodified |

Step 5 is decisive and I had it in front of me: the comment says in as many words
that nothing subtracts prefill. `program.md`'s "Both seed processing and decoding
are included in the same timed leg, even though prefill has no separate score" is
the plain reading after all — "no separate score" is sense (1), not sense (2). My
"once read carefully" gloss on that sentence was motivated reasoning.

#### The second error: the illustrative arithmetic double-charges prefill

The retracted section's own worked example is also wrong, and independently so:

```text
prefill-inclusive  (3.998045+38.179593)/(3.994842+25.330100) = 1.4383   <- WRONG
```

`38.179593` and `25.330100` are `decode_seconds`, which by step 3 above
**already contain** the `3.99 s` prefill. Adding `seed_prefill_seconds` to them
charges prefill twice. This mattered beyond my own notes: the E17 r1 assignment
quotes `Sp3 = 1.437971`, `Hp3 = 1.521771` and "scored delta 17.67% smaller",
which reproduces this double-charged pair to within 0.03 %. The advisor correctly
rejected my conclusion but inherited my bad worked example, so the corrected
convention arrived attached to numbers generated by the very error it was
correcting.

#### Corrected table

`research/e17_analyse.py --r3` recomputes both r3 arms from their own committed
reports (`.mlxfast-private/e11/runs/{Sp3,Hp3}/reports/0{3,4}-mtp-timed.json`,
n = 512, measured `seed_prefill_seconds` 3.994360–3.998045 s across all four
legs):

| convention | Sp3 (scalar 0.18) | Hp3 (merged curve) | pair delta | g on MTP leg |
| --- | --- | --- | --- | --- |
| **prefill-inclusive — the score** | **1.507282** | **1.609073** | **0.101791** | **+6.378 %** |
| decode-only (`spt − P/n`) | 1.602116 | 1.732572 | 0.130456 | +7.577 % |
| double-charged (`spt + P/n`) — the error | 1.438285 | 1.521191 | 0.082906 | n/a |

**The published r3 headline needs no numerical correction.** `Sp3 = 1.507282`,
`Hp3 = 1.609073` and `g = 6.378%` were computed straight from
`parent_measured_seconds_per_token` and are therefore *already* the
prefill-inclusive values, matching to six decimals. Only the prose was wrong.
Two further consequences worth recording:

* The direction of my "why this matters" claim is inverted. Going decode-only →
  prefill-inclusive *shrinks* a ratio, so if r3 had really published decode-only
  numbers the correct values would have been `1.602116 / 1.732572` — **larger**,
  not smaller. I claimed the assignment's formula would make me understate the
  lever; in fact the convention I adopted would have overstated it.
* Prefill genuinely does dilute, just not in the way I denied: at `P ≈ 3.995 s`
  it is ~10 % of the `38.2 s` serial leg and ~16 % of the `23.7–25.3 s` MTP leg,
  an unspeculatable fixed cost that caps any candidate's attainable ratio. That
  is a reason to take the convention seriously, not to argue it away.

E17 uses the prefill-inclusive convention throughout, with no subtraction; see
`research/e17-notes.md` §1.4 and §2.

---

<details>
<summary>RETRACTED r3 text, retained verbatim for the record — do not rely on it</summary>

### The score is decode-only, so prefill does not dilute any E11 delta

The r3 assignment states as a hard requirement that prefill is inside the score,
`raw_p = (P + D_serial) / (P + D_mtp)`, and records that this supersedes an
earlier "prefill excluded" claim. Checking the enforcing sources in the order
`program.md` prescribes, the earlier claim was the correct one and the
retraction is the error. Prefill is measured, reported, and thermally relevant,
but it is not in the scored ratio.

Enforcing evidence, in `program.md`'s own precedence order:

- `benchmark.json` `/scoring/mode` is the string `qwen-mtp-paired-decode-only`.
- `benchmark.json` `/scoring/aggregation` is
  `median_of_per_prompt_raw_serial_relative_speedup`, and its note defines the
  per-prompt quantity as "mean serial (depth-0) seconds/token over that
  prompt's accepted pairs divided by mean candidate seconds/token over the same
  pairs". There is no additive prefill term.
- `.github/workflows/qwen-mtp-ranked-benchmark.yml:129` states
  `raw_p = mean(serial depth-0 seconds/token) / mean(MTP seconds/token)`.
- The same workflow at `:3083` computes the per-prompt ratio as
  `(.serial_seconds_per_token_mean / .mtp_seconds_per_token_mean)`. That jq
  expression is the enforcing site and it contains no prefill operand.
- The workflow does collect `prefill_seconds_per_token` at `:3148-3217`, but
  only to average it into the emitted metrics block. It is a published
  diagnostic, never a divisor or an addend.

`Sources/MLXFastCore/Score.swift:18-48` does implement a prefill-weighted score,
a weighted geometric mean of decode and prefill speedups with
`scorePrefillWeight = 0.25` and `scoreDecodeWeight = 0.75`
(`Sources/MLXFastCore/Constants.swift:244-245`). That function is not on this
track's path. The local harness proves it numerically: `score.json` reports
`score` exactly equal to its own `mtp_decode_speedup`, which is
`serial_seconds_per_token / mtp_seconds_per_token` to every printed digit, while
`prefill_seconds_per_token` sits beside it unused.

Why this matters beyond bookkeeping. Prefill is a fixed ~3.99 s per leg that is
identical in the serial and MTP legs, because both legs process the same
512-token seed with the same target. Folding it in would compress every ratio
toward 1.0 and, worse, would compress *differences* between candidates. On the
r3 scalar arm the two readings are:

```text
decode-only        0.074569518 / 0.049472851          = 1.5073   <- the score
prefill-inclusive  (3.998045+38.179593)/(3.994842+25.330100) = 1.4383   <- not the score
```

A 4.6% gap on the level, and a similar proportional haircut on any delta between
two candidate arms. Had I accepted the assignment's formula I would have
reported the curve's headline as roughly 0.93x of its true score effect and
understated the lever. Because the score is decode-only, the decode-only deltas
this experiment measures carry into `raw_p` essentially one-for-one, and the r2
headline of -7.658% on `mtp_seconds_per_token` needs no prefill correction at
all.

`program.md`'s prose sentence "Both seed processing and decoding are included in
the same timed leg, even though prefill has no separate score" is consistent
with this once read carefully: both are inside the timed leg as *work*, and the
trailing clause is the disclaimer that prefill is not separately scored. It is
not a statement that prefill enters `raw_p`.

</details>

*(End of retracted r3 text. See the 2026-08-17 correction block above.)*


## r3 results

Base `fe38ecc21e4084e4d17dac3aa76264bb5897a614`, reached on this branch through
merge `d85d22d` rather than a rebase so that the r2 result commit `2062fea`
stays in history. `git diff --stat fe38ecc d85d22d -- Sources Vendor
mtp-head.manifest.json` is empty, so the merge carried no editable drift.

All arms: 512-token seed, 512 decode tokens, golden
`e11_prose_512_512.json` (sha256 `615a1f20cae333fdb540f29e3cad71a187c449b74ca62ed195a020fa75ceb219`),
declared head staged at `head_provenance_sha256`
`07293af742df4599d94eda6e9db5782e7f5be10cd1b5fdef7691f4ef404ea81c`, no
`MLX_QWEN_MTP_*` variable set, clean worktree.

### Q1 - does the r2 lever survive the base move?

Yes, at roughly five sixths of its r2 size.

```text
Sp3  S18  scalar 0.18   mtp_seconds_per_token = 0.049472851   speedup 1.50728
Hp3  HV   per-depth h   mtp_seconds_per_token = 0.046317363   speedup 1.60907
                                              delta = -6.378 %   (r2: -7.658 %)
```

Both arms report `all_tokens_matched: true` and `residual_divergence_count: 0`
on both legs across the full 512-token window, with `parity_all_ok: true` and
`max_rejected_tail_logit_delta: 0`.

A free noise check falls out of the protocol. The serial control leg is
byte-identical work in both arms - depth 0, same target, same golden - so the
gap between the two serial readings measures cross-arm run-to-run noise directly
without spending an extra arm:

```text
Sp3 serial 0.074569518   Hp3 serial 0.074528022   -> -0.056 %
```

That is inside the clean-pair band of 0.019-0.092 % established in r2, so the
two arms ran under equivalent thermal and power conditions and the -6.378 %
candidate-leg difference is roughly 114x the observed noise on the control leg.

### Q2 - forced-depth marginal costs on the new base

Recovered from the aligned `block_request_seconds` and `effective_draft_lengths`
series in the timed reports rather than from new forced-depth arms, which the
frontier removed the hooks for. Round 0 dropped. Serial control gives `C(0)`.

```text
depth      Sp3 C(d)      Hp3 C(d)      old-base r2      C(d)/C(0) Sp3
  0        66.283 ms     66.230 ms     66.265 ms        1.0000
  1        72.657 ms     72.330 ms     72.659 ms        1.0962
  2        80.156 ms     80.125 ms     80.636 ms        1.2093
  3        96.515 ms     96.554 ms     97.106 ms        1.4561
  4       121.447 ms        --        121.890 ms        1.8322

marginal h[d] = (C(d+1) - C(d)) / C(0)
  d      E1 fit    Sp3        Hp3        old-base r2
  0      0.0842   +0.0962    +0.0921    +0.0965
  1      0.0775   +0.1131    +0.1177    +0.1204
  2      0.2426   +0.2468    +0.2481    +0.2485
  3      0.3754   +0.3761      --       +0.3740
```

Three independent readings agree, and they agree with the old base. The cost
structure did not move across either the base change or the declared-head repo
change, and the two arms agree with each other to within 0.1-0.5 % at every
shared depth - which is the evidence that the per-depth cost is a property of
the machine and model rather than of the schedule that sampled it.

The E1 vector remains accurate at `d = 0, 2, 3` and still understates `h[1]` by
about 1.5x (1.46x on Sp3, 1.52x on Hp3, 1.55x in r2). Re-fitting is out of r3
scope, so this is recorded as a recommendation, not a change.

One honesty note the doc block needs. The declared head changed between bases,
from `lowskillcoding/qwen38-mtp-head-4bit-g64@0966ddaf` to
`hf:dwsdubey/qwen3.8-27b-mtp-4bit@34ee76f6`. Both are 4-bit affine group-64
requantizations of the same pinned bf16 head with identical geometry, so the
doc block's "re-fit after any head change" trigger did fire on paper. The
measurements above are the answer to whether it mattered: it did not.

### Q3 - are the curve and the width wall still substitutes at wall 5?

Yes, and now provably rather than empirically. The extend rule is

```text
extend iff reach > h[depth] * (1 + expected) / (1 + cumH)
```

`reach` is a product of probabilities, so `reach <= 1`, and both confidence
clamps only lower it. Setting `reach = 1` therefore gives the supremum over
every acceptance profile and every prompt:

```text
curve:        d0->1 thr 0.0842 take    d1->2 thr 0.1430 take
              d2->3 thr 0.6265 take    d3->4 thr 1.0693 IMPOSSIBLE
  => max depth 3 for all inputs; width cap 5 unreachable; wall constant inert

scalar 0.18:  thresholds 0.1800 0.3051 0.3971 0.4675 0.5233 0.5684 0.6058 0.6372
  => reaches depth 8 at its supremum; the wall IS live for the scalar
```

So the wall move from 4 to 5 is a change to the baseline, not to the candidate.
"Ship the curve alone" still holds, and the two levers remain substitutes
because the curve makes the wall unreachable by construction.

The measured depth histograms confirm the analysis exactly:

```text
Sp3 scalar 0.18   maxD 4   {1:19, 2:138, 3:67, 4:20}    825 rows, 110 replays
Hp3 curve         maxD 3   {1:2,  2:236, 3:7}           743 rows,  74 replays
r2  curve         maxD 3   {1:2,  2:231, 3:13}          749 rows,  76 replays
```

The curve never reaches depth 4 on 246 rounds, and the r2 and r3 curve
histograms are near-identical, so the mechanism is reproducible across the base
move.

### Q4 - gain against the new 0.18 baseline, and what the frontier already took

```text
r2 old base, scalar 0.20   0.050634801
r3 new base, scalar 0.18   0.049472851    -2.295 % vs 0.20
r3 new base, curve         0.046317363    -8.527 % vs 0.20,  -6.378 % vs 0.18
```

Read against the correct 0.18 baseline the lever is -6.378 %, not -7.658 %. The
missing 1.28 points are not noise and not a regression in the idea: the
frontier's move from 0.20 to 0.18 is itself a step in the curve's direction and
already collected about 27 % of what the r2 measurement was crediting to the
curve. The remaining -6.378 % is the part a single scalar cannot reach, because
no scalar can simultaneously be small at depth 1 and large at depth 4.


## R3 measurement record — five arms, promoted-frontier base `fe38ecc`

**Base disclosure.** All five arms were measured with a merge-base of
`fe38ecc21e4084e4d17dac3aa76264bb5897a614`. The advisor branch then advanced
twice during the write-up, to `bc5e15fd` (the base named by the r3 assignment
event) and then to `ef16dea4`. Both moves are **byte-identical on the submitted
surface**: `git diff fe38ecc ef16dea4 -- Sources Vendor mtp-head.manifest.json
mtp-head benchmark.json fixtures` is empty, and the only changed paths are
`AGENTS.md` (a cosmetic note about where the Yukon CLI installs), campaign
records, and new advisor-side research and parity scripts. The branch has been
merged up to `ef16dea4`, so these numbers are current-base results, not stale
ones, and no arm needs replaying. Against `ef16dea4` the branch differs on the
submitted surface by exactly one file
(`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`); scope and budget checks
pass (source 2405727/3000000, growth 3524/262144).

All five arms on the same host, same 512-seed/512-decode prose golden
(`.mlxfast-private/e11/goldens/e11_prose_512_512.json`, sha256
`615a1f20cae333fdb540f29e3cad71a187c449b74ca62ed195a020fa75ceb219`), same
declared head (`head_provenance_sha256 07293af7...`, 2 files, 238937699 bytes =
manifest 238934129 + staged `config.json` 3570), no `MLX_QWEN_MTP_*` set,
`dirty=0`. Every arm: `all_tokens_matched: true`,
`residual_divergence_count: 0`, `parity_all_ok: true`,
`reference_checked_row_total == declared_rows_total`, `emitted_token_total 512`.

W&B (`wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, all `finished`,
`base_sha = fe38ecc2`):

| arm | build | W&B run | head sha at run time | window (UTC) |
|---|---|---|---|---|
| `Sp3`  | S18 | `5x66fb6b` | `075174ad` | 08:57:14-09:05:02 |
| `Sp3b` | S18 | `vw8tkex5` | `5db6e26c` | 09:16:43-09:24:55 |
| `S20p` | S20 | `8o6vg3ku` | `c91c2293` | 09:35:01-09:43:10 |
| `Hp3`  | HV  | `930e1t9u` | `e47e562f` | 09:06:18-09:14:24 |
| `Hp3b` | HV  | `83k6ipv6` | `aec51a1c` | 09:25:52-09:33:56 |

Build identity: CLI `e1d9980b...` for every arm; workers `741d3b37...` (S18),
`3ea7bb59...` (S20), `a698f506...` (HV). Inter-arm commits touched only
`research/`, never `Sources/`, so each arm's archived `source.swift` digest is
constant across its runs.

```
arm    h        mtp s/tok     speedup   meanD  accRate  maxD rounds rows replays gpuC
Sp3    0.18     0.049472851   1.507282  2.367  0.4603   4    245    825  110     37.56
Sp3b   0.18     0.049594029   1.501926  2.367  0.4603   4    245    825  110     39.65
S20p   0.20     0.049148680   1.509605  2.152  0.4888   4    250    788   85     41.71
Hp3    curve    0.046317363   1.609073  2.020  0.5352   3    246    743   74     41.19
Hp3b   curve    0.046259088   1.607356  2.020  0.5352   3    246    743   74     42.34
```

### Serial-leg noise floor (the honest error bar)

The serial leg is byte-identical work in all five arms, so its spread is a
direct measurement of session noise on this host:

```
mean 0.074426803 s/tok over 5 arms, peak-to-peak +0.503%
(0.125% relative sd over the first four)
```

Replicate-pair noise was `Sp3`/`Sp3b` +0.245% and `Hp3`/`Hp3b` -0.126%. Both
exceed the r2 clean-pair band 0.019-0.092%, so per the r3 protocol the pairs are
**not averaged**; the range is reported instead. The serial floor brackets both
pair figures, which is why these are read as session noise rather than defective
arms. `research/e11_pairnoise.py` on both pairs shows the schedule is
**bit-identical** (`effective_draft_lengths` equal element-for-element, meanD and
acceptance agreeing to 16 digits), the shift is uniform across quantiles rather
than a tail, and no stall. Stall ratios (max/p50 block latency) are 1.51-1.55 on
the MTP leg and 1.71-1.79 serial, far under the 4x guardrail.

### Thermal retraction

An earlier two-point fit claimed "hotter start -> slower, so the measured gain
understates the truth". **The sign flips with five points.** Regression of MTP
seconds/token on `gpu_c_before`:

```
slope -55.34e-6 s/tok/degC = -0.0744%/degC, r = -0.695
(critical |r| at n=5, p=0.05 is 0.878 -> not significant)
```

Colder start -> slower, consistent with clock/residency warmup rather than
throttling. The curve arms started 3.16 degC hotter on average (41.77 vs 38.60),
so the curve had a possible ~0.24 pp thermal *advantage*: -6.378% may be a
slight overstatement (~-6.14% thermally corrected). The best thermally matched
pair is `Hp3` (41.19) vs `S20p` (41.71), 0.5 degC apart, giving **-5.761%** —
simultaneously the most conservative and the best-controlled number.

### Per-depth marginal costs (Q2): the profile did not move

Round cost C(d) from aligned `block_request_seconds` and
`effective_draft_lengths`; C(0) from the serial leg (median, n=511 per arm,
66.16-66.28 ms). `h[d] = (C(d+1) - C(d)) / C(0)`.

| d | Sp3 | Sp3b | S20p | Hp3 | Hp3b | E1 fit | best estimate |
|---|---|---|---|---|---|---|---|
| 0 | 0.0962 | 0.0982 | 0.0969 | 0.0921 (n=2) | 0.0874 (n=2) | 0.0842 | **0.0971** (fit 15% low) |
| 1 | 0.1131 | 0.1153 | 0.1171 | 0.1177 | 0.1222 | 0.0775 | **0.1152** (fit 33% low) |
| 2 | 0.2468 | 0.2480 | 0.2497 | 0.2481 | 0.2472 | 0.2426 | **0.2482** (fit 2.3% low) |
| 3 | 0.3761 | 0.3751 | 0.3772 | -- | -- | 0.3754 | **0.3761** (fit 0.2% low) |

Cross-arm agreement 0.1-0.5%; cross-base movement vs the r2 old-base values
(0.0965 / 0.1204 / 0.2485 / 0.3740) is <= 0.6%. **The declared head repository
changed between bases and the cost profile did not.**

New finding: the E1 fit is **non-monotone at the first step** (h[0]=0.0842 >
h[1]=0.0775, i.e. it prices the second draft step below the first) while
measurement is strictly monotone (0.0971 < 0.1152 < 0.2482 < 0.3761). That
non-monotonicity is a fitting artifact of the forced-depth design. The doc-block
claim "0.18 is ~2.2x TOO HIGH at d=1..2" is also wrong: it is ~1.85x high at
h[0] and ~1.56x high at h[1], and ~1.4-2.1x *low* at h[2..3]. Both corrected in
the source comment.

### Width wall vs curve (Q3): the wall was never binding

Direct evidence from the reports rather than inference: every arm has
`max_draft_depth_bound = 8` and `requested_draft_depth = 8`, so the parent
offered depth 8 every round and `cap = min(8, maxDepth 8, widthCap)` = widthCap,
which is 5 normally and 8 once `fullAcceptStreak >= segmentedStreakGate`. **Zero
rounds at depth >= 5 in any arm** (`Sp3`/`Sp3b` maxD 4, `S20p` maxD 4, curve
maxD 3). Note `sdpaWidthWallDepthCap` is a *depth* cap min'd against the offer,
not a width cap, so 5 genuinely permits depth 5 — the observed stop at 4 is the
cost-model threshold rule, not the wall. The streak gate can only raise cap
5 -> 8 and is therefore irrelevant here.

The wall moved 4 -> 5 between bases and unlocked nothing: old-base scalar 0.20
had 18/253 rounds at d=4 against a wall of 4; new-base `S20p` has 16/250 at d=4
and none at d=5 against a wall of 5. So the wall was already slack on the old
base too.

Analytically the curve caps *itself*. Extend iff
`reach > h[d]*(1+expected)/(1+cumH)`, and `reach <= 1`, so `reach = 1` is the
supremum over all acceptance profiles. Curve thresholds are 0.0842, 0.1430,
0.6265, then **1.0693 at d=3 -> impossible**, hence max depth 3 for every input
— exactly what 246 rounds show. The 0.18 scalar's thresholds (0.1800, 0.3051,
0.3971, 0.4675, 0.5233, 0.5684, 0.6058, 0.6372) are all < 1, so it would reach
depth 8 at the supremum and is the arm the wall could in principle bind.
Conclusion unchanged: curve and wall are substitutes, and shipping the curve
alone still holds.

### The 0.20 -> 0.18 scalar step is a null (Q4)

`S20p` (0.20) is **0.655-0.898% faster** than the 0.18 arms. That is only
1.3-1.8x the 0.503% serial floor and larger than the 0.245% base pair noise, so
it is not resolved — but 0.20 is certainly no worse than 0.18 on this prompt, and
the frontier's 0.18 step is not demonstrated here. Q4's premise (that 0.18 is
the stronger baseline) does not hold on this prompt, so measuring the curve
against 0.18 slightly *flatters* it. Against 0.20 on the same base the curve is
-5.761% / -5.879%.

The mechanistic trend is monotone in effective conservatism (0.18 -> 0.20 ->
curve) and supports the curve's thesis of drafting less at the right depths:

```
meanD    2.367 -> 2.152 -> 2.020
accRate  0.460 -> 0.489 -> 0.535
rows       825 ->   788 ->   743
replays    110 ->    85 ->    74
rejected   313 ->   275 ->   231
s/tok   0.049473 -> 0.049149 -> 0.046317
```

