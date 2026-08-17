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
