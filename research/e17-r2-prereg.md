# e17 r2 pre-registration

Assignment `qwen38-r1-e17-curve-transfer-and-refit`, revision `r2`, PR #19,
student `qwen-edward`.

Base `af80b0fc93cf20e8405631bb53365ace21a1f913`.
Committed **before** the first timed arm of r2. Nothing below is a measurement.

---

## 1. Index convention (deliverable 1), stated before anything else

Derived from the base source, not assumed. In
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift` at `af80b0fc`,
`costModelDepth` walks `depth = 0, 1, 2, ...` and the extend test for the step
**out of** depth `depth` reads the cost element at index `depth`:

```swift
let threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)   // l.629
if !(reach > threshold) { break }
```

Taking that step makes the round draft `depth + 1` tokens.

The round's verify width is settled from the **row ledger**, not from reading the
source. On r1's `english` arms, summing `(depth + 1)` over rounds reproduces
`declared_rows` exactly, while `depth` and `depth + 2` do not:

| arm | depth histogram | `sum d*n` | `sum (d+1)*n` | `sum (d+2)*n` | declared rows |
| --- | --- | --- | --- | --- | --- |
| FLAT18 | `{1:19, 2:138, 3:67, 4:21}` | 580 | **825** | 1070 | **825** |
| CURVE | `{1:2, 2:237, 3:7}` | 497 | **743** | 989 | **743** |

> **Convention, two parts, both used below:**
>
> 1. A round that drafts `depth` tokens is verified at width **`M = depth + 1`**
>    -- the pending primary plus the drafts. So **`M >= 5` is `depth >= 4`**.
> 2. **`h[i]` prices the step `depth i -> i+1`**, which takes that round's width
>    from `i+1` to `i+2`. So `h[3]` is the element that first buys `M = 5`.

Part 1 indexes a round total and part 2 indexes a step; they are consistent, and
conflating them is exactly the error this section exists to prevent. An earlier
draft of this file asserted `M = depth + 2` for the round total, which the table
above falsifies; it was corrected **before the first r2 timed arm**, and the
correction is reported rather than quietly applied.

Consequences that are fixed by this convention and not renegotiable later:

- The element that governs "go to depth 2" (i.e. draft 2 tokens, verify `M = 3`)
  is **`h[1]`**. Shipped scalar value `0.18`; r1 curve value `0.0775`; direct
  forced-depth measured marginal `0.1152`.
- r1's prose "conservatism at d2" and r1's table row "d=1 (M2->3)" name the
  **same element**, `h[1]`. There is no label slip between r1 and r2.
- `h[0]` prices `0 -> 1` / `M = 2`; `h[2]` prices `2 -> 3` / `M = 4`;
  `h[3]` prices `3 -> 4` / `M = 5`.

**This disagrees with the advisor's r2 comment §3**, which asks for a
perturbation direction that would close depth 3 by lowering the depth-2 element.
Working the algebra out (§4 below) shows the sign is the other way: on a flat
base vector, *lowering* `h[1]` cannot close depth 3, and *raising* it opens
depth 3. Per the advisor's own instruction ("if it disagrees with §3, say so and
follow your own"), r2 follows the algebra derived here.

## 2. Base-state correction: the merged depth curve is NOT on the live base

This invalidates the r2 assignment's stated premise that the control arm is the
"shipped curve", and it changes what r2 can deliver.

`git show af80b0fc:Sources/MLXFastModel/Qwen36MTPBlockSession.swift` declares:

```swift
private static let headStepCostRatio = 0.18        // l.530, a SCALAR
private static let segmentedStreakGate = 2         // l.597
```

`headStepCostRatioByDepth` and the `cumH` accumulator algebra are **absent**.
Bisected presence:

| commit | `headStepCostRatioByDepth` | `segmentedStreakGate` |
| --- | --- | --- |
| `e6e6f817` (r1 measurement base) | present | 3 |
| `b85e782` "Merge challenge frontier into advisor campaign" | **absent** | 2 |
| `d098212` | absent | 2 |
| `3c9317d` | absent | 2 |
| `af80b0fc` (r2 base) | absent | 2 |

The merge `b85e782` dropped the per-depth curve. So:

- The live shipped default **is r1's losing FLAT18 arm policy**, modulo the
  streak gate moving 3 -> 2.
- Restoring the curve is therefore not a no-op refactor. It is a genuine
  **submittable candidate**, worth about **+5%** local decode by r1 evidence.
- The r2 control arm is `S18` = unmodified `af80b0fc`, i.e. flat `0.18`. It is
  *not* the curve.

## 3. Counter-evidence already in the base: ranked M5 brackets the scalar

The base file's own doc comment (l.505-596) records ranked M5 submissions that
already swept this exact scalar. This is first-class evidence and it points the
opposite way from r1's local result.

| `headStepCostRatio` | ranked median | vs base | note |
| --- | --- | --- | --- |
| 0.14 | 2.766 | worse | |
| 0.15 | 2.667 | worse | |
| **0.18** | ~2.934 | **shipped** | local optimum |
| 0.32 (`fc62d1aa`) | 2.84585 | **-3%** | baseline leg flat 0.038092 -> 0.038070 |

At 0.32 the drafted depths per prompt fell `4.35/4.89/5.78/5.33/5.04 ->
3.36/4.01/4.53/4.03/4.76` and candidate decode time **rose 0.95%**. The comment
concludes **"this pool rewards depth"**.

Also documented in that comment: `segmentedStreakGate = 2` has ranked history
(newjordan 2.91995 promoted; hadakang 2.92976; `4650c96e` 2.93524 vs 2.93429
base); gate 1 is measured dead (2.833, -7.1%); gate 0 ties (2.9200). And
`sdpaWidthWallDepthCap = 5` has a bitwise justification: widths 6-9 drift in
top-2 values, so `attentionWithCacheUpdate` splits a 6..9-row attention into two
<=5-row sdpa calls, measured bit-exact.

**This contradicts r1's local mechanism.** r1 concluded the curve wins by
*declining* redundant target verification at `M = 4, 5`
(Spearman(g, extra-rows%) = +1.000, n = 4), i.e. by going shallower. Ranked M5
says going shallower costs 3%. Both can be true: they are different prompt
pools with different acceptance statistics. Establishing that this is a
**local-vs-ranked transfer failure, not a refit error**, is the main scientific
content of r2 and is reported whatever the timed arms say.

## 4. Sweep direction, derived

Writing the thresholds out from the code in §1, with
`reach_d = prod_{k<=d} p_k`:

```
d0 -> 1 :  reach0 > h[0]
d1 -> 2 :  reach1 > h[1] (1 + reach0) / (1 + h[0])
d2 -> 3 :  reach2 > h[2] (1 + reach0 + reach1) / (1 + h[0] + h[1])
```

`h[1]` sits in the **numerator** of the `1 -> 2` threshold and in the
**denominator** of every deeper threshold. So raising `h[1]`:

- raises the `1 -> 2` threshold (closes depth 2), and
- lowers the `2 -> 3` and deeper thresholds (**opens** depth 3+).

Lowering `h[1]` does the reverse and therefore **cannot** close depth 3 on a
flat-0.18 base. Mean drafted depth is consequently **not monotone in `h[1]`**.

A float note, checked: with a flat vector the restored `cumH` accumulator equals
the shipped `Double(depth) * h` **bitwise for depths 0-5**, and differs by 1 ulp
at depths 6-7. Over the reachable range the vector form reproduces the shipped
scalar rule term for term, so `S18` built through the vector patch would be
behaviourally identical -- which is why `S18` is instead built as the *untouched*
HEAD source and asserted byte-identical.

## 5. Arms (deliverable 3: shipped value + three perturbation points)

Sweep element: `h[1]`. Shipped `0.18`, plus three points bracketing the measured
marginal `0.1152` asymmetrically (two below the shipped value, one well above,
not a symmetric straddle):

| arm | `h[1]` | vector | role |
| --- | --- | --- | --- |
| `S18` | 0.18 | untouched HEAD scalar | **control** |
| `H1LO` | 0.0800 | flat 0.18, `h[1]` moved | below measured |
| `H1MEAS` | 0.1152 | flat 0.18, `h[1]` moved | measured marginal |
| `H1HI` | 0.3000 | flat 0.18, `h[1]` moved | above shipped |
| `CURVE` | 0.0775 | full r1 curve + `cumH` algebra | transfer candidate |
| `S18R` | 0.18 | byte-identical copy of `S18` | **noise floor** |

`S18R` is added because r1 had no arm-level noise floor. Each arm's raw ratio is
already internally thermally paired -- its serial and MTP legs come from the same
`--local-iterate` invocation -- so running five arms on one prompt does not need
a re-paired control. What r1 could *not* bound is how much of a reported `g%`
survives running the *same binary* in a different thermal slot of the session.
`S18R` is exactly that: same worker sha256 as `S18`
(`aa17ce5c064b5d1f35...`), different position. Registered before the first timed
arm with a live threshold: **a candidate whose `|g%|` does not clear `|g%(S18R)|`
is reported as not measured**, whatever its sign.

`research/e17-build.sh` asserts per arm that **only** `h[1]` moved for the `H1*`
arms, that `S18` is byte-identical to HEAD, and that the shared constants
(`acceptEMAAlpha = 0.15`, `sdpaWidthWallDepthCap = 5`,
`segmentedVerifyDepthCap = 8`, `segmentedStreakGate = 2`, both confidence lines)
are untouched on every arm.

## 6. Falsifiable predictions

Generated by `python3 research/e17_gate_sim.py --r2 --tokens 512 --trials 60`,
which reimplements `costModelDepth` line for line at `segmentedStreakGate = 2`.

### P1 -- the `S18` control reproduces r1's FLAT18 arm, not r1's CURVE arm

This is the runtime test of the static base-state finding in §2. If `af80b0fc`
really ships the scalar `0.18`, then the untouched-HEAD `S18` control must
behave like r1's **FLAT18** arm, not like r1's CURVE arm.

r1 on `english`, 512 decode tokens, base `e6e6f81`, gate 3:

| arm | depth histogram | mean depth | `M>=5` (depth>=4) | rows | accept |
| --- | --- | --- | --- | --- | --- |
| FLAT18 | `{1:19, 2:138, 3:67, 4:21}` | 2.367 | **8.57%** | 825 | 267/580 (46.0%) |
| CURVE | `{1:2, 2:237, 3:7}` | 2.020 | **0.00%** | 743 | 266/497 (53.5%) |

Predicted for `S18` on `af80b0fc`, `english`, 512 decode tokens:

- depth histogram close to FLAT18's `{1:19, 2:138, 3:67, 4:21}`, mean depth
  `~2.37`, **not** near CURVE's `2.020`;
- **`M>=5` share strictly greater than `0.00%`** and near `8.57%`; the curve
  cannot produce depth-4 rounds because `h[3] = 0.3754` prices that step out of
  reach, so any depth-4 round is positive proof the curve is absent;
- `max_depth >= 4`;
- rows/round ratio near FLAT18's `825/245 = 3.37`, not CURVE's `743/246 = 3.02`.

Honest attribution, stated up front: the gate move 3 -> 2 is **not** what this
tests. The simulator puts the gate's contribution at ~0.015 mean depth
(3.558 -> 3.573), because the width cap of 5 is rarely the binding constraint;
essentially all of the FLAT18-vs-CURVE depth gap is the **h vector** (flat 0.18
prices depth 2 and 3 at 0.18 against the curve's 0.2426 and 0.3754), worth ~1.2
mean depth in simulation. P1 tests the vector.

**Stop rule:** if the measured `S18` control instead resembles CURVE -- mean
depth near 2.02, `max_depth <= 3`, `M>=5` share `0.00%` -- then the curve *is*
live on `af80b0fc` and the §2 bisect is wrong. r2 then stops the sweep, retracts
§2, and reports that as the result.

### P2 -- mean drafted depth is not monotone in `h[1]`

Simulated mean drafted depth, 512 tokens, 60 trials, `Process(a=0.86, b=0.94,
m=4.0)`:

| arm | `h[1]` | mean depth | accept % |
| --- | --- | --- | --- |
| `S18` | 0.1800 | **3.573** | 62.51 |
| `H1MEAS` | 0.1152 | 3.430 | 63.53 |
| `H1LO` | 0.0800 | 3.366 | 63.88 |
| `H1HI` | 0.3000 | 3.075 | 61.72 |
| `CURVE` | 0.0775 | 2.388 | 72.93 |

The shipped `0.18` is a **local maximum of drafted depth**: both lower and
higher `h[1]` draft shallower. `H1HI` is predicted to be strongly **bimodal**
(`d1: 32.7%`, `d2: 1.2%`, then `d3: 17.0%`, `d4: 26.8%`, `d5: 19.7%`) -- it
declines depth entirely on the low-reach tail while opening depth 4-5 on the
high-reach rounds, exactly the numerator/denominator split of §4.

**Stop rule:** if measured mean drafted depth is monotone in `h[1]` across the
four arms, the sign analysis of §4 is wrong and it is reported as wrong.

### P3 -- the interesting coincidence, and what it predicts

P2 says the shipped `0.18` maximises drafted depth. The base's ranked M5 comment
(§3) says "this pool rewards depth" and that `0.18` is a ranked local optimum.
r1 says the local prose pool rewards *shallowness* (`g` correlates +1.000 with
declining extra rows).

Predicted, and the reason r2 is worth running at all: **`CURVE` wins locally
again on `af80b0fc`** (it is the shallowest arm by a wide margin), and the
already-recorded ranked evidence at `0.32` shows the same shallowing direction
**loses 3% on M5**. If both hold, the honest conclusion is that the local
`--local-iterate` prose fixture and the ranked hidden pool **rank depth
policies in opposite directions**, and no amount of refitting `h` on local
prompts can produce a ranked winner. That is a transfer-failure result, not a
curve-fitting result.

**Stop rule for promotion:** no arm is advanced as a submittable candidate on
local evidence alone if its mechanism is "draft shallower", because ranked M5
has already measured that direction at -3%. A shallowing arm can only be
reported as a *local* winner with the transfer caveat attached.

## 7. Measurement contract

- Base `af80b0fc93cf20e8405631bb53365ace21a1f913`; host = this M-series AWS mac;
  one model-holding process at a time via
  `research/await-lock-then-run.sh`; `macmon` thermal sampling per arm.
- 512-token seed, **512 decode tokens**, `./benchmark-qwen-mtp.sh
  --local-iterate`, prefill-inclusive. The window is never shortened.
- Sweep prompt: `english` (chosen for direct comparability with r1, whose
  measured `english` histograms were FLAT18 `{1:19, 2:138, 3:67, 4:21}` and
  CURVE `{1:2, 2:237, 3:7}`). Confirmation prompt for the winning arm:
  `technical` (r1's weakest register, `+3.923%`).
- ABBA interleave: within-prompt arm order alternates by prompt index
  (`research/e17-run.sh`).
- No `MLX_QWEN_MTP_*` environment variable is set on any timed arm
  (`research/e11-run.sh` clears them all).
- Pinned head only. No traced timing arms. No prompt selection after the fact to
  recover a headline.
- Every ratio reported in r2 is labelled with its window and with the index
  convention of §1.
- The decode -> score conversion factor (0.84228 on `e6e6f81`) is recomputed on
  `af80b0fc` from this run's own prefill and decode seconds rather than carried
  over.
- Correctness gates at 512 on **every** timed arm, in r1 form:
  `all_tokens_matched = True` and `residual_divergence_count = 0` are
  non-negotiable; `rejected_rows_reference_checked` is reported paired with
  `rejected_rows`.
- Per-arm worker sha256 recorded alongside the shared CLI sha256
  (`meta.txt` per run directory).
- r1's measured run directories were moved to `.mlxfast-private/e17/runs-r1/`
  before the first r2 arm, because `research/e11-run.sh` starts each run with
  `rm -rf` and r1 already held an `english-CURVE` built on `e6e6f81`. r2 writes
  a fresh `runs/`, so no r2 number can silently inherit an r1 measurement and
  the r1 arithmetic stays independently re-checkable via
  `research/e17_analyse.py --runs-root .mlxfast-private/e17/runs-r1`.

## 8. Signature

Committed before the first r2 timed arm. The predictions in §6 and the stop
rules attached to them are the standard r2 is judged against; a failed
prediction is reported as failed.

-- `qwen-edward`, r2, base `af80b0fc93cf20e8405631bb53365ace21a1f913`
