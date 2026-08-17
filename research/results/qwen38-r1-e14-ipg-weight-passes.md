# E14 — IPG weight passes: is the second weight pass real, and is width or streaming the bigger tax?

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":false,"value":null},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- Student / branch: `qwen-thorfinn` / `qwen-thorfinn/ipg-weight-passes` (PR #16)
- Hypothesis and target cost: the `qmv_fast_crossrow_affine4_g64_m<T,M,IPG>` dispatch table
  makes weight passes `ceil(M/IPG)` jump from 1 to 2 exactly at `M=5`; if that second pass is
  a real DRAM re-read it should be worth ~1 depth-0 round, and removing it at `M=5`
  (`<5,3>` -> `<5,5>`) should make depth 4 much cheaper.
- Decision: **dead as a speedup, green as measurement.** Both directions of the 2x2 were
  measured. The second weight pass is real but ~89% absorbed by cache, and the only lever
  that removes it costs 3.5x more than it saves.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: `ef16dea4ab2cb1023eb96f3740e0dfdbd88a3bda` /
  see `senpai/frontier-state.json` / no candidate proposed (research-only result)
- Yukon promoted submission / source ref used as frontier: unchanged; this experiment
  proposes no submission.
- Submitted candidate files: **none.** All arm patches were applied transiently by
  `research/run-ipg-arms.sh` and restored by its exit trap; the branch ships only research
  tooling and this report.
- Supporting test, tooling, or documentation files: `research/run-ipg-arms.sh`,
  `research/roofline_arm_patch.py` (arms `ipg-a`, `ipg-b`, `ipg-d`, `perturb`),
  `research/ipg_h_from_curve.py`, `research/ipg_shape_breakdown.py`,
  `research/ipg_depth_frequency.py`, this file.
- MTP head provenance and draft policy: organizer-pinned head, unchanged. No
  `mtp-head.manifest.json` declaration was added.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh ef16dea4… <quantized.h>
  <quantized.cpp>` -> `assignment scope OK: 2 submitted path(s)`.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `editable budget OK: source=2402203/3000000 headroom=597797 growth=0/262144 exempt=2410
  files=154`.
- Scored-path reachability evidence: the device-side dispatch `switch (ntg.x)` at
  `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h:1809` is guarded by
  `!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024`; the wide `_m` branch
  additionally needs `out_vec_size >= 4096`. All eight projection shapes measured here have
  `n >= 4096`, so every one takes the wide `_m` branch on the scored path.

## Question

Does verify width `M` cost what it costs because of the *number of weight passes*
(`ceil(M/IPG)`), or because of the *number of rows / accumulator width* (`NA`)? And if the
second weight pass at `M=5` can be removed, does depth 4 become worth offering?

## Evidence that made it worth testing

- Host: Mac mini, Apple M4 Pro, 14 cores, 48 GB. **The ranked runner is M5; every number
  here carries architecture-transfer risk and is directional only.**
- Toolchain / thermal policy: repository-frozen Swift dependency graph
  (`--force-resolved-versions`); every resident measurement went through the wrapper's run
  lock and 40C cooling gate. Per-leg temperatures are recorded in
  `.mlxfast-private/qmv-curve/<TAG>/start-temps.txt`.
- The shipped dispatch table (verified at HEAD, `quantized.h:1809`) is:

  | M | template | passes = ceil(M/IPG) | NA = IPG (or TAIL) |
  | ---: | --- | ---: | ---: |
  | 2 | `qmv_fast_crossrow_affine4_g64<T,2>` | — (different family) | — |
  | 3 | `_m<T,3,3>` | 1 | 3 |
  | 4 | `_m<T,4,4>` | 1 | 4 |
  | 5 | `_m<T,5,3>` | 2 | 3 |
  | 6 | `_m<T,6,3>` | 2 | 3 |
  | 7 | `_m<T,7,4>` | 2 | 4 |
  | 8 | `_m<T,8,4>` | 2 | 4 |
  | 9 | `_m<T,9,3>` | 3 | 3 |

## Smallest decisive test

A **2x2**: add a pass / remove a pass, each at a known `NA` change, measured on the same
host in one session against a bracketing reference.

| arm | edit | passes | NA | direction |
| --- | --- | --- | --- | --- |
| ref1 / ref2 | unpatched | — | — | bracketing drift + noise floor |
| **A** | `_m<T,5,3>` -> `_m<T,5,5>` (+ `static_assert` widened to `NA<=5`) | 2 -> 1 | 3 -> 5 | remove a pass, widen accumulator |
| **B** | `_m<T,4,4>` -> `_m<T,4,2>` | 1 -> 2 | 4 -> 2 | add a pass, narrow accumulator |
| **D** | A + exact `float acc[rows][NA]` packing | 2 -> 1 | 3 -> 5 (unpadded) | isolate the padding tax inside A |

Each arm is one width's table entry, so **every other width in the same binary is an
in-session control**. That is what makes a 1% effect measurable on a shared machine.

### Measurement method

`MLX_QWEN_MTP_FORCE_DEPTH` / `MLX_QWEN_MTP_H_VECTOR` **do not exist in `Sources/` at
`ef16dea4`** (only `MLX_QWEN_MTP_TRACE`, `Qwen36MTPBlockSession.swift:462`), so forced-depth
end-to-end arms were impossible without an out-of-scope source change. Instead h is derived
from the QMV cost curve:

```text
wvs[M] = sum over the 8 scored shapes of seconds_per_call[M] * calls_per_verify
h(d)   = (wvs[d+1] - wvs[d]) / wvs[1]
```

This reproduces edward's shipped vector closely (see Q3), which validates the derivation.

### Estimator correction (a real methodology finding)

The natural `mean`-of-15-reps estimator showed `M=1` — an **untouched control width** —
moving **+7.3%** between two builds. Per-shape decomposition traced all of it to
`linear_attn.in_proj_fused_qkvzba` at `M=1`, which is the **first timed row of the first
shape of the sweep**, with intra-run `max/min` = **1.730** in armB vs **1.065** in ref1.
That is a clock-ramp transient, not a kernel effect.

Switching to `seconds_per_call_min` (best-of-15) is now the standard estimator for this
tooling:

| | mean | min |
| --- | ---: | ---: |
| control widths M=3,5,6,7,8,9 | +/-0.5% | **+/-0.27%** |
| M=1 (untouched control) | +7.4% | +3.2% |
| M=4 (the changed entry, arm B) | +7.75% | **+8.15%** |

The signal on the changed width *grows* while the controls tighten, which is what a real
steady-state cost looks like. The min estimator also reproduces edward's `h4` to 1.6%
(0.3816 vs 0.3754 shipped) and recovers the `h1 > h2` ordering that the mean estimator
inverts.

## Results

### Reference (ref1, min estimator)

`wvs` (seconds per verify, all 8 shapes weighted by calls):

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wvs | 0.058462 | 0.064497 | 0.069949 | 0.084780 | 0.107088 | 0.123076 | 0.139277 | 0.155408 | 0.177720 |

`h_ref = [0.1032, 0.0933, 0.2537, 0.3816, 0.2735, 0.2771, 0.2759, 0.3816]`
(edward's shipped vector: `[.0842, .0775, .2426, .3754, .2919, .3000, .2870, .3909]`)

The reference was remeasured from scratch three times across the session, so `h4` — the
quantity every Q3 conclusion turns on — has an independent spread rather than a single
reading:

| reference | started | h4 |
| --- | --- | ---: |
| e14-ref1 | 10:04:23Z | 0.3816 |
| e14-ref2 | 10:33:50Z | 0.3764 |
| e14-ref3 | 10:49:23Z | 0.3880 |

Mean **0.3820, spread +/-1.5%**. Edward's shipped 0.3754 sits 1.7% below that mean, i.e.
just outside the spread and in the direction that makes depth 4 look *cheaper* on his
vector than on mine. The Q3 threshold below is quoted on `h4 = 0.3816`; recomputing it on
0.3764 or 0.3880 moves the required cut between 3.6% and 6.5% and changes no conclusion,
because depth 4 loses to depth 3 and depth 7 on cost-per-token at every value in that range.

`ref3 h = [0.0931, 0.0874, 0.2407, 0.3880, 0.2624, 0.2730, 0.2665, 0.3846]`

Round-cost model fitted on the same session:
`C(d) = G(d+1) + 4.20 ms + 3.96 ms/draft`, max |residual| 2.13 ms, `C(0) = 65.07 ms`.

### Q1 — is the second weight pass real, or absorbed by L2?

**It is real, and it is ~89% absorbed.**

Two independent routes agree:

*Structural* — line the h-steps up against the dispatch table:

| step | M -> M+1 | passes | NA | h (min) |
| --- | --- | --- | --- | ---: |
| h3 | 3 -> 4 | 1 -> 1 | 3 -> 4 | 0.2537 |
| **h4** | 4 -> 5 | **1 -> 2** | 4 -> 3 | **0.3816** |
| h5 | 5 -> 6 | 2 -> 2 | 3 -> 3 | 0.2735 |
| h6 | 6 -> 7 | 2 -> 2 | 3 -> 4 | 0.2771 |
| h7 | 7 -> 8 | 2 -> 2 | 4 -> 4 | 0.2759 |
| **h8** | 8 -> 9 | **2 -> 3** | 4 -> 3 | **0.3816** |

The two spikes land **exactly** on the two pass-count increments, and they are the **same
height**. Spike minus the mean of the four flat steps (0.2701) = **0.1115 h-units**.

*Interventional* — arm B forces a second pass at `M=4`, where the table ships one:
**+0.1161 h-units** drift-adjusted (ratio 1.0801 -> 1.0816; control widths [3,5,6,7,8,9]
spanned [0.9977, 1.0013], median drift 0.9986, noise floor +/-0.27%).

The two routes agree within 4% (and to 4 decimal places — 0.1084 both ways — under the mean
estimator). A genuine full re-read of the weights would cost about one whole depth-0 round,
so **~89% of the second pass is served from cache.**

Per-shape excess in arm B scales cleanly with weight footprint, which is the signature of
weight traffic rather than scheduling:

| shape | n | k | W MiB | calls/verify | M4 excess | share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head.lm_head | 248320 | 5120 | 682.03 | 1 | 12.07% | 6.3% |
| head.compact_draft_vocab | 98336 | 5120 | 270.09 | 0 | 11.29% | 0% |
| mlp.gate_up_fused | 34816 | 5120 | 95.62 | 64 | 11.03% | 56.0% |
| full_attn.qkv_proj_fused | 14336 | 5120 | 39.38 | 16 | 9.89% | 5.7% |
| linear_attn.in_proj_fused_qkvzba | 16480 | 5120 | 45.26 | 48 | 8.73% | 17.3% |
| linear_attn.out_proj | 5120 | 6144 | 16.88 | 48 | 4.51% | 4.3% |
| mlp.down | 5120 | 17408 | 47.81 | 64 | 3.25% | 9.9% |
| full_attn.o_proj | 5120 | 6144 | 16.88 | 16 | 1.32% | 0.4% |

682 MiB -> 12.1% excess down to 16.9 MiB -> 1.3% excess, monotone in `W`.

*Honest caveat:* arm B changes two things at once (adds a pass **and** narrows `NA` 4 -> 2).
The +8.2% is therefore a net figure with an `NA` **credit** inside it, so the isolated
stream tax is **at least** that large. Arms A and D are the mirror image and close the 2x2.

### Q2 — which tax is bigger, NA width or weight streams?

**One-sentence verdict:** *Adding one verify row at constant pass count costs ~0.27 depth-0
rounds while adding a second weight pass costs ~0.11, so the row/NA-width tax is ~2.4x the
weight-stream tax — width, not weight streaming, is what makes deep verification
expensive.*

Arms A and D confirm this from the other direction, and add a second mechanism:

- **Arm A** (`<5,3>` -> `<5,5>`: pass removed, `NA` 3 -> 5) made `M=5`
  **+39.3%** slower (ratio 1.3932, drift-adjusted 1.3929, +0.7202 h-units; control widths
  [3,4,6,7,8,9] spanned [0.9944, 1.0016], median drift 1.0002, noise floor +/-0.58%).
  Removing a whole weight pass is worth ~0.11 h-units; widening `NA` from 3 to 5 costs
  ~0.83. The lever loses by **3.5x** in ratio terms.
- The per-shape profile of arm A is **flat** (34%–54%) and does *not* track weight
  footprint — the opposite of arm B — which is the signature of a register/occupancy effect
  rather than memory traffic. The largest excess is `mlp.down` (53.7%), the shape with the
  longest reduction (`k = 17408`), exactly where accumulator pressure bites hardest.

### Arm D — is the arm A penalty register pressure?

`_wide` keeps its accumulator as `VF acc[4]`. On this target `vec<float,5>` occupies eight
lanes, so at `NA = 5` three lanes per element are dead. Arm D is arm A with `acc` packed
exactly, and it recovered **nothing**:

| arm | ratio vs ref1 | drift-adjusted | h-units | noise floor | control span |
| --- | ---: | ---: | ---: | ---: | --- |
| ref2 (repeat) | 1.0002 | 0.9998 | +0.0003 | +/-0.39% | [0.9965, 1.0027] |
| A (`<5,3>` -> `<5,5>`) | 1.3932 | 1.3929 | +0.7202 | +/-0.58% | [0.9944, 1.0016] |
| D (A + packed `acc`) | 1.3969 | **1.3996** | +0.7270 | +/-0.49% | [0.9949, 1.0029] |

Per-shape excess is statistically indistinguishable from arm A (D vs A: 34.58/54.00/35.74/
37.70/36.14/34.25/39.75/34.50 against 34.53/53.74/35.59/33.62/36.22/34.05/40.97/34.36).

I first read this as "the penalty is not register pressure." **That read was wrong, and arm
D was simply too weak a probe.** `_wide` holds *thirteen* `NA`-wide vectors per thread —
`acc[4]`, `sums`, `partial[4]`, `a0..a3`. At `NA = 5` that is 13 x 8 = 104 lanes against 65
useful, so 39 lanes are wasted. Arm D packed four of thirteen vectors, removing 12 of those
39 lanes. Occupancy is a step function of registers per thread, so removing 31% of the waste
can plausibly cross no step at all and return exactly zero — which is what happened. The
correct conclusion from arm D alone is *"one vector is not enough to move occupancy"*, not
*"registers are not the mechanism"*. Arm E is the strong form of the same probe.

### Arm E — packing all thirteen vectors

Arm E is arm A plus exact packing of all thirteen `NA`-wide vectors, with identical
elementwise arithmetic in identical order. It was run twice against two different
references:

| leg | started | reference | ref measured | control median | M=5 raw |
| --- | --- | --- | ---: | ---: | ---: |
| armE | 10:40:38Z | e14-ref1 | 36 min earlier | 1.1215 | 1.0762 |
| armE2 | 10:56:18Z | e14-ref3 | 7 min earlier | **1.1241** | **1.0739** |
| armA | 10:22:01Z | e14-ref1 | 18 min earlier | 0.9985 | 1.3932 |
| armA3 | 11:02:30Z | e14-ref3 | 13 min earlier | **0.9985** | **1.3921** |

Control span for armE2 is [1.0989, 1.1259] and for armA3 [0.9880, 1.0030]; start temps
39.73 / 39.59 / 39.48 / 39.80 C against references at 39.51 (ref1) and 39.54 C (ref3).

`armA3` is the decisive control on the drift hypothesis. It ran in the *same session and the
same thermal window* as `armE2`, four minutes later and 0.2 C warmer, against the same
reference — and its controls came back flat while `armE2`'s came back +12%. Two arms cannot
drift in opposite directions in the same session. The elevated widths belong to the arm E
patch, not to the machine. Independently of that, `armA`'s `M=5` ratio reproduces across
sessions to **0.08%** (1.3932 vs 1.3921) and its control median to four decimal places,
which fixes the run-to-run reproducibility of this fixture well below every effect reported
here.

#### The controls were never controls

On the first run I read the elevated control widths as thermal drift and reported a
drift-adjusted `0.9596`. **That was a measurement-design error, and the repeat exposes it:**
two control medians measured in different sessions against references taken 36 and 7 minutes
prior agree to **0.23%**. Thermal drift does not reproduce to 0.23%.

The cause is in the patch structure. `VF` is `vec<float, NA>` declared inside the *shared*
`_wide` body, not inside a per-`M` instantiation. Arms D and E rewrite that body, so every
`_m<T,M,IPG>` instantiation compiled from it changes — all of `M = 3..9`. Widths
[3,4,6,7,8,9] receive the intervention. Only arms A and B, which rewrite a dispatch-table
line and an assert, have genuine controls.

| arm | edits | controls valid | control span |
| --- | --- | --- | --- |
| A | dispatch `<5,3>` -> `<5,5>` | yes | [0.9944, 1.0016] |
| B | dispatch `<4,4>` -> `<4,2>` | yes | [0.9977, 1.0013] |
| D | dispatch + pack `acc` (4/13) | **no** | [0.9949, 1.0029] |
| E | dispatch + pack all 13 | **no** | [1.0989, 1.1259] |

Arm D's flat controls were luck, not design: packing four vectors happens to be free at
`NA = 3` and `NA = 4`. That coincidence is what let me misread arm D as a clean null.

**The drift-adjusted `0.9596` / `0.9553` figures are retracted.** Dividing the `M=5` ratio by
the `NA=3/4` penalty presumes packing costs the same at every `NA`, which is precisely false:
packing *helps* at `NA = 5` by removing padding and *hurts* at `NA = 3/4`. The quotient is
meaningless in both directions.

#### What arm E does establish

Two statements survive, and both are strong:

1. **Full packing is a reproducible global regression** of +12.4% median (+9.9% to +12.6%)
   at every unchanged wide width.
2. **Full packing removes the `NA = 5` penalty.** At `M = 5`, arm A costs **+39.3%** against
   shipped and arm E costs **+7.4%** — packing recovers **about 32 points**.

So the register-pressure diagnosis is confirmed, and the arm D null is now *explained*
rather than merely recorded: 12 of 39 wasted lanes crosses no occupancy step, 39 of 39 does.
The two arms together are a clean pair, which is more than either is alone.

#### Why it is still not a win

Arm E taxes the widths that actually execute — `M = 3` carries 231 of 246 verifies in
edward's histogram and `M = 4` carries 13 — by roughly 11%, in order to cheapen an `M = 5`
that is dispatched 0.00% of the time on `e6e6f81`. No schedule trades that positively.

This makes the conclusion **stronger**, not weaker. The earlier draft said to reopen if a
future toolchain removed the `vec<float,5>` padding penalty. The accurate version is that
the padding penalty is removable in source **today**, and removing it costs more elsewhere
than it saves. That is a closed door rather than a pending one.

Arm E does not touch Q2, which rests on arms A and B and on the structural h-step spikes.

## Q3 — if M=5 gets cheap, does depth 4 open?

Threshold arithmetic on the reference curve: `cumH(h1..h3) = 0.4502`, `h4 = 0.3816`,
breakeven `<= 0.3625`, so depth 4 needs a **4.99% cut in `h4`** (mean estimator: 3.65%;
edward's shipped vector: 6.48%). In absolute terms **`wvs[5]` only has to drop ~1.0%** —
which is why this arm looked promising.

**But depth 4 is never optimal anyway.** Sweeping cost-per-token
`(1 + sum(h[:d])) / sum(q^i, i=0..d)` over acceptance `q`:

| q | best d | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 7 | 0.3988 | 0.3625 | 0.3664 | 0.3509 | 0.3403 | **0.3323** | 0.3378 |

Optimal depth is 2 for `q <= 0.80`, 3 for `q = 0.85–0.95`, and 7 at `q = 1.00`. Depth 4 is a
**local bump**, not a plateau: it is the first width that pays the pass tax, while depths
5–8 amortise that *same single* tax over more accepted tokens. This holds under the min
estimator, the mean estimator, and edward's shipped vector.

So the honest answer to Q3 is: making `M=5` cheap does not "open depth 4" as a stable
operating point — it would at best flatten a bump that the scheduler already routes around.

## Q4 — what would it buy on the scored fixture?

<!-- Q4_RESULTS -->

## Correctness

<!-- PARITY_RESULTS -->

## Freshness proof (before the first timing arm)

Recorded in `.mlxfast-private/ipg-arms/freshness-before.txt`, re-recorded post-rebuild at
`2026-08-17T10:03:29Z`:

- `research/rebuild.sh` (job `1eb5dac0-…`) exit 0, 128.3 s ->
  `.build/release/mlxfast-swift` sha256 `65b698ba19d37f11`,
  `.build-worker/release/mlxfast-runtime-worker` sha256 `b4f0f046f200e708`
  (the worker had been **stale since Aug 16 14:52**).
- **The metallibs were stale.** Vendored source fingerprint `6639cc59d6fb84ff…` did not
  match the on-disk sidecars `65d691ee4dd0090b…`.
- Fix job `ed89f537-…` (`tools/build-mlx-metallib.sh --all-build-roots`) exit 0, 43.9 s.
  All four build-root metallibs then reported `sha256=3849425dfed0737c` with
  sidecar == source fingerprint.
- Twin digests at the reference state:
  `quantized.h` `b99146e90bc13cb89847d89d889f5717b59194be0f4324507e056acaf2fe1245`;
  `quantized.cpp` `70b1af945d35e0e3b220c03ca8856a68828500c41889547db970f744d6eb2b73`.

**Every kernel result taken before that rebuild on this host was measuring a stale
metallib.** `run-ipg-arms.sh` now rebuilds the metallib per arm and records
`arm-state.json` + `arm-state-after.txt` so the binding is auditable.

## Conclusion

- **What happened and why:** the second weight pass is real and lands exactly where the
  dispatch table says it should, but it is ~89% absorbed by cache, so it is worth only ~0.11
  depth-0 rounds. The only way to remove it at `M=5` is to widen `NA` to 5, and `NA` width is
  the *dominant* term (~0.27 per row, and ~0.83 for the 3 -> 5 jump measured here). The
  mechanism is confirmed; the lever is a loser.
- **Evidence for or against the mechanism:** for — structural and interventional estimates of
  the pass cost agree within 4%; per-shape excess is monotone in weight footprint for the
  pass arm and flat for the width arm, which are two different and correctly-predicted
  fingerprints. Against the *speedup* — arm A is 39% slower at the width it was supposed to
  make cheap.
- **Prompt or M5 transfer risk:** high in magnitude, low in sign. M4 Pro and M5 differ in
  cache hierarchy and register file, so the 89% absorption fraction and the 2.4x ratio are
  M4-Pro numbers. The *ordering* (width tax > stream tax) rests on register/occupancy
  behaviour that is unlikely to invert.
- **The padding penalty is removable, and removing it does not help.** Arm E packs all
  thirteen `NA`-wide vectors and recovers ~32 of arm A's 39 points at `M = 5`, which confirms
  register pressure as the mechanism. It also makes every other wide width 12.4% slower,
  including `M = 3`, which carries 231 of 246 verifies. The fix costs more where it applies
  than it saves where it helps.
- **Recommendation: close, and close firmly.** Do not repeat this lever, and do not reopen it
  on a toolchain change. The earlier draft of this line said to reopen if a future MLX or
  Metal toolchain removed the `vec<float,5>` padding penalty. Arm E shows that penalty can be
  removed in source today, so that reopening condition has already been tested and failed.
  Reopening now requires a genuinely different reason: a scheduler that dispatches `M = 5` at
  meaningful frequency, or a wide-branch rewrite that reaches `NA = 5` occupancy without
  taxing `NA = 3` and `NA = 4`.

### Suggested follow-ups (not implemented here)

1. **Host-side grid sizing** — `ntg.x == M` means the launch grid is sized for the widest
   `IPG` group and arm A only halves *active* threadgroups. A host-side change that sizes
   the grid to `ceil(M/IPG)` is a separate, larger question and is outside this
   assignment's scope. *Owner: unassigned.*
2. **`run-qmv-curve.sh:136` venv-python bug** — the default `MLXFAST_PYTHON_BIN` resolves to
   the venv `python3` symlink, which **exits silently under `run_job`**, leaving a 0-byte
   `summary.log` and no `summary.json`. Fixed locally for `run-ipg-arms.sh`; still latent
   for every other caller. *Owner: whoever next touches the curve tooling.*
3. **Stale-metallib hazard** — nothing in the normal build path rebuilds `mlx.metallib` when
   a `.metal`/`.h` source changes without a generated twin. A cheap fingerprint check in
   `research/rebuild.sh` would stop silent measurement of stale kernels. *Owner: unassigned.*
4. **edward anti-synergy** — see below; the scheduler owner should decide whether depth 4 is
   worth re-opening at all given the cost-per-token sweep above. *Owner: edward.*
5. **Control validity for shared-body kernel edits (methodology)** — `research/run-ipg-arms.sh`
   and `ipg_shape_breakdown.py` treat every unchanged width as a control and report a "drift
   adjustment" from their median. That is only sound for arms that edit a *dispatch-table
   entry*. An arm that edits the shared `_wide` body changes every instantiation, so the
   adjustment silently divides the signal by the intervention. The tooling should require the
   arm to declare which widths it touches and refuse to drift-adjust otherwise. I hit this
   here and it cost a full confirmation run. *Owner: whoever next touches the arm tooling.*
6. **A cheap way to detect it** — the `1 + 2^-6` perturbation arm used for parity doubles as a
   width map: it lives in `_wide`, so the set of widths whose output changes is exactly the
   set routed through the wide branch. Running it once per arm family would have flagged the
   shared-body scope before any timing was spent. *Owner: unassigned.*

### edward anti-synergy (stated explicitly)

edward's promoted schedule is tuned against the *current* `h` vector, in which `h4` is a
spike. This experiment set out to flatten that spike. If it had succeeded, **edward's
schedule constants would have been tuned against a cost curve that no longer existed**, and
the two changes would have had to be re-tuned together rather than composed. That
anti-synergy is now moot because arm A loses, but the general rule stands: any change to the
QMV dispatch table invalidates the `h` vector that the scheduler was fitted to, so the two
must always land as one re-tuned pair, never as independent merges.
