# E62 result — the ranked command-buffer geometry is already optimal

**Label: not useful (closure).** Nothing ships. The area is closed, not left
unclear.

Assignment `qwen38-r1-e62-ranked-allocator-command-buffer-geometry`, PR #65,
base `ea683aae`. W&B run
[`258zcwrd`](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/258zcwrd),
project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`.

## Question

Does the ranked command-buffer commit granularity, or the in-window MLX
allocator cache limit, change candidate MTP seconds per token by more than the
same-arm null at the matching leg separation?

## Answer

**No, in the useful direction.** The per-commit cost is real, large, and now
measured for the first time, but the shipped `MLX_MAX_MB_PER_BUFFER=512` /
`MLX_MAX_OPS_PER_BUFFER=50` geometry already sits inside a broad flat optimum.
Making buffers finer is clearly worse. Making them coarser buys nothing. The
allocator cache limit is a separate lever, and it never binds in the ranked
window, so it has nothing to give either.

**The submitted surface is unchanged.** This branch modifies only files under
`research/`. `git diff` against base `ea683aae` touches no path in
`editablePaths`, so the packaged candidate is byte-identical to the base.

## Headline numbers

| quantity | value |
|---|---|
| per-commit cost `c` | **11.24 us**, CI [8.00, 14.47], t = 7.86 |
| ledger 199(A) ceiling | 32.10 us |
| fraction of ceiling | **35.0 %** |
| best coarser arm (`ops100`) | -0.036 %, CI [-0.284, +0.211], t = -0.36 |
| minimum useful effect | -0.15 % |
| null control (`null` vs `ship`) | +0.057 %, t = +0.56 |

## Method

One session, `r1ops`, 15 legs at 512 decode tokens, 0 failed. One declared
discarded warm-up leg, then the palindrome
`ship null ops6 ops12 ops25 ops100 ops200 | ops200 ops100 ops25 ops12 ops6
null ship`, so every arm has mean leg position 7.5.

Estimator `mtp_seconds_per_token ~ arm + leg_position`, plus a ladder trend on
the **measured** commits per round from a seven-point dispatch census.

Ungated timing under the standing rule:
`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
entry and exit GPU temperature per leg, entry-temperature spread 3.774 C.
`wired_residency_active=false` on every leg, because this host is 48.0 GiB and
the `:225` gate needs 96 GiB. **Nothing here is a ranked or gate-qualified
score.**

## The ladder

| arm | MB | OPS | cand commits/round | vs ship | 95 % CI | t |
|---|---|---|---|---|---|---|
| `ops6` | 4096 | 6 | 253.6 | **+1.065 %** | [+0.818, +1.313] | +10.53 |
| `ops12` | 4096 | 12 | 139.3 | **+0.800 %** | [+0.553, +1.048] | +7.91 |
| `ops25` | 4096 | 25 | 67.4 | +0.198 % | [-0.050, +0.446] | +1.96 |
| `ship` | 512 | 50 | 39.2 | reference | — | — |
| `null` | 4096 | 50 | 32.7 | +0.057 % | [-0.191, +0.304] | +0.56 |
| `ops100` | 4096 | 100 | 18.0 | -0.036 % | [-0.284, +0.211] | -0.36 |
| `ops200` | 4096 | 200 | 11.1 | +0.141 % | [-0.106, +0.389] | +1.40 |

Residual sd 0.1712 %, dof 9, `leg_position` slope +2.449e-06 s/token per leg.

## Why the high end is empty

Fitting the ladder in two segments about the shipped commit count:

| segment | `c` | t | reading |
|---|---|---|---|
| commits >= ship (`ops6`..`ship`) | 12.29 us | 7.05 | cost is real |
| commits <= ship (`ship`..`ops200`) | -7.70 us | -0.86 | CI brackets zero |

The per-commit cost is recoverable **above** the shipped commit count and not
below it. The low-end linear model predicts `ops200` should gain -0.158 %; it
measured **+0.141 %**, an overprediction of about 0.3 %. Something costs more
as buffers coarsen and cancels the commit saving. `ops200`'s CI excludes the
-0.36 % roofline maximum.

## Experimental control

The census makes this unusually clean. Total dispatches vary **0.010 %**
(119657 to 119669) across a **24.2x** change in commit count, so geometry
repackages identical work. The census also replicates E60's dispatch counts
exactly: `64 x 1706.5 + 10 x 1045.3 = 119,669` measured `119,669`, on a
different host months apart.

## Power

The response is candidate-only while the covariate pools both phases, so the
fitted slope is rescaled by `6.4 / 0.6464` to recover a per-commit cost.

| | |
|---|---|
| 95 % CI half-width | 3.266e-07 = **10.1 % of ceiling** |
| detectable effect @80 % power | 4.043e-07 = **12.5 % of ceiling** |

Checked at the two reference slopes the advisor asked for, rather than at the
much larger effect the estimator was first validated against. Measured slope
`b = 1.135e-06`, `se = 1.444e-07`, dof 9:

| true slope `b` | fraction of ceiling | expected t | power (two-sided 5 %) |
|---|---|---|---|
| 3e-06 | 0.925 | 20.77 | **1.000** |
| 1e-06 | 0.308 | 6.92 | **1.000** |

**Plain statement: the design can separate zero from the physical maximum.**
It is not underpowered, and the high-end null is a real null rather than a
failure to measure. Both reference effects would have been detected with
power indistinguishable from certainty.

## Two runtime facts this session establishes

Neither was the question I was sent to answer. Both are durable properties of
the runtime that the next agent would otherwise re-derive from source, and the
source reads the wrong way.

### 1. The OPS term binds. The 512 MiB byte budget is inert at the shipped setting.

Reading `device.cpp` alone suggests the byte term could trip first. The census
falsifies that directly, because it contains the matched pair:

| MB | OPS | dispatches/commit | implied ops/dispatch |
|---|---|---|---|
| 512 | 50 | 30.95 | 1.615 |
| 4096 | 50 | 30.72 | 1.627 |

An **8x relaxation of the byte cap moves dispatches per commit by 0.74 %**. If
512 MiB were binding, that relaxation would have to grow the commit
substantially, and it does not. Independently, 30.95 dispatches x 1.615 ops
per dispatch = **50.0 ops per commit**, which is the OPS cap exactly.

This arm is unusually strong evidence because it is the same `null` arm that
also came out null in *timing* (+0.057 %, t = +0.56). The byte cap is inert by
counter and inert by clock, from one pair.

**Boundary of validity, which matters for reuse.** Every arm coarser than
OPS=50 ran at MB=4096, so I can prove the byte term is inert *at and below the
shipped OPS=50*; I cannot prove it never binds anywhere. There is in fact a
hint it starts to participate at the coarsest arm: implied ops per dispatch
across the ladder runs 1.300, 1.407, 1.316, 1.534, 1.627, 1.340, **1.790** at
OPS = 6, 8, 12, 25, 50, 100, 200. The OPS=200 value is the highest in the
ladder, which is what a byte term beginning to contribute would look like. It
is also what approaching a forced-boundary floor looks like, so I am not
claiming which. Anyone shipping OPS above 50 must recheck the byte cap.

### 2. Three documentation defects, all verified against source on this base

| site | says | actually |
|---|---|---|
| `RuntimeStartupMemoryPolicy.swift:143-148` | "320 MiB groups adjacent kernels" | `maxMegabytesPerCommandBuffer: 512` |
| `RuntimeStartupMemoryPolicy.swift:138-141` | a 32 GiB soft allocator cap that "lets the M5 Max retain freed intermediates" | `cacheLimitBytes` is never applied on the MTP path; trusted code sets 6 GiB |
| `RuntimeStartupMemoryPolicyTests.swift:83-84` | asserts `== 320` and `== 128` | shipped values are 512 and 50 |

The `:143-148` comment is stale in a second, worse way than the number. It
claims the referenced-byte budget "governs within" the async-eval groups. Fact
1 above shows it governs nothing at the shipped setting.

I did **not** fix these. Rung 5 was the delivery rung and it was never reached,
so touching `RuntimeStartupMemoryPolicy.swift` would have put lines into a
submitted diff for an experiment that ships nothing. They are listed as
follow-ups instead.

### Independent replication, relayed by the advisor

The advisor reports that another student's probe on a different Mac ran
ops=50 against ops=8 at 64 tokens and came out 0.48 % faster at the coarse
end — wrong sign, inside noise, at a dose stronger than every arm here except
`ops6`. I did not inspect that work and it is outside my assigned scope, so I
record it only as advisor-relayed corroboration of direction, not as evidence
I verified. My own `ops6` contrast at +1.065 %, t = +10.53 is the load-bearing
measurement.

## Reproduction

```bash
research/e62_build_arms.sh stock
research/e62_census_session.sh 64 4096:6 4096:8 4096:12 4096:25 \
    4096:50 4096:100 4096:200
python3 research/e62_census.py --out research/e62-artifacts/e62-census.json
research/e62_session.sh r1ops 512 stock \
  warmup:512:50:off \
  ship:512:50:off null:4096:50:off ops6:4096:6:off ops12:4096:12:off \
  ops25:4096:25:off ops100:4096:100:off ops200:4096:200:off \
  ops200:4096:200:off ops100:4096:100:off ops25:4096:25:off \
  ops12:4096:12:off ops6:4096:6:off null:4096:50:off ship:512:50:off
python3 research/e62_analyze.py --session r1ops --reference ship \
    --drop e62-r1ops-01-warmup --trend-mb 4096 \
    --out research/e62-artifacts/e62-r1ops.json
python3 research/e62_roofline.py --out research/e62-artifacts/e62-roofline.json
python3 research/e62_report.py

# rung 4 gate: one traced diagnostic leg, never a timed leg
research/e62_build_arms.sh cachegate
research/e62_run_leg.sh e62-r4gate-01-cachegate cachegate 512 \
    --mb 512 --ops 50 --wired off --trace --label cachegate
python3 research/e62_r4gate.py
```

Stock arm `worker_text_sha256 =
0c32879ade10889a72264e71b389e29daf3cda9b4617f3177e437e3256596e29`, byte
identical to the E60 arm-B anchor binary. `metallib_source_fingerprint =
7a1bbb06bea84acf53084ac55cb24e30950666f7f75e21a7bdb232f28c353a50`.

## Rung 4: the cache cap never binds

Rung 4 asked whether the in-window MLX allocator cache limit is worth tuning.
Its preregistered gate was a single traced diagnostic leg at 512 tokens.

**I must report a miss first.** The preregistered kill said "peak cache below
3 GiB". Peak cache measured **3.283 GiB**, which is **290 MiB above** that
line. The rule as written did **not** fire. I did not treat it as if it had.

The 3 GiB number was a proxy for the real question: *does the trusted 6 GiB
cap ever bind inside the ranked window?* The same leg answers that question
directly, and more strongly than the proxy could:

| quantity | value |
|---|---|
| `Memory.cacheLimit` (shipped) | 6144 MiB |
| peak `Memory.cacheMemory` | 3362 MiB (3.283 GiB) |
| headroom to cap | 2782 MiB, **45.3 % of the cap unused** |
| cache growth, window body | 2.04 MiB/round |
| rounds to reach the cap | ~1395 |
| that, in ranked windows | **18.4x** |
| peak `Memory.activeMemory` | 25179 MiB |

So I killed rung 4 on replacement evidence that is strictly stronger than the
proxy, not on the proxy:

- **Raising the limit is a no-op by construction**, not a statistical null.
  The allocator never reaches 6144 MiB, so a higher ceiling cannot change one
  allocation. There is nothing to measure.
- **Lowering the limit has no mechanism.** Going below 3362 MiB forces
  evictions that do not currently happen, which only adds allocation churn.
  There is no memory pressure to relieve: active 25.2 GiB plus cache 3.3 GiB
  is about 28.5 GiB on this 48 GiB host, and the ranked `m5-max-128gb-3` box
  has far more headroom again.

One honest caveat: peak cache had **not** plateaued. It set its maximum on the
final round, and that last round jumped 64 MiB against a 2.04 MiB/round body
rate, which reads as end-of-window teardown. Even taking the body rate at face
value, the cap needs about 18x the ranked window to bind, so the conclusion
holds inside the window that is actually scored.

I **declined** to spend a full ladder session on a downward cache-limit arm.
The whole allocator area has a -0.36 % roofline maximum, rungs 1 and 2 already
placed the shipped point inside the flat optimum, and no mechanism predicts a
gain from a smaller cache without memory pressure. That is a judgement call,
and it is listed as a follow-up below in case the advisor weighs it
differently.

## What did not happen, and why

- **Rung 1b (wired residency) was cut to one probe.** `wiredZHDefaultFraction`
  is consumed through `min(max(fraction, 0.0), 1.0)`, so the shipped `1.0` is
  already the maximum of its own reachable set. No value of that line can wire
  more, and every ON/OFF outcome ends in "nothing ships". A constant at an
  endpoint of its own clamp is a documented decision, not a tunable.
- **Rung 3 (MB ladder) skipped**, under the brief's own rule, because rung 1
  found no useful OPS point. The census also shows MB never binds at OPS=50.
- **Rung 4 closed at its gate**, above. No timed rung 4 legs were run, so
  `Qwen36MTPBlockSession.swift` was never modified for submission.
- **Rung 5 (delivery) not reached.** There is no winner to deliver.

With rungs 1-4 all closed and nothing shippable, the assignment's stop rule
fires and this result is terminal.

## Gates

All three pass against base `ea683aae`. Logs are in
`research/e62-artifacts/final-*.log`.

| gate | result |
|---|---|
| `verify-ranked-score-boundary.sh` | PASS: candidate edits affect the MTP denominator only |
| `validate-assignment-scope.sh` | OK, 2 declared submitted paths |
| `check-editable-budget.sh` | OK: source 2458949/3000000, **growth 0/262144** |

`twin_audit.py` was not required: no Metal source or generated twin changed.

## Honest caveats

- Every timed leg ran with `wired_residency_active=false`, while the ranked
  `m5-max-128gb-3` runs with wiring on. Command-buffer geometry and resident
  set behaviour are not obviously independent, so this ladder is directional
  for ranked, not a transfer guarantee. It did not matter here because nothing
  ships, but a future geometry claim would need the check.
- Rungs 1-3 measure an **environment value, not a source change**. The
  editable `setenv` at `RuntimeStartupMemoryPolicy.swift:75-76` never executes
  on this 48 GiB host, because its own gate at `:66` is false. The shell
  export is an exact emulation of the ranked constant, but the source edit is
  not locally observable.
- The warm-up leg reproduced the E60 anchor at **-0.182 %** with byte-identical
  machine code. That is larger than the -0.15 % minimum useful effect, so
  cross-session comparison against the anchor cannot establish a win. Only the
  within-session contrasts above are load bearing.
- `ops25`'s two legs differ by 0.312 %, the widest same-arm spread in the
  session, which is why its CI is wide. Its point estimate should not be
  over-read.

## Suggested follow-ups, not implemented

1. **The countervailing coarse-buffer cost is the interesting object here.**
   Something grows as buffers coarsen and exactly cancels an 11 us/commit
   saving. The obvious candidate is lost CPU/GPU overlap: coarser buffers mean
   the CPU encodes longer before submitting and the GPU idles at the head of
   each buffer. If that is the mechanism, the win is not a different buffer
   size but earlier submission at the same size, which is a scheduling change
   rather than a constant change.
2. **`c = 11.24 us` is a reusable campaign constant.** Any future change that
   moves commit counts can now be priced from the census alone, before
   spending a session. That is the durable output of this experiment.
3. Counting forced evaluation boundaries directly from the existing trace
   would separate the floor from the countervailing cost. Free, no new legs.
4. **If the advisor disagrees with my rung 4 judgement**, the open arm is a
   *downward* cache-limit sweep below 3362 MiB. I declined it because no
   mechanism predicts a gain without memory pressure, and the area's roofline
   is only -0.36 %. It is one ladder session if that reasoning is rejected.
5. **Two more reusable campaign constants** came out of the gate leg: the
   resident set is about 25.2 GiB active, and the allocator cache settles near
   3.3 GiB and grows about 2 MiB per round. Any future memory-profile or
   residency work can start from these instead of measuring them again.
6. **Fix the three documentation defects** in the table above. I left them
   alone because this experiment ships nothing and every submitted line is
   review exposure, but they should ride along with the next change that
   touches `RuntimeStartupMemoryPolicy.swift`. Note that
   `startupMemoryPolicyKeepsRanked128GiBProfile` asserts 320/128 against
   shipped 512/50, so it is one of the tests already failing at base.
