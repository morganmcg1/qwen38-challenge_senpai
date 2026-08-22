# E130 — remove the wide-QMV entry-point occupancy tax on the ranked architecture

**Verdict: terminal negative, correctly attributed. The occupancy axis is closed with a
ranked receipt behind it.**

The tax this experiment was written to remove no longer exists. E129 removed it structurally.
The pricing model that made it look valuable is refuted in magnitude and in sign. Two durable
assets came out of the work that are worth more than the arm would have been: a wired-residency
placement rule, and a runtime-witness method that the campaign has adopted.

- Student: qwen-alphonse
- PR: [#130](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/130)
- Advisor base at close: `35d8cf586b8671dc3d01faf3cdbd724ec603801b`
- **Submitted surface is byte-identical to the base.** This experiment ships no scored change.

---

## 1. What was asked, and what the answer is

**Hypothesis.** The wide-QMV entry point pays the maximum register count over every inlined
width, so on `applegpu_g17s` a dispatch at a narrow width runs at the worst width's occupancy.
Giving each width its own register footprint should raise F83-weighted resident simdgroups and
convert that into ranked candidate-leg time at roughly `0.10` to `0.30 %` per `1 %` residency.

**Answer.** The structural half already shipped, and the pricing half is false.

1. **The tax was removed structurally by E129.** `Qwen35.swift:1929` tiers entry points by
   `ipg`: `tier(m) = plan(m).ipg`. Under the promoted `onePass67` plan
   `{3:3:4, 4:4:4, 5:5:4, 6:6:4, 7:7:4, 8:4:4, 9:3:4}` there are five distinct entry points,
   not one shared switch. No width pays another width's maximum any more. Landmarks:
   `:1763 Entry.tiered`, `:1811 Table.onePass67` as `compiledDefault`, `:1873 MLX_E120_QMV_TABLE`,
   `:1887 defaultRouteWitness = "e120_default_route/tiered_switch/onepass67"`.
2. **Occupancy does not price on this kernel.** Three independent instruments agree, one of
   them a ranked receipt. See FINDING 174.

---

## 2. FINDING 174 — occupancy carries no ranked value on the wide-QMV kernel

### Instrument 1 — a ranked receipt against a measured residency loss

I censused the promoted `onepass67` plan and the `shipped` plan on the same base with the same
tool, so the promoted change is an exact counterfactual. Zero GPU. Simdgroups are derived from
registers through the floor law, budget `3968` on `applegpu_g17s` (Rule 89: the simdgroup
number is a model output; registers and spill are measurements).

| width | shipped plan | promoted `onepass67` | change |
|---|---|---|---:|
| M=6 | tier 3, 94 regs → **42 sg** | tier 6, 105 regs → **37 sg** | **−11.90 %** |
| M=7 | tier 4, 96 regs → **41 sg** | tier 7, 118 regs → **33 sg** | **−19.51 %** |
| others | unchanged | unchanged | 0 |
| **F83-weighted** | **41.129** | **38.594** | **−6.16 %** |

`623e77af` measured that exact arm at **−0.2032 % F83-weighted on the candidate leg**, which
is `0.20 %` **faster**, and it took the crown.

```
predicted   -6.16 % residency  x  0.10..0.30 %/%  =  +0.62 to +1.85 % ranked time COST
measured                                             -0.2032 %          ranked time GAIN
```

The model is refuted in magnitude and in sign. Residency fell by a sixth of the weighted
surface and time fell too. Any reading of the coefficient from this point is below the
`0.05 %/%` kill gate.

### Instruments 2 and 3

| instrument | value | kill gate |
|---|---:|---|
| ranked receipt `623e77af` against a −6.16 % weighted residency change | wrong sign, \|k\| ≤ 0.033 %/% | 0.05 %/% |
| thorfinn's E129 three-lever occupancy exponent | −0.0096 | 0.05 %/% |
| my local g16s coefficient, 2σ `[−0.0044, +0.0116]` | +0.0036 %/% | 0.05 %/% |

A fourth, weaker line agrees. E129's own census gate prints `gate 5 % STOP` for the
shared-switch to tiered comparison on both architectures, `+2.82 %` on g17s and `+1.15 %` on
g16s, so even the templating gain never cleared its own bar.

### One correction to the campaign record

The `39 → 33 sg, −15.4 %` figure in circulation came from a pre-E129 census where the shared
entry point was 101 registers. On the current base the before-value is 41 sg, so the M=7 change
is `41 → 33` = **−19.51 %** and the weighted change is **−6.16 %**. The conclusion strengthens;
only the arithmetic moves.

---

## 3. FINDING 181 — the local absolute instrument can invert in sign, confirmed independently

The advisor raised this from my own level table. My census confirms the mechanism on the
current base with a different tool, and extends it.

```
arm               arch  tier   regs  spill      arm               arch  tier   regs  spill
sumtable          g17s    5      98     0       sumtable          g16s    5      96     0
sumtable          g17s    6     105     0       sumtable          g16s    6      96     0
sumtable          g17s    7     118     0       sumtable          g16s    7      96    32 B
replica_no_table  g17s    5      98     0       replica_no_table  g16s    5      96     0
replica_no_table  g17s    6     108     0       replica_no_table  g16s    6      96    16 B
replica_no_table  g17s    7     122     0       replica_no_table  g16s    7      96    48 B
```

The `sumtable` rows reproduce thorfinn's E129 table digit for digit. The new part is the
no-table arm: **on g16s both wide-QMV arms are pinned flat at the 96-register ceiling from
NA=5 upward, and the no-table arm begins spilling one tier earlier than the chunk-sum arm**.
g17s grants 108 and 122 there with no spill at all.

So on my host the promoted mechanism is a spill generator, and on the ranked host it is a
pass-count reduction. My absolute local instrument reads `+1.07 %` slower where the ranked
frame reads `0.20 %` faster.

**Operational rule, adopted.** Before screening a mechanism on absolute local g16s candidate
seconds per token, state whether it changes the per-cell register count of any routed wide-QMV
entry point. If it does, the local instrument can be **sign-inverted**, not merely noisy, and
the mechanism must go to a register census plus a ranked receipt instead.

---

## 4. FINDING 173 and 173.1 — the wired slack is a fully subscribed FCFS queue, not headroom

The wired-residency ladder was the second half of E130. It is also a clean negative, and it
produced the durable asset below.

- The identity `capacity == active_at_sizing + (slack_mb << 20)` holds **exactly** on all 24
  resize draws, with `active_at_sizing = 26,146,704,372` bytes.

  | arm | applied bytes |
  |---|---:|
  | s64 | 26,213,813,236 |
  | s512 | 26,683,575,284 |
  | s1024 | 27,220,446,196 |
  | s2048 | 28,294,188,020 |

- The wired set at sizing is **byte-identical at every rung**. Raising the slack does not admit
  one extra byte of weight; it only widens a queue that is already spent.
- Headroom at steady state, worst over roles, is `0.023` to `0.117` MiB at every rung, so the
  slack is **98.3 % to 99.9 %** consumed. A post-wiring consumer can absorb **zero** bytes
  without displacing something.
- The ladder above 64 MiB is **null at 512 tokens**. Six contrasts, all not significant, on a
  residual sd of `0.0650 %` with 7 df:

  | contrast | pct | se | CI95 |
  |---|---:|---:|---|
  | s64→s512 | +0.0179 | 0.0531 | [−0.1077, +0.1436] |
  | s512→s1024 | +0.0437 | 0.0532 | [−0.0822, +0.1695] |
  | s1024→s2048 | −0.0009 | 0.0534 | [−0.1270, +0.1253] |
  | s512→s2048 | +0.0428 | 0.0531 | [−0.0828, +0.1684] |
  | s64→s2048 | +0.0607 | 0.0532 | [−0.0651, +0.1866] |
  | s64→s1024 | +0.0616 | 0.0531 | [−0.0640, +0.1872] |

  Ladder argmax is `s64`, the shipped value.

- Marginal rate above 64 MiB: `+3.060e-05 %/MiB`, 95 % bound `9.405e-05 %/MiB`.
- Admission screen at 64 tokens: slope `1.000` with no saturation, constant page tax
  `0.9746208190917969` MiB, head exclusion `0/24`, `largest_unwired_bytes_over_all_draws =
  254,279,680`, and **zero** wired-class shrink events.

**Honest gap.** A one-time warmup or seed-prefill cost is not refuted by this design. The
residual sd caps the prize at about `±0.13 %`.

**Retrospective, on record.** The rung-10a `−0.1968 %` anchor that motivated the ladder was
never significant: CI `[−0.4112, +0.0176]` on 2 df. Rungs 9 and 10 are withdrawn as wiring
measurements. Rungs 10a and 11 stand.

---

## 5. THE PLACEMENT RULE — the durable asset from this experiment

> A resident consumer allocated **before** `wireResidentWeightsIfEnabled()` joins
> `active_at_sizing` and costs zero slack. A resident consumer allocated **after** it competes
> for a slack that is already 98 to 100 percent spent, and there is no eviction.

**Standing instruction to the campaign:**

1. Keep the admission cut at or above **64 MiB**.
2. Raise `wiredZHDefaultSlackMB` by **X** for any post-wiring resident consumer of **X** MiB.

Worked example, Askeladd's C1 consumer at 31.84 MB = 30.36 MiB: allocated before wiring it is
free, because the physical spare is 13.09 GiB. Allocated after wiring it costs `+0.00093 %`
at the point estimate and `0.00286 %` at 95 %, and the safe slack becomes 94.36 MiB.

---

## 6. THE GATE-LOWERING RECIPE — reusable, reproduced verbatim

This is the method for measuring a mechanism whose guard does not pass on the local host. It is
recorded here rather than in a PR comment because it is a reusable research tool.

1. Patch `Qwen36MTPBlockSession.swift`, replacing the `UInt64(96)` guard with:

```swift
let gateGiB = environment["MLX_E130_WIRED_GATE_GIB"]
    .flatMap(UInt64.init) ?? 96
guard ProcessInfo.processInfo.physicalMemory >= (gateGiB << 30)
else { e130ResidencyProbe(clearsWarmCache: true); return }
```

Known-good text is at `git show cbf87ee8:Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
lines 311–332.

2. `export MLX_E130_WIRED_GATE_GIB=32`. The host is 48 GiB and the `MLX_` prefix is on the
   worker environment allowlist.
3. `export MLX_E130_RESIDENCY_PROBE_PATH=<per-leg path>`. Assert one line per leg with
   `applied > 0`, and run one `DARKBLOOM_QWEN_MTP_WIRED_ZH=0` leg to prove the check can fail.
4. Remove the patch before submission. `69a6d26e` is the exact revert.

**The patch must never be present on a submitted surface.** Every timed leg in this experiment
records its own `base_sha` and `worker_sha256`, and the patch state of every rung is on record:

| rung | timed base | patched? | threshold |
|---|---|---|---|
| rung 9 | `1470f0c4` | no | shipped 96 GiB |
| rung 10 | `e76bcda2` | no | shipped 96 GiB |
| rung 10a | `0bcfc5d7`, `ec300cfb` | **yes** on 4 `s` legs, deliberately **no** on 2 `none` legs | 32 / 96 GiB |
| rung 11 | `cbf87ee8` | **yes**, all 13 legs | 32 GiB |
| rung 12 | `42cdd52d` | no | shipped 96 GiB |

---

## 7. Rules owned, in my own words

**Rule 109 — prove the guard passes at runtime, on the measuring host.**
A compiled-in mechanism is not a running mechanism. Every gated arm needs a per-leg runtime
witness, and that witness is worthless unless I have also shown it can fail. In this experiment
the witness is a line written to `MLX_E130_RESIDENCY_PROBE_PATH` at
`Qwen36MTPBlockSession.swift:271-275`, captured per leg at `research/e130_rung10a_leg.sh:81-84`.
It fires on all 13 rung-11 legs with `slack_mb` matching the arm label, it fires with the wrong
polarity on both rung-10a `none` legs, and it is **absent on all twelve rung-12 legs** exactly
as the shipped 96 GiB guard on a 48 GiB host requires. That absence is the demonstrated failing
polarity, and it is why the witness is evidence rather than decoration.

**Rule 101, third occurrence — a string in the binary is not an executed code path.**
I used `senpai/rebuild-and-assert-worker.sh --require` to witness kernel content and treated a
present string as proof the path ran. A string table proves a literal was compiled in, not that
a dispatch reached it. For Swift identifiers the correct flag is `--require-symbol`, and for
behaviour the correct evidence is a runtime witness of the kind above. This is the third time
this campaign has paid for the same confusion, and I paid it after reading the rule.

**Rule 110 — a warm-phase change is worth zero unless it moves a pipeline cache key.**
Before I propose or price any warm-parity mechanism I must read the `kname` construction, the
function-constant list, and the `hash_name` construction for the kernel I claim to warm, and
name the exact field my change moves. **If I cannot name the field, the mechanism has no cost
to recover and I do not get to run it.** My qL warm-parity proposal died here:
`scaled_dot_product_attention.cpp:341-348` keys on dtype and two shape entries, `:364-371` sets
six function constants, `:374-378` builds `hash_name`, and `:392` passes the key length as a
**runtime argument** through `set_bytes(N, 5)`. Neither `qL` nor `kL` is in the key, so
`[1, 5, 4]` and `[1, 2, 3, 4, 5]` compile identical pipelines.

**Rules 62 and 63, my own restatement.** Before attributing a board delta to a mechanism,
extract the per-prompt candidate vector. If I cannot, I have a rumour and not a measurement.
A published median is a median of eight ratios of independently drawn legs; a byte-identical
redraw moved one by `−0.39 %`.

**Rule 112, accepted.** The `se` printed by `candmean.py` is a within-run per-prompt term and
ignores the run-level candidate random effect at sd `0.0470 %`. The null sd of a pair difference
is `0.0665 %`. Divide any printed `z` by about three; the 2σ single-receipt bar on the candidate
mean is `0.133 %`.

---

## 8. The rung-12 anchor — the campaign's current same-host absolute level

13 legs, 512 decode tokens, all untraced, one base, one binary, `all_tokens_matched=true` ×13.
Base `42cdd52d`, submitted surface byte-identical to `35d8cf58`. Worker
`e3bfdf904fc9eeed1a078f7dd55011fa14d3634e641ce304e49aeff11e47b3c9`.

| estimate | s/token |
|---|---:|
| **pooled over 12 fitted legs** | **0.03229839** |
| robust, drop leg 07 | 0.03228942 |

Both arms are provably identical machine code, so the level is pooled. Leg 07 sits at
`+0.3145 %`, a leave-one-out excursion of `5.6 sd`, unexplained and not chased. Residual sd is
`0.1092 %` on the full fit and `0.0561 %` on the robust fit.

**Null control.** `none → s512` is a contrast between two identical binaries.

| fit | n | df | effect | se | CI95 | t | verdict |
|---|--:|--:|---:|---:|---|---:|---|
| all 12 legs | 12 | 9 | +0.0853 % | 0.0631 | [−0.0574, +0.2280] | 1.352 | ns |
| drop leg 07 | 11 | 8 | +0.0329 % | 0.0340 | [−0.0455, +0.1112] | 0.967 | ns |

The ladder design and fit code do not manufacture effects. This retro-validates the six null
contrasts in §4.

**The warmup leg works.** Entry spread over the twelve fitted legs is `1.757 C`, against
`23.27 C` inside the rung-11 fit which had no warmup leg. Carry the warmup leg forward as
standard practice.

---

## 9. 🔴 Defect found while closing out — the pre-submit occupancy gate is blind and fails open

`senpai/entry-point-cliff-census.sh --base 35d8cf58…` returns `verdict: PASS` while reporting
`no complete Route B QMV surface`. It cannot see a single scored wide-QMV entry point.

| where | what |
|---|---|
| `Qwen35.swift:1572` | signature is now `private func qwen35E120QMVSource(table: Bool, tier: Int?) -> String {` |
| `research/e131_kernel_sources.py:123` | searches for `func qwen35E120QMVSource(table: Bool) -> String {` |
| `research/e131_kernel_sources.py:125` | raises `SourceUnavailable("qwen35E120QMVSource is gone")` |
| `research/e131_cliff_gate.py:134` | catches it and continues |
| `research/e131_cliff_gate.py:433-435` | appends a **warning** |
| `research/e131_cliff_gate.py:476` | `"verdict": "fail" if failures else "pass"` → **pass** |

E129 added the `tier:` parameter. The shell wrapper documents `exit 2` as "the gate could not
run" and never reaches it. Two further drifts sit behind the signature: `:132` expects
`let cases = [...]` where the plan is now `Qwen35CustomQMV.widthPlan.filter{…}.map{…}`, and the
case template now interpolates `\(plan.m)`, `\(plan.ipg)`, `\(plan.rps)` and `\(2 * plan.rps)`.

**Working substitute today:** `research/e129_entry_point_census.py --table <plan>` is
tier-aware and correct. Everything in §2 and §3 used it.

**Repair recipe.** Match the current signature; read the compiled default from
`defaultRouteWitness` at `Qwen35.swift:1887` and parse the `m:ipg:rps` triples from that table's
`witness` literal at `:1852-1863`, both of which are pinned to the plan by
`planWitnessMatchesWidthPlan` and `defaultRouteWitnessNamesTheCompiledDefaults`; emit one
library member per distinct `ipg` named by `qwen35E120QMVName(table:tier:)` at `:1620-1637`;
then make an unavailable ranked Route B surface a **failure**. The last step must not land
without the first three, or every submission blocks.

I did not make these changes. `research/e131_*.py` is shared pre-submit tooling and outside
this experiment's question.

---

## 10. Reproduction

```bash
# rung 12, the absolute level and the null control (13 GPU legs, 512 tokens)
research/e130_rung12_session.sh e130-r12base 512
python3 research/e130_rung12_base.py --wandb

# FINDING 174, zero GPU
python3 research/e129_entry_point_census.py --table onepass67 \
  --out research/e130-artifacts/rung13-tier-census-onepass67.json
python3 research/e129_entry_point_census.py --table shipped \
  --out research/e130-artifacts/rung13-tier-census-shipped.json

# FINDING 173 headroom read, zero GPU
python3 research/e130_headroom_read.py --selftest
python3 research/e130_headroom_read.py --wandb

# the blind gate, zero GPU
senpai/entry-point-cliff-census.sh --base 35d8cf586b8671dc3d01faf3cdbd724ec603801b
```

Every reader carries `--selftest`, and all of them pass.

## 11. W&B runs

Project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`.

| run | link |
|---|---|
| `e130r12` rung 12 fresh base and null control | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130r12 |
| `e130r11` rung 11 slack ladder | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130r11 |
| `e130hdrm` headroom read, FINDING 173 | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130hdrm |
| `e130adm` admission screen | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130adm |
| `e130rung10a` wiring ladder | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130rung10a |
| `e130rung10` withdrawn as a wiring measurement | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130rung10 |
| `e130rung9` withdrawn as a wiring measurement | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130rung9 |
| `e130rung2` occupancy ladder, the local coefficient | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130rung2 |
| `e130rung2m1` occupancy ladder, second design | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130rung2m1 |
| `e130rcpt0` receipt readout | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130rcpt0 |
| `e130final` E130 summary run | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e130final |
| `e126rng40` the E126 census this experiment was built on | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/e126rng40 |

---

## 12. Suggested follow-ups I did not implement

1. **Repair the pre-submit occupancy gate** (§9). Recipe supplied. Small, zero GPU, and it
   protects every future submission. I can take it.
2. **Retire or re-anchor `research/e130_rung11_class_predict.py`.** It bakes in
   `ANCHOR_PCT = -0.1968`, an anchor now known never to have been significant.
3. **A one-time warmup or seed-prefill cost is still unrefuted** (§4). The 512-token steady-state
   ladder cannot see it. A design that separates round 1 from the tail would close it.
4. **`largest_unwired_bytes_over_all_draws = 254,279,680`.** One 242 MiB allocation never joins
   the wired set at any rung. Identifying it would say whether the wired set is complete.
5. **Leg 07 remains unexplained** at `5.6 sd` with clean thermals and clean safety counters. If
   another ladder shows a single excursion of that size, the host, not the design, is the
   suspect.
