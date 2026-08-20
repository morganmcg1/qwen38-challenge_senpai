# E74 — the in-situ working-threadgroup knee

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"grid_contrast_D_over_level_at_two_groups","available":true,"value":0.0571},"test_metric":{"name":"all_tokens_matched","available":false,"value":null}}
```

- Student / branch: `qwen-askeladd` / `qwen-askeladd/e74-in-situ-threadgroup-knee`
- Hypothesis and target cost: the QMV cost per byte carries a working-threadgroup
  occupancy term in situ, not only in a shapes-only microbenchmark. The target
  cost is the verify-width tax, which E71 measured at 60.7 ms per decode block at
  M=6 on this host.
- Decision: **green on the question, amber on the number.** The knee exists in
  situ and is decisive. Its numeric location is measured but is **not certified**,
  because the pre-registered cross-instrument positive control failed.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit: `d19d6f5c9612da313785eb32038d9e3781fcc9a4` /
  unchanged by this work / no candidate commit. **This experiment changes zero
  candidate files.**
- Yukon promoted submission / source ref used as frontier: `9ad17378`, source
  `bfab0de58d43453e506523707e1720a3485570f4`, score 3.25238228. Not exercised:
  no ranked run, no submission.
- Candidate build fingerprint: the unmodified base worker. The census arms
  intercept QMV calls at run time; they do not change the shipped kernel.
- Submitted-surface / generated-twin / metallib digests: not applicable. No
  submitted path changed, so no twin or metallib was rebuilt.
- Submitted candidate files: **none**.
- Supporting test, tooling, or documentation files:
  `research/e74_rung0.py`, `research/e74_report.py`, `research/e74-results.md`,
  `research/e71_census.sh`, `research/e71_wandb_stream.py`,
  `Tests/MLXFastTests/E71WidthTaxCensusTests.swift`.
- MTP head provenance, digest, and draft policy: unchanged. This census pins
  verify rows directly and never proposes drafts.
- Token window, fixture, reference source, and harness: seed 768 on
  `correctness_prompts/public_longcopy_gate_english_512_1024.json`,
  **`harness=local`**. No number in this report is a ranked score, and none is
  converted to one.
- Exact cell: five quantized-linear families at widths 6, 7, 8 and 9, dispatched
  through `affine_qmv_fast` and its `switch (ntg.x)` at `quantized.h:1922`.
  Source form is the JIT twin `mlx-generated/quantized.cpp`. The `_nax` variant
  is **not** exercised: this host is `applegpu_g16s`.
- Official causal path and score equation: not exercised. This is a measurement,
  and the assignment forbids converting it into a score.
- Assignment-scope preflight: all six files report `outside … editablePaths`.
- Editable source bytes / headroom / growth / exempt-head bytes:
  `source=2464949/3000000 headroom=535051 growth=0/262144 exempt=2410 files=154`.
- Scored-path reachability evidence: the census intercepts the same
  `affine_qmv_fast` entry the scored worker calls, at the same shapes, inside a
  real decode block. The null arm proves the interception itself is free.

## Evidence

- Host, instance, chip, memory profile, toolchain, and thermal policy:
  `mac-mini.local`, Apple M4 Pro, **20 GPU cores** read from `ioreg`
  `gpu-core-count`, 48 GiB, `applegpu_g16s`,
  `max_recommended_working_set_size` 40200896512.
  **Ungated and counterbalanced.** `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`, `official_or_ranked_score=false`, all
  preserved verbatim in `research/out/e74-census-r1/census.json`. Entry
  temperature over 130 blocks: min 51.09 °C, max 64.42 °C, mean 62.01 °C,
  **spread 13.33 °C**. Exit temperature mean 73.33 °C. Every arm is measured as
  an ABBA quartet, so monotone thermal drift cancels to first order within a
  quartet. **This is directional causal evidence within one counterbalanced
  session. It is not gate-qualified and it is not comparable to a gated
  historical run.**

- Exact baseline and candidate commands:

  ```bash
  # rung 0, before any GPU work
  python3 research/e74_rung0.py --json research/out/e74-rung0.json

  # rung 1, one session, 130 blocks, about 25 minutes
  export MLXFAST_CENSUS_EXPERIMENT=e74-in-situ-threadgroup-knee
  export MLXFAST_CENSUS_GROUP=e74-threadgroup-knee
  export MLXFAST_E71_ARM_WIDTHS=6,7,8,9
  research/e71_census.sh e74-census-r1 full

  # rungs 2 and 3
  python3 research/e74_report.py research/out/e74-census-r1/census.json \
      --json research/out/e74-census-r1/report.json
  ```

  There is no baseline arm in the usual sense. Each quartet is
  `baseline, arm, arm, baseline` at one width, and the tax is the arm minus the
  bracketing baselines.

- W&B runs:
  - main session `0orl4f8u` — <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/0orl4f8u>, state finished
  - smoke `g29ofoa9` — <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/g29ofoa9>, state finished
  - E71 census `clfgswy8` — <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/clfgswy8>,
    state finished. This is the source of every M = 4 and M = 5 cell used here
    and of the E71 side of the cross-session bridge at M = 6 and M = 9.
  - E71 plumbing smoke `3wu6kmdk` — <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/3wu6kmdk>,
    state finished.
  - E33 and E38 are other students' experiments and I have no W&B run for
    either. Their provenance here is the ledger only: item 130 at
    `senpai/campaign-ledger.md:3457` for the eight-shape ratio table, item 157
    at `:5010` (E38, thorfinn, PR #43) for the imported group term `G` and the
    independent knee bracket. Every number I take from them is quoted, never
    re-derived.

- Runtime and resources: main session 1300 s wall, model load 15.2 s, one
  model-holding process. Peak memory is the ordinary resident model; this census
  adds no allocation.

### Rung 0, pre-registered before any GPU work

Committed in `58148a0`, output in `research/out/e74-rung0.json`.

`working_tgs = ceil(M/IPG) · ceil(n/8) · B`. The host launches `M · ceil(n/8) · B`
threadgroups of 64 threads (`quantized.cpp:250-254`); the surplus x-groups
early-return at `quantized.h:1171-1174`. The shipped IPG table at
`out_vec_size >= 4096` is `{2:2, 3:3, 4:4, 5:5, 6:6, 7:4, 8:4, 9:5}`, so the
group count is **1 for M <= 6 and 2 for M = 7, 8, 9**.

The rung-0 stop rule did not fire: Kendall tau-b of ms/GB against working
threadgroups is −0.837 at M = 5, 6 and 9, and −0.598 at M = 4 with one
0.004 ms/GB inversion in a cell that failed its own null gate.

Registered bands for the primary statistic
`D = mean{fa_o_proj, gdn_out_proj} − mean{lm_head, mlp_gate_up}`:

| hypothesis | predicted `D/level` at M=7 and M=8 |
| --- | --- |
| H_knee | [0.02, 0.17] |
| H_null | [0.20, 0.42] |

Falsification required `D/level >= 0.20` at **both** new widths.

### Rung 1, the measured surface

`harness=local`. Cost curve, mean block ms, compared with the E71 session on the
same host:

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E74 | 64.745 | 67.898 | 70.535 | 80.280 | 94.318 | 122.111 | 138.407 | 149.333 | 164.703 |
| E71 | 64.706 | 67.701 | 70.652 | 80.377 | 94.349 | 122.110 | 138.334 | 149.266 | 164.790 |
| diff | +0.039 | +0.197 | −0.117 | −0.097 | −0.031 | +0.001 | +0.073 | +0.067 | −0.087 |

**The two sessions agree to better than 0.3 % at every width.** The session null
is 0.464 ms.

Null control, per width: passed at 6, 7, 8 and 9, with **no unresolved cells at
any width**. E71 left five cells unresolved; this session leaves none.

| M | null tax ms | smallest arm tax ms | fraction of smallest arm |
| ---: | ---: | ---: | ---: |
| 6 | −0.221 | 1.194 | 0.185 |
| 7 | −0.252 | 1.481 | 0.170 |
| 8 | −0.162 | 1.660 | 0.098 |
| 9 | −0.187 | 1.965 | 0.095 |

The cell surface, 20 resolved cells:

| M | family | working TGs | TGs/core | k-blocks | ms/GB | tax ms | GB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | lm_head | 31040 | 1552.0 | 80 | 3.088 | 2.208 | 0.715 |
| 6 | mlp_gate_up | 4352 | 217.6 | 80 | 3.316 | 21.281 | 6.417 |
| 6 | mlp_down | 640 | 32.0 | 272 | 5.266 | 16.896 | 3.209 |
| 6 | gdn_out_proj | 640 | 32.0 | 96 | 3.730 | 3.168 | 0.849 |
| 6 | fa_o_proj | 640 | 32.0 | 96 | 4.216 | 1.194 | 0.283 |
| 7 | lm_head | 62080 | 3104.0 | 80 | 5.021 | 3.590 | 0.715 |
| 7 | mlp_gate_up | 8704 | 435.2 | 80 | 4.770 | 30.608 | 6.417 |
| 7 | mlp_down | 1280 | 64.0 | 272 | 5.231 | 16.784 | 3.209 |
| 7 | gdn_out_proj | 1280 | 64.0 | 96 | 4.745 | 4.030 | 0.849 |
| 7 | fa_o_proj | 1280 | 64.0 | 96 | 5.232 | 1.481 | 0.283 |
| 8 | lm_head | 62080 | 3104.0 | 80 | 4.809 | 3.439 | 0.715 |
| 8 | mlp_gate_up | 8704 | 435.2 | 80 | 5.546 | 35.589 | 6.417 |
| 8 | mlp_down | 1280 | 64.0 | 272 | 5.959 | 19.120 | 3.209 |
| 8 | gdn_out_proj | 1280 | 64.0 | 96 | 5.551 | 4.715 | 0.849 |
| 8 | fa_o_proj | 1280 | 64.0 | 96 | 5.864 | 1.660 | 0.283 |
| 9 | lm_head | 62080 | 3104.0 | 80 | 6.209 | 4.440 | 0.715 |
| 9 | mlp_gate_up | 8704 | 435.2 | 80 | 6.443 | 41.350 | 6.417 |
| 9 | mlp_down | 1280 | 64.0 | 272 | 7.025 | 22.540 | 3.209 |
| 9 | gdn_out_proj | 1280 | 64.0 | 96 | 6.095 | 5.176 | 0.849 |
| 9 | fa_o_proj | 1280 | 64.0 | 96 | 6.942 | 1.965 | 0.283 |

### Cheapest real falsification gate, and the verdict

| M | groups | IPG | level ms/GB | D | **D/level** | R | spread |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1 | 6 | 3.923 | +0.771 | **+0.197** | +1.293 | 2.178 |
| 7 | 2 | 4 | 5.000 | +0.093 | **+0.019** | +0.242 | 0.486 |
| 8 | 2 | 4 | 5.546 | +0.530 | **+0.096** | +0.252 | 1.150 |
| 9 | 2 | 5 | 6.543 | +0.192 | **+0.029** | +0.507 | 0.930 |

E71 measured `D/level` = 0.256 at M=4, 0.410 at M=5 and 0.224 at M=6, all at one
group.

**Verdict: the knee is confirmed in situ.** `D/level` is 0.019 at M=7 and 0.096
at M=8. Both sit inside or below the H_knee band and far below the H_null lower
bound of 0.20. H_null is falsified. The M=7 value of 0.019 is marginally below
the registered H_knee lower bound of 0.02, which means the contrast collapsed
slightly further than H_knee itself predicted.

The **model-free** form of the same result is the strongest statement this
experiment supports:

> The three small families sit at **32 working threadgroups per core** at
> M <= 6, where the grid contrast is 0.197 to 0.410 of the level. The shipped
> table doubles the group count at M = 7, which moves them to **64 per core**,
> and the contrast falls to 0.019 to 0.096. The knee therefore lies between
> 32 and 64 working threadgroups per core on this host, at or just above 64.

Pre-registered point predictions also select the model. Four tables were
registered before measurement, one for each corner of
{grid term, no grid term} × {M=6 depth cliff resolves, persists}:

| registered table | RMSE ms/GB | max abs error |
| --- | ---: | ---: |
| **grid_responds_and_cliff_resolves** | **0.223** | 0.453 |
| no_grid_term_and_cliff_resolves | 0.354 | 0.742 |
| grid_responds_and_cliff_persists | 0.603 | 1.281 |
| no_grid_term_and_cliff_persists | 0.726 | 1.418 |

Two conclusions follow. The grid term is real, because both `grid_responds`
tables beat their `no_grid_term` twins. And **E71's M=6 reduction-depth cliff
resolves rather than persists**: `R` falls from +1.293 at M=6 to +0.242 to +0.507
at M = 7, 8 and 9. That cliff was itself an occupancy effect, not a fixed
reduction-depth tax. The registered level predictions were also close: M=7
predicted 4.966 against 5.000 measured, M=8 predicted 5.704 against 5.546, both
inside the registered ±10 %.

Cross-session bridge on the two shared widths, E74 over E71 per family: at M=6
the ratios run 0.956 to 1.031 with median 1.018; at M=9 they run 0.911 to 1.002
with median 0.988. `gdn_out_proj` is the least reproducible family.

### Rung 2, the knee, and the control that failed

Model, fitted on the 20 resolved E74 cells:

```text
ln(ms/GB) = a_M + A · max(0, ln(knee) − ln(working_tgs)) + B_w · (k_blocks − 96)
```

| quantity | value |
| --- | --- |
| knee | **1558 working threadgroups**, that is **77.9 per core** |
| knee 68 % profile interval | [1344, 2357], that is [67.2, 117.8] per core |
| A | 0.2132 per log unit of deficit |
| bootstrap knee p16 / p50 / p84 | 1305 / 1806 / 24346 |
| bootstrap A p16 / p50 / p84 | 0.042 / 0.212 / 0.312 |
| RMSE in log units | 0.0456, max residual 0.100 |

The bootstrap resamples families with replacement and keeps all widths of a
family together. Its p84 runs away to 24346 because some resamples drop the
families that carry the identifying variation. Read the profile interval, not
the bootstrap tail.

**Residual against the session null**, which the rung asks for explicitly. The
log residual above converts to milliseconds of tax by multiplying by each cell's
own traffic, so it can be compared with the 0.464 ms session null directly:

| quantity | value |
| --- | --- |
| session null | 0.464 ms |
| RMS residual over the 20 cells | 0.478 ms, **1.03× the null** |
| cells with abs residual below the null | **17 of 20** |
| the three that exceed it | `mlp_gate_up` at M = 8 (+1.483 ms), M = 9 (+1.066 ms), M = 6 (+0.755 ms) |
| largest residual in any other family | `gdn_out_proj` at M = 9, −0.413 ms |

So the fit is at the noise floor everywhere except one family. All three misses
are the same family, all have the same sign, and it is the largest-traffic cell
in the census at 6.417 GB, where a 4 % per-byte error is already 1.5 ms. A
one-family systematic of one sign is a model gap, not noise: `mlp_gate_up` is
the only census family whose `n` sits between the small three and `lm_head`, so
it is the only cell that carries information about the shape of the knee's
shoulder, and the hard-knee form has no shoulder. That is the same stiffness
that shows up as the failed E33 control below, seen from the other side.

**Independent corroboration.** Ledger item 157 R3 measured grid thinning at
identical `n` and identical traffic: "+7.4 pp at 1280 TGs decaying to ~0 at
>=4120 TGs". Under a hard knee that requires knee <= 2060 and knee >= 1280 and
implies A = 0.074/ln2 = 0.107. The fitted knee 1558 sits **inside** that prior
bracket [1280, 2060]. The fitted A is **2.0× the prior A**.

**The pre-registered positive control FAILED.**

| | value |
| --- | --- |
| gate | the fit must place E33's sign flip between 3584 and 4120 shipped working threadgroups |
| predicted flip, tax-share converted (primary) | **2719** |
| predicted flip, raw | **2577** |
| verdict | **fail** |
| knee values that would pass, at the fitted A | [2285, 2677] |
| overlap with the fitted 68 % interval | **[2285, 2357]**, non-empty |

Why it fails, mechanically: E33's flip sits between 1792 and 2060 arm
threadgroups. A hard knee at 1558 gives both of those cells a **zero** penalty,
so no flip can occur there at any level term. The control demands a knee about
1.5× larger than the point estimate. The demand and the fit are not disjoint —
they overlap in the top 3 % of the profile interval — but the point estimate
fails.

The ordering is still reproduced. Kendall tau-b of predicted against observed
E33 ratios is +0.713 in the converted variant and +0.732 raw, with **zero
discordant pairs** in both. Maximum absolute residual is 0.028 converted.

**I stopped rung 2 here, as the pre-registered stop rule requires.** I did not
retune the model, widen the gate, or add arms to chase a passing knee.

### Rung 3, the recommendation

Cost model at fixed width and fixed shape:

```text
cost(IPG) = A · max(0, ln(knee) − ln(working_tgs))   # occupancy, fitted here
          + G · (groups − 1)                          # weight passes, imported
```

**`G` is not identified by this design.** Every census family has
`out_vec_size >= 4096`, so one width has one IPG for all five families, and the
per-width intercept absorbs the whole group term. `G` is imported from ledger
item 157 R1, "the second weight pass: +0.1196" at M=6 on this host family, and
every cell is scored at three values: `G = 0`, `G = ln(1.1196) = 0.113`, and
`G = ln(1.200) = 0.182`, the top of that item's registered band.

Only three cells recommend a change on this host. Every other cell keeps the
shipped IPG.

| cell | shipped IPG | recommended | groups | robust to G | gain at G=0 | gain at G point | gain at G high |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| M4 / 4096-8191 | 4 | 2 | 1 → 2 | **no** | +0.137 | +0.034 | −0.035 |
| M5 / 4096-8191 | 5 | 3 | 1 → 2 | **no** | +0.137 | +0.034 | −0.035 |
| M6 / 4096-8191 | 6 | 3 | 1 → 2 | **no** | +0.137 | +0.034 | −0.035 |

Gains are fractions of that cell's own width tax. The 4096-8191 band carries
`fa_o_proj` 0.0267, `gdn_out_proj` 0.0708 and `mlp_down` 0.3776 of the measured
width tax, so **47.5 % of the tax sits in the one band that moves**.

Weighted by the ranked width mixture, the whole lever is worth:

| G assumption | total gain, as a share of the verify-width tax |
| --- | ---: |
| G = 0, an extra weight pass is free | **+4.68 %** |
| G = 0.113, item 157's conservative attribution | **+1.17 %** |
| G = 0.182, top of item 157's registered band | **−1.20 %** |

**The sign is not robust.** At the top of the prior's own registered band the
change is a loss.

Under each ranked core assumption, holding the knee at the fitted 77.9 per core:

| assumed cores | evidence | knee working TGs | cells that change | total gain at G point |
| ---: | --- | ---: | --- | ---: |
| 20 | the local M4 Pro count, the null assumption that nothing scales | 1558 | M4/M5/M6 at 4096-8191 | +1.17 % |
| 24 | a mid M5 Pro tier, carried only to show the shape of the dependence | 1870 | same three | +1.17 % |
| 40 | ledger 205(D): `m5-max-128gb-3` is offered only on the 40-core SKU | 3116 | the same three, **plus M4/M5/M6 at 8192-16383** | +1.17 % |

The 40-core total is **understated**, and the reason is a real gap in this
census: no family lands in the 8192-16383 or 16384-32767 bands, so a cell that
changes there contributes zero to the weighted total. Two scored shapes do live
there, `full_attn.qkv_proj` at n=14336 and `linear_attn.in_proj` at n=16480. At
40 cores those two shapes fall below the knee at M=6, but this census cannot
weight them.

**Which of the eight scored shapes fall below the ranked knee**, at every M in
the shipped table and under each core assumption. Working threadgroups take only
two values per shape across the shipped table, because the group count is 1 at
M = 3 to 6 and 2 at M = 7 to 9, so the whole table collapses to two columns of
counts. Every column here is an **extrapolation**, not a measurement: it applies
the locally fitted 77.9 threadgroups per core to an assumed ranked core count.

| shape | n | TGs at M 3-6 | TGs at M 7-9 | below knee, 20 cores | below knee, 24 cores | below knee, 40 cores |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `head.lm_head` | 248320 | 31040 | 62080 | none | none | none |
| `head.compact_draft_vocab` | 98336 | 12292 | 24584 | none | none | none |
| `mlp.gate_up_fused` | 34816 | 4352 | 8704 | none | none | none |
| `linear_attn.in_proj` | 16480 | 2060 | 4120 | none | none | M 3,4,5,6 |
| `full_attn.qkv_proj` | 14336 | 1792 | 3584 | none | M 3,4,5,6 | M 3,4,5,6 |
| `full_attn.o_proj` | 5120 | 640 | 1280 | all M | all M | all M |
| `linear_attn.out_proj` | 5120 | 640 | 1280 | all M | all M | all M |
| `mlp.down` | 5120 | 640 | 1280 | all M | all M | all M |

**Is the shipped `out_vec_size >= 4096` threshold in the right place?** Read at
M = 6 and one group, the fitted knee is an `n` threshold: everything below
**n = 12464** is grid-starved at 20 cores, below **n = 14957** at 24 and below
**n = 24928** at 40. The source gate is 4096. The gate's *direction* is right —
below it the grid is thin and the pair kernel is kept — but on these numbers it
is **3× too low locally and 6× too low at 40 cores**, and the practical
consequence is sharp: **no scored shape is below 4096 at all.** The smallest,
`mlp.down`, `full_attn.o_proj` and `linear_attn.out_proj` at n = 5120, sit just
above the gate and take the wide path while sitting at 32 threadgroups per core,
the most starved cells in the whole census. That is ledger item 130's "the tier
boundary is one shape too low", now measured in situ rather than inferred from a
microbenchmark ratio.

I still do not recommend moving the gate on this evidence, for a reason specific
to this census rather than a general caution: the gate selects between two
kernels that differ in **both** grid width and weight passes, and this design
cannot identify the weight-pass term `G` at all — it is imported from ledger
item 157, as the cost model above says. A threshold move is a
trade of one against the other, so it needs the term this experiment could not
measure. What the census can say is that the three shapes at n = 5120 are on the
wrong side of the knee under every core assumption, and that the shape most
likely to be misclassified by the gate as cores grow is `full_attn.qkv_proj`.

Extrapolation flags on every ranked column: the knee is assumed to be a per-core
capacity boundary; generation 17 is assumed to keep the same resident
threadgroups per core; and the ranked core count is itself an inference. Ledger
`:17127` records that the ranked runner's tier was never probed. None of the
three is measured.

## Conclusion

- **What happened and why.** The working-threadgroup occupancy term that E33 saw
  in a shapes-only microbenchmark is real inside a live decode block. The
  cleanest evidence needs no model: the shipped IPG table doubles the group count
  at M=7, which moves the three small families from 32 to 64 working
  threadgroups per core, and the grid contrast collapses by a factor of about
  four to ten. A pre-registered falsification band for the null was fixed before
  any GPU work and the null is falsified.

- **Is the shipped `out_vec_size >= 4096` threshold in the right place?** The
  direction is right and the location is too low. The fitted knee puts the
  starvation boundary at n = 12464 at 20 cores and n = 24928 at 40, so **no
  scored shape is below the gate**: the three shapes at n = 5120 clear it by one
  step and then run at 32 threadgroups per core, the most starved cells measured.
  I am not recommending a move, because the gate trades grid width against weight
  passes and this design cannot identify the weight-pass term.

- **Evidence for the mechanism.** Four independent readings agree. The
  pre-registered `D/level` bands select H_knee. The pre-registered point-prediction
  tables select `grid_responds` over `no_grid_term` by RMSE 0.223 against 0.354.
  The fitted knee at 1558 sits inside ledger item 157 R3's independent bracket
  [1280, 2060], measured at identical traffic with a different instrument. And
  the fit reproduces E33's per-shape ordering with zero discordant pairs.

- **Evidence against, and it is specific.** The pre-registered E33 sign-flip
  control **failed**. It needs a knee in [2285, 2677] and the point estimate is
  1558. The two overlap only in [2285, 2357], the top 3 % of the profile
  interval. The fitted A is also 2.0× the prior A. My leading explanation is that
  the two instruments should not agree: a shapes-only microbenchmark runs the QMV
  with the rest of the machine idle, so it needs more threadgroups to saturate,
  while in situ the neighbouring work of a decode block already fills the
  machine. That predicts the in-situ knee is **lower** than the microbenchmark
  knee, which is the direction observed, 78 per core against at least 114 per
  core. I did not test that explanation, so it is a hypothesis, not a result.

- **Prompt or M5 transfer risk.** High, and I flag it rather than discount it.
  One public prompt, one host, `applegpu_g16s` with `_nax` off, ungated with a
  13.33 °C entry spread. Every ranked column rests on three unmeasured
  assumptions. This experiment cannot say where the knee sits on
  `applegpu_g17s`, and the value it prices is not sign-robust even locally.

- **What this means for E73 and E72.** On the strength of these numbers the knee
  is a **weak lever on this host**: about +1.2 % of the verify-width tax at the
  prior's conservative group cost, and a loss at the top of that prior's own
  registered band. It moves only the 4096-8191 band at M = 4, 5 and 6. I would
  not spend an IPG-table change on it locally. The 40-core column is the one
  place it could matter more, and this census cannot weight that column.

- **Smallest useful next action.** Add one family in the 8192-16383 band, for
  example `full_attn.qkv_proj` at n=14336, to a single census session. That is
  the only missing weight in the whole rung-3 table, and it is the band the
  40-core extrapolation turns on. One session, same harness, no kernel change.

- **Recommendation: close the measurement, do not promote a kernel change.**
  The knee question is answered. The numeric knee should be carried as
  "77.9 per core, 68 % interval [67.2, 117.8], not certified across
  instruments", never as a transferable constant.

### Suggested follow-ups, not implemented

1. **Price the group term in situ.** The one clean way to identify `G` here is a
   family with `1024 <= out_vec_size < 4096`, which the shipped table routes to
   the pair kernel at NA=2 for every width. That breaks the confound where one
   width has one IPG for all families. Until then the rung-3 table depends on an
   imported constant.
2. **Test the idle-machine explanation directly.** Run the same five families as
   a shapes-only microbenchmark on this host in the same session as the in-situ
   census. If the microbenchmark knee comes out higher than the in-situ knee on
   one host in one session, the instrument gap is explained and E33's numbers
   can be translated rather than discarded.
3. **`gdn_out_proj` is the least reproducible family**, 0.956 and 0.911 across
   sessions while the others hold within 3 %. Worth one look before anyone
   builds on its cell.
4. **The M=6 depth cliff resolves.** E71 priced a large reduction-depth penalty
   at M=6, `R` = +1.293. It falls to +0.242 to +0.507 once the group count
   doubles. Any plan that budgets a fixed reduction-depth tax at M >= 7 should be
   rechecked against this.
