# E142 — certified verify readout: refuted at rung 0

- PR: #142, branch `qwen-askeladd/e142-certified-verify-readout`
- Base: `senpai/qwen38-mtp-r1` @ `5800f88e1666f1a01895ef03ab5dfacfbbb729e8`
- W&B: `e142-r0-certified-survival`, run `6dm53tak`
  (<https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/6dm53tak>)
- Host: `Mac16,11`, Apple M4 Pro, 20-core GPU, 48 GiB
- Result label: **not useful**. The rung 0 stop rule fires. Rungs 1 to 4 were
  not run and no candidate change reaches the submitted surface.

## Question

The scored verify readout at `Qwen36MTPBlockSession.swift:1632-1645` streams
the whole affine-4 group-64 `lm_head` (248,320 x 5,120, 715 MB) once per round
and returns only `(top2IDs, top2Values)` per row. Can a screen with an
on-device certificate and a dense fallback return bit-identical top-2 evidence
while reading a fraction of that vocabulary?

Pre-registered price: +3.0 % ranked candidate leg, band +1.5 % to +5.0 %,
minimum useful +0.8 %, refuted below +0.5 %. Measured `lm_head` share of the
medpair round: 6.6 % of 52,860 us.

## Answer

No. Every screen in this family returns the whole vocabulary.

| screen | mechanism | row survival (median) | round bytes / dense (median) |
|---|---|---|---|
| S-A | per-block max-norm bound | 1.0000 | 1.0000 |
| S-B | derived affine-2 group-64 copy | 1.0000 | 1.5556 |
| S-C | sampled 7-NN leaf radius (optimistic ceiling) | 0.9978 | not built |
| S-D | rank-2048 right singular basis of `lm_head` | 0.9977 | 1.7098 |

Stop rule: best median byte fraction 1.0000 > 0.45 **and** best p95 1.0000 >
0.70. It fires on the first clause alone. S-B and S-D read *more* than the
dense pass, so shipping either would cost ranked time, not save it.

## Why every screen fails, in one number

Each screen certifies `logit_v < s2` through a bound of the form

```text
|logit_v - approx_v|  <=  slack_v
```

and prunes row `v` only when `approx_v + slack_v < s2`. Two measured
quantities decide the whole family:

- **Certificate budget** = `s2 - typical logit` = **18.31** logits
  (`s2` median 15.69, mean logit median -2.51).
- **Cauchy-Schwarz scale** = `||h|| * ||w_v||` = **129.24**
  (`||h||` median 132.58, row norm median 0.9748).

So a screen prunes only when its relative bound quality

```text
epsilon = slack / (||h|| * ||w_v||)
```

falls below `18.31 / 129.24 = 0.1417`. Measured `epsilon`:

| screen | epsilon | slack / budget |
|---|---|---|
| S-A block max-norm | 1.192 | 8.41 |
| S-B affine-2 copy | 0.564 | 3.98 |
| S-C leaf radius (ceiling) | 0.479 | 3.38 |
| S-D rank 128 | 0.870 | 6.14 |
| S-D rank 256 | 0.823 | 5.81 |
| S-D rank 512 | 0.757 | 5.35 |
| S-D rank 1024 | 0.653 | 4.61 |
| S-D rank 2048 | 0.478 | 3.37 |

The tightest bound anyone in this family produced is 3.4x too loose. The
target `epsilon` is 0.14 and the measured floor is 0.48.

### Two independent causes

**1. The logit is a small residue of a large norm product.** The hidden state
has `||h||_1 / ||h||_2 = 51.9` against `sqrt(5120) = 71.6`, so it is close to
dense and isotropic. A logit of 22.6 comes out of a norm product of 129.2, so
the answer lives at 17 % of the scale that any norm-based bound must work in.
The 18-logit budget is 14 % of that scale, and no worst-case bound over 5,120
coordinates gets that tight.

**2. `lm_head` is not low rank, and the hidden state does not live in its
dominant subspace.** S-D was added precisely because it is the only variant
whose slack does not carry a full `sqrt(d)` factor: it bounds the residual
`(w_v - P_k w_v) . (h - P_k h)` instead of the whole inner product. It fails
anyway on measured spectra:

| rank k | weight energy in top-k | `||h - P_k h|| / ||h||` |
|---|---|---|
| 128 | 0.130 | 0.949 |
| 256 | 0.172 | 0.925 |
| 512 | 0.245 | 0.894 |
| 1024 | 0.371 | 0.847 |
| 2048 | 0.582 | 0.761 |

Rank 2048 of 5,120 holds only 58 % of the weight energy, barely above the 40 %
an isotropic matrix would give, and the hidden state keeps 76 % of its norm
outside that subspace. To reach `epsilon = 0.1417` the product
`(1 - E_w) * (1 - E_h)` must fall below 0.0201; at rank 2048 it is 0.242. The
spectra are close enough to linear that only a near-full-rank basis would
qualify, and a near-full-rank projected table is not a screen.

### S-B closes the "just use a cheaper copy" idea exactly

S-B is a derived affine-2 group-64 copy of the head, so it reads 397 MB in the
coarse pass and its bound is a worst-case L1 error. Its **actual** median error
is 0.635 logits, only 0.035 of the budget — a coarse copy is accurate enough in
practice. Its **provable** error is 72.87 logits, 3.98x the budget. Inverting
the bound gives the decisive number:

```text
sb_required_bits_for_certificate = 3.694
```

A certifying copy needs more than 3.7 bits per weight, and the exact head is
4 bits. There is no cheaper representation of this head that still certifies,
so the coarse-plus-exact structure can only add bandwidth. That is exactly what
the measurement shows: 1.5556x dense.

## Evidence

### Capture

`research/e142_capture.sh` replays the cached E133 512-token goldens through an
instrumented, untimed `mtp-verify` leg and dumps the post-norm hidden rows that
reach the scored readout, together with the device `(top2IDs, top2Values)` for
each row. Six seeds, 734 rounds, 3,664 verify rows, widths 2 to 8.

| seed | rounds | rows | accepted draft rate | mean draft |
|---|---|---|---|---|
| beagle_a | 118 | 608 | 0.804 | 4.153 |
| beagle_f | 114 | 577 | 0.860 | 4.061 |
| essays_montaigne | 151 | 669 | 0.697 | 3.430 |
| medicine_hippoc | 81 | 542 | 0.935 | 5.691 |
| republic_jowett | 105 | 583 | 0.851 | 4.552 |
| travel_eothen | 165 | 685 | 0.667 | 3.152 |

Every seed reports `parity_all_ok=true`, `all_tokens_matched=true`,
`residual_divergence_count=0`. Head provenance `dadbfb806d80...` on all six.

### Thresholds

A screen needs a threshold `tau <= s2` before it knows `s2`. Three were
evaluated, and none of them is optimistic:

- `oracle`: `tau = s2` as the device measured it. Unreachable in practice, and
  therefore a strict upper bound on what any screen can prune.
- `seed`: `tau = min` of two exactly evaluated rows, the draft token and the
  screen's best-scoring row, at a cost of 5,760 B per row. The draft token is
  the argmax 85.9 % of the time.
- `noseed`: `tau =` second largest screen lower bound.

`threshold_check` confirms validity: 0 rows over tolerance for both constructed
thresholds. `seed` exceeds the device `s2` by at most 0.073 logits, inside the
offline-to-device arithmetic gap; `noseed` sits 38.3 logits below it.

Even the unreachable `oracle` threshold prunes nothing, so no better threshold
construction can rescue the family.

### Offline arithmetic validation

The offline pass dequantizes and accumulates in float32; the scored QMV
multiplies in the model dtype. Over all 3,664 rows:

- offline argmax matches the device argmax on 99.26 % of rows,
- `|delta s1| <= 0.105`, median 0.030,
- `|delta s2| <= 0.107`, median 0.021,
- the 27 argmax mismatches have median `s1 - s2 = 0.0`, that is, exact ties.

This gap is 0.107 logits against the 61.7-logit slack that decides the
question, three orders of magnitude below the effect. It cannot change any
conclusion here.

## What was measured versus what was not

- `e142_certified_survival_fraction_median` = 1.0 (primary rung 0 metric,
  minimize; stop threshold 0.45).
- `e142_net_ranked_pct` is **null**. Rung 3 measures it and rung 3 did not run.
  A missing measurement and a measured zero are different claims.
- `e142_top2_bit_identity` and `e142_screen_added_us_per_round` are **null**
  for the same reason.
- `timing_valid=false`, `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`, `official_or_ranked_score=false`. Rung 0
  is an untimed replay plus offline analysis and produces no score.

## Reproduce

```bash
research/e142_rung0.sh                                    # build + capture
research/e133_job.sh research/e142_r0.py \
  --out research/e142-r0.json                             # S-A, S-B, S-C
research/e133_job.sh research/e142_r0_lowrank.py \
  --out research/e142-r0-lowrank.json                     # S-D
research/e142_wandb_log.py --rung r0
```

Capture 745 s, S-A/B/C analysis 12 s, S-D analysis 18 s.

`research/e142_rebuild.sh` builds the instrumented worker and refuses to
proceed unless `MLX_E142_VERIFY_DUMP` and the symbol `E142VerifyDump` are both
present in the built binary, so a capture cannot silently run against a stale
worker.

## Submitted surface

The Swift instrument (`Qwen36MTPE142VerifyDump.swift` and its three-line call
at `Qwen36MTPBlockSession.swift:1649-1651`) is research-only and is reverted in
this branch. The branch leaves the submitted surface byte-identical to the
base. Only `research/` files remain.

## Suggested follow-ups

1. **Close the certified-screen family.** S-A, S-B, S-C and S-D all fail
   through the same measured ratio. Any new proposal in this family should be
   priced by its `epsilon` before implementation: it must beat 0.1417, and the
   best of four families reached 0.478. Record `epsilon` in the ledger so this
   does not get re-opened without a mechanism that changes the geometry.
2. **The 6.6 % `lm_head` share is still live, but not through pruning.** The
   pass is bandwidth bound at about 196 GB/s. The remaining levers read the
   same bytes faster or fewer times: fusing the readout with the preceding
   norm, avoiding a separate materialisation of the `[width, 248320]` logit
   tensor, or a top-2 reduction that never writes full logits to memory. None
   of those needs a certificate.
3. **Approximate screens are viable but out of contract.** S-B's *actual*
   median error is 0.035 of the budget, so a 2-bit screen would in practice
   pick the right rows almost always. It cannot be used here because the
   contract needs exact top-two evidence per row and an approximate screen has
   no certificate. Do not confuse the two: the failure is in the provable
   bound, not in the practical accuracy.
4. **Reuse the capture.** The 3,664 dumped hidden rows and their device top-2
   are cached at
   `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e142/verifyrows/`. Any future readout
   experiment that needs real post-norm inputs can skip the 745 s replay.
