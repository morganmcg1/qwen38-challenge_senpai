SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"round_weighted_max_width_step_ms","available":true,"value":25.052},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

# E212 — The per-round verify cost law R_local(m) on the composed base

- Student / branch: qwen-alphonse / `qwen-alphonse/e212-width-cost-law`
- Hypothesis and target cost: the standing local width-cost law is stale. It
  predates E195 selective-m6, E193 head prefetch, and E208 staged (9,5). A
  fresh table either exposes the next staged-substitution target or closes the
  local kernel-step axis.
- Decision: **census complete, axis stays open.** Six of eight adjacent width
  boundaries carry a round-weighted step above the 0.2 ms/round bar, by factors
  of 31 to 125. No staged-substitution target follows, for the reason in
  "Owning cell" below.
- `BASE_SHA`: `523da3be` (cap-8 + staged (9,5) + E195 selective-m6).
  `UPSTREAM_SHA`: unchanged this assignment. Candidate commit: none. This
  experiment changes no scored file.
- Yukon promoted submission / source ref used as frontier: not applicable. This
  is a measurement census, not a candidate.
- Candidate build fingerprint: worker `114a8c9755d728ca8cb71301976b69a192ca8eb29f3dfb46c2f6c95bc2fc625b`,
  identical and digest-stable across all 26 legs.
- Submitted-surface / generated-twin / metallib digests: unchanged. The scored
  diff against `523da3be` across every path in `editablePaths` is empty.
- Submitted candidate files: none.
- Supporting test, tooling, or documentation files: `research/e212_session.sh`,
  `research/e212_report.py`, `research/e212_wandb.py`,
  `research/e212-report.json`, `research/e212-result.md`.
- MTP head provenance, digest, and draft policy: `head_provenance_sha256`
  `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9` on every
  one of the 26 legs. Draft policy is the shipped `draftPolicy` for the `ref`
  legs and the shipped policy under the research pin
  `MLX_E159_FIXED_DRAFT_DEPTH=N` for the forced-width legs.
- Token window, fixture, reference source, harness: 512 decode tokens, the
  public local fixture, candidate-generated local reference rows,
  **harness=local**.
- Exact cell: the seven fused routed affine-4/group-64 QMV decode cells
  (`mlp.gate_up`, `mlp.down`, `gdn.in_proj`, `gdn.out_proj`, `fa.qkv`,
  `fa.o_proj`, `lm_head`) dispatched through `qwen35QMVVariant(m:cell:)`, plus
  the 48 Gated DeltaNet and 16 full-attention mixers. Source form: JIT from the
  generated twin. M5 variant: not measured here; this host is M4 Pro.
- Official causal path and score equation: none claimed. **No number in this
  report is a ranked or official score.** The forced-width legs measure
  absolute candidate MTP seconds per token, which is ranked-relevant, but the
  width pin is a research switch and not a shipped policy.
- Assignment-scope preflight: passed. Only `research/` files changed.
- Editable source bytes / headroom / growth: unchanged, growth 0 bytes.
- Scored-path reachability evidence: every measured round ran through the
  scored `Qwen36MTPBlockSession` verify path and emitted a matched token
  stream. The census is of the live path, not a microbenchmark.
- Written promotion rule and verdict: the census itself was the terminal
  deliverable. Delivered.
- Pre-official evidence budget / timed legs used: 20 census legs (two
  counterbalanced sweeps) plus 6 band-attribution legs. Budget allowed ~18 legs
  plus at most two cell-census sessions; I used one band session of 6 legs and
  needed no tie-breaker leg.
- Frozen candidate SHA: not applicable.
- Submission owner / receipt-watcher job ID: not applicable.

## Evidence

- Host, chip, memory, toolchain, thermal policy: AWS Mac, Apple M4 Pro, 48 GiB
  unified, macOS 26.5.2, Apple Swift 6.3.3 (swiftlang-6.3.3.1.3), target
  `arm64-apple-macosx26.0`. **Thermal policy: `MLXFAST_LOCAL_COOL_GATE=0`**
  under the standing three conditions. Every leg records
  `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
  verbatim. Entry temperatures spanned 32.7–57.9 C across the 26 legs; per-leg
  entry and exit temperatures are in `research/e212-report.json` under `legs`.
- Exact commands:

  ```bash
  research/e212_session.sh s1 ref w0 w1 w2 w3 w4 w5 w6 w7 w8 \
                              w8 w7 w6 w5 w4 w3 w2 w1 w0 ref
  research/e212_session.sh s2band b4 b5 b6 b6 b5 b4
  research/e212_report.py s1 s2band --out research/e212-report.json
  research/e212_wandb.py research/e212-report.json
  ```

- Cheapest real falsification gate and positive control: `all_tokens_matched`
  on every leg, with the session driver stopping on the first mismatch. The
  gate is capable of failing: a width pin that corrupted the accept walk would
  change the emitted stream, and the driver treats any non-`true` value as a
  defect and stops the session. Additionally the m=1 arm is a positive control
  on the instrument itself, since a zero-draft MTP leg must reproduce serial
  cost, and it does to 0.16%.
- Exact-token and row-ledger verdict: **26 of 26 legs
  `all_tokens_matched=true`, `residual_divergence_count=0`.**
- Divergent tokens or failure category: none.
- Generated-twin audit: not relevant, no Metal source changed.

## R_local(m), harness=local, NOT gate-qualified

| m | R_local ms/round | leg spread | ms/token | accept rate | leg s/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 64.832 | 0.007 | 64.83 | — | 0.073260 |
| 2 | 67.724 | 0.030 | 34.26 | 0.981 | 0.042317 |
| 3 | 69.545 | 0.071 | 23.91 | 0.954 | 0.031985 |
| 4 | 76.202 | 0.018 | 19.79 | 0.951 | 0.027921 |
| 5 | 89.468 | 0.044 | **19.05** | 0.923 | **0.026996** |
| 6 | 117.989 | 0.082 | 21.89 | 0.866 | 0.029934 |
| 7 | 136.846 | 0.064 | 21.65 | 0.873 | 0.029820 |
| 8 | 145.346 | 0.194 | 20.44 | 0.864 | 0.028464 |
| 9 | 164.459 | 0.123 | 21.20 | 0.826 | 0.029262 |
| shipped adaptive | — | — | — | — | 0.029317 |

Every width's two counterbalanced legs agree inside the FINDING 503 noise floor
of 0.24 ms/round. Largest spread 0.194 ms at m=8. No tie-breaker needed.

Instrument check: the m=1 leg runs 0.073260 s/token against the same session's
serial leg at 0.073140 s/token, a gap of 0.16%. The zero-draft MTP round costs
a serial round, so R_local(m) carries no fixed session overhead.

## Step verdicts, round-weighted with the measured shipped distribution

Measured served-width shares from the two `ref` legs: m=9 0.568, m=8 0.162,
m=6 0.095, m=5 0.081, m=7 0.054, m=4 0.041.

| boundary | 1→2 | 2→3 | 3→4 | 4→5 | 5→6 | 6→7 | 7→8 | 8→9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw ms/round | 2.892 | 1.821 | 6.657 | 13.266 | 28.521 | 18.857 | 8.500 | 19.113 |
| weighted ms/round | 0 | 0 | 6.657 | 12.728 | **25.052** | 14.780 | 6.203 | 10.848 |

Six boundaries clear the 0.2 ms/round bar. The local kernel-step axis does not
close.

## Owning cell: none. The owning quantity is the group width IPG

Band-attribution legs at m=5, 6, 7 (attribution only; 129 extra syncs per
forward inflate every band-arm round, so no band-arm round is a round cost):

| boundary | gdn_mixer | gdn_mlp | fa_mixer | fa_mlp | pre |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5→6 | +8.039 (27%) | +14.402 (48%) | +2.847 (10%) | +4.462 (15%) | 0 |
| 6→7 | +3.534 (23%) | +7.803 (52%) | +1.128 (7%) | +2.588 (17%) | −0.001 |

Two facts follow, and together they refute the weight-traffic account.

**1. Weight traffic is anti-correlated with the step across these boundaries.**

| boundary | extra weight bytes | cells changing weight passes | measured step |
| --- | ---: | --- | ---: |
| 5→6 | +3.209 GB | `mlp.down` only | 28.521 ms |
| 6→7 | +11.204 GB | the other six cells | 18.857 ms |

The boundary that adds 3.5 times more weight traffic costs 34% less.

**2. At 5→6, 37% of the step lands in cells whose weight passes do not change.**
No mixer cell changes its pass count at 5→6, yet the mixer bands rise
10.886 ms. That residue is width-proportional arithmetic and occupancy cost,
and it is the component the weight-pass count cannot see.

The step is not localized to any cell. Normalized per layer it is uniform:
at 5→6 the MLP band rises 0.3000 ms per GDN layer and 0.2789 ms per FA layer,
and the mixer band rises 0.1675 and 0.1779 ms per layer respectively. All 64
layers pay it in proportion to their share of the round.

**The structural law.** Every width except m=6 dispatches one staged `(m, IPG)`
pair across all seven cells, so `(G, IPG)` is a scalar there. Reading the table
that way:

Single-pass cost against group width, at G=1:

| IPG | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: |
| f(IPG) ms/round | 67.724 | 69.545 | 76.202 | 89.468 |
| step | — | +1.821 | +6.657 | +13.266 |

The second weight pass, at fixed IPG:

| IPG | one pass | two passes | ratio |
| --- | --- | --- | ---: |
| 4 | m=4, 76.202 | m=7, 136.846 | 1.796 |
| 4 | m=4, 76.202 | m=8, 145.346 | 1.907 |
| 5 | m=5, 89.468 | m=9, 164.459 | 1.838 |

So `R(m) ≈ G(m) × f(IPG(m))` less a fixed overhead of roughly 7–16 ms, where
`f` is steeply convex in IPG: each IPG increment roughly doubles the previous
increment. This single reading explains the whole table, including the
plateau-interior growth that refuted the weight-pass model, the size of the
5→6 step (`mlp.down` gains a pass while every other cell moves from IPG 5 to
IPG 6), and why 6→7 is cheaper despite far more weight traffic (the other cells
gain a pass but drop from IPG 6 to the much cheaper IPG 4).

I stated earlier in this PR that the law is an "irregular lookup table". That
was incomplete and the band legs corrected it: the law is structured, but by
passes times per-pass cost at the dispatched group width, not by pass count.

## Comparisons

**5(a) Old versus new.** Measured minus the E182-era vintage table:

| m | 1 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| delta ms/round | −0.47 | −1.11 | −1.02 | **−8.09** | −0.52 | −0.43 | **−21.37** |

The dated table remains accurate to about 1 ms at every width no scored change
touched, and is wrong at exactly the two changed widths.

**5(b) m=9 sanity, post-(9,5).** My m=9 sits 21.37 ms/round below the dated
vintage. FINDING 543 brackets the (9,5) correction at −20.774 ms/round paired
on Edward's host. Two hosts and two instruments agree within 0.6 ms. This is
cross-host confirmation of FINDING 543, not a new claim. E195's claimed
−7.31 ms/round at m=6 likewise reproduces here at −8.09.

**5(c) Shape.** Stepped, and the steps are real, but not the steps the
weight-pass model predicts. That model predicts flat cost inside m=2..5 and
inside m=7..9; measured interior growth is +21.7 and +27.6 ms/round, each about
100 times the noise floor. A quadratic, a pure weight-pass step function, and
their sum all leave residuals of 2 to 12 ms/round. The IPG law above is the
shape that fits. Under RULE 399 this local corpus nominates the reading; only
the ranked instrument can decide a family-level claim.

## Result labels

- **Not a candidate.** No scored file changed and no speed claim is made.
- The census is a **positive measurement result**: it refreshes every desk
  instrument that consumed the dated law and it replaces the weight-pass cost
  model with a measured IPG law.

## Metrics

| Metric | Value |
| --- | ---: |
| legs run / legs matched | 26 / 26 |
| residual divergence, all legs | 0 |
| max leg spread vs 0.24 ms floor | 0.194 ms |
| largest round-weighted step | 25.052 ms/round at 5→6 |
| boundaries above the 0.2 ms bar | 6 of 8 |
| m=1 control vs same-session serial | +0.16% |
| serial seconds/token (session mean) | 0.073140 |
| shipped adaptive MTP seconds/token | 0.029317 |

## Suggested follow-ups, not implemented

1. **The shipped staged plan is already optimal at every width under this law.**
   Evaluating `c + G × (f(IPG) − c)` for every uniform alternative IPG puts the
   shipped entry first at m=7, m=8 and m=9, for every fixed overhead `c` in the
   7–16 ms range the pass-doubling pairs imply. Margin to the nearest
   alternative:

   | m | shipped | nearest uniform alternative | margin |
   | --- | --- | --- | ---: |
   | 7 | IPG 4, G=2 | IPG 7, G=1 | +23.62 ms |
   | 8 | IPG 4, G=2 | IPG 5, G=2 | +17.59 to +26.59 ms |
   | 9 | IPG 5, G=2 | IPG 3, G=3 | +12.18 to +30.18 ms |

   At m=6 the model can only score uniform plans, and the shipped mixed plan
   (117.99 measured) beats the best uniform alternative (IPG 6, G=1, 121.49
   inferred from E195). This predicts Edward's E213 retune of
   `(6,x)/(7,x)/(8,x)` finds no win. It is a falsifiable prediction from my
   table, not a request to stop that work — if E213 finds a win, my `f` is
   wrong and that is worth knowing.
2. **Measure `f(6)` and `f(7)` directly.** Both are currently inferred from
   E195's end-to-end deltas rather than measured. A short forced-width session
   with the single-pass variant forced at m=6 and m=7 would pin the convex tail
   of `f` and make the plan-optimality claim above a measurement instead of an
   extrapolation.
3. **The per-token minimum sits at m=5, four widths below where the shipped
   schedule concentrates.** Fixed width 5 runs 0.026996 s/token against the
   shipped adaptive 0.029317, which is 7.9% less candidate MTP time on this
   fixture. Acceptance falls monotonically from 0.981 at m=2 to 0.826 at m=9
   while `R_local(m)` rises faster than the extra accepted tokens repay. One
   public fixture cannot decide this and a `draftPolicy` change is a scored
   change outside this assignment. The advisor has this and is pricing it
   against the FINDING 520 ranked desk instrument.
4. **Reproduce the IPG ladder on M5.** `f` is a register and occupancy
   property, so its convexity is the part of this law most likely to differ on
   the ranked host.

## W&B

- Run `2el8cb01`, group `qwen38-r1-e212-width-cost-law`:
  https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/2el8cb01
- Tables: `width_table`, `legs` (per-leg temps, gate flags, digests, head
  provenance), `steps`, `served_width_distribution`, `band_arms`, `band_steps`,
  `ipg_pass_doubling`.
