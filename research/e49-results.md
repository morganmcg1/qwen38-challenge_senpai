# E49 — the 2-stream M=9 prize is real, and the shared register ceiling is not what E27 paid

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"arm1_m9_isolated_delta_pct","available":true,"value":-12.255},"test_metric":{"name":"arm2_ceiling_tax_refuted","available":true,"value":1}}
```

- Student / branch: `qwen-thorfinn` / `qwen-thorfinn/m9-two-stream-local-vs-shared`
- Hypothesis and target cost: separate `H_local_eaten` from `H_shared_tax`. Arm 1
  times `<T,9,5>` against `<T,9,3>` as isolated kernels. Arm 2 raises the
  kernel-wide register allocation with an unreachable switch case and times the
  widths whose instructions did not change.
- Decision: **both arms decisive.** Arm 1 → `H_local_win`. Arm 2 → `tax_refuted`.
- `BASE_SHA` `fb0a09d3912477d94ed631bdb90fd04172d7b4cf` / leg driver commit
  `f3d0819413c75183dd1250ddbedf421bcdca0141`
- Submitted candidate files: **none.** This is a probe-only experiment. The
  branch scored surface is byte-identical to `fb0a09d`
  (`git diff --stat fb0a09d -- Sources/ Vendor/ mtp-head.manifest.json benchmark.json`
  is empty).
- Supporting files: `research/e49_*.py`, `research/e49_*.sh`,
  `research/e49-prereg.json`, `research/e49-reg-census.json`,
  `research/e49-artifacts/*.json`
- MTP head provenance: organizer-pinned head, `bf16`, 849,400,347 bytes,
  `head_repo_declares_proposal_head=false`, identical across every leg.
- Scored-path reachability: dispatch read back from each run, not assumed.
  M=9 resolves to `qmv_fast_crossrow_affine4_g64_m<T,9,3,true>` in `iso3` and
  `<T,9,5,true>` in `iso5`.

## W&B evidence

Complete record, both arms, with the pre-registration, the dose census, the
contention self-tests, per-leg dispatch and digests, the width curves, and the
thermal record:

- **`92a0u0fl`** — https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/92a0u0fl

The nine arm-2 legs also logged live runs while they were timing:

| leg | run | leg | run |
|---|---|---|---|
| `shipped-c1` | `twd7gz0z` | `dhuge-c2` | `su8juixr` |
| `dnull-c1` | `2ldt9cdr` | `dbig-c2` | `uwi1ugww` |
| `d129-c1` | `4ihkl7iu` | `d129-c2` | `1yve8l1e` |
| `dbig-c1` | `9dtauxn6` | `dnull-c2` | `ykt6ddif` |
| `dhuge-c1` | `vnpdosmq` | | |

The four arm-1 legs have **no live run**: W&B logging for this experiment ran
once at the end of the session, and the session was destroyed before it reached
that step. `92a0u0fl` is therefore logged post-hoc from the recovered leg
artifacts and is labelled `logged_post_hoc=true` with its measurement window in
config. Reproduce it with:

```bash
python3 research/e49_analyze.py --out research/e49-artifacts/e49-metrics.json
python3 research/e49_dose_contrasts.py research/e49-artifacts/e49-metrics.json
python3 research/e49_price.py
python3 research/e49_wandb_log.py research/e49-artifacts/e49-metrics.json
```

The timed legs themselves were produced by
`research/e49_session.sh ARM:TAG [ARM:TAG ...]`, which drives
`research/e49_run_leg.sh` (patch → commit → GPU gate → measure → unwind).

## Arm 1 — isolated `<T,9,5>` vs `<T,9,3>`

Four ABBA legs (`iso3 · iso5 | iso5 · iso3`), 21 reps, widths 1..10. In these
builds cases 2..8 are deleted in **both** arms and fall through to
`qmv_fast_impl`, so nine widths execute byte-identical code and form an
empirical null measured in the same session as the signal.

| M | iso3 ms | iso5 ms | Δ ms | Δ % | replicate MDE ms |
|---|---|---|---|---|---|
| 1 | 59.633 | 60.158 | +0.526 | +0.88 | 1.504 |
| 2 | 66.769 | 66.987 | +0.217 | +0.33 | 0.275 |
| 3 | 94.129 | 94.132 | +0.003 | +0.00 | 0.146 |
| 4 | 122.017 | 121.963 | −0.055 | −0.05 | 0.107 |
| 5 | 149.813 | 149.971 | +0.158 | +0.11 | 0.330 |
| 6 | 177.700 | 177.616 | −0.084 | −0.05 | 0.498 |
| 7 | 205.531 | 205.598 | +0.067 | +0.03 | 0.392 |
| 8 | 233.649 | 233.525 | −0.124 | −0.05 | 0.422 |
| **9** | **186.113** | **163.305** | **−22.808** | **−12.255** | **0.333** |
| 10 | 271.757 | 271.875 | +0.118 | +0.04 | 0.454 |

Predicted −14.80 % (E46 refit) and −15.77 % (E46 contrast B transferred);
measured **−12.255 %**, ~68× the replicate spread at M=9. Same sign, same order,
consistently 2–3 points smaller than both predictions. The M=1 +0.88 % is inside
its own 1.504 ms replicate spread and is not claimed as an effect.

**`H_local_eaten` is refuted: the +21 registers are not paid locally.**

## Arm 2 — does raising the ceiling tax code that did not change?

The dose is an **unreachable `case 10:`** in the `>=4096` tier. `ntg.x == M` and
nothing verifies more than 9 rows, so the fat cell is compiled and allocated for
but never executed, and **no dispatched width's instructions change**. Every
width is therefore a control. Order is palindromic so each arm has the same mean
sweep position.

| dose | injected cell | census entry heuristic | Δ vs shipped | pooled tax M=3..9 | worst width |
|---|---|---|---|---|---|
| `dose_null` | `<T,4,4>` (104) | 164 | +1 | **+0.083 %** | +0.413 % |
| `dose_129` | `<T,9,5>` (129) | 181 | +18 | **−0.035 %** | +0.271 % |
| `dose_big` | `<T,12,6>` (144) | 197 | +34 | **+0.213 %** | +0.755 % |
| `dose_huge` | `<T,16,8>` (177) | 230 | +67 | **+0.078 %** | +0.282 % |

**There is no dose–response.** Ledger 173(C) needed **+10.6 %** here. The
largest pooled value is +0.213 % at `dose_big`, and at +67 registers the tax is
+0.078 %. Pre-registered `arm2_refuted` fires (`|tax| ≤ 2 %` at 129 and at every
larger dose). Restricted to the advisor's named subset M=3,4,6,7,8 the picture is
the same.

### Control-free confirmation, because the control has one leg

The launch was recreated before `shipped` got its second replicate, so one leg
carried the whole reference. Every **dose** has both replicates, so dose-vs-dose
contrasts answer the same question with no dependence on `shipped`
(`research/e49_dose_contrasts.py`):

| contrast | register delta | pooled | worst width |
|---|---|---|---|
| `dose_129` − `dose_null` | +17 | −0.117 % | +0.014 % |
| `dose_big` − `dose_null` | +33 | +0.130 % | +0.340 % |
| `dose_huge` − `dose_null` | **+66** | **−0.005 %** | +0.206 % |

Same-arm leg-to-leg noise over the same widths is 0.066–0.377 % mean and up to
0.770 % max, i.e. **larger than every contrast**. The single control leg's mean
rank is 4.43/9 against 5.0 for an unbiased reference, so the shipped-referenced
ladder is not an artifact of it. The refutation does not rest on the missing leg.

## Price — `harness=ranked` (`research/e49_price.py`)

Timing is the deliverable; this section is only the price, and it is kept
separate as asked. Coefficients come from `research/qmv_score_leverage.py` at
advisor ref `ccd1af6`, imported from that ref rather than copied into this
branch. `psi_mtp = 0.6736`, kink `1.0551 %`, saturation cap `4.7156 %`.
The advisor's own check reproduces mechanically: **`target_for(10.6) → None`**,
so the retired `c_ceiling = +10.6 %` was outside the reachable range of the
score function.

The measured M=9 win, by M=9's share of scored candidate-leg QMV cost:

| share | QMV removed | leg gain | **score (order-stat)** |
|---|---|---|---|
| 15.00 % | 1.838 % | 1.238 % | **+1.1437 %** |
| **20.48 %** (E48 P1, relayed) | 2.510 % | 1.691 % | **+1.3625 %** |
| 30.00 % | 3.676 % | 2.476 % | **+1.7426 %** |
| 53.45 % (retracted) | 6.550 % | 4.412 % | **+2.6789 %** |

20.48 % is **not hard-coded**: it is quoted with a band because the advisor has
since corrected the prompt weights it was computed at. So 173(C)'s +5.36 %
becomes **≈ +1.36 %**, still 2.6× the 0.5193 % crown gap and 1.8× the 0.7678 %
board floor.

The ceiling, priced as a uniform slowdown: worst pooled tax 0.213 % ⇒
**|dScore| ≤ 0.1435 %**; control-free 0.130 % ⇒ **≤ 0.0876 %**. Both are **below
the board floor**, so the shared-allocation ceiling is not resolvable on ranked.
This tightens the restated E44 bound of ≤ 0.4466 % by 3.1×.

## Where this disagrees with E27, stated plainly

```text
E27 observed score change            -0.3321 %   (board, trusted anchor)
E49 measured M=9 half, priced        +1.3625 %
residual                             -1.6946 %
E49 upper bound on the ceiling        0.1435 %
STILL UNEXPLAINED                    -1.5511 %
```

The ceiling explains at most **8.5 %** of E27's residual. E27's loss is real and
it is **not** the shared allocation. E27 changed *two* cells; only the M=9 half
is measured here. To close the residual, the M=5 half must **add +2.516 %** of
scored QMV cost:

| if M=5 is …% of scored QMV cost | `<T,5,5>` must be … |
|---|---|
| 10 % | **+25.2 % slower** than `<T,5,3>` |
| 15 % | +16.8 % slower |
| 20 % | +12.6 % slower |
| 30 % | +8.4 % slower |

The E46 refit instead predicts `<T,5,5>` is −22.95 % *faster*, or −19.0 % after
arm 1's measured shrinkage factor. That is a 26–41 point contradiction, and it is
the same failure mode as 173(C) one cell over: the refit's `27.532 ms/stream`
term was fit on 1→2→3 stream transitions at `NA ≤ 4`, while `<T,5,5>` is **1
stream at NA=5**, out of support in both coordinates. `<T,5,5>` has never been
timed and sits in `f{4,5,6}` ≈ 65 % of scored QMV cost.

## Provenance, and an infrastructure event that cost three legs

- Host `Apple M4 Pro`, `applegpu_g16s` (gen 16), 48 GiB, one model-holding
  process, `MLXFAST_LOCAL_RUN_LOCK_DIR=/tmp/mlxfast-shared` exported per leg.
- **All thirteen legs: `cool_gate_vendored=passed` on the real 40 °C gate.**
  Entry temperatures 38.51–39.87 °C, 1.36 °C total spread.
  `MLXFAST_LOCAL_COOL_GATE` was never disabled, so
  `cool_gate_passed_real_gate=true` and `gate_qualified_for_timing=true` for
  every leg. This **deviates from my own pre-registration**, which predicted
  `false` from E46's 43.2 °C floor observation. Recorded as a deviation rather
  than edited after the fact.
- GPU-idle gate: verdict `idle` on all thirteen legs. The first sample is
  discarded as priming, BUSY needs 3 consecutive samples over threshold, and a
  missing counter is distinct from idle. Three legs recorded a non-zero priming
  sample (`[4,9,0,0,0]`, `[10,0,0,0,0]`, `[9,0,0,0,0]`), which is exactly the
  documented interval-counter artifact; no leg ever reached 3 consecutive busy
  samples.
- Each leg commits its arm bytes, measures, and unwinds, so no turn observes a
  dirty scored surface and the timed bytes are pinned to a reachable commit.
  `dirty=0` on all thirteen legs, and seven distinct `sources_as_measured`
  digest sets, one per arm.
- 🟡 **Correcting my own earlier claim on the PR.** I reported
  `run_qmv_curve_rc=0` for every leg. That is true for the nine arm-2 legs and
  **false for the four arm-1 legs, which all exited 1.** The cause is
  `qmv_cost_curve_summary.py:212`: `staircase_fit` divides by an empty interior
  width set, because an isolated build contains a single crossrow case and so has
  no interior to fit. That is the optional **post-measurement report** stage.
  `vendored.json` is written and complete before it runs, is what
  `e49_analyze.py` reads, and parsed for all four legs. No arm-1 timing is
  affected; the only consequence is that those four legs have no `summary.json`.
  This is the exact failure class the driver was rebuilt to survive rather than
  hide, and the recorded rc is how it stayed visible.
- 🔴 **At 15:02 UTC the launch tag was recreated as `qwen38-mlx-senpai-r2` and
  the role workspace was replaced with a fresh clone.** This killed the tail
  session mid-leg and deleted the working tree, which had never been pushed.
  Recovered from the prior `r1` role checkout: all thirteen legs, the harness,
  the pre-registration, and the census. The nine arm-2 legs had already logged
  live W&B runs; the four arm-1 legs had not, which is why they were the ones at
  risk. Everything is now in one durable run, and re-analysis in the fresh
  checkout reproduces every number unchanged.
- **Three legs were not run**: `e49-shipped-c2` (killed at 14:54, control
  replicate 2) and `e49-e27rep` ×2 (the full-table composite). The pre-registered
  stopping rules had already fired on both arms, and no pending leg can change
  either conclusion. `run_job` correctly refuses a working directory outside the
  current workspace, so an identity-matched leg is no longer launchable.
- Correctness: eight bitwise deltas appear identically in the byte-identical
  `shipped` control, so they are a pre-existing base property of the `qmm` path
  at **M=10 only** — outside the scored range (max verify width 9 = 1 primary +
  8 drafts). **No leg shows any delta at M ≤ 9.**

## Compliance

- `<T,9,5>` is a general template instantiation in the existing runtime
  `switch (ntg.x)`, structurally identical to shipped cases 2..9. It is not keyed
  to benchmark shapes, and arm 1's isolation is a probe-only build, never a shape
  conditional on a submitted surface.
- **No bit-width change anywhere.** Every arm stays on `affine4_g64`; register
  pressure is bought with stream and IPG geometry only, never with a narrower
  stored or accumulated width.
- The `static_assert(NA >= 2 && NA <= 4)` relaxation is probe-only and unwound on
  every exit path. It is not shipped.
- Registers are the **dose**, not the readout. The census entry numbers only
  ordered the ladder; no conclusion rests on them. There is no true register or
  occupancy readout on this box.

## Conclusion

- `H_local_win`: a 2-stream M=9 really is ~12.3 % faster in isolation, so the
  mechanism exists and the roadmap item is not empty.
- `tax_refuted`: raising the kernel-wide allocation does **not** tax untouched
  widths, at four doses up to +67 registers, confirmed control-free. Ledger
  173(C)'s attribution of E27's loss to the shared ceiling is dead, and the
  ceiling is below the board floor as a ranked quantity.
- Prize, re-priced honestly: **≈ +1.36 % of score**, not +5.36 %.
- Recommendation: **close E49** and open the M=5 question. Nothing further is
  measurable here.

### Suggested follow-ups, not implemented

1. **Time `<T,5,5>`.** A 2×2 factorial on the full shipped table
   (`shipped`, `e27_replica`, `e27_m5`, `e27_full`) decomposes E27 exactly and
   gives the board anchor its first matched local measurement. `e27_replica` is
   already implemented in `research/e49_arms.py`; `e27_m5` and `e27_full` are a
   two-line addition. One ~22 min job.
2. **Look at M ∈ {4,5,6}.** It carries ~65 % of scored QMV cost and nobody has
   measured it. M=9-only mechanisms now compete against it at 3× the weight.
3. Re-run the palindrome tail if a matched environment returns, purely to close
   `shipped-c2`; it cannot change the verdict.
