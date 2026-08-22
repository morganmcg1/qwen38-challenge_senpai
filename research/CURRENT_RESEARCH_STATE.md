# SENPAI Research State

- **2026-08-22 09:05Z.** Advisor base `6724db07`. Campaign base `origin/main` `770a3ff2`.
  Live crown `bc070b7b` francip `3.35922017`, unmoved for over seven hours. Our submission
  `d3c491b5-902f-4f80-8d33-b7938f980d2d` is **validating**, queued 08:27:51Z: Route B alone on the
  post-revert base.

- **Most recent research direction from the human researcher team:** none received this round. The
  campaign is running autonomously under `senpai/program.md`.

## Current research focus

The round found that the largest unexploited lever on the board is **GPU occupancy at the wide-QMV
entry point on the ranked architecture**, and that it was invisible for 273 ledger entries because
it does not exist on the hardware we develop on.

1. **The entry-point occupancy tax (F102, F107, F108).** The wide QMV compiles every width into one
   Metal function, so its register allocation is the maximum over all inlined width bodies and every
   dispatch runs at that occupancy. On `applegpu_g17s` the maximum is **M=5 at 101 registers, giving
   39 resident simdgroups**, while every other width compiles at 89 or 90 and would run at 44. M=5
   carries 3.4 % of the work and taxes the other 96.6 %. On `applegpu_g16s` the maximum is NA=4,
   which is M4, M7 and M8 — the dominant widths — so the lever reads as an exact null locally. The
   crown ships `<T,5,3>` at 90 registers; we adopted `<T,5,5>` across E27, E55, t55 and E100, every
   time on local evidence where the change is free.

2. **The residency coefficient (F110).** `c = 0.445 % leg gain per 1 % entry-point residency`,
   bracket `[0.139, 0.819]`, instruction channel `I = -0.958 %` for E121. The solve has zero degrees
   of freedom and needs an identifying design. Saturation above 39 simdgroups is the single
   unmodelled risk and is the reason the bracket is wide.

3. **Two independent routes to the same tax, running in parallel and not colliding.** Alphonse holds
   `quantized.h` and attacks the incumbent kernel by removing the NA=5 body or shrinking it.
   Thorfinn holds `Qwen35.swift` and attacks Route B's replica kernel, where per-width templating is
   fully reachable through `MLXFast.metalKernel`'s template parameters and carries an **instruction
   channel of exactly zero** (F109).

4. **The scheduler is measurably mis-calibrated (F111, F112).** The ranked round count R is now
   pinned out of sample to within 3 % on all seven drafting prompts, and the shipped reach estimator
   is **biased low by 9 to 24 % with slope ~1.0 and correlation 0.86 to 0.95** — a level defect, not
   a discrimination defect. Hypothesis J attributes it to Jensen's inequality over per-round
   acceptance heterogeneity, which predicts the slope, the depth scaling and the one unbiased
   fixture from a single mechanism.

5. **Two instruments were downgraded (F113, F114).** Fifteen of twenty E123 price-ladder cells carry
   an occupancy term rather than a pure instruction cost. The bandwidth axis is dead; the deleted
   instruction count alone explains 90 % of the variance and the pooled correlations were width
   effects.

## In flight

| PR | student | experiment | state |
|---|---|---|---|
| #128 | thorfinn | E129 — Route B submitted (`d3c491b5`); rung 2 restored: per-width templating of the Route B replica kernel | wip, awaiting receipt |
| #130 | alphonse | E130 — remove the `applegpu_g17s` entry-point occupancy tax; rung 2 is the identifying design for `c` | wip |
| #129 | edward | E128 — reach-estimator calibration; hypothesis J, the sign decomposition, and the joint price sweep | wip, rung 1 running |
| — | askeladd | E125 merged. Free for the next assignment. | available |

## Potential next research directions

- **Compose the two occupancy arms.** If Route B is promoted and templating passes, the incumbent
  `quantized.h` kernel still serves every wide QMV that Route B declines. The two arms are disjoint
  by construction and should be shown so under Rule 75 before composing.
- **A pre-submission entry-point cliff census as a standing gate.** Askeladd's 7.5 s zero-GPU census
  would have caught the E121 ranked regression before it cost a submission slot. Make it a
  mandatory pre-submit step alongside `twin_audit.py` and the scope and budget checks.
- **Re-audit every historical register-channel decision against `applegpu_g17s`.** E27, E55, t55 and
  E100 were all decided on `applegpu_g16s`. F107 says at least one of them inverted on the ranked
  host. There may be more.
- **Resolve the 26x disagreement between E27's `-20.1 %` and E100's `-0.775 %`** on the identical
  M=5 cell contrast. One of them is measuring something other than what it claims.
- **Ship a mechanism-based calibration of the reach estimator**, not a fitted scalar, if hypothesis J
  survives. A variance correction computed from measured per-round heterogeneity has a first-
  principles justification that a gamma does not.
- **Rebuild the E123 price ladder under Rule 86**, with the register delta of both rungs on every
  cell. Five of twenty cells survive today, and several campaign decisions rest on the other fifteen.
- **Free re-ranking of the 37 E87 decision cells** at the corrected coefficient 203 rather than the
  206.6 hardcoded at `research/e87_decide.py:74`.
- **The C1 sign-sketch draft-path arm** remains designed and unassigned at `+0.23 to +0.34 %` ranked.
  It is now the largest queued arm that does not touch the occupancy axis.
- **Post-promotion cleanup.** Delete the dead `qkv(_:)` fast path at `Qwen35.swift:2281-2300`, the
  reverted E121 remnants, and the research-only `Qwen35IslandArm` selector, so the winning behaviour
  is the only path.
