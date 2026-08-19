#!/usr/bin/env python3
"""Pass-count-aware verify cost model for the Qwen 3.8 native-MTP scored path.

Why this file exists (E34 deliverable (e))
------------------------------------------
Every depth policy in this campaign has priced an extra draft row *smoothly*:
the shipped rule extends while `reach > h * (1 + expected) / (1 + d * h)`,
which is a monotone function of `d` alone. A smooth price provably cannot
express the cost structure the machine actually has, because the dominant term
in a verify forward is the number of times the 4-bit backbone is STREAMED, and
that number is `ceil(M / IPG)` read off a discrete dispatch table:

    Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
      case 5: qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>   -> 1 stream
      case 6: qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>   -> 2 streams

So M=5 carries 0.200 weight streams per row and M=6 carries 0.333 - a 67 %
jump in the dominant per-row cost for ONE extra row. E25 r3 measured that step
directly (T(4)=108.3 ms at M=5, T(5)=143.9 ms at M=6) and showed that an
unrelated kernel change (E27) MOVED the step one row deeper rather than
removing it. A model that hard-codes "the cliff is at 5" is wrong the moment
the dispatch table changes; a model that READS the table is correct under both
kernels. That is the whole point.

This module is deliberately dependency-free and offline: it reads the source of
truth (the dispatch table), fits a cost curve to measured per-width round
times, and reimplements the shipped depth policy exactly so counterfactual caps
can be evaluated without spending a GPU slot.

    python3 research/e34_cost_model.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from dataclasses import dataclass, field

REPO = pathlib.Path(__file__).resolve().parent.parent
DISPATCH_SOURCE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

# `qmv_fast_crossrow_affine4_g64_m<T, M, IPG, DIRECT_NIBBLES>`
_WIDE_M = re.compile(r"qmv_fast_crossrow_affine4_g64_m<\s*T\s*,\s*(\d+)\s*,\s*(\d+)\s*,")
# `qmv_fast_crossrow_affine4_g64<T, M>` - one group, one weight stream.
_WIDE_1 = re.compile(r"qmv_fast_crossrow_affine4_g64<\s*T\s*,\s*(\d+)\s*>")


def dispatch_ipg(source: pathlib.Path = DISPATCH_SOURCE) -> dict[int, int]:
    """M -> inputs-per-group for the affine-4/g64 cross-row QMV dispatch.

    The table is read from the runtime-effective generated twin, not from the
    readable `.metal`/`.h` source, because the JIT compiles the string in the
    `.cpp`. Widths with no cross-row instantiation (M in 1..2) fall through to
    the single-row path, which streams the weights once.
    """
    text = source.read_text()
    table: dict[int, int] = {}
    for m, ipg in _WIDE_M.findall(text):
        table[int(m)] = int(ipg)
    for m in _WIDE_1.findall(text):
        table.setdefault(int(m), int(m))
    return table


def weight_passes(width: int, table: dict[int, int] | None = None) -> int:
    """Weight streams for a verify forward of `width` rows.

    This is `ceil(M / IPG)` from the dispatch table - the quantity the machine
    charges - and NOT a smooth function of depth. Widths outside the table use
    one stream (the single-row QMV path).
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    if table is None:
        table = dispatch_ipg()
    ipg = table.get(width)
    if ipg is None:
        return 1
    return math.ceil(width / ipg)


def rows_per_pass(width: int, table: dict[int, int] | None = None) -> float:
    """Weight streams per emitted-candidate row: the number to minimise."""
    return weight_passes(width, table) / width


@dataclass
class StepCostModel:
    """`T(M) = intercept + per_row*M + per_row2*M^2 + per_pass*passes(M)`.

    `per_row2` is the advisor's "F proportional to M" end of the bracket: it
    is the only term that can express a per-row cost that itself grows with
    the width (more query rows attending over the same window).
    """

    intercept: float
    per_row: float
    per_pass: float
    per_row2: float = 0.0
    table: dict[int, int] = field(default_factory=dict)
    passes: dict[int, int] = field(default_factory=dict)
    residuals: dict[int, float] = field(default_factory=dict)
    r_squared: float = float("nan")
    max_abs_residual: float = float("nan")
    name: str = "step"

    def __call__(self, width: int) -> float:
        return self.predict(width)

    def predict(self, width: int) -> float:
        if self.passes:
            passes = self.passes[width]
        else:
            passes = weight_passes(width, self.table) if self.table else 1
        return (
            self.intercept
            + self.per_row * width
            + self.per_row2 * width * width
            + self.per_pass * passes
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "intercept_ms": self.intercept,
            "per_row_ms": self.per_row,
            "per_row2_ms": self.per_row2,
            "per_weight_pass_ms": self.per_pass,
            "r_squared": self.r_squared,
            "max_abs_residual_ms": self.max_abs_residual,
            "residuals_ms": self.residuals,
            "predicted_ms": {w: self.predict(w) for w in sorted(self.residuals)},
        }


def _lstsq(design: list[list[float]], target: list[float]) -> list[float]:
    """Normal-equation least squares with Gaussian elimination.

    The design matrices here are 2x2 to 4x4 and well conditioned; a dependency
    on numpy is not worth it for a research script that must run anywhere.
    """
    k = len(design[0])
    ata = [[sum(r[i] * r[j] for r in design) for j in range(k)] for i in range(k)]
    atb = [sum(r[i] * t for r, t in zip(design, target)) for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(ata[r][col]))
        if abs(ata[pivot][col]) < 1e-12:
            raise ValueError("singular design matrix")
        ata[col], ata[pivot] = ata[pivot], ata[col]
        atb[col], atb[pivot] = atb[pivot], atb[col]
        for row in range(col + 1, k):
            factor = ata[row][col] / ata[col][col]
            for c in range(col, k):
                ata[row][c] -= factor * ata[col][c]
            atb[row] -= factor * atb[col]
    out = [0.0] * k
    for row in reversed(range(k)):
        acc = atb[row] - sum(ata[row][c] * out[c] for c in range(row + 1, k))
        out[row] = acc / ata[row][row]
    return out


def fit_cost_model(
    times_ms: dict[int, float],
    *,
    quadratic: bool = False,
    use_passes: bool = True,
    table: dict[int, int] | None = None,
    passes: dict[int, int] | None = None,
    name: str | None = None,
) -> StepCostModel:
    """Fit `T(M)` to measured per-width round times.

    `use_passes=False` gives the smooth control model - the one every shipped
    depth rule implicitly assumes - so the step term can be tested rather than
    asserted. `passes` overrides the dispatch table for a curve measured on a
    build whose table is not the one checked out here.
    """
    if table is None:
        table = dispatch_ipg()
    widths = sorted(times_ms)

    def passes_of(w: int) -> int:
        return passes[w] if passes else weight_passes(w, table)

    design, target = [], []
    for w in widths:
        row = [1.0, float(w)]
        if quadratic:
            row.append(float(w * w))
        if use_passes:
            row.append(float(passes_of(w)))
        design.append(row)
        target.append(times_ms[w])
    beta = _lstsq(design, target)
    idx = 2
    per_row2 = 0.0
    if quadratic:
        per_row2 = beta[idx]
        idx += 1
    per_pass = beta[idx] if use_passes else 0.0
    model = StepCostModel(
        intercept=beta[0],
        per_row=beta[1],
        per_pass=per_pass,
        per_row2=per_row2,
        table=table if use_passes else {},
        passes=dict(passes) if (use_passes and passes) else {},
        name=name or ("quad" if quadratic else "linear") + ("+step" if use_passes else "+smooth"),
    )
    mean = sum(target) / len(target)
    ss_tot = sum((t - mean) ** 2 for t in target)
    ss_res = 0.0
    for w in widths:
        resid = times_ms[w] - model.predict(w)
        model.residuals[w] = resid
        ss_res += resid * resid
    model.max_abs_residual = max(abs(r) for r in model.residuals.values())
    model.r_squared = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return model


@dataclass
class PolicySim:
    """Exact reimplementation of the shipped `costModelDepth` walk.

    Mirrors `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`:
    per-position accept EMAs (prior `0.85 * 0.98^i`, alpha 0.15), the
    optimism transfer capped at 0.95 on a fully accepted round, the
    `fullAcceptStreak >= 2` gate that swaps `sdpaWidthWallDepthCap` for
    `segmentedVerifyDepthCap`, and the marginal price
    `h * (1 + expected) / (1 + d * h)`.

    The per-round target top-2 margin clamp is NOT modelled: it can only
    SHORTEN a round, it needs a quantity the ranked telemetry does not
    publish, and on the central prompts it is provably inactive often enough
    that every drafting round drafts (non_drafting_round_count == 0).
    """

    max_depth: int = 8
    width_wall_cap: int = 5
    segmented_cap: int = 8
    streak_gate: int = 2
    h: float = 0.18
    ema_alpha: float = 0.15
    optimism_cap: float = 0.95

    def new_state(self) -> dict:
        return {
            "ema": [0.85 * (0.98 ** i) for i in range(self.max_depth)],
            "streak": 0,
        }

    def choose_depth(self, state: dict, offered: int = 8,
                     margin: float | None = None) -> tuple[int, int]:
        """`margin` is the pending primary's target top-2 logit gap.

        The shipped rule clamps the first two positions' accept estimates by
        `sigmoid(margin/2)` and `sigmoid(margin/3)`. That clamp is the only
        thing in the policy that can shorten a round WITHOUT the prompt's
        acceptance being low, so it is the degree of freedom that lets a
        ranked prompt draft 4.53 rows while accepting 84 % of them.
        """
        width_cap = self.segmented_cap if state["streak"] >= self.streak_gate else self.width_wall_cap
        cap = min(min(offered, self.max_depth), width_cap)
        if cap <= 0:
            return 0, width_cap
        reach, expected, depth = 1.0, 0.0, 0
        while depth < cap:
            p = state["ema"][depth]
            if margin is not None and depth == 0:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 2.0)))
            elif margin is not None and depth == 1:
                p = min(p, 1.0 / (1.0 + math.exp(-margin / 3.0)))
            reach *= p
            threshold = self.h * (1.0 + expected) / (1.0 + depth * self.h)
            if not reach > threshold:
                break
            expected += reach
            depth += 1
        return depth, width_cap

    def record(self, state: dict, accepted: int, drafted: int) -> None:
        ema = state["ema"]
        for i in range(min(accepted, len(ema))):
            ema[i] += self.ema_alpha * (1.0 - ema[i])
        if accepted < drafted and accepted < len(ema):
            ema[accepted] += self.ema_alpha * (0.0 - ema[accepted])
        elif accepted == drafted and drafted > 0 and accepted < len(ema):
            if ema[accepted] < self.optimism_cap:
                ema[accepted] += self.ema_alpha * (self.optimism_cap - ema[accepted])
        state["streak"] = state["streak"] + 1 if accepted == drafted else 0


def _self_test() -> int:
    table = dispatch_ipg()
    expect_ipg = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}
    failures = []
    for width, ipg in expect_ipg.items():
        if table.get(width) != ipg:
            failures.append("IPG[%d] = %r, expected %d" % (width, table.get(width), ipg))
    expect_passes = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}
    for width, passes in expect_passes.items():
        got = weight_passes(width, table)
        if got != passes:
            failures.append("passes(%d) = %d, expected %d" % (width, got, passes))

    # E25 r3 post-E27 forced-depth curve, parent clock, M = depth + 1.
    measured = {
        1: 68.07211760816902,
        2: 71.36809877280531,
        3: 78.82930860890971,
        4: 91.7211723955054,
        5: 108.34561419068721,
        6: 143.87177260576095,
        7: 156.54867119259305,
        8: 170.32108455896378,
    }
    step = fit_cost_model(measured, table=table)
    smooth = fit_cost_model(measured, use_passes=False, table=table)
    quad_step = fit_cost_model(measured, quadratic=True, table=table)
    if not step.max_abs_residual < smooth.max_abs_residual:
        failures.append("step model did not beat the smooth control on max |residual|")
    if not quad_step.max_abs_residual < step.max_abs_residual:
        failures.append("curvature did not improve the step model")
    # The fitted per-pass term is the quantity the counterfactual leans on, so
    # the fit is only usable if its worst residual is small next to that term.
    if not quad_step.max_abs_residual < 0.25 * abs(quad_step.per_pass):
        failures.append(
            "quadratic+step max |residual| %.3f ms is not small next to the fitted "
            "per-pass term %.3f ms" % (quad_step.max_abs_residual, quad_step.per_pass)
        )

    sim = PolicySim()
    state = sim.new_state()
    depth, cap = sim.choose_depth(state)
    if cap != 5:
        failures.append("cold width cap %d, expected 5" % cap)
    for _ in range(40):  # a perfectly accepting prompt must open the segmented cap
        depth, cap = sim.choose_depth(state)
        sim.record(state, accepted=depth, drafted=depth)
    if cap != 8 or depth != 8:
        failures.append("hot prompt settled at depth %d cap %d, expected 8/8" % (depth, cap))

    print(json.dumps({
        "dispatch_ipg": {str(k): v for k, v in sorted(table.items())},
        "weight_passes": {str(w): weight_passes(w, table) for w in range(1, 10)},
        "passes_per_row": {str(w): round(rows_per_pass(w, table), 4) for w in range(1, 10)},
        "fit_linear_step": step.as_dict(),
        "fit_linear_smooth": smooth.as_dict(),
        "fit_quadratic_step": quad_step.as_dict(),
        "failures": failures,
    }, indent=1))
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(_self_test())
    print(json.dumps({str(k): v for k, v in sorted(dispatch_ipg().items())}, indent=1))


if __name__ == "__main__":
    main()
