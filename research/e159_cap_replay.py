#!/usr/bin/env python3
"""E159 r2 step 2: replay the ranked published median of receipt 5a9f130a
under a hard verify-width cap, with no GPU work.

The instrument has four parts.

1. Board collinearity audit. The board cell `width` is a MEAN verify width,
   not an integer, so a per-integer-width marginal is not directly readable
   from it. Worse, within one prompt the distinct mean widths are digit
   identical across hundreds of cells, which means width varies only when the
   whole schedule varies. The audit measures that collinearity.

2. Ranked per-width marginals, identified from the eight anchor prompts. Each
   prompt's observed ranked round cost is the expectation of the per-width cost
   over that prompt's proposed-depth distribution. With the FINDING 330
   zero-draft round pinned, the eight prompt costs constrain the seven
   marginals m[2..8]. The design is near collinear, so the fit reports an
   identified SET for the width-6 marginal rather than one number.

3. Acceptance. The local pinned depth sweep gives a per-position acceptance
   survival S(i). Each ranked prompt gets S_p(i) = S(i) ** theta_p with
   theta_p fitted to that prompt's observed accepted rows per round.

4. Cap replay. For each cap C the proposed depth truncates to min(D, C),
   tokens stay at exactly 512, the round count rises, and the published median
   of the eight raw ratios is recomputed.

Every quantity that comes from the board or from a receipt is harness=ranked.
Every quantity that comes from the advisor host sweep is harness=local.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
ARTIFACTS = ROOT / "e159-artifacts"
TRANSFER = ARTIFACTS / "e159_ranked_transfer.json"
BUDGET = ARTIFACTS / "e159_round_budget.json"
COST_LAW = ARTIFACTS / "e159_round_cost_law.json"

DECODE_TOKENS = 512
SHIPPED_CAP = 7


# ---------------------------------------------------------------- utilities


def pava_nonincreasing(values: list[float]) -> list[float]:
    """Pool adjacent violators, preserving the sum."""
    blocks: list[tuple[float, int]] = []
    for value in values:
        blocks.append((value, 1))
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            (v_a, n_a), (v_b, n_b) = blocks[-2], blocks[-1]
            blocks[-2:] = [((v_a * n_a + v_b * n_b) / (n_a + n_b), n_a + n_b)]
    out: list[float] = []
    for value, count in blocks:
        out.extend([value] * count)
    return out


def solve_normal_equations(a: list[list[float]], b: list[float]) -> list[float]:
    """Least squares by Gaussian elimination on the normal equations."""
    cols = len(a[0])
    ata = [[sum(row[i] * row[j] for row in a) for j in range(cols)]
           for i in range(cols)]
    atb = [sum(row[i] * rhs for row, rhs in zip(a, b)) for i in range(cols)]
    for i in range(cols):
        ata[i][i] += 1e-9
    for i in range(cols):
        pivot = max(range(i, cols), key=lambda r: abs(ata[r][i]))
        ata[i], ata[pivot] = ata[pivot], ata[i]
        atb[i], atb[pivot] = atb[pivot], atb[i]
        head = ata[i][i]
        if abs(head) < 1e-18:
            continue
        for r in range(cols):
            if r == i:
                continue
            factor = ata[r][i] / head
            if factor == 0.0:
                continue
            for c in range(i, cols):
                ata[r][c] -= factor * ata[i][c]
            atb[r] -= factor * atb[i]
    return [atb[i] / ata[i][i] if abs(ata[i][i]) > 1e-18 else 0.0
            for i in range(cols)]


def solve_nonnegative(a: list[list[float]], b: list[float]) -> list[float]:
    """Active-set nonnegative least squares.

    A marginal cost cannot be negative. An unconstrained fit on a design whose
    highest widths carry almost no probability mass produces large negative
    entries there, and a negative marginal makes a cap look good for a reason
    that is arithmetic rather than physical.
    """
    cols = len(a[0])
    free = list(range(cols))
    while True:
        sub = [[row[j] for j in free] for row in a]
        partial = solve_normal_equations(sub, b)
        if all(v >= 0.0 for v in partial):
            out = [0.0] * cols
            for j, value in zip(free, partial):
                out[j] = value
            return out
        worst = min(range(len(partial)), key=lambda i: partial[i])
        free.pop(worst)
        if not free:
            return [0.0] * cols


def median_of(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def median_pair(values: dict[str, float]) -> tuple[str, str]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    return ordered[n // 2 - 1][0], ordered[n // 2][0]


# ------------------------------------------------------- depth distributions


def depth_distribution(kind: str, f0: float, q: float,
                       cap: int) -> dict[int, float]:
    """Proposed-depth distribution for one prompt.

    `f0` is the observed non-drafting fraction and `q` the observed mean
    proposed rows per round. Every family reproduces both exactly.
    """
    drafting = 1.0 - f0
    if drafting <= 1e-12:
        return {0: 1.0}
    mean_when_drafting = q / drafting
    mean_when_drafting = min(mean_when_drafting, float(cap))

    if kind == "two_point":
        low = int(math.floor(mean_when_drafting))
        high = min(low + 1, cap)
        if high == low:
            inner = {low: 1.0}
        else:
            upper_share = mean_when_drafting - low
            inner = {low: 1.0 - upper_share, high: upper_share}
    elif kind == "local_shape":
        # The shipped schedule's own proposed-depth histogram on the advisor
        # fixture, shifted so its mean matches this prompt.
        base = {3: 4, 4: 7, 5: 5, 6: 3, 7: 59}
        total = sum(base.values())
        base_mean = sum(d * n for d, n in base.items()) / total
        shift = mean_when_drafting - base_mean
        inner = collections.defaultdict(float)
        for depth, count in base.items():
            target = depth + shift
            low = int(math.floor(target))
            frac = target - low
            for slot, weight in ((low, 1.0 - frac), (low + 1, frac)):
                slot = max(1, min(cap, slot))
                inner[slot] += weight * count / total
        inner = dict(inner)
        # Re-centre after clamping so the mean is exact.
        realised = sum(d * w for d, w in inner.items())
        if abs(realised - mean_when_drafting) > 1e-9:
            inner = _rebalance(inner, mean_when_drafting, cap)
    elif kind == "geometric":
        # Truncated geometric over 1..cap with the required mean.
        inner = _geometric_with_mean(mean_when_drafting, cap)
    else:
        raise ValueError(f"unknown depth distribution family {kind}")

    out = collections.defaultdict(float)
    out[0] += f0
    for depth, weight in inner.items():
        out[depth] += drafting * weight
    return dict(out)


def _rebalance(inner: dict[int, float], target_mean: float,
               cap: int) -> dict[int, float]:
    low = int(math.floor(target_mean))
    high = min(low + 1, cap)
    if high == low:
        return {low: 1.0}
    upper = target_mean - low
    blend = 0.5
    mixed = collections.defaultdict(float)
    for depth, weight in inner.items():
        mixed[depth] += blend * weight
    mixed[low] += (1.0 - blend) * (1.0 - upper)
    mixed[high] += (1.0 - blend) * upper
    realised = sum(d * w for d, w in mixed.items())
    delta = target_mean - realised
    # One linear correction between the two central slots.
    mixed[high] += delta
    mixed[low] -= delta
    return {d: w for d, w in mixed.items() if w > 1e-12}


def _geometric_with_mean(target_mean: float, cap: int) -> dict[int, float]:
    lo, hi = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        rate = 0.5 * (lo + hi)
        weights = [rate ** (d - 1) for d in range(1, cap + 1)]
        total = sum(weights)
        mean = sum(d * w for d, w in zip(range(1, cap + 1), weights)) / total
        if mean < target_mean:
            lo = rate
        else:
            hi = rate
    rate = 0.5 * (lo + hi)
    weights = [rate ** (d - 1) for d in range(1, cap + 1)]
    total = sum(weights)
    return {d: w / total for d, w in zip(range(1, cap + 1), weights)}


# ------------------------------------------------------------- acceptance


def accepted_given_depth(survival: list[float], theta: float) -> list[float]:
    """Expected accepted rows for each proposed depth 0..len(survival)."""
    out = [0.0]
    running = 0.0
    for value in survival:
        running += value ** theta
        out.append(running)
    return out


def fit_theta(survival: list[float], dist: dict[int, float],
              target_a: float) -> float:
    lo, hi = 1e-3, 200.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        table = accepted_given_depth(survival, mid)
        value = sum(w * table[d] for d, w in dist.items())
        if value > target_a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- cost law fit


def fit_ranked_marginals(prompts: list[dict], dists: dict[str, dict[int, float]],
                         zero_draft_round: float,
                         max_width: int) -> tuple[list[float], float]:
    """Least squares for the marginals into widths 2..max_width.

    Row p of the design holds the probability that prompt p reaches each
    width, because the cumulative cost of a round at width w includes every
    marginal up to w.
    """
    design: list[list[float]] = []
    rhs: list[float] = []
    for prompt in prompts:
        dist = dists[prompt["prompt"]]
        row = []
        for width in range(2, max_width + 1):
            reach = sum(w for d, w in dist.items() if d + 1 >= width)
            row.append(reach)
        design.append(row)
        rhs.append(prompt["R_ranked_seconds"] - zero_draft_round)
    solution = solve_nonnegative(design, rhs)
    residual = math.sqrt(sum(
        (sum(r * s for r, s in zip(row, solution)) - target) ** 2
        for row, target in zip(design, rhs)) / len(rhs))
    return solution, residual


def constrained_marginals(prompts: list[dict], dists, zero_draft_round: float,
                          max_width: int, wall_width: int,
                          wall_value: float) -> tuple[list[float], float]:
    """Least squares with the wall marginal pinned to `wall_value`."""
    index = wall_width - 2
    design: list[list[float]] = []
    rhs: list[float] = []
    for prompt in prompts:
        dist = dists[prompt["prompt"]]
        full = []
        for width in range(2, max_width + 1):
            reach = sum(w for d, w in dist.items() if d + 1 >= width)
            full.append(reach)
        target = prompt["R_ranked_seconds"] - zero_draft_round
        target -= full[index] * wall_value
        design.append([v for i, v in enumerate(full) if i != index])
        rhs.append(target)
    solution = solve_nonnegative(design, rhs)
    full_solution = solution[:index] + [wall_value] + solution[index:]
    residual = math.sqrt(sum(
        (sum(r * s for r, s in zip(row, solution)) - target) ** 2
        for row, target in zip(design, rhs)) / len(rhs))
    return full_solution, residual


def round_cost(zero_draft_round: float, marginals: list[float],
               width: int) -> float:
    return zero_draft_round + sum(marginals[: max(0, width - 1)])


# ------------------------------------------------------------------ replay


def replay(prompts, dists, thetas, survival, zero_draft_round, marginals,
           offsets, cap):
    out = {}
    for prompt in prompts:
        name = prompt["prompt"]
        dist = dists[name]
        table = accepted_given_depth(survival, thetas[name])
        a_capped = 0.0
        q_capped = 0.0
        cost = 0.0
        for depth, weight in dist.items():
            eff = min(depth, cap)
            a_capped += weight * table[eff]
            q_capped += weight * eff
            cost += weight * round_cost(zero_draft_round, marginals, eff + 1)
        cost += offsets[name]
        rounds = DECODE_TOKENS / (1.0 + a_capped)
        mtp = prompt["prefill_seconds_per_token"] + rounds * cost / DECODE_TOKENS
        out[name] = {
            "q": q_capped,
            "a": a_capped,
            "rounds": rounds,
            "R_seconds": cost,
            "mtp_seconds_per_token": mtp,
            "raw": prompt["serial_seconds_per_token_mean"] / mtp,
        }
    return out


def run(depth_family: str, wall_pin: float | None):
    transfer = json.loads(TRANSFER.read_text())
    budget = json.loads(BUDGET.read_text())
    prompts = transfer["prompts"]
    zero_draft_round = transfer["roofline"]["ranked_batch_1_round_seconds"]

    dense = budget["dense_depth_table"]
    increments = []
    previous = 0.0
    for row in dense[1:]:
        increments.append(row["a_accepted_per_round"] - previous)
        previous = row["a_accepted_per_round"]
    survival = pava_nonincreasing(increments)

    dists = {}
    for prompt in prompts:
        rounds = prompt["rounds"]
        f0 = prompt["non_drafting_round_count"] / rounds
        dists[prompt["prompt"]] = depth_distribution(
            depth_family, f0, prompt["q_proposed_per_round"], SHIPPED_CAP)

    thetas = {}
    for prompt in prompts:
        thetas[prompt["prompt"]] = fit_theta(
            survival, dists[prompt["prompt"]],
            prompt["a_accepted_per_round"])

    max_width = SHIPPED_CAP + 1
    if wall_pin is None:
        marginals, residual = fit_ranked_marginals(
            prompts, dists, zero_draft_round, max_width)
    else:
        marginals, residual = constrained_marginals(
            prompts, dists, zero_draft_round, max_width, 6, wall_pin)

    # Level calibration: the shipped cap must reproduce each observed round
    # cost exactly, so cap deltas are differences against a matched baseline.
    offsets = {}
    for prompt in prompts:
        dist = dists[prompt["prompt"]]
        modelled = sum(
            w * round_cost(zero_draft_round, marginals, min(d, SHIPPED_CAP) + 1)
            for d, w in dist.items())
        offsets[prompt["prompt"]] = prompt["R_ranked_seconds"] - modelled

    base = replay(prompts, dists, thetas, survival, zero_draft_round,
                  marginals, offsets, SHIPPED_CAP)
    base_median = median_of([v["raw"] for v in base.values()])

    caps = {}
    for cap in range(0, 9):
        state = replay(prompts, dists, thetas, survival, zero_draft_round,
                       marginals, offsets, cap)
        raws = {k: v["raw"] for k, v in state.items()}
        caps[cap] = {
            "published_median": median_of(list(raws.values())),
            "median_pair": list(median_pair(raws)),
            "per_prompt": {
                k: {
                    "raw": v["raw"],
                    "mtp_pct_change": 100.0 * (
                        v["mtp_seconds_per_token"]
                        / base[k]["mtp_seconds_per_token"] - 1.0),
                    "q": v["q"],
                    "a": v["a"],
                    "rounds": v["rounds"],
                    "R_ms": v["R_seconds"] * 1e3,
                } for k, v in state.items()},
        }
        caps[cap]["median_pct_change"] = 100.0 * (
            caps[cap]["published_median"] / base_median - 1.0)

    return {
        "depth_family": depth_family,
        "wall_pin_seconds": wall_pin,
        "survival": survival,
        "thetas": thetas,
        "marginals_into_width_2_to_8": marginals,
        "fit_residual_seconds": residual,
        "zero_draft_round_seconds": zero_draft_round,
        "base_published_median": base_median,
        "observed_published_median": 3.70784519,
        "caps": caps,
        "depth_distributions": {k: {str(d): w for d, w in v.items()}
                                for k, v in dists.items()},
    }


# ----------------------------------------------------------- board audit


def board_collinearity_audit():
    law = json.loads(COST_LAW.read_text())
    cells = law["cells"]
    by_prompt = collections.defaultdict(list)
    for cell in cells:
        by_prompt[cell["prompt"]].append(cell)

    audit = {}
    for prompt, group in by_prompt.items():
        by_width = collections.defaultdict(list)
        for cell in group:
            by_width[round(cell["width"], 4)].append(cell)
        widths = sorted(by_width)
        clusters = [(w, len(v)) for w, v in sorted(
            by_width.items(), key=lambda kv: -len(kv[1]))[:4]]
        # Efficiency is the candidate round cost per emitted token, expressed
        # in units of the same tree's own width-1 round, so tree speed cancels.
        eff_by_group = {}
        for cell in group:
            key = (cell["head"][:8], round(cell["width"], 4))
            eff_by_group.setdefault(key, []).append(
                cell["cost_over_anchor"] / cell["tokens_per_round"])
        rows = [(head, width, len(vals), statistics.median(vals))
                for (head, width), vals in eff_by_group.items()
                if len(vals) >= 5]
        rows.sort()
        audit[prompt] = {
            "n_cells": len(group),
            "distinct_mean_widths": len(widths),
            "mean_width_min": widths[0],
            "mean_width_max": widths[-1],
            "top_clusters": clusters,
            "efficiency_by_head_and_width": [
                {"head": h, "mean_width": w, "n": n, "median_efficiency": e}
                for h, w, n, e in rows],
        }
    return audit


def board_deconvolved_marginals(head_filter: str | None = None,
                                family: str = "two_point"):
    """Deconvolve per-integer-width marginals from the whole board.

    `cost_over_anchor` is a cell's mean round cost divided by the SAME tree's
    width-1 round, so tree speed cancels and the fitted marginals are relative
    to each tree's own zero-draft round. Every cell contributes its own
    proposed-depth distribution built from its mean width and its non-drafting
    fraction, so the fit inverts the mixing that makes a mean-width regression
    unable to see an integer step.
    """
    law = json.loads(COST_LAW.read_text())
    cells = law["cells"]
    max_width = 9
    design: list[list[float]] = []
    rhs: list[float] = []
    weights: list[float] = []
    used = 0
    for cell in cells:
        if head_filter is not None and not cell["head"].startswith(head_filter):
            continue
        rounds = cell["rounds"]
        if not rounds or cell["width"] is None:
            continue
        q = cell["width"] - 1.0
        if q < 0:
            continue
        f0 = min(1.0, max(0.0, cell["non_drafting_rounds"] / rounds))
        dist = depth_distribution(family, f0, q, max_width - 1)
        row = [sum(w for d, w in dist.items() if d + 1 >= width)
               for width in range(2, max_width + 1)]
        design.append(row)
        rhs.append(cell["cost_over_anchor"] - 1.0)
        weights.append(math.sqrt(rounds))
        used += 1
    scaled_design = [[v * w for v in row] for row, w in zip(design, weights)]
    scaled_rhs = [v * w for v, w in zip(rhs, weights)]
    solution = solve_nonnegative(scaled_design, scaled_rhs)
    predicted = [sum(r * s for r, s in zip(row, solution)) for row in design]
    residuals = [p - t for p, t in zip(predicted, rhs)]
    resid_sd = math.sqrt(sum(r * r for r in residuals) / max(1, len(residuals)))

    # Standard errors from the weighted normal equations.
    cols = len(solution)
    ata = [[sum(row[i] * row[j] for row in scaled_design) for j in range(cols)]
           for i in range(cols)]
    inv = _invert(ata)
    weighted_resid_var = sum(
        (p - t) ** 2 * w * w for p, t, w in zip(predicted, rhs, weights))
    dof = max(1, len(design) - cols)
    sigma2 = weighted_resid_var / dof
    se = [math.sqrt(max(0.0, sigma2 * inv[i][i])) for i in range(cols)]
    return {
        "head_filter": head_filter,
        "family": family,
        "n_cells": used,
        "relative_marginals_into_width_2_to_9": solution,
        "relative_marginal_se": se,
        "residual_sd_of_cost_over_anchor": resid_sd,
    }


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(aug[r][i]))
        aug[i], aug[pivot] = aug[pivot], aug[i]
        head = aug[i][i]
        if abs(head) < 1e-18:
            head = 1e-18
        aug[i] = [v / head for v in aug[i]]
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            if factor == 0.0:
                continue
            aug[r] = [v - factor * u for v, u in zip(aug[r], aug[i])]
    return [row[n:] for row in aug]


def cap_sweep_with_marginals(marginals: list[float], family: str,
                             label: str) -> dict:
    """Replay caps 0..8 with an externally supplied ranked marginal vector."""
    transfer = json.loads(TRANSFER.read_text())
    budget = json.loads(BUDGET.read_text())
    prompts = transfer["prompts"]
    zero = transfer["roofline"]["ranked_batch_1_round_seconds"]

    dense = budget["dense_depth_table"]
    increments = []
    previous = 0.0
    for row in dense[1:]:
        increments.append(row["a_accepted_per_round"] - previous)
        previous = row["a_accepted_per_round"]
    survival = pava_nonincreasing(increments)

    dists, thetas = {}, {}
    for prompt in prompts:
        f0 = prompt["non_drafting_round_count"] / prompt["rounds"]
        dists[prompt["prompt"]] = depth_distribution(
            family, f0, prompt["q_proposed_per_round"], SHIPPED_CAP)
        thetas[prompt["prompt"]] = fit_theta(
            survival, dists[prompt["prompt"]], prompt["a_accepted_per_round"])

    offsets = {}
    for prompt in prompts:
        modelled = sum(
            w * round_cost(zero, marginals, min(d, SHIPPED_CAP) + 1)
            for d, w in dists[prompt["prompt"]].items())
        offsets[prompt["prompt"]] = prompt["R_ranked_seconds"] - modelled

    base = replay(prompts, dists, thetas, survival, zero, marginals,
                  offsets, SHIPPED_CAP)
    base_median = median_of([v["raw"] for v in base.values()])
    caps = {}
    for cap in range(0, 9):
        state = replay(prompts, dists, thetas, survival, zero, marginals,
                       offsets, cap)
        raws = {k: v["raw"] for k, v in state.items()}
        caps[cap] = {
            "published_median": median_of(list(raws.values())),
            "median_pct_change": 100.0 * (
                median_of(list(raws.values())) / base_median - 1.0),
            "median_pair": list(median_pair(raws)),
            "per_prompt_mtp_pct": {
                k: 100.0 * (v["mtp_seconds_per_token"]
                            / base[k]["mtp_seconds_per_token"] - 1.0)
                for k, v in state.items()},
        }
        # RULE 177 prices a ranked mechanism on the paired per-prompt
        # candidate leg, so a saving is a negative change in mtp.
        gains = [-value for value in caps[cap]["per_prompt_mtp_pct"].values()]
        caps[cap]["candidate_leg_mean_pct"] = statistics.mean(gains)
        caps[cap]["candidate_leg_se_pct"] = (
            statistics.stdev(gains) / math.sqrt(len(gains))
            if len(gains) > 1 else 0.0)
        caps[cap]["candidate_leg_same_sign"] = max(
            sum(1 for g in gains if g > 0), sum(1 for g in gains if g < 0))
    best = max(caps, key=lambda c: caps[c]["published_median"])
    best_leg = max(caps, key=lambda c: caps[c]["candidate_leg_mean_pct"])

    # Leave one prompt out. FINDING 306 says the identity of the fifth prompt
    # is a lottery, so the fold that matters is over the SELECTION objective:
    # drop one prompt, take the median of the remaining seven, and ask whether
    # the winning cap moves.
    folds = {}
    per_cap_raws = {}
    for cap in range(0, 9):
        state = replay(prompts, dists, thetas, survival, zero, marginals,
                       offsets, cap)
        per_cap_raws[cap] = {k: v["raw"] for k, v in state.items()}
    base_raws = {k: v["raw"] for k, v in base.items()}
    for held in [p["prompt"] for p in prompts]:
        held_base = median_of([v for k, v in base_raws.items() if k != held])
        by_cap = {
            cap: median_of([v for k, v in raws.items() if k != held])
            for cap, raws in per_cap_raws.items()}
        fold_best = max(by_cap, key=lambda c: by_cap[c])
        folds[held] = {
            "argmax_cap": fold_best,
            "best_median_pct": 100.0 * (by_cap[fold_best] / held_base - 1.0),
            "median_pct_by_cap": {c: 100.0 * (v / held_base - 1.0)
                                  for c, v in by_cap.items()},
        }
    fold_caps = [v["argmax_cap"] for v in folds.values()]
    return {
        "label": label,
        "family": family,
        "marginals_into_width_2_to_8_ms": [v * 1e3 for v in marginals],
        "argmax_cap": best,
        "argmax_cap_median_pct": caps[best]["median_pct_change"],
        "argmax_cap_candidate_leg": best_leg,
        "argmax_cap_candidate_leg_pct": caps[best_leg]["candidate_leg_mean_pct"],
        "argmax_cap_candidate_leg_same_sign":
            caps[best_leg]["candidate_leg_same_sign"],
        "caps": caps,
        "loo_folds": folds,
        "loo_cap_spread": [min(fold_caps), max(fold_caps)],
        "loo_stable": min(fold_caps) == max(fold_caps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ARTIFACTS / "e159_cap_replay.json"))
    args = parser.parse_args()

    results = {}
    for family in ("two_point", "local_shape", "geometric"):
        results[family] = run(family, None)

    # Identified set for the width-6 marginal: pin it and measure how badly
    # the eight ranked prompt costs are then fitted.
    free_residual = results["two_point"]["fit_residual_seconds"]
    wall_scan = []
    for pinned_ms in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0,
                      24.0, 28.0, 32.0, 36.0]:
        scan = run("two_point", pinned_ms / 1e3)
        wall_scan.append({
            "pinned_width6_marginal_ms": pinned_ms,
            "fit_residual_ms": scan["fit_residual_seconds"] * 1e3,
            "residual_over_free_fit": (scan["fit_residual_seconds"]
                                       / max(free_residual, 1e-12)),
            "best_cap": max(scan["caps"], key=lambda c: scan["caps"][c]
                            ["published_median"]),
            "best_cap_median_pct": max(
                scan["caps"][c]["median_pct_change"] for c in scan["caps"]),
        })

    # Board-wide deconvolution: thousands of ranked cells, each contributing
    # its own proposed-depth distribution, fix the per-integer-width shape far
    # more tightly than the eight anchor prompts can on their own.
    transfer = json.loads(TRANSFER.read_text())
    zero = transfer["roofline"]["ranked_batch_1_round_seconds"]
    board_fits = {}
    for label, head in (("all_heads", None), ("our_head_559b24eb", "559b24eb")):
        board_fits[label] = board_deconvolved_marginals(head)
        rel = board_fits[label]["relative_marginals_into_width_2_to_9"]
        board_fits[label]["absolute_marginals_ms"] = [v * zero * 1e3
                                                      for v in rel]
        board_fits[label]["absolute_marginal_se_ms"] = [
            v * zero * 1e3
            for v in board_fits[label]["relative_marginal_se"]]

    board_marginals = [v * zero for v in
                       board_fits["all_heads"]
                       ["relative_marginals_into_width_2_to_9"][:7]]
    neighbour_mean = statistics.mean(
        [board_marginals[3], board_marginals[5]])  # into widths 5 and 7
    healed = list(board_marginals)
    healed[4] = neighbour_mean

    scenarios = {
        "board_deconvolved_current_price": cap_sweep_with_marginals(
            board_marginals, "two_point", "board deconvolved, width-6 as fitted"),
        "board_deconvolved_width6_at_neighbour_mean": cap_sweep_with_marginals(
            healed, "two_point", "width-6 priced at the neighbour mean"),
    }
    # The advisor's counterfactual with the LOCAL wall transferred verbatim:
    # what the cap is worth if the ranked width-6 step really does cost what
    # the advisor host measures, scaled by the zero-draft round ratio.
    budget = json.loads(BUDGET.read_text())
    dense = budget["dense_depth_table"]
    local_increments = [
        dense[i + 1]["R_decode_seconds_mean"] - dense[i]["R_decode_seconds_mean"]
        for i in range(len(dense) - 1)]
    local_scale = zero / dense[0]["R_decode_seconds_mean"]
    transferred = [v * local_scale for v in local_increments[:7]]
    scenarios["local_wall_transferred"] = cap_sweep_with_marginals(
        transferred, "two_point", "local per-width wall scaled to the ranked "
        "zero-draft round")
    healed_local = list(transferred)
    healed_local[4] = statistics.mean([transferred[3], transferred[5]])
    scenarios["local_wall_transferred_width6_healed"] = cap_sweep_with_marginals(
        healed_local, "two_point", "local wall transferred, width-6 healed to "
        "the neighbour mean")

    payload = {
        "experiment": "e159-r2-cap-replay",
        "harness_of_board_and_receipt_quantities": "ranked",
        "harness_of_sweep_quantities": "local",
        "official_or_ranked_score": False,
        "anchor_receipt": "5a9f130a",
        "decode_tokens": DECODE_TOKENS,
        "shipped_cap": SHIPPED_CAP,
        "families": results,
        "wall_identified_set_scan": wall_scan,
        "board_deconvolved_fits": board_fits,
        "cap_scenarios": scenarios,
        "board_collinearity_audit": board_collinearity_audit(),
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")

    two = results["two_point"]
    print("\nsurvival S(i), local, isotonic:",
          [round(v, 5) for v in two["survival"]])
    print("theta per prompt:",
          {k: round(v, 4) for k, v in two["thetas"].items()})
    print("free-fit ranked marginals into widths 2..8 (ms):",
          [round(v * 1e3, 3) for v in two["marginals_into_width_2_to_8"]])
    print("free-fit residual (ms):", round(two["fit_residual_seconds"] * 1e3, 4))
    print("\ncap table, two_point family")
    print(" C   published_median   delta_%   median_pair")
    for cap in range(9):
        row = two["caps"][cap]
        print(f" {cap}   {row['published_median']:.8f}  "
              f"{row['median_pct_change']:+8.4f}   {row['median_pair']}")

    print("\nwidth-6 marginal identified-set scan, eight anchor prompts only")
    print("  pinned_ms  residual_ms  x_free_fit  best_cap  best_cap_delta_%")
    for row in wall_scan:
        print(f"  {row['pinned_width6_marginal_ms']:>8.1f} "
              f"{row['fit_residual_ms']:>11.4f} "
              f"{row['residual_over_free_fit']:>10.3f} "
              f"{row['best_cap']:>9} {row['best_cap_median_pct']:>15.4f}")

    print("\nboard-deconvolved ranked marginals, absolute ms at our anchor")
    for label, fit in board_fits.items():
        print(f"  {label}  n_cells={fit['n_cells']}  "
              f"resid_sd={fit['residual_sd_of_cost_over_anchor']:.4f}")
        for i, width in enumerate(range(2, 10)):
            print(f"    into width {width}: "
                  f"{fit['absolute_marginals_ms'][i]:8.3f} ms  "
                  f"+- {fit['absolute_marginal_se_ms'][i]:.3f}")

    print("\ncap scenarios")
    for key, scenario in scenarios.items():
        print(f"\n  --- {key}: {scenario['label']}")
        print("      marginals into widths 2..8 (ms): "
              + ", ".join(f"{v:.2f}" for v in
                          scenario["marginals_into_width_2_to_8_ms"]))
        print("       C   median     median_%    cand_leg_%  same_sign  pair")
        for cap in range(9):
            row = scenario["caps"][cap]
            print(f"       {cap}   {row['published_median']:.6f} "
                  f"{row['median_pct_change']:+9.4f} "
                  f"{row['candidate_leg_mean_pct']:+11.4f} "
                  f"{row['candidate_leg_same_sign']:>7}/8   "
                  f"{row['median_pair']}")
        print(f"      median argmax cap {scenario['argmax_cap']} at "
              f"{scenario['argmax_cap_median_pct']:+.4f} %; "
              f"candidate-leg argmax cap "
              f"{scenario['argmax_cap_candidate_leg']} at "
              f"{scenario['argmax_cap_candidate_leg_pct']:+.4f} % "
              f"({scenario['argmax_cap_candidate_leg_same_sign']}/8 same sign)")
        print(f"      LOO argmax spread {scenario['loo_cap_spread']} "
              f"stable={scenario['loo_stable']} folds="
              + str({k: v['argmax_cap']
                     for k, v in scenario['loo_folds'].items()}))


if __name__ == "__main__":
    main()
