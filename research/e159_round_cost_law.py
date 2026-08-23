"""E159 Part A: the ranked round-cost law over verify widths 1 to 9.

harness=ranked. Zero GPU. Read-only over the Yukon list endpoint cache.
RULE 177 clean: every number here comes from the per-prompt candidate leg
`mtp_seconds_per_token_mean`. No published median enters this file.

E157 identified the ranked row price only over verify widths 1.05 to 3.24,
because that is the range the `drama` sweep covers inside one head digest. The
depth price `costModelDepth` reads is a vector over verify widths 2 to 9, so
the whole supra-wall half of it was never measured on the ranked host.

THE ROUND COUNT IS THE WHOLE PROBLEM. `effective_mean_draft_len` is the
rational `drafted / rounds`, so the round count is a multiple of that
fraction's reduced denominator `q`. Arithmetic alone pins the multiple only
when `q > 256`, which happens when a round emits fewer than two tokens. Every
wide prompt therefore leaves at least the pair `{q, 2q}` open, and E157's
strict rule discards it. Under that strict rule the board holds ZERO cells at
widths 5 to 8, so the supra-wall price is not identified at all.

ONE PHYSICAL CONSTRAINT CLOSES IT, AND IT IS A CHAIN, NOT A CELL TEST. The
eight cells of one submission share one build, one host and one head. A round
that verifies more rows and runs more head steps cannot cost less than a
narrower round of that same build, so sorting a submission's cells by verify
width must sort them by round cost as well. Each cell offers a descending list
of candidate costs, one per candidate round count. A forward pass keeps the
candidates that admit a valid prefix and a backward pass keeps the ones that
admit a valid suffix, so the survivors are exactly the candidates that lie on
a complete monotone chain. Where one candidate survives, the round count is
identified. One arithmetically unique cell propagates through its whole
submission, which is why a chain identifies what no single-cell test can.

Three validations are printed and stored:

  crown          The chain recovers all eight cells of crown receipt
                 `ec24d59` and every one matches the tokens per round E157
                 published in FINDING 281, worst case 0.31 %. The two prompts
                 that break a naive "smallest round count" reading, `drama`
                 and `travel`, both at `3q`, are recovered correctly. A
                 smallest-candidate reader misses drama by 50 %.
  anchor rule    An independent sound test from a same-build cost anchor `A`:
                 `0.98 A <= cost <= 9 A`. It is conservative, because it can
                 only decide a cell whose cost lies between one and two
                 anchors, which truncates exactly the wide cells this
                 experiment needs. Where it decides, it must agree.
  slack scan     The monotone test carries multiplicative slack for thermal
                 and measurement noise. Identification counts, the anchor
                 agreement and the crown check are reported across a scan.

THE FRAME MATTERS MORE THAN THE BASIS. A log-cost fit CANNOT test convexity
here: `exp(b (w - 1))` has strictly increasing differences for any `b > 0`, so
it manufactures a rising marginal from a straight line. The law is therefore
fitted in LEVELS, on each cell's cost divided by its own submission's
near-width-one anchor. That ratio removes build speed multiplicatively without
assuming a functional form, and it leaves the coefficients in the exact unit
the scheduler uses.

THE WIDTH BASIS IS NOT EIGHT FREE STEPS EITHER. A cell publishes a MEAN width
and a MEAN cost over a distribution of per-round depths, and the eight prompts
occupy eight narrow width bands, so free unit steps are ill-conditioned where
no prompt lives. The reported segment basis places knots only at widths that
carry support. Widths with no supporting cell are marked INFERRED, never
measured.

Reported in the units the scheduler uses:
`cumulative[d]` is the cost of a round at verify width `d + 1` divided by the
cost of a width-one round, and `marginal[d] = cumulative[d + 1] -
cumulative[d]` prices the step into verify width `d + 2`.

    python3 research/e159_round_cost_law.py [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
from fractions import Fraction

import numpy as np

CACHE = "/tmp/yukon-board/full.json"
DECODE_TOKENS = 512
MAX_WIDTH = 9
MAX_DEPTH = 8
OUR_SOLVER = "morganmcg1"
MONOTONE_TOLERANCE = 0.02
TOLERANCE_SCAN = (0.0, 0.01, 0.02, 0.05, 0.10)
ANCHOR_FLOOR_TOLERANCE = 0.98
ANCHOR_CEILING_MULTIPLE = 9.0
WALL_MIN_ROUNDS_AT_WIDTH_6_PLUS = 120
WALL_SIGMA = 2.0
# A width is measured, not inferred, only if a bin of +-0.5 rows around it
# holds at least this many cells and this many distinct head digests.
SUPPORT_MIN_CELLS = 30
SUPPORT_MIN_HEADS = 2
SEGMENT_KNOTS = (1.0, 3.0, 4.0, 5.0, 6.0)
SHIPPED_HEAD_STEP_COST_RATIO = 0.18
E157_RANKED_H_HEAD_CLEAN = 0.11052

PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
# FINDING 281, harness=ranked, crown receipt ec24d59: tokens per round.
FINDING_281_TOKENS_PER_ROUND = {
    "plutarch": 1.0519,
    "drama": 2.0317,
    "travel": 2.4113,
    "beagle": 4.6545,
    "republic": 5.5054,
    "essays": 5.5628,
    "medicine": 5.6883,
    "botany": 6.3179,
}


def load_rows() -> list[dict]:
    with open(CACHE) as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else payload["submissions"]


def round_candidates(edl: float, non_drafting: int) -> list[int] | None:
    """Every round count consistent with the published columns."""
    frac = Fraction(edl).limit_denominator(DECODE_TOKENS)
    if abs(float(frac) - edl) > 1e-12:
        return None
    step = frac.denominator
    low = max(
        math.ceil(DECODE_TOKENS / (1.0 + edl) - 1e-9), int(non_drafting), step
    )
    out = [r for r in range(step, DECODE_TOKENS + 1, step) if r >= low]
    if edl < 1.0 and non_drafting > 0:
        out = [r for r in out if r * (1.0 - edl) <= non_drafting + 1e-9]
    return out


def monotone_survivors(chain: list[dict], tolerance: float):
    """Candidate indices that lie on a complete monotone cost chain."""
    forward: list[list[int]] = []
    lowest = None
    for cell in chain:
        if lowest is None:
            keep = list(range(len(cell["costs"])))
        else:
            keep = [
                j
                for j, cost in enumerate(cell["costs"])
                if cost >= lowest * (1.0 - tolerance)
            ]
        if not keep:
            return None
        forward.append(keep)
        lowest = min(cell["costs"][j] for j in keep)
    survivors: list[list[int]] = [[] for _ in chain]
    highest = None
    for i in range(len(chain) - 1, -1, -1):
        cell = chain[i]
        if highest is None:
            keep = list(forward[i])
        else:
            keep = [
                j
                for j in forward[i]
                if cell["costs"][j] * (1.0 - tolerance) <= highest
            ]
        if not keep:
            return None
        survivors[i] = keep
        highest = max(cell["costs"][j] for j in keep)
    return survivors


def read_row(row: dict) -> list[dict]:
    metrics = row.get("officialMetrics") or {}
    per_prompt = metrics.get("per_prompt")
    if not per_prompt:
        return []
    if metrics.get("decode_tokens") not in (None, DECODE_TOKENS):
        return []
    row_prefill = metrics.get("prefill_seconds_per_token")
    chain: list[dict] = []
    for entry in per_prompt:
        name = PROMPT_NAMES.get(entry["prompt_sha256"][:8])
        if name is None or not entry.get("parity_ok", True):
            continue
        edl = entry.get("effective_mean_draft_len")
        non_drafting = entry.get("non_drafting_round_count")
        spt = entry.get("mtp_seconds_per_token_mean")
        prefill = entry.get("prefill_seconds_per_token")
        if prefill is None:
            prefill = row_prefill
        if None in (edl, non_drafting, spt, prefill):
            continue
        candidates = round_candidates(float(edl), int(non_drafting))
        if not candidates:
            continue
        total = 1e6 * DECODE_TOKENS * (float(spt) - float(prefill))
        chain.append(
            {
                "id8": row["id"][:8],
                "solver": row.get("solverUsername"),
                "created": row.get("createdAt"),
                "official_score": row.get("officialScore"),
                "commit": row.get("submissionCommitSha"),
                "prompt": name,
                "head": str(entry.get("head_provenance_sha256"))[:16],
                "edl": float(edl),
                "width": float(edl) + 1.0,
                "candidates": candidates,
                "costs": [total / r for r in candidates],
                "total_decode_us": total,
                "non_drafting_rounds": int(non_drafting),
                "raw_ratio": entry.get("raw_ratio_of_means"),
            }
        )
    chain.sort(key=lambda c: c["width"])
    return chain


def anchor_pick(cell: dict, anchor_cost: float) -> int | None:
    feasible = [
        r
        for r, cost in zip(cell["candidates"], cell["costs"])
        if ANCHOR_FLOOR_TOLERANCE * anchor_cost
        <= cost
        <= ANCHOR_CEILING_MULTIPLE * anchor_cost
    ]
    return feasible[0] if len(feasible) == 1 else None


def collect_cells(rows: list[dict], tolerance: float):
    census: collections.Counter = collections.Counter()
    cells: list[dict] = []
    survivors_seen: collections.Counter = collections.Counter()
    for row in rows:
        chain = read_row(row)
        if len(chain) < 2:
            census["row_too_few_cells"] += 1
            continue
        survivors = monotone_survivors(chain, tolerance)
        if survivors is None:
            census["row_no_monotone_chain"] += 1
            continue
        census["row_used"] += 1
        anchors = [
            c for c in chain if len(c["candidates"]) == 1 and c["width"] < 2.0
        ]
        anchor = min(anchors, key=lambda c: c["width"]) if anchors else None
        anchor_cost = anchor["costs"][0] if anchor else None
        if anchor is None:
            census["row_without_anchor"] += 1
        for cell, keep in zip(chain, survivors):
            census["cell_seen"] += 1
            survivors_seen[min(len(keep), 4)] += 1
            if len(keep) != 1:
                census["cell_chain_ambiguous"] += 1
                continue
            rounds = cell["candidates"][keep[0]]
            census["cell_used"] += 1
            census[
                "cell_by_arithmetic"
                if len(cell["candidates"]) == 1
                else "cell_by_monotone_chain"
            ] += 1
            anchor_confirmed = False
            if anchor_cost is not None:
                decided = anchor_pick(cell, anchor_cost)
                if decided is not None:
                    anchor_confirmed = decided == rounds
                    census[
                        "cell_anchor_agrees"
                        if anchor_confirmed
                        else "cell_anchor_disagrees"
                    ] += 1
            tokens_per_round = DECODE_TOKENS / rounds
            cells.append(
                {
                    "id8": cell["id8"],
                    "solver": cell["solver"],
                    "created": cell["created"],
                    "official_score": cell["official_score"],
                    "commit": cell["commit"],
                    "prompt": cell["prompt"],
                    "head": cell["head"],
                    "width": cell["width"],
                    "rounds": rounds,
                    "n_candidates": len(cell["candidates"]),
                    "anchor_confirmed": anchor_confirmed,
                    "tokens_per_round": tokens_per_round,
                    "rejected_per_round": cell["width"] - tokens_per_round,
                    "accept_rate": (
                        (tokens_per_round - 1.0) / cell["edl"]
                        if cell["edl"] > 0
                        else 0.0
                    ),
                    "non_drafting_rounds": cell["non_drafting_rounds"],
                    "clean_us_per_round": cell["costs"][keep[0]],
                    "anchor_us_per_round": anchor_cost,
                    "anchor_width": anchor["width"] if anchor else None,
                    "cost_over_anchor": (
                        cell["costs"][keep[0]] / anchor_cost
                        if anchor_cost
                        else None
                    ),
                    "raw_ratio": cell["raw_ratio"],
                }
            )
    census["survivors_1_2_3_4plus"] = [survivors_seen[k] for k in (1, 2, 3, 4)]
    return cells, dict(census)


def crown_control(cells: list[dict]) -> dict:
    crown: list[dict] = []
    for prefix in ("ec24d59", "0cf1637e"):
        crown = [c for c in cells if c["id8"].startswith(prefix)]
        if crown:
            break
    out = {
        "receipt": crown[0]["id8"] if crown else None,
        "cells_recovered": len(crown),
        "by_prompt": {},
    }
    worst = 0.0
    for cell in sorted(crown, key=lambda c: c["width"]):
        reference = FINDING_281_TOKENS_PER_ROUND.get(cell["prompt"])
        if reference is None:
            continue
        relative = abs(cell["tokens_per_round"] - reference) / reference
        worst = max(worst, relative)
        out["by_prompt"][cell["prompt"]] = {
            "width": cell["width"],
            "rounds": cell["rounds"],
            "n_candidates": cell["n_candidates"],
            "recovered_tokens_per_round": cell["tokens_per_round"],
            "finding_281": reference,
            "relative_difference": relative,
            "cost_over_anchor": cell["cost_over_anchor"],
        }
    out["worst_relative_difference"] = worst
    return out


def dummies(values: list[str]) -> tuple[list[list[float]], list[str]]:
    levels = sorted(set(values))
    keep = levels[1:]
    index = {level: i for i, level in enumerate(keep)}
    matrix = []
    for value in values:
        row = [0.0] * len(keep)
        if value in index:
            row[index[value]] = 1.0
        matrix.append(row)
    return matrix, keep


def wls_cluster(design, target, weights, clusters) -> dict:
    sqrt_w = np.sqrt(weights)
    x_w = design * sqrt_w[:, None]
    y_w = target * sqrt_w
    xtx = x_w.T @ x_w
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x_w.T @ y_w)
    residual = target - design @ beta
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(clusters):
        groups.setdefault(key, []).append(i)
    meat = np.zeros_like(xtx)
    for rows_index in groups.values():
        idx = np.array(rows_index)
        score = (design[idx] * (weights[idx] * residual[idx])[:, None]).sum(0)
        meat += np.outer(score, score)
    n_clusters = len(groups)
    n_obs, n_par = design.shape
    scale = (n_clusters / max(n_clusters - 1, 1)) * (
        (n_obs - 1) / max(n_obs - n_par, 1)
    )
    cov = xtx_inv @ meat @ xtx_inv * scale
    weighted_mean = float(np.sum(weights * target) / np.sum(weights))
    ss_total = float(np.sum(weights * (target - weighted_mean) ** 2))
    ss_residual = float(np.sum(weights * residual**2))
    weight_mean = float(np.mean(weights))
    return {
        "beta": beta,
        "cov": cov,
        "stderr": np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        "n_obs": int(n_obs),
        "n_clusters": int(n_clusters),
        "r_squared": 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0,
        # per-observation residual scale, with the round weighting divided out
        "residual_sd": math.sqrt(
            ss_residual / weight_mean / max(n_obs - n_par, 1)
        ),
    }


def width_columns(widths: list[float], basis: str):
    if basis == "linear":
        return [("width", [w - 1.0 for w in widths])]
    if basis == "quadratic":
        return [
            ("width", [w - 1.0 for w in widths]),
            ("width_sq", [(w - 1.0) ** 2 for w in widths]),
        ]
    if basis.startswith("piecewise@"):
        knot = float(basis.split("@")[1])
        return [
            ("width", [w - 1.0 for w in widths]),
            ("width_above", [max(w - knot, 0.0) for w in widths]),
        ]
    if basis == "segments":
        columns = []
        for i, low in enumerate(SEGMENT_KNOTS):
            high = (
                SEGMENT_KNOTS[i + 1]
                if i + 1 < len(SEGMENT_KNOTS)
                else float("inf")
            )
            span = high - low
            columns.append(
                (
                    "seg_%g_%s" % (low, "up" if math.isinf(high) else "%g" % high),
                    [
                        min(max(w - low, 0.0), span) if span != float("inf")
                        else max(w - low, 0.0)
                        for w in widths
                    ],
                )
            )
        return columns
    raise ValueError(basis)


def fit(cells: list[dict], spec: str, basis: str) -> dict:
    widths = [c["width"] for c in cells]
    names = ["const"]
    columns: list[list[float]] = [[1.0] * len(cells)]
    width_names = []
    for name, column in width_columns(widths, basis):
        width_names.append(name)
        names.append(name)
        columns.append(column)
    names.append("rejected_per_round")
    columns.append([c["rejected_per_round"] for c in cells])
    names.append("anchor_width")
    columns.append([c["anchor_width"] for c in cells])
    for key, prefix in (("head", "head_"), ("prompt", "prompt_")):
        if key not in spec:
            continue
        block, levels = dummies([c[key] for c in cells])
        for j, level in enumerate(levels):
            names.append(prefix + level[:8])
            columns.append([row[j] for row in block])
    design = np.array(columns, dtype=float).T
    target = np.array([c["cost_over_anchor"] for c in cells], dtype=float)
    weights = np.array([float(c["rounds"]) for c in cells], dtype=float)
    out = wls_cluster(design, target, weights, [c["id8"] for c in cells])
    out.update(
        {
            "names": names,
            "width_names": width_names,
            "spec": spec,
            "basis": basis,
            "design": design,
            "weights": weights,
        }
    )
    return out


def coefficient(result: dict, name: str) -> tuple[float, float]:
    i = result["names"].index(name)
    return float(result["beta"][i]), float(result["stderr"][i])


def summarise(result: dict) -> dict:
    keep = [
        n
        for n in result["names"]
        if not n.startswith(("head_", "prompt_"))
    ]
    return {
        "spec": result["spec"],
        "basis": result["basis"],
        "target": "cost_over_own_submission_anchor",
        "n_cells": result["n_obs"],
        "n_submission_clusters": result["n_clusters"],
        "r_squared": result["r_squared"],
        "residual_sd": result["residual_sd"],
        "beta": {n: coefficient(result, n)[0] for n in keep},
        "se": {n: coefficient(result, n)[1] for n in keep},
    }


def reference_row(result: dict, width: float) -> np.ndarray:
    """A design row at `width`, every other covariate at its round-weighted
    mean, so a ratio of two such rows is the pure width effect."""
    design, weights = result["design"], result["weights"]
    row = (design * weights[:, None]).sum(0) / weights.sum()
    row[result["names"].index("const")] = 1.0
    for name, column in width_columns([width], result["basis"]):
        row[result["names"].index(name)] = column[0]
    return row


def price_vector(result: dict) -> dict:
    """cumulative[d] and marginal[d] in the scheduler's own units."""
    beta, cov = result["beta"], result["cov"]
    base_row = reference_row(result, 1.0)
    base = float(base_row @ beta)
    cumulative, errors = [], []
    for depth in range(MAX_DEPTH + 1):
        row = reference_row(result, depth + 1.0)
        value = float(row @ beta)
        gradient = (row * base - base_row * value) / (base * base)
        cumulative.append(value / base)
        errors.append(math.sqrt(max(float(gradient @ cov @ gradient), 0.0)))
    return {
        "fitted_width_one_ratio_to_anchor": base,
        "cumulative": cumulative,
        "cumulative_se": errors,
        "marginal": [
            cumulative[d + 1] - cumulative[d] for d in range(MAX_DEPTH)
        ],
        "monotone": all(
            cumulative[d + 1] >= cumulative[d] for d in range(MAX_DEPTH)
        ),
    }


def width_support(cells: list[dict]) -> dict:
    out = {}
    for width in range(1, MAX_WIDTH + 1):
        sample = [c for c in cells if abs(c["width"] - width) <= 0.5]
        entry = {
            "n_cells": len(sample),
            "n_rounds": int(sum(c["rounds"] for c in sample)),
            "n_heads": len({c["head"] for c in sample}),
            "n_submissions": len({c["id8"] for c in sample}),
            "prompts": sorted({c["prompt"] for c in sample}),
        }
        entry["measured"] = bool(
            entry["n_cells"] >= SUPPORT_MIN_CELLS
            and entry["n_heads"] >= SUPPORT_MIN_HEADS
        )
        if sample:
            values = np.array([c["cost_over_anchor"] for c in sample])
            weights = np.array([float(c["rounds"]) for c in sample])
            mean = float(np.sum(weights * values) / np.sum(weights))
            var = float(
                np.sum(weights * (values - mean) ** 2) / np.sum(weights)
            )
            entry["raw_cost_over_anchor_mean"] = mean
            entry["raw_cost_over_anchor_se"] = math.sqrt(
                var / max(len(sample) - 1, 1)
            )
            entry["mean_us_per_round"] = float(
                np.sum(weights * np.array([c["clean_us_per_round"] for c in sample]))
                / np.sum(weights)
            )
        out[str(width)] = entry
    return out


def scan_knot(cells: list[dict], spec: str) -> dict:
    scan = {}
    for knot in [float(k) for k in range(3, MAX_WIDTH)]:
        result = fit(cells, spec, "piecewise@%.1f" % knot)
        above, above_se = coefficient(result, "width_above")
        base, base_se = coefficient(result, "width")
        scan["%.1f" % knot] = {
            "residual_sd": result["residual_sd"],
            "r_squared": result["r_squared"],
            "slope_below": base,
            "slope_below_se": base_se,
            "extra_slope_above": above,
            "extra_slope_above_se": above_se,
            "extra_sigma": above / above_se if above_se > 0 else None,
        }
    best = min(scan, key=lambda k: scan[k]["residual_sd"])
    return {"scan": scan, "best_knot": float(best), "best": scan[best]}


def segment_profile(summary: dict) -> dict:
    """Per-row price of each fitted width segment, with the wall test."""
    beta = {n: v for n, v in summary["beta"].items() if n.startswith("seg_")}
    se = {n: summary["se"][n] for n in beta}
    order = list(beta)
    ranked = sorted(order, key=lambda n: beta[n], reverse=True)
    top, second = ranked[0], ranked[1]
    gap = beta[top] - beta[second]
    sigma = gap / math.sqrt(se[top] ** 2 + se[second] ** 2)
    return {
        "slopes": beta,
        "se": se,
        "order": order,
        "largest": top,
        "largest_slope": beta[top],
        "second_largest": second,
        "second_largest_slope": beta[second],
        "dominance_gap": gap,
        "dominance_sigma": sigma,
        # A wall means ONE step is decisively the most expensive, not merely
        # that the price is not flat.
        "wall_present": bool(sigma >= WALL_SIGMA),
        "wall_index": int(SEGMENT_KNOTS[order.index(top)]) + 1
        if sigma >= WALL_SIGMA
        else None,
    }


def analyse(cells: list[dict], label: str) -> dict:
    specs = ["prompt", "head+prompt"]
    preferred = fit(cells, "head+prompt", "segments")
    price = price_vector(preferred)
    fits = {
        "%s|%s" % (spec, basis): summarise(fit(cells, spec, basis))
        for spec in specs
        for basis in ("linear", "quadratic", "segments")
    }
    profile = segment_profile(fits["head+prompt|segments"])
    # A stratum whose prompts carry no within-prompt width variation cannot
    # identify the law: the prompt fixed effects absorb the whole width axis
    # and the fit returns a cost curve that is not even positive.
    degenerate = not price["monotone"] or min(price["cumulative"]) <= 0.0
    return {
        "label": label,
        "n_cells": len(cells),
        "n_submissions": len({c["id8"] for c in cells}),
        "n_heads": len({c["head"] for c in cells}),
        "rounds_total": int(sum(c["rounds"] for c in cells)),
        "rounds_at_width_6_plus": int(
            sum(c["rounds"] for c in cells if c["width"] >= 6.0)
        ),
        "cells_at_width_6_plus": sum(1 for c in cells if c["width"] >= 6.0),
        "heads_at_width_6_plus": len(
            {c["head"] for c in cells if c["width"] >= 6.0}
        ),
        "submissions_at_width_6_plus": len(
            {c["id8"] for c in cells if c["width"] >= 6.0}
        ),
        "median_anchor_us_per_round": float(
            np.median([c["anchor_us_per_round"] for c in cells])
        ),
        "width_support": width_support(cells),
        "fits": fits,
        "knot_scan": scan_knot(cells, "head+prompt"),
        "preferred_spec": "head+prompt|segments",
        "segment_profile": profile,
        "price_vector": price,
        "price_vector_linear": price_vector(fit(cells, "head+prompt", "linear")),
        "law_identified": not degenerate,
    }


def tolerance_sensitivity(rows: list[dict]) -> dict:
    out = {}
    for tolerance in TOLERANCE_SCAN:
        cells, census = collect_cells(rows, tolerance)
        control = crown_control(cells)
        out["%.2f" % tolerance] = {
            "cells_identified": census.get("cell_used", 0),
            "cells_ambiguous": census.get("cell_chain_ambiguous", 0),
            "rows_without_monotone_chain": census.get(
                "row_no_monotone_chain", 0
            ),
            "anchor_agrees": census.get("cell_anchor_agrees", 0),
            "anchor_disagrees": census.get("cell_anchor_disagrees", 0),
            "rounds_at_width_6_plus": int(
                sum(c["rounds"] for c in cells if c["width"] >= 6.0)
            ),
            "crown_cells_recovered": control["cells_recovered"],
            "crown_worst_relative_difference": control[
                "worst_relative_difference"
            ],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=pathlib.Path,
        default=pathlib.Path("research/e159-artifacts/e159_round_cost_law.json"),
    )
    args = parser.parse_args()

    rows = load_rows()
    cells, census = collect_cells(rows, MONOTONE_TOLERANCE)
    cells = [c for c in cells if c["cost_over_anchor"] is not None]
    cells.sort(key=lambda c: c["width"])

    print("harness=ranked  E159 Part A  zero GPU  RULE 177 candidate leg only")
    print("board rows %d, monotone slack %.2f" % (len(rows), MONOTONE_TOLERANCE))
    for key in sorted(census):
        print("  %-30s %s" % (key, census[key]))
    print("  cells with a build anchor        %d" % len(cells))

    control = crown_control(cells)
    print(
        "\ncrown positive control %s: %d of 8 cells recovered, worst relative "
        "difference against FINDING 281 %.4f %%"
        % (
            control["receipt"],
            control["cells_recovered"],
            100.0 * control["worst_relative_difference"],
        )
    )
    print(
        "%-10s %8s %6s %6s %10s %10s %8s"
        % ("prompt", "width", "cands", "R", "tok/round", "FINDING", "rel %")
    )
    for prompt, e in control["by_prompt"].items():
        print(
            "%-10s %8.3f %6d %6d %10.4f %10.4f %8.4f"
            % (
                prompt,
                e["width"],
                e["n_candidates"],
                e["rounds"],
                e["recovered_tokens_per_round"],
                e["finding_281"],
                100.0 * e["relative_difference"],
            )
        )

    print("\nmonotone slack sensitivity")
    sensitivity = tolerance_sensitivity(rows)
    print(
        "%6s %10s %10s %9s %9s %12s %8s"
        % ("slack", "identified", "ambig", "agree", "disagree", "rounds w>=6", "crown %")
    )
    for key in sorted(sensitivity, key=float):
        s = sensitivity[key]
        print(
            "%6s %10d %10d %9d %9d %12d %8.4f"
            % (
                key,
                s["cells_identified"],
                s["cells_ambiguous"],
                s["anchor_agrees"],
                s["anchor_disagrees"],
                s["rounds_at_width_6_plus"],
                100.0 * s["crown_worst_relative_difference"],
            )
        )

    conservative = [
        c for c in cells if c["n_candidates"] == 1 or c["anchor_confirmed"]
    ]
    # The anchor rule only decides a cell when exactly one candidate falls in
    # its cost window, so retention depends on the cell's own cost. Measure
    # that selection before using the conservative subset for anything.
    conservative_selection = []
    for width in range(1, MAX_WIDTH + 1):
        every = [c["cost_over_anchor"] for c in cells if round(c["width"]) == width]
        kept = [
            c["cost_over_anchor"]
            for c in conservative
            if round(c["width"]) == width
        ]
        if not every:
            continue
        med_all = statistics.median(every)
        med_kept = statistics.median(kept) if kept else None
        conservative_selection.append(
            {
                "width": width,
                "cells_all": len(every),
                "cells_kept": len(kept),
                "median_cost_over_anchor_all": med_all,
                "median_cost_over_anchor_kept": med_kept,
                "selection_shift_pct": (
                    100.0 * (med_kept / med_all - 1.0)
                    if med_kept is not None
                    else None
                ),
            }
        )
    print("\nconservative subset selection, by verify width")
    print("%6s %9s %9s %9s %10s %11s" % (
        "width", "cells", "kept", "keep %", "med c/A", "shift %"))
    for entry in conservative_selection:
        print(
            "%6d %9d %9d %9.1f %10.4f %11s"
            % (
                entry["width"],
                entry["cells_all"],
                entry["cells_kept"],
                100.0 * entry["cells_kept"] / entry["cells_all"],
                entry["median_cost_over_anchor_all"],
                "-"
                if entry["selection_shift_pct"] is None
                else "%.3f" % entry["selection_shift_pct"],
            )
        )

    strata = {
        "board": cells,
        "conservative": conservative,
        "ours": [c for c in cells if c["solver"] == OUR_SOLVER],
    }
    results = {}
    for label, sample in strata.items():
        if len(sample) < 40:
            print("\n%s: %d cells, too few for a fit" % (label, len(sample)))
            continue
        results[label] = analyse(sample, label)
        s = results[label]
        print(
            "\n%s: %d cells, %d submissions, %d heads, %d rounds, median "
            "anchor %.0f us per round"
            % (
                label,
                s["n_cells"],
                s["n_submissions"],
                s["n_heads"],
                s["rounds_total"],
                s["median_anchor_us_per_round"],
            )
        )

    board = results["board"]
    print("\nsupport by verify width, board")
    print(
        "%6s %9s %10s %7s %6s %10s %12s %9s  %s"
        % (
            "width",
            "cells",
            "rounds",
            "heads",
            "subs",
            "raw c/A",
            "mean us/round",
            "measured",
            "prompts",
        )
    )
    for width in range(1, MAX_WIDTH + 1):
        e = board["width_support"][str(width)]
        print(
            "%6d %9d %10d %7d %6d %10s %12s %9s  %s"
            % (
                width,
                e["n_cells"],
                e["n_rounds"],
                e["n_heads"],
                e["n_submissions"],
                "%.4f" % e["raw_cost_over_anchor_mean"]
                if e["n_cells"]
                else "-",
                "%.0f" % e["mean_us_per_round"] if e["n_cells"] else "-",
                "yes" if e["measured"] else "INFERRED",
                ",".join(e["prompts"]),
            )
        )

    print("\nfit comparison, board, level frame (cost over own build anchor)")
    print("%-26s %10s %12s" % ("spec|basis", "r_squared", "residual_sd"))
    for key in sorted(board["fits"]):
        f = board["fits"][key]
        print("%-26s %10.4f %12.5f" % (key, f["r_squared"], f["residual_sd"]))
    quad = board["fits"]["head+prompt|quadratic"]
    curvature_sigma = (
        quad["beta"]["width_sq"] / quad["se"]["width_sq"]
        if quad["se"]["width_sq"] > 0
        else None
    )
    print(
        "quadratic width^2 = %.5f +- %.5f  (%.2f sigma)"
        % (
            quad["beta"]["width_sq"],
            quad["se"]["width_sq"],
            curvature_sigma or float("nan"),
        )
    )

    print("\nsegment slopes, board, head+prompt fixed effects, per row")
    seg = board["fits"]["head+prompt|segments"]
    for name in seg["beta"]:
        if name.startswith("seg_"):
            print(
                "  %-14s %8.5f +- %.5f"
                % (name, seg["beta"][name], seg["se"][name])
            )

    print("\nknot scan, level frame, head+prompt fixed effects, board")
    print(
        "%6s %12s %12s %14s %8s"
        % ("knot", "residual_sd", "slope_below", "extra_above", "sigma")
    )
    scan = board["knot_scan"]
    for knot in sorted(scan["scan"], key=float):
        e = scan["scan"][knot]
        print(
            "%6s %12.5f %12.5f %14.5f %8s"
            % (
                knot,
                e["residual_sd"],
                e["slope_below"],
                e["extra_slope_above"],
                "%.2f" % e["extra_sigma"] if e["extra_sigma"] else "n/a",
            )
        )
    print("best knot %.1f" % scan["best_knot"])

    for label in results:
        price = results[label]["price_vector"]
        support = results[label]["width_support"]
        anchor_us = results[label]["median_anchor_us_per_round"]
        print(
            "\nranked round cost by verify width, %s, segment basis, "
            "head+prompt fixed effects" % label
        )
        print(
            "%6s %8s %12s %9s %12s %12s %10s %9s"
            % (
                "width",
                "depth d",
                "cumulative",
                "se",
                "marginal[d]",
                "mean_us",
                "n_rounds",
                "measured",
            )
        )
        for depth in range(MAX_DEPTH + 1):
            width = depth + 1
            e = support[str(width)]
            print(
                "%6d %8d %12.4f %9.4f %12s %12.0f %10d %9s"
                % (
                    width,
                    depth,
                    price["cumulative"][depth],
                    price["cumulative_se"][depth],
                    "%.4f" % price["marginal"][depth]
                    if depth < MAX_DEPTH
                    else "-",
                    price["cumulative"][depth] * anchor_us,
                    e["n_rounds"],
                    "yes" if e["measured"] else "INFERRED",
                )
            )
        print("  monotone across the whole range: %s" % price["monotone"])

    rounds_6 = board["rounds_at_width_6_plus"]
    data_gate = (
        rounds_6 >= WALL_MIN_ROUNDS_AT_WIDTH_6_PLUS
        and board["heads_at_width_6_plus"] >= 2
    )
    price = board["price_vector"]
    profile = board["segment_profile"]
    measured_widths = [
        w
        for w in range(1, MAX_WIDTH + 1)
        if board["width_support"][str(w)]["measured"]
    ]
    wall_identified = bool(data_gate and profile["wall_present"])
    wall_index = profile["wall_index"] if wall_identified else None
    wall_marginal = profile["largest_slope"] if wall_identified else None
    sub_wall = [
        profile["slopes"][name]
        for i, name in enumerate(profile["order"])
        if wall_index is None or SEGMENT_KNOTS[i] + 1 < wall_index
    ]
    h_subwall = (
        float(np.mean(sub_wall))
        if sub_wall
        else profile["slopes"][profile["order"][0]]
    )
    mean_measured_slope = (price["cumulative"][6] - 1.0) / 6.0
    law_form = (
        "convex"
        if profile["wall_present"] or (curvature_sigma or 0.0) >= WALL_SIGMA
        else (
            "concave"
            if (curvature_sigma or 0.0) <= -WALL_SIGMA
            else "linear"
        )
    )

    print(
        "\nrounds recovered at width >= 6: board %d, ours %s (gate %d)"
        % (
            rounds_6,
            results.get("ours", {}).get("rounds_at_width_6_plus"),
            WALL_MIN_ROUNDS_AT_WIDTH_6_PLUS,
        )
    )
    print(
        "distinct heads at width >= 6: %d, submissions %d"
        % (board["heads_at_width_6_plus"], board["submissions_at_width_6_plus"])
    )
    print("measured widths            = %s" % measured_widths)
    print(
        "largest segment             = %s at %.5f, next %s at %.5f, "
        "dominance %.2f sigma"
        % (
            profile["largest"],
            profile["largest_slope"],
            profile["second_largest"],
            profile["second_largest_slope"],
            profile["dominance_sigma"],
        )
    )
    print("data gate passed           = %s" % data_gate)
    print(
        "one step dominates at %.0f sigma = %s" % (WALL_SIGMA, profile["wall_present"])
    )
    print("e159_wall_identified       = %s" % wall_identified)
    print("e159_wall_index            = %s" % wall_index)
    print(
        "e159_wall_marginal_W       = %s"
        % ("%.5f" % wall_marginal if wall_marginal is not None else None)
    )
    print("e159_h_subwall             = %.5f" % h_subwall)
    print(
        "mean measured per-row price = %.5f over widths 1 to 7" % mean_measured_slope
    )
    print(
        "  shipped headStepCostRatio %.5f, E157 ranked h_head_clean %.5f"
        % (SHIPPED_HEAD_STEP_COST_RATIO, E157_RANKED_H_HEAD_CLEAN)
    )
    print("e159_law_form              = %s" % law_form)
    for label in results:
        print(
            "law identified in stratum %-13s = %s"
            % (label, results[label]["law_identified"])
        )

    out = {
        "experiment": "e159-part-a-ranked-round-cost-law",
        "harness": "ranked",
        "rule_177_clean": True,
        "source": "Yukon list endpoint officialMetrics.per_prompt, read only",
        "identification": {
            "rule": (
                "the round count is a multiple of the reduced denominator of "
                "effective_mean_draft_len; within one submission the cells "
                "sorted by verify width must also sort by round cost, and a "
                "forward and a backward pass keep only the candidates that "
                "lie on a complete monotone chain"
            ),
            "monotone_tolerance": MONOTONE_TOLERANCE,
            "anchor_floor_tolerance": ANCHOR_FLOOR_TOLERANCE,
            "anchor_ceiling_multiple": ANCHOR_CEILING_MULTIPLE,
            "cells_where_anchor_rule_agrees": census.get(
                "cell_anchor_agrees", 0
            ),
            "cells_where_anchor_rule_disagrees": census.get(
                "cell_anchor_disagrees", 0
            ),
            "tolerance_sensitivity": sensitivity,
        },
        "frame": (
            "levels, cost divided by the same submission's near-width-one "
            "anchor; a log frame cannot test convexity because exp(b(w-1)) "
            "has increasing differences for every b > 0"
        ),
        "census": census,
        "crown_positive_control": control,
        "strata": results,
        "e159_round_us_by_width": {
            label: {
                str(d + 1): {
                    "depth_index": d,
                    "cumulative_over_width_one": results[label][
                        "price_vector"
                    ]["cumulative"][d],
                    "cumulative_se": results[label]["price_vector"][
                        "cumulative_se"
                    ][d],
                    "mean_us": results[label]["price_vector"]["cumulative"][d]
                    * results[label]["median_anchor_us_per_round"],
                    "se_us": results[label]["price_vector"]["cumulative_se"][d]
                    * results[label]["median_anchor_us_per_round"],
                    "marginal_into_next_width": (
                        results[label]["price_vector"]["marginal"][d]
                        if d < MAX_DEPTH
                        else None
                    ),
                    "n_cells": results[label]["width_support"][str(d + 1)][
                        "n_cells"
                    ],
                    "n_rounds": results[label]["width_support"][str(d + 1)][
                        "n_rounds"
                    ],
                    "n_heads": results[label]["width_support"][str(d + 1)][
                        "n_heads"
                    ],
                    "measured": results[label]["width_support"][str(d + 1)][
                        "measured"
                    ],
                }
                for d in range(MAX_DEPTH + 1)
            }
            for label in results
        },
        "e159_measured_widths": measured_widths,
        "e159_conservative_selection": conservative_selection,
        "e159_wall_data_gate_passed": bool(data_gate),
        "e159_wall_present": bool(profile["wall_present"]),
        "e159_wall_identified": wall_identified,
        "e159_wall_index": wall_index,
        "e159_wall_marginal_W": wall_marginal,
        "e159_h_subwall": h_subwall,
        "e159_law_form": law_form,
        "quadratic_curvature_sigma": curvature_sigma,
        "segment_profile": profile,
        "shipped_head_step_cost_ratio": SHIPPED_HEAD_STEP_COST_RATIO,
        "e157_ranked_h_head_clean": E157_RANKED_H_HEAD_CLEAN,
        "wall_gate_rounds": WALL_MIN_ROUNDS_AT_WIDTH_6_PLUS,
        "cells": cells,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
