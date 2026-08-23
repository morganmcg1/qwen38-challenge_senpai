#!/usr/bin/env python3
"""E149 rung D. Solve the ranked M5 per-width round-cost curve from receipts.

`harness=ranked` for every fitted curve, every observable and every verdict.
The synthetic ground truth is `harness=local` (edward's E145 R2 measured width
cost curve on `applegpu_g16s`) EXPRESSED IN RANKED UNITS by the published level
transfer `k = 2.1034`. The two frames never touch: the local curve enters only
as a generator of synthetic observations inside the controls, and no ranked
observation is ever divided by `k`.

Zero GPU.

The question
------------
Rule 138 (admissible width set `[1,2,3,4,5]`, width 7 never optimal) is a
statement about edward's LOCAL measured curve divided by one scalar. F130.2
shows the two hosts diverge exactly where the decision lives: `g16s` spills at
`NA = 6` and `g17s` does not spill until `NA = 8`. So the local 5->6 cliff may
be a spill artefact the ranked host never pays. This module asks whether the
ranked receipts can separate the two competing curves on their own.

The observation model
---------------------
For one published row and one prompt, everything below is public:

    leg_us     = 512 * mtp_seconds_per_token_mean * 1e6
    prefill_us = 512 * prefill_seconds_per_token * 1e6
    decode_us  = leg_us - prefill_us
    decode_us_per_round = decode_us / R

and the model is a mass-weighted sum over verify widths, `M = d + 1` (F78):

    decode_us_per_round(row, prompt) = alpha_row * sum_w m_w(row, prompt) * C_w

`C_9` is reported as UNIDENTIFIED by construction, not by data: the shipped
`SEGMENTED_VERIFY_DEPTH_CAP = 7` (`research/e128_replay.py:44`, mirrored from
`Qwen36MTPLimits`) caps the drafted depth at 7, so `M <= 8` on every round of
every row in the dataset and no receipt can carry width-9 mass.

Round counts are EXACT, not fitted
----------------------------------
`effective_mean_draft_len` is a ratio of two integers, `D / R`, where `D` is
the total drafted token count and `R` the round count of a 512-token leg. The
published double therefore recovers the reduced fraction exactly, and the leg
identity `R + A = 512` with `0 <= A <= D` prunes the multiples. Plutarch is
pinned uniquely at `R = 487` on `572b2cc4` with no assumption at all. This
replaces the F111 `R_assumed` heuristic with an arithmetic identity and is
reported as its own result.

Advisor Error 156 is respected everywhere: no cost is ever evaluated at a
mixture mean, and no curve is ever interpolated at a mean width. Only full mass
vectors enter the design.

Usage:
  python3 research/e149_rungD_curve.py --json research/e149-rungD.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections import Counter
from fractions import Fraction

import numpy as np
from scipy.optimize import nnls

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_ourcurve import (  # noqa: E402
    MAX_ROWS, PROMPT_FIXTURES, fixture_histograms, tilt_to_mean,
)
from e128_replay import (  # noqa: E402
    ACCEPT_EMA_ALPHA, EMA_PRIOR, MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP,
    record_accept_outcome,
)

BOARD = pathlib.Path("/tmp/yukon-board/full.json")
DECODE_TOKENS = 512
MAX_WIDTH = min(MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP) + 1  # 8

PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
ORDER = ["beagle", "medicine", "essays", "republic", "botany", "drama",
         "travel", "plutarch"]

# F111 `R_assumed`, kept only as a cross-check on the exact recovery.
R_ASSUMED = {"beagle": 110, "medicine": 90, "essays": 92, "botany": 81,
             "republic": 93, "drama": 252, "travel": 212, "plutarch": 487}

# E149 rung 0a n=11 at-zero block, per-prompt sd of the TOTAL leg, percent.
# These are PAIR sds (parent -> child), so they bound a single-receipt sd from
# above by about sqrt(2). Used as given, which is the conservative choice.
NOISE_PCT = {"beagle": 0.1202, "botany": 0.1059, "drama": 0.1709,
             "essays": 0.0407, "medicine": 0.0605, "plutarch": 0.3348,
             "republic": 0.0663, "travel": 0.1993}

# H-null: the replayed curve, already in ranked units (F5).
H_NULL = {1: 31173.2, 2: 34619.3, 3: 38065.4, 4: 41511.4, 5: 44957.5,
          6: 61198.8, 7: 62824.9, 8: 70315.4, 9: 75638.9}

# H-alt: edward's E145 R2 measured LOCAL curve divided by the level transfer
# k = 2.1034 to put it in ranked units. Width 1 is not in R2, so the plutarch
# anchor supplies it; the admissibility test needs C_1 and R2 never timed it.
E145_MEASURED_LOCAL_US = {2: 70429.4, 3: 75197.3, 4: 83188.0, 5: 95301.5,
                          6: 124436.0, 7: 150297.5, 8: 154243.9}
E145_LEVEL_TRANSFER_K = 2.1034

# The shipped depth price and the pb6 boundary price, read this session from
# Sources/MLXFastModel/Qwen36MTPBlockSession.swift:937 (headStepCostRatio
# 0.18), :983 (passBoundaryVerifyWidth 6), :1013 (passBoundaryTierFactor 1.45),
# :1019-1027 (makeUniformDepthPrice) and :1030-1041 (makeBoundaryDepthPrice).
HEAD_STEP_COST_RATIO = 0.18
PASS_BOUNDARY_WIDTH = 6
PASS_BOUNDARY_TIER = 1.45


def uniform_price() -> tuple[list[float], list[float]]:
    marginal = [HEAD_STEP_COST_RATIO] * MAX_DEPTH
    cumulative = [1.0 + i * HEAD_STEP_COST_RATIO for i in range(MAX_DEPTH + 1)]
    return marginal, cumulative


def boundary_price(width: int, tier: float) -> tuple[list[float], list[float]]:
    count = MAX_DEPTH
    within = count * HEAD_STEP_COST_RATIO / ((count - 1) + tier)
    marginal = [within] * count
    marginal[width - 2] = within * tier
    cumulative = [1.0]
    for step in marginal:
        cumulative.append(cumulative[-1] + step)
    return marginal, cumulative


PRICE_ARMS = {
    "ship": uniform_price(),
    "pb6": boundary_price(PASS_BOUNDARY_WIDTH, PASS_BOUNDARY_TIER),
}

# The rows this rung uses, and why.
ROW_SPEC = {
    "572b2cc4": {"arm": "ship", "core": True,
                 "why": "our tree, flat depth price, 3.66218564"},
    "e003a86d": {"arm": "pb6", "core": True,
                 "why": "our tree, pb6, 3.57502547; same tree as 572b2cc4 so "
                        "the pair shares one alpha and one C"},
    "684821ed": {"arm": "ship", "core": False,
                 "why": "the promoted crown, newjordan; a different tree, so "
                        "it carries its own alpha and adds only a second draw "
                        "of the flat mass matrix"},
    "1db9d63e": {"arm": "ship", "core": False,
                 "why": "drops onePass67 and adds width-2 routing, so it "
                        "changes the QMV dispatch TABLE. That is a shape "
                        "change, not a level change, and it cannot share C"},
    "7226dc9a": {"arm": "pb6", "core": False,
                 "why": "pb6 + rung A + rung B; kept only for the free "
                        "consistency check against e003a86d"},
}


# --------------------------------------------------------------- receipts

def load_rows(board: pathlib.Path, ids: list[str]) -> dict:
    rows = json.loads(board.read_text())
    if isinstance(rows, dict):
        rows = rows["submissions"]
    out = {}
    for row in rows:
        rid = (row.get("id") or "")[:8]
        if rid not in ids:
            continue
        metrics = row.get("officialMetrics") or {}
        per_prompt = metrics.get("per_prompt") or []
        if not per_prompt:
            out[rid] = {"id": rid, "score": row.get("officialScore"),
                        "status": row.get("status"), "per_prompt": None}
            continue
        table = {}
        for entry in per_prompt:
            name = PROMPT_NAMES[entry["prompt_sha256"][:8]]
            table[name] = {
                "candidate": entry["mtp_seconds_per_token_mean"],
                "prefill": entry["prefill_seconds_per_token"],
                "serial": entry["serial_seconds_per_token_mean"],
                "edl": entry["effective_mean_draft_len"],
                "non_drafting": entry["non_drafting_round_count"],
                "head": entry["head_provenance_sha256"][:12],
            }
        out[rid] = {"id": rid, "score": row.get("officialScore"),
                    "status": row.get("status"), "per_prompt": table}
    return out


def leg_us(entry: dict) -> float:
    return DECODE_TOKENS * entry["candidate"] * 1e6


def decode_us(entry: dict) -> float:
    return DECODE_TOKENS * (entry["candidate"] - entry["prefill"]) * 1e6


# ---------------------------------------------------- exact round counts

def exact_round_candidates(edl: float, non_drafting: int) -> list[dict]:
    """Every integer (R, D, A) consistent with the published `edl`.

    `edl = D / R` exactly, `R + A = 512`, `0 <= A <= D`, `R >= non_drafting`
    and `R <= 512`. The published double is the nearest representable value of
    a rational with denominator at most 512, so `limit_denominator` recovers
    the reduced fraction with no tolerance.
    """
    frac = Fraction(edl).limit_denominator(DECODE_TOKENS)
    num, den = frac.numerator, frac.denominator
    out = []
    if num == 0:
        return out
    k = 1
    while den * k <= DECODE_TOKENS:
        rounds, drafts = den * k, num * k
        accepted = DECODE_TOKENS - rounds
        if 0 <= accepted <= drafts and rounds >= non_drafting:
            out.append({"R": rounds, "D": drafts, "A": accepted,
                        "rate": accepted / drafts, "k": k,
                        "exact_fraction": "%d/%d" % (num, den)})
        k += 1
    return out


def pick_round_count(prompt: str, entry: dict, reference_rate: float) -> dict:
    """Choose the multiple with the acceptance-continuity rule.

    The per-position acceptance of a prompt is a property of the prompt and the
    target model, not of the depth price, so the realised accept rate moves
    slowly between arms. `R_ref = 512 / (1 + reference_rate * edl)` is the
    round count that rate would produce at this row's realised draft length,
    and the feasible multiple nearest to it is taken. The rule never reads a
    cost curve, so it cannot import either hypothesis into the fit.
    """
    cands = exact_round_candidates(entry["edl"], entry["non_drafting"])
    if not cands:
        raise SystemExit("no feasible round count for %s" % prompt)
    ref = DECODE_TOKENS / (1.0 + reference_rate * entry["edl"])
    best = min(cands, key=lambda c: abs(c["R"] - ref))
    best = dict(best)
    best["R_reference"] = ref
    best["n_candidates"] = len(cands)
    best["unique"] = len(cands) == 1
    best["alternatives"] = [c["R"] for c in cands if c["R"] != best["R"]]
    return best


def reference_rates(flat_row: dict) -> dict:
    """Per-prompt accept rate of the flat arm, from its own exact solution."""
    out = {}
    for prompt in ORDER:
        entry = flat_row["per_prompt"][prompt]
        cands = exact_round_candidates(entry["edl"], entry["non_drafting"])
        target = R_ASSUMED[prompt]
        match = [c for c in cands if c["R"] == target]
        chosen = match[0] if match else cands[0]
        out[prompt] = chosen["rate"]
    return out


# ------------------------------------------------------------ mass models

def tilt_mass(prompt: str, entry: dict, rounds: int, hists: dict) -> np.ndarray:
    """Mass model M1. Fixture shape, receipt mean, exact non-drafting pin.

    `research/e128_ourcurve.py:154-193` `prompt_width_histogram`, read this
    session and used unchanged. The shape is a LOCAL proxy and the mean is
    exact, so any per-width mass carries the proxy's error. That error is the
    subject of the third control below.
    """
    counter: Counter = Counter()
    for fixture in PROMPT_FIXTURES[prompt]:
        counter.update(hists[fixture])
    support = np.arange(0, MAX_ROWS, dtype=float)
    weights = np.array([float(counter.get(int(d), 0)) for d in support])
    weights = np.maximum(weights, 0.0)
    share = min(max(entry["non_drafting"] / rounds, 0.0), 1.0)
    target_depth = entry["edl"]
    if share > 0.0:
        conditional = target_depth / (1.0 - share)
        drafting = weights.copy()
        drafting[0] = 0.0
        probs = tilt_to_mean(drafting, support, conditional) * (1.0 - share)
        probs[0] += share
    else:
        weights[0] = 0.0
        probs = tilt_to_mean(weights, support, target_depth)
    return probs[:MAX_WIDTH] / probs[:MAX_WIDTH].sum()


def logit(x: float) -> float:
    x = min(max(x, 1e-9), 1.0 - 1e-9)
    return math.log(x / (1.0 - x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def forced_rounds(path: pathlib.Path) -> dict:
    """Recorded (margin, uncensored capability) pairs per local fixture."""
    legs = json.loads(path.read_text())["legs"]
    out: dict[str, list[tuple[float, int]]] = {}
    for leg in legs:
        rows = out.setdefault(leg["prompt_id"], [])
        for record in leg["rounds_detail"]:
            rows.append((float(record["margin"]), int(record["accepted"])))
    return out


def positions_from_capability(caps: list[int], cap_max: int) -> list[float]:
    """Per-position conditional acceptance from uncensored run lengths."""
    out = []
    for j in range(cap_max):
        at_risk = sum(1 for c in caps if c >= j)
        passed = sum(1 for c in caps if c >= j + 1)
        out.append(passed / at_risk if at_risk else 0.0)
    return out


def walk(ema: list[float], margin: float, price) -> int:
    """The shipped walk, `research/e134_rung2.py:45-84`, read this session.

    Reproduced here rather than imported so this module can drive it with an
    arbitrary price arm without touching edward's merged file.
    """
    marginal, cumulative = price
    cap = min(MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)
    reach, expected, depth = 1.0, 0.0, 0
    have_margin = not math.isnan(margin)
    while depth < cap:
        p = ema[depth]
        scale = {0: 2.0, 1: 3.0}.get(depth)
        if scale is not None and have_margin:
            p = min(p, 1.0 / (1.0 + math.exp(-margin / scale)))
        reach *= p
        threshold = marginal[depth] * (1.0 + expected) / cumulative[depth]
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def replay_leg(pairs: list[tuple[float, int]], delta: float, price,
               seed: int) -> dict:
    """Mass model M2. Simulate one 512-token leg of the shipped scheduler.

    The recorded round supplies the margin and its own quantile in the
    capability distribution; the logit shift `delta` moves the per-position
    acceptance onto the ranked prompt (the F92 transfer). Nothing in the walk
    reads a cost curve: the depth price is the frozen compiled table, so the
    mass vector is not circular with `C_w`.
    """
    cap_max = SEGMENTED_VERIFY_DEPTH_CAP
    caps = [min(c, cap_max) for _, c in pairs]
    base_p = positions_from_capability(caps, cap_max)
    shifted = [sigmoid(logit(p) + delta) for p in base_p]
    base_s = [1.0]
    for p in base_p:
        base_s.append(base_s[-1] * p)
    shift_s = [1.0]
    for p in shifted:
        shift_s.append(shift_s[-1] * p)
    rng = random.Random(seed)
    ema = list(EMA_PRIOR)
    emitted, rounds, drafts, accepted_total, non_drafting = 0, 0, 0, 0, 0
    widths = Counter()
    guard = 0
    while emitted < DECODE_TOKENS and guard < 4 * DECODE_TOKENS:
        guard += 1
        margin, recorded = pairs[rng.randrange(len(pairs))]
        # E128 model item 4: resample the recorded round's capability through
        # its OWN quantile band, so a strong recorded round stays strong after
        # the transfer and keeps its margin. The band is the survival interval
        # the recorded capability occupies under the unshifted chain; the same
        # quantile is then read off the shifted chain.
        recorded = min(recorded, cap_max)
        hi = base_s[recorded]
        lo = base_s[recorded + 1] if recorded < cap_max else 0.0
        quantile = lo + (hi - lo) * rng.random()
        capability = 0
        while capability < cap_max and shift_s[capability + 1] > quantile:
            capability += 1
        depth = walk(ema, margin, price)
        accepted = min(capability, depth)
        rounds += 1
        drafts += depth
        accepted_total += accepted
        if depth == 0:
            non_drafting += 1
        widths[depth + 1] += 1
        emitted += 1 + accepted
        ema = record_accept_outcome(ema, accepted, depth)
    probs = np.zeros(MAX_WIDTH)
    for width, count in widths.items():
        probs[min(width, MAX_WIDTH) - 1] += count
    probs /= probs.sum()
    return {"rounds": rounds, "edl": drafts / rounds,
            "non_drafting": non_drafting,
            "accept_rate": accepted_total / drafts if drafts else 0.0,
            "probs": probs}


def fit_delta(pairs, price, target_edl: float, seeds: int) -> dict:
    """One logit shift per (row, prompt), fitted to the published draft length.

    `non_drafting_round_count` is NOT used in the fit, so it is the held-out
    observable and its residual is the validation this rung reports.
    """
    lo, hi = -6.0, 6.0

    def mean_edl(delta):
        return float(np.mean([replay_leg(pairs, delta, price, 1000 + s)["edl"]
                              for s in range(seeds)]))

    if mean_edl(lo) > target_edl:
        best = lo
    elif mean_edl(hi) < target_edl:
        best = hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if mean_edl(mid) < target_edl:
                lo = mid
            else:
                hi = mid
        best = 0.5 * (lo + hi)
    draws = [replay_leg(pairs, best, price, 2000 + s) for s in range(seeds)]
    probs = np.mean([d["probs"] for d in draws], axis=0)
    return {"delta": best,
            "edl": float(np.mean([d["edl"] for d in draws])),
            "rounds": float(np.mean([d["rounds"] for d in draws])),
            "non_drafting": float(np.mean([d["non_drafting"] for d in draws])),
            "accept_rate": float(np.mean([d["accept_rate"] for d in draws])),
            "probs": probs}


# ----------------------------------------------------------------- solver

def solve_curve(mass: np.ndarray, y: np.ndarray, sigma: np.ndarray,
                ridge: float) -> np.ndarray:
    """Monotone least squares for `C_1..C_8`.

    `C` is reparametrised as a non-negative cumulative sum, `C_1 = t_0` and
    `C_{w+1} = C_w + t_w` with `t >= 0`, which imposes constraint 1 (a wider
    verify cannot cost less) exactly and turns the problem into NNLS.
    Convexity is NOT imposed: a spill cliff is genuinely non-convex and forcing
    convexity would assume away the answer. `ridge > 0` adds a weak penalty on
    the SECOND difference of `C` only, and the fit is reported with and
    without it.
    """
    n_w = mass.shape[1]
    tri = np.tril(np.ones((n_w, n_w)))
    design = (mass @ tri) / sigma[:, None]
    target = y / sigma
    if ridge > 0.0:
        extra = []
        for i in range(1, n_w - 1):
            row = np.zeros(n_w)
            row[i] = -ridge
            row[i + 1] = ridge
            extra.append(row)
        design = np.vstack([design, np.array(extra)])
        target = np.concatenate([target, np.zeros(len(extra))])
    theta, _ = nnls(design, target)
    return tri @ theta


def solve_curve_free_alpha(mass_by_row: list[np.ndarray],
                           y_by_row: list[np.ndarray],
                           sigma_by_row: list[np.ndarray],
                           ridge: float) -> tuple[np.ndarray, list[float]]:
    """Alternate between the shared curve and one free level per row.

    The advisor's core model asserts that `572b2cc4` and `e003a86d` share one
    alpha because they are the same tree with one changed constant. This
    variant does not assert it: it fits a level per row and reports what the
    data says the levels are. `alpha` of the first row is pinned at 1 so the
    scale stays with `C`.
    """
    alphas = [1.0] * len(mass_by_row)
    curve = None
    for _ in range(60):
        mass = np.vstack([a * m for a, m in zip(alphas, mass_by_row)])
        curve = solve_curve(mass, np.concatenate(y_by_row),
                            np.concatenate(sigma_by_row), ridge)
        for i in range(1, len(mass_by_row)):
            pred = mass_by_row[i] @ curve
            w = 1.0 / sigma_by_row[i] ** 2
            alphas[i] = float(np.sum(w * pred * y_by_row[i])
                              / np.sum(w * pred * pred))
    return curve, alphas


def fit_level_only(mass: np.ndarray, y: np.ndarray, sigma: np.ndarray,
                   curve: dict) -> dict:
    """Score one FIXED hypothesis by giving it its best single level.

    This is the well-conditioned form of the question. Instead of inverting
    eight free widths from sixteen noisy rows, it asks which of the two named
    curves explains the ranked receipts better once each is allowed its own
    scalar level. One parameter, sixteen observations.
    """
    base = np.array([curve[w + 1] for w in range(mass.shape[1])])
    pred = mass @ base
    w = 1.0 / sigma ** 2
    scale = float(np.sum(w * pred * y) / np.sum(w * pred * pred))
    resid = scale * pred - y
    return {"scale": scale,
            "chi2": float(np.sum((resid / sigma) ** 2)),
            "chi2_per_dof": float(np.sum((resid / sigma) ** 2)
                                  / max(len(y) - 1, 1)),
            "rms_resid_pct": float(np.sqrt(np.mean((resid / y) ** 2)) * 100.0),
            "resid_pct": (resid / y * 100.0).tolist()}


def admissible_set(curve: dict) -> list[int]:
    """`research/e145_r7.py:68-98`: a width is admissible exactly when its cost
    per token is a new strict running minimum."""
    out, best = [], math.inf
    for width in sorted(curve):
        per_token = curve[width] / width
        if per_token < best:
            out.append(width)
            best = per_token
    return out


# ------------------------------------------------------------------ build

def build_design(rows: dict, hists: dict, ref_rates: dict, model: str,
                 forced: dict, seeds: int) -> dict:
    out = {}
    for rid, spec in ROW_SPEC.items():
        row = rows.get(rid)
        if row is None or row["per_prompt"] is None:
            continue
        arm = PRICE_ARMS[spec["arm"]]
        cells = {}
        for prompt in ORDER:
            entry = row["per_prompt"][prompt]
            pick = pick_round_count(prompt, entry, ref_rates[prompt])
            if model == "tilt":
                probs = tilt_mass(prompt, entry, pick["R"], hists)
                sim = None
            else:
                pairs = []
                for fixture in PROMPT_FIXTURES[prompt]:
                    pairs.extend(forced[fixture])
                sim = fit_delta(pairs, arm, entry["edl"], seeds)
                probs = sim["probs"]
            dec = decode_us(entry)
            cells[prompt] = {
                "R": pick["R"], "R_exact": pick,
                "decode_us": dec, "leg_us": leg_us(entry),
                "decode_us_per_round": dec / pick["R"],
                "probs": probs.tolist(),
                "sigma_us_per_round":
                    NOISE_PCT[prompt] / 100.0 * leg_us(entry) / pick["R"],
                "sim": None if sim is None else {
                    k: v for k, v in sim.items() if k != "probs"},
            }
        out[rid] = {"arm": spec["arm"], "core": spec["core"],
                    "why": spec["why"], "score": row["score"], "cells": cells}
    return out


def stack(design: dict, ids: list[str]):
    mass, y, sigma, labels = [], [], [], []
    for rid in ids:
        for prompt in ORDER:
            cell = design[rid]["cells"][prompt]
            mass.append(cell["probs"])
            y.append(cell["decode_us_per_round"])
            sigma.append(cell["sigma_us_per_round"])
            labels.append("%s/%s" % (rid, prompt))
    return (np.array(mass), np.array(y), np.array(sigma), labels)


def curve_dict(vec: np.ndarray) -> dict:
    return {w + 1: float(vec[w]) for w in range(len(vec))}


def bootstrap(mass, y, sigma, ridge, draws, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        noisy = y + rng.normal(0.0, sigma)
        out.append(solve_curve(mass, noisy, sigma, ridge))
    return np.array(out)


def verdicts(samples: np.ndarray) -> dict:
    hits = {w: 0 for w in range(1, MAX_WIDTH + 1)}
    sets = Counter()
    for row in samples:
        adm = admissible_set(curve_dict(row))
        for width in adm:
            hits[width] += 1
        sets[tuple(adm)] += 1
    total = len(samples)
    return {"p_admissible": {w: hits[w] / total for w in hits},
            "modal_set": list(sets.most_common(1)[0][0]),
            "modal_share": sets.most_common(1)[0][1] / total}


# --------------------------------------------------------------- controls

def synthetic(mass, sigma, truth: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([truth.get(w + 1, 0.0) for w in range(mass.shape[1])])
    return mass @ base + rng.normal(0.0, sigma)


def run_control(mass, sigma, truth: dict, ridge: float, draws: int,
                seed: int) -> dict:
    truth_adm = admissible_set({w: truth[w] for w in sorted(truth)
                                if w <= MAX_WIDTH})
    fits, matches = [], 0
    for i in range(draws):
        y = synthetic(mass, sigma, truth, seed + i)
        fit = solve_curve(mass, y, sigma, ridge)
        fits.append(fit)
        if admissible_set(curve_dict(fit)) == truth_adm:
            matches += 1
    fits = np.array(fits)
    return {
        "truth_admissible": truth_adm,
        "recovery_rate_exact_set": matches / draws,
        "p_admissible": verdicts(fits)["p_admissible"],
        "modal_set": verdicts(fits)["modal_set"],
        "mean_fit": curve_dict(fits.mean(axis=0)),
        "sd_fit": {w + 1: float(fits[:, w].std(ddof=1))
                   for w in range(fits.shape[1])},
        "mean_abs_error_pct": {
            w + 1: float(np.mean(np.abs(fits[:, w] - truth.get(w + 1, 0.0)))
                         / truth.get(w + 1, 1.0) * 100.0)
            for w in range(fits.shape[1])},
    }


def h_alt_curve(c1: float) -> dict:
    out = {1: c1}
    for width, value in E145_MEASURED_LOCAL_US.items():
        out[width] = value / E145_LEVEL_TRANSFER_K
    return out


def plutarch_anchor(design: dict, rid: str, tail_curve: dict) -> dict:
    """`C_1` from the one prompt that spends 92 % of its rounds at width 1."""
    cell = design[rid]["cells"]["plutarch"]
    probs = np.array(cell["probs"])
    share = probs[0]
    tail = sum(probs[w] * tail_curve.get(w + 1, 0.0)
               for w in range(1, len(probs)))
    return {"row": rid, "share_at_width1": float(share),
            "decode_us_per_round": cell["decode_us_per_round"],
            "R": cell["R"],
            "c1": float((cell["decode_us_per_round"] - tail) / share)}


# -------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=pathlib.Path, default=BOARD)
    parser.add_argument("--json", type=pathlib.Path,
                        default=HERE / "e149-rungD.json")
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--ridge", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_rows(args.board, list(ROW_SPEC) + ["0cf1637e"])
    hists = fixture_histograms(HERE / "e128-artifacts/rung1-shipped.json")
    forced = forced_rounds(HERE / "e128-artifacts/rung1-forced.json")
    ref_rates = reference_rates(rows["572b2cc4"])

    report: dict = {
        "harness": "ranked",
        "synthetic_ground_truth_harness":
            "local (E145 R2, applegpu_g16s) expressed in ranked units by "
            "k = 2.1034; never mixed with a ranked observation",
        "max_width": MAX_WIDTH,
        "c9": "UNIDENTIFIED: SEGMENTED_VERIFY_DEPTH_CAP = 7 caps the drafted "
              "depth at 7, so M <= 8 on every round of every row",
        "reference_rates": ref_rates,
    }

    # --- exact round counts, reported as their own result
    exact = {}
    for rid in ROW_SPEC:
        if rows.get(rid) is None or rows[rid]["per_prompt"] is None:
            continue
        exact[rid] = {}
        for prompt in ORDER:
            entry = rows[rid]["per_prompt"][prompt]
            pick = pick_round_count(prompt, entry, ref_rates[prompt])
            exact[rid][prompt] = pick
    report["exact_round_counts"] = exact
    report["exact_round_counts_vs_f111_assumed"] = {
        prompt: {"exact": exact["572b2cc4"][prompt]["R"],
                 "f111_assumed": R_ASSUMED[prompt],
                 "agree": exact["572b2cc4"][prompt]["R"] == R_ASSUMED[prompt]}
        for prompt in ORDER}

    # --- the free consistency check, computed BEFORE any fit
    check = {}
    for prompt in ORDER:
        a = rows["e003a86d"]["per_prompt"][prompt]
        b = rows["7226dc9a"]["per_prompt"][prompt]
        check[prompt] = {
            "e003a86d_decode_us": decode_us(a),
            "7226dc9a_decode_us": decode_us(b),
            "delta_pct": (decode_us(b) / decode_us(a) - 1.0) * 100.0}
    deltas = [v["delta_pct"] for v in check.values()]
    report["consistency_7226dc9a_vs_e003a86d"] = {
        "per_prompt": check,
        "min_pct": min(deltas), "max_pct": max(deltas),
        "mean_pct": float(np.mean(deltas)),
        "matches": bool(max(abs(d) for d in deltas) < 0.5)}

    designs = {}
    for model in ("tilt", "replay"):
        designs[model] = build_design(rows, hists, ref_rates, model, forced,
                                      args.seeds)

    # --- the empirical per-prompt reproducibility of a ranked decode leg.
    #
    # `drama` and `travel` carry ZERO median weight and their schedules are
    # digit-frozen between the two core rows (edl 2.297619 -> 2.294821 and
    # 2.655660 -> 2.634259). Under the advisor's core model those two cells
    # must reproduce, because the mass is the same and the alpha is shared.
    # Whatever they do instead is the level reproducibility of a ranked
    # candidate decode leg, measured rather than assumed.
    frozen = {}
    for prompt in ORDER:
        a = designs["tilt"]["572b2cc4"]["cells"][prompt]
        b = designs["tilt"]["e003a86d"]["cells"][prompt]
        mass_move = float(np.max(np.abs(np.array(a["probs"])
                                        - np.array(b["probs"]))))
        frozen[prompt] = {
            "max_abs_mass_move": mass_move,
            "decode_us_per_round_572b2cc4": a["decode_us_per_round"],
            "decode_us_per_round_e003a86d": b["decode_us_per_round"],
            "delta_pct": (b["decode_us_per_round"]
                          / a["decode_us_per_round"] - 1.0) * 100.0,
            "assumed_sigma_pct": NOISE_PCT[prompt],
        }
    control_prompts = [p for p in ORDER if frozen[p]["max_abs_mass_move"] < 0.01]
    control_deltas = [frozen[p]["delta_pct"] for p in control_prompts]
    empirical_level_pct = float(np.mean(control_deltas)) if control_deltas \
        else 0.0
    empirical_spread_pct = float(np.std(control_deltas, ddof=1)) \
        if len(control_deltas) > 1 else float("nan")
    report["schedule_frozen_level_check"] = {
        "prompts": control_prompts,
        "per_prompt": frozen,
        "common_level_shift_pct": empirical_level_pct,
        "residual_spread_pct": empirical_spread_pct,
        "shared_alpha_holds": bool(abs(empirical_level_pct) < 0.5),
        "note": "the core model predicts 0.0 on both counts",
    }

    report["p_inference_residual_by_prompt"] = {
        rid: {prompt: {
            "target_edl": rows[rid]["per_prompt"][prompt]["edl"],
            "sim_edl": cell["sim"]["edl"],
            "delta": cell["sim"]["delta"],
            "published_non_drafting":
                rows[rid]["per_prompt"][prompt]["non_drafting"],
            "sim_non_drafting": cell["sim"]["non_drafting"],
            "published_R": cell["R"],
            "sim_R": cell["sim"]["rounds"],
            "held_out_non_drafting_residual":
                cell["sim"]["non_drafting"]
                - rows[rid]["per_prompt"][prompt]["non_drafting"],
        } for prompt, cell in designs["replay"][rid]["cells"].items()}
        for rid in designs["replay"]}

    core_ids = [rid for rid, spec in ROW_SPEC.items() if spec["core"]]
    results = {}
    for model in ("tilt", "replay"):
        design = designs[model]
        mass, y, sigma, labels = stack(design, core_ids)
        entry: dict = {"labels": labels,
                       "mass": mass.tolist(),
                       "y_us_per_round": y.tolist(),
                       "sigma_us_per_round": sigma.tolist(),
                       "design_condition_number":
                           float(np.linalg.cond(mass / sigma[:, None]))}
        for ridge_name, ridge in (("no_ridge", 0.0), ("weak_ridge", 5e-5)):
            fit = solve_curve(mass, y, sigma, ridge)
            samples = bootstrap(mass, y, sigma, ridge, args.draws, 7)
            fitted = curve_dict(fit)
            entry[ridge_name] = {
                "curve_us": fitted,
                "cost_per_token": {w: fitted[w] / w for w in fitted},
                "admissible": admissible_set(fitted),
                "sd_us": {w + 1: float(samples[:, w].std(ddof=1))
                          for w in range(samples.shape[1])},
                "ci95_us": {w + 1: [float(np.percentile(samples[:, w], 2.5)),
                                    float(np.percentile(samples[:, w], 97.5))]
                            for w in range(samples.shape[1])},
                "verdicts": verdicts(samples),
                "residual_us_per_round": (mass @ fit - y).tolist(),
                "chi2_per_dof": float(
                    np.sum(((mass @ fit - y) / sigma) ** 2)
                    / max(len(y) - MAX_WIDTH, 1)),
            }
        anchor_tail = {w: H_NULL[w] for w in range(2, MAX_WIDTH + 1)}
        entry["plutarch_anchor"] = {
            "h_null_tail": plutarch_anchor(design, "572b2cc4", anchor_tail),
            "h_alt_tail": plutarch_anchor(
                design, "572b2cc4",
                {w: E145_MEASURED_LOCAL_US[w] / E145_LEVEL_TRANSFER_K
                 for w in range(2, MAX_WIDTH + 1)}),
            "crown_h_null_tail": plutarch_anchor(design, "684821ed",
                                                 anchor_tail),
        }

        # Free level per row: the shared-alpha assertion, not assumed.
        mass_rows = [mass[:len(ORDER)], mass[len(ORDER):]]
        y_rows = [y[:len(ORDER)], y[len(ORDER):]]
        sig_rows = [sigma[:len(ORDER)], sigma[len(ORDER):]]
        free_fit, alphas = solve_curve_free_alpha(mass_rows, y_rows, sig_rows,
                                                  0.0)
        free_curve = curve_dict(free_fit)
        stacked = np.vstack([a * m for a, m in zip(alphas, mass_rows)])
        entry["free_alpha"] = {
            "alphas": dict(zip(core_ids, alphas)),
            "curve_us": free_curve,
            "cost_per_token": {w: free_curve[w] / w for w in free_curve},
            "admissible": admissible_set(free_curve),
            "chi2_per_dof": float(
                np.sum(((stacked @ free_fit - y) / sigma) ** 2)
                / max(len(y) - MAX_WIDTH - 1, 1)),
        }

        c1 = entry["plutarch_anchor"]["h_null_tail"]["c1"]
        truth_alt = h_alt_curve(c1)
        truth_null = {w: H_NULL[w] for w in range(1, MAX_WIDTH + 1)}

        # The well-conditioned form of the question: give each named curve its
        # own best scalar level and ask which one the ranked receipts prefer.
        null_fit = fit_level_only(mass, y, sigma, truth_null)
        alt_fit = fit_level_only(mass, y, sigma, truth_alt)
        # Each hypothesis at its OWN best level, so the statistic measures the
        # shape disagreement the receipts can see and not the level offset the
        # level fit already absorbed.
        pred_null = null_fit["scale"] * (
            mass @ np.array([truth_null[w + 1] for w in range(MAX_WIDTH)]))
        pred_alt = alt_fit["scale"] * (
            mass @ np.array([truth_alt[w + 1] for w in range(MAX_WIDTH)]))
        sep_us = float(np.sqrt(np.mean((pred_null - pred_alt) ** 2)))
        sep_pct = float(100.0 * np.sqrt(np.mean(
            ((pred_null - pred_alt) / y) ** 2)))
        entry["hypothesis_test"] = {
            "h_null": null_fit, "h_alt": alt_fit,
            "delta_chi2_null_minus_alt": null_fit["chi2"] - alt_fit["chi2"],
            "preferred": "h_alt" if alt_fit["chi2"] < null_fit["chi2"]
                         else "h_null",
            "rms_model_misfit_pct": min(null_fit["rms_resid_pct"],
                                        alt_fit["rms_resid_pct"]),
            "separation_after_own_best_level_us_per_round": sep_us,
            "separation_after_own_best_level_pct_of_y": sep_pct,
            "empirical_reproducibility_floor_pct": abs(empirical_spread_pct),
            "separation_exceeds_floor": sep_pct > abs(empirical_spread_pct),
        }

        # F6: the step marginals are the decisive cells, so report them beside
        # both reference curves rather than only the levels.
        fit_curve = entry["no_ridge"]["curve_us"]
        h_alt_ref = {w: E145_MEASURED_LOCAL_US[w] / E145_LEVEL_TRANSFER_K
                     for w in E145_MEASURED_LOCAL_US}
        steps = {}
        for w in range(1, MAX_WIDTH):
            rec = {"fitted_us": fit_curve[w + 1] - fit_curve[w],
                   "replayed_us": H_NULL[w + 1] - H_NULL[w]}
            if w in h_alt_ref and (w + 1) in h_alt_ref:
                rec["measured_over_k_us"] = h_alt_ref[w + 1] - h_alt_ref[w]
                rec["replayed_over_measured"] = (
                    rec["replayed_us"] / rec["measured_over_k_us"])
                rec["fitted_over_measured"] = (
                    rec["fitted_us"] / rec["measured_over_k_us"])
            rec["fitted_over_replayed"] = rec["fitted_us"] / rec["replayed_us"]
            steps["%d_to_%d" % (w, w + 1)] = rec
        # The 5->6 cell is the advisor's sanity check: the two reference curves
        # agree there to 0.85x, so a fit that finds real structure must land
        # near them.
        s56 = steps["5_to_6"]
        entry["step_marginals"] = steps
        entry["marginal_5_to_6_sanity"] = {
            "fitted_us": s56["fitted_us"],
            "replayed_us": s56["replayed_us"],
            "measured_over_k_us": s56["measured_over_k_us"],
            "reference_agreement_ratio":
                s56["replayed_us"] / s56["measured_over_k_us"],
            "fitted_over_replayed": s56["fitted_over_replayed"],
            "fitted_over_measured": s56["fitted_over_measured"],
            "passes": (0.5 <= s56["fitted_over_replayed"] <= 2.0
                       and 0.5 <= s56["fitted_over_measured"] <= 2.0),
        }

        entry["positive_control_h_alt"] = run_control(
            mass, sigma, truth_alt, 0.0, min(args.draws, 200), 11)
        entry["negative_control_h_null"] = run_control(
            mass, sigma, truth_null, 0.0, min(args.draws, 200), 23)

        # The controls above use the ADVISOR'S assumed noise. Repeat them at
        # the noise the receipts actually show, so the verdict is scored
        # against the instrument we have rather than the one we hoped for.
        emp_sigma = sigma * (abs(empirical_spread_pct) / 100.0) \
            / (np.array([NOISE_PCT[p] for p in ORDER] * 2) / 100.0)
        entry["empirical_sigma_us_per_round"] = emp_sigma.tolist()
        entry["positive_control_h_alt_empirical_noise"] = run_control(
            mass, emp_sigma, truth_alt, 0.0, min(args.draws, 200), 31)
        entry["negative_control_h_null_empirical_noise"] = run_control(
            mass, emp_sigma, truth_null, 0.0, min(args.draws, 200), 37)

        entry["h_alt_truth_used"] = truth_alt
        entry["h_null_truth_used"] = truth_null
        results[model] = entry

    report["results"] = results

    # ------------------------------------------------------- F6 deliverables
    # The tilt mass model is the primary; the replay model is the sensitivity.
    primary = results["tilt"]
    fit = primary["no_ridge"]
    pos_emp = primary["positive_control_h_alt_empirical_noise"]
    neg_emp = primary["negative_control_h_null_empirical_noise"]
    pos_ass = primary["positive_control_h_alt"]
    neg_ass = primary["negative_control_h_null"]
    steps = primary["step_marginals"]
    controls_pass_at_empirical_noise = (
        pos_emp["modal_set"] == pos_emp["truth_admissible"]
        and neg_emp["modal_set"] == neg_emp["truth_admissible"])
    method_adequate = bool(controls_pass_at_empirical_noise
                           and primary["marginal_5_to_6_sanity"]["passes"])
    report["f6_deliverables"] = {
        "harness": "ranked",
        "primary_mass_model": "tilt",
        "e149_rungD_ranked_curve_us": fit["curve_us"],
        "e149_rungD_ranked_curve_sd_us": fit["sd_us"],
        "e149_rungD_cost_per_token": fit["cost_per_token"],
        "e149_rungD_admissible_set": fit["admissible"],
        "e149_rungD_width6_admissible_ranked": 6 in fit["admissible"],
        "e149_rungD_width6_confidence": fit["verdicts"]["p_admissible"][6],
        "e149_rungD_width7_admissible_ranked": 7 in fit["admissible"],
        "e149_rungD_width7_confidence": fit["verdicts"]["p_admissible"][7],
        "e149_rungD_width8_admissible_ranked": 8 in fit["admissible"],
        "e149_rungD_width8_confidence": fit["verdicts"]["p_admissible"][8],
        "e149_rungD_marginal_5_to_6_us": steps["5_to_6"]["fitted_us"],
        "e149_rungD_marginal_6_to_7_us": steps["6_to_7"]["fitted_us"],
        "e149_rungD_marginal_7_to_8_us": steps["7_to_8"]["fitted_us"],
        "e149_rungD_marginal_5_to_6_sanity": primary["marginal_5_to_6_sanity"],
        "e149_rungD_rule138_holds_on_g17s":
            fit["admissible"] == [1, 2, 3, 4, 5],
        "e149_rungD_synthetic_recovery_verdict_correct": {
            "at_advisor_assumed_noise":
                pos_ass["modal_set"] == pos_ass["truth_admissible"],
            "at_empirical_noise":
                pos_emp["modal_set"] == pos_emp["truth_admissible"],
            "recovery_rate_assumed": pos_ass["recovery_rate_exact_set"],
            "recovery_rate_empirical": pos_emp["recovery_rate_exact_set"],
        },
        "e149_rungD_negative_control_returns_replayed_answer": {
            "at_advisor_assumed_noise":
                neg_ass["modal_set"] == neg_ass["truth_admissible"],
            "at_empirical_noise":
                neg_emp["modal_set"] == neg_emp["truth_admissible"],
            "recovery_rate_assumed": neg_ass["recovery_rate_exact_set"],
            "recovery_rate_empirical": neg_emp["recovery_rate_exact_set"],
            "modal_set_at_empirical_noise": neg_emp["modal_set"],
        },
        "e149_rungD_c1_plutarch_anchor_us":
            primary["plutarch_anchor"]["h_null_tail"]["c1"],
        "e149_rungD_c1_plutarch_anchor_sensitivity_us": {
            "h_null_drafting_tail":
                primary["plutarch_anchor"]["h_null_tail"]["c1"],
            "h_alt_drafting_tail":
                primary["plutarch_anchor"]["h_alt_tail"]["c1"],
            "crown_row_h_null_tail":
                primary["plutarch_anchor"]["crown_h_null_tail"]["c1"],
        },
        "e149_rungD_7226dc9a_matches_e003a86d_decode":
            report["consistency_7226dc9a_vs_e003a86d"]["matches"],
        "e149_rungD_7226dc9a_decode_gap_pct":
            report["consistency_7226dc9a_vs_e003a86d"]["mean_pct"],
        "e149_rungD_p_inference_residual_by_prompt":
            report.get("p_inference_residual_by_prompt"),
        "e149_rungD_method_adequate": method_adequate,
        "e149_rungD_stop_rule_fired": not method_adequate,
        "e149_rungD_stop_rule_reason": (
            None if method_adequate else
            "the negative control does not return the replayed answer at the "
            "empirical per-prompt reproducibility floor, and the fitted 5->6 "
            "marginal misses both reference curves at a cell where the two "
            "references agree to 0.85x"),
        "e149_rungD_empirical_reproducibility_floor_pct":
            abs(report["schedule_frozen_level_check"]["residual_spread_pct"]),
        "e149_rungD_sensitivity_replay_model": {
            "admissible": results["replay"]["no_ridge"]["admissible"],
            "chi2_per_dof": results["replay"]["no_ridge"]["chi2_per_dof"],
            "marginal_5_to_6_us":
                results["replay"]["step_marginals"]["5_to_6"]["fitted_us"],
            "marginal_6_to_7_us":
                results["replay"]["step_marginals"]["6_to_7"]["fitted_us"],
            "marginal_7_to_8_us":
                results["replay"]["step_marginals"]["7_to_8"]["fitted_us"],
        },
    }

    report["rows_used"] = {rid: ROW_SPEC[rid]["why"] for rid in core_ids}
    report["rows_excluded_with_reason"] = {
        rid: ROW_SPEC[rid]["why"] for rid in ROW_SPEC if rid not in core_ids}
    report["rows_excluded_with_reason"]["0cf1637e"] = (
        "still validating at the fetch time; officialMetrics.per_prompt is "
        "empty, so it carries no observation")

    args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("wrote %s" % args.json)

    for model in ("tilt", "replay"):
        entry = results[model]
        print("\n=== mass model %s  cond=%.3g" %
              (model, entry["design_condition_number"]))
        fit = entry["no_ridge"]
        print("  w   C_w us      sd      C_w/w      P(admissible)")
        for width in range(1, MAX_WIDTH + 1):
            print("  %d %10.1f %8.1f %10.1f     %.3f" % (
                width, fit["curve_us"][width], fit["sd_us"][width],
                fit["cost_per_token"][width],
                fit["verdicts"]["p_admissible"][width]))
        print("  admissible=%s  modal=%s (%.2f)  chi2/dof=%.2f" % (
            fit["admissible"], fit["verdicts"]["modal_set"],
            fit["verdicts"]["modal_share"], fit["chi2_per_dof"]))
        free = entry["free_alpha"]
        print("  free-alpha alphas=%s chi2/dof=%.2f admissible=%s" % (
            {k: round(v, 5) for k, v in free["alphas"].items()},
            free["chi2_per_dof"], free["admissible"]))
        test = entry["hypothesis_test"]
        print("  hypothesis test: H-null chi2=%.1f rms=%.3f%% | "
              "H-alt chi2=%.1f rms=%.3f%% | prefers %s (dchi2=%.1f)" % (
                  test["h_null"]["chi2"], test["h_null"]["rms_resid_pct"],
                  test["h_alt"]["chi2"], test["h_alt"]["rms_resid_pct"],
                  test["preferred"], test["delta_chi2_null_minus_alt"]))
        print("  H-null/H-alt separation after each takes its own best level "
              "= %.1f us/round = %.3f %% of y; floor %.3f %%; separable=%s; "
              "model misfit rms = %.3f %%" % (
                  test["separation_after_own_best_level_us_per_round"],
                  test["separation_after_own_best_level_pct_of_y"],
                  test["empirical_reproducibility_floor_pct"],
                  test["separation_exceeds_floor"],
                  test["rms_model_misfit_pct"]))
        for tag in ("", "_empirical_noise"):
            pos = entry["positive_control_h_alt" + tag]
            neg = entry["negative_control_h_null" + tag]
            label = "assumed" if not tag else "EMPIRICAL"
            print("  [%s noise] POSITIVE (truth %s) recovery=%.3f modal=%s | "
                  "NEGATIVE (truth %s) recovery=%.3f modal=%s" % (
                      label, pos["truth_admissible"],
                      pos["recovery_rate_exact_set"], pos["modal_set"],
                      neg["truth_admissible"],
                      neg["recovery_rate_exact_set"], neg["modal_set"]))
    frozen = report["schedule_frozen_level_check"]
    print("\n=== schedule-frozen level check on %s: common shift %+.4f %%, "
          "spread %.4f %% (core model predicts 0.0)" % (
              frozen["prompts"], frozen["common_level_shift_pct"],
              frozen["residual_spread_pct"]))

    dlv = report["f6_deliverables"]
    print("\n=== F6 step marginals (harness=ranked, us/round)")
    print("  step    fitted    replayed    meas/k   fit/repl   fit/meas")
    for name, rec in results["tilt"]["step_marginals"].items():
        mk = rec.get("measured_over_k_us")
        print("  %-7s %9.1f %11.1f %9s %10.2f %10s" % (
            name.replace("_to_", "->").replace("_", ""), rec["fitted_us"],
            rec["replayed_us"], "-" if mk is None else "%.1f" % mk,
            rec["fitted_over_replayed"],
            "-" if mk is None else "%.2f" % rec["fitted_over_measured"]))
    print("\n=== F6 verdicts (harness=ranked)")
    for key in ("e149_rungD_width6_admissible_ranked",
                "e149_rungD_width7_admissible_ranked",
                "e149_rungD_width8_admissible_ranked",
                "e149_rungD_rule138_holds_on_g17s",
                "e149_rungD_7226dc9a_matches_e003a86d_decode",
                "e149_rungD_method_adequate",
                "e149_rungD_stop_rule_fired"):
        print("  %-52s %s" % (key, dlv[key]))
    print("  %-52s %.1f" % ("e149_rungD_c1_plutarch_anchor_us",
                            dlv["e149_rungD_c1_plutarch_anchor_us"]))
    print("  %-52s %.4f" % ("empirical reproducibility floor pct",
                            dlv["e149_rungD_empirical_reproducibility_floor_pct"]))
    print("  5->6 sanity: %s" % dlv["e149_rungD_marginal_5_to_6_sanity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
