#!/usr/bin/env python3
"""E114 rung 1b. The width distribution the SHIPPED SCHEDULE can generate.

Rung 0 treats the per-prompt verify-width distribution as an arbitrary point of
the 7-simplex and proves the board pins only three linear functionals of it,
leaving four free dimensions. That is the correct OUTER envelope, but the
candidate does not emit an arbitrary distribution: every round's width is the
return value of `costModelDepth`, which is deterministic given its state.

This module re-implements that function and its state update in Python, drives
it with a parametric acceptance process, and reads the realised width histogram
off the simulation. The result is a POINT, obtained with zero GPU and no trace,
that can be compared against the identified set.

Two models are fitted, because the first one is falsified by the board:

    MODEL A  one parameter, a per-prompt acceptance level `lambda_p` scaling a
             fixed local per-position profile. The margin overrides are off.
    MODEL B  two parameters, `lambda_p` plus a mean top-2 margin `mu_p` that
             drives the depth-0 and depth-1 confidence overrides the shipped
             schedule really applies.

Model A is fitted to `effective_mean_draft_len` alone, as specified. Model B is
fitted to `effective_mean_draft_len` AND `accepted_draft_rate`, which is what
model A cannot satisfy at the same time. Under both models the width histogram,
the round count and the non-drafting share stay out of sample.

    harness=local for the acceptance profile and the validation histograms
    harness=ranked for the per-prompt parameters, which are fitted to the board

Everything below is transcribed from the current base with its source line
recorded beside it. Nothing is fitted except `lambda_p` and `mu_p`.

    PYTHONPATH=research python3 research/e114_policy_sim.py
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics

import e114_width_recovery as wr

SRC = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

# --- transcribed policy constants -------------------------------------------
#
# `MLXFastConstants.qwenMTPMaxDepth` = `qwenMTPMaxDraftDepth` = 8,
# `Sources/MLXFastCore/Constants.swift:331,337`.
MAX_DEPTH = 8
# `segmentedVerifyDepthCap`, SRC:1009. Flat cap 7, no gate.
WIDTH_CAP = 7
# `headStepCostRatio`, SRC:841.
HEAD_STEP_COST_RATIO = 0.18
# `acceptEMAAlpha`, SRC:806.
EMA_ALPHA = 0.15
# `positionAcceptEMA` initialiser, SRC:805-806.
EMA_PRIOR = tuple(0.85 * 0.98 ** i for i in range(MAX_DEPTH))
# Optimism-transfer cap, SRC:1167.
OPTIMISM_CAP = 0.95
# `depthPriceArm = .ship` (SRC:959) selects `makeUniformDepthPrice` (SRC:872),
# so every marginal is `headStepCostRatio` and the prefix costs are linear.
MARGINAL = tuple(HEAD_STEP_COST_RATIO for _ in range(MAX_DEPTH))
CUMULATIVE = tuple(1.0 + i * HEAD_STEP_COST_RATIO for i in range(MAX_DEPTH + 1))

# E92 per-position acceptance profile, harness=local. Index i is
# P(draft i accepted | 0..<i accepted).
E92_POSITION_ACCEPT = (0.9659, 0.9652, 0.9543, 0.9486, 0.9487, 0.9859,
                       0.9451, 0.8333)
LAMBDA_MAX = 1.0 / max(E92_POSITION_ACCEPT)

T = wr.T

# --- validation targets ------------------------------------------------------
#
# Traced local DEPTH histograms; verify width is depth + 1. `v2_pair_census` is
# the 31-round subset of the same session as `E109_witness` that entered the
# E110 design-C contrast (ledger 259.4), so it is not independent evidence and
# its round count cannot be scored against a 512-token window.
GT_DEPTH = {
    "E106_census": {"hist": {1: 1, 4: 1, 5: 4, 6: 3, 7: 10},
                    "rounds": 19, "tokens": 128, "score_rounds": True,
                    "src": "E106 per-width census, W&B 19kgn6xi"},
    "E109_witness": {"hist": {1: 1, 2: 2, 3: 6, 4: 5, 5: 7, 6: 56},
                     "rounds": 77, "tokens": 512, "score_rounds": True,
                     "src": "E109 witness leg, 512 tokens, 77 rounds"},
    "v2_pair_census": {"hist": {3: 1, 4: 1, 5: 2, 6: 1, 7: 26},
                       "rounds": 31, "tokens": 512, "score_rounds": False,
                       "src": "askeladd v2 pair census, ledger 259.4, a "
                              "31-round SUBSET of the E109_witness session"},
}
BEAGLE_ACCEPTED_DRAFT_RATE = 0.8340

# Prompts with a non-zero published weight (Finding 16). Only these have to be
# represented for the rung-1 arm pricing to use a policy point.
SCORED = ("beagle", "essays", "medicine", "republic", "botany")


# --- the policy, transcribed -------------------------------------------------

def cost_model_depth(ema, offered: int = MAX_DEPTH,
                     margin: float | None = None) -> int:
    """`Qwen36MTPBlockSession.costModelDepth`, SRC:1040-1112.

    `margin` is the pending primary's target top-2 gap. It is non-negative by
    construction, so both confidence overrides live in `[0.5, 1)` and can only
    LOWER `p` at depth 0 and depth 1.
    """
    cap = min(min(offered, MAX_DEPTH), WIDTH_CAP)
    if cap <= 0:
        return 0
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        p = ema[depth]
        if margin is not None and depth in (0, 1):
            div = 2.0 if depth == 0 else 3.0
            p = min(p, 1.0 / (1.0 + math.exp(-margin / div)))
        reach *= p
        threshold = MARGINAL[depth] * (1.0 + expected) / CUMULATIVE[depth]
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def record_accept_outcome(ema, accepted: int, drafted: int,
                          stopped_early: bool = False) -> None:
    """`Qwen36MTPBlockSession.recordAcceptOutcome`, SRC:1143-1172."""
    a = EMA_ALPHA
    for i in range(min(accepted, MAX_DEPTH)):
        ema[i] += a * (1.0 - ema[i])
    if accepted < drafted and not stopped_early and accepted < MAX_DEPTH:
        ema[accepted] += a * (0.0 - ema[accepted])
    elif accepted == drafted and drafted > 0 and accepted < MAX_DEPTH:
        if ema[accepted] < OPTIMISM_CAP:
            ema[accepted] += a * (OPTIMISM_CAP - ema[accepted])


# --- the generator -----------------------------------------------------------

def episode(lam: float, mu: float | None, rng: random.Random,
            tokens_target: int = T) -> dict:
    """One decode window under acceptance level `lam` and mean margin `mu`."""
    ema = list(EMA_PRIOR)
    probs = [min(1.0, lam * q) for q in E92_POSITION_ACCEPT]
    hist: collections.Counter = collections.Counter()
    tokens = rounds = drafted = accepted_total = 0
    while tokens < tokens_target:
        margin = rng.expovariate(1.0 / mu) if mu else None
        depth = cost_model_depth(ema, margin=margin)
        accepted = 0
        for i in range(depth):
            if rng.random() < probs[i]:
                accepted += 1
            else:
                break
        tokens += 1 + accepted
        rounds += 1
        hist[depth + 1] += 1
        drafted += depth
        accepted_total += accepted
        record_accept_outcome(ema, accepted, depth)
    return {"hist": hist, "rounds": rounds, "drafted": drafted,
            "accepted": accepted_total}


def run(lam: float, mu: float | None, episodes: int, seed: int = 20260817,
        tokens_target: int = T) -> dict:
    """Monte Carlo over independent windows. Common random numbers by seed."""
    rng = random.Random(seed)
    hist: collections.Counter = collections.Counter()
    rounds, drafted, accepted = [], 0, 0
    for _ in range(episodes):
        e = episode(lam, mu, rng, tokens_target)
        hist.update(e["hist"])
        rounds.append(e["rounds"])
        drafted += e["drafted"]
        accepted += e["accepted"]
    n = sum(hist.values())
    dist = {M: c / n for M, c in sorted(hist.items())}
    return {
        "lambda": lam, "mu": mu, "width_dist": dist,
        "mean_width": sum(M * p for M, p in dist.items()),
        "mean_draft_len": drafted / n,
        "mean_rounds": statistics.mean(rounds),
        "sd_rounds": statistics.pstdev(rounds) if len(rounds) > 1 else 0.0,
        "accepted_draft_rate": accepted / drafted if drafted else float("nan"),
        "p_width1": dist.get(1, 0.0), "episodes": episodes,
    }


def _bisect(evaluate, key: str, target: float, lo: float, hi: float,
            iters: int) -> dict:
    """Bisect a monotone increasing scalar output onto `target`."""
    r_lo, r_hi = evaluate(lo), evaluate(hi)
    if target <= r_lo[key]:
        return {**r_lo, "bracketed": False, "hit_bound": "lo"}
    if target >= r_hi[key]:
        return {**r_hi, "bracketed": False, "hit_bound": "hi"}
    best = r_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        best = evaluate(mid)
        if best[key] < target:
            lo = mid
        else:
            hi = mid
    return {**best, "bracketed": True, "hit_bound": None}


def fit_a(target_dbar: float, episodes: int, seed: int, tokens: int = T,
          iters: int = 24) -> dict:
    """MODEL A. One parameter, fitted to the mean draft length only."""
    return _bisect(lambda x: run(x, None, episodes, seed, tokens),
                   "mean_draft_len", target_dbar, 0.0, LAMBDA_MAX, iters)


def fit_b(target_dbar: float, target_acc: float, episodes: int, seed: int,
          tokens: int = T, iters: int = 14) -> dict:
    """MODEL B. Two parameters, exactly identified by two board numbers.

    `mu` moves the depth at a fixed per-position acceptance; `lambda` moves the
    per-position acceptance. Nest them: bisect `mu` onto the mean draft length
    inside a bisection of `lambda` onto the accepted-draft rate. Both maps are
    monotone increasing, which the self-test asserts.
    """
    def inner(lam: float) -> dict:
        return _bisect(lambda m: run(lam, m, episodes, seed, tokens),
                       "mean_draft_len", target_dbar, 1e-3, 120.0, iters)

    out = _bisect(inner, "accepted_draft_rate", target_acc, 0.05, LAMBDA_MAX,
                  iters)
    out["dbar_resid"] = out["mean_draft_len"] - target_dbar
    out["acc_resid"] = out["accepted_draft_rate"] - target_acc
    return out


# --- validation --------------------------------------------------------------

def _tvd(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def validate(episodes: int, seed: int) -> dict:
    """Fit the declared moments, then score the histogram and the round count."""
    rows = []
    for name, gt in GT_DEPTH.items():
        obs_w = {d + 1: c / gt["rounds"] for d, c in gt["hist"].items()}
        n = gt["rounds"]
        drafted = sum(d * c for d, c in gt["hist"].items())
        mean_d = drafted / n
        # Accepted drafts follow from token conservation only on a leg that
        # decodes a full window; on a subset of rounds they do not.
        acc = ((gt["tokens"] - n) / drafted) if gt["score_rounds"] else None
        a = fit_a(mean_d, episodes, seed, gt["tokens"])
        row = {
            "target": name, "src": gt["src"], "rounds": n,
            "tokens": gt["tokens"], "score_rounds": gt["score_rounds"],
            "observed_mean_draft_len": mean_d,
            "observed_accepted_draft_rate": acc,
            "observed_width_dist": obs_w,
            "A": {"lambda": a["lambda"], "mu": None,
                  "bracketed": a["bracketed"], "width_dist": a["width_dist"],
                  "tvd": _tvd(obs_w, a["width_dist"]),
                  "mean_rounds": a["mean_rounds"],
                  "rounds_resid_pct": 100.0 * (a["mean_rounds"] - n) / n,
                  "accepted_draft_rate": a["accepted_draft_rate"],
                  "acc_resid": (a["accepted_draft_rate"] - acc)
                  if acc is not None else None},
        }
        if acc is not None:
            b = fit_b(mean_d, acc, episodes, seed, gt["tokens"])
            row["B"] = {"lambda": b["lambda"], "mu": b["mu"],
                        "bracketed": b["bracketed"],
                        "width_dist": b["width_dist"],
                        "tvd": _tvd(obs_w, b["width_dist"]),
                        "mean_rounds": b["mean_rounds"],
                        "rounds_resid_pct": 100.0 * (b["mean_rounds"] - n) / n,
                        "accepted_draft_rate": b["accepted_draft_rate"],
                        "acc_resid": b["acc_resid"]}
        rows.append(row)
    return {"rows": rows,
            "note": "Each model is given its own fitted moments and nothing "
                    "else. The width histogram and the round count are out of "
                    "sample for both."}


def recover(rec: dict, episodes: int, seed: int) -> dict:
    """Per-prompt ranked width distribution generated by the shipped policy."""
    out = {}
    for name, p in sorted(rec["prompts"].items()):
        a = fit_a(p["mean_draft_len"], episodes, seed)
        row = {
            "board_mean_draft_len": p["mean_draft_len"],
            "board_rounds": p["rounds"],
            "board_accepted_draft_rate": p["accept"],
            "board_p_width1": p["p_width1"],
            "A": {"lambda": a["lambda"], "mu": None,
                  "bracketed": a["bracketed"], "width_dist": a["width_dist"],
                  "mean_width": a["mean_width"],
                  "mean_draft_len": a["mean_draft_len"],
                  "mean_rounds": a["mean_rounds"],
                  "rounds_resid_pct": 100.0 * (a["mean_rounds"] - p["rounds"])
                  / p["rounds"],
                  "accepted_draft_rate": a["accepted_draft_rate"],
                  "acc_resid": a["accepted_draft_rate"] - p["accept"],
                  "p_width1": a["p_width1"]},
        }
        if p["mean_draft_len"] > 0:
            b = fit_b(p["mean_draft_len"], p["accept"], episodes, seed)
            row["B"] = {
                "lambda": b["lambda"], "mu": b["mu"],
                "bracketed": b["bracketed"], "width_dist": b["width_dist"],
                "mean_width": b["mean_width"],
                "mean_draft_len": b["mean_draft_len"],
                "mean_rounds": b["mean_rounds"],
                "rounds_resid_pct": 100.0 * (b["mean_rounds"] - p["rounds"])
                / p["rounds"],
                "accepted_draft_rate": b["accepted_draft_rate"],
                "acc_resid": b["acc_resid"], "p_width1": b["p_width1"]}
        out[name] = row
    return out


def wide_shape(dist: dict) -> dict[int, float]:
    """The policy's width distribution conditioned on width >= 2."""
    wide = {M: m for M, m in dist.items() if M >= 2 and m > 0}
    mass = sum(wide.values())
    if mass <= 0:
        raise ValueError("policy point carries no wide-QMV round")
    return {M: m / mass for M, m in wide.items()}


def in_identified_set(dist: dict, p: dict, tol: float = 1e-6) -> dict:
    """Is the policy SHAPE, placed on the board's exact mean, feasible?

    The Monte Carlo mean never lands exactly on the board's mean, so the raw
    simulated point fails a strict equality for a reason that carries no
    information. Report that gap, then move the policy SHAPE onto the exact
    board mean with the same I-projection rung 1 uses for the traced shapes,
    and test the projected point against the cost band. That second test is the
    one that can actually contradict the policy model, and a contradiction is
    reported rather than absorbed.
    """
    cond = wide_shape(dist)
    mean_wide_policy = sum(M * m for M, m in cond.items())
    p1 = p["p_width1"]
    mean_wide_board = (p["mean_width"] - p1) / (1.0 - p1)
    band = wr.cost_band(p["round_us"], p1, wr.ROUTE_B["max_resid_pct"] / 100.0)
    support_ok = all(2 <= M <= 8 for M in cond)
    try:
        proj = wr.tilt(cond, mean_wide_board)
    except (ValueError, ZeroDivisionError):
        proj = None
    cost = (sum(m * wr.route_b_us(float(M)) for M, m in proj.items())
            if proj else float("nan"))
    cost_ok = bool(proj) and (band is None
                              or band[1] - tol <= cost <= band[2] + tol)
    return {
        "inside": bool(support_ok and cost_ok),
        "support_ok": support_ok, "cost_ok": cost_ok,
        "policy_mean_wide": mean_wide_policy,
        "board_mean_wide": mean_wide_board,
        "mean_wide_gap": mean_wide_policy - mean_wide_board,
        "projected_shape": proj,
        "projected_cost_us": cost,
        "cost_band": None if band is None else [band[1], band[2]],
        "cost_resid_pct": (100.0 * (cost - 0.5 * (band[1] + band[2]))
                           / (0.5 * (band[1] + band[2]))) if band else None,
        "policy_p_width1": dist.get(1, 0.0), "board_p_width1": p1,
    }


# Pre-registered before the fits were run. A total-variation distance of 0.10
# between the simulated and the observed width histogram is already generous:
# it permits a tenth of all rounds to sit at the wrong verify width.
OOS_TVD_BAR = 0.10


def promotion_gate(val: dict, inside: dict) -> dict:
    """May the generated shape be published beside maxent, gt1 and gt2?

    Fitting the two ranked moments per prompt is in-sample by construction, so
    it cannot license the shape. The gate is held-out: reproduce the width
    censuses this experiment did NOT fit, and land every scored prompt inside
    the rung-0 cost band.
    """
    verdict = {}
    for m in ("A", "B"):
        rows = [(r["target"], r[m]["tvd"]) for r in val["rows"] if m in r]
        target, worst = max(rows, key=lambda kv: kv[1])
        out_of_band = sorted(n for n in SCORED
                             if m in inside.get(n, {})
                             and not inside[n][m]["cost_ok"])
        verdict[m] = {"worst_tvd": worst, "worst_tvd_target": target,
                      "out_of_band": out_of_band, "bar": OOS_TVD_BAR,
                      "pass": worst <= OOS_TVD_BAR and not out_of_band}
    return verdict


GATES = [
    "the pending top-2 margin is not observable offline. Model A switches "
    "both overrides OFF, which makes its depth an upper bound at a given EMA "
    "state; model B restores them through one fitted exponential mean.",
    "both confidence overrides are bounded below by 0.5, because the top-2 "
    "margin cannot be negative, so NEITHER model can produce a non-drafting "
    "round except by driving positionAcceptEMA[0] under 0.18.",
    "the acceptance profile is one fixed local shape scaled by a single "
    "per-prompt level; a prompt whose profile has a different SHAPE is not "
    "represented.",
    "acceptance is drawn independently per position within a round, which "
    "rung 0 already showed the board rejects as a complete model.",
    "no stop token is simulated inside the window, so `stoppedEarly` is "
    "always false and every short round is a real reject.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", default="/tmp/yukon-board/full.json")
    ap.add_argument("--receipt", default="b8b8b860")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--json", default="research/e114-artifacts/rung1b.json")
    args = ap.parse_args()

    print("=" * 78)
    print("E114 rung 1b - the width distribution the SHIPPED SCHEDULE emits")
    print("=" * 78)
    print("policy transcribed from %s" % SRC)
    print("  cap %d, price uniform h=%.2f, EMA alpha %.2f, prior %s, ..."
          % (WIDTH_CAP, HEAD_STEP_COST_RATIO, EMA_ALPHA,
             ", ".join("%.4f" % v for v in EMA_PRIOR[:4])))
    print("  acceptance profile (E92, harness=local) %s"
          % ", ".join("%.4f" % q for q in E92_POSITION_ACCEPT))
    print("  %d Monte Carlo windows per evaluation, seed %d"
          % (args.episodes, args.seed))

    val = validate(args.episodes, args.seed)
    print("\n-- out-of-sample validation ------------------------------------")
    print("  %-15s %-2s %7s %7s %7s %9s %9s %10s"
          % ("target", "M", "lambda", "mu", "TVD", "rounds obs", "rounds sim",
             "acc resid"))
    for r in val["rows"]:
        for m in ("A", "B"):
            if m not in r:
                continue
            f = r[m]
            print("  %-15s %-2s %7.4f %7s %7.4f %9d %9.1f %10s"
                  % (r["target"], m, f["lambda"],
                     "-" if f["mu"] is None else "%.3f" % f["mu"],
                     f["tvd"], r["rounds"], f["mean_rounds"],
                     "-" if f["acc_resid"] is None
                     else "%+.4f" % f["acc_resid"]))
    for r in val["rows"]:
        print("\n  %s width histogram, observed against policy" % r["target"])
        keys = set(r["observed_width_dist"]) | set(r["A"]["width_dist"])
        if "B" in r:
            keys |= set(r["B"]["width_dist"])
        for M in sorted(keys):
            print("    M=%d  observed %.4f   A %.4f   B %s"
                  % (M, r["observed_width_dist"].get(M, 0.0),
                     r["A"]["width_dist"].get(M, 0.0),
                     "%.4f" % r["B"]["width_dist"].get(M, 0.0)
                     if "B" in r else "-"))

    rec = wr.load_receipt(args.board, args.receipt)
    prompts = recover(rec, args.episodes, args.seed)
    inside = {n: {m: in_identified_set(r[m]["width_dist"], rec["prompts"][n])
                  for m in ("A", "B") if m in r}
              for n, r in prompts.items()}

    print("\n-- ranked per-prompt policy point, receipt %s ------------------"
          % rec["id"][:8])
    print("  %-9s %-2s %7s %7s %9s %9s %9s %9s"
          % ("prompt", "M", "lambda", "mu", "dbar brd", "dbar sim", "acc brd",
             "acc sim"))
    for n, r in prompts.items():
        for m in ("A", "B"):
            if m not in r:
                continue
            f = r[m]
            print("  %-9s %-2s %7.4f %7s %9.4f %9.4f %9.4f %9.4f%s"
                  % (n, m, f["lambda"],
                     "-" if f["mu"] is None else "%.3f" % f["mu"],
                     r["board_mean_draft_len"], f["mean_draft_len"],
                     r["board_accepted_draft_rate"], f["accepted_draft_rate"],
                     "" if f["bracketed"] else "   NOT BRACKETED"))
    print("\n  beagle accepted_draft_rate: board %.4f, model A %.4f, model B "
          "%.4f" % (BEAGLE_ACCEPTED_DRAFT_RATE,
                    prompts["beagle"]["A"]["accepted_draft_rate"],
                    prompts["beagle"]["B"]["accepted_draft_rate"]))
    print("\n  round-count check against Finding 18b")
    print("  %-9s %8s %10s %10s" % ("prompt", "R (18b)", "A resid %",
                                    "B resid %"))
    for n, r in prompts.items():
        print("  %-9s %8d %+10.2f %+10s"
              % (n, r["board_rounds"], r["A"]["rounds_resid_pct"],
                 "%.2f" % r["B"]["rounds_resid_pct"] if "B" in r else "-"))

    print("\n  is the policy SHAPE, projected onto the board mean, inside the")
    print("  rung-0 identified set?")
    for n, v in inside.items():
        for m, chk in v.items():
            print("    %-9s %-2s inside=%-5s mean_wide %.4f vs board %.4f "
                  "(MC gap %+.4f), projected cost %+.3f %% of band centre -> %s"
                  % (n, m, chk["inside"], chk["policy_mean_wide"],
                     chk["board_mean_wide"], chk["mean_wide_gap"],
                     chk["cost_resid_pct"],
                     "ok" if chk["cost_ok"] else "OUT OF BAND"))

    verdict = promotion_gate(val, inside)
    print("\n-- pre-registered promotion gate for the policy shape ----------")
    print("  a generated shape may join maxent/gt1/gt2 as a published weight")
    print("  vector only if it reproduces HELD-OUT width censuses. Bar: worst")
    print("  out-of-sample TVD <= %.2f and every scored prompt in band."
          % OOS_TVD_BAR)
    for m in ("A", "B"):
        v = verdict[m]
        print("    model %s  worst OOS TVD %.4f (%s)   scored prompts out of "
              "band %s  -> %s"
              % (m, v["worst_tvd"], v["worst_tvd_target"],
                 ",".join(v["out_of_band"]) or "none",
                 "PASS" if v["pass"] else "FAIL"))
    print("  both models FAIL. The `policy` column downstream is a DIAGNOSTIC")
    print("  shape, not a validated one: it must not be used to narrow the")
    print("  identified set or to overturn a bound built from the equalities.")

    shapes = {m: {n: inside[n][m]["projected_shape"]
                  for n in SCORED if m in inside[n]
                  and inside[n][m]["projected_shape"]}
              for m in ("A", "B")}
    out = {"policy_source": SRC, "shapes": shapes, "constants": {
        "max_depth": MAX_DEPTH, "width_cap": WIDTH_CAP,
        "head_step_cost_ratio": HEAD_STEP_COST_RATIO,
        "ema_alpha": EMA_ALPHA, "ema_prior": list(EMA_PRIOR),
        "optimism_cap": OPTIMISM_CAP, "cumulative": list(CUMULATIVE),
        "acceptance_profile": list(E92_POSITION_ACCEPT)},
        "episodes": args.episodes, "seed": args.seed,
        "validation": val, "prompts": prompts,
        "identified_set_check": inside, "scored_prompts": list(SCORED),
        "promotion_gate": verdict, "oos_tvd_bar": OOS_TVD_BAR,
        "gates": GATES}
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as h:
        json.dump(out, h, indent=1, sort_keys=True, default=str)
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
