#!/usr/bin/env python3
"""E128 sections 3 and 4 -- hypothesis J, and the SIGN of the level correction.

Zero GPU. Both sections read traces that already exist.

Section 3 -- hypothesis J
-------------------------
The shipped walk predicts a round's accepted count as

    expected = sum_{k=1..d} prod_{j<k} p_j

where `p_j` is a per-position acceptance EMA, overridden downward by the
pending primary's top-2 margin at j = 0 and j = 1. E128's F1 answer measured
that `expected` under-predicts the realised accepted count by 9 to 24 percent
on eleven of twelve fixtures, with slope near one.

Hypothesis J says the cause is Jensen / positive association: rounds are
heterogeneous, the realised acceptance is `E_round[prod_j p_j(round)]`, and a
product of position-wise averages is a lower bound on it.

This module does not argue the point. It splits the measured bias into three
terms that sum to it EXACTLY, so the data decides which one carries it:

    measured_bias = mean(expected) - mean(accepted)
                  = margin_component        (walk p vs the round's own EMA)
                  + ema_component           (EMA vs the true uncensored chain)
                  + selection_component     (chain vs realised, at the round's
                                             own depth = the Jensen / FKG term)

`selection_component` is the only term hypothesis J can own. It is measured as
`mean_r E[min(C, d_r)] - mean_r min(c_r, d_r)`, where `C` is drawn from the
fixture's UNCENSORED capability distribution and `c_r` is the round's realised
capability. It is negative exactly when the scheduler drafts deeper on the
rounds that were going to accept more, which is what positive association
between a round's positions means operationally.

Hypothesis J's own quantitative prediction is then formed independently, by
direct resampling over margin bins, and regressed against the measured bias
across the twelve fixtures. That regression is the test.

Section 4 -- the sign of the correction
---------------------------------------
The walk consumes the same biased estimate twice, with opposite effect:

    reach     -> the guard `reach > threshold`. Biased low => break earlier
                 => SHALLOWER. Correcting it makes rounds deeper.
    expected  -> `threshold = marginal[d] * (1 + expected) / cumulative[d]`.
                 Biased low => threshold too small => guard too easy
                 => DEEPER. Correcting it makes rounds shallower.

At `expected ~ 3` the two are the same order of magnitude, so the net sign is
an empirical question. This module replays every recorded round three ways --
`reachonly`, `expectedonly`, `levelfix` (both) -- and reports the depth change
and the ranked-priced effect of each. The realised accepted count is imputed
from the uncensored capability distribution, so a deeper round is charged for
its extra work and credited only with the tokens it would really have won.

The replay is MYOPIC: it holds each round's recorded EMA fixed and does not
propagate the counterfactual outcome into later rounds. That is deliberate for
a sign test, because it removes the simulator from the causal chain. The full
feedback path is priced by `research/e128_price.py`.

  usage: research/e128_jensen.py --shipped DIR [DIR ...] \
                                 --forced DIR [DIR ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from e128_price import MAX_DEPTH, ranked_round_us
from e128_replay import (
    EMA_PRIOR,
    SEGMENTED_VERIFY_DEPTH_CAP,
    cost_model_depth,
    read_meta,
    read_rounds,
)

DECODE_TOKENS = 512
MARGIN_BINS = 4
GAMMA_GRID = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45]


# ------------------------------------------------------------------ utilities

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def pearson(xs: list[float], ys: list[float]) -> tuple:
    """Returns (r, slope, intercept) of `ys` regressed on `xs`."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = sxy / sxx
    return sxy / math.sqrt(sxx * syy), slope, my - slope * mx


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) \
                    and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    return pearson(ranks(xs), ranks(ys))[0]


def conditional_chain(capabilities: list[int], depths: list[int]) -> list[float]:
    """`q_j = P(accept at j | reached j)` with a Jeffreys prior.

    A round with `c < d` pins its capability exactly. A round with `c == d`
    is right censored at `d`: it reached every position below `d` and accepted
    all of them, and it says nothing about positions at or beyond `d`.
    """
    q = []
    for j in range(MAX_DEPTH):
        reached = accepted = 0
        for c, d in zip(capabilities, depths):
            if d <= j:
                continue
            reached += 1
            accepted += 1 if c > j else 0
        q.append((accepted + 0.5) / (reached + 1.0) if reached else
                 (q[-1] if q else 0.5))
    return q


def survival(q: list[float]) -> list[float]:
    """`S[k] = P(capability >= k)`, `S[0] = 1`."""
    out = [1.0]
    running = 1.0
    for value in q:
        running *= value
        out.append(running)
    return out


def truncated_mean(surv: list[float], depth: int) -> float:
    """`E[min(C, depth)] = sum_{k=1..depth} S[k]`."""
    return sum(surv[k] for k in range(1, min(depth, len(surv) - 1) + 1))


# ------------------------------------------------------------------- reading

def load_leg(run_dir: Path) -> dict:
    meta = read_meta(run_dir)
    rounds = read_rounds(run_dir)
    total = int(meta.get("tokens", str(DECODE_TOKENS)))
    emitted = 0
    for record in rounds:
        record["offer"] = max(1, min(MAX_DEPTH, total - emitted - 1))
        emitted += min(1 + record["accepted"], total - emitted)
    return {
        "run_dir": str(run_dir),
        "prompt_id": run_dir.name,
        "forced_depth": meta.get("forced_depth", "none"),
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "tokens": total,
        "rounds": rounds,
    }


def walk_expected(record: dict, **kwargs) -> tuple:
    """Replays one round's walk and returns (depth, expected, per-step p)."""
    depth, _, walked = cost_model_depth(
        record["ema"], record["margin"],
        offered_depth=record["offer"],
        width_cap=record.get("cap", SEGMENTED_VERIFY_DEPTH_CAP), **kwargs)
    expected = 0.0
    running = 1.0
    for k in range(depth):
        running *= walked[k]
        expected += running
    return depth, expected, walked


def chain_expected(chain: list[float], depth: int) -> float:
    total = 0.0
    running = 1.0
    for k in range(depth):
        running *= chain[k]
        total += running
    return total


# --------------------------------------------------- section 3: hypothesis J

def margin_bin_chains(leg: dict, bins: int = MARGIN_BINS) -> dict:
    """The uncensored chain inside each margin bin, plus the Jensen gains.

    Rounds are split by the pending primary's top-2 margin, which the F1
    answer measured to be a real per-round covariate (position-0 and
    position-1 AUC of 0.78 to 0.91 on beagle). Each bin gets its own
    conditional chain. Two survival curves then follow:

      S_mix[k] = sum_b w_b * prod_{j<k} q_j^b   -- heterogeneous rounds
      S_hom[k] = prod_{j<k} (sum_b w_b q_j^b)   -- one average round

    `S_mix >= S_hom` is the Jensen inequality this hypothesis rests on, and
    `G[k] = S_mix[k] / S_hom[k]` is the measured gain that the `jensen` arm
    applies. Nothing here is fitted.
    """
    rounds = leg["rounds"]
    margins = sorted(r["margin"] for r in rounds)
    edges = [margins[int(len(margins) * (i + 1) / bins) - 1]
             for i in range(bins - 1)]

    def which(margin: float) -> int:
        for index, edge in enumerate(edges):
            if margin <= edge:
                return index
        return bins - 1

    grouped: dict[int, list[dict]] = {b: [] for b in range(bins)}
    for record in rounds:
        grouped[which(record["margin"])].append(record)

    weights, chains = [], []
    for b in range(bins):
        members = grouped[b]
        if not members:
            continue
        weights.append(len(members) / len(rounds))
        chains.append(conditional_chain(
            [r["accepted"] for r in members], [r["depth"] for r in members]))

    s_mix, s_hom = [1.0], [1.0]
    hom_running = 1.0
    for k in range(MAX_DEPTH):
        s_mix.append(sum(
            w * math.prod(chain[:k + 1]) for w, chain in zip(weights, chains)))
        hom_running *= sum(w * chain[k] for w, chain in zip(weights, chains))
        s_hom.append(hom_running)
    gains = [s_mix[k] / s_hom[k] if s_hom[k] > 0 else 1.0
             for k in range(len(s_mix))]

    # The same quantity by second-order expansion, as a cross check on the
    # direct resampling above. cov is taken over bins, weighted by w_b.
    q_bar = [sum(w * chain[j] for w, chain in zip(weights, chains))
             for j in range(MAX_DEPTH)]

    def cov(j: int, jp: int) -> float:
        return sum(w * (chain[j] - q_bar[j]) * (chain[jp] - q_bar[jp])
                   for w, chain in zip(weights, chains))

    second_order = [1.0]
    for k in range(1, MAX_DEPTH + 1):
        term = 0.0
        for j in range(k):
            for jp in range(j + 1, k):
                term += cov(j, jp) / (q_bar[j] * q_bar[jp])
        second_order.append(1.0 + term)

    return {
        "bins": len(weights),
        "bin_weights": weights,
        "bin_chains": chains,
        "bin_edges": edges,
        "s_mix": s_mix,
        "s_hom": s_hom,
        "jensen_gain": gains,
        "jensen_gain_second_order": second_order,
    }


def association_stats(leg: dict) -> dict:
    """Round-level heterogeneity evidence from the uncensored forced leg."""
    rounds = leg["rounds"]
    caps = [r["accepted"] for r in rounds]
    margins = [r["margin"] for r in rounds]
    chain = conditional_chain(caps, [r["depth"] for r in rounds])

    # Does the conditional acceptance rate RISE with position? A rise is the
    # signature of within-round selection: survivors of the early positions
    # are the stronger rounds. A flat or falling profile falsifies that
    # mechanism for this fixture regardless of what the level bias does.
    reached = []
    for j in range(MAX_DEPTH):
        reached.append(sum(1 for r in rounds if r["depth"] > j))
    xs = [j for j in range(MAX_DEPTH) if reached[j] >= 10]
    q_slope = pearson(xs, [chain[j] for j in xs])[1] if len(xs) >= 3 else \
        float("nan")

    # Between-bin share of the capability variance: the fraction of round-level
    # capability spread that an OBSERVABLE round covariate already explains.
    order = sorted(range(len(rounds)), key=lambda i: margins[i])
    size = max(1, len(order) // MARGIN_BINS)
    total_var = statistics.pvariance(caps) if len(caps) > 1 else 0.0
    within = 0.0
    for b in range(MARGIN_BINS):
        chunk = order[b * size:(b + 1) * size if b < MARGIN_BINS - 1
                      else len(order)]
        if len(chunk) > 1:
            within += (len(chunk) / len(rounds)) * \
                statistics.pvariance([caps[i] for i in chunk])
    serial = pearson(caps[:-1], caps[1:])[0] if len(caps) > 3 else float("nan")

    return {
        "chain": chain,
        "reached": reached,
        "q_slope_per_position": q_slope,
        "capability_mean": mean([float(c) for c in caps]),
        "capability_variance": total_var,
        "eta2_margin": (1.0 - within / total_var) if total_var > 0 else 0.0,
        "spearman_margin_capability": spearman(margins, [float(c) for c in caps]),
        "serial_corr_capability": serial,
    }


def hypothesis_j(shipped: dict, forced: dict) -> dict:
    """The exact three-way split of the measured level bias, plus J's own
    independent prediction of it."""
    assoc = association_stats(forced)
    chain = assoc["chain"]
    surv = survival(chain)
    mix = margin_bin_chains(forced)

    expected, expected_nomargin, pred_chain, realised, depths = [], [], [], [], []
    pred_mix, pred_hom = [], []
    for record in shipped["rounds"]:
        depth, exp_value, _ = walk_expected(record)
        if depth != record["depth"]:
            raise SystemExit("replay disagrees with the trace at round %d of %s"
                             % (record["round"], shipped["prompt_id"]))
        _, exp_nomargin, _ = walk_expected(
            record, margin_scale_0=None, margin_scale_1=None)
        expected.append(exp_value)
        expected_nomargin.append(exp_nomargin)
        pred_chain.append(truncated_mean(surv, depth))
        pred_mix.append(sum(mix["s_mix"][k] for k in range(1, depth + 1)))
        pred_hom.append(sum(mix["s_hom"][k] for k in range(1, depth + 1)))
        realised.append(float(record["accepted"]))
        depths.append(depth)

    measured_bias = mean(expected) - mean(realised)
    by_depth = {}
    for d in sorted(set(depths)):
        rows = [(e, a) for e, a, dd in zip(expected, realised, depths) if dd == d]
        by_depth[d] = {
            "rounds": len(rows),
            "mean_expected": mean([e for e, _ in rows]),
            "mean_accepted": mean([a for _, a in rows]),
            "bias": mean([e for e, _ in rows]) - mean([a for _, a in rows]),
        }

    return {
        "prompt_id": shipped["prompt_id"],
        "rounds": len(shipped["rounds"]),
        "mean_depth": mean([float(d) for d in depths]),
        "mean_expected": mean(expected),
        "mean_accepted": mean(realised),
        "measured_bias": measured_bias,
        "gamma": mean(realised) / mean(expected) if mean(expected) else float("nan"),
        # exact split; the three components sum to measured_bias
        "margin_component": mean(expected) - mean(expected_nomargin),
        "ema_component": mean(expected_nomargin) - mean(pred_chain),
        "selection_component": mean(pred_chain) - mean(realised),
        # hypothesis J's own prediction, formed without touching `realised`
        "jensen_predicted_bias": mean(pred_hom) - mean(pred_mix),
        "jensen_gain": mix["jensen_gain"],
        "jensen_gain_second_order": mix["jensen_gain_second_order"],
        "bias_by_depth": by_depth,
        "association": {k: v for k, v in assoc.items() if k != "chain"},
        "uncensored_chain": chain,
        "uncensored_survival": surv,
    }


# ----------------------------------------------- section 4: the sign decomposition

def impute_accepted(record: dict, surv: list[float], new_depth: int) -> float:
    """`E[min(C, new_depth)]` given what the recorded round revealed.

    A round that rejected at `a < d` revealed its capability exactly. A round
    that accepted all `d` drafts is censored at `d`, so the tail beyond it is
    drawn from the fixture's own uncensored survival curve, conditioned on
    having already survived to `d`.
    """
    accepted, depth = record["accepted"], record["depth"]
    if accepted < depth:
        return float(min(accepted, new_depth))
    if new_depth <= depth:
        return float(new_depth)
    base = surv[min(depth, len(surv) - 1)]
    if base <= 0:
        return float(depth)
    extra = sum(surv[min(k, len(surv) - 1)] / base
                for k in range(depth + 1, new_depth + 1))
    return depth + extra


def price_replay(shipped: dict, surv: list[float], chooser=None,
                 **kwargs) -> dict:
    """Depth and ranked cost per token for one myopic counterfactual."""
    total_us = 0.0
    total_tokens = 0.0
    depths = []
    for record in shipped["rounds"]:
        if chooser is None:
            depth, _, _ = walk_expected(record, **kwargs)
        else:
            depth = chooser(record, surv)
        depths.append(depth)
        total_us += ranked_round_us(depth + 1)
        total_tokens += 1.0 + impute_accepted(record, surv, depth)
    return {
        "mean_depth": mean([float(d) for d in depths]),
        "us_per_token": total_us / total_tokens,
        "tokens_per_round": total_tokens / len(depths),
    }


def make_static(depth: int):
    return lambda record, surv: min(
        depth, record["offer"], MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)


def oracle_depth(record: dict, surv: list[float]) -> int:
    """Per-round argmin of the ranked cost per token, given the round's own
    revealed capability. This is the ceiling of the whole scheduler axis: no
    depth rule that reads only pre-proposal signals can beat it."""
    cap = min(record["offer"], MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)
    best, best_cost = 0, ranked_round_us(1)
    for depth in range(1, cap + 1):
        cost = ranked_round_us(depth + 1) / (
            1.0 + impute_accepted(record, surv, depth))
        if cost < best_cost:
            best, best_cost = depth, cost
    return best


def sign_decomposition(shipped: dict, jrow: dict) -> dict:
    surv = jrow["uncensored_survival"]
    gamma = jrow["gamma"]
    gains = jrow["jensen_gain"][1:]
    base = price_replay(shipped, surv)
    # `expected` is a sum of reaches, so the same measured gain has to be
    # available to it as well; applying the correction only to `reach` is the
    # one-sided arm, not the corrected estimator.
    expected_gain_from_gains = mean(gains[:4])
    arms = {
        "reachonly": {"reach_gain": gamma},
        "expectedonly": {"expected_gain": gamma},
        "levelfix": {"reach_gain": gamma, "expected_gain": gamma},
        "jensen": {"reach_gain_by_step": gains},
        "jensen_both": {"reach_gain_by_step": gains,
                        "expected_gain": expected_gain_from_gains},
    }
    out = {"prompt_id": shipped["prompt_id"], "gamma": gamma, "base": base,
           "arms": {}, "gamma_grid": {}, "static": {}}
    for name, kwargs in arms.items():
        row = price_replay(shipped, surv, **kwargs)
        row["delta_depth"] = row["mean_depth"] - base["mean_depth"]
        row["ranked_gain_pct"] = 100.0 * (
            base["us_per_token"] / row["us_per_token"] - 1.0)
        out["arms"][name] = row
    for name, chooser in [("oracle", oracle_depth)] + \
            [("static%d" % d, make_static(d)) for d in range(MAX_DEPTH)]:
        row = price_replay(shipped, surv, chooser=chooser)
        row["delta_depth"] = row["mean_depth"] - base["mean_depth"]
        row["ranked_gain_pct"] = 100.0 * (
            base["us_per_token"] / row["us_per_token"] - 1.0)
        out["static" if name.startswith("static") else "arms"][name] = row
    for value in GAMMA_GRID:
        row = price_replay(shipped, surv, reach_gain=value, expected_gain=value)
        out["gamma_grid"]["%.2f" % value] = {
            "mean_depth": row["mean_depth"],
            "ranked_gain_pct": 100.0 * (
                base["us_per_token"] / row["us_per_token"] - 1.0),
        }
    return out


# ----------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shipped", nargs="+", required=True)
    parser.add_argument("--forced", nargs="+", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()

    forced = {Path(p).name: load_leg(Path(p)) for p in args.forced}
    rows, signs = [], []
    for path in args.shipped:
        leg = load_leg(Path(path))
        if leg["prompt_id"] not in forced:
            raise SystemExit("no forced leg for %s" % leg["prompt_id"])
        row = hypothesis_j(leg, forced[leg["prompt_id"]])
        rows.append(row)
        signs.append(sign_decomposition(leg, row))

    print("section 3 -- exact split of the measured level bias "
          "(negative = the walk under-predicts)\n")
    print("%-18s %6s %7s %7s %7s | %8s %8s %8s | %8s" % (
        "fixture", "rounds", "E[exp]", "E[acc]", "gamma",
        "margin", "ema", "select", "J-pred"))
    for row in rows:
        print("%-18s %6d %7.3f %7.3f %7.3f | %8.3f %8.3f %8.3f | %8.3f" % (
            row["prompt_id"], row["rounds"], row["mean_expected"],
            row["mean_accepted"], row["gamma"], row["margin_component"],
            row["ema_component"], row["selection_component"],
            row["jensen_predicted_bias"]))

    measured = [r["measured_bias"] for r in rows]
    tests = {}
    for name in ("jensen_predicted_bias", "selection_component",
                 "margin_component", "ema_component"):
        r, slope, intercept = pearson([row[name] for row in rows], measured)
        tests[name] = {"r": r, "slope": slope, "intercept": intercept}
    # Internal validation. Hypothesis J's resampled prediction can only own the
    # selection component, so this is the regression that tests the RESAMPLING,
    # separately from the regression that tests whether that mechanism is the
    # one carrying the measured bias.
    r, slope, intercept = pearson(
        [row["jensen_predicted_bias"] for row in rows],
        [row["selection_component"] for row in rows])
    tests["jensen_predicts_selection"] = {
        "r": r, "slope": slope, "intercept": intercept}
    print("\nhypothesis J test -- measured_bias regressed on each candidate:")
    for name, stat in tests.items():
        print("  %-24s r=%+.3f slope=%+.3f intercept=%+.3f"
              % (name, stat["r"], stat["slope"], stat["intercept"]))
    share = [abs(row["selection_component"]) /
             (abs(row["margin_component"]) + abs(row["ema_component"])
              + abs(row["selection_component"])) for row in rows]
    print("  selection component carries %.1f %% of the split on the median "
          "fixture (range %.1f to %.1f)" % (
              100.0 * statistics.median(share), 100.0 * min(share),
              100.0 * max(share)))

    lead = tests["jensen_predicted_bias"]
    residual = [m - (lead["intercept"] + lead["slope"] * row["jensen_predicted_bias"])
                for m, row in zip(measured, rows)]
    covariates = {
        "margin_component": [r["margin_component"] for r in rows],
        "ema_component": [r["ema_component"] for r in rows],
        "selection_component": [r["selection_component"] for r in rows],
        "mean_depth": [r["mean_depth"] for r in rows],
        "gamma": [r["gamma"] for r in rows],
        "q0": [r["uncensored_chain"][0] for r in rows],
        "q_slope": [r["association"]["q_slope_per_position"] for r in rows],
        "eta2_margin": [r["association"]["eta2_margin"] for r in rows],
        "spearman_margin_capability":
            [r["association"]["spearman_margin_capability"] for r in rows],
    }
    residual_fit = {name: pearson(values, residual)[0]
                    for name, values in covariates.items()}
    print("\nresidual of the hypothesis J fit, correlated with:")
    for name, value in sorted(residual_fit.items(),
                              key=lambda kv: -abs(kv[1])):
        print("  %-30s r=%+.3f" % (name, value))

    print("\nsection 4 -- SIGN of the level correction "
          "(myopic replay, ranked price, imputed acceptance)\n")
    print("%-18s %7s | %6s %8s | %6s %8s | %6s %8s | %6s %8s" % (
        "fixture", "gamma", "d:rch", "%rch", "d:exp", "%exp",
        "d:both", "%both", "d:jns", "%jns"))
    for row in signs:
        a = row["arms"]
        print("%-18s %7.3f | %+6.3f %+8.3f | %+6.3f %+8.3f | "
              "%+6.3f %+8.3f | %+6.3f %+8.3f" % (
                  row["prompt_id"], row["gamma"],
                  a["reachonly"]["delta_depth"], a["reachonly"]["ranked_gain_pct"],
                  a["expectedonly"]["delta_depth"],
                  a["expectedonly"]["ranked_gain_pct"],
                  a["levelfix"]["delta_depth"], a["levelfix"]["ranked_gain_pct"],
                  a["jensen"]["delta_depth"], a["jensen"]["ranked_gain_pct"]))

    for name in ("reachonly", "expectedonly", "levelfix", "jensen",
                 "jensen_both", "oracle"):
        deltas = [row["arms"][name]["delta_depth"] for row in signs]
        gains = [row["arms"][name]["ranked_gain_pct"] for row in signs]
        print("  %-13s median depth %+0.3f, median ranked %+0.3f %%, "
              "%d of %d positive" % (
                  name, statistics.median(deltas), statistics.median(gains),
                  sum(1 for g in gains if g > 0), len(gains)))

    print("\nfixed depth, median fixture-level ranked gain over shipped:")
    for d in range(MAX_DEPTH):
        key = "static%d" % d
        gains = [row["static"][key]["ranked_gain_pct"] for row in signs]
        print("  depth=%d  median ranked %+0.3f %%, %d of %d positive"
              % (d, statistics.median(gains),
                 sum(1 for g in gains if g > 0), len(gains)))

    print("\nglobal gamma sweep, median fixture-level ranked gain:")
    for value in GAMMA_GRID:
        key = "%.2f" % value
        gains = [row["gamma_grid"][key]["ranked_gain_pct"] for row in signs]
        depths = [row["gamma_grid"][key]["mean_depth"] for row in signs]
        print("  gamma=%s  median depth %.3f  median ranked %+0.3f %%"
              % (key, statistics.median(depths), statistics.median(gains)))

    payload = {
        "harness": "local",
        "hypothesis_j": rows,
        "hypothesis_j_regression": tests,
        "hypothesis_j_residual_correlations": residual_fit,
        "sign_decomposition": signs,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
