#!/usr/bin/env python3
"""E168: is the pending-primary top-2 margin predictive of draft acceptance,
and is `min(EMA, sigmoid(margin/T))` a calibrated way to use it?

Reads `mtp-trace:` round lines written by `MLX_QWEN_MTP_TRACE=1`. Two leg kinds
answer two different questions and must not be pooled:

  pinned  `MLX_E159_FIXED_DRAFT_DEPTH=7` replaces the schedule, so every round
          proposes 7 drafts whatever the margin says. This is the ONLY leg that
          observes acceptance at a position the shipped clamp would have
          refused to draft, so the calibration curve is fitted here.
  adapt   the shipped adaptive schedule. Used only to count how often the clamp
          binds in production and how much offered depth it removes; its
          acceptance observations are censored BY the clamp under test.

Usage:
  research/e168_margin_calibration.py --pinned DIR [DIR ...] \
      [--adapt DIR [DIR ...]] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

# Shipped constants, Qwen36MTPBlockSession.swift.
HEAD_STEP_COST_RATIO = 0.18
MAX_DEPTH = 8
WIDTH_CAP = 7
CLAMP_T = {0: 2.0, 1: 3.0}

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
# The schedule snapshot is written only when the round consulted the schedule.
# A leg built before the pinned-depth path snapshotted it reports the round
# bookkeeping without it; that leg still answers every counts question and
# answers no calibration question.
SCHEDULE_RE = re.compile(
    r" m=(\S+) streak=(\d+) (?:offer=(\d+) wcap=(\d+) )?cap=(\d+) ema=(\S+)"
)
ROW_RE = re.compile(r"^mtp-row: pos=(\d+) ids=(\d+),(\d+) v=(\S+)")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def parse_rounds(path: str) -> tuple[list[dict], dict[int, dict]]:
    """Round records, plus the absolute-position row evidence they emitted.

    Each round writes its accepted-trajectory rows (`mtp-row`) before its own
    summary line, so the rows seen since the previous summary belong to the
    round that follows them. Those rows carry the ABSOLUTE token position, which
    is the only key that survives a change of draft depth: two arms partition
    the same serial token stream into different rounds, but position `p` is the
    same token in both.
    """
    rounds: list[dict] = []
    rows: dict[int, dict] = {}
    pending: list[int] = []
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            row = ROW_RE.match(line)
            if row:
                position = int(row.group(1))
                # A census leg writes rows for the reference pass before the
                # MTP pass, and both start at the same absolute position. Row
                # positions run contiguously inside one pass, so a position
                # that does not continue the previous one starts a new pass and
                # cannot belong to the round that follows.
                if pending and position != pending[-1] + 1:
                    pending = []
                pending.append(position)
                rows[position] = {
                    "ids": (int(row.group(2)), int(row.group(3))),
                    "v": row.group(4),
                }
                continue
            match = ROUND_RE.match(line)
            if not match:
                continue
            record = {
                "round": int(match.group(1)),
                "d": int(match.group(2)),
                "acc": int(match.group(3)),
                "margin": float("nan"),
                "streak": None,
                "offer": None,
                "wcap": WIDTH_CAP,
                "cap": None,
                "ema": None,
                "start": pending[0] if pending else None,
                "rows": len(pending),
            }
            schedule = SCHEDULE_RE.search(line)
            if schedule:
                record.update(
                    margin=float(schedule.group(1)),
                    streak=int(schedule.group(2)),
                    offer=(
                        int(schedule.group(3)) if schedule.group(3) else None
                    ),
                    wcap=(
                        int(schedule.group(4))
                        if schedule.group(4)
                        else WIDTH_CAP
                    ),
                    cap=int(schedule.group(5)),
                    ema=[float(v) for v in schedule.group(6).split(",")],
                )
            rounds.append(record)
            pending = []
    return rounds, rows


def load_legs(dirs: list[str], kind: str) -> list[dict]:
    legs = []
    for directory in dirs:
        trace = os.path.join(directory, "trace.txt")
        if not os.path.exists(trace):
            sys.exit(f"e168: no trace.txt in {directory}")
        meta = {}
        meta_path = os.path.join(directory, "meta.txt")
        if os.path.exists(meta_path):
            for line in open(meta_path):
                if "=" in line:
                    key, _, value = line.partition("=")
                    meta[key.strip()] = value.strip()
        rounds, rows = parse_rounds(trace)
        legs.append(
            {
                "dir": directory,
                "kind": kind,
                "label": meta.get("prompt_id", os.path.basename(directory)),
                "meta": meta,
                "rounds": rounds,
                "rows": rows,
            }
        )
    return legs


def walk(
    margin: float,
    ema: list[float],
    offered: int,
    clamped: bool,
    ratio: float = HEAD_STEP_COST_RATIO,
    temps: dict[int, float] | None = None,
) -> int:
    """Replay `costModelDepth` exactly, with or without the margin clamps.

    `ratio` and `temps` default to the shipped constants, so the default call
    is the shipped controller. Overriding them replays a counterfactual
    controller on the same recorded inputs.
    """
    cap = min(min(offered, MAX_DEPTH), WIDTH_CAP)
    if cap <= 0:
        return 0
    table = CLAMP_T if temps is None else temps
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        p = ema[depth]
        if clamped and depth in table and not math.isnan(margin):
            p = min(p, sigmoid(margin / table[depth]))
        reach *= p
        threshold = ratio * (1.0 + expected) / (1.0 + depth * ratio)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def observations(rounds: list[dict], position: int) -> list[tuple[float, int, float]]:
    """(margin, accepted, ema_at_position) for rounds that RESOLVED `position`.

    Position k is resolved when the round proposed at least k+1 drafts and the
    accept walk reached it: every position below k was accepted. Positions past
    the first reject were verified but never walked, so they carry no outcome.
    """
    out = []
    for record in rounds:
        if record["d"] <= position or record["acc"] < position:
            continue
        if math.isnan(record["margin"]):
            continue
        accepted = 1 if record["acc"] > position else 0
        out.append((record["margin"], accepted, record["ema"][position]))
    return out


def auc_interval(area: float, positives: int, negatives: int) -> tuple[float, float]:
    """Hanley-McNeil 95% interval, so a null AUC is stated with its precision.

    An AUC near 0.5 only closes a signal if the interval is narrow enough to
    exclude a useful effect. Reporting the point estimate alone would let a
    small sample masquerade as a negative result.
    """
    if positives < 1 or negatives < 1 or math.isnan(area):
        return float("nan"), float("nan")
    q1 = area / (2.0 - area)
    q2 = 2.0 * area * area / (1.0 + area)
    variance = (
        area * (1.0 - area)
        + (positives - 1) * (q1 - area * area)
        + (negatives - 1) * (q2 - area * area)
    ) / (positives * negatives)
    spread = 1.959963985 * math.sqrt(max(variance, 0.0))
    return max(0.0, area - spread), min(1.0, area + spread)


def signal_auc(rounds: list[dict], position: int) -> dict:
    """AUC of every signal the round already has, not just the margin.

    The clamp uses the margin, and the streak gate uses the run of full
    accepts. If none of them ranks acceptance better than chance, then no
    re-tuning of this controller can help: the controller cannot separate
    rounds it cannot tell apart, and the ceiling is a constant depth.
    """
    rows = []
    for record in rounds:
        if record["d"] <= position or record["acc"] < position:
            continue
        if math.isnan(record["margin"]) or record["streak"] is None:
            continue
        rows.append(
            {
                "accepted": 1 if record["acc"] > position else 0,
                "margin": record["margin"],
                "ema": record["ema"][position],
                "streak": float(record["streak"]),
            }
        )
    if not rows:
        return {}
    positives = sum(r["accepted"] for r in rows)
    negatives = len(rows) - positives
    scores = {}
    for name in ("margin", "ema", "streak"):
        value = auc([(r[name], r["accepted"], 0.0) for r in rows])
        low, high = auc_interval(value, positives, negatives)
        scores[name] = {"auc": value, "ci_low": low, "ci_high": high}
    return {
        "n": len(rows),
        "positives": positives,
        "base_rate": positives / len(rows),
        "auc": scores,
    }


def autocorrelation(series: list[list[float]], lags: int = 5) -> dict:
    """Lag 1..`lags` autocorrelation of several independent sequences.

    Each sequence is one leg, so no lag ever spans two prompts.
    """
    total = sum(len(s) for s in series)
    if total < 30:
        return {}
    mean = sum(sum(s) for s in series) / total
    variance = sum((v - mean) ** 2 for s in series for v in s) / total
    out = {"n": total, "mean": mean, "lags": {}}
    if variance <= 0:
        return out
    for lag in range(1, lags + 1):
        covariance = 0.0
        pairs = 0
        for sequence in series:
            for index in range(len(sequence) - lag):
                covariance += (sequence[index] - mean) * (sequence[index + lag] - mean)
                pairs += 1
        if pairs < 10:
            continue
        rho = covariance / pairs / variance
        # Bartlett's band for white noise; a lag inside it is not evidence of
        # structure at that lag.
        band = 1.959963985 / math.sqrt(pairs)
        out["lags"][lag] = {
            "rho": rho,
            "pairs": pairs,
            "band": band,
            "significant": abs(rho) > band,
        }
    return out


def history_structure(legs: list[dict], position: int, lags: int = 5) -> dict:
    """Does acceptance have memory, and is the margin a real signal at all?

    `accept` answers the first question: the EMA and the streak are history
    summaries, so they can only carry information if the outcome sequence is
    autocorrelated.

    `margin` is descriptive rather than a test. It asks whether the margin
    itself carries memory, which decides whether a smoothed margin could work
    where the instantaneous one does not.

    Neither series can settle whether the recording is sound. Two other checks
    do that, and both are reported elsewhere: the offline replay of
    `costModelDepth` reproduces every recorded depth from the recorded margin,
    EMA and offer, and the splice finds the same margin at the same absolute
    token in two independently scheduled arms.
    """
    accepts, margins = [], []
    for leg in legs:
        rounds = [
            record
            for record in leg["rounds"]
            if record["d"] > position and record["acc"] >= position
        ]
        if len(rounds) <= lags + 1:
            continue
        accepts.append([1.0 if r["acc"] > position else 0.0 for r in rounds])
        usable = [r["margin"] for r in rounds if not math.isnan(r["margin"])]
        if len(usable) == len(rounds):
            margins.append(usable)
    return {
        "accept": autocorrelation(accepts, lags),
        "margin": autocorrelation(margins, lags),
    }


def oracle_ceiling(legs: list[dict]) -> list[dict]:
    """The best any per-round depth rule could do, given perfect foresight.

    A round that will accept `a` drafts gains nothing from proposing more than
    `a` and loses a token for every one fewer, so `d = a` is the per-round
    optimum whenever a marginal accepted token outprices a verify row. This is
    an upper bound on adaptivity itself: no signal, however good, can beat it.
    """
    out = []
    for leg in legs:
        rounds = leg["rounds"]
        if not rounds:
            continue
        mean_a = sum(record["acc"] for record in rounds) / len(rounds)
        out.append(
            {
                "label": leg["label"],
                "rounds": len(rounds),
                "mean_depth": mean_a,
                "mean_accepted": mean_a,
                "raw": ranked_price(mean_a, mean_a)["raw"],
            }
        )
    return out


def fit_temperature(samples: list[tuple[float, int, float]]) -> tuple[float, float]:
    """MLE of T in P(accept) = sigmoid(margin / T) by golden-section search."""
    if not samples:
        return float("nan"), float("nan")

    def negative_log_likelihood(temperature: float) -> float:
        total = 0.0
        for margin, accepted, _ in samples:
            p = min(max(sigmoid(margin / temperature), 1e-12), 1 - 1e-12)
            total -= math.log(p) if accepted else math.log(1.0 - p)
        return total

    lo, hi = 1e-3, 400.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = math.log(lo), math.log(hi)
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = negative_log_likelihood(math.exp(c)), negative_log_likelihood(math.exp(d))
    for _ in range(200):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = negative_log_likelihood(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = negative_log_likelihood(math.exp(d))
    best = math.exp((a + b) / 2.0)
    return best, negative_log_likelihood(best)


def score_predictor(samples, predict) -> dict:
    """Log-loss and Brier score of one probability model."""
    if not samples:
        return {"n": 0}
    log_loss = 0.0
    brier = 0.0
    for margin, accepted, ema in samples:
        p = min(max(predict(margin, ema), 1e-12), 1 - 1e-12)
        log_loss -= math.log(p) if accepted else math.log(1.0 - p)
        brier += (p - accepted) ** 2
    n = len(samples)
    return {
        "n": n,
        "log_loss": log_loss / n,
        "brier": brier / n,
        "mean_prediction": sum(predict(m, e) for m, _, e in samples) / n,
    }


def auc(samples: list[tuple[float, int, float]]) -> float:
    """Mann-Whitney AUC of margin as a ranker of acceptance."""
    positives = [m for m, a, _ in samples if a == 1]
    negatives = [m for m, a, _ in samples if a == 0]
    if not positives or not negatives:
        return float("nan")
    ordered = sorted((m, a) for m, a, _ in samples)
    ranks: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    positive_rank_sum = sum(
        ranks[i] for i, (_, a) in enumerate(ordered) if a == 1
    )
    n_pos, n_neg = len(positives), len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963985
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - spread, centre + spread


BIN_EDGES = [-1e9, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 1e9]


def bin_index(margin: float) -> int:
    for index in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[index] <= margin < BIN_EDGES[index + 1]:
            return index
    return len(BIN_EDGES) - 2


def calibration_table(samples, position: int) -> list[dict]:
    buckets = defaultdict(lambda: {"n": 0, "accepted": 0, "margin_sum": 0.0,
                                   "ema_sum": 0.0})
    for margin, accepted, ema in samples:
        bucket = buckets[bin_index(margin)]
        bucket["n"] += 1
        bucket["accepted"] += accepted
        bucket["margin_sum"] += margin
        bucket["ema_sum"] += ema
    rows = []
    for index in sorted(buckets):
        bucket = buckets[index]
        n = bucket["n"]
        mean_margin = bucket["margin_sum"] / n
        low, high = wilson(bucket["accepted"], n)
        rows.append(
            {
                "bin": f"[{BIN_EDGES[index]:g},{BIN_EDGES[index + 1]:g})".replace(
                    "-1e+09", "-inf"
                ).replace("1e+09", "inf"),
                "n": n,
                "mean_margin": mean_margin,
                "empirical": bucket["accepted"] / n,
                "ci_low": low,
                "ci_high": high,
                "clamp_sigmoid": sigmoid(mean_margin / CLAMP_T[position]),
                "mean_ema": bucket["ema_sum"] / n,
            }
        )
    return rows


def analyse_pinned(legs: list[dict]) -> dict:
    result = {"legs": [], "positions": {}, "profile_by_leg": {}}
    pooled: dict[int, list] = {0: [], 1: [], 2: []}
    pooled_rounds: list[dict] = []
    for leg in legs:
        rounds = leg["rounds"]
        depths = [r["d"] for r in rounds]
        result["legs"].append(
            {
                "label": leg["label"],
                "dir": leg["dir"],
                "rounds": len(rounds),
                "mean_d": sum(depths) / len(depths) if depths else 0.0,
                "mean_acc": (
                    sum(r["acc"] for r in rounds) / len(rounds) if rounds else 0.0
                ),
                "accept_fraction": (
                    sum(r["acc"] for r in rounds) / sum(depths) if sum(depths) else 0.0
                ),
                "margin_mean": (
                    sum(r["margin"] for r in rounds) / len(rounds) if rounds else 0.0
                ),
            }
        )
        for position in pooled:
            pooled[position].extend(observations(rounds, position))
        pooled_rounds.extend(rounds)
        result["profile_by_leg"][leg["label"]] = position_profile(rounds)

    result["profile"] = position_profile(pooled_rounds)
    for position in (0, 1):
        samples = pooled[position]
        temperature, _ = fit_temperature(samples)
        shipped_t = CLAMP_T[position]
        result["positions"][position] = {
            "n": len(samples),
            "base_rate": (
                sum(a for _, a, _ in samples) / len(samples) if samples else 0.0
            ),
            "fitted_temperature": temperature,
            "shipped_temperature": shipped_t,
            "auc_margin": auc(samples),
            "table": calibration_table(samples, position),
            "predictors": {
                "ema_only": score_predictor(samples, lambda m, e: e),
                "shipped_clamp": score_predictor(
                    samples, lambda m, e, t=shipped_t: min(e, sigmoid(m / t))
                ),
                "fitted_clamp": score_predictor(
                    samples, lambda m, e, t=temperature: min(e, sigmoid(m / t))
                ),
                "margin_only_fitted": score_predictor(
                    samples, lambda m, e, t=temperature: sigmoid(m / t)
                ),
                "base_rate": score_predictor(
                    samples,
                    lambda m, e, b=(
                        sum(a for _, a, _ in samples) / len(samples)
                        if samples
                        else 0.5
                    ): b,
                ),
            },
        }
    return result


# Ranked M5 Max constants, advisor F2 sections 4 and 5. `R(M)` is ms per round
# against `M = 1 + proposed draft rows`; prefill and the pinned serial time are
# near-constant across the eight ranked prompts.
RANKED_ROUND_FIXED_MS = 16.1585
RANKED_ROW_MS = 5.3350
RANKED_PREFILL_MS_PER_TOKEN = 1.0310
RANKED_SERIAL_MS_PER_TOKEN = 37.92
RANKED_WINDOW = 512


def ranked_price(mean_depth: float, mean_accepted: float) -> dict:
    """Convert one arm's round bookkeeping into a ranked `raw` ratio.

    This is a TRANSFER, not a measurement. It carries only two counts, the
    proposed depth and the accepted extra tokens per round, onto the ranked
    round law. It assumes the counts hold on the ranked host and that the law
    is affine in proposed rows; it assumes nothing about local wall time, so no
    ungated local timing enters it.
    """
    tokens_per_round = 1.0 + mean_accepted
    rounds = RANKED_WINDOW / tokens_per_round
    round_ms = RANKED_ROUND_FIXED_MS + RANKED_ROW_MS * (1.0 + mean_depth)
    decode_ms = rounds * round_ms
    leg_ms = decode_ms + RANKED_WINDOW * RANKED_PREFILL_MS_PER_TOKEN
    return {
        "tokens_per_round": tokens_per_round,
        "rounds": rounds,
        "round_ms": round_ms,
        "decode_ms": decode_ms,
        "leg_ms": leg_ms,
        "ms_per_token": leg_ms / RANKED_WINDOW,
        "raw": RANKED_SERIAL_MS_PER_TOKEN / (leg_ms / RANKED_WINDOW),
    }


def profile_decay(profile: list[dict], min_reached: int = 20) -> dict:
    """Log-linear decay of the per-position conditional acceptance.

    The advisor's break-even table is stated as a percentage fall per extra
    position, so report the same shape: a least-squares slope of `log q` on
    position, over the positions with enough observations to mean anything.
    """
    points = [
        (row["position"], math.log(row["q"]))
        for row in profile
        if row["reached"] >= min_reached and row["q"] > 0.0
    ]
    if len(points) < 2:
        return {"positions_used": len(points)}
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return {
        "positions_used": n,
        "q0_fit": math.exp(mean_y - slope * mean_x),
        "decay_per_position": 1.0 - math.exp(slope),
        "first_step_decay": (
            1.0 - profile[1]["q"] / profile[0]["q"] if len(profile) > 1 else None
        ),
    }


def leg_bookkeeping(rounds: list[dict]) -> dict:
    """The counts the ranked conversion needs, and nothing modelled.

    The halves split is a warm-up control: the proposal head and the schedule
    both carry state across a leg, so an acceptance rate that differs between
    the halves is a transient, not the prompt's acceptance.
    """
    count = len(rounds)
    proposed = sum(record["d"] for record in rounds)
    accepted = sum(record["acc"] for record in rounds)
    half = count // 2

    def fraction(subset: list[dict]) -> float:
        rows = sum(record["d"] for record in subset)
        return sum(record["acc"] for record in subset) / rows if rows else float("nan")

    capped = [record for record in rounds if record["cap"] is not None]
    histogram: dict[int, int] = defaultdict(int)
    bins: dict[int, dict] = defaultdict(
        lambda: {
            "rounds": 0,
            "proposed": 0,
            "accepted": 0,
            "reached0": 0,
            "accepted0": 0,
            "margin_sum": 0.0,
            "margin_n": 0,
        }
    )
    for record in rounds:
        histogram[record["d"]] += 1
        bucket = bins[record["d"]]
        bucket["rounds"] += 1
        bucket["proposed"] += record["d"]
        bucket["accepted"] += record["acc"]
        # Accepted-over-proposed inside a depth bin falls with depth for any
        # chain with q < 1, because a deeper round proposes rows the chain
        # rarely reaches. That is arithmetic, not selection. `q0` is the
        # unconfounded comparison: it asks whether the FIRST draft of a deep
        # round is more likely to be right than the first draft of a shallow
        # one, which is what "the controller picks its spots" has to mean.
        if record["d"] >= 1:
            bucket["reached0"] += 1
            bucket["accepted0"] += 1 if record["acc"] >= 1 else 0
        if not math.isnan(record["margin"]):
            bucket["margin_sum"] += record["margin"]
            bucket["margin_n"] += 1
    for bucket in bins.values():
        bucket["accept_fraction"] = (
            bucket["accepted"] / bucket["proposed"] if bucket["proposed"] else 0.0
        )
        bucket["tokens_per_round"] = 1.0 + bucket["accepted"] / bucket["rounds"]
        bucket["q0"] = (
            bucket["accepted0"] / bucket["reached0"] if bucket["reached0"] else None
        )
        bucket["mean_margin"] = (
            bucket["margin_sum"] / bucket["margin_n"] if bucket["margin_n"] else None
        )
    return {
        "rounds": count,
        "proposed": proposed,
        "accepted": accepted,
        "mean_depth": proposed / count if count else 0.0,
        "mean_accepted": accepted / count if count else 0.0,
        "accept_fraction": accepted / proposed if proposed else 0.0,
        "tokens_per_round": 1.0 + (accepted / count if count else 0.0),
        "accept_fraction_first_half": fraction(rounds[:half]),
        "accept_fraction_second_half": fraction(rounds[half:]),
        # The chosen-depth histogram and the at-cap fraction answer the same
        # question from two sides: how much of the depth axis is already spent.
        "depth_histogram": dict(sorted(histogram.items())),
        "depth_bins": {depth: bins[depth] for depth in sorted(bins)},
        "cap_stop_fraction": (
            sum(1 for record in capped if record["d"] >= record["cap"]) / len(capped)
            if capped
            else None
        ),
        "at_cap_fraction": (
            sum(1 for record in capped if record["d"] == record["cap"]) / len(capped)
            if capped
            else None
        ),
        "schedule_rounds": len(capped),
    }


def compare_arms(pinned_legs: list[dict], adapt_legs: list[dict]) -> dict:
    """Per prompt, both arms' round bookkeeping and their ranked prices.

    The comparison of the two arms' accept fractions is the selection test: the
    adaptive arm chooses when to be deep, so if it accepts a higher share of
    what it proposes than the fixed-depth arm does on the SAME text, the
    controller is picking its spots and no stationary profile can stand in for
    it.
    """
    pinned_by_label = {leg["label"]: leg for leg in pinned_legs}
    rows = []
    for leg in adapt_legs:
        pinned = pinned_by_label.get(leg["label"])
        if pinned is None:
            continue
        adapt_counts = leg_bookkeeping(leg["rounds"])
        pinned_counts = leg_bookkeeping(pinned["rounds"])
        adapt_price = ranked_price(
            adapt_counts["mean_depth"], adapt_counts["mean_accepted"]
        )
        pinned_price = ranked_price(
            pinned_counts["mean_depth"], pinned_counts["mean_accepted"]
        )
        rows.append(
            {
                "label": leg["label"],
                "adapt": adapt_counts,
                "pinned": pinned_counts,
                "raw_adapt": adapt_price["raw"],
                "raw_pinned": pinned_price["raw"],
                "raw_ratio": pinned_price["raw"] / adapt_price["raw"],
                "adapt_price": adapt_price,
                "pinned_price": pinned_price,
                "selection_gain": (
                    adapt_counts["accept_fraction"]
                    - pinned_counts["accept_fraction"]
                ),
                "pinned_profile": position_profile(pinned["rounds"]),
                "pinned_decay": profile_decay(position_profile(pinned["rounds"])),
                "adapt_profile": position_profile(leg["rounds"]),
            }
        )
    return {"prompts": rows}


def position_profile(rounds: list[dict], max_position: int = 8) -> list[dict]:
    """Per-position conditional acceptance, the quantity the walk models.

    `P(position k accepted | positions 0..<k accepted and k was drafted)`. This
    is the same conditional the EMA tracks, so a flat profile means the walk's
    chain-rule `reach` is well specified and a decaying one means it is not.
    """
    rows = []
    for position in range(max_position):
        reached = sum(
            1 for r in rounds if r["d"] > position and r["acc"] >= position
        )
        accepted = sum(1 for r in rounds if r["acc"] > position)
        if reached == 0:
            continue
        low, high = wilson(accepted, reached)
        rows.append(
            {
                "position": position,
                "reached": reached,
                "accepted": accepted,
                "q": accepted / reached,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def analyse_adapt(legs: list[dict], offered_default: int) -> dict:
    result = {"legs": [], "pooled": {}}
    total = binds0 = binds1 = 0
    depth_loss_rounds = 0
    depth_loss_sum = 0
    pure_loss_rounds = 0
    replay_mismatch = 0
    cap_stops = 0
    cap_histogram: dict[int, int] = defaultdict(int)
    pooled_rounds: list[dict] = []
    for leg in legs:
        leg_total = leg_binds = leg_loss = leg_cap_stops = 0
        leg_loss_sum = 0
        for record in leg["rounds"]:
            margin, ema = record["margin"], record["ema"]
            if math.isnan(margin):
                continue
            # The parent narrows its own offer as the decode window runs down,
            # so a round replayed at a fixed offer 8 is not the round that ran.
            # A trace without `offer=` predates that field; its tail round is
            # the only one the default can misreplay.
            offered = record["offer"] if record["offer"] is not None else (
                offered_default
            )
            cap = record["cap"] if record["offer"] is not None else min(
                min(offered, MAX_DEPTH), record["wcap"]
            )
            leg_total += 1
            total += 1
            cap_histogram[cap] += 1
            bind0 = sigmoid(margin / 2.0) < ema[0]
            bind1 = sigmoid(margin / 3.0) < ema[1]
            binds0 += bind0
            binds1 += bind1
            if bind0 or bind1:
                leg_binds += 1
            clamped = walk(margin, ema, offered, clamped=True)
            unclamped = walk(margin, ema, offered, clamped=False)
            # The offline walk must reproduce the depth the session actually
            # chose. Any mismatch means the counterfactual below is fiction.
            if clamped != record["d"]:
                replay_mismatch += 1
            # A round that reached its cap was stopped from outside the walk;
            # any other round stopped on its own marginal threshold.
            if record["d"] >= cap:
                cap_stops += 1
                leg_cap_stops += 1
            if unclamped > clamped:
                depth_loss_rounds += 1
                leg_loss += 1
                depth_loss_sum += unclamped - clamped
                leg_loss_sum += unclamped - clamped
                # The clamp removed positions the round then never resolved.
                # A round that accepted every draft it DID make is a round
                # where the removed position was very likely acceptable too.
                if record["acc"] == record["d"]:
                    pure_loss_rounds += 1
        pooled_rounds.extend(leg["rounds"])
        depths = [r["d"] for r in leg["rounds"]]
        result["legs"].append(
            {
                "label": leg["label"],
                "counts": leg_bookkeeping(leg["rounds"]),
                "rounds": leg_total,
                "bind_fraction": leg_binds / leg_total if leg_total else 0.0,
                "depth_loss_fraction": leg_loss / leg_total if leg_total else 0.0,
                "mean_depth_loss": leg_loss_sum / leg_total if leg_total else 0.0,
                "cap_stop_fraction": (
                    leg_cap_stops / leg_total if leg_total else 0.0
                ),
                "mean_d": sum(depths) / len(depths) if depths else 0.0,
                "accept_fraction": (
                    sum(r["acc"] for r in leg["rounds"]) / sum(depths)
                    if sum(depths)
                    else 0.0
                ),
            }
        )
    result["pooled"] = {
        "rounds": total,
        "replay_mismatch": replay_mismatch,
        "bind_pos0_fraction": binds0 / total if total else 0.0,
        "bind_pos1_fraction": binds1 / total if total else 0.0,
        "depth_loss_fraction": depth_loss_rounds / total if total else 0.0,
        "mean_depth_removed": depth_loss_sum / total if total else 0.0,
        "pure_loss_fraction": pure_loss_rounds / total if total else 0.0,
        "cap_stop_fraction": cap_stops / total if total else 0.0,
        "cap_histogram": dict(sorted(cap_histogram.items())),
    }
    result["profile"] = position_profile(pooled_rounds)
    return result


def chain_survival(profile: list[float], position: int) -> float:
    """P(the draft at `position` is accepted) under a position-only model.

    Every position below `position` must also be accepted, so this is the
    running product of the conditional profile. Positions past the profile's
    reach reuse its last measured value; those tails are rare and the report
    states the reach so the extrapolation is visible.
    """
    if not profile:
        return float("nan")
    survival = 1.0
    for index in range(position + 1):
        survival *= profile[min(index, len(profile) - 1)]
    return survival


def expected_accepted(profile: list[float], depth: int) -> float:
    """Accepted drafts per round for a chain of `depth` proposals."""
    return sum(chain_survival(profile, k) for k in range(depth))


def profile_for(profiles: dict[str, list[float]], label: str) -> list[float]:
    """The prompt's OWN acceptance profile, never a pooled stand-in.

    Acceptance varies far more between prompts than between positions, so
    pricing a hard prompt with a pooled profile would import the easy prompts'
    acceptance into it and invert the answer. A prompt with no pinned twin is
    dropped rather than priced against someone else's text.
    """
    return profiles.get(label, [])


def clamp_counterfactual(
    adapt_legs: list[dict], profiles: dict[str, list[float]], offered_default: int
) -> dict:
    """Price the clamp itself: replay each round with and without it.

    The comparison holds the recorded margin, EMA and offered cap fixed and
    changes only whether the clamp term applies, so the depth difference is
    caused by the clamp alone. Accepted tokens cannot be observed for a depth
    the arm never took, so they come from the pinned arm's measured position
    profile. That model is the same for both policies, which makes the
    DIFFERENCE a fair comparison even where the absolute level is approximate.
    """
    per_leg = []
    for leg in adapt_legs:
        rounds = [r for r in leg["rounds"] if not math.isnan(r["margin"])]
        profile = profile_for(profiles, leg["label"])
        if not rounds or not profile:
            continue
        clamped_depth = unclamped_depth = 0
        predicted_on = predicted_off = 0.0
        for record in rounds:
            offered = (
                record["offer"] if record["offer"] is not None else offered_default
            )
            other = walk(record["margin"], record["ema"], offered, False)
            clamped_depth += record["d"]
            unclamped_depth += other
            predicted_on += expected_accepted(profile, record["d"])
            predicted_off += expected_accepted(profile, other)
        count = len(rounds)
        entry = {
            "label": leg["label"],
            "rounds": count,
            "mean_depth_clamped": clamped_depth / count,
            "mean_depth_unclamped": unclamped_depth / count,
            "predicted_accepted_clamped": predicted_on / count,
            "predicted_accepted_unclamped": predicted_off / count,
            # Validation: the SAME model, run on the depths the arm actually
            # took, must reproduce the accepted tokens the arm actually got.
            # Without this the unclamped number is an unchecked extrapolation.
            "measured_accepted": sum(r["acc"] for r in rounds) / count,
        }
        entry["model_error"] = (
            entry["predicted_accepted_clamped"] - entry["measured_accepted"]
        )
        entry["raw_clamped"] = ranked_price(
            entry["mean_depth_clamped"], entry["predicted_accepted_clamped"]
        )["raw"]
        entry["raw_unclamped"] = ranked_price(
            entry["mean_depth_unclamped"], entry["predicted_accepted_unclamped"]
        )["raw"]
        entry["raw_gain_from_clamp"] = entry["raw_clamped"] - entry["raw_unclamped"]
        per_leg.append(entry)
    pooled: dict = {}
    if per_leg:
        total = sum(entry["rounds"] for entry in per_leg)

        def weighted(key: str) -> float:
            return sum(e[key] * e["rounds"] for e in per_leg) / total

        pooled = {
            "rounds": total,
            "mean_depth_clamped": weighted("mean_depth_clamped"),
            "mean_depth_unclamped": weighted("mean_depth_unclamped"),
            "mean_model_error": weighted("model_error"),
            "mean_raw_gain_from_clamp": weighted("raw_gain_from_clamp"),
        }
    return {"pooled": pooled, "legs": per_leg}


RANKED_COST_RATIO = RANKED_ROW_MS / (RANKED_ROUND_FIXED_MS + RANKED_ROW_MS)

# Local M4 Pro round law, sealed gated E159 512-token sweep (qwen-askeladd PR
# 158, base 3cdab4e2), raw form. It is used only to PREDICT local decode time,
# never to price ranked value.
LOCAL_ROUND_FIXED_MS = 29.367
LOCAL_ROW_MS = 14.775
LOCAL_COST_RATIO = LOCAL_ROW_MS / (LOCAL_ROUND_FIXED_MS + LOCAL_ROW_MS)


def local_decode_ms_per_token(mean_depth: float, mean_accepted: float) -> float:
    """Predicted local decode ms per token, prefill excluded."""
    return (LOCAL_ROUND_FIXED_MS + LOCAL_ROW_MS * (1.0 + mean_depth)) / (
        1.0 + mean_accepted
    )


def policy_sweep(
    adapt_legs: list[dict],
    profiles: dict[str, list[float]],
    offered_default: int,
    fitted: dict[int, float] | None = None,
) -> dict:
    """Price alternative controllers on the recorded rounds, without a GPU.

    Every policy sees the same recorded margin, EMA and offered cap, and every
    policy is priced through the same measured position profile and the same
    ranked round law. The comparison therefore isolates the decision rule. It
    inherits the profile's assumption that acceptance depends on position and
    not on the round, which is exactly the assumption the AUC and skill tests
    above are there to check.
    """
    def offered_of(record: dict) -> int:
        return record["offer"] if record["offer"] is not None else offered_default

    policies: list[tuple[str, object]] = [
        ("shipped (clamp, ratio 0.180)", lambda r: r["d"]),
        (
            "no clamp, ratio 0.180",
            lambda r: walk(r["margin"], r["ema"], offered_of(r), False),
        ),
        (
            f"no clamp, ranked ratio {RANKED_COST_RATIO:.3f}",
            lambda r: walk(
                r["margin"], r["ema"], offered_of(r), False, ratio=RANKED_COST_RATIO
            ),
        ),
        (
            f"clamp, ranked ratio {RANKED_COST_RATIO:.3f}",
            lambda r: walk(
                r["margin"], r["ema"], offered_of(r), True, ratio=RANKED_COST_RATIO
            ),
        ),
        (
            f"clamp, local ratio {LOCAL_COST_RATIO:.3f}",
            lambda r: walk(
                r["margin"], r["ema"], offered_of(r), True, ratio=LOCAL_COST_RATIO
            ),
        ),
    ]
    if fitted:
        policies.append(
            (
                "clamp at fitted T, ratio 0.180",
                lambda r: walk(r["margin"], r["ema"], offered_of(r), True, temps=fitted),
            )
        )
    for fixed in range(0, min(offered_default, WIDTH_CAP) + 1):
        policies.append((f"fixed depth {fixed}", lambda r, f=fixed: f))

    per_prompt: dict[str, list[dict]] = {}
    for leg in adapt_legs:
        rounds = [r for r in leg["rounds"] if not math.isnan(r["margin"])]
        profile = profile_for(profiles, leg["label"])
        if not rounds or not profile:
            continue
        rows = []
        for name, rule in policies:
            depths = [rule(record) for record in rounds]
            mean_depth = sum(depths) / len(depths)
            accepted = sum(expected_accepted(profile, d) for d in depths) / len(depths)
            rows.append(
                {
                    "policy": name,
                    "mean_depth": mean_depth,
                    "predicted_accepted": accepted,
                    "raw": ranked_price(mean_depth, accepted)["raw"],
                    "local_decode_ms_per_token": local_decode_ms_per_token(
                        mean_depth, accepted
                    ),
                }
            )
        best = max(row["raw"] for row in rows)
        for row in rows:
            row["raw_deficit_vs_best"] = row["raw"] - best
        per_prompt[leg["label"]] = rows

    # The published score is a MEDIAN of per-prompt ratios, so a policy is
    # summarised by the median of its per-prompt `raw` and never by a pooled
    # round set, which would let the prompts with the most rounds decide.
    pooled = []
    for index, (name, _) in enumerate(policies):
        values = sorted(rows[index]["raw"] for rows in per_prompt.values())
        if not values:
            continue
        middle = len(values) // 2
        median = (
            values[middle]
            if len(values) % 2
            else (values[middle - 1] + values[middle]) / 2.0
        )
        pooled.append(
            {
                "policy": name,
                "median_raw": median,
                "min_raw": values[0],
                "max_raw": values[-1],
                "prompts": len(values),
            }
        )
    if pooled:
        best = max(row["median_raw"] for row in pooled)
        for row in pooled:
            row["median_deficit_vs_best"] = row["median_raw"] - best
    return {"per_prompt": per_prompt, "pooled": pooled}


def splice_arms(
    adapt_legs: list[dict],
    pinned_legs: list[dict],
    profiles: dict[str, list[float]] | None = None,
) -> dict:
    """Answer the counterfactual DIRECTLY instead of by proxy.

    The two arms decode the same prompt to the same serial tokens, so absolute
    token position `p` names the same token in both. A round is spliceable when
    both arms began a round at the same position: the pinned arm then drafted
    positions the shipped clamp refused, and its accept walk says outright
    whether the refused position was acceptable.

    Two invariants are checked before any splice is believed:
      rows     the target's top-2 ids and raw logit bits at every shared
               position must be identical. They are produced by the fixed
               target on the same prefix, so a difference means the arms are
               not decoding the same stream and no splice is valid.
      margins  a matched round's pending-primary margin must agree, which
               confirms the round-start alignment is real and not a coincidence
               of cumulative sums.
    """
    pinned_by_label = {leg["label"]: leg for leg in pinned_legs}
    totals = {
        "legs_matched": 0,
        "rows_compared": 0,
        "rows_differing": 0,
        "rounds_matched": 0,
        "margin_mismatch": 0,
        "accept_inconsistent": 0,
        "clamp_removed_rounds": 0,
        "removed_resolved": 0,
        "removed_accepted": 0,
        "extra_tokens_available": 0,
        "removed_predicted": 0.0,
    }
    per_leg = []
    for leg in adapt_legs:
        pinned = pinned_by_label.get(leg["label"])
        if pinned is None:
            continue
        profile = profile_for(profiles or {}, leg["label"])
        totals["legs_matched"] += 1
        shared = set(leg["rows"]) & set(pinned["rows"])
        differing = [
            p for p in sorted(shared) if leg["rows"][p] != pinned["rows"][p]
        ]
        index = {r["start"]: r for r in pinned["rounds"] if r["start"] is not None}
        entry = {
            "label": leg["label"],
            "rows_compared": len(shared),
            "rows_differing": len(differing),
            "first_differing_row": differing[0] if differing else None,
            "rounds_matched": 0,
            "margin_mismatch": 0,
            "accept_inconsistent": 0,
            "clamp_removed_rounds": 0,
            "removed_resolved": 0,
            "removed_accepted": 0,
            "extra_tokens_available": 0,
            "removed_predicted": 0.0,
        }
        for record in leg["rounds"]:
            twin = index.get(record["start"])
            if twin is None or math.isnan(record["margin"]):
                continue
            entry["rounds_matched"] += 1
            if abs(record["margin"] - twin["margin"]) > 1e-3:
                entry["margin_mismatch"] += 1
            depth = min(record["d"], twin["d"])
            reject_a = record["acc"] if record["acc"] < record["d"] else None
            reject_p = twin["acc"] if twin["acc"] < twin["d"] else None
            first_a = reject_a if reject_a is not None and reject_a < depth else None
            first_p = reject_p if reject_p is not None and reject_p < depth else None
            if first_a != first_p:
                entry["accept_inconsistent"] += 1
                continue
            offered = record["offer"] if record["offer"] is not None else MAX_DEPTH
            unclamped = walk(record["margin"], record["ema"], offered, clamped=False)
            if unclamped <= record["d"]:
                continue
            entry["clamp_removed_rounds"] += 1
            # The clamp refused positions d .. unclamped-1. The pinned twin
            # drafted them if it went deeper, and its accept count says whether
            # the first refused position was acceptable.
            if twin["d"] <= record["d"]:
                continue
            entry["removed_resolved"] += 1
            # Skill test. A clamp that reads the round correctly must refuse
            # positions that were LESS acceptable than a position-only rule
            # predicts. The prediction uses the pinned arm's profile at the
            # same position, so a match means the margin added nothing beyond
            # knowing how deep the round already was.
            if profile:
                entry["removed_predicted"] += chain_survival(profile, record["d"])
            if twin["acc"] > record["d"]:
                entry["removed_accepted"] += 1
                entry["extra_tokens_available"] += (
                    min(twin["acc"], unclamped, twin["d"]) - record["d"]
                )
        per_leg.append(entry)
        for key in totals:
            if key != "legs_matched" and key in entry:
                totals[key] += entry[key]
    return {"pooled": totals, "legs": per_leg}


def print_profile(title: str, profile: list[dict]) -> None:
    print(f"  per-position conditional acceptance, {title}:")
    for row in profile:
        print(
            f"    q[{row['position']}] = {row['q']:.4f}  "
            f"[{row['ci_low']:.4f},{row['ci_high']:.4f}]  "
            f"n={row['reached']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinned", nargs="*", default=[])
    parser.add_argument("--adapt", nargs="*", default=[])
    parser.add_argument("--offered", type=int, default=8)
    parser.add_argument("--json")
    args = parser.parse_args()

    report: dict = {"offered_depth": args.offered}
    pinned_legs = load_legs(args.pinned, "pinned") if args.pinned else []
    adapt_legs = load_legs(args.adapt, "adapt") if args.adapt else []
    if pinned_legs:
        report["pinned"] = analyse_pinned(pinned_legs)
    if adapt_legs:
        report["adapt"] = analyse_adapt(adapt_legs, args.offered)
    if pinned_legs and adapt_legs:
        profiles = {
            label: [row["q"] for row in rows]
            for label, rows in report["pinned"]["profile_by_leg"].items()
        }
        report["position_profiles_used"] = profiles
        report["splice"] = splice_arms(adapt_legs, pinned_legs, profiles)
        report["arms"] = compare_arms(pinned_legs, adapt_legs)
        report["clamp_counterfactual"] = clamp_counterfactual(
            adapt_legs, profiles, args.offered
        )
        pooled_pinned = [r for leg in pinned_legs for r in leg["rounds"]]
        report["signal_auc"] = {
            position: signal_auc(pooled_pinned, position) for position in (0, 1, 2)
        }
        report["oracle_ceiling"] = oracle_ceiling(pinned_legs)
        report["autocorrelation"] = {
            position: history_structure(pinned_legs, position)
            for position in (0, 1)
        }
        report["policy_sweep"] = policy_sweep(
            adapt_legs,
            profiles,
            args.offered,
            {
                position: block["fitted_temperature"]
                for position, block in report["pinned"]["positions"].items()
            },
        )

    if "arms" in report:
        print("=== per prompt, both arms, counts then ranked price ===")
        print(
            f"{'prompt':<18}{'arm':<7}{'N':>5}{'depth':>7}{'A':>7}"
            f"{'acc/prop':>10}{'capstop':>9}{'tok/rnd':>9}{'raw':>8}"
        )
        for row in report["arms"]["prompts"]:
            for arm in ("adapt", "pinned"):
                counts = row[arm]
                cap_stop = counts["cap_stop_fraction"]
                print(
                    f"{row['label'] if arm == 'adapt' else '':<18}{arm:<7}"
                    f"{counts['rounds']:>5}{counts['mean_depth']:>7.3f}"
                    f"{counts['mean_accepted']:>7.3f}"
                    f"{counts['accept_fraction']:>10.4f}"
                    f"{('n/a' if cap_stop is None else f'{cap_stop:.3f}'):>9}"
                    f"{counts['tokens_per_round']:>9.3f}"
                    f"{row['raw_adapt' if arm == 'adapt' else 'raw_pinned']:>8.3f}"
                )
            decay = row["pinned_decay"]
            print(
                f"{'':<18}{'->':<7}raw_p7/raw_adapt {row['raw_ratio']:.4f}   "
                f"selection gain in acc/prop {row['selection_gain']:+.4f}   "
                f"pinned decay/position "
                f"{decay.get('decay_per_position', float('nan')):.4f}   "
                f"halves {row['pinned']['accept_fraction_first_half']:.3f}"
                f"/{row['pinned']['accept_fraction_second_half']:.3f}"
            )
        print()
        print("=== chosen-depth histogram and the easy/hard split, adaptive arm ===")
        for row in report["arms"]["prompts"]:
            counts = row["adapt"]
            at_cap = counts["at_cap_fraction"]
            print(
                f"  {row['label']:<20} at_cap="
                f"{('n/a' if at_cap is None else f'{at_cap:.3f}')}  "
                f"depth histogram {counts['depth_histogram']}"
            )
            split = "  ".join(
                f"d={depth}: n={bucket['rounds']} "
                f"acc/prop={bucket['accept_fraction']:.3f}"
                for depth, bucket in counts["depth_bins"].items()
            )
            print(f"  {'':<20} {split}")
        print()

    if "clamp_counterfactual" in report:
        counter = report["clamp_counterfactual"]
        print("=== what the clamp itself buys: same rounds, clamp on vs off ===")
        print(
            f"{'prompt':<20}{'d_on':>7}{'d_off':>7}{'A_model':>9}{'A_real':>8}"
            f"{'err':>8}{'raw_on':>8}{'raw_off':>9}{'raw gain':>10}"
        )
        for entry in counter["legs"]:
            print(
                f"{entry['label']:<20}{entry['mean_depth_clamped']:>7.3f}"
                f"{entry['mean_depth_unclamped']:>7.3f}"
                f"{entry['predicted_accepted_clamped']:>9.3f}"
                f"{entry['measured_accepted']:>8.3f}{entry['model_error']:>+8.3f}"
                f"{entry['raw_clamped']:>8.3f}{entry['raw_unclamped']:>9.3f}"
                f"{entry['raw_gain_from_clamp']:>+10.3f}"
            )
        if counter["pooled"]:
            print(
                f"{'POOLED':<20}{counter['pooled']['mean_depth_clamped']:>7.3f}"
                f"{counter['pooled']['mean_depth_unclamped']:>7.3f}"
                f"{'':>9}{'':>8}{counter['pooled']['mean_model_error']:>+8.3f}"
                f"{'':>8}{'':>9}"
                f"{counter['pooled']['mean_raw_gain_from_clamp']:>+10.3f}"
            )
        print()

    if report.get("signal_auc"):
        print("=== can ANY recorded signal rank acceptance? (pinned arm) ===")
        print(f"{'position':<9}{'n':>5}{'base':>7}   AUC with 95% interval")
        for position, block in report["signal_auc"].items():
            if not block:
                continue
            cells = "  ".join(
                f"{name} {s['auc']:.3f} [{s['ci_low']:.3f},{s['ci_high']:.3f}]"
                for name, s in block["auc"].items()
            )
            print(
                f"{position:<9}{block['n']:>5}{block['base_rate']:>7.3f}   {cells}"
            )
        print("0.5 is chance; a useful controller signal needs clearly more.")
        print()

    if report.get("autocorrelation"):
        print("=== is acceptance predictable from its own history? ===")
        for position, block in report["autocorrelation"].items():
            for series, rows in block.items():
                if not rows:
                    continue
                cells = "  ".join(
                    f"lag{lag} {row['rho']:+.3f}"
                    f"{'*' if row['significant'] else ' '}"
                    for lag, row in rows["lags"].items()
                )
                band = next(iter(rows["lags"].values()))["band"]
                print(
                    f"  position {position} {series:<7} n={rows['n']:<5}"
                    f"band +-{band:.3f}   {cells}"
                )
        print(
            "  An EMA or a streak can only work on an autocorrelated sequence.\n"
            "  The margin row is descriptive: it says whether a smoothed margin\n"
            "  could carry information where the instantaneous one does not.\n"
            "  * marks a lag outside Bartlett's white-noise band."
        )
        print()

    if report.get("oracle_ceiling"):
        print("=== oracle ceiling: perfect per-round foresight ===")
        for entry in report["oracle_ceiling"]:
            print(
                f"  {entry['label']:<20} mean d = mean A = "
                f"{entry['mean_depth']:.3f}  raw {entry['raw']:.3f}"
            )
        print()

    if report.get("policy_sweep", {}).get("pooled"):
        sweep = report["policy_sweep"]
        print("=== controllers priced on the same recorded rounds ===")
        print("median of per-prompt raw, which is the published score's shape")
        print(
            f"{'policy':<34}{'median':>9}{'min':>8}{'max':>8}"
            f"{'vs best':>10}{'prompts':>9}"
        )
        for row in sweep["pooled"]:
            print(
                f"{row['policy']:<34}{row['median_raw']:>9.3f}"
                f"{row['min_raw']:>8.3f}{row['max_raw']:>8.3f}"
                f"{row['median_deficit_vs_best']:>+10.3f}{row['prompts']:>9}"
            )
        print()
        for label, rows in sweep["per_prompt"].items():
            best = max(rows, key=lambda row: row["raw"])
            shipped = rows[0]
            print(
                f"  {label:<20} shipped raw {shipped['raw']:.3f} at d="
                f"{shipped['mean_depth']:.2f}   best '{best['policy']}' raw "
                f"{best['raw']:.3f}   gap {best['raw'] - shipped['raw']:+.3f}"
            )
        print()

    if "adapt" in report:
        pooled = report["adapt"]["pooled"]
        print(f"=== shipped adaptive policy, {pooled['rounds']} rounds ===")
        print(f"offline walk replay mismatches: {pooled['replay_mismatch']}")
        print(
            f"clamp binds at pos0 {pooled['bind_pos0_fraction']:.3f}  "
            f"pos1 {pooled['bind_pos1_fraction']:.3f}"
        )
        print(
            f"rounds where the clamp REMOVED depth "
            f"{pooled['depth_loss_fraction']:.3f}, "
            f"mean rows removed/round {pooled['mean_depth_removed']:.3f}"
        )
        print(
            f"of those, rounds that then accepted every draft they made "
            f"{pooled['pure_loss_fraction']:.3f}"
        )
        print(
            f"rounds that stopped ON THE CAP rather than on the threshold "
            f"{pooled['cap_stop_fraction']:.3f}; "
            f"effective cap histogram {pooled['cap_histogram']}"
        )
        for leg in report["adapt"]["legs"]:
            print(
                f"  {leg['label']:<24} n={leg['rounds']:<4} "
                f"bind={leg['bind_fraction']:.3f} "
                f"depth_loss={leg['depth_loss_fraction']:.3f} "
                f"rows_removed={leg['mean_depth_loss']:.3f} "
                f"cap_stop={leg['cap_stop_fraction']:.3f} "
                f"mean_d={leg['mean_d']:.3f} "
                f"acc_frac={leg['accept_fraction']:.3f}"
            )
            counts = leg["counts"]
            print(f"  {'':<24} depth histogram {counts['depth_histogram']}")
            print(
                f"  {'':<24} "
                + "  ".join(
                    f"d={depth}: n={bucket['rounds']} "
                    f"acc/prop={bucket['accept_fraction']:.3f} "
                    f"q0={bucket['q0']:.3f} m={bucket['mean_margin']:.2f}"
                    for depth, bucket in counts["depth_bins"].items()
                    if depth > 0
                )
            )
        print_profile("shipped adaptive", report["adapt"]["profile"])
        print()

    if "pinned" in report:
        print("=== pinned-depth legs (uncensored) ===")
        for leg in report["pinned"]["legs"]:
            print(
                f"  {leg['label']:<24} rounds={leg['rounds']:<4} "
                f"mean_d={leg['mean_d']:.2f} mean_acc={leg['mean_acc']:.3f} "
                f"accept_frac={leg['accept_fraction']:.3f} "
                f"mean_margin={leg['margin_mean']:.2f}"
            )
        print_profile("pinned depth (pooled)", report["pinned"]["profile"])
        for label, profile in report["pinned"]["profile_by_leg"].items():
            values = " ".join(f"{row['q']:.4f}" for row in profile)
            print(f"  q[{label:<20}] {values}")
        for position in (0, 1):
            block = report["pinned"]["positions"][position]
            print(
                f"\n--- position {position}: n={block['n']} "
                f"base_rate={block['base_rate']:.4f} "
                f"AUC(margin)={block['auc_margin']:.4f} ---"
            )
            print(
                f"shipped T={block['shipped_temperature']:.2f}  "
                f"fitted T*={block['fitted_temperature']:.4g}"
            )
            print(
                f"{'bin':<12}{'n':>6}{'mean_m':>9}{'empirical':>11}"
                f"{'95% CI':>18}{'clamp':>9}{'mean_ema':>10}"
            )
            for row in block["table"]:
                interval = f"[{row['ci_low']:.3f},{row['ci_high']:.3f}]"
                print(
                    f"{row['bin']:<12}{row['n']:>6}{row['mean_margin']:>9.3f}"
                    f"{row['empirical']:>11.4f}{interval:>18}"
                    f"{row['clamp_sigmoid']:>9.4f}{row['mean_ema']:>10.4f}"
                )
            print(f"{'predictor':<20}{'log_loss':>11}{'brier':>10}{'mean_p':>10}")
            for name, scores in block["predictors"].items():
                if scores.get("n"):
                    print(
                        f"{name:<20}{scores['log_loss']:>11.5f}"
                        f"{scores['brier']:>10.5f}{scores['mean_prediction']:>10.4f}"
                    )

    if "splice" in report:
        pooled = report["splice"]["pooled"]
        print("\n=== arm splice: the counterfactual measured, not modelled ===")
        print(
            f"target rows compared across arms {pooled['rows_compared']}, "
            f"differing {pooled['rows_differing']}"
        )
        print(
            f"rounds that began at the same absolute token in both arms "
            f"{pooled['rounds_matched']} "
            f"(margin mismatches {pooled['margin_mismatch']}, "
            f"accept inconsistencies {pooled['accept_inconsistent']})"
        )
        print(
            f"of those, clamp removed depth in {pooled['clamp_removed_rounds']}; "
            f"the pinned twin resolved the first removed position in "
            f"{pooled['removed_resolved']}; it was ACCEPTED in "
            f"{pooled['removed_accepted']}"
        )
        if pooled["removed_resolved"]:
            low, high = wilson(
                pooled["removed_accepted"], pooled["removed_resolved"]
            )
            print(
                f"pure-loss rate {pooled['removed_accepted'] / pooled['removed_resolved']:.4f} "
                f"[{low:.4f},{high:.4f}]  "
                f"tokens the clamp gave up {pooled['extra_tokens_available']}"
            )
            predicted = pooled["removed_predicted"] / pooled["removed_resolved"]
            observed = pooled["removed_accepted"] / pooled["removed_resolved"]
            print(
                f"skill test: a position-only rule predicts {predicted:.4f}, "
                f"the clamp's refusals actually paid {observed:.4f}  "
                f"(skill {observed - predicted:+.4f}; negative means the clamp "
                f"refused better-than-average positions)"
            )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
