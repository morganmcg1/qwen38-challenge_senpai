#!/usr/bin/env python3
"""E134 rung 1 - what does the scheduler NOT know that it could know for free?

harness=local instrument, ranked target. Zero GPU.

Every E128 arm reshaped the same three inputs: `positionAcceptEMA`, the tail
top-2 margin and the depth price. This rung asks whether new INFORMATION, which
the round already computes and then throws away, discriminates the accept
outcome at each depth boundary better than the shipped input set does.

The information is already on disk. `Qwen36MTPBlockSession.swift:1598-1608`
calls `traceRow` for rows `0 ... acceptedCount` of every round, and `:783-787`
writes each one as

    mtp-row: pos=<abs> ids=<t1>,<t2> v=<hexfloat>,<hexfloat>

so the archived forced-depth traces carry the exact top-2 target evidence at
every position the round verified, not only the tail the scheduler reads
through `pendingTop2`. The rows are emitted BEFORE that round's `mtp-trace`
line, and there are exactly `acc + 1` of them.

Two parse gates run before any AUC:

  1. `acc + 1` rows per round, always.
  2. round `t`'s shipped `m=` equals the LAST row margin of round `t - 1`. That
     is the identity that proves the alignment, because the pending primary
     comes from row `acceptedCount`.

A third gate reproduces the shipped `sched=` trace field from the parsed EMAs
and margin, so the reconstructed shipped statistic is the shipped one.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e128_ourcurve import F83_WEIGHT  # noqa: E402
from e128_signals import auc  # noqa: E402

# `Qwen36MTPBlockSession.swift:840, 871-878, 958`, and `Constants.swift:331`.
HEAD_STEP_COST_RATIO = 0.18
MAX_DEPTH = 8
PRICE_MARGINAL = [HEAD_STEP_COST_RATIO] * MAX_DEPTH
PRICE_CUMULATIVE = [1.0 + d * HEAD_STEP_COST_RATIO for d in range(MAX_DEPTH + 1)]
# The shipped `min(p, conf)` override divisors at depths 0 and 1.
MARGIN_DIVISOR = {0: 2.0, 1: 3.0}

PROMPT_FIXTURES = {
    "beagle": ["beagle_a", "beagle_b"],
    "medicine": ["medicine_hist", "medicine_hippoc"],
    "essays": ["essays_bacon", "essays_montaigne"],
    "botany": ["botany_andrews"],
    "republic": ["republic_jowett"],
    "drama": ["drama_dollhouse"],
    "travel": ["travel_eothen"],
    "plutarch": ["plutarch_lives"],
}
FIXTURE_PROMPT = {f: p for p, fs in PROMPT_FIXTURES.items() for f in fs}

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?"
    r"arm=(\S+) m=(\S+) streak=(\d+) cap=(\d+) ema=([0-9.,]+)(.*)$")
SCHED_RE = re.compile(r"sched=(\S+)")
ROW_RE = re.compile(r"mtp-row: pos=(\d+) ids=(\d+),(\d+) v=(\S+),(\S+)")

SHIPPED = ("margin", "ema_d", "reach_shipped")
CANDIDATES = (
    "prev_margin_at_d", "prev_margin_mean", "prev_margin_min",
    "prev_margin_first", "prev_margin_slope", "prev_margin_last",
    "prev_len", "streak", "prev_acc", "width_mean3", "km_reach",
)
EXTRA_COLUMNS = [
    "ALL_PREV_MARGIN", "ALL_NEW", "NULL_FLOOR",
    "HELDOUT_ALL_NEW", "HELDOUT_PREV_MARGIN",
]
SHORT = {
    "prev_margin_at_d": "pm_at_d",
    "prev_margin_mean": "pm_mean",
    "prev_margin_min": "pm_min",
    "prev_margin_first": "pm_first",
    "prev_margin_slope": "pm_slope",
    "prev_margin_last": "pm_last",
    "prev_len": "prev_len",
    "streak": "streak",
    "prev_acc": "prev_acc",
    "width_mean3": "wmean3",
    "km_reach": "km_reach",
    "ALL_PREV_MARGIN": "ALLPM",
    "ALL_NEW": "ALLNEW",
    "NULL_FLOOR": "null",
    "HELDOUT_ALL_NEW": "heldnew",
    "HELDOUT_PREV_MARGIN": "heldpm",
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def parse_trace(path: pathlib.Path) -> tuple[list[dict], dict]:
    rounds, pending = [], []
    for line in path.read_text().splitlines():
        row = ROW_RE.search(line)
        if row is not None:
            pending.append({
                "pos": int(row.group(1)),
                "margin": float.fromhex(row.group(4))
                - float.fromhex(row.group(5))})
            continue
        hit = ROUND_RE.search(line)
        if hit is None:
            continue
        sched = SCHED_RE.search(hit.group(9))
        rounds.append({
            "round": int(hit.group(1)),
            "depth": int(hit.group(2)),
            "acc": int(hit.group(3)),
            "margin": float(hit.group(5)),
            "streak": int(hit.group(6)),
            "cap": int(hit.group(7)),
            "ema": [float(v) for v in hit.group(8).split(",")],
            "rows": [r["margin"] for r in pending],
            "sched": sched.group(1) if sched else "",
        })
        pending = []

    gate = {"rounds": len(rounds), "row_count_ok": 0, "row_count_bad": 0,
            "margin_identity_ok": 0, "margin_identity_bad": 0,
            "sched_max_abs_error": 0.0, "sched_checked": 0}
    for index, record in enumerate(rounds):
        if len(record["rows"]) == record["acc"] + 1:
            gate["row_count_ok"] += 1
        else:
            gate["row_count_bad"] += 1
        if index > 0 and rounds[index - 1]["rows"]:
            want = rounds[index - 1]["rows"][-1]
            if abs(record["margin"] - want) < 1e-9:
                gate["margin_identity_ok"] += 1
            else:
                gate["margin_identity_bad"] += 1
        gate["sched_max_abs_error"] = max(
            gate["sched_max_abs_error"], check_sched(record, gate))
    return rounds, gate


def shipped_path(record: dict) -> tuple[list[float], list[float], list[float]]:
    """The shipped per-depth p, reach and threshold, from the trace inputs."""
    ema, margin = record["ema"], record["margin"]
    probs, reach, thresholds = [], [], []
    running, expected = 1.0, 0.0
    for depth in range(MAX_DEPTH):
        p = ema[depth] if depth < len(ema) else ema[-1]
        div = MARGIN_DIVISOR.get(depth)
        if div is not None:
            p = min(p, sigmoid(margin / div))
        running *= p
        probs.append(p)
        reach.append(running)
        thresholds.append(PRICE_MARGINAL[depth] * (1.0 + expected)
                          / PRICE_CUMULATIVE[depth])
        expected += running
    return probs, reach, thresholds


def check_sched(record: dict, gate: dict) -> float:
    if not record["sched"]:
        return 0.0
    probs, reach, thresholds = shipped_path(record)
    worst = 0.0
    for chunk in record["sched"].split(";"):
        if not chunk:
            continue
        head, _, rest = chunk.partition(":")
        parts = rest.split("/")
        if len(parts) != 3:
            continue
        depth = int(head)
        got = (float(parts[0]), float(parts[1]), float(parts[2]))
        want = (probs[depth], reach[depth], thresholds[depth])
        worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
        gate["sched_checked"] += 1
    return worst


def km_reach(rounds: list[dict], index: int, depth: int) -> float:
    """Censoring-aware reach: Kaplan-Meier over this run's history so far.

    The realised accepted counts are right-censored by the offered depth, so a
    naive mean of `acc` understates reach. This uses only rounds strictly
    before `index`, so it stays a legal round-start signal.
    """
    survivors = 1.0
    for step in range(depth + 1):
        at_risk = hazard = 0
        for past in rounds[:index]:
            if past["depth"] <= step or past["acc"] < step:
                continue
            at_risk += 1
            if past["acc"] == step:
                hazard += 1
        if at_risk == 0:
            return survivors * (0.85 * 0.98 ** step)
        survivors *= 1.0 - hazard / at_risk
    return survivors


def features_at(rounds: list[dict], depth: int) -> dict:
    cols = {name: [] for name in SHIPPED + CANDIDATES}
    cols["label"] = []
    for index, record in enumerate(rounds):
        if record["depth"] <= depth or record["acc"] < depth:
            continue
        if index == 0:
            continue  # no previous round, so no margin vector
        prev = rounds[index - 1]
        vec = np.array(prev["rows"], dtype=float)
        if vec.size == 0:
            continue
        _, reach, _ = shipped_path(record)
        ema = record["ema"]
        cols["margin"].append(record["margin"])
        cols["ema_d"].append(ema[depth] if depth < len(ema) else ema[-1])
        cols["reach_shipped"].append(reach[depth])

        cols["prev_margin_at_d"].append(
            float(vec[depth]) if depth < vec.size else float(vec[-1]))
        cols["prev_margin_mean"].append(float(vec.mean()))
        cols["prev_margin_min"].append(float(vec.min()))
        cols["prev_margin_first"].append(float(vec[0]))
        cols["prev_margin_last"].append(float(vec[-1]))
        if vec.size >= 2:
            idx = np.arange(vec.size, dtype=float)
            slope = float(np.polyfit(idx, vec, 1)[0])
        else:
            slope = 0.0
        cols["prev_margin_slope"].append(slope)
        cols["prev_len"].append(float(vec.size))
        cols["streak"].append(float(record["streak"]))
        cols["prev_acc"].append(float(prev["acc"]))
        window = [r["acc"] for r in rounds[max(0, index - 3):index]]
        cols["width_mean3"].append(float(np.mean(window)) if window else 0.0)
        cols["km_reach"].append(km_reach(rounds, index, depth))
        cols["label"].append(1.0 if record["acc"] > depth else 0.0)
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> tuple:
    """IRLS with a small ridge, on standardised columns plus an intercept."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    beta = np.zeros(z.shape[1])
    penalty = ridge * np.eye(z.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(60):
        eta = np.clip(z @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        grad = z.T @ (y - p) - penalty @ beta
        hess = (z * w[:, None]).T @ z + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta, mu, sd


def score_with(beta, mu, sd, x: np.ndarray) -> np.ndarray:
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    return z @ beta


def fit_auc(cols: dict, names, labels: np.ndarray) -> float:
    x = np.column_stack([cols[n] for n in names])
    beta, mu, sd = logistic(x, labels)
    return auc(score_with(beta, mu, sd, x), labels)[0]


def split_auc(cols: dict, names, labels: np.ndarray) -> float:
    """Fit on the first half of the decode window, score on the second.

    In-sample AUC with fourteen columns on ~100 rounds is inflated by
    construction. This is the cheapest honest check available inside one
    fixture; rung 2 does leave-one-fixture-out.
    """
    cut = len(labels) // 2
    if cut < 15 or len(labels) - cut < 15:
        return float("nan")
    train = slice(0, cut)
    test = slice(cut, len(labels))
    if labels[train].min() == labels[train].max():
        return float("nan")
    if labels[test].min() == labels[test].max():
        return float("nan")
    x = np.column_stack([cols[n] for n in names])
    beta, mu, sd = logistic(x[train], labels[train])
    return auc(score_with(beta, mu, sd, x[test]), labels[test])[0]


def null_incremental(cols: dict, labels: np.ndarray, draws: int,
                     seed: int) -> float:
    """In-sample incremental AUC when the new columns carry no signal.

    The labels stay put and the NEW columns are permuted together, so the
    shipped baseline is unchanged and only the added block is broken. The mean
    of this is the overfitting floor that `ALL_NEW` must clear.
    """
    rng = np.random.default_rng(seed)
    base = fit_auc(cols, SHIPPED, labels)
    order = np.arange(len(labels))
    values = []
    for _ in range(draws):
        rng.shuffle(order)
        shuffled = dict(cols)
        for name in CANDIDATES:
            shuffled[name] = cols[name][order]
        values.append(fit_auc(shuffled, SHIPPED + CANDIDATES, labels) - base)
    return float(np.mean(values))


def pooled_blocks(traces: dict, weight: dict, depth: int) -> dict:
    """Per-fixture feature blocks at one boundary, keyed by fixture."""
    per = {}
    for fixture, rounds in sorted(traces.items()):
        if fixture not in weight:
            continue
        cols = features_at(rounds, depth)
        labels = cols["label"]
        if len(labels) < 30 or labels.min() == labels.max():
            continue
        per[fixture] = cols
    return per


def pooled_folds(per: dict, weight: dict, names, held_out: bool) -> dict:
    """One AUC per scored fixture under a single pooled global policy.

    With `held_out`, the scored fixture is removed from the training set, so
    each value is a genuine out-of-sample AUC.
    """
    folds = {}
    keys = sorted(per)
    for scored in keys:
        train = [k for k in keys if k != scored] if held_out else keys
        y_train = np.concatenate([per[k]["label"] for k in train])
        if y_train.min() == y_train.max():
            continue
        x_train = np.concatenate(
            [np.column_stack([per[k][n] for n in names]) for k in train])
        beta, mu, sd = logistic(x_train, y_train)
        y_test = per[scored]["label"]
        if y_test.min() == y_test.max():
            continue
        x_test = np.column_stack([per[scored][n] for n in names])
        folds[scored] = (auc(score_with(beta, mu, sd, x_test), y_test)[0],
                         weight[scored] * len(y_test))
    return folds


def weighted_mean(folds: dict) -> float:
    den = sum(w for _, w in folds.values())
    if not den:
        return float("nan")
    return sum(v * w for v, w in folds.values()) / den


def pooled_auc(per: dict, weight: dict, names, held_out: bool) -> float:
    return weighted_mean(pooled_folds(per, weight, names, held_out))


def is_usable(per: dict, name: str) -> bool:
    """A column is usable when the POOLED column varies.

    A column that is flat inside one fixture is still informative to a single
    global policy, so the constancy test belongs on the pooled column.
    """
    stacked = np.concatenate([cols[name] for cols in per.values()])
    return bool(stacked.std() > 1e-12)


def pooled_null(per: dict, weight: dict, draws: int, seed: int,
                held_out: bool) -> float:
    """The same pooled protocol after breaking the candidate columns.

    Every candidate column is permuted inside its own fixture, so base rates,
    fixture sizes and the shipped inputs are untouched and only the added
    block loses its link to the label.
    """
    rng = np.random.default_rng(seed)
    base = pooled_auc(per, weight, SHIPPED, held_out)
    values = []
    for _ in range(draws):
        broken = {}
        for fixture, cols in per.items():
            copy = dict(cols)
            order = rng.permutation(len(cols["label"]))
            for name in CANDIDATES:
                copy[name] = cols[name][order]
            broken[fixture] = copy
        values.append(
            pooled_auc(broken, weight, SHIPPED + CANDIDATES, held_out) - base)
    return float(np.mean(values))


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=here.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--json", type=pathlib.Path,
                    default=here / "e134-artifacts/rung1-incremental-auc.json")
    ap.add_argument("--depths", type=int, nargs="+",
                    default=list(range(0, 7)))
    ap.add_argument("--null-draws", type=int, default=20)
    args = ap.parse_args()

    print("harness=local instrument  E134 rung 1  zero GPU")
    print("forced depth 7 traces, uncensored by the shipped estimator\n")

    traces, gates = {}, {}
    for directory in sorted(args.runs.iterdir()):
        path = directory / "trace.txt"
        if not path.is_file():
            continue
        rounds, gate = parse_trace(path)
        if not rounds:
            continue
        traces[directory.name] = rounds
        gates[directory.name] = gate

    print("## parse gates")
    print("%-18s %6s %9s %9s %9s %9s %11s" % (
        "fixture", "rounds", "rowcnt-ok", "rowcnt-x", "ident-ok", "ident-x",
        "sched-maxerr"))
    bad = 0
    for name in sorted(gates):
        g = gates[name]
        bad += g["row_count_bad"] + g["margin_identity_bad"]
        print("%-18s %6d %9d %9d %9d %9d %11.2e" % (
            name, g["rounds"], g["row_count_ok"], g["row_count_bad"],
            g["margin_identity_ok"], g["margin_identity_bad"],
            g["sched_max_abs_error"]))
    print("\ntotal alignment failures: %d" % bad)
    if bad:
        print("ALIGNMENT IS NOT PROVEN - every AUC below is void")

    # Finding 83 weights, mapped from fixtures to ranked prompts.
    weight = {}
    for fixture in traces:
        prompt = FIXTURE_PROMPT.get(fixture)
        if prompt is None:
            continue
        weight[fixture] = F83_WEIGHT[prompt] / len(PROMPT_FIXTURES[prompt])

    pooled = {}
    per_depth = {}
    for depth in args.depths:
        parts, rows = [], []
        for fixture, rounds in sorted(traces.items()):
            if fixture not in weight:
                continue
            cols = features_at(rounds, depth)
            labels = cols["label"]
            if len(labels) < 30 or labels.min() == labels.max():
                continue
            base = fit_auc(cols, SHIPPED, labels)
            entry = {"fixture": fixture, "n": int(len(labels)),
                     "p": float(labels.mean()), "weight": weight[fixture],
                     "shipped_auc": base, "alone": {}, "incremental": {}}
            for name in CANDIDATES:
                if np.allclose(cols[name], cols[name][0]):
                    continue
                entry["alone"][name] = auc(cols[name], labels)[0]
                entry["incremental"][name] = (
                    fit_auc(cols, SHIPPED + (name,), labels) - base)
            entry["incremental"]["ALL_PREV_MARGIN"] = fit_auc(
                cols, SHIPPED + ("prev_margin_at_d", "prev_margin_mean",
                                 "prev_margin_min", "prev_margin_slope"),
                labels) - base
            entry["incremental"]["ALL_NEW"] = fit_auc(
                cols, SHIPPED + CANDIDATES, labels) - base
            entry["incremental"]["NULL_FLOOR"] = null_incremental(
                cols, labels, args.null_draws, depth * 1000 + len(parts))
            split_base = split_auc(cols, SHIPPED, labels)
            entry["shipped_auc_heldout"] = split_base
            entry["incremental"]["HELDOUT_ALL_NEW"] = (
                split_auc(cols, SHIPPED + CANDIDATES, labels) - split_base)
            entry["incremental"]["HELDOUT_PREV_MARGIN"] = (
                split_auc(cols, SHIPPED + ("prev_margin_at_d",
                                           "prev_margin_mean",
                                           "prev_margin_min",
                                           "prev_margin_slope"), labels)
                - split_base)
            parts.append(entry)
            rows.append(entry["n"] * weight[fixture])
        if not parts:
            continue
        per_depth[depth] = parts
        total = sum(rows)
        agg = {"weighted_rounds": total,
               "observations": sum(e["n"] for e in parts),
               "fixtures": len(parts),
               "shipped_auc": sum(e["shipped_auc"] * w
                                  for e, w in zip(parts, rows)) / total,
               "alone": {}, "incremental": {}}
        for name in list(CANDIDATES) + EXTRA_COLUMNS:
            for key in ("alone", "incremental"):
                num = den = 0.0
                for entry, w in zip(parts, rows):
                    got = entry[key].get(name)
                    if got is None or math.isnan(got):
                        continue
                    num += w * got
                    den += w
                if den:
                    agg[key][name] = num / den
        pooled[depth] = agg

    print("\n## F83-weighted AUC of each signal ALONE, per boundary")
    print("   `pm_last` is a NEGATIVE CONTROL: it IS the shipped `m=` field,")
    print("   so its incremental AUC must be zero if the pipeline is honest.")
    print("%-3s %6s %7s %8s %s" % ("d", "obs", "w-rnds", "shipped",
                                   " ".join("%8s" % SHORT[n]
                                            for n in CANDIDATES)))
    for depth in sorted(pooled):
        agg = pooled[depth]
        print("%-3d %6d %7.1f %8.4f %s" % (
            depth, agg["observations"], agg["weighted_rounds"],
            agg["shipped_auc"],
            " ".join("%8.4f" % agg["alone"].get(n, float("nan"))
                     for n in CANDIDATES)))

    print("\n## F83-weighted INCREMENTAL AUC over the shipped input set")
    print("   shipped set = %s. Logistic, IN SAMPLE." % ", ".join(SHIPPED))
    cols_out = list(CANDIDATES) + EXTRA_COLUMNS
    print("%-3s %s" % ("d", " ".join("%8s" % SHORT[n] for n in cols_out)))
    for depth in sorted(pooled):
        agg = pooled[depth]
        print("%-3d %s" % (depth, " ".join(
            "%+8.4f" % agg["incremental"].get(n, float("nan"))
            for n in cols_out)))

    print("\n## the honest columns, side by side")
    print("   `null` is the in-sample gain from the SAME eleven columns after")
    print("   permuting them, so it is the overfitting floor. `held` fits on")
    print("   the first half of each decode window and scores the second.")
    print("%-3s %6s %9s %9s %9s %9s %9s" % (
        "d", "obs", "in-sample", "null", "net", "held-out", "held-pm"))
    for depth in sorted(pooled):
        inc = pooled[depth]["incremental"]
        raw = inc.get("ALL_NEW", float("nan"))
        floor = inc.get("NULL_FLOOR", float("nan"))
        print("%-3d %6d %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f" % (
            depth, pooled[depth]["observations"], raw, floor, raw - floor,
            inc.get("HELDOUT_ALL_NEW", float("nan")),
            inc.get("HELDOUT_PREV_MARGIN", float("nan"))))

    print("\n## ONE GLOBAL POLICY, fitted pooled, scored per fixture")
    print("   A scheduler ships one rule for every prompt, so this is the")
    print("   design that matches the decision. `lofo` drops the scored")
    print("   fixture from the training set. AUCs are F83-weighted.")
    print("%-3s %5s %4s %8s %8s %8s %8s %8s %8s %8s" % (
        "d", "obs", "fix", "ship-in", "ship-lofo", "all-in", "all-lofo",
        "inc-lofo", "null-lof", "net"))
    global_out = {}
    for depth in args.depths:
        per = pooled_blocks(traces, weight, depth)
        if len(per) < 3:
            continue
        obs = sum(len(c["label"]) for c in per.values())
        ship_in = pooled_auc(per, weight, SHIPPED, False)
        base_folds = pooled_folds(per, weight, SHIPPED, True)
        ship_lofo = weighted_mean(base_folds)
        all_in = pooled_auc(per, weight, SHIPPED + CANDIDATES, False)
        all_lofo = pooled_auc(per, weight, SHIPPED + CANDIDATES, True)
        null_lofo = pooled_null(per, weight, args.null_draws,
                                7000 + depth, True)
        row = {"observations": obs, "fixtures": len(per),
               "shipped_in_sample": ship_in, "shipped_heldout": ship_lofo,
               "all_in_sample": all_in, "all_heldout": all_lofo,
               "incremental_heldout": all_lofo - ship_lofo,
               "null_floor_heldout": null_lofo,
               "net_heldout": all_lofo - ship_lofo - null_lofo,
               "single_incremental_heldout": {}, "single_folds": {}}
        for name in CANDIDATES:
            if not is_usable(per, name):
                continue
            folds = pooled_folds(per, weight, SHIPPED + (name,), True)
            delta = {f: folds[f][0] - base_folds[f][0]
                     for f in folds if f in base_folds}
            row["single_incremental_heldout"][name] = weighted_mean(
                {f: (delta[f], base_folds[f][1]) for f in delta})
            row["single_folds"][name] = delta
        global_out[depth] = row
        print("%-3d %5d %4d %8.4f %8.4f %8.4f %8.4f %+8.4f %+8.4f %+8.4f" % (
            depth, obs, len(per), ship_in, ship_lofo, all_in, all_lofo,
            row["incremental_heldout"], null_lofo, row["net_heldout"]))

    print("\n## held-out incremental AUC of ONE added input at a time")
    print("   One extra column cannot overfit the way eleven can, so a")
    print("   positive value here is the load-bearing evidence.")
    print("%-3s %s" % ("d", " ".join("%8s" % SHORT[n] for n in CANDIDATES)))
    for depth in sorted(global_out):
        singles = global_out[depth]["single_incremental_heldout"]
        print("%-3d %s" % (depth, " ".join(
            "%+8.4f" % singles.get(n, float("nan")) for n in CANDIDATES)))

    print("\n## how many held-out fixtures each input helps, `wins/folds`")
    print("   A weighted mean carried by one fixture is not a global rule.")
    print("%-3s %s" % ("d", " ".join("%8s" % SHORT[n] for n in CANDIDATES)))
    for depth in sorted(global_out):
        folds = global_out[depth]["single_folds"]
        cells = []
        for name in CANDIDATES:
            delta = folds.get(name)
            if delta is None:
                cells.append("%8s" % "-")
                continue
            wins = sum(1 for v in delta.values() if v > 0.0)
            cells.append("%8s" % ("%d/%d" % (wins, len(delta))))
        print("%-3d %s" % (depth, " ".join(cells)))

    print("\n## fit-free F83-weighted AUC of each shipped input")
    print("   `ship-lofo` above REFITS the shipped inputs, so it is an upper")
    print("   bound on the shipped closed form. These need no fit at all.")
    print("   Below 0.5000 means the input is ANTI-predictive.")
    print("%-3s %5s %9s %9s %9s" % ("d", "obs", "margin", "ema_d", "reach"))
    fitfree = {}
    for depth in args.depths:
        per = pooled_blocks(traces, weight, depth)
        if not per:
            continue
        row = {}
        for name in SHIPPED:
            num = den = 0.0
            for fixture, cols in per.items():
                labels = cols["label"]
                if cols[name].std() < 1e-12:
                    continue
                w = weight[fixture] * len(labels)
                num += w * auc(cols[name], labels)[0]
                den += w
            row[name] = num / den if den else float("nan")
        fitfree[depth] = row
        print("%-3d %5d %9.4f %9.4f %9.4f" % (
            depth, sum(len(c["label"]) for c in per.values()),
            row["margin"], row["ema_d"], row["reach_shipped"]))

    blocks = {d: pooled_blocks(traces, weight, d) for d in args.depths}
    common = set.intersection(*(set(b) for b in blocks.values() if b))
    balanced = {}
    if len(common) >= 3:
        print("\n## the same %d fixtures at every boundary, so the depth"
              % len(common))
        print("   profile below is not a composition effect")
        print("   %s" % ", ".join(sorted(common)))
        print("%-3s %5s %10s %10s %10s" % (
            "d", "obs", "ship-lofo", "best-inc", "which"))
        for depth in sorted(blocks):
            per = {k: v for k, v in blocks[depth].items() if k in common}
            if len(per) < 3:
                continue
            base = pooled_folds(per, weight, SHIPPED, True)
            ship = weighted_mean(base)
            best = (float("-inf"), "-")
            singles = {}
            for name in CANDIDATES:
                if not is_usable(per, name):
                    continue
                folds = pooled_folds(per, weight, SHIPPED + (name,), True)
                value = weighted_mean({f: (folds[f][0] - base[f][0], base[f][1])
                                       for f in folds if f in base})
                singles[name] = value
                best = max(best, (value, name))
            obs = sum(len(c["label"]) for c in per.values())
            balanced[depth] = {"observations": obs, "shipped_heldout": ship,
                               "single_incremental_heldout": singles}
            print("%-3d %5d %10.4f %+10.4f %10s" % (
                depth, obs, ship, best[0], SHORT.get(best[1], best[1])))

    step = {}
    if 3 in blocks and 4 in blocks:
        pair = sorted(set(blocks[3]) & set(blocks[4]))
        left = {k: blocks[3][k] for k in pair}
        right = {k: blocks[4][k] for k in pair}
        a = pooled_folds(left, weight, SHIPPED, True)
        b = pooled_folds(right, weight, SHIPPED, True)
        print("\n## the depth-3 to depth-4 step on an IDENTICAL %d-fixture"
              % len(pair))
        print("   panel, so this step cannot be a composition effect")
        print("%-18s %8s %8s %9s %7s %7s" % (
            "fixture", "d3 auc", "d4 auc", "delta", "d3 p", "d4 p"))
        for fixture in sorted(pair, key=lambda f: b[f][0] - a[f][0]):
            step[fixture] = {"d3": a[fixture][0], "d4": b[fixture][0],
                             "delta": b[fixture][0] - a[fixture][0],
                             "d3_rate": float(left[fixture]["label"].mean()),
                             "d4_rate": float(right[fixture]["label"].mean())}
            print("%-18s %8.4f %8.4f %+9.4f %7.3f %7.3f" % (
                fixture, a[fixture][0], b[fixture][0],
                b[fixture][0] - a[fixture][0],
                step[fixture]["d3_rate"], step[fixture]["d4_rate"]))
        print("%-18s %8.4f %8.4f %+9.4f" % (
            "F83-weighted", weighted_mean(a), weighted_mean(b),
            weighted_mean(b) - weighted_mean(a)))

    if 4 in global_out:
        row = global_out[4]
        agg = pooled.get(4, {})
        singles = row["single_incremental_heldout"]
        best = max(((v, k) for k, v in singles.items()),
                   default=(float("nan"), "-"))
        print("\n## the depth-4 boundary, which the assignment names")
        print("  observations                      %d over %d fixtures"
              % (row["observations"], row["fixtures"]))
        print("  shipped AUC, held out             %.4f"
              % row["shipped_heldout"])
        print("  best single input, held out       %+.4f  (%s)" % best)
        print("  every new input, held out         %+.4f"
              % row["incremental_heldout"])
        print("  permuted null floor, held out     %+.4f"
              % row["null_floor_heldout"])
        print("  held out net of the null floor    %+.4f" % row["net_heldout"])
        print("  per-fixture in-sample ALL_NEW     %+.4f  (inflated)"
              % agg.get("incremental", {}).get("ALL_NEW", float("nan")))
        print("  per-fixture in-sample null floor  %+.4f  (the same size)"
              % agg.get("incremental", {}).get("NULL_FLOOR", float("nan")))
        print("  negative control `pm_last`        %+.4f  (must be ~0)"
              % agg.get("incremental", {}).get("prev_margin_last",
                                               float("nan")))
        best_value = best[0] if best[0] == best[0] else 0.0
        detail = row["single_folds"].get(best[1], {})
        if detail:
            wins = sum(1 for v in detail.values() if v > 0.0)
            print("  `%s` per held-out fixture      %d of %d positive"
                  % (best[1], wins, len(detail)))
            for fixture in sorted(detail, key=lambda f: -detail[f]):
                print("      %-18s %+8.4f" % (fixture, detail[fixture]))
        headline = max(row["net_heldout"], best_value)
        print("\n  stop rule: below +0.0200 falsifies the section C hypothesis")
        print("  headline (best of net-of-null and best single) %+.4f"
              % headline)
        print("  VERDICT: %s" % (
            "PASS, section C survives at depth 4" if headline >= 0.02
            else "FAIL, section C is falsified at depth 4"))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "harness": "local instrument", "gpu_used": False,
        "shipped_inputs": list(SHIPPED), "candidates": list(CANDIDATES),
        "gates": gates, "alignment_failures": bad,
        "per_depth": {str(k): v for k, v in per_depth.items()},
        "pooled": {str(k): v for k, v in pooled.items()},
        "one_global_policy": {str(k): v for k, v in global_out.items()},
        "balanced_panel": {str(k): v for k, v in balanced.items()},
        "depth3_to_depth4_step": step,
        "fit_free_shipped_auc": {str(k): v for k, v in fitfree.items()},
    }, indent=2) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
