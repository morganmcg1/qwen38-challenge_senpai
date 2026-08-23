#!/usr/bin/env python3
"""E150 R1 and R2: how much per-round headroom does pre-draft information buy?

`harness=local instrument`. Zero GPU. Zero Swift. Campaign Rule 79 binds this
assignment, and this module publishes no timing contrast of any kind: every
number is an offline price against the E145 R3 measured per-width round cost
curve, through the E128 replayer.

E145 R7-2 split the acceptance axis in two on that curve:

    shipped EMA state                    +0.1338 %
    perfect per-position marginal        -0.3806 %   (worse than shipped)
    perfect per-round realised capability +6.3508 %  (the open axis)

R1 asks how much of the second gap a predictor can reach when it may read
only what the scored session already holds BEFORE the round's drafts exist.
R2 then splits that answer by information rung.

Method

  1. Collect, under a held-fixed policy, one pre-draft observation per round
     together with that round's realised capability `K`.
  2. Fit one conditional acceptance model per draft position,
     `P(K > i | K >= i, x)`. That is exactly the object the shipped walk
     multiplies into `reach`, so the fitted vector is a drop-in replacement
     for `positionAcceptEMA` and the decision rule never changes.
  3. Cross-validate leave one ranked prompt out. Every priced prompt is
     scored by a model that never saw it.
  4. Price the deployed predictor through the same scorer that produced the
     published R7 cells, and reproduce four published cells as controls.

Units. E145 R7-3 adds `N(0, sigma)` INDEPENDENTLY TO EVERY ENTRY of the
state vector whose truth is the step `1[i < K]`, then clips to `[0, 1]`. Its
`sigma` is therefore a per-entry state-vector error, not an error on `K`. Two
consequences this module measures rather than asserts: the clip makes the
realised per-entry error smaller than the nominal `sigma`, and a real
predictor's error is correlated across entries and shrunk toward the base
rate, which the independent ladder cannot represent. So the ladder is
re-indexed on its own REALISED per-entry error, a second ladder is run in the
displacement units a scalar `K` predictor naturally produces, and the
headline number is the measured price of the actual predictor, which needs no
conversion at all.

Usage:
  python3 e150_r1.py --json r1.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
import e128_replay  # noqa: E402
import e150_lib  # noqa: E402
from e128_price import RANKED_PROMPTS  # noqa: E402
from e128_replay import MAX_DEPTH  # noqa: E402
from e134_rung2 import concordance, simulate  # noqa: E402
from e140_cells import install  # noqa: E402
from e150_lib import (  # noqa: E402
    EMA_PRIOR_LIST, R7_CLAIRVOYANT_ARGMAX_PCT, R7_CLAMP_NONE_PCT,
    R7_ORACLE_DEPTH_PCT, R7_SHIPPED_PCT, R7_TRUEP_ARGMAX_PCT,
    REPRO_TOLERANCE_PP, build_env, clamp_state, score_arm, write_artifact,
)
from e150_predict import (  # noqa: E402
    LADDER, Observer, SHARED_COLUMNS, USES_EMA, apply_rule, features,
    fit_hazard, fixed_state_walker, hit_vector, predictor_walker,
    select_temperature,
)

# E145 R7-3's own noise grid, reproduced rung for rung.
NOISE_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
# The displacement ladder, in units of draft positions on `K` itself.
SHIFT_GRID = (0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
# Logit temperature. `1.0` is the fitted model; above `1.0` is more hedged.
TEMPERATURE_GRID = (0.6, 0.8, 1.0, 1.25, 1.6, 2.0)
# Constant running-ratio control, in the same normalised units as `cumulative`.
LAMBDA_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00)
# The stop rule the assignment set for this rung.
R1_GATE_PCT = R7_SHIPPED_PCT + 0.30
RESIDUAL_STRIDE = 41


# ----------------------------------------------------------- data collection

class Collector:
    """One pre-draft observation per round, plus the round's realised `K`.

    The label is read here and nowhere else. `e150_predict.features` never
    receives it, so no priced round can see the outcome it is predicting.
    """

    def __init__(self):
        self.rows = {name: [] for name in SHARED_COLUMNS}
        self.ema = []
        self.capability = []
        self.depth = []
        self.prev_acc_mismatch = 0
        self.rounds = 0
        self._last = None

    def sink(self, feat, ema, capability, depth, claimed_prev_acc, restarted):
        if restarted or self._last is None:
            truth = 0
        else:
            truth = min(self._last[0], self._last[1])
        if truth != claimed_prev_acc:
            self.prev_acc_mismatch += 1
        self._last = (capability, depth)
        for name in SHARED_COLUMNS:
            self.rows[name].append(feat[name])
        self.ema.append(ema)
        self.capability.append(capability)
        self.depth.append(depth)
        self.rounds += 1

    def arrays(self) -> dict:
        out = {name: np.asarray(values, dtype=float)
               for name, values in self.rows.items()}
        out["ema"] = np.asarray(self.ema, dtype=float)
        out["capability"] = np.asarray(self.capability, dtype=float)
        out["depth"] = np.asarray(self.depth, dtype=float)
        return out


def collector_walker(collector: Collector, rule: str, clamp, model=None):
    """The held-fixed policy, instrumented. `model=None` is the shipped tree."""
    obs = Observer()

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        ema_list = list(ema)
        restarted = ema_list == EMA_PRIOR_LIST
        obs.observe(ema_list, ctx["base_rate"])
        feat = features(margin, offer, ctx["prev_rows"], obs,
                        ctx["base_rate"])
        state = ema_list if model is None else model.state(feat, ema_list)
        if clamp is None:
            depth = apply_rule(rule, state, margin, offer, price)
        else:
            depth = apply_rule(rule, clamp_state(state, margin, clamp),
                               float("nan"), offer, price)
        collector.sink(feat, ema_list, ctx["capability"], depth,
                       obs.prev_acc, restarted)
        obs.chose(depth)
        return depth

    return chooser


def collect(env, seeds, windows, rule="greedy", clamp=None,
            models=None) -> tuple[dict, dict]:
    """One `Collector` per ranked prompt, under a held-fixed policy."""
    per_prompt, gate = {}, {"rounds": 0, "prev_acc_mismatch": 0}
    for prompt in RANKED_PROMPTS:
        collector = Collector()
        model = None if models is None else models[prompt]
        for seed in seeds:
            entry = env.cache[(seed, prompt)]
            install(env.measured)
            simulate(None, entry["factory"](entry["p_target"]), windows,
                     price=env.measured_price,
                     walker=collector_walker(collector, rule, clamp, model))
        per_prompt[prompt] = collector.arrays()
        gate["rounds"] += collector.rounds
        gate["prev_acc_mismatch"] += collector.prev_acc_mismatch
    return per_prompt, gate


def pool(per_prompt: dict, names) -> dict:
    keys = list(SHARED_COLUMNS) + ["ema", "capability", "depth"]
    return {key: np.concatenate([per_prompt[name][key] for name in names])
            for key in keys}


# --------------------------------------------------------------- diagnostics

class ErrorTally:
    """State-vector error in three different units, all measured together.

    `rms` is the all-position per-entry error, which is the unit R7-3's
    Gaussian ladder perturbs. `reachable_rms` restricts the same quantity to
    positions the round actually reached (`i <= K`), because a conditional
    hazard is never consulted past the first rejection and its value there
    cannot move any decision. `scalar_sd` is the error of the induced scalar
    `E[K]` against the realised `K`, which is the unit the displacement ladder
    perturbs. The three differ by a large factor, so a price ladder read at
    the wrong one is not an approximation but a different experiment.
    """

    def __init__(self, stride: int = RESIDUAL_STRIDE):
        self.sq = 0.0
        self.n = 0
        self.reach_sq = 0.0
        self.reach_n = 0
        self.scalar_sq = 0.0
        self.scalar_sum = 0.0
        self.scalar_n = 0
        self.sample = []
        self.stride = stride
        self._seen = 0

    def record(self, state, capability: int) -> None:
        hit = hit_vector(capability)
        for i in range(MAX_DEPTH):
            err = state[i] - hit[i]
            self.sq += err * err
            if i <= capability:
                self.reach_sq += err * err
                self.reach_n += 1
            self._seen += 1
            if self._seen % self.stride == 0:
                self.sample.append(err)
        self.n += MAX_DEPTH
        reach, expected = 1.0, 0.0
        for i in range(MAX_DEPTH):
            reach *= state[i]
            expected += reach
        gap = expected - capability
        self.scalar_sq += gap * gap
        self.scalar_sum += gap
        self.scalar_n += 1

    def rms(self) -> float:
        return math.sqrt(self.sq / self.n) if self.n else float("nan")

    def reachable_rms(self) -> float:
        if not self.reach_n:
            return float("nan")
        return math.sqrt(self.reach_sq / self.reach_n)

    def scalar_bias(self) -> float:
        return self.scalar_sum / self.scalar_n if self.scalar_n else float("nan")

    def scalar_sd(self) -> float:
        """Spread of the `E[K]` error, with its own bias removed."""
        if self.scalar_n < 2:
            return float("nan")
        mean = self.scalar_sum / self.scalar_n
        var = self.scalar_sq / self.scalar_n - mean * mean
        return math.sqrt(max(0.0, var))

    def scalar_rms(self) -> float:
        if not self.scalar_n:
            return float("nan")
        return math.sqrt(self.scalar_sq / self.scalar_n)

    def stats(self) -> dict:
        if not self.sample:
            return {}
        values = np.asarray(self.sample)
        mean = float(values.mean())
        sd = float(values.std())
        centred = values - mean
        skew = float((centred ** 3).mean() / sd ** 3) if sd else 0.0
        kurt = float((centred ** 4).mean() / sd ** 4 - 3.0) if sd else 0.0
        quantiles = [float(np.quantile(values, q))
                     for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)]
        beyond2 = float(np.mean(np.abs(centred) > 2 * sd)) if sd else 0.0
        beyond3 = float(np.mean(np.abs(centred) > 3 * sd)) if sd else 0.0
        return {
            "rms": self.rms(),
            "reachable_rms": self.reachable_rms(),
            "reachable_entries": int(self.reach_n),
            "scalar_k_sd": self.scalar_sd(),
            "scalar_k_rms": self.scalar_rms(),
            "scalar_k_bias": self.scalar_bias(),
            "rounds": int(self.scalar_n),
            "sampled": int(values.size),
            "mean": mean, "sd": sd, "skew": skew, "excess_kurtosis": kurt,
            "quantiles_01_05_25_50_75_95_99": quantiles,
            "frac_beyond_2sd": beyond2, "gaussian_frac_beyond_2sd": 0.0455,
            "frac_beyond_3sd": beyond3, "gaussian_frac_beyond_3sd": 0.0027,
            "is_gaussian": abs(skew) < 0.5 and abs(kurt) < 1.0,
        }


def predictive_diagnostics(model, rows: dict) -> dict:
    """Out-of-sample discrimination of the fitted state vector on one prompt."""
    cap = rows["capability"]
    n = len(cap)
    stride = max(1, n // 20000)
    index = range(0, n, stride)
    expected, per_position = [], {}
    scores = {i: [] for i in range(MAX_DEPTH)}
    labels = {i: [] for i in range(MAX_DEPTH)}
    for j in index:
        feat = {name: rows[name][j] for name in SHARED_COLUMNS}
        state = model.state(feat, rows["ema"][j])
        reach, total = 1.0, 0.0
        for i in range(MAX_DEPTH):
            reach *= state[i]
            total += reach
        expected.append(total)
        for i in range(MAX_DEPTH):
            if cap[j] >= i:
                scores[i].append(state[i])
                labels[i].append(1.0 if cap[j] > i else 0.0)
    for i in range(MAX_DEPTH):
        per_position["auc_position_%d" % i] = (
            concordance(scores[i], labels[i]) if scores[i] else float("nan"))
    truth = [cap[j] for j in index]
    return {
        "expected_vs_capability_pearson_r": pearson(expected, truth),
        "rounds_scored": len(expected),
        **per_position,
    }


def pearson(a, b) -> float:
    if len(a) < 2:
        return float("nan")
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


# -------------------------------------------------------------------- arms

def pooled_level() -> dict:
    """The measured level bias E128 fitted, reused unchanged."""
    rows = {row["prompt_id"]: row for row in json.loads(
        (HERE / "e128-artifacts" / "jensen-and-sign.json").read_text()
    )["hypothesis_j"]}
    return e128_price.pooled_level(
        rows, e128_price.RANKED_PROMPTS["beagle"]["fixture"])


def e128_walker(arm: str, level: dict):
    """The published four-line E128 adapter, so R0 arms replay identically."""
    policy = e128_price.make_policy(arm, level=level)

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        return policy(ema, margin, offer, ctx["capability"])

    return lambda seed, prompt, entry: chooser


def composed_walker(level: dict):
    """`rankedprice` x `expectedonly`: swap the price table AND debias the
    level term in one policy. R0 scored these two separately; the advisor
    asked whether they compose, because they correct different halves of the
    same first-break test -- `rankedprice` fixes what a step costs and
    `expectedonly` fixes what a step is expected to return."""
    marginal, cumulative = e128_price.ranked_price_table()
    gamma = level["gamma"]

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        return e128_replay.cost_model_depth(
            ema, margin, offered_depth=offer, expected_gain=gamma,
            marginal=marginal, cumulative=cumulative)[0]

    return lambda seed, prompt, entry: chooser


def price_predictor(env, models, rung: str, rule: str, clamp,
                    tally: ErrorTally | None = None, fixed_lam=None) -> dict:
    def make(seed, prompt, entry):
        record = None if tally is None else tally.record
        return predictor_walker(models[prompt][rung], rule, clamp, record,
                                fixed_lam)
    return score_arm(env, "measured", env.measured, env.measured_price, make)


def price_state(env, state_of, rule: str, clamp,
                tally: ErrorTally | None = None) -> dict:
    def make(seed, prompt, entry):
        record = None if tally is None else tally.record
        return fixed_state_walker(state_of(seed, prompt, entry), rule, clamp,
                                  record)
    return score_arm(env, "measured", env.measured, env.measured_price, make)


def truth_state(seed, prompt, entry):
    def state_of(ema, margin, offer, ctx):
        return hit_vector(ctx["capability"])
    return state_of


def truep_state(seed, prompt, entry):
    vector = [entry["p_target"][min(i, len(entry["p_target"]) - 1)]
              for i in range(MAX_DEPTH)]

    def state_of(ema, margin, offer, ctx):
        return vector
    return state_of


def noise_state(sigma: float):
    def factory(seed, prompt, entry):
        rng = random.Random(seed)

        def state_of(ema, margin, offer, ctx):
            hit = hit_vector(ctx["capability"])
            return [min(1.0, max(0.0, hit[i] + rng.gauss(0.0, sigma)))
                    for i in range(MAX_DEPTH)]
        return state_of
    return factory


def shift_state(sigma: float):
    """A scalar `K` predictor with displacement error, in `K`'s own units."""
    def factory(seed, prompt, entry):
        rng = random.Random(seed)

        def state_of(ema, margin, offer, ctx):
            noisy = ctx["capability"] + (rng.gauss(0.0, sigma)
                                         if sigma > 0 else 0.0)
            k = int(min(MAX_DEPTH, max(0, round(noisy))))
            return hit_vector(k)
        return state_of
    return factory


def price_shipped(env, rule: str, clamp, tally=None) -> dict:
    def state_of(seed, prompt, entry):
        def inner(ema, margin, offer, ctx):
            return ema
        return inner
    return price_state(env, state_of, rule, clamp, tally)


# -------------------------------------------------------------------- main

def interpolate(ladder, sigma: float, key: str = "realised_sigma") -> float:
    """Linear read of a price ladder that is indexed by realised error.

    `key` names the error unit the ladder is indexed on. Reading a ladder at a
    sigma measured in a different unit is the conversion error this experiment
    exists to expose, so the unit is always explicit here.
    """
    points = sorted((row[key], row["median_pct"]) for row in ladder)
    if sigma <= points[0][0]:
        return points[0][1]
    if sigma >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= sigma <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (sigma - x0) / (x1 - x0)
    return points[-1][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--fit-seeds", type=int, default=2)
    ap.add_argument("--fit-sim-windows", type=int, default=100)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--refit-iteration", action="store_true",
                    help="refit under the deployed policy and reprice")
    ap.add_argument("--json", type=pathlib.Path,
                    default=e150_lib.ARTIFACTS / "r1.json")
    args = ap.parse_args()

    print("harness=local instrument  E150 R1 and R2  zero GPU  zero Swift")
    env = build_env(windows=args.windows, seeds=args.seeds)
    print("  attachment gate %s" % env.gate)
    fit_seeds = env.seeds[:args.fit_seeds]

    print("\n## collect pre-draft observations under the shipped tree")
    per_prompt, gate = collect(env, fit_seeds, args.fit_sim_windows)
    print("  rounds %d ; previous-accept reconstruction mismatches %d"
          % (gate["rounds"], gate["prev_acc_mismatch"]))
    if gate["prev_acc_mismatch"]:
        raise SystemExit("the pre-draft observer is not exact; R1 is void")
    names = list(RANKED_PROMPTS)
    caps = np.concatenate([per_prompt[p]["capability"] for p in names])
    print("  realised capability: mean %.4f  sd %.4f  histogram %s"
          % (caps.mean(), caps.std(),
             [int((caps == k).sum()) for k in range(MAX_DEPTH + 1)]))

    print("\n## fit leave one ranked prompt out")
    models = {}
    for held in names:
        train = pool(per_prompt, [n for n in names if n != held])
        models[held] = {rung: fit_hazard(train, rung, ridge=args.ridge)
                        for rung in LADDER}
    all_rows = pool(per_prompt, names)
    in_sample = {rung: fit_hazard(all_rows, rung, ridge=args.ridge)
                 for rung in LADDER}
    in_sample_models = {p: in_sample for p in names}

    diagnostics = {}
    for held in names:
        diagnostics[held] = predictive_diagnostics(models[held]["L4"],
                                                   per_prompt[held])
    print("  out-of-sample E[K] to K correlation per prompt")
    for held in names:
        print("    %-10s r = %+0.4f   auc(pos0) %.4f  auc(pos3) %.4f"
              % (held,
                 diagnostics[held]["expected_vs_capability_pearson_r"],
                 diagnostics[held]["auc_position_0"],
                 diagnostics[held]["auc_position_3"]))

    out = dict(env.identity())
    out.update({
        "rung": "E150 R1,R2",
        "fit_seeds": fit_seeds,
        "fit_sim_windows": args.fit_sim_windows,
        "ridge": args.ridge,
        "collection_gate": gate,
        "capability_histogram": [int((caps == k).sum())
                                 for k in range(MAX_DEPTH + 1)],
        "e150_predictor_diagnostics": diagnostics,
        "e150_predictor_columns": {r: list(LADDER[r]) for r in LADDER},
        "e150_predictor_uses_ema": USES_EMA,
        "e150_predictor_model_L4": {
            held: models[held]["L4"].to_json() for held in names},
    })

    # ---------------------------------------------------------- controls
    print("\n## controls that must reproduce published R7 cells")
    controls = {}
    controls["shipped_greedy"] = price_shipped(env, "greedy", None)
    controls["shipped_argmax"] = price_shipped(env, "argmax", None)
    controls["shipped_greedy_noclamp"] = price_shipped(env, "greedy", {})
    controls["truep_argmax"] = price_state(env, truep_state, "argmax", None)
    truth_tally = ErrorTally()
    controls["truth_argmax_clamped"] = price_state(
        env, truth_state, "argmax", None, truth_tally)
    controls["truth_argmax_noclamp"] = price_state(
        env, truth_state, "argmax", {})
    checks = {
        "shipped_greedy_reproduces_r7": abs(
            controls["shipped_greedy"]["median_pct_mean"] - R7_SHIPPED_PCT)
        <= REPRO_TOLERANCE_PP,
        "shipped_argmax_reproduces_r7": abs(
            controls["shipped_argmax"]["median_pct_mean"] - R7_SHIPPED_PCT)
        <= REPRO_TOLERANCE_PP,
        "clamp_none_reproduces_r7": abs(
            controls["shipped_greedy_noclamp"]["median_pct_mean"]
            - R7_CLAMP_NONE_PCT) <= REPRO_TOLERANCE_PP,
        "truep_reproduces_r7": abs(
            controls["truep_argmax"]["median_pct_mean"]
            - R7_TRUEP_ARGMAX_PCT) <= REPRO_TOLERANCE_PP,
        "truth_clamped_reproduces_r7_ladder_top": abs(
            controls["truth_argmax_clamped"]["median_pct_mean"]
            - R7_CLAIRVOYANT_ARGMAX_PCT) <= REPRO_TOLERANCE_PP,
        "truth_unclamped_reproduces_decision_oracle": abs(
            controls["truth_argmax_noclamp"]["median_pct_mean"]
            - R7_ORACLE_DEPTH_PCT) <= REPRO_TOLERANCE_PP,
    }
    for name, row in controls.items():
        print("  %-28s %+9.4f  sd %.4f  depth %.4f"
              % (name, row["median_pct_mean"], row["median_pct_sd"],
                 row["weighted_mean_depth"]))
    for name, ok in checks.items():
        print("  check %-46s %s" % (name, ok))

    # ------------------------------------------------- the fitted predictor
    print("\n## the fitted pre-draft predictor, leave one prompt out")
    arms, tallies = {}, {}
    for rung in ("L0c", "L1e", "L2", "L3", "L4"):
        for rule in ("argmax", "greedy"):
            for clamp_name, clamp in (("noclamp", {}), ("clamped", None)):
                name = "%s_%s_%s" % (rung, rule, clamp_name)
                tally = ErrorTally() if rule == "argmax" else None
                arms[name] = price_predictor(env, models, rung, rule, clamp,
                                             tally)
                if tally is not None:
                    tallies[name] = tally
                print("  %-26s %+9.4f  sd %.4f  depth %.4f  width1 %.4f"
                      % (name, arms[name]["median_pct_mean"],
                         arms[name]["median_pct_sd"],
                         arms[name]["weighted_mean_depth"],
                         arms[name]["width_histogram"]["1"]))

    in_tally = ErrorTally()
    arms["L4_argmax_noclamp_in_sample"] = price_predictor(
        env, in_sample_models, "L4", "argmax", {}, in_tally)
    print("  %-26s %+9.4f  (in sample)"
          % ("L4_argmax_noclamp", arms["L4_argmax_noclamp_in_sample"]
             ["median_pct_mean"]))

    primary_name = "L4_argmax_noclamp"
    primary = arms[primary_name]["median_pct_mean"]
    sigma_out = tallies[primary_name].rms()
    reachable_out = tallies[primary_name].reachable_rms()
    scalar_sd_out = tallies[primary_name].scalar_sd()
    scalar_bias_out = tallies[primary_name].scalar_bias()
    sigma_in = in_tally.rms()
    print("  primary error units: all-position %.4f  reachable %.4f  "
          "scalar K sd %.4f  bias %+.4f"
          % (sigma_out, reachable_out, scalar_sd_out, scalar_bias_out))

    # ------------------------------------- the ratio rule and its own ceiling
    print("\n## the ratio-of-sums rule the published score actually measures")
    ratio_arms = {}
    ratio_tally = ErrorTally()
    ratio_arms["L4_ratio_noclamp"] = price_predictor(
        env, models, "L4", "ratio", {}, ratio_tally)
    ratio_arms["L4_ratio_clamped"] = price_predictor(
        env, models, "L4", "ratio", None)
    ratio_arms["truth_ratio_noclamp"] = price_state(
        env, truth_state, "ratio", {})
    ratio_arms["truth_ratio_clamped"] = price_state(
        env, truth_state, "ratio", None)
    # The decisive control for the reframe: shipped information, new rule only.
    ratio_arms["shipped_ratio_noclamp"] = price_shipped(env, "ratio", {})
    ratio_arms["shipped_ratio_clamped"] = price_shipped(env, "ratio", None)
    for name, row in ratio_arms.items():
        print("  %-26s %+9.4f  sd %.4f  depth %.4f"
              % (name, row["median_pct_mean"], row["median_pct_sd"],
                 row["weighted_mean_depth"]))
    arms.update(ratio_arms)

    # ---------------------------------- R0 follow-up: do the two E128 fixes
    # compose? R0 repriced every E128 arm on the measured curve and found
    # exactly one that beat shipped, `expectedonly` at +0.3563. `rankedprice`
    # corrects the other half of the same test. This block scores both alone
    # and together inside R1's harness so all three share one env and one
    # base run. All three keep the SHIPPED first-break rule and shipped
    # information, so any gain here is orthogonal to R1's own reframe.
    print("\n## R0 follow-up: rankedprice x expectedonly composition")
    level = pooled_level()
    compose = {}
    for name, walker in (
            ("e128_ship", e128_walker("ship", level)),
            ("e128_expectedonly", e128_walker("expectedonly", level)),
            ("e128_rankedprice", e128_walker("rankedprice", level)),
            ("e128_rankedprice_x_expectedonly", composed_walker(level))):
        compose[name] = score_arm(env, "measured", env.measured,
                                  env.measured_price, walker)
        print("  %-32s %+9.4f  sd %.4f  depth %.4f  inadm %.4f"
              % (name, compose[name]["median_pct_mean"],
                 compose[name]["median_pct_sd"],
                 compose[name]["weighted_mean_depth"],
                 compose[name]["frac_rounds_inadmissible"]))
    compose_best = compose["e128_rankedprice_x_expectedonly"][
        "median_pct_mean"]
    compose_super = (
        compose_best - compose["e128_ship"]["median_pct_mean"]
        - (compose["e128_expectedonly"]["median_pct_mean"]
           - compose["e128_ship"]["median_pct_mean"])
        - (compose["e128_rankedprice"]["median_pct_mean"]
           - compose["e128_ship"]["median_pct_mean"]))
    print("  super-additive term %+0.4f pp ; composition beats the R1 "
          "headline: %s"
          % (compose_super, compose_best > arms["L4_ratio_noclamp"]
             ["median_pct_mean"]))
    # Control, and a load-bearing identity. `make_policy("ship")` and the env
    # base run are the SAME first-break walk reading the SAME uniform belief
    # table `1 + 0.18d`, which is what `makeUniformDepthPrice()` compiles in
    # Swift. So this arm must return exactly zero, and it does. It follows
    # that the published `+0.1338` shipped cell is not the deployed policy:
    # it is the deployed rule holding a DIFFERENT price table. The gap below
    # is the price-table swap priced on its own.
    compose_control = abs(compose["e128_ship"]["median_pct_mean"])
    swap_pp = R7_SHIPPED_PCT - compose["e128_ship"]["median_pct_mean"]
    print("  control |e128_ship - env base| %.3e  (must be 0)"
          % compose_control)
    print("  price-table swap alone %+0.4f pp : the published +%.4f shipped "
          "cell is the shipped RULE on a swapped TABLE, not what compiles"
          % (swap_pp, R7_SHIPPED_PCT))
    checks["e128_ship_equals_env_base"] = compose_control < 1e-9

    # The headline is the best DEPLOYABLE arm, not the best arm. Deployable
    # means: full pre-draft information, a rule the scored path can evaluate
    # before any draft exists, and predictor coefficients fitted leave one
    # prompt out. The rule form is not selected on R1's own outcome; R0.5
    # settles it on independent grounds, so naming the linearised rule here
    # is a pre-registration, not a search.
    deployable = {name: arms[name] for name in (
        "L4_argmax_noclamp", "L4_argmax_clamped",
        "L4_ratio_noclamp", "L4_ratio_clamped")}
    capturable_name = max(deployable,
                          key=lambda n: deployable[n]["median_pct_mean"])
    capturable = deployable[capturable_name]["median_pct_mean"]
    print("\n## deployable selection over %d arms" % len(deployable))
    print("  %-26s %9s %6s %9s %9s"
          % ("arm", "median%", "sd", "inadm", "depth"))
    for name in sorted(deployable, key=lambda n:
                       -deployable[n]["median_pct_mean"]):
        print("  %-26s %+9.4f %6.4f %9.4f %9.4f"
              % (name, deployable[name]["median_pct_mean"],
                 deployable[name]["median_pct_sd"],
                 deployable[name]["frac_rounds_inadmissible"],
                 deployable[name]["weighted_mean_depth"]))
    print("  headline arm %s at %+0.4f ; argmax-rule-only reading %+0.4f"
          % (capturable_name, capturable, primary))
    # `deployable` names the arms whose rule form is implementable. It does
    # NOT assert that the schedule each one produces stays inside the Rule 138
    # admissible width set, so that share is reported next to the headline
    # instead of being left for a reader to infer.
    headline_inadmissible = deployable[capturable_name][
        "frac_rounds_inadmissible"]
    print("  headline inadmissible width share %.4f ; Rule 138 admissible %s"
          % (headline_inadmissible, sorted(env.admitted)))

    print("\n## constant-lambda control: is the gain adaptivity or just depth?")
    lam_rows = []
    for lam in LAMBDA_GRID:
        row = price_predictor(env, models, "L4", "ratio", {}, None, lam)
        lam_rows.append({
            "fixed_lambda": lam,
            "median_pct": row["median_pct_mean"],
            "median_pct_sd": row["median_pct_sd"],
            "weighted_mean_depth": row["weighted_mean_depth"],
        })
        print("  lambda %.2f  %+9.4f  depth %.4f"
              % (lam, row["median_pct_mean"], row["weighted_mean_depth"]))
    best_lam = max(lam_rows, key=lambda r: r["median_pct"])
    adaptive = ratio_arms["L4_ratio_noclamp"]["median_pct_mean"]
    print("  best constant lambda %.2f at %+0.4f ; adaptive %+0.4f ; "
          "adaptivity premium %+0.4f pp"
          % (best_lam["fixed_lambda"], best_lam["median_pct"], adaptive,
             adaptive - best_lam["median_pct"]))

    # --------------------------------- temperature, chosen on training rows
    print("\n## temperature picked per fold by the offline training surrogate")
    tempered, temp_log = {}, {}
    for held in names:
        train = pool(per_prompt, [n for n in names if n != held])
        best_t, scored = select_temperature(
            models[held]["L4"], train, "argmax", {}, env.measured_price,
            TEMPERATURE_GRID)
        tempered[held] = {"L4": models[held]["L4"].at_temperature(best_t)}
        temp_log[held] = {"chosen_temperature": best_t, "surrogate": scored}
        print("  hold out %-10s T = %.2f" % (held, best_t))
    temp_tally = ErrorTally()
    arms["L4_argmax_noclamp_tempered"] = price_predictor(
        env, tempered, "L4", "argmax", {}, temp_tally)
    print("  %-26s %+9.4f  (T chosen without the held-out prompt)"
          % ("L4_argmax_noclamp_tempered",
             arms["L4_argmax_noclamp_tempered"]["median_pct_mean"]))

    # ---------------------------------------------------- the two ladders
    print("\n## R7-3's independent-noise ladder, re-indexed on realised error")
    noise_rows = []
    for sigma in NOISE_GRID:
        tally = ErrorTally()
        row = price_state(env, noise_state(sigma), "argmax", None, tally)
        noise_rows.append({
            "nominal_sigma": sigma,
            "realised_sigma": tally.rms(),
            "realised_reachable_sigma": tally.reachable_rms(),
            "realised_scalar_k_sd": tally.scalar_sd(),
            "median_pct": row["median_pct_mean"],
            "median_pct_sd": row["median_pct_sd"],
            "weighted_mean_depth": row["weighted_mean_depth"],
            "error_stats": tally.stats(),
        })
        print("  sigma %.2f  realised %.4f  reachable %.4f  K sd %.4f  %+9.4f"
              % (sigma, tally.rms(), tally.reachable_rms(), tally.scalar_sd(),
                 row["median_pct_mean"]))

    print("\n## the displacement ladder, in the units a K predictor produces")
    shift_rows = []
    for sigma in SHIFT_GRID:
        tally = ErrorTally()
        row = price_state(env, shift_state(sigma), "argmax", None, tally)
        shift_rows.append({
            "shift_sigma_positions": sigma,
            "realised_sigma": tally.rms(),
            "realised_reachable_sigma": tally.reachable_rms(),
            "realised_scalar_k_sd": tally.scalar_sd(),
            "median_pct": row["median_pct_mean"],
            "median_pct_sd": row["median_pct_sd"],
            "weighted_mean_depth": row["weighted_mean_depth"],
        })
        print("  shift %.2f  realised %.4f  reachable %.4f  K sd %.4f  %+9.4f"
              % (sigma, tally.rms(), tally.reachable_rms(), tally.scalar_sd(),
                 row["median_pct_mean"]))

    # Four reads of the same two ladders. Only the last one matches units on
    # both sides; the first is the naive read the assignment warned about.
    equivalent_noise = interpolate(noise_rows, sigma_out)
    equivalent_noise_reachable = interpolate(
        noise_rows, reachable_out, "realised_reachable_sigma")
    equivalent_shift_wrong_unit = interpolate(shift_rows, sigma_out)
    equivalent_shift = interpolate(shift_rows, scalar_sd_out,
                                   "realised_scalar_k_sd")

    ladder_rows = []
    for rung in ("L0c", "L1e", "L2", "L3", "L4"):
        name = "%s_argmax_noclamp" % rung
        clamped = "%s_argmax_clamped" % rung
        ladder_rows.append({
            "rung": rung,
            "columns": list(LADDER[rung]),
            "uses_ema": USES_EMA[rung],
            "median_pct_noclamp": arms[name]["median_pct_mean"],
            "median_pct_clamped": arms[clamped]["median_pct_mean"],
            "median_pct_sd": arms[name]["median_pct_sd"],
            "out_of_sample_sigma": tallies[name].rms(),
            "weighted_mean_depth": arms[name]["weighted_mean_depth"],
            "width1_share": arms[name]["width_histogram"]["1"],
        })
    # L0 and L1 of the assignment's ladder are the shipped information state,
    # not a fitted model, so they come from the controls.
    information_ladder = [
        {"rung": "L0", "information": "shipped EMA, no clamp",
         "median_pct": controls["shipped_greedy_noclamp"]["median_pct_mean"]},
        {"rung": "L1", "information": "shipped EMA plus the shipped clamps",
         "median_pct": controls["shipped_greedy"]["median_pct_mean"]},
    ] + [
        {"rung": {"L0c": "Lc", "L1e": "L1f", "L2": "L2", "L3": "L3",
                  "L4": "L4"}[row["rung"]],
         "information": {
             "L0c": "constant hazard, no state at all",
             "L1e": "fitted function of the shipped EMA only",
             "L2": "L1f plus the pending top-2 margin",
             "L3": "L2 plus recent margin history",
             "L4": "L3 plus recent realised acceptance",
         }[row["rung"]],
         "median_pct": row["median_pct_noclamp"],
         "out_of_sample_sigma": row["out_of_sample_sigma"]}
        for row in ladder_rows
    ]

    residuals = tallies[primary_name].stats()
    out.update({
        "e150_controls": {k: v for k, v in controls.items()},
        "e150_control_checks": checks,
        "e150_controls_all_reproduced": all(checks.values()),
        "e150_control_truth_sigma": truth_tally.rms(),
        "e150_positive_control_truth_pp":
            controls["truth_argmax_noclamp"]["median_pct_mean"],
        "e150_positive_control_truth_clamped_pp":
            controls["truth_argmax_clamped"]["median_pct_mean"],
        "e150_positive_control_constant_pp":
            arms["L0c_argmax_noclamp"]["median_pct_mean"],
        "e150_positive_control_constant_clamped_pp":
            arms["L0c_argmax_clamped"]["median_pct_mean"],
        "e150_predictor_arms": arms,
        "e150_primary_arm": capturable_name,
        "e150_capturable_pp_at_measured_sigma": capturable,
        "e150_capturable_over_shipped_pp": capturable - R7_SHIPPED_PCT,
        # The pre-registered single-arm reading, kept so the headline cannot
        # be mistaken for the only number this rung produced.
        "e150_capturable_pp_argmax_rule_only": primary,
        "e150_capturable_arm_selection_n": len(deployable),
        "e150_capturable_deployable_arms": {
            name: deployable[name]["median_pct_mean"] for name in deployable},
        "e150_capturable_per_prompt_ratio":
            deployable[capturable_name].get("per_prompt_ratio"),
        "e150_capturable_median_pct_sd":
            deployable[capturable_name]["median_pct_sd"],
        "e150_capturable_frac_rounds_inadmissible": headline_inadmissible,
        "e150_capturable_width_histogram":
            deployable[capturable_name]["width_histogram"],
        # The headline clears the score gate only. Rule 138 admits widths 1
        # to 5, so a non-zero share here means the arm is not deployable as
        # measured, however large its replay gain is.
        "e150_capturable_within_admissible_widths": headline_inadmissible == 0,
        "e150_pre_draft_predictor_sigma": sigma_out,
        "e150_pre_draft_predictor_reachable_sigma": reachable_out,
        "e150_pre_draft_predictor_scalar_k_sd": scalar_sd_out,
        "e150_pre_draft_predictor_scalar_k_bias": scalar_bias_out,
        "e150_predictor_sigma_in_sample": sigma_in,
        "e150_predictor_overfit_gap_pp": (
            arms["L4_argmax_noclamp_in_sample"]["median_pct_mean"] - primary),
        "e150_predictor_overfit_sigma_gap": sigma_out - sigma_in,
        "e150_r73_noise_ladder_reindexed": noise_rows,
        "e150_displacement_ladder": shift_rows,
        "e150_iid_gaussian_equivalent_pct": equivalent_noise,
        "e150_iid_gaussian_reachable_equivalent_pct":
            equivalent_noise_reachable,
        "e150_displacement_equivalent_pct": equivalent_shift,
        "e150_displacement_wrong_unit_equivalent_pct":
            equivalent_shift_wrong_unit,
        "e150_iid_ladder_overstates_by_pp": equivalent_noise - primary,
        "e150_iid_reachable_ladder_overstates_by_pp":
            equivalent_noise_reachable - primary,
        "e150_displacement_ladder_overstates_by_pp": equivalent_shift - primary,
        "e150_ladder_conversion_reads": {
            "iid_all_position": {
                "ladder": "e150_r73_noise_ladder_reindexed",
                "read_at_unit": "all-position per-entry rms",
                "read_at_value": sigma_out,
                "units_match": False,
                "predicted_pct": equivalent_noise,
            },
            "iid_reachable_only": {
                "ladder": "e150_r73_noise_ladder_reindexed",
                "read_at_unit": "per-entry rms restricted to i <= K",
                "read_at_value": reachable_out,
                "units_match": False,
                "predicted_pct": equivalent_noise_reachable,
            },
            "displacement_wrong_unit": {
                "ladder": "e150_displacement_ladder",
                "read_at_unit": "all-position per-entry rms",
                "read_at_value": sigma_out,
                "units_match": False,
                "predicted_pct": equivalent_shift_wrong_unit,
            },
            "displacement_matched_unit": {
                "ladder": "e150_displacement_ladder",
                "read_at_unit": "scalar E[K] error sd",
                "read_at_value": scalar_sd_out,
                "units_match": True,
                "predicted_pct": equivalent_shift,
            },
            "measured_no_conversion": {
                "ladder": None,
                "read_at_unit": "none, the predictor is priced directly",
                "read_at_value": None,
                "units_match": True,
                "predicted_pct": primary,
            },
        },
        "e150_ratio_rule_arms": {
            name: {
                "median_pct": row["median_pct_mean"],
                "median_pct_sd": row["median_pct_sd"],
                "weighted_mean_depth": row["weighted_mean_depth"],
                "width1_share": row["width_histogram"]["1"],
                # Rule 138 excludes width 8, and E150 R0.5 showed the
                # linearised rule reaches it. The headline arm comes from
                # this table, so its admissible-width profile has to be
                # recorded here or the result cannot be judged deployable.
                "frac_rounds_inadmissible": row["frac_rounds_inadmissible"],
                "width_histogram": row["width_histogram"],
                "weighted_accept_rate": row["weighted_accept_rate"],
            } for name, row in ratio_arms.items()
        },
        "e150_r0_composition_arms": {
            name: {
                "median_pct": row["median_pct_mean"],
                "median_pct_sd": row["median_pct_sd"],
                "weighted_mean_depth": row["weighted_mean_depth"],
                "frac_rounds_inadmissible": row["frac_rounds_inadmissible"],
                "width_histogram": row["width_histogram"],
                "weighted_accept_rate": row["weighted_accept_rate"],
            } for name, row in compose.items()
        },
        "e150_r0_composition_pct": compose_best,
        "e150_r0_composition_over_shipped_pp": (
            compose_best - compose["e128_ship"]["median_pct_mean"]),
        "e150_r0_composition_super_additive_pp": compose_super,
        "e150_r0_composition_beats_headline": bool(
            compose_best > arms["L4_ratio_noclamp"]["median_pct_mean"]),
        "e150_r0_composition_control_pp": compose_control,
        "e150_r0_composition_control_passed": bool(compose_control < 1e-9),
        "e150_shipped_cell_price_table_swap_pp": swap_pp,
        "e150_shipped_cell_is_price_table_swap": bool(compose_control < 1e-9),
        "e150_ratio_over_argmax_pp": (
            ratio_arms["L4_ratio_noclamp"]["median_pct_mean"] - primary),
        "e150_ratio_truth_ceiling_pp":
            ratio_arms["truth_ratio_noclamp"]["median_pct_mean"],
        "e150_ratio_truth_over_argmax_truth_pp": (
            ratio_arms["truth_ratio_noclamp"]["median_pct_mean"]
            - controls["truth_argmax_noclamp"]["median_pct_mean"]),
        "e150_constant_lambda_control": lam_rows,
        "e150_best_constant_lambda": best_lam["fixed_lambda"],
        "e150_best_constant_lambda_pct": best_lam["median_pct"],
        "e150_ratio_adaptivity_premium_pp": adaptive - best_lam["median_pct"],
        "e150_ratio_arm_sigma": ratio_tally.rms(),
        "e150_ratio_arm_scalar_k_sd": ratio_tally.scalar_sd(),
        "e150_shipped_information_ratio_pp":
            ratio_arms["shipped_ratio_noclamp"]["median_pct_mean"],
        "e150_rule_only_gain_pp": (
            ratio_arms["shipped_ratio_noclamp"]["median_pct_mean"]
            - controls["shipped_greedy_noclamp"]["median_pct_mean"]),
        "e150_information_gain_at_fixed_rule_pp": (
            ratio_arms["L4_ratio_noclamp"]["median_pct_mean"]
            - ratio_arms["shipped_ratio_noclamp"]["median_pct_mean"]),
        "e150_temperature_grid": list(TEMPERATURE_GRID),
        "e150_temperature_selection": temp_log,
        "e150_tempered_median_pct":
            arms["L4_argmax_noclamp_tempered"]["median_pct_mean"],
        "e150_tempered_over_untempered_pp": (
            arms["L4_argmax_noclamp_tempered"]["median_pct_mean"] - primary),
        "e150_tempered_sigma": temp_tally.rms(),
        "e150_information_ladder": information_ladder,
        "e150_information_ladder_detail": ladder_rows,
        "e150_error_distribution": residuals,
        "e150_error_distribution_is_gaussian": residuals.get("is_gaussian"),
        "e150_r1_gate_pct": R1_GATE_PCT,
        "e150_r1_gate_cleared": capturable >= R1_GATE_PCT,
        "e150_r1_gate_cleared_argmax_rule_only": primary >= R1_GATE_PCT,
        "e150_realised_headroom_axis_pp": (R7_ORACLE_DEPTH_PCT
                                           - R7_SHIPPED_PCT),
        "e150_axis_captured_frac": (
            (capturable - R7_SHIPPED_PCT)
            / (R7_ORACLE_DEPTH_PCT - R7_SHIPPED_PCT)),
    })

    if args.refit_iteration:
        print("\n## one refit iteration under the deployed policy")
        deployed = {p: models[p]["L4"] for p in names}
        redo, redo_gate = collect(env, fit_seeds, args.fit_sim_windows,
                                  rule="argmax", clamp={}, models=deployed)
        print("  rounds %d ; mismatches %d"
              % (redo_gate["rounds"], redo_gate["prev_acc_mismatch"]))
        refit = {}
        for held in names:
            train = pool(redo, [n for n in names if n != held])
            refit[held] = {"L4": fit_hazard(train, "L4", ridge=args.ridge)}
        tally = ErrorTally()
        row = price_predictor(env, refit, "L4", "argmax", {}, tally)
        print("  refitted L4_argmax_noclamp %+9.4f  sigma %.4f"
              % (row["median_pct_mean"], tally.rms()))
        out["e150_refit_under_deployed_policy"] = {
            "median_pct": row["median_pct_mean"],
            "median_pct_sd": row["median_pct_sd"],
            "out_of_sample_sigma": tally.rms(),
            "collection_gate": redo_gate,
            "shift_from_first_fit_pp": row["median_pct_mean"] - primary,
        }

    path = write_artifact(args.json.name, out)

    print("\n## R1 headline")
    print("  shipped reference                       %+8.4f %%"
          % R7_SHIPPED_PCT)
    print("  perfect per-round, shipped clamp on     %+8.4f %%"
          % controls["truth_argmax_clamped"]["median_pct_mean"])
    print("  perfect per-round, no clamp             %+8.4f %%"
          % controls["truth_argmax_noclamp"]["median_pct_mean"])
    print("  fitted pre-draft predictor, out of sample %+8.4f %%"
          % primary)
    print("  that is %+0.4f pp over shipped, %.2f %% of the axis"
          % (primary - R7_SHIPPED_PCT,
             100.0 * out["e150_axis_captured_frac"]))
    print("  out-of-sample per-entry sigma           %8.4f" % sigma_out)
    print("  in-sample per-entry sigma               %8.4f" % sigma_in)
    print("  reachable-only per-entry sigma          %8.4f" % reachable_out)
    print("  scalar E[K] error sd                    %8.4f" % scalar_sd_out)
    print("  iid ladder read at all-position sigma   %+8.4f %%  units differ"
          % equivalent_noise)
    print("  iid ladder read at reachable sigma      %+8.4f %%  units differ"
          % equivalent_noise_reachable)
    print("  displacement ladder at per-entry sigma  %+8.4f %%  units differ"
          % equivalent_shift_wrong_unit)
    print("  displacement ladder at scalar K sd      %+8.4f %%  units match"
          % equivalent_shift)
    print("\n  rule and information do not add:")
    print("    shipped information, shipped rule     %+8.4f %%"
          % controls["shipped_greedy_noclamp"]["median_pct_mean"])
    print("    shipped information, ratio rule       %+8.4f %%  (rule alone "
          "%+0.4f pp)"
          % (out["e150_shipped_information_ratio_pp"],
             out["e150_rule_only_gain_pp"]))
    print("    fitted information, argmax rule       %+8.4f %%" % primary)
    print("    fitted information, ratio rule        %+8.4f %%  (information "
          "at that rule %+0.4f pp)"
          % (arms["L4_ratio_noclamp"]["median_pct_mean"],
             out["e150_information_gain_at_fixed_rule_pp"]))
    print("    best constant lambda %.2f              %+8.4f %%  "
          "(adaptivity %+0.4f pp)"
          % (best_lam["fixed_lambda"], best_lam["median_pct"],
             out["e150_ratio_adaptivity_premium_pp"]))
    print("    perfect information, ratio rule       %+8.4f %%  (above the "
          "published oracle by %+0.4f pp)"
          % (out["e150_ratio_truth_ceiling_pp"],
             out["e150_ratio_truth_ceiling_pp"] - R7_ORACLE_DEPTH_PCT))
    print("  temperature over untempered             %+8.4f pp"
          % out["e150_tempered_over_untempered_pp"])
    print("  R1 gate at %+0.4f %% : %s"
          % (R1_GATE_PCT, "CLEARED" if out["e150_r1_gate_cleared"]
             else "NOT CLEARED"))
    print("  same gate, argmax-rule-only reading  : %s"
          % ("CLEARED" if out["e150_r1_gate_cleared_argmax_rule_only"]
             else "NOT CLEARED"))
    print("  headline arm inside Rule 138 widths  : %s  (inadmissible share "
          "%.4f)"
          % (out["e150_capturable_within_admissible_widths"],
             out["e150_capturable_frac_rounds_inadmissible"]))
    print("\nwrote %s" % path.relative_to(HERE.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
