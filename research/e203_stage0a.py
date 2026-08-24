#!/usr/bin/env python3
"""E203 stage 0a: does WITHIN-prompt acceptance structure exist at all?

harness=ranked is not claimed here. This stage prints no score and no time; it
measures the serial structure of the accepted-draft-length sequence inside a
prompt. Stage 0b converts whatever structure survives into ranked economics.

WHY THE `p7` ARM. Every within-prompt depth rule -- drift tracking, EMA
lookahead, a per-prompt price scaled jointly with lookahead -- is a map from
the acceptance HISTORY of the current prompt to the next round's depth. Such a
rule can only pay if the history predicts the next round. The shipped arm
cannot answer that question, because its own depth choice is a function of the
same history and its acceptance is censored at that depth, so serial structure
in the shipped `acc=` sequence is partly manufactured by the rule. The e168
`p7` arm holds the drafted depth at 7 for every round except the few the token
budget narrows at the tail, so its `acc=` sequence is an open-loop observation
of the process a lookahead rule would have to predict.

THE NULL. Exchangeability: the rounds of one prompt are a random order of the
same multiset. Under that null no causal rule can beat the best constant
depth for that prompt, whatever it observes. The null is enforced by
permutation, so it needs no distributional assumption and it automatically
carries the finite-sample bias of every statistic reported against it.

FOUR STATISTICS, all one-sided in the direction a rule would need:

  acf(l)        Pearson autocorrelation at lags 1..16. A drift or clustering
                rule needs acf > 0 at short lags.
  drift         correlation of accepted length with the round index. A rule
                that changes depth over the decode window needs this.
  ema_skill     one-step-ahead prediction-error reduction of the implementable
                EMA predictor over the causal running mean, swept over alpha.
                This is the shipped tracker's own functional form.
  ar_skill      same, for a causal expanding-window AR(L) least-squares
                predictor. This is the best LINEAR use of the history and
                upper-bounds the EMA family.

Both skills are scored ONE STEP AHEAD from a causal expanding window, so no
number here uses information a live rule would not have.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e203_traces as T  # noqa: E402

LAGS = list(range(1, 17))
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]
AR_ORDERS = [1, 2, 4, 8]
WARMUP = 16
ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e203-artifacts"


# ------------------------------------------------------------------ series

def ema_expected_accept(ema: list[float], depth: int) -> float:
    """The accepted count the SHIPPED tracker itself expects for this round.

    `costModelDepth` reads the position EMA vector as a reach product: the
    chance of accepting position k is the product of the first k entries. The
    expected accepted count is the sum of those reaches over the drafted
    positions. This is the shipped rule's own forecast, so correlating it with
    the realised accepted count asks whether the shipped tracker is tracking
    anything.
    """
    reach, total = 1.0, 0.0
    for k in range(min(depth, len(ema))):
        reach *= ema[k]
        total += reach
    return total


def series(arm: str, fixed_depth: int | None):
    """Per-prompt accepted-length sequences, with the tail-narrowed rounds cut.

    `QwenRuntimeMTPDriver` narrows the offered depth at the end of the 512-token
    window, so the last rounds of a `p7` leg run at d < 7. Those rounds are
    censored differently from the rest and are dropped; the count is reported.
    """
    out = {}
    for name, leg in T.corpus(arm).items():
        rounds = leg["rounds"]
        if fixed_depth is None:
            kept = rounds
            dropped = 0
        else:
            kept = [r for r in rounds if r["depth"] == fixed_depth]
            dropped = len(rounds) - len(kept)
        out[name] = {
            "acc": np.array([r["accepted"] for r in kept], dtype=float),
            "depth": np.array([r["depth"] for r in kept], dtype=float),
            "forecast": np.array(
                [ema_expected_accept(r["ema"], r["depth"]) for r in kept],
                dtype=float),
            "n_all": len(rounds), "n_kept": len(kept), "dropped": dropped,
            "censored": float(np.mean([r["accepted"] == r["depth"]
                                       for r in kept])),
        }
    return out


# --------------------------------------------------------------- statistics

def acf(x: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at `lag`, mean removed once over the series."""
    if x.shape[-1] <= lag + 2:
        return math.nan
    c = x - x.mean(axis=-1, keepdims=True)
    num = (c[..., :-lag] * c[..., lag:]).sum(axis=-1)
    den = (c * c).sum(axis=-1)
    return num / np.where(den > 0, den, np.nan)


def drift(x: np.ndarray) -> float:
    """Pearson correlation of the series with its own round index."""
    n = x.shape[-1]
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    c = x - x.mean(axis=-1, keepdims=True)
    den = np.sqrt((c * c).sum(axis=-1) * (t * t).sum())
    return (c * t).sum(axis=-1) / np.where(den > 0, den, np.nan)


def ema_errors(x: np.ndarray, alpha: float, warmup: int):
    """Squared one-step-ahead errors of the EMA predictor. x may be (B, N)."""
    x = np.atleast_2d(x)
    b, n = x.shape
    state = x[:, 0].copy()
    err = np.zeros((b, n))
    for i in range(1, n):
        err[:, i] = (x[:, i] - state) ** 2
        state = alpha * x[:, i] + (1.0 - alpha) * state
    return err[:, warmup:]


def runmean_errors(x: np.ndarray, warmup: int):
    """Squared one-step-ahead errors of the CAUSAL running mean.

    This is the exchangeable-optimal predictor: under the null the best
    estimate of the next round is the mean of the rounds already seen.
    """
    x = np.atleast_2d(x)
    b, n = x.shape
    csum = np.cumsum(x, axis=1)
    idx = np.arange(1, n, dtype=float)
    pred = csum[:, :-1] / idx
    err = np.zeros((b, n))
    err[:, 1:] = (x[:, 1:] - pred) ** 2
    return err[:, warmup:]


def ar_errors(x: np.ndarray, order: int, warmup: int):
    """Squared one-step-ahead errors of a causal expanding-window AR(order).

    The coefficients used at round r are the least-squares fit on rounds
    order+1..r-1 only, refit every round, so the predictor is strictly causal.
    The refits share one cumulative Gram matrix instead of re-accumulating the
    window, which is what makes the permutation null affordable. A small ridge
    keeps the early, nearly singular windows finite.
    """
    x = np.atleast_2d(x)
    b, n = x.shape
    k = order + 1
    first = max(warmup, 2 * order + 6)             # first SCORED round
    lags = np.stack([x[:, order - j: n - j] for j in range(1, order + 1)],
                    axis=-1)                      # (b, n-order, order)
    design = np.concatenate([np.ones(lags.shape[:2] + (1,)), lags], axis=-1)
    target = x[:, order:]
    gram = np.cumsum(design[..., :, None] * design[..., None, :], axis=1)
    rhs = np.cumsum(design * target[..., None], axis=1)
    rows = np.arange(first - order, n - order)     # index into the design
    A = gram[:, rows - 1] + 1e-6 * np.eye(k)
    # numpy 2 reads a non-1-D right-hand side as a matrix stack, so the
    # vector systems carry an explicit trailing axis.
    beta = np.linalg.solve(A, rhs[:, rows - 1][..., None])[..., 0]
    pred = np.einsum("bmi,bmi->bm", design[:, rows], beta)
    return (target[:, rows] - pred) ** 2, first


def skill(model_err: np.ndarray, base_err: np.ndarray) -> np.ndarray:
    """1 - MSE_model / MSE_base, pooled over the scored rounds."""
    num = model_err.sum(axis=-1)
    den = base_err.sum(axis=-1)
    return 1.0 - num / np.where(den > 0, den, np.nan)


# ------------------------------------------------------------- permutations

def permute(rng, x: np.ndarray, b: int) -> np.ndarray:
    """b exchangeable resamples of one prompt's sequence."""
    n = x.shape[0]
    idx = np.argsort(rng.random((b, n)), axis=1)
    return x[idx]


def block_permute(rng, x: np.ndarray, b: int, block: int) -> np.ndarray:
    """b resamples that keep blocks of `block` rounds intact and reorder them.

    This null keeps short-range dependence and destroys position, so it is the
    honest null for the DRIFT statistic: a drift that survives it is a real
    trend and not an artefact of local clustering.
    """
    n = x.shape[0]
    nb = n // block
    head = x[: nb * block].reshape(nb, block)
    tail = x[nb * block:]
    order = np.argsort(rng.random((b, nb)), axis=1)
    out = head[order].reshape(b, nb * block)
    if tail.size:
        out = np.concatenate([out, np.tile(tail, (b, 1))], axis=1)
    return out


def pvalue(obs: float, null: np.ndarray, side: str) -> float:
    """Permutation p-value with the +1 correction, never zero."""
    null = null[np.isfinite(null)]
    if null.size == 0 or not math.isfinite(obs):
        return math.nan
    if side == "greater":
        hits = int((null >= obs).sum())
    elif side == "less":
        hits = int((null <= obs).sum())
    else:
        hits = int((np.abs(null) >= abs(obs)).sum())
    return (1 + hits) / (null.size + 1)


# ------------------------------------------------------------------ analysis

def analyse(arm: str, fixed_depth: int | None, draws: int,
            ar_draws: int, seed: int):
    rng = np.random.default_rng(seed)
    data = series(arm, fixed_depth)
    names = list(data)

    # one shared permutation index set per prompt, reused by every statistic,
    # so the pooled null is a single coherent resample of the whole corpus.
    perms = {n: permute(rng, data[n]["acc"], draws) for n in names}
    blocks = {n: block_permute(rng, data[n]["acc"], draws, 16) for n in names}

    out = {"arm": arm, "fixed_depth": fixed_depth, "draws": draws,
           "seed": seed, "warmup": WARMUP, "prompts": {}, "pooled": {}}

    # ---------------------------------------------------------------- acf
    per_lag_obs, per_lag_null, weights = {}, {}, {}
    for lag in LAGS:
        obs_v, null_v, w_v = [], [], []
        for n in names:
            x = data[n]["acc"]
            if x.size <= lag + 2:
                continue
            obs_v.append(float(acf(x, lag)))
            null_v.append(acf(perms[n], lag))
            w_v.append(x.size - lag)
        per_lag_obs[lag] = obs_v
        per_lag_null[lag] = np.array(null_v)      # (prompts, draws)
        weights[lag] = np.array(w_v, dtype=float)

    acf_table = []
    for lag in LAGS:
        w = weights[lag]
        obs = float(np.dot(w, per_lag_obs[lag]) / w.sum())
        null = (w[:, None] * per_lag_null[lag]).sum(axis=0) / w.sum()
        acf_table.append({
            "lag": lag, "pooled": obs,
            "null_mean": float(np.nanmean(null)),
            "null_sd": float(np.nanstd(null)),
            "null_lo": float(np.nanpercentile(null, 2.5)),
            "null_hi": float(np.nanpercentile(null, 97.5)),
            "z": float((obs - np.nanmean(null)) / np.nanstd(null)),
            "p_greater": pvalue(obs, null, "greater"),
            "p_two_sided": pvalue(obs, null, "two"),
            "per_prompt": dict(zip(names, per_lag_obs[lag])),
        })
    out["acf"] = acf_table

    # maximum pooled acf over lags 1..16, with the selection paid for by the
    # same maximisation inside every null draw.
    obs_max = max(row["pooled"] for row in acf_table)
    null_stack = np.stack([
        (weights[lag][:, None] * per_lag_null[lag]).sum(axis=0)
        / weights[lag].sum() for lag in LAGS])
    out["acf_max"] = {
        "lag": int(max(acf_table, key=lambda r: r["pooled"])["lag"]),
        "pooled": obs_max,
        "null_hi": float(np.nanpercentile(null_stack.max(axis=0), 97.5)),
        "p_greater": pvalue(obs_max, null_stack.max(axis=0), "greater"),
    }

    # -------------------------------------------------------------- drift
    obs_v, null_v, w_v = [], [], []
    for n in names:
        x = data[n]["acc"]
        obs_v.append(float(drift(x)))
        null_v.append(drift(blocks[n]))
        w_v.append(float(x.size))
    w = np.array(w_v)
    null = (w[:, None] * np.array(null_v)).sum(axis=0) / w.sum()
    obs = float(np.dot(w, obs_v) / w.sum())
    out["drift"] = {
        "pooled": obs, "per_prompt": dict(zip(names, obs_v)),
        "null_lo": float(np.nanpercentile(null, 2.5)),
        "null_hi": float(np.nanpercentile(null, 97.5)),
        "p_two_sided": pvalue(obs, null, "two"),
        "null_kind": "block-16 reorder (keeps local dependence, kills position)",
    }

    # ------------------------------------------------------- predictive skill
    def pooled_skill(kind, param, resamples, n_draws):
        """Corpus-level skill: MSE summed over prompts, not a mean of ratios.

        A mean of per-prompt ratios would let a short, low-variance prompt
        outvote the rounds that carry the decode time.
        """
        obs_num = obs_den = 0.0
        null_num = np.zeros(n_draws)
        null_den = np.zeros(n_draws)
        per_prompt = {}
        for n in names:
            x = data[n]["acc"]
            if kind == "ema":
                mdl = ema_errors(x, param, WARMUP)
                base = runmean_errors(x, WARMUP)
                nm = ema_errors(resamples[n], param, WARMUP)
                nb = runmean_errors(resamples[n], WARMUP)
            else:
                mdl, start = ar_errors(x, param, WARMUP)
                base = runmean_errors(x, start)
                nm, _ = ar_errors(resamples[n], param, WARMUP)
                nb = runmean_errors(resamples[n], start)
            per_prompt[n] = float(skill(mdl, base)[0])
            obs_num += float(mdl.sum())
            obs_den += float(base.sum())
            null_num += nm.sum(axis=-1)
            null_den += nb.sum(axis=-1)
        obs = 1.0 - obs_num / obs_den
        null = 1.0 - null_num / null_den
        return {"skill": obs, "per_prompt": per_prompt,
                "null_mean": float(np.nanmean(null)),
                "null_hi": float(np.nanpercentile(null, 97.5)),
                "p_greater": pvalue(obs, null, "greater"),
                "null": null}

    ema_rows, ema_nulls = [], []
    for alpha in ALPHAS:
        row = pooled_skill("ema", alpha, perms, draws)
        ema_nulls.append(row.pop("null"))
        row["alpha"] = alpha
        ema_rows.append(row)
    out["ema_skill"] = ema_rows
    best = max(ema_rows, key=lambda r: r["skill"])
    null_max = np.nanmax(np.stack(ema_nulls), axis=0)
    out["ema_best"] = {
        "alpha": best["alpha"], "skill": best["skill"],
        "null_hi_over_alpha": float(np.nanpercentile(null_max, 97.5)),
        "p_greater_over_alpha": pvalue(best["skill"], null_max, "greater"),
    }

    # the AR null refits a least-squares problem for every round of every
    # draw, so it runs on its own smaller, independent resample set.
    ar_perms = {n: permute(rng, data[n]["acc"], ar_draws) for n in names}
    out["ar_draws"] = ar_draws
    ar_rows, ar_nulls = [], []
    for order in AR_ORDERS:
        row = pooled_skill("ar", order, ar_perms, ar_draws)
        ar_nulls.append(row.pop("null"))
        row["order"] = order
        ar_rows.append(row)
    out["ar_skill"] = ar_rows
    best_ar = max(ar_rows, key=lambda r: r["skill"])
    null_max_ar = np.nanmax(np.stack(ar_nulls), axis=0)
    out["ar_best"] = {
        "order": best_ar["order"], "skill": best_ar["skill"],
        "null_hi_over_order": float(np.nanpercentile(null_max_ar, 97.5)),
        "p_greater_over_order": pvalue(best_ar["skill"], null_max_ar,
                                       "greater"),
    }

    # ----------------------------------------------------- structure budget
    # The one number stage 0b needs: how much of the round-to-round variance
    # of accepted length can ANY causal rule predict? For a stationary process
    # the one-step linear bound is rho1**2, so the honest ceiling input is the
    # upper end of the rho1 interval, not its point estimate.
    lag1 = next(r for r in acf_table if r["lag"] == 1)
    rho1_hi = lag1["pooled"] + 1.96 * lag1["null_sd"]
    rho_any_hi = out["acf_max"]["pooled"] + 1.96 * lag1["null_sd"]
    out["structure_budget"] = {
        "rho1": lag1["pooled"], "rho1_sd": lag1["null_sd"],
        "rho1_upper95": rho1_hi,
        "rho_any_lag_upper95": rho_any_hi,
        "predictable_variance_fraction_upper95": max(0.0, rho1_hi) ** 2,
        "predictable_variance_fraction_any_lag_upper95":
            max(0.0, rho_any_hi) ** 2,
    }

    # ------------------------------------------------- shipped tracker state
    # The shipped EMA vector is the state the live depth rule actually reads.
    # Correlating its own forecast with the realised accepted count asks the
    # question directly: is the shipped within-prompt tracker tracking
    # anything? Both series are de-meaned INSIDE each prompt, so between-prompt
    # differences -- which a per-prompt constant already captures -- cannot
    # manufacture a correlation.
    fx, ax = [], []
    per_prompt_corr = {}
    for n in names:
        f = data[n]["forecast"]
        a = data[n]["acc"]
        f = f - f.mean()
        a = a - a.mean()
        den = math.sqrt(float((f * f).sum() * (a * a).sum()))
        per_prompt_corr[n] = float((f * a).sum() / den) if den > 0 else math.nan
        fx.append(f)
        ax.append(a)
    f = np.concatenate(fx)
    a = np.concatenate(ax)
    corr = float((f * a).sum() / math.sqrt(float((f * f).sum() * (a * a).sum())))
    out["shipped_tracker"] = {
        "corr_forecast_vs_realised": corr,
        "n_rounds": int(a.size),
        "sd_under_null": 1.0 / math.sqrt(a.size),
        "per_prompt": per_prompt_corr,
    }

    # ------------------------------------------------------ leave-one-out
    # benchfixture is a long-copy prompt: 6.0 accepted per round against 1.2
    # to 1.7 for the prose prompts, and 80 % of its rounds are censored at the
    # drafted depth. A pooled statistic must not rest on it, or on any one
    # prompt.
    loo = {}
    for drop in names:
        keep = [n for n in names if n != drop]
        w = np.array([data[n]["acc"].size - 1.0 for n in keep])
        r1 = float(np.dot(w, [acf(data[n]["acc"], 1) for n in keep]) / w.sum())
        num = den = 0.0
        alpha = out["ema_best"]["alpha"]
        for n in keep:
            x = data[n]["acc"]
            num += float(ema_errors(x, alpha, WARMUP).sum())
            den += float(runmean_errors(x, WARMUP).sum())
        loo[drop] = {"rho1": r1, "ema_skill_at_best_alpha": 1.0 - num / den}
    out["leave_one_out"] = loo

    # -------------------------------------------------- drift sign agreement
    signs = [1 if v > 0 else -1 for v in out["drift"]["per_prompt"].values()]
    pos = sum(1 for s in signs if s > 0)
    out["drift_sign"] = {
        "positive": pos, "n": len(signs),
        "binomial_p_two_sided": 2.0 * sum(
            math.comb(len(signs), k) for k in range(max(pos, len(signs) - pos),
                                                    len(signs) + 1))
        / 2.0 ** len(signs),
    }

    for n in names:
        d = data[n]
        out["prompts"][n] = {
            "n_all": d["n_all"], "n_kept": d["n_kept"],
            "dropped_tail_rounds": d["dropped"],
            "acc_mean": float(d["acc"].mean()), "acc_sd": float(d["acc"].std()),
            "censored_fraction": d["censored"],
        }
    return out


# ------------------------------------------------------------------- report

def report(res, handle):
    def emit(text=""):
        print(text, file=handle)

    emit("=" * 78)
    emit("E203 STAGE 0a  WITHIN-PROMPT ACCEPTANCE STRUCTURE  arm=%s"
         % res["arm"])
    emit("=" * 78)
    emit("  corpus: research/out/e168/%s, 512 decode tokens per prompt"
         % res["arm"])
    emit("  null:   exchangeability by permutation, %d draws, seed %d"
         % (res["draws"], res["seed"]))
    if res["fixed_depth"]:
        emit("  rounds: drafted depth held at %d; tail-narrowed rounds dropped"
             % res["fixed_depth"])
    emit()
    emit("  PROMPTS")
    emit("   %-16s %5s %5s %5s  %7s %7s %7s"
         % ("prompt", "all", "kept", "drop", "mean", "sd", "cens"))
    for name, row in res["prompts"].items():
        emit("   %-16s %5d %5d %5d  %7.3f %7.3f %7.3f"
             % (name, row["n_all"], row["n_kept"], row["dropped_tail_rounds"],
                row["acc_mean"], row["acc_sd"], row["censored_fraction"]))
    emit()
    emit("  AUTOCORRELATION of accepted length, pooled over prompts by")
    emit("  (N - lag). A drift or clustering rule needs acf > 0.")
    emit("   %3s %9s %9s %9s %7s %8s"
         % ("lag", "pooled", "null_lo", "null_hi", "z", "p>"))
    for row in res["acf"]:
        flag = "  *" if row["p_greater"] < 0.05 else ""
        emit("   %3d %9.4f %9.4f %9.4f %7.2f %8.4f%s"
             % (row["lag"], row["pooled"], row["null_lo"], row["null_hi"],
                row["z"], row["p_greater"], flag))
    m = res["acf_max"]
    emit("   max over lags 1..16: %.4f at lag %d, null 97.5%% %.4f, p=%.4f"
         % (m["pooled"], m["lag"], m["null_hi"], m["p_greater"]))
    emit()
    d = res["drift"]
    emit("  DRIFT over the decode window (corr with round index)")
    emit("   pooled %.4f, null [%.4f, %.4f], p=%.4f"
         % (d["pooled"], d["null_lo"], d["null_hi"], d["p_two_sided"]))
    emit("   null: %s" % d["null_kind"])
    emit("   per prompt: " + "  ".join("%s=%+.3f" % (k, v)
                                       for k, v in d["per_prompt"].items()))
    emit()
    emit("  ONE-STEP-AHEAD PREDICTION SKILL vs the causal running mean")
    emit("  (positive = the history predicts the next round; this is the")
    emit("  quantity every drift/EMA/lookahead depth rule consumes)")
    emit("   %-8s %10s %10s %9s" % ("EMA a", "skill", "null97.5", "p>"))
    for row in res["ema_skill"]:
        emit("   %-8.2f %10.4f %10.4f %9.4f"
             % (row["alpha"], row["skill"], row["null_hi"], row["p_greater"]))
    b = res["ema_best"]
    emit("   best alpha %.2f skill %+.4f, null 97.5%% over alpha %.4f, p=%.4f"
         % (b["alpha"], b["skill"], b["null_hi_over_alpha"],
            b["p_greater_over_alpha"]))
    emit("   %-8s %10s %10s %9s" % ("AR(L)", "skill", "null97.5", "p>"))
    for row in res["ar_skill"]:
        emit("   %-8d %10.4f %10.4f %9.4f"
             % (row["order"], row["skill"], row["null_hi"], row["p_greater"]))
    b = res["ar_best"]
    emit("   best order %d skill %+.4f, null 97.5%% over order %.4f, p=%.4f"
         % (b["order"], b["skill"], b["null_hi_over_order"],
            b["p_greater_over_order"]))
    emit()
    s = res["structure_budget"]
    emit("  STRUCTURE BUDGET, the input stage 0b prices")
    emit("   rho1                       %+.4f (sd %.4f)"
         % (s["rho1"], s["rho1_sd"]))
    emit("   rho1 95%% upper             %+.4f" % s["rho1_upper95"])
    emit("   best lag 95%% upper         %+.4f" % s["rho_any_lag_upper95"])
    emit("   predictable variance of accepted length, 95%% upper bound:")
    emit("     from lag 1               %.4f %%"
         % (100 * s["predictable_variance_fraction_upper95"]))
    emit("     from the best of 16 lags %.4f %%"
         % (100 * s["predictable_variance_fraction_any_lag_upper95"]))
    emit()
    t = res["shipped_tracker"]
    emit("  SHIPPED TRACKER STATE: does the live EMA forecast the round?")
    emit("   corr(EMA forecast, realised accepted) = %+.4f over %d rounds"
         % (t["corr_forecast_vs_realised"], t["n_rounds"]))
    emit("   sd under no association %.4f, so the 95%% band is +-%.4f"
         % (t["sd_under_null"], 1.96 * t["sd_under_null"]))
    emit("   per prompt: " + "  ".join("%s=%+.3f" % (k, v)
                                       for k, v in t["per_prompt"].items()))
    emit("   Both series are de-meaned inside each prompt, so this measures")
    emit("   WITHIN-prompt tracking only.")
    emit()
    d = res["drift_sign"]
    emit("  DRIFT SIGN AGREEMENT across prompts: %d of %d positive, "
         "binomial p=%.3f" % (d["positive"], d["n"],
                              d["binomial_p_two_sided"]))
    emit("   A drift rule with a fixed sign needs agreement; a rule that")
    emit("   learns the sign online is the EMA family scored above.")
    emit()
    emit("  LEAVE-ONE-PROMPT-OUT on the two headline statistics")
    emit("   %-16s %9s %14s" % ("dropped", "rho1", "ema_skill"))
    for name, row in res["leave_one_out"].items():
        emit("   %-16s %+9.4f %+14.4f"
             % (name, row["rho1"], row["ema_skill_at_best_alpha"]))
    emit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="p7")
    ap.add_argument("--fixed-depth", type=int, default=7)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--ar-draws", type=int, default=600)
    ap.add_argument("--seed", type=int, default=203)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    fixed = None if args.fixed_depth <= 0 else args.fixed_depth
    res = analyse(args.arm, fixed, args.draws, args.ar_draws, args.seed)
    report(res, sys.stdout)
    if args.json:
        ARTIFACTS.mkdir(exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(res, indent=1))
        print("  wrote %s" % args.json)


if __name__ == "__main__":
    main()
