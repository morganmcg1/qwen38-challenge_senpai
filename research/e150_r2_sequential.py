#!/usr/bin/env python3
"""E150 R2 L5/L6: price sequential head-side stopping parametrically.

`harness=local instrument`. Zero GPU. Zero Swift. Rule 79 binds: no timing
contrast is published here. Every number is an offline price against the E145
R3 measured per-width round cost curve.

Why this module exists, and what it is NOT
------------------------------------------
The brief asks for two rungs above L4:

    L5 = L4 plus the proposal head's own logits for draft position 1
    L6 = L5 plus the head's logits for positions 1..k, revealed sequentially

**The captured trace cannot supply either signal.** `research/e145_r7_state.py`
records the target's top-1 and top-2 values per verified row and the round's
realised capability. It records no proposal-head logits and no target hidden
state. So L5 and L6 cannot be measured on the evidence this campaign owns,
and advisor feedback F1 is explicit that a missing signal is reported as a
limit and never substituted with a different one.

What is priced instead is the DEMAND CURVE for that signal: given a head-side
signal of assumed discriminative power, what would sequential stopping be
worth? That converts an unmeasurable rung into a decision. If the premium is
small even at implausibly high signal quality, R3's host round trip is never
worth paying for and the campaign can close the axis without building the
readback at all.

The signal model
----------------
At draft position `i` the head emits a scalar whose discriminative power for
the binary event "position `i` is accepted" is a declared AUC. The standard
equal-variance bi-normal model gives

    s_i ~ N(d', 1) if position i is accepted, N(0, 1) otherwise
    d'  = sqrt(2) * Phi^{-1}(AUC)

so AUC is the only free parameter and AUC = 0.5 means d' = 0, a signal that
carries nothing. That is the module's own positive control: at AUC = 0.5 the
sequential arms must reproduce the one-shot greedy arm exactly, because the
posterior collapses to the prior.

Signals are generated from the round's realised capability, which makes them
a CEILING rather than an implementable arm. The decision rule reads only the
sampled scalars, never the capability itself.

Posterior
---------
With prior hazards `p[i]` the prior reach is `P(K >= k) = prod_{i<k} p[i]`.
Position `i` is accepted exactly when `K >= i`, so after drafting `d`
positions and observing `s_1..s_d` the posterior weight on `K = k` is

    w_k  =  P(K = k) * prod_{i=1..min(d,k)} LR(s_i)
    LR(s) = exp(d' * s - d'^2 / 2)

and the extension decision at depth `d` compares the expected marginal token
against the measured marginal cost under the same linearised objective R0.5
solved:

    continue iff  mu * P(K >= d+1 | s)  >  cumulative[d+1] - cumulative[d]

Usage:
  python3 e150_r2_sequential.py --json r2seq.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import e150_lib  # noqa: E402
from e128_price import MAX_DEPTH, PRICE_CUMULATIVE  # noqa: E402
from e128_replay import SEGMENTED_VERIFY_DEPTH_CAP  # noqa: E402
from e150_lib import (  # noqa: E402
    R7_SHIPPED_PCT, build_env, score_arm, write_artifact)
from e150_predict import EMA_PRIOR_LIST, clamp_state  # noqa: E402

# 0.5 is the no-information control. 0.865 is PAD's published target-side AUC
# (arXiv:2511.00351) and is the only value here with an external referent;
# every other point is a hypothetical. 1.0 is the perfect-signal ceiling.
AUC_GRID = (0.50, 0.60, 0.70, 0.80, 0.865, 0.90, 0.95, 1.00)

# Above this the bi-normal separation diverges, so the perfect-signal row is
# priced at a large but finite d'.
MAX_DPRIME = 12.0

# The shipped depth-0 and depth-1 margin clamp scales, lifted verbatim from
# `Qwen36MTPBlockSession` and from E145 R7-2's `clamped` cell.
SHIPPED_CLAMP_SCALES = {0: 2.0, 1: 3.0}


def dprime(auc: float) -> float:
    """Equal-variance bi-normal separation for a declared AUC."""
    if auc >= 1.0:
        return MAX_DPRIME
    if auc <= 0.5:
        return 0.0
    # Phi^{-1} without scipy: invert the error function by bisection, which
    # is exact enough for a parameter that is itself an assumption.
    lo, hi = 0.0, MAX_DPRIME
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / 2.0)) < auc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def prior_pmf(state, cap: int) -> np.ndarray:
    """`P(K = k)` for `k` in `0..cap` from a hazard vector."""
    reach = np.ones(cap + 2)
    acc = 1.0
    for k in range(1, cap + 2):
        acc *= float(state[k - 1]) if k - 1 < len(state) else 0.0
        reach[k] = acc
    pmf = reach[:cap + 1] - reach[1:cap + 2]
    pmf = np.clip(pmf, 0.0, None)
    total = pmf.sum()
    # A degenerate hazard vector can zero the whole mass; fall back to all
    # weight on K = 0 rather than dividing by zero.
    return pmf / total if total > 0 else np.eye(cap + 1)[0]


def sequential_depth(state, capability: int, mu: float, d_prime: float,
                     cumulative, cap: int, rng: np.random.Generator,
                     reveal_every_step: bool) -> int:
    """Greedy stopping under the posterior, one draft position at a time.

    `reveal_every_step` selects L6. When false only the first drafted
    position returns a signal, which is the cheap L5 variant that needs one
    host round trip per round instead of one per draft step.
    """
    if cap <= 0:
        return 0
    pmf = prior_pmf(state, cap)
    ks = np.arange(cap + 1)
    log_w = np.log(np.clip(pmf, 1e-300, None))
    depth = 0
    while depth < cap:
        # Expected marginal token against measured marginal cost.
        w = np.exp(log_w - log_w.max())
        p_reach = w[ks >= depth + 1].sum() / w.sum()
        gain = mu * p_reach
        cost = cumulative[depth + 1] - cumulative[depth]
        if gain <= cost:
            break
        depth += 1
        # The position was drafted, so its signal now exists.
        if d_prime > 0.0 and (reveal_every_step or depth == 1):
            accepted = capability >= depth
            s = rng.normal(d_prime if accepted else 0.0, 1.0)
            log_lr = d_prime * s - 0.5 * d_prime * d_prime
            log_w = log_w + np.where(ks >= depth, log_lr, 0.0)
    return depth


def sequential_walker(mu: float, auc: float, clamp, mode: str, seed_key: int,
                      cap_limit: int = SEGMENTED_VERIFY_DEPTH_CAP):
    """A chooser that walks depth sequentially under head-side signals.

    `mode` is `oneshot`, `l5` or `l6`. `oneshot` reveals nothing and is the
    reference the AUC = 0.5 rows must reproduce.
    """
    d_prime = 0.0 if mode == "oneshot" else dprime(auc)
    reveal_every = mode == "l6"
    # One generator per arm per prompt, advanced round by round, so an arm is
    # deterministic and two arms at the same AUC see the same noise draw.
    state = {"rng": np.random.default_rng(seed_key), "round": 0}

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        ema_list = list(ema)
        if ema_list == EMA_PRIOR_LIST:
            state["rng"] = np.random.default_rng(seed_key)
            state["round"] = 0
        state["round"] += 1
        # The shipped walk expresses "clamped" by passing the real margin so
        # its own internal sigmoid fires, and "no clamp" by passing NaN. This
        # loop reads the hazard vector directly and never calls that walk, so
        # the shipped clamp has to be applied here explicitly to keep the two
        # arms comparable with every other E150 cell.
        scales = SHIPPED_CLAMP_SCALES if clamp is None else clamp
        vector = clamp_state(ema_list, margin, scales)
        cumulative = (price or (None, PRICE_CUMULATIVE))[1]
        cap = min(min(offer, MAX_DEPTH), cap_limit)
        return sequential_depth(vector, int(ctx["capability"]), mu, d_prime,
                                cumulative, cap, state["rng"], reveal_every)

    return chooser


def price(env, mu: float, auc: float, clamp, mode: str) -> dict:
    def make(seed, prompt, entry):
        key = abs(hash((mode, round(auc, 4), seed, prompt))) % (2 ** 31)
        return sequential_walker(mu, auc, clamp, mode, key)
    return score_arm(env, "measured", env.measured, env.measured_price, make)


def summarise(row: dict) -> dict:
    return {
        "median_pct": row["median_pct_mean"],
        "median_pct_sd": row["median_pct_sd"],
        "weighted_mean_depth": row["weighted_mean_depth"],
        "weighted_accept_rate": row["weighted_accept_rate"],
        "width1_share": row["width_histogram"]["1"],
        "width_histogram": row["width_histogram"],
        "frac_rounds_inadmissible": row["frac_rounds_inadmissible"],
    }


def resolve_mu(args) -> tuple[float, str]:
    if args.mu is not None:
        return args.mu, "command line"
    path = HERE / "e150-artifacts" / "r05.json"
    if path.exists():
        solved = json.loads(path.read_text()).get(
            "e150_lambda_star_solution", {})
        mu = solved.get("mu_star_cost_per_token_normalised")
        if mu is not None:
            return float(mu), "r05.json fixed point"
    raise SystemExit("no mu available: run e150_r05.py first or pass --mu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--mu", type=float, default=None,
                    help="normalised cost per token; defaults to R0.5's fixed "
                         "point")
    ap.add_argument("--json", type=pathlib.Path,
                    default=pathlib.Path("r2seq.json"))
    args = ap.parse_args()

    mu, mu_source = resolve_mu(args)
    env = build_env(windows=args.windows, seeds=args.seeds)

    print("E150 R2 L5/L6 — parametric price of a head-side sequential signal")
    print("  harness=local  gpu_used=False  rule_79=not_engaged")
    print("  mu %.6f  (%s)" % (mu, mu_source))
    print("  windows %d  seeds %d" % (args.windows, args.seeds))
    print("\n## the limit this rung reports")
    print("  the captured trace carries no proposal-head logits and no target")
    print("  hidden state, so L5 and L6 are NOT measured. what follows is the")
    print("  demand curve for a signal of assumed AUC.")

    rows, table = [], {}
    for clamp_name, clamp in (("noclamp", {}), ("clamped", None)):
        name = "oneshot_%s" % clamp_name
        table[name] = summarise(price(env, mu, 0.5, clamp, "oneshot"))
        print("\n## one-shot greedy linearised reference, %s" % clamp_name)
        print("  %-28s %+9.4f  depth %.4f"
              % (name, table[name]["median_pct"],
                 table[name]["weighted_mean_depth"]))

    print("\n## the demand curve, no clamp")
    print("  %-7s %10s %10s %10s %9s %9s"
          % ("auc", "L5", "L6", "L6-L5", "L5 depth", "L6 depth"))
    reference = table["oneshot_noclamp"]["median_pct"]
    for auc in AUC_GRID:
        l5 = summarise(price(env, mu, auc, {}, "l5"))
        l6 = summarise(price(env, mu, auc, {}, "l6"))
        table["l5_auc%.3f" % auc] = l5
        table["l6_auc%.3f" % auc] = l6
        rows.append({
            "auc": auc,
            "dprime": dprime(auc),
            "l5_median_pct": l5["median_pct"],
            "l6_median_pct": l6["median_pct"],
            "premium_pp": l6["median_pct"] - l5["median_pct"],
            "l5_over_oneshot_pp": l5["median_pct"] - reference,
            "l6_over_oneshot_pp": l6["median_pct"] - reference,
            "l6_over_shipped_pp": l6["median_pct"] - R7_SHIPPED_PCT,
            "l5_mean_depth": l5["weighted_mean_depth"],
            "l6_mean_depth": l6["weighted_mean_depth"],
            "l6_frac_rounds_inadmissible": l6["frac_rounds_inadmissible"],
        })
        print("  %-7.3f %+10.4f %+10.4f %+10.4f %9.4f %9.4f"
              % (auc, l5["median_pct"], l6["median_pct"],
                 l6["median_pct"] - l5["median_pct"],
                 l5["weighted_mean_depth"], l6["weighted_mean_depth"]))

    # The control. At AUC 0.5 the posterior is the prior, so both sequential
    # arms must land on the one-shot arm. If they do not, the posterior or the
    # signal wiring is wrong and no row above is readable.
    zero = rows[0]
    control_l5 = abs(zero["l5_median_pct"] - reference)
    control_l6 = abs(zero["l6_median_pct"] - reference)
    controls_ok = control_l5 < 1e-9 and control_l6 < 1e-9
    print("\n## control: AUC 0.5 must reproduce the one-shot arm exactly")
    print("  |L5 - oneshot| %.3e   |L6 - oneshot| %.3e   pass %s"
          % (control_l5, control_l6, controls_ok))

    best = max(rows, key=lambda r: r["premium_pp"])
    at_pad = next(r for r in rows if r["auc"] == 0.865)
    perfect = rows[-1]

    out = {
        "rung": "E150 R2 L5/L6",
        "question": "what would a head-side sequential signal be worth?",
        "harness": "local",
        "gpu_used": False,
        "frame": "decode",
        "windows": args.windows,
        "seeds": env.seeds,
        "receipt": env.receipt,
        "mu": mu,
        "mu_source": mu_source,
        "e150_l5_l6_measured": False,
        "e150_l5_l6_limit": (
            "the captured trace carries no proposal-head logits and no target "
            "hidden state, so L5 and L6 are priced parametrically in assumed "
            "signal AUC and are not measured"),
        "e150_r2_sequential_table": table,
        "e150_r2_sequential_demand_curve": rows,
        "e150_r2_sequential_controls_ok": controls_ok,
        "e150_r2_sequential_control_l5_error": control_l5,
        "e150_r2_sequential_control_l6_error": control_l6,
        "e150_oneshot_reference_pct": reference,
        "e150_sequential_information_premium_pp": at_pad["premium_pp"],
        "e150_sequential_premium_at_pad_auc_pp": at_pad["premium_pp"],
        "e150_sequential_premium_best_pp": best["premium_pp"],
        "e150_sequential_premium_best_auc": best["auc"],
        "e150_sequential_premium_at_perfect_signal_pp": perfect["premium_pp"],
        "e150_l5_over_oneshot_at_pad_auc_pp": at_pad["l5_over_oneshot_pp"],
        "e150_l6_over_oneshot_at_pad_auc_pp": at_pad["l6_over_oneshot_pp"],
    }
    # Rule 134: a premium only survives if it outruns the host round trip R3
    # would have to pay for it. One readback per step is the L6 toll; L5 pays
    # one per round and is charged in R3's own units there.
    out["e150_sequential_premium_us_per_round_break_even"] = (
        515.2 * at_pad["premium_pp"] / 1.0 if at_pad["premium_pp"] > 0 else 0.0)

    path = write_artifact(args.json, out)

    print("\n## R2 L5/L6 headline")
    print("  one-shot greedy linearised          %+9.4f %%" % reference)
    print("  L6 minus L5 at PAD's AUC 0.865      %+9.4f pp"
          % at_pad["premium_pp"])
    print("  L6 minus L5 at a perfect signal     %+9.4f pp"
          % perfect["premium_pp"])
    print("  largest premium anywhere on the grid %+8.4f pp at AUC %.3f"
          % (best["premium_pp"], best["auc"]))
    print("  break-even host readback cost        %8.1f us/round"
          % out["e150_sequential_premium_us_per_round_break_even"])
    print("  R3 stop rule is 100 us/round, so R3 is %s"
          % ("WORTH RUNNING"
             if out["e150_sequential_premium_us_per_round_break_even"] > 100.0
             else "NOT worth running"))
    if not controls_ok:
        print("  WARNING: the AUC 0.5 control missed, so no row is readable.")
    print("\nwrote %s" % path.relative_to(HERE.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
