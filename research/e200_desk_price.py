#!/usr/bin/env python3
"""E200 desk pricing: what a width-aware depth price does to the RANKED median.

harness=ranked for every number this file prints.

RULE 79 carve-out. A local timing leg may not publish a depth-price contrast in
any direction. The decision instrument for a schedule policy is an offline
replay against a measured cost table, which is exactly what this file is: the
E197 merged ranked round-cost law (`research/e197_refit.py`, smooth-step,
AICc weight 0.567) plus the paid per-prompt schedule of receipt A.

THE LATENT-q REPLAY. `Qwen36MTPBlockSession.costModelDepth` is a hill climb on
one round's own state. To move it off the shipped uniform price we need the
per-round distribution of that state, which no receipt reports. The receipts do
report the per-prompt stopping-depth survival t_k under ONE known rule, so:

  * model a round by a single latent per-position acceptance q, so that
    `reach` after d steps is q**d and `expected` is sum_{j=1..d} q**j;
  * the shipped rule then reaches depth >= k iff q > Q_k, where Q_k is a
    constant of the rule alone (no prompt enters it);
  * therefore t_{k-1} = P(q > Q_k) pins the per-prompt survival function G of q
    at seven measured knots, exactly, with no family assumed;
  * a different price is a different Q'_k, and t'_{k-1} = G(Q'_k) is read off
    the same pinned G by monotone interpolation between those knots.

The one-dimensional q is the modelling assumption. The shipped rule's depth-0
and depth-1 margin clamps make the true per-position vector non-constant; the
clamps only LOWER p at those two positions, so G absorbs them at the shallow
knots. Every price compared here is read through the same G, so the assumption
is shared by the arms and cancels to first order in the contrast.

Levels stay measured. Round cost and mean acceptance are anchored on receipt A
and moved only by fit-based DIFFERENCES over the changed width mixture, which
is the same discipline as `e197_refit.chain`.
"""

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402
import e197_refit as E  # noqa: E402

ORDER = E.ORDER
TOKENS = E.TOKENS
CAP = 7                        # segmentedVerifyDepthCap; E199 owns this axis.
MAXD = 8                       # Qwen36MTPLimits.maxDepth
SHIP_H = 0.18                  # headStepCostRatio
GRID = 40001                   # q resolution of the rule thresholds
QGRID = 4001                   # q resolution of the per-prompt latent measure
ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e200-artifacts"


# ------------------------------------------------------------------ prices

def ship_price():
    """`makeUniformDepthPrice`, transcribed including its closed-form cumulative."""
    marginal = [SHIP_H] * MAXD
    cumulative = [1.0 + d * SHIP_H for d in range(MAXD + 1)]
    return {"marginal": marginal, "cumulative": cumulative}


def table_price(R_ms, level=None):
    """Width-aware price from a measured round-cost table R(1..9), ms.

    `marginal[d]` prices the step into verify width d + 2 and `cumulative[d]`
    is the round cost before that step, both normalised by the width-1 verify
    forward, which is the unit the shipped price already uses.

    `level` rescales the shape to a fixed total over the REACHABLE steps
    (0..CAP-1), holding the average price the scheduler can actually pay.
    Passing None keeps the measured level, which is the honest reading of the
    ranked law and is reported separately because it moves two things at once.
    """
    V = R_ms[0]
    raw = [(R_ms[d + 1] - R_ms[d]) / V for d in range(MAXD)]
    if level is not None:
        scale = level / sum(raw[:CAP])
        raw = [v * scale for v in raw]
    cumulative = [1.0]
    for v in raw:
        cumulative.append(cumulative[-1] + v)
    return {"marginal": raw, "cumulative": cumulative}


def blend_price(R_ms, w, level):
    """Shape interpolation between uniform (w=0) and the measured law (w=1)."""
    meas = table_price(R_ms, level=level)["marginal"]
    flat = level / CAP
    raw = [(1.0 - w) * flat + w * m for m in meas]
    scale = level / sum(raw[:CAP])
    raw = [v * scale for v in raw]
    cumulative = [1.0]
    for v in raw:
        cumulative.append(cumulative[-1] + v)
    return {"marginal": raw, "cumulative": cumulative}


# ------------------------------------------------------------------- rules

def greedy_depth(price, q, cap=CAP):
    """`costModelDepth` with a constant per-position acceptance q."""
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= q
        threshold = price["marginal"][depth] * (1.0 + expected) \
            / price["cumulative"][depth]
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def argmin_depth(price, q, cap=CAP):
    """Global argmin of round cost per token over the SAME price table.

    The shipped walk is a one-step hill climb, so a cost table with a step in
    it can trap it in a local minimum below the step. This rule reads the same
    table and takes the best depth outright.
    """
    best_d, best_v, expected = 0, price["cumulative"][0], 0.0
    for depth in range(cap):
        expected += q ** (depth + 1)
        value = price["cumulative"][depth + 1] / (1.0 + expected)
        if value < best_v:
            best_v, best_d = value, depth + 1
    return best_d


RULES = {"greedy": greedy_depth, "argmin": argmin_depth}


def thresholds(price, rule, cap=CAP):
    """Q_k: the smallest latent q that reaches depth >= k, k = 1..cap.

    Also returns whether depth is monotone in q on the grid; the survival
    inversion is only defined when it is.
    """
    fn = RULES[rule]
    prev, monotone = 0, True
    Q = [None] * (cap + 1)
    for i in range(1, GRID):
        q = i / GRID
        d = fn(price, q, cap)
        if d < prev:
            monotone = False
        for k in range(prev + 1, d + 1):
            Q[k] = q
        prev = max(prev, d)
    return Q, monotone


# --------------------------------------------------------------- survival

def survival_reader(t, Q_ship, mode="linear"):
    """G(x) = P(q > x), pinned at the shipped rule's knots by the paid t_k.

    Knots are (Q_k, t_{k-1}) for every reachable k, plus the trivial (0, 1)
    endpoint and (1, 0). Interpolation is monotone by construction because
    both sequences are monotone.
    """
    xs, ys = [0.0], [1.0]
    for k in range(1, CAP + 1):
        x = Q_ship[k]
        if x is None or x <= xs[-1]:
            continue
        xs.append(x)
        ys.append(min(t[k - 1], ys[-1]))
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(0.0)

    def G(x):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                lo, hi = i - 1, i
                break
        f = (x - xs[lo]) / (xs[hi] - xs[lo])
        if mode == "linear":
            return ys[lo] + f * (ys[hi] - ys[lo])
        a = math.log(max(ys[lo], 1e-9))
        b = math.log(max(ys[hi], 1e-9))
        return math.exp(a + f * (b - a))

    return G, (xs, ys)


# ---------------------------------------------------------------- pricing

def price_arm(data, tvar, acc, fit, price, rule, Q_ship, mode="linear",
              cap=CAP):
    """Per-prompt ranked raw ratios under one depth-price arm."""
    Q_new, monotone = thresholds(price, rule, cap)
    rows, raws = {}, []
    extrapolated = False
    for name in ORDER:
        t = tvar[name]
        S = acc[name]["S"]
        G, knots = survival_reader(t, Q_ship, mode)
        t_new = []
        prev = 1.0
        for k in range(1, cap + 1):
            x = Q_new[k]
            v = G(x) if x is not None else 0.0
            v = min(v, prev)
            t_new.append(v)
            prev = v
            if x is not None and (x < knots[0][1] or x > knots[0][-2]):
                extrapolated = True
        rec = data["A"]["rec"][name]
        abar = rec["abar"] + sum(S[k] * (t_new[k - 1] - t[k - 1])
                                 for k in range(1, cap + 1))
        R = rec["R"]
        for d in range(cap + 1):
            p_old = (t[d - 1] if d else 1.0) - (t[d] if d < cap else 0.0)
            p_new = (t_new[d - 1] if d else 1.0) - (t_new[d] if d < cap else 0.0)
            R += (p_new - p_old) * E.R_of(fit, d + 1)
        n_rounds = TOKENS / (1.0 + abar)
        raw = E.raw_of(data, name, n_rounds * R)
        raws.append(raw)
        rows[name] = {"t": t_new, "abar": abar, "R_ms": 1000 * R,
                      "edl": sum(t_new), "raw": raw,
                      "raw_A": data["A"]["vec"][name]["raw"]}
    return {"median": e177.published_median(raws), "rows": rows,
            "Q": Q_new, "monotone": monotone, "extrapolated": extrapolated}


# --------------------------------------- stage 2: q-conditional acceptance

def geom(p, d):
    """sum_{k=1..d} p**k, the expected accepted count at drafted depth d."""
    if d <= 0:
        return 0.0
    if abs(1.0 - p) < 1e-12:
        return float(d)
    return p * (1.0 - p ** d) / (1.0 - p)


def quadrature(t, Q_ship, mode="linear"):
    """Discrete latent-q measure implied by the pinned survival function."""
    G, _ = survival_reader(t, Q_ship, mode)
    qs, ws = [], []
    for i in range(QGRID):
        lo, hi = i / QGRID, (i + 1) / QGRID
        w = G(lo) - G(hi)
        if w > 0.0:
            qs.append(0.5 * (lo + hi))
            ws.append(w)
    return qs, ws


def fit_gamma(qs, ws, depths, abar_target):
    """Per-prompt EMA calibration: true per-position acceptance is q**gamma.

    The scheduler's own `reach` is q**k. If the position EMAs were calibrated,
    gamma would be 1. Anything below 1 means the rule under-estimates its own
    acceptance, which is the only way a price BELOW the measured marginal cost
    can be the end-to-end optimum. One parameter, fitted so the shipped arm
    reproduces the paid mean accepted count exactly.
    """
    def abar_of(gamma):
        return sum(w * geom(q ** gamma, d)
                   for q, w, d in zip(qs, ws, depths) if d)

    lo, hi = 0.05, 6.0
    if abar_of(hi) > abar_target:
        return hi, abar_of(hi)
    if abar_of(lo) < abar_target:
        return lo, abar_of(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if abar_of(mid) > abar_target:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    return g, abar_of(g)


def calibrate(data, tvar, Q_ship, ship, mode="linear"):
    """Build the per-prompt latent-q measure and its acceptance calibration."""
    out = {}
    for name in ORDER:
        qs, ws = quadrature(tvar[name], Q_ship, mode)
        depths = [greedy_depth(ship, q) for q in qs]
        rec = data["A"]["rec"][name]
        gamma, abar_fit = fit_gamma(qs, ws, depths, rec["abar"])
        S_pred = []
        for k in range(1, CAP + 1):
            num = sum(w * (q ** (gamma * k))
                      for q, w, d in zip(qs, ws, depths) if d >= k)
            den = sum(w for w, d in zip(ws, depths) if d >= k)
            S_pred.append(num / den if den > 1e-12 else 0.0)
        out[name] = {"q": qs, "w": ws, "depth_ship": depths, "gamma": gamma,
                     "abar_fit": abar_fit, "S_pred": S_pred}
    return out


def price_arm2(data, cal, fit, price, rule, cap=CAP, leg="A"):
    """Stage-2 pricing: the depth AND the acceptance both follow the latent q.

    This removes the stage-1 selection bias. Stage 1 credits every newly deep
    round with the per-position acceptance S_k that was measured on the rounds
    the SHIPPED rule already selected for that depth, so it over-values extra
    depth and under-values removed depth. Here a round that the new rule takes
    deeper is priced with its own q.
    """
    fn = RULES[rule]
    rows, raws = {}, []
    for name in ORDER:
        c = cal[name]
        gamma = c["gamma"]
        rec = data["A"]["rec"][name]
        abar, dR, edl = 0.0, 0.0, 0.0
        t_new = [0.0] * cap
        for q, w, d_ship in zip(c["q"], c["w"], c["depth_ship"]):
            d = fn(price, q, cap)
            abar += w * geom(q ** gamma, d)
            dR += w * (E.R_of(fit, d + 1) - E.R_of(fit, d_ship + 1))
            edl += w * d
            for k in range(1, d + 1):
                t_new[k - 1] += w
        R = rec["R"] + dR
        n_rounds = TOKENS / (1.0 + abar)
        raw = E.raw_of(data, name, n_rounds * R, leg)
        raws.append(raw)
        rows[name] = {"abar": abar, "R_ms": 1000 * R, "edl": edl, "raw": raw,
                      "raw_A": data["A"]["vec"][name]["raw"], "t": t_new}
    return {"median": e177.published_median(raws), "rows": rows}


# ----------------------------------------------------------------- report

def fmt_price(price):
    return "[" + " ".join("%.4f" % v for v in price["marginal"][:CAP]) + "]"


def main():
    data = E.build_receipts()
    tvar, _ = E.survival(data)
    acc = E.acceptance(data, tvar)
    obs = E.observations(data, tvar)
    fits = E.fit_all(obs)
    fit = fits["smooth-step"]
    R_ms = [1000 * E.R_of(fit, m) for m in range(1, E.MAXW + 1)]

    ship = ship_price()
    Q_ship, mono_ship = thresholds(ship, "greedy")

    print("=" * 78)
    print("E200 DESK PRICING  (harness=ranked)")
    print("=" * 78)
    print("  ranked law: smooth-step, AICc %.1f, wRMSE %.3f ms"
          % (fit["aicc"], fit["wrmse_ms"]))
    print("  R(1..9) ms: " + " ".join("%.2f" % v for v in R_ms))
    print("  dR(m->m+1): " + " ".join("%.3f" % (R_ms[i + 1] - R_ms[i])
                                      for i in range(8)))
    print("  cap %d, receipt A median %.8f, crown %.8f"
          % (CAP, E.BEST_A, E.CROWN))
    print()
    print("  shipped-rule latent-q knots Q_k (k = 1..%d), monotone=%s" %
          (CAP, mono_ship))
    print("   " + " ".join("Q%d=%.4f" % (k, Q_ship[k])
                           for k in range(1, CAP + 1)))
    print()

    base = price_arm(data, tvar, acc, fit, ship, "greedy", Q_ship)
    print("  MODEL CHECK: the shipped price replayed through its own knots")
    print("  reproduces receipt A exactly by construction: %.8f vs %.8f"
          % (base["median"], E.BEST_A))
    print()

    level_ship = SHIP_H * CAP
    arms = [
        ("ship uniform 0.18", ship, "greedy"),
        ("ranked dR, level held", table_price(R_ms, level_ship), "greedy"),
        ("ranked dR, measured level", table_price(R_ms), "greedy"),
        ("ranked dR, level held, argmin", table_price(R_ms, level_ship),
         "argmin"),
        ("ranked dR, measured level, argmin", table_price(R_ms), "argmin"),
        ("ship uniform, argmin", ship, "argmin"),
    ]
    print("  " + "-" * 74)
    print("  %-34s %11s %8s %8s" % ("arm", "median", "vs A", "vs crown"))
    print("  " + "-" * 74)
    results = {}
    for label, price, rule in arms:
        out = price_arm(data, tvar, acc, fit, price, rule, Q_ship)
        results[label] = out
        print("  %-34s %11.6f %+7.2f%% %+7.2f%%%s"
              % (label, out["median"], 100 * (out["median"] / E.BEST_A - 1),
                 100 * (out["median"] / E.CROWN - 1),
                 "" if out["monotone"] else "  NON-MONOTONE"))
    print("  " + "-" * 74)
    print()

    print("  price tables, marginal[0..%d] in units of the width-1 forward" %
          (CAP - 1))
    for label, price, _ in arms[:5]:
        print("   %-34s %s" % (label, fmt_price(price)))
    print()

    print("  Q_k under each arm (the latent-q bar for reaching depth k)")
    print("   %-34s %s" % ("arm", " ".join("Q%d" % k
                                           for k in range(1, CAP + 1))))
    for label, _, _ in arms:
        Q = results[label]["Q"]
        print("   %-34s %s" % (label, " ".join(
            "%.3f" % Q[k] if Q[k] is not None else "  -  "
            for k in range(1, CAP + 1))))
    print()

    for label in [a[0] for a in arms[1:]]:
        out = results[label]
        if abs(out["median"] / E.BEST_A - 1) < 0.001:
            continue
        print("  PER-PROMPT SHIFT: %s" % label)
        print("   %-9s %7s %7s %8s %8s %9s %9s" %
              ("prompt", "edl_A", "edl'", "R_A ms", "R' ms", "raw_A", "raw'"))
        for name in ORDER:
            r = out["rows"][name]
            print("   %-9s %7.3f %7.3f %8.3f %8.3f %9.5f %9.5f" %
                  (name, sum(tvar[name][:CAP]), r["edl"],
                   1000 * data["A"]["rec"][name]["R"], r["R_ms"],
                   r["raw_A"], r["raw"]))
        print()

    print("  SENSITIVITY: cost family, survival interpolation, S model")
    print("   %-34s %10s %10s %10s" %
          ("arm", "smooth-step", "step2+step6", "sat-exp"))
    for label, price, rule in arms[1:]:
        vals = []
        for key in ("smooth-step", "step2+step6", "sat-exp"):
            f = fits[key]
            Rk = [1000 * E.R_of(f, m) for m in range(1, E.MAXW + 1)]
            if "measured level" in label:
                p = table_price(Rk)
            elif "ranked dR" in label:
                p = table_price(Rk, level_ship)
            else:
                p = price
            Qs, _ = thresholds(ship, "greedy")
            vals.append(price_arm(data, tvar, acc, f, p, rule, Qs)["median"])
        print("   %-34s %10.6f %10.6f %10.6f" % (label, *vals))
    print()
    print("   %-34s %10s %10s" % ("arm", "G linear", "G log-linear"))
    for label, price, rule in arms[1:]:
        a = price_arm(data, tvar, acc, fit, price, rule, Q_ship, "linear")
        b = price_arm(data, tvar, acc, fit, price, rule, Q_ship, "log")
        print("   %-34s %10.6f %10.6f" % (label, a["median"], b["median"]))
    print()
    acc_curve = E.acceptance(data, tvar, model="curve")
    print("   %-34s %10s %10s" % ("arm", "S measured", "S curve"))
    for label, price, rule in arms[1:]:
        a = price_arm(data, tvar, acc, fit, price, rule, Q_ship)
        b = price_arm(data, tvar, acc_curve, fit, price, rule, Q_ship)
        print("   %-34s %10.6f %10.6f" % (label, a["median"], b["median"]))
    print()

    print("  SHAPE SWEEP: blend w from uniform to the measured ranked law,")
    print("  level held at %.2f, greedy walk" % level_ship)
    print("   %6s %11s %8s" % ("w", "median", "vs A"))
    sweep = []
    for i in range(0, 11):
        w = i / 10.0
        p = blend_price(R_ms, w, level_ship)
        out = price_arm(data, tvar, acc, fit, p, "greedy", Q_ship)
        sweep.append({"w": w, "median": out["median"]})
        print("   %6.1f %11.6f %+7.2f%%"
              % (w, out["median"], 100 * (out["median"] / E.BEST_A - 1)))
    print()

    print("  LEVEL SWEEP on the measured ranked SHAPE, greedy walk")
    print("   %6s %11s %8s" % ("level", "median", "vs A"))
    levels = []
    for i in range(6, 19):
        lv = i * 0.1
        p = table_price(R_ms, lv)
        out = price_arm(data, tvar, acc, fit, p, "greedy", Q_ship)
        levels.append({"level": lv, "median": out["median"]})
        print("   %6.2f %11.6f %+7.2f%%"
              % (lv, out["median"], 100 * (out["median"] / E.BEST_A - 1)))
    print()
    print("  LEVEL SWEEP on the UNIFORM shape, greedy walk (control axis:")
    print("  ranked receipts already bracket h = 0.18 on both sides)")
    print("   %6s %11s %8s" % ("h", "median", "vs A"))
    hsweep = []
    for i in range(8, 33, 2):
        h = i / 100.0
        p = {"marginal": [h] * MAXD,
             "cumulative": [1.0 + d * h for d in range(MAXD + 1)]}
        out = price_arm(data, tvar, acc, fit, p, "greedy", Q_ship)
        hsweep.append({"h": h, "median": out["median"]})
        print("   %6.2f %11.6f %+7.2f%%"
              % (h, out["median"], 100 * (out["median"] / E.BEST_A - 1)))

    print()
    print("=" * 78)
    print("STAGE 2  the same replay with q-CONDITIONAL acceptance")
    print("=" * 78)
    print("""  Stage 1 reuses the paid per-position rate S_k for every round the new
  rule sends to depth k. S_k was measured on the rounds the SHIPPED rule
  selected for that depth, so stage 1 over-values added depth and over-
  charges removed depth. Stage 2 gives every round its own acceptance
  q**gamma, with one gamma per prompt fitted so the shipped arm reproduces
  the paid mean accepted count exactly.""")
    print()
    cal = calibrate(data, tvar, Q_ship, ship)
    print("  %-9s %7s %9s %9s | %s" %
          ("prompt", "gamma", "abar fit", "abar paid", "S_k predicted vs paid"))
    for name in ORDER:
        c = cal[name]
        S = acc[name]["S"]
        cmp_s = " ".join("%.2f/%.2f" % (c["S_pred"][k - 1], S[k])
                         for k in range(1, CAP + 1) if tvar[name][k - 1] > 0.02)
        print("  %-9s %7.3f %9.4f %9.4f | %s"
              % (name, c["gamma"], c["abar_fit"],
                 data["A"]["rec"][name]["abar"], cmp_s))
    print()
    print("  gamma < 1 means the scheduler's own `reach` UNDER-estimates the")
    print("  acceptance it then gets, so a price below the measured marginal")
    print("  cost is the compensating error, not a mistake.")
    print()

    base2 = price_arm2(data, cal, fit, ship, "greedy")
    print("  MODEL CHECK: shipped arm reproduces receipt A: %.8f vs %.8f (%+0.3f%%)"
          % (base2["median"], E.BEST_A, 100 * (base2["median"] / E.BEST_A - 1)))
    for cap, paid, leg, lab in ((4, E.CAP4, "C", "cap4"), (5, E.CAP5, "E", "cap5")):
        out = price_arm2(data, cal, fit, ship, "greedy", cap, leg)
        print("  OUT OF SAMPLE: %s predicted %.6f, paid %.6f, %+0.2f%%"
              % (lab, out["median"], paid, 100 * (out["median"] / paid - 1)))
    print("  (receipt channel is %.3f%% 1-sigma, FINDING 460)"
          % (100 * E.SIGMA_PUBLISHED))
    print()
    print("  " + "-" * 74)
    print("  %-34s %11s %8s %8s" % ("arm", "median", "vs A", "vs crown"))
    print("  " + "-" * 74)
    results2 = {}
    for label, price, rule in arms:
        out = price_arm2(data, cal, fit, price, rule)
        results2[label] = out
        print("  %-34s %11.6f %+7.2f%% %+7.2f%%"
              % (label, out["median"], 100 * (out["median"] / E.BEST_A - 1),
                 100 * (out["median"] / E.CROWN - 1)))
    print("  " + "-" * 74)
    print()
    print("  STAGE 2 SHAPE SWEEP, level held at %.2f, greedy walk" % level_ship)
    print("   %6s %11s %8s" % ("w", "median", "vs A"))
    sweep2 = []
    for i in range(0, 11):
        w = i / 10.0
        out = price_arm2(data, cal, fit, blend_price(R_ms, w, level_ship),
                         "greedy")
        sweep2.append({"w": w, "median": out["median"]})
        print("   %6.1f %11.6f %+7.2f%%"
              % (w, out["median"], 100 * (out["median"] / E.BEST_A - 1)))
    print()
    print("  STAGE 2 LEVEL SWEEP on the UNIFORM shape (the axis three paid")
    print("  receipts already bracket: h 0.14 and 0.15 below, 0.32 above)")
    print("   %6s %11s %8s" % ("h", "median", "vs A"))
    hsweep2 = []
    for i in range(8, 33, 2):
        h = i / 100.0
        p = {"marginal": [h] * MAXD,
             "cumulative": [1.0 + d * h for d in range(MAXD + 1)]}
        out = price_arm2(data, cal, fit, p, "greedy")
        hsweep2.append({"h": h, "median": out["median"]})
        print("   %6.2f %11.6f %+7.2f%%"
              % (h, out["median"], 100 * (out["median"] / E.BEST_A - 1)))
    print()
    print("  STAGE 2 LEVEL SWEEP on the measured ranked SHAPE, greedy walk")
    print("   %6s %11s %8s" % ("level", "median", "vs A"))
    lsweep2 = []
    for i in range(4, 19, 1):
        lv = i * 0.1
        out = price_arm2(data, cal, fit, table_price(R_ms, lv), "greedy")
        lsweep2.append({"level": lv, "median": out["median"]})
        print("   %6.2f %11.6f %+7.2f%%"
              % (lv, out["median"], 100 * (out["median"] / E.BEST_A - 1)))

    ARTIFACTS.mkdir(exist_ok=True)
    payload = {
        "harness": "ranked",
        "cap": CAP,
        "law": {"family": "smooth-step", "R_ms": R_ms,
                "aicc": fit["aicc"], "wrmse_ms": fit["wrmse_ms"]},
        "Q_ship": Q_ship[1:],
        "arms": {label: {"median": results[label]["median"],
                         "marginal": price["marginal"],
                         "rule": rule,
                         "monotone": results[label]["monotone"],
                         "rows": results[label]["rows"]}
                 for label, price, rule in arms},
        "shape_sweep": sweep,
        "level_sweep": levels,
        "uniform_sweep": hsweep,
        "stage2": {
            "gamma": {n: cal[n]["gamma"] for n in ORDER},
            "S_pred": {n: cal[n]["S_pred"] for n in ORDER},
            "ship_median": base2["median"],
            "out_of_sample": {
                lab: price_arm2(data, cal, fit, ship, "greedy", cap,
                                leg)["median"]
                for cap, leg, lab in ((4, "C", "cap4"), (5, "E", "cap5"))},
            "arms": {label: {"median": results2[label]["median"],
                             "rows": results2[label]["rows"]}
                     for label, _, _ in arms},
            "shape_sweep": sweep2,
            "uniform_sweep": hsweep2,
            "level_sweep": lsweep2,
        },
        "constants": {"receiptA": E.BEST_A, "crown": E.CROWN,
                      "sigma_published": E.SIGMA_PUBLISHED},
    }
    (ARTIFACTS / "desk-price.json").write_text(json.dumps(payload, indent=1))
    print()
    print("  wrote %s" % (ARTIFACTS / "desk-price.json"))


if __name__ == "__main__":
    main()
