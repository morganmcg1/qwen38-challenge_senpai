#!/usr/bin/env python3
"""E211: the OPTIMAL step-aware marginal depth price, priced on the ranked
instrument.

    python3 research/e211_step_price.py [--json OUT]

harness=ranked for every number this file prints. No GPU, no timed leg, no
local millisecond is carried into a ranked number.

THE QUESTION. The shipped scheduler prices every marginal draft row at the
same `headStepCostRatio` (`makeUniformDepthPrice`, h = 0.18). The ranked cost
law is not uniform: the 6-row cell carries a measured step (FINDING 505) and
the 9-row cell may carry a weight-group step. E200 swept the SHAPE of the price
at a held level and found +0.17 % (FINDING 519). E209 closed depth LEVEL
controllers (FINDING 541). Neither closed the free optimum over the price
table itself. This file computes it.

WHY THE FREE OPTIMUM IS COMPUTABLE EXACTLY. `costModelDepth` is a one-step hill
climb: it takes the step into depth d + 1 iff

    q**(d+1) > marginal[d] * (1 + sum_{k=1..d} q**k) / cumulative[d]

The left side is 0 at q = 0 and the ratio on the right is strictly increasing
in q, so each row d has a single crossing q_d, and the walk stops at the first
row whose crossing is not cleared. A round therefore reaches depth >= k iff
q > Q_k with Q_k = max(q_0 .. q_{k-1}), which is monotone in k by construction.
Conversely any monotone threshold vector Q is realised by the recursion

    marginal[d] = Q_{d+1}**(d+1) * cumulative[d] / (1 + sum_{k=1..d} Q_{d+1}**k)

so the reachable set of the price-table family is EXACTLY the set of monotone
depth maps of the acceptance signal. Optimising the price table is optimising
the eight cut points of that map, which is a small exact search rather than a
heuristic sweep over price shapes. The recovered table is checked against
`e200_desk_price.greedy_depth` on every grid point, so the reported optimum is
a real price table and not only an abstract policy.

INPUT LEGALITY. The policy reads the marginal row index d and the scheduler's
own acceptance signal, and nothing else. It carries no prompt feature, no
benchmark-phase detection and no cross-request state, so a later implementation
stays inside the declared editable surface.

THE INSTRUMENT. FINDING 520 survival-pinned latent-q desk replay, imported from
`e201_online_cap` without modification, exactly as E203, E207 and E209 use it.
The reproduction gate below rebuilds the shipped cap-4..cap-8 arms through the
grid form of this file and requires bit-level agreement with `O.cap_grid`
before any optimum is reported.

THREE COST LAWS. The ranked 9-row cell is unobserved by every paid receipt at
cap <= 7, so the optimum is priced under all three live readings:

  smooth      the fit's own smooth continuation at 9 rows.
  step        E186's local 9-row weight-group step transferred at the FINDING
              505 6-row ratio; this is `e201_online_cap.cost_laws`'s `step9`.
  step_e208   the same step with E208's measured (9,5) staged-QMV saving
              removed from the LOCAL step before the SAME transfer. INFERRED:
              the E208 delta is a local M4 Pro paired measurement and only the
              6-row transfer ratio exists to move it onto ranked hardware.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e197_refit as E  # noqa: E402
import e200_desk_price as D  # noqa: E402
import e201_online_cap as O  # noqa: E402
import e203_stage0b as S  # noqa: E402

ORDER = E.ORDER
TOKENS = E.TOKENS
MAXD = 8                       # Qwen36MTPLimits.maxDepth
QGRID = D.QGRID                # latent-q quadrature resolution of the instrument
ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e211-artifacts"

E208_M9_LOCAL_MS = 20.774      # FINDING 543 paired m=9 delta, harness=local
E208_M9_LOCAL_CI = (20.396, 21.152)

LAW_KEYS = ("smooth", "step", "step_e208")
TIEBREAK = 1e-7                # breaks exact median ties toward the better mean


# ------------------------------------------------------------------- laws

def build_laws(fit, e208_local_ms=E208_M9_LOCAL_MS):
    """R(m) for m = 1..9 under the three live readings of the 9-row cell."""
    ranked, ratio = O.cost_laws(fit)
    head = [1000.0 * E.R_of(fit, m) for m in range(1, MAXD + 1)]
    residual_local = max(E.LOCAL_DR9 - e208_local_ms, 0.0)
    corrected = E.table_law(head + [head[-1] + ratio * residual_local])
    return ({"smooth": ranked["smooth"], "step": ranked["step9"],
             "step_e208": corrected},
            {"transfer_ratio": ratio, "local_dr9_ms": E.LOCAL_DR9,
             "e208_local_ms": e208_local_ms,
             "residual_local_dr9_ms": residual_local,
             "ranked_dr9_ms": {
                 "smooth": 1000.0 * (E.R_of(ranked["smooth"], 9)
                                     - E.R_of(ranked["smooth"], 8)),
                 "step": 1000.0 * (E.R_of(ranked["step9"], 9)
                                   - E.R_of(ranked["step9"], 8)),
                 "step_e208": 1000.0 * (E.R_of(corrected, 9)
                                        - E.R_of(corrected, 8))}})


# ------------------------------------------------------------- grid tables

def grid_q():
    """The instrument's own quadrature abscissae, reproduced bit for bit.

    `e200_desk_price.quadrature` builds each node as the midpoint of its bin,
    so the same expression is used here. Regenerating the node as
    `(i + 0.5) / QGRID` differs in the last unit in the last place and would
    put the reproduction gate below its own resolution.
    """
    i = np.arange(QGRID)
    return 0.5 * (i / QGRID + (i + 1) / QGRID)


def prompt_table(inst, law, name):
    """Prefix sums that price any monotone depth map in O(MAXD) per prompt."""
    c = inst["cal"][name]
    idx = np.rint(np.asarray(c["q"]) * QGRID - 0.5).astype(int)
    w = np.zeros(QGRID)
    w[idx] = np.asarray(c["w"])
    q = grid_q()
    p = q ** c["gamma"]

    accepted = np.zeros((MAXD + 1, QGRID))
    run = np.zeros(QGRID)
    for d in range(1, MAXD + 1):
        run = run + p ** d
        accepted[d] = run
    cost = np.array([E.R_of(law, d + 1) for d in range(MAXD + 1)])

    zero = np.zeros((MAXD + 1, 1))
    rec = inst["data"]["A"]["rec"][name]
    vec = inst["data"]["A"]["vec"][name]
    return {
        "PA": np.concatenate([zero, np.cumsum(w * accepted, axis=1)], axis=1),
        "PC": np.concatenate([zero, np.cumsum(w[None, :] * cost[:, None],
                                              axis=1)], axis=1),
        "PW": np.concatenate([[0.0], np.cumsum(w)]),
        "rec_R": rec["R"],
        "ship_R": sum(wi * E.R_of(law, di + 1)
                      for wi, di in zip(c["w"], c["depth_ship"])),
        "serial": vec["serial"], "prefill": vec["prefill"],
    }


def prompt_tables(inst, law):
    return {name: prompt_table(inst, law, name) for name in ORDER}


def evaluate(tables, cuts):
    """Per-prompt ranked raw speedup of one monotone depth map."""
    out = {}
    for name, T in tables.items():
        abar = cost = 0.0
        for d in range(MAXD + 1):
            abar += T["PA"][d][cuts[d + 1]] - T["PA"][d][cuts[d]]
            cost += T["PC"][d][cuts[d + 1]] - T["PC"][d][cuts[d]]
        R = T["rec_R"] + cost - T["ship_R"]
        out[name] = T["serial"] / (R / (1.0 + abar) + T["prefill"])
    return out


def edl_of(tables, cuts):
    """Expected drafted rows per round, per prompt."""
    return {name: sum(d * (T["PW"][cuts[d + 1]] - T["PW"][cuts[d]])
                      for d in range(MAXD + 1))
            for name, T in tables.items()}


def depth_mass(tables, cuts):
    """P(depth = d) per prompt, the shape the receipt anchor is walked from."""
    return {name: [T["PW"][cuts[d + 1]] - T["PW"][cuts[d]]
                   for d in range(MAXD + 1)]
            for name, T in tables.items()}


def median_of(values):
    """Published median rule, extended to the odd LOO pools."""
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


# --------------------------------------------------------------- policies

def ship_cuts(cap):
    """Cut points of `makeUniformDepthPrice` at one static cap."""
    depth = np.array([D.greedy_depth(D.ship_price(), qi, cap)
                      for qi in grid_q()])
    if np.any(np.diff(depth) < 0):
        raise AssertionError("shipped depth map is not monotone in q")
    cuts = [0]
    for k in range(1, MAXD + 1):
        hit = np.flatnonzero(depth >= k)
        cuts.append(int(hit[0]) if hit.size else QGRID)
    return cuts_of(cuts[1:])


def cuts_of(inner):
    return [0] + [int(v) for v in inner] + [QGRID]


def const_cuts(depth):
    return cuts_of([0] * depth + [QGRID] * (MAXD - depth))


def price_of_cuts(cuts):
    """Recover the marginal price table that realises a monotone depth map.

    Units are the shipped ones: the width-1 verify forward, so 0.18 is the
    shipped `headStepCostRatio` on every row.
    """
    marginal, cumulative = [], [1.0]
    for d in range(MAXD):
        Q = cuts[d + 1] / QGRID
        reach = sum(Q ** k for k in range(1, d + 1))
        marginal.append(Q ** (d + 1) * cumulative[d] / (1.0 + reach))
        cumulative.append(cumulative[d] + marginal[d])
    return {"marginal": marginal, "cumulative": cumulative}


def verify_price(cuts, price):
    """Agreement between the recovered table and the map it must reproduce."""
    want = np.searchsorted(np.array(cuts[1:MAXD + 1]),
                           np.arange(QGRID), side="right")
    got = np.array([D.greedy_depth(price, qi, MAXD) for qi in grid_q()])
    return float(np.mean(got == want)), int(np.sum(got != want))


# ------------------------------------------------------------- optimiser

def objective(tables, names, cuts):
    raws = evaluate(tables, cuts)
    picked = [raws[n] for n in names]
    return median_of(picked) + TIEBREAK * (sum(picked) / len(picked))


def _sweep_axis(tables, names, cuts, k):
    """Best cut for row k with every other cut held, evaluated in one shot."""
    lo, hi = cuts[k - 1], cuts[k + 1]
    if hi <= lo:
        return cuts[k], None
    ts = np.arange(lo, hi + 1)
    rows = np.empty((len(names), ts.size))
    for j, name in enumerate(names):
        T = tables[name]
        abar = cost = 0.0
        for d in range(MAXD + 1):
            if d in (k - 1, k):
                continue
            abar += T["PA"][d][cuts[d + 1]] - T["PA"][d][cuts[d]]
            cost += T["PC"][d][cuts[d + 1]] - T["PC"][d][cuts[d]]
        abar_t = (abar + (T["PA"][k - 1][ts] - T["PA"][k - 1][lo])
                  + (T["PA"][k][hi] - T["PA"][k][ts]))
        cost_t = (cost + (T["PC"][k - 1][ts] - T["PC"][k - 1][lo])
                  + (T["PC"][k][hi] - T["PC"][k][ts]))
        R = T["rec_R"] + cost_t - T["ship_R"]
        rows[j] = T["serial"] / (R / (1.0 + abar_t) + T["prefill"])
    ordered = np.sort(rows, axis=0)
    n = len(names)
    med = (ordered[n // 2] if n % 2
           else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2]))
    score = med + TIEBREAK * rows.mean(axis=0)
    best = int(np.argmax(score))
    return int(ts[best]), float(score[best])


def ascend(tables, names, start, passes=60):
    cuts = list(start)
    best = objective(tables, names, cuts)
    for _ in range(passes):
        moved = False
        for k in range(1, MAXD + 1):
            pos, score = _sweep_axis(tables, names, cuts, k)
            if score is not None and score > best + 1e-15:
                cuts[k], best, moved = pos, score, True
        if not moved:
            break
    return cuts, best


def starts(rng):
    seeds = [ship_cuts(8), ship_cuts(7), ship_cuts(6), ship_cuts(4)]
    seeds += [const_cuts(d) for d in range(MAXD + 1)]
    for _ in range(24):
        seeds.append(cuts_of(sorted(rng.integers(0, QGRID + 1, MAXD))))
    return seeds


def optimise(tables, names, rng):
    best_cuts, best_score = None, -1e18
    for seed in starts(rng):
        cuts, score = ascend(tables, names, seed)
        if score > best_score:
            best_cuts, best_score = cuts, score
    return best_cuts


# ------------------------------------------------------------------- gate

def reproduction_gate(inst, laws):
    """Rebuild the shipped static-cap arms through this file's grid form.

    `O.cap_grid` walks the prompt's own quadrature nodes; this file walks the
    full grid with zero-weight bins filled in. They must agree exactly, or the
    grid form is not the merged instrument and no optimum may be reported.
    """
    grid = O.cap_grid(inst, {"smooth": laws["smooth"], "step9": laws["step"]})
    worst, rows = 0.0, {}
    for key, gkey in (("smooth", "smooth"), ("step", "step9")):
        tables = prompt_tables(inst, laws[key])
        for cap in O.CAPS:
            got = evaluate(tables, ship_cuts(cap))
            for name in ORDER:
                want = grid[gkey][name][cap]["raw"]
                err = abs(got[name] - want) / want
                worst = max(worst, err)
            rows["%s/cap%d" % (key, cap)] = median_of(
                [got[n] for n in ORDER])
    return {"max_relative_error": worst, "pass": worst < 1e-12,
            "medians": rows}


def instrument_validation(inst, law):
    """FINDING 520 gate, restated on this file's own arms.

    The cap-4 and cap-5 receipts were paid on their own legs, so each is
    scored against its own serial and prefill vector rather than against
    receipt A's.
    """
    tables = prompt_tables(inst, law)

    def median_on_leg(cap, leg_key):
        out = []
        for name in ORDER:
            T = tables[name]
            cuts = ship_cuts(cap)
            abar = cost = 0.0
            for d in range(MAXD + 1):
                abar += T["PA"][d][cuts[d + 1]] - T["PA"][d][cuts[d]]
                cost += T["PC"][d][cuts[d + 1]] - T["PC"][d][cuts[d]]
            R = T["rec_R"] + cost - T["ship_R"]
            out.append(E.raw_of(inst["data"], name,
                                TOKENS / (1.0 + abar) * R, leg_key))
        return median_of(out)

    m7 = median_of([evaluate(tables, ship_cuts(7))[n] for n in ORDER])
    m4 = median_on_leg(4, "C")
    m5 = median_on_leg(5, "E")
    err = {"cap7_pct": 100.0 * (m7 - E.BEST_A) / E.BEST_A,
           "cap4_pct": 100.0 * (m4 - E.CAP4) / E.CAP4,
           "cap5_pct": 100.0 * (m5 - E.CAP5) / E.CAP5}
    channel = 100.0 * E.SIGMA_PUBLISHED
    return {"cap7": m7, "cap4": m4, "cap5": m5, "paid_cap7": E.BEST_A,
            "paid_cap4": E.CAP4, "paid_cap5": E.CAP5, "error_pct": err,
            "receipt_channel_pct": channel,
            "pass": (abs(err["cap7_pct"]) < 1e-9
                     and abs(err["cap4_pct"]) < channel
                     and abs(err["cap5_pct"]) < channel)}


# ------------------------------------------------------------- experiment

def oracle_ceiling(inst, law):
    """Per-prompt round-level oracle: the strict ceiling on this family."""
    return median_of([S.optimise(inst, law, name, S.items_oracle_q)["raw"]
                      for name in ORDER])


def run_law(inst, law, key, rng):
    tables = prompt_tables(inst, law)
    ship = ship_cuts(8)
    ship_raw = evaluate(tables, ship)
    ship_median = median_of([ship_raw[n] for n in ORDER])

    best = optimise(tables, ORDER, rng)
    best_raw = evaluate(tables, best)
    best_median = median_of([best_raw[n] for n in ORDER])

    price = price_of_cuts(best)
    agree, mismatches = verify_price(best, price)

    # LOO robustness: the fitted table, scored on each 7-prompt subpool.
    loo_robust = {}
    for held in ORDER:
        pool = [n for n in ORDER if n != held]
        s = median_of([ship_raw[n] for n in pool])
        b = median_of([best_raw[n] for n in pool])
        loo_robust[held] = {"shipped": s, "optimum": b,
                            "delta_pct": 100.0 * (b - s) / s}

    # LOO-honest: the held-out prompt is scored by a table it never saw.
    honest_raw, honest_tables = {}, {}
    for held in ORDER:
        pool = [n for n in ORDER if n != held]
        fold = optimise(tables, pool, rng)
        honest_tables[held] = {"cuts": fold[1:MAXD + 1],
                               "price": price_of_cuts(fold)["marginal"]}
        honest_raw[held] = evaluate(tables, fold)[held]
    honest_median = median_of([honest_raw[n] for n in ORDER])

    return {
        "law": key,
        "cost_table_ms": [1000.0 * E.R_of(law, m) for m in range(1, MAXD + 2)],
        "shipped": {"cuts": ship[1:MAXD + 1],
                    "price_marginal": D.ship_price()["marginal"],
                    "per_prompt_raw": ship_raw,
                    "published_median": ship_median,
                    "edl": edl_of(tables, ship),
                    "depth_mass": depth_mass(tables, ship)},
        "optimum": {"cuts": best[1:MAXD + 1],
                    "thresholds": [c / QGRID for c in best[1:MAXD + 1]],
                    "price_marginal": price["marginal"],
                    "price_cumulative": price["cumulative"],
                    "per_prompt_raw": best_raw,
                    "published_median": best_median,
                    "edl": edl_of(tables, best),
                    "depth_mass": depth_mass(tables, best),
                    "greedy_agreement": agree,
                    "greedy_mismatches": mismatches},
        "in_sample_delta_pct":
            100.0 * (best_median - ship_median) / ship_median,
        "loo_robust": loo_robust,
        "loo_worst_delta_pct": min(v["delta_pct"]
                                   for v in loo_robust.values()),
        "loo_worst_prompt": min(loo_robust, key=lambda n:
                                loo_robust[n]["delta_pct"]),
        "loo_honest": {"per_prompt_raw": honest_raw,
                       "published_median": honest_median,
                       "fold_tables": honest_tables},
        "loo_honest_delta_pct":
            100.0 * (honest_median - ship_median) / ship_median,
        "oracle_q_median": oracle_ceiling(inst, law),
    }


# ---------------------------------------------------------------- report

def pct(x, base):
    return 100.0 * (x - base) / base


def report(out):
    L = ["=" * 78,
         "E211  OPTIMAL STEP-AWARE MARGINAL DEPTH PRICE   (harness=ranked)",
         "=" * 78,
         "  instrument: FINDING 520 survival-pinned latent-q, imported from",
         "              e201_online_cap without modification",
         "  receipt A %.8f (cap 7)   crown %.8f" % (E.BEST_A, E.CROWN),
         "  MUE = 0.39 %% of the published median = %.4f score points"
         % (0.0039 * E.BEST_A), ""]

    g = out["reproduction_gate"]
    L.append("  REPRODUCTION GATE (grid form vs merged O.cap_grid)")
    L.append("   max relative error %.3e   %s"
             % (g["max_relative_error"], "PASS" if g["pass"] else "FAIL"))
    v = out["instrument_validation"]
    L.append("  INSTRUMENT VALIDATION GATE (FINDING 520 bar)")
    L.append("   cap 7 in-sample     %.8f vs paid %.8f  (%+0.6f%%)"
             % (v["cap7"], v["paid_cap7"], v["error_pct"]["cap7_pct"]))
    L.append("   cap 4 out-of-sample %.6f vs paid %.6f  (%+0.2f%%)"
             % (v["cap4"], v["paid_cap4"], v["error_pct"]["cap4_pct"]))
    L.append("   cap 5 out-of-sample %.6f vs paid %.6f  (%+0.2f%%)"
             % (v["cap5"], v["paid_cap5"], v["error_pct"]["cap5_pct"]))
    L.append("   receipt channel 1-sigma %.3f%%   %s"
             % (v["receipt_channel_pct"], "PASS" if v["pass"] else "FAIL"))
    L.append("")

    t = out["nine_row_cell"]
    L.append("  9-ROW CELL, THE THREE LIVE READINGS")
    L.append("   transfer ratio (FINDING 505 over E186) %.5f" %
             t["transfer_ratio"])
    L.append("   local dR9 %.3f ms; E208 (9,5) removes %.3f ms; residual "
             "%.3f ms [INFERRED transfer]"
             % (t["local_dr9_ms"], t["e208_local_ms"],
                t["residual_local_dr9_ms"]))
    for key in LAW_KEYS:
        L.append("   %-10s ranked dR9 %+7.3f ms" % (key,
                                                    t["ranked_dr9_ms"][key]))
    L.append("")

    for key in LAW_KEYS:
        r = out["laws"][key]
        ship, opt = r["shipped"], r["optimum"]
        L.append("-" * 78)
        L.append("  LAW %s" % key.upper())
        L.append("   R(1..9) ms  %s"
                 % "  ".join("%.2f" % x for x in r["cost_table_ms"]))
        L.append("   shipped uniform price h=0.18, cap 8   %.6f"
                 % ship["published_median"])
        L.append("   OPTIMAL price table                   %.6f  (%+.3f%% "
                 "in-sample)"
                 % (opt["published_median"], r["in_sample_delta_pct"]))
        L.append("   LOO-honest median (decision statistic) %.6f  (%+.3f%%)"
                 % (r["loo_honest"]["published_median"],
                    r["loo_honest_delta_pct"]))
        L.append("   round-level oracle_q ceiling           %.6f  (%+.3f%%)"
                 % (r["oracle_q_median"],
                    pct(r["oracle_q_median"], ship["published_median"])))
        L.append("   optimal marginal price by row d (shipped = 0.18 on every"
                 " row)")
        L.append("     d      %s"
                 % "  ".join("%6d" % d for d in range(MAXD)))
        L.append("     price  %s"
                 % "  ".join("%6.3f" % x for x in opt["price_marginal"]))
        L.append("     Q_(d+1)%s"
                 % "  ".join("%6.4f" % x for x in opt["thresholds"]))
        L.append("     greedy-table agreement %.4f (%d mismatching grid "
                 "points)" % (opt["greedy_agreement"],
                              opt["greedy_mismatches"]))
        L.append("   per-prompt raw: shipped -> optimum (edl shipped -> "
                 "optimum)")
        for name in ORDER:
            L.append("     %-9s %.4f -> %.4f (%+6.2f%%)   edl %.2f -> %.2f"
                     % (name, ship["per_prompt_raw"][name],
                        opt["per_prompt_raw"][name],
                        pct(opt["per_prompt_raw"][name],
                            ship["per_prompt_raw"][name]),
                        ship["edl"][name], opt["edl"][name]))
        L.append("   LOO robustness of the fitted table (drop one prompt)")
        for name in ORDER:
            row = r["loo_robust"][name]
            L.append("     drop %-9s shipped %.6f  optimum %.6f  %+.3f%%"
                     % (name, row["shipped"], row["optimum"],
                        row["delta_pct"]))
        L.append("   worst LOO prompt %s at %+.3f%%"
                 % (r["loo_worst_prompt"], r["loo_worst_delta_pct"]))
        L.append("   LOO-honest held-out raws")
        for name in ORDER:
            L.append("     %-9s honest %.4f  shipped %.4f  (%+6.2f%%)"
                     % (name, r["loo_honest"]["per_prompt_raw"][name],
                        ship["per_prompt_raw"][name],
                        pct(r["loo_honest"]["per_prompt_raw"][name],
                            ship["per_prompt_raw"][name])))
        L.append("")

    d = out["decision"]
    L.append("=" * 78)
    L.append("  DECISION (LOO-honest delta is the statistic)")
    for key in LAW_KEYS:
        L.append("   %-10s in-sample %+7.3f%%   LOO-honest %+7.3f%%   %s"
                 % (key, out["laws"][key]["in_sample_delta_pct"],
                    out["laws"][key]["loo_honest_delta_pct"],
                    d["per_law_verdict"][key]))
    L.append("   %s" % d["verdict"])
    return "\n".join(L)


def decide(laws_out):
    verdicts = {}
    for key in LAW_KEYS:
        x = laws_out[key]["loo_honest_delta_pct"]
        if x >= 1.0:
            verdicts[key] = "IMPLEMENT (>=1%)"
        elif x >= 0.5:
            verdicts[key] = "ADVISOR CALL (0.5-1%)"
        else:
            verdicts[key] = "CLOSE (<0.5%)"
    best = max(LAW_KEYS, key=lambda k: laws_out[k]["loo_honest_delta_pct"])
    worst_is_close = all(laws_out[k]["loo_honest_delta_pct"] < 0.5
                         for k in LAW_KEYS)
    if worst_is_close:
        verdict = ("STOP RULE FIRED: LOO-honest improvement < 0.5 % under "
                   "EVERY law. The depth-schedule family closes.")
    else:
        verdict = ("Best law %s at %+.3f%% LOO-honest; the receipt selects "
                   "the branch." % (best,
                                    laws_out[best]["loo_honest_delta_pct"]))
    return {"per_law_verdict": verdicts, "best_law": best, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ARTIFACTS / "step-price.json"))
    ap.add_argument("--seed", type=int, default=211)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    inst = O.build_instrument()
    laws, cell = build_laws(inst["fit"])

    out = {"harness": "ranked", "seed": args.seed,
           "receipt_A": E.BEST_A, "crown": E.CROWN,
           "reproduction_gate": reproduction_gate(inst, laws),
           "instrument_validation": instrument_validation(inst,
                                                          laws["smooth"]),
           "nine_row_cell": cell, "laws": {}}
    if not out["reproduction_gate"]["pass"]:
        raise SystemExit("reproduction gate FAILED; refusing to report an "
                         "optimum")
    for key in LAW_KEYS:
        out["laws"][key] = run_law(inst, laws[key], key, rng)
    out["decision"] = decide(out["laws"])

    text = report(out)
    print(text)
    ARTIFACTS.mkdir(exist_ok=True)
    path = pathlib.Path(args.json)
    path.write_text(json.dumps(out, indent=1, sort_keys=True,
                               default=float) + "\n")
    path.with_suffix(".txt").write_text(text + "\n")
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
