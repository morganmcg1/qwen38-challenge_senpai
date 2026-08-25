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


def uniform_price(h):
    return {"marginal": [h] * MAXD,
            "cumulative": [1.0 + d * h for d in range(MAXD + 1)]}


def _walk_terms():
    """`reach` and `1 + expected` at every row, for the whole q grid."""
    q = grid_q()
    powers = np.array([q ** k for k in range(1, MAXD + 1)])
    before = np.vstack([np.zeros((1, QGRID)), np.cumsum(powers, axis=0)[:-1]])
    return powers, 1.0 + before


_REACH, _BAR = _walk_terms()


def price_cuts(marginal):
    """Cut points of `costModelDepth` under an explicit marginal price table.

    The whole grid is walked at once. `reach` at row d is q**(d+1) and the
    walk's `expected` at that moment is sum_{k=1..d} q**k, so the shipped
    comparison becomes a matrix and the depth is the first row whose
    comparison fails.
    """
    cumulative, ok = 1.0, np.empty((MAXD, QGRID), dtype=bool)
    for d in range(MAXD):
        ok[d] = _REACH[d] > marginal[d] * _BAR[d] / cumulative
        cumulative += marginal[d]
    depth = np.where(ok.all(axis=0), MAXD, np.argmin(ok, axis=0))
    cuts = [0]
    for k in range(1, MAXD + 1):
        hit = np.flatnonzero(depth >= k)
        cuts.append(int(hit[0]) if hit.size else QGRID)
    return cuts_of(cuts[1:])


def level_of(marginal):
    """E200's price level: total marginal price over the reachable rows."""
    return float(sum(marginal))


# ------------------------------------------------------------- optimiser

def objective(tables, names, cuts):
    raws = evaluate(tables, cuts)
    picked = [raws[n] for n in names]
    return median_of(picked) + TIEBREAK * (sum(picked) / len(picked))


def _median_curve(tables, names, cuts, k, ts):
    """Published median and mean over every candidate cut for row k.

    Only the two segments that touch row k change, so the other segments are
    summed once and the sweep is a pair of prefix-sum lookups per prompt.
    """
    lo, hi = cuts[k - 1], cuts[k + 1]
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
    return med, rows.mean(axis=0), rows


def _sweep_axis(tables, names, cuts, k):
    """Best cut for row k with every other cut held, evaluated in one shot."""
    lo, hi = cuts[k - 1], cuts[k + 1]
    if hi <= lo:
        return cuts[k], None
    ts = np.arange(lo, hi + 1)
    med, mean, _rows = _median_curve(tables, names, cuts, k, ts)
    score = med + TIEBREAK * mean
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


# ------------------------------------------------- receipt-proof (minimax)

GUARD = 100.0     # weight on a per-prompt regression, relative to the delta


def minimax_objective(law_tables, ship_medians, names, cuts, floors=None):
    rel, penalty = [], 0.0
    for key, tables in law_tables.items():
        raws = evaluate(tables, cuts)
        med = median_of([raws[n] for n in names])
        rel.append((med - ship_medians[key]) / ship_medians[key])
        if floors is not None:
            for n in names:
                penalty += max(0.0, floors[key][n] - raws[n]) / floors[key][n]
    return (min(rel) + TIEBREAK * (sum(rel) / len(rel))
            - GUARD * penalty / len(law_tables))


def ascend_minimax(law_tables, ship_medians, names, start, floors=None,
                   passes=60):
    """Maximise the WORST per-law delta, optionally with no prompt regressing.

    The `aff4ad64` receipt names the true 9-row cell only after a table has to
    be chosen, so the table that may be implemented is the one whose weakest
    reading still pays, not the one that wins under a guessed reading. The
    optional floor forbids paying for the published median with a loss on any
    single prompt, because the hidden pool is not the proxy pool and a prompt
    that does not set the proxy median may set the hidden one.
    """
    cuts = list(start)
    best = minimax_objective(law_tables, ship_medians, names, cuts, floors)
    for _ in range(passes):
        moved = False
        for k in range(1, MAXD + 1):
            lo, hi = cuts[k - 1], cuts[k + 1]
            if hi <= lo:
                continue
            ts = np.arange(lo, hi + 1)
            worst = total = None
            penalty = np.zeros(ts.size)
            for key, tables in law_tables.items():
                med, _mean, rows = _median_curve(tables, names, cuts, k, ts)
                rel = (med - ship_medians[key]) / ship_medians[key]
                worst = rel if worst is None else np.minimum(worst, rel)
                total = rel if total is None else total + rel
                if floors is not None:
                    floor = np.array([floors[key][n] for n in names])[:, None]
                    penalty += (np.maximum(0.0, floor - rows)
                                / floor).sum(axis=0)
            score = (worst + TIEBREAK * total / len(law_tables)
                     - GUARD * penalty / len(law_tables))
            pick = int(np.argmax(score))
            if float(score[pick]) > best + 1e-15:
                cuts[k], best, moved = int(ts[pick]), float(score[pick]), True
        if not moved:
            break
    return cuts, best


def optimise_minimax(law_tables, ship_medians, names, rng, floors=None):
    best_cuts, best_score = None, -1e18
    for seed in starts(rng):
        cuts, score = ascend_minimax(law_tables, ship_medians, names, seed,
                                     floors)
        if score > best_score:
            best_cuts, best_score = cuts, score
    return best_cuts


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


# ------------------------------------------- level / structure decomposition

def _uniform_ladder(grid=6001):
    """Every distinct depth map the uniform price can produce.

    `costModelDepth` reads a step function of h, so the sweep collapses to the
    handful of distinct cut vectors on the grid and the LOO folds reuse them.
    """
    seen, ladder = set(), []
    for i in range(1, grid):
        h = 0.60 * i / grid
        cuts = price_cuts([h] * MAXD)
        key = tuple(cuts)
        if key not in seen:
            seen.add(key)
            ladder.append((h, cuts))
    return ladder


_UNIFORM = _uniform_ladder()


def optimise_uniform(tables, names):
    """Best UNIFORM price: the shipped shape with only its level free.

    This arm is the control the headline number has to be read against. It
    changes how deep the scheduler drafts and nothing about how the rows are
    priced relative to each other, so it carries no step awareness at all.
    """
    best = max(_UNIFORM, key=lambda hc: objective(tables, names, hc[1]))
    return {"h": best[0], "cuts": best[1], "marginal": [best[0]] * MAXD}


def measured_shape(law):
    """The honest ranked marginal cost of each row, in shipped price units."""
    R = [E.R_of(law, m) for m in range(1, MAXD + 2)]
    return [(R[d + 1] - R[d]) / R[0] for d in range(MAXD)]


def optimise_shape(tables, names, rng, level, law, rounds=12):
    """Best price SHAPE at a held level: E200's axis with a free shape.

    Coordinate descent in price space with a rescale to the held level after
    every move, restarted from the uniform shape, from the measured ranked
    cost shape and from random shapes. The level constraint is what separates
    a step-aware price from a cheaper price.
    """
    def rescale(vec):
        vec = [max(v, 1e-9) for v in vec]
        s = level / sum(vec)
        return [v * s for v in vec]

    def score_of(vec):
        return objective(tables, names, price_cuts(vec))

    seeds = [[level / MAXD] * MAXD, rescale(measured_shape(law))]
    for _ in range(16):
        seeds.append(rescale(list(rng.random(MAXD) + 0.05)))

    best_vec, best_score = None, -1e18
    for seed in seeds:
        vec, score = rescale(seed), None
        score = score_of(vec)
        for _ in range(rounds):
            moved = False
            for d in range(MAXD):
                for factor in (1.6, 1.25, 1.08, 1.02, 0.98, 0.93, 0.8, 0.625):
                    trial = list(vec)
                    trial[d] *= factor
                    trial = rescale(trial)
                    value = score_of(trial)
                    if value > score + 1e-15:
                        vec, score, moved = trial, value, True
            if not moved:
                break
        if score > best_score:
            best_vec, best_score = vec, score
    return {"marginal": best_vec, "cuts": price_cuts(best_vec)}


def arm(tables, names, marginal, cuts):
    raws = evaluate(tables, cuts)
    return {"marginal": list(marginal), "level": level_of(marginal),
            "cuts": cuts[1:MAXD + 1], "per_prompt_raw": raws,
            "published_median": median_of([raws[n] for n in names]),
            "edl": edl_of(tables, cuts)}


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


def anchor_distance(tables, ship, cuts):
    """How far the arm moves the round population from the paid anchor.

    The instrument walks receipt A's measured round cost to a new schedule, so
    the arm's credibility falls with the round mass it relocates and with the
    depth it reaches beyond the deepest paid cell. Both are reported rather
    than folded into the score.
    """
    ship_mass, arm_mass = depth_mass(tables, ship), depth_mass(tables, cuts)
    moved, deep = {}, {}
    for name in tables:
        moved[name] = 0.5 * sum(abs(a - b) for a, b in zip(arm_mass[name],
                                                           ship_mass[name]))
        deep[name] = arm_mass[name][MAXD]
    return {"round_mass_relocated": moved, "mass_at_depth_8": deep,
            "worst_round_mass_relocated": max(moved.values())}


def validated_envelope(inst, law, arms):
    """How far outside the PAID extrapolation range each arm sits.

    Receipt A paid cap 7. Caps 4 and 5 are the only other schedules the
    instrument was validated against a paid receipt on, so the round mass they
    relocate away from cap 7 is the largest relocation the instrument is known
    to survive. An arm that relocates far more mass than that is extrapolating
    beyond the validated envelope, whatever its fitted median says.
    """
    tables = prompt_tables(inst, law)
    anchor = ship_cuts(7)
    ref = {"cap4": ship_cuts(4), "cap5": ship_cuts(5), "cap8": ship_cuts(8)}
    validated = {k: anchor_distance(tables, anchor, c)
                 ["worst_round_mass_relocated"] for k, c in ref.items()}
    bar = max(validated["cap4"], validated["cap5"])
    out = {"validated_relocation": validated, "validated_bar": bar, "arms": {}}
    for name, cuts in arms.items():
        d = anchor_distance(tables, anchor, cuts)["worst_round_mass_relocated"]
        out["arms"][name] = {"relocation_vs_cap7": d,
                             "envelope_ratio": d / bar,
                             "inside_validated_envelope": bool(d <= bar)}
    return out


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

    # The control the headline number must be read against: the shipped shape
    # with only its level free carries no step awareness whatsoever.
    level_arm = optimise_uniform(tables, ORDER)
    level_only = arm(tables, ORDER, level_arm["marginal"], level_arm["cuts"])
    shape_ship = optimise_shape(tables, ORDER, rng,
                                level_of(D.ship_price()["marginal"]), law)
    shape_at_ship_level = arm(tables, ORDER, shape_ship["marginal"],
                              shape_ship["cuts"])
    shape_best = optimise_shape(tables, ORDER, rng, level_arm["h"] * MAXD, law)
    shape_at_best_level = arm(tables, ORDER, shape_best["marginal"],
                              shape_best["cuts"])

    # LOO robustness: the fitted table, scored on each 7-prompt subpool.
    loo_robust = {}
    for held in ORDER:
        pool = [n for n in ORDER if n != held]
        s = median_of([ship_raw[n] for n in pool])
        b = median_of([best_raw[n] for n in pool])
        loo_robust[held] = {"shipped": s, "optimum": b,
                            "delta_pct": 100.0 * (b - s) / s}

    # LOO-honest: the held-out prompt is scored by a table it never saw. The
    # one-parameter level control is refitted on the same folds, so the
    # structure premium compares two honestly fitted arms.
    honest_raw, honest_tables, honest_level_raw = {}, {}, {}
    for held in ORDER:
        pool = [n for n in ORDER if n != held]
        fold = optimise(tables, pool, rng)
        honest_tables[held] = {"cuts": fold[1:MAXD + 1],
                               "price": price_of_cuts(fold)["marginal"]}
        honest_raw[held] = evaluate(tables, fold)[held]
        fold_h = optimise_uniform(tables, pool)
        honest_level_raw[held] = evaluate(tables, fold_h["cuts"])[held]
    honest_median = median_of([honest_raw[n] for n in ORDER])
    honest_level_median = median_of([honest_level_raw[n] for n in ORDER])

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
        "decomposition": {
            "shipped_level": level_of(D.ship_price()["marginal"]),
            "level_only": level_only,
            "shape_at_shipped_level": shape_at_ship_level,
            "shape_at_best_level": shape_at_best_level,
            "level_pct": pct(level_only["published_median"], ship_median),
            "structure_at_shipped_level_pct":
                pct(shape_at_ship_level["published_median"], ship_median),
            "structure_premium_pct":
                pct(best_median, level_only["published_median"]),
            "loo_honest_level_median": honest_level_median,
            "loo_honest_level_pct": pct(honest_level_median, ship_median),
            "loo_honest_structure_premium_pct":
                pct(honest_median, honest_level_median),
        },
        "anchor_distance": {
            "optimum": anchor_distance(tables, ship, best),
            "level_only": anchor_distance(tables, ship, level_arm["cuts"]),
        },
    }


def cross_law(inst, laws, results):
    """A table fitted under one 9-row reading, paid under another.

    The `aff4ad64` receipt selects the true reading after the fact, so the
    only safe table is one that still pays under the reading it was not fitted
    on. This matrix is the transfer risk, priced.
    """
    out = {}
    for paid in LAW_KEYS:
        tables = prompt_tables(inst, laws[paid])
        ship = median_of([evaluate(tables, ship_cuts(8))[n] for n in ORDER])
        row = {}
        for fitted in LAW_KEYS:
            cuts = cuts_of(results[fitted]["optimum"]["cuts"])
            raws = evaluate(tables, cuts)
            value = median_of([raws[n] for n in ORDER])
            row[fitted] = {"published_median": value,
                           "delta_pct": pct(value, ship)}
        out[paid] = {"shipped": ship, "fitted": row}
    worst = min(out[paid]["fitted"][fit]["delta_pct"]
                for paid in LAW_KEYS for fit in LAW_KEYS)
    out["worst_transfer_delta_pct"] = worst
    return out


def _minimax_arm(law_tables, ship, floors, rng):
    cuts = optimise_minimax(law_tables, ship, ORDER, rng, floors)
    price = price_of_cuts(cuts)
    agree, mismatches = verify_price(cuts, price)

    per_law, worst_prompt = {}, {}
    for key in LAW_KEYS:
        raws = evaluate(law_tables[key], cuts)
        value = median_of([raws[n] for n in ORDER])
        ship_raw = evaluate(law_tables[key], ship_cuts(8))
        prompt_pct = {n: pct(raws[n], ship_raw[n]) for n in ORDER}
        worst_prompt[key] = min(prompt_pct, key=lambda n: prompt_pct[n])
        per_law[key] = {"published_median": value,
                        "delta_pct": pct(value, ship[key]),
                        "per_prompt_raw": raws,
                        "per_prompt_delta_pct": prompt_pct,
                        "worst_prompt_delta_pct":
                            min(prompt_pct.values())}

    honest = {key: {} for key in LAW_KEYS}
    for held in ORDER:
        pool = [n for n in ORDER if n != held]
        fold = optimise_minimax(law_tables, ship, pool, rng, floors)
        for key in LAW_KEYS:
            honest[key][held] = evaluate(law_tables[key], fold)[held]
    honest_out = {}
    for key in LAW_KEYS:
        value = median_of([honest[key][n] for n in ORDER])
        honest_out[key] = {"published_median": value,
                           "delta_pct": pct(value, ship[key]),
                           "per_prompt_raw": honest[key]}

    return {"cuts": cuts[1:MAXD + 1],
            "thresholds": [c / QGRID for c in cuts[1:MAXD + 1]],
            "price_marginal": price["marginal"],
            "price_cumulative": price["cumulative"],
            "greedy_agreement": agree, "greedy_mismatches": mismatches,
            "per_law": per_law, "loo_honest": honest_out,
            "worst_delta_pct": min(per_law[k]["delta_pct"] for k in LAW_KEYS),
            "worst_loo_honest_delta_pct":
                min(honest_out[k]["delta_pct"] for k in LAW_KEYS),
            "worst_prompt_delta_pct":
                min(per_law[k]["worst_prompt_delta_pct"] for k in LAW_KEYS),
            "worst_prompt": worst_prompt,
            "edl": edl_of(law_tables["step_e208"], cuts),
            "anchor_distance": anchor_distance(law_tables["step_e208"],
                                               ship_cuts(8), cuts)}


def receipt_proof(inst, laws, rng):
    """One table that must pay under EVERY live reading of the 9-row cell.

    Two arms are reported. The free arm maximises the weakest per-law
    published median. The guarded arm adds a floor that forbids any single
    prompt falling below its shipped raw ratio, because the hidden pool is
    not the proxy pool: a prompt that does not set the proxy median may set
    the hidden one.
    """
    law_tables = {key: prompt_tables(inst, laws[key]) for key in LAW_KEYS}
    ship = {key: median_of([evaluate(law_tables[key], ship_cuts(8))[n]
                            for n in ORDER]) for key in LAW_KEYS}
    floors = {key: evaluate(law_tables[key], ship_cuts(8))
              for key in LAW_KEYS}

    free = _minimax_arm(law_tables, ship, None, rng)
    guarded = _minimax_arm(law_tables, ship, floors, rng)
    out = dict(free)
    out["shipped"] = ship
    out["guarded"] = guarded
    return out


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
        dec = r["decomposition"]
        L.append("   LEVEL / STRUCTURE DECOMPOSITION. The price has a level "
                 "(how deep the")
        L.append("   scheduler drafts) and a shape (how the rows are priced "
                 "against each")
        L.append("   other). Only the shape carries step awareness.")
        L.append("     shipped uniform, level %.2f            %12.6f"
                 % (dec["shipped_level"], ship["published_median"]))
        L.append("     best SHAPE at the shipped level        %12.6f  "
                 "%+7.3f%%"
                 % (dec["shape_at_shipped_level"]["published_median"],
                    dec["structure_at_shipped_level_pct"]))
        L.append("     best uniform LEVEL h=%.4f (level %.2f) %12.6f  "
                 "%+7.3f%%"
                 % (dec["level_only"]["marginal"][0],
                    dec["level_only"]["level"],
                    dec["level_only"]["published_median"], dec["level_pct"]))
        L.append("     best SHAPE at that best level          %12.6f  "
                 "%+7.3f%%"
                 % (dec["shape_at_best_level"]["published_median"],
                    pct(dec["shape_at_best_level"]["published_median"],
                        ship["published_median"])))
        L.append("     free table (level + structure)         %12.6f  "
                 "%+7.3f%%"
                 % (opt["published_median"], r["in_sample_delta_pct"]))
        L.append("     STRUCTURE PREMIUM over the level control %+7.3f%% "
                 "in-sample, %+7.3f%% LOO-honest"
                 % (dec["structure_premium_pct"],
                    dec["loo_honest_structure_premium_pct"]))
        L.append("     LOO-honest level control alone           %+7.3f%%"
                 % dec["loo_honest_level_pct"])
        ad = r["anchor_distance"]["optimum"]
        L.append("   ANCHOR DISTANCE of the free table (round mass moved off "
                 "the paid schedule)")
        L.append("     worst prompt %.3f; mass at depth 8: %s"
                 % (ad["worst_round_mass_relocated"],
                    " ".join("%s %.2f" % (n[:4], ad["mass_at_depth_8"][n])
                             for n in ORDER)))
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

    x = out["cross_law"]
    L.append("=" * 78)
    L.append("  CROSS-LAW TRANSFER. A table fitted under one 9-row reading, "
             "paid under")
    L.append("  another. The receipt picks the paid row after the table is "
             "chosen.")
    L.append("   %-12s %10s | %s" % ("paid law", "shipped",
                                     "  ".join("fit %-10s" % k
                                               for k in LAW_KEYS)))
    for paid in LAW_KEYS:
        row = x[paid]
        L.append("   %-12s %10.6f | %s"
                 % (paid, row["shipped"],
                    "  ".join("%+8.3f%%     " % row["fitted"][f]["delta_pct"]
                              for f in LAW_KEYS)))
    L.append("   worst transfer %+.3f%%" % x["worst_transfer_delta_pct"])

    rp = out["receipt_proof"]
    L.append("")
    L.append("  RECEIPT-PROOF TABLE (maximises the WORST per-law delta)")
    L.append("     d      %s" % "  ".join("%6d" % d for d in range(MAXD)))
    L.append("     price  %s"
             % "  ".join("%6.3f" % v for v in rp["price_marginal"]))
    L.append("     Q_(d+1)%s"
             % "  ".join("%6.4f" % v for v in rp["thresholds"]))
    L.append("     greedy-table agreement %.4f (%d mismatching grid points)"
             % (rp["greedy_agreement"], rp["greedy_mismatches"]))
    for key in LAW_KEYS:
        L.append("     paid under %-10s %12.6f  %+7.3f%% in-sample   "
                 "%+7.3f%% LOO-honest"
                 % (key, rp["per_law"][key]["published_median"],
                    rp["per_law"][key]["delta_pct"],
                    rp["loo_honest"][key]["delta_pct"]))
    L.append("     worst reading %+.3f%% in-sample, %+.3f%% LOO-honest"
             % (rp["worst_delta_pct"], rp["worst_loo_honest_delta_pct"]))
    L.append("     worst-prompt round mass relocated %.3f"
             % rp["anchor_distance"]["worst_round_mass_relocated"])
    L.append("     WORST SINGLE PROMPT %+.3f%% (%s under %s). The published "
             "median does not"
             % (rp["worst_prompt_delta_pct"],
                rp["worst_prompt"][min(LAW_KEYS, key=lambda k:
                                       rp["per_law"][k]
                                       ["worst_prompt_delta_pct"])],
                min(LAW_KEYS, key=lambda k:
                    rp["per_law"][k]["worst_prompt_delta_pct"])))
    L.append("     see a prompt that loses, but the hidden pool is not the "
             "proxy pool.")

    gp = rp["guarded"]
    L.append("")
    L.append("  NO-REGRESSION GUARDED MINIMAX TABLE (no prompt may fall "
             "below its shipped raw)")
    L.append("     d      %s" % "  ".join("%6d" % d for d in range(MAXD)))
    L.append("     price  %s"
             % "  ".join("%6.3f" % v for v in gp["price_marginal"]))
    L.append("     Q_(d+1)%s"
             % "  ".join("%6.4f" % v for v in gp["thresholds"]))
    L.append("     greedy-table agreement %.4f (%d mismatching grid points)"
             % (gp["greedy_agreement"], gp["greedy_mismatches"]))
    for key in LAW_KEYS:
        L.append("     paid under %-10s %12.6f  %+7.3f%% in-sample   "
                 "%+7.3f%% LOO-honest   worst prompt %+7.3f%%"
                 % (key, gp["per_law"][key]["published_median"],
                    gp["per_law"][key]["delta_pct"],
                    gp["loo_honest"][key]["delta_pct"],
                    gp["per_law"][key]["worst_prompt_delta_pct"]))
    L.append("     worst reading %+.3f%% in-sample, %+.3f%% LOO-honest; "
             "worst single prompt %+.3f%%"
             % (gp["worst_delta_pct"], gp["worst_loo_honest_delta_pct"],
                gp["worst_prompt_delta_pct"]))

    ve = out["validated_envelope"]
    L.append("")
    L.append("  VALIDATED EXTRAPOLATION ENVELOPE (round mass moved away from "
             "the PAID cap-7 anchor)")
    L.append("   the instrument was checked against a paid receipt at caps 4 "
             "and 5 only, so")
    L.append("   their relocation is the largest the instrument is known to "
             "survive.")
    for name, value in sorted(ve["validated_relocation"].items()):
        L.append("     %-16s %.3f%s" % (name, value,
                                        "  <- validated" if name in
                                        ("cap4", "cap5") else ""))
    L.append("     validated bar    %.3f" % ve["validated_bar"])
    for name in sorted(ve["arms"]):
        a = ve["arms"][name]
        L.append("     %-16s %.3f   %.2fx the bar   %s"
                 % (name, a["relocation_vs_cap7"], a["envelope_ratio"],
                    "inside" if a["inside_validated_envelope"]
                    else "OUTSIDE the validated envelope"))

    d = out["decision"]
    L.append("=" * 78)
    L.append("  DECISION")
    L.append("   %-10s %12s %12s %12s %12s"
             % ("law", "in-sample", "LOO-honest", "level-only", "structure"))
    for key in LAW_KEYS:
        r = out["laws"][key]
        L.append("   %-10s %+11.3f%% %+11.3f%% %+11.3f%% %+11.3f%%  %s"
                 % (key, r["in_sample_delta_pct"], r["loo_honest_delta_pct"],
                    r["decomposition"]["loo_honest_level_pct"],
                    r["decomposition"]["loo_honest_structure_premium_pct"],
                    d["per_law_verdict"][key]))
    L.append("")
    for line in d["verdict"]:
        L.append("   %s" % line)
    return "\n".join(L)


def decide(laws_out, cross, proof, envelope):
    """The assigned stop rule, plus the reading the decomposition forces.

    The assignment's statistic is the LOO-honest published-median improvement
    of the step-aware price over the shipped uniform price. That number does
    not separate the two things the free table changes at once, so the
    structure premium over an honestly refitted level control is reported
    beside it and both readings are stated.
    """
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
    struct = {k: laws_out[k]["decomposition"]
              ["loo_honest_structure_premium_pct"] for k in LAW_KEYS}
    lines = []
    if all(laws_out[k]["loo_honest_delta_pct"] < 0.5 for k in LAW_KEYS):
        lines.append("ASSIGNED RULE: STOP. LOO-honest improvement < 0.5 % "
                     "under every law; the depth-schedule family closes.")
    elif all(laws_out[k]["loo_honest_delta_pct"] >= 1.0 for k in LAW_KEYS):
        lines.append("ASSIGNED RULE: IMPLEMENT under every law. Weakest law "
                     "%s at %+.3f%% LOO-honest."
                     % (min(LAW_KEYS, key=lambda k:
                            laws_out[k]["loo_honest_delta_pct"]),
                        min(laws_out[k]["loo_honest_delta_pct"]
                            for k in LAW_KEYS)))
    else:
        lines.append("ASSIGNED RULE: mixed; the receipt selects the branch. "
                     "Best law %s at %+.3f%% LOO-honest."
                     % (best, laws_out[best]["loo_honest_delta_pct"]))
    if max(struct.values()) < 0.5:
        lines.append("STRUCTURE READING: the step-aware SHAPE is worth "
                     "%+.3f%% at most (LOO-honest, over a refitted level "
                     "control). The price-STRUCTURE question is closed; the "
                     "whole gain is the price LEVEL, which is the depth "
                     "lever the cap-8 receipt already prices."
                     % max(struct.values()))
    else:
        lines.append("STRUCTURE READING: the step-aware SHAPE is worth "
                     "%+.3f%% LOO-honest over a refitted level control."
                     % max(struct.values()))
    lines.append("TRANSFER: worst cross-law transfer of a per-law fitted "
                 "table is %+.3f%%. A table fitted under a guessed reading "
                 "can lose." % cross["worst_transfer_delta_pct"])
    guarded = proof["guarded"]
    lines.append("RECOMMENDED TABLE: the no-regression guarded minimax "
                 "table, %+.3f%% under its WORST reading (%+.3f%% "
                 "LOO-honest) with no prompt below %+.3f%%. The unguarded "
                 "minimax table pays %+.3f%% but costs one prompt %+.3f%%, "
                 "and the hidden pool is not the proxy pool."
                 % (guarded["worst_delta_pct"],
                    guarded["worst_loo_honest_delta_pct"],
                    guarded["worst_prompt_delta_pct"],
                    proof["worst_delta_pct"],
                    proof["worst_prompt_delta_pct"]))
    outside = [n for n, a in envelope["arms"].items()
               if not a["inside_validated_envelope"]]
    lines.append("ENVELOPE: %s sit outside the paid cap-4/cap-5 "
                 "extrapolation envelope (bar %.3f). The predicted gain is a "
                 "desk prediction that a paid receipt has not yet covered; "
                 "an implementation must measure it."
                 % (", ".join(sorted(outside)) if outside
                    else "no arm", envelope["validated_bar"]))
    return {"per_law_verdict": verdicts, "best_law": best,
            "structure_premium_pct": struct,
            "receipt_proof_worst_pct": proof["worst_delta_pct"],
            "receipt_proof_worst_loo_honest_pct":
                proof["worst_loo_honest_delta_pct"],
            "guarded_worst_pct": guarded["worst_delta_pct"],
            "guarded_worst_loo_honest_pct":
                guarded["worst_loo_honest_delta_pct"],
            "guarded_worst_prompt_pct": guarded["worst_prompt_delta_pct"],
            "arms_outside_validated_envelope": sorted(outside),
            "verdict": lines}


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
    out["cross_law"] = cross_law(inst, laws, out["laws"])
    out["receipt_proof"] = receipt_proof(inst, laws, rng)
    out["validated_envelope"] = validated_envelope(
        inst, laws["step_e208"],
        {"free_step_e208": cuts_of(out["laws"]["step_e208"]["optimum"]
                                   ["cuts"]),
         "minimax": cuts_of(out["receipt_proof"]["cuts"]),
         "minimax_guarded": cuts_of(out["receipt_proof"]["guarded"]["cuts"])})
    out["decision"] = decide(out["laws"], out["cross_law"],
                             out["receipt_proof"], out["validated_envelope"])

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
