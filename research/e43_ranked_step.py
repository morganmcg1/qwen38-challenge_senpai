#!/usr/bin/env python3
"""E43: does the ranked per-round data admit a step at M >= 6, and how big?

Zero GPU seconds.  Everything here is arithmetic on published ranked telemetry
plus the local width ladder that other experiments measured.

The quantity under test is the per-round cost curve T(M), M = drafts + 1 rows
verified in a round.  A prompt's observable is

    y_p = 512 * mtp_seconds_per_token_mean_p / R_p          (ms per round)

and the model is

    y_p = sum_M rho_p(M) * T(M)

under two hypotheses:

    (i)  T(M) = a + b*M                       linear
    (ii) T(M) = a + b*M + s*[M >= 6]          linear plus a step

rho_p is *not* point identified.  Published telemetry pins only two functionals
of it -- mean(M) = effective_mean_draft_len + 1 and rho_p(M=1) =
non_drafting_round_count / R_p -- so every claim below is reported as an
interval over the admissible set, with each narrowing assumption named.

Structural fact that makes hypothesis (i) falsifiable at all: a linear T has
E[T(M)] = a + b*E[M], which depends on rho only through the published mean.  So
(i) is an over-determined 8-point / 2-parameter fit that can fail, while (ii)
carries one free nuisance share per prompt and cannot.  Comparing their raw
residuals therefore compares different numbers of free parameters; the honest
output is the feasible interval of s, which this script computes exactly.
"""
from __future__ import annotations

import argparse
import fractions
import json
import math
import os
import pathlib
import re
import subprocess
import sys

BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
API = ("https://api.yukon.org/api/benchmarks/%s/submissions?limit=2000"
       % BENCHMARK_ID)
CACHE = pathlib.Path(".mlxfast-private/e43-corpus.json")
CONTRACT = pathlib.Path("fixtures/qwen3_8_27b_mtp_track.json")
OUT = pathlib.Path("research/e43-ranked-step.json")

DECODE_TOKENS = 512
MAX_M = 9                      # offered depth 8 -> M = drafts + 1 <= 9
STEP_M = 6                     # ceil(M/5) goes 1 -> 2 here
ROUND_TOL = 5e-5               # board rounds effective_mean_draft_len to 4 dp

# Ascending by ratio on the frontier row; the score is the mean of ranks 4-5.
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine",
         "essays", "republic", "botany"]
CENTRAL = ("beagle", "medicine")

DECLARED_HEAD = "559b24eb"     # sha256 of the declared one-line tree manifest
OUR_ROW = "ca9251b8"           # first Senpai row on the declared head
FRONTIER_ROW = "0cd0a6b4"      # ofou, board top
ZERO_DRAFT_ROW = "c91581eb"    # 512 rounds, zero accepted on all eight prompts

# Local width ladder (ms/round), E25/E27 instrument, declared head, same host.
LOCAL_LADDER = {1: 58.676, 2: 63.212, 3: 72.507, 4: 82.774, 5: 96.163,
                6: 128.843, 7: 138.694, 8: 149.490, 9: 164.443}
LOCAL_STEP_5_6 = 32.850        # thorfinn E38, supersedes +32.680
LOCAL_STEP_WEIGHT_STREAM = 15.401
LOCAL_STEP_RESIDUAL = 17.448

# Pair-level per-prompt ratio noise from the fixture's six no-op calibration
# sessions (fractions, not percents).
PAIR_NOISE = {"beagle": 0.00104, "botany": 0.00281, "drama": 0.00116}
PAIR_NOISE_DEFAULT = 0.00281   # widest measured prompt, used where unmeasured

CROWN_GAP_PCT = 0.5193         # score gain that would take the board crown
SIGMA_SCORE_PCT = 0.0978       # run-to-run spread of the published score


def load_corpus(refresh: bool) -> list[dict]:
    if refresh or not CACHE.exists():
        token = os.environ.get("YUKON_API_TOKEN")
        if not token:
            sys.exit("YUKON_API_TOKEN unset and no cache at %s" % CACHE)
        raw = subprocess.run(
            ["curl", "-sS", "-H", "Authorization: Bearer %s" % token, API],
            capture_output=True, check=True).stdout
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(raw)
    return json.loads(CACHE.read_text())["submissions"]


def prompt_map() -> dict:
    contract = json.loads(CONTRACT.read_text())
    out: dict = {}

    def walk(node):
        if isinstance(node, dict):
            if "sha256" in node:
                path = node.get("r2_path") or node.get("path") or ""
                hit = re.search(r"pool-([a-z_]+)\.json", path)
                if hit:
                    out[node["sha256"]] = hit.group(1)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(contract)
    return out


def official(sub: dict) -> dict:
    om = sub.get("officialMetrics") or {}
    return json.loads(om) if isinstance(om, str) else om


def per_prompt(sub: dict, pmap: dict) -> dict:
    out = {}
    for entry in official(sub).get("per_prompt") or []:
        name = pmap.get(entry.get("prompt_sha256"))
        if name:
            out[name] = entry
    return out


def find_row(subs: list, prefix: str) -> dict:
    hits = [s for s in subs if s["id"].startswith(prefix)]
    if len(hits) != 1:
        sys.exit("row %r matched %d submissions" % (prefix, len(hits)))
    return hits[0]


def score_of(ratios: dict) -> float:
    """Published rule: mean of the two central order statistics of eight."""
    s = sorted(ratios.values())
    return 0.5 * (s[3] + s[4])


# --------------------------------------------------------------------------
# Round-count recovery, with the non-uniqueness carried rather than hidden
# --------------------------------------------------------------------------

def feasible_rounds(n: float, mtp_ms_total: float, non_drafting: int) -> list:
    """Every integer round count consistent with the published telemetry.

    `effective_mean_draft_len` is the mean of the per-round draft counts, so it
    equals D/R for integers D (drafts offered) and R (rounds).  The board
    publishes it rounded to four decimals, so the identity is an interval:
    |D/R - n| <= 5e-5.  The other constraints are hard:

      * R + A = 512 with A <= D (accepted <= offered) => R >= 512/(1+n);
      * D <= 8R, the parent's per-round draft cap;
      * R >= non_drafting_round_count, since those rounds are rounds;
      * R <= 512, one primary token per round.

    Returns (R, D, A, alpha, per_round_ms) per admissible reading.  This is the
    set edward/askeladd both collapsed with a monotonicity assumption; the
    collapse is done separately in `admissible_rounds` so the assumption is
    visible and its critical threshold is reportable.
    """
    lo = DECODE_TOKENS / (1.0 + n) if n > 0 else 1.0
    out = []
    for R in range(max(1, non_drafting), DECODE_TOKENS + 1):
        if R < lo - 1e-9:
            continue
        for D in (math.floor(n * R), math.ceil(n * R)):
            if D < 0 or D > (MAX_M - 1) * R or abs(D / R - n) > ROUND_TOL:
                continue
            A = DECODE_TOKENS - R
            if A > D:
                continue
            out.append({"R": R, "D": D, "A": A, "alpha": A / D if D else 0.0,
                        "per_round_ms": mtp_ms_total / R,
                        "mean_M_exact": 1.0 + D / R})
            break
    return out


def t1_bounds(anchor: dict, kappa: float) -> dict:
    """Bracket T(1) from the ranked zero-accept row, in the correct direction.

    Row c91581eb commits 512 primary tokens with zero accepted drafts, so it ran
    R = 512 rounds of which non_drafting_round_count were width 1 and the rest
    offered drafts that were all rejected.  Its published ms/token is therefore
    a *mean* round cost, which bounds T(1) from ABOVE, not below:

        y = rho1*T(1) + (1-rho1)*E[T | drafted] >= T(1).

    A lower bound needs a cap on how expensive the few drafting rounds can be.
    Monotone T gives E[T | drafted] <= T(9) <= kappa*T(1), so

        T(1) >= y / (1 + (R - nd)*(kappa - 1)/R),

    and the tightest valid bound is the max over prompts.  kappa is a named
    assumption: a larger kappa lowers the bound and admits more readings, so
    every exclusion made with it is conservative in that direction only.
    """
    lo, lo_prompt, hi, hi_prompt = 0.0, None, None, None
    for name, p in anchor["per_prompt"].items():
        r = p["rounds"]
        d = r - p["non_drafting"]
        cand = p["mtp_ms"] / (1.0 + d * (kappa - 1.0) / r)
        if cand > lo:
            lo, lo_prompt = cand, name
        if hi is None or p["mtp_ms"] < hi:
            hi, hi_prompt = p["mtp_ms"], name
    return {"kappa": kappa, "lower_ms": lo, "upper_ms": hi,
            "lower_from": lo_prompt, "upper_from": hi_prompt,
            "upper_is_assumption_free": True,
            "note": "upper needs only monotone T; lower needs T(9) <= kappa*T(1)"}


def admissible_rounds(rows: dict, floor_ms: float) -> dict:
    """Collapse each prompt's feasible set using the T(1) floor.

    Named assumption: our tree's width-1 round is no cheaper than `floor_ms`.
    The report prints, per prompt, the floor value at which the next reading
    would become admissible again, so the strength of the assumption is
    visible rather than asserted.
    """
    out = {}
    for name, row in rows.items():
        keep = [c for c in row["feasible"] if c["per_round_ms"] >= floor_ms]
        dropped = [c for c in row["feasible"] if c["per_round_ms"] < floor_ms]
        crit = max((c["per_round_ms"] for c in dropped), default=float("nan"))
        out[name] = {"kept": keep, "dropped": dropped,
                     "critical_floor_ms": crit,
                     "unique": len(keep) == 1}
    return out


def ranked_rows(subs: list, pmap: dict, row_id: str) -> dict:
    """Per-prompt observables for one submission, with round recovery."""
    sub = find_row(subs, row_id)
    entries = per_prompt(sub, pmap)
    heads = {e.get("head_provenance_sha256", "")[:8] for e in entries.values()}
    out = {"submission": {"id": sub["id"], "solver": sub["solverUsername"],
                          "status": sub["status"],
                          "score": sub.get("officialScore"),
                          "created": sub.get("createdAt"),
                          "heads": sorted(heads)},
           "prompts": {}}
    for name in ORDER:
        e = entries[name]
        mtp_ms_total = e["mtp_seconds_per_token_mean"] * DECODE_TOKENS * 1e3
        nd = e.get("non_drafting_round_count") or 0
        n = e["effective_mean_draft_len"]
        out["prompts"][name] = {
            "n": n, "mean_M": n + 1.0,
            "ratio": e["raw_ratio_of_means"],
            "serial_ms": e["serial_seconds_per_token_mean"] * 1e3,
            "mtp_ms": e["mtp_seconds_per_token_mean"] * 1e3,
            "mtp_ms_total": mtp_ms_total,
            "non_drafting": nd,
            "parity_ok": e.get("parity_ok"),
            "head": e.get("head_provenance_sha256", "")[:8],
            "feasible": feasible_rounds(n, mtp_ms_total, nd),
        }
    return out


def zero_draft_anchor(subs: list, pmap: dict) -> dict:
    """Measure k = serial/T(1) on the ranked zero-draft row, not by inference."""
    sub = find_row(subs, ZERO_DRAFT_ROW)
    entries = per_prompt(sub, pmap)
    per = {}
    for name in ORDER:
        e = entries[name]
        n = e["effective_mean_draft_len"]
        nd = e.get("non_drafting_round_count")
        cands = feasible_rounds(n, e["mtp_seconds_per_token_mean"] * 1e3
                               * DECODE_TOKENS, nd)
        per[name] = {"n": n, "non_drafting": nd,
                     "ratio": e["raw_ratio_of_means"],
                     "serial_ms": e["serial_seconds_per_token_mean"] * 1e3,
                     "mtp_ms": e["mtp_seconds_per_token_mean"] * 1e3,
                     "rounds": cands[0]["R"] if cands else None,
                     "rounds_unique": len(cands) == 1,
                     "accepted": cands[0]["A"] if cands else None,
                     "head": e.get("head_provenance_sha256", "")[:8]}
    ks = [p["serial_ms"] / p["mtp_ms"] for p in per.values()]
    return {"id": sub["id"], "solver": sub["solverUsername"],
            "status": sub["status"], "score": sub.get("officialScore"),
            "per_prompt": per,
            "k_mean": sum(ks) / len(ks), "k_min": min(ks), "k_max": max(ks),
            "T1_ms_from_k_mean": sum(p["mtp_ms"] for p in per.values())
                                 / len(per),
            "T1_ms_slowest": max(p["mtp_ms"] for p in per.values()),
            "T1_ms_fastest": min(p["mtp_ms"] for p in per.values()),
            "all_zero_draft": all(p["n"] == 0.0 for p in per.values())}


# --------------------------------------------------------------------------
# Pre-registration.  Written before any fit was run on the recovered rows.
# --------------------------------------------------------------------------

PREREG = {
    "decisive_residual_ratio": 3.0,
    "inconclusive_residual_ratio": 1.5,
    "residual_ratio_note":
        "Step beats the alternative decisively at >= 3.0 (the factor my E34 r1 "
        "local analysis found), and the instrument discriminates nothing below "
        "1.5.  The comparison is only meaningful between models with the same "
        "parameter count under a common named rho assumption, so the fair "
        "contest is step (a,b,s) against quadratic (a,b,c), not against "
        "linear (a,b).",
    "bracket_verdicts": {
        "contains_0_and_local":
            "ranked telemetry cannot price the M>=6 lever; redirect to the "
            "causal local route",
        "excludes_0_contains_local":
            "ranked data is consistent with the full local step; no discount "
            "is justified",
        "excludes_0_excludes_local":
            "ranked step magnitude differs from local; quote the signed "
            "discount with its interval",
    },
    "tolerance_frac": PAIR_NOISE_DEFAULT,
    "tolerance_note":
        "Primary tolerance is the widest measured pair-level per-prompt ratio "
        "spread (botany, 0.281 %).  Model-slack arms at 1 % and 2 % are "
        "reported alongside because T(M) prompt-independence is an assumption, "
        "not a measurement.",
}


# --------------------------------------------------------------------------
# Admissible width distributions
# --------------------------------------------------------------------------

def support(non_drafting: int) -> list:
    """M values a round may take.  M = drafts + 1, drafts in 0..8."""
    return list(range(1, MAX_M + 1)) if non_drafting else list(range(2, MAX_M + 1))


def vertices(mean_M: float, rho1, non_drafting: int) -> list:
    """Extreme points of the admissible distribution over M.

    Constraints: sum rho = 1, sum M*rho = mean_M, rho(1) = rho1 (exact, from
    the trusted parent's non_drafting_round_count / R), rho >= 0.  With rho(1)
    pinned the free block carries two equalities, so every vertex of the free
    block has at most two support points and enumerating ordered pairs is
    exhaustive rather than a search.  Passing rho1=None leaves rho(1) free,
    which is only used to reproduce a prior bracket that did not have the
    non-drafting count.
    """
    if rho1 is None:
        free = list(support(non_drafting))
        mass, rho1 = 1.0, 0.0
    else:
        free = [m for m in support(non_drafting) if m > 1]
        mass = 1.0 - rho1
    out = []
    if mass <= 1e-12:
        return [{1: 1.0}]
    mu = (mean_M - rho1) / mass                      # conditional mean over M>1
    for i in free:
        if abs(i - mu) < 1e-12:
            v = {i: mass}
            if rho1:
                v[1] = rho1
            out.append(v)
    for i in free:
        for j in free:
            if i >= j or not (i <= mu <= j):
                continue
            pi = (j - mu) / (j - i)
            v = {i: mass * pi, j: mass * (1.0 - pi)}
            if rho1:
                v[1] = rho1
            out.append(v)
    return out


def share_bracket(mean_M: float, rho1, non_drafting: int) -> dict:
    """Exact bracket on q = P(M >= 6) and on rho(6), plus the analytic form.

    Analytic (nd = 0, support {2..9}): the conditional mean mu obeys
    q' <= (mu - 2)/4 (step mass at M=6, remainder at M=2) and
    q' >= (mu - 5)/4 (step mass at M=9, remainder at M=5), and q = (1-rho1)*q'.
    Enumeration is the check on that algebra.
    """
    verts = vertices(mean_M, rho1, non_drafting)
    if rho1 is None:
        lo_free, mass, mu = min(support(non_drafting)), 1.0, mean_M
    else:
        lo_free, mass = 2, 1.0 - rho1
        mu = (mean_M - rho1) / mass if mass > 1e-12 else 1.0
    if not verts:
        # No distribution has this mean with rho(1) pinned here, so the reading
        # of R that produced the pair is impossible.  Report it instead of
        # crashing: callers must drop the reading, not clamp it.
        return {"feasible": False, "conditional_mean_M": mu, "rho1": rho1,
                "n_vertices": 0}
    qs = [sum(p for m, p in v.items() if m >= STEP_M) for v in verts]
    w6 = [v.get(STEP_M, 0.0) for v in verts]
    x2 = [sum(p * m * m for m, p in v.items()) for v in verts]
    lo_a = mass * max(0.0, (mu - (STEP_M - 1.0)) / (MAX_M - STEP_M + 1.0))
    hi_a = (mass * min(1.0, (mu - lo_free) / (STEP_M - lo_free))
            if mass > 1e-12 else 0.0)
    return {"feasible": True, "q_lo": min(qs), "q_hi": max(qs),
            "q_lo_analytic": lo_a, "q_hi_analytic": hi_a,
            "rho6_lo": min(w6), "rho6_hi": max(w6),
            "x2_lo": min(x2), "x2_hi": max(x2),
            "conditional_mean_M": mu, "rho1": rho1,
            "n_vertices": len(verts)}


# --------------------------------------------------------------------------
# Exact-enough linear programming by vertex enumeration
# --------------------------------------------------------------------------

def lp_extreme(rows: list, rhs: list, obj: list, sense: str = "max",
               tol: float = 1e-9) -> dict:
    """Optimise obj over {x : rows[i].x <= rhs[i]} by vertex enumeration.

    The caller supplies box constraints, so the polytope is bounded and an
    optimum is attained at a vertex.  n <= 3 here, so enumerating all n-subsets
    is cheap and needs no external solver.  `feasible` is reported separately
    from the optimum so an infeasible system is a result rather than an error.
    """
    n = len(obj)
    best = None
    best_x = None
    idx = range(len(rows))

    def solve(sub):
        a = [list(rows[i]) for i in sub]
        b = [rhs[i] for i in sub]
        for col in range(n):                          # Gauss-Jordan
            piv = max(range(col, n), key=lambda r: abs(a[r][col]))
            if abs(a[piv][col]) < 1e-12:
                return None
            a[col], a[piv] = a[piv], a[col]
            b[col], b[piv] = b[piv], b[col]
            f = a[col][col]
            a[col] = [v / f for v in a[col]]
            b[col] /= f
            for r in range(n):
                if r == col:
                    continue
                g = a[r][col]
                if g:
                    a[r] = [v - g * w for v, w in zip(a[r], a[col])]
                    b[r] -= g * b[col]
        return b

    def combos(k, start=0, acc=()):
        if len(acc) == k:
            yield acc
            return
        for i in range(start, len(rows)):
            yield from combos(k, i + 1, acc + (i,))

    for sub in combos(n):
        x = solve(sub)
        if x is None:
            continue
        if any(sum(r * v for r, v in zip(rows[i], x)) > rhs[i] + 1e-7
               for i in idx):
            continue
        val = sum(c * v for c, v in zip(obj, x))
        if best is None or (val > best + tol if sense == "max"
                            else val < best - tol):
            best, best_x = val, x
    return {"feasible": best is not None, "value": best, "x": best_x}


def lp_feasible(rows: list, rhs: list) -> bool:
    """True if the system has a point, stopping at the first satisfied vertex."""
    n = len(rows[0])

    def solve(sub):
        a = [list(rows[i]) for i in sub]
        b = [rhs[i] for i in sub]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(a[r][col]))
            if abs(a[piv][col]) < 1e-12:
                return None
            a[col], a[piv] = a[piv], a[col]
            b[col], b[piv] = b[piv], b[col]
            f = a[col][col]
            a[col] = [v / f for v in a[col]]
            b[col] /= f
            for r in range(n):
                if r != col and a[r][col]:
                    g = a[r][col]
                    a[r] = [v - g * w for v, w in zip(a[r], a[col])]
                    b[r] -= g * b[col]
        return b

    def combos(k, start=0, acc=()):
        if len(acc) == k:
            yield acc
            return
        for i in range(start, len(rows)):
            yield from combos(k, i + 1, acc + (i,))

    for sub in combos(n):
        x = solve(sub)
        if x is None:
            continue
        if all(sum(r * v for r, v in zip(rows[i], x)) <= rhs[i] + 1e-7
               for i in range(len(rows))):
            return True
    return False


BOX = [("a", -1000.0, 1000.0), ("b", 0.0, 200.0), ("s", 0.0, 400.0)]


def step_polytope(obs: list, tol_frac: float, t1_floor: float | None) -> tuple:
    """Constraint rows for y_p = a + b*x_p + s*q_p, q_p in its exact bracket.

    Only the step share q_p is a free nuisance parameter: a linear T has
    E[T(M)] = a + b*E[M], so the linear part of the prediction is fixed by the
    published mean and does not depend on rho at all.
    """
    rows, rhs, why = [], [], []
    for o in obs:
        eps = tol_frac * o["y"]
        rows.append([1.0, o["x"], o["q_lo"]])
        rhs.append(o["y"] + eps)
        why.append("%s upper (q=q_lo)" % o["name"])
        rows.append([-1.0, -o["x"], -o["q_hi"]])
        rhs.append(-(o["y"] - eps))
        why.append("%s lower (q=q_hi)" % o["name"])
    rows.append([-1.0, -1.0, 0.0])                    # T(1) = a + b >= 0
    rhs.append(0.0 if t1_floor is None else -t1_floor)
    why.append("T(1) floor")
    for i, (_, lo, hi) in enumerate(BOX):
        r = [0.0, 0.0, 0.0]
        r[i] = 1.0
        rows.append(list(r))
        rhs.append(hi)
        why.append("box hi %d" % i)
        r[i] = -1.0
        rows.append(list(r))
        rhs.append(-lo)
        why.append("box lo %d" % i)
    return rows, rhs, why


def step_bracket(obs: list, tol_frac: float,
                 t1_floor: float | None = None) -> dict:
    """Interval of step magnitudes s the ranked data admits, and what is identified.

    The feasible set in (a, b, s) is a polyhedron, so its projection onto s is
    an interval and the two ends are LPs.  The per-prompt product e_p = s*q_p
    is a linear functional of (a, b) at fixed data, so it gets its own interval
    -- and that product, not s, is what a fix at M >= 6 can actually remove.
    """
    rows, rhs, why = step_polytope(obs, tol_frac, t1_floor)
    smin = lp_extreme(rows, rhs, [0.0, 0.0, 1.0], "min")
    smax = lp_extreme(rows, rhs, [0.0, 0.0, 1.0], "max")
    out = {"feasible": smin["feasible"], "tol_frac": tol_frac,
           "t1_floor": t1_floor, "s_lo": None, "s_hi": None, "excess": {},
           "witness_lo": None, "witness_hi": None, "constraints": len(rows)}
    if not smin["feasible"]:
        return out
    out["s_lo"], out["s_hi"] = smin["value"], smax["value"]
    out["witness_lo"] = dict(zip(("a", "b", "s"), smin["x"]))
    out["witness_hi"] = dict(zip(("a", "b", "s"), smax["x"]))
    out["hit_box"] = bool(smax["value"] > BOX[2][2] - 1e-6)
    for o in obs:
        lo = lp_extreme(rows, rhs, [-1.0, -o["x"], 0.0], "max")
        hi = lp_extreme(rows, rhs, [1.0, o["x"], 0.0], "max")
        out["excess"][o["name"]] = {
            "e_lo_ms": o["y"] - hi["value"], "e_hi_ms": o["y"] + lo["value"],
            "e_lo_frac": (o["y"] - hi["value"]) / o["y"],
            "e_hi_frac": (o["y"] + lo["value"]) / o["y"],
            "y_ms": o["y"], "q_lo": o["q_lo"], "q_hi": o["q_hi"]}
    return out


# --------------------------------------------------------------------------
# Deliverable (a): the model comparison, with rho held the same way for both
# --------------------------------------------------------------------------

def maxent_rho(mean_M: float, rho1: float, non_drafting: int) -> dict:
    """Max-entropy distribution over M at the published mean, rho(1) pinned.

    This is a named narrowing assumption, not a measurement.  It is the least
    committal single point of the admissible polytope, and it is the same
    inference class askeladd used, so the two analyses stay comparable.
    """
    free = [m for m in support(non_drafting) if m > 1]
    mass = 1.0 - rho1
    if mass <= 1e-12:
        return {1: 1.0}
    mu = (mean_M - rho1) / mass
    lo, hi = -60.0, 60.0
    for _ in range(500):
        lam = 0.5 * (lo + hi)
        w = [math.exp(lam * (m - free[0])) for m in free]
        z = sum(w)
        if sum(m * x for m, x in zip(free, w)) / z < mu:
            lo = lam
        else:
            hi = lam
    w = [math.exp(lam * (m - free[0])) for m in free]
    z = sum(w)
    out = {m: mass * x / z for m, x in zip(free, w)}
    if rho1:
        out[1] = rho1
    return out


def wls(design: list, y: list, weight: list) -> dict:
    """Weighted least squares by normal equations; k <= 3 so this is exact enough."""
    k = len(design[0])
    ata = [[sum(w * r[i] * r[j] for r, w in zip(design, weight))
            for j in range(k)] for i in range(k)]
    atb = [sum(w * r[i] * v for r, v, w in zip(design, y, weight))
           for i in range(k)]
    aug = [row + [rhs] for row, rhs in zip(ata, atb)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        f = aug[col][col]
        aug[col] = [v / f for v in aug[col]]
        for r in range(k):
            if r != col and aug[r][col]:
                g = aug[r][col]
                aug[r] = [v - g * w for v, w in zip(aug[r], aug[col])]
    beta = [row[-1] for row in aug]
    pred = [sum(b * v for b, v in zip(beta, row)) for row in design]
    res = [v - p for v, p in zip(y, pred)]
    ss_res = sum(r * r for r in res)
    mean_y = sum(y) / len(y)
    ss_tot = sum((v - mean_y) ** 2 for v in y)
    chi2 = sum(w * r * r for r, w in zip(res, weight))
    return {"beta": beta, "pred": pred, "residual": res,
            "rms_ms": math.sqrt(ss_res / len(y)),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
            "chi2": chi2, "chi2_per_dof": chi2 / max(1, len(y) - k),
            "params": k}


def model_comparison(obs: list, tol_frac: float) -> dict:
    """Linear vs linear+step vs smooth-convex, all under max-entropy rho.

    Three models, so the comparison the brief asks for (linear against step)
    and the comparison that is actually fair (step against a 3-parameter smooth
    alternative) are both reported.  Weights are 1/eps^2 with eps the measured
    pair-level spread, so chi2/dof is interpretable against the instrument.
    """
    y = [o["y"] for o in obs]
    w = [1.0 / (tol_frac * o["y"]) ** 2 for o in obs]
    lin = wls([[1.0, o["x"]] for o in obs], y, w)
    step = wls([[1.0, o["x"], o["q_maxent"]] for o in obs], y, w)
    quad = wls([[1.0, o["x"], o["x2_maxent"]] for o in obs], y, w)
    ratios = {
        "linear_over_step": lin["rms_ms"] / step["rms_ms"]
                            if step["rms_ms"] else float("inf"),
        "quadratic_over_step": quad["rms_ms"] / step["rms_ms"]
                               if step["rms_ms"] else float("inf"),
    }
    verdict = ("step decisive" if ratios["quadratic_over_step"]
               >= PREREG["decisive_residual_ratio"]
               else "no discrimination" if ratios["quadratic_over_step"]
               <= PREREG["inconclusive_residual_ratio"]
               else "indeterminate")
    return {"linear": lin, "step": step, "quadratic": quad,
            "ratios": ratios,
            "prereg_verdict_step_vs_smooth": verdict,
            "linear_T1_ms": lin["beta"][0] + lin["beta"][1],
            "step_T1_ms": step["beta"][0] + step["beta"][1],
            "step_magnitude_ms_maxent": step["beta"][2]}


def linear_only_tolerance(obs: list) -> dict:
    """Smallest uniform per-prompt slack at which a pure linear T fits at all.

    This is the assumption-free form of deliverable (a): with s forced to 0 the
    prediction no longer depends on rho, so the question "can a straight line
    in M explain the eight ranked per-round costs" has a yes/no answer at each
    tolerance, and the crossing point is a scale-free effect size.
    """
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        rows, rhs, _ = step_polytope(obs, mid, None)
        rows.append([0.0, 0.0, 1.0])                  # force s = 0
        rhs.append(0.0)
        if lp_extreme(rows, rhs, [0.0, 0.0, 1.0], "max")["feasible"]:
            hi = mid
        else:
            lo = mid
    return {"required_tol_frac": hi,
            "measured_pair_noise_frac": PAIR_NOISE_DEFAULT,
            "rejection_factor": hi / PAIR_NOISE_DEFAULT}


def observations(prompts: dict, selection: dict) -> list:
    """Per-prompt (x, y) plus every admissible-set quantity the fits need."""
    obs = []
    for name in ORDER:
        p = prompts[name]
        R = selection[name]
        cand = next(c for c in p["feasible"] if c["R"] == R)
        rho1 = p["non_drafting"] / R
        br = share_bracket(p["mean_M"], rho1, p["non_drafting"])
        if not br["feasible"]:
            raise ValueError("selection %s=%d admits no rho" % (name, R))
        me = maxent_rho(p["mean_M"], rho1, p["non_drafting"])
        obs.append({
            "name": name, "x": p["mean_M"], "y": cand["per_round_ms"],
            "R": R, "D": cand["D"], "A": cand["A"], "alpha": cand["alpha"],
            "ratio": p["ratio"], "rho1": rho1,
            "q_lo": br["q_lo"], "q_hi": br["q_hi"],
            "rho6_lo": br["rho6_lo"], "rho6_hi": br["rho6_hi"],
            "q_maxent": sum(v for m, v in me.items() if m >= STEP_M),
            "x2_maxent": sum(v * m * m for m, v in me.items()),
            "rejects_per_round": (cand["D"] - cand["A"]) / R,
            "eps_frac": PAIR_NOISE.get(name, PAIR_NOISE_DEFAULT),
        })
    return obs


def prompt_rows(o: dict, tol_frac: float, family: str = "step") -> tuple:
    """The two half-planes prompt o contributes to the polytope of the family.

    Under `linear` there is no nuisance share at all, because E[a + b*M] is
    a + b*E[M] whatever rho is; that is exactly why a linear T is falsifiable
    from the published means and a step T is not.
    """
    eps = tol_frac * o["y"]
    rhs = [o["y"] + eps, -(o["y"] - eps)]
    if family == "linear":
        return ([[1.0, o["x"]], [-1.0, -o["x"]]], rhs)
    lo, hi = ((o["x2_lo"], o["x2_hi"]) if family == "quadratic"
              else (o["q_lo"], o["q_hi"]))
    return ([[1.0, o["x"], lo], [-1.0, -o["x"], -hi]], rhs)


def enumerate_selections(prompts: dict, tol_frac: float,
                         node_cap: int = 4_000_000,
                         family: str = "step", first_only: bool = False) -> dict:
    """Every reading combination the step model itself admits, with no floor.

    The round count R is a nuisance parameter with several published-consistent
    values per prompt, and the earlier recoveries collapsed it with a monotone-rho
    assumption the advisor rejected.  Rather than assume, enumerate: depth-first
    over readings, pruning any partial assignment whose polytope is already
    empty.  Pruning is what makes the full cross product (about 2e8 leaves)
    tractable, and no T(1) bound is used, so the surviving set depends on
    nothing but the published rows and the model family being tested.

    Ordering is narrow prompts first: they pin (a, b) hardest, so the wide
    prompts' cheap readings die at shallow depth.
    """
    cache: dict = {}
    for name in ORDER:
        p = prompts[name]
        cache[name] = []
        for cand in p["feasible"]:
            rho1 = p["non_drafting"] / cand["R"]
            br = share_bracket(p["mean_M"], rho1, p["non_drafting"])
            if not br["feasible"]:
                continue            # no rho at all: the reading is impossible
            cache[name].append((cand["R"], prompt_rows(
                dict(br, x=p["mean_M"], y=cand["per_round_ms"]),
                tol_frac, family)))
    order = sorted(ORDER, key=lambda nm: prompts[nm]["mean_M"])
    box = BOX[:2] if family == "linear" else BOX
    box_rows, box_rhs = [], []
    for i, (_, lo, hi) in enumerate(box):
        r = [0.0] * len(box)
        r[i] = 1.0
        box_rows.append(list(r)); box_rhs.append(hi)
        r[i] = -1.0
        box_rows.append(list(r)); box_rhs.append(-lo)

    out, nodes, capped = [], 0, False

    def walk(depth, rows, rhs, sel):
        nonlocal nodes, capped
        if capped or (first_only and out):
            return
        if depth == len(order):
            out.append(dict(sel))
            return
        name = order[depth]
        for R, (prows, prhs) in cache[name]:
            nodes += 1
            if nodes > node_cap:
                capped = True
                return
            nr, nh = rows + prows, rhs + prhs
            if not lp_feasible(nr, nh):
                continue
            sel[name] = R
            walk(depth + 1, nr, nh, sel)
            del sel[name]
            if first_only and out:
                return

    walk(0, box_rows, box_rhs, {})
    total = 1
    for name in ORDER:
        total *= len(cache[name])
    pinned = {}
    for nm in ORDER:
        vals = {s[nm] for s in out}
        if len(vals) == 1:
            pinned[nm] = vals.pop()
    return {"selections": out, "nodes_visited": nodes, "capped": capped,
            "cross_product_size": total, "order": order, "family": family,
            "readings_per_prompt": {nm: len(cache[nm]) for nm in ORDER},
            "readings_surviving": {nm: sorted({s[nm] for s in out})
                                   for nm in ORDER} if out else {},
            "pinned_rounds": pinned, "tol_frac": tol_frac}


def family_threshold(prompts: dict, family: str, hi: float = 0.40) -> dict:
    """Smallest per-prompt slack at which a family can explain the board row.

    This is the assumption-free version of deliverable (a): no rho point
    estimate, no round-count choice, no T(1) bound.  A family that needs more
    slack than the measured pair-level spread is rejected by the ranked data
    itself, and the ratio of two families' thresholds is a residual ratio that
    cannot be argued away as an artefact of how rho was filled in.
    """
    if enumerate_selections(prompts, 1e-9, family=family,
                            first_only=True)["selections"]:
        return {"family": family, "threshold_frac": 0.0, "bracketed": True}
    if not enumerate_selections(prompts, hi, family=family,
                                first_only=True)["selections"]:
        return {"family": family, "threshold_frac": None, "bracketed": False,
                "searched_to": hi}
    lo = 1e-9
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if enumerate_selections(prompts, mid, family=family,
                                first_only=True)["selections"]:
            hi = mid
        else:
            lo = mid
    return {"family": family, "threshold_frac": hi, "bracketed": True,
            "lower_bracket_frac": lo}


# --------------------------------------------------------------------------
# Which readings of R survive on published facts alone
# --------------------------------------------------------------------------

def reading_audit(prompts: dict, t1_lower_ms: float,
                 surviving: dict | None = None) -> dict:
    """Per prompt, every admissible reading and the two filters that touch it.

    The kappa filter is assumption-bearing: a prompt's mean per-round cost
    cannot fall below T(1), and T(1) is bounded below only once T(9)/T(1) is
    capped.  A larger kappa lowers the bound and re-admits readings, so this
    filter is reported and never used for a headline.  The model filter is the
    joint feasibility of the whole eight-prompt row, which needs no bound on
    T(1) at all -- that is the one the headline uses.
    """
    out = {"local_T9_over_T1": LOCAL_LADDER[MAX_M] / LOCAL_LADDER[1],
           "T1_lower_ms": t1_lower_ms, "prompts": {},
           "unique_under_kappa_filter": [], "unique_under_model_filter": []}
    for name in ORDER:
        p = prompts[name]
        rows = []
        for cand in p["feasible"]:
            rho1 = p["non_drafting"] / cand["R"]
            br = share_bracket(p["mean_M"], rho1, p["non_drafting"])
            rows.append({"R": cand["R"], "alpha": cand["alpha"],
                         "per_round_ms": cand["per_round_ms"],
                         "rho1": rho1, "q_lo": br["q_lo"], "q_hi": br["q_hi"],
                         "above_T1_lower": cand["per_round_ms"] >= t1_lower_ms,
                         "survives_model": (surviving is None or
                                            cand["R"] in surviving.get(name, []))})
        out["prompts"][name] = rows
        if sum(1 for r in rows if r["above_T1_lower"]) == 1:
            out["unique_under_kappa_filter"].append(name)
        if sum(1 for r in rows if r["survives_model"]) == 1:
            out["unique_under_model_filter"].append(name)
    return out


# --------------------------------------------------------------------------
# Deliverable (c): the M >= 6 share of quantised-matvec cost
# --------------------------------------------------------------------------

def passes(m: int) -> int:
    """Weight passes at width M: IPG = ceil(M/ceil(M/5)), passes = ceil(M/IPG)."""
    ipg = math.ceil(m / math.ceil(m / 5))
    return math.ceil(m / ipg)


def ladder_split() -> dict:
    """Split the local ladder into per-row and per-pass terms.

    T(M) = A + F*M + S*(passes(M) - 1).  F and S are what weight the QMV cost
    model in the second phi variant, so they come from a fit that is shown, not
    from a quoted constant.
    """
    ms = sorted(LOCAL_LADDER)
    design = [[1.0, float(m), float(passes(m) - 1)] for m in ms]
    y = [LOCAL_LADDER[m] for m in ms]
    fit = wls(design, y, [1.0] * len(ms))
    return {"A_ms": fit["beta"][0], "F_ms_per_row": fit["beta"][1],
            "S_ms_per_extra_pass": fit["beta"][2], "r2": fit["r2"],
            "rms_ms": fit["rms_ms"],
            "implied_step_5_to_6": fit["beta"][1] + fit["beta"][2]}


def phi_bracket(o: dict, non_drafting: int, weight, cell) -> dict:
    """Bracket a QMV-cost share over the admissible set.

    The share is linear-fractional in rho, so its extremes over the polytope
    are attained at vertices and enumeration is exact.
    """
    verts = vertices(o["x"], o["rho1"], non_drafting)
    vals = []
    for v in verts:
        den = sum(p * weight(m) for m, p in v.items())
        num = sum(p * weight(m) for m, p in v.items() if cell(m))
        vals.append(num / den)
    return {"lo": min(vals), "hi": max(vals), "n_vertices": len(vals)}


def phi_report(obs: list, prompts: dict) -> dict:
    split = ladder_split()
    models = {
        "pass_count": lambda m: float(passes(m)),
        "pass_plus_row": lambda m: (split["S_ms_per_extra_pass"] * passes(m)
                                    + split["F_ms_per_row"] * m),
    }
    cells = {"M_ge_6": lambda m: m >= STEP_M, "M_eq_6": lambda m: m == STEP_M}
    out = {"ladder_split": split, "prompts": {}}
    for o in obs:
        nd = prompts[o["name"]]["non_drafting"]
        entry = {}
        for mname, weight in models.items():
            for cname, cell in cells.items():
                entry["%s/%s" % (mname, cname)] = phi_bracket(
                    o, nd, weight, cell)
        entry["round_share_M_ge_6"] = {"lo": o["q_lo"], "hi": o["q_hi"]}
        entry["row_share_M_ge_6"] = phi_bracket(o, nd, float, cells["M_ge_6"])
        out["prompts"][o["name"]] = entry
    return out


# --------------------------------------------------------------------------
# Value: what the identified excess is worth on the published score
# --------------------------------------------------------------------------

def score_after(ratios: dict, deltas: dict) -> float:
    """Score after reducing named prompts' candidate-leg time by a fraction.

    raw_p = serial / mtp, so a fractional leg saving d maps to raw_p/(1-d).
    Recomputing the median rather than differentiating keeps the order-statistic
    gate honest: the central pair can stop being central.
    """
    new = {k: v / (1.0 - deltas.get(k, 0.0)) for k, v in ratios.items()}
    return score_of(new)


def value_report(obs: list, bracket: dict) -> dict:
    ratios = {o["name"]: o["ratio"] for o in obs}
    base = score_of(ratios)

    def gain_pct(frac, end):
        deltas = {nm: frac * bracket["excess"][nm][end] for nm in CENTRAL}
        return 100.0 * (score_after(ratios, deltas) / base - 1.0), deltas

    out = {"base_score": base, "arms": {}, "crown_gap_pct": CROWN_GAP_PCT,
           "sigma_score_pct": SIGMA_SCORE_PCT, "fraction_needed": {}}
    for frac in (0.10, 0.25, 0.469, 1.0):
        for end in ("e_lo_frac", "e_hi_frac"):
            g, deltas = gain_pct(frac, end)
            out["arms"]["removed_%.3f_%s" % (frac, end)] = {
                "delta_beagle": deltas["beagle"],
                "delta_medicine": deltas["medicine"],
                "score": base * (1.0 + g / 100.0), "score_gain_pct": g}

    # How much of the identified excess a fix must actually remove to be worth
    # running: the score is a median, so this is bisected on the recomputed
    # order statistic rather than read off a derivative.
    for target, label in ((SIGMA_SCORE_PCT, "one_sigma"),
                          (CROWN_GAP_PCT, "crown")):
        for end in ("e_lo_frac", "e_hi_frac"):
            if gain_pct(1.0, end)[0] < target:
                out["fraction_needed"]["%s_%s" % (label, end)] = None
                continue
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if gain_pct(mid, end)[0] >= target:
                    hi = mid
                else:
                    lo = mid
            out["fraction_needed"]["%s_%s" % (label, end)] = hi
    return out


def monotone_rho_selection(prompts: dict) -> dict:
    """The selection edward and askeladd both used: round cost non-decreasing in M.

    Kept only as a cross-reference.  It is a named assumption about the policy,
    it is the assumption class the advisor flagged as shared between our two
    recoveries, and the reading audit above replaces it with a bound.
    """
    by_M = sorted(ORDER, key=lambda nm: prompts[nm]["mean_M"])
    sols = []

    def cost(name, cand):
        return DECODE_TOKENS / (cand["R"] * prompts[name]["ratio"])

    def walk(i, chosen, last):
        if i == len(by_M):
            sols.append(dict(chosen))
            return
        name = by_M[i]
        for cand in prompts[name]["feasible"]:
            c = cost(name, cand)
            if c < last - 1e-9:
                continue
            chosen[name] = cand["R"]
            walk(i + 1, chosen, c)
            del chosen[name]

    walk(0, {}, 0.0)
    return {"solutions": sols, "unique": len(sols) == 1}


def reject_rate(obs: list, tol_frac: float, gen_sigma: float, s_true: float,
                draws: int, seed: int) -> float:
    """How often the s = 0 model is declared infeasible on synthetic data.

    Data is generated on the fitted line plus s_true * q_maxent and perturbed by
    the *measured* pair-level spread; only the polytope's tolerance band uses
    tol_frac.  At s_true = 0 this is the instrument's false-positive rate, which
    is why the two arguments must be allowed to differ.
    """
    import random
    rng = random.Random(seed)
    base = wls([[1.0, o["x"]] for o in obs], [o["y"] for o in obs],
               [1.0] * len(obs))
    a0, b0 = base["beta"]
    rejects = 0
    for _ in range(draws):
        trial = []
        for o in obs:
            clean = a0 + b0 * o["x"] + s_true * o["q_maxent"]
            trial.append(dict(o, y=clean * (1.0 + rng.gauss(0.0, gen_sigma))))
        rows, rhs, _ = step_polytope(trial, tol_frac, None)
        rows.append([0.0, 0.0, 1.0])                       # force s = 0
        rhs.append(0.0)
        if not lp_extreme(rows, rhs, [0.0, 0.0, 1.0], "max")["feasible"]:
            rejects += 1
    return rejects / draws


def calibrate_tolerance(obs: list, gen_sigma: float, target_fp: float = 0.05,
                        draws: int = 200, seed: int = 20260819) -> dict:
    """Smallest tolerance band whose false-positive rate meets target_fp.

    The pre-registered band is one measured sigma, but the feasibility test is a
    Chebyshev (worst-residual) criterion over eight prompts, so a one-sigma band
    rejects a true zero step most of the time.  This finds the multiplier that
    makes "s = 0 is infeasible" an honest rejection.
    """
    scan = []
    for z in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        fp = reject_rate(obs, z * gen_sigma, gen_sigma, 0.0, draws, seed)
        scan.append({"z": z, "tol_frac": z * gen_sigma, "fp": fp})
        if fp <= target_fp:
            break
    ok = [r for r in scan if r["fp"] <= target_fp]
    return {"target_fp": target_fp, "gen_sigma": gen_sigma, "scan": scan,
            "z": ok[0]["z"] if ok else None,
            "tol_frac": ok[0]["tol_frac"] if ok else None,
            "fp": ok[0]["fp"] if ok else None}


def mde_step(obs: list, tol_frac: float, gen_sigma: float, draws: int = 200,
             seed: int = 20260819) -> dict:
    """Power of this instrument to reject "no step" at a given true magnitude."""
    curve = [{"s_true_ms": s,
              "power": reject_rate(obs, tol_frac, gen_sigma, s, draws, seed)}
             for s in (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.85, 48.0)]
    hits = [c["s_true_ms"] for c in curve if c["power"] >= 0.8]
    return {"tol_frac": tol_frac, "gen_sigma": gen_sigma, "draws": draws,
            "curve": curve, "mde_80pct_ms": min(hits) if hits else None,
            "false_positive_rate_at_s0": curve[0]["power"]}


def verdicts(bracket: dict, comparison: dict, lin_tol: dict,
             thresholds: dict) -> dict:
    lo, hi = bracket["s_lo"], bracket["s_hi"]
    contains_zero = lo is not None and lo <= 1e-9
    contains_local = lo is not None and lo <= LOCAL_STEP_5_6 <= hi
    if contains_zero and contains_local:
        key = "contains_0_and_local"
    elif not contains_zero and contains_local:
        key = "excludes_0_contains_local"
    else:
        key = "excludes_0_excludes_local"
    lin = thresholds["linear"]["threshold_frac"]
    return {
        "bracket_class": key,
        "bracket_sentence": PREREG["bracket_verdicts"][key],
        "local_step_inside": contains_local,
        "zero_step_inside": contains_zero,
        "bracket_width_ms": None if lo is None else hi - lo,
        "bracket_width_over_local": None if lo is None
                                    else (hi - lo) / LOCAL_STEP_5_6,
        "e34r2_discount_claim_pct": 19.14,
        "e34r2_claim_supported": False if contains_local else None,
        "linear_rejection_factor_maxent": lin_tol["rejection_factor"],
        "linear_rejection_factor_assumption_free":
            None if lin is None else lin / PAIR_NOISE_DEFAULT,
        "superlinearity": ("decisive" if lin is not None
                           and lin >= PREREG["decisive_residual_ratio"]
                           * PAIR_NOISE_DEFAULT else "not decisive"),
        "step_vs_smooth_maxent": comparison["prereg_verdict_step_vs_smooth"],
        "step_vs_smooth_assumption_free":
            "no discrimination"
            if thresholds["quadratic"]["threshold_frac"] is not None
            and abs(thresholds["quadratic"]["threshold_frac"]
                    - thresholds["step"]["threshold_frac"])
            <= PAIR_NOISE_DEFAULT else "discriminated",
    }


def analyse(args) -> dict:
    subs = load_corpus(args.refresh)
    pmap = prompt_map()
    anchor = zero_draft_anchor(subs, pmap)
    t1 = {"kappa_local": t1_bounds(anchor, LOCAL_LADDER[MAX_M] / LOCAL_LADDER[1]),
          "kappa_10": t1_bounds(anchor, 10.0)}
    data = ranked_rows(subs, pmap, args.row)
    frontier = ranked_rows(subs, pmap, FRONTIER_ROW)
    prompts = data["prompts"]

    # The pre-registered one-sigma band is a Chebyshev criterion over eight
    # prompts, so it rejects a true zero step most of the time.  Calibrate the
    # band that makes "s = 0 is infeasible" an honest rejection first, on the
    # monotone-rho reading, and carry that band through every bracket.
    mono = monotone_rho_selection(prompts)
    cal = calibrate_tolerance(observations(prompts, mono["solutions"][0]),
                              PREREG["tolerance_frac"], draws=args.draws)
    headline_arm = "calibrated_fp5pct" if cal["tol_frac"] else "model_slack_2pct"
    headline_tol = cal["tol_frac"] or 0.02

    # Deliverable (a), assumption-free form: the smallest per-prompt slack each
    # family needs to explain the row at all, over every admissible reading and
    # every admissible rho.  No point estimate of rho enters.
    thresholds = {fam: family_threshold(prompts, fam)
                  for fam in ("linear", "step", "quadratic")}

    # Round recovery without any bound on T(1): keep every reading combination
    # the step family can actually fit, and take the union of the brackets.
    enum = enumerate_selections(prompts, headline_tol)
    audit = reading_audit(prompts, t1["kappa_local"]["lower_ms"],
                          enum["readings_surviving"])
    per_sel = []
    for sel in enum["selections"]:
        br = step_bracket(observations(prompts, sel), headline_tol, None)
        rec = {"selection": sel, "s_lo": br["s_lo"], "s_hi": br["s_hi"]}
        for nm in CENTRAL:
            e = br["excess"].get(nm)
            if e:
                rec["excess_" + nm] = [e["e_lo_ms"], e["e_hi_ms"]]
        per_sel.append(rec)
    union = {"n_selections": len(per_sel), "tol_frac": headline_tol}
    if per_sel:
        union["s_lo"] = min(s["s_lo"] for s in per_sel)
        union["s_hi"] = max(s["s_hi"] for s in per_sel)
        for nm in CENTRAL:
            vals = [s["excess_" + nm] for s in per_sel if "excess_" + nm in s]
            if vals:
                union["excess_" + nm] = [min(v[0] for v in vals),
                                         max(v[1] for v in vals)]

    mono_sel = mono["solutions"][0]
    primary_sel = (mono_sel if mono_sel in enum["selections"]
                   else enum["selections"][0])
    obs = observations(prompts, primary_sel)

    arms = [("measured_pair_noise", PREREG["tolerance_frac"]),
            ("model_slack_1pct", 0.01), ("model_slack_2pct", 0.02)]
    if cal["tol_frac"]:
        arms.insert(1, (headline_arm, cal["tol_frac"]))
    brackets = {}
    for label, tol in arms:
        brackets[label] = {
            "no_T1_bound": step_bracket(obs, tol, None),
            "with_kappa_local_T1_lower":
                step_bracket(obs, tol, t1["kappa_local"]["lower_ms"])}
    primary = brackets[headline_arm]["no_T1_bound"]
    comparison = model_comparison(obs, PREREG["tolerance_frac"])
    lin_tol = linear_only_tolerance(obs)
    out = {
        "generated_utc": args.now,
        "prereg": PREREG,
        "inputs": {
            "our_row": data["submission"], "frontier_row": frontier["submission"],
            "zero_draft_anchor": {k: v for k, v in anchor.items()
                                  if k != "per_prompt"},
            "zero_draft_anchor_per_prompt": anchor["per_prompt"],
            "T1_bounds_ms": t1,
            "local_ladder_ms": LOCAL_LADDER,
            "local_step_5_6_ms": LOCAL_STEP_5_6,
            "corpus_rows": len(subs),
            "declared_head": DECLARED_HEAD,
        },
        "round_recovery": {
            "audit": audit,
            "enumeration": {k: v for k, v in enum.items()
                            if k != "selections"},
            "union_over_feasible_selections": union,
            "per_selection_brackets": per_sel,
            "monotone_rho_selection": mono,
            "primary_selection": primary_sel,
            "primary_matches_monotone_rho": primary_sel == mono_sel,
        },
        "observations": obs,
        "deliverable_a_family_thresholds": thresholds,
        "deliverable_a_model_comparison": comparison,
        "deliverable_a_linear_only_tolerance": lin_tol,
        "deliverable_b_step_brackets": brackets,
        "deliverable_b_headline_arm": headline_arm,
        "deliverable_b_tolerance_calibration": cal,
        "deliverable_c_phi": phi_report(obs, prompts),
        "value": value_report(obs, primary),
        "mde": mde_step(obs, primary["tol_frac"], PREREG["tolerance_frac"],
                        draws=args.draws),
        "verdicts": verdicts(primary, comparison, lin_tol, thresholds),
    }
    return out


def print_report(rep: dict) -> None:
    inp = rep["inputs"]
    print("E43 ranked step vs linear -- row %s (%s, score %.8f)"
          % (inp["our_row"]["id"][:8], inp["our_row"]["status"],
             inp["our_row"]["score"]))
    t1 = inp["T1_bounds_ms"]
    print("corpus rows %d, declared head %s" % (inp["corpus_rows"],
                                                inp["declared_head"]))
    print("T(1) from the zero-accept row: <= %.4f ms (monotone T only), "
          ">= %.4f ms at kappa=%.3f, >= %.4f ms at kappa=10"
          % (t1["kappa_local"]["upper_ms"], t1["kappa_local"]["lower_ms"],
             t1["kappa_local"]["kappa"], t1["kappa_10"]["lower_ms"]))
    rec = rep["round_recovery"]
    en = rec["enumeration"]
    print("\nROUND RECOVERY  no T(1) bound used: %d of %d reading combinations "
          "are jointly feasible (%d nodes, capped=%s)"
          % (rec["union_over_feasible_selections"]["n_selections"],
             en["cross_product_size"], en["nodes_visited"], en["capped"]))
    print("  pinned by the model alone: %s"
          % ", ".join("%s=%d" % kv for kv in sorted(en["pinned_rounds"].items())))
    print("  monotone-rho selection unique: %s ; primary == monotone-rho: %s"
          % (rec["monotone_rho_selection"]["unique"],
             rec["primary_matches_monotone_rho"]))
    print("  %-9s %6s %8s %8s %8s %10s  readings surviving the model filter"
          % ("prompt", "R", "alpha", "meanM", "ms/rnd", "q bracket"))
    for o in rep["observations"]:
        alt = " ".join("R=%d:%.2fms" % (c["R"], c["per_round_ms"])
                       for c in rec["audit"]["prompts"][o["name"]]
                       if c["survives_model"])
        print("  %-9s %6d %8.4f %8.4f %8.3f %10s  %s"
              % (o["name"], o["R"], o["alpha"], o["x"], o["y"],
                 "%.3f-%.3f" % (o["q_lo"], o["q_hi"]), alt))
    print("  unique under the kappa filter: %s"
          % (", ".join(rec["audit"]["unique_under_kappa_filter"]) or "none"))
    print("  local T(9)/T(1) = %.3f" % rec["audit"]["local_T9_over_T1"])

    th = rep["deliverable_a_family_thresholds"]
    print("\n(a) ASSUMPTION-FREE FAMILY TEST  minimum per-prompt slack needed "
          "to explain the row")
    for fam in ("linear", "step", "quadratic"):
        t = th[fam]["threshold_frac"]
        print("  %-10s %s  = %s x measured pair noise (%.3f %%)"
              % (fam, "infeasible to 40 %" if t is None else "%.4f %%" % (100 * t),
                 "n/a" if t is None else "%.1f" % (t / PAIR_NOISE_DEFAULT),
                 100 * PAIR_NOISE_DEFAULT))
    cmp_ = rep["deliverable_a_model_comparison"]
    lt = rep["deliverable_a_linear_only_tolerance"]
    print("\n(a) MODEL COMPARISON under max-entropy rho (named assumption)")
    for key in ("linear", "step", "quadratic"):
        f = cmp_[key]
        print("  %-9s k=%d rms %7.4f ms  R2 %.6f  chi2/dof %10.1f"
              % (key, f["params"], f["rms_ms"], f["r2"], f["chi2_per_dof"]))
    print("  residual ratios: linear/step %.3f   quadratic/step %.3f -> %s"
          % (cmp_["ratios"]["linear_over_step"],
             cmp_["ratios"]["quadratic_over_step"],
             cmp_["prereg_verdict_step_vs_smooth"]))
    print("  linear-implied T(1) %.3f ms vs zero-accept upper bound %.3f ms"
          % (cmp_["linear_T1_ms"], t1["kappa_local"]["upper_ms"]))
    print("  same at this reading only: linear needs %.3f %% slack -> %.1fx "
          "the measured %.3f %%"
          % (100 * lt["required_tol_frac"], lt["rejection_factor"],
             100 * lt["measured_pair_noise_frac"]))

    cal = rep["deliverable_b_tolerance_calibration"]
    print("\n(b) STEP MAGNITUDE BRACKETS (ms/round), local reference %.3f"
          % LOCAL_STEP_5_6)
    print("  tolerance calibration to <= %.0f %% false positives at s=0: %s"
          % (100 * cal["target_fp"],
             " ".join("%.2fs:%.2f" % (r["z"], r["fp"]) for r in cal["scan"])))
    print("  headline arm %s (tol %.3f %%)"
          % (rep["deliverable_b_headline_arm"], 100 * (cal["tol_frac"] or 0.02)))
    for label, arms in rep["deliverable_b_step_brackets"].items():
        for arm, br in arms.items():
            if not br["feasible"]:
                print("  %-20s %-14s INFEASIBLE" % (label, arm))
                continue
            print("  %-20s %-14s s in [%8.3f, %8.3f]  local inside: %s"
                  % (label, arm, br["s_lo"], br["s_hi"],
                     br["s_lo"] <= LOCAL_STEP_5_6 <= br["s_hi"]))
    prim = rep["deliverable_b_step_brackets"][
        rep["deliverable_b_headline_arm"]]["no_T1_bound"]
    print("  identified product e_p = s*q_p (what a fix can actually remove):")
    for name in ORDER:
        e = prim["excess"][name]
        print("    %-9s %7.3f - %7.3f ms/round  (%5.2f - %5.2f %% of leg)"
              % (name, e["e_lo_ms"], e["e_hi_ms"],
                 100 * e["e_lo_frac"], 100 * e["e_hi_frac"]))
    un = rep["round_recovery"]["union_over_feasible_selections"]
    if un.get("n_selections"):
        print("  union over all %d feasible round readings (no R assumption):"
              % un["n_selections"])
        print("    s in [%.3f, %.3f]  local inside: %s"
              % (un["s_lo"], un["s_hi"],
                 un["s_lo"] <= LOCAL_STEP_5_6 <= un["s_hi"]))
        for nm in CENTRAL:
            if "excess_" + nm in un:
                lo, hi = un["excess_" + nm]
                print("    excess_%-9s %7.3f - %7.3f ms/round" % (nm, lo, hi))

    phi = rep["deliverable_c_phi"]
    sp = phi["ladder_split"]
    print("\n(c) RANKED phi BRACKETS  (local split: F %.3f ms/row, "
          "S %.3f ms/pass, R2 %.4f)"
          % (sp["F_ms_per_row"], sp["S_ms_per_extra_pass"], sp["r2"]))
    for name in CENTRAL:
        e = phi["prompts"][name]
        print("  %-9s pass-count phi(M>=6) %.4f-%.4f   phi(M=6) %.4f-%.4f"
              % (name, e["pass_count/M_ge_6"]["lo"], e["pass_count/M_ge_6"]["hi"],
                 e["pass_count/M_eq_6"]["lo"], e["pass_count/M_eq_6"]["hi"]))
        print("  %-9s row share(M>=6) %.4f-%.4f   pass+row phi(M>=6) %.4f-%.4f"
              % ("", e["row_share_M_ge_6"]["lo"], e["row_share_M_ge_6"]["hi"],
                 e["pass_plus_row/M_ge_6"]["lo"], e["pass_plus_row/M_ge_6"]["hi"]))

    val = rep["value"]
    print("\nVALUE  base score %.8f (order-statistic recomputed, not "
          "differentiated)" % val["base_score"])
    for key in sorted(val["arms"]):
        a = val["arms"][key]
        print("  %-24s beagle -%.2f %% medicine -%.2f %% -> %.6f (%+.4f %%)"
              % (key, 100 * a["delta_beagle"], 100 * a["delta_medicine"],
                 a["score"], a["score_gain_pct"]))
    print("  fraction of the identified excess a fix must remove:")
    for key in sorted(val["fraction_needed"]):
        f = val["fraction_needed"][key]
        print("    %-22s %s" % (key, "unreachable" if f is None
                                else "%.2f %%" % (100 * f)))

    mde = rep["mde"]
    print("\nMDE  80 %% power at s = %s ms (draws %d, tol %.3f %%); "
          "false positive rate at s=0: %.3f"
          % (mde["mde_80pct_ms"], mde["draws"], 100 * mde["tol_frac"],
             mde["false_positive_rate_at_s0"]))
    v = rep["verdicts"]
    print("\nVERDICT  %s\n  %s" % (v["bracket_class"], v["bracket_sentence"]))


def cmd_dump(args) -> int:
    subs = load_corpus(args.refresh)
    pmap = prompt_map()
    anchor = zero_draft_anchor(subs, pmap)
    print("zero-draft anchor %s (%s, %s, score %.6f)"
          % (anchor["id"][:8], anchor["solver"], anchor["status"],
             anchor["score"]))
    kap = LOCAL_LADDER[MAX_M] / LOCAL_LADDER[1]
    bounds = t1_bounds(anchor, kap)
    print("  all_zero_draft=%s  k in [%.4f, %.4f] mean %.4f  T(1) in "
          "[%.3f, %.3f] ms at kappa=%.3f"
          % (anchor["all_zero_draft"], anchor["k_min"], anchor["k_max"],
             anchor["k_mean"], bounds["lower_ms"], bounds["upper_ms"], kap))
    for name in ORDER:
        p = anchor["per_prompt"][name]
        print("    %-9s n=%.6f nd=%s R=%s A=%s ratio=%.6f serial=%.4f "
              "mtp=%.4f head=%s"
              % (name, p["n"], p["non_drafting"], p["rounds"], p["accepted"],
                 p["ratio"], p["serial_ms"], p["mtp_ms"], p["head"]))
    for row_id in (args.row, FRONTIER_ROW):
        data = ranked_rows(subs, pmap, row_id)
        s = data["submission"]
        print("\n=== %s  %s  %s  score %.8f  heads %s"
              % (s["id"][:8], s["solver"], s["status"], s["score"],
                 ",".join(s["heads"])))
        floor = bounds["lower_ms"]
        adm = admissible_rounds(data["prompts"], floor)
        print("  kappa-filter T(1) lower bound = %.4f ms" % floor)
        print("  %-9s %8s %8s %6s | %-28s | %s"
              % ("prompt", "n", "ratio", "nd", "kept (R,alpha,ms/round)",
                 "dropped R (ms/round)"))
        for name in ORDER:
            p = data["prompts"][name]
            a = adm[name]
            kept = " ".join("R=%d a=%.4f %.3f" % (c["R"], c["alpha"],
                                                  c["per_round_ms"])
                            for c in a["kept"])
            drop = " ".join("%d(%.2f)" % (c["R"], c["per_round_ms"])
                            for c in a["dropped"])
            print("  %-9s %8.4f %8.4f %6d | %-28s | %s"
                  % (name, p["n"], p["ratio"], p["non_drafting"], kept, drop))
    return 0


def self_test() -> int:
    """Every claim-bearing routine, plus a negative control for each instrument."""
    import random
    checks, fails = 0, []

    def ck(name, cond, detail=""):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append("%s %s" % (name, detail))

    # 1. share_bracket enumeration agrees with the analytic form.
    for mean_M, rho1, nd in ((5.5327, 0.0, 0), (5.7677, 0.0, 0),
                             (1.1540, 449 / 487, 449), (6.7765, 0.0, 0)):
        br = share_bracket(mean_M, rho1, nd)
        ck("bracket_analytic",
           abs(br["q_lo"] - br["q_lo_analytic"]) < 1e-9
           and abs(br["q_hi"] - br["q_hi_analytic"]) < 1e-9,
           "%s %s" % (br["q_lo"], br["q_lo_analytic"]))

    # 2. Reproduce askeladd's published brackets, and show the nd=0 correction.
    beagle_zero = share_bracket(5.5327, None, 1)     # rho(1) free: prior work
    beagle_nd0 = share_bracket(5.5327, 0.0, 0)       # nd = 0 published
    ck("askeladd_reproduced", abs(beagle_zero["q_hi"] - 0.90654) < 1e-4,
       str(beagle_zero["q_hi"]))
    ck("askeladd_lower_matches", abs(beagle_nd0["q_lo"] - 0.13318) < 1e-4,
       str(beagle_nd0["q_lo"]))
    ck("nd0_tightens_upper", beagle_nd0["q_hi"] < beagle_zero["q_hi"] - 1e-3,
       "%s vs %s" % (beagle_nd0["q_hi"], beagle_zero["q_hi"]))

    # 3. lp_extreme against brute force on random 2-variable problems.
    rng = random.Random(7)
    worst = 0.0
    for _ in range(40):
        rows, rhs = [], []
        for _ in range(6):
            rows.append([rng.uniform(-2, 2), rng.uniform(-2, 2)])
            rhs.append(rng.uniform(0.5, 4))
        for i in range(2):                            # box
            r = [0.0, 0.0]
            r[i] = 1.0
            rows.append(list(r)); rhs.append(5.0)
            r[i] = -1.0
            rows.append(list(r)); rhs.append(5.0)
        obj = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
        got = lp_extreme(rows, rhs, obj, "max")
        best = None
        n = 240
        for i in range(n + 1):
            for j in range(n + 1):
                x = [-5 + 10 * i / n, -5 + 10 * j / n]
                if all(sum(a * v for a, v in zip(r, x)) <= b + 1e-12
                       for r, b in zip(rows, rhs)):
                    val = sum(c * v for c, v in zip(obj, x))
                    best = val if best is None else max(best, val)
        if best is not None and got["feasible"]:
            worst = max(worst, best - got["value"])
    ck("lp_vs_bruteforce", worst < 5e-2, "grid exceeded LP by %.4f" % worst)

    # 3b. Negative control: an impossible system must report infeasible.
    ck("lp_negative_control",
       not lp_extreme([[1.0, 0.0], [-1.0, 0.0]], [1.0, -2.0],
                      [1.0, 0.0])["feasible"])

    # 4. passes() reproduces the dispatch table.
    ck("passes_table",
       [passes(m) for m in range(1, 10)] == [1, 1, 1, 1, 1, 2, 2, 2, 2],
       str([passes(m) for m in range(1, 10)]))

    # 5. maxent_rho hits its constraints.
    me = maxent_rho(5.5327, 0.0, 0)
    ck("maxent_normalised", abs(sum(me.values()) - 1.0) < 1e-9)
    ck("maxent_mean", abs(sum(m * p for m, p in me.items()) - 5.5327) < 1e-6)
    me2 = maxent_rho(1.1540, 449 / 487, 449)
    ck("maxent_rho1_pinned", abs(me2[1] - 449 / 487) < 1e-12)
    ck("maxent_mean_with_rho1",
       abs(sum(m * p for m, p in me2.items()) - 1.1540) < 1e-6)

    # 6. wls recovers exact coefficients when the model is exact.
    truth = [3.0, 5.0, 7.0]
    design = [[1.0, float(i), float(i * i)] for i in range(1, 7)]
    y = [sum(t * v for t, v in zip(truth, row)) for row in design]
    fit = wls(design, y, [1.0] * len(y))
    ck("wls_exact", all(abs(a - b) < 1e-8 for a, b in zip(fit["beta"], truth)),
       str(fit["beta"]))
    ck("wls_zero_residual", fit["rms_ms"] < 1e-8)

    # 7. step_bracket covers the truth on synthetic data built from a known step.
    obs = []
    for name, mean_M, nd in (("a", 1.154, 449), ("b", 3.298, 0),
                             ("c", 3.656, 0), ("d", 5.533, 0),
                             ("e", 5.768, 0), ("f", 6.270, 0),
                             ("g", 6.425, 0), ("h", 6.777, 0)):
        rho1 = 0.92197 if nd else 0.0
        rho = maxent_rho(mean_M, rho1, nd)
        br = share_bracket(mean_M, rho1, nd)
        q = sum(p for m, p in rho.items() if m >= STEP_M)
        obs.append({"name": name, "x": mean_M, "q_lo": br["q_lo"],
                    "q_hi": br["q_hi"], "q_maxent": q,
                    "x2_maxent": sum(p * m * m for m, p in rho.items()),
                    "y": 27.0 + 4.0 * mean_M + 32.85 * q})
    syn = step_bracket(obs, 1e-9, None)
    ck("synthetic_covers_truth",
       syn["feasible"] and syn["s_lo"] - 1e-6 <= 32.85 <= syn["s_hi"] + 1e-6,
       "%s" % [syn["s_lo"], syn["s_hi"]])
    ck("synthetic_excess_covers",
       syn["excess"]["d"]["e_lo_ms"] - 1e-6
       <= 32.85 * obs[3]["q_maxent"] <= syn["excess"]["d"]["e_hi_ms"] + 1e-6,
       str(syn["excess"]["d"]))
    # 7b. Negative control: data built with no step must admit s = 0.
    flat = [dict(o, y=27.0 + 4.0 * o["x"]) for o in obs]
    ck("synthetic_no_step_admits_zero",
       step_bracket(flat, 1e-9, None)["s_lo"] < 1e-6,
       str(step_bracket(flat, 1e-9, None)["s_lo"]))
    ck("linear_tolerance_zero_on_flat",
       linear_only_tolerance(flat)["required_tol_frac"] < 1e-6)

    # 7c. The calibrated band really does control the false-positive rate, and
    #     a one-sigma Chebyshev band really does not.
    cal = calibrate_tolerance(obs, PAIR_NOISE_DEFAULT, 0.05, draws=120)
    ck("one_sigma_band_is_anticonservative", cal["scan"][0]["fp"] > 0.2,
       str(cal["scan"][0]["fp"]))
    ck("calibration_reaches_target",
       cal["tol_frac"] is not None and cal["fp"] <= 0.05, str(cal))
    ck("calibrated_band_wider_than_measured",
       cal["tol_frac"] > PAIR_NOISE_DEFAULT, str(cal["tol_frac"]))
    ck("power_rises_with_true_step",
       reject_rate(obs, cal["tol_frac"], PAIR_NOISE_DEFAULT, 60.0, 60, 1)
       > reject_rate(obs, cal["tol_frac"], PAIR_NOISE_DEFAULT, 0.0, 60, 1))

    # 8. t1_bounds points the right way.  The zero-accept row's published
    #    ms/token is a mean over rounds, so it bounds T(1) from ABOVE; the
    #    lower bound exists only under a named kappa cap and must loosen as
    #    kappa grows.  An earlier version of this script had the direction
    #    backwards and used the mean as a floor.
    br = share_bracket(5.5327, 0.0, 0)
    anc = {"per_prompt": {
        "cheap": {"rounds": 512, "non_drafting": 500, "mtp_ms": 44.0},
        "dear": {"rounds": 512, "non_drafting": 502, "mtp_ms": 44.2}}}
    b2, b5 = t1_bounds(anc, 2.0), t1_bounds(anc, 5.0)
    ck("t1_upper_is_min_prompt_mean", abs(b2["upper_ms"] - 44.0) < 1e-12,
       str(b2))
    ck("t1_lower_below_upper", b2["lower_ms"] <= b2["upper_ms"], str(b2))
    ck("t1_larger_kappa_loosens_lower", b5["lower_ms"] < b2["lower_ms"],
       "%s vs %s" % (b5["lower_ms"], b2["lower_ms"]))
    ck("t1_kappa_one_collapses_to_mean",
       abs(t1_bounds(anc, 1.0)["lower_ms"] - 44.2) < 1e-12,
       str(t1_bounds(anc, 1.0)))
    # A single T(1) across prompts is an assumption; when the row's prompt
    # means spread too far for the kappa correction to absorb, the bracket
    # inverts.  That is the diagnostic, and it must not be hidden.
    spread = {"per_prompt": dict(anc["per_prompt"],
                                 dear={"rounds": 512, "non_drafting": 502,
                                       "mtp_ms": 46.0})}
    inv = t1_bounds(spread, 2.0)
    ck("t1_inconsistent_anchor_inverts", inv["lower_ms"] > inv["upper_ms"],
       str(inv))

    # 8b. enumerate_selections and family_threshold on data built from a known
    #     truth: the true reading must survive, a real step must cost the
    #     linear family slack, and linear data must cost it none.
    def synth(step_ms):
        out = {}
        for nm, x in zip(ORDER, (1.154, 2.298, 2.656, 4.533,
                                 4.768, 5.270, 5.425, 5.777)):
            nd, R = (449, 487) if x < 1.5 else (0, 100)
            me = maxent_rho(x, nd / R, nd)
            q = sum(v for m, v in me.items() if m >= STEP_M)
            y = 27.0 + 4.0 * x + step_ms * q
            out[nm] = {"mean_M": x, "non_drafting": nd, "ratio": 1.0,
                       "feasible": [{"R": R, "per_round_ms": y},
                                    {"R": 2 * R, "per_round_ms": y / 2}]}
        return out, {nm: out[nm]["feasible"][0]["R"] for nm in ORDER}

    s_prompts, s_truth = synth(32.85)
    l_prompts, l_truth = synth(0.0)
    en = enumerate_selections(s_prompts, 1e-6)
    ck("enum_keeps_true_reading", s_truth in en["selections"],
       "%d selections" % len(en["selections"]))
    ck("enum_not_capped", not en["capped"])
    # The narrow prompt's decoy doubles R, which pins rho(1) so high that no
    # distribution has the published mean; share_bracket must say so and
    # enumerate_selections must drop the reading rather than crash.
    ck("share_bracket_reports_infeasible",
       not share_bracket(1.154, 449 / 974, 449)["feasible"])
    ck("enum_drops_impossible_reading",
       en["readings_per_prompt"][ORDER[0]] == 1
       and en["cross_product_size"] == 2 ** (len(ORDER) - 1),
       str(en["readings_per_prompt"]))
    ck("enum_linear_data_keeps_true_reading",
       l_truth in enumerate_selections(l_prompts, 1e-6,
                                       family="linear")["selections"])
    th_step = family_threshold(s_prompts, "step")
    th_lin = family_threshold(s_prompts, "linear")
    ck("threshold_zero_for_true_family",
       th_step["threshold_frac"] == 0.0, str(th_step))
    ck("threshold_positive_for_wrong_family",
       th_lin["threshold_frac"] is None or th_lin["threshold_frac"] > 1e-3,
       str(th_lin))
    ck("threshold_zero_on_linear_data",
       family_threshold(l_prompts, "linear")["threshold_frac"] == 0.0)

    # 9. phi closed form: with a pass-count weighting phi(M>=6) = 2q/(1+q).
    o = {"x": 5.5327, "rho1": 0.0, "q_lo": br["q_lo"], "q_hi": br["q_hi"]}
    ph = phi_bracket(o, 0, lambda m: float(passes(m)), lambda m: m >= STEP_M)
    for end, q in (("lo", br["q_lo"]), ("hi", br["q_hi"])):
        ck("phi_closed_form_%s" % end,
           abs(ph[end] - 2 * q / (1 + q)) < 1e-9,
           "%s vs %s" % (ph[end], 2 * q / (1 + q)))

    # 10. score identity and the no-op control on score_after.
    ratios = {"plutarch": 1.2528, "drama": 1.9167, "travel": 2.1798,
              "beagle": 3.1202, "medicine": 3.3449, "essays": 3.3661,
              "republic": 3.3940, "botany": 3.4254}
    ck("score_identity", abs(score_of(ratios) - 0.5 * (3.1202 + 3.3449)) < 1e-12)
    ck("score_after_noop",
       abs(score_after(ratios, {}) - score_of(ratios)) < 1e-12)
    ck("score_after_saturates",
       score_after(ratios, {"beagle": 0.5, "medicine": 0.5})
       <= 0.5 * (3.3661 + 3.3940) + 1e-9,
       str(score_after(ratios, {"beagle": 0.5, "medicine": 0.5})))

    # 10b. value_report's bisection: at the returned fraction the recomputed
    #      median really does clear the target, and just below it does not.
    vobs = [{"name": nm, "ratio": r} for nm, r in ratios.items()]
    vbr = {"excess": {nm: {"e_lo_frac": 0.05, "e_hi_frac": 0.30}
                      for nm in CENTRAL}}
    vr = value_report(vobs, vbr)
    for key, target in (("crown_e_hi_frac", CROWN_GAP_PCT),
                        ("one_sigma_e_hi_frac", SIGMA_SCORE_PCT)):
        f = vr["fraction_needed"][key]
        got = 100.0 * (score_after(
            ratios, {nm: f * 0.30 for nm in CENTRAL}) / vr["base_score"] - 1.0)
        below = 100.0 * (score_after(
            ratios, {nm: 0.98 * f * 0.30 for nm in CENTRAL})
            / vr["base_score"] - 1.0)
        ck("fraction_needed_%s" % key, got >= target > below,
           "f=%s gain=%s below=%s target=%s" % (f, got, below, target))

    # 11. feasible_rounds respects both hard identities.
    fr = feasible_rounds(4.5327, 6188.0, 0)
    ck("feasible_alpha_le_1", all(c["alpha"] <= 1.0 + 1e-12 for c in fr))
    ck("feasible_mean_rounds_to_published",
       all(abs(c["D"] / c["R"] - 4.5327) <= ROUND_TOL for c in fr))
    ck("feasible_contains_107", any(c["R"] == 107 for c in fr))
    ck("feasible_rounds_plus_accepted",
       all(c["R"] + c["A"] == DECODE_TOKENS for c in fr))

    # 12. ladder_split reproduces the measured 5->6 step it was not told about.
    sp = ladder_split()
    ck("ladder_split_step",
       abs(sp["implied_step_5_to_6"] - LOCAL_STEP_5_6) < 6.0,
       "%s vs %s" % (sp["implied_step_5_to_6"], LOCAL_STEP_5_6))
    ck("ladder_split_positive",
       sp["F_ms_per_row"] > 0 and sp["S_ms_per_extra_pass"] > 0, str(sp))

    print("self-test: %d checks, %d failures" % (checks, len(fails)))
    for f in fails:
        print("  FAIL %s" % f)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the ranked corpus from Yukon")
    ap.add_argument("--row", default=OUR_ROW, help="our ranked row id prefix")
    ap.add_argument("--dump", action="store_true", help="print inputs only")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--draws", type=int, default=200,
                    help="parametric bootstrap draws per MDE grid point")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--now", default="")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.dump:
        return cmd_dump(args)
    rep = analyse(args)
    print_report(rep)
    pathlib.Path(args.out).write_text(json.dumps(rep, indent=1,
                                                 sort_keys=True) + "\n")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
