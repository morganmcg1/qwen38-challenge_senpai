#!/usr/bin/env python3
"""E128-F9 - known state step, Thorfinn's fixed instruction shape, and the
per-drafting-round term the shipped depth price does not carry.

harness=ranked. Zero GPU. Analysis only.

Three questions, kept separate.

1. F9.1. Alphonse identified the runner state as a categorical label with a
   KNOWN step of about 930 us per drafting round. Enter it as a fixed offset
   with no free parameter and ask whether our M>=6 break survives. Our eight
   points come from one receipt, `d3c491b5` (morganmcg1), which sits in
   Alphonse's cluster 1, so the whole curve carries one common state cost that
   is proportional to each prompt's drafting-round fraction `phi`.

2. F9.2. Thorfinn's deleted-instruction accounting predicts a row-keyed cost
   of `38 / IPG(M)` per output element. Scaled by the M output elements of a
   round that is `38 * M / IPG(M)`, a FIXED shape with no free break. Fit one
   slope on that shape and ask whether it beats a free two-parameter pass term
   and a free-break model. The board and our tree ship different IPG tables,
   so the shape predicts a different break location for each with no fitted
   parameter.

3. F9.4. A cost paid once per DRAFTING round is amortised over the accepted
   tokens of that round, so it raises the optimal depth. Audit the shipped
   depth price for such a term, then price the depth error it causes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from e128_ourcurve import (
    MAX_ROWS,
    build_points,
    fixture_histograms,
    load_receipt,
    prompt_probs,
    r_scenarios,
)
from e128_slopes import PASSES_POST_E100, ROWS, basis, design, ols
from e128_state_fe import attach_strata, build_panel, fe_fit
from e128_strata_curve import MAXM, gbar, gvec

# The shipped `let cases` at Qwen35.swift:1565 and the board's dominant table
# from F7. Both map verify width M to the IPG template argument.
TABLE_OURS = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
TABLE_BOARD = {2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# Thorfinn's deleted-instruction counts, re-derived by the advisor.
RK_STATEMENTS = 38.0

# Alphonse's identified step, and the two other readings of the same quantity.
S_ADVISOR = 930.0
S_ALPHONSE_REGRESSION = 903.6
S_ALPHONSE_CLUSTER = 928.1
S_E128_F8 = 739.0

# Shipped depth price, Qwen36MTPBlockSession.swift:840 and :871-878.
SHIPPED_H = 0.18
SHIPPED_MAXDEPTH = 8
SHIPPED_CAP = 7  # segmentedVerifyDepthCap, :1008

# Only positions 0..6 were measured uncensored at forced depth 7.
FORCED_POSITIONS = 7

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

PHI_COL = np.where(ROWS >= 2.0, 1.0, 0.0)


# ----------------------------------------------------------------- shapes

def ipg_of(table: dict, m: int) -> float:
    """IPG template argument for width `m`; widths 1-2 have no wide case."""
    return float(table.get(m, m))


def rk_ideal(table: dict) -> np.ndarray:
    """`38 * M / IPG(M)`: the advisor's per-output-element form times M.

    Exact only when `M % IPG == 0`. The shipped table has one tail group, at
    M=7 (IPG 4, tail 3), so this form and `rk_exact` differ only there.
    """
    return np.array([RK_STATEMENTS * m / ipg_of(table, int(m)) for m in ROWS])


def rk_exact(table: dict) -> np.ndarray:
    """`38 * ceil(M / IPG(M))`: total row-keyed statements over all groups."""
    return np.array([RK_STATEMENTS * math.ceil(m / ipg_of(table, int(m)))
                     for m in ROWS])


def pass_count(table: dict) -> np.ndarray:
    return np.array([math.ceil(m / ipg_of(table, int(m))) for m in ROWS])


def bar(vec: dict, mbar: float) -> float:
    """`gbar`'s linear interpolation, for any width-indexed vector."""
    lo = max(1, min(MAXM, int(math.floor(mbar))))
    hi = max(1, min(MAXM, lo + 1))
    frac = min(max(mbar - lo, 0.0), 1.0)
    return (1 - frac) * vec[lo] + frac * vec[hi]


# ------------------------------------------------------------------ fitting

def score_fit(fit: dict, n: int) -> dict:
    k = fit["params"]
    rss = max(fit["rss"], 1e-12)
    fit["bic"] = n * math.log(rss / n) + math.log(n) * (k + 1)
    return fit


def run(points: list[dict], cols: np.ndarray, offset: np.ndarray | None,
        label: str) -> dict:
    a = design(points, cols)
    y = np.array([p["round_us"] for p in points])
    if offset is not None:
        y = y - offset
    got = ols(a, y)
    got["label"] = label
    return score_fit(got, len(y))


def hinge_cols(star: float, extra: np.ndarray | None = None) -> np.ndarray:
    cols = [np.ones_like(ROWS), ROWS, np.maximum(ROWS - star, 0.0)]
    if extra is not None:
        cols.append(extra)
    return np.stack(cols, axis=1)


def piece_cols(star: int) -> np.ndarray:
    step = (ROWS >= star).astype(float)
    return np.stack([np.ones_like(ROWS), ROWS, step,
                     step * (ROWS - star)], axis=1)


def show(fit: dict, cols: list[str]) -> None:
    beta = " ".join("%s %9.1f+-%-8.1f" % (c, b, s)
                    for c, b, s in zip(cols, fit["beta"],
                                       fit.get("se", [float("nan")] * 9)))
    print("  %-26s k=%d rmse %8.1f bic %7.2f  %s"
          % (fit["label"], fit["params"], fit["rmse"], fit["bic"], beta))


# ----------------------------------------------------------------- section 1

def section1(points: list[dict], s_known: float) -> dict:
    print("\n=== F9.1  the state as a KNOWN offset, not a fitted intercept ===")
    print("\nOur curve is fitted on ONE receipt, d3c491b5 (morganmcg1).")
    print("Alphonse's cluster 1 = {fkiene, morganmcg1, nagaral} at +928.1 us.")
    print("Alphonse's cluster 0 = {Lieisyourlie, newjordan, noskillcoding}.")
    print("E128-F8 estimated d3c491b5 against exactly that cluster-0 set")
    print("  (cf79f7df 48423d09 3b376ba2 390ec878 c63eaa21) and got")
    print("  %.0f +- %.0f us. Alphonse gets %.1f +- %.1f us on the same"
          % (S_E128_F8, 188.0, S_ALPHONSE_CLUSTER, 31.9))
    print("  account partition, %.2f of our SE away. Independent confirmation."
          % (abs(S_ALPHONSE_CLUSTER - S_E128_F8) / 188.0))
    print("\nSo our whole curve carries ONE common slow-state cost. It enters")
    print("the mean round time as s * phi, phi = drafting rounds / rounds.")

    print("\n%-10s %6s %6s %7s %9s %11s %11s"
          % ("prompt", "R", "n0", "phi", "mbar", "round_us", "destated"))
    for p in points:
        phi = p["phi"]
        print("%-10s %6d %6d %7.3f %9.4f %11.1f %11.1f"
              % (p["prompt"], p["R"], p["n0"], phi, p["mbar"],
                 p["round_us"], p["round_us"] - s_known * phi))

    phi_vec = np.array([p["phi"] for p in points])
    offset = s_known * phi_vec

    line = basis("line", PASSES_POST_E100)
    slope6 = basis("passcount_slopeonly", PASSES_POST_E100)
    line_s = np.concatenate([line, PHI_COL[:, None]], axis=1)
    slope6_s = np.concatenate([slope6, PHI_COL[:, None]], axis=1)

    fits = {
        "line raw": run(points, line, None, "line, raw"),
        "line free s": run(points, line_s, None, "line, free s"),
        "line known s": run(points, line, offset, "line, known s=%.0f" % s_known),
        "b6 raw": run(points, slope6, None, "break M>=6, raw"),
        "b6 free s": run(points, slope6_s, None, "break M>=6, free s"),
        "b6 known s": run(points, slope6, offset,
                          "break M>=6, known s=%.0f" % s_known),
    }
    print("\n## fits  (design = per-prompt width-histogram expectation)")
    show(fits["line raw"], ["a", "b"])
    show(fits["line free s"], ["a", "b", "s"])
    show(fits["line known s"], ["a", "b"])
    show(fits["b6 raw"], ["a", "b", "k"])
    show(fits["b6 free s"], ["a", "b", "k", "s"])
    show(fits["b6 known s"], ["a", "b", "k"])

    for tag in ("raw", "known s"):
        f6, fl = fits["b6 " + tag], fits["line " + tag]
        b = f6["beta"]
        ratio = (b[1] + b[2]) / b[1]
        print("\n  %-8s slope ratio hi/lo %.3f   dBIC(line - break) %+.2f"
              % (tag, ratio, fl["bic"] - f6["bic"]))

    print("\n## break location sweep under the KNOWN offset")
    print("  %-6s %10s %10s %8s %8s" % ("break", "b_lo", "b_hi", "rmse", "bic"))
    best = None
    for star in range(3, 9):
        g = np.where(ROWS >= star, 2.0, 1.0)
        g[ROWS >= 9] = 3.0
        cols = np.stack([np.ones_like(ROWS), ROWS, (g - 1.0) * ROWS], axis=1)
        fit = run(points, cols, offset, "b%d" % star)
        b = fit["beta"]
        print("  M>=%-4d %10.1f %10.1f %8.1f %8.2f"
              % (star, b[1], b[1] + b[2], fit["rmse"], fit["bic"]))
        if best is None or fit["bic"] < best[1]:
            best = (star, fit["bic"])
    print("  best BIC at break M>=%d" % best[0])
    return {"fits": {k: {"rmse": v["rmse"], "bic": v["bic"],
                         "beta": list(map(float, v["beta"]))}
                     for k, v in fits.items()},
            "best_break_known_s": best[0]}


# ----------------------------------------------------------------- section 2

def section2(points: list[dict], s_known: float) -> dict:
    print("\n=== F9.2  Thorfinn's fixed instruction shape, no free break ===")
    print("\nrow-keyed statements per round, u(M) = 38 * M / IPG(M):")
    print("  M            " + " ".join("%7.0f" % m for m in ROWS))
    for name, tbl in (("ours ", TABLE_OURS), ("board", TABLE_BOARD)):
        print("  IPG %s    " % name
              + " ".join("%7.0f" % ipg_of(tbl, int(m)) for m in ROWS))
        print("  u   %s    " % name
              + " ".join("%7.1f" % v for v in rk_ideal(tbl)))
    print("\n  Our table holds u flat at 38.0 through M=5 and steps to 76.0 at")
    print("  M=6.  The board table steps at M=5 (38.0 -> 63.3).  Those are the")
    print("  two break locations measured independently: ours M>=6, board")
    print("  M>=5.  The shape predicts both with NO fitted break parameter.")

    phi_vec = np.array([p["phi"] for p in points])
    offset = s_known * phi_vec
    u_ours = rk_ideal(TABLE_OURS)
    u_ours_x = rk_exact(TABLE_OURS)
    u_board = rk_ideal(TABLE_BOARD)
    one = np.ones_like(ROWS)

    families = [
        ("line a+bM", np.stack([one, ROWS], axis=1), ["a", "b"]),
        ("P free pass", np.stack([one, ROWS, pass_count(TABLE_OURS)], axis=1),
         ["a", "b", "f"]),
        ("TH ideal 38M/IPG", np.stack([one, u_ours], axis=1), ["a", "g"]),
        ("TH exact 38ceil", np.stack([one, u_ours_x], axis=1), ["a", "g"]),
        ("TH ideal + M", np.stack([one, ROWS, u_ours], axis=1), ["a", "b", "g"]),
        ("TH exact + M", np.stack([one, ROWS, u_ours_x], axis=1),
         ["a", "b", "g"]),
        ("TH BOARD table", np.stack([one, u_board], axis=1), ["a", "g"]),
        ("R free break M>=6", piece_cols(6), ["a", "b", "jump", "dslope"]),
    ]
    print("\n  identity: 38*ceil(M/IPG) = 38 * pass count, so 'TH exact + M'")
    print("  and 'P free pass' are the same model. The only place the ideal")
    print("  per-output-element form adds anything is the M=7 tail group,")
    print("  where it predicts u(7)=66.5 BELOW u(6)=u(8)=76.0, a cost DIP.")
    out = {}
    for tag, offs in (("raw", None), ("known s", offset)):
        print("\n## %s" % tag)
        rows = []
        for name, cols, cn in families:
            fit = run(points, cols, offs, name)
            show(fit, cn)
            rows.append((name, fit["rmse"], fit["bic"], fit["params"]))
        out[tag] = rows
        base = dict((r[0], r[2]) for r in rows)
        print("  dBIC vs 'TH ideal 38M/IPG': "
              + "  ".join("%s %+0.2f" % (n, base[n] - base["TH ideal 38M/IPG"])
                          for n, _, _, _ in rows
                          if n != "TH ideal 38M/IPG"))
    return {"our_points": {k: [list(r) for r in v] for k, v in out.items()},
            "u_ours": list(u_ours), "u_board": list(u_board)}


def section2_board(s_known: float) -> dict:
    print("\n=== F9.2b  the same fixed shape on the board population ===")
    panel = attach_strata(build_panel()["panel"])
    sids = sorted({p["sid"] for p in panel})
    print("  %d table-bearing rows, %d points, each row's OWN IPG table"
          % (len(sids), len(panel)))

    y = np.array([p["round_us"] for p in panel])
    phi = np.array([p["phi"] for p in panel])
    mb = np.array([p["mbar"] for p in panel])
    # Per-row IPG table from the strata file the panel was built from.
    raw = json.load(open("/tmp/e128_strata.json"))["tables"]
    tbl_by_sid = {sid[:8]: {int(m): int(v) for m, v in t.items()}
                  for sid, t in raw.items()}
    ubar, uxbar, gb = [], [], []
    for p in panel:
        tbl = tbl_by_sid[p["sid"]]
        uv = {int(m): RK_STATEMENTS * m / ipg_of(tbl, int(m))
              for m in range(1, MAXM + 1)}
        ux = {int(m): RK_STATEMENTS * math.ceil(m / ipg_of(tbl, int(m)))
              for m in range(1, MAXM + 1)}
        ubar.append(bar(uv, p["mbar"]))
        uxbar.append(bar(ux, p["mbar"]))
        gb.append(p["gbar"])
    ubar = np.array(ubar)
    uxbar = np.array(uxbar)
    gb = np.array(gb)
    # The whole disagreement between the ideal and exact forms: tail groups.
    dip = ubar - uxbar

    groups = [[p["sid"] for p in panel], [p["prompt"] for p in panel]]
    cluster = [p["sid"] for p in panel]
    star = 4.375  # F8's board-wide free break, held fixed here
    hinge = np.maximum(mb - star, 0.0)

    designs = [
        ("M only", np.stack([mb], axis=1), ["b"]),
        ("P  M + passes", np.stack([mb, gb], axis=1), ["b", "f"]),
        ("TH ubar only", np.stack([ubar], axis=1), ["g"]),
        ("TH ubar exact", np.stack([uxbar], axis=1), ["g"]),
        ("TH ubar + M", np.stack([mb, ubar], axis=1), ["b", "g"]),
        ("dip test M+pass+dip", np.stack([mb, gb, dip], axis=1),
         ["b", "f", "dip"]),
        ("R  M + hinge4.375", np.stack([mb, hinge], axis=1), ["b_lo", "d"]),
        ("R + P", np.stack([mb, hinge, gb], axis=1), ["b_lo", "d", "f"]),
    ]
    rows = []
    for tag, offs, off_name in ((0, None, "raw"),
                                (1, s_known * phi, "known s")):
        print("\n## %s, row+prompt FE, SEs clustered by row" % off_name)
        yy = y if offs is None else y - offs
        for name, x, cn in designs:
            fit = fe_fit(yy, x, groups, cluster=cluster, names=cn)
            print("  %-20s rmse %8.1f aicc %10.1f  %s"
                  % (name, fit["rmse"], fit["aicc"],
                     " ".join("%s %8.1f+-%-7.1f" % (c, b, s)
                              for c, b, s in zip(cn, fit["beta"], fit["se"]))))
            rows.append((off_name, name, fit["rmse"], fit["aicc"]))
    print("\n  corr(ubar, gbar) %.4f   corr(ubar, mbar) %.4f"
          % (float(np.corrcoef(ubar, gb)[0, 1]),
             float(np.corrcoef(ubar, mb)[0, 1])))
    return {"board": [list(r) for r in rows]}


# ----------------------------------------------------------------- section 4

def load_probs(forced: Path) -> dict:
    legs = {lg["prompt_id"]: lg for lg in json.load(open(forced))["legs"]}
    out = {}
    for prompt, fixtures in PROMPT_FIXTURES.items():
        reached = np.zeros(FORCED_POSITIONS)
        accepted = np.zeros(FORCED_POSITIONS)
        for fx in fixtures:
            for q in legs[fx]["positions"]:
                j = q["position"] if isinstance(q["position"], int) else 0
                reached[j] += q["reached"]
                accepted[j] += q["reached"] * q["p"]
        out[prompt] = accepted / np.maximum(reached, 1.0)
    return out


def reach_vector(probs: np.ndarray) -> np.ndarray:
    """reach[d] = probability that the d-th draft is accepted, d = 1..7."""
    return np.cumprod(probs)


def policy_rate(phi: float, c0: np.ndarray, s: float, depth: int,
                probs: np.ndarray) -> tuple:
    """Mean cost (us) and mean tokens for a round, holding phi fixed."""
    reach = reach_vector(probs)
    tokens = 1.0 + (reach[:depth].sum() if depth > 0 else 0.0)
    cost = c0[0] if depth == 0 else c0[depth] + s
    mean_cost = (1.0 - phi) * c0[0] + phi * cost
    mean_tok = (1.0 - phi) * 1.0 + phi * tokens
    return mean_cost, mean_tok


def best_depth(phi: float, c0: np.ndarray, s: float,
               probs: np.ndarray) -> int:
    cand = []
    for d in range(0, SHIPPED_CAP + 1):
        mc, mt = policy_rate(phi, c0, s, d, probs)
        cand.append((mt / mc, -d))
    cand.sort(reverse=True)
    return -cand[0][1]


def critical_s(phi: float, c0: np.ndarray, probs: np.ndarray, d0: int,
               sign: int) -> float | None:
    """Smallest |s| that moves the optimal depth one step in `sign`.

    A positive `s` prices a fixed cost per drafting round, which can only
    push the optimum deeper; a negative `s` is the mirror probe.
    """
    step = 25.0
    for i in range(1, 4001):
        s = sign * step * i
        if best_depth(phi, c0, s, probs) != d0:
            return s
    return None


def greedy_depth(probs: np.ndarray, marginal: np.ndarray, cap: int) -> int:
    cumulative = [1.0]
    running = 1.0
    for m in marginal:
        running += m
        cumulative.append(running)
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= probs[depth]
        thr = marginal[depth] * (1.0 + expected) / cumulative[depth]
        if not reach > thr:
            break
        expected += reach
        depth += 1
    return depth


def median8(vals: list[float]) -> float:
    v = sorted(vals)
    return 0.5 * (v[3] + v[4])


def entry_boundary(shipped: Path, s_u: float) -> dict:
    """The one place a per-drafting-round term can bite: the d=0 -> 1 gate.

    The shipped rule enters drafting when `min(EMA[0], conf) > 0.18`, with
    `conf = sigmoid(margin / 2)` from the pending primary's top-2 margin
    (`costModelDepth`, :1083-1087). Pricing the state raises that gate to
    `0.18 + s_u`, so the question is which of the two inputs can reach it.
    """
    gate = SHIPPED_H + s_u
    print("\n## 4.4b  the entry gate, against the recorded per-round margins")
    print("  The gate moves %.4f -> %.4f. The margin channel cannot reach"
          % (SHIPPED_H, gate))
    print("  either value: the top-2 margin is score[0] - score[1] on a sorted")
    print("  pair, so it is NON-NEGATIVE by construction, and")
    print("      conf = 1 / (1 + exp(-margin / 2))  in  [0.5, 1).")
    print("  Both gates sit below 0.5, so `min(EMA[0], conf)` is EMA-limited")
    print("  at the entry decision on every round. Measured margins:")
    print("  %-18s %7s %9s %9s %9s"
          % ("fixture", "rounds", "min", "conf min", "median"))
    out = {}
    worst = 1.0
    for leg in json.load(open(shipped))["legs"]:
        m = sorted(r["margin"] for r in leg["rounds_detail"])
        cmin = 1.0 / (1.0 + math.exp(-m[0] / 2.0))
        worst = min(worst, cmin)
        out[leg["prompt_id"]] = {"rounds": len(m), "min_margin": m[0],
                                 "min_conf": cmin,
                                 "median_margin": m[len(m) // 2]}
        print("  %-18s %7d %9.4f %9.4f %9.4f"
              % (leg["prompt_id"], len(m), m[0], cmin, m[len(m) // 2]))
    print("\n  Lowest confidence anywhere in %d rounds: %.4f, which is %.2fx"
          % (sum(v["rounds"] for v in out.values()), worst, worst / gate))
    print("  the corrected gate. No round can flip through the margin.")
    print("  So the entry decision is set by `positionAcceptEMA[0]` alone, and")
    print("  a round flips only while that EMA sits in (%.4f, %.4f], a band"
          % (SHIPPED_H, gate))
    print("  %.2f pp wide. The shipped trace does not record the EMA, so this"
          % (100.0 * s_u))
    print("  bounds the mechanism but not the ranked count. plutarch is the")
    print("  only prompt that lives near the gate (449 of 487 ranked rounds")
    print("  non-drafting), it is the lowest raw of the eight, and it is not")
    print("  one of the two median prompts, so a plutarch-only entry change")
    print("  cannot move the published median.")
    return out


def curve_sensitivity(points: list[dict], probs: dict, curves: dict,
                      s_known: float) -> dict:
    """Does the depth answer survive a different cost-curve shape?"""
    print("\n## 4.2c  the same question under three de-stated cost curves")
    print("  %-16s %s" % ("curve", " ".join("%9s" % p["prompt"][:9]
                                            for p in points)))
    out = {}
    for name, c0 in curves.items():
        d0 = [best_depth(p["phi"], c0, 0.0, probs[p["prompt"]])
              for p in points]
        ds = [best_depth(p["phi"], c0, s_known, probs[p["prompt"]])
              for p in points]
        med = {}
        for tag, depths in (("d0", d0), ("ds", ds)):
            raws = []
            for p, d in zip(points, depths):
                mc, mt = policy_rate(p["phi"], c0, s_known, d,
                                     probs[p["prompt"]])
                raws.append(p["serial"] * mt / (mc * 1e-6))
            med[tag] = median8(raws)
        moved = sum(1 for a, b in zip(d0, ds) if a != b)
        print("  %-16s %s   d*(0)" % (name, " ".join("%9d" % v for v in d0)))
        print("  %-16s %s   d*(%.0f), %d moved, median %+.8f"
              % ("", " ".join("%9d" % v for v in ds), s_known, moved,
                 med["ds"] - med["d0"]))
        out[name] = {"d0": d0, "ds": ds, "moved": moved,
                     "delta_median": med["ds"] - med["d0"]}
    return out


def section4(points: list[dict], probs: dict, c0: np.ndarray,
             s_known: float) -> dict:
    print("\n=== F9.4  the per-drafting-round term and the shipped price ===")
    print("""
## 4.1  Does the shipped price vector carry a per-DRAFTING-round term?  NO.

  Qwen36MTPBlockSession.swift:840
      private static let headStepCostRatio = 0.18
  :871-878  makeUniformDepthPrice()
      marginal: [Double](repeating: headStepCostRatio, count: maxDepth),
      cumulative: (0 ... maxDepth).map { 1.0 + Double($0) * headStepCostRatio }
  :936-947  prefixCosts(_:)
      var out = [1.0]
      var running = 1.0
      for value in marginal { running += value; out.append(running) }
  :1094-1096  costModelDepth
      let threshold = price.marginal[depth] * (1.0 + expected) /
          price.cumulative[depth]
      guard reach > threshold else { break }

  There IS a fixed term: the literal `1.0` that `prefixCosts` and
  `makeUniformDepthPrice` seed the cumulative with. It is the whole cost of a
  round at depth 0, and every arm inherits it.

  It is a per-ROUND term, NOT a per-DRAFTING-round term. `cumulative[0] = 1.0`
  and `cumulative[d] = 1.0 + d*h`, so the same 1.0 is charged whether the
  round drafts or not. Alphonse's state cost is paid ONLY when the drafting
  path is entered, so the correct cost is
      C(0) = 1.0
      C(d) = 1.0 + s_u + d*h        for d >= 1
  and no `DepthPrice` the shipped code can build has that shape, because
  `prefixCosts` fixes the base at 1.0 and every arm holds the marginal total
  at `maxDepth * headStepCostRatio` (:850, :928).

  Consequences, both directions:
    entry  d=0 -> 1:  shipped extends iff reach > h; correct iff reach > s_u+h.
                      The shipped rule ENTERS DRAFTING TOO EAGERLY by s_u.
    within d >= 1:    the correct cumulative is LARGER by s_u, so the correct
                      threshold is SMALLER.  The shipped rule STOPS TOO EARLY.
  The advisor's direction is right once drafting has started, and reversed at
  the entry boundary. Both errors are exactly s_u in the cumulative.
""")
    s_u = s_known / c0[0]
    print("  base round C0(M=1) = %.1f us (de-stated curve)" % c0[0])
    print("  s_u = %.0f / %.1f = %.5f price units, %.1f %% of one 0.18 step"
          % (s_known, c0[0], s_u, 100.0 * s_u / SHIPPED_H))
    print("  marginal implied by the de-stated curve: "
          + " ".join("%.4f" % ((c0[d + 1] - c0[d]) / c0[0])
                     for d in range(SHIPPED_CAP)))

    print("\n## 4.2  optimal fixed depth at s = 0 and s = %.0f us" % s_known)
    print("  phi held at each prompt's shipped value; acceptance from the")
    print("  rung-1 forced-depth-7 legs, pooled over fixtures.")
    print("  CAVEAT: a fixed depth is not the shipped rule. The shipped rule")
    print("  is adaptive on per-round EMAs, so it can take a deep round only")
    print("  on a stretch the head is already proving. Comparing a fixed")
    print("  depth with the shipped rule at POOLED acceptance removes exactly")
    print("  that adaptivity, so the 'd shipped' column is a reference point,")
    print("  not a verdict on the shipped scheduler.")
    print("\n  %-10s %6s %8s %8s %8s %10s %10s %9s"
          % ("prompt", "phi", "d*(0)", "d*(s)", "d shipped", "raw d*(0)",
             "raw d*(s)", "delta %"))
    per = {}
    rows = []
    for p in points:
        pr = probs[p["prompt"]]
        best = {}
        for s_true in (0.0, s_known):
            best[s_true] = best_depth(p["phi"], c0, s_true, pr)
        d_ship = greedy_depth(pr, np.full(SHIPPED_MAXDEPTH, SHIPPED_H),
                              SHIPPED_CAP)
        raws = {}
        for d, tag in ((best[0.0], "d0"), (best[s_known], "ds"),
                       (d_ship, "ship")):
            mc, mt = policy_rate(p["phi"], c0, s_known, d, pr)
            raws[tag] = p["serial"] * mt / (mc * 1e-6)
        per[p["prompt"]] = {"phi": p["phi"], "d0": best[0.0],
                            "ds": best[s_known], "dship": d_ship, **raws}
        rows.append(per[p["prompt"]])
        print("  %-10s %6.3f %8d %8d %8d %10.5f %10.5f %+8.3f"
              % (p["prompt"], p["phi"], best[0.0], best[s_known], d_ship,
                 raws["d0"], raws["ds"],
                 100.0 * (raws["ds"] / raws["d0"] - 1.0)))
    m_d0 = median8([r["d0"] for r in rows])
    m_ds = median8([r["ds"] for r in rows])
    m_sh = median8([r["ship"] for r in rows])
    print("\n  median, all legs run in the SLOW state s=%.0f:" % s_known)
    print("    depth chosen as if s=0        %.8f" % m_d0)
    print("    depth chosen knowing s=%-6.0f %.8f" % (s_known, m_ds))
    print("    shipped greedy rule at h=0.18  %.8f  (see CAVEAT)" % m_sh)
    print("    recoverable by pricing s      %+.8f  (%+.4f %%)"
          % (m_ds - m_d0, 100.0 * (m_ds / m_d0 - 1.0)))

    print("\n## 4.2b  how large would the round cost have to be to move d*?")
    print("  critical s (us) at which each prompt's optimal depth changes.")
    print("  %-10s %6s %14s %14s %12s"
          % ("prompt", "d*(0)", "s to go deeper", "s to go shallower",
             "930 / |s|"))
    crit = {}
    for p in points:
        pr = probs[p["prompt"]]
        d0 = best_depth(p["phi"], c0, 0.0, pr)
        up = critical_s(p["phi"], c0, pr, d0, +1)
        dn = critical_s(p["phi"], c0, pr, d0, -1)
        crit[p["prompt"]] = {"d0": d0, "up": up, "down": dn}
        near = min(abs(v) for v in (up, dn) if v is not None) \
            if any(v is not None for v in (up, dn)) else float("inf")
        print("  %-10s %6d %14s %14s %12s"
              % (p["prompt"], d0,
                 "none" if up is None else "%+.0f" % up,
                 "none" if dn is None else "%+.0f" % dn,
                 "-" if near == float("inf") else "%.4f" % (s_known / near)))
    print("\n  A cost of %.0f us moves NO prompt's optimum. The depth ladder"
          % s_known)
    print("  is quantised by the M=6 pass cliff, C0(6)-C0(5) = %.0f us, which"
          % (c0[5] - c0[4]))
    print("  is %.1fx the state term. Inside a segment the step is %.0f us,"
          % ((c0[5] - c0[4]) / s_known, c0[1] - c0[0]))
    print("  still %.1fx the state term." % ((c0[1] - c0[0]) / s_known))

    print("\n## 4.3  is the flat 0.18 still optimal under a %.0f us term?"
          % s_known)
    print("  %-8s %14s %14s" % ("h", "median s=0", "median s=%.0f" % s_known))
    grid = [round(0.02 + 0.005 * i, 3) for i in range(0, 97)]
    best = {0.0: (None, -1.0), s_known: (None, -1.0)}
    table = []
    for h in grid:
        marg = np.full(SHIPPED_MAXDEPTH, h)
        line = {}
        for s_true in (0.0, s_known):
            raws = []
            for p in points:
                pr = probs[p["prompt"]]
                d = greedy_depth(pr, marg, SHIPPED_CAP)
                mc, mt = policy_rate(p["phi"], c0, s_true, d, pr)
                raws.append(p["serial"] * mt / (mc * 1e-6))
            med = median8(raws)
            line[s_true] = med
            if med > best[s_true][1]:
                best[s_true] = (h, med)
        table.append((h, line[0.0], line[s_known]))
    for h, a, b in table:
        if abs(h - round(h * 20) / 20) < 1e-9 or abs(h - SHIPPED_H) < 1e-9:
            print("  %-8.3f %14.8f %14.8f%s"
                  % (h, a, b, "   <- shipped" if abs(h - SHIPPED_H) < 1e-9
                     else ""))
    print("  argmax h at s=0        %.3f  median %.8f" % best[0.0])
    print("  argmax h at s=%-6.0f  %.3f  median %.8f"
          % (s_known, best[s_known][0], best[s_known][1]))

    print("\n## 4.4  the state-aware price and the hedge")
    print("  state-aware price: marginal[0] = h + s_u, marginal[j>0] = h.")
    print("  That is the exact corrected cumulative, and it is NOT")
    print("  expressible through `prefixCosts` at a held marginal total, so it")
    print("  needs a new constructor, not a new `DepthPriceArm`.")
    aware = np.full(SHIPPED_MAXDEPTH, SHIPPED_H)
    aware[0] = SHIPPED_H + s_u
    prices = {"shipped flat 0.18": np.full(SHIPPED_MAXDEPTH, SHIPPED_H),
              "state-aware 0.18+s_u": aware}
    if best[s_known][0] is not None:
        prices["retuned flat %.3f" % best[s_known][0]] = np.full(
            SHIPPED_MAXDEPTH, best[s_known][0])
    print("\n  %-24s %14s %14s %12s"
          % ("price", "true s=0", "true s=%.0f" % s_known, "worst case"))
    cells = {}
    for name, marg in prices.items():
        got = {}
        for s_true in (0.0, s_known):
            raws = []
            for p in points:
                pr = probs[p["prompt"]]
                d = greedy_depth(pr, marg, SHIPPED_CAP)
                mc, mt = policy_rate(p["phi"], c0, s_true, d, pr)
                raws.append(p["serial"] * mt / (mc * 1e-6))
            got[s_true] = median8(raws)
        cells[name] = got
        print("  %-24s %14.8f %14.8f %12.8f"
              % (name, got[0.0], got[s_known], min(got.values())))
    depths = {}
    for name, marg in prices.items():
        depths[name] = {p["prompt"]: greedy_depth(probs[p["prompt"]], marg,
                                                  SHIPPED_CAP)
                        for p in points}
    print("\n  depth chosen per prompt")
    print("  %-24s %s" % ("price", " ".join("%9s" % p["prompt"][:9]
                                            for p in points)))
    for name in prices:
        print("  %-24s %s"
              % (name, " ".join("%9d" % depths[name][p["prompt"]]
                                for p in points)))
    print("\n  per-prompt raw where the state-aware price changed the depth")
    for p in points:
        da, db = depths["shipped flat 0.18"][p["prompt"]], \
            depths["state-aware 0.18+s_u"][p["prompt"]]
        if da == db:
            continue
        pr = probs[p["prompt"]]
        ra = [p["serial"] * mt / (mc * 1e-6) for mc, mt in
              (policy_rate(p["phi"], c0, s_known, d, pr) for d in (da, db))]
        print("    %-10s depth %d -> %d   raw %.5f -> %.5f  (%+.4f %%)"
              % (p["prompt"], da, db, ra[0], ra[1],
                 100.0 * (ra[1] / ra[0] - 1.0)))
    ship = cells["shipped flat 0.18"]
    sa = cells["state-aware 0.18+s_u"]
    print("\n  hedge accounting, worst case over the two states:")
    print("    shipped      %.8f" % min(ship.values()))
    print("    state-aware  %.8f" % min(sa.values()))
    print("    gain if slow %+.8f  (%+.4f %%)"
          % (sa[s_known] - ship[s_known],
             100.0 * (sa[s_known] / ship[s_known] - 1.0)))
    print("    cost if fast %+.8f  (%+.4f %%)"
          % (sa[0.0] - ship[0.0], 100.0 * (sa[0.0] / ship[0.0] - 1.0)))
    return {"s_u": s_u, "per_prompt": per, "critical_s": crit,
            "median": {"d0": m_d0, "ds": m_ds, "ship": m_sh},
            "flat_argmax": {"s0": best[0.0][0], "s_known": best[s_known][0]},
            "cells": {k: {str(a): b for a, b in v.items()}
                      for k, v in cells.items()},
            "depths": depths}


# ---------------------------------------------------------------------- main

def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path,
                    default=Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--identity", type=Path,
                    default=here / "e128-artifacts/rung0-identity.json")
    ap.add_argument("--shipped", type=Path,
                    default=here / "e128-artifacts/rung1-shipped.json")
    ap.add_argument("--forced", type=Path,
                    default=here / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--scenario", default="assumed")
    ap.add_argument("--state-us", type=float, default=S_ADVISOR)
    ap.add_argument("--skip-board", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    hists = fixture_histograms(args.shipped)
    scenarios = r_scenarios(args.identity)
    receipt = load_receipt(args.board, args.receipt)
    points = build_points(receipt, scenarios[args.scenario], hists)
    for p in points:
        row = receipt["per_prompt"][p["prompt"]]
        p["n0"] = row["non_drafting"]
        p["serial"] = row["serial"]
        p["candidate"] = row["candidate"]
        p["phi"] = 1.0 - float(np.array(prompt_probs(p, True))[0])

    print("harness=ranked  E128-F9  zero GPU  analysis only")
    print("receipt %s  score %.8f  R scenario %s  state step %.0f us"
          % (receipt["id"][:8], receipt["score"], args.scenario,
             args.state_us))
    print("state readings: advisor %.0f, Alphonse regression %.1f, Alphonse"
          % (S_ADVISOR, S_ALPHONSE_REGRESSION))
    print("  cluster %.1f, E128-F8 two-channel %.0f"
          % (S_ALPHONSE_CLUSTER, S_E128_F8))

    out = {"receipt": receipt["id"][:8], "state_us": args.state_us}
    out["section1"] = section1(points, args.state_us)
    out["section2"] = section2(points, args.state_us)
    if not args.skip_board:
        out["section2b"] = section2_board(args.state_us)

    # De-stated cost curves. The headline is `slopeonly` at break 6; the line
    # and the free piecewise bracket it so section 4 is not one curve's answer.
    phi_vec = np.array([p["phi"] for p in points])
    offset = args.state_us * phi_vec
    curves = {}
    for name, cols in (("slopeonly b6",
                        basis("passcount_slopeonly", PASSES_POST_E100)),
                       ("line", basis("line", PASSES_POST_E100)),
                       ("piece b6", piece_cols(6))):
        beta = run(points, cols, offset, name)["beta"]
        curves[name] = np.array([float(beta @ cols[int(m) - 1]) for m in ROWS])
    c0 = curves["slopeonly b6"]
    print("\n## de-stated cost curves  (us per round)")
    print("  %-14s %s" % ("M", " ".join("%9.0f" % m for m in ROWS)))
    for name, vec in curves.items():
        print("  %-14s %s" % (name, " ".join("%9.1f" % v for v in vec)))
    out["curves"] = {k: list(map(float, v)) for k, v in curves.items()}

    probs = load_probs(args.forced)
    print("\n## uncensored per-position acceptance, pooled over fixtures")
    for name in sorted(probs):
        print("  %-10s %s" % (name,
                              " ".join("%.4f" % v for v in probs[name])))
    out["section4"] = section4(points, probs, c0, args.state_us)
    out["section4c"] = curve_sensitivity(points, probs, curves, args.state_us)
    out["section4d"] = entry_boundary(args.shipped,
                                      out["section4"]["s_u"])

    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=float))
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
