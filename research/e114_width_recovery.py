#!/usr/bin/env python3
"""E114 rung 0. Recover the ranked per-prompt verify-width distribution.

The board publishes two EXACT per-prompt facts and no histogram:

    effective_mean_draft_len   -> mean verify width, exactly (a small rational)
    non_drafting_round_count   -> the number of rounds at verify width 1

With the width support fixed at M = 1..8 by the shipped cap
(`segmentedVerifyDepthCap = 7`), those two facts plus normalisation are three
linear equalities on a seven-dimensional simplex. The width distribution is
therefore NOT identified, but every linear-fractional functional of it - and the
NA weight vector is exactly such a functional - has an exactly computable range.
This module computes that range by vertex enumeration instead of guessing a
shape, so the primary deliverable carries no unvalidated model.

Three estimands, all reduced through `research/scoring_weights.py`:

    bound      exact [min, max] of each NA weight over the identified set
    maxent     the least-committal point in the set (maximum entropy)
    cost       the set intersected with the ranked round-cost curve

The pre-registered Route A estimator is NOT here. See `e114-results.md`: the
advisor withdrew its input (the E99 per-round traces do not exist) and the
forward-simulation substitute failed its own pre-registered gate. Both are
reported as negative results.

    python3 research/e114_width_recovery.py --board /tmp/yukon-board/full.json
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from itertools import combinations

import scoring_weights as sw

T = 512
WIDE = tuple(range(2, 9))

PROMPT_BY_SHA8 = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

# Finding 18b round counts, reviewed in `research/ranked_cost_curve.py` and
# reused unchanged by thorfinn's E113. `check_round_counts()` re-derives the
# admissible set from token conservation and reports where the choice is free.
ROUNDS = {"plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
          "republic": 93, "essays": 92, "medicine": 90, "botany": 81}

# harness=ranked. E113 route A, Finding 31 identity `round_us = G * W / rate`.
# The G=1 entries are the ranked analogue of `sw.ONE_GROUP_GBPS`.
W_GB = 14.41235
RANKED_RATE = {1: (1, 462.3), 2: (1, 409.8), 3: (1, 368.0), 4: (1, 333.9),
               5: (1, 272.2), 6: (2, 477.7), 7: (2, 426.6), 8: (2, 385.3)}
RANKED_ROUND_US = {M: g * W_GB / r * 1e6 for M, (g, r) in RANKED_RATE.items()}

# harness=ranked. E113 route B, two-line refit of round us against verify width
# over the 16 per-prompt rows of our own post-E100 receipts. Break at M=5, max
# absolute residual 0.68 %.
ROUTE_B = {"break": 5, "a1": 27439.9, "c1": 3799.2, "a2": 16982.0,
           "c2": 7108.1, "max_resid_pct": 0.68}
ROUTE_B_PRE_E100 = {"break": 5, "a1": 27181.5, "c1": 3995.1, "a2": 16943.2,
                    "c2": 7233.0}


def route_b_us(M: float, fit: dict = ROUTE_B) -> float:
    if M < fit["break"]:
        return fit["a1"] + fit["c1"] * M
    return fit["a2"] + fit["c2"] * M


# --- ground truth, harness=local -------------------------------------------
#
# Traced local width distributions. Each is a genuine out-of-sample test: the
# method is given only (mean width, width-1 share) and has to return the rest.
GROUND_TRUTH = {
    # E106 per-width census, W&B 19kgn6xi, 128 decode tokens, 19 rounds.
    "GT1": {"hist": {2: 1, 5: 1, 6: 4, 7: 3, 8: 10}, "tol": 0.10,
            "source": "E106 census, W&B 19kgn6xi, 19 rounds"},
    # E109 witness leg, 512 decode tokens, 77 rounds, byte-identical across 17
    # replicate reports. Depth histogram {2:1,3:2,4:6,5:5,6:7,7:56}, width = d+1.
    "GT2": {"hist": {3: 1, 4: 2, 5: 6, 6: 5, 7: 7, 8: 56}, "tol": 0.05,
            "source": "E109 witness w512, 77 rounds, 512 tokens"},
}
# GT3 exposes no histogram, only the share of rounds that dispatch one group.
GT3 = {"g1_share": 0.0641, "mean_width": 7.359, "tol": 0.08,
       "source": "research/e99-artifacts/rung67.json null-off legs, 78 rounds"}


# --- the identified set ------------------------------------------------------

def vertices(mean_wide: float, support=WIDE, extra=None):
    """Vertices of {Q >= 0, sum Q = 1, sum M Q = mean_wide[, extra]}.

    `extra` is an optional second linear functional `(f, lo, hi)` constraining
    `sum f(M) Q(M)` to the closed interval `[lo, hi]`. Without it a vertex has
    at most two support points; with it, at most three.
    """
    out = []
    if not min(support) - 1e-12 <= mean_wide <= max(support) + 1e-12:
        return out
    plain = []
    for j in support:
        if abs(j - mean_wide) < 1e-12:
            plain.append({j: 1.0})
    for j, k in combinations(support, 2):
        if j <= mean_wide <= k:
            w = (mean_wide - j) / (k - j)
            plain.append({j: 1.0 - w, k: w})
    if extra is None:
        return plain
    f, blo, bhi = extra
    for band in sorted({blo, bhi}):
        for a, b, c in combinations(support, 3):
            q = _solve3([[1.0, 1.0, 1.0],
                         [float(a), float(b), float(c)],
                         [f(a), f(b), f(c)]], [1.0, mean_wide, band])
            if q and all(v >= -1e-12 for v in q):
                out.append({a: max(0.0, q[0]), b: max(0.0, q[1]),
                            c: max(0.0, q[2])})
    for v in plain:
        val = sum(f(M) * p for M, p in v.items())
        if blo - 1e-9 <= val <= bhi + 1e-9:
            out.append(v)
    return out


def tilt(base: dict[int, float], mean_wide: float) -> dict[int, float]:
    """Exponential tilt of `base` onto the mean constraint.

    `p(M) is proportional to base(M) * exp(theta * M)`. This is the
    minimum-KL - I-projection - point of the identified set relative to `base`,
    so with a uniform base it is exactly `maxent`, and with a traced schedule
    histogram as the base it transports that realised shape to a new operating
    point while changing it as little as the mean constraint allows.
    """
    support = sorted(base)
    tot = sum(base.values())
    p0 = [base[M] / tot for M in support]
    if mean_wide <= support[0] + 1e-9:
        return {support[0]: 1.0}
    if mean_wide >= support[-1] - 1e-9:
        return {support[-1]: 1.0}
    a, b = -60.0, 60.0
    for _ in range(300):
        th = 0.5 * (a + b)
        w = [x * math.exp(th * (M - mean_wide)) for M, x in zip(support, p0)]
        if sum(M * x for M, x in zip(support, w)) / sum(w) < mean_wide:
            a = th
        else:
            b = th
    th = 0.5 * (a + b)
    w = [x * math.exp(th * (M - mean_wide)) for M, x in zip(support, p0)]
    s = sum(w)
    return {M: x / s for M, x in zip(support, w) if x > 0}


def _solve3(m, rhs):
    a = [row[:] + [r] for row, r in zip(m, rhs)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(3):
            if r == col:
                continue
            fac = a[r][col] / a[col][col]
            for c in range(col, 4):
                a[r][c] -= fac * a[col][c]
    return [a[i][3] / a[i][i] for i in range(3)]


def maxent(mean_wide: float, support=WIDE) -> dict[int, float]:
    """The maximum-entropy distribution on `support` with the given mean.

    This is the least-committal point of the identified set: it adds no
    information beyond the two exact board facts.
    """
    return tilt({M: 1.0 for M in support}, mean_wide)


def pooled_traced_shape() -> dict[int, float]:
    """Both traced local histograms pooled, 96 rounds, as a transport base."""
    base = {M: 0.0 for M in WIDE}
    for gt in GROUND_TRUTH.values():
        for M, c in gt["hist"].items():
            base[M] += c
    return base


def weight_range(mean_wide, rates, extra=None):
    """Exact [min, max] of each NA weight over the identified set."""
    verts = vertices(mean_wide, extra=extra)
    if not verts:
        return None
    vecs = [sw.na_weights(v, rates=rates) for v in verts]
    return ({na: min(v[na] for v in vecs) for na in sw.NA_CELLS},
            {na: max(v[na] for v in vecs) for na in sw.NA_CELLS},
            len(verts))


def cost_band(round_us: float, p1: float, tol: float):
    """The mean-round-time constraint, moved onto the wide-width conditional.

    The measured mean round time is an exact linear functional of the FULL
    width distribution once a per-round cost curve is fixed. Conditioning on
    width >= 2 divides the band by `1 - p1`, so a prompt that almost never
    drafts amplifies the cost curve's own error by `1 / (1 - p1)` and the
    constraint stops carrying information. That amplification is reported, not
    hidden.
    """
    if p1 >= 1.0 - 1e-12:
        return None
    lo = (round_us * (1.0 - tol) - p1 * route_b_us(1.0)) / (1.0 - p1)
    hi = (round_us * (1.0 + tol) - p1 * route_b_us(1.0)) / (1.0 - p1)
    return (route_b_us, lo, hi)


# --- board ------------------------------------------------------------------

def load_receipt(board_path: str, prefix: str) -> dict:
    full = json.load(open(board_path))
    rows = full["submissions"] if isinstance(full, dict) else full
    hits = [r for r in rows
            if isinstance(r, dict) and str(r.get("id", "")).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("prefix %r matched %d board rows" % (prefix, len(hits)))
    row = hits[0]
    out = {}
    for e in row["officialMetrics"]["per_prompt"]:
        name = PROMPT_BY_SHA8[e["prompt_sha256"][:8]]
        dl = e["effective_mean_draft_len"]
        R = ROUNDS[name]
        out[name] = {
            "mean_draft_len": dl,
            "mean_draft_len_exact": str(Fraction(dl).limit_denominator(600)),
            "mean_width": 1.0 + dl,
            "rounds": R,
            "zero_draft_rounds": e["non_drafting_round_count"],
            "p_width1": e["non_drafting_round_count"] / R,
            "accept": (T - R) / (dl * R) if dl > 0 else float("nan"),
            "raw": e["raw_ratio_of_means"],
            "round_us": T * e["mtp_seconds_per_token_mean"] / R * 1e6,
            "mtp_us_per_token": e["mtp_seconds_per_token_mean"] * 1e6,
        }
    return {"id": row["id"], "status": row["status"],
            "published": row["officialMetrics"]["mtp_decode_speedup_raw_median"],
            "prompts": out}


def check_round_counts(rec: dict) -> list[dict]:
    """Which round counts R are admissible, and does the cost curve pin R?

    Token conservation gives `R + sum(accepted) = 512` with
    `0 <= sum(accepted) <= R * dbar`, and `R * dbar` must be an integer. The
    ranked cost curve then prices each admissible R, and the residual against
    the measured round time selects one.
    """
    rows = []
    for name, p in sorted(rec["prompts"].items()):
        f = Fraction(p["mean_draft_len"]).limit_denominator(600)
        den = f.denominator
        pred = route_b_us(p["mean_width"])
        adm = []
        for R in range(den, T + 1, den):
            if R * f < T - R:          # cannot emit 512 tokens
                continue
            if R < p["zero_draft_rounds"]:
                continue
            us = T * p["mtp_us_per_token"] / R
            adm.append({"R": R, "round_us": us,
                        "resid_pct": 100 * (us - pred) / pred,
                        "accept": (T - R) / float(R * f)})
        best = min(adm, key=lambda a: abs(a["resid_pct"])) if adm else None
        rows.append({"prompt": name, "chosen": p["rounds"],
                     "admissible": [a["R"] for a in adm],
                     "cost_curve_pick": best["R"] if best else None,
                     "chosen_resid_pct":
                         next((a["resid_pct"] for a in adm
                               if a["R"] == p["rounds"]), float("nan")),
                     "agrees": bool(best and best["R"] == p["rounds"])})
    return rows


# --- validation -------------------------------------------------------------

def _stats(hist):
    n = sum(hist.values())
    mean_w = sum(M * c for M, c in hist.items()) / n
    p1 = hist.get(1, 0) / n
    return n, mean_w, p1, (mean_w - p1) / (1.0 - p1)


def validate(rates, verbose=True) -> dict:
    """Run every estimator on traced ground truth it was not derived from.

    The bound gets a COVERAGE test: the traced vector must lie inside it, or the
    constraint arithmetic is wrong. Each point estimate gets the pre-registered
    numeric gate. `transport` is cross-validated: the base shape is always the
    OTHER ground truth, never the one being predicted.
    """
    res = {"coverage": {}, "points": {}, "pass": True, "bound_pass": True}
    ids = list(GROUND_TRUTH)
    for gid, gt in GROUND_TRUTH.items():
        hist = gt["hist"]
        n, mean_w, p1, mean_wide = _stats(hist)
        w_true = sw.na_weights(hist, rates=rates)
        lo, hi, nv = weight_range(mean_wide, rates)
        covered = all(lo[na] - 1e-9 <= w_true[na] <= hi[na] + 1e-9
                      for na in sw.NA_CELLS)
        other = GROUND_TRUTH[[i for i in ids if i != gid][0]]
        points = {
            "maxent": maxent(mean_wide),
            "transport": tilt(other["hist"], mean_wide),
        }
        res["coverage"][gid] = {
            "mean_width": mean_w, "rounds": n, "covered": covered,
            "vertices": nv, "source": gt["source"], "w_true": w_true,
            "lo": lo, "hi": hi,
            "widest_band": max(hi[na] - lo[na] for na in sw.NA_CELLS)}
        res["bound_pass"] &= covered
        if verbose:
            print("  %s  %-44s rounds %3d  mean width %.4f"
                  % (gid, gt["source"], n, mean_w))
            print("      traced     %s" % _fmt(w_true))
        for pname, dist in points.items():
            w = sw.na_weights(dist, rates=rates)
            err = max(abs(w[na] - w_true[na]) for na in sw.NA_CELLS)
            ok = err <= gt["tol"]
            res["points"].setdefault(pname, {})[gid] = {
                "max_abs_err": err, "tol": gt["tol"], "pass": ok, "w": w}
            if verbose:
                print("      %-10s %s   max|err| %.4f  tol %.2f  %s"
                      % (pname, _fmt(w), err, gt["tol"],
                         "PASS" if ok else "FAIL"))
        if verbose:
            print("      bound      %s"
                  % "  ".join("NA%d[%.3f,%.3f]" % (na, lo[na], hi[na])
                              for na in sw.NA_CELLS))
            print("      coverage   %s (%d vertices, widest band %.3f)"
                  % ("PASS" if covered else "FAIL", nv,
                     res["coverage"][gid]["widest_band"]))
    verts = vertices(GT3["mean_width"])
    g1_lo = min(sum(p for M, p in v.items() if M <= 5) for v in verts)
    g1_hi = max(sum(p for M, p in v.items() if M <= 5) for v in verts)
    res["gt3"] = {"observed": GT3["g1_share"], "lo": g1_lo, "hi": g1_hi,
                  "covered": g1_lo - 1e-9 <= GT3["g1_share"] <= g1_hi + 1e-9,
                  "source": GT3["source"]}
    for pname, dist in (("maxent", maxent(GT3["mean_width"])),
                        ("transport", tilt(GROUND_TRUTH["GT1"]["hist"],
                                           GT3["mean_width"]))):
        got = sum(p for M, p in dist.items() if M <= 5)
        res["gt3"][pname] = {"g1_share": got,
                             "max_abs_err": abs(got - GT3["g1_share"]),
                             "pass": abs(got - GT3["g1_share"]) <= GT3["tol"]}
    res["bound_pass"] &= res["gt3"]["covered"]
    for pname in ("maxent", "transport"):
        res["points"][pname]["overall_pass"] = bool(
            all(v["pass"] for k, v in res["points"][pname].items()
                if k in GROUND_TRUTH) and res["gt3"][pname]["pass"])
    res["pass"] = bool(res["bound_pass"])
    if verbose:
        print("  GT3 g1_share observed %.4f  bound [%.4f,%.4f] %s  |  %s"
              % (GT3["g1_share"], g1_lo, g1_hi,
                 "PASS" if res["gt3"]["covered"] else "FAIL",
                 "  ".join("%s %.4f %s" % (p, res["gt3"][p]["g1_share"],
                                           "PASS" if res["gt3"][p]["pass"]
                                           else "FAIL")
                           for p in ("maxent", "transport"))))
        for pname in ("maxent", "transport"):
            print("  point estimate %-10s pre-registered gate: %s"
                  % (pname, "PASS" if res["points"][pname]["overall_pass"]
                     else "FAIL"))
    return res


def _fmt(w):
    return "  ".join("NA%d %.4f" % (na, w[na]) for na in sw.NA_CELLS)


# --- per-prompt recovery -----------------------------------------------------

def recover(rec: dict, rates: dict, use_cost: bool, base=None) -> dict:
    """Per-prompt NA weight bound and point estimates, harness=ranked widths."""
    base = base or pooled_traced_shape()
    tol = ROUTE_B["max_resid_pct"] / 100.0
    out = {}
    for name, p in rec["prompts"].items():
        p1 = p["p_width1"]
        mean_wide = (p["mean_width"] - p1) / (1.0 - p1)
        extra = cost_band(p["round_us"], p1, tol) if use_cost else None
        rng = weight_range(mean_wide, rates, extra=extra)
        widened = 0.0
        while rng is None and widened < 0.30:
            widened += 0.005
            extra = cost_band(p["round_us"], p1, tol + widened)
            rng = weight_range(mean_wide, rates, extra=extra)
        lo, hi, nv = rng
        me, tr = maxent(mean_wide), tilt(base, mean_wide)
        out[name] = {
            "p_width1": p1, "mean_width": p["mean_width"],
            "mean_width_wide": mean_wide, "vertices": nv,
            "lo": lo, "hi": hi,
            "maxent": sw.na_weights(me, rates=rates),
            "transport": sw.na_weights(tr, rates=rates),
            "transport_dist": tr,
            "cost_tol_widened_pct": 100 * widened,
            "cost_band_amplification": 1.0 / (1.0 - p1),
            "widest_band": max(hi[na] - lo[na] for na in sw.NA_CELLS),
            # per-token wide-QMV group time: the work an arm can act on
            "qmv_us_per_token": (p["rounds"] / T) * sum(
                mass * sum(1.0 / rates[na] for na in sw.PARTITION[M]
                           if na in rates)
                for M, mass in full(tr, p1).items()) * 1e3,
            "p_width5": full(tr, p1).get(5, 0.0),
            "p_width5_lo": min(v.get(5, 0.0) * (1 - p1)
                               for v in vertices(mean_wide, extra=extra)),
            "p_width5_hi": max(v.get(5, 0.0) * (1 - p1)
                               for v in vertices(mean_wide, extra=extra)),
        }
    return out


def full(wide_dist, p1):
    d = {1: p1} if p1 > 0 else {}
    for M, m in wide_dist.items():
        d[M] = d.get(M, 0.0) + m * (1.0 - p1)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", default="/tmp/yukon-board/full.json")
    ap.add_argument("--receipt", default="b8b8b860")
    ap.add_argument("--json", default="research/e114-artifacts/rung0.json")
    args = ap.parse_args()

    rec = load_receipt(args.board, args.receipt)
    out = {"receipt": rec["id"], "status": rec["status"],
           "published": rec["published"], "prompts": rec["prompts"],
           "partition_source": sw.PARTITION_SOURCE}

    print("=" * 78)
    print("E114 rung 0 - ranked verify-width recovery   receipt %s (%s)"
          % (rec["id"][:8], rec["status"]))
    print("=" * 78)

    print("\n-- round-count admissibility, harness=ranked --")
    out["round_counts"] = check_round_counts(rec)
    print("  %-9s %6s %-28s %6s %9s" %
          ("prompt", "chosen", "admissible R", "curve", "resid%"))
    for r in out["round_counts"]:
        print("  %-9s %6d %-28s %6s %+8.2f%% %s"
              % (r["prompt"], r["chosen"], str(r["admissible"])[:28],
                 r["cost_curve_pick"], r["chosen_resid_pct"],
                 "" if r["agrees"] else "<- curve prefers another R"))

    for frame, rates in (("local", sw.ONE_GROUP_GBPS),
                         ("ranked", sw.RANKED_ONE_GROUP_GBPS)):
        print("\n-- validation on traced ground truth, rate table harness=%s --"
              % frame)
        out.setdefault("validation", {})[frame] = validate(rates)

    for frame, rates in (("local", sw.ONE_GROUP_GBPS),
                         ("ranked", sw.RANKED_ONE_GROUP_GBPS)):
        for tag, use_cost in (("mean_only", False), ("with_cost", True)):
            key = "%s_%s" % (frame, tag)
            rows = recover(rec, rates, use_cost)
            out.setdefault("recovery", {})[key] = rows
            print("\n-- per-prompt NA weights  harness=ranked widths, "
                  "rate table harness=%s, constraints=%s --" % (frame, tag))
            print("  %-9s %6s %6s %6s | %-44s | %s"
                  % ("prompt", "M", "P(M=1)", "band", "transport", "widen%"))
            for name in sorted(rows, key=lambda n: rows[n]["mean_width"]):
                r = rows[name]
                print("  %-9s %6.3f %6.3f %6.3f | %s | %5.2f"
                      % (name, r["mean_width"], r["p_width1"],
                         r["widest_band"], _fmt(r["transport"]),
                         r["cost_tol_widened_pct"]))
    import os
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as h:
        json.dump(out, h, indent=1, sort_keys=True, default=str)
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
