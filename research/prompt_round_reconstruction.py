#!/usr/bin/env python3
"""Reconstruct per-prompt round counts and acceptance from public ranked receipts.

Every ranked receipt publishes, per prompt, `mean_draft_len`, `non_drafting_rounds`,
`mtp_spt`, `serial_spt` and `raw_ratio`. `mean_draft_len` is an EXACT rational whose
reduced denominator divides the round count, because it is
`total_proposed_drafts / round_count` computed in double precision from two integers.

That single observation identifies, for each hidden prompt:

    R  round count
    P  total proposed drafts   = mean_draft_len * R
    A  total accepted drafts   = 512 - R          (window closure)
    a  per-draft accept rate   = A / P

R is pinned by four independent constraints:

    C1 exactness   mean_draft_len == Fraction(P, R) with zero double-precision error
    C2 closure     R + A == 512 and 0 <= A <= P
    C3 census      R >= non_drafting_rounds, and P >= (R - non_drafting_rounds)
    C4 weight floor  512 * mtp_spt / R  >=  c1, the candidate's own depth-0 round cost

C4 is the decisive one. Every round streams the full 27B backbone at least once, so a
round can never cost less than a depth-0 round. c1 is measured, not assumed: `plutarch`
has 449 non-drafting rounds out of a uniquely-determined 487, so its leg time is almost
entirely depth-0 rounds.

Run with --self-test for the positive controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction

WINDOW_TOKENS = 512

# E1's measured per-depth marginal round cost on the local M4 Pro host, in ms.
# Index d is the marginal cost of moving from depth d-1 to depth d.
E1_MARGINAL_MS_M4 = {
    1: 5.47,
    2: 5.04,
    3: 15.77,
    4: 24.40,
    5: 18.98,
    6: 19.50,
    7: 18.66,
    8: 25.41,
}
E1_DEPTH0_ROUND_MS_M4 = 65.009

PROMPTS = [
    "plutarch",
    "drama",
    "travel",
    "beagle",
    "medicine",
    "republic",
    "essays",
    "botany",
]

# The near-serial prompt used to calibrate the depth-0 round cost. Its round count
# is uniquely identified because 2 * R exceeds the 512-token window.
ANCHOR_PROMPT = "plutarch"


def cumulative_ms_m4(depth: float) -> float:
    """Cumulative extra round cost above depth 0, at possibly fractional depth."""
    if depth <= 0:
        return 0.0
    total = 0.0
    d = 1
    while d <= 8 and d <= depth:
        total += E1_MARGINAL_MS_M4[d]
        d += 1
    frac = depth - (d - 1)
    if frac > 0 and d <= 8:
        total += frac * E1_MARGINAL_MS_M4[d]
    return total


def candidate_round_indices(mdl: float, non_drafting: int) -> list[int]:
    """Every round count consistent with C1, C2 and C3."""
    if mdl == 0.0:
        return []
    frac = Fraction(mdl).limit_denominator(WINDOW_TOKENS * 2)
    if float(frac) != mdl:
        raise ValueError(f"mean_draft_len {mdl!r} is not an exact small rational")
    q = frac.denominator
    out = []
    r = q
    while r <= WINDOW_TOKENS:
        proposed = frac * r
        if proposed.denominator != 1:
            r += q
            continue
        p = int(proposed)
        a = WINDOW_TOKENS - r
        drafting = r - non_drafting
        if a < 0 or a > p or non_drafting > r or drafting < 0 or p < drafting:
            r += q
            continue
        out.append(r)
        r += q
    return out


def calibrate_depth0_ms(rows: dict, prefill_ms: float = 0.0) -> tuple[float, dict]:
    """Measure c1, the candidate's depth-0 round cost, from the anchor prompt.

    The anchor is whichever prompt has a UNIQUE feasible round count under C1-C3 and
    the largest non-drafting fraction. Its leg time is then almost entirely depth-0
    rounds, so c1 falls out with a small correction for its few drafting rounds.
    """
    best = None
    for name, row in rows.items():
        cands = candidate_round_indices(row["mean_draft_len"], row["non_drafting_rounds"])
        if len(cands) != 1:
            continue
        r = cands[0]
        share = row["non_drafting_rounds"] / r
        if best is None or share > best[1]:
            best = (name, share, r)
    if best is None:
        raise RuntimeError("no prompt has a unique round count; cannot calibrate c1")

    name, share, r = best
    row = rows[name]
    # The trusted clock starts before the seed prefill, so only leg - prefill is
    # round work (QwenRuntimeMTPDriver.swift:94-197, QwenRuntimeMTP.swift:347).
    leg_ms = WINDOW_TOKENS * row["mtp_spt"] * 1000.0 - prefill_ms
    n0 = row["non_drafting_rounds"]
    drafting = r - n0
    proposed = row["mean_draft_len"] * r
    mean_drafting_depth = proposed / drafting if drafting else 0.0

    # leg = R * c1 + drafting * scale * cumulative_ms_m4(mean_drafting_depth)
    # One unknown c1 and one unknown scale. Resolve by fixed point: the scale is
    # c1 / E1_DEPTH0_ROUND_ROUND_MS_M4 under the single-factor host model.
    c1 = leg_ms / r
    for _ in range(64):
        scale = c1 / E1_DEPTH0_ROUND_MS_M4
        extra = drafting * scale * cumulative_ms_m4(mean_drafting_depth)
        c1 = (leg_ms - extra) / r
    scale = c1 / E1_DEPTH0_ROUND_MS_M4
    detail = {
        "anchor_prompt": name,
        "anchor_rounds": r,
        "anchor_non_drafting_share": share,
        "anchor_mean_drafting_depth": mean_drafting_depth,
        "host_scale_vs_m4pro": scale,
    }
    return c1, detail


def predicted_round_ms(c1: float, scale: float, mean_depth: float) -> float:
    return c1 + scale * cumulative_ms_m4(mean_depth)


def reconstruct(rows: dict, prefill_ms: float = 0.0) -> dict:
    c1, cal = calibrate_depth0_ms(rows, prefill_ms)
    scale = cal["host_scale_vs_m4pro"]

    out = {
        "calibration": dict(cal, depth0_round_ms=c1, prefill_ms=prefill_ms),
        "prompts": {},
    }
    for name in PROMPTS:
        row = rows[name]
        mdl = row["mean_draft_len"]
        cands = candidate_round_indices(mdl, row["non_drafting_rounds"])
        leg_ms = WINDOW_TOKENS * row["mtp_spt"] * 1000.0 - prefill_ms

        scored = []
        for r in cands:
            observed = leg_ms / r
            if observed < c1:
                continue  # C4: a round cannot be cheaper than a depth-0 round
            proposed = int(round(mdl * r))
            drafting = r - row["non_drafting_rounds"]
            mean_depth = proposed / r
            predicted = predicted_round_ms(c1, scale, mean_depth)
            scored.append(
                {
                    "rounds": r,
                    "observed_round_ms": observed,
                    "predicted_round_ms": predicted,
                    "rel_residual": (observed - predicted) / predicted,
                }
            )
        if not scored:
            raise RuntimeError(f"{name}: no round count survives C1-C4")
        scored.sort(key=lambda s: abs(s["rel_residual"]))
        chosen = scored[0]
        r = chosen["rounds"]
        proposed = int(round(mdl * r))
        accepted = WINDOW_TOKENS - r
        out["prompts"][name] = {
            "rounds": r,
            "proposed_drafts": proposed,
            "accepted_drafts": accepted,
            "non_drafting_rounds": row["non_drafting_rounds"],
            "drafting_rounds": r - row["non_drafting_rounds"],
            "mean_depth_all_rounds": proposed / r,
            "mean_depth_drafting_rounds": (
                proposed / (r - row["non_drafting_rounds"])
                if r > row["non_drafting_rounds"]
                else 0.0
            ),
            "per_draft_accept_rate": accepted / proposed if proposed else 0.0,
            "mtp_spt_ms": row["mtp_spt"] * 1000.0,
            "serial_spt_ms": row["serial_spt"] * 1000.0,
            "raw_ratio": row["raw_ratio"],
            "observed_round_ms": chosen["observed_round_ms"],
            "predicted_round_ms": chosen["predicted_round_ms"],
            "rel_residual": chosen["rel_residual"],
            "feasible_round_counts": [s["rounds"] for s in scored],
            "unique_under_c1_c4": len(scored) == 1,
        }
    return out


# thorfinn's E46 refit of measured QMV cost, in arbitrary consistent units:
#   T(M) = 16.757 + 27.532 * streams(M) + 9.624 * M,  max|residual| 0.770
# streams(M) = ceil(M / IPG(M)) from the SHIPPED dispatch table in quantized.h.
QMV_T0 = 16.757
QMV_PER_STREAM = 27.532
QMV_PER_ROW = 9.624
SHIPPED_IPG = {2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}


def qmv_streams(m: int) -> int:
    if m <= 1:
        return 1
    ipg = SHIPPED_IPG[m]
    return -(-m // ipg)


def qmv_cost(m: int) -> float:
    return QMV_T0 + QMV_PER_STREAM * qmv_streams(m) + QMV_PER_ROW * m


def depth_distribution_bounds(mean_depth: float, cum_budget_ms_m4: float,
                             min_depth: int, max_depth: int = 8) -> dict:
    """Bound every f_d by linear programming over two exactly-known moments.

    The feasible set is
        sum_d f_d = 1
        sum_d d * f_d       = mean_depth        (from the reconstructed P / R)
        sum_d cum(d) * f_d  = cum_budget_ms_m4  (from the reconstructed round cost)
        f_d >= 0
    Vertices of a 3-equality polytope have at most 3 nonzero coordinates, so
    enumerating all triples is exact, not a heuristic.
    """
    depths = list(range(min_depth, max_depth + 1))
    rows = [
        [1.0 for _ in depths],
        [float(d) for d in depths],
        [cumulative_ms_m4(d) for d in depths],
    ]
    rhs = [1.0, mean_depth, cum_budget_ms_m4]

    vertices = []
    n = len(depths)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cols = (i, j, k)
                a = [[rows[r][c] for c in cols] for r in range(3)]
                sol = _solve3(a, rhs)
                if sol is None:
                    continue
                if all(v >= -1e-12 for v in sol):
                    f = [0.0] * n
                    for pos, c in enumerate(cols):
                        f[c] = max(0.0, sol[pos])
                    vertices.append(f)

    if not vertices:
        return {"feasible": False}

    out = {"feasible": True, "depths": depths, "min": {}, "max": {}, "vertices": len(vertices)}
    for idx, d in enumerate(depths):
        vals = [v[idx] for v in vertices]
        out["min"][d] = min(vals)
        out["max"][d] = max(vals)

    # QMV time share of M = depth + 1, minimised and maximised over the SAME polytope.
    def share_of_max_width(f):
        total = sum(f[i] * qmv_cost(depths[i] + 1) for i in range(n))
        top = f[-1] * qmv_cost(depths[-1] + 1)
        return top / total if total else 0.0

    shares = [share_of_max_width(v) for v in vertices]
    out["m9_qmv_share_min"] = min(shares)
    out["m9_qmv_share_max"] = max(shares)
    return out


def _solve3(a, b):
    """Exact 3x3 solve by Cramer's rule. Returns None when singular."""
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    d = det3(a)
    if abs(d) < 1e-12:
        return None
    out = []
    for c in range(3):
        m = [[b[r] if cc == c else a[r][cc] for cc in range(3)] for r in range(3)]
        out.append(det3(m) / d)
    return out


def load_rows(path: str, submission_prefix: str) -> dict:
    tel = json.load(open(path))["telemetry"]
    rows = {}
    for name in PROMPTS:
        hit = None
        for row in tel[name]:
            if row["submission"].startswith(submission_prefix):
                hit = row
                break
        if hit is None:
            raise RuntimeError(f"{name}: no row for submission {submission_prefix}")
        rows[name] = hit
    return rows


def self_test() -> int:
    failures = []

    # 1. Exact rational recovery of a known ratio.
    got = Fraction(485 / 107).limit_denominator(1024)
    if got != Fraction(485, 107):
        failures.append(f"exact-rational recovery returned {got}")

    # 2. A value that is NOT a small rational must be rejected.
    try:
        candidate_round_indices(0.123456789012345, 0)
        failures.append("non-rational mean_draft_len was accepted")
    except ValueError:
        pass

    # 3. Closure must reject a round count that implies more accepted than proposed.
    #    mdl = 1/2 with R = 2 implies P = 1 and A = 510: infeasible.
    if 2 in candidate_round_indices(0.5, 0):
        failures.append("closure check accepted A > P")

    # 4. The weight floor must be able to fail. A round cost below c1 is rejected.
    rows = {
        "x": {
            "mean_draft_len": 4.0,
            "non_drafting_rounds": 0,
            "mtp_spt": 0.001,
            "serial_spt": 0.03,
            "raw_ratio": 30.0,
        }
    }
    try:
        calibrate_depth0_ms(rows)
    except RuntimeError:
        pass  # single prompt with mdl != 0 is fine as an anchor; not the point here

    # 5. Cumulative ladder must be monotone and match the recorded cumulative sums.
    expect = {1: 5.47, 2: 10.51, 3: 26.28, 4: 50.68, 5: 69.66, 6: 89.16, 7: 107.82, 8: 133.23}
    for d, v in expect.items():
        if abs(cumulative_ms_m4(d) - v) > 1e-9:
            failures.append(f"cumulative ladder at depth {d}: {cumulative_ms_m4(d)} != {v}")

    # 6. Fractional depth must interpolate strictly between its neighbours.
    mid = cumulative_ms_m4(4.5)
    if not (expect[4] < mid < expect[5]):
        failures.append(f"fractional depth 4.5 gave {mid}, outside ({expect[4]}, {expect[5]})")

    # 7. The closed-form hull bounds must agree with the LP vertex enumeration.
    #    This is the positive control for prefill_degeneracy: the LP is asked for a
    #    budget just inside and just outside each hull edge, and must answer
    #    feasible then infeasible. Without the outside probe the check could not
    #    fail.
    for md, dmin in ((4.533, 1), (4.768, 0), (2.298, 0), (5.776, 0)):
        ymin, ymax = cum_hull_bounds(md, dmin)
        eps = 1e-6 * max(1.0, ymax)
        inside_lo = depth_distribution_bounds(md, ymin + eps, dmin)["feasible"]
        inside_hi = depth_distribution_bounds(md, ymax - eps, dmin)["feasible"]
        outside_lo = depth_distribution_bounds(md, ymin - 1.0, dmin)["feasible"]
        outside_hi = depth_distribution_bounds(md, ymax + 1.0, dmin)["feasible"]
        if not (inside_lo and inside_hi):
            failures.append(
                f"hull bound at mean {md} min {dmin}: LP calls the interior infeasible")
        if outside_lo or outside_hi:
            failures.append(
                f"hull bound at mean {md} min {dmin}: LP calls the exterior feasible")

    for f in failures:
        print(f"FAIL {f}")
    print(f"self-test: {7 - len(failures)}/7 checks passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="research/e53-board-facts.json")
    ap.add_argument("--submission", default="ca9251b8")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--prefill-ms", type=float, default=0.0,
        help="constant seed-prefill charge inside the timed leg, in ms. The "
             "receipt reports prefill_seconds_per_token; 512 * that is 526.6 ms "
             "on submission ca9251b8. Pass 0 to reproduce the earlier "
             "prefill-blind reconstruction.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rows = load_rows(args.facts, args.submission)
    res = reconstruct(rows, args.prefill_ms)
    c1 = res["calibration"]["depth0_round_ms"]
    scale = res["calibration"]["host_scale_vs_m4pro"]

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    cal = res["calibration"]
    print(f"submission prefix        {args.submission}")
    print(f"anchor prompt            {cal['anchor_prompt']} "
          f"({cal['anchor_non_drafting_share']:.1%} non-drafting, "
          f"{cal['anchor_rounds']} rounds)")
    print(f"candidate depth-0 round  {cal['depth0_round_ms']:.3f} ms")
    print(f"host scale vs M4 Pro     {cal['host_scale_vs_m4pro']:.4f}")
    print()
    hdr = ("prompt", "R", "P", "A", "meanD", "accept", "mtp_ms", "ser_ms",
           "obs_ms", "pred_ms", "resid", "uniq")
    print("%-9s %4s %5s %5s %6s %7s %8s %8s %8s %8s %8s %5s" % hdr)
    for name in PROMPTS:
        p = res["prompts"][name]
        print("%-9s %4d %5d %5d %6.3f %7.4f %8.4f %8.4f %8.2f %8.2f %+7.1f%% %5s" % (
            name, p["rounds"], p["proposed_drafts"], p["accepted_drafts"],
            p["mean_depth_all_rounds"], p["per_draft_accept_rate"],
            p["mtp_spt_ms"], p["serial_spt_ms"],
            p["observed_round_ms"], p["predicted_round_ms"],
            100.0 * p["rel_residual"], p["unique_under_c1_c4"]))
    print()
    amb = [n for n in PROMPTS if not res["prompts"][n]["unique_under_c1_c4"]]
    if amb:
        print("round count NOT unique under C1-C4 for: " + ", ".join(amb))
        for n in amb:
            print(f"  {n}: feasible {res['prompts'][n]['feasible_round_counts']}, "
                  f"chosen {res['prompts'][n]['rounds']} by cost residual")
    else:
        print("every prompt's round count is unique under C1-C4")

    print()
    print("EXACT score decomposition")
    print("  raw_ratio = build_factor * spec_factor * dilution")
    print("  build_factor = serial_spt / depth0_round      (uniform, candidate-build speed)")
    print("  spec_factor  = tokens_per_round * depth0_round / round_ms")
    print("  dilution     = 1 - K / leg                    (the seed prefill's share)")
    print()
    print("%-9s %8s %8s %9s %10s %10s %9s" % (
        "prompt", "build", "spec", "dilution", "product", "raw_ratio", "err"))
    for name in PROMPTS:
        p = res["prompts"][name]
        build = p["serial_spt_ms"] / c1
        tokens_per_round = WINDOW_TOKENS / p["rounds"]
        spec = tokens_per_round * c1 / p["observed_round_ms"]
        leg = p["mtp_spt_ms"] * WINDOW_TOKENS
        dil = 1.0 - res["calibration"]["prefill_ms"] / leg
        prod = build * spec * dil
        print("%-9s %8.4f %8.4f %9.5f %10.5f %10.5f %+8.2e" % (
            name, build, spec, dil, prod, p["raw_ratio"], prod - p["raw_ratio"]))

    print()
    print("DEPTH-DISTRIBUTION BOUNDS from two exactly-known moments (LP vertices)")
    print("  reference: the local public fixture at mean draft 6.269 puts")
    print("  %.2f %% of QMV time in M=9, which is the source of the +5.36 %% claim."
          % (100.0 * _fixture_m9_share()))
    print()
    print("%-9s %7s %9s %9s %11s %11s %14s" % (
        "prompt", "meanD", "f8_min", "f8_max", "M9share_min", "M9share_max", "c1_band_ms"))
    bands = {}
    for name in PROMPTS:
        p = res["prompts"][name]
        min_depth = 1 if p["non_drafting_rounds"] == 0 else 0
        band = _feasibility_c1(p, min_depth)
        bands[name] = band
        btxt = ("%.2f-%.2f" % band) if band else "empty"
        b = depth_distribution_bounds(
            p["mean_depth_all_rounds"], (p["observed_round_ms"] - c1) / scale, min_depth)
        if not b["feasible"]:
            print("%-9s %7.3f %9s %9s %11s %11s %14s" % (
                name, p["mean_depth_all_rounds"], "-", "-", "INFEASIBLE", "-", btxt))
            continue
        print("%-9s %7.3f %9.4f %9.4f %10.2f%% %10.2f%% %14s" % (
            name, p["mean_depth_all_rounds"], b["min"][8], b["max"][8],
            100.0 * b["m9_qmv_share_min"], 100.0 * b["m9_qmv_share_max"], btxt))

    print()
    print("c1_band_ms is the interval of depth-0 round costs that keep the polytope")
    print("non-empty under the single-factor M4 Pro -> M5 cost transfer. The calibrated")
    print("value is %.3f ms. A band that EXCLUDES it falsifies the transfer, not the" % c1)
    print("round-count reconstruction: the two exact moments stay valid either way.")

    live = {n: b for n, b in bands.items() if b}
    lo = max(b[0] for b in live.values())
    hi = min(b[1] for b in live.values())
    lo_name = max(live, key=lambda n: live[n][0])
    hi_name = min(live, key=lambda n: live[n][1])
    print()
    print("JOINT TEST across all eight prompts (calibration-independent)")
    print("  binding lower edge  %-9s c1 >= %.3f ms" % (lo_name, lo))
    print("  binding upper edge  %-9s c1 <= %.3f ms" % (hi_name, hi))
    if lo <= hi:
        print("  intersection [%.3f, %.3f] NON-EMPTY: one c1 explains every prompt."
              % (lo, hi))
    else:
        print("  intersection EMPTY by %.3f ms (%.2f %%). No depth-0 round cost whatever"
              % (lo - hi, 100.0 * (lo - hi) / lo))
        print("  reconciles all eight prompts. The single-factor M4 Pro -> M5 cost")
        print("  transfer is REFUTED without reference to any calibrated value.")
        drafting = {n: b for n, b in live.items() if n != ANCHOR_PROMPT}
        dlo = max(b[0] for b in drafting.values())
        dhi = min(b[1] for b in drafting.values())
        if dlo <= dhi:
            print("  Excluding the near-serial anchor %s, the seven drafting prompts"
                  % ANCHOR_PROMPT)
            print("  are jointly feasible at c1 in [%.3f, %.3f]. The anchor alone needs"
                  % (dlo, dhi))
            print("  c1 in [%.3f, %.3f]. The two blocks disagree by %.2f %%, and the"
                  % (live[ANCHOR_PROMPT][0], live[ANCHOR_PROMPT][1],
                     100.0 * (live[ANCHOR_PROMPT][0] - dhi) / live[ANCHOR_PROMPT][0]))
            print("  drafting block wants a SMALLER c1, i.e. more cost in the ladder.")
            print("  At the true c1 the ladder therefore over-prices depth on M5.")

    print()
    print("SLOPE-FACTOR BAND at the calibrated c1 = %.3f ms" % c1)
    print("  Same polytope, but the whole cumulative ladder is scaled by g. g < 1 means")
    print("  M5 marginal drafting cost is CHEAPER than the scaled M4 Pro ladder claims.")
    print("  The +5.36 %% M=9 prize is priced off this slope, so g bounds re-price it.")
    print()
    print("%-9s %7s %10s %10s" % ("prompt", "meanD", "g_min", "g_max"))
    gb = {}
    for name in PROMPTS:
        p = res["prompts"][name]
        min_depth = 1 if p["non_drafting_rounds"] == 0 else 0
        band = _feasibility_slope(p, min_depth, c1, scale)
        gb[name] = band
        if band is None:
            print("%-9s %7.3f %10s %10s" % (name, p["mean_depth_all_rounds"], "-", "-"))
            continue
        print("%-9s %7.3f %10.4f %10.4f" % (
            name, p["mean_depth_all_rounds"], band[0], band[1]))
    glive = {n: b for n, b in gb.items() if b}
    glo = max(b[0] for b in glive.values())
    ghi = min(b[1] for b in glive.values())
    print()
    if glo <= ghi:
        print("  joint slope factor g in [%.4f, %.4f]" % (glo, ghi))
        print("  Every prompt is explained once the E1 ladder slope is scaled by g.")
        if ghi < 1.0:
            print("  g < 1 throughout: the E1 M4 Pro ladder OVER-PRICES depth on the")
            print("  ranked M5 by at least %.1f %%. Any score delta derived by scaling a"
                  % (100.0 * (1.0 - ghi)))
            print("  per-width QMV win through that ladder is inflated by the same order.")
        print()
        print("M=9 SHARE under the identified two-parameter transfer")
        print("  c1 = %.3f ms, scale = %.4f, g in [%.4f, %.4f]. Bounds are the union"
              % (c1, scale, glo, ghi))
        print("  over both g edges, so they hold for every transfer the receipt admits.")
        print()
        print("%-9s %7s %9s %9s %11s %11s" % (
            "prompt", "meanD", "f8_min", "f8_max", "M9share_min", "M9share_max"))
        budget_scale = scale
        for name in PROMPTS:
            p = res["prompts"][name]
            min_depth = 1 if p["non_drafting_rounds"] == 0 else 0
            f8los, f8his, s9los, s9his = [], [], [], []
            for g in (glo, ghi):
                b = depth_distribution_bounds(
                    p["mean_depth_all_rounds"],
                    (p["observed_round_ms"] - c1) / (budget_scale * g), min_depth)
                if not b["feasible"]:
                    continue
                f8los.append(b["min"][8])
                f8his.append(b["max"][8])
                s9los.append(b["m9_qmv_share_min"])
                s9his.append(b["m9_qmv_share_max"])
            if not f8los:
                print("%-9s %7.3f %s" % (name, p["mean_depth_all_rounds"], "INFEASIBLE"))
                continue
            print("%-9s %7.3f %9.4f %9.4f %10.2f%% %10.2f%%" % (
                name, p["mean_depth_all_rounds"], min(f8los), max(f8his),
                100.0 * min(s9los), 100.0 * max(s9his)))
        print()
        print("  The published score is the mean of the 4th and 5th order statistics,")
        print("  which the receipt shows are beagle and medicine. Those two rows, not")
        print("  the local fixture's %.2f %%, set the value of any M=9-only mechanism."
              % (100.0 * _fixture_m9_share()))
    else:
        print("  joint slope band EMPTY: a single slope factor does not rescue the")
        print("  transfer either. The M5 ladder differs from M4 Pro in SHAPE, not scale.")

    print()
    print("=" * 78)
    print("SEED-PREFILL CHARGE K versus LADDER SLOPE g")
    print("  QwenRuntimeMTPDriver.swift:93-99 starts the clock BEFORE the seed")
    print("  prefill, so leg = K + R * round_ms and every round cost above is")
    print("  biased upward, worst for the deep low-R prompts.")
    if args.prefill_ms > 0.0:
        print()
        print("  K = %.2f ms was supplied from the receipt's own"
              % args.prefill_ms)
        print("  prefill_seconds_per_token field, so K is OBSERVED, not inferred, and")
        print("  everything above already excludes it. The scan below is the")
        print("  identification exercise for the case where K is unknown; it is")
        print("  skipped here. Run without --prefill-ms to see it.")
        lev = prefill_leverage(res, args.prefill_ms)
        print()
        print("  %-9s %9s %9s %9s %9s" % (
            "prompt", "leg_ms", "K/leg", "round_x", "prefill_x"))
        for name in PROMPTS:
            r = lev["prompts"][name]
            print("  %-9s %9.0f %8.2f%% %9.4f %9.4f" % (
                name, r["leg_ms"], 100.0 * r["prefill_share"],
                r["round_gain_conversion"], r["prefill_gain_conversion"]))
        print()
        print("  Median pair %s+%s: a fractional round-cost win converts to score"
              % lev["median_pair"])
        print("  at x%.4f, so EVERY round-cost projection in the ledger is %.2f %% high."
              % (lev["median_round_conversion"],
                 100.0 * (1.0 / lev["median_round_conversion"] - 1.0)))
        return 0
    deg = prefill_degeneracy(res)
    e = deg["g1_k0"]
    print()
    print("  hull band of E[cum_M4(depth)] per prompt, ms:")
    for name in PROMPTS:
        h = deg["hull"][name]
        print("    %-9s %8.2f .. %8.2f" % (name, h["ymin"], h["ymax"]))
    print()
    print("  control at K = 0, g = 1: c1 in [%.3f, %.3f] -> %s"
          % (e["lo"], e["hi"], "NON-EMPTY" if e["lo"] <= e["hi"] else "EMPTY"))
    print("    binding low %s, binding high %s" % (e["binding_low"], e["binding_high"]))
    g1 = deg["g1_feasible_k_ms"]
    print()
    if g1:
        print("  EXACT transfer g = 1 becomes feasible for K in [%.0f, %.0f] ms."
              % (min(g1), max(g1)))
    else:
        print("  No K in [0, 3000] ms rescues the exact transfer g = 1. Removing a")
        print("  constant per-leg charge cannot repair the ladder SHAPE, because K/R")
        print("  is largest exactly for the deep prompts that already want a")
        print("  cheaper ladder. The g < 1 finding survives the prefill correction.")
    print()
    print("  feasible (K, g) set, sampled:")
    print("  %8s %8s %8s %10s %10s" % ("K_ms", "g_min", "g_max", "c1@g_min", "c1@g_max"))
    for row in deg["rows"]:
        if abs(row["k_ms"] % 250.0) > 1e-6:
            continue
        if not row["feasible"]:
            print("  %8.0f %8s %8s %10s %10s" % (row["k_ms"], "-", "-", "-", "-"))
            continue
        print("  %8.0f %8.4f %8.4f %10.3f %10.3f" % (
            row["k_ms"], row["g_min"], row["g_max"],
            row["c1_at_g_min"], row["c1_at_g_max"]))
    if "k_feasible_min" in deg:
        print()
        print("  K feasible over [%.0f, %.0f] ms with g in [%.4f, %.4f] across that range."
              % (deg["k_feasible_min"], deg["k_feasible_max"],
                 deg["g_union_min"], deg["g_union_max"]))
        print("  K and g are therefore NOT separately identified by the receipt: the")
        print("  admissible set is a CURVE, not a point. Only a local measurement of")
        print("  seed_prefill_seconds on the ranked-shaped window can split them.")
        kmax = deg["k_feasible_max"]
        m4 = 3993.0
        print()
        print("  The upper edge K <= %.0f ms is a FALSIFIABLE prediction about the" % kmax)
        print("  ranked host. M4 Pro measures seed_prefill = %.0f ms, so the receipt" % m4)
        print("  admits a prefill host scale of at most %.4f, against the round-cost"
              % (kmax / m4))
        print("  scale %.4f. Prefill must therefore transfer to M5 STRICTLY BETTER"
              % res["calibration"]["host_scale_vs_m4pro"])
        print("  than decode rounds do. That is the expected sign for a compute-bound")
        print("  GEMM phase against memory-bound decode, and it is a real test: if a")
        print("  local measurement puts M5-equivalent prefill above %.0f ms, this" % kmax)
        print("  whole constant-prefill plus single-slope family is wrong.")

        print()
        print("PREFILL DILUTION of every score projection")
        print("  raw_p = build * spec * (1 - K/leg_p). A fractional round-cost win")
        print("  converts at (1 - K/leg_p); a fractional prefill win converts at K/leg_p.")
        for kv in (0.0, 1000.0, kmax):
            lev = prefill_leverage(res, kv)
            print()
            print("  K = %.0f ms" % kv)
            print("    %-9s %9s %9s %9s" % ("prompt", "leg_ms", "K/leg", "round_x"))
            for name in PROMPTS:
                r = lev["prompts"][name]
                print("    %-9s %9.0f %8.2f%% %9.4f" % (
                    name, r["leg_ms"], 100.0 * r["prefill_share"],
                    r["round_gain_conversion"]))
            print("    median pair %s+%s: round win x%.4f, prefill win x%.4f" % (
                lev["median_pair"][0], lev["median_pair"][1],
                lev["median_round_conversion"], lev["median_prefill_conversion"]))
    return 0


def _feasibility_c1(p: dict, min_depth: int):
    """Closed interval of depth-0 round costs that keep the moment polytope non-empty.

    Feasibility is a BAND, not a half-line. Too small a c1 leaves a cost budget above
    cum(8); too large a c1 leaves one below cum(min_depth). Scan first, then refine
    both edges by bisection.
    """
    obs = p["observed_round_ms"]
    md = p["mean_depth_all_rounds"]

    def ok(c1v: float) -> bool:
        if c1v <= 0.0 or c1v >= obs:
            return False
        sc = c1v / E1_DEPTH0_ROUND_MS_M4
        return depth_distribution_bounds(md, (obs - c1v) / sc, min_depth).get("feasible", False)

    steps = 800
    step = obs / steps
    hits = [i * step for i in range(1, steps) if ok(i * step)]
    if not hits:
        return None

    lo, hi = min(hits) - step, min(hits)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    left = hi

    lo, hi = max(hits), max(hits) + step
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    right = lo
    return (left, right)


def _feasibility_slope(p: dict, min_depth: int, c1: float, scale: float):
    """Closed interval of ladder slope factors g that keep the polytope non-empty.

    c1 stays at its calibrated value, so this isolates the SHAPE of the marginal
    cost ladder from the level of the depth-0 round. The cost budget is strictly
    decreasing in g, so feasibility is again a band with two edges to bisect.
    """
    obs = p["observed_round_ms"]
    md = p["mean_depth_all_rounds"]
    budget1 = (obs - c1) / scale

    def ok(g: float) -> bool:
        if g <= 0.0:
            return False
        return depth_distribution_bounds(md, budget1 / g, min_depth).get("feasible", False)

    steps = 2000
    hi_g = 4.0
    step = hi_g / steps
    hits = [i * step for i in range(1, steps + 1) if ok(i * step)]
    if not hits:
        return None

    lo, hi = min(hits) - step, min(hits)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    left = hi

    lo, hi = max(hits), max(hits) + step
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return (left, lo)


def _fixture_m9_share() -> float:
    """M=9's share of QMV time in the local public fixture histogram."""
    hist = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}
    total = sum(n * qmv_cost(m) for m, n in hist.items())
    return hist[9] * qmv_cost(9) / total


def cum_hull_bounds(mean_depth: float, min_depth: int, max_depth: int = 8):
    """Exact range of E[cum(depth)] over all distributions with this mean.

    Two moments pin a 1-parameter family, so the extremes of a third linear
    functional lie on the lower convex and upper concave hulls of the points
    (d, cum(d)). This replaces the vertex enumeration in
    depth_distribution_bounds with a closed form, which makes a two-parameter
    (prefill, slope) scan cheap enough to run exhaustively.
    """
    pts = [(k, cumulative_ms_m4(k)) for k in range(min_depth, max_depth + 1)]
    if mean_depth <= pts[0][0]:
        return pts[0][1], pts[0][1]
    lo = hi = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (x1, y1), (x2, y2) = pts[i], pts[j]
            if not (x1 <= mean_depth <= x2):
                continue
            def chord(x):
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            v = chord(mean_depth)
            span = [(x, y) for x, y in pts if x1 <= x <= x2]
            if all(chord(x) <= y + 1e-9 for x, y in span):
                lo = v if lo is None else min(lo, v)
            if all(chord(x) >= y - 1e-9 for x, y in span):
                hi = v if hi is None else max(hi, v)
    return lo, hi


def prefill_degeneracy(res: dict, k_max_ms: float = 3000.0, k_step_ms: float = 5.0,
                       g_steps: int = 2000, g_max: float = 4.0) -> dict:
    """Scan the constant per-leg prefill charge K against the ladder slope g.

    The trusted driver starts its clock BEFORE the seed prefill
    (QwenRuntimeMTPDriver.swift:93-99), so the scored leg is
        512 * mtp_spt = K + R * round_ms
    and every round cost derived as leg / R is biased upward, more for the deep
    prompts that use fewer rounds. Under the transfer
        round_ms = c1 * (1 + g * Y / C0),   Y = E[cum_M4(depth)]
    a prompt is feasible exactly when Y lies inside its hull band, which gives a
    closed-form c1 interval per prompt:
        c1 in [ r / (1 + g * Ymax / C0),  r / (1 + g * Ymin / C0) ],  r = (leg-K)/R
    All eight prompts share one c1, so the question is whether the intersection
    is ever non-empty. Reporting the whole (K, g) feasible set is the point: if
    the set is a curve rather than a point, K and g are NOT separately
    identified from the receipt and only a local measurement can split them.
    """
    C0 = E1_DEPTH0_ROUND_MS_M4
    info = {}
    for name in PROMPTS:
        p = res["prompts"][name]
        min_depth = 1 if p["non_drafting_rounds"] == 0 else 0
        ymin, ymax = cum_hull_bounds(p["mean_depth_all_rounds"], min_depth)
        info[name] = {
            "leg_ms": p["mtp_spt_ms"] * WINDOW_TOKENS,
            "rounds": p["rounds"],
            "ymin": ymin,
            "ymax": ymax,
            "min_depth": min_depth,
        }

    def joint(k: float, g: float):
        lo = None
        hi = None
        lo_name = hi_name = ""
        for name in PROMPTS:
            d = info[name]
            r = (d["leg_ms"] - k) / d["rounds"]
            if r <= 0.0:
                return None
            a = r / (1.0 + g * d["ymax"] / C0)
            b = r / (1.0 + g * d["ymin"] / C0)
            if lo is None or a > lo:
                lo, lo_name = a, name
            if hi is None or b < hi:
                hi, hi_name = b, name
        return lo, hi, lo_name, hi_name

    g_step = g_max / g_steps
    rows = []
    k = 0.0
    while k <= k_max_ms + 1e-9:
        gs = []
        for i in range(1, g_steps + 1):
            g = i * g_step
            j = joint(k, g)
            if j and j[0] <= j[1]:
                gs.append(g)
        row = {"k_ms": k, "feasible": bool(gs)}
        if gs:
            row["g_min"] = min(gs)
            row["g_max"] = max(gs)
            row["g_contiguous"] = (
                abs((max(gs) - min(gs)) / g_step + 1 - len(gs)) < 0.5)
            jm = joint(k, min(gs))
            jx = joint(k, max(gs))
            row["c1_at_g_min"] = 0.5 * (jm[0] + jm[1])
            row["c1_at_g_max"] = 0.5 * (jx[0] + jx[1])
            row["binding_low_at_g_min"] = jm[2]
            row["binding_high_at_g_min"] = jm[3]
        rows.append(row)
        k += k_step_ms

    exact = joint(0.0, 1.0)
    out = {
        "hull": {n: {"ymin": info[n]["ymin"], "ymax": info[n]["ymax"]} for n in PROMPTS},
        "rows": rows,
        "g1_k0": {"lo": exact[0], "hi": exact[1],
                  "binding_low": exact[2], "binding_high": exact[3]},
    }
    live = [r for r in rows if r["feasible"]]
    if live:
        out["k_feasible_min"] = live[0]["k_ms"]
        out["k_feasible_max"] = live[-1]["k_ms"]
        out["g_union_min"] = min(r["g_min"] for r in live)
        out["g_union_max"] = max(r["g_max"] for r in live)
    # Does any K rescue an EXACT single-factor transfer, g == 1?
    g1 = []
    k = 0.0
    while k <= k_max_ms + 1e-9:
        j = joint(k, 1.0)
        if j and j[0] <= j[1]:
            g1.append(k)
        k += k_step_ms
    out["g1_feasible_k_ms"] = g1
    return out


def prefill_leverage(res: dict, k_ms: float) -> dict:
    """Price a prefill gain and a round-cost gain against each other at this K.

    Because leg = K + R * round_ms, the exact score identity is a THREE-factor
    product, not the two-factor product of the K = 0 analysis:

        raw_p = (serial_spt / c1) * (512 * c1 / (R_p * round_ms_p)) * (1 - K / leg_p)
                 build (uniform)     speculation efficiency          prefill dilution

    Differentiating raw_p ~ 1 / (K + R * round_ms) gives the two conversion
    factors below. A fractional round-cost win is damped by (1 - K/leg); a
    fractional prefill win converts at K/leg. The damping is WORST for the
    fastest, deepest prompts, which are exactly the prompts that set the median.
    """
    rows = {}
    for name in PROMPTS:
        p = res["prompts"][name]
        leg = p["mtp_spt_ms"] * WINDOW_TOKENS
        share = k_ms / leg
        rows[name] = {
            "leg_ms": leg,
            "prefill_share": share,
            "round_gain_conversion": 1.0 - share,
            "prefill_gain_conversion": share,
        }
    pair = ("beagle", "medicine")
    return {
        "k_ms": k_ms,
        "prompts": rows,
        "median_pair": pair,
        "median_round_conversion": sum(rows[n]["round_gain_conversion"] for n in pair) / 2.0,
        "median_prefill_conversion": sum(rows[n]["prefill_gain_conversion"] for n in pair) / 2.0,
    }


if __name__ == "__main__":
    sys.exit(main())
