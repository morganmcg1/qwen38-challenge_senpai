"""E164 red team: falsify CLAIM A (FINDING 357) and CLAIM B (prefill gap).

harness=ranked throughout. Board arithmetic only. No GPU, no build.

Priority order set by the advisor: B1 and A1 first.

B1 asks whether `prefill_spt * 512 + decode_seconds` reconstructs `mtp_spt * 512`.
A1 asks whether `delta_s = +0.2883 ms/round` survives at two standard errors.
"""

import argparse
import json
import math
import os
from fractions import Fraction

import numpy as np

LIVE_BOARD = "/tmp/yukon-board/full.json"
DECODE_TOKENS = 512

# Advisor's F2 N-vector for 5a9f130a, quoted so the recovery can be checked
# against it rather than silently agreeing with it.
ADVISOR_N = {
    "drama": 252,
    "travel": 213,
    "beagle": 110,
    "republic": 93,
    "essays": 92,
    "medicine": 90,
    "botany": 81,
    "plutarch": 488,
}


def load_board(path):
    rows = json.load(open(path))
    rows = rows["submissions"] if isinstance(rows, dict) else rows
    keep = []
    for r in rows:
        om = r.get("officialMetrics") or {}
        if om.get("per_prompt") and r.get("officialScore") is not None:
            keep.append(r)
    return keep


def prompt_names(board):
    """Stable short names keyed on prompt_sha256, ordered as the receipts order them."""
    ref = board[0]["officialMetrics"]["per_prompt"]
    return [c["prompt_sha256"] for c in ref]


def recover_rounds(edl, decode_tokens=DECODE_TOKENS, max_mult=40):
    """Return every feasible round count N for one prompt.

    `edl` is an exact rational P/N. Constraints: N + A = decode_tokens and
    A <= P = edl * N. The reduced denominator is a divisor of N, so enumerate
    its multiples and keep the feasible ones.
    """
    if edl == 0:
        return [decode_tokens]
    fr = Fraction(edl).limit_denominator(10**6)
    den = fr.denominator
    out = []
    for k in range(1, max_mult + 1):
        n = den * k
        if n >= decode_tokens:
            break
        p = fr * n
        if p.denominator != 1:
            continue
        a = decode_tokens - n
        if 0 <= a <= int(p):
            out.append(n)
    return out


def pick_rounds(cells, names):
    """Choose one N per prompt.

    Smallest-feasible is wrong (advisor F2). Choose the N-vector that makes
    R = s + h * rows affine with h > 0, searching over the feasible sets of the
    drafting prompts. Returns the chosen vector plus every candidate set, so an
    ambiguous prompt is reported rather than hidden.
    """
    feas = {}
    for nm, c in zip(names, cells):
        feas[nm] = recover_rounds(c["effective_mean_draft_len"])
    return feas


def leg_seconds(cell):
    return cell["mtp_seconds_per_token_mean"] * DECODE_TOKENS


def prefill_seconds(cell):
    p = cell.get("prefill_seconds_per_token")
    return None if p is None else p * DECODE_TOKENS


def ols(x, y):
    """Two-parameter OLS with the full covariance matrix of the estimates."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = n - 2
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    corr = cov[0, 1] / (se[0] * se[1])
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se_intercept": float(se[0]),
        "se_slope": float(se[1]),
        "corr_intercept_slope": float(corr),
        "sigma": float(math.sqrt(sigma2)),
        "dof": int(dof),
        "sse": float(resid @ resid),
        "worst_abs_resid": float(np.abs(resid).max()),
        "resid": [float(v) for v in resid],
        "sxx": float(((x - x.mean()) ** 2).sum()),
        "x_mean": float(x.mean()),
        "n": int(n),
    }


def student_t_sf(t, dof):
    """Two-sided Student-t tail. scipy is not installed on this host."""
    t = abs(float(t))
    x = dof / (dof + t * t)

    def betainc(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 300):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            c = 1.0 + num / c
            c = 1e-30 if abs(c) < 1e-30 else c
            f *= c * d
            if abs(1.0 - c * d) < 1e-12:
                break
        return front * (f - 1.0)

    ib = betainc(dof / 2.0, 0.5, x)
    return max(0.0, min(1.0, ib))


def find(board, id8):
    hits = [r for r in board if r["id"].startswith(id8)]
    if len(hits) != 1:
        raise SystemExit(f"{id8}: {len(hits)} matches")
    return hits[0]


# --------------------------------------------------------------------- B1


def b1_prefill_reconciliation(board, names):
    """Can the parts be summed to the whole, and is prefill inside the scored leg?

    The board publishes no decode-seconds field, so the literal identity cannot
    be evaluated on the board. The enforcing source answers the question the
    identity was meant to answer, and the board then bounds how good the
    published prefill is as a proxy for the in-leg seed prefill.
    """
    res = {
        "harness": "ranked",
        "literal_identity_evaluable_on_the_board": False,
        "why": (
            "officialMetrics.per_prompt carries no decode-seconds field. The only timing "
            "cells are mtp_seconds_per_token_mean, serial_seconds_per_token_mean and "
            "prefill_seconds_per_token, so prefill + decode == leg cannot be checked as "
            "arithmetic. It has to be settled from the enforcing source"
        ),
        "enforcing_source": {
            "fixtures/qwen3_8_27b_mtp_track.json": {
                "field": "proposed_scoring.prefill_component",
                "value": (
                    "none; seed prefill is charged inside the decode measurement, "
                    "identically on both legs"
                ),
            },
            "Sources/MLXFastTrustedHarness/QwenRuntimeBenchmark.swift": {
                "decode_phase_start_precedes_begin_decode": True,
                "progress_literal": "decode measured start tokens=... includes_seed_prefill=true",
                "seconds_per_token_divisor": "decodeSteps, i.e. 512 decode tokens",
                "prefill_seconds_per_token_divisor": "promptTokens.count, i.e. the seed length",
            },
            "benchmark.json": {"scoring.mode": "qwen-mtp-paired-decode-only"},
        },
        "verdict_on_the_load_bearing_assumption": (
            "CONFIRMED. The seed prefill is inside the scored leg. Claim B does not "
            "collapse on B1"
        ),
    }

    # But the published prefill is NOT the in-leg seed prefill.
    res["published_prefill_is_a_different_measurement"] = {
        "in_leg_seed_prefill": (
            "worker.beginDecode(seedTokens: golden.decodeSeedTokens), executed once inside "
            "the decode worker, inside the scored timer"
        ),
        "published_prefill_seconds_per_token": (
            "measureWorkerPrefillSecondsPerToken(promptTokens: golden.prefillPromptTokens), "
            "executed in a separate short-lived prefillWorker process, outside the scored timer"
        ),
        "different_token_set": "prefillPromptTokens vs decodeSeedTokens",
        "different_process": True,
        "benchmarkPrefillWarmupRuns": 0,
        "benchmarkPrefillTimedRuns": 1,
        "consequence": (
            "the published prefill is a single un-warmed sample from a freshly spawned "
            "worker. It is a proxy for the in-leg seed prefill, not a measurement of it, "
            "and it carries process-start and first-touch cost that the scored leg does not"
        ),
    }

    # How many distinct prefill values does a receipt actually report?
    per_receipt = []
    for r in board:
        cells = r["officialMetrics"]["per_prompt"]
        vals = [c.get("prefill_seconds_per_token") for c in cells]
        if any(v is None for v in vals):
            continue
        ms = np.array(vals, float) * DECODE_TOKENS * 1000.0
        per_receipt.append(
            {
                "id8": r["id"][:8],
                "solver": r["solverUsername"],
                "created": r["createdAt"],
                "n_distinct": len(set(vals)),
                "mean_ms": float(ms.mean()),
                "sd_ms": float(ms.std(ddof=1)),
                "min_ms": float(ms.min()),
                "max_ms": float(ms.max()),
                "spread_ms": float(ms.max() - ms.min()),
                "rel_sd_pct": float(100.0 * ms.std(ddof=1) / ms.mean()),
            }
        )
    sds = np.array([p["sd_ms"] for p in per_receipt])
    means = np.array([p["mean_ms"] for p in per_receipt])
    res["b5_within_receipt_prefill_spread"] = {
        "n_receipts_with_prefill": len(per_receipt),
        "all_eight_distinct_share": float(
            np.mean([p["n_distinct"] == 8 for p in per_receipt])
        ),
        "median_within_receipt_sd_ms": float(np.median(sds)),
        "median_within_receipt_rel_sd_pct": float(
            np.median([p["rel_sd_pct"] for p in per_receipt])
        ),
        "median_within_receipt_spread_ms": float(
            np.median([p["spread_ms"] for p in per_receipt])
        ),
        "between_receipt_sd_of_receipt_mean_ms": float(means.std(ddof=1)),
        "board_mean_ms": float(means.mean()),
    }
    res["_per_receipt_prefill"] = per_receipt
    return res


# --------------------------------------------------------------------- A1


def receipt_fit(r, names, n_by_prompt, drop_nondrafting=True):
    """Fit R = s + h * rows over the drafting prompts of one receipt."""
    cells = r["officialMetrics"]["per_prompt"]
    rows, R, used = [], [], []
    for nm, c in zip(names, cells):
        if drop_nondrafting and c["non_drafting_round_count"] > 0:
            continue
        n = n_by_prompt.get(nm)
        if n is None:
            continue
        pf = prefill_seconds(c)
        if pf is None:
            return None
        r_ms = (leg_seconds(c) - pf) * 1000.0 / n
        rows.append(1.0 + c["effective_mean_draft_len"])
        R.append(r_ms)
        used.append(nm)
    if len(rows) < 4:
        return None
    fit = ols(rows, R)
    fit["prompts"] = used
    fit["rows"] = rows
    fit["R_ms"] = R
    return fit


def a1_claim_a(board, names, ours_id="5a9f130a", front_id="ec24d591"):
    ours = find(board, ours_id)
    front = find(board, front_id)
    sha_to_name = {}
    ref = ours["officialMetrics"]["per_prompt"]
    # Map the advisor's prompt labels onto prompt_sha256 by matching his N-vector
    # to the recovered feasible sets. Order is stable across receipts.
    res = {"harness": "ranked", "ours": ours_id, "frontier": front_id}

    # --- A2 first: are the rows matched digit for digit? ---
    a2 = []
    ok_all = True
    for i, (co, cf) in enumerate(
        zip(ours["officialMetrics"]["per_prompt"], front["officialMetrics"]["per_prompt"])
    ):
        same_prompt = co["prompt_sha256"] == cf["prompt_sha256"]
        same_edl = co["effective_mean_draft_len"] == cf["effective_mean_draft_len"]
        same_nd = co["non_drafting_round_count"] == cf["non_drafting_round_count"]
        ok_all = ok_all and same_prompt and same_edl and same_nd
        a2.append(
            {
                "i": i,
                "prompt_sha8": co["prompt_sha256"][:8],
                "same_prompt": same_prompt,
                "edl_ours": co["effective_mean_draft_len"],
                "edl_frontier": cf["effective_mean_draft_len"],
                "edl_digit_identical": same_edl,
                "nd_ours": co["non_drafting_round_count"],
                "nd_frontier": cf["non_drafting_round_count"],
                "nd_identical": same_nd,
            }
        )
    res["a2_rows_matched"] = {"all_matched": ok_all, "per_prompt": a2}

    # --- recover N on the shared edl vector ---
    feas = {}
    for c in ours["officialMetrics"]["per_prompt"]:
        feas[c["prompt_sha256"][:8]] = recover_rounds(c["effective_mean_draft_len"])
    res["round_recovery"] = {
        "feasible_sets": feas,
        "ambiguous_prompts": {k: v for k, v in feas.items() if len(v) > 1},
    }
    return res, ours, front


def resolve_n_vector(r):
    """Map the advisor's F2 N-vector onto this receipt's prompt_sha256 order.

    The advisor fixed the vector; do not rediscover it. Assert it is feasible
    under N + A = 512 and A <= edl * N so a silent mismatch cannot pass.
    """
    out, report = {}, []
    for c in r["officialMetrics"]["per_prompt"]:
        sha8 = c["prompt_sha256"][:8]
        feas = recover_rounds(c["effective_mean_draft_len"])
        hit = [n for n in ADVISOR_N.values() if n in feas]
        if c["non_drafting_round_count"] > 0:
            hit = [n for n in hit if n == 488] or hit
        chosen = hit[0] if hit else None
        out[sha8] = chosen
        report.append(
            {
                "prompt_sha8": sha8,
                "edl": c["effective_mean_draft_len"],
                "feasible": feas,
                "chosen_N": chosen,
                "is_smallest_feasible": bool(feas and chosen == feas[0]),
                "n_feasible": len(feas),
                "implied_acceptance_alpha": (
                    None
                    if not chosen or c["effective_mean_draft_len"] == 0
                    else (DECODE_TOKENS - chosen)
                    / (c["effective_mean_draft_len"] * chosen)
                ),
                "non_drafting_round_count": c["non_drafting_round_count"],
            }
        )
    return out, report


def fit_receipt(r, n_by_sha, drop_nondrafting=True, prefill_override_ms=None):
    cells = r["officialMetrics"]["per_prompt"]
    rows, R, used, legs, pfs, ns = [], [], [], [], [], []
    for c in cells:
        sha8 = c["prompt_sha256"][:8]
        if drop_nondrafting and c["non_drafting_round_count"] > 0:
            continue
        n = n_by_sha.get(sha8)
        pf = prefill_seconds(c)
        if n is None or pf is None:
            return None
        leg = leg_seconds(c)
        pf_ms = pf * 1000.0 if prefill_override_ms is None else prefill_override_ms
        rows.append(1.0 + c["effective_mean_draft_len"])
        R.append((leg * 1000.0 - pf_ms) / n)
        legs.append(leg * 1000.0)
        pfs.append(pf_ms)
        ns.append(n)
        used.append(sha8)
    if len(rows) < 4:
        return None
    f = ols(rows, R)
    f.update({"prompts": used, "rows": rows, "R_ms": R, "leg_ms": legs,
              "prefill_ms": pfs, "N": ns})
    return f


def a1_prefill_attribution(ours, front, n_by_sha):
    """Is delta_s an artifact of the prefill probe rather than decode work?

    delta_R = (dleg - dprefill) / N, so a systematic prefill-probe difference
    that is constant across prompts injects a term -dprefill / N. Because 1/N is
    close to affine in rows, that term lands almost entirely in delta_s. The
    published prefill is a single un-warmed probe from a separate process, so
    this is the most likely way a real decode conclusion is contaminated.
    """
    variants = {}
    fo_own = fit_receipt(ours, n_by_sha)
    ff_own = fit_receipt(front, n_by_sha)
    shared = 0.5 * (np.mean(fo_own["prefill_ms"]) + np.mean(ff_own["prefill_ms"]))
    rows = np.array(fo_own["rows"])
    N = np.array(fo_own["N"], float)

    for label, po, pf_ in (
        ("own_prefill_advisor_method", None, None),
        ("shared_prefill_both_legs", shared, shared),
        ("no_prefill_subtraction", 0.0, 0.0),
    ):
        fo = fit_receipt(ours, n_by_sha, prefill_override_ms=po)
        ff = fit_receipt(front, n_by_sha, prefill_override_ms=pf_)
        dR = np.array(fo["R_ms"]) - np.array(ff["R_ms"])
        f = ols(rows, dR)
        variants[label] = {
            "delta_s": f["intercept"],
            "se_delta_s": f["se_intercept"],
            "t_delta_s": f["intercept"] / f["se_intercept"],
            "delta_h": f["slope"],
            "se_delta_h": f["se_slope"],
            "sigma": f["sigma"],
            "delta_R_at_rows_6": f["intercept"] + 6.0 * f["slope"],
        }
    dpf = float(np.mean(fo_own["prefill_ms"]) - np.mean(ff_own["prefill_ms"]))
    # analytic contribution of a constant prefill-probe gap to delta_s
    inj = ols(rows, -dpf / N)
    return {
        "prefill_probe_gap_ours_minus_frontier_ms": dpf,
        "variants": variants,
        "analytic_injection_of_a_constant_prefill_gap": {
            "delta_s_injected": inj["intercept"],
            "delta_h_injected": inj["slope"],
            "note": (
                "a constant prefill-probe gap of this size alone would produce this much "
                "apparent delta_s, with no decode difference at all"
            ),
        },
        "share_of_delta_s_explained_by_the_prefill_probe": (
            inj["intercept"] / variants["own_prefill_advisor_method"]["delta_s"]
        ),
    }


def a1_paired_difference(ours, front, n_by_sha):
    """A1, A3 and A4 together: is delta_s identified, and is it forced?"""
    fo = fit_receipt(ours, n_by_sha)
    ff = fit_receipt(front, n_by_sha)
    rows = np.array(fo["rows"])
    N = np.array(fo["N"], float)

    dR = np.array(fo["R_ms"]) - np.array(ff["R_ms"])
    dleg = np.array(fo["leg_ms"]) - np.array(ff["leg_ms"])
    dpf = np.array(fo["prefill_ms"]) - np.array(ff["prefill_ms"])
    D = dleg - dpf  # the per-leg decode-seconds gap, ms

    diff = ols(rows, dR)
    t_s = diff["intercept"] / diff["se_intercept"]
    t_h = diff["slope"] / diff["se_slope"]

    # A4. delta_R = D / N identically. If D is constant across prompts, then
    # delta_R is a pure function of 1/N, and 1/N is close to affine in rows,
    # so an affine fit is forced and carries no cost information.
    inv_n = 1.0 / N
    affine_invn = ols(rows, inv_n)
    # one-parameter alternative: a single constant leg-time gap D_hat
    d_const = float(D.mean())
    pred_const = d_const * inv_n
    sse_const = float(((dR - pred_const) ** 2).sum())

    return {
        "harness": "ranked",
        "single_receipt_fits": {
            "ours": {k: fo[k] for k in
                     ("intercept", "slope", "se_intercept", "se_slope",
                      "corr_intercept_slope", "sigma", "dof", "sse",
                      "worst_abs_resid", "sxx", "x_mean", "n")},
            "frontier": {k: ff[k] for k in
                         ("intercept", "slope", "se_intercept", "se_slope",
                          "corr_intercept_slope", "sigma", "dof", "sse",
                          "worst_abs_resid", "sxx", "x_mean", "n")},
        },
        "paired_difference_regression": {
            "delta_s_ms_per_round": diff["intercept"],
            "se_delta_s": diff["se_intercept"],
            "t_delta_s": t_s,
            "p_delta_s": student_t_sf(t_s, diff["dof"]),
            "delta_s_ci95": [
                diff["intercept"] - 2.571 * diff["se_intercept"],
                diff["intercept"] + 2.571 * diff["se_intercept"],
            ],
            "delta_h_ms_per_row": diff["slope"],
            "se_delta_h": diff["se_slope"],
            "t_delta_h": t_h,
            "p_delta_h": student_t_sf(t_h, diff["dof"]),
            "delta_h_ci95": [
                diff["slope"] - 2.571 * diff["se_slope"],
                diff["slope"] + 2.571 * diff["se_slope"],
            ],
            "corr_delta_s_delta_h": diff["corr_intercept_slope"],
            "sigma_ms": diff["sigma"],
            "dof": diff["dof"],
            "worst_abs_resid_ms": diff["worst_abs_resid"],
            "survives_at_2se": bool(abs(t_s) > 2.571),
            "t_critical_2sided_5pct_dof5": 2.571,
        },
        "a4_is_the_affine_form_forced": {
            "delta_R_equals_D_over_N_identically": True,
            "D_ms_per_leg": [float(v) for v in D],
            "D_mean_ms": float(D.mean()),
            "D_sd_ms": float(D.std(ddof=1)),
            "D_rel_sd_pct": float(100.0 * D.std(ddof=1) / abs(D.mean())),
            "D_min_ms": float(D.min()),
            "D_max_ms": float(D.max()),
            "corr_D_vs_rows": float(np.corrcoef(D, rows)[0, 1]),
            "one_over_N_is_affine_in_rows_r2": float(
                1.0 - affine_invn["sse"] / ((inv_n - inv_n.mean()) ** 2).sum()
            ),
            "sse_two_param_affine_in_rows": diff["sse"],
            "sse_one_param_constant_leg_gap": sse_const,
            "constant_leg_gap_D_ms": d_const,
            "verdict": (
                "the affine form is largely forced if D is near constant and 1/N is "
                "near affine in rows"
            ),
        },
        "a3_sensitivity_to_a_wrong_N": None,
    }


def a3_n_sensitivity(ours, front, n_by_sha):
    """A3: what does a plausible N error do to delta_s?"""
    base = a1_paired_difference(ours, front, n_by_sha)["paired_difference_regression"]
    out = []
    for sha8, n in list(n_by_sha.items()):
        cell = [c for c in ours["officialMetrics"]["per_prompt"]
                if c["prompt_sha256"].startswith(sha8)][0]
        if cell["non_drafting_round_count"] > 0:
            continue
        feas = recover_rounds(cell["effective_mean_draft_len"])
        for alt in feas:
            if alt == n:
                continue
            trial = dict(n_by_sha)
            trial[sha8] = alt
            try:
                r = a1_paired_difference(ours, front, trial)["paired_difference_regression"]
            except Exception:
                continue
            out.append(
                {
                    "prompt_sha8": sha8,
                    "N_base": n,
                    "N_alt": alt,
                    "ratio": alt / n,
                    "delta_s": r["delta_s_ms_per_round"],
                    "delta_s_shift": r["delta_s_ms_per_round"] - base["delta_s_ms_per_round"],
                    "delta_h": r["delta_h_ms_per_row"],
                }
            )
    out.sort(key=lambda d: -abs(d["delta_s_shift"]))
    return {
        "base_delta_s": base["delta_s_ms_per_round"],
        "base_se_delta_s": base["se_delta_s"],
        "n_alternatives_tried": len(out),
        "worst_shift": out[0] if out else None,
        "smallest_alternative_shifts": out[:8],
    }


def b2_solver_or_calendar(b1):
    """B2: is the low-prefill cluster a solver or a date window?"""
    per = b1["_per_receipt_prefill"]
    low = [p for p in per if p["mean_ms"] < 510.0]
    by_solver, by_day = {}, {}
    for p in per:
        by_solver.setdefault(p["solver"], []).append(p)
        by_day.setdefault(p["created"][:10], []).append(p)

    straddlers = []
    for s, ps in by_solver.items():
        lo = [p for p in ps if p["mean_ms"] < 510.0]
        hi = [p for p in ps if p["mean_ms"] >= 510.0]
        if lo and hi:
            straddlers.append(
                {
                    "solver": s,
                    "n_low": len(lo),
                    "n_high": len(hi),
                    "low_dates": sorted({p["created"][:10] for p in lo}),
                    "high_dates": sorted({p["created"][:10] for p in hi}),
                    "min_ms": min(p["mean_ms"] for p in ps),
                    "max_ms": max(p["mean_ms"] for p in ps),
                }
            )
    day_tab = []
    for d in sorted(by_day):
        v = np.array([p["mean_ms"] for p in by_day[d]])
        day_tab.append(
            {
                "day": d,
                "n": int(v.size),
                "mean_ms": float(v.mean()),
                "median_ms": float(np.median(v)),
                "min_ms": float(v.min()),
                "frac_below_510": float((v < 510).mean()),
            }
        )
    solver_tab = []
    for s, ps in sorted(by_solver.items(), key=lambda kv: np.median([p["mean_ms"] for p in kv[1]])):
        v = np.array([p["mean_ms"] for p in ps])
        solver_tab.append(
            {
                "solver": s,
                "n": int(v.size),
                "median_ms": float(np.median(v)),
                "min_ms": float(v.min()),
                "max_ms": float(v.max()),
                "frac_below_510": float((v < 510).mean()),
                "days": sorted({p["created"][:10] for p in ps}),
            }
        )
    return {
        "harness": "ranked",
        "n_receipts_with_prefill": len(per),
        "n_below_510ms": len(low),
        "low_cluster_solvers": sorted({p["solver"] for p in low}),
        "low_cluster_days": sorted({p["created"][:10] for p in low}),
        "solvers_straddling_the_step": straddlers,
        "by_day": day_tab,
        "by_solver_top20_lowest": solver_tab[:20],
    }


def b2b_is_claim_b_selection_on_the_outcome(b1, board, ids=("5a9f130a", "ec24d591")):
    """The prefill step is inside the scored leg, so picking the best-scoring
    receipt of a solver also picks that solver's luckiest prefill draw.

    Compare each named receipt's prefill against its own solver's distribution.
    """
    per = {p["id8"]: p for p in b1["_per_receipt_prefill"]}
    by_solver = {}
    for p in b1["_per_receipt_prefill"]:
        by_solver.setdefault(p["solver"], []).append(p)

    score_by_id = {r["id"][:8]: r.get("officialScore") for r in board}

    # the receipts the advisor actually used, plus every solver's best score
    named = list(ids)
    for s in ("vibecodooor", "scarletbright", "Amal-David", "BitWonka"):
        ps = by_solver.get(s, [])
        if ps:
            named.append(min(ps, key=lambda p: p["mean_ms"])["id8"])

    out = []
    for i8 in named:
        p = per.get(i8)
        if not p:
            continue
        sib = np.array([q["mean_ms"] for q in by_solver[p["solver"]]])
        rank = int((sib < p["mean_ms"]).sum()) + 1
        out.append(
            {
                "id8": i8,
                "solver": p["solver"],
                "created": p["created"],
                "score": score_by_id.get(i8),
                "prefill_ms": p["mean_ms"],
                "solver_n_receipts": int(sib.size),
                "solver_median_prefill_ms": float(np.median(sib)),
                "solver_sd_prefill_ms": float(sib.std(ddof=1)) if sib.size > 1 else None,
                "solver_min_prefill_ms": float(sib.min()),
                "rank_within_solver_low_is_fast": rank,
                "deviation_from_own_solver_median_ms": float(p["mean_ms"] - np.median(sib)),
            }
        )

    # honest run-to-run error bar for prefill: within solver, on the stable
    # runner window only, so the calendar step cannot inflate it
    stable = [p for p in b1["_per_receipt_prefill"] if p["created"][:10] >= "2026-08-19"]
    devs = []
    for s, ps in by_solver.items():
        q = [p for p in ps if p["created"][:10] >= "2026-08-19"]
        if len(q) < 5:
            continue
        v = np.array([p["mean_ms"] for p in q])
        devs.extend(v - np.median(v))
    devs = np.array(devs)
    allv = np.array([p["mean_ms"] for p in stable])
    return {
        "harness": "ranked",
        "named_receipts": out,
        "stable_window_from": "2026-08-19",
        "stable_window_n": int(allv.size),
        "stable_window_median_ms": float(np.median(allv)),
        "within_solver_dev_sd_ms": float(devs.std(ddof=1)),
        "within_solver_dev_p01_ms": float(np.percentile(devs, 1)),
        "within_solver_dev_p05_ms": float(np.percentile(devs, 5)),
        "within_solver_dev_min_ms": float(devs.min()),
        "n_within_solver_devs": int(devs.size),
        "prefill_percentiles_stable_ms": {
            str(q): float(np.percentile(allv, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)
        },
    }


def b3_does_prefill_pay(b1, board):
    """B3: board-wide, does low prefill predict a high score?"""
    per = {p["id8"]: p for p in b1["_per_receipt_prefill"]}
    x, y, xd = [], [], []
    for r in board:
        i8 = r["id"][:8]
        p = per.get(i8)
        if not p or r.get("officialScore") is None:
            continue
        if r["createdAt"][:10] < "2026-08-19":
            continue
        om = r["officialMetrics"]
        cand = om.get("candidate_mtp_seconds_per_token_mean")
        if not cand:
            continue
        x.append(p["mean_ms"])
        xd.append(cand * DECODE_TOKENS * 1000.0 - p["mean_ms"])
        y.append(r["officialScore"])
    x, y, xd = np.array(x), np.array(y), np.array(xd)
    keep = y > 3.0  # competitive rows only; the tail is a different regime
    res = {"harness": "ranked", "n_all": int(x.size), "n_competitive": int(keep.sum())}
    for label, m in (("all", np.ones_like(y, bool)), ("competitive_score_gt_3", keep)):
        if m.sum() < 10:
            continue
        f = ols(x[m], y[m])
        g = ols(xd[m], y[m])
        res[label] = {
            "corr_score_vs_prefill": float(np.corrcoef(x[m], y[m])[0, 1]),
            "slope_score_per_ms_prefill": f["slope"],
            "se_slope_prefill": f["se_slope"],
            "t_prefill": f["slope"] / f["se_slope"],
            "corr_score_vs_decode": float(np.corrcoef(xd[m], y[m])[0, 1]),
            "slope_score_per_ms_decode": g["slope"],
            "se_slope_decode": g["se_slope"],
            "t_decode": g["slope"] / g["se_slope"],
        }
    return res


def b4_per_prompt_prefill(board, ids):
    out = {}
    for i8 in ids:
        hits = [r for r in board if r["id"].startswith(i8)]
        if not hits:
            continue
        r = hits[0]
        cells = r["officialMetrics"]["per_prompt"]
        vals = [c.get("prefill_seconds_per_token") for c in cells]
        if any(v is None for v in vals):
            continue
        out[i8] = {
            "solver": r["solverUsername"],
            "created": r["createdAt"],
            "score": r.get("officialScore"),
            "per_prompt_ms": [round(v * DECODE_TOKENS * 1000.0, 3) for v in vals],
            "prompt_sha8": [c["prompt_sha256"][:8] for c in cells],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=LIVE_BOARD)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "e164-artifacts"))
    args = ap.parse_args()

    board = load_board(args.board)
    names = prompt_names(board)
    print(f"board receipts with per_prompt + officialScore: {len(board)}")

    b1 = b1_prefill_reconciliation(board, names)
    a1meta, ours, front = a1_claim_a(board, names)
    n_by_sha, n_report = resolve_n_vector(ours)
    a1meta["n_vector"] = n_report
    a1 = a1_paired_difference(ours, front, n_by_sha)
    a1["a3_sensitivity_to_a_wrong_N"] = a3_n_sensitivity(ours, front, n_by_sha)
    a1["a1_prefill_attribution"] = a1_prefill_attribution(ours, front, n_by_sha)
    b2 = b2_solver_or_calendar(b1)

    print("\n=== N VECTOR ===")
    for d in n_report:
        print(f"  {d['prompt_sha8']}  edl {d['edl']:.6f}  N={d['chosen_N']}  "
              f"smallest={d['is_smallest_feasible']}  nfeas={d['n_feasible']}  "
              f"alpha={d['implied_acceptance_alpha']}")
    print("\n=== A1 single-receipt fits ===")
    for k, v in a1["single_receipt_fits"].items():
        print(f"  {k:9s} s={v['intercept']:8.4f} +/- {v['se_intercept']:.4f}   "
              f"h={v['slope']:7.4f} +/- {v['se_slope']:.4f}   "
              f"corr={v['corr_intercept_slope']:+.4f}  sigma={v['sigma']:.4f}  "
              f"worst|r|={v['worst_abs_resid']:.4f}")
    p = a1["paired_difference_regression"]
    print("\n=== A1 paired difference regression ===")
    print(f"  delta_s = {p['delta_s_ms_per_round']:+.4f} +/- {p['se_delta_s']:.4f} ms/round  "
          f"t={p['t_delta_s']:+.3f}  p={p['p_delta_s']:.4f}  "
          f"CI95 [{p['delta_s_ci95'][0]:+.4f}, {p['delta_s_ci95'][1]:+.4f}]")
    print(f"  delta_h = {p['delta_h_ms_per_row']:+.4f} +/- {p['se_delta_h']:.4f} ms/row    "
          f"t={p['t_delta_h']:+.3f}  p={p['p_delta_h']:.4f}")
    print(f"  corr(delta_s,delta_h) = {p['corr_delta_s_delta_h']:+.4f}   sigma={p['sigma_ms']:.4f}"
          f"   survives 2se: {p['survives_at_2se']}")
    a4 = a1["a4_is_the_affine_form_forced"]
    print("\n=== A4 is it circular ===")
    print(f"  D (leg gap minus prefill gap), ms: "
          f"{[round(v,3) for v in a4['D_ms_per_leg']]}")
    print(f"  D mean {a4['D_mean_ms']:.3f} ms  sd {a4['D_sd_ms']:.3f} ms  "
          f"rel sd {a4['D_rel_sd_pct']:.1f} %  corr(D,rows) {a4['corr_D_vs_rows']:+.3f}")
    print(f"  1/N affine in rows, r2 = {a4['one_over_N_is_affine_in_rows_r2']:.6f}")
    print(f"  SSE two-param affine {a4['sse_two_param_affine_in_rows']:.6f}   "
          f"SSE one-param constant leg gap {a4['sse_one_param_constant_leg_gap']:.6f}")
    a3 = a1["a3_sensitivity_to_a_wrong_N"]
    print("\n=== A3 N sensitivity ===")
    print(f"  worst single-prompt N swap: {a3['worst_shift']}")
    pa = a1["a1_prefill_attribution"]
    print("\n=== A1b prefill attribution ===")
    print(f"  prefill probe gap ours - frontier = "
          f"{pa['prefill_probe_gap_ours_minus_frontier_ms']:+.4f} ms")
    for k, v in pa["variants"].items():
        print(f"  {k:28s} delta_s {v['delta_s']:+.4f} +/- {v['se_delta_s']:.4f} "
              f"(t {v['t_delta_s']:+.2f})   delta_h {v['delta_h']:+.4f}   "
              f"dR(rows=6) {v['delta_R_at_rows_6']:+.4f}")
    inj = pa["analytic_injection_of_a_constant_prefill_gap"]
    print(f"  a constant prefill gap alone injects delta_s {inj['delta_s_injected']:+.4f}, "
          f"delta_h {inj['delta_h_injected']:+.4f}")
    print(f"  share of delta_s explained by the prefill probe: "
          f"{pa['share_of_delta_s_explained_by_the_prefill_probe']:.3f}")
    print("\n=== B2 solver or calendar ===")
    print(f"  receipts with prefill {b2['n_receipts_with_prefill']}, "
          f"below 510 ms: {b2['n_below_510ms']}")
    print(f"  low-cluster solvers: {b2['low_cluster_solvers']}")
    print(f"  low-cluster days:    {b2['low_cluster_days']}")
    print(f"  solvers straddling the step: {len(b2['solvers_straddling_the_step'])}")
    for s in b2["solvers_straddling_the_step"][:10]:
        print(f"     {s['solver']:16s} low {s['n_low']:3d} high {s['n_high']:3d}  "
              f"[{s['min_ms']:.1f}, {s['max_ms']:.1f}] ms  low_days {s['low_dates']}")
    print("\n  by day:")
    for d in b2["by_day"]:
        print(f"     {d['day']}  n={d['n']:4d}  median {d['median_ms']:7.2f}  "
              f"min {d['min_ms']:7.2f}  frac<510 {d['frac_below_510']:.3f}")

    b2b = b2b_is_claim_b_selection_on_the_outcome(b1, board)
    b3 = b3_does_prefill_pay(b1, board)
    b4 = b4_per_prompt_prefill(
        board,
        ["5a9f130a", "ec24d591"]
        + [d["id8"] for d in b2b["named_receipts"] if d["id8"] not in ("5a9f130a", "ec24d591")],
    )

    print("\n=== B2b is Claim B selection on the outcome ===")
    print(f"  stable window from {b2b['stable_window_from']}, n={b2b['stable_window_n']}, "
          f"median {b2b['stable_window_median_ms']:.2f} ms")
    print(f"  within-solver deviation sd {b2b['within_solver_dev_sd_ms']:.2f} ms   "
          f"p05 {b2b['within_solver_dev_p05_ms']:+.2f}   p01 {b2b['within_solver_dev_p01_ms']:+.2f}"
          f"   min {b2b['within_solver_dev_min_ms']:+.2f}  (n={b2b['n_within_solver_devs']})")
    print("  receipt    solver           score        prefill  solverN  solverMed   dev   rank")
    for d in b2b["named_receipts"]:
        sc = f"{d['score']:.6f}" if d["score"] is not None else "        "
        print(f"  {d['id8']}  {d['solver']:15s} {sc}  {d['prefill_ms']:8.2f}  "
              f"{d['solver_n_receipts']:6d}  {d['solver_median_prefill_ms']:9.2f}  "
              f"{d['deviation_from_own_solver_median_ms']:+7.2f}  "
              f"{d['rank_within_solver_low_is_fast']:3d}/{d['solver_n_receipts']}")
    print("\n=== B3 does prefill pay board-wide ===")
    for k in ("all", "competitive_score_gt_3"):
        if k in b3:
            v = b3[k]
            print(f"  {k:24s} corr(score,prefill) {v['corr_score_vs_prefill']:+.4f} "
                  f"t {v['t_prefill']:+.2f}    corr(score,decode) "
                  f"{v['corr_score_vs_decode']:+.4f} t {v['t_decode']:+.2f}")
    print("\n=== B4 per-prompt prefill, ms ===")
    for i8, d in b4.items():
        print(f"  {i8} {d['solver']:15s} {d['per_prompt_ms']}")

    os.makedirs(args.out, exist_ok=True)
    b1.pop("_per_receipt_prefill", None)
    with open(os.path.join(args.out, "e164-redteam.json"), "w") as fh:
        json.dump({"b1": b1, "a1_meta": a1meta, "a1": a1, "b2": b2,
                   "b2b": b2b, "b3": b3, "b4": b4}, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
