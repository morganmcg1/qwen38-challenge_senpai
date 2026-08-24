#!/usr/bin/env python3
"""E2/E3: check the prefill refutation and redo the neutral ranking honestly.

E2  Is any solver *consistently* below the main prefill mode across many
    receipts?  The advisor's straddle test says no; this re-runs it with
    per-solver medians, day residuals, and a per-solver share-below test
    instead of a single min/max span.

E3  FINDING 353's neutral ranking, recomputed with the 2026-08-23 low-prefill
    receipts handled honestly: once by exclusion, once by correcting every
    receipt's candidate leg to a common prefill reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict

DEFAULT_BOARD = "/tmp/yukon-board/full.json"
T = 512  # decode tokens and seed tokens

PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]
DRAFTING = [p for p in ORDER if p != "plutarch"]


def load_board(path):
    with open(path) as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else data["rows"]


def cells_by_name(row):
    metrics = row.get("officialMetrics") or {}
    out = {}
    for cell in metrics.get("per_prompt") or []:
        name = PROMPT_NAMES.get((cell.get("prompt_sha256") or "")[:8])
        if name:
            out[name] = cell
    return out


def eligible(row):
    metrics = row.get("officialMetrics") or {}
    if row.get("status") in ("failed", "cancelled", "validating"):
        return False
    if row.get("officialScore") is None:
        return False
    if metrics.get("prompt_count") != 8 or metrics.get("decode_tokens") != T:
        return False
    if not metrics.get("parity_all_ok"):
        return False
    cells = cells_by_name(row)
    if len(cells) != 8:
        return False
    for name in ORDER:
        c = cells[name]
        if c.get("serial_seconds_per_token_mean") is None:
            return False
        if c.get("mtp_seconds_per_token_mean") is None:
            return False
    return True


def median8(vals):
    v = sorted(vals)
    return 0.5 * (v[3] + v[4])


def day(row):
    return (row.get("createdAt") or "")[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=DEFAULT_BOARD)
    ap.add_argument("--out", default="research/e164-artifacts/e164-e2e3.json")
    args = ap.parse_args()

    rows = [r for r in load_board(args.board) if eligible(r)]
    report = []
    results = {"n_eligible": len(rows)}
    report.append("=" * 78)
    report.append("E2 / E3   prefill consistency and the honest neutral ranking")
    report.append("=" * 78)
    report.append(f"eligible receipts (parity_all_ok, 8 prompts, scored): {len(rows)}")

    # ------------------------------------------------------------------ E2
    report.append("")
    report.append("-" * 78)
    report.append("E2  is any solver CONSISTENTLY below the main prefill mode?")
    report.append("-" * 78)

    pref = []  # (row, mean prefill ms over 8 prompts)
    for r in rows:
        cells = cells_by_name(r)
        vals = [
            cells[n].get("prefill_seconds_per_token")
            for n in ORDER
            if cells[n].get("prefill_seconds_per_token") is not None
        ]
        if len(vals) == 8:
            pref.append((r, sum(vals) / 8.0 * T * 1000.0))
    report.append(f"receipts carrying per-prompt prefill on all 8 prompts: {len(pref)}")

    allp = [p for _, p in pref]
    main_mode = [p for p in allp if 520.0 <= p <= 535.0]
    mode_mean = statistics.mean(main_mode)
    mode_sd = statistics.pstdev(main_mode)
    report.append(
        f"main mode 520-535 ms: n={len(main_mode)} mean {mode_mean:.3f} sd {mode_sd:.3f} ms"
    )
    report.append(
        f"all receipts        : n={len(allp)} mean {statistics.mean(allp):.3f} "
        f"sd {statistics.pstdev(allp):.3f} ms  min {min(allp):.3f} max {max(allp):.3f}"
    )

    # Day medians: is there a calendar component?
    byday = defaultdict(list)
    for r, p in pref:
        byday[day(r)].append(p)
    report.append("")
    report.append("prefill by submission day (median, n)")
    for d in sorted(byday):
        v = byday[d]
        report.append(
            f"  {d}  n={len(v):>4}  median {statistics.median(v):8.3f}  "
            f"min {min(v):8.3f}  max {max(v):8.3f}"
        )
    results["day_medians"] = {
        d: {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)}
        for d, v in byday.items()
    }

    # Day-residualised prefill removes the calendar step before judging solvers.
    day_med = {d: statistics.median(v) for d, v in byday.items()}
    resid = [(r, p, p - day_med[day(r)]) for r, p in pref]

    bysolver = defaultdict(list)
    for r, p, e in resid:
        bysolver[r.get("solverUsername") or "?"].append((r, p, e))

    LOW = 512.0
    report.append("")
    report.append("per-solver prefill: raw and day-residualised")
    report.append(
        f"  {'solver':<18} {'n':>4} {'median':>9} {'min':>9} {'max':>9} "
        f"{'span':>8} {'<512':>5} {'medResid':>9} {'residMin':>9} {'residMax':>9}"
    )
    solver_rows = []
    for name in sorted(bysolver, key=lambda s: -len(bysolver[s])):
        recs = bysolver[name]
        if len(recs) < 3:
            continue
        ps = [p for _, p, _ in recs]
        es = [e for _, _, e in recs]
        nlow = sum(1 for p in ps if p < LOW)
        solver_rows.append(
            {
                "solver": name,
                "n": len(recs),
                "median": statistics.median(ps),
                "min": min(ps),
                "max": max(ps),
                "span": max(ps) - min(ps),
                "n_below_512": nlow,
                "share_below_512": nlow / len(recs),
                "median_day_residual": statistics.median(es),
                "resid_min": min(es),
                "resid_max": max(es),
            }
        )
    solver_rows.sort(key=lambda d: d["median_day_residual"])
    for d in solver_rows:
        report.append(
            f"  {d['solver']:<18} {d['n']:>4} {d['median']:>9.3f} {d['min']:>9.3f} "
            f"{d['max']:>9.3f} {d['span']:>8.3f} {d['n_below_512']:>5} "
            f"{d['median_day_residual']:>+9.3f} {d['resid_min']:>+9.3f} {d['resid_max']:>+9.3f}"
        )
    results["e2_solvers"] = solver_rows

    consistent = [
        d for d in solver_rows if d["share_below_512"] >= 0.5 and d["n"] >= 5
    ]
    report.append("")
    report.append(
        f"solvers with >=5 receipts and >=50% of them below {LOW:.0f} ms: "
        f"{[d['solver'] for d in consistent] or 'NONE'}"
    )
    # The tighter version of the same question, on day residuals.
    strong = [
        d
        for d in solver_rows
        if d["n"] >= 5 and d["resid_max"] < -2.0
    ]
    report.append(
        f"solvers with >=5 receipts whose WORST day-residual is still below "
        f"-2.0 ms (i.e. never in the pack): {[d['solver'] for d in strong] or 'NONE'}"
    )
    results["e2_consistent_low_solvers"] = [d["solver"] for d in consistent]
    results["e2_always_below_pack_solvers"] = [d["solver"] for d in strong]

    # The five low receipts, named, with each solver's own distribution around them.
    low_recs = sorted([(p, r) for r, p in pref if p < LOW])
    report.append("")
    report.append(f"every receipt below {LOW:.0f} ms")
    for p, r in low_recs:
        name = r.get("solverUsername")
        own = [q for _, q, _ in bysolver[name]]
        report.append(
            f"  {day(r)}  {name:<16} {str(r.get('id'))[:8]}  {p:8.3f} ms  "
            f"status={r.get('status'):<9} solver median {statistics.median(own):8.3f} "
            f"(n={len(own)}, min {min(own):.3f}, max {max(own):.3f})"
        )
    results["e2_low_receipts"] = [
        {
            "day": day(r),
            "solver": r.get("solverUsername"),
            "id": r.get("id"),
            "prefill_ms": p,
            "status": r.get("status"),
        }
        for p, r in low_recs
    ]

    # Does a solver's prefill predict anything, once the day is removed?
    # Between-solver variance of the day residual against within-solver variance.
    grand = statistics.mean(e for _, _, e in resid)
    groups = [
        [e for _, _, e in bysolver[s]] for s in bysolver if len(bysolver[s]) >= 3
    ]
    if len(groups) >= 2:
        n_tot = sum(len(g) for g in groups)
        ss_between = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups)
        ss_within = sum((x - statistics.mean(g)) ** 2 for g in groups for x in g)
        df_b = len(groups) - 1
        df_w = n_tot - len(groups)
        ms_b = ss_between / df_b
        ms_w = ss_within / df_w
        f_stat = ms_b / ms_w
        report.append("")
        report.append("one-way ANOVA of day-residual prefill by solver (groups with n>=3)")
        report.append(
            f"  groups={len(groups)}  n={n_tot}  MS_between={ms_b:.4f}  "
            f"MS_within={ms_w:.4f}  F={f_stat:.4f}  df=({df_b}, {df_w})"
        )
        # Variance component: how much of the residual sd is solver identity?
        k = n_tot / len(groups)
        var_solver = max(0.0, (ms_b - ms_w) / k)
        report.append(
            f"  solver variance component = {var_solver:.4f} ms^2 "
            f"(sd {math.sqrt(var_solver):.3f} ms) against within-solver sd "
            f"{math.sqrt(ms_w):.3f} ms"
        )
        med_res = [d["median_day_residual"] for d in solver_rows]
        report.append(
            f"  spread of the {len(med_res)} per-solver MEDIAN day-residuals: "
            f"min {min(med_res):+.3f}  max {max(med_res):+.3f}  "
            f"sd {statistics.pstdev(med_res):.3f} ms"
        )
        inner = [x for x in med_res if abs(x) <= 0.5]
        report.append(
            f"  {len(inner)} of {len(med_res)} solver medians are within +/-0.5 ms of "
            f"their own day, against a 22 ms low cluster."
        )
        results["e2_solver_median_residual_spread"] = {
            "min": min(med_res),
            "max": max(med_res),
            "sd": statistics.pstdev(med_res),
            "n_within_0p5ms": len(inner),
            "n_solvers": len(med_res),
        }
        results["e2_anova"] = {
            "groups": len(groups),
            "n": n_tot,
            "ms_between": ms_b,
            "ms_within": ms_w,
            "F": f_stat,
            "df": [df_b, df_w],
            "solver_sd_ms": math.sqrt(var_solver),
            "within_sd_ms": math.sqrt(ms_w),
        }

    # ------------------------------------------------------------------ E3
    report.append("")
    report.append("-" * 78)
    report.append("E3  neutral ranking, with the low-prefill day handled honestly")
    report.append("-" * 78)

    # Board-wide reference serial leg per prompt (the pinned, uneditable build).
    serial_ref = {}
    for name in ORDER:
        vals = [cells_by_name(r)[name]["serial_seconds_per_token_mean"] for r in rows]
        serial_ref[name] = statistics.mean(vals)
    report.append("reference serial_spt per prompt (mean over all eligible receipts)")
    report.append(
        "  " + "  ".join(f"{n}={serial_ref[n]:.7f}" for n in ORDER[:4])
    )
    report.append(
        "  " + "  ".join(f"{n}={serial_ref[n]:.7f}" for n in ORDER[4:])
    )
    results["serial_reference"] = serial_ref

    prefill_ref = mode_mean  # ms per leg, main-mode mean

    def neutral(row, prefill_correct=False):
        cells = cells_by_name(row)
        raws = []
        for name in ORDER:
            c = cells[name]
            m = c["mtp_seconds_per_token_mean"]
            if prefill_correct:
                p = c.get("prefill_seconds_per_token")
                if p is None:
                    return None
                own_ms = p * T * 1000.0
                # Put every receipt on the same prefill: add back what the host gave it.
                m = m + ((prefill_ref - own_ms) / 1000.0) / T
            raws.append(serial_ref[name] / m)
        return median8(raws)

    recs = []
    for r in rows:
        n0 = neutral(r)
        n1 = neutral(r, prefill_correct=True)
        cells = cells_by_name(r)
        pv = [
            cells[n].get("prefill_seconds_per_token")
            for n in ORDER
            if cells[n].get("prefill_seconds_per_token") is not None
        ]
        pm = sum(pv) / 8.0 * T * 1000.0 if len(pv) == 8 else None
        recs.append(
            {
                "id": r.get("id"),
                "solver": r.get("solverUsername"),
                "day": day(r),
                "official": r.get("officialScore"),
                "neutral": n0,
                "neutral_prefill_corrected": n1,
                "prefill_ms": pm,
                "commit": r.get("submissionCommitSha"),
            }
        )

    anomaly_ids = {d["id"] for d in results["e2_low_receipts"]}

    def top_table(key, exclude=(), title=""):
        pool = [x for x in recs if x[key] is not None and x["id"] not in exclude]
        pool.sort(key=lambda x: -x[key])
        report.append("")
        report.append(title)
        report.append(
            f"  {'rank':>4} {'solver':<16} {'receipt':<10} {'day':<11} "
            f"{key:>12} {'official':>11} {'prefill_ms':>11}"
        )
        seen = set()
        out = []
        for x in pool:
            if x["solver"] in seen:
                continue
            seen.add(x["solver"])
            out.append(x)
            if len(out) >= 12:
                break
        for i, x in enumerate(out, start=1):
            pm = f"{x['prefill_ms']:11.3f}" if x["prefill_ms"] else " " * 11
            report.append(
                f"  {i:>4} {x['solver']:<16} {str(x['id'])[:8]:<10} {x['day']:<11} "
                f"{x[key]:>12.6f} {x['official']:>11.6f} {pm}"
            )
        return out

    t_raw = top_table("neutral", title="A) neutral ranking, best receipt per solver, NOTHING excluded")
    t_excl = top_table(
        "neutral",
        exclude=anomaly_ids,
        title="B) neutral ranking, best receipt per solver, low-prefill receipts EXCLUDED",
    )
    t_corr = top_table(
        "neutral_prefill_corrected",
        title=(
            f"C) neutral ranking, every receipt corrected to a common prefill of "
            f"{prefill_ref:.3f} ms"
        ),
    )
    results["e3_neutral_all"] = t_raw
    results["e3_neutral_excluding_anomaly"] = t_excl
    results["e3_neutral_prefill_corrected"] = t_corr

    # What the correction does to the named anomaly receipts.
    report.append("")
    report.append("effect of the prefill correction on the five anomalous receipts")
    report.append(
        f"  {'solver':<16} {'receipt':<10} {'prefill_ms':>11} {'neutral':>11} "
        f"{'corrected':>11} {'shift':>10}"
    )
    for x in recs:
        if x["id"] in anomaly_ids:
            report.append(
                f"  {x['solver']:<16} {str(x['id'])[:8]:<10} {x['prefill_ms']:>11.3f} "
                f"{x['neutral']:>11.6f} {x['neutral_prefill_corrected']:>11.6f} "
                f"{x['neutral_prefill_corrected'] - x['neutral']:>+10.6f}"
            )

    # Our own and the crown's rows, always shown.
    report.append("")
    report.append("anchors")
    for want in ("5a9f130a", "ec24d591"):
        for x in recs:
            if str(x["id"]).startswith(want):
                report.append(
                    f"  {want} {x['solver']:<16} official {x['official']:.6f}  "
                    f"neutral {x['neutral']:.6f}  corrected "
                    f"{x['neutral_prefill_corrected']:.6f}  prefill {x['prefill_ms']:.3f} ms"
                )
                results[f"anchor_{want}"] = x

    # Draw share: official minus neutral, for the top of each list.
    report.append("")
    report.append("serial-draw share (official - neutral), corrected ranking, top 12")
    for i, x in enumerate(t_corr, start=1):
        report.append(
            f"  {i:>2} {x['solver']:<16} official {x['official']:.6f}  corrected "
            f"{x['neutral_prefill_corrected']:.6f}  draw "
            f"{100.0 * (x['official'] - x['neutral_prefill_corrected']) / x['neutral_prefill_corrected']:+.4f}%"
        )

    # A 512 ms cut misses the milder anomalies; list everything below 525 ms.
    report.append("")
    report.append("all receipts below 525 ms, which a 512 ms cut would keep")
    mild = sorted([(p, r) for r, p in pref if p < 525.0])
    for p, r in mild:
        report.append(
            f"  {day(r)}  {r.get('solverUsername'):<16} {str(r.get('id'))[:8]}  {p:8.3f} ms"
        )
    results["e3_receipts_below_525ms"] = [
        {"day": day(r), "solver": r.get("solverUsername"), "id": r.get("id"), "prefill_ms": p}
        for p, r in mild
    ]

    # ------------------------------------------------------- selection bias
    report.append("")
    report.append("-" * 78)
    report.append("E3b  best-of-n selection bias in the neutral ranking")
    report.append("-" * 78)
    per_solver = defaultdict(list)
    for x in recs:
        if x["neutral_prefill_corrected"] is not None:
            per_solver[x["solver"]].append(x)
    pts = []
    for s, xs in per_solver.items():
        if len(xs) < 2:
            continue
        best = max(v["neutral_prefill_corrected"] for v in xs)
        med = statistics.median(v["neutral_prefill_corrected"] for v in xs)
        pts.append((math.log(len(xs)), best, med, s, len(xs)))
    if len(pts) >= 5:
        xs_ = [p[0] for p in pts]
        xbar = statistics.mean(xs_)
        sxx = sum((x - xbar) ** 2 for x in xs_)
        for label_, col in (("best", 1), ("median", 2)):
            ys_ = [p[col] for p in pts]
            ybar = statistics.mean(ys_)
            sxy = sum((p[0] - xbar) * (p[col] - ybar) for p in pts)
            slope = sxy / sxx
            resid = [p[col] - (ybar + slope * (p[0] - xbar)) for p in pts]
            sig = math.sqrt(sum(r * r for r in resid) / (len(pts) - 2))
            se = sig / math.sqrt(sxx)
            report.append(
                f"  regression of solver {label_:<6} neutral on log(n receipts): "
                f"slope {slope:+.6f} +/- {se:.6f}  t={slope / se:+.3f}  n={len(pts)}"
            )
            results[f"e3b_slope_{label_}"] = {"slope": slope, "se": se, "t": slope / se, "n": len(pts)}
        report.append(
            "  a positive 'best' slope with a flat 'median' slope is best-of-n "
            "selection on noise, not a faster implementation."
        )

    # Ranking with no selection on the outcome: each solver's LATEST receipt.
    by_time = defaultdict(list)
    for r in rows:
        by_time[r.get("solverUsername")].append(r)
    latest_rows = {}
    for s, rs in by_time.items():
        rs2 = sorted(rs, key=lambda r: r.get("createdAt") or "")
        latest_rows[s] = rs2[-1].get("id")
    pool = [x for x in recs if x["id"] == latest_rows.get(x["solver"]) and x["neutral_prefill_corrected"]]
    pool.sort(key=lambda x: -x["neutral_prefill_corrected"])
    report.append("")
    report.append("D) prefill-corrected neutral, each solver's LATEST receipt (no selection on score)")
    report.append(
        f"  {'rank':>4} {'solver':<16} {'receipt':<10} {'day':<11} "
        f"{'corrected':>11} {'official':>11} {'n receipts':>10}"
    )
    for i, x in enumerate(pool[:12], start=1):
        report.append(
            f"  {i:>4} {x['solver']:<16} {str(x['id'])[:8]:<10} {x['day']:<11} "
            f"{x['neutral_prefill_corrected']:>11.6f} {x['official']:>11.6f} "
            f"{len(per_solver[x['solver']]):>10}"
        )
    results["e3_neutral_latest_receipt"] = pool[:12]

    # Ranking that averages run noise without selecting on the outcome:
    # the mean of everything a solver shipped on the last board day.
    LAST_DAY = max(x["day"] for x in recs)
    lastday = defaultdict(list)
    for x in recs:
        if x["day"] == LAST_DAY and x["neutral_prefill_corrected"] is not None:
            lastday[x["solver"]].append(x["neutral_prefill_corrected"])
    ed = []
    for s, vs in lastday.items():
        if len(vs) < 3:
            continue
        m = statistics.mean(vs)
        sd = statistics.stdev(vs)
        ed.append((m, sd / math.sqrt(len(vs)), len(vs), s, max(vs)))
    ed.sort(reverse=True)
    report.append("")
    report.append(
        f"E) prefill-corrected neutral, MEAN over everything a solver shipped on "
        f"{LAST_DAY} (n>=3, no selection on score, run noise averaged)"
    )
    report.append(
        f"  {'rank':>4} {'solver':<16} {'n':>3} {'mean':>11} {'sem':>9} {'best that day':>14}"
    )
    for i, (m, sem, n, s, mx) in enumerate(ed[:12], start=1):
        report.append(
            f"  {i:>4} {s:<16} {n:>3} {m:>11.6f} {sem:>9.6f} {mx:>14.6f}"
        )
    results["e3_neutral_lastday_mean"] = [
        {"solver": s, "n": n, "mean": m, "sem": sem, "best": mx} for m, sem, n, s, mx in ed[:12]
    ]
    report.append(
        "  caution: this mean includes each solver's failed experiments that day, "
        "so it measures what they shipped, not their best code."
    )

    # Best-of-n exposure of the top of ranking C.  A solver's receipt set is a
    # mixture of experiments, not ability plus noise, so the median is not an
    # ability estimate.  What matters is how repeatable the maximum is.
    report.append("")
    report.append("best-of-n exposure of the corrected top 12")
    report.append(
        f"  {'solver':<16} {'n':>4} {'max':>11} {'2nd':>11} {'max-2nd':>9} "
        f"{'within .01':>10} {'top2 mean':>11}"
    )
    top2 = []
    for x in t_corr:
        vs = sorted(
            (v["neutral_prefill_corrected"] for v in per_solver[x["solver"]]), reverse=True
        )
        second = vs[1] if len(vs) > 1 else float("nan")
        near = sum(1 for v in vs if v >= vs[0] - 0.01)
        t2 = statistics.mean(vs[:2]) if len(vs) > 1 else vs[0]
        report.append(
            f"  {x['solver']:<16} {len(vs):>4} {vs[0]:>11.6f} {second:>11.6f} "
            f"{vs[0] - second:>9.6f} {near:>10} {t2:>11.6f}"
        )
        top2.append({"solver": x["solver"], "n": len(vs), "max": vs[0], "second": second,
                     "n_within_0p01_of_max": near, "top2_mean": t2})
    results["e3_best_of_n_exposure"] = top2

    report.append("")
    report.append("F) ranking by the mean of each solver's two best corrected neutrals")
    report.append(f"  {'rank':>4} {'solver':<16} {'top2 mean':>11} {'n':>4}")
    allsolvers = []
    for s, xs in per_solver.items():
        vs = sorted((v["neutral_prefill_corrected"] for v in xs), reverse=True)
        if len(vs) < 2:
            continue
        allsolvers.append((statistics.mean(vs[:2]), s, len(vs)))
    allsolvers.sort(reverse=True)
    for i, (m, s, n) in enumerate(allsolvers[:12], start=1):
        report.append(f"  {i:>4} {s:<16} {m:>11.6f} {n:>4}")
    results["e3_neutral_top2_mean"] = [
        {"solver": s, "top2_mean": m, "n": n} for m, s, n in allsolvers[:12]
    ]

    # ------------------------------------------------- is the top separable?
    report.append("")
    report.append("-" * 78)
    report.append("E3c  is the top of the ranking separable at all?")
    report.append("-" * 78)
    top_span = t_corr[0]["neutral_prefill_corrected"] - t_corr[11]["neutral_prefill_corrected"]
    report.append(
        f"  spread of the corrected top 12: {top_span:.6f} "
        f"({100.0 * top_span / t_corr[0]['neutral_prefill_corrected']:.4f} %)"
    )
    report.append(
        f"  1st minus 2nd: "
        f"{t_corr[0]['neutral_prefill_corrected'] - t_corr[1]['neutral_prefill_corrected']:.6f}"
    )
    report.append(
        f"  1st minus 4th: "
        f"{t_corr[0]['neutral_prefill_corrected'] - t_corr[3]['neutral_prefill_corrected']:.6f}"
    )
    # Neutral keeps the whole run-level candidate shift, because it divides by a
    # fixed reference serial leg instead of the run's own serial leg.
    serial_run_sd_pct = 0.0990  # E164 Q1, common-mode serial sd across runs
    for beta, tag in ((1.914, "point estimate"), (0.0, "CI lower"), (4.14, "CI upper")):
        sd_pct = beta * serial_run_sd_pct
        report.append(
            f"  run-level neutral sd at beta={beta:.3f} ({tag}): {sd_pct:.4f} % = "
            f"{sd_pct / 100.0 * t_corr[0]['neutral_prefill_corrected']:.6f} score units"
        )
    results["e3c"] = {
        "top12_spread": top_span,
        "first_minus_second": t_corr[0]["neutral_prefill_corrected"]
        - t_corr[1]["neutral_prefill_corrected"],
        "first_minus_fourth": t_corr[0]["neutral_prefill_corrected"]
        - t_corr[3]["neutral_prefill_corrected"],
        "run_level_neutral_sd_units_at_beta_point": 1.914
        * serial_run_sd_pct
        / 100.0
        * t_corr[0]["neutral_prefill_corrected"],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    text = "\n".join(report)
    print(text)
    with open(args.out.replace(".json", ".txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
