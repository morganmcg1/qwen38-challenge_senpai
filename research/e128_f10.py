#!/usr/bin/env python3
"""E128-F10 - Finding 153 replicated, then the advisor's three queued items.

harness=ranked. Zero GPU. Analysis only. Run from `research/`.

Section 0 replicates Finding 153 from the same 806-row board population the
advisor used, so that Rule 100 and the carrier weights rest on a check I ran
myself rather than on a quoted table.

Section 1 re-reads the E128 headline with beagle as a first-class line and
with the second half of the median treated as a `min` over four prompts.
Every arm number here comes from the candidate leg, which Rule 100 prefers.

Section 2 answers queue item 1. It publishes the ranked per-prompt width
histograms weighted by measured carrier share and prices thorfinn's one-pass
tables in both competing frames, so the frame question is settled by measured
ranked mass rather than by argument.

Section 3 answers queue item 2, the one-parameter `row_keyed(M) = 38/IPG(M)`
shape.

Section 4 answers queue item 3, the `costModelDepth` fixed term, with the
runner state entered as a categorical factor instead of a free scalar.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from e128_arm_prices import (
    ONEPASS_RESIDENCY_LOSS,
    TEMPLATING,
    median_of,
    single_pass_curve,
)
from e128_f9 import RK_STATEMENTS, TABLE_BOARD, TABLE_OURS, bar, ipg_of, run, show
from e128_reprice_onepass import F_HAT, F_SE, onepass_fixed
from e128_ourcurve import (
    F83_WEIGHT,
    build_points,
    curve_us,
    fixture_histograms,
    load_receipt,
    prompt_probs,
    r_scenarios,
)
from e128_slopes import ROWS
from e128_state_fe import attach_strata, build_panel, fe_fit
from e128_state_s45 import kmeans1d
from e128_strata_curve import MAXM

PROMPT_BY_SHA = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
ORDER = ["beagle", "medicine", "essays", "republic", "botany",
         "travel", "plutarch", "drama"]

# The advisor's approximation: beagle carries one half of the median and the
# other half is the slowest of four. Section 0 measures how good it is.
MIN_FOUR = ["medicine", "essays", "republic", "botany"]

ONEPASS = {"{6:6}": [6], "{6:6,7:7}": [6, 7], "{6:6,7:7,8:8}": [6, 7, 8]}
S_ALPHONSE = 930.0
F76_THRESHOLD = -12.9


def board_rows(path: Path) -> list[dict]:
    """Every board row that carries complete per-prompt official metrics."""
    raw = json.load(open(path))["submissions"]
    out = []
    for row in raw:
        met = row.get("officialMetrics")
        if not met:
            continue
        per = {}
        for entry in met.get("per_prompt") or []:
            name = PROMPT_BY_SHA.get(entry["prompt_sha256"][:8])
            if name is None:
                continue
            per[name] = entry
        if len(per) != 8:
            continue
        out.append({"id": (row.get("submissionCommitSha")
                           or row.get("id") or "")[:8],
                    "solver": row.get("solverUsername"),
                    "score": row.get("officialScore"),
                    "per": per})
    return out


# ----------------------------------------------------------------- section 0

def section0(rows: list[dict]) -> dict:
    print("=== F10.0  Finding 153 replicated on the same population ===")
    print("  %d board rows carry complete per-prompt official metrics" % len(rows))

    print("\n## the pinned serial numerator, which no candidate edit can move")
    print("  %-10s %5s %12s %7s %10s %10s"
          % ("prompt", "n", "mean s/tok", "sd %", "p5-p95 %", "min-max %"))
    serial_tbl = {}
    for name in ORDER:
        v = np.array([r["per"][name]["serial_seconds_per_token_mean"]
                      for r in rows])
        mean = float(v.mean())
        sd = 100.0 * float(v.std(ddof=1)) / mean
        p5, p95 = np.percentile(v, [5, 95])
        span_p = 100.0 * (p95 - p5) / mean
        span_m = 100.0 * (v.max() - v.min()) / mean
        serial_tbl[name] = {"mean": mean, "sd_pct": sd, "p5_p95_pct": span_p,
                            "min_max_pct": span_m}
        print("  %-10s %5d %12.6f %7.3f %10.3f %10.3f"
              % (name, len(v), mean, sd, span_p, span_m))

    mat = np.array([[r["per"][n]["serial_seconds_per_token_mean"]
                     for n in ORDER] for r in rows])
    run_mean = mat.mean(axis=1)
    run_sd = 100.0 * float(run_mean.std(ddof=1)) / float(run_mean.mean())
    resid = mat / run_mean[:, None] - 1.0
    resid_sd = 100.0 * float(resid.std(ddof=1))
    print("\n  run-level serial mean sd      %.4f %%" % run_sd)
    print("  within-run per-prompt resid sd %.4f %%" % resid_sd)
    print("  the draw is per pair, not per run: the within-run term is %.2fx"
          % (resid_sd / run_sd))

    print("\n## which prompts occupy median order statistics 4 and 5")
    carrier = Counter()
    approx_err = []
    for r in rows:
        raws = {n: r["per"][n]["raw_ratio_of_means"] for n in ORDER}
        ordered = sorted(raws, key=lambda n: raws[n])
        for n in ordered[3:5]:
            carrier[n] += 1
        exact = 0.5 * (raws[ordered[3]] + raws[ordered[4]])
        approx = 0.5 * raws["beagle"] + 0.5 * min(raws[n] for n in MIN_FOUR)
        approx_err.append(100.0 * (approx / exact - 1.0))
    weights = {}
    for name, cnt in carrier.most_common():
        weights[name] = cnt / len(rows)
        print("  %-10s %5d / %d  %6.1f %%" % (name, cnt, len(rows),
                                              100.0 * cnt / len(rows)))
    err = np.array(approx_err)
    print("\n  approximation  0.5*beagle + 0.5*min(medicine,essays,"
          "republic,botany)")
    print("    mean error %+.4f %%   sd %.4f %%   max abs %.4f %%   "
          "exact on %d / %d rows"
          % (err.mean(), err.std(ddof=1), np.abs(err).max(),
             int((np.abs(err) < 1e-12).sum()), len(err)))

    print("\n## Rule 100: rank the frontier cluster on a common serial leg")
    top = sorted(rows, key=lambda r: -(r["score"] or 0.0))[:12]
    ref = np.array([serial_tbl[n]["mean"] for n in ORDER])
    print("  %-10s %-14s %12s %12s %8s"
          % ("id", "solver", "published", "common-serial", "rank move"))
    common = []
    for r in top:
        cand = np.array([r["per"][n]["mtp_seconds_per_token_mean"]
                         for n in ORDER])
        common.append((r, float(median_of(list(ref / cand)))))
    pub_rank = {r["id"]: i for i, (r, _) in
                enumerate(sorted(common, key=lambda t: -(t[0]["score"] or 0)))}
    com_rank = {r["id"]: i for i, (r, _) in
                enumerate(sorted(common, key=lambda t: -t[1]))}
    for r, cm in sorted(common, key=lambda t: -(t[0]["score"] or 0)):
        print("  %-10s %-14s %12.8f %12.8f %+8d"
              % (r["id"], (r["solver"] or "")[:14], r["score"] or 0.0, cm,
                 pub_rank[r["id"]] - com_rank[r["id"]]))
    return {"serial": serial_tbl, "run_sd_pct": run_sd,
            "resid_sd_pct": resid_sd, "carrier_share": weights,
            "approx_mean_err_pct": float(err.mean()),
            "approx_sd_err_pct": float(err.std(ddof=1)),
            "approx_max_abs_err_pct": float(np.abs(err).max()),
            "approx_exact_rows": int((np.abs(err) < 1e-12).sum()),
            "common_serial": [{"id": r["id"], "solver": r["solver"],
                               "published": r["score"], "common": cm}
                              for r, cm in common]}


# ----------------------------------------------------------------- section 1

def section1(pricing: dict, rows: list[dict]) -> dict:
    print("\n=== F10.1  the E128 arms re-read with beagle first-class ===")
    print("  every column below is a CANDIDATE-LEG quantity, so Rule 100's")
    print("  0.0967 % serial-numerator term does NOT apply to it. The exact")
    print("  Rule 67 median column is a ratio, so it does carry that term.")
    gains = pricing["per_prompt_candidate_gain_pct"]
    med = pricing["median_gain_pct_vs_ship"]

    # Ordering robustness. The arm's per-prompt candidate effect is held
    # fixed and applied to every board row's own realised raw vector, so the
    # spread below is caused ONLY by which prompts carry that row's median.
    # It is not a claim about anybody else's candidate.
    def replay(arm: str) -> tuple[float, float, float]:
        mult = {n: 1.0 + gains[arm][n] / 100.0 for n in ORDER}
        deltas = []
        for r in rows:
            raws = {n: r["per"][n]["raw_ratio_of_means"] for n in ORDER}
            old = median_of(list(raws.values()))
            new = median_of([raws[n] * mult[n] for n in ORDER])
            deltas.append(100.0 * (new / old - 1.0))
        v = np.array(deltas)
        return float(v.mean()), float(v.std(ddof=1)), float(np.median(v))

    order = sorted(med, key=lambda k: -med[k])
    print("\n  %-16s %9s %9s %9s %9s %8s  %s"
          % ("arm", "median", "beagle", "min4", "replay", "sd",
             "the four, worst first"))
    out = {}
    for arm in order:
        g = gains[arm]
        four = {n: g[n] for n in MIN_FOUR}
        worst = min(four, key=lambda n: four[n])
        mean, sd, _ = replay(arm)
        out[arm] = {"median_pct": med[arm], "beagle_pct": g["beagle"],
                    "min4_pct": four[worst], "min4_prompt": worst,
                    "replay_mean_pct": mean, "replay_sd_pct": sd}
        print("  %-16s %+9.4f %+9.4f %+9.4f %+9.4f %8.4f  %s"
              % (arm, med[arm], g["beagle"], four[worst], mean, sd,
                 " ".join("%s %+.3f" % (n[:4], four[n])
                          for n in sorted(four, key=lambda n: four[n]))))
    print("\n  'replay' holds each arm's per-prompt candidate effect fixed and")
    print("  recomputes the exact Rule 67 median on all %d board raw vectors,"
          % len(rows))
    print("  so it prices the arm against the POPULATION carrier structure")
    print("  instead of against the single ordering our own receipt drew.")
    return out


# ----------------------------------------------------------------- section 2

def bench_histogram(hists: dict) -> np.ndarray:
    """The local benchfixture width histogram, over M = draft depth + 1.

    `fixture_histograms` counts recorded draft depths per fixture. The local
    `benchfixture` leg maps to no ranked prompt, which is exactly why it is
    the frame askeladd priced on.
    """
    counter = hists["benchfixture"]
    probs = np.zeros(MAXM)
    for depth, count in counter.items():
        probs[min(int(depth), MAXM - 1)] += count
    return probs / probs.sum()


def hist_stats(probs: np.ndarray) -> dict:
    rows = np.arange(1, len(probs) + 1, dtype=float)
    return {"mean_M": float((probs * rows).sum()),
            "p_M_eq_8": float(probs[7]) if len(probs) > 7 else 0.0,
            "p_M_ge_6": float(probs[5:].sum())}


def _break_mult(widths, curve, single, c):
    """The WITHDRAWN F4 3b reading: the whole tier break is a pass cost."""
    loss = 1.0 + c * ONEPASS_RESIDENCY_LOSS * TEMPLATING["round_share"]

    def mult(m: float) -> float:
        if int(m) not in widths:
            return 1.0
        return curve_us(single, m) / curve_us(curve, m) * loss
    return mult


def price_frames(points: dict, curve: dict, mult, forced) -> dict:
    """Median delta for a per-width multiplier, optionally forcing one shape.

    `forced` replaces every prompt's own width histogram with a single shared
    one. That is askeladd's local benchfixture frame. `forced=None` keeps each
    prompt's measured ranked mass, which is the ranked frame.
    """
    rows = np.arange(1, MAXM + 1, dtype=float)
    per = {}
    for prompt, point in points.items():
        probs = np.array(point["hist"]["probs"]) if forced is None else forced
        base = float(sum(p * curve_us(curve, m) for p, m in zip(probs, rows)))
        new = float(sum(p * curve_us(curve, m) * mult(m)
                        for p, m in zip(probs, rows)))
        per[prompt] = {"raw": point["raw"] * base / new,
                       "delta_pct": 100.0 * (base / new - 1.0)}
    old = median_of([p["raw"] for p in points.values()])
    new = median_of([p["raw"] for p in per.values()])
    return {"median_delta_pct": 100.0 * (new / old - 1.0),
            "beagle_delta_pct": per["beagle"]["delta_pct"],
            "per_prompt": {k: v["delta_pct"] for k, v in per.items()}}


def section2(points: dict, curve: dict, bench: np.ndarray,
             carrier: dict) -> dict:
    print("\n=== F10.2  ranked width histograms and the frame question ===")

    print("\n## ranked per-prompt mass, carrier order, beagle first")
    print("  %-10s %8s %8s %8s %8s   %s"
          % ("prompt", "carrier", "mean M", "P(M=8)", "P(M>=6)", "probs 1..9"))
    ranked = {}
    for name in ORDER:
        probs = np.array(points[name]["hist"]["probs"])
        st = hist_stats(probs)
        ranked[name] = st | {"probs": [float(x) for x in probs]}
        print("  %-10s %7.1f%% %8.4f %8.4f %8.4f   %s"
              % (name, 100.0 * carrier.get(name, 0.0), st["mean_M"],
                 st["p_M_eq_8"], st["p_M_ge_6"],
                 " ".join("%.3f" % x for x in probs)))

    cw = np.zeros(MAXM)
    tot = 0.0
    for name in ORDER:
        w = carrier.get(name, 0.0)
        cw += w * np.array(points[name]["hist"]["probs"])
        tot += w
    cw /= tot
    f83 = np.zeros(MAXM)
    for name in ORDER:
        f83 += F83_WEIGHT[name] * np.array(points[name]["hist"]["probs"])
    f83 /= f83.sum()
    print("\n  %-22s %8s %8s %8s" % ("aggregate", "mean M", "P(M=8)", "P(M>=6)"))
    aggs = {"carrier-weighted (F10)": cw, "F83-weighted": f83,
            "benchfixture (local)": bench}
    for tag, vec in aggs.items():
        st = hist_stats(vec)
        print("  %-22s %8.4f %8.4f %8.4f"
              % (tag, st["mean_M"], st["p_M_eq_8"], st["p_M_ge_6"]))
    print("\n  beagle alone: mean M %.4f, P(M=8) %.4f."
          % (ranked["beagle"]["mean_M"], ranked["beagle"]["p_M_eq_8"]))
    print("  The local benchfixture puts %.1f%% of its mass at M=8; the"
          % (100.0 * hist_stats(bench)["p_M_eq_8"]))
    print("  carrier-weighted ranked mass puts %.1f%% there, and beagle %.1f%%."
          % (100.0 * hist_stats(cw)["p_M_eq_8"],
             100.0 * ranked["beagle"]["p_M_eq_8"]))

    print("\n## thorfinn's one-pass tables, priced in both frames")
    print("  The pass saving is priced at the BOARD-MEASURED pass price")
    print("  f = %.1f +- %.1f us from F7, not at the fitted tier break. F7"
          % (F_HAT, F_SE))
    print("  withdrew the tier-break reading, so it is shown only as the")
    print("  last column, to make the size of that error explicit.")
    print("\n  %-16s %5s %11s %11s %11s %11s"
          % ("table", "c", "ranked med", "ranked bgl", "bench frame",
             "WITHDRAWN"))
    prices = {}
    single = single_pass_curve(curve)
    for tag, widths in ONEPASS.items():
        for c in (0.0, 0.445):
            mult = onepass_fixed(widths, curve, F_HAT, c, False)
            rk = price_frames(points, curve, mult, None)
            bf = price_frames(points, curve, mult, bench)
            old = price_frames(points, curve, _break_mult(widths, curve,
                                                          single, c), None)
            prices["%s c=%.3f" % (tag, c)] = {
                "ranked_median_pct": rk["median_delta_pct"],
                "ranked_beagle_pct": rk["beagle_delta_pct"],
                "bench_frame_median_pct": bf["median_delta_pct"],
                "withdrawn_tier_break_pct": old["median_delta_pct"],
                "ranked_per_prompt": rk["per_prompt"]}
            print("  %-16s %5.3f %+11.4f %+11.4f %+11.4f %+11.4f"
                  % (tag, c, rk["median_delta_pct"], rk["beagle_delta_pct"],
                     bf["median_delta_pct"], old["median_delta_pct"]))

    print("\n  f is statistically zero, so the honest interval is f +- 1 SE:")
    print("  %-16s %5s %11s %11s"
          % ("table", "c", "f - 1 SE", "f + 1 SE"))
    for tag, widths in ONEPASS.items():
        for c in (0.0, 0.445):
            lo = price_frames(points, curve,
                              onepass_fixed(widths, curve,
                                            max(F_HAT - F_SE, 0.0), c, False),
                              None)["median_delta_pct"]
            hi = price_frames(points, curve,
                              onepass_fixed(widths, curve, F_HAT + F_SE, c,
                                            False),
                              None)["median_delta_pct"]
            prices["%s c=%.3f" % (tag, c)]["f_band_pct"] = [lo, hi]
            print("  %-16s %5.3f %+11.4f %+11.4f" % (tag, c, lo, hi))
    return {"ranked": ranked,
            "aggregates": {k: hist_stats(v) | {"probs": [float(x) for x in v]}
                           for k, v in aggs.items()},
            "onepass_prices": prices}


# ----------------------------------------------------------------- section 3

def rk_row(table: dict) -> np.ndarray:
    """`38 / IPG(M)`: row-keyed statements per verified row, not per round."""
    return np.array([RK_STATEMENTS / ipg_of(table, int(m)) for m in ROWS])


def section3(pts: list[dict], s_known: float) -> dict:
    print("\n=== F10.3  the one-parameter row-keyed shape 38 / IPG(M) ===")
    print("\n  M                " + " ".join("%7.0f" % m for m in ROWS))
    for tag, tbl in (("ours ", TABLE_OURS), ("board", TABLE_BOARD)):
        print("  38/IPG %s     " % tag
              + " ".join("%7.2f" % v for v in rk_row(tbl)))
    print("\n  This is a per-ROW cost, so it does not grow with M. It falls as")
    print("  the template packs more rows per group and jumps back up when the")
    print("  table drops IPG. Ours jumps at M=6, the board at M=5. One free")
    print("  scale reproduces a level jump at the right place with no fitted")
    print("  break location and no fitted jump size.")

    one = np.ones_like(ROWS)
    rk_o = rk_row(TABLE_OURS)
    phi = np.array([p["phi"] for p in pts])
    step = (ROWS >= 6).astype(float)
    families = [
        ("line a+bM", np.stack([one, ROWS], axis=1), ["a", "b"]),
        ("row-keyed only", np.stack([one, rk_o], axis=1), ["a", "q"]),
        ("row-keyed + M", np.stack([one, ROWS, rk_o], axis=1), ["a", "b", "q"]),
        ("row-keyed x M", np.stack([one, ROWS, rk_o * ROWS], axis=1),
         ["a", "b", "q"]),
        ("R free break M>=6",
         np.stack([one, ROWS, step, step * (ROWS - 6)], axis=1),
         ["a", "b", "jump", "dslope"]),
    ]
    out = {}
    for tag, offs in (("raw", None), ("known s=930", s_known * phi)):
        print("\n## %s" % tag)
        got = []
        for name, cols, cn in families:
            fit = run(pts, cols, offs, name)
            show(fit, cn)
            got.append({"name": name, "k": fit["params"], "rmse": fit["rmse"],
                        "bic": fit["bic"],
                        "beta": [float(b) for b in fit["beta"]]})
        out[tag] = got
        best = min(got, key=lambda r: r["bic"])
        print("  best BIC: %s (%.2f)" % (best["name"], best["bic"]))
    return out


def section3_board(s_known: float) -> dict:
    print("\n=== F10.3b  the row-keyed shape on the board population ===")
    panel = attach_strata(build_panel()["panel"])
    tables = json.load(open("/tmp/e128_strata.json"))["tables"]
    tbl_by_sid = {sid[:8]: {int(m): int(v) for m, v in t.items()}
                  for sid, t in tables.items()}
    y = np.array([p["round_us"] for p in panel])
    phi = np.array([p["phi"] for p in panel])
    mb = np.array([p["mbar"] for p in panel])
    gb = np.array([p["gbar"] for p in panel])
    rkbar = np.array([bar({m: RK_STATEMENTS / ipg_of(tbl_by_sid[p["sid"]], m)
                           for m in range(1, MAXM + 1)}, p["mbar"])
                      for p in panel])
    hinge = np.maximum(mb - 4.375, 0.0)
    groups = [[p["sid"] for p in panel], [p["prompt"] for p in panel]]
    cluster = [p["sid"] for p in panel]
    designs = [
        ("M only", np.stack([mb], axis=1), ["b"]),
        ("row-keyed only", np.stack([rkbar], axis=1), ["q"]),
        ("row-keyed + M", np.stack([mb, rkbar], axis=1), ["b", "q"]),
        ("P  M + passes", np.stack([mb, gb], axis=1), ["b", "f"]),
        ("R  M + hinge4.375", np.stack([mb, hinge], axis=1), ["b_lo", "d"]),
        ("R + row-keyed", np.stack([mb, hinge, rkbar], axis=1),
         ["b_lo", "d", "q"]),
    ]
    out = []
    for name_off, offs in (("raw", None), ("known s=930", s_known * phi)):
        print("\n## %s, row+prompt FE, SEs clustered by row" % name_off)
        yy = y if offs is None else y - offs
        for name, x, cn in designs:
            fit = fe_fit(yy, x, groups, cluster=cluster, names=cn)
            print("  %-20s rmse %8.1f aicc %10.1f  %s"
                  % (name, fit["rmse"], fit["aicc"],
                     " ".join("%s %8.1f+-%-7.1f" % (c, b, s)
                              for c, b, s in zip(cn, fit["beta"], fit["se"]))))
            out.append({"offset": name_off, "name": name, "rmse": fit["rmse"],
                        "aicc": fit["aicc"],
                        "beta": [float(b) for b in fit["beta"]]})
    print("\n  corr(row-keyed, mbar) %.4f   corr(row-keyed, gbar) %.4f"
          % (float(np.corrcoef(rkbar, mb)[0, 1]),
             float(np.corrcoef(rkbar, gb)[0, 1])))
    return out


# ----------------------------------------------------------------- section 4

def section4(k_levels: int = 3) -> dict:
    print("\n=== F10.4  the state term as a categorical factor, not a scalar ===")
    wanted = Path("e130-artifacts/rung8-state-model.json")
    print("  alphonse's research/%s is ABSENT from this branch, and his E130"
          % wanted.as_posix())
    print("  branch is outside my read scope for this launch. I therefore")
    print("  recover the same factor from the public board myself and publish")
    print("  the labels, so he can check them against his file. The label")
    print("  source is the F76 mode index, which F8 checked against the ALS")
    print("  state assignment and matched on 13 of 13 rows.")

    panel = attach_strata(build_panel()["panel"])
    # The label source is the F76 mode index, an independent classifier that
    # F8 already validated: it agreed with the ALS state assignment on 13 of
    # 13 rows of the family it could check. Clustering the index is therefore
    # not a blind state fit. A full-panel ALS is NOT usable here: with eight
    # prompts per row the scale `c_i` and the state `s_i` are near collinear,
    # and the recovered levels run to tens of thousands of microseconds.
    idx_by_sid = {}
    for p in panel:
        idx_by_sid.setdefault(p["sid"], p["f76_index"])
    sids = sorted(idx_by_sid)
    idx = np.array([idx_by_sid[s] for s in sids], dtype=float)
    centres, wss, tss = kmeans1d(idx, k_levels)
    labels = np.argmin(np.abs(idx[:, None] - centres[None, :]), axis=1)
    print("\n  %d rows, labelled by F76 mode index into %d levels:"
          % (len(sids), k_levels))
    for j in range(k_levels):
        n = int((labels == j).sum())
        print("    mode %d  index centre %+9.3f  %4d rows"
              % (j, centres[j], n))
    print("  index variance explained %.3f"
          % (1.0 - wss / max(tss, 1e-12)))

    lab_by_sid = {s: int(l) for s, l in zip(sids, labels)}
    y = np.array([p["round_us"] for p in panel])
    mb = np.array([p["mbar"] for p in panel])
    phi = np.array([p["phi"] for p in panel])
    gb = np.array([p["gbar"] for p in panel])
    hinge = np.maximum(mb - 4.375, 0.0)
    dummies = [np.array([1.0 if lab_by_sid[p["sid"]] == j else 0.0
                         for p in panel]) * phi
               for j in range(1, k_levels)]
    names_s = ["s%d" % j for j in range(1, k_levels)]
    # Row FE absorbs any per-row constant, so the state enters only through
    # its interaction with the drafting-round share. A common `phi` slope is
    # carried separately, which makes s_k a DIFFERENCE from mode 0.
    groups = [[p["sid"] for p in panel], [p["prompt"] for p in panel]]
    cluster = [p["sid"] for p in panel]

    print("\n## row+prompt FE; the mode factor enters as s_k * phi")
    print("   s_k is a difference from mode 0, in us per drafting round")
    designs = [
        ("M + phi", [mb, phi], ["b", "a"]),
        ("M + phi + mode", [mb, phi] + dummies, ["b", "a"] + names_s),
        ("M + hinge + phi + mode", [mb, hinge, phi] + dummies,
         ["b_lo", "d", "a"] + names_s),
        ("M + pass + phi + mode", [mb, gb, phi] + dummies,
         ["b", "f", "a"] + names_s),
    ]
    out_fits = []
    for name, cols, cn in designs:
        fit = fe_fit(y, np.stack(cols, axis=1), groups, cluster=cluster,
                     names=cn)
        print("  %-24s rmse %8.1f aicc %10.1f  %s"
              % (name, fit["rmse"], fit["aicc"],
                 " ".join("%s %8.1f+-%-7.1f" % (c, b, s)
                          for c, b, s in zip(cn, fit["beta"], fit["se"]))))
        out_fits.append({"name": name, "rmse": fit["rmse"],
                         "aicc": fit["aicc"],
                         "beta": [float(b) for b in fit["beta"]],
                         "se": [float(v) for v in fit["se"]],
                         "names": cn})

    # The validated F76 classifier is a two-way rule at index -12.9, not a
    # three-way k-means. Report it as well, so the label choice is visible.
    slow = np.array([1.0 if idx_by_sid[p["sid"]] < F76_THRESHOLD else 0.0
                     for p in panel]) * phi
    n_slow = sum(1 for s in sids if idx_by_sid[s] < F76_THRESHOLD)
    two = fe_fit(y, np.stack([mb, phi, slow], axis=1), groups,
                 cluster=cluster, names=["b", "a", "s_slow"])
    print("\n  the VALIDATED two-way F76 rule, index < %.1f (%d of %d rows):"
          % (F76_THRESHOLD, n_slow, len(sids)))
    print("  %-24s rmse %8.1f aicc %10.1f  %s"
          % ("M + phi + F76 slow", two["rmse"], two["aicc"],
             " ".join("%s %8.1f+-%-7.1f" % (c, b, s)
                      for c, b, s in zip(["b", "a", "s_slow"],
                                         two["beta"], two["se"]))))
    out_fits.append({"name": "M + phi + F76 slow", "rmse": two["rmse"],
                     "aicc": two["aicc"], "n_slow": n_slow,
                     "beta": [float(b) for b in two["beta"]],
                     "se": [float(v) for v in two["se"]],
                     "names": ["b", "a", "s_slow"]})
    print("  Only %d of %d table-bearing rows fall on the far side of the"
          % (n_slow, len(sids)))
    print("  validated threshold, so THIS PANEL CANNOT IDENTIFY the state")
    print("  step with alphonse's own rule. The three-way k-means split is")
    print("  finer than the validated rule and its labels are therefore not")
    print("  his. Note also that the widest index gap, mode 0 to mode 2,")
    print("  carries no step at all, which is what a contaminated label")
    print("  looks like. Item 3 needs his file or a mode-spanning panel.")

    main_fit = out_fits[1]
    steps = [abs(main_fit["beta"][2 + j]) for j in range(k_levels - 1)]
    steps.append(abs(float(two["beta"][2])))
    span = max(steps) if steps else 0.0
    ours = [s for s in sids if s.startswith("d3c491b5")]
    if ours:
        print("\n  our curve receipt d3c491b5 sits in mode %d"
              % lab_by_sid[ours[0]])
    print("\n## the costModelDepth answer, restated under the factor")
    print("  F9.4 priced the missing per-drafting-round term at the known")
    print("  930 us and found the ranked median delta is exactly zero: the")
    print("  M=6 pass cliff on our curve is 15,645 us, 16.8x that term, and")
    print("  the smallest cost that would move ANY prompt's chosen depth is")
    print("  10,525 us, 11.3x it.")
    print("  The largest mode difference recovered here is %.1f us per" % span)
    print("  drafting round. %s"
          % ("That is still %.1fx below the 10,525 us threshold, so the "
             "lever stays closed." % (10525.0 / span) if span > 1e-9
             else "That is zero, so the lever stays closed."))
    print("  The factor form does not change the F9.4 conclusion. It does")
    print("  change the ERROR BAR: fitting s blind on one receipt gave a")
    print("  plutarch dummy, and this design cannot, because the mode label")
    print("  comes from outside the fit.")
    return {"sids": sids, "f76_index": {s: float(idx_by_sid[s]) for s in sids},
            "mode_of": lab_by_sid,
            "index_centres": [float(c) for c in centres],
            "index_variance_explained": float(1.0 - wss / max(tss, 1e-12)),
            "mode_steps_us": [float(s) for s in steps],
            "span_us": float(span), "fits": out_fits}


# --------------------------------------------------------------------- main

def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path,
                    default=Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--identity", type=Path,
                    default=here / "e128-artifacts/rung0-identity.json")
    ap.add_argument("--shipped", type=Path,
                    default=here / "e128-artifacts/rung1-shipped.json")
    ap.add_argument("--curves", type=Path,
                    default=here / "e128-artifacts/f4-candidate-curves.json")
    ap.add_argument("--pricing", type=Path,
                    default=here / "e128-artifacts/rung2-ours-pricing.json")
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--skip-board", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    print("harness=ranked  E128-F10  zero GPU  analysis only\n")
    rows = board_rows(args.board)
    doc = {"harness": "ranked", "n_rows": len(rows)}
    doc["section0"] = section0(rows)

    pricing = json.loads(args.pricing.read_text())
    doc["section1"] = section1(pricing, rows)

    hists = fixture_histograms(args.shipped)
    scen = r_scenarios(args.identity)
    receipt = load_receipt(args.board, args.receipt)
    ordered = build_points(receipt, scen["assumed"], hists)
    for point in ordered:
        row = receipt["per_prompt"][point["prompt"]]
        point["raw"] = row["raw"]
        point["n0"] = row["non_drafting"]
        point["phi"] = 1.0 - float(np.array(prompt_probs(point, True))[0])
    points = {p["prompt"]: p for p in ordered}
    curves = json.loads(args.curves.read_text())["curves"]
    curve = curves["slopeonly_b6"]
    doc["section2"] = section2(points, curve, bench_histogram(hists),
                               doc["section0"]["carrier_share"])

    doc["section3"] = section3(ordered, S_ALPHONSE)
    if not args.skip_board:
        doc["section3_board"] = section3_board(S_ALPHONSE)
        doc["section4"] = section4()

    if args.json:
        args.json.write_text(json.dumps(doc, indent=2) + "\n")
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
