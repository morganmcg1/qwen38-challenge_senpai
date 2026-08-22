#!/usr/bin/env python3
"""E128 section 5 - fit OUR ranked round-cost curve from OUR official receipts.

harness=ranked.

F97's curve was fitted over 147 official runs of other solvers. Route B
(`d3c491b5`) changed our candidate and did not change theirs, so the board
curve is no longer our curve. This script fits the curve from our own
receipts, locates the tier step empirically, and writes the fitted curve in
the shape `e128_price.py --curve-json` consumes.

Two things make a naive fit wrong, and both are handled here.

1. `round_us = candidate_seconds_per_token * 512e6 / R` needs the ranked round
   count `R`, which is not published. Every fit is therefore repeated over the
   four R scenarios pinned in `rung0-identity.json`.

2. The published width is the mean `M̄ = effective_mean_draft_len + 1`, the
   cost curve is convex, and the realised per-round width distribution is
   strongly bimodal. Fitting through `M̄` therefore carries Jensen bias in the
   same direction as Hypothesis J. The `dist` fit replaces `f(M̄)` with
   `E_hist[f(M)]` over a per-prompt width histogram, which removes that bias
   exactly, because the piecewise-linear cost is linear in its parameters and
   the expectation of a linear form is a linear form.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

DECODE_TOKENS = 512
MAX_ROWS = 9  # M = d + 1 with d in 0..8; the shipped cap keeps d <= 7.

PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

# F83 marginal weights, as used everywhere else in E128.
F83_WEIGHT = {
    "beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598, "botany": 0.0124,
    "republic": 0.0100, "drama": 0.0, "travel": 0.0, "plutarch": 0.0,
}

# The E124 prose fixtures that supply a realised per-round width histogram.
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

# F97, fitted over 147 official runs of other solvers. Kept as the control.
BOARD_CURVE = {"breakpoint": 5, "lo": (27215.4, 3966.4), "hi": (17020.7, 7154.2)}


# ----------------------------------------------------------------- receipts

def load_receipt(board: Path, prefix: str) -> dict:
    rows = json.loads(board.read_text())
    if isinstance(rows, dict):
        rows = rows["submissions"]
    for row in rows:
        if row["id"].startswith(prefix):
            metrics = row["officialMetrics"]
            per_prompt = {}
            for entry in metrics["per_prompt"]:
                name = PROMPT_NAMES[entry["prompt_sha256"][:8]]
                per_prompt[name] = {
                    "candidate": entry["mtp_seconds_per_token_mean"],
                    "serial": entry["serial_seconds_per_token_mean"],
                    "raw": entry["raw_ratio_of_means"],
                    "draft_len": entry["effective_mean_draft_len"],
                    "non_drafting": entry["non_drafting_round_count"],
                }
            return {
                "id": row["id"], "score": row["officialScore"],
                "status": row["status"], "mode": metrics.get("mode"),
                "commit": metrics.get("commit"), "per_prompt": per_prompt,
            }
    raise SystemExit("no board row with id prefix %r" % prefix)


def r_scenarios(identity: Path) -> dict:
    """The four R vectors E128-F2 requires, from the F111 pinning."""
    pred = json.loads(identity.read_text())["manifold"]["predictions"]
    out = {"predicted": {}, "assumed": {}, "band_lo": {}, "band_hi": {}}
    for prompt, row in pred.items():
        r_pred = row["R_predicted"]
        band = row["R_band"]
        if r_pred is None or math.isnan(r_pred):
            # plutarch is outside the local eff range, so the manifold cannot
            # predict it. Its analytic bound from `non_drafting_round_count`
            # is exact, so the bound's midpoint and edges stand in.
            lo, hi = row["R_bound_from_non_drafting_rounds"]
            r_pred, band = 0.5 * (lo + hi), [lo, hi]
        out["predicted"][prompt] = r_pred
        out["assumed"][prompt] = float(row["R_assumed"])
        out["band_lo"][prompt] = band[0]
        out["band_hi"][prompt] = band[1]
    return out


# ---------------------------------------------------------------- histograms

def fixture_histograms(shipped: Path) -> dict:
    legs = json.loads(shipped.read_text())["legs"]
    out: dict[str, Counter] = {}
    for leg in legs:
        counter = out.setdefault(leg["prompt_id"], Counter())
        for record in leg["rounds_detail"]:
            counter[record["depth"]] += 1
    return out


def tilt_to_mean(weights: np.ndarray, support: np.ndarray,
                 target: float) -> np.ndarray:
    """Max-entropy exponential tilt of a histogram onto a target mean.

    `p_k` proportional to `w_k * exp(lam * k)` is the distribution closest to
    `w` in KL divergence among those with the required mean, so the fixture's
    shape is preserved as far as the moment constraint allows. A shift-only
    rule would have to clamp at the ends and would then miss the mean.
    """
    lo, hi = support[0], support[-1]
    if not lo - 1e-9 <= target <= hi + 1e-9:
        raise SystemExit("target mean %.4f outside support [%g, %g]"
                         % (target, lo, hi))
    left, right = -60.0, 60.0
    for _ in range(200):
        lam = 0.5 * (left + right)
        tilted = weights * np.exp(lam * (support - support.mean()))
        mean = float((tilted * support).sum() / tilted.sum())
        if mean < target:
            left = lam
        else:
            right = lam
    lam = 0.5 * (left + right)
    tilted = weights * np.exp(lam * (support - support.mean()))
    return tilted / tilted.sum()


def prompt_width_histogram(prompt: str, mbar: float, non_drafting: int,
                           rounds: float, hists: dict) -> dict:
    """Per-round verify-width distribution for one ranked prompt.

    The fixture supplies the shape and the receipt supplies the mean. For
    plutarch the receipt also supplies the exact non-drafting share, which no
    fixture can reproduce, so that mass is placed at `M = 1` first and the
    fixture shape is tilted onto the conditional mean of the rest.
    """
    counter = Counter()
    for fixture in PROMPT_FIXTURES[prompt]:
        counter.update(hists[fixture])
    support = np.arange(0, MAX_ROWS, dtype=float)  # depth d
    weights = np.array([float(counter.get(int(d), 0)) for d in support])
    weights = np.maximum(weights, 0.0)

    share = min(max(non_drafting / rounds, 0.0), 1.0) if rounds > 0 else 0.0
    target_depth = mbar - 1.0
    if share > 0.0:
        if share >= 1.0 - 1e-9:
            probs = np.zeros_like(support)
            probs[0] = 1.0
            return {"support_rows": (support + 1.0).tolist(),
                    "probs": probs.tolist(), "non_drafting_share": share,
                    "conditional_mean_depth": 0.0}
        conditional = target_depth / (1.0 - share)
        drafting = weights.copy()
        drafting[0] = 0.0
        probs = tilt_to_mean(drafting, support, conditional)
        probs = (1.0 - share) * probs
        probs[0] += share
        return {"support_rows": (support + 1.0).tolist(),
                "probs": probs.tolist(), "non_drafting_share": share,
                "conditional_mean_depth": conditional}
    weights[0] = 0.0  # every drafting prompt records zero rounds at depth 0
    probs = tilt_to_mean(weights, support, target_depth)
    return {"support_rows": (support + 1.0).tolist(), "probs": probs.tolist(),
            "non_drafting_share": 0.0, "conditional_mean_depth": target_depth}


# --------------------------------------------------------------------- fits

MODEL_TERMS = {"line": 2, "step": 3, "piece": 4}


def design_row(rows: np.ndarray, probs: np.ndarray, breakpoint: int,
               terms: int) -> np.ndarray:
    """Expected regressor row for one prompt under a monotone tiered cost.

    The cost is written in an incremental basis that makes every physical
    constraint a simple sign constraint:

        round_us(M) = a + b M + [M >= B] * (jump + db * (M - B))

    so `b >= 0` forces cost to rise with width, `jump >= 0` forbids a tier
    boundary that makes a round cheaper, and `db >= 0` forbids the upper tier
    being flatter than the lower one. An unconstrained two-segment fit on
    eight points does produce negative slopes and downward steps, which are
    not cost curves.

    Cost is linear in these parameters, so the expectation over the per-round
    width histogram is the expectation of each regressor. A degenerate
    histogram at `M̄` recovers the ordinary mean fit.
    """
    above = rows >= breakpoint
    row = [1.0, float((probs * rows).sum())]
    if terms >= 3:
        row.append(float(probs[above].sum()))
    if terms >= 4:
        row.append(float((probs[above] * (rows[above] - breakpoint)).sum()))
    return np.array(row)


def bounded_lstsq(a: np.ndarray, y: np.ndarray) -> tuple:
    """Least squares with variable 0 free and every later variable >= 0.

    Only three variables can be constrained, so the eight possible active
    sets are enumerated exactly. The constrained optimum's active set is one
    of them, so the feasible solution with the lowest residual sum of squares
    is the exact answer.
    """
    n = a.shape[1]
    best = None
    for mask in range(1 << (n - 1)):
        keep = [0] + [i for i in range(1, n) if mask >> (i - 1) & 1]
        sub = a[:, keep]
        if np.linalg.matrix_rank(sub) < sub.shape[1]:
            continue
        beta_sub, *_ = np.linalg.lstsq(sub, y, rcond=None)
        if any(v < -1e-9 for v, idx in zip(beta_sub, keep) if idx > 0):
            continue
        beta = np.zeros(n)
        beta[keep] = beta_sub
        rss = float(((y - a @ beta) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, beta)
    return best


def prompt_probs(point: dict, use_hist: bool) -> np.ndarray:
    if use_hist:
        return np.array(point["hist"]["probs"])
    probs = np.zeros(MAX_ROWS)
    mbar = point["mbar"]
    k = min(max(int(math.floor(mbar - 1.0)), 0), MAX_ROWS - 2)
    frac = (mbar - 1.0) - k
    probs[k] = 1.0 - frac
    probs[k + 1] = frac
    return probs


def fit(points: list[dict], breakpoint: int, model: str,
        use_hist: bool) -> dict:
    rows = np.arange(0, MAX_ROWS, dtype=float) + 1.0
    terms = MODEL_TERMS[model]
    a = np.array([design_row(rows, prompt_probs(p, use_hist), breakpoint, terms)
                  for p in points])
    y = np.array([p["round_us"] for p in points])
    got = bounded_lstsq(a, y)
    if got is None:
        return {"ok": False, "breakpoint": breakpoint, "model": model}
    rss, beta = got
    resid = y - a @ beta
    n, k = len(y), terms
    aicc = n * math.log(max(rss, 1e-12) / n) + 2 * k
    aicc += 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("inf")
    intercept, slope = float(beta[0]), float(beta[1])
    jump = float(beta[2]) if terms >= 3 else 0.0
    dslope = float(beta[3]) if terms >= 4 else 0.0
    return {
        "ok": True, "model": model, "breakpoint": breakpoint,
        "lo": (intercept, slope),
        "hi": (intercept + jump - dslope * breakpoint, slope + dslope),
        "jump_us": jump, "dslope_us": dslope,
        "rss": rss, "aicc": aicc, "params": k,
        "residuals": {p["prompt"]: float(r) for p, r in zip(points, resid)},
        "rmse": math.sqrt(rss / n),
    }


def curve_entry(scenario: dict, breakpoint: int, name: str) -> dict:
    for row in scenario["dist_fit_table"]:
        if row["model"] == "piece" and row["breakpoint"] == breakpoint:
            return {"name": name, "model": "piece", "breakpoint": breakpoint,
                    "rmse_us": row["rmse"], "lo": list(row["lo"]),
                    "hi": list(row["hi"])}
    raise SystemExit("no piece fit at break %d" % breakpoint)


def curve_us(curve: dict, rows: float) -> float:
    intercept, slope = curve["lo"] if rows < curve["breakpoint"] else curve["hi"]
    return intercept + slope * rows


def build_points(receipt: dict, r_vec: dict, hists: dict) -> list[dict]:
    points = []
    for prompt, row in receipt["per_prompt"].items():
        rounds = r_vec[prompt]
        mbar = row["draft_len"] + 1.0
        round_us = row["candidate"] * DECODE_TOKENS * 1e6 / rounds
        points.append({
            "prompt": prompt, "R": rounds, "mbar": mbar,
            "round_us": round_us, "weight": F83_WEIGHT[prompt],
            "tokens_per_round": DECODE_TOKENS / rounds,
            "hist": prompt_width_histogram(prompt, mbar, row["non_drafting"],
                                           rounds, hists),
        })
    points.sort(key=lambda p: p["mbar"])
    return points


def select(points: list[dict], use_hist: bool) -> dict:
    """Sweeps the tier step and the model shape, and reports every candidate."""
    table = []
    for model in ("line", "step", "piece"):
        breaks = [MAX_ROWS + 1] if model == "line" else list(range(2, MAX_ROWS))
        for breakpoint in breaks:
            got = fit(points, breakpoint, model, use_hist)
            if got.get("ok"):
                table.append(got)
    table.sort(key=lambda row: row["aicc"])
    return {"best": table[0], "table": table}


def r_sweep(receipt: dict, scenarios: dict, hists: dict,
            steps: int = 81) -> dict:
    """One-parameter R sweep between the two pre-registered R vectors.

    `R(t) = (1 - t) R_assumed + t R_predicted` is a single free parameter, so
    minimising the curve residual over `t` leaves three degrees of freedom on
    eight points and is a legitimate estimate rather than an exact fit. A
    uniform rescale of R cannot move the residual at all, because it rescales
    every cost by the same factor, so any structure found here is genuine
    shape information about the R vector.
    """
    out = []
    for index in range(steps):
        t = -0.5 + 2.0 * index / (steps - 1)
        r_vec = {p: (1.0 - t) * scenarios["assumed"][p]
                 + t * scenarios["predicted"][p]
                 for p in scenarios["assumed"]}
        points = build_points(receipt, r_vec, hists)
        best = min((fit(points, b, "piece", True) for b in range(2, MAX_ROWS)),
                   key=lambda row: row["rmse"])
        out.append({"t": t, "rmse": best["rmse"],
                    "breakpoint": best["breakpoint"],
                    "relative_rmse": best["rmse"]
                    / (sum(p["round_us"] for p in points) / len(points))})
    best = min(out, key=lambda row: row["rmse"])
    return {"grid": out, "best": best}


def loo_breakpoint(points: list[dict], model: str, use_hist: bool) -> list:
    out = []
    for drop in range(len(points)):
        subset = [p for i, p in enumerate(points) if i != drop]
        best, best_aicc = None, float("inf")
        for breakpoint in range(2, MAX_ROWS):
            got = fit(subset, breakpoint, model, use_hist)
            if got.get("ok") and got["aicc"] < best_aicc:
                best, best_aicc = got, got["aicc"]
        out.append({"dropped": points[drop]["prompt"],
                    "breakpoint": best["breakpoint"] if best else None,
                    "rmse": best["rmse"] if best else None})
    return out


def jensen_report(points: list[dict], curve: dict) -> list:
    out = []
    rows = np.arange(0, MAX_ROWS, dtype=float) + 1.0
    for point in points:
        probs = np.array(point["hist"]["probs"])
        at_mean = curve_us(curve, point["mbar"])
        over_hist = float(sum(pr * curve_us(curve, m)
                              for pr, m in zip(probs, rows)))
        out.append({
            "prompt": point["prompt"], "mbar": point["mbar"],
            "cost_at_mean_us": at_mean, "cost_over_hist_us": over_hist,
            "jensen_bias_us": over_hist - at_mean,
            "jensen_bias_pct": 100.0 * (over_hist - at_mean) / at_mean,
            "hist_sd_rows": float(math.sqrt(
                max((probs * (rows - point["mbar"]) ** 2).sum(), 0.0))),
        })
    return out


# -------------------------------------------------------------------- report

def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--board", type=Path,
                        default=Path("/tmp/yukon-board/full.json"))
    parser.add_argument("--identity", type=Path,
                        default=here / "e128-artifacts/rung0-identity.json")
    parser.add_argument("--shipped", type=Path,
                        default=here / "e128-artifacts/rung1-shipped.json")
    parser.add_argument("--receipt", default="d3c491b5")
    parser.add_argument("--control", default="44559d02")
    parser.add_argument("--extra", nargs="*", default=["b8b8b860", "f04b102e"])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--curve-json", type=Path)
    args = parser.parse_args()

    hists = fixture_histograms(args.shipped)
    scenarios = r_scenarios(args.identity)
    receipts = {name: load_receipt(args.board, name)
                for name in [args.receipt, args.control] + list(args.extra)}

    print("harness=ranked  E128 section 5 - our own ranked cost curve\n")
    for name, receipt in receipts.items():
        print("%-9s %-9s official %.8f  mode %s  commit %s"
              % (name, receipt["status"], receipt["score"], receipt["mode"],
                 (receipt["commit"] or "")[:12]))

    ref = build_points(receipts[args.receipt], scenarios["predicted"], hists)
    ctl = build_points(receipts[args.control], scenarios["predicted"], hists)
    ctl_by_prompt = {p["prompt"]: p for p in ctl}

    print("\nper-prompt points at R_predicted (F111 pinned):")
    print("%-9s %7s %9s %8s %13s %13s %8s" % (
        "prompt", "R", "tok/round", "Mbar", "round us new", "round us ctl",
        "delta %"))
    for point in ref:
        other = ctl_by_prompt[point["prompt"]]
        print("%-9s %7.1f %9.5f %8.3f %13.0f %13.0f %+8.2f" % (
            point["prompt"], point["R"], point["tokens_per_round"],
            point["mbar"], point["round_us"], other["round_us"],
            100.0 * (point["round_us"] / other["round_us"] - 1.0)))

    results = {}
    for scenario, r_vec in scenarios.items():
        points = build_points(receipts[args.receipt], r_vec, hists)
        mean_fit = select(points, use_hist=False)
        dist_fit = select(points, use_hist=True)
        results[scenario] = {
            "points": [{k: v for k, v in p.items() if k != "hist"}
                       for p in points],
            "mean_fit_best": mean_fit["best"],
            "mean_fit_table": mean_fit["table"],
            "dist_fit_best": dist_fit["best"],
            "dist_fit_table": dist_fit["table"],
            "jensen": jensen_report(points, dist_fit["best"]),
        }

    # The four R vectors are alternatives for the same eight measurements, so
    # their residuals are in the same units on the same points and are
    # directly comparable. The scenario whose best curve fits tightest is the
    # one the physics prefers.
    primary = min(results, key=lambda s: results[s]["dist_fit_best"]["rmse"])
    print("\nprimary R scenario chosen by curve residual: %s" % primary)

    print("\nmodel selection at R_%s, ranked by AICc (dist fit):" % primary)
    print("%-6s %6s %7s %9s %9s %9s %9s  %s" % (
        "model", "break", "params", "rmse us", "aicc", "jump us", "dslope",
        "lo / hi"))
    for row in results[primary]["dist_fit_table"][:10]:
        print("%-6s %6s %7d %9.0f %9.2f %9.0f %9.0f  %.0f+%.0fM / %.0f+%.0fM"
              % (row["model"],
                 row["breakpoint"] if row["model"] != "line" else "-",
                 row["params"], row["rmse"], row["aicc"], row["jump_us"],
                 row["dslope_us"], row["lo"][0], row["lo"][1],
                 row["hi"][0], row["hi"][1]))

    best = results[primary]["dist_fit_best"]
    mean_best = results[primary]["mean_fit_best"]
    print("\nchosen dist  model %s break M>=%s  %.1f + %.1f M  |  %.1f + %.1f M"
          % (best["model"], best["breakpoint"], best["lo"][0], best["lo"][1],
             best["hi"][0], best["hi"][1]))
    print("chosen mean  model %s break M>=%s  %.1f + %.1f M  |  %.1f + %.1f M"
          % (mean_best["model"], mean_best["breakpoint"], mean_best["lo"][0],
             mean_best["lo"][1], mean_best["hi"][0], mean_best["hi"][1]))

    print("\nrmse (us) of the 4-parameter tiered fit, by tier step and R:")
    header = "  ".join("M>=%d" % b for b in range(2, MAX_ROWS))
    print("%-9s  %s" % ("scenario", header))
    grid = {}
    for scenario, r_vec in scenarios.items():
        points = build_points(receipts[args.receipt], r_vec, hists)
        row = {b: fit(points, b, "piece", True) for b in range(2, MAX_ROWS)}
        grid[scenario] = {b: v["rmse"] for b, v in row.items() if v.get("ok")}
        print("%-9s  %s" % (scenario, "  ".join(
            "%5.0f" % grid[scenario][b] for b in range(2, MAX_ROWS))))
    results["rmse_grid_piece"] = grid

    sweep = r_sweep(receipts[args.receipt], scenarios, hists)
    print("\none-parameter R sweep, R(t) = (1-t) assumed + t predicted:")
    print("%8s %10s %10s %8s" % ("t", "rmse us", "rel rmse", "break"))
    for row in sweep["grid"]:
        if abs(row["t"] * 8 - round(row["t"] * 8)) < 1e-9:
            print("%8.3f %10.0f %10.5f %8d"
                  % (row["t"], row["rmse"], row["relative_rmse"],
                     row["breakpoint"]))
    print("minimum at t = %.4f  rmse %.0f us  relative %.5f  break M>=%d"
          % (sweep["best"]["t"], sweep["best"]["rmse"],
             sweep["best"]["relative_rmse"], sweep["best"]["breakpoint"]))
    results["r_sweep"] = sweep

    loo = loo_breakpoint(build_points(receipts[args.receipt],
                                      scenarios[primary], hists),
                         best["model"], True)
    print("\nleave-one-prompt-out tier step (%s model):" % best["model"])
    for row in loo:
        print("  drop %-9s -> break M>=%s  rmse %.0f us"
              % (row["dropped"], row["breakpoint"], row["rmse"]))

    print("\nbreakpoint stability across R scenarios (dist fit):")
    for scenario in ("predicted", "assumed", "band_lo", "band_hi"):
        row = results[scenario]["dist_fit_best"]
        print("  %-9s model %-6s break M>=%-2s  rmse %6.0f us  "
              "marginal lo %.0f hi %.0f"
              % (scenario, row["model"], row["breakpoint"], row["rmse"],
                 row["lo"][1], row["hi"][1]))

    print("\nJensen bias of a fit through Mbar, at R_%s:" % primary)
    print("%-9s %8s %9s %14s %14s %10s" % (
        "prompt", "Mbar", "sd rows", "cost at mean", "cost over hist",
        "bias %"))
    for row in results[primary]["jensen"]:
        print("%-9s %8.3f %9.3f %14.0f %14.0f %+10.2f" % (
            row["prompt"], row["mbar"], row["hist_sd_rows"],
            row["cost_at_mean_us"], row["cost_over_hist_us"],
            row["jensen_bias_pct"]))

    board = BOARD_CURVE
    print("\nmeasured points against the F97 board curve, at the realised "
          "Mbar (no fit):")
    print("%-9s %8s %12s %12s %11s %12s %11s" % (
        "prompt", "Mbar", "board us", "obs R_pred", "low %",
        "obs R_asm", "low %"))
    assumed = build_points(receipts[args.receipt], scenarios["assumed"], hists)
    assumed_by_prompt = {p["prompt"]: p for p in assumed}
    for point in ref:
        theirs = curve_us(board, point["mbar"])
        other = assumed_by_prompt[point["prompt"]]
        print("%-9s %8.3f %12.0f %12.0f %+11.2f %12.0f %+11.2f" % (
            point["prompt"], point["mbar"], theirs, point["round_us"],
            100.0 * (point["round_us"] / theirs - 1.0), other["round_us"],
            100.0 * (other["round_us"] / theirs - 1.0)))

    print("\nour fitted curve against the F97 board curve, at the realised "
          "Mbar:")
    print("%-9s %8s %12s %12s %9s" % (
        "prompt", "Mbar", "ours us", "board us", "ours low %"))
    for point in ref:
        ours = curve_us(best, point["mbar"])
        theirs = curve_us(board, point["mbar"])
        print("%-9s %8.3f %12.0f %12.0f %+9.2f" % (
            point["prompt"], point["mbar"], ours, theirs,
            100.0 * (ours / theirs - 1.0)))

    ratio_ours = best["hi"][1] / curve_us(best, 1.0)
    ratio_board = board["hi"][1] / curve_us(board, 1.0)
    print("\nmarginal-to-fixed ratio  ours %.4f  board %.4f  shipped price 0.18"
          % (ratio_ours, ratio_board))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "harness": "ranked",
            "receipt": {k: {kk: vv for kk, vv in v.items()
                            if kk != "per_prompt"}
                        for k, v in receipts.items()},
            "control_points": [{k: v for k, v in p.items() if k != "hist"}
                               for p in ctl],
            "histograms": {p["prompt"]: p["hist"] for p in ref},
            "primary_r_scenario": primary,
            "r_scenarios": scenarios,
            "scenarios": results,
            "loo_breakpoint": loo,
            "board_curve": {"breakpoint": board["breakpoint"],
                            "lo": list(board["lo"]), "hi": list(board["hi"])},
            "marginal_to_fixed_ratio": {"ours": ratio_ours,
                                        "board": ratio_board},
        }, indent=1) + "\n")
        print("\nwrote %s" % args.json)

    if args.curve_json:
        args.curve_json.parent.mkdir(parents=True, exist_ok=True)
        args.curve_json.write_text(json.dumps({
            "harness": "ranked",
            "source": "e128_ourcurve.py",
            "receipt": args.receipt,
            "fit": "dist",
            "r_scenario": primary,
            "r_vector": scenarios[primary],
            "rmse_us": best["rmse"],
            "model": best["model"],
            "breakpoint": best["breakpoint"],
            "lo": list(best["lo"]),
            "hi": list(best["hi"]),
            "curves": {
                scenario: {
                    "name": "ours_%s_%s" % (args.receipt, scenario),
                    "r_scenario": scenario,
                    "rmse_us": results[scenario]["dist_fit_best"]["rmse"],
                    "model": results[scenario]["dist_fit_best"]["model"],
                    "breakpoint":
                        results[scenario]["dist_fit_best"]["breakpoint"],
                    "lo": list(results[scenario]["dist_fit_best"]["lo"]),
                    "hi": list(results[scenario]["dist_fit_best"]["hi"]),
                }
                for scenario in ("predicted", "assumed", "band_lo", "band_hi")
            } | {
                # The tier step is the one structural choice in the fit, so the
                # two next-best steps and the mean-fit curve are carried
                # forward as named alternatives. Any conclusion that survives
                # all four does not rest on the step location.
                "board": {"name": "board_f97", "breakpoint": board["breakpoint"],
                          "lo": list(board["lo"]), "hi": list(board["hi"]),
                          "rmse_us": None, "model": "piece"},
                "ours_b4": curve_entry(results[primary], 4, "ours_b4"),
                "ours_b5": curve_entry(results[primary], 5, "ours_b5"),
                "ours_meanfit": {
                    "name": "ours_meanfit",
                    "model": mean_best["model"],
                    "breakpoint": mean_best["breakpoint"],
                    "rmse_us": mean_best["rmse"],
                    "lo": list(mean_best["lo"]), "hi": list(mean_best["hi"])},
            },
            "name": "ours_%s_%s" % (args.receipt, primary),
        }, indent=1) + "\n")
        print("wrote %s" % args.curve_json)


if __name__ == "__main__":
    main()
