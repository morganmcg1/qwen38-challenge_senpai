#!/usr/bin/env python3
"""E190: predeclared cap-5 ranked predictions under the two admissible laws.

The bare quadratic (FINDING 456) and the affine-rescaled staircase
(FINDING 488) both claim to fit the paid ranked anchors. They disagree most at
draft cap 5. This script prices the cap-5 published median under each law with
E177's incremental walk, so every prediction is on the record before the
receipt lands.

    usage: python3 research/e190_predict.py

harness=ranked. Nothing here is measured; all numbers are model predictions.

ROW-FORM AUDIT. `basis_quadratic` regresses on E[rows] and E[rows^2], where
rows = drafts + 1, so FINDING 456's fitted coefficients (30.640, -0.939,
0.576) are already in ROW form: R(rows=1) = 30.277 ms, which is exactly the
held-out zero-draft anchor FINDING 456 reports. `e186_analyze.py` instead read
those coefficients as draft-count form and re-expressed them to
(32.155, -2.091, 0.576), which evaluates the law one row low. FINDING 488's
alpha/beta were therefore fitted to shifted targets, and its published
disagreement column compared staircase(m) against quadratic(m-1). This script
reports the published variant, the corrected variant, and a direct refit of
the staircase SHAPE on the same ranked observations the quadratic was fitted
on, which is the specification that compares the two forms on equal terms.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402

E186_ANALYSIS = pathlib.Path(__file__).resolve().parent / "out/e186/analysis.json"
TARGET_CAP = 5
SIGMA_PUBLISHED = 0.00689  # FINDING 460 receipt channel, fraction of score
MUE_MS = 0.567             # minimum useful effect, ms/round
ANCHOR_ROWS = [1, 5, 8]    # zero-draft floor, paid cap-4, paid cap-7
WIDTHS = list(range(1, 10))


def local_shape():
    """E186's measured local round reconstruction, ms, indexed by rows - 1."""
    return json.load(open(E186_ANALYSIS))["round_reconstruction"]["abs_ms"]


def affine_fit(xs, ys):
    """Least squares y = a + b*x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def table_model(table_ms):
    """Wrap a deterministic R(rows) table, ms, as an e177 cost model, seconds."""
    table = [v / 1000.0 for v in table_ms]

    def basis(probs):
        return [sum(q * table[d] for d, q in enumerate(probs))]

    return {"names": ["stair"], "beta": [1.0], "basis": basis,
            "table_ms": list(table_ms)}


def staircase_variants(quad):
    """Published and corrected affine rescalings of the measured staircase."""
    shape = local_shape()
    F, b, c = [1000 * v for v in quad["beta"]]

    def quad_at(rows):
        return F + b * rows + c * rows * rows

    out = {}
    for key, offset in (("staircase-published", -1), ("staircase-corrected", 0)):
        xs = [shape[r - 1] for r in ANCHOR_ROWS]
        ys = [quad_at(r + offset) for r in ANCHOR_ROWS]
        a, s = affine_fit(xs, ys)
        resid = [y - (a + s * x) for x, y in zip(xs, ys)]
        model = table_model([a + s * v for v in shape])
        model.update({"alpha_ms": a, "scale": s, "anchor_target_ms": ys,
                      "max_anchor_residual_ms": max(abs(r) for r in resid)})
        out[key] = model
    return out, shape


def build():
    rows = e177.load_board()
    data = {}
    for label, prefix, cap, note in e177.RECEIPTS:
        row = next(r for r in rows if r["id"].startswith(prefix))
        vec = e177.per_prompt(row)
        data[label] = {"id": row["id"][:8], "cap": cap, "score": row["officialScore"],
                       "vec": vec, "rec": e177.recover_rounds(vec)}
    surv, bounds = {}, {}
    for name in e177.ORDER:
        a, c = data["A"]["rec"][name], data["C"]["rec"][name]
        t0 = 1.0 - a["nondraft"] / a["rounds"]
        _, _, _, s = e177.fit_survival(c["edl"], a["edl"], t0)
        surv[name] = s
        bounds[name] = e177.tail_bounds(c["edl"], a["edl"], s[3])
    mu = {}
    for name in e177.ORDER:
        a, c = data["A"]["rec"][name], data["C"]["rec"][name]
        d_edl = a["edl"] - c["edl"]
        mu[name] = (a["abar"] - c["abar"]) / d_edl if d_edl > 1e-9 else 0.0
    core = []
    for label in ("A", "C"):
        for name in e177.ORDER:
            r = data[label]["rec"][name]
            core.append((label, name, e177.width_probs(surv[name], data[label]["cap"]),
                         r["R"], r["rounds"]))
    fits = e177.fit_cost(core, ["A", "C"], free_intercepts=[])
    return data, surv, bounds, mu, fits, core


def refit_shape_on_ranked(core, shape):
    """R = F + s * localShape(rows), refitted on the ranked observations."""
    scaled = [v / 1000.0 for v in shape]

    def basis(probs):
        return [1.0, sum(q * scaled[d] for d, q in enumerate(probs))]

    e177.COST_MODELS["staircase-ranked-refit"] = (["F", "s"], basis)
    try:
        return e177.fit_cost(core, ["A", "C"],
                             free_intercepts=[])["staircase-ranked-refit"]
    finally:
        del e177.COST_MODELS["staircase-ranked-refit"]


def chain(data, tvar, mu, model, name, target_cap, anchor):
    rec = data[anchor]["rec"][name]
    cap, abar, R, t = data[anchor]["cap"], rec["abar"], rec["R"], tvar[name]
    while cap < target_cap:
        abar += t[cap] * mu[name]
        R += t[cap] * (e177.cost_at_width(model, cap + 2)
                       - e177.cost_at_width(model, cap + 1))
        cap += 1
    while cap > target_cap:
        abar -= t[cap - 1] * mu[name]
        R -= t[cap - 1] * (e177.cost_at_width(model, cap + 1)
                           - e177.cost_at_width(model, cap))
        cap -= 1
    n_rounds = e177.TOKENS / (1.0 + abar)
    return abar, R, n_rounds, n_rounds * R


def score_of(data, tvar, mu, model, target_cap, anchor):
    raws = []
    for name in e177.ORDER:
        vec = data["A"]["vec"][name]
        _, _, _, total = chain(data, tvar, mu, model, name, target_cap, anchor)
        raws.append(vec["serial"] / (total / e177.TOKENS + vec["prefill"]))
    return e177.published_median(raws)


def main():
    data, surv, bounds, mu, fits, core = build()
    quad = fits["quadratic"]
    stairs, shape = staircase_variants(quad)
    stairs["staircase-ranked-refit"] = refit_shape_on_ranked(core, shape)

    variants = {"fitted": {n: list(surv[n]) for n in e177.ORDER},
                "tail-heavy": {}, "tail-light": {}}
    for name in e177.ORDER:
        b = bounds[name]
        heavy = list(surv[name])
        heavy[4] = b["t4"][1]
        heavy[5] = max(0.0, b["delta"] - heavy[4])
        heavy[6] = heavy[7] = 0.0
        light = list(surv[name])
        light[4] = light[5] = light[6] = light[7] = b["delta"] / 3.0
        variants["tail-heavy"][name] = heavy
        variants["tail-light"][name] = light

    print("=" * 108)
    print("E190 PREDECLARED cap-5 predictions (harness=ranked)")
    print("=" * 108)

    print("\n  ROW-FORM AUDIT")
    print("    fitted quadratic betas (ms): F=%.4f b=%.4f c=%.4f, wRMSE %.4f ms"
          % tuple([1000 * v for v in quad["beta"]] + [quad["wrmse_ms"]]))
    print("    R(rows=1) = %.3f ms; FINDING 456 held-out anchor 30.277 ms, floor "
          "30.2519 ms -> the fit is in ROW form"
          % (1000 * e177.cost_at_width(quad, 1)))

    print("\n  local measured staircase shape (E186 round reconstruction, ms)")
    print("    %-24s %s" % ("rows", "  ".join("%7d" % m for m in WIDTHS)))
    print("    %-24s %s" % ("shape", "  ".join("%7.1f" % v for v in shape)))

    print("\n  ranked round cost R(rows), ms")
    print("    %-24s %s" % ("model", "  ".join("%7d" % m for m in WIDTHS)))
    qv = [1000 * e177.cost_at_width(quad, m) for m in WIDTHS]
    print("    %-24s %s" % ("quadratic", "  ".join("%7.2f" % v for v in qv)))
    for key, model in stairs.items():
        sv = [1000 * e177.cost_at_width(model, m) for m in WIDTHS]
        print("    %-24s %s" % (key, "  ".join("%7.2f" % v for v in sv)))
        print("    %-24s %s" % ("  minus quadratic",
                                "  ".join("%+7.2f" % (s - q) for q, s in zip(qv, sv))))

    print("\n  anchor quality (the three ranked cells both forms claim to fit)")
    for key in ("staircase-published", "staircase-corrected"):
        m = stairs[key]
        print("    %-24s alpha %8.4f ms  scale %.6f  max|resid| %.3f ms = %.2f MUE"
              % (key, m["alpha_ms"], m["scale"], m["max_anchor_residual_ms"],
                 m["max_anchor_residual_ms"] / MUE_MS))
    rr = stairs["staircase-ranked-refit"]
    print("    %-24s wRMSE %.4f ms  maxres %.3f ms  AIC %.2f   (quadratic wRMSE "
          "%.4f ms, AIC %.2f)"
          % ("staircase-ranked-refit", rr["wrmse_ms"], rr["max_resid_ms"], rr["aic"],
             quad["wrmse_ms"], quad["aic"]))

    print("\n  cap 5 opens the 6-row cell; that is where the forms separate")
    for key, model in stairs.items():
        s6 = 1000 * e177.cost_at_width(model, 6)
        print("    %-24s R(6) = %6.2f ms vs quadratic %6.2f ms, gap %+6.2f ms = "
              "%5.2f MUE" % (key, s6, qv[5], s6 - qv[5], (s6 - qv[5]) / MUE_MS))

    print("\n  PREDICTED cap-5 published median, walked from each paid anchor")
    header = "    %-12s %-7s %14s" % ("survival", "anchor", "quadratic")
    header += "".join("%24s" % k for k in stairs)
    print(header)
    preds = {"quadratic": []}
    for k in stairs:
        preds[k] = []
    for vname, tvar in variants.items():
        for anchor in ("A", "C"):
            line = "    %-12s %-7s" % (vname, anchor)
            q = score_of(data, tvar, mu, quad, TARGET_CAP, anchor)
            line += "%14.8f" % q
            if vname == "fitted":
                preds["quadratic"].append(q)
            for key, model in stairs.items():
                s = score_of(data, tvar, mu, model, TARGET_CAP, anchor)
                line += "%24.8f" % s
                if vname == "fitted":
                    preds[key].append(s)
            print(line)

    means = {k: sum(v) / len(v) for k, v in preds.items()}
    pq = means["quadratic"]
    print("\n  HEADLINE (fitted survival, mean of both anchors)")
    for k, v in means.items():
        print("    %-24s %.8f" % (k, v))
    print("\n  SEPARATION from the quadratic, in receipt-channel sigma "
          "(sigma = %.3f%% of score)" % (100 * SIGMA_PUBLISHED))
    for k, v in means.items():
        if k == "quadratic":
            continue
        print("    %-24s %+.8f  = %5.2f sigma"
              % (k, v - pq, abs(v - pq) / (SIGMA_PUBLISHED * pq)))

    print("\n  PREDECLARED DECISION BOUNDARY")
    print("    quadratic 1-sigma window     [%.6f, %.6f]"
          % (pq * (1 - SIGMA_PUBLISHED), pq * (1 + SIGMA_PUBLISHED)))
    for k, v in means.items():
        if k == "quadratic":
            continue
        print("    %-24s [%.6f, %.6f], midpoint vs quadratic %.6f"
              % (k, v * (1 - SIGMA_PUBLISHED), v * (1 + SIGMA_PUBLISHED),
                 0.5 * (pq + v)))
    print("    Rule: the receipt selects the one law whose 1-sigma window contains")
    print("    it. Inside both, inside neither, or between the windows is Unclear.")

    print("\n  reference: paid cap-7 receipt A %.8f, paid cap-4 receipt C %.8f, "
          "crown %.8f" % (e177.BEST_A, e177.CAP4_PUBLISHED, e177.CROWN))


if __name__ == "__main__":
    main()
