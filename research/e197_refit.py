#!/usr/bin/env python3
"""E197: refit the ranked round-cost law on THREE paid caps and reprice depth.

    usage: python3 research/e197_refit.py [stage1|stage2|all]

harness=ranked. Every number here is inverted from paid ranked receipts or is
a model prediction built on them. The one local input is E186's measured
round shape (harness=local), used only as a SHAPE prior and always labelled.

WHAT THE THIRD RECEIPT BUYS. With paid caps {4, 5, 7} the per-prompt schedule
identity D = min(D*, cap) gives, per prompt, exact and family-free values:

    t4      = edl(5) - edl(4)                     tail mass at depth 4
    t5 + t6 = edl(7) - edl(5)                     joint tail mass at 5 and 6
    S5      = (abar(5) - abar(4)) / t4            5-deep cumulative acceptance
    mu(5:7) = (abar(7) - abar(5)) / (edl(7) - edl(5))

E177 had to fit a two-parameter survival family to reach these; they are now
measured. Only t7, S8 and the cost of the 9-row cell remain extrapolated, and
the 9-row cell is invisible to every ranked receipt at cap <= 7.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402

ORDER = e177.ORDER
TOKENS = e177.TOKENS
CROWN = e177.CROWN
BEST_A = e177.BEST_A
CAP4 = e177.CAP4_PUBLISHED
CAP5 = 3.52363580183674
SIGMA_PUBLISHED = 0.00689      # FINDING 460 receipt channel, whole-score
MUE_MS = 0.567                 # 0.39% of the 145.963 ms round median
R1_FLOOR_MS = 30.2519          # FINDING 330 held-out zero-draft ranked round
DR6_MEASURED_MS = 8.940        # FINDING 505
DR6_BAND_MS = (8.345, 9.544)
ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e197-artifacts"
CAP5_RECEIPT = ARTIFACTS / "receipt-cap5-04710829.json"
E186_SHAPE = ARTIFACTS / "e186-local-round-shape.json"
MAXW = 9                       # widths 1..9; cap 8 verifies at most 9 rows

# FINDING 503 per-prompt receipt-channel sigma, fraction of the MTP leg.
PROMPT_SIGMA = {"travel": 0.00954, "drama": 0.00921, "beagle": 0.00753,
                "medicine": 0.00647, "botany": 0.00630, "essays": 0.00621,
                "republic": 0.00583, "plutarch": 0.00105}


# ---------------------------------------------------------------- receipts

def build_receipts():
    rows = e177.load_board()
    data = {}
    for label, prefix, cap, _ in e177.RECEIPTS:
        row = next(r for r in rows if r["id"].startswith(prefix))
        vec = e177.per_prompt(row)
        data[label] = {"id": row["id"][:8], "cap": cap,
                       "score": row["officialScore"], "vec": vec,
                       "rec": e177.recover_rounds(vec)}
    receipt = json.loads(CAP5_RECEIPT.read_text())
    vec = e177.per_prompt(receipt)
    data["E"] = {"id": receipt["id"][:8], "cap": 5,
                 "score": receipt["officialScore"], "vec": vec,
                 "rec": e177.recover_rounds(vec)}
    return data


def sigma_R_ms(name, rec):
    """1-sigma on the inverted round cost, ms, from the per-prompt channel."""
    inflate = rec["mtp"] / (rec["mtp"] - rec["prefill"])
    return 1000.0 * PROMPT_SIGMA[name] * rec["R"] * inflate


# ------------------------------------------------- schedule identification

def survival(data, tail="family"):
    """Per-prompt tail masses t_0..t_8 for D* = greedy stopping depth.

    t4 and t5+t6 are exact. t0..t3 are split by a discrete Weibull fitted to
    edl(4); the split affects only the shallow width mixture, whose total is
    exact. t7 is the one extrapolated mass; ``tail`` selects its treatment.
    """
    out, exact = {}, {}
    for name in ORDER:
        c, e, a = (data["C"]["rec"][name], data["E"]["rec"][name],
                   data["A"]["rec"][name])
        t0 = 1.0 - a["nondraft"] / a["rounds"]
        edl4, edl5, edl7 = c["edl"], e["edl"], a["edl"]
        t4 = edl5 - edl4
        t56 = edl7 - edl5
        best = None
        for li in range(1, 801):
            lam = li * 0.02
            for ki in range(1, 301):
                kappa = ki * 0.02
                surv = [t0 * math.exp(-((d / lam) ** kappa)) if d else t0
                        for d in range(MAXW + 1)]
                err = ((sum(surv[:4]) - edl4) ** 2 + (surv[4] - t4) ** 2
                       + (surv[5] + surv[6] - t56) ** 2)
                if best is None or err < best[0]:
                    best = (err, lam, kappa, surv)
        _, lam, kappa, surv = best
        # exact-match corrections. t0 is measured (the drafting rate), so only
        # t1..t3 absorb the shallow-block residual; their sum is exact.
        rest = sum(surv[1:4])
        scale = (edl4 - t0) / rest if rest > 1e-12 else 0.0
        t = [t0] + [surv[d] * scale for d in range(1, 4)]
        t = [min(v, t[i - 1]) if i else v for i, v in enumerate(t)]
        t.append(t4)
        pair = surv[5] + surv[6]
        if pair > 1e-12:
            t += [surv[5] * t56 / pair, surv[6] * t56 / pair]
        else:
            t += [0.5 * t56, 0.5 * t56]
        t = [min(v, t[i - 1]) if i else v for i, v in enumerate(t)]  # monotone
        if tail == "family":
            t7 = surv[7] * (t[6] / surv[6]) if surv[6] > 1e-12 else 0.0
        elif tail == "heavy":
            t7 = t[6]
        else:
            t7 = 0.0
        t.append(min(t7, t[6]))
        t.append(min(t7, t[6]))          # t8, never offered (cap <= 8)
        out[name] = t
        exact[name] = {"t0": t0, "edl4": edl4, "edl5": edl5, "edl7": edl7,
                       "t4": t4, "t56": t56, "lam": lam, "kappa": kappa,
                       "fit_err": best[0]}
    return out, exact


def acceptance(data, tvar, deep="family", model="measured"):
    """Per-prompt acceptance rate S_k for the k-th drafted row.

    ``model="measured"`` is the primary pricing model and is family-free on
    every observed cell: S5 = (abar(5) - abar(4)) / t4 reproduces abar(5) from
    abar(4) exactly, and S6 = S7 = mu(5:7) reproduces abar(7) from abar(5)
    exactly. Only S8 is extrapolated, flat by default.

    ``model="curve"`` is the conservative alternative: a monotone
    S_k = exp(-(k/theta)^eta) fitted to the same three abar values. It is
    admissible as a cumulative acceptance probability but does not reproduce
    the measured values; the measured cells are mildly non-nested, because a
    cap change perturbs the token trajectory, so D = min(D*, cap) holds only
    approximately and can imply S6 > S5 (beagle, republic, essays, botany).
    """
    out = {}
    for name in ORDER:
        c, e, a = (data["C"]["rec"][name], data["E"]["rec"][name],
                   data["A"]["rec"][name])
        t = tvar[name]
        s5 = (e["abar"] - c["abar"]) / t[4] if t[4] > 1e-9 else 0.0
        d_edl = a["edl"] - e["edl"]
        mu57 = (a["abar"] - e["abar"]) / d_edl if d_edl > 1e-9 else 0.0
        targets = {4: c["abar"], 5: e["abar"], 7: a["abar"]}

        def curve(theta, eta):
            return [0.0] + [math.exp(-((k / theta) ** eta))
                            for k in range(1, MAXW + 1)]

        def abar_of(S, cap):
            return sum(S[k] * t[k - 1] for k in range(1, cap + 1))

        best = None
        for ti in range(1, 601):
            theta = ti * 0.05
            for ei in range(1, 301):
                eta = ei * 0.02
                S = curve(theta, eta)
                err = sum((abar_of(S, cap) - val) ** 2
                          for cap, val in targets.items())
                if best is None or err < best[0]:
                    best = (err, theta, eta, S)
        err, theta, eta, S_curve = best
        p = None
        if model == "measured":
            lo, hi = 0.0, 1.0
            for _ in range(200):
                p = 0.5 * (lo + hi)
                if sum((p ** k) * t[k - 1] for k in range(1, 5)) < c["abar"]:
                    lo = p
                else:
                    hi = p
            p = 0.5 * (lo + hi)
            S = [0.0] + [p ** k for k in range(1, 5)] + [s5, mu57, mu57, mu57]
        else:
            S = list(S_curve)
        if deep == "pessimistic":
            ratio = S[7] / S[6] if S[6] > 1e-9 else 1.0
            S[8] = S[7] * ratio
        out[name] = {"S": S, "s5": s5, "mu57": mu57, "theta": theta, "eta": eta,
                     "p_shallow": p, "model": model,
                     "abar_fit": {cap: abar_of(S, cap) for cap in targets},
                     "abar_obs": targets}
    return out


# ----------------------------------------------------------- cost families

def local_shape_ms():
    return json.loads(E186_SHAPE.read_text())["abs_ms"]


LOCAL_MS = local_shape_ms()
LOCAL_GROUP = [0, 1, 1, 1, 1, 2, 2, 2, 3]      # FINDING 484 weight-pass index


def _widths(probs):
    return [(d + 1, q) for d, q in enumerate(probs)]


def make_basis(fn):
    def basis(probs):
        cols = None
        for m, q in _widths(probs):
            v = fn(m)
            cols = [q * x for x in v] if cols is None else \
                [c + q * x for c, x in zip(cols, v)]
        return cols
    return basis


FAMILIES = {
    "affine": make_basis(lambda m: [1.0, m]),
    "quadratic": make_basis(lambda m: [1.0, m, m * m]),
    "step6": make_basis(lambda m: [1.0, m, 1.0 if m >= 6 else 0.0]),
    "step2+step6": make_basis(lambda m: [1.0, m, 1.0 if m >= 2 else 0.0,
                                         1.0 if m >= 6 else 0.0]),
    "localshape": make_basis(lambda m: [1.0, LOCAL_MS[m - 1] / 1000.0]),
    "grouppass": make_basis(lambda m: [1.0, m, float(LOCAL_GROUP[m - 1])]),
}
FAMILY_NAMES = {
    "affine": ["F", "b"],
    "quadratic": ["F", "b", "c"],
    "step6": ["F", "b", "step6"],
    "step2+step6": ["F", "b", "step2", "step6"],
    "localshape": ["F", "scale"],
    "grouppass": ["F", "b", "gstep"],
}


def satexp_basis(tau):
    return make_basis(lambda m: [1.0, m, 1.0 - math.exp(-(m - 1) / tau)])


def smoothstep_basis(m0, w):
    return make_basis(lambda m: [1.0, m, 1.0 / (1.0 + math.exp(-(m - m0) / w))])


def fit_family(basis, obs):
    design, target, weights = [], [], []
    for _, _, probs, R, sig in obs:
        design.append(basis(probs))
        target.append(R)
        weights.append(1.0 / (sig / 1000.0) ** 2)
    beta = e177.lstsq(design, target, weights)
    if beta is None:
        return None
    resid = [t - sum(b * v for b, v in zip(beta, row))
             for row, t in zip(design, target)]
    chi2 = sum(w * r * r for r, w in zip(resid, weights))
    n = len(obs)
    k = len(beta)
    wsse = sum(w * r * r for r, w in zip(resid, weights))
    return {"beta": beta, "basis": basis, "chi2": chi2,
            "wrmse_ms": 1000 * math.sqrt(sum(r * r for r in resid) / n),
            "chi_rmse": math.sqrt(chi2 / n),
            "aic": chi2 + 2 * k,
            "aicc": chi2 + 2 * k + 2 * k * (k + 1) / max(1, n - k - 1),
            "max_resid_ms": 1000 * max(abs(r) for r in resid),
            "resid_ms": {"%s/%s" % (o[0], o[1]): 1000 * r
                         for o, r in zip(obs, resid)},
            "npar": k}


def fit_nonlinear(kind, obs):
    best = None
    if kind == "satexp":
        grid = [(tau,) for tau in [0.2 + 0.05 * i for i in range(120)]]
        maker = lambda g: satexp_basis(*g)          # noqa: E731
    else:
        grid = [(m0, w) for m0 in [3.0 + 0.1 * i for i in range(60)]
                for w in [0.05 + 0.05 * i for i in range(40)]]
        maker = lambda g: smoothstep_basis(*g)      # noqa: E731
    for g in grid:
        fit = fit_family(maker(g), obs)
        if fit is None:
            continue
        if best is None or fit["chi2"] < best["chi2"]:
            fit["hyper"] = g
            fit["npar"] += len(g)
            fit["aic"] = fit["chi2"] + 2 * fit["npar"]
            n = len(obs)
            k = fit["npar"]
            fit["aicc"] = fit["chi2"] + 2 * k + 2 * k * (k + 1) / max(1, n - k - 1)
            best = fit
    return best


def R_of(fit, m):
    """Round cost, seconds, at exactly m verified rows."""
    return sum(b * v for b, v in zip(fit["beta"],
                                     fit["basis"]([0.0] * (m - 1) + [1.0])))


def table_law(table_ms):
    """Wrap an explicit R(m) table, ms, in the cost-model interface."""
    tab = [v / 1000.0 for v in table_ms]
    return {"beta": [1.0], "table_ms": list(table_ms),
            "basis": lambda probs: [sum(q * tab[d] for d, q in enumerate(probs))]}


LOCAL_DR6 = LOCAL_MS[5] - LOCAL_MS[4]
LOCAL_DR9 = LOCAL_MS[8] - LOCAL_MS[7]


def step9_scenario(fit, mode="measured"):
    """Price the 9-row cell that no cap<=7 receipt can observe.

    E186 measured a SECOND weight-streaming group pass at 9 rows
    (harness=local, +39.83 ms, the same class of step as the 6-row cell). No
    ranked receipt at cap <= 7 verifies a 9-row round, so a ranked fit can only
    continue its own smooth form there. This scenario instead transfers the
    local 9-row step onto ranked hardware with the transfer ratio measured at
    the 6-row cell.
    """
    table = [1000 * R_of(fit, m) for m in range(1, MAXW)]
    if mode == "measured":
        ratio = DR6_MEASURED_MS / LOCAL_DR6      # FINDING 505 over E186
    else:
        ratio = 1000 * (R_of(fit, 6) - R_of(fit, 5)) / LOCAL_DR6
    return table_law(table + [table[-1] + ratio * LOCAL_DR9]), ratio


def observations(data, tvar, labels=("C", "E", "A")):
    obs = []
    for label in labels:
        cap = data[label]["cap"]
        for name in ORDER:
            rec = data[label]["rec"][name]
            probs = e177.width_probs(tvar[name], cap)
            obs.append((label, name, probs, rec["R"], sigma_R_ms(name, rec)))
    return obs


def fit_all(obs):
    fits = {}
    for key, basis in FAMILIES.items():
        fit = fit_family(basis, obs)
        if fit:
            fit["names"] = FAMILY_NAMES[key]
            fits[key] = fit
    for kind, label in (("satexp", "sat-exp"), ("smoothstep", "smooth-step")):
        fit = fit_nonlinear(kind, obs)
        if fit:
            fit["names"] = ["F", "b", "amp"] + ["hyper"]
            fits[label] = fit
    return fits


# --------------------------------------------------------------- pipeline

def chain(data, tvar, acc, fit, name, target_cap, anchor):
    """Walk a paid receipt's own state to another cap. Levels stay measured."""
    rec = data[anchor]["rec"][name]
    cap, abar, R = data[anchor]["cap"], rec["abar"], rec["R"]
    t, S = tvar[name], acc[name]["S"]
    while cap < target_cap:
        abar += t[cap] * S[cap + 1]
        R += t[cap] * (R_of(fit, cap + 2) - R_of(fit, cap + 1))
        cap += 1
    while cap > target_cap:
        abar -= t[cap - 1] * S[cap]
        R -= t[cap - 1] * (R_of(fit, cap + 1) - R_of(fit, cap))
        cap -= 1
    n_rounds = TOKENS / (1.0 + abar)
    return abar, R, n_rounds, n_rounds * R


def raw_of(data, name, decode_total, leg="A"):
    vec = data[leg]["vec"][name]
    return vec["serial"] / (decode_total / TOKENS + vec["prefill"])


def median_at(data, tvar, acc, fit, target_cap, anchor, leg=None):
    leg = leg or anchor
    raws = []
    for name in ORDER:
        _, _, _, total = chain(data, tvar, acc, fit, name, target_cap, anchor)
        raws.append(raw_of(data, name, total, leg))
    return e177.published_median(raws)


def dr6_of(fit):
    return 1000 * (R_of(fit, 6) - R_of(fit, 5))


# -------------------------------------------------------------- reporting

def table_constraints(data):
    print("\n" + "=" * 108)
    print("STAGE 1a  CONSTRAINT TABLE: per prompt x per paid receipt "
          "(harness=ranked)")
    print("=" * 108)
    print("  %-9s %-3s %6s %8s %8s %8s %9s %8s %8s"
          % ("prompt", "rcp", "rounds", "edl", "Mbar", "abar", "R_ms",
             "sigma_ms", "weight"))
    for name in ORDER:
        for label in ("C", "E", "A"):
            rec = data[label]["rec"][name]
            sig = sigma_R_ms(name, rec)
            print("  %-9s %-3s %6d %8.4f %8.4f %8.4f %9.4f %8.3f %8.2f"
                  % (name, "%s%d" % (label, data[label]["cap"]), rec["rounds"],
                     rec["edl"], rec["Mbar"], rec["abar"], 1000 * rec["R"],
                     sig, 1.0 / sig ** 2))
    print("\n  E159 round-count cross-check on receipt A: %s"
          % ("PASS" if all(data["A"]["rec"][n]["rounds"] == e177.E159_ROUNDS_A[n]
                           for n in ORDER) else "FAIL"))
    print("  replicate instrument (FINDING 506): prompts whose edl is identical")
    print("  across all three receipts measure the channel directly")
    for name in ORDER:
        edls = {round(data[l]["rec"][name]["edl"], 9) for l in ("C", "E", "A")}
        if len(edls) == 1:
            vals = [1000 * data[l]["rec"][name]["R"] for l in ("C", "E", "A")]
            pred = sigma_R_ms(name, data["A"]["rec"][name])
            print("    %-9s R = %s ms   spread %.3f ms   FINDING 503 sigma "
                  "%.3f ms" % (name, " ".join("%.3f" % v for v in vals),
                               max(vals) - min(vals), pred))


def table_schedule(data, tvar, exact, acc):
    print("\n" + "=" * 108)
    print("STAGE 1b  SCHEDULE IDENTIFICATION: exact tail masses and acceptance "
          "(harness=ranked)")
    print("=" * 108)
    print("  %-9s %7s %7s %7s %8s %8s %8s %8s %8s %8s"
          % ("prompt", "t0", "edl4", "t4*", "t5+t6*", "t5", "t6", "t7~",
             "S5*", "mu(5:7)*"))
    for name in ORDER:
        t, x, a = tvar[name], exact[name], acc[name]
        print("  %-9s %7.4f %7.4f %7.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (name, t[0], x["edl4"], x["t4"], x["t56"], t[5], t[6], t[7],
                 a["s5"], a["mu57"]))
    print("  * = exact, family-free. ~ = extrapolated.")
    print("\n  PRIMARY acceptance model: measured rates. S5 and mu(5:7) are the")
    print("  measured values, so abar(5) and abar(7) reproduce exactly; only S8")
    print("  is extrapolated (flat continuation of the measured deep rate).")
    print("  %-9s %7s %7s %7s %7s %7s | %9s %9s %9s"
          % ("prompt", "S4", "S5*", "S6*", "S7*", "S8~", "d abar4", "d abar5",
             "d abar7"))
    for name in ORDER:
        a = acc[name]
        S = a["S"]
        print("  %-9s %7.4f %7.4f %7.4f %7.4f %7.4f | %9.5f %9.5f %9.5f"
              % (name, S[4], S[5], S[6], S[7], S[8],
                 a["abar_fit"][4] - a["abar_obs"][4],
                 a["abar_fit"][5] - a["abar_obs"][5],
                 a["abar_fit"][7] - a["abar_obs"][7]))
    print("\n  CONSERVATIVE acceptance model: monotone S_k = exp(-(k/theta)^eta)")
    print("  fitted to the same three abar values. Its abar residuals show how")
    print("  far the measured cells are from an admissible nested schedule: a")
    print("  cap change perturbs the trajectory, so S6 > S5 can appear.")
    curve = acceptance(data, tvar, model="curve")
    print("  %-9s %7s %7s %7s %7s %7s | %9s %9s %9s | %s"
          % ("prompt", "S4", "S5", "S6", "S7", "S8", "d abar4", "d abar5",
             "d abar7", "note"))
    for name in ORDER:
        a = curve[name]
        S = a["S"]
        note = "measured S6>S5" if a["mu57"] > a["s5"] + 1e-9 else ""
        print("  %-9s %7.4f %7.4f %7.4f %7.4f %7.4f | %9.5f %9.5f %9.5f | %s"
              % (name, S[4], S[5], S[6], S[7], S[8],
                 a["abar_fit"][4] - a["abar_obs"][4],
                 a["abar_fit"][5] - a["abar_obs"][5],
                 a["abar_fit"][7] - a["abar_obs"][7], note))


def table_fits(fits, data, tvar, acc):
    print("\n" + "=" * 118)
    print("STAGE 1c  FIT COMPARISON on 24 paid observations (harness=ranked)")
    print("=" * 118)
    order = sorted(fits, key=lambda k: fits[k]["aicc"])
    print("  %-13s %4s %9s %9s %9s %10s | %11s %11s %11s | %9s %9s"
          % ("family", "par", "chi2", "AICc", "wRMSE", "maxres",
             "cap4 med", "cap5 med", "cap7 med", "dR6 ms", "R(1) ms"))
    for key in order:
        f = fits[key]
        m4 = median_at(data, tvar, acc, f, 4, "A", "C")
        m5 = median_at(data, tvar, acc, f, 5, "A", "E")
        m7 = median_at(data, tvar, acc, f, 7, "C", "A")
        print("  %-13s %4d %9.2f %9.2f %9.4f %10.3f | %11.6f %11.6f %11.6f "
              "| %9.3f %9.3f"
              % (key, f["npar"], f["chi2"], f["aicc"], f["wrmse_ms"],
                 f["max_resid_ms"], m4, m5, m7, dr6_of(f),
                 1000 * R_of(f, 1)))
    print("  paid medians: cap4 %.6f  cap5 %.6f  cap7 %.6f" % (CAP4, CAP5, BEST_A))
    print("  mandatory checks: dR6 must land in [%.3f, %.3f] ms (FINDING 505);"
          % DR6_BAND_MS)
    print("  R(1) is a held-out anchor at %.3f ms (FINDING 330)" % R1_FLOOR_MS)

    print("\n  MANDATORY CHECK TABLE")
    print("  %-13s %11s %11s %11s %11s %11s %11s"
          % ("family", "cap4 err%", "cap5 err%", "cap7 err%", "dR6 dev",
             "R(1) err", "verdict"))
    verdicts = {}
    for key in order:
        f = fits[key]
        e4 = 100 * (median_at(data, tvar, acc, f, 4, "A", "C") / CAP4 - 1)
        e5 = 100 * (median_at(data, tvar, acc, f, 5, "A", "E") / CAP5 - 1)
        e7 = 100 * (median_at(data, tvar, acc, f, 7, "C", "A") / BEST_A - 1)
        dr6 = dr6_of(f)
        inband = DR6_BAND_MS[0] <= dr6 <= DR6_BAND_MS[1]
        edge = min(abs(dr6 - DR6_BAND_MS[0]), abs(dr6 - DR6_BAND_MS[1]))
        r1err = 1000 * R_of(f, 1) - R1_FLOOR_MS
        med_ok = max(abs(e4), abs(e5), abs(e7)) < 100 * SIGMA_PUBLISHED
        r1_ok = abs(r1err) < 3.0
        if med_ok and inband and r1_ok:
            grade = "PASS"
        elif med_ok and r1_ok and edge <= 1.0:
            grade = "marginal"
        else:
            grade = "fail"
        verdicts[key] = grade
        print("  %-13s %11.3f %11.3f %11.3f %11s %11.3f %11s"
              % (key, e4, e5, e7, "IN" if inband else "%+.2f" %
                 (dr6 - DR6_MEASURED_MS), r1err, grade))
    print("  median errors are judged against the 1-sigma receipt channel "
          "(+-%.3f%%)" % (100 * SIGMA_PUBLISHED))
    print("  PASS = all three checks; marginal = medians and R(1) pass and dR6 "
          "sits within 1.0 ms of the FINDING 505 band; fail otherwise")
    return order, verdicts


def table_loo(data, tvar, acc):
    """Leave-one-receipt-out: the only honest test of depth extrapolation."""
    print("\n" + "=" * 108)
    print("STAGE 1d  LEAVE-ONE-RECEIPT-OUT EXTRAPOLATION (harness=ranked)")
    print("=" * 108)
    print("  Fit the law on two paid caps, predict the third published median.")
    print("  The C+E -> A row is the direction cap-8 pricing needs: fit shallow,")
    print("  predict deep.")
    cases = [(("C", "E"), "A", 7, "E", "A", BEST_A),
             (("C", "A"), "E", 5, "A", "E", CAP5),
             (("E", "A"), "C", 4, "A", "C", CAP4)]
    out = {}
    for train, held, cap, anchor, leg, paid in cases:
        obs = observations(data, tvar, train)
        fits = fit_all(obs)
        print("\n  train %s -> predict %s (cap %d, paid %.8f)"
              % ("+".join(train), held, cap, paid))
        print("    %-13s %9s %13s %11s %10s"
              % ("family", "chi2", "predicted", "err %", "err sigma"))
        for key in sorted(fits, key=lambda k: fits[k]["aicc"]):
            f = fits[key]
            pred = median_at(data, tvar, acc, f, cap, anchor, leg)
            err = 100 * (pred / paid - 1)
            print("    %-13s %9.2f %13.8f %11.3f %10.2f"
                  % (key, f["chi2"], pred, err, err / (100 * SIGMA_PUBLISHED)))
            out.setdefault(key, {})["+".join(train)] = err
    print("\n  mean |err| across the three held-out receipts, per family")
    for key in sorted(out, key=lambda k: sum(abs(v) for v in out[k].values())):
        vals = out[key]
        print("    %-13s %7.3f %%   (%s)"
              % (key, sum(abs(v) for v in vals.values()) / len(vals),
                 " ".join("%s %+.2f%%" % (k, v) for k, v in vals.items())))
    return out


def table_widths(fits, order):
    print("\n" + "=" * 108)
    print("STAGE 1e  ROUND COST AND MARGINAL ROW COST BY WIDTH (harness=ranked)")
    print("=" * 108)
    print("  R(m) ms")
    print("  %-13s %s" % ("family", " ".join("%7s" % ("m=%d" % m)
                                             for m in range(1, MAXW + 1))))
    for key in order:
        f = fits[key]
        print("  %-13s %s" % (key, " ".join("%7.2f" % (1000 * R_of(f, m))
                                            for m in range(1, MAXW + 1))))
    print("\n  marginal ms for the m-th row, dR(m) = R(m) - R(m-1)")
    print("  %-13s %s" % ("family", " ".join("%7s" % ("dR%d" % m)
                                             for m in range(2, MAXW + 1))))
    for key in order:
        f = fits[key]
        print("  %-13s %s"
              % (key, " ".join("%7.2f" % (1000 * (R_of(f, m) - R_of(f, m - 1)))
                               for m in range(2, MAXW + 1))))
    print("  %-13s %s" % ("local (E186)",
                          " ".join("%7.2f" % (LOCAL_MS[m - 1] - LOCAL_MS[m - 2])
                                   for m in range(2, MAXW + 1))))
    print("  harness=local row is E186's measured round reconstruction; it is a")
    print("  SHAPE reference only and is never used to price ranked value.")
    print("\n  ranked/local transfer ratio per cell, dR_ranked / dR_local")
    for key in order:
        f = fits[key]
        vals = []
        for m in range(2, MAXW + 1):
            loc = LOCAL_MS[m - 1] - LOCAL_MS[m - 2]
            vals.append(1000 * (R_of(f, m) - R_of(f, m - 1)) / loc)
        print("  %-13s %s" % (key, " ".join("%7.3f" % v for v in vals)))


def table_breakeven(data, tvar, acc, fits, order):
    print("\n" + "=" * 108)
    print("STAGE 1f  PER-PROMPT BREAK-EVEN FOR THE 9-ROW CELL (harness=ranked)")
    print("=" * 108)
    print("  Raising the cap 7 -> 8 pays on a prompt when the 9-row marginal is")
    print("  below Rbar * S8 / (1 + abar) at that prompt's cap-7 state.")
    print("  %-9s %9s %8s %8s %8s %12s"
          % ("prompt", "Rbar_ms", "abar", "t7~", "S8~", "break-even"))
    be = {}
    for name in ORDER:
        rec = data["A"]["rec"][name]
        S8 = acc[name]["S"][8]
        val = 1000 * rec["R"] * S8 / (1.0 + rec["abar"])
        be[name] = val
        print("  %-9s %9.3f %8.4f %8.4f %8.4f %12.3f"
              % (name, 1000 * rec["R"], rec["abar"], tvar[name][7], S8, val))
    print("\n  predicted 9-row marginal dR9 by family, against that band "
          "[%.2f, %.2f] ms" % (min(be.values()), max(be.values())))
    for key in order:
        f = fits[key]
        d9 = 1000 * (R_of(f, 9) - R_of(f, 8))
        pays = sum(1 for n in ORDER if d9 < be[n] and tvar[n][7] > 1e-6)
        print("    %-13s dR9 = %8.3f ms   pays on %d of 8 prompts" % (key, d9, pays))
    return be


def table_pricing(data, tvar, acc, fits, order):
    print("\n" + "=" * 108)
    print("STAGE 1g  PRICED CELLS: published median by cap (harness=ranked)")
    print("=" * 108)
    print("  walked from the paid cap-7 receipt A; leg vectors from A")
    print("  %-13s %12s %12s %12s %12s %12s"
          % ("family", "cap 4", "cap 5", "cap 6", "cap 7", "cap 8"))
    priced = {}
    for key in order:
        f = fits[key]
        vals = [median_at(data, tvar, acc, f, cap, "A") for cap in (4, 5, 6, 7, 8)]
        priced[key] = dict(zip((4, 5, 6, 7, 8), vals))
        print("  %-13s %12.8f %12.8f %12.8f %12.8f %12.8f" % tuple([key] + vals))
    print("  paid anchors:  cap4 %.8f   cap5 %.8f   cap7 %.8f"
          % (CAP4, CAP5, BEST_A))
    print("  crown %.8f   receipt A %.8f   1 sigma %.5f"
          % (CROWN, BEST_A, SIGMA_PUBLISHED * BEST_A))
    return priced


def table_oracle(data, tvar, acc, fits, best_key):
    f = fits[best_key]
    print("\n" + "=" * 108)
    print("STAGE 1h  PER-PROMPT ORACLE CAP under %s (harness=ranked)" % best_key)
    print("=" * 108)
    print("  %-9s %6s %11s %11s %9s" % ("prompt", "cap*", "raw*", "raw@cap7",
                                        "rank@7"))
    order7 = sorted(ORDER, key=lambda n: data["A"]["vec"][n]["raw"])
    raws = []
    for name in ORDER:
        opts = []
        for cap in range(4, 9):
            _, _, _, total = chain(data, tvar, acc, f, name, cap, "A")
            opts.append((total, cap))
        total, cap = min(opts)
        raw = raw_of(data, name, total)
        raws.append(raw)
        print("  %-9s %6d %11.5f %11.5f %9d"
              % (name, cap, raw, data["A"]["vec"][name]["raw"],
                 order7.index(name) + 1))
    med = e177.published_median(raws)
    print("  oracle per-prompt-cap median %.8f  (paid cap-7 %.8f, %+0.3f%%)"
          % (med, BEST_A, 100 * (med / BEST_A - 1)))
    return med


def table_step9(data, tvar, acc, fits, order):
    """Cap-8 price under the two admissible treatments of the 9-row cell."""
    print("\n" + "=" * 108)
    print("STAGE 1g2  THE 9-ROW CELL IS INVISIBLE TO EVERY PAID RECEIPT")
    print("=" * 108)
    print("  A cap-7 receipt verifies at most 8 rows per round, so no ranked")
    print("  observation contains R(9). Each family's cap-8 price is therefore")
    print("  its own smooth continuation. E186 measured a second weight-pass")
    print("  step exactly at 9 rows (harness=local, dR9 = %.2f ms vs dR6 = %.2f"
          % (LOCAL_DR9, LOCAL_DR6))
    print("  ms). Transferring it with the ratio measured at the 6-row cell,")
    print("  dR6_ranked / dR6_local = %.3f, gives dR9_ranked = %.2f ms."
          % (DR6_MEASURED_MS / LOCAL_DR6,
             LOCAL_DR9 * DR6_MEASURED_MS / LOCAL_DR6))
    print("  %-13s %10s %13s | %10s %13s | %11s"
          % ("family", "dR9 flat", "cap8 flat", "dR9 step", "cap8 step",
             "swing"))
    out = {}
    for key in order:
        f = fits[key]
        flat_d9 = 1000 * (R_of(f, 9) - R_of(f, 8))
        flat = median_at(data, tvar, acc, f, 8, "A")
        law, ratio = step9_scenario(f, "measured")
        step_d9 = law["table_ms"][8] - law["table_ms"][7]
        stepped = median_at(data, tvar, acc, law, 8, "A")
        out[key] = {"flat_dr9_ms": flat_d9, "flat_cap8": flat,
                    "step_dr9_ms": step_d9, "step_cap8": stepped,
                    "transfer_ratio": ratio}
        print("  %-13s %10.3f %13.8f | %10.3f %13.8f | %11.8f"
              % (key, flat_d9, flat, step_d9, stepped, flat - stepped))
    print("  crown %.8f   receipt A %.8f" % (CROWN, BEST_A))
    return out


def cell_law(index, value_ms):
    """A law whose only non-zero marginal is the cell R(index) - R(index-1)."""
    table = [0.0] * (index - 1) + [value_ms] * (MAXW - index + 1)
    return table_law(table)


def table_cell_inversion(data, tvar, acc, fits, order):
    """Reduce each open cap to the one cost cell that decides it.

    Walking the paid cap-7 receipt to cap 8 uses exactly one cost-law value,
    t7 * (R(9) - R(8)); walking it down to cap 6 uses only t6 * (R(8) - R(7)).
    Everything else in the walk is measured. Each open cap is therefore a
    monotone function of a single unknown marginal, and the decision reduces to
    a threshold on that marginal.
    """
    print("\n" + "=" * 108)
    print("STAGE 1g3  SINGLE-CELL INVERSION OF THE OPEN CAPS (harness=ranked)")
    print("=" * 108)
    out = {}
    for cap, index, label in ((8, 9, "dR9 = R(9) - R(8)"), (6, 8, "dR8 = R(8) - R(7)")):
        def med(v):
            return median_at(data, tvar, acc, cell_law(index, v), cap, "A")
        lo, hi = -5.0, 40.0
        rising = med(hi) > med(lo)
        print("\n  cap %d as a function of %s  (%s in the cell cost)"
              % (cap, label, "rising" if rising else "falling"))
        print("    %-10s %s" % ("cell ms", " ".join("%8.1f" % v
                                                    for v in (0, 2, 4, 6, 8,
                                                              10, 12, 15))))
        print("    %-10s %s" % ("cap %d" % cap,
                                " ".join("%8.5f" % med(v)
                                         for v in (0, 2, 4, 6, 8, 10, 12, 15))))
        thresholds = {}
        for tname, target in (("receipt A", BEST_A), ("crown", CROWN)):
            a, b = -5.0, 60.0
            for _ in range(200):
                mid = 0.5 * (a + b)
                if (med(mid) > target) == rising:
                    b = mid
                else:
                    a = mid
            thresholds[tname] = 0.5 * (a + b)
            print("    cap %d equals %-9s (%.8f) at %s = %+8.3f ms"
                  % (cap, tname, target, label.split(" =")[0],
                     thresholds[tname]))
        evidence = {k: 1000 * (R_of(fits[k], index) - R_of(fits[k], index - 1))
                    for k in order}
        print("    ranked-fit estimates of that cell: %s"
              % "  ".join("%s %.2f" % (k, v) for k, v in evidence.items()))
        loc = LOCAL_MS[index - 1] - LOCAL_MS[index - 2]
        print("    E186 local cell %.2f ms (harness=local); transferred at the "
              "6-row ratio %.3f -> %.2f ms"
              % (loc, DR6_MEASURED_MS / LOCAL_DR6,
                 loc * DR6_MEASURED_MS / LOCAL_DR6))
        out[cap] = {"cell": label, "thresholds": thresholds,
                    "fit_estimates_ms": evidence, "local_ms": loc,
                    "local_transferred_ms": loc * DR6_MEASURED_MS / LOCAL_DR6}
    return out


def montecarlo(data, tvar, acc, fits, order, loo, draws=6000, scheme="mixed"):
    """Win probability against the crown for cap 6 and cap 8.

    Five uncertainty sources are combined: (1) family choice, sampled by the
    selected weight scheme; (2) the 9-row cell treatment, flat continuation or
    transferred local step, 50/50; (3) the extrapolated tail mass t7 and the
    deep acceptance cell S8; (4) the measured one-cap extrapolation error of
    that family, from the leave-one-receipt-out table, as a Gaussian
    systematic; (5) the receipt channel, sigma = 0.689% per draw.

    AICc scores in-sample fit only. This experiment needs one-cap
    extrapolation, and the leave-one-receipt-out table shows the in-sample
    ranking does not predict extrapolation. The default `mixed` scheme
    therefore averages the AICc weight with a leave-one-out weight so a single
    family cannot dominate the band on in-sample evidence alone.
    """
    import random
    random.seed(197)
    variants = {}
    for key, tail, model in (("central", "family", "measured"),
                             ("t7-heavy", "heavy", "measured"),
                             ("t7-zero", "zero", "measured"),
                             ("accept-curve", "family", "curve")):
        t, _ = survival(data, tail)
        variants[key] = (t, acceptance(data, t, model=model))
    vkeys = list(variants)
    vprob = [0.4, 0.2, 0.2, 0.2]
    amin = min(fits[k]["aicc"] for k in order)
    aic_w = {k: math.exp(-0.5 * (fits[k]["aicc"] - amin)) for k in order}
    tot = sum(aic_w.values())
    aic_w = {k: v / tot for k, v in aic_w.items()}
    loo_score = {k: sum(abs(v) for v in loo[k].values()) / len(loo[k])
                 for k in order}
    scale = min(loo_score.values())
    loo_w = {k: math.exp(-0.5 * (loo_score[k] / scale) ** 2) for k in order}
    tot = sum(loo_w.values())
    loo_w = {k: v / tot for k, v in loo_w.items()}
    if scheme == "aicc":
        wts = aic_w
    elif scheme == "loo":
        wts = loo_w
    else:
        wts = {k: 0.5 * aic_w[k] + 0.5 * loo_w[k] for k in order}
    print("\n" + "=" * 108)
    print("STAGE 1i  WIN PROBABILITY vs the crown %.6f (harness=ranked), "
          "weights=%s" % (CROWN, scheme))
    print("=" * 108)
    print("  AICc weights (in-sample): %s"
          % "  ".join("%s %.3f" % (k, aic_w[k]) for k in order))
    print("  LOO weights (one-cap extrapolation): %s"
          % "  ".join("%s %.3f" % (k, loo_w[k]) for k in order))
    print("  sampled weights: %s"
          % "  ".join("%s %.3f" % (k, wts[k]) for k in order))
    sysmatic = {k: math.sqrt(sum(v * v for v in loo[k].values())
                             / len(loo[k])) / 100.0 for k in order}
    print("  one-cap extrapolation systematic per family (RMS leave-one-out): %s"
          % "  ".join("%s %.2f%%" % (k, 100 * sysmatic[k]) for k in order))
    out = {}
    for cap in (6, 8):
        vals = []
        for _ in range(draws):
            u, tail_u, scen_u = random.random(), random.random(), random.random()
            acc_run = 0.0
            key = order[-1]
            for k in order:
                acc_run += wts[k]
                if u <= acc_run:
                    key = k
                    break
            run = 0.0
            vkey = vkeys[-1]
            for name, prob in zip(vkeys, vprob):
                run += prob
                if tail_u <= run:
                    vkey = name
                    break
            tv, av = variants[vkey]
            law = fits[key]
            if cap == 8 and scen_u < 0.5:
                law = step9_scenario(fits[key], "measured")[0]
            central = median_at(data, tv, av, law, cap, "A")
            central *= (1.0 + random.gauss(0.0, sysmatic[key]))
            vals.append(central * (1.0 + random.gauss(0.0, SIGMA_PUBLISHED)))
        vals.sort()
        central = vals[len(vals) // 2]
        pcrown = sum(1 for v in vals if v > CROWN) / len(vals)
        pa = sum(1 for v in vals if v > BEST_A) / len(vals)
        out[cap] = {"median": central, "p05": vals[int(0.05 * len(vals))],
                    "p95": vals[int(0.95 * len(vals))],
                    "p_beat_crown": pcrown, "p_beat_A": pa}
        print("  cap %d  median %.8f  90%% band [%.8f, %.8f]  "
              "P(beat crown) %.3f  P(beat receipt A) %.3f"
              % (cap, central, out[cap]["p05"], out[cap]["p95"], pcrown, pa))
    return out, wts


def stage2(data, tvar, acc, fits, order):
    """Conditional repricing if E195/E196 shave the 6-row cell by X ms."""
    print("\n" + "=" * 108)
    print("STAGE 2  CONDITIONAL REPRICING: shave the 6-row step by X ms "
          "(harness=ranked)")
    print("=" * 108)
    print("  A shave of X reduces R(m) for every m >= 6, so it also lowers the")
    print("  paid cap-7 anchor by X * P(rows >= 6) on each prompt. Both effects")
    print("  are applied; the depth optimum is then re-solved.")
    rows = []
    for key in order[:3]:
        base = fits[key]
        for X in (0.0, 2.0, 5.0, 8.9):
            shifted = {"beta": base["beta"], "basis": base["basis"]}

            def make(fit, X):
                def R(m):
                    return R_of(fit, m) - (X / 1000.0 if m >= 6 else 0.0)
                return R
            Rfn = make(base, X)
            # patched cost model with the same interface as R_of
            patched = {"beta": [1.0], "basis": (lambda probs, Rfn=Rfn:
                                                [sum(q * Rfn(d + 1)
                                                     for d, q in enumerate(probs))])}
            meds = {}
            for cap in (5, 6, 7, 8):
                raws = []
                for name in ORDER:
                    rec = data["A"]["rec"][name]
                    probs = e177.width_probs(tvar[name], 7)
                    shave = X / 1000.0 * sum(q for d, q in enumerate(probs)
                                             if d + 1 >= 6)
                    saved = dict(rec)
                    saved["R"] = rec["R"] - shave
                    stash = data["A"]["rec"][name]
                    data["A"]["rec"][name] = saved
                    _, _, _, total = chain(data, tvar, acc, patched, name, cap, "A")
                    data["A"]["rec"][name] = stash
                    raws.append(raw_of(data, name, total))
                meds[cap] = e177.published_median(raws)
            best_cap = max(meds, key=lambda c: meds[c])
            rows.append((key, X, meds, best_cap))
            del shifted
    print("  %-13s %7s %12s %12s %12s %12s %8s"
          % ("family", "shave", "cap 5", "cap 6", "cap 7", "cap 8", "cap*"))
    for key, X, meds, best_cap in rows:
        print("  %-13s %7.1f %12.8f %12.8f %12.8f %12.8f %8d"
              % (key, X, meds[5], meds[6], meds[7], meds[8], best_cap))
    return rows


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = build_receipts()
    tvar, exact = survival(data)
    acc = acceptance(data, tvar)
    obs = observations(data, tvar)
    fits = fit_all(obs)

    table_constraints(data)
    table_schedule(data, tvar, exact, acc)
    order, verdicts = table_fits(fits, data, tvar, acc)
    loo = table_loo(data, tvar, acc)
    table_widths(fits, order)
    be = table_breakeven(data, tvar, acc, fits, order)
    priced = table_pricing(data, tvar, acc, fits, order)
    step9 = table_step9(data, tvar, acc, fits, order)
    cells = table_cell_inversion(data, tvar, acc, fits, order)
    oracle = table_oracle(data, tvar, acc, fits, order[0])
    mc, wts = montecarlo(data, tvar, acc, fits, order, loo, scheme="mixed")
    mc_aicc, _ = montecarlo(data, tvar, acc, fits, order, loo, scheme="aicc")
    s2 = stage2(data, tvar, acc, fits, order) if cmd in ("all", "stage2") else []

    payload = {
        "harness": "ranked",
        "receipts": {l: {"id": data[l]["id"], "cap": data[l]["cap"],
                         "score": data[l]["score"]} for l in ("A", "C", "E")},
        "constraints": {n: {l: {k: data[l]["rec"][n][k]
                                for k in ("rounds", "edl", "Mbar", "abar", "R")}
                            for l in ("C", "E", "A")} for n in ORDER},
        "schedule": {n: {"t": tvar[n], "S": acc[n]["S"], "s5": acc[n]["s5"],
                         "mu57": acc[n]["mu57"]} for n in ORDER},
        "fits": {k: {"beta": fits[k]["beta"], "chi2": fits[k]["chi2"],
                     "aicc": fits[k]["aicc"], "wrmse_ms": fits[k]["wrmse_ms"],
                     "npar": fits[k]["npar"],
                     "R_ms": [1000 * R_of(fits[k], m)
                              for m in range(1, MAXW + 1)],
                     "dR6_ms": dr6_of(fits[k]),
                     "dR9_ms": 1000 * (R_of(fits[k], 9) - R_of(fits[k], 8)),
                     "check_grade": verdicts[k],
                     "sample_weight": wts.get(k)} for k in fits},
        "loo_percent_error": loo,
        "breakeven_ms": be,
        "priced": priced,
        "step9_scenarios": step9,
        "cell_inversion": cells,
        "local_shape_ms": LOCAL_MS,
        "oracle_median": oracle,
        "montecarlo": mc,
        "montecarlo_aicc_weights": mc_aicc,
        "stage2": [{"family": k, "shave_ms": X, "medians": m, "best_cap": c}
                   for k, X, m, c in s2],
        "constants": {"crown": CROWN, "receiptA": BEST_A, "cap4": CAP4,
                      "cap5": CAP5, "sigma_published": SIGMA_PUBLISHED,
                      "mue_ms": MUE_MS, "dr6_measured_ms": DR6_MEASURED_MS},
    }
    dest = pathlib.Path("research/out/e197")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "refit.json").write_text(json.dumps(payload, indent=2))
    print("\n  wrote research/out/e197/refit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
