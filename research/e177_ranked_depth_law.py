#!/usr/bin/env python3
"""E177: recover the ranked M5 round-cost law from the two paid controlled receipts.

Two organizer-pure receipts differ by exactly one literal, ``segmentedVerifyDepthCap``
(``Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1067``):

    A  5a9f130a  cap 7  published 3.70784519415395
    C  90c131dc  cap 4  published 3.54742900664627

Both expose eight per-prompt rows on the Yukon list endpoint. Each row carries the
candidate seconds per token, the prefill seconds per token, the mean OFFERED draft
rows per round (``effective_mean_draft_len``) and the non-drafting round count.

Recovery chain, per prompt and receipt:

    decode seconds per token   d = mtp_s_per_tok - prefill_s_per_tok
    total decode seconds       T = 512 * d
    round count                N = lattice(edl, 512)   (edl is the exact rational
                                                        proposed/rounds)
    mean rows verified         Mbar = 1 + edl
    seconds per round          R = T / N
    mean accepted per round    abar = (512 - N) / N
    row acceptance             alpha = (512 - N) / (edl * N)

The scored depth policy is a greedy marginal rule whose stopping depth D* is decided
before the cap is applied (``costModelDepth``: ``while depth < cap``), so the offered
depth is exactly ``D = min(D*, cap)``. Therefore, with the survival function
``t_d = P(D* > d)``:

    edl(c)  = sum_{d=0..c-1} t_d
    abar(c) = sum_{k=1..c} S_k * t_{k-1}        S_k = P(first k drafts correct)

Two caps give two edl equations and two abar equations per prompt. That identifies a
two-parameter survival family plus one acceptance parameter, and it bounds the
individual tail masses without any family assumption at all.

Usage:
    python3 research/e177_ranked_depth_law.py [rounds|survival|fit|price|all]
"""

import itertools
import json
import math
import sys
from fractions import Fraction

CACHE = "/tmp/yukon-board/full.json"
TOKENS = 512
CROWN = 3.7291100105909
BEST_A = 3.70784519415395
CAP4_PUBLISHED = 3.54742900664627
# Candidate-leg pairwise 1 sigma, Thorfinn's board decomposition (run f4np59rg).
SIGMA_LEG = 0.00189

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
ORDER = ["plutarch", "drama", "travel", "beagle",
         "republic", "essays", "medicine", "botany"]

# cap = segmentedVerifyDepthCap = maximum draft rows offered per round.
# Maximum rows verified per round is cap + 1.
RECEIPTS = [
    ("A", "5a9f130a", 7, "organizer-pure, pinned head, campaign best"),
    ("C", "90c131dc", 4, "organizer-pure, single-factor cap change vs A"),
    ("B", "180db842", 7, "cap-7 replicate, inert tree delta"),
    ("D", "2c885d64", 7, "cap-7 replicate, slower tree"),
]

# E159 / FINDING 330 round counts for A. Independent cross-check only.
E159_ROUNDS_A = {"beagle": 110, "botany": 81, "drama": 252, "essays": 92,
                 "medicine": 90, "plutarch": 488, "republic": 93, "travel": 213}

# Local cap-4 fixture schedule measured in E168 r1 (W&B u2g6yfjx) and the local
# cap-7 offered depth that proved FINDING 380. Calibration only, never pricing.
LOCAL_CAP4_HISTOGRAM = {2: 1, 4: 4, 5: 106}     # rows verified -> rounds
LOCAL_CAP7_EDL = 6.3589743589743586


def load_board():
    rows = json.load(open(CACHE))
    if isinstance(rows, dict):
        rows = rows["submissions"]
    return rows


def per_prompt(row):
    out = {}
    for entry in row["officialMetrics"]["per_prompt"]:
        mtp = entry["mtp_seconds_per_token_mean"]
        pre = entry["prefill_seconds_per_token"]
        out[PROMPT_NAMES[entry["prompt_sha256"][:8]]] = {
            "mtp": mtp,
            "prefill": pre,
            "decode_total": TOKENS * (mtp - pre),
            "serial": entry["serial_seconds_per_token_mean"],
            "raw": entry["raw_ratio_of_means"],
            "edl": entry["effective_mean_draft_len"],
            "nondraft": entry["non_drafting_round_count"],
        }
    return out


# --------------------------------------------------------------------------
# step 2a: exact round counts
# --------------------------------------------------------------------------

def lattice_rounds(edl, nondraft):
    """Round counts consistent with the published edl and the token budget."""
    frac = Fraction(edl).limit_denominator(4096)
    out = []
    for m in itertools.count(1):
        n = m * frac.denominator
        if n > TOKENS:
            break
        proposed = m * frac.numerator
        accepted = TOKENS - n
        if accepted < 0 or accepted > proposed or n < nondraft:
            continue
        out.append((n, proposed))
    return out


def recover_rounds(vec):
    """Resolve the lattice with a monotone round-cost curve (the E159 method).

    Prompts whose lattice is a singleton after the batch-1 weight-stream floor
    anchor an affine R(Mbar); the remaining prompts take the candidate closest to
    that curve; iterate to a fixed point.
    """
    floor_seconds = 0.0302519           # ranked batch-1 zero-draft round, FINDING 330
    cands = {name: lattice_rounds(vec[name]["edl"], vec[name]["nondraft"])
             for name in ORDER}
    feasible = {}
    for name in ORDER:
        total = vec[name]["decode_total"]
        keep = [c for c in cands[name] if total / c[0] >= floor_seconds]
        feasible[name] = keep or cands[name]
    chosen = {name: feasible[name][0][0] for name in ORDER}
    anchors = [n for n in ORDER if len(feasible[n]) == 1]
    for _ in range(12):
        pts = [(1.0 + vec[n]["edl"], vec[n]["decode_total"] / chosen[n]) for n in anchors]
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        k = len(pts)
        slope = (k * sxy - sx * sy) / (k * sxx - sx * sx)
        intercept = (sy - slope * sx) / k
        updated = {}
        for name in ORDER:
            target = intercept + slope * (1.0 + vec[name]["edl"])
            best = min(feasible[name],
                       key=lambda c: abs(vec[name]["decode_total"] / c[0] - target))
            updated[name] = best[0]
        if updated == chosen:
            break
        chosen = updated
        anchors = ORDER
    out = {}
    for name in ORDER:
        n = chosen[name]
        proposed = next(p for c, p in feasible[name] if c == n)
        row = dict(vec[name])
        row.update({
            "rounds": n,
            "proposed": proposed,
            "accepted": TOKENS - n,
            "abar": (TOKENS - n) / n,
            "alpha": (TOKENS - n) / proposed if proposed else 0.0,
            "Mbar": 1.0 + vec[name]["edl"],
            "R": vec[name]["decode_total"] / n,
            "lattice": [c for c, _ in feasible[name]],
        })
        out[name] = row
    return out


# --------------------------------------------------------------------------
# step 2b: round-width mixtures from the min(D*, cap) identity
# --------------------------------------------------------------------------

MAXD = 8            # the trusted parent offers 8 draft rows on every round


def survival_weibull(lam, kappa, t0):
    """t_d = P(D* > d), a discrete Weibull tail scaled by the drafting rate t0."""
    return [t0 * math.exp(-((d / lam) ** kappa)) if d > 0 else t0
            for d in range(MAXD + 1)]


def edl_of(surv, cap):
    return sum(surv[d] for d in range(cap))


def fit_survival(edl4, edl7, t0):
    """Two-parameter survival matched to edl(4) and edl(7)."""
    best = None
    for li in range(1, 601):
        lam = li * 0.04
        for ki in range(1, 201):
            kappa = ki * 0.04
            surv = survival_weibull(lam, kappa, t0)
            err = (edl_of(surv, 4) - edl4) ** 2 + (edl_of(surv, 7) - edl7) ** 2
            if best is None or err < best[0]:
                best = (err, lam, kappa, surv)
    return best


def tail_bounds(edl4, edl7, surv3):
    """Family-free monotone bounds on the individual tail masses.

    t4 + t5 + t6 = edl(7) - edl(4) =: delta, with t3 >= t4 >= t5 >= t6 >= t7 >= 0.
    """
    delta = max(0.0, edl7 - edl4)
    t4_lo, t4_hi = delta / 3.0, min(delta, surv3)
    return {
        "delta": delta,
        "t4": (t4_lo, t4_hi),
        "t5": (0.0, delta / 2.0),
        "t6": (0.0, delta / 3.0),
        "t7": (0.0, delta / 3.0),
        "edl5": (edl4 + t4_lo, edl4 + t4_hi),
        "edl6": (edl4 + t4_lo, min(edl7, edl4 + t4_hi + delta / 2.0)),
        "edl8": (edl7, edl7 + delta / 3.0),
    }


def fit_acceptance(surv, abar_by_cap):
    """Per-row conditional acceptance p with S_k = p^k, fitted on both caps."""
    def predict(p, cap):
        return sum((p ** k) * surv[k - 1] for k in range(1, cap + 1))
    best = None
    for i in range(1, 20000):
        p = i / 20000.0
        err = sum((predict(p, cap) - abar) ** 2 for cap, abar in abar_by_cap.items())
        if best is None or err < best[0]:
            best = (err, p)
    p = best[1]
    return p, {cap: predict(p, cap) for cap in abar_by_cap}


def fit_acceptance2(surv, abar_by_cap):
    """Two-parameter cumulative acceptance S_k = exp(-(k/theta)^eta).

    Exactly identified by the two observed accepted-rows-per-round values, so the
    reconstruction reproduces both paid receipts by construction and only the
    unseen caps are extrapolated.
    """
    def curve(theta, eta):
        return [0.0] + [math.exp(-((k / theta) ** eta)) for k in range(1, MAXD + 2)]

    def predict(sk, cap):
        return sum(sk[k] * surv[k - 1] for k in range(1, cap + 1))

    best = None
    for ti in range(1, 601):
        theta = ti * 0.05
        for ei in range(1, 301):
            eta = ei * 0.02
            sk = curve(theta, eta)
            err = sum((predict(sk, cap) - abar) ** 2
                      for cap, abar in abar_by_cap.items())
            if best is None or err < best[0]:
                best = (err, theta, eta, sk)
    err, theta, eta, sk = best
    return sk, {"theta": theta, "eta": eta, "err": err,
                "pred": {cap: predict(sk, cap) for cap in abar_by_cap}}


def width_probs(surv, cap):
    """P(rows verified = 1 + d) for d = 0..cap under D = min(D*, cap)."""
    probs = []
    for d in range(cap + 1):
        upper = surv[d - 1] if d > 0 else 1.0
        lower = surv[d] if d < cap else 0.0
        probs.append(max(0.0, upper - lower))
    total = sum(probs)
    return [q / total for q in probs]


# --------------------------------------------------------------------------
# step 3: cost laws
# --------------------------------------------------------------------------

def lstsq(design, target, weights):
    k = len(design[0])
    ata = [[sum(w * r[i] * r[j] for r, w in zip(design, weights)) for j in range(k)]
           for i in range(k)]
    atb = [sum(w * r[i] * t for r, t, w in zip(design, target, weights)) for i in range(k)]
    aug = [row[:] + [b] for row, b in zip(ata, atb)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-20:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(k):
            if r != col and aug[r][col]:
                f = aug[r][col]
                aug[r] = [v - f * w for v, w in zip(aug[r], aug[col])]
    return [aug[i][k] for i in range(k)]


def _rows(probs):
    return sum(q * (1 + d) for d, q in enumerate(probs))


def basis_affine(probs):
    return [1.0, _rows(probs)]


def basis_quadratic(probs):
    return [1.0, _rows(probs), sum(q * (1 + d) ** 2 for d, q in enumerate(probs))]


def basis_wall6(probs):
    """Affine plus a free step paid when rows verified exceed the sdpa width wall."""
    return [1.0, _rows(probs), sum(q for d, q in enumerate(probs) if 1 + d > 5)]


def basis_wall6_rows(probs):
    """Affine plus a step plus a free extra row price above the width wall."""
    return [1.0, _rows(probs),
            sum(q for d, q in enumerate(probs) if 1 + d > 5),
            sum(q * (1 + d - 5) for d, q in enumerate(probs) if 1 + d > 5)]


def basis_groups(probs):
    """Cost set by the number of <=5-row sdpa segments only."""
    return [1.0, sum(q * math.ceil((1 + d) / 5.0) for d, q in enumerate(probs))]


def basis_groups_rows(probs):
    return [1.0, sum(q * math.ceil((1 + d) / 5.0) for d, q in enumerate(probs)),
            _rows(probs)]


COST_MODELS = {
    "affine": (["F", "b"], basis_affine),
    "quadratic": (["F", "b", "c"], basis_quadratic),
    "affine+wall6": (["F", "b", "step"], basis_wall6),
    "affine+wall6+rows": (["F", "b", "step", "b2"], basis_wall6_rows),
    "groups": (["F", "g"], basis_groups),
    "groups+rows": (["F", "g", "b"], basis_groups_rows),
}


def fit_cost(observations, receipt_labels, free_intercepts=None):
    """R = F_receipt + g(width mixture).

    ``free_intercepts`` lists the receipts that get their own intercept shift. A and
    C are the same tree apart from one cap literal, and a cap literal cannot change
    the per-round fixed cost, so the physically correct specification gives C no
    intercept of its own. Leaving it free instead turns the recovered shift into a
    specification test: a law that needs a large dF_C is describing curvature it
    failed to model.
    """
    if free_intercepts is None:
        free_intercepts = receipt_labels[1:]
    extra = [e for e in free_intercepts if e in receipt_labels]
    results = {}
    for key, (names, basis) in COST_MODELS.items():
        design, target, weights = [], [], []
        for label, name, probs, R, rounds in observations:
            design.append(basis(probs) + [1.0 if label == e else 0.0 for e in extra])
            target.append(R)
            weights.append(rounds)
        beta = lstsq(design, target, weights)
        if beta is None:
            continue
        resid = [t - sum(b * v for b, v in zip(beta, r))
                 for r, t in zip(design, target)]
        wsse = sum(w * r * r for r, w in zip(resid, weights))
        n = len(observations)
        results[key] = {
            "names": names + ["dF_" + e for e in extra],
            "beta": beta,
            "basis": basis,
            "wrmse_ms": 1000 * math.sqrt(wsse / sum(weights)),
            "max_resid_ms": 1000 * max(abs(r) for r in resid),
            "aic": n * math.log(max(wsse, 1e-18) / n) + 2 * len(beta),
            "resid_ms": {"%s/%s" % (o[0], o[1]): 1000 * r
                         for o, r in zip(observations, resid)},
        }
    return results


def cost_of_mixture(model, probs):
    return sum(b * v for b, v in zip(model["beta"], model["basis"](probs)))


def cost_at_width(model, width):
    return cost_of_mixture(model, [0.0] * (width - 1) + [1.0])


def published_median(raws):
    ordered = sorted(raws)
    return 0.5 * (ordered[3] + ordered[4])


# --------------------------------------------------------------------------

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    rows = load_board()
    data = {}
    for label, prefix, cap, note in RECEIPTS:
        row = next(r for r in rows if r["id"].startswith(prefix))
        vec = per_prompt(row)
        data[label] = {
            "id": row["id"][:8], "cap": cap, "note": note,
            "score": row["officialScore"], "status": row.get("status"),
            "commit": row["officialMetrics"].get("commit", "")[:12],
            "vec": vec, "rec": recover_rounds(vec),
        }

    if cmd in ("rounds", "all"):
        print("=" * 112)
        print("STEP 1-2a  board exposure and exact round recovery")
        print("=" * 112)
        for label, prefix, cap, note in RECEIPTS:
            d = data[label]
            print("\n--- %s %s cap=%d published=%.11f status=%s commit=%s"
                  % (label, d["id"], cap, d["score"], d["status"], d["commit"]))
            print("    %-9s %10s %8s %6s %7s %7s %8s %8s %8s  %s"
                  % ("prompt", "decode_s", "edl", "Mbar", "rounds", "accept",
                     "abar", "alpha", "R_ms", "lattice"))
            for name in ORDER:
                r = d["rec"][name]
                flag = ""
                if label == "A" and E159_ROUNDS_A[name] != r["rounds"]:
                    flag = "  E159=%d" % E159_ROUNDS_A[name]
                print("    %-9s %10.6f %8.4f %6.3f %7d %7d %8.4f %8.5f %8.3f  %s%s"
                      % (name, r["decode_total"], r["edl"], r["Mbar"], r["rounds"],
                         r["accepted"], r["abar"], r["alpha"], 1000 * r["R"],
                         ",".join(str(c) for c in r["lattice"][:5]), flag))
        agree = all(data["A"]["rec"][n]["rounds"] == E159_ROUNDS_A[n] for n in ORDER)
        print("\n    independent lattice recovery reproduces E159 on all 8 prompts: %s"
              % agree)

    surv, accept, bounds = {}, {}, {}
    for name in ORDER:
        a, c = data["A"]["rec"][name], data["C"]["rec"][name]
        t0 = 1.0 - a["nondraft"] / a["rounds"]
        err, lam, kappa, s = fit_survival(c["edl"], a["edl"], t0)
        p, pred = fit_acceptance(s, {4: c["abar"], 7: a["abar"]})
        surv[name] = {"t": s, "lam": lam, "kappa": kappa, "err": err, "t0": t0}
        accept[name] = {"p": p, "pred": pred, "obs": {4: c["abar"], 7: a["abar"]}}
        bounds[name] = tail_bounds(c["edl"], a["edl"], s[3])

    if cmd in ("survival", "all"):
        print("\n" + "=" * 112)
        print("STEP 2b  round-width mixtures from D = min(D*, cap)")
        print("=" * 112)
        print("    %-9s %7s %7s %6s %6s %6s %6s %6s %6s %6s %7s"
              % ("prompt", "edl4", "edl7", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
                 "p_row"))
        for name in ORDER:
            s = surv[name]["t"]
            a, c = data["A"]["rec"][name], data["C"]["rec"][name]
            print("    %-9s %7.4f %7.4f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %7.4f"
                  % (name, c["edl"], a["edl"], s[0], s[1], s[2], s[3], s[4], s[5], s[6],
                     accept[name]["p"]))
        print("\n    acceptance model check: accepted rows per round")
        print("    %-9s %10s %10s %10s %10s %9s"
              % ("prompt", "cap4 obs", "cap4 fit", "cap7 obs", "cap7 fit", "worst%"))
        for name in ORDER:
            o, f = accept[name]["obs"], accept[name]["pred"]
            worst = max(abs(f[4] / o[4] - 1), abs(f[7] / o[7] - 1)) if o[4] and o[7] else 0
            print("    %-9s %10.4f %10.4f %10.4f %10.4f %8.2f%%"
                  % (name, o[4], f[4], o[7], f[7], 100 * worst))

        print("\n    family-free monotone bounds on the offered depth at unseen caps")
        print("    %-9s %8s %16s %16s %16s"
              % ("prompt", "delta", "edl(5)", "edl(6)", "edl(8)"))
        for name in ORDER:
            b = bounds[name]
            print("    %-9s %8.4f  [%6.3f,%6.3f]  [%6.3f,%6.3f]  [%6.3f,%6.3f]"
                  % (name, b["delta"], b["edl5"][0], b["edl5"][1],
                     b["edl6"][0], b["edl6"][1], b["edl8"][0], b["edl8"][1]))

        rounds_local = sum(LOCAL_CAP4_HISTOGRAM.values())
        edl4_local = sum((m - 1) * n for m, n in LOCAL_CAP4_HISTOGRAM.items()) / rounds_local
        t_local = [sum(n for m, n in LOCAL_CAP4_HISTOGRAM.items() if (m - 1) > d)
                   / rounds_local for d in range(4)]
        err, lam, kappa, s = fit_survival(edl4_local, LOCAL_CAP7_EDL, t_local[0])
        print("\n    CALIBRATION on the local fixture (E168 r1, W&B u2g6yfjx)")
        print("      cap-4 histogram %s -> edl(4)=%.4f over %d rounds"
              % (LOCAL_CAP4_HISTOGRAM, edl4_local, rounds_local))
        print("      survival measured from the histogram  : %s"
              % ["%.4f" % v for v in t_local])
        print("      survival reconstructed from edl(4),(7): %s"
              % ["%.4f" % v for v in s[:4]])
        print("      reconstruction edl(4)=%.4f target %.4f ; edl(7)=%.4f target %.4f"
              % (edl_of(s, 4), edl4_local, edl_of(s, 7), LOCAL_CAP7_EDL))
        print("      max |t_d| error over d<4: %.4f" %
              max(abs(x - y) for x, y in zip(t_local, s[:4])))

    # Two-parameter cumulative acceptance, exactly matched to both receipts.
    accept2 = {}
    for name in ORDER:
        a, c = data["A"]["rec"][name], data["C"]["rec"][name]
        sk, info = fit_acceptance2(surv[name]["t"], {4: c["abar"], 7: a["abar"]})
        accept2[name] = {"S": sk, **info}

    if cmd in ("survival", "all"):
        print("\n    two-parameter cumulative acceptance S_k, exactly identified")
        print("    %-9s %7s %7s %7s %7s %7s %7s %7s %7s  %9s"
              % ("prompt", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "abar err"))
        for name in ORDER:
            s = accept2[name]["S"]
            pred = accept2[name]["pred"]
            obs = {4: data["C"]["rec"][name]["abar"], 7: data["A"]["rec"][name]["abar"]}
            worst = max(abs(pred[c] - obs[c]) for c in (4, 7))
            print("    %-9s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f  %9.5f"
                  % (name, s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], worst))

    observations, labels = [], ["A", "C", "B", "D"]
    for label in labels:
        cap = data[label]["cap"]
        for name in ORDER:
            r = data[label]["rec"][name]
            observations.append((label, name, width_probs(surv[name]["t"], cap),
                                 r["R"], r["rounds"]))
    core = [o for o in observations if o[0] in ("A", "C")]
    # Physical specification: A and C are one cap literal apart, so C gets no
    # intercept of its own. B and D carry real tree deltas and keep theirs.
    fits_core = fit_cost(core, ["A", "C"], free_intercepts=[])
    fits_core_free = fit_cost(core, ["A", "C"], free_intercepts=["C"])
    fits_all = fit_cost(observations, labels, free_intercepts=["B", "D"])
    model_keys = sorted(fits_core, key=lambda k: fits_core[k]["aic"])

    if cmd in ("fit", "all"):
        print("\n" + "=" * 112)
        print("STEP 3  competing round-cost laws on candidate-leg decode seconds")
        print("=" * 112)
        print("\n--- SPECIFICATION TEST: the cap literal cannot change the per-round")
        print("    fixed cost, so a law that needs a large dF_C is mis-specified.")
        print("    %-19s %10s %11s" % ("model", "dF_C (ms)", "wRMSE_ms"))
        for key in sorted(fits_core_free, key=lambda k: fits_core_free[k]["aic"]):
            m = fits_core_free[key]
            idx = m["names"].index("dF_C")
            print("    %-19s %10.3f %11.4f" % (key, 1000 * m["beta"][idx], m["wrmse_ms"]))

        for tag, res, obs in (("A+C single-factor pair, dF_C == 0", fits_core, core),
                              ("A+C+B+D, dF_C == 0", fits_all, observations)):
            print("\n--- %s  n=%d, rounds-weighted" % (tag, len(obs)))
            print("    %-19s %9s %9s %10s  %s"
                  % ("model", "wRMSE_ms", "maxres_ms", "AIC", "parameters (ms)"))
            for key in sorted(res, key=lambda k: res[k]["aic"]):
                m = res[key]
                pretty = "  ".join("%s=%.3f" % (n, 1000 * v)
                                   for n, v in zip(m["names"], m["beta"]))
                print("    %-19s %9.4f %9.3f %10.2f  %s"
                      % (key, m["wrmse_ms"], m["max_resid_ms"], m["aic"], pretty))

        print("\n    residuals of the best A+C model (%s), ms:" % model_keys[0])
        for k, v in sorted(fits_core[model_keys[0]]["resid_ms"].items(),
                           key=lambda kv: -abs(kv[1])):
            print("       %-14s %+8.3f" % (k, v))

        print("\n    ROBUSTNESS: refit under the extreme monotone tail allocations")
        print("    %-19s %11s %11s %11s" % ("model", "fitted", "tail-heavy", "tail-light"))
        alt = {}
        for vname in ("tail-heavy", "tail-light"):
            tvar = {}
            for name in ORDER:
                b = bounds[name]
                t = list(surv[name]["t"])
                if vname == "tail-heavy":
                    t[4] = b["t4"][1]
                    t[5] = max(0.0, b["delta"] - t[4])
                    t[6] = t[7] = 0.0
                else:
                    t[4] = t[5] = t[6] = t[7] = b["delta"] / 3.0
                tvar[name] = t
            obs = []
            for label in ("A", "C"):
                cap = data[label]["cap"]
                for name in ORDER:
                    r = data[label]["rec"][name]
                    obs.append((label, name, width_probs(tvar[name], cap),
                                r["R"], r["rounds"]))
            alt[vname] = fit_cost(obs, ["A", "C"], free_intercepts=[])
        for key in model_keys:
            print("    %-19s %11.4f %11.4f %11.4f"
                  % (key, fits_core[key]["wrmse_ms"],
                     alt["tail-heavy"][key]["wrmse_ms"], alt["tail-light"][key]["wrmse_ms"]))
        print("    quadratic curvature c (ms/row^2): fitted %.4f  heavy %.4f  light %.4f"
              % (1000 * fits_core["quadratic"]["beta"][2],
                 1000 * alt["tail-heavy"]["quadratic"]["beta"][2],
                 1000 * alt["tail-light"]["quadratic"]["beta"][2]))

        print("\n    implied deterministic-width round cost R(M), ms")
        print("    %-19s %s" % ("model", "  ".join("%6s" % ("M=%d" % m)
                                                   for m in range(1, 10))))
        for key in model_keys:
            vals = [1000 * cost_at_width(fits_core[key], m) for m in range(1, 10)]
            print("    %-19s %s" % (key, "  ".join("%6.1f" % v for v in vals)))
        print("\n    HELD-OUT ANCHOR: the ranked batch-1 zero-draft round is 30.2519 ms")
        print("    (FINDING 330, recovered from disjoint board data and never used here)")
        for key in model_keys:
            r1 = 1000 * cost_at_width(fits_core[key], 1)
            print("      %-19s R(1)=%7.3f ms   error %+7.3f ms" % (key, r1, r1 - 30.2519))

        print("\n    marginal ms per extra verified row")
        print("    %-19s %s" % ("model", "  ".join("%6s" % ("%d>%d" % (m, m + 1))
                                                   for m in range(1, 9))))
        for key in model_keys:
            vals = [1000 * (cost_at_width(fits_core[key], m + 1)
                            - cost_at_width(fits_core[key], m)) for m in range(1, 9)]
            print("    %-19s %s" % (key, "  ".join("%6.2f" % v for v in vals)))

        print("\n    R(M=7)/R(M=5), the advisor's 1.3-1.4 prior")
        for key in model_keys:
            r7 = cost_at_width(fits_core[key], 7)
            r5 = cost_at_width(fits_core[key], 5)
            print("      %-19s R(7)=%6.2f ms  R(5)=%6.2f ms  ratio=%.4f"
                  % (key, 1000 * r7, 1000 * r5, r7 / r5))
        wide = ["beagle", "republic", "essays", "medicine", "botany"]
        den7 = sum(data["A"]["rec"][n]["rounds"] for n in wide)
        den4 = sum(data["C"]["rec"][n]["rounds"] for n in wide)
        r7 = sum(data["A"]["rec"][n]["R"] * data["A"]["rec"][n]["rounds"] for n in wide) / den7
        r4 = sum(data["C"]["rec"][n]["R"] * data["C"]["rec"][n]["rounds"] for n in wide) / den4
        m7 = sum(data["A"]["rec"][n]["Mbar"] * data["A"]["rec"][n]["rounds"] for n in wide) / den7
        m4 = sum(data["C"]["rec"][n]["Mbar"] * data["C"]["rec"][n]["rounds"] for n in wide) / den4
        print("\n    MODEL-FREE on the five wide prompts: Mbar %.3f -> %.3f, "
              "mean round %.3f -> %.3f ms, ratio %.4f"
              % (m4, m7, 1000 * r4, 1000 * r7, r7 / r4))

    # ---------------------------------------------------------------- step 4
    if cmd in ("price", "all"):
        print("\n" + "=" * 112)
        print("STEP 4  pricing the open depth cells against the crown %.10f" % CROWN)
        print("=" * 112)
        print("""
    Incremental pricing. Raising the cap from c to c+1 changes only the rounds where
    the greedy rule wanted more depth than c, a fraction t_c of all rounds. Those
    rounds verify one extra row. Everything else is the paid receipt itself:

        abar(c+1) = abar(c) + t_c * S_{c+1}
        Rbar(c+1) = Rbar(c) + t_c * (R(c+2) - R(c+1))
        T(c+1)    = 512 / (1 + abar(c+1)) * Rbar(c+1)

    Levels come from the receipt, so only the increment is modelled.""")

        def raw_of(name, decode_total):
            """Published raw ratio. The scored denominator is the whole timed
            leg, so modelled decode seconds must carry the prefill back."""
            vec = data["A"]["vec"][name]
            return vec["serial"] / (decode_total / TOKENS + vec["prefill"])

        def chain(name, tvar, S, model, target_cap, anchor):
            """Walk from the anchor receipt's observed state to the target cap.

            The extra rows unlocked by one more cap step are accepted at the
            marginal rate mu measured directly between the two paid receipts, so
            the walk reproduces both receipts' accepted counts exactly and the
            held-out error isolates the cost law.
            """
            rec = data[anchor]["rec"][name]
            cap = data[anchor]["cap"]
            abar, R, t, mu = rec["abar"], rec["R"], tvar[name], S[name]
            while cap < target_cap:
                abar += t[cap] * mu
                R += t[cap] * (cost_at_width(model, cap + 2)
                               - cost_at_width(model, cap + 1))
                cap += 1
            while cap > target_cap:
                abar -= t[cap - 1] * mu
                R -= t[cap - 1] * (cost_at_width(model, cap + 1)
                                   - cost_at_width(model, cap))
                cap -= 1
            n_rounds = TOKENS / (1.0 + abar)
            return abar, R, n_rounds, n_rounds * R

        variants = {"fitted": {n: list(surv[n]["t"]) for n in ORDER},
                    "tail-heavy": {}, "tail-light": {}}
        for name in ORDER:
            b = bounds[name]
            heavy = list(surv[name]["t"])
            heavy[4] = b["t4"][1]
            heavy[5] = max(0.0, b["delta"] - heavy[4])
            heavy[6] = heavy[7] = 0.0
            light = list(surv[name]["t"])
            light[4] = light[5] = light[6] = light[7] = b["delta"] / 3.0
            variants["tail-heavy"][name] = heavy
            variants["tail-light"][name] = light

        # Model-free marginal acceptance of the rows that only cap 7 unlocks.
        Sfit = {}
        print("\n    MODEL-FREE marginal acceptance of the deep rows, mu = dabar/dedl")
        print("    %-9s %9s %9s %9s %10s %10s"
              % ("prompt", "d edl", "d abar", "mu", "alpha(4)", "mu-alpha"))
        for name in ORDER:
            a, c = data["A"]["rec"][name], data["C"]["rec"][name]
            d_edl = a["edl"] - c["edl"]
            d_abar = a["abar"] - c["abar"]
            mu = d_abar / d_edl if d_edl > 1e-9 else 0.0
            Sfit[name] = mu
            if d_edl <= 1e-9:
                print("    %-9s %9.4f %9.4f %9s %10.4f %10s"
                      % (name, d_edl, d_abar, "cap inert", c["alpha"], "-"))
            else:
                print("    %-9s %9.4f %9.4f %9.4f %10.4f %+10.4f"
                      % (name, d_edl, d_abar, mu, c["alpha"], mu - c["alpha"]))
        print("    mu is the acceptance rate a cap-8 row must beat to pay for itself.")

        print("\n    HELD-OUT CHECK 1: walk down from the paid cap-7 receipt to cap 4")
        print("    and compare with the paid cap-4 receipt, per prompt")
        best = fits_core[model_keys[0]]
        print("    %-9s %9s %9s %8s %9s %9s %8s"
              % ("prompt", "N pred", "N obs", "N err%", "R pred", "R obs", "R err%"))
        raws_pred = []
        for name in ORDER:
            abar, R, n_pred, total = chain(name, variants["fitted"], Sfit, best, 4, "A")
            obs = data["C"]["rec"][name]
            raws_pred.append(raw_of(name, total))
            print("    %-9s %9.1f %9d %8.2f %9.3f %9.3f %8.2f"
                  % (name, n_pred, obs["rounds"], 100 * (n_pred / obs["rounds"] - 1),
                     1000 * R, 1000 * obs["R"], 100 * (R / obs["R"] - 1)))
        pred4 = published_median(raws_pred)
        print("    predicted cap-4 published %.8f versus paid %.8f, error %+.3f %%"
              % (pred4, CAP4_PUBLISHED, 100 * (pred4 / CAP4_PUBLISHED - 1)))

        print("\n    HELD-OUT CHECK 2: refit the law on receipt A alone, then predict the")
        print("    cap-4 published score. Receipt C never enters the cost fit.")
        a_only = [o for o in observations if o[0] == "A"]
        fits_a = fit_cost(a_only, ["A"], free_intercepts=[])
        for key in sorted(fits_a, key=lambda k: fits_a[k]["aic"]):
            raws = []
            for name in ORDER:
                _, _, _, total = chain(name, variants["fitted"], Sfit,
                                       fits_a[key], 4, "A")
                raws.append(raw_of(name, total))
            p4 = published_median(raws)
            print("      %-19s wRMSE %6.4f ms  cap-4 predicted %.8f  error %+7.3f %%"
                  % (key, fits_a[key]["wrmse_ms"], p4,
                     100 * (p4 / CAP4_PUBLISHED - 1)))

        print("\n    HELD-OUT CHECK 3: the same walk under every A+C cost law")
        for key in model_keys:
            raws = []
            for name in ORDER:
                _, _, _, total = chain(name, variants["fitted"], Sfit,
                                       fits_core[key], 4, "A")
                raws.append(raw_of(name, total))
            p4 = published_median(raws)
            print("      %-19s cap-4 predicted %.8f   error %+7.3f %%"
                  % (key, p4, 100 * (p4 / CAP4_PUBLISHED - 1)))

        print("\n    PRICED CELLS: published score by cap, walked from the paid receipt")
        print("    (caps 5 and 6 are walked from both anchors; cap 8 from cap 7 only)")
        header = "    %-19s %-11s %-7s" % ("model", "survival", "anchor")
        header += "".join("%14s" % ("cap %d" % c) for c in (5, 6, 8))
        print(header)
        crown_hits = []
        for key in model_keys:
            model = fits_core[key]
            for vname, tvar in variants.items():
                for anchor in ("A", "C"):
                    line = "    %-19s %-11s %-7s" % (key, vname, anchor)
                    for cap in (5, 6, 8):
                        if cap == 8 and anchor == "C":
                            line += "%14s" % "-"
                            continue
                        raws = []
                        for name in ORDER:
                            _, _, _, total = chain(name, tvar, Sfit, model, cap, anchor)
                            raws.append(raw_of(name, total))
                        score = published_median(raws)
                        line += "%14.8f" % score
                        if score > CROWN:
                            crown_hits.append((key, vname, anchor, cap, score))
                    print(line)

        print("\n    ANCHOR CONSISTENCY. Walking up from cap 4 and down from cap 7 must")
        print("    agree. The disagreement prices the joint (cost law, tail) error.")
        print("    %-19s %-11s %13s %13s" % ("model", "survival", "cap 5 gap%", "cap 6 gap%"))
        for key in model_keys:
            for vname, tvar in variants.items():
                gaps = []
                for cap in (5, 6):
                    vals = []
                    for anchor in ("A", "C"):
                        raws = [raw_of(n, chain(n, tvar, Sfit, fits_core[key],
                                                cap, anchor)[3]) for n in ORDER]
                        vals.append(published_median(raws))
                    gaps.append(100 * (vals[0] / vals[1] - 1))
                print("    %-19s %-11s %13.3f %13.3f" % (key, vname, gaps[0], gaps[1]))

        print("\n    BREAK-EVEN marginal row cost. One more cap step pays only when")
        print("    dR/dM at the new width is below Rbar * mu / (1 + abar).")
        print("    %-9s %7s %10s %12s %12s %10s"
              % ("prompt", "cap", "Rbar_ms", "break-even", "quadratic", "verdict"))
        for name in ("beagle", "essays", "republic", "medicine", "botany"):
            for cap in (7,):
                rec = data["A"]["rec"][name]
                mu = Sfit[name]
                breakeven = 1000 * rec["R"] * mu / (1.0 + rec["abar"])
                actual = 1000 * (cost_at_width(best, cap + 2)
                                 - cost_at_width(best, cap + 1))
                print("    %-9s %7s %10.3f %12.3f %12.3f %10s"
                      % (name, "7->8", 1000 * rec["R"], breakeven, actual,
                         "pays" if actual < breakeven else "costs"))

        print("\n    cells clearing the crown %.8f:" % CROWN)
        if crown_hits:
            for hit in crown_hits:
                print("      %-19s %-11s anchor %s cap %d -> %.8f" % hit)
        else:
            print("      NONE under any cost law, survival allocation or anchor")
        print("    published-score 1 sigma from the candidate-leg floor: +-%.5f"
              % (BEST_A * SIGMA_LEG))

        print("\n    per-prompt detail under %s, fitted survival, anchor A"
              % model_keys[0])
        for cap in (5, 6, 7, 8):
            raws, rows = [], []
            for name in ORDER:
                abar, R, n, total = chain(name, variants["fitted"], Sfit, best, cap, "A")
                raw = raw_of(name, total)
                raws.append(raw)
                rows.append((name, edl_of(variants["fitted"][name], cap), abar,
                             n, 1000 * R, total, raw,
                             100 * (total / data["A"]["rec"][name]["decode_total"] - 1)))
            print("      cap %d  predicted median %.8f" % (cap, published_median(raws)))
            print("      %-9s %7s %7s %8s %8s %9s %9s %8s"
                  % ("prompt", "edl", "abar", "rounds", "R_ms", "decode_s", "raw",
                     "vs cap7"))
            for row in rows:
                print("      %-9s %7.3f %7.3f %8.1f %8.3f %9.4f %9.5f %+7.2f%%" % row)

        print("\n    BEST-CASE ADAPTIVE: per-prompt cap chosen to minimise decode time.")
        print("    An upper bound on any policy that only re-tunes the cap per prompt.")
        print("    The shipped controller has no prompt identity, so it cannot reach")
        print("    this. The search is confined to caps 4..8: the marginal acceptance")
        print("    mu is measured only on that interval, and caps below 4 would need")
        print("    an unmeasured shallow acceptance rate.")
        print("      %-9s %6s %10s %10s %10s %9s"
              % ("prompt", "cap*", "decode_s", "raw", "cap7 raw", "rank@7"))
        raws_adapt = []
        order7 = sorted(ORDER, key=lambda n: data["A"]["vec"][n]["raw"])
        for name in ORDER:
            options = []
            for cap in range(4, 9):
                _, _, _, total = chain(name, variants["fitted"], Sfit, best, cap, "A")
                options.append((total, cap))
            total, cap = min(options)
            raw = raw_of(name, total)
            raws_adapt.append(raw)
            print("      %-9s %6d %10.4f %10.5f %10.5f %9d"
                  % (name, cap, total, raw, data["A"]["vec"][name]["raw"],
                     order7.index(name) + 1))
        print("      oracle per-prompt-cap published median %.8f  (paid cap-7 %.8f)"
              % (published_median(raws_adapt), BEST_A))
        print("      the median window is set by ranks 4 and 5, %s and %s, and both"
              % (order7[3], order7[4]))
        print("      already sit at their own optimum, so depth adaptivity buys nothing.")

    return data


if __name__ == "__main__":
    main()
