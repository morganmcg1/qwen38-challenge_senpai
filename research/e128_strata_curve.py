"""E128 F7 item 3: fit the ranked round-cost curve inside each QMV dispatch
stratum and test Model P against Model R.

Model P (advisor)  T(M) = a_i + b*M + f*G_s(M).  The break MOVES with the
                   stratum's own pass-count vector G_s.
Model R (student)  T(M) = a_i + b_lo*M + d*max(0, M - M*).  One hinge at the
                   same width in every stratum, no pass term.

Two data problems have to be solved first.

1. Round count. The board publishes `effective_mean_draft_len` = D/R, which
   pins R only up to an integer multiple of the minimal denominator. Curve
   consistency resolves it: on the reference-schedule rows the wrong drama
   multiple raises the pooled hinge RSS by 200x, so the multiple is very
   strongly identified relative to the P-versus-R difference the fit is asked
   to resolve. Multiples are chosen here by coordinate descent on a
   free-break hinge fit, which is neutral between P and R because both are
   hinge-shaped inside one stratum.

2. Coverage. Only 202 of the 456 local trees carry an extractable
   `qmv_fast_crossrow_affine4_g64_m<T, M, IPG>` table, and none of the 164
   reference-schedule rows has a local tree at all.
"""

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

from e128_rounds import load_rows, per_prompt

TOKENS = 512
MAXM = 9
ALPHA_LO, ALPHA_HI = 0.15, 1.0
GRID = np.arange(1.5, 8.01, 0.125)


def legal_rounds(dl, n0):
    if dl == 0:
        return [n0 if n0 else TOKENS]
    base = Fraction(repr(dl)).limit_denominator(TOKENS).denominator
    lo = max(1, n0, int(TOKENS / (1.0 + dl)))
    out = []
    for k in range(1, TOKENS // base + 1):
        r = k * base
        if not (lo <= r <= TOKENS):
            continue
        alpha = (TOKENS / r - 1) / dl
        if ALPHA_LO <= alpha <= ALPHA_HI + 1e-12:
            out.append(r)
    return out or [max(1, int(math.ceil(TOKENS / (1.0 + dl))))]


def hinge_rss(xs, ys):
    """Best RSS over the break grid for a single-intercept hinge fit."""
    x = np.asarray(xs)
    y = np.asarray(ys)
    best = None
    for m in GRID:
        a = np.column_stack([np.ones(len(x)), x, np.maximum(0.0, x - m)])
        beta, *_ = np.linalg.lstsq(a, y, rcond=None)
        r = y - a @ beta
        rss = float(r @ r)
        if best is None or rss < best[0]:
            best = (rss, float(m))
    return best


def choose_rounds(entries):
    """Coordinate descent over the legal round multiples of each prompt."""
    names = list(entries)
    cands = {n: legal_rounds(entries[n]["dl"], entries[n]["n0"]) for n in names}
    pick = {n: cands[n][0] for n in names}
    xs = [entries[n]["mbar"] for n in names]

    def cost(p):
        ys = [TOKENS * entries[n]["spt"] / p[n] * 1e6 for n in names]
        return hinge_rss(xs, ys)[0]

    cur = cost(pick)
    for _ in range(6):
        moved = False
        for n in names:
            if len(cands[n]) == 1:
                continue
            for r in cands[n]:
                if r == pick[n]:
                    continue
                trial = dict(pick)
                trial[n] = r
                c = cost(trial)
                if c < cur * (1 - 1e-9):
                    pick, cur, moved = trial, c, True
        if not moved:
            break
    return pick, cur, {n: len(cands[n]) for n in names}


def gvec(tbl):
    return {m: math.ceil(m / max(tbl.get(m, m), 1)) for m in range(1, MAXM + 1)}


def gbar(g, mbar):
    lo = max(1, min(MAXM, int(math.floor(mbar))))
    hi = max(1, min(MAXM, lo + 1))
    frac = min(max(mbar - lo, 0.0), 1.0)
    return (1 - frac) * g[lo] + frac * g[hi]


def build_panel():
    tables = {
        sid: {int(m): v for m, v in t.items()}
        for sid, t in json.load(open("/tmp/e128_strata.json"))["tables"].items()
    }
    rows = {r["id"]: r for r in load_rows() if r.get("id")}
    panel = []
    skipped = Counter()
    for sid, tbl in sorted(tables.items()):
        row = rows.get(sid)
        if row is None:
            skipped["no-board-row"] += 1
            continue
        e = per_prompt(row)
        if len(e) != 8:
            skipped["incomplete"] += 1
            continue
        entries = {}
        bad = False
        for name, x in e.items():
            dl = x.get("effective_mean_draft_len")
            spt = x.get("mtp_seconds_per_token_mean")
            if dl is None or spt is None or dl + 1.0 > MAXM:
                bad = True
                break
            entries[name] = {
                "dl": dl,
                "n0": x.get("non_drafting_round_count") or 0,
                "spt": spt,
                "mbar": dl + 1.0,
            }
        if bad:
            skipped["bad-record"] += 1
            continue
        pick, rss, nc = choose_rounds(entries)
        g = gvec(tbl)
        key = ",".join(str(g[m]) for m in range(1, MAXM + 1))
        for name, x in entries.items():
            panel.append(
                {
                    "sid": sid,
                    "prompt": name,
                    "mbar": x["mbar"],
                    "round_us": TOKENS * x["spt"] / pick[name] * 1e6,
                    "R": pick[name],
                    "R_choices": nc[name],
                    "gbar": gbar(g, x["mbar"]),
                    "gkey": key,
                    "score": row.get("officialScore"),
                    "created": row.get("createdAt"),
                }
            )
        skipped["kept"] += 1
    return panel, skipped


def fe_design(panel, cols):
    sids = sorted({p["sid"] for p in panel})
    idx = {s: i for i, s in enumerate(sids)}
    a = np.zeros((len(panel), len(sids) + len(cols)))
    for r, p in enumerate(panel):
        a[r, idx[p["sid"]]] = 1.0
        for c, fn in enumerate(cols):
            a[r, len(sids) + c] = fn(p)
    return a, np.array([p["round_us"] for p in panel]), len(sids)


def fit(a, y, extra_params=0):
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    rss = float(resid @ resid)
    n = len(y)
    k = int(np.linalg.matrix_rank(a)) + extra_params + 1
    sigma2 = rss / max(n - k, 1)
    cov = sigma2 * np.linalg.pinv(a.T @ a)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    aic = n * math.log(rss / n) + 2 * k
    return {
        "beta": beta, "se": se, "rss": rss, "rmse": math.sqrt(rss / n), "n": n,
        "k": k, "aic": aic,
        "aicc": aic + 2 * k * (k + 1) / max(n - k - 1, 1),
        "bic": n * math.log(rss / n) + math.log(n) * k,
    }


def main():
    panel, skipped = build_panel()
    print("panel:", dict(skipped))
    print(f"points {len(panel)}   submissions {len({p['sid'] for p in panel})}")
    nmulti = sum(1 for p in panel if p["R_choices"] > 1)
    print(f"prompt-rows with more than one legal round count: {nmulti}/{len(panel)}")

    strata = defaultdict(list)
    for p in panel:
        strata[p["gkey"]].append(p)

    print("\n=== stratum table ===")
    print(
        f"{'G(1..9)':22s}{'subs':>6s}{'pts':>6s}{'mbar p10':>10s}{'mbar p90':>10s}"
        f"{'score p50':>11s}{'first':>12s}"
    )
    for key, pts in sorted(strata.items(), key=lambda kv: -len({p['sid'] for p in kv[1]})):
        subs = {p["sid"] for p in pts}
        mb = sorted(p["mbar"] for p in pts)
        sc = sorted(p["score"] for p in pts if p["score"] is not None)
        cr = sorted(p["created"] for p in pts if p["created"])
        print(
            f"{key:22s}{len(subs):6d}{len(pts):6d}{mb[int(0.1*(len(mb)-1))]:10.3f}"
            f"{mb[int(0.9*(len(mb)-1))]:10.3f}"
            f"{(statistics.median(sc) if sc else float('nan')):11.4f}"
            f"{(cr[0][:10] if cr else '-'):>12s}"
        )

    print("\n=== per-stratum free-break hinge fit (submission intercepts) ===")
    print(
        f"{'G(1..9)':22s}{'subs':>6s}{'M*':>7s}{'b_lo':>10s}{'b_hi':>10s}"
        f"{'rmse':>10s}{'G steps at':>14s}"
    )
    per_stratum = {}
    for key, pts in sorted(strata.items(), key=lambda kv: -len({p['sid'] for p in kv[1]})):
        subs = {p["sid"] for p in pts}
        if len(subs) < 5:
            continue
        gv = [int(c) for c in key.split(",")]
        steps = [m + 1 for m in range(1, MAXM) if gv[m] > gv[m - 1]]
        best = None
        for m in GRID:
            a, y, ns = fe_design(pts, [lambda p: p["mbar"], lambda p, m=m: max(0.0, p["mbar"] - m)])
            f = fit(a, y, extra_params=1)
            if best is None or f["rss"] < best[1]["rss"]:
                best = (float(m), f, ns)
        m, f, ns = best
        b_lo = f["beta"][ns]
        print(
            f"{key:22s}{len(subs):6d}{m:7.3f}{b_lo:10.1f}{b_lo+f['beta'][ns+1]:10.1f}"
            f"{f['rmse']:10.1f}{str(steps):>14s}"
        )
        per_stratum[key] = {
            "subs": len(subs), "mstar": m, "b_lo": float(b_lo),
            "b_hi": float(b_lo + f["beta"][ns + 1]), "rmse": f["rmse"],
            "g_steps": steps,
        }

    print("\n=== joint fits, per-submission intercepts, b constrained equal ===")
    a, y, ns = fe_design(panel, [lambda p: p["mbar"], lambda p: p["gbar"]])
    fp = fit(a, y)
    b, f_ = fp["beta"][ns], fp["beta"][ns + 1]
    sb, sf = fp["se"][ns], fp["se"][ns + 1]
    print(
        f"  Model P  b = {b:9.1f} +- {sb:6.1f}   f = {f_:9.1f} +- {sf:6.1f}"
        f"   95 % CI [{f_-1.96*sf:8.1f}, {f_+1.96*sf:8.1f}]"
    )
    print(f"           rmse {fp['rmse']:9.1f}   aicc {fp['aicc']:10.1f}   bic {fp['bic']:10.1f}")

    best = None
    for m in GRID:
        a, y, ns = fe_design(panel, [lambda p: p["mbar"], lambda p, m=m: max(0.0, p["mbar"] - m)])
        fr = fit(a, y, extra_params=1)
        if best is None or fr["rss"] < best[1]["rss"]:
            best = (float(m), fr, ns)
    mstar, fr, ns = best
    print(
        f"  Model R  M* = {mstar:5.3f}   b_lo = {fr['beta'][ns]:9.1f}"
        f"   b_hi = {fr['beta'][ns]+fr['beta'][ns+1]:9.1f}"
    )
    print(f"           rmse {fr['rmse']:9.1f}   aicc {fr['aicc']:10.1f}   bic {fr['bic']:10.1f}")

    a, y, ns = fe_design(
        panel,
        [lambda p: p["mbar"], lambda p: p["gbar"],
         lambda p: max(0.0, p["mbar"] - mstar)],
    )
    fb = fit(a, y, extra_params=1)
    print(
        f"  P + R    f = {fb['beta'][ns+1]:9.1f} +- {fb['se'][ns+1]:6.1f}"
        f"   step = {fb['beta'][ns+2]:9.1f} +- {fb['se'][ns+2]:6.1f}"
        f"   rmse {fb['rmse']:9.1f}"
    )
    a, y, ns = fe_design(panel, [lambda p: p["mbar"]])
    f0 = fit(a, y)
    print(f"  linear   b = {f0['beta'][ns]:9.1f}   rmse {f0['rmse']:9.1f}   aicc {f0['aicc']:10.1f}")
    print(
        f"\n  AICc prefers {'Model R' if fr['aicc'] < fp['aicc'] else 'Model P'}"
        f"   (dAICc = {abs(fr['aicc']-fp['aicc']):.1f})"
    )
    print(
        f"  BIC  prefers {'Model R' if fr['bic'] < fp['bic'] else 'Model P'}"
        f"   (dBIC  = {abs(fr['bic']-fp['bic']):.1f})"
    )

    out = {
        "per_stratum": per_stratum,
        "joint": {
            "P": {"b": float(b), "b_se": float(sb), "f": float(f_), "f_se": float(sf),
                  "rmse": fp["rmse"], "aicc": fp["aicc"], "bic": fp["bic"]},
            "R": {"mstar": mstar, "b_lo": float(fr["beta"][ns - 0]),
                  "rmse": fr["rmse"], "aicc": fr["aicc"], "bic": fr["bic"]},
            "PR": {"f": float(fb["beta"][ns + 1]), "f_se": float(fb["se"][ns + 1]),
                   "step": float(fb["beta"][ns + 2]), "step_se": float(fb["se"][ns + 2]),
                   "rmse": fb["rmse"]},
            "linear": {"b": float(f0["beta"][ns]), "rmse": f0["rmse"], "aicc": f0["aicc"]},
        },
        "n_points": len(panel),
        "n_subs": len({p["sid"] for p in panel}),
        "n_ambiguous_R": nmulti,
    }
    json.dump(out, open("/tmp/e128_strata_curve.json", "w"))
    print("\nwrote /tmp/e128_strata_curve.json")


if __name__ == "__main__":
    main()
