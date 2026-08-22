"""E128 F7 item 3: fit the ranked round-cost curve inside each QMV dispatch
stratum and test Model P against Model R.

Model P (advisor)  T(M) = a_i + b*M + f*G_s(M)
    a_i is a per-submission intercept, b a shared per-row slope, f the ranked
    price of one extra QMV pass, and G_s the stratum's own pass-count vector.
    The break in the observed curve MOVES with the stratum table.

Model R (student)  T(M) = a_i + b_lo*M + (b_hi - b_lo)*max(0, M - M*)
    one shared hinge at the same width M* in every stratum, no pass term.

Both carry the same per-submission intercepts, so b, f, b_lo, b_hi and M* are
identified from WITHIN-submission variation across the eight prompts plus
BETWEEN-stratum variation in G.
"""

import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

from e128_rounds import load_rows, per_prompt, recover_rounds

TOKENS = 512
MAXM = 9
REF_BEAGLE = 4.381818181818182
# Validated in e128_rounds_check.py: rational reconstruction reproduces this
# vector on 7 of 8 prompts; drama's minimal denominator is 84 and rankedcurve
# picks 3x84 = 252 rather than the minimal legal 2x84 = 168.
REF_ROUNDS = {
    "plutarch": 487,
    "drama": 252,
    "travel": 212,
    "beagle": 110,
    "republic": 93,
    "essays": 92,
    "medicine": 90,
    "botany": 81,
}


def load_strata():
    d = json.load(open("/tmp/e128_strata.json"))
    tables = {sid: {int(m): v for m, v in t.items()} for sid, t in d["tables"].items()}
    return tables


def gvec(tbl):
    return {m: math.ceil(m / max(tbl.get(m, m), 1)) for m in range(1, MAXM + 1)}


def gbar(g, mbar):
    """Expected pass count at a fractional mean width, linear between the two
    bracketing integer widths."""
    lo = max(1, min(MAXM, int(math.floor(mbar))))
    hi = max(1, min(MAXM, lo + 1))
    frac = min(max(mbar - lo, 0.0), 1.0)
    return (1 - frac) * g[lo] + frac * g[hi]


def build_panel(tables, use_ref_only):
    """One record per (submission, prompt)."""
    rows = load_rows()
    by_id = {r["id"]: r for r in rows if r.get("id")}
    panel = []
    kept = Counter()
    for sid, tbl in tables.items():
        row = by_id.get(sid)
        if row is None:
            kept["no-board-row"] += 1
            continue
        e = per_prompt(row)
        if len(e) != 8:
            kept["incomplete-per-prompt"] += 1
            continue
        is_ref = abs(e["beagle"]["effective_mean_draft_len"] - REF_BEAGLE) <= 1e-9
        if use_ref_only and not is_ref:
            kept["not-reference-schedule"] += 1
            continue
        g = gvec(tbl)
        recs = []
        bad = False
        for name, entry in e.items():
            dl = entry["effective_mean_draft_len"]
            n0 = entry["non_drafting_round_count"]
            spt = entry["mtp_seconds_per_token_mean"]
            if spt is None or dl is None:
                bad = True
                break
            if is_ref:
                R, mult = REF_ROUNDS[name], 1
            else:
                R, mult = recover_rounds(dl, n0)
            if not R:
                bad = True
                break
            mbar = dl + 1.0
            if mbar > MAXM:
                bad = True
                break
            recs.append(
                {
                    "sid": sid,
                    "prompt": name,
                    "mbar": mbar,
                    "round_us": TOKENS * spt / R * 1e6,
                    "R": R,
                    "R_mult": mult,
                    "gbar": gbar(g, mbar),
                    "gkey": ",".join(str(g[m]) for m in range(1, MAXM + 1)),
                    "score": row.get("officialScore"),
                    "created": row.get("createdAt"),
                    "is_ref": is_ref,
                }
            )
        if bad:
            kept["bad-prompt-record"] += 1
            continue
        kept["kept"] += 1
        panel.extend(recs)
    return panel, kept


def fe_design(panel, extra_cols):
    """Per-submission intercept dummies plus the named regressors."""
    sids = sorted({p["sid"] for p in panel})
    idx = {s: i for i, s in enumerate(sids)}
    n, k = len(panel), len(sids) + len(extra_cols)
    a = np.zeros((n, k))
    for r, p in enumerate(panel):
        a[r, idx[p["sid"]]] = 1.0
        for c, fn in enumerate(extra_cols):
            a[r, len(sids) + c] = fn(p)
    y = np.array([p["round_us"] for p in panel])
    return a, y, len(sids)


def fit(a, y):
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    rss = float(resid @ resid)
    n, k = a.shape
    rank = int(np.linalg.matrix_rank(a))
    sigma2 = rss / max(n - rank, 1)
    try:
        cov = sigma2 * np.linalg.pinv(a.T @ a)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    aic = n * math.log(rss / n) + 2 * (rank + 1)
    aicc = aic + 2 * (rank + 1) * (rank + 2) / max(n - rank - 2, 1)
    bic = n * math.log(rss / n) + math.log(n) * (rank + 1)
    return {
        "beta": beta,
        "se": se,
        "rss": rss,
        "rmse": math.sqrt(rss / n),
        "n": n,
        "rank": rank,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
    }


def hinge(mstar):
    return lambda p: max(0.0, p["mbar"] - mstar)


def report_panel(panel, label):
    print(f"\n=== {label} ===")
    strata = defaultdict(list)
    for p in panel:
        strata[p["gkey"]].append(p)
    print(f"points {len(panel)}   submissions {len({p['sid'] for p in panel})}")
    print(
        f"{'G(1..9)':22s}{'subs':>6s}{'pts':>6s}{'mbar min':>10s}"
        f"{'mbar max':>10s}{'score p50':>11s}{'first seen':>13s}"
    )
    for key, pts in sorted(strata.items(), key=lambda kv: -len({p['sid'] for p in kv[1]})):
        subs = {p["sid"] for p in pts}
        mb = [p["mbar"] for p in pts]
        sc = sorted(p["score"] for p in pts if p["score"] is not None)
        created = sorted(p["created"] for p in pts if p["created"])
        print(
            f"{key:22s}{len(subs):6d}{len(pts):6d}{min(mb):10.3f}{max(mb):10.3f}"
            f"{(sc[len(sc)//2] if sc else float('nan')):11.4f}"
            f"{(created[0][:10] if created else '-'):>13s}"
        )
    return strata


def per_stratum_fits(strata, min_subs=3):
    print("\nper-stratum Model R hinge scan (break M* that minimises RSS)")
    print(f"{'G(1..9)':22s}{'subs':>6s}{'M*':>6s}{'b_lo':>10s}{'b_hi':>10s}{'rmse':>10s}")
    out = {}
    for key, pts in sorted(strata.items(), key=lambda kv: -len({p['sid'] for p in kv[1]})):
        subs = {p["sid"] for p in pts}
        if len(subs) < min_subs:
            continue
        best = None
        for mstar in np.arange(2.0, 8.01, 0.25):
            a, y, ns = fe_design(pts, [lambda p: p["mbar"], hinge(mstar)])
            f = fit(a, y)
            if best is None or f["rss"] < best[1]["rss"]:
                best = (mstar, f, ns)
        mstar, f, ns = best
        b_lo = f["beta"][ns]
        b_hi = b_lo + f["beta"][ns + 1]
        print(
            f"{key:22s}{len(subs):6d}{mstar:6.2f}{b_lo:10.1f}{b_hi:10.1f}{f['rmse']:10.1f}"
        )
        out[key] = {
            "mstar": float(mstar),
            "b_lo": float(b_lo),
            "b_hi": float(b_hi),
            "rmse": f["rmse"],
            "subs": len(subs),
        }
    return out


def joint_fits(panel):
    print("\njoint fits with per-submission intercepts")
    res = {}

    a, y, ns = fe_design(panel, [lambda p: p["mbar"], lambda p: p["gbar"]])
    fp = fit(a, y)
    b, f = fp["beta"][ns], fp["beta"][ns + 1]
    sb, sf = fp["se"][ns], fp["se"][ns + 1]
    print(
        f"  Model P   b = {b:9.1f} +- {sb:7.1f}   f = {f:9.1f} +- {sf:7.1f}"
        f"   rmse {fp['rmse']:8.1f}  aicc {fp['aicc']:10.1f}  bic {fp['bic']:10.1f}"
    )
    print(f"            f 95 % CI [{f-1.96*sf:9.1f}, {f+1.96*sf:9.1f}]")
    res["P"] = {
        "b": float(b), "b_se": float(sb), "f": float(f), "f_se": float(sf),
        "rmse": fp["rmse"], "aicc": fp["aicc"], "bic": fp["bic"], "n": fp["n"],
    }

    best = None
    for mstar in np.arange(2.0, 8.01, 0.25):
        a, y, ns = fe_design(panel, [lambda p: p["mbar"], hinge(mstar)])
        fr = fit(a, y)
        if best is None or fr["rss"] < best[1]["rss"]:
            best = (mstar, fr, ns)
    mstar, fr, ns = best
    b_lo = fr["beta"][ns]
    d = fr["beta"][ns + 1]
    print(
        f"  Model R   M* = {mstar:4.2f}   b_lo = {b_lo:9.1f} +- {fr['se'][ns]:7.1f}"
        f"   b_hi = {b_lo+d:9.1f}"
    )
    print(
        f"            rmse {fr['rmse']:8.1f}  aicc {fr['aicc']:10.1f}"
        f"  bic {fr['bic']:10.1f}   (M* costs one extra parameter)"
    )
    res["R"] = {
        "mstar": float(mstar), "b_lo": float(b_lo), "b_hi": float(b_lo + d),
        "rmse": fr["rmse"], "aicc": fr["aicc"] + 2, "bic": fr["bic"] + math.log(fr["n"]),
        "n": fr["n"],
    }

    a, y, ns = fe_design(
        panel, [lambda p: p["mbar"], lambda p: p["gbar"], hinge(mstar)]
    )
    fb = fit(a, y)
    print(
        f"  P + R     b = {fb['beta'][ns]:9.1f}   f = {fb['beta'][ns+1]:9.1f}"
        f" +- {fb['se'][ns+1]:7.1f}   step = {fb['beta'][ns+2]:9.1f}"
        f" +- {fb['se'][ns+2]:7.1f}   rmse {fb['rmse']:8.1f}"
    )
    res["PR"] = {
        "b": float(fb["beta"][ns]), "f": float(fb["beta"][ns + 1]),
        "f_se": float(fb["se"][ns + 1]), "step": float(fb["beta"][ns + 2]),
        "step_se": float(fb["se"][ns + 2]), "rmse": fb["rmse"],
        "aicc": fb["aicc"], "bic": fb["bic"],
    }

    a, y, ns = fe_design(panel, [lambda p: p["mbar"]])
    f0 = fit(a, y)
    print(f"  linear    b = {f0['beta'][ns]:9.1f}   rmse {f0['rmse']:8.1f}  aicc {f0['aicc']:10.1f}")
    res["linear"] = {"b": float(f0["beta"][ns]), "rmse": f0["rmse"], "aicc": f0["aicc"]}

    dr = fp["rss"] - fr["rss"]
    print(
        f"\n  RSS  Model P {fp['rss']:.4g}   Model R {fr['rss']:.4g}"
        f"   difference {dr:+.4g}"
    )
    print(f"  preferred by AICc: {'R' if res['R']['aicc'] < res['P']['aicc'] else 'P'}")
    print(f"  preferred by BIC : {'R' if res['R']['bic'] < res['P']['bic'] else 'P'}")
    return res


def main():
    use_ref_only = "--all-rows" not in sys.argv
    tables = load_strata()
    panel, kept = build_panel(tables, use_ref_only)
    print("panel construction:", dict(kept))
    if not panel:
        print("no usable points")
        return
    label = "reference-schedule rows" if use_ref_only else "all rows, recovered R"
    strata = report_panel(panel, label)
    ps = per_stratum_fits(strata)
    js = joint_fits(panel)
    out = "/tmp/e128_strata_fit%s.json" % ("" if use_ref_only else "_all")
    json.dump({"per_stratum": ps, "joint": js, "label": label}, open(out, "w"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
