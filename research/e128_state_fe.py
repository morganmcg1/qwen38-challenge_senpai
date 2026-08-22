"""E128-F8: Finding 150's per-row runner state as a term in the cost curves.

harness=ranked (board receipts only). Zero GPU.

Finding 150 says a third runner state exists and that it adds a constant
`s in {0, ~820, ~1640} us` to every DRAFTING round of a run, and nothing to a
non-drafting round. The advisor writes that as a per-row intercept

    round_us(row, M) = a + s_row + b * M.

That collapse is not exact in this frame. `round_us` here is the run's mean
over ALL rounds, drafting and non-drafting together, because both
`effective_mean_draft_len = D / R` and `round_us = 512 * spt / R` divide by the
same total round count `R`. A constant added only to drafting rounds therefore
enters the mean scaled by the drafting fraction

    phi = 1 - non_drafting_round_count / R,

so the exact form is

    round_us(row, prompt) = a + b * mbar + s_row * phi.

Both forms are fitted below. The flat form is the advisor's literal request and
is the conservative one, because a per-row free intercept absorbs strictly more
than the state does. The `phi` form is the one Finding 150 actually predicts,
and it is testable inside a single submission row, because `phi` varies across
the eight prompts of one run while the flat term does not.

Sections:
  1  our ranked curve with a per-row intercept
  2  our ranked curve, two segments, with the state term
  3  the board population, joint constrained fit with a per-row intercept
  4  the fitted state against the F76 mode index
  5  the estimated state of every morganmcg1 board row
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter, defaultdict

import numpy as np

from e128_ourcurve import (
    MAX_ROWS,
    build_points,
    fixture_histograms,
    load_receipt,
    prompt_probs,
    r_scenarios,
)
from e128_rounds import PROMPTS, TOKENS, load_rows, per_prompt
from e128_strata_curve import MAXM, choose_rounds, gbar, gvec

ROWS = np.arange(1.0, MAX_ROWS + 1.0)
PANEL_CACHE = pathlib.Path("/tmp/e128_f8_panel.json")

# F76 mode index weights. index = sum_p w_p * 100 * ln(spt_p); threshold -12.9.
F76_W = {
    "plutarch": -0.3852, "drama": 0.0215, "travel": 0.4945, "beagle": 0.2068,
    "medicine": -0.1480, "republic": -0.0917, "essays": -0.0041,
    "botany": -0.0939,
}

# Our headline receipt and the rows Finding 150 groups with the rival frontier.
OUR_RECEIPT = "d3c491b5"
F150_CLUSTER = ["cf79f7df", "48423d09", "3b376ba2", "390ec878", "c63eaa21"]


# --------------------------------------------------------------- FE plumbing

def demean(y, x, groups):
    """Iterated within-transform over one or two grouping vectors."""
    y = y.astype(float).copy()
    x = x.astype(float).copy()
    idx = []
    for g in groups:
        keys = sorted(set(g))
        pos = {k: i for i, k in enumerate(keys)}
        idx.append((np.array([pos[v] for v in g]), len(keys)))
    for _ in range(60 if len(groups) > 1 else 1):
        before = x.copy()
        for ids, n in idx:
            cnt = np.bincount(ids, minlength=n).astype(float)
            y -= (np.bincount(ids, weights=y, minlength=n) / cnt)[ids]
            for c in range(x.shape[1]):
                m = np.bincount(ids, weights=x[:, c], minlength=n) / cnt
                x[:, c] -= m[ids]
        if len(groups) == 1 or np.max(np.abs(x - before)) < 1e-9:
            break
    return y, x


def fe_fit(y, x, groups, cluster=None, names=None):
    """Fixed-effect OLS with optional cluster-robust standard errors."""
    yd, xd = demean(np.asarray(y, float), np.asarray(x, float), groups)
    beta, *_ = np.linalg.lstsq(xd, yd, rcond=None)
    u = yd - xd @ beta
    n = len(yd)
    nfe = sum(len(set(g)) for g in groups) - (len(groups) - 1)
    k = xd.shape[1] + nfe
    rss = float(u @ u)
    xtx = np.linalg.pinv(xd.T @ xd)
    if cluster is None:
        cov = xtx * rss / max(n - k, 1)
    else:
        keys = sorted(set(cluster))
        meat = np.zeros((xd.shape[1], xd.shape[1]))
        for key in keys:
            m = np.array([c == key for c in cluster])
            s = xd[m].T @ u[m]
            meat += np.outer(s, s)
        g = len(keys)
        scale = g / max(g - 1, 1) * (n - 1) / max(n - k, 1)
        cov = xtx @ meat @ xtx * scale
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    kk = k + 1
    aic = n * math.log(rss / n) + 2 * kk
    return {
        "beta": beta, "se": se, "rss": rss, "rmse": math.sqrt(rss / n),
        "n": n, "k": kk, "aic": aic,
        "aicc": aic + 2 * kk * (kk + 1) / max(n - kk - 1, 1),
        "bic": n * math.log(rss / n) + math.log(n) * kk,
        "names": names or ["b%d" % i for i in range(xd.shape[1])],
    }


def fe_levels(y, x, beta, group):
    """Row intercepts implied by a within fit, centred on their mean."""
    y = np.asarray(y, float)
    resid = y - np.asarray(x, float) @ beta
    acc = defaultdict(list)
    for key, r in zip(group, resid):
        acc[key].append(r)
    lvl = {k: float(np.mean(v)) for k, v in acc.items()}
    mu = float(np.mean(list(lvl.values())))
    return {k: v - mu for k, v in lvl.items()}


def plain_fit(y, x, names):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    u = y - x @ beta
    n, k = len(y), x.shape[1] + 1
    rss = float(u @ u)
    se = np.sqrt(np.clip(
        np.diag(np.linalg.pinv(x.T @ x) * rss / max(n - k, 1)), 0.0, None))
    aic = n * math.log(rss / n) + 2 * k
    return {
        "beta": beta, "se": se, "rss": rss, "rmse": math.sqrt(rss / n),
        "n": n, "k": k, "aic": aic,
        "aicc": aic + 2 * k * (k + 1) / max(n - k - 1, 1),
        "bic": n * math.log(rss / n) + math.log(n) * k, "names": names,
    }


def show(tag, fit):
    body = "  ".join(
        "%s %9.1f +- %7.1f" % (nm, b, s)
        for nm, b, s in zip(fit["names"], fit["beta"], fit["se"]))
    print("  %-34s %s | rmse %8.1f  aicc %9.1f  bic %9.1f  n %d k %d"
          % (tag, body, fit["rmse"], fit["aicc"], fit["bic"],
             fit["n"], fit["k"]))


# ------------------------------------------------------------------- section 1

def our_points(args):
    hists = fixture_histograms(args.shipped)
    receipt = load_receipt(args.board, OUR_RECEIPT)
    points = build_points(receipt,
                          r_scenarios(args.identity)[args.scenario], hists)
    for p in points:
        raw = receipt["per_prompt"][p["prompt"]]
        p["n0"] = raw["non_drafting"]
        p["phi"] = 1.0 - raw["non_drafting"] / p["R"]
        p["spt"] = raw["candidate"]
    return receipt, points


def our_design(points, breakpoint, terms, extra):
    out = []
    for p in points:
        probs = prompt_probs(p, True)
        row = [1.0, float((probs * ROWS).sum())]
        if terms >= 3:
            above = ROWS >= breakpoint
            row.append(float(probs[above].sum()))
        if terms >= 4:
            row.append(float((probs[above] * (ROWS[above] - breakpoint)).sum()))
        for fn in extra:
            row.append(fn(p))
        out.append(row)
    return np.array(out)


def section1(points):
    print("\n=== 1  our ranked curve with a per-row intercept ===")
    print("  receipt %s, one submission row, eight prompts" % OUR_RECEIPT)
    print("  %-10s %5s %6s %6s %8s %10s"
          % ("prompt", "R", "n0", "phi", "mbar", "round_us"))
    for p in sorted(points, key=lambda q: q["mbar"]):
        print("  %-10s %5d %6d %6.3f %8.4f %10.1f"
              % (p["prompt"], p["R"], p["n0"], p["phi"], p["mbar"],
                 p["round_us"]))

    y = np.array([p["round_us"] for p in points])
    base = our_design(points, MAX_ROWS + 1, 2, [])
    flat = np.column_stack([base, np.ones(len(points))])
    print("\n  literal form  round_us = a + s_row + b*M, one submission row")
    print("  design rank with the s_row column %d, without it %d, columns %d"
          % (np.linalg.matrix_rank(flat), np.linalg.matrix_rank(base),
             flat.shape[1]))
    a = plain_fit(y, base, ["a", "b"])
    show("line, no s_row", a)
    b = plain_fit(y, flat, ["a", "b", "s_row"])
    show("line, + flat s_row", b)
    print("  slope moves by %.6f us/row; residual rmse moves by %.6f us"
          % (b["beta"][1] - a["beta"][1], b["rmse"] - a["rmse"]))

    print("\n  exact form    round_us = a + b*M + s*phi")
    c = plain_fit(y, np.column_stack([base, [p["phi"] for p in points]]),
                  ["a", "b", "s"])
    show("line, + s*phi", c)
    print("  phi correlation with mbar %.4f"
          % np.corrcoef([p["phi"] for p in points],
                        [p["mbar"] for p in points])[0, 1])
    return {"line": a, "line_flat": b, "line_phi": c, "y": y}


# ------------------------------------------------------------------- section 2

def section2(points, sec1):
    print("\n=== 2  our ranked curve, two segments, with the state term ===")
    y = sec1["y"]
    out = {}
    for label, extra, names in (
            ("no s_row", [], []),
            ("+ flat s_row", [lambda p: 1.0], ["s_row"]),
            ("+ s*phi", [lambda p: p["phi"]], ["s"])):
        print("  %s" % label)
        rows = []
        for bp in range(2, MAX_ROWS):
            x = our_design(points, bp, 4, extra)
            fit = plain_fit(y, x, ["a", "b", "jump", "dslope"] + names)
            fit["breakpoint"] = bp
            rows.append(fit)
        lin = plain_fit(y, our_design(points, MAX_ROWS + 1, 2, extra),
                        ["a", "b"] + names)
        rows.sort(key=lambda r: r["aicc"])
        for r in rows[:3]:
            show("break M>=%d" % r["breakpoint"], r)
        show("line (no break)", lin)
        best = rows[0]
        print("    best break M>=%d, advantage over the line "
              "dAICc %.2f  dBIC %.2f"
              % (best["breakpoint"], lin["aicc"] - best["aicc"],
                 lin["bic"] - best["bic"]))
        out[label] = {"best": best, "line": lin, "table": rows}
    return out


# ------------------------------------------------------------------- panel

def build_panel(limit=None):
    if PANEL_CACHE.exists():
        return json.loads(PANEL_CACHE.read_text())
    rows = load_rows()
    panel = []
    skipped = Counter()
    for row in rows:
        sid = (row.get("id") or "")[:8]
        if not sid:
            skipped["no-id"] += 1
            continue
        e = per_prompt(row)
        if len(e) != 8:
            skipped["incomplete"] += 1
            continue
        entries, bad = {}, False
        for name, x in e.items():
            dl = x.get("effective_mean_draft_len")
            spt = x.get("mtp_seconds_per_token_mean")
            if dl is None or spt is None or dl + 1.0 > MAXM:
                bad = True
                break
            entries[name] = {"dl": dl, "spt": spt, "mbar": dl + 1.0,
                             "n0": x.get("non_drafting_round_count") or 0}
        if bad:
            skipped["bad-record"] += 1
            continue
        pick, _, nc = choose_rounds(entries)
        index = sum(F76_W[n] * 100.0 * math.log(entries[n]["spt"])
                    for n in entries)
        for name, x in entries.items():
            r = pick[name]
            panel.append({
                "sid": sid, "prompt": name, "mbar": x["mbar"],
                "round_us": TOKENS * x["spt"] / r * 1e6, "R": r,
                "n0": x["n0"], "phi": 1.0 - x["n0"] / r, "spt": x["spt"],
                "solver": row.get("solverUsername"),
                "score": row.get("officialScore"),
                "created": row.get("createdAt"),
                "status": row.get("status"),
                "promotion": row.get("promotionStatus"),
                "f76_index": index,
            })
        skipped["kept"] += 1
        if limit and skipped["kept"] >= limit:
            break
    PANEL_CACHE.write_text(json.dumps({"panel": panel,
                                       "skipped": dict(skipped)}))
    return {"panel": panel, "skipped": dict(skipped)}


def attach_strata(panel):
    tables = {sid[:8]: {int(m): v for m, v in t.items()}
              for sid, t in json.load(
                  open("/tmp/e128_strata.json"))["tables"].items()}
    kept = []
    for p in panel:
        tbl = tables.get(p["sid"])
        if tbl is None:
            continue
        g = gvec(tbl)
        q = dict(p)
        q["gbar"] = gbar(g, p["mbar"])
        q["gkey"] = ",".join(str(g[m]) for m in range(1, MAXM + 1))
        kept.append(q)
    return kept


# ------------------------------------------------------------------- section 3

def hinge(m, star):
    return max(0.0, m - star)


def section3(panel, strata):
    print("\n=== 3  the board population with a per-row intercept ===")
    print("  full panel %d rows x 8 prompts = %d points"
          % (len({p["sid"] for p in panel}), len(panel)))
    print("  table-bearing panel %d rows, %d points"
          % (len({p["sid"] for p in strata}), len(strata)))

    y = np.array([p["round_us"] for p in strata])
    m = np.array([p["mbar"] for p in strata])
    g = np.array([p["gbar"] for p in strata])
    phi = np.array([p["phi"] for p in strata])
    sid = [p["sid"] for p in strata]
    prompt = [p["prompt"] for p in strata]

    print("\n  Model P   T = a + s_row + b*M + f*ceil(M/IPG),"
          " b common across strata")
    pooled = plain_fit(y, np.column_stack([np.ones(len(y)), m, g]),
                       ["a", "b", "f"])
    show("no s_row", pooled)
    fe1 = fe_fit(y, np.column_stack([m, g]), [sid], cluster=sid,
                 names=["b", "f"])
    show("row FE, clustered", fe1)
    fe2 = fe_fit(y, np.column_stack([m, g]), [sid, prompt], cluster=sid,
                 names=["b", "f"])
    show("row+prompt FE, clustered", fe2)
    fephi = fe_fit(y, np.column_stack([m, g, phi]), [sid, prompt], cluster=sid,
                   names=["b", "f", "s"])
    show("row+prompt FE, + s*phi", fephi)

    print("\n  Model R   T = a + s_row + b_lo*M + d*max(0, M - M*)")
    best = None
    for star in np.arange(1.5, 8.01, 0.125):
        h = np.array([hinge(v, star) for v in m])
        fit = fe_fit(y, np.column_stack([m, h]), [sid, prompt], cluster=sid,
                     names=["b_lo", "d"])
        fit["star"] = float(star)
        if best is None or fit["rss"] < best["rss"]:
            best = fit
    show("row+prompt FE, M*=%.3f" % best["star"], best)
    line = fe_fit(y, m.reshape(-1, 1), [sid, prompt], cluster=sid, names=["b"])
    show("row+prompt FE, single slope", line)

    both = None
    h = np.array([hinge(v, best["star"]) for v in m])
    both = fe_fit(y, np.column_stack([m, h, g]), [sid, prompt], cluster=sid,
                  names=["b_lo", "d", "f"])
    show("row+prompt FE, R + P together", both)

    print("\n  P vs R with the state term in both:"
          " dAICc %.1f  dBIC %.1f  (positive favours R)"
          % (fe2["aicc"] - best["aicc"], fe2["bic"] - best["bic"]))

    hg = np.array([hinge(v, best["star"]) for v in m])
    step5 = (m >= 5.0).astype(float)
    print("\n  is f identified separately from the break?")
    print("    corr(gbar, hinge at M*=%.3f) %.4f   corr(gbar, step M>=5) %.4f"
          % (best["star"], np.corrcoef(g, hg)[0, 1],
             np.corrcoef(g, step5)[0, 1]))
    hi = np.array([p["mbar"] >= 4.5 for p in strata])
    print("    high-width segment only, mbar >= 4.5: %d points, %d rows"
          % (int(hi.sum()), len({s for s, keep in zip(sid, hi) if keep})))
    hifit = fe_fit(y[hi], np.column_stack([m[hi], g[hi]]),
                   [[s for s, keep in zip(sid, hi) if keep],
                    [q for q, keep in zip(prompt, hi) if keep]],
                   cluster=[s for s, keep in zip(sid, hi) if keep],
                   names=["b", "f"])
    show("high width, row+prompt FE", hifit)

    # Per-stratum break location with the row intercept present.
    print("\n  per-stratum free break, row FE inside the stratum")
    groups = defaultdict(list)
    for p in strata:
        groups[p["gkey"]].append(p)
    print("  %-24s %5s %6s %10s %10s %9s   %s"
          % ("G(1..9)", "subs", "M*", "b_lo", "b_hi", "rmse", "G steps at"))
    per_stratum = []
    for key, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        subs = len({q["sid"] for q in pts})
        if subs < 5:
            continue
        yy = np.array([q["round_us"] for q in pts])
        mm = np.array([q["mbar"] for q in pts])
        ss = [q["sid"] for q in pts]
        bb = None
        for star in np.arange(1.5, 8.01, 0.125):
            hh = np.array([hinge(v, star) for v in mm])
            f = fe_fit(yy, np.column_stack([mm, hh]), [ss], names=["b_lo", "d"])
            f["star"] = float(star)
            if bb is None or f["rss"] < bb["rss"]:
                bb = f
        gv = [int(v) for v in key.split(",")]
        steps = [i + 1 for i in range(1, MAXM) if gv[i] != gv[i - 1]]
        print("  %-24s %5d %6.3f %10.1f %10.1f %9.1f   %s"
              % (key, subs, bb["star"], bb["beta"][0],
                 bb["beta"][0] + bb["beta"][1], bb["rmse"], steps))
        per_stratum.append({"gkey": key, "subs": subs, "star": bb["star"],
                            "b_lo": float(bb["beta"][0]),
                            "b_hi": float(bb["beta"][0] + bb["beta"][1]),
                            "rmse": bb["rmse"]})
    return {"pooled": pooled, "fe_row": fe1, "fe_two": fe2, "fe_phi": fephi,
            "hinge": best, "line": line, "both": both,
            "per_stratum": per_stratum}


# ------------------------------------------------------------------- section 4

from e128_state_s45 import section4, section5

# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument("--identity", type=pathlib.Path,
                    default=here / "e128-artifacts/rung0-identity.json")
    ap.add_argument("--shipped", type=pathlib.Path,
                    default=here / "e128-artifacts/rung1-shipped.json")
    ap.add_argument("--scenario", default="assumed")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    receipt, points = our_points(args)
    sec1 = section1(points)
    sec2 = section2(points, sec1)

    cache = build_panel()
    panel = cache["panel"]
    print("\n  panel build: %s" % cache["skipped"])
    strata = attach_strata(panel)
    sec3 = section3(panel, strata)
    sec4 = section4(panel, sec3, fe_fit, fe_levels, hinge, show,
                    OUR_RECEIPT, F150_CLUSTER)
    sec5 = section5(panel, sec4, OUR_RECEIPT, F150_CLUSTER)

    if args.json:
        def pack(f):
            return {"beta": [float(v) for v in f["beta"]],
                    "se": [float(v) for v in f["se"]],
                    "names": f["names"], "rmse": f["rmse"], "aicc": f["aicc"],
                    "bic": f["bic"], "n": f["n"], "k": f["k"],
                    "star": f.get("star"), "breakpoint": f.get("breakpoint")}
        args.json.write_text(json.dumps({
            "receipt": receipt["id"],
            "section1": {k: pack(v) for k, v in sec1.items() if k != "y"},
            "section2": {k: {"best": pack(v["best"]), "line": pack(v["line"])}
                         for k, v in sec2.items()},
            "section3": {k: pack(v) for k, v in sec3.items()
                         if k != "per_stratum"},
            "section3_per_stratum": sec3["per_stratum"],
            "section4": {
                "flat": pack(sec4["flat"]),
                "star": sec4["star"],
                "als_rel_rmse": sec4["als"]["rel_rmse"],
                "als_base": {q: float(v) for q, v in zip(
                    sec4["als"]["prompts"], sec4["als"]["base"])},
                "centres_us": [float(v) for v in sec4["centres"]],
                "six_row": sec4["six"],
                "families": sec4["families"],
                "ours_family": sec4.get("ours_family"),
                "s_rel_us": {k: float(v) for k, v in zip(
                    sec4["sids"], sec4["srel"])},
                "tree_speed": {k: float(v) for k, v in zip(
                    sec4["sids"], sec4["c"])},
                "s_flat": sec4["s_flat"]},
            "section5": sec5,
        }, indent=1, sort_keys=True))
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
