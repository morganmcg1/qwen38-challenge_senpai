"""Replacement body for E128-F8 sections 4 and 5.

Section 4 asks whether the fitted per-row term is a calibrated state
classifier. A per-row intercept in a global cost curve is not one: it absorbs
the whole quality difference between one solver tree and another, which on this
board is tens of thousands of microseconds per round, four orders above the
820 us the state model predicts. The state needs an estimator that separates a
multiplicative tree-speed channel from an additive per-drafting-round channel.

Finding 150's own physics gives that estimator directly. Writing `spt` for
candidate seconds per token, `dr` for drafting rounds and `W = 512` tokens,

    spt_ip = c_i * base_p  +  s_i * 1e-6 * dr_ip / W

`c_i` is the row's multiplicative tree speed, `base_p` is the common per-prompt
shape, and `s_i` is the additive state in microseconds per drafting round. The
two channels have different per-prompt shapes: on our crown receipt the
drafting-round share runs from 0.074 on plutarch to 0.492 on drama while the
per-token time runs the other way, 0.0303 s on plutarch down to 0.0102 s on
botany. That shape difference is what identifies the split, and it is the same
reason Finding 150 reads plutarch as the state's null.

The absolute level of `s` is only weakly identified, because a constant shift
of every `s_i` can be partly absorbed into `base_p` whenever every row runs a
similar schedule. Every number here is therefore reported relative to a named
reference row.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def kmeans1d(x, k, iters=300):
    x = np.sort(np.asarray(x, float))
    cent = np.quantile(x, np.linspace(0.15, 0.85, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - cent[None, :]), axis=1)
        new = np.array([x[lab == j].mean() if (lab == j).any() else cent[j]
                        for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    lab = np.argmin(np.abs(x[:, None] - cent[None, :]), axis=1)
    wss = float(((x - cent[lab]) ** 2).sum())
    tss = float(((x - x.mean()) ** 2).sum())
    return np.sort(cent), wss, tss


def als_state(panel, window=512, iters=400):
    """Alternating least squares for spt = c_i base_p + s_i 1e-6 dr_ip / W."""
    sids = sorted({p["sid"] for p in panel})
    prompts = sorted({p["prompt"] for p in panel})
    si = {s: i for i, s in enumerate(sids)}
    pi = {q: i for i, q in enumerate(prompts)}
    y = np.full((len(sids), len(prompts)), np.nan)
    z = np.zeros_like(y)
    for p in panel:
        y[si[p["sid"]], pi[p["prompt"]]] = p["spt"]
        z[si[p["sid"]], pi[p["prompt"]]] = (p["R"] - p["n0"]) / window * 1e-6
    keep = ~np.isnan(y).any(axis=1)
    y, z = y[keep], z[keep]
    sids = [s for s, k in zip(sids, keep) if k]

    base = y.mean(axis=0)
    c = np.ones(len(sids))
    s = np.zeros(len(sids))
    w = 1.0 / base ** 2
    for _ in range(iters):
        for i in range(len(sids)):
            a = np.column_stack([base, z[i]]) * np.sqrt(w)[:, None]
            beta, *_ = np.linalg.lstsq(a, y[i] * np.sqrt(w), rcond=None)
            c[i], s[i] = beta
        num = (c[:, None] * (y - s[:, None] * z)).sum(axis=0)
        base = num / float((c ** 2).sum())
        scale = float(c.mean())
        base, c = base * scale, c / scale
    resid = y - (c[:, None] * base[None, :] + s[:, None] * z)
    rel = np.abs(resid) / y
    se = np.zeros(len(sids))
    for i in range(len(sids)):
        a = np.column_stack([base, z[i]]) * np.sqrt(w)[:, None]
        sigma2 = float((resid[i] ** 2 * w).sum()) / max(len(base) - 2, 1)
        cov = np.linalg.pinv(a.T @ a) * sigma2
        se[i] = math.sqrt(max(cov[1, 1], 0.0))
    return {"sids": sids, "prompts": prompts, "base": base, "c": c, "s": s,
            "se": se, "rel_row": np.sqrt((rel ** 2).mean(axis=1)),
            "rel_rmse": float(np.sqrt((rel ** 2).mean())),
            "rel_p95": float(np.percentile(rel, 95))}


def shape_families(panel, k=8, iters=200):
    """Group rows by the shape of their per-prompt drafting-round vector.

    Two runs of the same draft schedule spend their rounds on the eight
    prompts in the same proportions, so the normalised vector is a schedule
    fingerprint that needs no source access.
    """
    prompts = sorted({p["prompt"] for p in panel})
    pi = {q: i for i, q in enumerate(prompts)}
    vec = {}
    for p in panel:
        vec.setdefault(p["sid"], np.zeros(len(prompts)))
        vec[p["sid"]][pi[p["prompt"]]] = p["R"] - p["n0"]
    sids = sorted(vec)
    x = np.array([vec[s] / max(vec[s].sum(), 1.0) for s in sids])
    rng = np.random.default_rng(0)
    cent = x[rng.choice(len(x), size=k, replace=False)]
    lab = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                cent[j] = x[lab == j].mean(axis=0)
    return {s: int(v) for s, v in zip(sids, lab)}


def section4(panel, sec3, fe_fit, fe_levels, hinge, show, our, cluster_ids):
    print("\n=== 4  a calibrated state estimator, and what a per-row"
          " intercept really absorbs ===")
    y = np.array([p["round_us"] for p in panel])
    m = np.array([p["mbar"] for p in panel])
    sid = [p["sid"] for p in panel]
    prompt = [p["prompt"] for p in panel]
    star = sec3["hinge"]["star"]
    h = np.array([hinge(v, star) for v in m])

    flat = fe_fit(y, np.column_stack([m, h]), [sid, prompt], cluster=sid,
                  names=["b_lo", "d"])
    show("flat s_row, full board", flat)
    s_flat = fe_levels(y, np.column_stack([m, h]), flat["beta"], sid)
    vals = np.array([s_flat[k] for k in sorted(s_flat)])
    print("  flat s_row spread p5..p95 %.0f .. %.0f us, full range %.0f us"
          % (np.percentile(vals, 5), np.percentile(vals, 95),
             vals.max() - vals.min()))
    print("  Finding 150 predicts a total spread of about 1640 us, so the"
          " per-row intercept is %.0fx too wide to be the state"
          % ((vals.max() - vals.min()) / 1640.0))

    als = als_state(panel)
    s = als["s"]
    c = als["c"]
    sids = als["sids"]
    print("\n  two-channel fit  spt = c_i base_p + s_i 1e-6 dr_ip / 512")
    print("    rows %d, relative rmse %.5f, relative p95 %.5f"
          % (len(sids), als["rel_rmse"], als["rel_p95"]))
    print("    base_p %s"
          % "  ".join("%s %.5f" % (q, v)
                      for q, v in zip(als["prompts"], als["base"])))
    print("    tree speed c: p5 %.4f  median %.4f  p95 %.4f"
          % (np.percentile(c, 5), np.median(c), np.percentile(c, 95)))

    idx = {p["sid"]: p["f76_index"] for p in panel}
    ref = sids.index(cluster_ids[0]) if cluster_ids[0] in sids else None
    anchor = float(np.mean([s[sids.index(k)] for k in cluster_ids
                            if k in sids]))
    srel = s - anchor
    print("    state s relative to the Finding 150 reference cluster mean:"
          " p5 %.0f  median %.0f  p95 %.0f us/drafting round"
          % (np.percentile(srel, 5), np.median(srel), np.percentile(srel, 95)))
    print("    correlation of s with the tree speed c %.4f, with the F76"
          " index %.4f"
          % (np.corrcoef(s, c)[0, 1],
             np.corrcoef(s, [idx[k] for k in sids])[0, 1]))

    print("\n  one common base_p cannot serve the whole board: relative"
          " misfit %.4f is five times the 1.15 %% the state is worth."
          % als["rel_rmse"])
    print("  base_p is only a valid shape inside one draft-schedule family,"
          " so the estimator is refitted family by family.")
    fam = shape_families(panel, k=8)
    local = {}
    print("  %-5s %5s %8s %9s %9s %8s %8s %6s   %s"
          % ("fam", "rows", "misfit", "s p5", "s p95", "gap k=2", "gap k=3",
             "F76", "3-way centres"))
    good_fams = []
    fam_table = []
    for f in sorted(set(fam.values())):
        members = {s for s, v in fam.items() if v == f}
        if len(members) < 10:
            continue
        sub = [p for p in panel if p["sid"] in members]
        a = als_state(sub)
        v = a["s"] - float(np.median(a["s"]))
        cent2, _, _ = kmeans1d(v, 2)
        cent3, _, _ = kmeans1d(v, 3)
        for j, k2 in enumerate(a["sids"]):
            local[k2] = {"s": float(v[j]), "se": float(a["se"][j]),
                         "fit": float(a["rel_row"][j]), "fam": int(f)}
        lab2 = np.argmin(np.abs(v[:, None] - cent2[None, :]), axis=1)
        band = np.array([idx[k2] > -12.9 for k2 in a["sids"]]).astype(int)
        agree = max((lab2 == band).mean(), (lab2 != band).mean())
        print("  %-5d %5d %8.4f %9.0f %9.0f %8.0f %8.0f %6.3f   %s"
              % (f, len(a["sids"]), a["rel_rmse"], np.percentile(v, 5),
                 np.percentile(v, 95), float(np.diff(cent2)[0]),
                 float(np.diff(cent3).mean()), agree,
                 " ".join("%8.0f" % t for t in cent3)))
        fam_table.append({"family": int(f), "rows": len(a["sids"]),
                          "misfit": a["rel_rmse"],
                          "gap_k2_us": float(np.diff(cent2)[0]),
                          "gap_k3_us": float(np.diff(cent3).mean()),
                          "f76_agreement": float(agree),
                          "centres_k3_us": [float(t) for t in cent3]})
        if a["rel_rmse"] < 0.02:
            good_fams.append(int(f))
    print("  families whose rank-1 tree model holds to 2 %%: %s"
          % good_fams)
    gsid = [k2 for k2, v in local.items() if v["fam"] in good_fams]
    gv = np.array([local[k2]["s"] for k2 in gsid])
    gb = np.array([idx[k2] > -12.9 for k2 in gsid]).astype(int)
    gc2, _, _ = kmeans1d(gv, 2)
    gl = np.argmin(np.abs(gv[:, None] - gc2[None, :]), axis=1)
    print("  union of those families: %d rows, 2-way centres %s,"
          " spacing %.0f us, F76 band agreement %.3f, correlation %.4f"
          % (len(gsid), " ".join("%.0f" % t for t in gc2),
             float(np.diff(gc2)[0]),
             max((gl == gb).mean(), (gl != gb).mean()),
             np.corrcoef(gv, [idx[k2] for k2 in gsid])[0, 1]))

    have = [k2 for k2 in sids if k2 in local and local[k2]["fam"] in good_fams]
    vloc = np.array([local[k2]["s"] for k2 in have])
    iloc = np.array([idx[k2] for k2 in have])
    print("\n  within-family state, %d rows, median se %.0f us"
          % (len(have), np.median([local[k2]["se"] for k2 in have])))
    for k2 in (2, 3, 4):
        cent, wss, tss = kmeans1d(vloc, k2)
        print("    k=%d centres %s  gaps %s  variance explained %.3f"
              % (k2, " ".join("%8.0f" % v for v in cent),
                 " ".join("%8.0f" % v for v in np.diff(cent)),
                 1.0 - wss / tss))
    cent, _, _ = kmeans1d(vloc, 3)
    lab = np.argmin(np.abs(vloc[:, None] - cent[None, :]), axis=1)
    ib = (iloc > -12.9).astype(int)
    best = 0.0
    for cut in (0, 1):
        pred = (lab > cut).astype(int)
        best = max(best, (pred == ib).mean(), (pred != ib).mean())
    print("    correlation with the F76 index %.4f;"
          " agreement with the -12.9 band, best 2-way split %.3f"
          % (np.corrcoef(vloc, iloc)[0, 1], best))
    ok = np.array([k2 in local for k2 in sids])

    print("\n  the decisive check: our crown against the Finding 150 cluster")
    six = [k2 for k2 in [our] + list(cluster_ids) if k2 in sids]
    a6 = als_state([p for p in panel if p["sid"] in set(six)])
    ref = float(np.mean([a6["s"][a6["sids"].index(k2)] for k2 in cluster_ids
                         if k2 in a6["sids"]]))
    by = {p["sid"]: p for p in panel}
    print("  six-row local fit, relative misfit %.5f" % a6["rel_rmse"])
    print("  %-9s %-14s %9s %9s %8s %8s %9s"
          % ("id", "solver", "c", "s_rel us", "se", "fit", "f76"))
    for k2 in six:
        i = a6["sids"].index(k2)
        print("  %-9s %-14s %9.4f %9.0f %8.0f %8.5f %9.2f"
              % (k2, by[k2]["solver"], a6["c"][i], a6["s"][i] - ref,
                 a6["se"][i], a6["rel_row"][i], idx[k2]))
    if our in a6["sids"]:
        i = a6["sids"].index(our)
        print("  advisor modetest offset 817 us/drafting round;"
              " this estimator gives %.0f +- %.0f us"
              % (a6["s"][i] - ref, a6["se"][i]))
        peers = [a6["s"][a6["sids"].index(k2)] for k2 in cluster_ids
                 if k2 in a6["sids"]]
        print("  the five cluster rows span %.0f us and cover three solver"
              " accounts and at least three trees"
              % (max(peers) - min(peers)))
    return {"flat": flat, "s_flat": s_flat, "als": als, "srel": srel,
            "sids": sids, "c": c, "centres": cent, "index": idx,
            "ok": ok, "local": local, "families": fam_table,
            "six": {"misfit": a6["rel_rmse"],
                    "rows": {k3: {"c": float(a6["c"][a6["sids"].index(k3)]),
                                  "s_rel_us": float(
                                      a6["s"][a6["sids"].index(k3)] - ref),
                                                                                                                                                           "fit": float(
                                      a6["rel_row"][a6["sids"].index(k3)])}
                             for k3 in six}},
            "star": float(star)}


def section5(panel, sec4, our, cluster_ids):
    print("\n=== 5  the estimated state of every morganmcg1 board row ===")
    sids = sec4["sids"]
    srel = sec4["srel"]
    c = sec4["c"]
    cent = sec4["centres"]
    by = {}
    for p in panel:
        by.setdefault(p["sid"], p)
    ours = sorted((s for s in sids if by[s]["solver"] == "morganmcg1"),
                  key=lambda s: by[s]["created"])
    print("  state levels from the 3-way clustering %s us/drafting round,"
          " each referenced to its own draft-schedule family median"
          % " ".join("%.0f" % v for v in cent))
    print("  %-9s %-19s %8s %8s %4s %9s %7s %8s %6s %8s %s"
          % ("id", "created", "score", "c", "fam", "s_local", "se", "fit",
             "state", "f76", "promotion"))
    out = []
    local = sec4["local"]
    als = sec4["als"]
    for s in ours:
        i = sids.index(s)
        p = by[s]
        loc = local.get(s)
        sv = loc["s"] if loc else float("nan")
        lab = int(np.argmin(np.abs(cent - sv))) if loc else -1
        print("  %-9s %-19s %8.5f %8.4f %4s %9.0f %7.0f %8.5f %6d %8.2f %s"
              % (s, p["created"][:19], p["score"] or float("nan"), c[i],
                 loc["fam"] if loc else "-", sv,
                 loc["se"] if loc else float("nan"),
                 loc["fit"] if loc else float("nan"), lab,
                 p["f76_index"], p["promotion"]))
        out.append({"sid": s, "created": p["created"], "score": p["score"],
                    "c": float(c[i]),
                    "s_local_us": float(sv) if loc else None,
                    "family": loc["fam"] if loc else None,
                    "se_us": float(loc["se"]) if loc else None,
                    "rel_fit": float(loc["fit"]) if loc else None,
                    "state": lab, "s_global_us": float(srel[i]),
                    "s_global_se_us": float(als["se"][i]),
                    "f76_index": p["f76_index"],
                    "promotion": p["promotion"],
                    "s_flat": sec4["s_flat"].get(s)})
    keep = [r for r in out if r["s_local_us"] is not None]
    if keep:
        v = np.array([r["s_local_us"] for r in keep])
        print("\n  %d of %d morganmcg1 rows sit in a family large enough to"
              " fit; their within-family state runs %.0f to %.0f us,"
              " median %.0f us"
              % (len(keep), len(out), v.min(), v.max(), np.median(v)))
    big = Counter(r["family"] for r in keep).most_common(1)
    if big and big[0][1] >= 6:
        f = big[0][0]
        sub = [r for r in keep if r["family"] == f]
        v = np.array([r["s_local_us"] for r in sub])
        c2, wss, tss = kmeans1d(v, 2)
        lab = np.argmin(np.abs(v[:, None] - c2[None, :]), axis=1)
        band = np.array([r["f76_index"] > -12.9 for r in sub]).astype(int)
        agree = max((lab == band).mean(), (lab != band).mean())
        print("  our own rows inside schedule family %d, the family that"
              " holds our recent frontier: %d rows" % (f, len(sub)))
        print("    2-way centres %.0f and %.0f us, spacing %.0f us,"
              " variance explained %.3f"
              % (c2[0], c2[1], c2[1] - c2[0], 1.0 - wss / tss))
        print("    agreement with the F76 -12.9 band %.3f (%d of %d)"
              % (agree, int(round(agree * len(sub))), len(sub)))
        for r, la in sorted(zip(sub, lab), key=lambda t: t[0]["s_local_us"]):
            r["state_local"] = int(la)
        sec4["ours_family"] = {
            "family": int(f), "rows": len(sub),
            "centre_lo_us": float(c2[0]), "centre_hi_us": float(c2[1]),
            "spacing_us": float(c2[1] - c2[0]),
            "variance_explained": float(1.0 - wss / tss),
            "f76_agreement": float(agree)}
        print("    slow-state rows %s"
              % " ".join(r["sid"] for r, la in zip(sub, lab)
                         if la == int(np.argmax(c2))))
    return out
