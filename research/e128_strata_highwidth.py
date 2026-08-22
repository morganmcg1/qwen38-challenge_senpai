"""E128 F7 item 3, decisive arm: the high-width segment.

Every stratum has G(1..4) = 1, so the strata are indistinguishable below M = 5.
Above M = 5 they diverge sharply:

  S1  G(5..9) = 2,2,2,2,3     121 submissions
  S2  G(5..9) = 2,2,2,3,3      39
  S3  G(5..9) = 2,1,2,1,1      12
  S4  G(5..9) = 2,1,1,1,1       9
  S5  G(5..9) = 2,1,2,3,1       9

Model P says the round cost above M = 5 tracks each stratum's own G, so S4
must fall by one pass price f between M = 5 and M = 6 while S1 stays flat.
Model R says the segment is one straight line in every stratum.

This fits only the prompt-rows with mean width >= 4.5 and compares:
  line      T = a_i + b*M
  P         T = a_i + b*M + f*Gbar_s(M)
  P (dev)   T = a_i + b*M + f*(Gbar_s(M) - Gbar_S1(M))   pass DEVIATION from
            the majority table, which removes any common curvature
"""

import json
import math
import statistics
from collections import defaultdict

import numpy as np

from e128_strata_curve import build_panel, fe_design, fit, gbar, gvec

MIN_MBAR = 4.5
S1_KEY = "1,1,1,1,2,2,2,2,3"


def twoway(pts, cols=None):
    """Submission and prompt fixed effects plus mean width and pass count.

    Errors are clustered on the submission, because the eight prompt-rows of
    one run share a thermal state and a build.
    """
    cols = cols or [lambda p: p["mbar"], lambda p: p["gbar"]]
    sids = sorted({p["sid"] for p in pts})
    prompts = sorted({p["prompt"] for p in pts})
    si = {s: i for i, s in enumerate(sids)}
    pi = {q: len(sids) + i for i, q in enumerate(prompts[1:])}
    k = len(sids) + len(prompts) - 1 + len(cols)
    a = np.zeros((len(pts), k))
    for r, p in enumerate(pts):
        a[r, si[p["sid"]]] = 1.0
        if p["prompt"] in pi:
            a[r, pi[p["prompt"]]] = 1.0
        for c, fn in enumerate(cols):
            a[r, k - len(cols) + c] = fn(p)
    y = np.array([p["round_us"] for p in pts])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    xtxi = np.linalg.pinv(a.T @ a)
    meat = np.zeros((k, k))
    by_cluster = defaultdict(list)
    for r, p in enumerate(pts):
        by_cluster[p["sid"]].append(r)
    for rows_ in by_cluster.values():
        u = a[rows_].T @ resid[rows_]
        meat += np.outer(u, u)
    g = len(by_cluster)
    n = len(pts)
    dof = int(np.linalg.matrix_rank(a))
    scale = g / max(g - 1, 1) * (n - 1) / max(n - dof, 1)
    cov = scale * xtxi @ meat @ xtxi
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    rss = float(resid @ resid)
    names = ["b", "f", "step"][: len(cols)]
    out = {"rmse": math.sqrt(rss / n), "n": n, "clusters": g}
    for i, nm in enumerate(names):
        v = float(beta[k - len(cols) + i])
        s = float(se[k - len(cols) + i])
        out[nm] = v
        out[nm + "_se"] = s
        print(
            f"    {nm:5s} = {v:10.1f} +- {s:7.1f}  (clustered)   "
            f"95 % CI [{v-1.96*s:9.1f}, {v+1.96*s:9.1f}]   t = {v/s if s else float('nan'):6.2f}"
        )
    print(f"    rmse {out['rmse']:8.1f}   n {n}   clusters {g}")
    return out


def main():
    panel, _ = build_panel()
    tables = {
        sid: {int(m): v for m, v in t.items()}
        for sid, t in json.load(open("/tmp/e128_strata.json"))["tables"].items()
    }
    g1 = gvec({3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3})
    hi = [p for p in panel if p["mbar"] >= MIN_MBAR]
    for p in hi:
        p["gdev"] = p["gbar"] - gbar(g1, p["mbar"])
    strata = defaultdict(list)
    for p in hi:
        strata[p["gkey"]].append(p)

    print(f"high-width points (mbar >= {MIN_MBAR}): {len(hi)} over "
          f"{len({p['sid'] for p in hi})} submissions")
    print(f"\n{'G(1..9)':22s}{'subs':>6s}{'pts':>6s}{'gbar p10':>10s}{'gbar p90':>10s}")
    for key, pts in sorted(strata.items(), key=lambda kv: -len({p['sid'] for p in kv[1]})):
        gb = sorted(p["gbar"] for p in pts)
        print(
            f"{key:22s}{len({p['sid'] for p in pts}):6d}{len(pts):6d}"
            f"{gb[int(0.1*(len(gb)-1))]:10.3f}{gb[int(0.9*(len(gb)-1))]:10.3f}"
        )

    a, y, ns = fe_design(hi, [lambda p: p["mbar"]])
    f0 = fit(a, y)
    print(f"\n  line   b = {f0['beta'][ns]:9.1f} +- {f0['se'][ns]:6.1f}"
          f"   rmse {f0['rmse']:8.1f}  aicc {f0['aicc']:9.1f}  bic {f0['bic']:9.1f}")

    a, y, ns = fe_design(hi, [lambda p: p["mbar"], lambda p: p["gbar"]])
    fp = fit(a, y)
    print(f"  P      b = {fp['beta'][ns]:9.1f} +- {fp['se'][ns]:6.1f}"
          f"   f = {fp['beta'][ns+1]:9.1f} +- {fp['se'][ns+1]:6.1f}"
          f"   rmse {fp['rmse']:8.1f}  aicc {fp['aicc']:9.1f}  bic {fp['bic']:9.1f}")
    f_, sf = fp["beta"][ns + 1], fp["se"][ns + 1]
    print(f"         f 95 % CI [{f_-1.96*sf:9.1f}, {f_+1.96*sf:9.1f}]"
          f"   t = {f_/sf if sf else float('nan'):6.2f}")

    a, y, ns = fe_design(hi, [lambda p: p["mbar"], lambda p: p["gdev"]])
    fd = fit(a, y)
    fd_, sfd = fd["beta"][ns + 1], fd["se"][ns + 1]
    print(f"  P dev  b = {fd['beta'][ns]:9.1f} +- {fd['se'][ns]:6.1f}"
          f"   f = {fd_:9.1f} +- {sfd:6.1f}"
          f"   rmse {fd['rmse']:8.1f}  aicc {fd['aicc']:9.1f}  bic {fd['bic']:9.1f}")
    print(f"         f 95 % CI [{fd_-1.96*sfd:9.1f}, {fd_+1.96*sfd:9.1f}]"
          f"   t = {fd_/sfd if sfd else float('nan'):6.2f}")

    print("\n  per-stratum slope of the high segment (own intercepts, own slope)")
    print(f"{'G(1..9)':22s}{'subs':>6s}{'b':>10s}{'se':>8s}{'rmse':>10s}")
    for key, pts in sorted(strata.items(), key=lambda kv: -len({p['sid'] for p in kv[1]})):
        if len({p["sid"] for p in pts}) < 5:
            continue
        a, y, ns = fe_design(pts, [lambda p: p["mbar"]])
        f = fit(a, y)
        print(f"{key:22s}{len({p['sid'] for p in pts}):6d}{f['beta'][ns]:10.1f}"
              f"{f['se'][ns]:8.1f}{f['rmse']:10.1f}")

    print("\n  residual of each stratum against the COMMON high-width line")
    a, y, ns = fe_design(hi, [lambda p: p["mbar"]])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    for i, p in enumerate(hi):
        p["resid"] = float(resid[i])
    print(f"{'G(1..9)':22s}{'prompt':11s}{'n':>5s}{'mbar':>8s}{'Gbar':>7s}"
          f"{'resid us':>11s}{'pred if P':>11s}")
    fhat = fp["beta"][ns + 1]
    for key, pts in sorted(strata.items(), key=lambda kv: -len({p['sid'] for p in kv[1]})):
        if len({p["sid"] for p in pts}) < 5:
            continue
        byp = defaultdict(list)
        for p in pts:
            byp[p["prompt"]].append(p)
        for name in sorted(byp, key=lambda n: statistics.median(q["mbar"] for q in byp[n])):
            v = byp[name]
            print(
                f"{key:22s}{name:11s}{len(v):5d}"
                f"{statistics.median(q['mbar'] for q in v):8.3f}"
                f"{statistics.median(q['gbar'] for q in v):7.3f}"
                f"{statistics.median(q['resid'] for q in v):11.1f}"
                f"{fhat*statistics.median(q['gbar'] for q in v):11.1f}"
            )

    print("\n  two-way fixed effects: submission AND prompt")
    print("  (f is then identified only by the same prompt sitting in strata")
    print("   with different pass counts, so any per-prompt round-count bias")
    print("   drops out)")
    tw = twoway(hi)
    out = {
        "n_points": len(hi),
        "line": {"b": float(f0["beta"][ns]), "rmse": f0["rmse"], "aicc": f0["aicc"], "bic": f0["bic"]},
        "P": {"b": float(fp["beta"][ns]), "f": float(f_), "f_se": float(sf),
              "rmse": fp["rmse"], "aicc": fp["aicc"], "bic": fp["bic"]},
        "P_dev": {"b": float(fd["beta"][ns]), "f": float(fd_), "f_se": float(sfd),
                  "rmse": fd["rmse"], "aicc": fd["aicc"], "bic": fd["bic"]},
        "twoway": tw,
    }
    json.dump(out, open("/tmp/e128_strata_highwidth.json", "w"))
    print("\nwrote /tmp/e128_strata_highwidth.json")


if __name__ == "__main__":
    main()
