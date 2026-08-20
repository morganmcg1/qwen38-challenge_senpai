#!/usr/bin/env python3
"""E73 rung 2 - fit the QMV group-partition cost model, run both positive
controls, then predict the ranked-optimal partition table.

Model forms compared, all fitted to the same rung-1 session:

  A  additive (the brief's original form, kept as the refutation baseline)
       t = a * groups * W + b(IPG) + c(shape)

  B  occupancy (the respecified form)
       t = [ groups*W + beta*M*k*Tn ] * rho0 * q(IPG) * (1 + lam*S(IPG)/x)
       x  = groups * Tn / cores          working threadgroups per core
       Tn = ceil(n/8)                    row groups
       S(IPG)                            rung-0 derived resident simdgroups
                                         per core; a fixed input, not fitted

  B0 occupancy with q == 1, i.e. no free per-IPG level term.

`cores` is the only host parameter. It is read from the device.
"""
import argparse
import itertools
import json
import math
import statistics

SG_PER_TG = 2  # group_dims(32, 2, 1)

# Rung-0 derived resident simdgroups per core (384 KiB register file assumed).
# Fixed input to the fit, never a free parameter.
SG_REG = {2: 37, 3: 31, 4: 26, 5: 23, 6: 16}

# name, n, k, calls per verify round (research/xgroup_census.py, E33 8.2)
SCORED_SHAPES = [
    ("mlp.gate_up_fused", 34816, 5120, 64),
    ("mlp.down", 5120, 17408, 64),
    ("linear_attn.in_proj_fused_qkvzba", 16480, 5120, 48),
    ("linear_attn.out_proj", 5120, 6144, 48),
    ("full_attn.qkv_proj_fused", 14336, 5120, 16),
    ("full_attn.o_proj", 5120, 6144, 16),
    ("head.lm_head", 248320, 5120, 1),
]

# E33's measured per-shape cost ratio at M=6, ledger item 137.
# arm  = <T,6,6> with 2 sequential row blocks  (groups 1, row_blocks 2)
# base = <T,6,3>                               (groups 2, row_blocks 1)
E33 = [
    ("head.lm_head", 248320, 5120, 0.9830),
    ("head.compact_draft_vocab", 98336, 5120, 0.9868),
    ("mlp.gate_up_fused", 34816, 5120, 0.9941),
    ("linear_attn.in_proj_fused_qkvzba", 16480, 5120, 0.9947),
    ("full_attn.qkv_proj_fused", 14336, 5120, 1.0148),
    ("full_attn.o_proj", 5120, 6144, 1.0414),
    ("linear_attn.out_proj", 5120, 6144, 1.0492),
    ("mlp.down", 5120, 17408, 1.0592),
]

# askeladd E71 in-situ ms/GB at M=6 on the shipped partition, merged.
E71_MS_PER_GB = {
    "head.lm_head": 2.995,
    "mlp.gate_up_fused": 3.259,
    "linear_attn.out_proj": 3.901,
    "full_attn.o_proj": 4.099,
    "mlp.down": 5.263,
}

SHIPPED_LOCAL = {3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 4, 9: 5}
CROWN = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}


def legal_ipg(m):
    return [i for i in range(2, 7) if m % i != 1 and i <= m]


def weight_bytes(n, k):
    """affine 4-bit group-64: packed weights plus fp16 scale and bias."""
    return n * k * 0.5 + (n * k / 64.0) * 2 * 2


# ---------------------------------------------------------------- data


def load(path):
    d = json.load(open(path))
    cells = []
    for sh in d["shapes"]:
        by_arm = {}
        for leg in sh["legs"]:
            by_arm.setdefault(leg["arm"], []).append(leg)
        for arm, legs in by_arm.items():
            pos = {}
            for l in legs:
                pos.setdefault(l["position"], []).append(l["seconds_per_dispatch"])
            m, ipg = legs[0]["m"], legs[0]["ipg"]
            cells.append(dict(
                shape=sh["shape"], n=sh["n"], k=sh["k"], w=sh["bytes_per_stream"],
                m=m, ipg=ipg, groups=math.ceil(m / ipg), tn=sh["n"] // 8,
                t=statistics.median(l["seconds_per_dispatch"] for l in legs),
                pos=sorted(statistics.median(v) for v in pos.values()),
            ))
    return d, cells


def session_null(cells):
    v = [abs(c["pos"][1] - c["pos"][0]) / c["pos"][0] for c in cells if len(c["pos"]) == 2]
    v.sort()
    return dict(n=len(v), median=statistics.median(v), p90=v[int(0.9 * len(v))], max=v[-1])


# ------------------------------------------------------------- solvers


def nelder_mead(f, x0, step=0.15, iters=4000, tol=1e-12):
    n = len(x0)
    pts = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step * (abs(p[i]) + 1e-3)
        pts.append(p)
    val = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: val[i])
        pts = [pts[i] for i in order]
        val = [val[i] for i in order]
        if abs(val[-1] - val[0]) < tol * (abs(val[0]) + tol):
            break
        cen = [sum(p[i] for p in pts[:-1]) / n for i in range(n)]
        ref = [cen[i] + (cen[i] - pts[-1][i]) for i in range(n)]
        fr = f(ref)
        if fr < val[0]:
            exp = [cen[i] + 2.0 * (cen[i] - pts[-1][i]) for i in range(n)]
            fe = f(exp)
            pts[-1], val[-1] = (exp, fe) if fe < fr else (ref, fr)
        elif fr < val[-2]:
            pts[-1], val[-1] = ref, fr
        else:
            con = [cen[i] + 0.5 * (pts[-1][i] - cen[i]) for i in range(n)]
            fc = f(con)
            if fc < val[-1]:
                pts[-1], val[-1] = con, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(n)]
                    val[i] = f(pts[i])
    i = min(range(n + 1), key=lambda i: val[i])
    return pts[i], val[i]


def rel_rms(pred, obs):
    r = [(p - o) / o for p, o in zip(pred, obs)]
    return math.sqrt(sum(x * x for x in r) / len(r)), r


# -------------------------------------------------------------- models


def _shape_of(n, k):
    for name, nn, kk, _ in SCORED_SHAPES:
        if nn == n and kk == k:
            return name
    return None


def fit_additive(cells):
    """A: t = a*groups*W + b(IPG) + c(shape); the brief's original form."""
    ipgs = sorted({c["ipg"] for c in cells})
    shapes = sorted({c["shape"] for c in cells})

    def unpack(p):
        a = p[0]
        b = {ipgs[0]: 0.0}
        b.update({ipgs[i]: p[i] for i in range(1, len(ipgs))})
        c = {shapes[i]: p[len(ipgs) - 1 + i] for i in range(len(shapes))}
        return a, b, c

    def pred(p):
        a, b, c = unpack(p)
        return [a * x["groups"] * x["w"] + b[x["ipg"]] + c[x["shape"]] for x in cells]

    obs = [x["t"] for x in cells]
    x0 = [3e-12] + [0.0] * (len(ipgs) - 1) + [1e-4] * len(shapes)
    best, _ = nelder_mead(lambda p: rel_rms(pred(p), obs)[0], x0, step=0.4, iters=20000)
    best, _ = nelder_mead(lambda p: rel_rms(pred(p), obs)[0], best, step=0.05, iters=20000)
    rms, res = rel_rms(pred(best), obs)
    a, b, c = unpack(best)
    cbar = statistics.mean(c.values())

    def t(m, ipg, n, k, cores=None, groups=None, row_blocks=1):
        g = groups if groups is not None else math.ceil(m / ipg)
        return a * g * weight_bytes(n, k) + b.get(ipg, 0.0) + c.get(_shape_of(n, k), cbar)

    return dict(name="A additive", nparam=len(best), rms=rms, res=res,
                a=a, b=b, c=c, params=best, t=t)


def make_occupancy(free_q, free_lam):
    """free_lam=False pins the occupancy scale to the rung-0 register census."""
    ipgs = [2, 3, 4, 5, 6]
    rest = [i for i in ipgs if i != 3]

    def unpack(p):
        rho0, beta = math.exp(p[0]), math.exp(p[1])
        j = 2
        if free_lam:
            lam = {ipgs[i]: math.exp(p[j + i]) for i in range(5)}
            j += 5
        else:
            l0 = math.exp(p[j])
            lam = {i: l0 * SG_REG[i] for i in ipgs}
            j += 1
        q = {3: 1.0}
        if free_q:
            q.update({rest[i]: math.exp(p[j + i]) for i in range(4)})
            j += 4
        else:
            q.update({i: 1.0 for i in ipgs})
        return rho0, beta, lam, q

    def t_of(p, m, ipg, n, k, cores, groups=None, row_blocks=1):
        rho0, beta, lam, q = unpack(p)
        g = groups if groups is not None else math.ceil(m / ipg)
        tn = math.ceil(n / 8)
        x = g * tn / cores
        work = g * weight_bytes(n, k) + beta * row_blocks * m * k * tn
        return work * rho0 * q[ipg] * (1.0 + lam[ipg] / x)

    n_free = 2 + (5 if free_lam else 1) + (4 if free_q else 0)
    return unpack, t_of, n_free


def fit_occupancy(cells, cores, name, free_q=True, free_lam=False):
    unpack, t_of, n_free = make_occupancy(free_q, free_lam)
    obs = [c["t"] for c in cells]

    def pred(p):
        return [t_of(p, c["m"], c["ipg"], c["n"], c["k"], cores) for c in cells]

    x0 = [math.log(2.0e-12), math.log(0.7)]
    x0 += [math.log(9.0)] * 5 if free_lam else [math.log(0.3)]
    x0 += [0.0] * (4 if free_q else 0)
    best, _ = nelder_mead(lambda p: rel_rms(pred(p), obs)[0], x0, step=0.3, iters=40000)
    best, _ = nelder_mead(lambda p: rel_rms(pred(p), obs)[0], best, step=0.03, iters=40000)
    rms, res = rel_rms(pred(best), obs)
    rho0, beta, lam, q = unpack(best)
    return dict(name=name, nparam=n_free, rms=rms, res=res, rho0=rho0, beta=beta,
                lam=lam, q=q, params=best, cells=cells,
                t=lambda m, ipg, n, k, cores=cores, groups=None, row_blocks=1:
                    t_of(best, m, ipg, n, k, cores, groups, row_blocks))


# ------------------------------------------------------------ controls


def round_cost(tfun, m, ipg, cores, shapes=SCORED_SHAPES):
    return sum(calls * tfun(m, ipg, n, k, cores=cores) for _, n, k, calls in shapes)


def argmin_table(tfun, cores):
    out = {}
    for m in range(3, 10):
        cand = [(round_cost(tfun, m, i, cores), i) for i in legal_ipg(m)]
        cand.sort()
        out[m] = dict(best=cand[0][1],
                      margin_pct=100 * (cand[1][0] - cand[0][0]) / cand[0][0] if len(cand) > 1 else None,
                      ranking=[(i, c) for c, i in cand])
    return out


def measured_argmin(cells):
    """Round-weighted argmin straight from the measured cells, no model."""
    by = {(c["shape"], c["m"], c["ipg"]): c["t"] for c in cells}
    out = {}
    for m in range(3, 10):
        tot = []
        for ipg in legal_ipg(m):
            s = 0.0
            for name, n, k, calls in SCORED_SHAPES:
                key = (name, m, ipg)
                if key not in by:  # fa.o_proj shares the gdn.out_proj cell
                    key = ("linear_attn.out_proj", m, ipg)
                s += calls * by[key]
            tot.append((s, ipg))
        tot.sort()
        out[m] = dict(best=tot[0][1],
                      margin_pct=100 * (tot[1][0] - tot[0][0]) / tot[0][0] if len(tot) > 1 else None,
                      ranking=[(i, c) for c, i in tot])
    return out


def kendall_tau(xs, ys):
    c = d = 0
    for i, j in itertools.combinations(range(len(xs)), 2):
        a, b = (xs[i] - xs[j]), (ys[i] - ys[j])
        if a * b > 0:
            c += 1
        elif a * b < 0:
            d += 1
    return (c - d) / (c + d) if c + d else 0.0, c, d


def control_e33(tfun, cores):
    """Out-of-sample: predict E33's eight per-shape ratios with zero new params.

    base = <T,6,3>: groups 2, one row block.
    arm  = <T,6,6>: groups 1, two sequential row blocks (x re-read per block).
    """
    rows = []
    for name, n, k, obs in E33:
        tn = math.ceil(n / 8)
        base = tfun(6, 3, n, k, cores=cores, groups=2, row_blocks=1)
        arm = tfun(6, 6, n, k, cores=cores, groups=1, row_blocks=2)
        rows.append(dict(shape=name, n=n, k=k, base_tgs=2 * tn, arm_tgs=tn,
                         obs=obs, pred=arm / base))
    tau, c, d = kendall_tau([r["base_tgs"] for r in rows], [r["pred"] for r in rows])
    obs_tau, oc, od = kendall_tau([r["base_tgs"] for r in rows], [r["obs"] for r in rows])
    return rows, dict(pred_tau=tau, pred_conc=c, pred_disc=d,
                      obs_tau=obs_tau, obs_conc=oc, obs_disc=od)


def measured_e33_contrast(cells):
    """The same contrast without E33's row-block confound, straight from rung 1."""
    by = {(c["shape"], c["m"], c["ipg"]): c for c in cells}
    rows = []
    for name in sorted({c["shape"] for c in cells}):
        b, a = by.get((name, 6, 3)), by.get((name, 6, 6))
        if not b or not a:
            continue
        rows.append(dict(shape=name, n=b["n"], k=b["k"], base_tgs=2 * b["tn"],
                         ratio=a["t"] / b["t"]))
    rows.sort(key=lambda r: r["base_tgs"])
    tau, c, d = kendall_tau([r["base_tgs"] for r in rows], [r["ratio"] for r in rows])
    return rows, dict(tau=tau, conc=c, disc=d)


def model_unconfounded_e33(tfun, cores, cells):
    """Zero-parameter model prediction of the contrast rung 1 actually measured."""
    by = {(c["shape"], c["m"], c["ipg"]): c for c in cells}
    rows = []
    for name in sorted({c["shape"] for c in cells}):
        b, a = by.get((name, 6, 3)), by.get((name, 6, 6))
        if not b or not a:
            continue
        pb = tfun(6, 3, b["n"], b["k"], cores=cores)
        pa = tfun(6, 6, b["n"], b["k"], cores=cores)
        rows.append(dict(shape=name, base_tgs=2 * b["tn"], obs=a["t"] / b["t"], pred=pa / pb))
    rows.sort(key=lambda r: r["base_tgs"])
    tau, c, d = kendall_tau([r["base_tgs"] for r in rows], [r["pred"] for r in rows])
    return rows, dict(tau=tau, conc=c, disc=d)


def critical_cores(tfun, target, cores_lo=20, cores_hi=100000):
    """Smallest core count at which `target` IPG becomes the round-weighted argmin."""
    out = {}
    for m, want in target.items():
        hit = None
        c = cores_lo
        while c <= cores_hi:
            cand = sorted((round_cost(tfun, m, i, c), i) for i in legal_ipg(m))
            if cand[0][1] == want:
                hit = c
                break
            c = int(c * 1.05) + 1
        out[m] = hit
    return out


def control_e71(tfun, cores):
    """Relative in-situ ms/GB profile at M=6 on the shipped partition."""
    rows = []
    for name, ms_gb in E71_MS_PER_GB.items():
        n, k = next((nn, kk) for nm, nn, kk, _ in SCORED_SHAPES if nm == name)
        t = tfun(6, 6, n, k, cores=cores)
        rows.append(dict(shape=name, n=n, k=k, tgs=math.ceil(n / 8),
                         obs_ms_gb=ms_gb, pred_per_gb=t / (weight_bytes(n, k) / 1e9)))
    ref = next(r for r in rows if r["shape"] == "head.lm_head")
    for r in rows:
        r["obs_rel"] = r["obs_ms_gb"] / ref["obs_ms_gb"]
        r["pred_rel"] = r["pred_per_gb"] / ref["pred_per_gb"]
    rows.sort(key=lambda r: -r["tgs"])
    tau, c, d = kendall_tau([r["obs_rel"] for r in rows], [r["pred_rel"] for r in rows])
    return rows, dict(tau=tau, conc=c, disc=d)


# --------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="research/e73-artifacts/rung1-s1.json")
    ap.add_argument("--cores", type=int, required=True, help="local GPU cores, read from device")
    ap.add_argument("--ranked-cores", type=int, required=True)
    ap.add_argument("--out", default="research/e73-artifacts/rung2.json")
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()

    d, cells = load(args.session)
    null = session_null(cells)
    print(f"session: {d['device']}  cells {len(cells)}  reps {d['reps']}  "
          f"entry {d['session_entry_gpu_temp_c']:.1f}C exit {d['session_exit_gpu_temp_c']:.1f}C")
    print(f"session null: median {null['median']*100:.4f}%  p90 {null['p90']*100:.4f}%  "
          f"max {null['max']*100:.4f}%  (n={null['n']})\n")

    models = [fit_additive(cells),
              fit_occupancy(cells, args.cores, "B0 occ, S-pinned, q=1", False, False),
              fit_occupancy(cells, args.cores, "B  occ, S-pinned", True, False),
              fit_occupancy(cells, args.cores, "B2 occ, lam(IPG) free", True, True)]

    print("== fits (114 cells) ==")
    for mo in models:
        ar = [abs(x) for x in mo["res"]]
        print(f"{mo['name']:24s} params {mo['nparam']:2d}  rel-rms {mo['rms']*100:6.2f}%  "
              f"median {statistics.median(ar)*100:5.2f}%  max {max(ar)*100:6.2f}%  "
              f"= {mo['rms']/null['median']:7.1f} x session null")
    B = models[-1]
    print(f"\n{B['name']} fitted: rho0 {B['rho0']*1e12:.4f} ps/byte  beta {B['beta']:.4f}")
    print("  q(IPG)                      " + " ".join(f"{i}:{v:7.4f}" for i, v in sorted(B["q"].items())))
    print("  saturated rate 1/(rho0*q)   " + " ".join(
        f"{i}:{1/(B['rho0']*v)/1e9:6.1f}" for i, v in sorted(B["q"].items())) + "  GB/s")
    print("  lam(IPG) = working TGs/core that doubles the per-byte rate")
    print("                              " + " ".join(f"{i}:{v:7.2f}" for i, v in sorted(B["lam"].items())))
    print("  rung-0 S(IPG) for contrast  " + " ".join(f"{i}:{SG_REG[i]:7d}" for i in sorted(SG_REG)))
    for mo in models[1:]:
        by_ipg = {}
        for c, r in zip(cells, mo["res"]):
            by_ipg.setdefault(c["ipg"], []).append(abs(r))
        print(f"  {mo['name']:24s} |res| by IPG  " +
              " ".join(f"{i}:{statistics.median(v)*100:5.2f}%" for i, v in sorted(by_ipg.items())))

    print("\n== control 1: reproduce the measured local optimum (round-weighted, cores="
          f"{args.cores}) ==")
    meas = measured_argmin(cells)
    tabs = [argmin_table(mo["t"], args.cores) for mo in models]
    print("  M  shipped  measured(margin)   " + " ".join(f"{mo['name'][:9]:>10s}" for mo in models))
    for m in range(3, 10):
        marg = f"({meas[m]['margin_pct']:.2f}%)" if meas[m]["margin_pct"] is not None else "(-)"
        print(f"  {m}     {SHIPPED_LOCAL[m]}       {meas[m]['best']} {marg:9s}   " +
              " ".join(f"{t[m]['best']:10d}" for t in tabs))
    ok = {mo["name"]: all(t[m]["best"] == SHIPPED_LOCAL[m] for m in range(3, 10))
          for mo, t in zip(models, tabs)}
    meas_ok = all(meas[m]["best"] == SHIPPED_LOCAL[m] for m in range(3, 10))
    print(f"  measured reproduces shipped table: {meas_ok}")
    for k_, v in ok.items():
        print(f"  {k_:22s} reproduces shipped table: {v}")

    print("\n== control 2: the E33 sign flip (out-of-sample, zero new parameters) ==")
    mrows, mstat = measured_e33_contrast(cells)
    print("  2a measured, unconfounded <T,6,3> -> <T,6,6> from rung 1:")
    print("     shape                              base TGs   ratio")
    for r in mrows:
        print(f"     {r['shape'][:32]:32s} {r['base_tgs']:9d}  {r['ratio']:6.4f}")
    print(f"     Kendall tau vs base TGs {mstat['tau']:+.3f} "
          f"({mstat['conc']} conc / {mstat['disc']} disc); no row-block confound")
    e33rows, e33stat = control_e33(B["t"], args.cores)
    print("  2b model prediction of E33's confounded arm (groups 2->1, row blocks 1->2):")
    print("     shape                              base TGs    obs    pred    err")
    for r in sorted(e33rows, key=lambda r: -r["base_tgs"]):
        print(f"     {r['shape'][:32]:32s} {r['base_tgs']:9d}  {r['obs']:.4f}  "
              f"{r['pred']:.4f}  {100*(r['pred']-r['obs'])/r['obs']:+6.2f}%")
    print(f"     obs tau {e33stat['obs_tau']:+.3f}   pred tau {e33stat['pred_tau']:+.3f} "
          f"({e33stat['pred_conc']} conc / {e33stat['pred_disc']} disc)")
    lo = [r["base_tgs"] for r in e33rows if r["pred"] > 1.0]
    hi = [r["base_tgs"] for r in e33rows if r["pred"] <= 1.0]
    print(f"     predicted loss (>1) at or below {max(lo, default=0)} base TGs; "
          f"predicted win (<=1) at or above {min(hi, default=0)} base TGs")
    urows, ustat = model_unconfounded_e33(B["t"], args.cores, cells)
    print("  2c model prediction of the SAME contrast rung 1 measured (row blocks 1->1):")
    print("     shape                              base TGs    obs    pred    err")
    for r in sorted(urows, key=lambda r: -r["base_tgs"]):
        print(f"     {r['shape'][:32]:32s} {r['base_tgs']:9d}  {r['obs']:.4f}  "
              f"{r['pred']:.4f}  {100*(r['pred']-r['obs'])/r['obs']:+6.2f}%")
    print(f"     pred tau vs base TGs {ustat['tau']:+.3f} "
          f"({ustat['conc']} conc / {ustat['disc']} disc)")

    print("\n== cross-check: askeladd E71 in-situ ms/GB profile at M=6 ==")
    e71rows, e71stat = control_e71(B["t"], args.cores)
    print("     shape                          working TGs  obs rel  pred rel")
    for r in e71rows:
        print(f"     {r['shape'][:28]:28s} {r['tgs']:11d}  {r['obs_rel']:7.3f}  {r['pred_rel']:8.3f}")
    print(f"     Kendall tau obs vs pred {e71stat['tau']:+.3f}")

    print(f"\n== ranked prediction (cores {args.cores} -> {args.ranked_cores}) ==")
    rk = argmin_table(B["t"], args.ranked_cores)
    print("  M  legal        local  ranked  crown  margin over 2nd (%)")
    for m in range(3, 10):
        mg = rk[m]["margin_pct"]
        print(f"  {m}  {str(legal_ipg(m)):13s} {tabs[-1][m]['best']:5d}  {rk[m]['best']:6d}  "
              f"{CROWN[m]:5d}  {'-' if mg is None else round(mg, 2)}")
    agree = sum(1 for m in range(3, 10) if rk[m]["best"] == CROWN[m])
    print(f"  agreement with crown table: {agree}/7")

    print("\n== how much occupancy pressure would the crown table need? ==")
    crit = critical_cores(B["t"], CROWN)
    print("  argmin over IPG is invariant to any pure rescaling of the per-byte rate,")
    print("  so a bandwidth-headroom difference alone cannot move a row. Only `cores` can.")
    print("  M  crown  smallest core count that makes crown optimal  (ranked M5 Max = "
          f"{args.ranked_cores})")
    for m in range(3, 10):
        c = crit[m]
        print(f"  {m}    {CROWN[m]}     " + ("already optimal at 20" if c == args.cores else
              (f"{c}" if c else "never, up to 100000 cores")))

    print("\n== two-dimensional deliverable: optimal IPG by (M, out_vec_size band) ==")
    bands = [("n=5120  k=6144  gdn.out_proj / fa.o_proj", 5120, 6144),
             ("n=5120  k=17408 mlp.down", 5120, 17408),
             ("n=14336 k=5120  fa.qkv", 14336, 5120),
             ("n=16480 k=5120  gdn.in_proj", 16480, 5120),
             ("n=34816 k=5120  mlp.gate_up", 34816, 5120),
             ("n=248320 k=5120 lm_head", 248320, 5120)]
    per_band = {}
    for label, n, k in bands:
        loc, ran = [], []
        for m in range(3, 10):
            cl = sorted((B["t"](m, i, n, k, cores=args.cores), i) for i in legal_ipg(m))
            cr = sorted((B["t"](m, i, n, k, cores=args.ranked_cores), i) for i in legal_ipg(m))
            loc.append(cl[0][1])
            ran.append(cr[0][1])
        per_band[label] = dict(local=loc, ranked=ran, n=n, k=k)
        print(f"  {label:38s} local {loc}  ranked {ran}")
    print("  (M = 3..9 left to right)")

    payload = dict(
        harness="local", experiment="e73", rung=2, device=d["device"],
        cores_local=args.cores, cores_ranked=args.ranked_cores,
        session=args.session, session_null=null,
        models=[{k_: v for k_, v in mo.items() if k_ not in ("t", "cells")} for mo in models],
        control1=dict(shipped=SHIPPED_LOCAL, measured=meas,
                      measured_ok=meas_ok, model_ok=ok,
                      model_tables={mo["name"]: t for mo, t in zip(models, tabs)}),
        control2=dict(measured_unconfounded=mrows, measured_stat=mstat,
                      model_e33=e33rows, model_stat=e33stat),
        crosscheck_e71=dict(rows=e71rows, stat=e71stat),
        ranked_prediction=rk, crown=CROWN, per_band=per_band,
        critical_cores_for_crown=crit,
        model_unconfounded_e33=dict(rows=urows, stat=ustat),
    )
    json.dump(payload, open(args.out, "w"), indent=1, default=str)
    print(f"\nwrote {args.out}")

    if args.wandb_name:
        log_wandb(args, payload, models, cells, null)


def log_wandb(args, payload, models, cells, null):
    import wandb

    B = models[-1]
    run = wandb.init(entity="wandb-applied-ai-team", project="qwen38-mlx-challenge-senpai",
                     name=args.wandb_name, group="e73-rung2", job_type="e73-fit",
                     tags=["e73", "qmv-crossrow", "ipg", "rung2", "cost-model"],
                     config=dict(harness="local", device=payload["device"],
                                 cores_local=args.cores, cores_ranked=args.ranked_cores,
                                 rung1_session=args.session, sg_reg_rung0=SG_REG,
                                 session_null_median=null["median"]))
    for mo, resids in ((m, m["res"]) for m in models):
        ar = [abs(x) for x in resids]
        run.summary[f"fit/{mo['name']}/rel_rms"] = mo["rms"]
        run.summary[f"fit/{mo['name']}/rel_median"] = statistics.median(ar)
        run.summary[f"fit/{mo['name']}/rel_max"] = max(ar)
        run.summary[f"fit/{mo['name']}/nparam"] = mo["nparam"]
        run.summary[f"fit/{mo['name']}/x_session_null"] = mo["rms"] / null["median"]
    run.summary["best/rho0_s_per_byte"] = B["rho0"]
    run.summary["best/beta"] = B["beta"]
    for i in sorted(B["q"]):
        run.summary[f"best/q_ipg{i}"] = B["q"][i]
        run.summary[f"best/lam_ipg{i}"] = B["lam"][i]
        run.summary[f"best/knee10pct_tgs_per_core_ipg{i}"] = B["lam"][i] / 0.1
    run.summary["control1/measured_reproduces_shipped"] = payload["control1"]["measured_ok"]
    for name, v in payload["control1"]["model_ok"].items():
        run.summary[f"control1/{name}/reproduces_shipped"] = v
    run.summary["control2/measured_unconfounded_tau"] = payload["control2"]["measured_stat"]["tau"]
    run.summary["control2/model_e33_tau"] = payload["control2"]["model_stat"]["pred_tau"]
    run.summary["control2/model_e33_max_abs_err"] = max(
        abs(r["pred"] - r["obs"]) / r["obs"] for r in payload["control2"]["model_e33"])
    run.summary["crosscheck/e71_tau"] = payload["crosscheck_e71"]["stat"]["tau"]
    run.summary["session/null_median"] = null["median"]
    run.summary["session/null_p90"] = null["p90"]

    cell_tab = wandb.Table(columns=["shape", "n", "k", "M", "IPG", "groups", "working_tgs",
                                    "tgs_per_core", "t_s", "pred_s", "rel_res"])
    for c, r in zip(cells, B["res"]):
        cell_tab.add_data(c["shape"], c["n"], c["k"], c["m"], c["ipg"], c["groups"],
                          c["groups"] * c["tn"], c["groups"] * c["tn"] / args.cores,
                          c["t"], c["t"] * (1 + r), r)
    pred_tab = wandb.Table(columns=["M", "legal_ipg", "shipped", "measured", "model_local",
                                    "model_ranked", "crown", "critical_cores_for_crown"])
    for m in range(3, 10):
        pred_tab.add_data(m, str(legal_ipg(m)), SHIPPED_LOCAL[m],
                          payload["control1"]["measured"][m]["best"],
                          payload["control1"]["model_tables"][B["name"]][m]["best"],
                          payload["ranked_prediction"][m]["best"], CROWN[m],
                          str(payload["critical_cores_for_crown"][m]))
    run.log({"cells": cell_tab, "prediction_table": pred_tab})
    print(f"wandb run_id={run.id} url={run.url}")
    run.finish()


if __name__ == "__main__":
    main()
