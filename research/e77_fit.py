#!/usr/bin/env python3
"""E77 rung 2 and rung 3 - measure the occupancy coefficient, refit the E73
surface with it fixed, then predict the ranked table.

Rung 1 measures `time(R)` at ONE cell, ONE shape family, fixed traffic, fixed
group count and fixed grid, so the only thing that moves is the per-thread
register count `R`. Resident simdgroups per core are `S_h(R) = floor(B_h/128R)`,
so the sweep measures the occupancy factor directly:

  Omega_hat(S) = t(S) / t(S_ref)      at fixed everything else

Rung 2 puts that fixed factor into the E73 form and refits the rest:

  t = [ groups*W + beta*M*k*Tn ] * rho0 * c(IPG) * Omega(S_h(R_h(IPG)))
                                        * (1 + lam(IPG)/x_h)
  x_h = groups*Tn/cores_h

Rung 0 showed `R` is a function of IPG alone on both hosts, so
`Omega(S_L(R_L(IPG)))` is perfectly collinear with E73's free level `q(IPG)`.
The refit therefore returns `c(IPG) = q(IPG)/Omega_L(IPG)` and the SAME
residuals. That is a stated consequence, not a failure; the sweep, not the
surface, identifies the coefficient. The ranked prediction is where the term
does work, because it re-evaluates `Omega` at the ranked register counts.

  python3 research/e77_fit.py --cores 20 --ranked-cores 40
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e73_fit import (CROWN, E33, SCORED_SHAPES, SHIPPED_LOCAL,  # noqa: E402
                     control_e33, kendall_tau, legal_ipg, nelder_mead,
                     rel_rms, weight_bytes)
from e73_fit import load as load_e73  # noqa: E402
from e73_fit import session_null as e73_session_null  # noqa: E402

LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"

# Register-file bytes per core. The local value is the rung-0 hypothesis the
# sweep tests. The ranked value is an EXTRAPOLATION from the 124-register
# ranked allocator ceiling: 124 registers * 32 lanes * 4 bytes * 32 simdgroups.
LOCAL_FILE_BYTES = 384 * 1024
RANKED_FILE_BYTES = 124 * 128 * 32

# Ranked QMV time share by proposal width, campaign ledger 200(B).
RANKED_WIDTH_SHARE = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334, 7: 0.122,
                      8: 0.0735, 9: 0.0575}
# QMV share of the ranked candidate leg, campaign ledger 200(B).
QMV_SHARE_OF_RANKED_LEG = 0.826
# Crown table minus our table, ranked candidate seconds/token, scoring prompts,
# after subtracting the plutarch drift floor. PR #80 baseline table.
MEASURED_SCORING_DELTA_PCT = -0.298


def sg_per_core(regs: int, file_bytes: int) -> int:
    return max(1, file_bytes // (128 * regs))


# ---------------------------------------------------------------- rung 1 data


def load_sweep(path: str, regs_path: str):
    d = json.loads(pathlib.Path(path).read_text())
    census = json.loads(pathlib.Path(regs_path).read_text())
    local = {k.removeprefix("e77_"): v
             for k, v in census["sweep"][LOCAL_ARCH].items()}
    rows = []
    for sh in d["shapes"]:
        by_arm = {}
        for leg in sh["legs"]:
            by_arm.setdefault(leg["arm"], []).append(leg)
        for arm, legs in by_arm.items():
            pos = {}
            for leg in legs:
                pos.setdefault(leg["position"], []).append(
                    leg["seconds_per_dispatch"])
            oracle = local[arm]
            rows.append(dict(
                shape=sh["shape"], n=sh["n"], k=sh["k"], arm=arm,
                m=legs[0]["m"], ipg=legs[0]["ipg"], groups=legs[0]["groups"],
                pressure=legs[0]["pressure"], kind=legs[0]["kind"],
                regs=oracle["registers"], frame_bytes=oracle["spill_bytes"],
                t=statistics.median(l["seconds_per_dispatch"] for l in legs),
                nlegs=len(legs),
                pos=sorted(statistics.median(v) for v in pos.values()),
            ))
    return d, rows


def sweep_null(rows):
    v = sorted(abs(r["pos"][1] - r["pos"][0]) / r["pos"][0]
               for r in rows if len(r["pos"]) == 2)
    return dict(n=len(v), median=statistics.median(v),
                p90=v[int(0.9 * len(v))], max=v[-1])


def curves(rows, file_bytes):
    """Per (shape, carrier) normalised response, keyed by register count."""
    out = {}
    for r in rows:
        out.setdefault((r["shape"], r["m"], r["ipg"]), []).append(r)
    for key, group in out.items():
        base = next(r for r in group if r["kind"] == "p" and r["pressure"] == 0)
        for r in group:
            r["ratio"] = r["t"] / base["t"]
            r["sg"] = sg_per_core(r["regs"], file_bytes)
            r["sg_base"] = sg_per_core(base["regs"], file_bytes)
        group.sort(key=lambda r: (r["kind"], r["pressure"]))
    return out


def fit_gamma(rows, file_bytes, shapes=None):
    """Omega(S) = (S_ref/S)^gamma on the spill-free arms, pooled over shapes.

    Least squares on log ratio against log(S_base/S), no intercept, so the
    unpadded arm anchors the curve at 1 by construction.
    """
    pts = [(math.log(r["sg_base"] / r["sg"]), math.log(r["ratio"]))
           for r in rows
           if r["kind"] == "p" and r["frame_bytes"] == 0 and r["sg"] != r["sg_base"]
           and (shapes is None or r["shape"] in shapes)]
    if not pts:
        return None
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    gamma = sxy / sxx
    res = [y - gamma * x for x, y in pts]
    return dict(gamma=gamma, n=len(pts),
                rms=math.sqrt(sum(e * e for e in res) / len(res)),
                max_abs=max(abs(e) for e in res))


def omega_table(rows, file_bytes):
    """Non-parametric Omega on measured S, pooled over shapes (spill-free)."""
    by_sg = {}
    for r in rows:
        if r["kind"] != "p" or r["frame_bytes"]:
            continue
        by_sg.setdefault(r["sg"], []).append(r["ratio"] * r["sg_base"] and r)
    out = {}
    for sg, group in by_sg.items():
        # Each arm's ratio is against its own carrier's unpadded arm, whose
        # occupancy is `sg_base`; renormalise every point onto one anchor.
        out[sg] = statistics.median(r["ratio"] for r in group)
    return out


# ------------------------------------------------------------------- rung 2


def make_model(omega_local, omega_ranked):
    ipgs = [2, 3, 4, 5, 6]
    rest = [i for i in ipgs if i != 3]

    def unpack(p):
        rho0, beta = math.exp(p[0]), math.exp(p[1])
        lam = {ipgs[i]: math.exp(p[2 + i]) for i in range(5)}
        c = {3: 1.0}
        c.update({rest[i]: math.exp(p[7 + i]) for i in range(4)})
        return rho0, beta, lam, c

    def t_of(p, m, ipg, n, k, cores, groups=None, row_blocks=1, ranked=False):
        rho0, beta, lam, c = unpack(p)
        g = groups if groups is not None else math.ceil(m / ipg)
        tn = math.ceil(n / 8)
        x = g * tn / cores
        work = g * weight_bytes(n, k) + beta * row_blocks * m * k * tn
        omega = (omega_ranked if ranked else omega_local)[ipg]
        return work * rho0 * c[ipg] * omega * (1.0 + lam[ipg] / x)

    return unpack, t_of


def fit_surface(cells, cores, omega_local, omega_ranked):
    unpack, t_of = make_model(omega_local, omega_ranked)
    obs = [c["t"] for c in cells]

    def pred(p):
        return [t_of(p, c["m"], c["ipg"], c["n"], c["k"], cores) for c in cells]

    x0 = [math.log(2.0e-12), math.log(0.7)] + [math.log(9.0)] * 5 + [0.0] * 4
    best, _ = nelder_mead(lambda p: rel_rms(pred(p), obs)[0], x0, step=0.3,
                          iters=40000)
    best, _ = nelder_mead(lambda p: rel_rms(pred(p), obs)[0], best, step=0.03,
                          iters=40000)
    rms, res = rel_rms(pred(best), obs)
    rho0, beta, lam, c = unpack(best)
    return dict(rms=rms, res=res, rho0=rho0, beta=beta, lam=lam, c=c,
                params=best,
                t_local=lambda m, ipg, n, k, cores=cores, groups=None,
                row_blocks=1: t_of(best, m, ipg, n, k, cores, groups,
                                   row_blocks, False),
                t_ranked=lambda m, ipg, n, k, cores, groups=None,
                row_blocks=1: t_of(best, m, ipg, n, k, cores, groups,
                                   row_blocks, True))


def round_cost(tfun, m, ipg, cores):
    return sum(calls * tfun(m, ipg, n, k, cores=cores)
               for _, n, k, calls in SCORED_SHAPES)


def argmin_table(tfun, cores):
    out = {}
    for m in range(3, 10):
        cand = sorted((round_cost(tfun, m, i, cores), i) for i in legal_ipg(m))
        out[m] = dict(best=cand[0][1],
                      margin_pct=(100 * (cand[1][0] - cand[0][0]) / cand[0][0]
                                  if len(cand) > 1 else None),
                      ranking=[(i, c) for c, i in cand])
    return out


def table_delta(tfun, cores, table_a, table_b):
    """Per-M and share-weighted cost of table_a relative to table_b."""
    per_m, weighted = {}, 0.0
    for m in range(3, 10):
        ca = round_cost(tfun, m, table_a[m], cores)
        cb = round_cost(tfun, m, table_b[m], cores)
        d = 100.0 * (ca - cb) / cb
        per_m[m] = d
        weighted += RANKED_WIDTH_SHARE[m] * d
    return per_m, weighted


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="research/e77-artifacts/rung1-s1.json")
    ap.add_argument("--regs", default="research/e77-artifacts/rung0-regs.json")
    ap.add_argument("--session", default="research/e73-artifacts/rung1-s1.json")
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--ranked-cores", type=int, required=True)
    ap.add_argument("--out", default="research/e77-artifacts/rung2.json")
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()

    census = json.loads(pathlib.Path(args.regs).read_text())
    cell_regs = {arch: {int(k.split("ipg")[1]): v["registers"]
                        for k, v in census["cells"][arch].items()
                        if k.startswith("e77_cell_m6_")}
                 for arch in (LOCAL_ARCH, RANKED_ARCH)}
    sg_local = {i: sg_per_core(cell_regs[LOCAL_ARCH][i], LOCAL_FILE_BYTES)
                for i in range(2, 7)}
    sg_ranked = {i: sg_per_core(cell_regs[RANKED_ARCH][i], RANKED_FILE_BYTES)
                 for i in range(2, 7)}

    d, rows = load_sweep(args.sweep, args.regs)
    null = sweep_null(rows)
    cur = curves(rows, LOCAL_FILE_BYTES)

    print(f"sweep: {d['device']}  arms {len(d['arms'])}  reps {d['reps']}  "
          f"entry {d['session_entry_gpu_temp_c']:.1f}C "
          f"exit {d['session_exit_gpu_temp_c']:.1f}C")
    print(f"gate: real_gate={d.get('cool_gate_passed_real_gate')} "
          f"qualified={d.get('gate_qualified_for_timing')}")
    print(f"sweep null: median {null['median']*100:.4f}%  "
          f"p90 {null['p90']*100:.4f}%  max {null['max']*100:.4f}%  "
          f"(n={null['n']})\n")

    print("== rung 1: time(R) at fixed traffic, group count and grid ==")
    for (shape, m, ipg), group in sorted(cur.items()):
        clean = [r for r in group if r["kind"] == "p" and not r["frame_bytes"]]
        spill = [r for r in group if r["kind"] == "p" and r["frame_bytes"]]
        ctrl = [r for r in group if r["kind"] == "q"]
        clean.sort(key=lambda r: r["regs"])
        print(f"  {shape}  m{m} ipg{ipg}  spill-free R "
              f"{clean[0]['regs']}..{clean[-1]['regs']}  "
              f"S {clean[0]['sg']}..{clean[-1]['sg']}")
        print("     R    S   P   t/t(p0)   dev vs null")
        for r in clean:
            print(f"    {r['regs']:3d}  {r['sg']:3d} {r['pressure']:3d}  "
                  f"{r['ratio']:8.5f}   "
                  f"{(r['ratio']-1)*100/ (null['median']*100):+7.2f}x")
        for r in spill:
            print(f"    {r['regs']:3d}  {r['sg']:3d} {r['pressure']:3d}  "
                  f"{r['ratio']:8.5f}   frame {r['frame_bytes']} B")
        for r in ctrl:
            base = next(x for x in group
                        if x["kind"] == "p" and x["pressure"] == r["pressure"])
            print(f"    control q{r['pressure']:<3d} ratio {r['ratio']:8.5f}  "
                  f"vs p{r['pressure']} {100*(r['ratio']/base['ratio']-1):+6.3f}%")

    fits = {"pooled": fit_gamma(rows, LOCAL_FILE_BYTES)}
    for shape in sorted({r["shape"] for r in rows}):
        fits[shape] = fit_gamma(rows, LOCAL_FILE_BYTES, shapes={shape})
    print("\n== occupancy exponent  Omega(S) = (S_ref/S)^gamma ==")
    for name, f in fits.items():
        if f:
            print(f"  {name:26s} gamma {f['gamma']:+8.5f}  n {f['n']:3d}  "
                  f"log-rms {f['rms']:.5f}  max {f['max_abs']:.5f}")
    gamma = fits["pooled"]["gamma"]

    def omega(sg):
        return (32.0 / sg) ** gamma

    omega_local = {i: omega(sg_local[i]) for i in range(2, 7)}
    omega_ranked = {i: omega(sg_ranked[i]) for i in range(2, 7)}
    measured_sg = sorted({r["sg"] for r in rows
                          if r["kind"] == "p" and not r["frame_bytes"]})
    print(f"  measured S range {measured_sg[0]}..{measured_sg[-1]}")
    print("  IPG  local R/S  Omega_L   ranked R/S  Omega_R   extrapolated")
    for i in range(2, 7):
        ex = not (measured_sg[0] <= sg_ranked[i] <= measured_sg[-1])
        print(f"   {i}   {cell_regs[LOCAL_ARCH][i]:3d}/{sg_local[i]:<3d} "
              f"{omega_local[i]:8.5f}   {cell_regs[RANKED_ARCH][i]:3d}/"
              f"{sg_ranked[i]:<3d} {omega_ranked[i]:8.5f}   {ex}")

    _, cells = load_e73(args.session)
    e73null = e73_session_null(cells)
    fit = fit_surface(cells, args.cores, omega_local, omega_ranked)
    ar = [abs(x) for x in fit["res"]]
    print(f"\n== rung 2 refit on the {len(cells)}-cell E73 surface "
          f"(Omega fixed from rung 1) ==")
    print(f"  rel-rms {fit['rms']*100:.2f}%  median {statistics.median(ar)*100:.2f}%"
          f"  max {max(ar)*100:.2f}%  = {fit['rms']/e73null['median']:.1f}x "
          f"E73 session null")
    print(f"  rho0 {fit['rho0']*1e12:.4f} ps/byte   beta {fit['beta']:.4f}")
    print("  c(IPG)        " + " ".join(f"{i}:{v:7.4f}"
                                        for i, v in sorted(fit["c"].items())))
    print("  c*Omega_L     " + " ".join(
        f"{i}:{fit['c'][i]*omega_local[i]:7.4f}" for i in range(2, 7)))
    print("  lam(IPG)      " + " ".join(f"{i}:{v:7.2f}"
                                        for i, v in sorted(fit["lam"].items())))

    print(f"\n== validation 1: local table at cores={args.cores} ==")
    tab = argmin_table(fit["t_local"], args.cores)
    print("  M  shipped  model  margin over 2nd (%)")
    for m in range(3, 10):
        mg = tab[m]["margin_pct"]
        print(f"  {m}     {SHIPPED_LOCAL[m]}       {tab[m]['best']}    "
              f"{'-' if mg is None else round(mg, 2)}")
    v1 = all(tab[m]["best"] == SHIPPED_LOCAL[m] for m in range(3, 10))
    print(f"  reproduces the shipped local table 7/7: {v1}")

    print(f"\n== validation 2: ranked ordering at cores={args.ranked_cores} "
          "(EXTRAPOLATED core count) ==")
    rk = argmin_table(fit["t_ranked"], args.ranked_cores)
    per_m, weighted = table_delta(fit["t_ranked"], args.ranked_cores, CROWN,
                                  SHIPPED_LOCAL)
    print("  M  ours  crown  model ranked argmin  crown-minus-ours QMV (%)  share")
    for m in range(3, 10):
        print(f"  {m}    {SHIPPED_LOCAL[m]}      {CROWN[m]}          "
              f"{rk[m]['best']}              {per_m[m]:+8.3f}          "
              f"{RANKED_WIDTH_SHARE[m]*100:5.2f}%")
    v2_m5 = per_m[5] < 0
    v2_m6 = per_m[6] < 0
    leg = weighted * QMV_SHARE_OF_RANKED_LEG
    scoring = (sum(RANKED_WIDTH_SHARE[m] * per_m[m] for m in (5, 6))
               / sum(RANKED_WIDTH_SHARE[m] for m in (5, 6))
               * QMV_SHARE_OF_RANKED_LEG)
    print(f"  crown faster than ours at M=5: {v2_m5}   at M=6: {v2_m6}")
    print(f"  pool-weighted QMV delta {weighted:+.4f}%  -> candidate leg "
          f"{leg:+.4f}%")
    print(f"  M5/M6-only delta -> candidate leg {scoring:+.4f}%  "
          f"vs measured {MEASURED_SCORING_DELTA_PCT:+.3f}%")

    print("\n== validation 3: control 2, E33 ==")
    e33rows, e33stat = control_e33(fit["t_local"], args.cores)
    print("     shape                              base TGs    obs    pred    err")
    for r in sorted(e33rows, key=lambda r: -r["base_tgs"]):
        print(f"     {r['shape'][:32]:32s} {r['base_tgs']:9d}  {r['obs']:.4f}  "
              f"{r['pred']:.4f}  {100*(r['pred']-r['obs'])/r['obs']:+6.2f}%")
    print(f"     obs tau {e33stat['obs_tau']:+.3f}   pred tau "
          f"{e33stat['pred_tau']:+.3f}")
    e33span = (min(r["pred"] for r in e33rows), max(r["pred"] for r in e33rows))
    print(f"     predicted ratio span {e33span[0]:.4f}..{e33span[1]:.4f} "
          f"(E73 gave 1.5048..1.7394; observed 0.9830..1.0592)")

    result = dict(
        experiment="e77", rung=2, harness="local",
        sweep_device=d["device"], sweep_reps=d["reps"],
        cool_gate_passed_real_gate=d.get("cool_gate_passed_real_gate"),
        gate_qualified_for_timing=d.get("gate_qualified_for_timing"),
        sweep_entry_gpu_temp_c=d["session_entry_gpu_temp_c"],
        sweep_exit_gpu_temp_c=d["session_exit_gpu_temp_c"],
        sweep_null=null, e73_session_null=e73null,
        local_file_bytes=LOCAL_FILE_BYTES,
        ranked_file_bytes_extrapolated=RANKED_FILE_BYTES,
        cores=args.cores, ranked_cores_extrapolated=args.ranked_cores,
        gamma=gamma, gamma_fits=fits,
        measured_sg_range=[measured_sg[0], measured_sg[-1]],
        cell_registers=cell_regs, sg_local=sg_local, sg_ranked=sg_ranked,
        omega_local=omega_local, omega_ranked=omega_ranked,
        points=[{k: v for k, v in r.items() if k != "pos"} for r in rows],
        refit=dict(rms=fit["rms"], rho0=fit["rho0"], beta=fit["beta"],
                   c=fit["c"], lam=fit["lam"]),
        validation1_local_table=v1,
        local_table={m: tab[m]["best"] for m in range(3, 10)},
        ranked_table={m: rk[m]["best"] for m in range(3, 10)},
        ranked_margin_pct={m: rk[m]["margin_pct"] for m in range(3, 10)},
        crown_minus_ours_qmv_pct=per_m,
        crown_minus_ours_leg_pct=leg,
        crown_minus_ours_leg_pct_m5m6=scoring,
        measured_scoring_delta_pct=MEASURED_SCORING_DELTA_PCT,
        validation2_m5=v2_m5, validation2_m6=v2_m6,
        e33_pred_span=e33span,
        e33_rows=e33rows, e33_stat=e33stat,
    )

    if v2_m5 and v2_m6:
        print("\n== rung 3: ranked-optimal table (EXTRAPOLATED core count and "
              "register file) ==")
        print("  M  legal          ours  crown  model  margin (%)  vs crown QMV (%)")
        vs_crown, _ = table_delta(fit["t_ranked"], args.ranked_cores,
                                  {m: rk[m]["best"] for m in range(3, 10)},
                                  CROWN)
        for m in range(3, 10):
            mg = rk[m]["margin_pct"]
            print(f"  {m}  {str(legal_ipg(m)):14s} {SHIPPED_LOCAL[m]}      "
                  f"{CROWN[m]}      {rk[m]['best']}    "
                  f"{'-' if mg is None else round(mg, 2):>8}   {vs_crown[m]:+8.3f}")
        gain = sum(RANKED_WIDTH_SHARE[m] * vs_crown[m] for m in range(3, 10))
        print(f"  model table vs crown, pool-weighted QMV {gain:+.4f}%  -> "
              f"candidate leg {gain*QMV_SHARE_OF_RANKED_LEG:+.4f}%")
        result["rung3_vs_crown_qmv_pct"] = vs_crown
        result["rung3_vs_crown_leg_pct"] = gain * QMV_SHARE_OF_RANKED_LEG
    else:
        print("\n== rung 3 not started: the ranked-ordering validation did not "
              "reproduce the correct sign at both M=5 and M=6 ==")

    pathlib.Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {args.out}")

    if args.wandb_name:
        import wandb
        run = wandb.init(entity="wandb-applied-ai-team",
                         project="qwen38-mlx-challenge-senpai",
                         name=args.wandb_name, group="e77-rung2",
                         job_type="e77-occupancy",
                         tags=["e77", "occupancy", "registers", "rung2"],
                         config=dict(cores=args.cores,
                                     ranked_cores=args.ranked_cores,
                                     sweep=args.sweep, session=args.session))
        run.summary.update({k: v for k, v in result.items()
                            if k not in {"points", "e33_rows"}})
        run.summary["points"] = result["points"]
        print(f"e77_fit: run_id={run.id} url={run.url}")
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
