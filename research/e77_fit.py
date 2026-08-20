#!/usr/bin/env python3
"""E77 rungs 2 and 3 - measure the occupancy coefficient, refit the E73 surface
with it fixed, then predict the ranked table.

Rung 1 carries two arm families and they answer different questions.

The NATURAL family holds M and the group count fixed and moves IPG, so it
measures E73's per-IPG level `q(IPG)` with the weight-traffic term removed. It
cannot separate occupancy from partition shape, because on both hosts the
register count is a function of the largest group alone, so registers and IPG
move together. Its value is that the same cells span 91 to 111 registers on
g17s, where the local host compresses them into 93 to 96.

The SYNTHETIC ladder holds the cell fixed and moves only inert live state, so it
is the ONLY arm family in which the register count varies at fixed IPG. It is
therefore the only source that can identify the occupancy factor:

  Omega_hat(S) = t(S) / t(S_base)      S_h(R) = floor(B_h / (128*R))

Per the advisor's revised constraint, `Omega` is fitted only on spill-free arms
at or below 96 registers. Arms with frame bytes are reported as a separate
spill curve and are excluded from the fit.

Rung 2 puts the fixed factor into the E73 form and refits the rest:

  t = [ groups*W + beta*M*k*Tn ] * rho0 * c(IPG) * Omega(S_h(R_h(IPG)))
                                        * (1 + lam(IPG)/x_h)
  x_h = groups*Tn/cores_h        c(IPG) = q(IPG) / Omega(S_L(R_L(IPG)))

  python3 research/e77_fit.py --cores 20 --ranked-cores 40
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e73_fit import (CROWN, SCORED_SHAPES, SHIPPED_LOCAL,  # noqa: E402
                     control_e33, legal_ipg, nelder_mead, rel_rms,
                     weight_bytes)
from e73_fit import load as load_e73  # noqa: E402
from e73_fit import session_null as e73_session_null  # noqa: E402
from e77_probe import arms as probe_arms  # noqa: E402

LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"

# Register-file bytes per core. The local value is the rung-0 hypothesis the
# sweep tests. The ranked value is an EXTRAPOLATION from the 124-register
# ranked allocator ceiling: 124 registers * 32 lanes * 4 bytes * 32 simdgroups.
LOCAL_FILE_BYTES = 384 * 1024
RANKED_FILE_BYTES = 124 * 128 * 32
LOCAL_REG_CEILING = 96

# Ranked QMV time share by proposal width, campaign ledger 200(B).
RANKED_WIDTH_SHARE = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334, 7: 0.122,
                      8: 0.0735, 9: 0.0575}
# QMV share of the ranked candidate leg, campaign ledger 200(B).
QMV_SHARE_OF_RANKED_LEG = 0.826
# Crown table minus our table, ranked candidate seconds/token, scoring prompts,
# after subtracting the plutarch drift floor. PR #80 baseline table.
MEASURED_SCORING_DELTA_PCT = -0.298
# Crown minus ours on the wide prompts near M = 6, PR #80 feedback section 4.
MEASURED_WIDE_DELTA_PCT = -0.44

# Fixed-group-count natural families: M -> IPGs that all give the same groups.
NATURAL_FAMILIES = {8: [4, 5, 6], 6: [3, 4], 7: [4, 5], 9: [5, 6]}


def sg_per_core(regs: int, file_bytes: int) -> int:
    return max(1, file_bytes // (128 * regs))


def ratio_ci(a, b, iters=2000, seed=20260820):
    """Percentile bootstrap on median(a)/median(b)."""
    rng = random.Random(seed)
    point = statistics.median(a) / statistics.median(b)
    n = min(len(a), len(b))
    draws = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(statistics.median([a[i] for i in idx])
                     / statistics.median([b[i] for i in idx]))
    draws.sort()
    return point, draws[int(0.025 * iters)], draws[int(0.975 * iters)]


# ---------------------------------------------------------------- rung 1 data


def load_sweep(path: str, regs_path: str):
    d = json.loads(pathlib.Path(path).read_text())
    census = json.loads(pathlib.Path(regs_path).read_text())
    local = {k.removeprefix("e77_"): v
             for k, v in census["sweep"][LOCAL_ARCH].items()}
    ranked = {k.removeprefix("e77_"): v
              for k, v in census["sweep"][RANKED_ARCH].items()}
    family = {s["arm"]: s["family"] for s in probe_arms()}
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
            rows.append(dict(
                shape=sh["shape"], n=sh["n"], k=sh["k"], arm=arm,
                m=legs[0]["m"], ipg=legs[0]["ipg"], groups=legs[0]["groups"],
                pressure=legs[0]["pressure"], kind=legs[0]["kind"],
                family=family[arm],
                regs=local[arm]["registers"],
                frame_bytes=local[arm]["spill_bytes"],
                ranked_regs=ranked[arm]["registers"],
                ranked_frame_bytes=ranked[arm]["spill_bytes"],
                sg=sg_per_core(local[arm]["registers"], LOCAL_FILE_BYTES),
                t=statistics.median(l["seconds_per_dispatch"] for l in legs),
                samples=[l["seconds_per_dispatch"] for l in legs],
                pos=sorted(statistics.median(v) for v in pos.values()),
            ))
    return d, rows


def sweep_null(rows):
    v = sorted(abs(r["pos"][1] - r["pos"][0]) / r["pos"][0]
               for r in rows if len(r["pos"]) == 2)
    return dict(n=len(v), median=statistics.median(v),
                p90=v[int(0.9 * len(v))], max=v[-1])


# --------------------------------------------------- rung 1: natural contrast


def natural_contrast(rows, null):
    by = {(r["shape"], r["arm"]): r for r in rows}
    shapes = sorted({r["shape"] for r in rows})
    out = []
    for m, ipgs in NATURAL_FAMILIES.items():
        ref_ipg = ipgs[0]
        for ipg in ipgs[1:]:
            per_shape = []
            for shape in shapes:
                a = by.get((shape, f"m{m}_ipg{ipg}_p0"))
                b = by.get((shape, f"m{m}_ipg{ref_ipg}_p0"))
                if not a or not b:
                    continue
                point, lo, hi = ratio_ci(a["samples"], b["samples"])
                per_shape.append(dict(
                    shape=shape, ratio=point, lo=lo, hi=hi, regs=a["regs"],
                    ref_regs=b["regs"], frame_bytes=a["frame_bytes"],
                    ranked_regs=a["ranked_regs"],
                    ref_ranked_regs=b["ranked_regs"],
                    flat=abs(point - 1.0) <= 3 * null["median"]))
            if not per_shape:
                continue
            out.append(dict(
                m=m, ipg=ipg, ref_ipg=ref_ipg, per_shape=per_shape,
                pooled=math.exp(statistics.mean(math.log(p["ratio"])
                                                for p in per_shape)),
                regs=per_shape[0]["regs"], ref_regs=per_shape[0]["ref_regs"],
                ranked_regs=per_shape[0]["ranked_regs"],
                ref_ranked_regs=per_shape[0]["ref_ranked_regs"],
                spill_free=per_shape[0]["frame_bytes"] == 0))
    return out


def cross_session(rows, cells):
    """Replicate the E77 probe arms against the independent E73 session.

    Every natural arm has an E73 cell on all four sweep shapes. The probe adds
    only a dead branch to the shipped body, so the absolute times must agree
    across sessions, and the natural ratios must reproduce.
    """
    e73 = {(c["shape"], c["m"], c["ipg"]): c["t"] for c in cells}
    abs_rows = []
    for r in rows:
        if r["kind"] != "p" or r["pressure"] != 0:
            continue
        ref = e73.get((r["shape"], r["m"], r["ipg"]))
        if ref is None:
            continue
        abs_rows.append(dict(shape=r["shape"], arm=r["arm"], m=r["m"],
                             ipg=r["ipg"], regs=r["regs"], sg=r["sg"],
                             e77=r["t"], e73=ref, rel=(r["t"] - ref) / ref))
    ratio_rows = []
    for m, ipgs in NATURAL_FAMILIES.items():
        for ipg in ipgs[1:]:
            for shape in sorted({r["shape"] for r in rows}):
                a = next((x for x in abs_rows if x["shape"] == shape
                          and x["m"] == m and x["ipg"] == ipg), None)
                b = next((x for x in abs_rows if x["shape"] == shape
                          and x["m"] == m and x["ipg"] == ipgs[0]), None)
                if a and b:
                    ratio_rows.append(dict(
                        shape=shape, m=m, ipg=ipg, ref_ipg=ipgs[0],
                        e77=a["e77"] / b["e77"], e73=a["e73"] / b["e73"]))
    ad = sorted(abs(r["rel"]) for r in abs_rows)
    rd = sorted(abs(r["e77"] / r["e73"] - 1.0) for r in ratio_rows)
    return dict(
        abs_rows=abs_rows, ratio_rows=ratio_rows,
        abs_n=len(ad), abs_median=statistics.median(ad) if ad else None,
        abs_max=ad[-1] if ad else None,
        ratio_n=len(rd), ratio_median=statistics.median(rd) if rd else None,
        ratio_max=rd[-1] if rd else None)


# -------------------------------------------------- rung 1: synthetic ladder


def ladders(rows):
    out = {}
    for r in rows:
        if not (r["m"] == 6 and r["ipg"] in (2, 3)):
            continue
        out.setdefault((r["shape"], r["m"], r["ipg"]), []).append(r)
    for group in out.values():
        base = next(r for r in group if r["kind"] == "p" and r["pressure"] == 0)
        for r in group:
            r["ratio"], r["lo"], r["hi"] = ratio_ci(r["samples"],
                                                    base["samples"])
            r["sg_base"] = base["sg"]
            r["base_regs"] = base["regs"]
        group.sort(key=lambda r: (r["kind"], r["pressure"]))
    return out


def fit_gamma(rows, shapes=None):
    """Omega(S) = (S_ref/S)^gamma, spill-free synthetic arms at or below 96."""
    pts = [(math.log(r["sg_base"] / r["sg"]), math.log(r["ratio"]))
           for r in rows
           if r["kind"] == "p" and r["frame_bytes"] == 0
           and r["regs"] <= LOCAL_REG_CEILING and r["sg"] != r["sg_base"]
           and (shapes is None or r["shape"] in shapes)]
    if not pts:
        return None
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    gamma = sxy / sxx
    res = [y - gamma * x for x, y in pts]
    sigma = math.sqrt(sum(e * e for e in res) / max(1, len(pts) - 1))
    return dict(gamma=gamma, n=len(pts), se=sigma / math.sqrt(sxx),
                rms=math.sqrt(sum(e * e for e in res) / len(res)),
                max_abs=max(abs(e) for e in res))


def staircase(rows, null):
    """Is time(R) a staircase in S, or smooth in R?

    Within-tier spread is the response across register counts that share one
    `S`; a step is the response across a tier boundary. A staircase has
    within-tier spread at the session null and steps above it.
    """
    clean = [r for r in rows
             if r["kind"] == "p" and r["frame_bytes"] == 0
             and r["regs"] <= LOCAL_REG_CEILING]
    tiers, within = {}, []
    for r in clean:
        tiers.setdefault((r["shape"], r["m"], r["ipg"], r["sg"]), []).append(r)
    for group in tiers.values():
        if len(group) > 1:
            vals = [r["ratio"] for r in group]
            within.append((max(vals) - min(vals)) / min(vals))
    steps = []
    for shape, m, ipg in {(k[0], k[1], k[2]) for k in tiers}:
        seq = sorted((r for r in clean if r["shape"] == shape
                      and r["m"] == m and r["ipg"] == ipg),
                     key=lambda r: r["regs"])
        for a, b in zip(seq, seq[1:]):
            if a["sg"] != b["sg"]:
                steps.append(dict(shape=shape, m=m, ipg=ipg,
                                  regs_from=a["regs"], regs_to=b["regs"],
                                  sg_from=a["sg"], sg_to=b["sg"],
                                  step=(b["ratio"] - a["ratio"]) / a["ratio"]))
    return dict(
        within_tier_n=len(within),
        within_tier_median=statistics.median(within) if within else None,
        within_tier_max=max(within) if within else None,
        step_n=len(steps),
        step_median=statistics.median(s["step"] for s in steps) if steps else None,
        step_max=max((s["step"] for s in steps), default=None),
        steps_above_null=sum(1 for s in steps
                             if abs(s["step"]) > 3 * null["median"]),
        steps=sorted(steps, key=lambda s: (s["shape"], s["regs_from"])))


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
                params=best, t_of=t_of,
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


def gamma_required(fit, cores, omega_local, omega_ranked, sg_local, sg_ranked):
    """Smallest occupancy exponent that makes the crown cell win at each M.

    `Omega_L(IPG)` is a per-IPG constant, so the rung-2 refit absorbs it exactly
    into `c(IPG)`; dividing it back out recovers the occupancy-free ranked cost.
    The ranked cost at any exponent is then that cost times
    `(S_L/S_R)**gamma`, which inverts in closed form.
    """
    out = {}
    for m in range(3, 10):
        o, c = SHIPPED_LOCAL[m], CROWN[m]
        if o == c:
            out[m] = dict(ours=o, crown=c, required=None, note="same cell")
            continue
        neutral = {}
        for i in (o, c):
            neutral[i] = (round_cost(fit["t_ranked"], m, i, cores)
                          * omega_local[i] / omega_ranked[i])
        a_o = sg_local[o] / sg_ranked[o]
        a_c = sg_local[c] / sg_ranked[c]
        if a_c >= a_o:
            out[m] = dict(ours=o, crown=c, required=None,
                          note="occupancy cannot flip this pair in this "
                               "direction at any exponent")
            continue
        out[m] = dict(ours=o, crown=c, a_ours=a_o, a_crown=a_c,
                      deficit=neutral[c] / neutral[o],
                      required=math.log(neutral[c] / neutral[o])
                      / math.log(a_o / a_c))
    return out


def table_delta(tfun, cores, table_a, table_b):
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
    _, cells = load_e73(args.session)
    null = sweep_null(rows)
    print(f"sweep: {d['device']}  arms {len(d['arms'])}  reps {d['reps']}  "
          f"entry {d['session_entry_gpu_temp_c']:.1f}C "
          f"exit {d['session_exit_gpu_temp_c']:.1f}C")
    print(f"gate: real_gate={d.get('cool_gate_passed_real_gate')} "
          f"qualified={d.get('gate_qualified_for_timing')}")
    print(f"sweep null: median {null['median']*100:.4f}%  "
          f"p90 {null['p90']*100:.4f}%  max {null['max']*100:.4f}%  "
          f"(n={null['n']})")
    print("arms with local frame bytes: "
          f"{sorted({r['arm'] for r in rows if r['frame_bytes']})}\n")

    print("== rung 1a: NATURAL contrast, fixed M and fixed group count ==")
    print("   measures q(IPG); cannot separate occupancy from partition shape")
    nat = natural_contrast(rows, null)
    for row in nat:
        tag = "" if row["spill_free"] else "   [LOCAL SPILL]"
        print(f"  M={row['m']}  IPG {row['ref_ipg']} -> {row['ipg']}   local "
              f"{row['ref_regs']} -> {row['regs']} regs   ranked "
              f"{row['ref_ranked_regs']} -> {row['ranked_regs']} regs{tag}")
        for p in row["per_shape"]:
            print(f"     {p['shape'][:28]:28s} {p['ratio']:8.5f}  "
                  f"95% CI [{p['lo']:.5f}, {p['hi']:.5f}]  "
                  f"{'null' if p['flat'] else 'MOVES'}")
        print(f"     pooled over shapes {row['pooled']:.5f}")

    xs = cross_session(rows, cells)
    print("\n== rung 1a control: replicate the probe arms against E73 ==")
    print("   the p0 probe adds only a dead branch, so both sessions must agree")
    print("     arm            R    S   shape                        E77/E73-1")
    for r in sorted(xs["abs_rows"], key=lambda r: (r["m"], r["ipg"],
                                                   r["shape"])):
        print(f"     {r['arm']:14s} {r['regs']:3d} {r['sg']:3d}  "
              f"{r['shape'][:28]:28s} {100*r['rel']:+7.3f}%")
    print(f"   absolute agreement: median {100*xs['abs_median']:.3f}%  "
          f"max {100*xs['abs_max']:.3f}%  (n={xs['abs_n']})")
    print(f"   natural ratios reproduce: median "
          f"{100*xs['ratio_median']:.3f}%  max {100*xs['ratio_max']:.3f}%  "
          f"(n={xs['ratio_n']})")

    print("\n== rung 1b: SYNTHETIC ladder, fixed cell, only registers move ==")
    lad = ladders(rows)
    for (shape, m, ipg), group in sorted(lad.items()):
        clean = sorted((r for r in group if r["kind"] == "p"
                        and not r["frame_bytes"]), key=lambda r: r["regs"])
        spill = sorted((r for r in group if r["kind"] == "p"
                        and r["frame_bytes"]), key=lambda r: r["frame_bytes"])
        ctrl = [r for r in group if r["kind"] == "q"]
        if len(clean) < 3:
            continue
        print(f"  {shape}  m{m} ipg{ipg}   R {clean[0]['regs']}.."
              f"{clean[-1]['regs']}   S {clean[0]['sg']}..{clean[-1]['sg']}")
        print("       R    S   P   t/t(p0)      95% CI")
        for r in clean:
            print(f"     {r['regs']:3d}  {r['sg']:3d} {r['pressure']:3d}  "
                  f"{r['ratio']:8.5f}  [{r['lo']:.5f}, {r['hi']:.5f}]")
        for r in spill:
            print(f"     {r['regs']:3d}  {r['sg']:3d} {r['pressure']:3d}  "
                  f"{r['ratio']:8.5f}  [{r['lo']:.5f}, {r['hi']:.5f}]  "
                  f"SPILL {r['frame_bytes']} B, excluded from the fit")
        for r in ctrl:
            base = next(x for x in group if x["kind"] == "p"
                        and x["pressure"] == r["pressure"])
            print(f"     control q{r['pressure']:<3d} {r['ratio']:8.5f} vs "
                  f"p{r['pressure']} {base['ratio']:.5f}  "
                  f"{100*(r['ratio']/base['ratio']-1):+6.3f}%")

    ladder_rows = [r for group in lad.values() for r in group]
    stairs = staircase(ladder_rows, null)
    print(f"\n  staircase test: {stairs['within_tier_n']} within-tier spreads, "
          f"median {100*(stairs['within_tier_median'] or 0):.4f}%, "
          f"max {100*(stairs['within_tier_max'] or 0):.4f}%")
    print(f"  {stairs['step_n']} tier boundaries, median step "
          f"{100*(stairs['step_median'] or 0):+.4f}%, "
          f"{stairs['steps_above_null']} above 3x the session null")

    fits = {"pooled": fit_gamma(ladder_rows)}
    for shape in sorted({r["shape"] for r in ladder_rows}):
        fits[shape] = fit_gamma(ladder_rows, shapes={shape})
    print("\n== occupancy exponent  Omega(S) = (S_ref/S)^gamma ==")
    for name, f in fits.items():
        if f:
            print(f"  {name:26s} gamma {f['gamma']:+8.5f} +- {f['se']:.5f}  "
                  f"n {f['n']:3d}  log-rms {f['rms']:.5f}")
    gamma = fits["pooled"]["gamma"]
    gamma_se = fits["pooled"]["se"]

    def omega(sg):
        return (32.0 / sg) ** gamma

    omega_local = {i: omega(sg_local[i]) for i in range(2, 7)}
    omega_ranked = {i: omega(sg_ranked[i]) for i in range(2, 7)}
    measured_sg = sorted({r["sg"] for r in ladder_rows
                          if r["kind"] == "p" and not r["frame_bytes"]
                          and r["regs"] <= LOCAL_REG_CEILING})
    print(f"  measured S range {measured_sg[0]}..{measured_sg[-1]}")
    print("  IPG  local R/S  Omega_L   ranked R/S  Omega_R  S extrapolated")
    for i in range(2, 7):
        ex = not (measured_sg[0] <= sg_ranked[i] <= measured_sg[-1])
        print(f"   {i}   {cell_regs[LOCAL_ARCH][i]:3d}/{sg_local[i]:<3d} "
              f"{omega_local[i]:8.5f}   {cell_regs[RANKED_ARCH][i]:3d}/"
              f"{sg_ranked[i]:<3d} {omega_ranked[i]:8.5f}  {ex}")

    e73null = e73_session_null(cells)
    fit = fit_surface(cells, args.cores, omega_local, omega_ranked)
    ar = [abs(x) for x in fit["res"]]
    print(f"\n== rung 2 refit on the {len(cells)}-cell E73 surface "
          "(Omega fixed from rung 1) ==")
    print(f"  rel-rms {fit['rms']*100:.2f}%  median "
          f"{statistics.median(ar)*100:.2f}%  max {max(ar)*100:.2f}%  = "
          f"{fit['rms']/e73null['median']:.1f}x E73 session null")
    print(f"  rho0 {fit['rho0']*1e12:.4f} ps/byte   beta {fit['beta']:.4f}")
    print("  c(IPG)      " + " ".join(f"{i}:{v:7.4f}"
                                      for i, v in sorted(fit["c"].items())))
    print("  c*Omega_L   " + " ".join(
        f"{i}:{fit['c'][i]*omega_local[i]:7.4f}" for i in range(2, 7))
        + "   (E73's q(IPG))")
    print("  lam(IPG)    " + " ".join(f"{i}:{v:7.2f}"
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
    print("  M  ours crown  model argmin   crown-minus-ours QMV (%)   share")
    for m in range(3, 10):
        print(f"  {m}    {SHIPPED_LOCAL[m]}     {CROWN[m]}           "
              f"{rk[m]['best']}          {per_m[m]:+9.3f}          "
              f"{RANKED_WIDTH_SHARE[m]*100:5.2f}%")
    v2_m5, v2_m6, v2_m9 = per_m[5] < 0, per_m[6] < 0, per_m[9] < 0
    leg = weighted * QMV_SHARE_OF_RANKED_LEG
    scoring = (sum(RANKED_WIDTH_SHARE[m] * per_m[m] for m in (5, 6))
               / sum(RANKED_WIDTH_SHARE[m] for m in (5, 6))
               * QMV_SHARE_OF_RANKED_LEG)
    print(f"  crown faster at M=5: {v2_m5}   M=6: {v2_m6}   M=9: {v2_m9}")
    print(f"  pool-weighted QMV {weighted:+.4f}%  -> candidate leg {leg:+.4f}%")
    print(f"  M5/M6 only -> candidate leg {scoring:+.4f}%   vs measured "
          f"{MEASURED_SCORING_DELTA_PCT:+.3f}%")

    print("\n== validation 2 in exponent units: what gamma would each M need? ==")
    req = gamma_required(fit, args.ranked_cores, omega_local, omega_ranked,
                         sg_local, sg_ranked)
    print(f"  measured gamma {gamma:+.5f} +- {gamma_se:.5f}")
    print("  M  ours crown  crown deficit  gamma required  sigma away")
    for m in range(3, 10):
        r = req[m]
        if r["required"] is None:
            print(f"  {m}    {r['ours']}     {r['crown']}         -"
                  f"              -            {r['note']}")
            continue
        sig = (r["required"] - gamma) / gamma_se if gamma_se else float("inf")
        print(f"  {m}    {r['ours']}     {r['crown']}      "
              f"{r['deficit']:8.5f}      {r['required']:+9.4f}     "
              f"{sig:+9.1f}")

    print("\n== validation 2b: the advisor's M=6 inequality ==")
    print("  occupancy_penalty(111) - occupancy_penalty(90) must exceed the")
    print("  extra weight stream that <T,6,3> pays over <T,6,6>.")
    _, t_flat = make_model({i: 1.0 for i in range(2, 7)},
                           {i: 1.0 for i in range(2, 7)})
    c6 = round_cost(fit["t_ranked"], 6, 6, args.ranked_cores)
    c3 = round_cost(fit["t_ranked"], 6, 3, args.ranked_cores)
    c6f = sum(calls * t_flat(fit["params"], 6, 6, n, k, args.ranked_cores,
                             ranked=True) for _, n, k, calls in SCORED_SHAPES)
    c3f = sum(calls * t_flat(fit["params"], 6, 3, n, k, args.ranked_cores,
                             ranked=True) for _, n, k, calls in SCORED_SHAPES)
    omega_gap = omega_ranked[6] / omega_ranked[3]
    print(f"  with Omega:    <T,6,3>/<T,6,6> = {c3/c6:.5f}   crown faster: "
          f"{c3 < c6}")
    print(f"  without Omega: <T,6,3>/<T,6,6> = {c3f/c6f:.5f}   crown faster: "
          f"{c3f < c6f}")
    print(f"  Omega_R(111)/Omega_R(90) = {omega_gap:.5f}; the inequality needs "
          f"it above {c3f/c6f:.5f}")
    v2b = c3 < c6
    print(f"  inequality satisfied: {v2b}   (measured wide-prompt effect "
          f"{MEASURED_WIDE_DELTA_PCT:+.2f}%)")

    print("\n== validation 3: control 2, E33 ==")
    e33rows, e33stat = control_e33(fit["t_local"], args.cores)
    print("     shape                              base TGs    obs    pred    err")
    for r in sorted(e33rows, key=lambda r: -r["base_tgs"]):
        print(f"     {r['shape'][:32]:32s} {r['base_tgs']:9d}  {r['obs']:.4f}  "
              f"{r['pred']:.4f}  {100*(r['pred']-r['obs'])/r['obs']:+6.2f}%")
    print(f"     obs tau {e33stat['obs_tau']:+.3f}   pred tau "
          f"{e33stat['pred_tau']:+.3f}")
    e33span = (min(r["pred"] for r in e33rows), max(r["pred"] for r in e33rows))
    print(f"     predicted span {e33span[0]:.4f}..{e33span[1]:.4f}  "
          "(E73 gave 1.5048..1.7394; observed 0.9830..1.0592)")

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
        gamma=gamma, gamma_se=gamma_se, gamma_fits=fits,
        staircase=stairs, natural_contrast=nat, cross_session=xs,
        measured_sg_range=[measured_sg[0], measured_sg[-1]],
        cell_registers=cell_regs, sg_local=sg_local, sg_ranked=sg_ranked,
        omega_local=omega_local, omega_ranked=omega_ranked,
        points=[{k: v for k, v in r.items() if k not in {"pos", "samples"}}
                for r in rows],
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
        validation2_m5=v2_m5, validation2_m6=v2_m6, validation2_m9=v2_m9,
        gamma_required=req,
        validation2b_inequality=v2b,
        m6_ratio_with_omega=c3 / c6, m6_ratio_without_omega=c3f / c6f,
        e33_pred_span=e33span, e33_rows=e33rows, e33_stat=e33stat,
    )

    if v2_m5 and v2_m6:
        print("\n== rung 3: ranked-optimal table (EXTRAPOLATED core count and "
              "register file) ==")
        model_table = {m: rk[m]["best"] for m in range(3, 10)}
        vs_crown, w_crown = table_delta(fit["t_ranked"], args.ranked_cores,
                                        model_table, CROWN)
        vs_ours, w_ours = table_delta(fit["t_ranked"], args.ranked_cores,
                                      model_table, SHIPPED_LOCAL)
        print("  M  legal          ours crown model  margin(%)  vs crown(%)  vs ours(%)")
        for m in range(3, 10):
            mg = rk[m]["margin_pct"]
            print(f"  {m}  {str(legal_ipg(m)):14s} {SHIPPED_LOCAL[m]}     "
                  f"{CROWN[m]}     {model_table[m]}   "
                  f"{'-' if mg is None else round(mg, 2):>8}   "
                  f"{vs_crown[m]:+9.3f}   {vs_ours[m]:+9.3f}")
        print(f"  model vs crown: QMV {w_crown:+.4f}%  -> candidate leg "
              f"{w_crown*QMV_SHARE_OF_RANKED_LEG:+.4f}%")
        print(f"  model vs ours:  QMV {w_ours:+.4f}%  -> candidate leg "
              f"{w_ours*QMV_SHARE_OF_RANKED_LEG:+.4f}%")
        result["rung3_table"] = model_table
        result["rung3_vs_crown_qmv_pct"] = vs_crown
        result["rung3_vs_crown_leg_pct"] = w_crown * QMV_SHARE_OF_RANKED_LEG
        result["rung3_vs_ours_leg_pct"] = w_ours * QMV_SHARE_OF_RANKED_LEG
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
        run.summary.update(result)
        print(f"e77_fit: run_id={run.id} url={run.url}")
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
