#!/usr/bin/env python3
"""E50. Has anyone on the board ever moved the SERIAL leg, and what is the
cross-leg slope d ln(serial) / d ln(mtp) across distinct trees?

Ledger 173(A) predicts, from ONE injection experiment on ONE tree:
    psi_mtp = 0.6736, psi_serial = 0.8525
    a purely-uniform field would show beta = psi_serial/psi_mtp = 1.266
    a purely-MTP-gated field would show beta = 0
This is the independent observational read from hundreds of distinct kernels.

WHY beta COMES OUT AT ~0, AND WHY THAT IS NOT A MEASUREMENT OF psi (E50 result):
the ranked serial leg is not our binary. `.github/workflows/qwen-mtp-ranked-
benchmark.yml` runs `baseline: pinned baseline tree` (a pre-built workspace at
MLXFAST_QWEN_MTP_BASELINE_WS, line 224) against `candidate: this workspace`, so
no candidate edit can move the ranked denominator. beta is therefore structurally
0 on this board and carries NO information about psi_serial. The psi_serial term
only applies to LOCAL --local-iterate pairs, where both legs share one build.
Read beta here as a check on the anchoring, not as kernel physics.

DATA PROVENANCE -- refresh before trusting any number:
  curl -sS -H "Authorization: Bearer $YUKON_API_TOKEN" \
    'https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?limit=2000' \
    | python3 -c 'import json,sys; json.dump(json.load(sys.stdin)["submissions"],open("/tmp/rows_live.json","w"))'
  python3 research/board_tree_identity.py      # -> /tmp/tree_ids.json

METHOD NOTES (each answers a named trap in the E50 assignment):
 1. Trees are deduped by CONTENT (git tree of the build-affecting paths), not by
    submission id and not by commit sha. Commit-sha dedupe is measurably wrong
    here: it merges every row with an absent sha and splits 17 byte-identical
    trees.
 2. Repeated submissions of an identical tree are treated as REPLICATES, never
    as independent draws. They are the null model and carry the noise estimate.
 3. Everything is per-prompt and in logs. Prompt effects cancel exactly in the
    run-mean because every row carries the same 8 prompts. Nothing is regressed
    on officialScore, which is a non-differentiable order statistic.
 4. The cross-leg slope is corrected for BOTH opposing biases at once by
    subtracting the within-tree (replicate) covariance matrix from the
    between-tree covariance matrix: the Var(m) subtraction removes attenuation,
    the Cov(s,m) subtraction removes thermal/contention common mode.
 5. Indexing is entry["key"]; a missing field raises rather than defaulting.
"""
import json
import math
import sys

import numpy as np

ROWS = "/tmp/rows_live.json"
TREES = "/tmp/tree_ids.json"
PSI_MTP, PSI_SERIAL = 0.6736, 0.8525
BETA_UNIFORM = PSI_SERIAL / PSI_MTP
NAMES = ["plutarch", "drama", "travel", "beagle", "medicine", "essays",
         "republic", "botany"]
SCORING = (3, 4)          # 4th and 5th order statistics carry the score
RNG = np.random.default_rng(20260819)


def load():
    rows = json.load(open(ROWS))
    tid = json.load(open(TREES))
    trees, pgroups = tid["build"], tid["groups"]
    out = []
    for r in rows:
        om = r.get("officialMetrics") or {}
        pp = om.get("per_prompt")
        if not (isinstance(pp, list) and len(pp) == 8):
            continue
        if r.get("officialScore") is None or r["id"] not in trees:
            continue
        fps = {p["head_provenance_sha256"][:8] for p in pp}
        if len(fps) != 1:
            continue
        out.append({
            "id": r["id"], "tree": trees[r["id"]], "cohort": fps.pop(),
            "solver": r["solverUsername"], "score": r["officialScore"],
            "sha": str(r.get("submissionCommitSha") or "")[:8],
            "qmv": pgroups["qmv"][r["id"]],
            "serial_shared": pgroups["serial_shared"][r["id"]],
            "mtp_only": pgroups["mtp_only"][r["id"]],
            "per_prompt": {p["prompt_sha256"][:8]:
                           (p["serial_seconds_per_token_mean"],
                            p["mtp_seconds_per_token_mean"]) for p in pp},
        })
    if not out:
        sys.exit("FAIL CLOSED: zero rows selected")
    return out


def prompt_order(rows):
    qs = sorted(rows[0]["per_prompt"])
    med = {q: np.median([r["per_prompt"][q][0] / r["per_prompt"][q][1]
                         for r in rows]) for q in qs}
    return sorted(qs, key=lambda q: med[q])


def matrices(rows, order):
    """100*log seconds/token. S, M are (n_rows, 8)."""
    S = np.array([[100 * math.log(r["per_prompt"][q][0]) for q in order] for r in rows])
    M = np.array([[100 * math.log(r["per_prompt"][q][1]) for q in order] for r in rows])
    return S, M


def group(rows):
    g = {}
    for i, r in enumerate(rows):
        g.setdefault(r["tree"], []).append(i)
    return g


def variance_components(S, M, groups):
    """Q3. Noise from replicate trees only. Returns per-prompt and run-mean
    2x2 noise covariance matrices plus their dof, and the run/prompt split."""
    reps = {k: v for k, v in groups.items() if len(v) > 1}
    sbar, mbar = S.mean(axis=1), M.mean(axis=1)
    pp_ss = np.zeros((2, 2))
    pp_dof = 0
    rm_ss = np.zeros((2, 2))
    rm_dof = 0
    for idx in reps.values():
        n = len(idx)
        ds = S[idx] - S[idx].mean(axis=0)
        dm = M[idx] - M[idx].mean(axis=0)
        pp_ss += np.array([[(ds * ds).sum(), (ds * dm).sum()],
                           [(ds * dm).sum(), (dm * dm).sum()]])
        pp_dof += (n - 1) * S.shape[1]
        a = sbar[idx] - sbar[idx].mean()
        b = mbar[idx] - mbar[idx].mean()
        rm_ss += np.array([[(a * a).sum(), (a * b).sum()],
                           [(a * b).sum(), (b * b).sum()]])
        rm_dof += n - 1
    if rm_dof < 2:
        return None
    C_pp, C_rm = pp_ss / pp_dof, rm_ss / rm_dof
    nq = S.shape[1]
    C_ind = (C_pp - C_rm) * nq / (nq - 1)      # per-prompt independent part
    C_run = C_pp - C_ind                       # whole-run common mode
    return {"n_groups": len(reps), "n_rows": sum(len(v) for v in reps.values()),
            "sizes": sorted((len(v) for v in reps.values()), reverse=True),
            "C_pp": C_pp, "pp_dof": pp_dof, "C_rm": C_rm, "rm_dof": rm_dof,
            "C_ind": C_ind, "C_run": C_run}


def tree_level(S, M, groups):
    keys = sorted(groups)
    sbar, mbar = S.mean(axis=1), M.mean(axis=1)
    ts = np.array([sbar[groups[k]].mean() for k in keys])
    tm = np.array([mbar[groups[k]].mean() for k in keys])
    n = np.array([len(groups[k]) for k in keys], float)
    return keys, ts, tm, n


def eiv(ts, tm, n, C_rm):
    """Q2. Observed between-tree covariance minus the expected replicate-noise
    covariance of a tree mean. One subtraction removes both named biases."""
    obs = np.cov(np.vstack([ts, tm]))
    shrink = C_rm * np.mean(1.0 / n)
    sig = obs - shrink
    d = {"obs": obs, "noise": shrink, "sig": sig}
    d["beta_ols"] = obs[0, 1] / obs[1, 1]
    d["beta_rev"] = obs[0, 0] / obs[0, 1] if obs[0, 1] != 0 else math.nan
    d["beta_sig"] = sig[0, 1] / sig[1, 1] if sig[1, 1] > 0 else math.nan
    d["r_obs"] = obs[0, 1] / math.sqrt(obs[0, 0] * obs[1, 1])
    return d


def boot(ts, tm, n, C_rm, B=4000):
    out = []
    k = len(ts)
    for _ in range(B):
        j = RNG.integers(0, k, k)
        try:
            e = eiv(ts[j], tm[j], n[j], C_rm)
        except Exception:
            continue
        out.append([e["beta_ols"], e["beta_rev"], e["beta_sig"]])
    a = np.array(out, float)
    return {name: np.nanpercentile(a[:, i], [2.5, 50, 97.5])
            for i, name in enumerate(["ols", "rev", "sig"])}


def score_of(v):
    w = np.sort(np.asarray(v))
    return (w[SCORING[0]] + w[SCORING[1]]) / 2.0


def signal_sd_bounds(v_obs, sigma_noise, nrep, dof, B=3000):
    """Monte-Carlo confidence bounds on the TRUE between-tree sd of a leg.

    Null: every tree has the same true time and the observed tree-to-tree
    scatter is only replicate noise. Bisects for the largest true signal sd
    still consistent with v_obs at 5%, and the smallest at 95%. The noise floor
    itself is resampled on its own dof, so the limited number of repeated trees
    is priced in rather than assumed away. No F/normal tail approximation."""
    k = len(nrep)

    def frac_below(sig):
        s = RNG.chisquare(dof, B) / dof            # noise-floor uncertainty
        v = np.empty(B)
        for i in range(B):
            x = (RNG.normal(0, sig, k)
                 + RNG.normal(0, sigma_noise * math.sqrt(s[i]), k) / np.sqrt(nrep))
            v[i] = x.var(ddof=1)
        return (v <= v_obs).mean()

    def bisect(target):
        lo, hi = 0.0, max(10 * math.sqrt(v_obs), 1e-6)
        if frac_below(hi) > target:
            return hi
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if frac_below(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return bisect(0.95), bisect(0.05)     # (lower 95% bound, upper 95% bound)


def variant_anova(S, M, groups, keys, variant_of, C_rm, label):
    """Q1b. Condition on trees that actually EDITED shared serial-path code.

    The board cannot, on its own, tell 'nobody moved the serial leg' apart from
    'nobody can move it'. This can: if the ranked serial leg executed candidate
    code, then trees carrying genuinely different quantized-matvec kernels would
    have to differ in serial time. Trees sharing byte-identical code in the
    group are the internal control."""
    sbar, mbar = S.mean(axis=1), M.mean(axis=1)
    ts = np.array([sbar[groups[k]].mean() for k in keys])
    tm = np.array([mbar[groups[k]].mean() for k in keys])
    var = np.array([variant_of[k] for k in keys])
    uniq = sorted(set(var))
    print(f"\n  [{label}] {len(uniq)} distinct code variants across {len(keys)} trees")
    out = {}
    for name, x in (("serial", ts), ("mtp", tm)):
        gm = {u: x[var == u] for u in uniq}
        multi = {u: v for u, v in gm.items() if len(v) > 1}
        wss = sum(((v - v.mean()) ** 2).sum() for v in multi.values())
        wdof = sum(len(v) - 1 for v in multi.values())
        means = np.array([v.mean() for v in gm.values()])
        bss = ((means - means.mean()) ** 2).sum()
        bdof = len(uniq) - 1
        if wdof < 2:
            print(f"    {name}: too few within-variant replicates")
            continue
        f = (bss / bdof) / (wss / wdof)
        print(f"    {name:<6s} between-variant sd {math.sqrt(bss/bdof):7.4f} % | "
              f"within-variant sd {math.sqrt(wss/wdof):7.4f} % (dof {wdof}) | "
              f"F = {f:8.2f}")
        out[name] = f
    return out


def report(rows, label, pooled=None):
    order = prompt_order(rows)
    S, M = matrices(rows, order)
    groups = group(rows)
    keys, ts, tm, nrep = tree_level(S, M, groups)
    print(f"\n{'=' * 78}\nCOHORT {label}: {len(rows)} rows -> {len(keys)} distinct trees "
          f"({len(rows) - len(keys)} replicate rows)\n{'=' * 78}")

    own = variance_components(S, M, groups)
    vc = pooled if pooled is not None else own
    print("\n--- Q3. NULL MODEL FROM REPEATED IDENTICAL TREES ---")
    if vc is None:
        print("  too few replicate trees to estimate a noise covariance; "
              "Q2 not estimable in this cohort")
        return None
    if pooled is not None and own is not None:
        print(f"  cohort-local floor: sd_serial {math.sqrt(own['C_rm'][0,0]):.4f} % "
              f"(dof {own['rm_dof']})  -- using POOLED floor below for power")
    print(f"  {vc['n_groups']} repeated trees, {vc['n_rows']} rows, sizes {vc['sizes']}")
    print(f"  per-prompt noise  sd_serial {math.sqrt(vc['C_pp'][0,0]):.4f} %  "
          f"sd_mtp {math.sqrt(vc['C_pp'][1,1]):.4f} %  dof {vc['pp_dof']}")
    print(f"  run-mean   noise  sd_serial {math.sqrt(vc['C_rm'][0,0]):.4f} %  "
          f"sd_mtp {math.sqrt(vc['C_rm'][1,1]):.4f} %  dof {vc['rm_dof']}")
    r_pp = vc["C_pp"][0, 1] / math.sqrt(vc["C_pp"][0, 0] * vc["C_pp"][1, 1])
    r_rm = vc["C_rm"][0, 1] / math.sqrt(vc["C_rm"][0, 0] * vc["C_rm"][1, 1])
    print(f"  COMMON-MODE cross-leg correlation: per-prompt r {r_pp:+.3f}, "
          f"run-mean r {r_rm:+.3f}")
    print(f"  split: whole-run common sd_serial {math.sqrt(max(vc['C_run'][0,0],0)):.4f} % "
          f"| per-prompt independent sd_serial {math.sqrt(max(vc['C_ind'][0,0],0)):.4f} %")

    print("\n--- Q1. HAS THE SERIAL LEG EVER MOVED? ---")
    v_obs = ts.var(ddof=1)
    v_noise = vc["C_rm"][0, 0] * np.mean(1.0 / nrep)
    print(f"  between-tree sd of the serial leg   {math.sqrt(v_obs):.4f} %")
    print(f"  replicate noise floor for a tree    {math.sqrt(v_noise):.4f} %")
    print(f"  variance ratio F = {v_obs / v_noise:6.2f}   "
          f"(excess sd {math.sqrt(max(v_obs - v_noise, 0)):.4f} %)")
    vm_obs = tm.var(ddof=1)
    vm_noise = vc["C_rm"][1, 1] * np.mean(1.0 / nrep)
    print(f"  [mtp leg for contrast] between-tree sd {math.sqrt(vm_obs):.4f} %, "
          f"floor {math.sqrt(vm_noise):.4f} %, F = {vm_obs / vm_noise:6.2f}, "
          f"excess sd {math.sqrt(max(vm_obs - vm_noise, 0)):.4f} %")

    slo, shi = signal_sd_bounds(v_obs, math.sqrt(vc["C_rm"][0, 0]), nrep, vc["rm_dof"])
    print(f"  TRUE between-tree serial sd, 90% MC interval: "
          f"[{slo:.4f} %, {shi:.4f} %]  (noise-floor dof {vc['rm_dof']} resampled)")
    forced = BETA_UNIFORM * math.sqrt(vm_obs)
    print(f"  under a purely-uniform field the observed mtp sd {math.sqrt(vm_obs):.3f} % "
          f"would force a serial sd of {forced:.3f} %")
    print(f"  => at most {100*shi/forced:.2f} % of the field's mtp-leg movement can be "
          f"uniform (two-leg) in character")

    se = math.sqrt(vc["C_rm"][0, 0]) / np.sqrt(nrep)
    z = (ts - np.median(ts)) / se
    print(f"  trees with |z| > 3 on the serial leg: {int((np.abs(z) > 3).sum())} "
          f"of {len(ts)}")
    for i in np.argsort(-np.abs(z))[:6]:
        idx = groups[keys[i]][0]
        print(f"     z {z[i]:+7.2f}  serial {ts[i] - np.median(ts):+7.3f} %  "
              f"mtp {tm[i] - np.median(tm):+8.3f} %  n={int(nrep[i])}  "
              f"{rows[idx]['solver']:<14s} {rows[idx]['sha']}")

    print("\n  SO-WHAT: how much score spread does each leg carry?")
    med_serial = np.median(S, axis=0)
    real = np.array([score_of(np.exp((S[groups[k][0]] - M[groups[k][0]]) / 100))
                     for k in keys])
    frozen = np.array([score_of(np.exp((med_serial - M[groups[k][0]]) / 100))
                       for k in keys])
    print(f"    sd of real scores                       {real.std(ddof=1):.6f}")
    print(f"    sd with every serial leg frozen at median {frozen.std(ddof=1):.6f}")
    print(f"    => serial-leg movement contributes {100*(1 - frozen.std(ddof=1)/real.std(ddof=1)):+.2f} % "
          f"of board score sd")
    d = real - frozen
    print(f"    per-tree score shift from its own serial draw: sd {d.std(ddof=1):.6f}, "
          f"max |{np.abs(d).max():.6f}|")

    print("\n--- Q1b. DID THE TREES THAT EDITED SHARED SERIAL-PATH CODE MOVE IT? ---")
    for gname in ("qmv", "serial_shared", "mtp_only"):
        vo = {k: rows[groups[k][0]][gname] for k in keys}
        variant_anova(S, M, groups, keys, vo, vc["C_rm"], gname)

    print("\n--- Q2. CROSS-LEG SLOPE beta = d ln(serial)/d ln(mtp) ---")
    e = eiv(ts, tm, nrep, vc["C_rm"])
    b = boot(ts, tm, nrep, vc["C_rm"])
    print(f"  observed between-tree correlation r = {e['r_obs']:+.4f}")
    print(f"  raw OLS  (serial ~ mtp)      beta = {e['beta_ols']:+.4f}   "
          f"95% CI [{b['ols'][0]:+.4f}, {b['ols'][2]:+.4f}]   <- attenuated toward 0")
    print(f"  reverse  (1 / OLS mtp~serial) beta = {e['beta_rev']:+.4f}   "
          f"95% CI [{b['rev'][0]:+.4f}, {b['rev'][2]:+.4f}]   <- upper bracket")
    print(f"  noise-corrected (Q3 subtracted) beta = {e['beta_sig']:+.4f}   "
          f"95% CI [{b['sig'][0]:+.4f}, {b['sig'][2]:+.4f}]")
    print(f"  between-tree Cov(s,m) {e['obs'][0,1]:+.6f}  minus within-tree "
          f"{e['noise'][0,1]:+.6f}  = excess {e['sig'][0,1]:+.6f}")
    lo, hi = sorted([e["beta_ols"], e["beta_rev"]])
    print(f"  reverse-regression bracket: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  predictions: uniform QMV {BETA_UNIFORM:+.4f} | fully MTP-gated {0.0:+.4f}")
    for nm, v in (("uniform 1.266", BETA_UNIFORM), ("gated 0", 0.0)):
        inb = b["sig"][0] <= v <= b["sig"][2]
        print(f"    {nm:<14s} {'INSIDE' if inb else 'OUTSIDE'} the corrected 95% CI")
    return {"trees": len(keys), "vc": vc, "eiv": e, "boot": b,
            "serial_sd": math.sqrt(v_obs), "serial_floor": math.sqrt(v_noise),
            "mtp_sd": math.sqrt(vm_obs)}


# fixtures/qwen3_8_27b_mtp_track.json .calibration -- the on-box pinned serial
# calibration. The fixture also asserts the depth-0 leg "does prompt-independent
# work, so all serial readings are interchangeable".
PINNED_SERIAL = 0.037994794617407023


def anchor_check(rows, order):
    S, M = matrices(rows, order)
    s_abs = np.exp(S / 100).mean(axis=1)
    m_abs = np.exp(M / 100).mean(axis=1)
    print(f"\n--- Q1c. IS THE BOARD'S SERIAL LEG THE PINNED DENOMINATOR? ---")
    print(f"  pinned on-box serial calibration  {PINNED_SERIAL:.12f} s/tok")
    print(f"  board median row-mean serial      {np.median(s_abs):.12f} s/tok  "
          f"({100*(np.median(s_abs)/PINNED_SERIAL - 1):+.4f} % vs pin)")
    print(f"  serial leg: sd {100*s_abs.std(ddof=1)/s_abs.mean():.4f} %, "
          f"full range {100*(s_abs.max()/s_abs.min()-1):.2f} %  "
          f"[{s_abs.min():.6f}, {s_abs.max():.6f}]")
    print(f"  mtp    leg: sd {100*m_abs.std(ddof=1)/m_abs.mean():.4f} %, "
          f"full range {100*(m_abs.max()/m_abs.min()-1):.1f} %  "
          f"[{m_abs.min():.6f}, {m_abs.max():.6f}]")
    colmed = np.median(np.exp(S / 100), axis=0)
    print(f"  prompt-independence of depth-0: per-prompt median serial spread "
          f"{100*(colmed.max()/colmed.min()-1):.4f} %")


def main():
    rows = load()
    cohorts = {}
    for r in rows:
        cohorts.setdefault(r["cohort"], []).append(r)
    print(f"loaded {len(rows)} metric rows with a resolvable content tree")
    print("cohort sizes: " + ", ".join(
        f"{k} {len(v)}" for k, v in sorted(cohorts.items(), key=lambda x: -len(x[1]))))
    # Noise is a property of the runner, not of a head-provenance cohort, and
    # every replicate group is inside one cohort by construction (an identical
    # tree carries an identical head). Pooling buys the dof that make the Q1
    # upper bound meaningful.
    order_all = prompt_order(rows)
    Sa, Ma = matrices(rows, order_all)
    pooled = variance_components(Sa, Ma, group(rows))
    print(f"\nPOOLED noise floor over all cohorts: run-mean sd_serial "
          f"{math.sqrt(pooled['C_rm'][0,0]):.4f} %, sd_mtp "
          f"{math.sqrt(pooled['C_rm'][1,1]):.4f} %, dof {pooled['rm_dof']} "
          f"({pooled['n_groups']} repeated trees)")
    anchor_check(rows, order_all)

    big = [k for k, v in sorted(cohorts.items(), key=lambda x: -len(x[1])) if len(v) >= 40]
    out = {}
    for k in big:
        out[k] = report(cohorts[k], k, pooled=pooled)
    print(f"\n\n{'#' * 78}\n# SENSITIVITY ACROSS PROVENANCE COHORTS\n{'#' * 78}")
    print(f"{'cohort':<10s} {'trees':>6s} {'serial sd':>10s} {'floor':>8s} "
          f"{'F':>7s} {'mtp sd':>8s} {'beta_sig':>9s} {'CI':>22s}")
    for k, v in out.items():
        if v is None:
            continue
        f = v["serial_sd"] ** 2 / v["serial_floor"] ** 2
        ci = f"[{v['boot']['sig'][0]:+.2f},{v['boot']['sig'][2]:+.2f}]"
        print(f"{k:<10s} {v['trees']:6d} {v['serial_sd']:10.4f} "
              f"{v['serial_floor']:8.4f} {f:7.2f} {v['mtp_sd']:8.4f} "
              f"{v['eiv']['beta_sig']:+9.4f} {ci:>22s}")


if __name__ == "__main__":
    main()
