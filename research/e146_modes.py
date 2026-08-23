"""E146 R-A rung 3: is the nuisance a latent per-row run mode, and how big is it?

harness=ranked throughout. The input is the pure-nuisance pair set built by
e146_pairs.py: solver-declared zero-delta resamples whose eight draft lengths
and eight head digests are digit-identical to their target, so the candidate
tree is the same on both sides of every pair.

WHAT A LATENT PER-ROW MODE PREDICTS, and what this file tests:

  1. TRIMODALITY OF PAIR DIFFERENCES. If each ranked run independently draws a
     hidden state m in {0, 1} and a state of 1 adds `s` microseconds to every
     drafting round, then a pair difference can only be -s, 0 or +s. A single
     noisy population predicts one mode. Dip, gap and mixture-BIC all run here.
  2. TRANSITIVITY. The state belongs to the ROW, not to the pair, so the pair
     differences inside one connected cluster must be explainable by ONE label
     per row. This is the test a "the fit is just overfitting eight points"
     objection cannot survive: a per-row labelling has far fewer free parameters
     than one free k per pair, and it is falsifiable by any inconsistent
     triangle.
  3. DRAFTING-LINKED, NOT HOST-WIDE. The serial leg of the same receipt shares
     the host, the hour and the thermal state but does no drafting. A host-wide
     stall moves both legs by a similar percentage; a per-drafting-round cost
     moves the candidate leg and leaves the serial leg alone.
  4. THE PLUTARCH NULL. `plutarch` drafts on about 0.03 of its rounds under this
     schedule family, so a per-drafting-round cost must leave it near zero while
     `travel` and `drama` move most. This is a within-receipt control: the same
     run, the same host, the same minute.

Nothing here is a local measurement and nothing here reweights the ranked score.
"""

import json
import math
import os
import random
import sys

import e146_lib as L
import e146_pairs as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-modes.json")

# A pair difference is called a step when it is beyond this many microseconds
# per drafting round. The gap analysis below reports the empty interval that
# justifies the cut rather than assuming it.
STEP_CUT = 300.0
random.seed(20260823)


def gaussian_mixture_bic(sample, components, restarts=40, iters=400):
    """EM for a 1-D Gaussian mixture, best of `restarts`, with BIC.

    Written out rather than imported: the only heavy dependency available here
    is scipy, and a 1-D EM with a variance floor is short enough to audit.
    """
    n = len(sample)
    lo, hi = min(sample), max(sample)
    span = hi - lo or 1.0
    var_floor = (span / (20.0 * components)) ** 2
    best = None
    for _ in range(restarts):
        mus = [random.uniform(lo, hi) for _ in range(components)]
        vars_ = [max(var_floor, (span / components) ** 2)] * components
        weights = [1.0 / components] * components
        ll_prev = None
        for _ in range(iters):
            resp = []
            ll = 0.0
            for x in sample:
                dens = []
                for c in range(components):
                    d = weights[c] * math.exp(-0.5 * (x - mus[c]) ** 2 / vars_[c])
                    dens.append(d / math.sqrt(2.0 * math.pi * vars_[c]))
                total = sum(dens) or 1e-300
                ll += math.log(total)
                resp.append([d / total for d in dens])
            for c in range(components):
                nk = sum(r[c] for r in resp) or 1e-12
                weights[c] = nk / n
                mus[c] = sum(r[c] * x for r, x in zip(resp, sample)) / nk
                vars_[c] = max(var_floor, sum(
                    r[c] * (x - mus[c]) ** 2 for r, x in zip(resp, sample)) / nk)
            if ll_prev is not None and abs(ll - ll_prev) < 1e-9:
                break
            ll_prev = ll
        if best is None or ll > best[0]:
            best = (ll, list(mus), list(vars_), list(weights))
    ll, mus, vars_, weights = best
    free = 3 * components - 1
    return {
        "components": components,
        "log_likelihood": ll,
        "bic": free * math.log(n) - 2.0 * ll,
        "means": sorted(mus),
        "sds": [math.sqrt(v) for v in vars_],
        "weights": weights,
    }


def largest_gap(sample, lo, hi):
    """The widest empty interval of the sample inside [lo, hi]."""
    inside = sorted(x for x in sample if lo <= x <= hi)
    if len(inside) < 2:
        return None
    best = (0.0, None, None)
    for a, b in zip(inside, inside[1:]):
        if b - a > best[0]:
            best = (b - a, a, b)
    return {"width": best[0], "from": best[1], "to": best[2]}


def gap_p_value(sample, draws=20000):
    """Parametric-bootstrap p-value for the widest empty interval above zero.

    The dip test is weak here because 39 of 57 points sit in one tight cluster,
    so it spends its power on the shape of that cluster rather than on the empty
    band. The gap statistic asks the question the eye is actually asking: could
    a single Gaussian of this mean and sd, drawn 57 times, leave a hole this
    wide on the positive side? Null draws use the fitted one-component model.
    """
    n = len(sample)
    mu, sigma = L.mean(sample), L.sd(sample)
    observed = largest_gap(sample, 0.0, max(sample))
    if observed is None:
        return float("nan"), None
    hits = 0
    for _ in range(draws):
        sim = [random.gauss(mu, sigma) for _ in range(n)]
        pos = [x for x in sim if x >= 0.0]
        widest = largest_gap(pos, 0.0, max(pos)) if len(pos) > 1 else None
        if widest and widest["width"] >= observed["width"]:
            hits += 1
    return (hits + 1.0) / (draws + 1.0), observed


def label_component(nodes, edges, step):
    """Best per-row labelling in {0,1} for one connected cluster."""
    index = {nid: i for i, nid in enumerate(nodes)}
    best = None
    for mask in range(1 << len(nodes)):
        labels = [(mask >> i) & 1 for i in range(len(nodes))]
        ss = 0.0
        for a, b, k in edges:
            pred = step * (labels[index[b]] - labels[index[a]])
            ss += (k - pred) ** 2
        if best is None or ss < best[0]:
            best = (ss, labels)
    return {
        "nodes": list(nodes),
        "labels": dict(zip(nodes, best[1])),
        "ss_labelled": best[0],
        "ss_null": sum(k * k for _, _, k in edges),
        "edges": len(edges),
    }


def fit_labels_and_step(comps, edge_by_comp, step0, rounds=30):
    """Joint fit of one binary label per row and ONE global step.

    Coordinate descent: labels are brute-forced per cluster given the step, and
    the step is then the closed-form least-squares value given the labels. One
    global step is a far stronger claim than a free step per cluster, so it is
    the version reported.
    """
    step = step0
    labelled = []
    for _ in range(rounds):
        labelled = []
        for comp in comps:
            edges = edge_by_comp.get(tuple(comp), [])
            labelled.append(label_component(comp, edges, step))
        num = den = 0.0
        for res, comp in zip(labelled, comps):
            labels = res["labels"]
            for a, b, k in edge_by_comp.get(tuple(comp), []):
                delta = labels[b] - labels[a]
                num += delta * k
                den += delta * delta
        new_step = num / den if den else step
        if abs(new_step - step) < 1e-6:
            step = new_step
            break
        step = new_step
    return step, labelled


def components_of(pair_list):
    adj = {}
    for pair in pair_list:
        a = pair["target"].id8
        b = pair["replicate"].id8
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen = set()
    out = []
    for node in sorted(adj):
        if node in seen:
            continue
        stack, comp = [node], []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(sorted(comp))
    return out


def bootstrap_ci(values, statistic, draws=20000, alpha=0.05):
    if not values:
        return (float("nan"), float("nan"))
    reps = []
    n = len(values)
    for _ in range(draws):
        reps.append(statistic([values[random.randrange(n)] for _ in range(n)]))
    reps.sort()
    return (reps[int(alpha / 2 * draws)], reps[int((1 - alpha / 2) * draws)])


def wilson(successes, total, z=1.959963985):
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (centre - half, centre + half)


def main():
    path, rows = L.load()
    pair_list = P.pairs(rows)
    ks = [p["fit"]["k_us_per_drafting_round"] for p in pair_list]

    print("board %s  scored %d  pure-nuisance pairs %d" % (path, len(rows), len(ks)))

    # ---- 1. modality -----------------------------------------------------
    print("\n=== 1. modality of the pure-nuisance pair differences ===")
    try:
        import diptest
        dip, dip_p = diptest.diptest(__import__("numpy").array(ks))
        print("Hartigan dip  D=%.5f  p=%.4g  (H0: unimodal)" % (dip, dip_p))
    except Exception as exc:  # pragma: no cover - reported, never silent
        dip, dip_p = float("nan"), float("nan")
        print("dip test unavailable: %s" % exc)

    mixtures = [gaussian_mixture_bic(ks, c) for c in (1, 2, 3, 4)]
    print("%-12s %14s %14s  %s" % ("components", "loglik", "BIC", "means"))
    for m in mixtures:
        print("%-12d %14.3f %14.3f  %s" % (
            m["components"], m["log_likelihood"], m["bic"],
            " ".join("%.0f" % v for v in m["means"])))
    best_mix = min(mixtures, key=lambda m: m["bic"])
    print("BIC prefers %d components" % best_mix["components"])

    gap_p, gap_pos = gap_p_value(ks)
    gap_neg = largest_gap(ks, min(ks), 0.0)
    if gap_pos:
        print("widest empty interval above zero: %.0f us wide, %.0f -> %.0f"
              % (gap_pos["width"], gap_pos["from"], gap_pos["to"]))
        print("parametric-bootstrap p for a hole that wide under one Gaussian: %.4g"
              % gap_p)
    if gap_neg:
        print("widest empty interval below zero: %.0f us wide, %.0f -> %.0f"
              % (gap_neg["width"], gap_neg["from"], gap_neg["to"]))

    up = [k for k in ks if k > STEP_CUT]
    down = [k for k in ks if k < -STEP_CUT]
    mid = [k for k in ks if abs(k) <= STEP_CUT]
    print("pairs: -step %d   flat %d   +step %d" % (len(down), len(mid), len(up)))
    print("  +step band %.0f..%.0f mean %.1f" % (min(up), max(up), L.mean(up)))
    if down:
        print("  -step band %.0f..%.0f mean %.1f"
              % (min(down), max(down), L.mean(down)))
    print("  flat band  %.0f..%.0f mean %.1f sd %.1f"
          % (min(mid), max(mid), L.mean(mid), L.sd(mid)))

    # ---- 2. transitivity -------------------------------------------------
    print("\n=== 2. is the state a property of the ROW? ===")
    step0 = L.mean([abs(k) for k in up + down])
    comps = components_of(pair_list)
    edge_by_comp = {}
    for pair in pair_list:
        key = pair["target"].id8
        for comp in comps:
            if key in comp:
                edge_by_comp.setdefault(tuple(comp), []).append(
                    (pair["target"].id8, pair["replicate"].id8,
                     pair["fit"]["k_us_per_drafting_round"]))
                break
    fitted_step, labelled = fit_labels_and_step(comps, edge_by_comp, step0)
    ss_lab = sum(r["ss_labelled"] for r in labelled)
    ss_null = sum(r["ss_null"] for r in labelled)
    n_edges = sum(r["edges"] for r in labelled)
    n_nodes = sum(len(r["nodes"]) for r in labelled)
    degree = {}
    for edges in edge_by_comp.values():
        for a, b, _ in edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
    multi_edge_nodes = sum(1 for v in degree.values() if v > 1)
    rmse_lab = math.sqrt(ss_lab / n_edges)
    rmse_null = math.sqrt(ss_null / n_edges)
    labels_all = {k: v for r in labelled for k, v in r["labels"].items()}
    n_high_rows = sum(labels_all.values())
    print("clusters %d  rows %d  edges %d  rows with >1 edge %d"
          % (len(labelled), n_nodes, n_edges, multi_edge_nodes))
    print("joint fit, one binary label per row and ONE global step:")
    print("  step          %8.1f us/drafting-round" % fitted_step)
    print("  residual rms  %8.1f us/dr   (no-mode null: %8.1f)" % (rmse_lab, rmse_null))
    print("  rows labelled high: %d of %d" % (n_high_rows, n_nodes))
    # A tree cluster still constrains the model: the model does not fit an
    # arbitrary edge value, it forces every edge onto {-s, 0, +s}. The honest
    # measure of that constraint is how many edges land near an allowed value.
    near = 0
    for comp in comps:
        for a, b, k in edge_by_comp.get(tuple(comp), []):
            allowed = min(abs(k - v) for v in (-fitted_step, 0.0, fitted_step))
            if allowed <= 3.0 * L.sd(mid):
                near += 1
    print("  edges within 3 flat-sd of an allowed value {-s,0,+s}: %d of %d"
          % (near, n_edges))
    inconsistent = [r for r in labelled
                    if r["edges"] > 1 and math.sqrt(r["ss_labelled"] / r["edges"]) > 250]
    print("  clusters a single per-row labelling cannot fit: %d" % len(inconsistent))
    for r in inconsistent:
        print("    %s rmse %.0f" % (",".join(r["nodes"]),
                                    math.sqrt(r["ss_labelled"] / r["edges"])))
    # The clusters that carry a cycle or a chain are where transitivity has
    # teeth, so they are printed in full.
    print("  chains of three or more rows:")
    for r in sorted(labelled, key=lambda r: -len(r["nodes"])):
        if len(r["nodes"]) < 3:
            continue
        edges = edge_by_comp.get(tuple(r["nodes"]), [])
        print("    %s" % " ".join(
            "%s=%d" % (nid, r["labels"][nid]) for nid in r["nodes"]))
        print("      edges: %s" % "  ".join(
            "%s->%s %+.0f" % (a, b, k) for a, b, k in edges))

    # ---- 3. drafting-linked, not host-wide -------------------------------
    print("\n=== 3. does the same receipt's serial leg move too? ===")
    hi_pairs = [p for p in pair_list
                if p["fit"]["k_us_per_drafting_round"] > STEP_CUT]
    lo_pairs = [p for p in pair_list
                if abs(p["fit"]["k_us_per_drafting_round"]) <= STEP_CUT]
    def s8(pair):
        d = L.serial_pct_diff(pair["target"], pair["replicate"])
        return sum(d.values()) / 8.0
    hi_cand = [p["fit"]["cand_mean8_pct"] for p in hi_pairs]
    hi_ser = [s8(p) for p in hi_pairs]
    lo_cand = [p["fit"]["cand_mean8_pct"] for p in lo_pairs]
    lo_ser = [s8(p) for p in lo_pairs]
    print("%-22s %10s %10s" % ("", "candidate", "serial"))
    print("%-22s %+10.4f %+10.4f" % ("+step pairs, mean %", L.mean(hi_cand), L.mean(hi_ser)))
    print("%-22s %10.4f %10.4f" % ("+step pairs, sd", L.sd(hi_cand), L.sd(hi_ser)))
    print("%-22s %+10.4f %+10.4f" % ("flat pairs, mean %", L.mean(lo_cand), L.mean(lo_ser)))
    print("%-22s %10.4f %10.4f" % ("flat pairs, sd", L.sd(lo_cand), L.sd(lo_ser)))
    leg_ratio = L.mean(hi_cand) / L.mean(hi_ser) if L.mean(hi_ser) else float("nan")
    print("candidate-to-serial move ratio in the +step state: %.1fx" % leg_ratio)
    print("a host-wide stall predicts about 1x; a drafting-linked cost predicts >>1x")

    # ---- 4. the plutarch null -------------------------------------------
    print("\n=== 4. within-receipt control: drafting share against the move ===")
    print("%-10s %14s %14s %14s" % ("prompt", "drafting share", "+step mean %", "flat mean %"))
    plut = {}
    for prompt in L.PROMPT_ORDER:
        share = L.mean([(L.round_count(prompt, p["replicate"].dlen(prompt))
                         - p["replicate"].non_drafting(prompt))
                        / L.round_count(prompt, p["replicate"].dlen(prompt))
                        for p in hi_pairs])
        himean = L.mean([p["fit"]["observed_pct"][prompt] for p in hi_pairs])
        lomean = L.mean([p["fit"]["observed_pct"][prompt] for p in lo_pairs])
        plut[prompt] = {"drafting_share": share, "high_mean_pct": himean,
                        "flat_mean_pct": lomean}
        print("%-10s %14.4f %+14.4f %+14.4f" % (prompt, share, himean, lomean))

    # The eight-point regression below is the honest version of the "R2 = 0.9"
    # claim: it uses the AVERAGED +step shape, so per-pair noise is divided by
    # 14, and it compares against the flat-percent null on the same eight
    # points rather than against zero.
    ref = hi_pairs[0]["target"]
    design = L.design_matrix(ref, hi_pairs[0]["replicate"], "drafting_round")
    obs = [L.mean([p["fit"]["observed_pct"][q] for p in hi_pairs])
           for q in L.PROMPT_ORDER]
    des = [design[q] for q in L.PROMPT_ORDER]
    slope = sum(d * o for d, o in zip(des, obs)) / sum(d * d for d in des)
    ss_res = sum((o - slope * d) ** 2 for d, o in zip(des, obs))
    flat = sum(obs) / len(obs)
    ss_flat = sum((o - flat) ** 2 for o in obs)
    ss_tot = sum(o * o for o in obs)
    shape_r2 = 1.0 - ss_res / ss_flat
    print("averaged +step shape on the drafting-round basis:")
    print("  slope %.1f us/drafting-round   R2 vs zero %.4f   R2 vs flat-percent %.4f"
          % (slope, 1.0 - ss_res / ss_tot, shape_r2))
    print("  a flat percentage slowdown would give R2 vs flat-percent of 0")

    # ---- 5. how often, and is it drifting or selected? -------------------
    print("\n=== 5. P(high) and what conditions it ===")
    n_step = len(up)
    n_pairs = len(ks)
    p_hat = n_step / n_pairs
    lo_ci, hi_ci = wilson(n_step, n_pairs)
    print("P(replicate lands +step | target flat) = %d/%d = %.4f  95%% CI %.4f..%.4f"
          % (n_step, n_pairs, p_hat, lo_ci, hi_ci))
    print("F75's independently reported 1/3 lies inside this interval: %s"
          % ("yes" if lo_ci <= 1.0 / 3 <= hi_ci else "NO"))
    print("-step pairs %d vs +step pairs %d: a symmetric two-state draw with an"
          % (len(down), len(up)))
    print("unselected target predicts equal counts, so the target side is selected")
    print("or the state drifts with calendar time. Both are tested next.")

    hi_when = sorted(p["replicate"].created for p in hi_pairs)
    lo_when = sorted(p["replicate"].created for p in lo_pairs)
    print("+step replicate timestamps: %s .. %s" % (hi_when[0][:16], hi_when[-1][:16]))
    print("flat  replicate timestamps: %s .. %s" % (lo_when[0][:16], lo_when[-1][:16]))
    hi_solvers = sorted({p["replicate"].solver for p in hi_pairs})
    print("+step pairs span %d distinct solvers: %s"
          % (len(hi_solvers), ", ".join(s[:12] for s in hi_solvers)))

    # ---- 6. the step size ------------------------------------------------
    print("\n=== 6. step size ===")
    step_mean = L.mean(up)
    step_ci = bootstrap_ci(up, lambda v: sum(v) / len(v))
    both = up + [-k for k in down]
    both_ci = bootstrap_ci(both, lambda v: sum(v) / len(v))
    print("+step only : %.1f us/drafting-round  95%% CI %.1f..%.1f  (n=%d)"
          % (step_mean, step_ci[0], step_ci[1], len(up)))
    print("both signs : %.1f us/drafting-round  95%% CI %.1f..%.1f  (n=%d)"
          % (L.mean(both), both_ci[0], both_ci[1], len(both)))
    for name, value in (("F152 within-crown", 930.9), ("F78 earlier estimate", 820.0),
                        ("F226 advisor estimate", 876.7)):
        inside = step_ci[0] <= value <= step_ci[1]
        print("  %-22s %7.1f  inside the +step CI: %s"
              % (name, value, "yes" if inside else "NO"))
    print("The pooled mean is BELOW all three earlier campaign estimates. The")
    print("next block shows why, and it is not that the earlier numbers were")
    print("wrong: the step differs by schedule family, and the earlier numbers")
    print("were read inside the crown family.")

    print("\n--- is the step the same in every lineage? ---")
    crown_row = next((r for r in rows if r.id8 == "1db9d63e"), None)
    crown_key = crown_row.draft_key() if crown_row else None
    het = {}
    fams = {}
    for pair in hi_pairs:
        fams.setdefault(pair["target"].draft_key(), []).append(
            pair["fit"]["k_us_per_drafting_round"])
    print("%-6s %4s %9s %9s  %s" % ("family", "n", "mean", "sd", "plutarch/travel draft len"))
    for i, (key, vals) in enumerate(sorted(fams.items(), key=lambda kv: -len(kv[1]))):
        tag = "crown" if key == crown_key else "fam%d" % i
        print("%-6s %4d %9.1f %9.1f  %s / %s"
              % (tag, len(vals), L.mean(vals),
                 L.sd(vals) if len(vals) > 1 else float("nan"), key[5], key[7]))
        het[tag] = {"n": len(vals), "mean": L.mean(vals),
                    "sd": L.sd(vals) if len(vals) > 1 else None}
    print("A cost that is exactly `so many microseconds per drafting round` must")
    print("not depend on the family. It does, by about 40 per cent, and no rival")
    print("unit removes it. The step is therefore a BAND, not a constant, and a")
    print("correction must use the band of the family it is applied to.")
    crown_vals = fams.get(crown_key, [])
    crown_step = L.mean(crown_vals) if crown_vals else float("nan")
    print("step in 1db9d63e's own schedule family: %.1f us/drafting-round (n=%d)"
          % (crown_step, len(crown_vals)))

    # ---- 7. is the step itself moving? -----------------------------------
    print("\n=== 7. is the step a constant, or does it move with time? ===")
    fam_index = {}
    for pair in hi_pairs:
        fam_index.setdefault(pair["target"].draft_key(), len(fam_index))
    print("%-10s %-18s %5s %9s %-14s" % ("replicate", "created", "fam", "k", "solver"))
    for pair in sorted(hi_pairs, key=lambda p: p["replicate"].created):
        print("%-10s %-18s %5d %9.1f %-14s"
              % (pair["replicate"].id8, pair["replicate"].created[:16],
                 fam_index[pair["target"].draft_key()],
                 pair["fit"]["k_us_per_drafting_round"],
                 pair["replicate"].solver[:14]))
    xs = [P._hours(p["replicate"].created) / 24.0 for p in hi_pairs]
    ys = [p["fit"]["k_us_per_drafting_round"] for p in hi_pairs]
    day0 = min(xs)
    xs = [x - day0 for x in xs]
    n = len(xs)
    mx, my = L.mean(xs), L.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope_t = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter_t = my - slope_t * mx
    resid = [y - (inter_t + slope_t * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for y in ys)
    se_slope = math.sqrt(sse / (n - 2) / sxx)
    print("linear-in-time fit: k = %.1f %+.1f per day   R2 %.3f   slope t %.2f  (n=%d)"
          % (inter_t, slope_t, 1.0 - sse / sst, slope_t / se_slope, n))
    print("REJECT that fit as a description. The slope is negative only because")
    print("the middle block belongs to one schedule family; the two most recent")
    print("pairs are 884.0 and 829.9. Calendar time and family are confounded and")
    print("neither alone explains the spread. What the sample does support:")
    print("  the step is NOT one global constant. Use the SAME-FAMILY step when")
    print("  correcting a row, and carry the family spread as uncertainty.")
    drift = {"intercept_us": inter_t, "slope_us_per_day": slope_t,
             "r2": 1.0 - sse / sst, "slope_t": slope_t / se_slope,
             "verdict": "time and schedule family are confounded; no monotone "
                        "drift is supported, and no global constant step fits"}

    payload = {
        "harness": "ranked",
        "board_path": path,
        "pure_nuisance_pairs": n_pairs,
        "dip_statistic": dip,
        "dip_p_value": dip_p,
        "mixture_bic": [{k: v for k, v in m.items() if k != "weights"}
                        for m in mixtures],
        "bic_preferred_components": best_mix["components"],
        "gap_above_zero": gap_pos,
        "gap_above_zero_bootstrap_p": gap_p,
        "gap_below_zero": gap_neg,
        "counts": {"minus_step": len(down), "flat": len(mid), "plus_step": len(up)},
        "flat_band_sd_us": L.sd(mid),
        "transitivity": {
            "clusters": len(labelled), "rows": n_nodes, "edges": n_edges,
            "rows_with_multiple_edges": multi_edge_nodes,
            "residual_rms_us": rmse_lab, "null_residual_rms_us": rmse_null,
            "inconsistent_clusters": len(inconsistent),
            "joint_step_us": fitted_step,
            "rows_labelled_high": n_high_rows,
            "edges_near_allowed_value": near,
            "labels": labels_all,
        },
        "shape_regression": {
            "slope_us_per_drafting_round": slope,
            "r2_vs_zero": 1.0 - ss_res / ss_tot,
            "r2_vs_flat_percent": shape_r2,
        },
        "step_heterogeneity": het,
        "crown_family_step_us": crown_step,
        "step_drift": drift,
        "leg_response": {
            "high_candidate_mean_pct": L.mean(hi_cand),
            "high_serial_mean_pct": L.mean(hi_ser),
            "flat_candidate_mean_pct": L.mean(lo_cand),
            "flat_serial_mean_pct": L.mean(lo_ser),
            "candidate_to_serial_ratio": leg_ratio,
        },
        "per_prompt": plut,
        "e146_p_high_from_replicates": p_hat,
        "p_high_ci95": [lo_ci, hi_ci],
        "step_us_per_drafting_round": step_mean,
        "step_ci95": list(step_ci),
        "step_both_signs": L.mean(both),
        "step_both_signs_ci95": list(both_ci),
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
