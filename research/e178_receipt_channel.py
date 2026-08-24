#!/usr/bin/env python3
"""E178. Ranked receipt-channel distribution and A-replay option pricing.

harness=ranked throughout. Every number here comes from the public Yukon board
(official M5 receipts). No local timing is used, and no local ratio enters any
estimate.

Method
------
1. Identity. Each board submission has an upstream `refs/heads/submissions/<id>`
   snapshot. `e178_tree_groups.py` content-addresses the whole snapshot (tree
   sha); `e178_editable_digest.py` digests only `benchmark.json:editablePaths`,
   which is exactly what Yukon packages. Two receipts in one group therefore ran
   the same candidate archive; every score difference between them is receipt
   channel, not mechanism.
2. Decomposition. For each prompt the board exposes the serial numerator draw,
   the candidate leg mean, and (for most rows) the prefill share. Decode-only
   candidate time uses 512*(mtp - prefill) per FINDING 455. Deviations are taken
   in logs about the group mean per prompt, so they are relative offsets.
3. Pooling. Pooled within-group variance with sum(n_g - 1) degrees of freedom
   gives the per-receipt channel sigma. The receipt-coherent component is the
   mean over the eight prompts; the idiosyncratic component is the residual.
4. Structure. Time-of-day, within-group elapsed time and submission adjacency
   are tested against the deviations. Board-wide serial drift uses all scored
   receipts, because the serial leg is the runner's own pinned baseline and is
   identical work for every submission.
5. Option. The A-lineage group is the pricing sample: replaying A's tree draws
   again from this distribution.

Usage: python3 research/e178_receipt_channel.py [--json OUT]
"""
import argparse
import collections
import datetime
import json
import math
import statistics as st

BOARD = "/tmp/yukon-board/full.json"
GROUPS = "/tmp/e178/groups.json"
DIGEST = "/tmp/e178/editable_digest.json"

PROMPTS = ["beagle", "botany", "drama", "essays", "medicine", "plutarch",
           "republic", "travel"]
A_TREE_IDS = ["ec24d591", "0f961a85", "5a9f130a", "b3868faa"]
CROWN = 3.7291100105909
OURS_A = 3.70784519415395
XSUMS_LOWER, XSUMS_UPPER = 0.0038, 0.0052   # FINDING 451 leg upper bound band


def hours(iso):
    return datetime.datetime.fromisoformat(
        iso.replace("Z", "+00:00")).timestamp() / 3600.0


def hour_of_day(iso):
    return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).hour \
        + datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).minute / 60.0


def load():
    board = json.load(open(BOARD))
    full = {r["id"][:8]: r["id"] for r in board}
    scored = json.load(open(GROUPS))["scored"]
    digest = json.load(open(DIGEST))
    for rec in scored:
        rec["archive"] = digest[full[rec["id8"]]]
        rec["t"] = hours(rec["createdAt"])
        rec["hod"] = hour_of_day(rec["createdAt"])
    return scored


def group_by(scored, key):
    g = collections.defaultdict(list)
    for rec in scored:
        g[rec[key]].append(rec)
    for members in g.values():
        members.sort(key=lambda m: m["t"])
    return {k: v for k, v in g.items() if len(v) >= 2}


def channels(rec):
    """Per-prompt log observables for one receipt."""
    out = {}
    for p in PROMPTS:
        d = rec["per_prompt"][p]
        row = {"serial": math.log(d["serial"]), "mtp": math.log(d["mtp"]),
               "raw": math.log(d["serial"] / d["mtp"])}
        if d["prefill"] is not None and d["mtp"] > d["prefill"] > 0:
            row["decode"] = math.log(d["mtp"] - d["prefill"])
            row["prefill"] = math.log(d["prefill"])
        out[p] = row
    return out


CHANS = ["serial", "mtp", "decode", "prefill", "raw"]


def deviations(members):
    """Within-group log deviations about the per-prompt group mean."""
    per = [channels(m) for m in members]
    dev = [collections.defaultdict(dict) for _ in members]
    for p in PROMPTS:
        for chan in CHANS:
            vals = [x[p].get(chan) for x in per]
            have = [i for i, v in enumerate(vals) if v is not None]
            if len(have) < 2:
                continue
            mean = sum(vals[i] for i in have) / len(have)
            for i in have:
                dev[i][chan][p] = vals[i] - mean
    return dev


def coherent(dev_i, chan):
    vals = list(dev_i[chan].values())
    if len(vals) < 8:
        return None, None
    m = sum(vals) / len(vals)
    spread = st.pstdev([v - m for v in vals]) * math.sqrt(len(vals) / (len(vals) - 1))
    return m, spread


def pooled_sigma(dev_by_group):
    """Pooled within-group sd of a per-receipt statistic, sum(n_g-1) dof.

    Deviations are already about the group mean, so they carry variance
    sigma^2 (1 - 1/n); the (n-1)/n correction is applied per group.
    """
    ss, dof = 0.0, 0
    for vals in dev_by_group:
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue
        ss += sum(v * v for v in vals)
        dof += len(vals) - 1
    return math.sqrt(ss / dof) if dof else float("nan"), dof


def analyse(scored, key, label):
    groups = group_by(scored, key)
    per_receipt = []
    for gid, members in groups.items():
        devs = deviations(members)
        scores = [math.log(m["score"]) for m in members]
        smean = sum(scores) / len(scores)
        for m, d, s in zip(members, devs, scores):
            row = {"group": gid[:12], "n": len(members), "id8": m["id8"],
                   "ladder": m["ladder"], "solver": m["solver"],
                   "promotion": m["promotion"],
                   "createdAt": m["createdAt"], "score": m["score"],
                   "t": m["t"], "hod": m["hod"],
                   "t_rel": m["t"] - members[0]["t"],
                   "score_dev": s - smean}
            for chan in CHANS:
                c, spread = coherent(d, chan)
                row[chan] = c
                row[chan + "_spread"] = spread
                row[chan + "_pp"] = dict(d[chan])
            per_receipt.append(row)

    out = {"label": label, "identity": key, "n_groups": len(groups),
           "n_receipts": len(per_receipt),
           "sizes": dict(sorted(collections.Counter(
               len(v) for v in groups.values()).items()))}

    by_group = collections.defaultdict(list)
    for r in per_receipt:
        by_group[r["group"]].append(r)

    sig = {}
    for chan in CHANS + ["score_dev"]:
        s, dof = pooled_sigma([[r[chan] for r in rs] for rs in by_group.values()])
        sig[chan] = {"sigma_pct": 100 * s, "dof": dof}
    # idiosyncratic (per-prompt) component, pooled over receipts
    for chan in CHANS:
        vals = [r[chan + "_spread"] for r in per_receipt if r[chan + "_spread"]]
        sig[chan]["per_prompt_spread_pct"] = 100 * st.mean(vals) if vals else None
    out["sigma"] = sig

    # sign coherence: how many of the 8 prompts share the receipt's offset sign
    coh = []
    for r in per_receipt:
        pp = r["mtp_pp"]
        if len(pp) == 8 and r["mtp"] is not None:
            same = sum(1 for v in pp.values() if (v > 0) == (r["mtp"] > 0))
            coh.append(same)
    out["mtp_sign_coherence"] = {"mean_same_sign_of_8": st.mean(coh),
                                 "hist": dict(sorted(collections.Counter(coh).items()))}

    # score consequence of a coherent candidate offset
    pairs = [(r["raw"], r["score_dev"]) for r in per_receipt if r["raw"] is not None]
    out["score_vs_raw_slope"] = ols(pairs)
    pairs = [(-r["mtp"], r["score_dev"]) for r in per_receipt if r["mtp"] is not None]
    out["score_vs_negmtp_slope"] = ols(pairs)
    pairs = [(r["serial"], r["score_dev"]) for r in per_receipt if r["serial"] is not None]
    out["score_vs_serial_slope"] = ols(pairs)
    # do the two channels move together inside one receipt?
    pairs = [(r["serial"], r["mtp"]) for r in per_receipt
             if r["serial"] is not None and r["mtp"] is not None]
    out["serial_vs_mtp_corr"] = pearson(pairs)

    out["standardised"] = tails(per_receipt, sig)
    out["structure"] = structure(per_receipt)
    out["per_receipt"] = per_receipt
    out["groups"] = {g: [r["id8"] for r in rs] for g, rs in by_group.items()}
    return out


def ols(pairs):
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    if sxx == 0:
        return None
    b = sxy / sxx
    resid = [y - my - b * (x - mx) for x, y in pairs]
    se = math.sqrt(sum(r * r for r in resid) / (n - 2) / sxx)
    return {"slope": b, "se": se, "t": b / se if se else None, "n": n,
            "r": pearson(pairs)}


def pearson(pairs):
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    sy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def tails(per_receipt, sig):
    """Standardised deviations: skew, kurtosis, extremes."""
    out = {}
    for chan in ["mtp", "serial", "score_dev"]:
        s = sig[chan]["sigma_pct"] / 100.0
        z = []
        for r in per_receipt:
            if r[chan] is None:
                continue
            n = r["n"]
            # deviation about a group mean of n has sd sigma*sqrt(1-1/n)
            z.append((r["id8"], r[chan] / (s * math.sqrt(1 - 1.0 / n))))
        vals = [v for _, v in z]
        m = st.mean(vals)
        sd = st.pstdev(vals)
        skew = sum((v - m) ** 3 for v in vals) / len(vals) / sd ** 3
        kurt = sum((v - m) ** 4 for v in vals) / len(vals) / sd ** 4
        z.sort(key=lambda kv: kv[1])
        out[chan] = {"n": len(vals), "skew": skew, "excess_kurtosis": kurt - 3,
                     "min": z[0], "max": z[-1],
                     "frac_abs_gt_2": sum(1 for v in vals if abs(v) > 2) / len(vals),
                     "frac_abs_gt_2p5": sum(1 for v in vals if abs(v) > 2.5) / len(vals)}
    return out


def structure(per_receipt):
    out = {}
    for chan in ["mtp", "serial", "score_dev"]:
        rows = [r for r in per_receipt if r[chan] is not None]
        y = [(r, r[chan]) for r in rows]
        out[chan] = {
            "vs_hour_of_day_sin": ols([(math.sin(2 * math.pi * r["hod"] / 24), v)
                                       for r, v in y]),
            "vs_hour_of_day_cos": ols([(math.cos(2 * math.pi * r["hod"] / 24), v)
                                       for r, v in y]),
            "vs_within_group_elapsed_h": ols([(r["t_rel"], v) for r, v in y]),
            "vs_date_day": ols([(r["t"] / 24.0, v) for r, v in y]),
        }
    # adjacency: are receipts that are close in time correlated?
    rows = sorted([r for r in per_receipt if r["mtp"] is not None], key=lambda r: r["t"])
    adj = []
    for a, b in zip(rows, rows[1:]):
        if a["group"] != b["group"]:            # different tree, same clock window
            adj.append((a["mtp"], b["mtp"], b["t"] - a["t"]))
    out["adjacent_pairs_diff_group"] = {
        "n": len(adj),
        "corr_all": pearson([(a, b) for a, b, _ in adj]),
        "corr_within_2h": pearson([(a, b) for a, b, dt in adj if dt <= 2.0]),
        "n_within_2h": sum(1 for _, _, dt in adj if dt <= 2.0),
    }
    return out


def selection_diagnostics(per_receipt):
    """A group exists mostly because its first receipt scored well enough to be
    promoted and copied. That first draw is therefore selected upward, and it
    inflates any within-group variance that includes it. Estimate the channel
    again from copier draws only, which face no such selection: the board keeps
    full metrics for rejected receipts, so a copier's score is recorded whatever
    it is.
    """
    by_group = collections.defaultdict(list)
    for r in per_receipt:
        by_group[r["group"]].append(r)
    for rs in by_group.values():
        rs.sort(key=lambda r: r["t"])

    out = {}
    first = [rs[0] for rs in by_group.values()]
    rest = [r for rs in by_group.values() for r in rs[1:]]
    for chan in ["mtp", "serial", "score_dev"]:
        fv = [r[chan] for r in first if r[chan] is not None]
        rv = [r[chan] for r in rest if r[chan] is not None]
        se = math.sqrt(st.pstdev(fv) ** 2 / len(fv) + st.pstdev(rv) ** 2 / len(rv))
        out["first_vs_rest_" + chan] = {
            "first_mean_pct": 100 * st.mean(fv), "n_first": len(fv),
            "rest_mean_pct": 100 * st.mean(rv), "n_rest": len(rv),
            "diff_pct": 100 * (st.mean(fv) - st.mean(rv)),
            "t": (st.mean(fv) - st.mean(rv)) / se if se else None}
    out["first_promoted_fraction"] = st.mean(
        [1.0 if r.get("promotion") == "promoted" else 0.0 for r in first])

    # Copier-only channel: re-centre each group on its copier draws alone.
    clean = {}
    for chan in CHANS + ["score"]:
        groups_vals = []
        for rs in by_group.values():
            copiers = rs[1:]
            if len(copiers) < 2:
                continue
            if chan == "score":
                vals = [math.log(r["score"]) for r in copiers]
            else:
                vals = [r[chan] for r in copiers if r[chan] is not None]
                if len(vals) != len(copiers):
                    continue
            m = sum(vals) / len(vals)
            groups_vals.append([v - m for v in vals])
        s, dof = pooled_sigma(groups_vals)
        clean[chan] = {"sigma_pct": 100 * s, "dof": dof,
                       "n_groups": len(groups_vals)}
    out["copier_only_sigma"] = clean

    # Same-solver resubmissions versus cross-solver copies.
    for name, pick in (("same_solver", True), ("cross_solver", False)):
        groups_vals = []
        for rs in by_group.values():
            same = len({r["solver"] for r in rs}) == 1
            if same != pick:
                continue
            vals = [math.log(r["score"]) for r in rs]
            m = sum(vals) / len(vals)
            groups_vals.append([v - m for v in vals])
        s, dof = pooled_sigma(groups_vals)
        out[name + "_score_sigma"] = {"sigma_pct": 100 * s, "dof": dof,
                                      "n_groups": len(groups_vals)}
    return out


def heteroskedasticity(per_receipt):
    """Is the channel one width, or does it depend on level, date or spacing?

    A single pooled sigma is only useful if the width is stable. Each group
    gives one within-group sd; regress its log on the group score level, the
    group date and the mean spacing between members.
    """
    by_group = collections.defaultdict(list)
    for r in per_receipt:
        by_group[r["group"]].append(r)
    rows = []
    for g, rs in by_group.items():
        rs.sort(key=lambda r: r["t"])
        n = len(rs)
        sd = math.sqrt(sum(r["score_dev"] ** 2 for r in rs) / (n - 1))
        gaps = [abs(a["t"] - b["t"]) for i, a in enumerate(rs) for b in rs[i + 1:]]
        rows.append({"group": g, "n": n,
                     "mean_score": st.mean([r["score"] for r in rs]),
                     "date": rs[0]["createdAt"][:10], "sd_pct": 100 * sd,
                     "mean_gap_h": st.mean(gaps),
                     "same_solver": len({r["solver"] for r in rs}) == 1,
                     "members": [r["id8"] for r in rs]})
    rows.sort(key=lambda r: r["mean_score"])
    pos = [r for r in rows if r["sd_pct"] > 0]
    day0 = min(r["t"] for r in per_receipt) / 24.0
    out = {"per_group": rows,
           "median_sd_pct": st.median([r["sd_pct"] for r in rows]),
           "log_sd_vs_score_level": ols([(r["mean_score"], math.log(r["sd_pct"]))
                                         for r in pos]),
           "log_sd_vs_day": ols([(min(x["t"] for x in by_group[r["group"]]) / 24.0 - day0,
                                  math.log(r["sd_pct"])) for r in pos]),
           "log_sd_vs_mean_gap_h": ols([(r["mean_gap_h"], math.log(r["sd_pct"]))
                                        for r in pos])}
    # pair-level: does a shorter gap between two receipts of one tree mean a
    # smaller score difference? A slow-moving runner state would say yes.
    pairs = []
    for rs in by_group.values():
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                pairs.append({"gap_h": abs(a["t"] - b["t"]),
                              "abs_diff_pct": 100 * abs(math.log(a["score"] / b["score"])),
                              "same_solver": a["solver"] == b["solver"]})
    out["pairs"] = {
        "n": len(pairs),
        "abs_diff_vs_gap": ols([(p["gap_h"], p["abs_diff_pct"]) for p in pairs]),
        "by_gap_bin": {},
        "same_solver": {"n": sum(1 for p in pairs if p["same_solver"]),
                        "mean_abs_diff_pct": st.mean(
                            [p["abs_diff_pct"] for p in pairs if p["same_solver"]])},
        "cross_solver": {"n": sum(1 for p in pairs if not p["same_solver"]),
                         "mean_abs_diff_pct": st.mean(
                             [p["abs_diff_pct"] for p in pairs if not p["same_solver"]])},
    }
    for lo, hi in ((0, 2), (2, 6), (6, 12), (12, 1e9)):
        sel = [p["abs_diff_pct"] for p in pairs if lo <= p["gap_h"] < hi]
        if sel:
            out["pairs"]["by_gap_bin"]["%g-%gh" % (lo, hi)] = {
                "n": len(sel), "mean_abs_diff_pct": st.mean(sel),
                "rms_pct": math.sqrt(st.mean([v * v for v in sel]))}
    return out


def sigma_by_band(per_receipt, bands=((0, 3.0), (3.0, 3.6), (3.6, 9.9))):
    """Channel width by score level; the top band is our operating point."""
    by_group = collections.defaultdict(list)
    for r in per_receipt:
        by_group[r["group"]].append(r)
    out = {}
    for lo, hi in bands:
        sel = [rs for rs in by_group.values()
               if lo <= st.mean([r["score"] for r in rs]) < hi]
        s, dof = pooled_sigma([[r["score_dev"] for r in rs] for rs in sel])
        s_mtp, _ = pooled_sigma([[r["mtp"] for r in rs] for rs in sel])
        out["%.1f-%.1f" % (lo, hi)] = {
            "n_groups": len(sel), "n_receipts": sum(len(rs) for rs in sel),
            "dof": dof, "score_sigma_pct": 100 * s, "mtp_sigma_pct": 100 * s_mtp,
            "ci95_pct": chi_ci(100 * s, dof)}
    return out


def chi_ci(sigma_pct, dof):
    """Approximate 95% CI for a pooled sd from a chi-square with `dof` dof."""
    if dof < 1:
        return [None, None]
    # Wilson-Hilferty normal approximation to the chi-square quantiles
    def q(p):
        z = 1.959963985 * (1 if p > 0.5 else -1)
        return dof * (1 - 2.0 / (9 * dof) + z * math.sqrt(2.0 / (9 * dof))) ** 3
    return [sigma_pct * math.sqrt(dof / q(0.975)), sigma_pct * math.sqrt(dof / q(0.025))]


def schedule_reproducibility(scored, key="archive"):
    """Does one tree always run the same schedule?

    The board exposes `effective_mean_draft_len` and `non_drafting_round_count`
    per prompt. Those are candidate decisions, not timings. If two receipts of
    one tree report different counters, that tree's schedule reacts to run
    state, and its score channel then carries a schedule term on top of pure
    timing noise. If the counters match exactly, every score difference between
    those receipts is timing alone.
    """
    groups = group_by(scored, key)
    rows = []
    for gid, members in groups.items():
        dl = {p: {round(m["per_prompt"][p]["draft_len"], 6) for m in members}
              for p in PROMPTS}
        nd = {p: {m["per_prompt"][p]["non_drafting"] for m in members}
              for p in PROMPTS}
        n_dl = sum(1 for p in PROMPTS if len(dl[p]) > 1)
        n_nd = sum(1 for p in PROMPTS if len(nd[p]) > 1)
        scores = [math.log(m["score"]) for m in members]
        mean = sum(scores) / len(scores)
        sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (len(members) - 1))
        # relative spread of the draft-length counter itself
        spread = []
        for p in PROMPTS:
            vals = [m["per_prompt"][p]["draft_len"] for m in members]
            if max(vals) > 0:
                spread.append((max(vals) - min(vals)) / st.mean(vals))
        rows.append({"group": gid[:12], "n": len(members),
                     "mean_score": st.mean([m["score"] for m in members]),
                     "score_sd_pct": 100 * sd,
                     "prompts_with_varying_draft_len": n_dl,
                     "prompts_with_varying_non_drafting": n_nd,
                     "max_draft_len_spread_pct": 100 * max(spread) if spread else 0.0,
                     "deterministic": n_dl == 0 and n_nd == 0,
                     "mean_draft_len": st.mean(
                         [m["per_prompt"][p]["draft_len"] for m in members
                          for p in PROMPTS])})
    det = [r for r in rows if r["deterministic"]]
    ada = [r for r in rows if not r["deterministic"]]

    def pooled(rs):
        ss = sum((r["score_sd_pct"] / 100.0) ** 2 * (r["n"] - 1) for r in rs)
        dof = sum(r["n"] - 1 for r in rs)
        return {"n_groups": len(rs), "dof": dof,
                "sigma_pct": 100 * math.sqrt(ss / dof) if dof else None,
                "median_sd_pct": st.median([r["score_sd_pct"] for r in rs]) if rs else None}

    return {"per_group": sorted(rows, key=lambda r: r["score_sd_pct"]),
            "deterministic_schedule": pooled(det),
            "adaptive_schedule": pooled(ada),
            "corr_draft_spread_vs_score_sd": pearson(
                [(r["max_draft_len_spread_pct"], r["score_sd_pct"]) for r in rows])}


def board_serial_drift(scored):
    """Runner-side drift, using every scored receipt.

    The serial leg is the runner's own prebuilt baseline workspace: identical
    work on every submission. Its per-prompt time is therefore a direct probe of
    runner state over the whole campaign.
    """
    by_prompt = collections.defaultdict(list)
    for rec in scored:
        for p in PROMPTS:
            d = rec["per_prompt"][p]
            by_prompt[p].append((rec["t"] / 24.0, math.log(d["serial"]), rec))
    out = {"per_prompt": {}}
    day0 = min(t for v in by_prompt.values() for t, _, _ in v)
    for p, vals in by_prompt.items():
        fit = ols([(t - day0, y) for t, y, _ in vals])
        out["per_prompt"][p] = {
            "n": len(vals), "cv_pct": 100 * st.pstdev([y for _, y, _ in vals]),
            "drift_pct_per_day": 100 * fit["slope"], "t": fit["t"]}
    # pooled receipt-mean serial level vs date and hour of day
    lvl = []
    for rec in scored:
        m = st.mean([math.log(rec["per_prompt"][p]["serial"]) for p in PROMPTS])
        lvl.append((rec, m))
    gm = st.mean([m for _, m in lvl])
    out["receipt_mean_serial"] = {
        "n": len(lvl), "sd_pct": 100 * st.pstdev([m for _, m in lvl]),
        "vs_date_day": ols([(r["t"] / 24.0 - day0, m - gm) for r, m in lvl]),
        "vs_hod_sin": ols([(math.sin(2 * math.pi * r["hod"] / 24), m - gm) for r, m in lvl]),
        "vs_hod_cos": ols([(math.cos(2 * math.pi * r["hod"] / 24), m - gm) for r, m in lvl]),
    }
    # day-by-day level, to expose harness/base steps rather than smooth aging
    byday = collections.defaultdict(list)
    for r, m in lvl:
        byday[r["createdAt"][:10]].append(m)
    out["by_day"] = {d: {"n": len(v), "mean_ms_per_token": 1000 * math.exp(st.mean(v)),
                         "sd_pct": 100 * st.pstdev(v) if len(v) > 1 else None}
                     for d, v in sorted(byday.items())}

    # Control: a faster candidate does less work in the paired leg, so the
    # machine may be cooler when the serial leg runs. Scores rose over the
    # campaign, so a raw date slope can be thermal coupling rather than runner
    # aging. Fit date and the receipt's own score together.
    X = [[r["t"] / 24.0 - day0, r["score"]] for r, _ in lvl]
    y = [m - gm for _, m in lvl]
    out["multivariate"] = mols(X, y, ["day", "own_score"])
    recent = [(r, m) for r, m in lvl if r["createdAt"] >= "2026-08-21"]
    Xr = [[r["t"] / 24.0 - day0, r["score"]] for r, _ in recent]
    yr = [m - gm for _, m in recent]
    out["multivariate_since_0821"] = mols(Xr, yr, ["day", "own_score"])
    out["multivariate_since_0821"]["n"] = len(recent)
    return out


def mols(X, y, names):
    """Least squares with intercept, normal equations, no dependencies."""
    n, k = len(X), len(X[0]) + 1
    A = [[1.0] + row for row in X]
    xtx = [[sum(A[i][a] * A[i][b] for i in range(n)) for b in range(k)]
           for a in range(k)]
    xty = [sum(A[i][a] * y[i] for i in range(n)) for a in range(k)]
    inv = invert(xtx)
    beta = [sum(inv[a][b] * xty[b] for b in range(k)) for a in range(k)]
    resid = [y[i] - sum(beta[a] * A[i][a] for a in range(k)) for i in range(n)]
    s2 = sum(r * r for r in resid) / (n - k)
    return {"n": n, "coef": {nm: {"beta": beta[j + 1],
                                  "se": math.sqrt(s2 * inv[j + 1][j + 1]),
                                  "t": beta[j + 1] / math.sqrt(s2 * inv[j + 1][j + 1])}
                             for j, nm in enumerate(names)}}


def invert(m):
    k = len(m)
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(m)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        a[col] = [v / d for v in a[col]]
        for r in range(k):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [v - f * w for v, w in zip(a[r], a[col])]
    return [row[k:] for row in a]


LADDER_RECEIPTS = [("A", "5a9f130a"), ("B", "180db842"), ("C", "fda590bb"),
                   ("D", "2c885d64"), ("E", "90c131dc"), ("E175", "15017ddf")]


def campaign_implications(scored, sigmas, a_level, n_a):
    """Re-price the campaign's single-receipt comparisons against the channel.

    Every ladder receipt is one draw of its own tree. A difference between two
    single receipts carries sd sigma*sqrt(2); a difference against the A level,
    which has n_a draws, carries sigma*sqrt(1 + 1/n_a).
    """
    byid = {r["id8"]: r for r in scored}
    sigma_pct = sigmas["score_dev"]["sigma_pct"]
    s = sigma_pct / 100.0
    out = {"sigma_pct": sigma_pct,
           "pair_contrast_sd_pct": 100 * s * math.sqrt(2),
           "min_resolvable_2sigma_pair_pct": 200 * s * math.sqrt(2),
           "vs_A_level_sd_pct": 100 * s * math.sqrt(1 + 1.0 / n_a),
           "A_level": a_level, "receipts": {}}
    sd_vs_level = s * math.sqrt(1 + 1.0 / n_a)
    for name, id8 in LADDER_RECEIPTS:
        r = byid.get(id8)
        if not r:
            continue
        rel = (r["score"] - a_level) / a_level
        out["receipts"][name] = {
            "id8": id8, "score": r["score"], "createdAt": r["createdAt"],
            "pct_vs_A_level": 100 * rel, "z_vs_A_level": rel / sd_vs_level,
            "resolved_at_2sigma": abs(rel / sd_vs_level) >= 2.0}
    # the two published claims this experiment was asked to check
    s_mtp = sigmas["mtp"]["sigma_pct"]
    s_dec = sigmas["decode"]["sigma_pct"]
    out["finding_452_recheck"] = {
        "claimed_coherent_offset_pct": -0.2346,
        "channel_pair_sd_pct": math.sqrt(2) * s_mtp,
        "z": -0.2346 / (math.sqrt(2) * s_mtp),
        "claimed_floor_pct": 0.25,
        "measured_pair_floor_2sigma_pct": 2 * math.sqrt(2) * sigma_pct}
    out["finding_453_recheck"] = {
        "claimed_decode_anomaly_pct": 0.9889, "claimed_sigma": 7.0,
        "channel_pair_sd_pct": math.sqrt(2) * s_dec,
        "z": 0.9889 / (math.sqrt(2) * s_dec)}
    # per-prompt scatter understates cross-receipt uncertainty by this factor
    out["per_prompt_understatement_factor"] = \
        s_mtp / (sigmas["mtp"]["per_prompt_spread_pct"] / math.sqrt(8))
    return out


def scenario(mu, sigma, target, ks=(1, 2, 3, 4, 5)):
    """One replay-option scenario: mean level mu, per-receipt channel sd sigma."""
    z = (target - mu) / sigma
    p1 = 1 - norm_cdf(z)
    return {"mu": mu, "sigma": sigma, "sigma_pct": 100 * sigma / mu,
            "z_to_target": z, "p_single": p1,
            "p_best_of_k": {k: 1 - (1 - p1) ** k for k in ks},
            "e_best_of_k": {k: mu + sigma * expected_max_normal(k) for k in ks},
            "e_gain_best_of_k_vs_our_A_pct":
                {k: 100 * (mu + sigma * expected_max_normal(k) - OURS_A) / OURS_A
                 for k in ks}}


def price_replay(scored, sigma_all_pct, sigma_clean_pct, drift_pct_per_day=None):
    """Option value of replaying A's exact tree, at ranked harness.

    The crown draw of this tree is upward selected: copiers only appear because
    that draw beat the live frontier. The unbiased level estimate therefore uses
    the three unselected copier draws, and the channel width uses the pooled
    copier-only estimate. Both selected and unselected variants are reported.
    """
    draws = [r for r in scored if r["id8"] in A_TREE_IDS]
    draws.sort(key=lambda r: r["t"])
    scores = [r["score"] for r in draws]
    n = len(scores)
    mean_all = st.mean(scores)
    copiers = [r["score"] for r in draws[1:]]
    mean_copier = st.mean(copiers)
    out = {"draws": [{"id8": r["id8"], "solver": r["solver"], "score": r["score"],
                      "createdAt": r["createdAt"], "promotion": r["promotion"]}
                     for r in draws],
           "n": n, "mean_all_draws": mean_all, "sd_all_draws": st.stdev(scores),
           "sd_all_draws_pct": 100 * st.stdev(scores) / mean_all,
           "range_pct": 100 * (max(scores) - min(scores)) / mean_all,
           "mean_copier_draws": mean_copier, "n_copier": len(copiers),
           "sd_copier_draws_pct": 100 * st.stdev(copiers) / mean_copier,
           "crown": CROWN, "ours_A": OURS_A,
           "crown_pct_above_copier_mean": 100 * (CROWN - mean_copier) / mean_copier}

    # distribution-free bound, valid only if the crown draw were exchangeable
    out["nonparametric_upper_bound"] = {
        "p_single_replay_is_max": 1.0 / (n + 1),
        "p_best_of_k_is_max": {k: k / float(n + k) for k in (1, 2, 3, 4, 5)},
        "note": "P(new draw is the max of n+1 exchangeable draws) = 1/(n+1). "
                "The crown draw is selected, so this over-states the true "
                "probability and is used as an upper bound only.",
    }

    sd_abs = sigma_all_pct / 100.0 * mean_all
    sd_clean = sigma_clean_pct / 100.0 * mean_copier
    pred = lambda s, m: s * math.sqrt(1 + 1.0 / m)   # noqa: E731
    out["scenarios"] = {
        "selected_mean_all_sigma":
            scenario(mean_all, pred(sd_abs, n), CROWN),
        "unselected_mean_copier_sigma":
            scenario(mean_copier, pred(sd_clean, len(copiers)), CROWN),
        "unselected_mean_all_sigma":
            scenario(mean_copier, pred(sd_abs, len(copiers)), CROWN),
    }
    out["headline"] = out["scenarios"]["unselected_mean_copier_sigma"]

    # a real mechanism gain instead of another draw, on the same channel
    band = {}
    for name, g in (("lower", XSUMS_LOWER), ("upper", XSUMS_UPPER)):
        mu = mean_copier * (1 + g)
        band[name] = scenario(mu, pred(sd_clean, len(copiers)), CROWN, ks=(1, 2))
    out["mechanism_comparison"] = {
        "xsums_band_pct": [100 * XSUMS_LOWER, 100 * XSUMS_UPPER], "band": band,
        "note": "FINDING 451 leg upper bound applied to the unselected A level, "
                "then exposed to the same receipt channel.",
    }
    # The serial numerator is falling, so the same tree scores lower each day.
    if drift_pct_per_day:
        t_now = max(r["t"] for r in scored)
        t_copier = st.mean([r["t"] for r in draws[1:]])
        days = (t_now - t_copier) / 24.0
        mu_now = mean_copier * (1 + drift_pct_per_day / 100.0 * days)
        out["drift_adjustment"] = {
            "drift_pct_per_day": drift_pct_per_day, "days_since_copier_draws": days,
            "mu_today": mu_now,
            "note": "score = serial / candidate; a falling pinned serial "
                    "numerator lowers the score of an unchanged tree."}
        out["scenarios"]["drift_adjusted"] = scenario(
            mu_now, pred(sd_clean, len(copiers)) * mu_now / mean_copier, CROWN)

    # how large must a mechanism be for one receipt to take the crown at 50/80%?
    out["mechanism_size_for_confidence"] = {
        "gain_pct_for_p50": 100 * (CROWN - mean_copier) / mean_copier,
        "gain_pct_for_p80": 100 * (CROWN + 0.8416 * pred(sd_clean, len(copiers))
                                   - mean_copier) / mean_copier,
        "gain_pct_for_p95": 100 * (CROWN + 1.6449 * pred(sd_clean, len(copiers))
                                   - mean_copier) / mean_copier,
    }
    return out


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def t_cdf(x, df):
    """Student-t cdf via the regularised incomplete beta function."""
    if df <= 0:
        return float("nan")
    xx = df / (df + x * x)
    p = 0.5 * betainc(df / 2.0, 0.5, xx)
    return 1 - p if x > 0 else p


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b + lbeta) / a
    # Lentz continued fraction
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1 - c * d) < 1e-12:
            break
    val = front * (f - 1)
    if x < (a + 1) / (a + b + 2):
        return val
    return 1 - betainc(b, a, 1 - x) if False else val


EMAX = {1: 0.0, 2: 0.5641895835, 3: 0.8462843753, 4: 1.0293753730,
        5: 1.1629644736}


def expected_max_normal(k):
    """E[max of k iid standard normals] (exact for k<=5)."""
    return EMAX[k]


def fmt(x, n=4):
    return "None" if x is None else ("%.*f" % (n, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/e178-receipt-channel.json")
    args = ap.parse_args()

    scored = load()
    primary = analyse(scored, "archive", "editable-archive identity")
    strict = analyse(scored, "tree", "exact-tree identity")
    sel = selection_diagnostics(primary["per_receipt"])
    het = heteroskedasticity(primary["per_receipt"])
    bands = sigma_by_band(primary["per_receipt"])
    sched = schedule_reproducibility(scored)
    drift = board_serial_drift(scored)
    top_band = bands["3.6-9.9"]["score_sigma_pct"]
    recent_drift = 100 * drift["multivariate_since_0821"]["coef"]["day"]["beta"]
    pricing = price_replay(scored, top_band,
                           sel["copier_only_sigma"]["score"]["sigma_pct"],
                           drift_pct_per_day=recent_drift)
    impl = campaign_implications(scored, primary["sigma"],
                                 pricing["mean_copier_draws"],
                                 pricing["n_copier"])
    pricing["scenarios"]["top_band_sigma"] = scenario(
        pricing["mean_copier_draws"],
        top_band / 100.0 * pricing["mean_copier_draws"] * math.sqrt(1 + 1.0 / 3),
        CROWN)

    report = {"harness": "ranked", "board_rows_scored": len(scored),
              "primary": primary, "strict": strict, "selection": sel,
              "heteroskedasticity": het, "sigma_by_score_band": bands,
              "schedule_reproducibility": sched, "serial_drift": drift,
              "replay_option": pricing, "campaign_implications": impl}
    slim = json.loads(json.dumps(report))
    json.dump(slim, open(args.json, "w"), indent=1)

    print("=" * 78)
    print("E178 ranked receipt channel  (harness=ranked, %d scored board receipts)"
          % len(scored))
    print("=" * 78)
    for res in (primary, strict):
        print("\n## %s: %d groups, %d receipts, sizes %s"
              % (res["label"], res["n_groups"], res["n_receipts"], res["sizes"]))
        print("   channel                sigma%    dof   per-prompt spread%")
        for chan in CHANS + ["score_dev"]:
            s = res["sigma"][chan]
            print("   %-20s %7.4f %5d   %s"
                  % (chan, s["sigma_pct"], s["dof"],
                     fmt(s.get("per_prompt_spread_pct"))))
        print("   mtp offset sign coherence over 8 prompts: mean %.2f/8  %s"
              % (res["mtp_sign_coherence"]["mean_same_sign_of_8"],
                 res["mtp_sign_coherence"]["hist"]))
        for k in ("score_vs_raw_slope", "score_vs_negmtp_slope",
                  "score_vs_serial_slope"):
            o = res[k]
            print("   %-26s slope %+.3f +- %.3f (t=%+.1f, r=%+.3f, n=%d)"
                  % (k, o["slope"], o["se"], o["t"], o["r"], o["n"]))
        print("   corr(serial offset, mtp offset) within receipt: %s"
              % fmt(res["serial_vs_mtp_corr"], 3))
        for chan, tl in res["standardised"].items():
            print("   tails %-9s skew %+.2f  excess kurt %+.2f  |z|>2 %.3f  "
                  "min %s(%.2f)  max %s(%.2f)"
                  % (chan, tl["skew"], tl["excess_kurtosis"], tl["frac_abs_gt_2"],
                     tl["min"][0], tl["min"][1], tl["max"][0], tl["max"][1]))

    print("\n## structure tests (primary identity)")
    for chan, tests in primary["structure"].items():
        if chan == "adjacent_pairs_diff_group":
            continue
        for name, o in tests.items():
            if o:
                print("   %-10s %-26s slope %+.5f  t=%+.2f  (n=%d)"
                      % (chan, name, o["slope"], o["t"], o["n"]))
    adj = primary["structure"]["adjacent_pairs_diff_group"]
    print("   adjacency (different trees, consecutive in time): r=%s over %d pairs; "
          "within 2h r=%s over %d"
          % (fmt(adj["corr_all"], 3), adj["n"], fmt(adj["corr_within_2h"], 3),
             adj["n_within_2h"]))

    print("\n## selection check: is the first draw of a group upward selected?")
    print("   fraction of first draws promoted: %.2f" % sel["first_promoted_fraction"])
    for chan in ("mtp", "serial", "score_dev"):
        o = sel["first_vs_rest_" + chan]
        print("   %-10s first %+.4f%% (n=%d) vs rest %+.4f%% (n=%d): diff %+.4f%% "
              "t=%+.2f" % (chan, o["first_mean_pct"], o["n_first"],
                           o["rest_mean_pct"], o["n_rest"], o["diff_pct"], o["t"]))
    print("   copier-only channel sigma (first draw of each group dropped):")
    for chan, o in sel["copier_only_sigma"].items():
        print("      %-10s %.4f%%  dof %d over %d groups"
              % (chan, o["sigma_pct"], o["dof"], o["n_groups"]))
    for name in ("same_solver_score_sigma", "cross_solver_score_sigma"):
        o = sel[name]
        print("   %-26s %.4f%%  dof %d over %d groups"
              % (name, o["sigma_pct"], o["dof"], o["n_groups"]))

    print("\n## schedule reproducibility inside a group")
    for name in ("deterministic_schedule", "adaptive_schedule"):
        o = sched[name]
        print("   %-24s %d groups, dof %d, sigma %s"
              % (name, o["n_groups"], o["dof"], fmt(o["sigma_pct"])))

    print("\n## channel width by score band")
    for band, o in bands.items():
        print("   %-9s %2d groups %3d receipts dof %2d  score sigma %.3f%% "
              "[%.3f, %.3f]  mtp sigma %.3f%%"
              % (band, o["n_groups"], o["n_receipts"], o["dof"],
                 o["score_sigma_pct"], o["ci95_pct"][0], o["ci95_pct"][1],
                 o["mtp_sigma_pct"]))

    print("\n## is the channel one width? (%d groups)" % len(het["per_group"]))
    print("   median within-group score sd: %.3f%%" % het["median_sd_pct"])
    for name in ("log_sd_vs_score_level", "log_sd_vs_day", "log_sd_vs_mean_gap_h"):
        o = het[name]
        print("   %-24s slope %+.4f  t=%+.2f  n=%d" % (name, o["slope"], o["t"], o["n"]))
    pp = het["pairs"]
    print("   pair |score diff| vs gap: slope %+.5f %%/h  t=%+.2f  (n=%d pairs)"
          % (pp["abs_diff_vs_gap"]["slope"], pp["abs_diff_vs_gap"]["t"], pp["n"]))
    for b, v in pp["by_gap_bin"].items():
        print("      gap %-8s n=%-3d mean |diff| %.3f%%  rms %.3f%%"
              % (b, v["n"], v["mean_abs_diff_pct"], v["rms_pct"]))
    print("      same-solver pairs n=%d mean |diff| %.3f%%; cross-solver n=%d "
          "mean |diff| %.3f%%"
          % (pp["same_solver"]["n"], pp["same_solver"]["mean_abs_diff_pct"],
             pp["cross_solver"]["n"], pp["cross_solver"]["mean_abs_diff_pct"]))
    print("   groups sorted by score level (n, level, date, sd%, gap_h, same-solver):")
    for r in het["per_group"]:
        print("      %-14s %d %.3f %s %6.3f%% %6.2fh %s"
              % (r["group"], r["n"], r["mean_score"], r["date"], r["sd_pct"],
                 r["mean_gap_h"], r["same_solver"]))

    print("\n## runner drift, serial leg, all %d scored receipts"
          % drift["receipt_mean_serial"]["n"])
    print("   receipt-mean serial sd across whole board: %.3f%%"
          % drift["receipt_mean_serial"]["sd_pct"])
    for name in ("vs_date_day", "vs_hod_sin", "vs_hod_cos"):
        o = drift["receipt_mean_serial"][name]
        print("   %-12s slope %+.6f/day  t=%+.2f" % (name, o["slope"], o["t"]))
    for name in ("multivariate", "multivariate_since_0821"):
        o = drift[name]
        print("   %-26s n=%d  %s" % (name, o["n"], "  ".join(
            "%s beta %+.6f (t=%+.2f)" % (k, v["beta"], v["t"])
            for k, v in o["coef"].items())))
    print("   per-day mean serial ms/token:")
    for d, v in drift["by_day"].items():
        print("      %s n=%-4d %.4f ms  sd %s%%" % (d, v["n"], v["mean_ms_per_token"],
                                                    fmt(v["sd_pct"], 3)))

    pr = pricing
    print("\n## A-lineage: %d ranked draws of one byte-identical tree" % pr["n"])
    for d in pr["draws"]:
        print("   %-9s %-15s %.10f  %-26s %s"
              % (d["id8"], d["solver"], d["score"], d["createdAt"],
                 d["promotion"] or "rejected"))
    print("   all 4 draws: mean %.6f  sd %.3f%%  range %.3f%%"
          % (pr["mean_all_draws"], pr["sd_all_draws_pct"], pr["range_pct"]))
    print("   3 unselected copier draws: mean %.6f  sd %.3f%%; crown sits %+.3f%% "
          "above that level"
          % (pr["mean_copier_draws"], pr["sd_copier_draws_pct"],
             pr["crown_pct_above_copier_mean"]))
    nb = pr["nonparametric_upper_bound"]
    print("   distribution-free upper bound P(one replay is max) = %.3f; best of k: %s"
          % (nb["p_single_replay_is_max"],
             {k: round(v, 3) for k, v in nb["p_best_of_k_is_max"].items()}))
    for name, sc in pr["scenarios"].items():
        print("   %-30s mu %.5f sigma %.3f%%  z %+.2f  P1 %.3f  "
              "P(best of k) %s  E[best of k] %s"
              % (name, sc["mu"], sc["sigma_pct"], sc["z_to_target"], sc["p_single"],
                 {k: round(v, 3) for k, v in sc["p_best_of_k"].items()},
                 {k: round(v, 4) for k, v in sc["e_best_of_k"].items()}))
    mc = pr["mechanism_comparison"]
    for name, sc in mc["band"].items():
        print("   xsums %s bound: mu %.5f -> P(one receipt >= crown) %.3f, "
              "P(best of 2) %.3f"
              % (name, sc["mu"], sc["p_single"], sc["p_best_of_k"][2]))
    ms = pr["mechanism_size_for_confidence"]
    print("   mechanism gain needed for one receipt to take the crown: "
          "%.3f%% at p=0.50, %.3f%% at p=0.80, %.3f%% at p=0.95"
          % (ms["gain_pct_for_p50"], ms["gain_pct_for_p80"], ms["gain_pct_for_p95"]))
    if "drift_adjustment" in pr:
        da = pr["drift_adjustment"]
        print("   drift adjustment: %+.4f%%/day over %.2f days -> mu today %.5f"
              % (da["drift_pct_per_day"], da["days_since_copier_draws"],
                 da["mu_today"]))

    print("\n## campaign implications: single-receipt comparisons vs the channel")
    print("   channel sigma %.3f%%; two single receipts differ with sd %.3f%%; "
          "2 sigma needs %.3f%%"
          % (impl["sigma_pct"], impl["pair_contrast_sd_pct"],
             impl["min_resolvable_2sigma_pair_pct"]))
    print("   against the A level (%d draws, sd %.3f%%):"
          % (pr["n_copier"], impl["vs_A_level_sd_pct"]))
    for name, o in impl["receipts"].items():
        print("      %-5s %-9s %.6f  %+.3f%%  z=%+.2f  resolved@2sigma=%s"
              % (name, o["id8"], o["score"], o["pct_vs_A_level"],
                 o["z_vs_A_level"], o["resolved_at_2sigma"]))
    f452 = impl["finding_452_recheck"]
    print("   FINDING 452 recheck: %+.4f%% offset against pair sd %.3f%% -> z=%+.2f; "
          "the 0.25%% floor should be %.2f%% at 2 sigma"
          % (f452["claimed_coherent_offset_pct"], f452["channel_pair_sd_pct"],
             f452["z"], f452["measured_pair_floor_2sigma_pct"]))
    f453 = impl["finding_453_recheck"]
    print("   FINDING 453 recheck: %+.4f%% decode anomaly against pair sd %.3f%% "
          "-> z=%+.2f, not %.1f sigma"
          % (f453["claimed_decode_anomaly_pct"], f453["channel_pair_sd_pct"],
             f453["z"], f453["claimed_sigma"]))
    print("   per-prompt scatter understates cross-receipt uncertainty by %.1fx"
          % impl["per_prompt_understatement_factor"])
    print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
