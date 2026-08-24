#!/usr/bin/env python3
"""E176: per-prompt null calibration and mechanism fit for factor Q.

Read-only desk analysis over the cached Yukon board payload written by
``research/board_per_prompt.py fetch``.  No GPU, no build.

Factor Q is the software-pipelined, double-buffered weight tile inside the
transposed quantized GEMM family (``qmm_t_impl`` / ``qmm_t_nax_tgp_impl``).
It is measured on the board as the receipt contrast B - C:

    A 5a9f130a  organizer-pure main
    B 180db842  organizer + Q + instrumentation
    C fda590bb  organizer + instrumentation, Q reverted

Task 2 calibrates a per-prompt pairwise 1 sigma for the candidate leg, which
the campaign has only ever had at mean7 granularity (0.189 %).
Task 3 fits two one-parameter mechanism models to the Q vector:

    prefill  Q_p = -100 * D / leg_p                (constant seconds per leg)
    round    Q_p = -100 * d * rounds_p / leg_p     (constant seconds per round)

``harness=ranked`` for every measured number in this file.
"""

import collections
import json
import math
import statistics as st

CACHE = "/tmp/yukon-board/full.json"
PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
ALL8 = sorted(PROMPT_NAMES.values())
DRAFTING = ["drama", "travel", "beagle", "republic", "essays", "medicine",
            "botany"]
RECEIPTS = {"A": "5a9f130a", "B": "180db842", "C": "fda590bb",
            "D": "2c885d64", "crown": "ec24d591", "K": "90c131dc"}
# What each receipt carries beyond organizer-pure A.
#   Q     software-pipelined qmm_t / qmm_t_nax (4 kernel files)
#   I     candidate instrumentation, per drafting round
#   E165  head prefetch
#   K     organizer main with segmentedVerifyDepthCap 7 -> 4, nothing else
CONTENT = {"A": "-", "B": "Q + I", "C": "I", "D": "Q + I + E165",
           "crown": "- (organizer main, other solver)", "K": "cap-4"}
DECODE_TOKENS = 512
MAD_TO_SD = 1.4826


def load():
    rows = json.load(open(CACHE))
    return [r for r in rows
            if isinstance(r, dict)
            and (r.get("officialMetrics") or {}).get("per_prompt")
            and r.get("officialScore") is not None
            and r["officialMetrics"].get("decode_tokens") == DECODE_TOKENS]


def vec(row):
    return {PROMPT_NAMES.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8]): e
            for e in row["officialMetrics"]["per_prompt"]}


def mtp(row):
    return {k: e["mtp_seconds_per_token_mean"] for k, e in vec(row).items()}


def pick(scored, prefix):
    hits = [r for r in scored if r["id"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("prefix %r matched %d rows" % (prefix, len(hits)))
    return hits[0]


def schedule_fingerprint(row):
    """Exact per-prompt (edl, non-drafting round count) 8-tuple.

    Two receipts that agree here emitted the same token stream under the same
    accept walk, so any candidate-leg difference between them is either a
    pure kernel/runtime effect or measurement draw.
    """
    return tuple(sorted(
        (PROMPT_NAMES.get(e["prompt_sha256"][:8]),
         round(e["effective_mean_draft_len"], 9),
         e["non_drafting_round_count"])
        for e in row["officialMetrics"]["per_prompt"]))


def mad_sd(xs):
    m = st.median(xs)
    return MAD_TO_SD * st.median([abs(x - m) for x in xs])


def two_way_residual_sd(rows, field):
    """Robust per-prompt residual scale after removing a per-receipt offset.

    Model log(t[p][i]) = a_i + b_p + eps.  ``a_i`` absorbs every effect that
    scales the whole receipt uniformly (host draw, thermal offset, a
    prompt-independent code change), so the residual isolates prompt-shaped
    dispersion -- exactly the channel a per-prompt claim such as
    "plutarch is different" has to clear.  Median polish keeps a handful of
    genuinely prompt-shaped mechanisms in the cluster from setting the scale.
    """
    mat = []
    for r in rows:
        v = vec(r)
        mat.append([math.log(v[p][field]) for p in ALL8])
    col = [0.0] * len(ALL8)
    row_off = [0.0] * len(mat)
    for _ in range(50):
        for j in range(len(ALL8)):
            col[j] += st.median([mat[i][j] - row_off[i] - col[j]
                                 for i in range(len(mat))])
        for i in range(len(mat)):
            row_off[i] += st.median([mat[i][j] - row_off[i] - col[j]
                                     for j in range(len(ALL8))])
    resid = {}
    for j, p in enumerate(ALL8):
        rs = [(mat[i][j] - row_off[i] - col[j]) * 100.0
              for i in range(len(mat))]
        resid[p] = mad_sd(rs)
    return resid, row_off


def main():
    scored = load()
    rows = {k: pick(scored, v) for k, v in RECEIPTS.items()}
    out = {"harness": "ranked", "board_rows_512": len(scored)}

    print("=" * 78)
    print("0. COHORT -- the four receipts of the factor system")
    print("=" * 78)
    for k in ("A", "B", "C", "crown"):
        r = rows[k]
        om = r["officialMetrics"]
        heads = {e["head_provenance_sha256"][:12] for e in om["per_prompt"]}
        print("  %-5s %s score=%.8f depth=%s maxdepth=%s tokens=%s head=%s"
              % (k, r["id"][:8], r["officialScore"], om.get("mtp_depth"),
                 om.get("mtp_max_draft_depth"), om.get("decode_tokens"),
                 ",".join(sorted(heads))))
    fps = {k: schedule_fingerprint(rows[k]) for k in rows}
    print("  schedule fingerprint A==B :", fps["A"] == fps["B"])
    print("  schedule fingerprint A==C :", fps["A"] == fps["C"])
    print("  schedule fingerprint A==crown:", fps["A"] == fps["crown"])

    # ---------------------------------------------------------------- task 2
    print()
    print("=" * 78)
    print("2. PER-PROMPT NULL CALIBRATION, candidate leg")
    print("=" * 78)

    counts = collections.Counter(schedule_fingerprint(r) for r in scored)
    main_fp = fps["A"]
    cluster = [r for r in scored if schedule_fingerprint(r) == main_fp]
    print("  distinct 512-token schedule fingerprints on the board: %d"
          % len(counts))
    print("  organizer-main-schedule cluster (A/B/C/crown token stream): %d"
          % len(cluster))

    resid_c, row_off_c = two_way_residual_sd(cluster, "mtp_seconds_per_token_mean")
    resid_s, row_off_s = two_way_residual_sd(scored, "serial_seconds_per_token_mean")
    common_c = mad_sd([o * 100.0 for o in row_off_c])
    common_s = mad_sd([o * 100.0 for o in row_off_s])
    print()
    print("  (a) candidate leg, %d-receipt same-schedule cluster" % len(cluster))
    print("      per-receipt common offset, robust sd: %.4f %%" % common_c)
    print("      per-prompt residual, robust sd:")
    for p in ALL8:
        print("        %-9s %.4f %%" % (p, resid_c[p]))
    print()
    print("  (b) serial leg, all %d board receipts, source identical by"
          % len(scored))
    print("      per-receipt common offset, robust sd: %.4f %% (pairwise %.4f)"
          % (common_s, common_s * math.sqrt(2.0)))
    print("      construction (runner-owned prebuilt baseline workspace)")
    for p in ALL8:
        print("        %-9s %.4f %%" % (p, resid_s[p]))

    # single byte-identical pair, kept as an independent check
    dpair = {p: (mtp(rows["crown"])[p] - mtp(rows["A"])[p])
             / mtp(rows["A"])[p] * 100.0 for p in ALL8}
    print()
    print("  (c) byte-identical pair crown ec24d591 vs A 5a9f130a")
    print("      mean over 8: %+.4f %%   sd of the 8: %.4f %%"
          % (st.mean(dpair.values()), st.stdev(dpair.values())))

    # The 115-receipt cluster shares a token stream but not its kernels, so its
    # candidate-leg spread contains real mechanism and is an upper bound only.
    # The serial leg is byte-identical across every board run by construction,
    # so its dispersion is pure measurement draw on the same host, same
    # 512-token window, inside the same alternating pair.
    common_null = common_s * math.sqrt(2.0)
    sigma = {p: math.sqrt(2.0 * resid_s[p] ** 2 + common_null ** 2)
             for p in ALL8}
    sigma_ub = {p: math.sqrt(2.0 * resid_c[p] ** 2 + 2.0 * common_c ** 2)
                for p in ALL8}
    print()
    print("  ADOPTED per-prompt pairwise 1 sigma, candidate leg")
    print("      sigma_p = sqrt(2*serial_resid_p^2 + common_null^2),"
          " common_null = %.3f %%" % common_null)
    print("      prompt      adopted   contaminated upper bound")
    for p in ALL8:
        print("        %-9s %.4f %%   %.4f %%" % (p, sigma[p], sigma_ub[p]))
    out["null"] = {
        "cluster_n": len(cluster),
        "cluster_common_offset_sd_pct": common_c,
        "cluster_per_prompt_residual_sd_pct": resid_c,
        "serial_per_prompt_residual_sd_pct": resid_s,
        "common_pairwise_null_pct": common_null,
        "per_prompt_pairwise_sigma_pct": sigma,
        "per_prompt_pairwise_sigma_upper_bound_pct": sigma_ub,
        "byte_identical_pair_mean_pct": st.mean(dpair.values()),
        "byte_identical_pair_sd_pct": st.stdev(dpair.values()),
    }

    # ---------------------------------------------------------------- task 3
    print()
    print("=" * 78)
    print("3. THE Q VECTOR AND ITS MECHANISM MODELS")
    print("=" * 78)
    mB, mC = mtp(rows["B"]), mtp(rows["C"])
    v = vec(rows["C"])
    q_pct = {p: (mB[p] - mC[p]) / mC[p] * 100.0 for p in ALL8}
    leg = {p: DECODE_TOKENS * mC[p] for p in ALL8}
    q_ms = {p: (mB[p] - mC[p]) * DECODE_TOKENS * 1000.0 for p in ALL8}
    edl = {p: v[p]["effective_mean_draft_len"] for p in ALL8}
    ndrc = {p: v[p]["non_drafting_round_count"] for p in ALL8}
    # tokens = rounds + accepted, accepted = edl * drafting_rounds
    rounds = {p: (DECODE_TOKENS + edl[p] * ndrc[p]) / (1.0 + edl[p])
              for p in ALL8}

    print("  prompt      edl   rounds   leg_s    Q %%      Q ms    sigma%%  n_sig")
    for p in sorted(ALL8, key=lambda x: edl[x]):
        print("  %-9s %5.3f  %6.1f  %6.3f  %+7.4f %+8.1f  %.3f  %+5.2f"
              % (p, edl[p], rounds[p], leg[p], q_pct[p], q_ms[p], sigma[p],
                 q_pct[p] / sigma[p]))
    print("  mean7 (drafting prompts): %+.4f %%"
          % st.mean(q_pct[p] for p in DRAFTING))

    def fit(basis):
        """Weighted least squares of Q_pct on one predictor, no intercept."""
        num = sum(basis[p] * q_pct[p] / sigma[p] ** 2 for p in ALL8)
        den = sum(basis[p] ** 2 / sigma[p] ** 2 for p in ALL8)
        beta = num / den
        chi2 = sum(((q_pct[p] - beta * basis[p]) / sigma[p]) ** 2 for p in ALL8)
        se = math.sqrt(1.0 / den)
        pred = {p: beta * basis[p] for p in ALL8}
        return beta, se, chi2, pred

    models = {
        "prefill (constant s/leg)": {p: -100.0 / leg[p] for p in ALL8},
        "round (constant s/round)": {p: -100.0 * rounds[p] / leg[p]
                                     for p in ALL8},
        "uniform (constant %)": {p: 1.0 for p in ALL8},
    }
    chi2_null = sum((q_pct[p] / sigma[p]) ** 2 for p in ALL8)
    print()
    print("  chi2 of Q == 0 (8 dof): %.2f" % chi2_null)
    out["models"] = {}
    for name, basis in models.items():
        beta, se, chi2, pred = fit(basis)
        if name.startswith("prefill"):
            unit = "%.1f +- %.1f ms saved per leg" % (beta * 1000.0,
                                                      se * 1000.0)
        elif name.startswith("round"):
            unit = "%.1f +- %.1f us saved per round" % (beta * 1e6, se * 1e6)
        else:
            unit = "%+.4f +- %.4f %%" % (beta, se)
        print()
        print("  MODEL %s : %s   chi2=%.2f (7 dof)" % (name, unit, chi2))
        for p in sorted(ALL8, key=lambda x: edl[x]):
            print("      %-9s obs %+7.4f  pred %+7.4f  resid %+7.4f (%+.2f s)"
                  % (p, q_pct[p], pred[p], q_pct[p] - pred[p],
                     (q_pct[p] - pred[p]) / sigma[p]))
        out["models"][name] = {"beta": beta, "se": se, "chi2": chi2,
                               "pred_pct": pred}

    # Two-parameter fit: does a per-round term survive next to a fixed
    # per-leg term?  The per-round coefficient is the quantity the cap-4
    # composition question actually needs.
    b1 = models["prefill (constant s/leg)"]
    b2 = models["round (constant s/round)"]
    s11 = sum(b1[p] ** 2 / sigma[p] ** 2 for p in ALL8)
    s22 = sum(b2[p] ** 2 / sigma[p] ** 2 for p in ALL8)
    s12 = sum(b1[p] * b2[p] / sigma[p] ** 2 for p in ALL8)
    y1 = sum(b1[p] * q_pct[p] / sigma[p] ** 2 for p in ALL8)
    y2 = sum(b2[p] * q_pct[p] / sigma[p] ** 2 for p in ALL8)
    det = s11 * s22 - s12 * s12
    c1 = (s22 * y1 - s12 * y2) / det
    c2 = (s11 * y2 - s12 * y1) / det
    se1 = math.sqrt(s22 / det)
    se2 = math.sqrt(s11 / det)
    chi2_2p = sum(((q_pct[p] - c1 * b1[p] - c2 * b2[p]) / sigma[p]) ** 2
                  for p in ALL8)
    print()
    print("  MODEL prefill + round (2 parameters), chi2=%.2f (6 dof)" % chi2_2p)
    print("      per-leg   term: %+.1f +- %.1f ms   (%.1f sigma)"
          % (c1 * 1000.0, se1 * 1000.0, c1 / se1))
    print("      per-round term: %+.1f +- %.1f us   (%.1f sigma)"
          % (c2 * 1e6, se2 * 1e6, c2 / se2))
    print("      95%% upper bound on |per-round saving|: %.1f us/round"
          % (abs(c2 * 1e6) + 1.96 * se2 * 1e6))
    out["two_param"] = {"per_leg_s": c1, "per_leg_se": se1,
                        "per_round_s": c2, "per_round_se": se2,
                        "chi2": chi2_2p}

    # ------------------------------------------------- direct channel split
    # `prefill_seconds_per_token` is a candidate-side per-prompt field on the
    # receipt, so the leg splits without any model:
    #     leg      = 512 * mtp_seconds_per_token_mean
    #     prefill  = 512 * prefill_seconds_per_token
    #     decode   = leg - prefill
    print()
    print("=" * 78)
    print("3b. DIRECT CHANNEL SPLIT FROM THE RECEIPTS (no model)")
    print("=" * 78)
    tags = ("A", "B", "C", "D", "crown", "K")
    vv = {t: vec(rows[t]) for t in tags}
    pre = {t: {p: DECODE_TOKENS * vv[t][p]["prefill_seconds_per_token"]
               for p in ALL8} for t in tags}
    lg = {t: {p: DECODE_TOKENS * vv[t][p]["mtp_seconds_per_token_mean"]
              for p in ALL8} for t in tags}
    dec = {t: {p: lg[t][p] - pre[t][p] for p in ALL8} for t in tags}
    print("  prefill share of the candidate leg (receipt C):")
    for p in sorted(ALL8, key=lambda x: edl[x]):
        print("    %-9s leg %6.3f s  prefill %6.4f s  share %5.2f %%"
              % (p, lg["C"][p], pre["C"][p], 100.0 * pre["C"][p] / lg["C"][p]))
    print()
    print("  Q = B - C, split by channel (negative = Q faster)")
    print("    prompt     prefill %%   prefill ms    decode %%   decode ms")
    for p in sorted(ALL8, key=lambda x: edl[x]):
        print("    %-9s %+8.4f %+10.1f %+11.4f %+10.1f"
              % (p,
                 100.0 * (pre["B"][p] - pre["C"][p]) / pre["C"][p],
                 1000.0 * (pre["B"][p] - pre["C"][p]),
                 100.0 * (dec["B"][p] - dec["C"][p]) / dec["C"][p],
                 1000.0 * (dec["B"][p] - dec["C"][p])))
    pre_pct = st.mean(100.0 * (pre["B"][p] - pre["C"][p]) / pre["C"][p]
                      for p in ALL8)
    dec7_pct = st.mean(100.0 * (dec["B"][p] - dec["C"][p]) / dec["C"][p]
                       for p in DRAFTING)
    print("    mean8 prefill %+.4f %%   mean7 decode %+.4f %%"
          % (pre_pct, dec7_pct))
    # Prefill is a pure Q instrument: neither the instrumentation nor the E165
    # head prefetch runs during the seed.  The four pairs below therefore test
    # C's flagged provenance (FINDING 443) on a channel where C must read zero.
    print()
    print("  FACTOR SYSTEM BY CHANNEL, mean over prompts (negative = faster)")
    print("    pair   content            prefill mean8   decode mean7"
          "   leg mean7")
    pairs = [("B", "A", "Q + I"), ("C", "A", "I"), ("B", "C", "Q"),
             ("D", "B", "E165 - I"), ("D", "A", "Q + I + E165"),
             ("crown", "A", "null, byte-identical")]
    out["factor_by_channel"] = {}
    for hi, lo, what in pairs:
        pp = st.mean(100.0 * (pre[hi][p] - pre[lo][p]) / pre[lo][p]
                     for p in ALL8)
        dd = st.mean(100.0 * (dec[hi][p] - dec[lo][p]) / dec[lo][p]
                     for p in DRAFTING)
        ll = st.mean(100.0 * (lg[hi][p] - lg[lo][p]) / lg[lo][p]
                     for p in DRAFTING)
        print("    %-6s %-18s %+10.4f %% %+13.4f %% %+10.4f %%"
              % ("%s-%s" % (hi, lo), what, pp, dd, ll))
        out["factor_by_channel"]["%s-%s" % (hi, lo)] = {
            "content": what, "prefill_mean8_pct": pp,
            "decode_mean7_pct": dd, "leg_mean7_pct": ll}

    # Closure test.  If a factor acts only on the seed prefill, its leg effect
    # must equal its prefill effect times the measured prefill share.  This
    # form does not depend on the subtracted decode residual being exact, so
    # it survives the `prefill_seconds_per_token` probe caveat.
    print()
    print("  PREFILL-ONLY CLOSURE TEST, mean7")
    print("    pair   content         prefill%% x share   leg%%      residual")
    for hi, lo, what in pairs:
        share = st.mean(pre[lo][p] / lg[lo][p] for p in DRAFTING)
        pp = st.mean(100.0 * (pre[hi][p] - pre[lo][p]) / pre[lo][p]
                     for p in DRAFTING)
        ll = st.mean(100.0 * (lg[hi][p] - lg[lo][p]) / lg[lo][p]
                     for p in DRAFTING)
        print("    %-6s %-16s %+10.4f %%   %+8.4f %% %+8.4f %%"
              % ("%s-%s" % (hi, lo), what, pp * share, ll, ll - pp * share))
        out["factor_by_channel"]["%s-%s" % (hi, lo)]["prefill_only_"
                                                     "prediction_pct"] = (
            pp * share)
        out["factor_by_channel"]["%s-%s" % (hi, lo)]["closure_residual_pct"] = (
            ll - pp * share)

    # B - A is the clean Q instrument: both trees were inspected first hand and
    # only A is free of the instrumentation that contaminates C.
    ba_pre = {p: 100.0 * (pre["B"][p] - pre["A"][p]) / pre["A"][p]
              for p in ALL8}
    ba_dec = {p: 100.0 * (dec["B"][p] - dec["A"][p]) / dec["A"][p]
              for p in ALL8}
    out["channel_split"] = {
        "prefill_share_pct": {p: 100.0 * pre["C"][p] / lg["C"][p]
                              for p in ALL8},
        "bc_prefill_pct": {p: 100.0 * (pre["B"][p] - pre["C"][p]) / pre["C"][p]
                           for p in ALL8},
        "bc_decode_pct": {p: 100.0 * (dec["B"][p] - dec["C"][p]) / dec["C"][p]
                          for p in ALL8},
        "bc_prefill_mean8_pct": pre_pct,
        "bc_decode_mean7_pct": dec7_pct,
        "q_prefill_pct": ba_pre,
        "q_decode_pct": ba_dec,
        "q_prefill_mean8_pct": st.mean(ba_pre[p] for p in ALL8),
        "q_decode_mean7_pct": st.mean(ba_dec[p] for p in DRAFTING),
    }

    # --------------------------------------------------- cap-4 composition
    # Receipt K is organizer main with the one segmentedVerifyDepthCap literal
    # changed.  Q and cap-4 act on disjoint channels, so the composed leg is
    # K's own prefill and decode each scaled by the measured B - A factor.
    print()
    print("=" * 78)
    print("4. CAP-4 COMPOSITION RULING")
    print("=" * 78)
    vk = vec(rows["K"])
    print("    prompt      A edl   K edl   prefill %%   decode %%     leg %%")
    for p in sorted(ALL8, key=lambda x: edl[x]):
        print("    %-9s %6.3f  %6.3f %+10.4f %+10.4f %+10.4f"
              % (p, vv["A"][p]["effective_mean_draft_len"],
                 vk[p]["effective_mean_draft_len"],
                 100.0 * (pre["K"][p] - pre["A"][p]) / pre["A"][p],
                 100.0 * (dec["K"][p] - dec["A"][p]) / dec["A"][p],
                 100.0 * (lg["K"][p] - lg["A"][p]) / lg["A"][p]))

    def median8(xs):
        xs = sorted(xs)
        return 0.5 * (xs[3] + xs[4])

    composed = {}
    for tag in ("A", "K"):
        base = median8(vv[tag][p]["raw_ratio_of_means"] for p in ALL8)
        with_q = median8(
            vv[tag][p]["serial_seconds_per_token_mean"] * DECODE_TOKENS
            / (pre[tag][p] * (1.0 + ba_pre[p] / 100.0)
               + dec[tag][p] * (1.0 + ba_dec[p] / 100.0))
            for p in ALL8)
        composed[tag] = {"published": rows[tag]["officialScore"],
                         "reconstructed": base, "plus_q": with_q,
                         "plus_q_delta_pct": 100.0 * (with_q - base) / base}
        print("    %-6s published %.6f  reconstructed %.6f  + Q %.6f"
              "  %+.4f %%" % (CONTENT[tag], rows[tag]["officialScore"], base,
                              with_q, composed[tag]["plus_q_delta_pct"]))
    out["cap4"] = {
        "receipt": rows["K"]["id"],
        "status": rows["K"]["status"],
        "published": rows["K"]["officialScore"],
        "vs_A_pct": 100.0 * (rows["K"]["officialScore"]
                             - rows["A"]["officialScore"])
        / rows["A"]["officialScore"],
        "prefill_mean8_pct": st.mean(
            100.0 * (pre["K"][p] - pre["A"][p]) / pre["A"][p] for p in ALL8),
        "decode_mean7_pct": st.mean(
            100.0 * (dec["K"][p] - dec["A"][p]) / dec["A"][p]
            for p in DRAFTING),
        "edl_A": {p: vv["A"][p]["effective_mean_draft_len"] for p in ALL8},
        "edl_K": {p: vk[p]["effective_mean_draft_len"] for p in ALL8},
        "composed": composed,
    }

    out["q_vector"] = {"pct": q_pct, "ms": q_ms, "leg_s": leg, "edl": edl,
                       "rounds": rounds, "ndrc": ndrc,
                       "mean7_pct": st.mean(q_pct[p] for p in DRAFTING),
                       "chi2_zero": chi2_null}
    with open("research/e176-q-census.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print()
    print("wrote research/e176-q-census.json")


if __name__ == "__main__":
    main()
