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
            "crown": "ec24d591"}
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
    resid_s, _ = two_way_residual_sd(scored, "serial_seconds_per_token_mean")
    common_c = mad_sd([o * 100.0 for o in row_off_c])
    print()
    print("  (a) candidate leg, %d-receipt same-schedule cluster" % len(cluster))
    print("      per-receipt common offset, robust sd: %.4f %%" % common_c)
    print("      per-prompt residual, robust sd:")
    for p in ALL8:
        print("        %-9s %.4f %%" % (p, resid_c[p]))
    print()
    print("  (b) serial leg, all %d board receipts, source identical by"
          % len(scored))
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

    sigma = {p: math.sqrt(2.0 * resid_c[p] ** 2 + 2.0 * common_c ** 2)
             for p in ALL8}
    print()
    print("  ADOPTED per-prompt pairwise 1 sigma, candidate leg")
    print("      sigma_p = sqrt(2*resid_p^2 + 2*common^2)")
    for p in ALL8:
        print("        %-9s %.4f %%" % (p, sigma[p]))
    out["null"] = {
        "cluster_n": len(cluster),
        "common_offset_sd_pct": common_c,
        "per_prompt_residual_sd_pct": resid_c,
        "serial_per_prompt_residual_sd_pct": resid_s,
        "per_prompt_pairwise_sigma_pct": sigma,
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
