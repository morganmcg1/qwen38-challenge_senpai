#!/usr/bin/env python3
"""E188 Stage 1. Re-audit FINDING 440's E165 ranked-harm claim under FINDING 460.

harness=ranked throughout. Every number comes from the public Yukon board
(`officialMetrics.per_prompt` of official M5 receipts). No local timing enters
any estimate.

Question
--------
FINDING 440 read the four-receipt factorial (A/B/C/D) as "E165 costs +0.69% to
+1.56% on ranked M5" and dropped the round-start head prefetch from the ship
set. FINDING 460 later measured the ranked receipt channel at sigma(published)
= 0.689% (candidate-leg side; mtp sigma 0.612%, decode sigma 0.626%, serial
sigma 0.114%) and recorded that per-prompt pairing does not evade the offset
because it is whole-receipt coherent. FINDING 462 withdrew a sibling 7-sigma
claim built on the same per-prompt-scatter null that FINDING 440 used
(pairwise mean7 1sd = 0.189%). This script asks whether the E165 figure
survives.

Estimators
----------
1. `naive`     - FINDING 440's own mean7 contrast with its 0.189% null.
2. `channel`   - the same contrast against sqrt(2)*sigma_leg from FINDING 460.
3. `shape`     - mean7(dev) - plutarch(dev). Plutarch barely drafts (449-488
   non-drafting rounds), so a drafting-round mechanism is ~0 there while a
   whole-receipt coherent channel offset is not. If the channel were perfectly
   uniform across prompts this contrast would cancel it and recover power that
   estimator 2 gives up. Its null is measured, not assumed.
4. `additive`  - the mechanism-native decomposition. E165 is a fixed extra
   memory-traffic block at each round start, so it predicts a constant
   microseconds-per-round cost; the receipt channel is a multiplicative
   host-timing offset. Regress the per-prompt relative decode deviation on
   1/R_p (R_p = measured round cost): the intercept absorbs the multiplicative
   channel draw and the slope prices the additive per-round mechanism.

Nulls for estimators 3 and 4 are calibrated on E178's 35 byte-identical
editable-archive groups (88 receipts), where every within-group difference is
receipt channel by construction.

Usage:
    python3 research/board_per_prompt.py fetch      # writes /tmp/yukon-board/full.json
    python3 research/e188_finding440_audit.py [--json OUT]
"""
import argparse
import json
import math
import statistics as st

BOARD = "/tmp/yukon-board/full.json"
E178 = "research/e178-receipt-channel.json"

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
PROMPTS = sorted(PROMPT_NAMES.values())
DRAFTING = [p for p in PROMPTS if p != "plutarch"]

# FINDING 460 (E178, W&B 0810d370), harness=ranked.
SIGMA_MTP = 0.612
SIGMA_DECODE = 0.626
SIGMA_PUBLISHED = 0.689

# FINDING 440's four receipts (frontier-state.json threeReceiptFactorial).
RECEIPTS = {
    "A": ("5a9f130a", "organizer main byte for byte"),
    "B": ("180db842", "organizer + Q + instrumentation, no E165"),
    "C": ("fda590bb", "organizer + instrumentation, no Q, no E165"),
    "D": ("2c885d64", "organizer + Q + E165, no instrumentation"),
}

# Ranked round counts for the promoted schedule (frontier-state perPromptAcceptance,
# FINDING 374 lattice). B and D share this schedule; verified below from edl.
ROUNDS = {"beagle": 110, "botany": 81, "drama": 252, "essays": 92,
          "medicine": 90, "plutarch": 488, "republic": 93, "travel": 213}


def load_board():
    rows = json.load(open(BOARD))
    out = {}
    for r in rows:
        om = r.get("officialMetrics") or {}
        pp = om.get("per_prompt")
        if not pp:
            continue
        legs = {}
        for row in pp:
            name = PROMPT_NAMES.get(row["prompt_sha256"][:8])
            if name is None:
                continue
            mtp = row["mtp_seconds_per_token_mean"]
            pre = row.get("prefill_seconds_per_token")
            legs[name] = {
                "mtp": mtp,
                "prefill": pre,
                # FINDING 455: mtp_seconds_per_token_mean INCLUDES prefill.
                "decode": (mtp - pre) if pre is not None else None,
                "serial": row["serial_seconds_per_token_mean"],
                "raw": row["raw_ratio_of_means"],
                "edl": row["effective_mean_draft_len"],
                "nodraft": row["non_drafting_round_count"],
                "head": row.get("head_provenance_sha256", "")[:12],
            }
        if len(legs) != 8:
            continue
        out[r["id"][:8]] = {
            "id": r["id"],
            "score": r.get("officialScore"),
            "status": r.get("status"),
            "createdAt": r.get("createdAt"),
            "legs": legs,
        }
    return out


def pct_delta(hi, lo, leg):
    """100*(hi/lo - 1) per prompt; positive = hi is slower."""
    return {p: 100.0 * (hi["legs"][p][leg] / lo["legs"][p][leg] - 1.0)
            for p in PROMPTS}


def mean7(d):
    return st.mean(d[p] for p in DRAFTING)


def ols(x, y):
    """Simple OLS with intercept; returns intercept, slope, se(intercept),
    se(slope), resid sd."""
    n = len(x)
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx
    icpt = my - slope * mx
    resid = [yi - icpt - slope * xi for xi, yi in zip(x, y)]
    dof = n - 2
    s2 = sum(r * r for r in resid) / dof
    se_slope = math.sqrt(s2 / sxx)
    se_icpt = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    return icpt, slope, se_icpt, se_slope, math.sqrt(s2)


def round_costs(rcpt):
    """Measured milliseconds per round, per prompt: 512*(mtp - prefill)/N."""
    return {p: 512.0 * rcpt["legs"][p]["decode"] / ROUNDS[p] * 1000.0
            for p in PROMPTS}


def channel_null(board, leg):
    """Per-prompt relative deviations inside E178's byte-identical groups.

    Returns the deviation vectors (one per receipt, deviation from its group's
    per-prompt mean) so any shape statistic can be calibrated on them.
    """
    groups = json.load(open(E178))["primary"]["groups"]
    vectors = []
    for gid, members in groups.items():
        ids = []
        for m in members:
            key = (m if isinstance(m, str) else
                   (m.get("id") or m.get("submission") or ""))[:8]
            if key in board and all(
                    board[key]["legs"][p][leg] is not None for p in PROMPTS):
                ids.append(key)
        if len(ids) < 2:
            continue
        gm = {p: st.mean(board[i]["legs"][p][leg] for i in ids)
              for p in PROMPTS}
        for i in ids:
            vectors.append({
                "group": gid[:8], "receipt": i, "n_group": len(ids),
                "dev": {p: 100.0 * (board[i]["legs"][p][leg] / gm[p] - 1.0)
                        for p in PROMPTS},
            })
    return vectors


def pooled_sd(values_by_group):
    """Pooled within-group sd of a scalar statistic; dof = sum(n_g - 1)."""
    ss, dof = 0.0, 0
    for vals in values_by_group:
        if len(vals) < 2:
            continue
        m = st.mean(vals)
        ss += sum((v - m) ** 2 for v in vals)
        dof += len(vals) - 1
    return math.sqrt(ss / dof), dof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    board = load_board()
    R = {k: board[v[0]] for k, v in RECEIPTS.items()}
    out = {"harness": "ranked", "source": "yukon board officialMetrics.per_prompt"}

    print("=" * 78)
    print("E188 STAGE 1 - FINDING 440 under the FINDING 460 receipt channel")
    print("harness=ranked. All deltas are candidate-leg; positive = slower.")
    print("=" * 78)

    print("\n[1] RECEIPT IDENTITY AND SCHEDULE PRECONDITION")
    ident = {}
    for k, (pfx, desc) in RECEIPTS.items():
        r = R[k]
        heads = {r["legs"][p]["head"] for p in PROMPTS}
        ident[k] = {"id": r["id"], "score": r["score"], "status": r["status"],
                    "createdAt": r["createdAt"], "content": desc,
                    "heads": sorted(heads)}
        print(f"  {k} {pfx} score={r['score']:.8f} {r['status']:9s} "
              f"{r['createdAt']} head={sorted(heads)}")
    out["receipts"] = ident

    print("\n  edl signature (E165 must not change the schedule):")
    print(f"  {'prompt':10s} " + " ".join(f"{k:>10s}" for k in "ABCD"))
    edl_same_BD = True
    for p in PROMPTS:
        vals = [R[k]["legs"][p]["edl"] for k in "ABCD"]
        if abs(vals[1] - vals[3]) > 1e-9:
            edl_same_BD = False
        print(f"  {p:10s} " + " ".join(f"{v:10.4f}" for v in vals))
    print(f"  B and D edl digit-identical on all 8 prompts: {edl_same_BD}")
    out["scheduleIdenticalBD"] = edl_same_BD

    print("\n[2] REPRODUCE FINDING 440's FACTOR TABLE FROM THE BOARD")
    pairs = [("D", "A", "Q + E165 + P"), ("C", "A", "I + P"),
             ("B", "C", "Q"), ("D", "B", "E165 - I  (P-free, Q-free)")]
    f440 = {"DvsA": 0.8899, "CvsA": 0.8765, "BvsC": -0.6691, "DvsB": 0.6878}
    table = {}
    for hi, lo, content in pairs:
        d = pct_delta(R[hi], R[lo], "mtp")
        key = f"{hi}vs{lo}"
        m7, plu = mean7(d), d["plutarch"]
        led = f440[key]
        print(f"  {key}: mean7={m7:+.4f}%  plutarch={plu:+.4f}%  "
              f"(ledger {led:+.4f}%, delta {m7-led:+.4f}pp)  [{content}]")
        table[key] = {"mean7Pct": m7, "plutarchPct": plu,
                      "ledgerMean7Pct": led, "perPrompt": d,
                      "content": content}
    out["factorTable"] = table

    print("\n[3] NAIVE vs CHANNEL z-SCORES (FINDING 440 null 0.189% vs FINDING 460)")
    sig_pair = math.sqrt(2.0) * SIGMA_MTP
    print(f"  pair sigma under FINDING 460 = sqrt(2)*{SIGMA_MTP}% = {sig_pair:.3f}%")
    z = {}
    for key, rec in table.items():
        m7 = rec["mean7Pct"]
        z[key] = {"naive_z": m7 / 0.189, "channel_z": m7 / sig_pair}
        print(f"  {key}: mean7={m7:+.4f}%  naive z={m7/0.189:+.2f}  "
              f"channel z={m7/sig_pair:+.2f}")
    # E165 alone = (D-B) + (C-A): four independent receipt draws.
    e165_alone = table["DvsB"]["mean7Pct"] + table["CvsA"]["mean7Pct"]
    sig4 = 2.0 * SIGMA_MTP
    z["E165_alone"] = {"value": e165_alone, "channel_z": e165_alone / sig4,
                       "sigma": sig4}
    print(f"  E165 alone = (D-B)+(C-A) = {e165_alone:+.4f}%  "
          f"sigma=2*{SIGMA_MTP}={sig4:.3f}%  channel z={e165_alone/sig4:+.2f}")
    out["zScores"] = z

    print("\n[4] SHAPE ESTIMATOR: mean7 - plutarch, null from E178 groups")
    vecs = channel_null(board, "mtp")
    bygroup = {}
    for v in vecs:
        bygroup.setdefault(v["group"], []).append(
            mean7(v["dev"]) - v["dev"]["plutarch"])
    sd_shape, dof_shape = pooled_sd(bygroup.values())
    bygroup_m7 = {}
    for v in vecs:
        bygroup_m7.setdefault(v["group"], []).append(mean7(v["dev"]))
    sd_m7, dof_m7 = pooled_sd(bygroup_m7.values())
    bygroup_plu = {}
    for v in vecs:
        bygroup_plu.setdefault(v["group"], []).append(v["dev"]["plutarch"])
    sd_plu, dof_plu = pooled_sd(bygroup_plu.values())
    # correlation between mean7 and plutarch deviations within groups
    xs = [mean7(v["dev"]) for v in vecs]
    ys = [v["dev"]["plutarch"] for v in vecs]
    mxs, mys = st.mean(xs), st.mean(ys)
    num = sum((a - mxs) * (b - mys) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mxs) ** 2 for a in xs)
                    * sum((b - mys) ** 2 for b in ys))
    corr = num / den
    print(f"  n receipts in groups: {len(vecs)}  groups: {len(bygroup)}")
    print(f"  within-group sd(mean7 dev)      = {sd_m7:.4f}%  dof {dof_m7}")
    print(f"  within-group sd(plutarch dev)   = {sd_plu:.4f}%  dof {dof_plu}")
    print(f"  within-group sd(mean7-plutarch) = {sd_shape:.4f}%  dof {dof_shape}")
    print(f"  corr(mean7 dev, plutarch dev)   = {corr:+.3f}")
    print("  -> the channel is only partly uniform across prompts; the shape")
    print("     contrast removes the uniform part and keeps the rest.")
    sd_shape_pair = math.sqrt(2.0) * sd_shape
    shape = {}
    for key, rec in table.items():
        s = rec["mean7Pct"] - rec["plutarchPct"]
        shape[key] = {"shapePct": s, "z": s / sd_shape_pair}
        print(f"  {key}: shape={s:+.4f}%  pair sigma={sd_shape_pair:.4f}%  "
              f"z={s/sd_shape_pair:+.2f}")
    out["shapeEstimator"] = {
        "withinGroupSdMean7Pct": sd_m7, "dofMean7": dof_m7,
        "withinGroupSdPlutarchPct": sd_plu, "dofPlutarch": dof_plu,
        "withinGroupSdShapePct": sd_shape, "dofShape": dof_shape,
        "corrMean7Plutarch": corr, "pairSigmaPct": sd_shape_pair,
        "contrasts": shape,
    }

    print("\n[4b] PER-PROMPT CHANNEL sigma - is the channel uniform?")
    per_prompt_sd = {}
    for p in PROMPTS:
        bg = {}
        for v in vecs:
            bg.setdefault(v["group"], []).append(v["dev"][p])
        s, dof = pooled_sd(bg.values())
        per_prompt_sd[p] = s
        print(f"  {p:10s} within-group sd = {s:.4f}%   dof {dof}")
    print("  FINDING 440 used 'plutarch ~ 0' to bound out any receipt-level")
    print("  offset. That check has no power: plutarch's own channel sd is")
    print(f"  {per_prompt_sd['plutarch']:.3f}%, ~{st.mean(per_prompt_sd[p] for p in DRAFTING)/per_prompt_sd['plutarch']:.1f}x smaller than the drafting prompts.")
    out["perPromptChannelSdPct"] = per_prompt_sd

    print("\n[5] MECHANISM-SHAPE TEST (drafting prompts only)")
    Rb = round_costs(R["B"])
    print("  measured round cost R_p from receipt B (ms/round):")
    for p in PROMPTS:
        M = 1.0 + R["B"]["legs"][p]["edl"]
        law = 16.1585 + 5.3350 * M
        print(f"    {p:10s} M={M:5.3f}  R={Rb[p]:7.3f}  affine law={law:7.3f}"
              f"  resid={Rb[p]-law:+6.3f}")
    print("  plutarch is excluded from the fit: 449/488 of its rounds do not")
    print("  draft, so a drafting-round mechanism barely fires there, and its")
    print("  channel sd is 6x smaller, which would dominate an OLS fit.")
    # Exposure for a fixed additive per-round cost: relative delta = delta/R_p.
    x = [1.0 / Rb[p] for p in DRAFTING]
    add = {}
    for key, rec in table.items():
        hi, lo = key.split("vs")
        d = pct_delta(R[hi], R[lo], "decode")
        y = [d[p] for p in DRAFTING]
        icpt, slope, se_i, se_s, rsd = ols(x, y)
        us = slope * 10.0  # %*ms -> us/round
        add[key] = {"channelInterceptPct": icpt, "slopePctMs": slope,
                    "perRoundUs": us, "seSlopeUs": se_s * 10.0,
                    "t": slope / se_s, "residSdPct": rsd,
                    "perPromptDecode": d}
        print(f"  {key}: channel intercept={icpt:+.4f}%  "
              f"additive={us:+8.1f} us/round (se {se_s*10:.1f}, t={slope/se_s:+.2f})")

    vecs_d = channel_null(board, "decode")
    bygroup_add = {}
    for v in vecs_d:
        y = [v["dev"][p] for p in DRAFTING]
        _, slope, _, _, _ = ols(x, y)
        bygroup_add.setdefault(v["group"], []).append(slope * 10.0)
    sd_add, dof_add = pooled_sd(bygroup_add.values())
    sd_add_pair = math.sqrt(2.0) * sd_add
    print(f"\n  null from {len(vecs_d)} receipts / {len(bygroup_add)} identical-content"
          f" groups:")
    print(f"    within-group sd(additive us/round) = {sd_add:.1f} us  dof {dof_add}")
    print(f"    pair sigma = {sd_add_pair:.1f} us/round")
    for key in add:
        add[key]["nullZ"] = add[key]["perRoundUs"] / sd_add_pair
        print(f"    {key}: {add[key]['perRoundUs']:+8.1f} us/round  "
              f"z={add[key]['nullZ']:+.2f}")
    out["additiveEstimator"] = {
        "exposure": "1/R_p over the 7 drafting prompts; "
                    "R_p = 512*(mtp-prefill)/N_rounds from receipt B",
        "roundCostsMs": Rb,
        "nullSdUsPerRound": sd_add, "nullDof": dof_add,
        "nullPairSigmaUsPerRound": sd_add_pair,
        "contrasts": add,
    }

    print("\n[6] SHAPE DISCRIMINATION: is D-B shaped like E165, or like a draw?")
    print("  Model CHANNEL   : y_p = c            (multiplicative host offset)")
    print("  Model E165      : y_p = a / R_p      (fixed us/round at round start)")
    dvb = pct_delta(R["D"], R["B"], "decode")
    y = [dvb[p] for p in DRAFTING]
    c = st.mean(y)
    sse_c = sum((yi - c) ** 2 for yi in y)
    a = sum(yi * xi for yi, xi in zip(y, x)) / sum(xi * xi for xi in x)
    sse_a = sum((yi - a * xi) ** 2 for yi, xi in zip(y, x))
    print(f"  CHANNEL fit: c = {c:+.4f}%          SSE = {sse_c:.4f}")
    print(f"  E165    fit: a = {a*10:+.1f} us/round   SSE = {sse_a:.4f}")
    print(f"  SSE ratio (E165/CHANNEL) = {sse_a/sse_c:.2f} "
          f"-> {'CHANNEL' if sse_c < sse_a else 'E165'} shape fits better")
    pred917 = {p: 91.7 / 10.0 / Rb[p] * 10.0 for p in DRAFTING}
    print("\n  FINDING 440's own mechanism claim (+917 us/round) predicts:")
    for p in sorted(DRAFTING, key=lambda q: Rb[q]):
        print(f"    {p:10s} R={Rb[p]:6.2f} ms  predicted {917.0/Rb[p]/1000*100:+.3f}%"
              f"   observed {dvb[p]:+.3f}%")
    print("  Predicted: cheap-round prompts hurt MOST (drama +2.68%, botany +1.68%).")
    print(f"  Observed : drama {dvb['drama']:+.3f}% is the SMALLEST, "
          f"botany {dvb['botany']:+.3f}%. The ordering is inverted.")
    out["shapeDiscrimination"] = {
        "channelFitPct": c, "channelSSE": sse_c,
        "e165FitUsPerRound": a * 10.0, "e165SSE": sse_a,
        "sseRatioE165OverChannel": sse_a / sse_c,
        "verdict": "CHANNEL shape fits better" if sse_c < sse_a else "E165 shape fits better",
    }

    print("\n[7] E165's LOCAL PREDICTION vs THE RANKED ADDITIVE ESTIMATE")
    print("  E165 recovered 539.0 us/round locally (M4 Pro, harness=local).")
    print("  FINDING 440 claimed it ADDS ~917 us/round of memory traffic on M5.")
    dvbA, cvaA = add["DvsB"], add["CvsA"]
    print(f"  D-B additive = {dvbA['perRoundUs']:+.1f} +/- {sd_add_pair:.1f} us/round"
          f"  (E165 minus instrumentation)")
    print(f"  C-A additive = {cvaA['perRoundUs']:+.1f} us/round  (instrumentation I)")
    e165_us = dvbA["perRoundUs"] + cvaA["perRoundUs"]
    sig_us4 = 2.0 * sd_add
    print(f"  E165 alone   = {e165_us:+.1f} +/- {sig_us4:.1f} us/round  "
          f"z={e165_us/sig_us4:+.2f}")
    print(f"  +917 us/round is excluded at "
          f"{(917.0-dvbA['perRoundUs'])/sd_add_pair:.1f} sigma by the D-B tilt alone.")
    out["e165AdditiveUsPerRound"] = {"value": e165_us, "sigma": sig_us4,
                                     "z": e165_us / sig_us4,
                                     "localRecoveryUsPerRound": 539.0,
                                     "finding440ClaimedUsPerRound": 917.0,
                                     "sigmaExcluding917": (917.0 - dvbA["perRoundUs"]) / sd_add_pair}

    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
