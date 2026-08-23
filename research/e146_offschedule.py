"""E146 F3 section 4: can the state be read off a row with no same-schedule family?

harness=ranked, zero GPU. The advisor asks whether the serial and prefill
channels, which F228 says the state does not touch, give a state readout for a
row like `e003a86d` whose mechanism changes every draft length and which is
therefore outside every digit-identical cohort.

Three tests, in increasing order of usefulness.

1. The literal proposal. On the pure-nuisance pairs the mechanism is zero by
   construction, so the candidate response IS the state. Regress it on the
   serial and prefill responses. If those channels carried a readout, this
   regression would explain some of it.

2. The real question. Does the residual statistic that reached AUC 0.820 on
   schedule survive when the anchor has a DIFFERENT draft-length tuple? This is
   directly testable: take the labelled cohort rows and recompute their
   residual against off-schedule anchors.

3. The identification. A schedule-changing mechanism moves the round counts,
   and a pure schedule change at unchanged per-round cost moves decode time by
   exactly the fractional change in round count. That is a second basis, and it
   is not parallel to the state basis. Report whether the two-parameter fit is
   identified and what it says about `e003a86d`.
"""

import json
import os
import random
import sys

import e146_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-offschedule.json")
PAIRS = os.path.join(HERE, "e146-pairs.json")
COHORT = os.path.join(HERE, "e146-cohort.json")
ANCHOR = "572b2cc4"
PB6 = "e003a86d"
PERMUTATIONS = 20000


def corr(xs, ys):
    n = len(xs)
    mx, my = L.mean(xs), L.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy and n > 1 else float("nan")


def r2_two_regressors(y, x1, x2):
    """R2 of y on [1, x1, x2] by normal equations on centred regressors."""
    my = L.mean(y)
    a = [v - L.mean(x1) for v in x1]
    b = [v - L.mean(x2) for v in x2]
    t = [v - my for v in y]
    saa = sum(v * v for v in a)
    sbb = sum(v * v for v in b)
    sab = sum(u * v for u, v in zip(a, b))
    sat = sum(u * v for u, v in zip(a, t))
    sbt = sum(u * v for u, v in zip(b, t))
    det = saa * sbb - sab * sab
    if abs(det) < 1e-18:
        return float("nan")
    ca = (sbb * sat - sab * sbt) / det
    cb = (saa * sbt - sab * sat) / det
    ss_res = sum((ti - ca * ai - cb * bi) ** 2 for ti, ai, bi in zip(t, a, b))
    ss_tot = sum(v * v for v in t)
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def auc(pos, neg):
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (len(pos) * len(neg))


def permuted_auc(values, n_pos, seed=20146):
    rng = random.Random(seed)
    pool = list(values)
    draws = []
    for _ in range(PERMUTATIONS):
        rng.shuffle(pool)
        draws.append(auc(pool[:n_pos], pool[n_pos:]))
    draws.sort()
    return draws


def two_basis_fit(anchor, row):
    """Split the observed decode response into a state term and a schedule term.

    b_state[p] is the per-drafting-round basis the state rides on.
    b_sched[p] is the fractional change in round count, which is exactly what a
    pure schedule change at unchanged per-round cost costs.
    """
    diffs = L.pct_diff(anchor, row, field="decode")
    b_state = L.design_matrix(anchor, row, "drafting_round", "decode")
    b_sched = {}
    for p in L.PROMPT_ORDER:
        r_row = L.round_count(p, row.dlen(p))
        r_anc = L.round_count(p, anchor.dlen(p))
        b_sched[p] = 100.0 * (r_row / r_anc - 1.0)
    names = list(L.PROMPT_ORDER)
    y = [diffs[p] for p in names]
    a = [b_state[p] for p in names]
    b = [b_sched[p] for p in names]
    saa = sum(v * v for v in a)
    sbb = sum(v * v for v in b)
    sab = sum(u * v for u, v in zip(a, b))
    say = sum(u * v for u, v in zip(a, y))
    sby = sum(u * v for u, v in zip(b, y))
    det = saa * sbb - sab * sab
    k_state = (sbb * say - sab * sby) / det
    k_sched = (saa * sby - sab * say) / det
    fitted = [k_state * ai + k_sched * bi for ai, bi in zip(a, b)]
    resid = [yi - fi for yi, fi in zip(y, fitted)]
    return {
        "k_state_us_per_drafting_round": k_state,
        "k_schedule_multiplier": k_sched,
        "basis_correlation": corr(a, b),
        "residual_sd_pp": L.sd(resid),
        "state_share_pct_of_decode": k_state * L.mean(a),
        "schedule_share_pct_of_decode": k_sched * L.mean(b),
        "observed_decode_pct": L.mean(y),
    }


def main():
    path, rows = L.load()
    by_id = {r.id8: r for r in rows}
    pairs = json.load(open(PAIRS))["pairs"]
    cohort = json.load(open(COHORT))
    out = {"harness": "ranked", "board_path": path}

    print("=== TEST 1. do the two state-free channels carry a state readout? ===")
    print("pure-nuisance pairs, mechanism is zero by construction, so the")
    print("candidate response IS the state.\n")
    y, xs, xp = [], [], []
    for pair in pairs:
        a, b = by_id.get(pair["target"]), by_id.get(pair["replicate"])
        if a is None or b is None or not (a.has_prefill() and b.has_prefill()):
            continue
        y.append(L.mean(L.pct_diff(a, b, field="decode").values()))
        xs.append(L.mean(L.pct_diff(a, b, field="serial").values()))
        xp.append(L.mean(L.pct_diff(a, b, field="prefill").values()))
    r_serial, r_prefill = corr(y, xs), corr(y, xp)
    r2 = r2_two_regressors(y, xs, xp)
    print("n pairs with all three channels        %d" % len(y))
    print("corr(candidate decode, serial)         %+.4f" % r_serial)
    print("corr(candidate decode, prefill)        %+.4f" % r_prefill)
    print("R2 of candidate decode on both         %.4f" % r2)
    print("residual sd after removing both, pp    %.4f"
          % (L.sd(y) * (1.0 - r2) ** 0.5))
    print("raw candidate decode sd, pp            %.4f" % L.sd(y))
    out["channel_regression"] = {
        "n_pairs": len(y), "corr_serial": r_serial, "corr_prefill": r_prefill,
        "r2_on_serial_and_prefill": r2, "candidate_decode_sd_pp": L.sd(y),
        "residual_sd_pp": L.sd(y) * (1.0 - r2) ** 0.5,
    }

    print("\n=== TEST 2. does the residual statistic survive off schedule? ===")
    group_of = {}
    for g in cohort["groups"]:
        for row_id in g["ids"]:
            group_of[row_id] = g["name"]
    labelled = [(rid, name) for rid, name in group_of.items()
                if name in ("A main", "B/C") and rid in by_id]
    bar_key = by_id[cohort["bar"]].draft_key()

    panel = []
    for rid, name in sorted(labelled):
        if name == "B/C":
            panel.append((rid, "control: cohort HIGH, on schedule"))
    for rid, name in sorted(labelled):
        if name == "A main":
            panel.append((rid, "control: cohort MAIN, on schedule"))
    for cand in (ANCHOR, "623e77af", "0c6191b7", PB6):
        row = by_id.get(cand)
        if row is not None and row.draft_key() != bar_key:
            panel.append((cand, "off schedule, mode unknown"))

    print("labelled cohort rows: %d" % len(labelled))
    print("\nRead the direction, not only the size. If the anchor itself drew")
    print("the state then the step cancels against a HIGH row and appears")
    print("against a MAIN row, so the ordering inverts and the AUC falls below")
    print("the null. The two control blocks calibrate that reading.\n")
    print("%-12s %8s %8s %10s  %s"
          % ("anchor", "AUC", "null p95", "p", "role"))
    table = []
    for anchor_id, role in panel:
        anchor = by_id[anchor_id]
        stats = {}
        for rid, name in labelled:
            if rid == anchor_id:
                continue
            stats[rid] = L.fit_k(anchor, by_id[rid], field="decode")["residual_sd_pp"]
        pos = [v for rid, v in stats.items() if group_of[rid] == "B/C"]
        neg = [v for rid, v in stats.items() if group_of[rid] == "A main"]
        observed = auc(pos, neg)
        draws = permuted_auc(list(stats.values()), len(pos))
        p95 = draws[int(0.95 * len(draws))]
        p_value = sum(1 for d in draws if d >= observed) / float(len(draws))
        call = "state" if observed < 0.5 else "flat"
        print("%-12s %8.4f %8.4f %10.5f  %s -> %s"
              % (anchor_id, observed, p95, p_value, role, call))
        table.append({"anchor": anchor_id, "role": role, "auc": observed,
                      "null_p95": p95, "n_high": len(pos), "n_main": len(neg),
                      "p_value_one_sided": p_value, "call": call})
    out["anchor_inversion_panel"] = table

    highs = [t["auc"] for t in table if t["role"].endswith("HIGH, on schedule")]
    mains = [t["auc"] for t in table if t["role"].endswith("MAIN, on schedule")]
    offs = [t for t in table if t["role"].startswith("off schedule")]
    print("\ncontrol calibration")
    print("  known HIGH anchors, n=%d, AUC mean %.4f, max %.4f"
          % (len(highs), L.mean(highs), max(highs)))
    print("  known MAIN anchors, n=%d, AUC mean %.4f, min %.4f"
          % (len(mains), L.mean(mains), min(mains)))
    separated = max(highs) < min(mains)
    print("  controls fully separated: %s" % separated)
    out["control_calibration"] = {
        "high_anchor_auc_mean": L.mean(highs), "high_anchor_auc_max": max(highs),
        "main_anchor_auc_mean": L.mean(mains), "main_anchor_auc_min": min(mains),
        "controls_fully_separated": separated,
    }
    for entry in offs:
        print("  %-10s AUC %.4f -> %s" % (entry["anchor"], entry["auc"],
                                          entry["call"]))

    print("\n=== TEST 3. two-basis identification on %s ===" % PB6)
    fit = two_basis_fit(by_id[ANCHOR], by_id[PB6])
    for key in ("basis_correlation", "k_state_us_per_drafting_round",
                "k_schedule_multiplier", "state_share_pct_of_decode",
                "schedule_share_pct_of_decode", "observed_decode_pct",
                "residual_sd_pp"):
        print("%-34s %+12.4f" % (key, fit[key]))
    one_basis = L.fit_k(by_id[ANCHOR], by_id[PB6], field="decode")
    print("%-34s %+12.4f" % ("one-basis k us/dr",
                             one_basis["k_us_per_drafting_round"]))
    print("%-34s %+12.4f" % ("one-basis residual sd pp",
                             one_basis["residual_sd_pp"]))
    out["pb6_two_basis"] = fit
    out["pb6_one_basis"] = {
        "k_us_per_drafting_round": one_basis["k_us_per_drafting_round"],
        "residual_sd_pp": one_basis["residual_sd_pp"],
    }

    # Calibrate against the control blocks instead of a fixed threshold. A call
    # is decisive only outside the span of the known-HIGH controls, because
    # three of eight known-HIGH anchors also clear the permuted null.
    high_lo, high_hi = min(highs), max(highs)
    main_lo = min(mains)
    margin = main_lo - high_hi
    decisive = 0
    for entry in offs:
        auc_value = entry["auc"]
        if auc_value < high_lo:
            entry["calibrated_call"] = "state"
            decisive += 1
        elif auc_value >= main_lo:
            entry["calibrated_call"] = "flat"
            decisive += 1
        else:
            entry["calibrated_call"] = "ambiguous"
    fraction = decisive / float(len(offs)) if offs else 0.0
    possible = 1.0 if (separated and decisive > 0) else 0.0
    out["e146_offschedule_state_readout_possible"] = possible
    out["e146_offschedule_decisive_fraction"] = fraction
    out["e146_offschedule_control_separation_margin_auc"] = margin
    out["e146_offschedule_high_control_span_auc"] = [high_lo, high_hi]
    print("\ncalibrated off-schedule calls "
          "(decisive outside the known-HIGH span %.4f..%.4f)"
          % (high_lo, high_hi))
    for entry in offs:
        print("  %-10s AUC %.4f -> %s"
              % (entry["anchor"], entry["auc"], entry["calibrated_call"]))
    print("  control separation margin, AUC %.4f" % margin)
    print("  decisive fraction %.2f (%d of %d)" % (fraction, decisive, len(offs)))
    print("\ne146_offschedule_state_readout_possible = %.1f" % possible)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
