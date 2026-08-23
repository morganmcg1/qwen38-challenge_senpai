"""Does the missing Rule 137 warmup leg threaten the rung A conclusion?

Rung A ran before Rule 137 existed, so its 16 legs span an 18.27 C entry
temperature range instead of the 1.757 C a warmup leg produces. The ABBA order
cancels monotone drift to first order, but "to first order" is a claim, not a
measurement. This script measures the residual risk three ways and prints the
verdict, using only the recorded rung A legs. Zero GPU.

  1. Within-arm slope of seed prefill against entry temperature. If prefill is
     insensitive to entry temperature, the spread cannot carry the effect.
  2. The arm effect after removing that slope from every leg.
  3. An entry-temperature-matched contrast: the coldest cand legs against the
     warmest base legs, which is the direction that would ERASE a real effect
     if temperature were producing it.

  usage: research/e147_rungA_thermal_check.py [--json research/e147-rungA.json]
"""
import argparse
import json
import statistics


def slope(xs, ys):
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return 0.0, my
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    sx = statistics.stdev(xs)
    sy = statistics.stdev(ys)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((n - 1) * sx * sy)


def welch(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = (va / len(a) + vb / len(b)) ** 0.5
    return ma, mb, 100.0 * (mb - ma) / ma, (mb - ma) / se if se else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/e147-rungA.json")
    args = ap.parse_args()

    legs = json.load(open(args.json))["legs"]
    legs = [l for l in legs if l["witness"] == "ok" and l["entry_c"] is not None]
    base = [l for l in legs if l["arm"] == "base"]
    cand = [l for l in legs if l["arm"] == "cand"]

    def pf(rows):
        return [r["seed_prefill_seconds"] for r in rows]

    def ec(rows):
        return [r["entry_c"] for r in rows]

    out = {"legs_used": len(legs), "base_legs": len(base), "cand_legs": len(cand)}
    all_ec = ec(legs)
    out["entry_c_spread"] = max(all_ec) - min(all_ec)
    out["entry_c_sd"] = statistics.stdev(all_ec)
    # a warmup leg would have removed most of this; E130 rung 12 reached 1.757 C
    out["entry_c_spread_target_rule137"] = 1.757

    # 1. within-arm sensitivity of prefill to entry temperature
    for name, rows in (("base", base), ("cand", cand)):
        b, _ = slope(ec(rows), pf(rows))
        out[f"prefill_seconds_per_deg_c_{name}"] = b
        out[f"prefill_vs_entry_c_r_{name}"] = pearson(ec(rows), pf(rows))

    # pooled within-arm slope, arms centred so the arm effect cannot leak in
    cxs, cys = [], []
    for rows in (base, cand):
        m_e = statistics.fmean(ec(rows))
        m_p = statistics.fmean(pf(rows))
        cxs += [e - m_e for e in ec(rows)]
        cys += [p - m_p for p in pf(rows)]
    b_pooled, _ = slope(cxs, cys)
    out["prefill_seconds_per_deg_c_pooled_within_arm"] = b_pooled
    out["prefill_pct_per_deg_c"] = 100.0 * b_pooled / statistics.fmean(pf(base))

    # The direct confound bound. A temperature confound can only bias the arm
    # contrast through the difference in MEAN entry temperature between arms,
    # which ABBA is designed to drive to zero. Slope times that imbalance is
    # the whole effect the missing warmup leg could have bought.
    imbalance = statistics.fmean(ec(cand)) - statistics.fmean(ec(base))
    out["arm_mean_entry_c_base"] = statistics.fmean(ec(base))
    out["arm_mean_entry_c_cand"] = statistics.fmean(ec(cand))
    out["arm_entry_c_imbalance"] = imbalance
    out["thermal_confound_bound_pct"] = abs(out["prefill_pct_per_deg_c"] * imbalance)

    # 2. raw contrast, then the same contrast after removing the pooled slope
    grand_e = statistics.fmean(all_ec)
    adj = {
        id(r): r["seed_prefill_seconds"] - b_pooled * (r["entry_c"] - grand_e)
        for r in legs
    }
    mb, mc, pct_raw, t_raw = welch(pf(base), pf(cand))
    _, _, pct_adj, t_adj = welch([adj[id(r)] for r in base], [adj[id(r)] for r in cand])
    out["prefill_pct_raw"] = pct_raw
    out["prefill_t_raw"] = t_raw
    out["prefill_pct_entry_c_adjusted"] = pct_adj
    out["prefill_t_entry_c_adjusted"] = t_adj
    out["prefill_pct_shift_from_adjustment"] = pct_adj - pct_raw

    # 3. adversarial matched contrast: coldest cand against warmest base.
    # If entry temperature drove the result, this pairing removes or reverses
    # it, because cand is then handed the colder half of the range.
    k = min(len(base), len(cand)) // 2
    warm_base = sorted(base, key=lambda r: -r["entry_c"])[:k]
    cold_cand = sorted(cand, key=lambda r: r["entry_c"])[:k]
    _, _, pct_adv, t_adv = welch(pf(warm_base), pf(cold_cand))
    out["adversarial_k_per_arm"] = k
    out["adversarial_mean_entry_c_base"] = statistics.fmean(ec(warm_base))
    out["adversarial_mean_entry_c_cand"] = statistics.fmean(ec(cold_cand))
    out["adversarial_prefill_pct"] = pct_adv
    out["adversarial_prefill_t"] = t_adv

    # the honest inverse: hand cand the WARMEST legs and base the coldest
    cold_base = sorted(base, key=lambda r: r["entry_c"])[:k]
    warm_cand = sorted(cand, key=lambda r: -r["entry_c"])[:k]
    _, _, pct_inv, t_inv = welch(pf(cold_base), pf(warm_cand))
    out["inverse_mean_entry_c_base"] = statistics.fmean(ec(cold_base))
    out["inverse_mean_entry_c_cand"] = statistics.fmean(ec(warm_cand))
    out["inverse_prefill_pct"] = pct_inv
    out["inverse_prefill_t"] = t_inv

    out["rank_separation_complete"] = max(pf(cand)) < min(pf(base))
    # the effect survives if both adversarial pairings still favour cand
    out["conclusion_survives_thermal_stress"] = bool(
        pct_adv < 0.0 and pct_inv < 0.0 and pct_adj < 0.0
    )
    out["rungA_warmup_leg_present"] = 0.0

    for k_, v in out.items():
        print(f"{k_:44s} {v}")


if __name__ == "__main__":
    main()
