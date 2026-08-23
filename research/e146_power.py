"""E146 R-B: variance components, the corrected MDE table, and error rates.

harness=ranked for the board terms, harness=local for the R-C terms. The two are
never added together; they are reported side by side because they answer
different questions. The ranked terms price an official submission. The local
terms price a local iterate.

THE ERROR THE CAMPAIGN HAS BEEN MAKING. F211 and F216 read a 0.0721 % noise
floor off a set of "null pairs". Those pairs were SELECTED for being tight, so
0.0721 % estimates the width of the tightest part of the nuisance distribution,
not the width of the distribution. Conditioning on smallness and then quoting
the result as the noise floor is the same error as reading the standard
deviation of a sample after discarding its tails.

WHAT THE NUISANCE ACTUALLY LOOKS LIKE. It is not Gaussian. Each timed leg draws
a hidden run mode, and a mode difference moves the candidate mean-8 by a whole
step. A single contrast therefore carries

    N in {-S, 0, +S} with probabilities {p(1-p), p^2 + (1-p)^2, p(1-p)}

plus a small Gaussian within-mode term. The consequence is not mainly a loss of
power. It is that the SIGN of a single-contrast conclusion is wrong with
probability p(1-p), whatever the size of the true effect, until the effect
itself exceeds a step.
"""

import json
import math
import os
import sys

import e146_lib as L
import e146_modes as M
import e146_pairs as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-power.json")
LOCAL_LEGS = os.path.join(
    os.path.dirname(HERE), ".mlxfast-private", "e146", "rc", "legs.jsonl")

# The campaign's incumbent floor, F211 and F216.
INCUMBENT_FLOOR_PCT = 0.0721
# The decision threshold the incumbent floor implies at two standard deviations.
THRESHOLD_PCT = 2.0 * INCUMBENT_FLOOR_PCT


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def call_rates(delta, step_pct, p_high, within_sd_pct, threshold):
    """P(call faster), P(no call), P(call slower) for one contrast.

    `delta` is the true candidate-minus-baseline percentage; negative is an
    improvement. The three nuisance branches are exact, and the within-mode term
    is Gaussian inside each branch.
    """
    branches = [(-step_pct, p_high * (1.0 - p_high)),
                (0.0, p_high ** 2 + (1.0 - p_high) ** 2),
                (step_pct, p_high * (1.0 - p_high))]
    faster = slower = 0.0
    for shift, weight in branches:
        mu = delta + shift
        faster += weight * normal_cdf((-threshold - mu) / within_sd_pct)
        slower += weight * (1.0 - normal_cdf((threshold - mu) / within_sd_pct))
    return faster, 1.0 - faster - slower, slower


def main():
    path, rows = L.load()
    pair_list = P.pairs(rows)
    ks = [p["fit"]["k_us_per_drafting_round"] for p in pair_list]
    flat_pairs = [p for p in pair_list
                  if abs(p["fit"]["k_us_per_drafting_round"]) <= M.STEP_CUT]
    high_pairs = [p for p in pair_list
                  if p["fit"]["k_us_per_drafting_round"] > M.STEP_CUT]
    p_high = len(high_pairs) / len(ks)

    # The crown family carries the campaign's own comparisons, so its step is
    # the one that prices campaign decisions.
    crown = next(r for r in rows if r.id8 == "1db9d63e")
    fam_high = [p["fit"]["k_us_per_drafting_round"] for p in high_pairs
                if p["target"].draft_key() == crown.draft_key()]
    step_us = L.mean(fam_high) if fam_high else L.mean(
        [p["fit"]["k_us_per_drafting_round"] for p in high_pairs])
    anchor = next(r for r in rows if r.id8 == "572b2cc4")
    fit = L.fit_k(anchor, crown, basis_row=anchor)
    pct_per_us = fit["cand_mean8_pct"] / fit["k_us_per_drafting_round"]
    step_pct = step_us * pct_per_us

    # Within-mode scatter, measured on the flat band. A pair difference carries
    # two legs, so one leg is that divided by root two.
    flat_pair_sd_pct = L.sd([p["fit"]["cand_mean8_pct"] for p in flat_pairs])
    within_leg_sd_pct = flat_pair_sd_pct / math.sqrt(2.0)

    print("=== variance components of one ranked candidate mean-8 contrast ===")
    print("harness=ranked, %d pure-nuisance pairs, %d of them +step" % (len(ks), len(high_pairs)))
    print("%-42s %12s %12s" % ("component", "sd (pp)", "share of var"))
    var_mode = 2.0 * p_high * (1.0 - p_high) * step_pct ** 2
    var_within = 2.0 * within_leg_sd_pct ** 2
    total = var_mode + var_within
    print("%-42s %12.4f %12.4f" % ("run mode, two legs", math.sqrt(var_mode), var_mode / total))
    print("%-42s %12.4f %12.4f" % ("within mode, two legs", math.sqrt(var_within), var_within / total))
    print("%-42s %12.4f %12s" % ("TOTAL", math.sqrt(total), "1"))
    print("step %.1f us/drafting-round = %.4f pp of the candidate mean-8"
          % (step_us, step_pct))
    print("P(a leg carries the state) = %.4f" % p_high)
    print()
    print("the campaign's incumbent floor F211/F216 %.4f pp understates this by %.1fx"
          % (INCUMBENT_FLOOR_PCT, math.sqrt(total) / INCUMBENT_FLOOR_PCT))
    print("F211/F216 selected pairs BY tightness, so it measures the flat band,")
    print("and even for the flat band the honest value is %.4f pp." % flat_pair_sd_pct)

    # ---- F227: what each published leg is worth as an instrument ----------
    # Measured on the pure-nuisance pairs, so every number below is the
    # run-to-run spread of a quantity whose true difference is exactly zero.
    print("\n=== F227: precision of each published leg, on zero-delta pairs ===")
    print("%-34s %10s %10s %10s" % ("published quantity", "sd (pp)", "flat only", "n"))
    leg_sd = {}
    for label, field in (("candidate leg, prefill included", "total"),
                         ("candidate leg, decode only", "decode"),
                         ("serial leg", "serial"),
                         ("seed prefill", "prefill")):
        vals, flat_vals = [], []
        for pair in pair_list:
            t, r = pair["target"], pair["replicate"]
            if field == "prefill" and not (t.has_prefill() and r.has_prefill()):
                continue
            d = L.pct_diff(t, r, field)
            v = sum(d[p] for p in L.PROMPT_ORDER) / 8.0
            vals.append(v)
            if abs(pair["fit"]["k_us_per_drafting_round"]) <= M.STEP_CUT:
                flat_vals.append(v)
        leg_sd[field] = {"sd_pct": L.sd(vals), "flat_sd_pct": L.sd(flat_vals),
                         "n": len(vals)}
        print("%-34s %10.4f %10.4f %10d"
              % (label, L.sd(vals), L.sd(flat_vals), len(vals)))
    print("The prefill is the board's most precise published leg and it does")
    print("NOT carry the state, so it is the natural negative control and the")
    print("natural check that a receipt pair is comparable at all.")

    # ---- the corrected MDE table -----------------------------------------
    print("\n=== minimum detectable effect, 5 per cent two-sided, 80 per cent power ===")
    z_a, z_b = 1.959963985, 0.8416212336
    designs = []

    def mde(sd_pct, n):
        return (z_a + z_b) * sd_pct / math.sqrt(n)

    sd_one = math.sqrt(total)
    for label, sd_pct, note in (
            ("incumbent assumption (F211/F216)", INCUMBENT_FLOOR_PCT,
             "wrong: selected by tightness"),
            ("one contrast, mode not handled", sd_one, "what the campaign has been doing"),
            ("one contrast, both rows mode classified", math.sqrt(var_within),
             "needs 4+ same-decision rows per family"),
            ("one contrast, best of 2 runs per side", None, "min statistic"),
            ("one contrast, best of 3 runs per side", None, "min statistic")):
        if sd_pct is None:
            n_runs = 2 if "of 2" in label else 3
            residual_p = p_high ** n_runs
            var = 2.0 * residual_p * (1.0 - residual_p) * step_pct ** 2 + var_within
            sd_pct = math.sqrt(var)
            note = "%s, P(no flat draw) = %.4f" % (note, residual_p)
        designs.append({"design": label, "sd_pct": sd_pct,
                        "mde_n1_pct": mde(sd_pct, 1), "note": note})
        print("%-42s sd %7.4f  MDE %7.4f pp   %s" % (label, sd_pct, mde(sd_pct, 1), note))

    # The brief asks for these two at 2 sigma, in the leg frame, by name.
    mde_classified = 2.0 * math.sqrt(var_within)
    paired_sd = L.sd([p["fit"]["cand_mean8_pct"] for p in pair_list])
    mde_paired = 2.0 * paired_sd
    print("\n=== the two named 2 sigma figures, leg frame ===")
    print("e146_mde_2sigma_pct_mode_classified = %.4f pp" % mde_classified)
    print("  definition: both rows of the contrast have their state removed by")
    print("  the R-A family-placement classifier, so only within-mode scatter")
    print("  is left. This is the design the campaign should adopt.")
    print("e146_mde_2sigma_pct_paired_replicate = %.4f pp" % mde_paired)
    print("  definition: the candidate is compared against a same-tree replicate")
    print("  and NOTHING is corrected. This is measured directly, not modelled:")
    print("  it is 2x the observed sd of the %d pure-nuisance pairs." % len(pair_list))
    print("  It is LARGER than the classified design because pairing removes")
    print("  tree-generation drift but does not remove the state; both legs")
    print("  still draw independently.")

    print("\nranked submissions needed to resolve a true effect of this size,")
    print("with the mode left unhandled:")
    print("%-14s %12s" % ("effect (pp)", "n contrasts"))
    need = {}
    for effect in (0.10, 0.20, 0.30, 0.50, 1.00, 1.50):
        n = math.ceil(((z_a + z_b) * sd_one / effect) ** 2)
        need["%.2f" % effect] = n
        print("%-14.2f %12d" % (effect, n))
    print("the same table once both rows are mode classified:")
    for effect in (0.10, 0.20, 0.30, 0.50, 1.00, 1.50):
        n = math.ceil(((z_a + z_b) * math.sqrt(var_within) / effect) ** 2)
        print("%-14.2f %12d" % (effect, n))

    # ---- error rates at the campaign's own threshold ---------------------
    print("\n=== error rates of ONE contrast at the incumbent threshold %.4f pp ==="
          % THRESHOLD_PCT)
    f0, n0, s0 = call_rates(0.0, step_pct, p_high, math.sqrt(var_within), THRESHOLD_PCT)
    type1 = f0 + s0
    print("truth is zero:")
    print("  P(call faster) %.4f   P(no call) %.4f   P(call slower) %.4f" % (f0, n0, s0))
    print("  e146_type1_at_zero = %.4f" % type1)

    delta = -0.30
    f1, n1, s1 = call_rates(delta, step_pct, p_high, math.sqrt(var_within), THRESHOLD_PCT)
    type2 = 1.0 - f1
    print("truth is a real 0.30 pp improvement (delta = %.2f pp):" % delta)
    print("  P(call faster, correct) %.4f" % f1)
    print("  P(no call)              %.4f" % n1)
    print("  P(call SLOWER, wrong sign) %.4f" % s1)
    print("  e146_type2_at_030pct = %.4f" % type2)
    print("  the failure is not weak power, it is that the sign is inverted in")
    print("  %.1f per cent of single contrasts." % (100.0 * s1))

    # ---- the local floor, when R-C has produced it -----------------------
    local = None
    if os.path.exists(LOCAL_LEGS):
        legs = [json.loads(line) for line in open(LOCAL_LEGS)]
        mtp = [l for l in legs if l["kind"] == "mtp"
               and l["gate_qualified_for_timing"]]
        ser = [l for l in legs if l["kind"] == "serial"
               and l["gate_qualified_for_timing"]]
        if len(mtp) >= 2:
            spt = [l["metrics"]["parent_measured_seconds_per_token"] for l in mtp]
            blocks = [l["metrics"]["blocks_only_seconds"] for l in mtp
                      if l["metrics"].get("blocks_only_seconds")]
            rounds = sorted({l["metrics"]["round_count"] for l in mtp})
            mean_spt = L.mean(spt)
            sd_pct = 100.0 * L.sd(spt) / mean_spt
            blk_pct = (100.0 * L.sd(blocks) / L.mean(blocks)) if len(blocks) > 1 else None
            ser_pct = None
            if len(ser) >= 2:
                sv = [l["metrics"]["parent_measured_seconds_per_token"] for l in ser]
                ser_pct = 100.0 * L.sd(sv) / L.mean(sv)
            # The first resident leg of a session pays cold caches and first
            # compilation, so the raw sd mixes one warm-up transient with the
            # steady-state floor. Report both, and the drift slope that says
            # which one dominates.
            tail = spt[1:]
            sd_tail_pct = (100.0 * L.sd(tail) / L.mean(tail)) if len(tail) > 1 else None
            slope_pct = None
            if len(spt) > 2:
                xs = list(range(len(spt)))
                xm, ym = L.mean(xs), mean_spt
                num = sum((x - xm) * (y - ym) for x, y in zip(xs, spt))
                den = sum((x - xm) ** 2 for x in xs)
                slope_pct = 100.0 * (num / den) / mean_spt
            local = {
                "harness": "local",
                # F229: Qwen36MTPBlockSession.swift:222 wireResidentWeightsIfEnabled()
                # guards on physicalMemory >= 96 GiB. This host has 48 GiB, so
                # the body never runs and these legs cannot draw the ranked
                # state. This sd is a lower bound on the ranked leg sd, never a
                # substitute for it.
                "e146_local_floor_excludes_wired_residency": 1.0,
                "n_mtp_legs": len(mtp),
                "n_serial_legs": len(ser),
                "e146_local_fixed_binary_sd_pct": sd_pct,
                "e146_local_fixed_binary_sd_excl_first_pct": sd_tail_pct,
                "e146_local_drift_slope_pct_per_leg": slope_pct,
                "e146_local_blocks_only_sd_pct": blk_pct,
                "e146_local_serial_sd_pct": ser_pct,
                "e146_local_round_count_unique": len(rounds),
                "round_counts": rounds,
                "mean_seconds_per_token": mean_spt,
                "seconds_per_token": spt,
                "e146_local_to_ranked_candidate_noise_ratio":
                    sd_pct / INCUMBENT_FLOOR_PCT,
            }
            print("\n=== harness=local: the fixed-binary floor from R-C ===")
            print("gate-qualified MTP legs %d, serial legs %d" % (len(mtp), len(ser)))
            print("round_count values observed: %s" % rounds)
            print("e146_local_fixed_binary_sd_pct   %.4f" % sd_pct)
            if sd_tail_pct is not None:
                print("  excluding the first leg          %.4f" % sd_tail_pct)
            if slope_pct is not None:
                print("  drift slope, per MTP leg         %+.4f pp" % slope_pct)
            if blk_pct is not None:
                print("e146_local_blocks_only_sd_pct    %.4f" % blk_pct)
            if ser_pct is not None:
                print("e146_local_serial_sd_pct         %.4f" % ser_pct)
            print("e146_local_to_ranked_candidate_noise_ratio %.2f"
                  % (sd_pct / INCUMBENT_FLOOR_PCT))
            print("\nlocal ABBA MDE per arm. The 2 sigma column is the one the")
            print("brief asks for; the power column is what a screen should use")
            print("if it wants an 80 per cent chance of seeing a true effect.")
            print("%-10s %14s %16s %16s"
                  % ("n per arm", "2 sigma (pp)", "80 % power (pp)",
                     "2 sig, steady"))
            local["mde_by_n"] = {}
            local["mde_2sigma_by_n"] = {}
            local["mde_2sigma_steady_by_n"] = {}
            for n in (1, 2, 4, 8):
                two = 2.0 * sd_pct * math.sqrt(2.0 / n)
                pw = (z_a + z_b) * sd_pct * math.sqrt(2.0 / n)
                steady = (2.0 * sd_tail_pct * math.sqrt(2.0 / n)
                          if sd_tail_pct else float("nan"))
                local["mde_by_n"][str(n)] = pw
                local["mde_2sigma_by_n"][str(n)] = two
                local["mde_2sigma_steady_by_n"][str(n)] = steady
                print("%-10d %14.4f %16.4f %16.4f" % (n, two, pw, steady))
            print("`steady` drops the first resident leg of the session, which")
            print("pays cold caches and first compilation. An ABBA screen that")
            print("discards one warm-up leg per session is entitled to it.")
    else:
        print("\nR-C legs are not present yet; the local block is omitted.")

    payload = {
        "ranked": {
            "harness": "ranked",
            "board_path": path,
            "pairs": len(ks),
            "p_high": p_high,
            "step_us_per_drafting_round": step_us,
            "step_pct_of_candidate_mean8": step_pct,
            "flat_pair_sd_pct": flat_pair_sd_pct,
            "within_leg_sd_pct": within_leg_sd_pct,
            "contrast_sd_pct": sd_one,
            "mode_share_of_variance": var_mode / total,
            "incumbent_floor_pct": INCUMBENT_FLOOR_PCT,
            "understatement_factor": sd_one / INCUMBENT_FLOOR_PCT,
            "designs": designs,
            "contrasts_needed": need,
            "published_leg_precision": leg_sd,
            "e146_mde_2sigma_pct_mode_classified": mde_classified,
            "e146_mde_2sigma_pct_paired_replicate": mde_paired,
            "e146_type1_at_zero": type1,
            "e146_type2_at_030pct": type2,
            "wrong_sign_rate_at_030pct": s1,
        },
        "local": local,
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
